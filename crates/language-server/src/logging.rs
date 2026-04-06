//! Tracing subscriber and panic hook for the language server.
//!
//! # Layering
//!
//! Two subscribers layered via a `Registry`:
//!
//! 1. **File layer** — structured output at `info` for our crates and `warn`
//!    for dependencies, written to `$FE_LSP_LOG_FILE` (default
//!    `$XDG_CACHE_HOME/fe-language-server/<pid>.log`). This is the primary
//!    forensics sink: everything useful for debugging lands here.
//!
//! 2. **Client layer** — `warn` and above only, forwarded to the LSP client
//!    as `window/logMessage` notifications. This is what the user actually
//!    sees in their editor's log panel. Kept narrow on purpose: the editor
//!    log is a lousy debugging surface and floods drown real errors.
//!
//! Filters are independent: the file log gets the full story; the client
//! log gets what an end user needs to see. Controlled by `FE_LSP_LOG`
//! (file) and `FE_LSP_CLIENT_LOG` (client), both using standard
//! `tracing_subscriber::EnvFilter` syntax.
//!
//! # Panic hook
//!
//! [`setup_panic_hook`] installs a process-wide hook via `Once`. On panic
//! it writes **directly** to `panics-<pid>.log` via `OpenOptions::append`,
//! bypassing the tracing subscriber entirely — so it works even during
//! shutdown when subscribers are being torn down. The written record
//! includes the panic payload, location, thread name,
//! [`crate::panic_context::format()`] stack, `salsa::Backtrace::capture()`
//! query stack, and `std::backtrace::Backtrace::force_capture()`.

use async_lsp::{
    ClientSocket, LanguageClient,
    lsp_types::{LogMessageParams, MessageType},
};
use std::{
    backtrace::Backtrace,
    fs::{File, OpenOptions},
    io::Write as _,
    path::{Path, PathBuf},
    sync::{Arc, Once},
};
use tracing::{Level, Metadata, subscriber::DefaultGuard};
use tracing_subscriber::{
    EnvFilter, Layer,
    fmt::{MakeWriter, writer::BoxMakeWriter},
    layer::SubscriberExt,
    registry::Registry,
};

/// Default filter applied to the file log when `FE_LSP_LOG` is not set.
///
/// Mirrors rust-analyzer's philosophy: default to `warn` globally, then
/// explicitly opt our own crates into `info` so dependency noise doesn't
/// drown out the signal. Override with `FE_LSP_LOG` using standard
/// `EnvFilter` syntax, e.g. `FE_LSP_LOG="fe_language_server=debug,warn"`.
///
/// Note: `EnvFilter` matches target **prefixes**, where the target is
/// normally the module path (like `fe_language_server::functionality::handlers`).
/// We also emit some events with **custom** `target:` attributes
/// (e.g. `target: "fe::lsp::startup"`) — those do NOT share a prefix with
/// `fe_language_server`, so they need their own directive (`fe::lsp=info`)
/// or they get dropped.
const DEFAULT_FILE_FILTER: &str = "warn,\
    fe_language_server=info,\
    fe_driver=info,\
    fe_hir=info,\
    fe_hir_analysis=info,\
    fe_mir=info,\
    fe_parser=info,\
    fe_resolver=info,\
    fe_common=info,\
    fe_codegen=info,\
    fe_fmt=info,\
    fe::lsp=info,\
    fe::lsp::startup=info,\
    fe::lsp::workspace=info,\
    fe::lsp::request=info,\
    fe::lsp::panic=error,\
    salsa=warn,\
    act_locally=warn,\
    tower=warn,\
    hyper=warn,\
    async_lsp=warn,\
    axum=warn,\
    tokio=warn";

/// Default filter applied to the client (editor) log when
/// `FE_LSP_CLIENT_LOG` is not set. `warn` means the editor only sees real
/// warnings/errors, not informational chatter.
const DEFAULT_CLIENT_FILTER: &str = "warn";

/// Resolve the directory file logs should be written to.
///
/// Precedence:
///   1. `FE_LSP_LOG_DIR` env var (explicit override)
///   2. `$XDG_CACHE_HOME/fe-language-server`
///   3. `$HOME/.cache/fe-language-server`
///
/// Returns `None` if no usable base path exists (e.g. neither env var set).
fn resolve_log_dir() -> Option<PathBuf> {
    if let Ok(dir) = std::env::var("FE_LSP_LOG_DIR") {
        return Some(PathBuf::from(dir));
    }
    let base = std::env::var("XDG_CACHE_HOME")
        .ok()
        .or_else(|| std::env::var("HOME").ok().map(|h| format!("{h}/.cache")))?;
    Some(PathBuf::from(base).join("fe-language-server"))
}

/// Resolve the full path of the file log for this process.
///
/// Precedence:
///   1. `FE_LSP_LOG_FILE` env var (absolute path override)
///   2. `<resolve_log_dir()>/<pid>.log`
///
/// Returns `None` if no path can be resolved.
pub fn default_log_file_path() -> Option<PathBuf> {
    if let Ok(path) = std::env::var("FE_LSP_LOG_FILE") {
        return Some(PathBuf::from(path));
    }
    let dir = resolve_log_dir()?;
    Some(dir.join(format!("{}.log", std::process::id())))
}

/// Path of the panics log for this process. Lives alongside the main log.
pub fn default_panic_file_path() -> Option<PathBuf> {
    let dir = resolve_log_dir()?;
    Some(dir.join(format!("panics-{}.log", std::process::id())))
}

/// Return the parent process's pid, if it can be determined.
///
/// On Linux, reads `/proc/self/status` and parses the `PPid:` field.
/// On other platforms, returns `None` (graceful degrade — we use this
/// only for diagnostic logging, not for correctness).
pub fn parent_pid() -> Option<u32> {
    #[cfg(target_os = "linux")]
    {
        let status = std::fs::read_to_string("/proc/self/status").ok()?;
        for line in status.lines() {
            if let Some(rest) = line.strip_prefix("PPid:") {
                return rest.trim().parse().ok();
            }
        }
        None
    }
    #[cfg(not(target_os = "linux"))]
    {
        None
    }
}

/// Emit a single structured log line at `info` with everything needed to
/// identify which process this is in a multi-instance forensic scenario.
///
/// Intended to be called exactly once, at server startup, before any other
/// work. The line includes:
///   * `pid` — our process id
///   * `parent_pid` — the spawning process (the editor, typically Zed)
///   * `argv` — our command-line arguments
///   * `cwd` — current working directory at startup (often but not always
///     the workspace root; the editor decides)
///   * `binary` — path to the running executable (via `/proc/self/exe` on
///     Linux, best-effort elsewhere)
///   * `log_file` — the file log path we chose, so grepping the log line
///     tells you where to `tail -F` the rest of the output
///
/// This is the line Grant would copy-paste when reporting "I've got three
/// `fe` processes running and I don't know which one Zed is actually
/// talking to." Every field is a join key.
pub fn log_startup_info() {
    let pid = std::process::id();
    let ppid = parent_pid();
    let argv: Vec<String> = std::env::args().collect();
    let cwd = std::env::current_dir()
        .map(|p| p.display().to_string())
        .unwrap_or_else(|e| format!("<error: {e}>"));
    let binary = std::env::current_exe()
        .map(|p| p.display().to_string())
        .unwrap_or_else(|e| format!("<error: {e}>"));
    let log_file = default_log_file_path()
        .map(|p| p.display().to_string())
        .unwrap_or_else(|| "<none>".to_owned());
    let version = env!("CARGO_PKG_VERSION");

    tracing::info!(
        target: "fe::lsp::startup",
        %pid,
        ppid = ?ppid,
        ?argv,
        %cwd,
        %binary,
        %log_file,
        %version,
        "fe-language-server starting"
    );
}

/// Open a log file for appending, creating the parent directory if needed.
fn open_log_file(path: &Path) -> std::io::Result<File> {
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    OpenOptions::new().create(true).append(true).open(path)
}

/// Build the default tracing subscriber for the language server and install
/// it as the thread-local default. The returned [`DefaultGuard`] must be
/// held for as long as the subscriber should remain active — dropping it
/// resets the thread-local dispatch.
pub fn setup_default_subscriber(client: ClientSocket) -> Option<DefaultGuard> {
    use tracing::subscriber::set_default;

    // Client layer: WARN+ only, forwarded to the LSP client as
    // `window/logMessage` notifications. This is what the editor sees.
    let client_filter = std::env::var("FE_LSP_CLIENT_LOG")
        .ok()
        .and_then(|s| EnvFilter::try_new(s).ok())
        .unwrap_or_else(|| EnvFilter::new(DEFAULT_CLIENT_FILTER));

    let client_layer = tracing_subscriber::fmt::layer()
        .with_ansi(false)
        .with_target(true)
        .with_writer(ClientSocketWriterMaker::new(client))
        .with_filter(client_filter);

    // File layer: DEFAULT_FILE_FILTER (or `FE_LSP_LOG`), written to the
    // per-process log file. If the file can't be opened, the layer is
    // silently omitted — we don't want to break the LSP because we can't
    // write logs.
    let file_filter = std::env::var("FE_LSP_LOG")
        .ok()
        .and_then(|s| EnvFilter::try_new(s).ok())
        .unwrap_or_else(|| EnvFilter::new(DEFAULT_FILE_FILTER));

    let file_layer: Option<_> = default_log_file_path()
        .and_then(|path| match open_log_file(&path) {
            Ok(file) => {
                // Write the path to stderr so a human watching the process
                // knows where to find the log. This is one line at startup,
                // not per-message.
                eprintln!("fe-language-server: log file at {}", path.display());
                Some(file)
            }
            Err(e) => {
                eprintln!(
                    "fe-language-server: failed to open log file at {}: {e}",
                    path.display()
                );
                None
            }
        })
        .map(|file| {
            let writer = BoxMakeWriter::new(Arc::new(file));
            tracing_subscriber::fmt::layer()
                .with_ansi(false)
                .with_target(true)
                .with_writer(writer)
                .with_filter(file_filter)
                .boxed()
        });

    let subscriber = Registry::default().with(client_layer).with(file_layer);

    Some(set_default(subscriber))
}

/// Returns a thunk that installs the subscriber when invoked. Used by the
/// act-locally `ActorBuilder::with_subscriber_init` hook so the actor
/// thread gets its own thread-local subscriber at startup.
pub fn init_fn(client: ClientSocket) -> impl FnOnce() -> Option<DefaultGuard> {
    move || setup_default_subscriber(client)
}

/// Install the process-wide panic hook. Idempotent: subsequent calls are
/// no-ops (guarded by `Once`).
///
/// On panic the hook writes a complete record to `panics-<pid>.log` via
/// `OpenOptions::append`. This write path does **not** go through the
/// tracing subscriber — it uses direct filesystem IO so it works during
/// shutdown when subscriber guards are being dropped. It also logs via
/// `tracing::error!` as a best-effort secondary path.
///
/// The record includes:
///   * panic payload + location + thread name
///   * [`crate::panic_context::format_stack`] — per-request metadata
///     pushed by [`crate::lsp_actor::service::LspActorService::call`]
///   * `std::backtrace::Backtrace::force_capture()` — Rust call stack
///
/// Note: rust-analyzer additionally captures `salsa::Backtrace` (the
/// active query stack), but our pinned `salsa = 0.20.0` does not
/// expose that API — it was added in a later version. Consider
/// adopting it on the next salsa bump.
pub fn setup_panic_hook() {
    static HOOK_INSTALLED: Once = Once::new();
    HOOK_INSTALLED.call_once(|| {
        let panic_file_path = default_panic_file_path();
        let previous_hook = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |panic_info| {
            // Preserve the default stderr panic output so `cargo test`
            // and `cargo run` users still see panics the familiar way.
            previous_hook(panic_info);

            let payload = panic_info.payload();
            let message = if let Some(s) = payload.downcast_ref::<&str>() {
                (*s).to_owned()
            } else if let Some(s) = payload.downcast_ref::<String>() {
                s.clone()
            } else {
                "<non-string panic payload>".to_owned()
            };

            let location = panic_info
                .location()
                .map(|l| format!("{}:{}:{}", l.file(), l.line(), l.column()))
                .unwrap_or_else(|| "<unknown>".to_owned());

            let backtrace = Backtrace::force_capture();
            let context_stack = crate::panic_context::format_stack();
            let thread_name = std::thread::current()
                .name()
                .unwrap_or("<unnamed>")
                .to_owned();

            if let Some(ref path) = panic_file_path {
                if let Some(parent) = path.parent() {
                    let _ = std::fs::create_dir_all(parent);
                }
                if let Ok(mut file) = OpenOptions::new().create(true).append(true).open(path) {
                    let _ = writeln!(file, "==== fe-lsp panic ====");
                    let _ = writeln!(file, "pid:      {}", std::process::id());
                    let _ = writeln!(file, "time:     {:?}", std::time::SystemTime::now());
                    let _ = writeln!(file, "thread:   {thread_name}");
                    let _ = writeln!(file, "location: {location}");
                    let _ = writeln!(file, "message:  {message}");
                    if !context_stack.is_empty() {
                        let _ = writeln!(file, "\n---- request context stack ----");
                        let _ = write!(file, "{context_stack}");
                    }
                    let _ = writeln!(file, "\n---- rust backtrace ----");
                    let _ = writeln!(file, "{backtrace}");
                    let _ = writeln!(file, "===================================\n");
                    let _ = file.sync_all();
                }
            }

            // Secondary path via tracing. May not reach the client / file
            // subscriber if we're already mid-shutdown, but worth trying.
            tracing::error!(
                target: "fe::lsp::panic",
                location = %location,
                thread = %thread_name,
                "panic captured: {message}"
            );
        }));
    });
}

pub(crate) struct ClientSocketWriterMaker {
    pub(crate) client_socket: Arc<ClientSocket>,
}

impl ClientSocketWriterMaker {
    pub fn new(client_socket: ClientSocket) -> Self {
        ClientSocketWriterMaker {
            client_socket: Arc::new(client_socket),
        }
    }
}

pub(crate) struct ClientSocketWriter {
    client_socket: Arc<ClientSocket>,
    typ: MessageType,
}

impl std::io::Write for ClientSocketWriter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let message = String::from_utf8_lossy(buf).to_string();
        let params = LogMessageParams {
            typ: self.typ,
            message,
        };

        let mut client_socket = self.client_socket.as_ref();
        _ = client_socket.log_message(params);
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

impl<'a> MakeWriter<'a> for ClientSocketWriterMaker {
    type Writer = ClientSocketWriter;

    fn make_writer(&'a self) -> Self::Writer {
        ClientSocketWriter {
            client_socket: self.client_socket.clone(),
            typ: MessageType::LOG,
        }
    }

    fn make_writer_for(&'a self, meta: &Metadata<'_>) -> Self::Writer {
        let typ = match *meta.level() {
            Level::ERROR => MessageType::ERROR,
            Level::WARN => MessageType::WARNING,
            Level::INFO => MessageType::INFO,
            Level::DEBUG => MessageType::LOG,
            Level::TRACE => MessageType::LOG,
        };

        ClientSocketWriter {
            client_socket: self.client_socket.clone(),
            typ,
        }
    }
}
