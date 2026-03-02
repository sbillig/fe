use camino::{Utf8Path, Utf8PathBuf};
use std::io;
use std::path::{Path, PathBuf};
use typed_path::{Utf8WindowsPath, Utf8WindowsPrefix};
use url::Url;

fn non_utf8_path_error(path: PathBuf) -> io::Error {
    io::Error::new(
        io::ErrorKind::InvalidData,
        format!("path is not UTF-8: {}", path.display()),
    )
}

pub fn normalize_slashes(raw: &str) -> String {
    if !raw.contains('\\') {
        return raw.to_string();
    }

    let windows_path = Utf8WindowsPath::new(raw);
    let unix_path = windows_path.with_unix_encoding();
    let unix = unix_path.as_str();
    let components = windows_path.components();

    match components.prefix_kind() {
        Some(Utf8WindowsPrefix::Disk(drive)) => {
            if windows_path.has_root() {
                return format!("{drive}:{unix}");
            }

            let drive_prefix = format!("{drive}:/");
            if let Some(path) = unix.strip_prefix(&drive_prefix) {
                return format!("{drive}:{path}");
            }

            unix.to_string()
        }
        Some(prefix) => {
            let normalized_prefix = normalize_windows_prefix(prefix);
            if unix == "/" && !components.has_physical_root() {
                normalized_prefix
            } else {
                format!("{normalized_prefix}{unix}")
            }
        }
        None => unix.to_string(),
    }
}

pub fn glob_pattern(path: &Path) -> String {
    normalize_slashes(path.to_string_lossy().as_ref())
}

fn normalize_windows_prefix(prefix: Utf8WindowsPrefix<'_>) -> String {
    match prefix {
        Utf8WindowsPrefix::Verbatim(component) => format!("//?/{component}"),
        Utf8WindowsPrefix::VerbatimUNC(server, share) => {
            format!("//?/UNC/{server}/{share}")
        }
        Utf8WindowsPrefix::VerbatimDisk(drive) => format!("//?/{drive}:"),
        Utf8WindowsPrefix::DeviceNS(device) => format!("//./{device}"),
        Utf8WindowsPrefix::UNC(server, share) => format!("//{server}/{share}"),
        Utf8WindowsPrefix::Disk(drive) => format!("{drive}:"),
    }
}

pub fn file_url_to_utf8_path(url: &Url) -> Option<Utf8PathBuf> {
    #[cfg(not(target_arch = "wasm32"))]
    let path = url.to_file_path().ok()?;
    #[cfg(target_arch = "wasm32")]
    let path = {
        if url.scheme() != "file" {
            return None;
        }
        PathBuf::from(url.path())
    };
    Utf8PathBuf::from_path_buf(path).ok()
}

pub fn canonicalize_utf8(path: &Path) -> io::Result<Utf8PathBuf> {
    let canonical = path.canonicalize()?;
    Utf8PathBuf::from_path_buf(canonical).map_err(non_utf8_path_error)
}

pub fn absolute_utf8(path: &Utf8Path) -> io::Result<Utf8PathBuf> {
    if path.is_absolute() {
        return Ok(path.to_path_buf());
    }

    let cwd = std::env::current_dir()?;
    let cwd = Utf8PathBuf::from_path_buf(cwd).map_err(non_utf8_path_error)?;
    Ok(cwd.join(path))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalizes_slashes() {
        assert_eq!(normalize_slashes(r"a\b\c"), "a/b/c");
    }

    #[test]
    fn normalizes_disk_paths() {
        assert_eq!(normalize_slashes(r"C:\a\b\c"), "C:/a/b/c");
        assert_eq!(normalize_slashes(r"C:a\b\c"), "C:a/b/c");
    }

    #[test]
    fn normalizes_unc_paths() {
        assert_eq!(
            normalize_slashes(r"\\server\share\path\to\file"),
            "//server/share/path/to/file"
        );
    }

    #[test]
    fn converts_file_urls_to_utf8_paths() {
        let cwd = std::env::current_dir().expect("current dir");
        let url = Url::from_directory_path(&cwd).expect("directory url");
        let path = file_url_to_utf8_path(&url).expect("file url to path");
        assert_eq!(path, Utf8PathBuf::from_path_buf(cwd).unwrap());
    }

    #[test]
    fn builds_glob_patterns_with_forward_slashes() {
        let pattern = glob_pattern(Path::new(r"a\b\**\fe.toml"));
        assert_eq!(pattern, "a/b/**/fe.toml");
    }

    #[test]
    fn makes_relative_paths_absolute() {
        let absolute = absolute_utf8(Utf8Path::new("src")).expect("absolute path");
        assert!(absolute.is_absolute());
    }

    #[test]
    fn canonicalizes_paths_to_utf8() {
        let cwd = std::env::current_dir().expect("current dir");
        let path = canonicalize_utf8(&cwd).expect("canonicalize");
        let expected = cwd.canonicalize().expect("canonicalize cwd");
        let expected = Utf8PathBuf::from_path_buf(expected).unwrap();
        assert_eq!(path, expected);
    }
}
