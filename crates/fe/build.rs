use std::{env, process::Command};

fn main() {
    println!("cargo:rerun-if-changed=tests/fixtures");
    println!("cargo:rerun-if-changed=tests/doc_fixtures");
    println!("cargo:rerun-if-env-changed=FE_GIT_HASH");

    if let Ok(override_hash) = env::var("FE_GIT_HASH")
        && !override_hash.trim().is_empty()
    {
        println!("cargo:rustc-env=FE_GIT_HASH={}", override_hash.trim());
        return;
    }

    let Some(hash) = git_output(&["rev-parse", "--short", "HEAD"]) else {
        return;
    };

    emit_git_rerun_paths();
    println!("cargo:rustc-env=FE_GIT_HASH={hash}");
}

fn emit_git_rerun_paths() {
    for path in [
        git_output(&["rev-parse", "--git-path", "HEAD"]),
        git_output(&["symbolic-ref", "-q", "HEAD"])
            .and_then(|branch| git_output(&["rev-parse", "--git-path", branch.as_str()])),
        git_output(&["rev-parse", "--git-path", "packed-refs"]),
    ]
    .into_iter()
    .flatten()
    {
        println!("cargo:rerun-if-changed={path}");
    }
}

fn git_output(args: &[&str]) -> Option<String> {
    let output = Command::new("git").args(args).output().ok()?;
    if !output.status.success() {
        return None;
    }

    let output = String::from_utf8_lossy(&output.stdout).trim().to_string();
    (!output.is_empty()).then_some(output)
}
