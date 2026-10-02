use std::{fs, path::Path, process::Command};

use tempfile::tempdir;

pub(crate) fn link_executable(inputs: &[&Path], executable: &Path) -> Result<(), String> {
    // The C compiler driver supplies the platform startup objects and linker
    // configuration for both ordinary executables and test dispatchers.
    let runtime_dir = tempdir().map_err(|err| format!("native runtime directory: {err}"))?;
    let runtime = runtime_dir.path().join("runtime.c");
    fs::write(&runtime, include_str!("native/runtime.c"))
        .map_err(|err| format!("write native runtime: {err}"))?;
    let output = Command::new("cc")
        .args(["-std=c11", "-O2", "-fno-lto"])
        .arg(&runtime)
        .args(inputs)
        .arg("-o")
        .arg(executable)
        .output()
        .map_err(|err| format!("failed to run host linker `cc`: {err}"))?;
    if !output.status.success() {
        return Err(format!(
            "host linker failed\nstdout:\n{}\nstderr:\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    Ok(())
}
