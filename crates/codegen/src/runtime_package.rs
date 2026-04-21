use driver::DriverDataBase;
use mir2::RuntimePackage;

pub(crate) fn ensure_runtime_package_has_roots(
    db: &DriverDataBase,
    package: &RuntimePackage<'_>,
    artifact: &str,
) -> Result<(), mir2::LowerError> {
    if package.root_objects(db).is_empty() {
        return Err(mir2::LowerError::Unsupported(format!(
            "runtime package has no root objects; refusing to emit target-only {artifact}"
        )));
    }
    Ok(())
}
