mod doc;
mod emitter;
mod errors;
mod legalize;
mod state;

use common::ingot::Ingot;
use driver::DriverDataBase;
use hir::hir_def::{HirIngot, ManualContractRootAttr, TopLevelMod};
use mir2::{RuntimePackage, build_runtime_package, build_test_runtime_package};

use crate::{
    TargetDataLayout, TestModuleOutput, runtime_package::ensure_runtime_package_has_roots,
};

pub use errors::YulError;

#[derive(Debug)]
pub enum EmitModuleError {
    RuntimeLower(mir2::LowerError),
    Yul(YulError),
}

impl std::fmt::Display for EmitModuleError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EmitModuleError::RuntimeLower(err) => write!(f, "{err}"),
            EmitModuleError::Yul(err) => write!(f, "{err}"),
        }
    }
}

impl std::error::Error for EmitModuleError {}

impl From<mir2::LowerError> for EmitModuleError {
    fn from(err: mir2::LowerError) -> Self {
        EmitModuleError::RuntimeLower(err)
    }
}

impl From<YulError> for EmitModuleError {
    fn from(err: YulError) -> Self {
        EmitModuleError::Yul(err)
    }
}

pub fn emit_runtime_package_yul<'db>(
    db: &'db DriverDataBase,
    package: &RuntimePackage<'db>,
    layout: TargetDataLayout,
) -> Result<String, EmitModuleError> {
    ensure_runtime_package_has_roots(db, package, "Yul")?;
    let package = legalize::legalize_runtime_package(db, package, layout)?;
    emitter::emit_runtime_package_yul(db, &package).map_err(Into::into)
}

pub fn emit_runtime_package_object_yul<'db>(
    db: &'db DriverDataBase,
    package: &RuntimePackage<'db>,
    layout: TargetDataLayout,
    object_name: &str,
) -> Result<String, EmitModuleError> {
    let package = legalize::legalize_runtime_package(db, package, layout)?;
    emitter::emit_runtime_package_object_yul(db, &package, object_name).map_err(Into::into)
}

pub fn emit_test_runtime_package_yul<'db>(
    db: &'db DriverDataBase,
    package: &RuntimePackage<'db>,
    layout: TargetDataLayout,
    filter: Option<&str>,
) -> Result<TestModuleOutput, EmitModuleError> {
    let package = legalize::legalize_runtime_package(db, package, layout)?;
    emitter::emit_test_runtime_package_yul(db, &package, filter).map_err(Into::into)
}

pub fn emit_module_yul(
    db: &DriverDataBase,
    top_mod: TopLevelMod<'_>,
) -> Result<String, EmitModuleError> {
    emit_module_yul_with_layout(db, top_mod, crate::EVM_LAYOUT)
}

pub fn emit_module_yul_with_layout(
    db: &DriverDataBase,
    top_mod: TopLevelMod<'_>,
    layout: TargetDataLayout,
) -> Result<String, EmitModuleError> {
    let package = build_runtime_package(db, top_mod)?;
    emit_runtime_package_yul(db, &package, layout)
}

pub fn emit_module_object_yul(
    db: &DriverDataBase,
    top_mod: TopLevelMod<'_>,
    object_name: &str,
) -> Result<String, EmitModuleError> {
    emit_module_object_yul_with_layout(db, top_mod, object_name, crate::EVM_LAYOUT)
}

pub fn emit_module_object_yul_with_layout(
    db: &DriverDataBase,
    top_mod: TopLevelMod<'_>,
    object_name: &str,
    layout: TargetDataLayout,
) -> Result<String, EmitModuleError> {
    let package = build_runtime_package(db, top_mod)?;
    emit_runtime_package_object_yul(db, &package, layout, object_name)
}

pub fn emit_ingot_yul(db: &DriverDataBase, ingot: Ingot<'_>) -> Result<String, EmitModuleError> {
    emit_ingot_yul_with_layout(db, ingot, crate::EVM_LAYOUT)
}

pub fn emit_ingot_yul_with_layout(
    db: &DriverDataBase,
    ingot: Ingot<'_>,
    layout: TargetDataLayout,
) -> Result<String, EmitModuleError> {
    let mut modules = Vec::new();
    for &top_mod in ingot.all_modules(db) {
        modules.push(emit_module_yul_with_layout(db, top_mod, layout)?);
    }
    Ok(modules.join("\n\n"))
}

pub fn emit_ingot_object_yul(
    db: &DriverDataBase,
    ingot: Ingot<'_>,
    object_name: &str,
) -> Result<String, EmitModuleError> {
    emit_ingot_object_yul_with_layout(db, ingot, object_name, crate::EVM_LAYOUT)
}

pub fn emit_ingot_object_yul_with_layout(
    db: &DriverDataBase,
    ingot: Ingot<'_>,
    object_name: &str,
    layout: TargetDataLayout,
) -> Result<String, EmitModuleError> {
    let mut found = None;
    let root_top_mod = ingot.root_file(db).ok().map(|file| db.top_mod(file));
    let mut top_mods = root_top_mod.into_iter().collect::<Vec<_>>();
    let mut other_top_mods = ingot
        .all_modules(db)
        .iter()
        .copied()
        .filter(|top_mod| Some(*top_mod) != root_top_mod)
        .collect::<Vec<_>>();
    other_top_mods.sort_by(|left, right| left.name(db).cmp(&right.name(db)));
    top_mods.extend(other_top_mods);
    for top_mod in top_mods {
        if Some(top_mod) != root_top_mod
            && !top_mod_declares_runtime_object(db, top_mod, object_name)
        {
            continue;
        }
        let package = build_runtime_package(db, top_mod)?;
        if !package
            .root_objects(db)
            .iter()
            .any(|object| object.name(db) == object_name)
        {
            continue;
        }
        if found.is_some() {
            return Err(YulError::InvalidYulPackage(format!(
                "multiple root objects named `{object_name}` in ingot"
            ))
            .into());
        }
        found = Some(emit_runtime_package_object_yul(
            db,
            &package,
            layout,
            object_name,
        )?);
    }
    found.ok_or_else(|| {
        YulError::InvalidYulPackage(format!("missing root object `{object_name}` in ingot")).into()
    })
}

fn top_mod_declares_runtime_object(
    db: &DriverDataBase,
    top_mod: TopLevelMod<'_>,
    object_name: &str,
) -> bool {
    top_mod.all_contracts(db).iter().any(|contract| {
        contract
            .name(db)
            .to_opt()
            .is_some_and(|name| yul_object_name(name.data(db)) == object_name)
    }) || top_mod.all_funcs(db).iter().any(|func| {
        if func.top_mod(db) != top_mod {
            return false;
        }
        match func.manual_contract_root_attr(db) {
            Some(ManualContractRootAttr::Init { contract_name })
            | Some(ManualContractRootAttr::Runtime { contract_name }) => {
                yul_object_name(contract_name.data(db)) == object_name
            }
            Some(ManualContractRootAttr::Error(_)) | None => false,
        }
    })
}

fn yul_object_name(value: &str) -> String {
    let sanitized: String = value
        .chars()
        .map(|ch| if ch.is_ascii_alphanumeric() { ch } else { '_' })
        .collect();
    if sanitized.is_empty() {
        "object".to_string()
    } else {
        sanitized
    }
}

pub fn emit_test_module_yul(
    db: &DriverDataBase,
    top_mod: TopLevelMod<'_>,
    filter: Option<&str>,
) -> Result<TestModuleOutput, EmitModuleError> {
    emit_test_module_yul_with_layout(db, top_mod, filter, crate::EVM_LAYOUT)
}

pub fn emit_test_module_yul_with_layout(
    db: &DriverDataBase,
    top_mod: TopLevelMod<'_>,
    filter: Option<&str>,
    layout: TargetDataLayout,
) -> Result<TestModuleOutput, EmitModuleError> {
    let package = build_test_runtime_package(db, top_mod, filter)?;
    emit_test_runtime_package_yul(db, &package, layout, filter)
}

pub fn emit_test_ingot_yul(
    db: &DriverDataBase,
    ingot: Ingot<'_>,
    filter: Option<&str>,
) -> Result<TestModuleOutput, EmitModuleError> {
    emit_test_ingot_yul_with_layout(db, ingot, filter, crate::EVM_LAYOUT)
}

pub fn emit_test_ingot_yul_with_layout(
    db: &DriverDataBase,
    ingot: Ingot<'_>,
    filter: Option<&str>,
    layout: TargetDataLayout,
) -> Result<TestModuleOutput, EmitModuleError> {
    let mut top_mods = ingot.all_modules(db).to_vec();
    top_mods.sort_by(|left, right| left.name(db).cmp(&right.name(db)));

    let mut output = TestModuleOutput { tests: Vec::new() };
    for top_mod in top_mods {
        output.extend(emit_test_module_yul_with_layout(
            db, top_mod, filter, layout,
        )?);
    }
    output.sort_tests();
    Ok(output)
}
