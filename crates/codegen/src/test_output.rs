use driver::DriverDataBase;
use hir::{
    HirDb,
    analysis::ty::ty_check::BodyOwner,
    hir_def::{ItemKind, LitKind, attr::AttrArgValue},
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TestMetadata {
    pub display_name: String,
    pub hir_name: String,
    pub symbol_name: String,
    pub object_name: String,
    pub yul: String,
    pub bytecode: Vec<u8>,
    pub sonatina_observability_json: Option<String>,
    pub value_param_count: usize,
    pub effect_param_count: usize,
    pub init_bytecode: Vec<u8>,
    pub expected_revert: Option<ExpectedRevert>,
    pub initial_balance: Option<Vec<u8>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExpectedRevert {
    Any,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TestModuleOutput {
    pub tests: Vec<TestMetadata>,
}

impl TestModuleOutput {
    pub(crate) fn extend(&mut self, other: Self) {
        self.tests.extend(other.tests);
    }

    pub(crate) fn sort_tests(&mut self) {
        self.tests.sort_by(|left, right| {
            left.display_name
                .cmp(&right.display_name)
                .then_with(|| left.hir_name.cmp(&right.hir_name))
                .then_with(|| left.symbol_name.cmp(&right.symbol_name))
                .then_with(|| left.object_name.cmp(&right.object_name))
        });
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct TestRootMetadata {
    pub hir_name: String,
    pub display_name: String,
    pub expected_revert: Option<ExpectedRevert>,
    pub initial_balance: Option<Vec<u8>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum TestRootMetadataError {
    InvalidPackage(String),
    Unsupported(String),
}

pub(crate) fn runtime_test_root_metadata<'db>(
    db: &'db DriverDataBase,
    owner: &mir2::RuntimeFunctionOwner<'db>,
    section_name: &mir2::RuntimeSectionName,
) -> Result<TestRootMetadata, TestRootMetadataError> {
    let mir2::RuntimeSectionName::Test(hir_name) = section_name else {
        return Err(TestRootMetadataError::InvalidPackage(format!(
            "non-test section `{section_name:?}` used as a test root"
        )));
    };
    let mir2::RuntimeFunctionOwner::Synthetic(mir2::RuntimeSyntheticSpec::TestRoot {
        callee, ..
    }) = owner
    else {
        return Err(TestRootMetadataError::InvalidPackage(format!(
            "test section `{hir_name}` does not use a TestRoot wrapper"
        )));
    };
    let Some(semantic) = callee.key(db).semantic(db) else {
        return Err(TestRootMetadataError::InvalidPackage(format!(
            "test section `{hir_name}` does not target a semantic instance"
        )));
    };
    let BodyOwner::Func(func) = semantic.key(db).owner(db) else {
        return Err(TestRootMetadataError::InvalidPackage(format!(
            "test section `{hir_name}` does not target a function body"
        )));
    };
    let attrs = ItemKind::from(func).attrs(db).ok_or_else(|| {
        TestRootMetadataError::InvalidPackage(format!("test function `{hir_name}` has no attrs"))
    })?;
    let test_attr = attrs.get_attr(db, "test").ok_or_else(|| {
        TestRootMetadataError::InvalidPackage(format!(
            "test function `{hir_name}` is missing #[test]"
        ))
    })?;
    let expected_revert = test_attr
        .has_arg(db, "should_revert")
        .then_some(ExpectedRevert::Any);
    let initial_balance = parse_test_balance_arg(db, hir_name, test_attr)?;
    Ok(TestRootMetadata {
        hir_name: hir_name.clone(),
        display_name: hir_name.clone(),
        expected_revert,
        initial_balance,
    })
}

fn parse_test_balance_arg<'db>(
    db: &'db dyn HirDb,
    test_name: &str,
    test_attr: &hir::hir_def::attr::NormalAttr<'db>,
) -> Result<Option<Vec<u8>>, TestRootMetadataError> {
    for arg in &test_attr.args {
        if arg.key_str(db) != Some("balance") {
            continue;
        }
        let Some(value) = arg.value.as_ref() else {
            return Err(TestRootMetadataError::Unsupported(format!(
                "invalid #[test] function `{test_name}`: #[test(balance = ...)] expects an integer literal"
            )));
        };
        let AttrArgValue::Lit(LitKind::Int(int_id)) = value else {
            return Err(TestRootMetadataError::Unsupported(format!(
                "invalid #[test] function `{test_name}`: #[test(balance = ...)] expects an integer literal"
            )));
        };
        let balance = int_id.data(db).clone();
        if balance.to_bytes_be().len() > 32 {
            return Err(TestRootMetadataError::Unsupported(format!(
                "invalid #[test] function `{test_name}`: #[test(balance = ...)] must fit in u256"
            )));
        }
        return Ok(Some(balance.to_bytes_be()));
    }

    Ok(None)
}
