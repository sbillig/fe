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
