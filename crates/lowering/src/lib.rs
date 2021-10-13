//! Fe Lowering.

use fe_analyzer::namespace::items::{IngotId, ModuleId};

mod context;
pub mod db;
mod mappers;
mod names;
mod utils;

pub use db::{LoweringDb, TestDb};

/// Lowers the Fe source AST to a Fe HIR AST.
pub fn lower_module(db: &dyn LoweringDb, module_id: ModuleId) -> ModuleId {
    db.lowered_module(module_id)
}

pub fn lower_ingot(db: &dyn LoweringDb, ingot_id: IngotId) -> IngotId {
    db.lowered_ingot(ingot_id)
}
