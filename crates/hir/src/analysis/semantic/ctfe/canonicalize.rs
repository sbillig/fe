use crate::analysis::{
    HirAnalysisDb,
    semantic::{SConst, SExpr, SStmt, SemanticBody, instance::SemanticInstance},
};

use super::eval_const_ref;

#[salsa::tracked]
pub fn canonicalize_semantic_consts<'db>(
    db: &'db dyn HirAnalysisDb,
    instance: SemanticInstance<'db>,
) -> SemanticBody<'db> {
    let mut body = instance.body(db).clone();
    for block in &mut body.blocks {
        for stmt in &mut block.stmts {
            canonicalize_stmt(db, stmt);
        }
    }
    body
}

fn canonicalize_stmt<'db>(db: &'db dyn HirAnalysisDb, stmt: &mut SStmt<'db>) {
    match stmt {
        SStmt::Assign { expr, .. } => canonicalize_expr(db, expr),
        SStmt::Store { .. } => {}
    }
}

fn canonicalize_expr<'db>(db: &'db dyn HirAnalysisDb, expr: &mut SExpr<'db>) {
    match expr {
        SExpr::Const(SConst::Ref(cref)) => {
            let value = eval_const_ref(db, *cref)
                .unwrap_or_else(|err| panic!("CTFE failed for {cref:?}: {err:?}"));
            *expr = SExpr::Const(SConst::Value(value));
        }
        SExpr::Use(_)
        | SExpr::Const(SConst::Value(_))
        | SExpr::Unary { .. }
        | SExpr::Binary { .. }
        | SExpr::Cast { .. }
        | SExpr::AggregateMake { .. }
        | SExpr::EnumMake { .. }
        | SExpr::Field { .. }
        | SExpr::Index { .. }
        | SExpr::Borrow { .. }
        | SExpr::GetEnumTag { .. }
        | SExpr::IsEnumVariant { .. }
        | SExpr::ExtractEnumField { .. }
        | SExpr::Call { .. } => {}
    }
}
