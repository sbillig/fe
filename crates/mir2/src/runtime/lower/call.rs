use cranelift_entity::EntityRef;
use hir::analysis::semantic::{
    NExpr, NSStmtKind, borrowck::normalize_semantic_body, ctfe::canonicalize_semantic_consts,
};

use crate::{db::MirDb, instance::RuntimeInstance, runtime::RuntimeCallEdge};

use super::class::{
    RuntimeSemanticCallContext, infer_local_runtime_state, runtime_callee_for_semantic_call,
    runtime_param_locals,
};

pub fn collect_runtime_calls<'db>(
    db: &'db dyn MirDb,
    instance: RuntimeInstance<'db>,
) -> Vec<RuntimeCallEdge<'db>> {
    let Some(semantic) = instance.key(db).semantic(db) else {
        let body = instance.body(db);
        return body
            .blocks
            .iter()
            .flat_map(|block| {
                block.stmts.iter().filter_map(|stmt| match stmt {
                    crate::runtime::RStmt::Assign {
                        expr: crate::runtime::RExpr::Call { callee, .. },
                        ..
                    } => Some(RuntimeCallEdge { callee: *callee }),
                    crate::runtime::RStmt::Assign { .. }
                    | crate::runtime::RStmt::Store { .. }
                    | crate::runtime::RStmt::CopyInto { .. }
                    | crate::runtime::RStmt::EnumSetTag { .. }
                    | crate::runtime::RStmt::EnumWriteVariant { .. } => None,
                })
            })
            .chain(
                body.blocks
                    .iter()
                    .filter_map(|block| match &block.terminator {
                        crate::runtime::RTerminator::TerminalCall { callee, .. } => {
                            Some(RuntimeCallEdge { callee: *callee })
                        }
                        crate::runtime::RTerminator::Goto(_)
                        | crate::runtime::RTerminator::Branch { .. }
                        | crate::runtime::RTerminator::SwitchScalar { .. }
                        | crate::runtime::RTerminator::MatchEnumTag { .. }
                        | crate::runtime::RTerminator::Return(_)
                        | crate::runtime::RTerminator::ReturnData { .. }
                        | crate::runtime::RTerminator::Revert { .. }
                        | crate::runtime::RTerminator::SelfDestruct { .. }
                        | crate::runtime::RTerminator::Trap
                        | crate::runtime::RTerminator::Stop => None,
                    }),
            )
            .collect();
    };

    let raw_body = canonicalize_semantic_consts(db, semantic);
    let body = normalize_semantic_body(db, semantic).unwrap_or_else(|err| {
        panic!(
            "semantic normalization failed while collecting runtime calls for {:?}: {err:?}",
            semantic.key(db)
        )
    });
    let typed_body = semantic.key(db).instantiate_typed_body(db);
    let carriers = infer_local_runtime_state(
        db,
        &raw_body,
        &body,
        instance.key(db).params(db),
        &runtime_param_locals(db, semantic, instance.key(db).params(db)),
        typed_body.body().map(|body| body.scope()),
        typed_body.assumptions(),
    );
    let carrier_classes = carriers
        .iter()
        .map(|local| local.carrier.clone())
        .collect::<Vec<_>>();

    body.blocks
        .iter()
        .flat_map(|block| {
            block.stmts.iter().filter_map(|stmt| {
                let NSStmtKind::Assign { dst, expr } = &stmt.kind else {
                    return None;
                };
                let NExpr::Call {
                    callee,
                    args,
                    effect_args,
                } = expr
                else {
                    return None;
                };
                runtime_callee_for_semantic_call(
                    db,
                    RuntimeSemanticCallContext {
                        caller: semantic,
                        raw_body: &raw_body,
                        body: &body,
                        carriers: &carrier_classes,
                        result_ty: body.locals[dst.index()].ty,
                    },
                    *callee,
                    args,
                    effect_args,
                )
                .map(|callee| RuntimeCallEdge { callee })
            })
        })
        .collect()
}
