use rustc_hash::FxHashSet;

use crate::{
    analysis::{
        HirAnalysisDb,
        semantic::{
            SBlockId, SExpr, SStmt, STerminator, SemanticBody, SemanticCalleeRef,
            lower::{expr_lowers_to_semantic_call, lower_to_smir},
            verify_semantic_body,
        },
        ty::{
            corelib::resolve_lib_func_path,
            ty_check::{BodyOwner, TypedBody},
        },
    },
    hir_def::Partial,
};

use super::{
    GenericSubst, ImplEnv, instantiate_typed_body, semantic_callee_key, typed_body_template,
};

#[salsa::interned]
#[derive(Debug)]
pub struct SemanticInstanceKey<'db> {
    pub owner: BodyOwner<'db>,
    pub subst: GenericSubst<'db>,
    pub impl_env: ImplEnv<'db>,
}

impl<'db> SemanticInstanceKey<'db> {
    pub fn instantiate_typed_body(self, db: &'db dyn HirAnalysisDb) -> TypedBody<'db> {
        instantiate_typed_body(db, typed_body_template(db, self.owner(db)), self.subst(db))
    }
}

#[salsa::tracked]
#[derive(Debug)]
pub struct SemanticInstance<'db> {
    pub key: SemanticInstanceKey<'db>,
}

#[salsa::tracked]
impl<'db> SemanticInstance<'db> {
    #[salsa::tracked]
    pub fn body(self, db: &'db dyn HirAnalysisDb) -> SemanticBody<'db> {
        lower_semantic_body(db, self)
    }

    #[salsa::tracked(return_ref)]
    pub fn callees(self, db: &'db dyn HirAnalysisDb) -> Vec<SemanticCalleeRef<'db>> {
        collect_semantic_callees(db, self)
    }
}

#[salsa::tracked(
    cycle_fn=semantic_may_return_normally_cycle_recover,
    cycle_initial=semantic_may_return_normally_cycle_initial
)]
pub fn semantic_may_return_normally<'db>(
    db: &'db dyn HirAnalysisDb,
    instance: SemanticInstance<'db>,
) -> bool {
    if semantic_is_nonreturning_builtin(db, instance) {
        return false;
    }

    let body = instance.body(db);
    if body.blocks.is_empty() {
        return true;
    }

    let mut pending = vec![SBlockId::from_u32(0)];
    let mut visited = FxHashSet::default();
    while let Some(block_id) = pending.pop() {
        if !visited.insert(block_id) {
            continue;
        }
        let Some(block) = body.block(block_id) else {
            continue;
        };
        let mut terminated_in_stmt = false;
        for stmt in &block.stmts {
            let SStmt::Assign {
                expr: SExpr::Call { callee, .. },
                ..
            } = stmt
            else {
                continue;
            };
            if !semantic_may_return_normally(db, SemanticInstance::new(db, callee.key)) {
                terminated_in_stmt = true;
                break;
            }
        }
        if terminated_in_stmt {
            continue;
        }

        match &block.terminator {
            STerminator::Return(_) => return true,
            STerminator::Goto(next) => pending.push(*next),
            STerminator::Branch {
                then_bb, else_bb, ..
            } => {
                pending.push(*then_bb);
                pending.push(*else_bb);
            }
            STerminator::MatchEnum { cases, default, .. } => {
                pending.extend(cases.iter().map(|(_, block)| *block));
                if let Some(default) = default {
                    pending.push(*default);
                }
            }
        }
    }

    false
}

#[salsa::tracked]
pub fn get_or_build_semantic_instance<'db>(
    db: &'db dyn HirAnalysisDb,
    key: SemanticInstanceKey<'db>,
) -> SemanticInstance<'db> {
    let instance = SemanticInstance::new(db, key);
    for callee in instance.callees(db) {
        get_or_build_semantic_instance(db, callee.key);
    }
    instance
}

fn lower_semantic_body<'db>(
    db: &'db dyn HirAnalysisDb,
    instance: SemanticInstance<'db>,
) -> SemanticBody<'db> {
    let key = instance.key(db);
    let typed_body = key.instantiate_typed_body(db);
    let body = lower_to_smir(db, instance, key.owner(db), typed_body);
    verify_semantic_body(&body).expect("invalid semantic MIR");
    body
}

fn collect_semantic_callees<'db>(
    db: &'db dyn HirAnalysisDb,
    instance: SemanticInstance<'db>,
) -> Vec<SemanticCalleeRef<'db>> {
    let key = instance.key(db);
    let typed_body = key.instantiate_typed_body(db);
    let Some(body) = typed_body.body() else {
        return Vec::new();
    };

    let mut seen = FxHashSet::default();
    let mut callees = Vec::new();
    for (expr_id, expr) in body.exprs(db).iter() {
        let Partial::Present(expr) = expr else {
            continue;
        };
        if !expr_lowers_to_semantic_call(db, &typed_body, body, expr_id, expr) {
            continue;
        }

        let Some(callable) = typed_body.callable_expr(expr_id) else {
            continue;
        };
        let Some(callee_key) = semantic_callee_key(db, key, &typed_body, callable) else {
            continue;
        };

        if seen.insert(callee_key) {
            callees.push(SemanticCalleeRef { key: callee_key });
        }
    }

    callees
}

fn semantic_is_nonreturning_builtin<'db>(
    db: &'db dyn HirAnalysisDb,
    instance: SemanticInstance<'db>,
) -> bool {
    let BodyOwner::Func(func) = instance.key(db).owner(db) else {
        return false;
    };
    let scope = func.scope();

    resolve_lib_func_path(db, scope, "std::evm::ops::return_data")
        .is_some_and(|builtin| builtin == func)
        || resolve_lib_func_path(db, scope, "std::evm::ops::revert")
            .is_some_and(|builtin| builtin == func)
        || resolve_lib_func_path(db, scope, "std::evm::ops::selfdestruct")
            .is_some_and(|builtin| builtin == func)
        || resolve_lib_func_path(db, scope, "std::evm::ops::stop")
            .is_some_and(|builtin| builtin == func)
        || resolve_lib_func_path(db, scope, "core::panic").is_some_and(|builtin| builtin == func)
        || resolve_lib_func_path(db, scope, "core::todo").is_some_and(|builtin| builtin == func)
        || resolve_lib_func_path(db, scope, "core::panic_with_value")
            .is_some_and(|builtin| builtin == func)
}

fn semantic_may_return_normally_cycle_initial<'db>(
    _db: &'db dyn HirAnalysisDb,
    _instance: SemanticInstance<'db>,
) -> bool {
    true
}

fn semantic_may_return_normally_cycle_recover<'db>(
    _db: &'db dyn HirAnalysisDb,
    _value: &bool,
    _count: u32,
    _instance: SemanticInstance<'db>,
) -> salsa::CycleRecoveryAction<bool> {
    salsa::CycleRecoveryAction::Iterate
}
