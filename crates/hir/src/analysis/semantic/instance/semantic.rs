use cranelift_entity::EntityRef;
use rustc_hash::FxHashSet;

use crate::{
    analysis::{
        HirAnalysisDb,
        semantic::{
            SBlockId, SExpr, SStmtKind, STerminatorKind, SemanticBindingLowering, SemanticBody,
            SemanticCalleeRef, SemanticLocalRole, ValueProvenance,
            lower::{expr_lowers_to_semantic_call, lower_to_smir},
            resolved_provider_binding_for_owner_effect, verify_semantic_body,
        },
        ty::{
            corelib::resolve_lib_func_path,
            effect_handle_metadata,
            normalize::normalize_ty,
            provider::ProviderKind,
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

#[salsa::tracked]
pub fn semantic_binding_lowering<'db>(
    db: &'db dyn HirAnalysisDb,
    instance: SemanticInstance<'db>,
    binding: crate::analysis::ty::ty_check::LocalBinding<'db>,
) -> SemanticBindingLowering<'db> {
    classify_binding_lowering(db, instance, binding)
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
            let SStmtKind::Assign {
                expr: SExpr::Call { callee, .. },
                ..
            } = &stmt.kind
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

        match &block.terminator.kind {
            STerminatorKind::Return(_) => return true,
            STerminatorKind::Goto(next) => pending.push(*next),
            STerminatorKind::Branch {
                then_bb, else_bb, ..
            } => {
                pending.push(*then_bb);
                pending.push(*else_bb);
            }
            STerminatorKind::MatchEnum { cases, default, .. } => {
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
    let mut body = lower_to_smir(db, instance, key.owner(db), typed_body);
    assign_semantic_local_roles(db, instance, &mut body);
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

fn classify_binding_lowering<'db>(
    db: &'db dyn HirAnalysisDb,
    instance: SemanticInstance<'db>,
    binding: crate::analysis::ty::ty_check::LocalBinding<'db>,
) -> SemanticBindingLowering<'db> {
    let owner = instance.key(db).owner(db);
    let typed_body = instance.key(db).instantiate_typed_body(db);
    let scope = owner.scope();
    let assumptions = typed_body.assumptions();
    let ty = normalize_ty(db, typed_body.binding_ty(db, binding), scope, assumptions);
    if let Some((_, value_ty)) = ty.as_capability(db) {
        let value_ty = normalize_ty(db, value_ty, scope, assumptions);
        return SemanticBindingLowering::PlaceCarrier { value_ty };
    }
    if let Some(metadata) = effect_handle_metadata(db, scope, assumptions, ty) {
        return SemanticBindingLowering::DirectCarrier {
            provider: resolved_provider_binding_for_owner_effect(db, owner, binding),
            target_ty: metadata.target_ty,
        };
    }
    if let Some(provider) = resolved_provider_binding_for_owner_effect(db, owner, binding) {
        return match provider.semantics.kind {
            ProviderKind::RootObject => SemanticBindingLowering::DirectValue {
                provenance: ValueProvenance::RootProvider(provider),
            },
            ProviderKind::Handle | ProviderKind::RawAddress => {
                SemanticBindingLowering::PlaceBoundValue {
                    provider,
                    value_ty: ty,
                }
            }
        };
    }
    SemanticBindingLowering::DirectValue {
        provenance: ValueProvenance::Ordinary,
    }
}

fn assign_semantic_local_roles<'db>(
    db: &'db dyn HirAnalysisDb,
    instance: SemanticInstance<'db>,
    body: &mut SemanticBody<'db>,
) {
    let owner = instance.key(db).owner(db);
    let typed_body = instance.key(db).instantiate_typed_body(db);
    let scope = owner.scope();
    let assumptions = typed_body.assumptions();

    for local in &mut body.locals {
        local.role = local.source.map_or_else(
            || SemanticLocalRole::DirectValue {
                provenance: ValueProvenance::Ordinary,
            },
            |binding| {
                binding_lowering_to_local_role(semantic_binding_lowering(db, instance, binding))
            },
        );
    }

    let assignments = body
        .blocks
        .iter()
        .flat_map(|block| block.stmts.iter())
        .filter_map(|stmt| match &stmt.kind {
            SStmtKind::Assign { dst, expr } => Some((*dst, expr.clone())),
            SStmtKind::Store { .. } => None,
        })
        .collect::<Vec<_>>();
    for (dst, expr) in assignments {
        if body.locals[dst.index()].source.is_some() {
            continue;
        }
        let fallback = fallback_local_role(db, scope, assumptions, body.locals[dst.index()].ty);
        let role = classify_expr_local_role(
            db,
            scope,
            assumptions,
            body.locals[dst.index()].ty,
            &expr,
            &body.locals,
        );
        body.locals[dst.index()].role =
            merge_local_roles(body.locals[dst.index()].role.clone(), role, fallback);
    }
}

fn binding_lowering_to_local_role<'db>(
    lowering: SemanticBindingLowering<'db>,
) -> SemanticLocalRole<'db> {
    match lowering {
        SemanticBindingLowering::Erased => SemanticLocalRole::Erased,
        SemanticBindingLowering::DirectValue { provenance } => {
            SemanticLocalRole::DirectValue { provenance }
        }
        SemanticBindingLowering::PlaceCarrier { value_ty } => {
            SemanticLocalRole::PlaceCarrier { value_ty }
        }
        SemanticBindingLowering::PlaceBoundValue { provider, value_ty } => {
            SemanticLocalRole::PlaceBoundValue { provider, value_ty }
        }
        SemanticBindingLowering::DirectCarrier {
            provider,
            target_ty,
        } => SemanticLocalRole::DirectCarrier {
            provider,
            target_ty,
        },
    }
}

fn fallback_local_role<'db>(
    db: &'db dyn HirAnalysisDb,
    scope: crate::hir_def::scope_graph::ScopeId<'db>,
    assumptions: crate::analysis::ty::trait_resolution::PredicateListId<'db>,
    ty: crate::analysis::ty::ty_def::TyId<'db>,
) -> SemanticLocalRole<'db> {
    let ty = normalize_ty(db, ty, scope, assumptions);
    if let Some((_, value_ty)) = ty.as_capability(db) {
        return SemanticLocalRole::PlaceCarrier {
            value_ty: normalize_ty(db, value_ty, scope, assumptions),
        };
    }
    effect_handle_metadata(db, scope, assumptions, ty).map_or(
        SemanticLocalRole::DirectValue {
            provenance: ValueProvenance::Ordinary,
        },
        |metadata| SemanticLocalRole::DirectCarrier {
            provider: None,
            target_ty: metadata.target_ty,
        },
    )
}

fn classify_expr_local_role<'db>(
    db: &'db dyn HirAnalysisDb,
    scope: crate::hir_def::scope_graph::ScopeId<'db>,
    assumptions: crate::analysis::ty::trait_resolution::PredicateListId<'db>,
    dst_ty: crate::analysis::ty::ty_def::TyId<'db>,
    expr: &SExpr<'db>,
    locals: &[crate::analysis::semantic::SLocal<'db>],
) -> SemanticLocalRole<'db> {
    match expr {
        SExpr::Use(value) => match locals[value.index()].role.clone() {
            SemanticLocalRole::DirectValue { provenance } => {
                SemanticLocalRole::DirectValue { provenance }
            }
            SemanticLocalRole::PlaceCarrier { value_ty } => {
                SemanticLocalRole::PlaceCarrier { value_ty }
            }
            SemanticLocalRole::DirectCarrier {
                provider,
                target_ty,
            } => SemanticLocalRole::DirectCarrier {
                provider,
                target_ty,
            },
            SemanticLocalRole::Erased => SemanticLocalRole::Erased,
            SemanticLocalRole::PlaceBoundValue { .. } => {
                fallback_local_role(db, scope, assumptions, dst_ty)
            }
        },
        SExpr::Borrow { .. } => fallback_local_role(db, scope, assumptions, dst_ty),
        SExpr::Field { .. }
        | SExpr::Index { .. }
        | SExpr::ExtractEnumField { .. }
        | SExpr::Call { .. } => fallback_local_role(db, scope, assumptions, dst_ty),
        SExpr::Const(_)
        | SExpr::Unary { .. }
        | SExpr::Binary { .. }
        | SExpr::Cast { .. }
        | SExpr::AggregateMake { .. }
        | SExpr::EnumMake { .. }
        | SExpr::GetEnumTag { .. }
        | SExpr::IsEnumVariant { .. }
        | SExpr::CodeRegionOffset { .. }
        | SExpr::CodeRegionLen { .. } => SemanticLocalRole::DirectValue {
            provenance: ValueProvenance::Ordinary,
        },
    }
}

fn merge_local_roles<'db>(
    current: SemanticLocalRole<'db>,
    next: SemanticLocalRole<'db>,
    fallback: SemanticLocalRole<'db>,
) -> SemanticLocalRole<'db> {
    if current == next {
        return current;
    }
    match (current, next) {
        (
            SemanticLocalRole::DirectValue {
                provenance: left_provenance,
            },
            SemanticLocalRole::DirectValue {
                provenance: right_provenance,
            },
        ) => merge_direct_value_role(left_provenance, right_provenance).unwrap_or(fallback),
        (
            SemanticLocalRole::DirectCarrier {
                provider: left_provider,
                target_ty: left_target_ty,
            },
            SemanticLocalRole::DirectCarrier {
                provider: right_provider,
                target_ty: right_target_ty,
            },
        ) if left_target_ty == right_target_ty => SemanticLocalRole::DirectCarrier {
            provider: (left_provider == right_provider)
                .then_some(left_provider)
                .flatten(),
            target_ty: left_target_ty,
        },
        (
            SemanticLocalRole::PlaceCarrier {
                value_ty: left_value_ty,
            },
            SemanticLocalRole::PlaceCarrier {
                value_ty: right_value_ty,
            },
        ) if left_value_ty == right_value_ty => SemanticLocalRole::PlaceCarrier {
            value_ty: left_value_ty,
        },
        (
            SemanticLocalRole::DirectValue {
                provenance: ValueProvenance::Ordinary,
            },
            next,
        ) => next,
        (
            current,
            SemanticLocalRole::DirectValue {
                provenance: ValueProvenance::Ordinary,
            },
        ) => current,
        _ => fallback,
    }
}

fn merge_direct_value_role<'db>(
    left: ValueProvenance<'db>,
    right: ValueProvenance<'db>,
) -> Option<SemanticLocalRole<'db>> {
    let provenance = match (left, right) {
        (ValueProvenance::Ordinary, other) | (other, ValueProvenance::Ordinary) => other,
        (ValueProvenance::RootProvider(left), ValueProvenance::RootProvider(right))
            if left == right =>
        {
            ValueProvenance::RootProvider(left)
        }
        (ValueProvenance::RootProvider(_), ValueProvenance::RootProvider(_)) => return None,
    };
    Some(SemanticLocalRole::DirectValue { provenance })
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
