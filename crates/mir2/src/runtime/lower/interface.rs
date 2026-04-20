use hir::analysis::{
    semantic::{
        SemanticInstance, owner_effect_bindings, same_owner_effect_binding, semantic_binding_ty,
        semantic_instance_assumptions,
    },
    ty::ty_check::{BodyOwner, LocalBinding, ParamSite},
    ty::ty_def::TyId,
};

use crate::{
    db::MirDb,
    runtime::{AddressSpaceKind, RuntimeBoundarySpec, RuntimeParamPlan},
};

use super::{
    classify::{
        RuntimeVisibleBindingPlan, desired_runtime_param_plan, owner_effect_binding_boundary,
    },
    type_info::{RuntimeTypeEnv, top_level_class_for_ty_in_env},
};

pub(crate) fn runtime_param_locals<'db>(
    db: &'db dyn MirDb,
    semantic: SemanticInstance<'db>,
    params: &[crate::runtime::RuntimeClass<'db>],
) -> Vec<hir::analysis::semantic::SLocalId> {
    let entries = runtime_visible_binding_plans(db, semantic);
    if entries.len() != params.len() {
        let owner = semantic.key(db).owner(db);
        let binding_debug = entries
            .iter()
            .map(|entry| {
                let ty = semantic_binding_ty(db, semantic, entry.binding)
                    .pretty_print(db)
                    .to_string();
                format!(
                    "{:?}:{ty}:plan={:?}:local={:?}",
                    entry.binding, entry.plan, entry.local
                )
            })
            .collect::<Vec<_>>()
            .join("; ");
        panic!(
            "failed to map runtime params to semantic locals for {:?} owner={:?}: expected {} runtime-visible params, got {}; params={params:?}; visible_bindings=[{}]",
            semantic.key(db),
            owner,
            entries.len(),
            params.len(),
            binding_debug,
        );
    }
    entries.iter().map(|entry| entry.local).collect()
}

fn runtime_visible_binding_semantic_ty<'db>(
    db: &'db dyn MirDb,
    semantic: SemanticInstance<'db>,
    typed_body: &hir::analysis::ty::ty_check::TypedBody<'db>,
    binding: LocalBinding<'db>,
) -> TyId<'db> {
    match binding {
        LocalBinding::EffectParam { .. }
        | LocalBinding::Param {
            site: ParamSite::EffectField(_),
            ..
        } => semantic_binding_ty(db, semantic, binding),
        LocalBinding::Local { .. } | LocalBinding::Param { .. } => {
            typed_body.binding_ty(db, binding)
        }
    }
}

#[salsa::tracked(return_ref)]
pub(crate) fn runtime_param_plans<'db>(
    db: &'db dyn MirDb,
    semantic: SemanticInstance<'db>,
) -> Vec<RuntimeParamPlan<'db>> {
    let typed_body = semantic.key(db).typed_body(db);
    let mut plans = Vec::new();
    let mut idx = 0;
    while typed_body.param_binding(idx).is_some() {
        plans.push(desired_runtime_param_plan(db, semantic, typed_body, idx));
        idx += 1;
    }
    plans
}

#[salsa::tracked(return_ref)]
pub(crate) fn runtime_visible_binding_plans<'db>(
    db: &'db dyn MirDb,
    semantic: SemanticInstance<'db>,
) -> Vec<RuntimeVisibleBindingPlan<'db>> {
    let owner = semantic.key(db).owner(db);
    let typed_body = semantic.key(db).typed_body(db);
    let param_plans = runtime_param_plans(db, semantic);
    let mut entries = Vec::new();
    let mut push = |binding, plan| {
        if !matches!(plan, RuntimeParamPlan::Erased) {
            entries.push(RuntimeVisibleBindingPlan {
                binding,
                local: runtime_visible_binding_local(db, owner, typed_body, binding),
                semantic_ty: runtime_visible_binding_semantic_ty(db, semantic, typed_body, binding),
                plan,
            });
        }
    };

    let mut idx = 0;
    while let Some(binding) = typed_body.param_binding(idx) {
        push(
            binding,
            param_plans
                .get(idx)
                .cloned()
                .unwrap_or(RuntimeParamPlan::Erased),
        );
        idx += 1;
    }

    if let BodyOwner::ContractRecvArm {
        contract,
        recv_idx,
        arm_idx,
    } = owner
    {
        let recv = hir::semantic::RecvView::new(db, contract, recv_idx);
        let arm = hir::semantic::RecvArmView::new(db, recv, arm_idx);
        let env = RuntimeTypeEnv::new(
            Some(owner.scope()),
            semantic_instance_assumptions(db, semantic),
        );
        for arg_binding in arm.arg_bindings(db) {
            let Some(binding) = typed_body.pat_binding(arg_binding.pat) else {
                continue;
            };
            let ty = semantic_binding_ty(db, semantic, binding);
            let plan = top_level_class_for_ty_in_env(db, env, ty, AddressSpaceKind::Memory)
                .map(RuntimeBoundarySpec::ExactTransport)
                .map(RuntimeParamPlan::Boundary)
                .unwrap_or(RuntimeParamPlan::Erased);
            push(binding, plan);
        }
    }

    for binding in owner_effect_bindings(db, owner) {
        let plan = owner_effect_binding_boundary(db, semantic, binding)
            .map(RuntimeParamPlan::Boundary)
            .unwrap_or(RuntimeParamPlan::Erased);
        push(binding, plan);
    }

    entries
}

fn runtime_visible_binding_local<'db>(
    db: &'db dyn MirDb,
    owner: BodyOwner<'db>,
    typed_body: &hir::analysis::ty::ty_check::TypedBody<'db>,
    binding: LocalBinding<'db>,
) -> hir::analysis::semantic::SLocalId {
    let mut next = 0u32;
    let mut param_idx = 0;
    while let Some(param_binding) = typed_body.param_binding(param_idx) {
        if param_binding == binding {
            return hir::analysis::semantic::SLocalId::from_u32(next);
        }
        next += 1;
        param_idx += 1;
    }
    if let BodyOwner::ContractRecvArm {
        contract,
        recv_idx,
        arm_idx,
    } = owner
    {
        let recv = hir::semantic::RecvView::new(db, contract, recv_idx);
        let arm = hir::semantic::RecvArmView::new(db, recv, arm_idx);
        for arg_binding in arm.arg_bindings(db) {
            let Some(pat_binding) = typed_body.pat_binding(arg_binding.pat) else {
                continue;
            };
            if pat_binding == binding {
                return hir::analysis::semantic::SLocalId::from_u32(next);
            }
            next += 1;
        }
    }
    for effect_binding in owner_effect_bindings(db, owner) {
        if same_owner_effect_binding(effect_binding, binding) {
            return hir::analysis::semantic::SLocalId::from_u32(next);
        }
        next += 1;
    }
    panic!("missing semantic local for runtime-visible binding {binding:?}")
}
