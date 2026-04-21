use super::{
    EffectProviderSubst, GenericSubst, ImplEnv, SemanticInstance, SemanticInstanceKey,
    resolved_provider_binding_for_instance_effect, semantic_instance_assumptions,
};
use crate::{
    analysis::{
        HirAnalysisDb,
        semantic::{SemOrigin, SemanticConstRef},
        ty::{
            assoc_const::AssocConstUse,
            effects::place_effect_provider_param_index_map,
            trait_def::{
                assoc_const_body_and_impl_args_for_trait_inst, resolve_trait_method_instance,
            },
            trait_resolution::TraitSolveCx,
            ty_check::{
                BodyOwner, Callable, ConstRef, EffectParamSite, EffectProviderSpecialization,
            },
            ty_def::TyId,
            ty_lower::instantiate_callable_effect_layout_args,
        },
    },
    core::semantic::EffectEnvView,
    hir_def::{CallableDef, Const},
};
use common::indexmap::IndexSet;
use rustc_hash::FxHashMap;

pub(crate) fn semantic_callee_key<'db>(
    db: &'db dyn HirAnalysisDb,
    caller_key: SemanticInstanceKey<'db>,
    callable: &Callable<'db>,
) -> Option<SemanticInstanceKey<'db>> {
    let impl_env = caller_key.impl_env(db);
    let assumptions = semantic_instance_assumptions(db, SemanticInstance::new(db, caller_key));
    let (owner, mut subst_args) = match callable.callable_def() {
        CallableDef::Func(func) => {
            let mut subst_args = callable.generic_args().to_vec();
            let owner = if let Some(inst) = callable.trait_inst()
                && let Some(name) = func.name(db).to_opt()
                && let Some((impl_func, impl_args)) = resolve_trait_method_instance(
                    db,
                    TraitSolveCx::new(db, impl_env.normalization_scope(db))
                        .with_assumptions(assumptions),
                    inst,
                    name,
                ) {
                let trait_arg_len = inst.args(db).len();
                let mut resolved_args = impl_args;
                let tail = subst_args
                    .get(trait_arg_len..)
                    .unwrap_or(subst_args.as_slice());
                resolved_args.extend_from_slice(tail);
                subst_args = resolved_args;
                BodyOwner::Func(impl_func)
            } else {
                BodyOwner::Func(func)
            };
            (owner, subst_args)
        }
        CallableDef::VariantCtor(_) => return None,
    };
    let effect_providers =
        resolve_callable_effect_providers(db, caller_key, owner, &mut subst_args, callable);

    let mut witnesses: IndexSet<_> = impl_env.witnesses(db).iter().copied().collect();
    if let Some(witness) = callable.trait_inst() {
        witnesses.insert(witness);
    }
    let impl_env = ImplEnv::new(
        db,
        impl_env.normalization_scope(db),
        assumptions,
        witnesses.into_iter().collect::<Vec<_>>(),
    );

    Some(SemanticInstanceKey::new(
        db,
        owner,
        GenericSubst::new(db, subst_args),
        EffectProviderSubst::new(db, effect_providers),
        impl_env,
    ))
}

fn resolve_callable_effect_providers<'db>(
    db: &'db dyn HirAnalysisDb,
    caller_key: SemanticInstanceKey<'db>,
    owner: BodyOwner<'db>,
    subst_args: &mut [TyId<'db>],
    callable: &Callable<'db>,
) -> Vec<EffectProviderSpecialization<'db>> {
    let caller = SemanticInstance::new(db, caller_key);
    let mut providers = callable
        .effect_providers()
        .iter()
        .map(|specialization| {
            let provider_idx = specialization.provider.provider_idx;
            let provider = match specialization.provenance {
                crate::analysis::ty::ty_check::EffectProviderProvenance::Binding {
                    binding,
                    ..
                } => resolved_provider_binding_for_instance_effect(db, caller, binding)
                    .map(|provider| crate::semantic::ProviderBinding {
                        provider_idx,
                        ..provider
                    })
                    .unwrap_or(specialization.provider.clone()),
                crate::analysis::ty::ty_check::EffectProviderProvenance::Expr { .. } => {
                    specialization.provider.clone()
                }
            };
            EffectProviderSpecialization {
                provider,
                provenance: specialization.provenance,
            }
        })
        .collect::<Vec<_>>();
    providers.sort_by_key(|provider| provider.provider.provider_idx);
    if let BodyOwner::Func(func) = owner {
        let resolution_by_req = EffectEnvView::new(EffectParamSite::Func(func))
            .resolutions(db)
            .into_iter()
            .map(|resolution| (resolution.requirement_idx as usize, resolution.provider_idx))
            .collect::<FxHashMap<_, _>>();
        let provider_by_idx = providers
            .iter()
            .map(|provider| (provider.provider.provider_idx, provider))
            .collect::<FxHashMap<_, _>>();
        for (effect_idx, param_idx) in place_effect_provider_param_index_map(db, func)
            .iter()
            .enumerate()
            .filter_map(|(effect_idx, param_idx)| {
                param_idx.map(|param_idx| (effect_idx, param_idx))
            })
        {
            let Some(provider_idx) = resolution_by_req.get(&effect_idx).copied() else {
                continue;
            };
            let Some(provider) = provider_by_idx.get(&provider_idx) else {
                continue;
            };
            if let Some(slot) = subst_args.get_mut(param_idx) {
                *slot = provider.provider.provider_ty;
            }
            let actual_key_ty = provider
                .provider
                .semantics
                .target_ty
                .unwrap_or(provider.provider.provider_ty);
            instantiate_callable_effect_layout_args(
                db,
                func,
                effect_idx,
                actual_key_ty,
                subst_args,
            );
        }
    }
    providers
}

pub(crate) fn resolve_semantic_const_ref<'db>(
    db: &'db dyn HirAnalysisDb,
    const_ref: ConstRef<'db>,
    ty: TyId<'db>,
    origin: SemOrigin<'db>,
) -> Option<SemanticConstRef<'db>> {
    let instance = match const_ref {
        ConstRef::Const(const_) => semantic_const_key_for_const(db, const_),
        ConstRef::TraitConst(assoc) => semantic_const_key_for_assoc_const(db, assoc, ty),
    }?;
    Some(SemanticConstRef::new(db, instance, ty, origin))
}

fn semantic_const_key_for_const<'db>(
    db: &'db dyn HirAnalysisDb,
    const_: Const<'db>,
) -> Option<SemanticInstanceKey<'db>> {
    let owner = BodyOwner::Const(const_);
    Some(SemanticInstanceKey::new(
        db,
        owner,
        GenericSubst::empty(db),
        EffectProviderSubst::empty(db),
        ImplEnv::empty(db, owner.scope()),
    ))
}

fn semantic_const_key_for_assoc_const<'db>(
    db: &'db dyn HirAnalysisDb,
    assoc: AssocConstUse<'db>,
    ty: TyId<'db>,
) -> Option<SemanticInstanceKey<'db>> {
    let (body, impl_args) = assoc_const_body_and_impl_args_for_trait_inst(
        db,
        assoc.solve_cx(db),
        assoc.inst(),
        assoc.name(),
    )?;
    Some(SemanticInstanceKey::new(
        db,
        BodyOwner::AnonConstBody { body, expected: ty },
        GenericSubst::new(db, impl_args),
        EffectProviderSubst::empty(db),
        ImplEnv::new(
            db,
            assoc.origin_scope(),
            assoc.assumptions(),
            vec![assoc.inst()],
        ),
    ))
}
