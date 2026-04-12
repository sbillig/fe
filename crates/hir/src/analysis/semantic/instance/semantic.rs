use cranelift_entity::EntityRef;
use rustc_hash::FxHashSet;

use crate::{
    analysis::{
        HirAnalysisDb,
        semantic::{
            PlaceProvenance, SBlockId, SExpr, SStmtKind, STerminatorKind, SemanticBindingLowering,
            SemanticBody, SemanticCalleeRef, SemanticLocalRole, SemanticProjection,
            ValueProvenance, effect_param_site, lower::lower_to_smir, verify_semantic_body,
        },
        ty::{
            corelib::resolve_lib_func_path,
            effect_handle_metadata,
            effects::place_effect_provider_param_index_map,
            fold::TyFoldable,
            instantiate_trait_self,
            normalize::normalize_ty,
            provider::{ProviderKind, provider_semantics},
            trait_resolution::PredicateListId,
            ty_check::{BodyOwner, LocalBinding, SemanticExprLowering, TypedBody},
        },
    },
    hir_def::FuncParamMode,
    hir_def::{CallableDef, Partial, scope_graph::ScopeId},
    semantic::{
        EffectEnvView, EffectRequirement, EffectRequirementKey, ProviderBinding, ProviderSource,
        ResolvedEffectBinding,
    },
};
use common::indexmap::IndexMap;
use indexmap::IndexSet;

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

#[derive(Debug, Clone)]
pub struct SemanticEffectEnvInstantiationError<'db> {
    pub owner: BodyOwner<'db>,
    pub owner_scope: ScopeId<'db>,
    pub offending_ty: crate::analysis::ty::ty_def::TyId<'db>,
    pub param_idx: usize,
    pub args_len: usize,
}

#[derive(Debug, Clone)]
pub enum RootSemanticInstanceError<'db> {
    UnsupportedGenericParam {
        owner: BodyOwner<'db>,
        owner_scope: ScopeId<'db>,
        offending_ty: crate::analysis::ty::ty_def::TyId<'db>,
        param_idx: usize,
    },
    MissingRootProvider {
        owner: BodyOwner<'db>,
    },
    UnclosedEffectEnv(SemanticEffectEnvInstantiationError<'db>),
}

type InstantiatedEffectEnvData<'db> = (
    crate::analysis::ty::ty_check::EffectParamSite<'db>,
    Vec<EffectRequirement<'db>>,
    Vec<ProviderBinding<'db>>,
    Vec<ResolvedEffectBinding>,
    Vec<crate::analysis::ty::trait_def::TraitInstId<'db>>,
    PredicateListId<'db>,
);

#[salsa::tracked]
#[derive(Debug)]
pub struct InstantiatedEffectEnv<'db> {
    pub site: crate::analysis::ty::ty_check::EffectParamSite<'db>,
    #[return_ref]
    pub requirements: Vec<EffectRequirement<'db>>,
    #[return_ref]
    pub providers: Vec<ProviderBinding<'db>>,
    #[return_ref]
    pub resolutions: Vec<ResolvedEffectBinding>,
    #[return_ref]
    pub forwarded_witnesses: Vec<crate::analysis::ty::trait_def::TraitInstId<'db>>,
    pub assumptions: PredicateListId<'db>,
}

#[salsa::tracked]
pub fn instantiated_effect_env<'db>(
    db: &'db dyn HirAnalysisDb,
    instance: SemanticInstance<'db>,
) -> Option<InstantiatedEffectEnv<'db>> {
    let (site, requirements, providers, resolutions, forwarded_witnesses, assumptions) =
        instantiate_effect_env_data(db, instance).unwrap_or_else(|err| {
            panic!(
                "failed to instantiate effect env for {:?}: owner_scope={:?} param_idx={} args_len={} offending_ty={}",
                err.owner,
                err.owner_scope,
                err.param_idx,
                err.args_len,
                err.offending_ty.pretty_print(db),
            )
        })?;
    Some(InstantiatedEffectEnv::new(
        db,
        site,
        requirements,
        providers,
        resolutions,
        forwarded_witnesses,
        assumptions,
    ))
}

#[salsa::tracked]
pub fn semantic_instance_assumptions<'db>(
    db: &'db dyn HirAnalysisDb,
    instance: SemanticInstance<'db>,
) -> PredicateListId<'db> {
    instantiated_effect_env(db, instance)
        .map(|env| env.assumptions(db))
        .unwrap_or_else(|| semantic_instance_base_assumptions_for_key(db, instance.key(db)))
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

#[salsa::tracked]
pub fn semantic_binding_ty<'db>(
    db: &'db dyn HirAnalysisDb,
    instance: SemanticInstance<'db>,
    binding: LocalBinding<'db>,
) -> crate::analysis::ty::ty_def::TyId<'db> {
    match binding {
        LocalBinding::EffectParam {
            idx, provider_idx, ..
        } => effect_binding_ty_from_env(
            db,
            instantiated_effect_env(db, instance),
            idx,
            Some(provider_idx),
        ),
        LocalBinding::Param {
            site: crate::analysis::ty::ty_check::ParamSite::EffectField(_),
            idx,
            ..
        } => effect_binding_ty_from_env(db, instantiated_effect_env(db, instance), idx, None),
        LocalBinding::Local { .. } | LocalBinding::Param { .. } => instance
            .key(db)
            .instantiate_typed_body(db)
            .binding_ty(db, binding),
    }
}

#[salsa::tracked]
pub fn resolved_provider_binding_for_instance_effect<'db>(
    db: &'db dyn HirAnalysisDb,
    instance: SemanticInstance<'db>,
    binding: LocalBinding<'db>,
) -> Option<ProviderBinding<'db>> {
    let env = instantiated_effect_env(db, instance)?;
    let (binding_idx, provider_idx) = match binding {
        LocalBinding::EffectParam {
            idx, provider_idx, ..
        } => (idx, Some(provider_idx)),
        LocalBinding::Param {
            site: crate::analysis::ty::ty_check::ParamSite::EffectField(_),
            idx,
            ..
        } => (idx, None),
        LocalBinding::Local { .. } | LocalBinding::Param { .. } => return None,
    };
    let requirement = env
        .requirements(db)
        .iter()
        .find(|requirement| requirement.binding_idx as usize == binding_idx)
        .cloned();
    provider_idx
        .and_then(|provider_idx| {
            env.providers(db)
                .iter()
                .find(|provider| provider.provider_idx == provider_idx)
                .cloned()
        })
        .or_else(|| {
            instantiated_resolved_binding(env, db, binding_idx).map(|binding| binding.provider)
        })
        .map(|mut provider| {
            if matches!(
                provider.semantics.kind,
                crate::analysis::ty::provider::ProviderKind::RootObject
            ) && let Some(target_ty) =
                requirement.and_then(|requirement| requirement.key.binding_ty(db))
            {
                provider.semantics.target_ty = Some(target_ty);
            }
            provider
        })
}

fn effect_binding_ty_from_env<'db>(
    db: &'db dyn HirAnalysisDb,
    env: Option<InstantiatedEffectEnv<'db>>,
    idx: usize,
    provider_idx: Option<u32>,
) -> crate::analysis::ty::ty_def::TyId<'db> {
    let Some(env) = env else {
        return crate::analysis::ty::ty_def::TyId::invalid(
            db,
            crate::analysis::ty::ty_def::InvalidCause::Other,
        );
    };
    let requirement = env
        .requirements(db)
        .iter()
        .find(|requirement| requirement.binding_idx as usize == idx)
        .cloned();
    let provider = provider_idx
        .and_then(|provider_idx| {
            env.providers(db)
                .iter()
                .find(|provider| provider.provider_idx == provider_idx)
                .cloned()
        })
        .or_else(|| instantiated_resolved_binding(env, db, idx).map(|binding| binding.provider));
    match requirement.as_ref().map(|requirement| &requirement.key) {
        Some(crate::core::semantic::EffectRequirementKey::Trait(_)) => provider
            .map(|binding| binding.provider_ty)
            .or_else(|| requirement.and_then(|requirement| requirement.key.binding_ty(db))),
        Some(
            crate::core::semantic::EffectRequirementKey::Type(_)
            | crate::core::semantic::EffectRequirementKey::Other,
        ) => requirement
            .and_then(|requirement| requirement.key.binding_ty(db))
            .or_else(|| provider.map(|binding| binding.provider_ty)),
        None => None,
    }
    .unwrap_or_else(|| {
        crate::analysis::ty::ty_def::TyId::invalid(
            db,
            crate::analysis::ty::ty_def::InvalidCause::Other,
        )
    })
}

fn instantiated_resolved_binding<'db>(
    env: InstantiatedEffectEnv<'db>,
    db: &'db dyn HirAnalysisDb,
    idx: usize,
) -> Option<crate::core::semantic::ResolvedEffectBindingInfo<'db>> {
    let requirement = env
        .requirements(db)
        .iter()
        .find(|requirement| requirement.binding_idx as usize == idx)
        .cloned()?;
    let provider_idx = env
        .resolutions(db)
        .iter()
        .find(|resolution| resolution.requirement_idx as usize == idx)?
        .provider_idx;
    let provider = env
        .providers(db)
        .iter()
        .find(|provider| provider.provider_idx == provider_idx)
        .cloned()?;
    Some(crate::core::semantic::ResolvedEffectBindingInfo {
        requirement,
        provider,
    })
}

pub fn root_semantic_instance_key<'db>(
    db: &'db dyn HirAnalysisDb,
    owner: BodyOwner<'db>,
) -> Result<SemanticInstanceKey<'db>, RootSemanticInstanceError<'db>> {
    let generic_args = root_owner_generic_args(db, owner)?;
    let key = SemanticInstanceKey::new(
        db,
        owner,
        GenericSubst::new(db, generic_args),
        ImplEnv::empty(db, owner.scope()),
    );
    validate_instantiated_effect_env_key(db, key)
        .map_err(RootSemanticInstanceError::UnclosedEffectEnv)?;
    Ok(key)
}

pub fn identity_semantic_instance_key<'db>(
    db: &'db dyn HirAnalysisDb,
    owner: BodyOwner<'db>,
) -> SemanticInstanceKey<'db> {
    SemanticInstanceKey::new(
        db,
        owner,
        GenericSubst::new(db, owner_identity_generic_args(db, owner)),
        ImplEnv::empty(db, owner.scope()),
    )
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
    SemanticInstance::new(db, key)
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
        let Partial::Present(_) = expr else {
            continue;
        };
        let Some(SemanticExprLowering::Call { callable }) =
            typed_body.semantic_expr_lowering(expr_id)
        else {
            continue;
        };
        let Some(callee_key) = semantic_callee_key(db, key, callable) else {
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
    binding: LocalBinding<'db>,
) -> SemanticBindingLowering<'db> {
    let owner = instance.key(db).owner(db);
    let scope = owner.scope();
    let assumptions = semantic_instance_assumptions(db, instance);
    let mut ty = normalize_ty(
        db,
        semantic_binding_ty(db, instance, binding),
        scope,
        assumptions,
    );
    if let LocalBinding::Param {
        mode: FuncParamMode::View,
        ..
    } = binding
        && let Some(inner) = ty.as_view(db)
    {
        ty = normalize_ty(db, inner, scope, assumptions);
    }
    if let Some((_, value_ty)) = ty.as_capability(db) {
        let value_ty = normalize_ty(db, value_ty, scope, assumptions);
        return SemanticBindingLowering::PlaceCarrier { value_ty };
    }
    if let Some(metadata) = effect_handle_metadata(db, scope, assumptions, ty) {
        return SemanticBindingLowering::DirectCarrier {
            provider: resolved_provider_binding_for_instance_effect(db, instance, binding),
            target_ty: metadata.target_ty,
        };
    }
    if let Some(provider) = resolved_provider_binding_for_instance_effect(db, instance, binding) {
        return match provider.semantics.kind {
            ProviderKind::RootObject => SemanticBindingLowering::DirectValue {
                provenance: ValueProvenance::RootProvider(provider),
            },
            ProviderKind::Handle | ProviderKind::RawAddress => {
                SemanticBindingLowering::PlaceBoundValue {
                    provenance: PlaceProvenance::RootProvider(provider),
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
    let scope = owner.scope();
    let assumptions = semantic_instance_assumptions(db, instance);

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
        SemanticBindingLowering::PlaceBoundValue {
            provenance,
            value_ty,
        } => SemanticLocalRole::PlaceBoundValue {
            provenance,
            value_ty,
        },
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

pub fn validate_instantiated_effect_env<'db>(
    db: &'db dyn HirAnalysisDb,
    instance: SemanticInstance<'db>,
) -> Result<(), SemanticEffectEnvInstantiationError<'db>> {
    instantiate_effect_env_data_for_key(db, instance.key(db)).map(|_| ())
}

pub fn validate_instantiated_effect_env_key<'db>(
    db: &'db dyn HirAnalysisDb,
    key: SemanticInstanceKey<'db>,
) -> Result<(), SemanticEffectEnvInstantiationError<'db>> {
    instantiate_effect_env_data_for_key(db, key).map(|_| ())
}

fn instantiate_effect_env_data<'db>(
    db: &'db dyn HirAnalysisDb,
    instance: SemanticInstance<'db>,
) -> Result<Option<InstantiatedEffectEnvData<'db>>, SemanticEffectEnvInstantiationError<'db>> {
    instantiate_effect_env_data_for_key(db, instance.key(db))
}

fn instantiate_effect_env_data_for_key<'db>(
    db: &'db dyn HirAnalysisDb,
    key: SemanticInstanceKey<'db>,
) -> Result<Option<InstantiatedEffectEnvData<'db>>, SemanticEffectEnvInstantiationError<'db>> {
    let owner = key.owner(db);
    let Some(site) = effect_param_site(owner) else {
        return Ok(None);
    };
    let base_assumptions = semantic_instance_base_assumptions_for_key(db, key);
    let view = EffectEnvView::new(site);
    let requirements = view
        .requirements(db)
        .into_iter()
        .map(|requirement| instantiate_effect_requirement(db, key, requirement))
        .collect::<Result<Vec<_>, _>>()?;
    let providers = view
        .providers(db)
        .into_iter()
        .map(|provider| instantiate_provider_binding(db, key, provider))
        .collect::<Result<Vec<_>, _>>()?;
    let resolutions = view.resolutions(db);
    let forwarded_witnesses =
        instantiated_effect_env_forwarded_witnesses(db, &requirements, &providers, &resolutions);
    let assumptions = if forwarded_witnesses.is_empty() {
        base_assumptions
    } else {
        let mut predicates: IndexSet<_> = base_assumptions.list(db).iter().copied().collect();
        predicates.extend(forwarded_witnesses.iter().copied());
        PredicateListId::new(db, predicates.into_iter().collect::<Vec<_>>()).extend_all_bounds(db)
    };
    Ok(Some((
        site,
        requirements,
        providers,
        resolutions,
        forwarded_witnesses,
        assumptions,
    )))
}

fn semantic_instance_base_assumptions_for_key<'db>(
    db: &'db dyn HirAnalysisDb,
    key: SemanticInstanceKey<'db>,
) -> PredicateListId<'db> {
    let typed_body = key.instantiate_typed_body(db);
    let impl_env = key.impl_env(db);
    let mut predicates: IndexSet<_> = typed_body.assumptions().list(db).iter().copied().collect();
    predicates.extend(impl_env.assumptions(db).list(db).iter().copied());
    predicates.extend(impl_env.witnesses(db).iter().copied());
    PredicateListId::new(db, predicates.into_iter().collect::<Vec<_>>()).extend_all_bounds(db)
}

fn instantiated_effect_env_forwarded_witnesses<'db>(
    db: &'db dyn HirAnalysisDb,
    requirements: &[EffectRequirement<'db>],
    providers: &[ProviderBinding<'db>],
    resolutions: &[ResolvedEffectBinding],
) -> Vec<crate::analysis::ty::trait_def::TraitInstId<'db>> {
    let provider_by_idx = providers
        .iter()
        .map(|provider| (provider.provider_idx, provider.provider_ty))
        .collect::<IndexMap<_, _>>();
    let resolution_by_req = resolutions
        .iter()
        .map(|resolution| (resolution.requirement_idx, resolution.provider_idx))
        .collect::<IndexMap<_, _>>();
    let mut witnesses = IndexSet::new();
    for requirement in requirements {
        let Some(trait_inst) = requirement.key.key_trait() else {
            continue;
        };
        let witness = resolution_by_req
            .get(&requirement.binding_idx)
            .and_then(|provider_idx| provider_by_idx.get(provider_idx))
            .copied()
            .map_or(trait_inst, |provider_ty| {
                instantiate_trait_self(db, trait_inst, provider_ty)
            });
        witnesses.insert(witness);
    }
    witnesses.into_iter().collect()
}

fn root_owner_generic_args<'db>(
    db: &'db dyn HirAnalysisDb,
    owner: BodyOwner<'db>,
) -> Result<Vec<crate::analysis::ty::ty_def::TyId<'db>>, RootSemanticInstanceError<'db>> {
    match owner {
        BodyOwner::Func(func) => root_func_generic_args(db, func),
        BodyOwner::Const(_)
        | BodyOwner::AnonConstBody { .. }
        | BodyOwner::ContractInit { .. }
        | BodyOwner::ContractRecvArm { .. } => Ok(Vec::new()),
    }
}

fn owner_identity_generic_args<'db>(
    db: &'db dyn HirAnalysisDb,
    owner: BodyOwner<'db>,
) -> Vec<crate::analysis::ty::ty_def::TyId<'db>> {
    match owner {
        BodyOwner::Func(func) => CallableDef::Func(func).params(db).to_vec(),
        BodyOwner::Const(_)
        | BodyOwner::AnonConstBody { .. }
        | BodyOwner::ContractInit { .. }
        | BodyOwner::ContractRecvArm { .. } => Vec::new(),
    }
}

fn root_func_generic_args<'db>(
    db: &'db dyn HirAnalysisDb,
    func: crate::hir_def::Func<'db>,
) -> Result<Vec<crate::analysis::ty::ty_def::TyId<'db>>, RootSemanticInstanceError<'db>> {
    let owner = BodyOwner::Func(func);
    let owner_scope = func.scope();
    let provider_param_idxs = place_effect_provider_param_index_map(db, func)
        .iter()
        .flatten()
        .copied()
        .collect::<FxHashSet<_>>();
    let params = CallableDef::Func(func).params(db);
    if provider_param_idxs.is_empty() {
        if let Some((param_idx, &offending_ty)) = params.iter().enumerate().next() {
            return Err(RootSemanticInstanceError::UnsupportedGenericParam {
                owner,
                owner_scope,
                offending_ty,
                param_idx,
            });
        }
        return Ok(Vec::new());
    }
    for (param_idx, &param_ty) in params.iter().enumerate() {
        let is_effect_provider = matches!(
            param_ty.data(db),
            crate::analysis::ty::ty_def::TyData::TyParam(param)
                if param.owner == owner_scope && param.is_effect_provider() && provider_param_idxs.contains(&param_idx)
        );
        if !is_effect_provider {
            return Err(RootSemanticInstanceError::UnsupportedGenericParam {
                owner,
                owner_scope,
                offending_ty: param_ty,
                param_idx,
            });
        }
    }
    let site = effect_param_site(owner).expect("function owners should always have an effect site");
    let root_provider_ty = EffectEnvView::new(site)
        .providers(db)
        .iter()
        .find_map(|provider| {
            matches!(provider.source, ProviderSource::RootProvider { .. })
                .then_some(provider.provider_ty)
        })
        .ok_or(RootSemanticInstanceError::MissingRootProvider { owner })?;
    Ok(vec![root_provider_ty; params.len()])
}

fn instantiate_effect_requirement<'db>(
    db: &'db dyn HirAnalysisDb,
    key: SemanticInstanceKey<'db>,
    requirement: EffectRequirement<'db>,
) -> Result<EffectRequirement<'db>, SemanticEffectEnvInstantiationError<'db>> {
    Ok(EffectRequirement {
        key: instantiate_effect_requirement_key(db, key, requirement.key.clone())?,
        ..requirement
    })
}

fn instantiate_effect_requirement_key<'db>(
    db: &'db dyn HirAnalysisDb,
    key: SemanticInstanceKey<'db>,
    requirement_key: EffectRequirementKey<'db>,
) -> Result<EffectRequirementKey<'db>, SemanticEffectEnvInstantiationError<'db>> {
    Ok(match requirement_key {
        EffectRequirementKey::Type(ty) => {
            EffectRequirementKey::Type(instantiate_normalized_ty(db, key, ty)?)
        }
        EffectRequirementKey::Trait(trait_inst) => {
            EffectRequirementKey::Trait(instantiate_normalized_trait_inst(db, key, trait_inst)?)
        }
        EffectRequirementKey::Other => EffectRequirementKey::Other,
    })
}

fn instantiate_provider_binding<'db>(
    db: &'db dyn HirAnalysisDb,
    key: SemanticInstanceKey<'db>,
    provider: ProviderBinding<'db>,
) -> Result<ProviderBinding<'db>, SemanticEffectEnvInstantiationError<'db>> {
    let scope = key.owner(db).scope();
    let assumptions = semantic_instance_base_assumptions_for_key(db, key);
    let provider_ty = instantiate_normalized_ty(db, key, provider.provider_ty)?;
    let source = match provider.source.clone() {
        ProviderSource::RootProvider { site, registration } => ProviderSource::RootProvider {
            site,
            registration: crate::analysis::ty::provider::RootProviderRegistration {
                provider_ty: instantiate_normalized_ty(db, key, registration.provider_ty)?,
                ..registration
            },
        },
        source => source,
    };
    let semantics = match source {
        ProviderSource::ContractField { .. } => crate::analysis::ty::provider::ProviderSemantics {
            provider_ty,
            target_ty: provider
                .semantics
                .target_ty
                .map(|ty| instantiate_normalized_ty(db, key, ty))
                .transpose()?,
            ..provider.semantics
        },
        ProviderSource::UsesParam { .. } | ProviderSource::RootProvider { .. } => {
            provider_semantics(db, scope, assumptions, provider_ty)
        }
    };
    Ok(ProviderBinding {
        provider_ty,
        source,
        semantics,
        ..provider
    })
}

fn instantiate_normalized_ty<'db>(
    db: &'db dyn HirAnalysisDb,
    key: SemanticInstanceKey<'db>,
    ty: crate::analysis::ty::ty_def::TyId<'db>,
) -> Result<crate::analysis::ty::ty_def::TyId<'db>, SemanticEffectEnvInstantiationError<'db>> {
    let scope = key.owner(db).scope();
    let assumptions = semantic_instance_base_assumptions_for_key(db, key);
    let ty = instantiate_checked(db, key.owner(db), scope, ty, key.subst(db).generic_args(db))?;
    Ok(normalize_ty(db, ty, scope, assumptions))
}

fn instantiate_normalized_trait_inst<'db>(
    db: &'db dyn HirAnalysisDb,
    key: SemanticInstanceKey<'db>,
    trait_inst: crate::analysis::ty::trait_def::TraitInstId<'db>,
) -> Result<
    crate::analysis::ty::trait_def::TraitInstId<'db>,
    SemanticEffectEnvInstantiationError<'db>,
> {
    let scope = key.owner(db).scope();
    let assumptions = semantic_instance_base_assumptions_for_key(db, key);
    let trait_inst = instantiate_checked(
        db,
        key.owner(db),
        scope,
        trait_inst,
        key.subst(db).generic_args(db),
    )?;
    let args = trait_inst
        .args(db)
        .iter()
        .map(|&arg| normalize_ty(db, arg, scope, assumptions))
        .collect::<Vec<_>>();
    let assoc_type_bindings = trait_inst
        .assoc_type_bindings(db)
        .iter()
        .map(|(&name, &ty)| (name, normalize_ty(db, ty, scope, assumptions)))
        .collect::<IndexMap<_, _>>();
    Ok(crate::analysis::ty::trait_def::TraitInstId::new(
        db,
        trait_inst.def(db),
        args,
        assoc_type_bindings,
    ))
}

fn instantiate_checked<'db, T>(
    db: &'db dyn HirAnalysisDb,
    owner: BodyOwner<'db>,
    owner_scope: ScopeId<'db>,
    value: T,
    args: &[crate::analysis::ty::ty_def::TyId<'db>],
) -> Result<T, SemanticEffectEnvInstantiationError<'db>>
where
    T: crate::analysis::ty::fold::TyFoldable<'db>,
{
    let mut folder = CheckedInstantiateFolder {
        owner,
        owner_scope,
        args,
        error: None,
    };
    let value = value.fold_with(db, &mut folder);
    folder.error.map_or(Ok(value), Err)
}

struct CheckedInstantiateFolder<'db, 'a> {
    owner: BodyOwner<'db>,
    owner_scope: ScopeId<'db>,
    args: &'a [crate::analysis::ty::ty_def::TyId<'db>],
    error: Option<SemanticEffectEnvInstantiationError<'db>>,
}

impl<'db> crate::analysis::ty::fold::TyFolder<'db> for CheckedInstantiateFolder<'db, '_> {
    fn fold_ty(
        &mut self,
        db: &'db dyn HirAnalysisDb,
        ty: crate::analysis::ty::ty_def::TyId<'db>,
    ) -> crate::analysis::ty::ty_def::TyId<'db> {
        match ty.data(db) {
            crate::analysis::ty::ty_def::TyData::TyParam(param)
                if param.owner == self.owner_scope && !param.is_effect() =>
            {
                if let Some(arg) = self.args.get(param.idx).copied() {
                    return arg;
                }
                self.error
                    .get_or_insert(SemanticEffectEnvInstantiationError {
                        owner: self.owner,
                        owner_scope: self.owner_scope,
                        offending_ty: ty,
                        param_idx: param.idx,
                        args_len: self.args.len(),
                    });
                ty
            }
            crate::analysis::ty::ty_def::TyData::ConstTy(const_ty) => {
                if let crate::analysis::ty::const_ty::ConstTyData::TyParam(param, _) =
                    const_ty.data(db)
                    && param.owner == self.owner_scope
                {
                    if let Some(arg) = self.args.get(param.idx).copied() {
                        return arg;
                    }
                    self.error
                        .get_or_insert(SemanticEffectEnvInstantiationError {
                            owner: self.owner,
                            owner_scope: self.owner_scope,
                            offending_ty: ty,
                            param_idx: param.idx,
                            args_len: self.args.len(),
                        });
                    return ty;
                }
                ty.super_fold_with(db, self)
            }
            _ => ty.super_fold_with(db, self),
        }
    }
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
            SemanticLocalRole::PlaceBoundValue {
                provenance,
                value_ty,
            } => SemanticLocalRole::PlaceBoundValue {
                provenance,
                value_ty,
            },
            SemanticLocalRole::DirectCarrier {
                provider,
                target_ty,
            } => SemanticLocalRole::DirectCarrier {
                provider,
                target_ty,
            },
            SemanticLocalRole::Erased => SemanticLocalRole::Erased,
        },
        SExpr::Borrow { .. } => fallback_local_role(db, scope, assumptions, dst_ty),
        SExpr::Field { base, field } => classify_projection_local_role(
            db,
            scope,
            assumptions,
            dst_ty,
            *base,
            vec![SemanticProjection::Field(field.0 as usize)].into_boxed_slice(),
            locals,
        ),
        SExpr::Index { base, index } => classify_projection_local_role(
            db,
            scope,
            assumptions,
            dst_ty,
            *base,
            vec![SemanticProjection::Index(*index)].into_boxed_slice(),
            locals,
        ),
        SExpr::ExtractEnumField {
            value,
            variant,
            field,
        } => classify_projection_local_role(
            db,
            scope,
            assumptions,
            dst_ty,
            *value,
            vec![SemanticProjection::VariantField {
                variant: *variant,
                enum_ty: locals[value.index()].ty,
                field_idx: field.0 as usize,
            }]
            .into_boxed_slice(),
            locals,
        ),
        SExpr::Call { .. } => fallback_local_role(db, scope, assumptions, dst_ty),
        SExpr::AggregateMake { ty, .. } => {
            let fallback = fallback_local_role(db, scope, assumptions, *ty);
            match fallback {
                SemanticLocalRole::PlaceCarrier { .. }
                | SemanticLocalRole::PlaceBoundValue { .. }
                | SemanticLocalRole::DirectCarrier { .. } => fallback,
                SemanticLocalRole::Erased | SemanticLocalRole::DirectValue { .. } => {
                    SemanticLocalRole::DirectValue {
                        provenance: ValueProvenance::Ordinary,
                    }
                }
            }
        }
        SExpr::Const(_)
        | SExpr::Unary { .. }
        | SExpr::Binary { .. }
        | SExpr::Cast { .. }
        | SExpr::EnumMake { .. }
        | SExpr::GetEnumTag { .. }
        | SExpr::IsEnumVariant { .. }
        | SExpr::CodeRegionOffset { .. }
        | SExpr::CodeRegionLen { .. } => SemanticLocalRole::DirectValue {
            provenance: ValueProvenance::Ordinary,
        },
    }
}

fn classify_projection_local_role<'db>(
    db: &'db dyn HirAnalysisDb,
    scope: crate::hir_def::scope_graph::ScopeId<'db>,
    assumptions: crate::analysis::ty::trait_resolution::PredicateListId<'db>,
    dst_ty: crate::analysis::ty::ty_def::TyId<'db>,
    base: crate::analysis::semantic::SLocalId,
    path: Box<[SemanticProjection<'db>]>,
    locals: &[crate::analysis::semantic::SLocal<'db>],
) -> SemanticLocalRole<'db> {
    let fallback = fallback_local_role(db, scope, assumptions, dst_ty);
    let base_role = locals[base.index()].role.clone();
    match fallback {
        SemanticLocalRole::Erased => SemanticLocalRole::Erased,
        SemanticLocalRole::DirectValue { .. }
            if !matches!(base_role, SemanticLocalRole::Erased) =>
        {
            SemanticLocalRole::DirectValue {
                provenance: ValueProvenance::DerivedPlace { base, path },
            }
        }
        SemanticLocalRole::DirectValue { .. } => fallback,
        SemanticLocalRole::PlaceCarrier { value_ty }
        | SemanticLocalRole::PlaceBoundValue { value_ty, .. }
            if local_role_supports_place_provenance(&base_role) =>
        {
            SemanticLocalRole::PlaceBoundValue {
                provenance: PlaceProvenance::Derived { base, path },
                value_ty,
            }
        }
        SemanticLocalRole::PlaceCarrier { .. } | SemanticLocalRole::PlaceBoundValue { .. } => {
            fallback
        }
        SemanticLocalRole::DirectCarrier { target_ty, .. } => base_role
            .root_provider(locals)
            .map_or(fallback, |provider| SemanticLocalRole::DirectCarrier {
                provider: Some(provider),
                target_ty,
            }),
    }
}

fn local_role_supports_place_provenance(role: &SemanticLocalRole<'_>) -> bool {
    match role {
        SemanticLocalRole::Erased
        | SemanticLocalRole::DirectValue {
            provenance: ValueProvenance::Ordinary,
        }
        | SemanticLocalRole::DirectCarrier { provider: None, .. } => false,
        SemanticLocalRole::DirectValue { .. }
        | SemanticLocalRole::PlaceCarrier { .. }
        | SemanticLocalRole::PlaceBoundValue { .. }
        | SemanticLocalRole::DirectCarrier {
            provider: Some(_), ..
        } => true,
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
        (
            ValueProvenance::DerivedPlace {
                base: left_base,
                path: left_path,
            },
            ValueProvenance::DerivedPlace {
                base: right_base,
                path: right_path,
            },
        ) if left_base == right_base && left_path == right_path => ValueProvenance::DerivedPlace {
            base: left_base,
            path: left_path,
        },
        (ValueProvenance::RootProvider(_), ValueProvenance::RootProvider(_))
        | (ValueProvenance::DerivedPlace { .. }, ValueProvenance::DerivedPlace { .. })
        | (ValueProvenance::RootProvider(_), ValueProvenance::DerivedPlace { .. })
        | (ValueProvenance::DerivedPlace { .. }, ValueProvenance::RootProvider(_)) => {
            return None;
        }
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
