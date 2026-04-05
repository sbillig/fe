use common::indexmap::IndexSet;
use cranelift_entity::EntityRef;
use hir::analysis::{
    semantic::{
        FieldIndex, GenericSubst, ImplEnv, Mutability, NEffectArg, NEffectArgValue, NSStmtKind,
        SConst, SLocalId, SemanticBody, SemanticCalleeRef, SemanticInstance, SemanticInstanceKey,
        SemanticLocalRole, ValueProvenance, VariantIndex,
        borrowck::{
            NBorrowRoot, NExpr, NOperand, NSLocal, NSPlace, NSPlaceRoot, NSTerminatorKind,
            NormalizedSemanticBody, normalize_semantic_body,
        },
        canonicalize_semantic_consts, get_or_build_semantic_instance, owner_effect_bindings,
        same_owner_effect_binding, sem_const_ty, semantic_binding_lowering, semantic_binding_ty,
    },
    ty::{
        ProviderAddressSpace, ProviderKind,
        corelib::resolve_lib_func_path,
        normalize::normalize_ty,
        provider::{provider_semantics, registered_root_providers},
        trait_def::{TraitInstId, resolve_trait_method_instance},
        trait_resolution::{PredicateListId, TraitSolveCx},
        ty_check::{BodyOwner, EffectParamSite, EffectPassMode, LocalBinding, ParamSite},
        ty_def::{
            MAX_INLINE_STRING_BYTES, PrimTy, TyBase, TyData, TyId, strip_derived_adt_layout_args,
        },
    },
};
use hir::hir_def::ArithBinOp;
use hir::projection::Projection;
use hir::semantic::ProviderBinding;

use crate::{
    db::MirDb,
    instance::{RuntimeInstanceKey, RuntimeInstanceSource, get_or_build_runtime_instance},
    runtime::{
        AddressSpaceKind, HandleKind, HandleView, Layout, RuntimeCarrier, RuntimeClass,
        RuntimeCodeRegion, RuntimeCodeRegionKey, RuntimeLocalLowering, RuntimeLocalRoot,
        RuntimeParam, RuntimeProviderBinding, RuntimeProviderBindingId, RuntimeSignature,
        SaturatingBinOp, ScalarClass, ScalarRepr, ScalarRole,
    },
};

use super::{
    layout::{
        RuntimeTypeEnv, is_zero_sized_in_context, layout_for_ty_in_context, layout_for_ty_in_env,
        runtime_repr_ty_in_context,
    },
    place::{
        address_space_from_provider, project_field_class, project_index_class,
        resolved_address_space,
    },
};

#[derive(Clone, Debug)]
pub(super) struct InferredRuntimeLocal<'db> {
    pub(super) carrier: RuntimeCarrier<'db>,
    pub(super) root: RuntimeLocalRoot<'db>,
}

pub(crate) struct RuntimeSemanticCallContext<'a, 'db> {
    pub(crate) caller: SemanticInstance<'db>,
    pub(crate) raw_body: &'a SemanticBody<'db>,
    pub(crate) body: &'a NormalizedSemanticBody<'db>,
    pub(crate) carriers: &'a [RuntimeCarrier<'db>],
    pub(crate) result_ty: TyId<'db>,
}

pub fn runtime_signature_for_key<'db>(
    db: &'db dyn MirDb,
    semantic: SemanticInstance<'db>,
    params: &[RuntimeClass<'db>],
) -> RuntimeSignature<'db> {
    let key = RuntimeInstanceKey::new(
        db,
        RuntimeInstanceSource::Semantic(semantic),
        params.to_vec(),
    );
    RuntimeSignature {
        params: params
            .iter()
            .zip(runtime_param_locals(db, semantic, params))
            .map(|(class, local)| RuntimeParam {
                local: crate::runtime::RLocalId::from_u32(local.index() as u32),
                class: class.clone(),
            })
            .collect(),
        ret: runtime_return_class_for_key(db, key),
    }
}

#[salsa::tracked(
    cycle_fn=runtime_return_class_cycle_recover,
    cycle_initial=runtime_return_class_cycle_initial
)]
pub fn runtime_return_class_for_key<'db>(
    db: &'db dyn MirDb,
    key: RuntimeInstanceKey<'db>,
) -> Option<RuntimeClass<'db>> {
    let semantic = key
        .semantic(db)
        .expect("return-class inference only applies to semantic runtime instances");
    let typed_body = semantic.key(db).instantiate_typed_body(db);
    if !return_ty_requires_runtime_body_inference(
        db,
        typed_body.result_ty(),
        typed_body.body().map(|body| body.scope()),
        typed_body.assumptions(),
    ) {
        return default_return_class(db, &typed_body);
    }
    let raw_semantic_body = canonicalize_semantic_consts(db, semantic);
    let semantic_body = normalize_semantic_body(db, semantic)
        .unwrap_or_else(|err| panic!("semantic normalization failed for {:?}: {err:?}", key));
    let states = infer_local_runtime_state(
        db,
        &raw_semantic_body,
        &semantic_body,
        key.params(db),
        &runtime_param_locals(db, semantic, key.params(db)),
        typed_body.body().map(|body| body.scope()),
        typed_body.assumptions(),
    );
    let mut returned = semantic_body
        .blocks
        .iter()
        .filter_map(|block| match &block.terminator.kind {
            NSTerminatorKind::Return(Some(value)) => {
                match &states.get(value.local.index())?.carrier {
                    RuntimeCarrier::Erased => None,
                    RuntimeCarrier::Value(class) => Some(class.clone()),
                }
            }
            NSTerminatorKind::Goto(_)
            | NSTerminatorKind::Branch { .. }
            | NSTerminatorKind::MatchEnum { .. }
            | NSTerminatorKind::Return(None) => None,
        })
        .collect::<Vec<_>>();
    let Some(first) = returned.pop() else {
        return default_return_class(db, &typed_body);
    };
    if returned.iter().all(|class| class == &first) {
        Some(first)
    } else {
        default_return_class(db, &typed_body)
    }
}

pub(super) fn infer_local_runtime_state<'db>(
    db: &'db dyn MirDb,
    raw_body: &SemanticBody<'db>,
    body: &NormalizedSemanticBody<'db>,
    params: &[RuntimeClass<'db>],
    param_locals: &[SLocalId],
    scope: Option<hir::hir_def::scope_graph::ScopeId<'db>>,
    assumptions: PredicateListId<'db>,
) -> Vec<InferredRuntimeLocal<'db>> {
    let mut carriers = vec![RuntimeCarrier::Erased; body.locals.len()];
    for (class, local) in params.iter().zip(param_locals.iter().copied()) {
        carriers[local.index()] = RuntimeCarrier::Value(class.clone());
    }

    loop {
        let mut changed = false;
        for block in &body.blocks {
            for stmt in &block.stmts {
                let NSStmtKind::Assign { dst, expr } = &stmt.kind else {
                    continue;
                };
                if matches!(carriers[dst.index()], RuntimeCarrier::Value(_)) {
                    continue;
                }
                let desired = match &body.locals[dst.index()] {
                    NSLocal {
                        ty,
                        mutability: Mutability::Mutable,
                        source: Some(_),
                        ..
                    } if matches!(
                        top_level_class_for_ty_in_context(
                            db,
                            *ty,
                            AddressSpaceKind::Memory,
                            scope,
                            assumptions,
                        ),
                        Some(RuntimeClass::AggregateValue { .. })
                    ) =>
                    {
                        let ty = runtime_repr_ty_in_context(db, *ty, scope, assumptions);
                        RuntimeCarrier::Value(RuntimeClass::Handle {
                            layout: layout_for_ty_in_context(db, ty, scope, assumptions),
                            kind: HandleKind::ObjectValue,
                            view: HandleView::Whole,
                        })
                    }
                    local => {
                        match expr_direct_class(db, body, raw_body, expr, local.ty, &carriers) {
                            Some(class) => RuntimeCarrier::Value(class),
                            None => continue,
                        }
                    }
                };
                carriers[dst.index()] = desired;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }

    for (idx, local) in raw_body.locals.iter().enumerate() {
        let SemanticLocalRole::DirectValue {
            provenance: ValueProvenance::RootProvider(provider),
        } = &local.role
        else {
            continue;
        };
        if !local
            .source
            .is_some_and(owner_effect_binding_source_is_runtime_root)
            || !matches!(carriers[idx], RuntimeCarrier::Erased)
        {
            continue;
        }
        let class = runtime_class_for_effect_binding_provider_in_context(
            db, provider, scope, assumptions,
        )
        .or_else(|| runtime_class_for_direct_value_provider_in_context(
            db, provider, scope, assumptions,
        ))
        .unwrap_or_else(|| {
            panic!(
                "missing runtime provider class for root-provider direct value local SLocalId({idx}): {provider:?}"
            )
        });
        carriers[idx] = RuntimeCarrier::Value(class);
    }

    let mut needs_root = vec![false; body.locals.len()];
    for block in &body.blocks {
        for stmt in &block.stmts {
            match &stmt.kind {
                NSStmtKind::Assign { expr, .. } => {
                    mark_expr_place_roots(expr, &mut needs_root, body)
                }
                NSStmtKind::Store { dst, .. } => mark_place_root(dst, &mut needs_root, body),
            }
        }
    }

    for local in &body.locals {
        if let NSLocal {
            lowering:
                hir::analysis::semantic::borrowck::NormalizedBindingLowering::ValueLocal { place },
            ..
        } = local
        {
            mark_place_root(place, &mut needs_root, body);
        }
    }

    body.locals
        .iter()
        .enumerate()
        .map(|(idx, local)| {
            let local_id = SLocalId::from_u32(idx as u32);
            let mut carrier = carriers[idx].clone();
            let role = &raw_body.locals[idx].role;
            let root = if !needs_root[idx] && !semantic_local_requires_runtime_root(role) {
                RuntimeLocalRoot::None
            } else {
                infer_runtime_local_root(
                    db,
                    raw_body,
                    local_id,
                    local.ty,
                    &mut carrier,
                    scope,
                    assumptions,
                )
            };
            InferredRuntimeLocal { carrier, root }
        })
        .collect()
}

fn semantic_local_requires_runtime_root(role: &SemanticLocalRole<'_>) -> bool {
    matches!(
        role,
        SemanticLocalRole::PlaceCarrier { .. } | SemanticLocalRole::PlaceBoundValue { .. }
    )
}

pub(super) fn lower_semantic_locals<'db>(
    db: &'db dyn MirDb,
    raw_body: &SemanticBody<'db>,
    body: &NormalizedSemanticBody<'db>,
    states: &[InferredRuntimeLocal<'db>],
    scope: Option<hir::hir_def::scope_graph::ScopeId<'db>>,
    assumptions: PredicateListId<'db>,
) -> (
    Vec<RuntimeLocalLowering<'db>>,
    Vec<RuntimeProviderBinding<'db>>,
) {
    let mut provider_bindings = Vec::new();
    for (idx, local) in raw_body.locals.iter().enumerate() {
        let local_id = SLocalId::from_u32(idx as u32);
        let Some((provider, provider_class, place_class)) = direct_value_provider_binding(
            db,
            local,
            inferred_carrier_value_class(local_id, states),
            scope,
            assumptions,
        ) else {
            continue;
        };
        if provider_bindings
            .iter()
            .any(|binding: &RuntimeProviderBinding<'db>| binding.provider == provider)
        {
            continue;
        }
        push_runtime_provider_binding(
            &mut provider_bindings,
            provider,
            local_id,
            provider_class,
            place_class,
        );
    }
    let lowerings = body
        .locals
        .iter()
        .enumerate()
        .map(|(idx, local)| {
            match &raw_body.locals[idx].role {
                SemanticLocalRole::Erased => RuntimeLocalLowering::Erased,
                SemanticLocalRole::DirectValue {
                    provenance: ValueProvenance::RootProvider(provider),
                } =>
                {
                    let provider = provider_bindings
                        .iter()
                        .enumerate()
                        .find_map(|(binding_idx, binding)| {
                            (binding.provider == *provider)
                                .then(|| RuntimeProviderBindingId::from_u32(binding_idx as u32))
                        })
                        .unwrap_or_else(|| {
                            panic!(
                                "missing runtime provider binding for root-provider direct value local {idx}: {provider:?}"
                            )
                        });
                    RuntimeLocalLowering::PlaceBoundValue {
                        provider,
                        place_class: provider_bindings[provider.index()].place_class.clone(),
                    }
                }
                SemanticLocalRole::DirectValue { .. } => RuntimeLocalLowering::DirectValue,
                SemanticLocalRole::PlaceCarrier { value_ty } => {
                    RuntimeLocalLowering::PlaceCarrier {
                        place_class: stored_class_for_ty_in_context(
                            db,
                            *value_ty,
                            scope,
                            assumptions,
                        ),
                    }
                }
                SemanticLocalRole::PlaceBoundValue { provider, value_ty } => {
                    let provider_class = inferred_carrier_value_class(
                        SLocalId::from_u32(idx as u32),
                        states,
                    )
                        .or_else(|| {
                            runtime_class_for_effect_binding_provider_in_context(
                                db,
                                provider,
                                scope,
                                assumptions,
                            )
                        })
                        .unwrap_or_else(|| {
                            panic!(
                                "missing provider runtime class for place-bound semantic local {idx}: {provider:?}"
                            )
                        });
                    let place_class = place_bound_value_class(
                        db,
                        *value_ty,
                        &provider_class,
                        scope,
                        assumptions,
                    );
                    let provider = push_runtime_provider_binding(
                        &mut provider_bindings,
                        provider.clone(),
                        SLocalId::from_u32(idx as u32),
                        provider_class,
                        place_class.clone(),
                    );
                    RuntimeLocalLowering::PlaceBoundValue {
                        provider,
                        place_class,
                    }
                }
                SemanticLocalRole::DirectCarrier {
                    provider,
                    target_ty,
                } => {
                    let place_class =
                        stored_class_for_ty_in_context(db, *target_ty, scope, assumptions);
                    let provider = provider.as_ref().map(|provider| {
                        let provider_class = inferred_carrier_value_class(
                            SLocalId::from_u32(idx as u32),
                            states,
                        )
                            .or_else(|| {
                                top_level_class_for_ty_in_context(
                                    db,
                                    local.ty,
                                    AddressSpaceKind::Memory,
                                    scope,
                                    assumptions,
                                )
                            })
                            .unwrap_or_else(|| {
                                panic!(
                                    "missing direct-carrier runtime class for semantic local {idx}: {}",
                                    local.ty.pretty_print(db),
                                )
                            });
                        push_runtime_provider_binding(
                            &mut provider_bindings,
                            provider.clone(),
                            SLocalId::from_u32(idx as u32),
                            provider_class,
                            place_class.clone(),
                        )
                    });
                    RuntimeLocalLowering::DirectCarrier {
                        provider,
                        place_class,
                    }
                }
            }
        })
        .collect();
    (lowerings, provider_bindings)
}

fn direct_value_provider_binding<'db>(
    db: &'db dyn MirDb,
    local: &hir::analysis::semantic::SLocal<'db>,
    inferred_provider_class: Option<RuntimeClass<'db>>,
    scope: Option<hir::hir_def::scope_graph::ScopeId<'db>>,
    assumptions: PredicateListId<'db>,
) -> Option<(ProviderBinding<'db>, RuntimeClass<'db>, RuntimeClass<'db>)> {
    let SemanticLocalRole::DirectValue {
        provenance: ValueProvenance::RootProvider(provider),
    } = &local.role
    else {
        return None;
    };
    let provider_class = inferred_provider_class
        .or_else(|| runtime_class_for_direct_value_provider_in_context(db, provider, scope, assumptions))
        .unwrap_or_else(|| {
            panic!(
                "missing runtime provider class for root-provider direct value local {:?}: {provider:?}",
                local.source
            )
        });
    let place_class = place_bound_value_class(db, local.ty, &provider_class, scope, assumptions);
    Some((provider.clone(), provider_class, place_class))
}

fn owner_effect_binding_source_is_runtime_root(source: LocalBinding<'_>) -> bool {
    matches!(
        source,
        LocalBinding::EffectParam { .. }
            | LocalBinding::Param {
                site: ParamSite::EffectField(_),
                ..
            }
    )
}

fn inferred_carrier_value_class<'db>(
    local: SLocalId,
    states: &[InferredRuntimeLocal<'db>],
) -> Option<RuntimeClass<'db>> {
    match &states.get(local.index())?.carrier {
        RuntimeCarrier::Erased => None,
        RuntimeCarrier::Value(class) => Some(class.clone()),
    }
}

fn mark_expr_place_roots<'db>(
    expr: &NExpr<'db>,
    needs_root: &mut [bool],
    body: &NormalizedSemanticBody<'db>,
) {
    match expr {
        NExpr::Use(_)
        | NExpr::Const(_)
        | NExpr::Unary { .. }
        | NExpr::Binary { .. }
        | NExpr::Cast { .. }
        | NExpr::AggregateMake { .. }
        | NExpr::EnumMake { .. }
        | NExpr::GetEnumTag { .. }
        | NExpr::IsEnumVariant { .. }
        | NExpr::ExtractEnumField { .. }
        | NExpr::CodeRegionOffset { .. }
        | NExpr::CodeRegionLen { .. } => {}
        NExpr::ReadPlace { place, .. } | NExpr::Borrow { place, .. } => {
            mark_place_root(place, needs_root, body);
        }
        NExpr::Call { effect_args, .. } => {
            for arg in effect_args {
                if let NEffectArgValue::Place(place) = &arg.arg {
                    mark_place_root(place, needs_root, body);
                }
            }
        }
    }
}

fn mark_place_root(
    place: &NSPlace<'_>,
    needs_root: &mut [bool],
    body: &NormalizedSemanticBody<'_>,
) {
    let local = match place.root {
        NSPlaceRoot::CarrierDerefLocal(local) => Some(local),
        NSPlaceRoot::Root(root) => match body.root(root) {
            Some(NBorrowRoot::Param { local, .. }) | Some(NBorrowRoot::LocalSlot { local }) => {
                Some(*local)
            }
            Some(NBorrowRoot::Provider { .. }) | None => None,
        },
    };
    if let Some(local) = local
        && let Some(entry) = needs_root.get_mut(local.index())
    {
        *entry = true;
    }
}

fn infer_runtime_local_root<'db>(
    db: &'db dyn MirDb,
    raw_body: &SemanticBody<'db>,
    local: SLocalId,
    ty: TyId<'db>,
    carrier: &mut RuntimeCarrier<'db>,
    scope: Option<hir::hir_def::scope_graph::ScopeId<'db>>,
    assumptions: PredicateListId<'db>,
) -> RuntimeLocalRoot<'db> {
    let role = &raw_body.locals[local.index()].role;
    let place_class = local_place_root_class(db, role, ty, carrier, scope, assumptions);
    let transport_class = match carrier {
        RuntimeCarrier::Value(class) => Some(class.clone()),
        RuntimeCarrier::Erased => fallback_root_transport_class(db, role, ty, scope, assumptions),
    };
    let Some(place_class) = place_class else {
        return RuntimeLocalRoot::None;
    };
    let Some(transport_class) = transport_class else {
        return RuntimeLocalRoot::Slot(place_class);
    };
    if matches!(
        (&*carrier, &transport_class),
        (
            RuntimeCarrier::Erased,
            RuntimeClass::RawAddr { .. } | RuntimeClass::Handle { .. }
        )
    ) {
        *carrier = RuntimeCarrier::Value(transport_class.clone());
    }
    match transport_class {
        RuntimeClass::RawAddr { space, .. } => RuntimeLocalRoot::Ptr {
            space,
            class: place_class,
        },
        RuntimeClass::Handle {
            kind: HandleKind::Provider { space, .. },
            ..
        } if space != AddressSpaceKind::Memory => RuntimeLocalRoot::Ptr {
            space,
            class: place_class,
        },
        RuntimeClass::Handle { .. } => RuntimeLocalRoot::Handle(transport_class),
        RuntimeClass::Scalar(_) | RuntimeClass::AggregateValue { .. } => {
            RuntimeLocalRoot::Slot(place_class)
        }
    }
}

fn local_place_root_class<'db>(
    db: &'db dyn MirDb,
    role: &SemanticLocalRole<'db>,
    ty: TyId<'db>,
    carrier: &RuntimeCarrier<'db>,
    scope: Option<hir::hir_def::scope_graph::ScopeId<'db>>,
    assumptions: PredicateListId<'db>,
) -> Option<RuntimeClass<'db>> {
    match role {
        SemanticLocalRole::Erased => None,
        SemanticLocalRole::DirectValue { .. } => Some(
            carrier_value_class_for_runtime(carrier)
                .filter(|class| !matches!(class, RuntimeClass::RawAddr { .. }))
                .unwrap_or_else(|| stored_class_for_ty_in_context(db, ty, scope, assumptions)),
        ),
        SemanticLocalRole::PlaceCarrier { value_ty } => Some(stored_class_for_ty_in_context(
            db,
            *value_ty,
            scope,
            assumptions,
        )),
        SemanticLocalRole::PlaceBoundValue { provider, value_ty } => Some(
            runtime_class_for_effect_binding_provider_in_context(db, provider, scope, assumptions)
                .map(|provider_class| {
                    place_bound_value_class(db, *value_ty, &provider_class, scope, assumptions)
                })
                .unwrap_or_else(|| {
                    stored_class_for_ty_in_context(db, *value_ty, scope, assumptions)
                }),
        ),
        SemanticLocalRole::DirectCarrier { target_ty, .. } => Some(stored_class_for_ty_in_context(
            db,
            *target_ty,
            scope,
            assumptions,
        )),
    }
}

fn fallback_root_transport_class<'db>(
    db: &'db dyn MirDb,
    role: &SemanticLocalRole<'db>,
    ty: TyId<'db>,
    scope: Option<hir::hir_def::scope_graph::ScopeId<'db>>,
    assumptions: PredicateListId<'db>,
) -> Option<RuntimeClass<'db>> {
    match role {
        SemanticLocalRole::Erased | SemanticLocalRole::DirectValue { .. } => None,
        SemanticLocalRole::PlaceBoundValue { .. } => None,
        SemanticLocalRole::PlaceCarrier { value_ty } => {
            top_level_class_for_ty_in_context(db, ty, AddressSpaceKind::Memory, scope, assumptions)
                .or_else(|| {
                    Some(provider_class_for_target_in_context(
                        db,
                        Some(*value_ty),
                        AddressSpaceKind::Memory,
                        scope,
                        assumptions,
                    ))
                })
        }
        SemanticLocalRole::DirectCarrier {
            provider,
            target_ty,
        } => provider
            .as_ref()
            .and_then(|provider| {
                runtime_class_for_provider_binding(db, provider, scope, assumptions)
            })
            .or_else(|| {
                top_level_class_for_ty_in_context(
                    db,
                    ty,
                    AddressSpaceKind::Memory,
                    scope,
                    assumptions,
                )
            })
            .or_else(|| {
                Some(provider_class_for_target_in_context(
                    db,
                    Some(*target_ty),
                    AddressSpaceKind::Memory,
                    scope,
                    assumptions,
                ))
            }),
    }
}

fn carrier_value_class_for_runtime<'db>(
    carrier: &RuntimeCarrier<'db>,
) -> Option<RuntimeClass<'db>> {
    match carrier {
        RuntimeCarrier::Erased => None,
        RuntimeCarrier::Value(class) => Some(class.clone()),
    }
}

fn place_bound_value_class<'db>(
    db: &'db dyn MirDb,
    value_ty: TyId<'db>,
    provider_class: &RuntimeClass<'db>,
    scope: Option<hir::hir_def::scope_graph::ScopeId<'db>>,
    assumptions: PredicateListId<'db>,
) -> RuntimeClass<'db> {
    match provider_class {
        RuntimeClass::Handle {
            layout,
            kind: HandleKind::Provider { .. },
            ..
        }
        | RuntimeClass::RawAddr {
            target: Some(layout),
            ..
        } => RuntimeClass::AggregateValue { layout: *layout },
        RuntimeClass::Scalar(_)
        | RuntimeClass::AggregateValue { .. }
        | RuntimeClass::RawAddr { target: None, .. }
        | RuntimeClass::Handle {
            kind: HandleKind::ConstValue | HandleKind::ObjectValue,
            ..
        } => stored_class_for_ty_in_context(db, value_ty, scope, assumptions),
    }
}

pub(crate) fn runtime_param_locals<'db>(
    db: &'db dyn MirDb,
    semantic: SemanticInstance<'db>,
    params: &[RuntimeClass<'db>],
) -> Vec<SLocalId> {
    let owner = semantic.key(db).owner(db);
    let typed_body = semantic.key(db).instantiate_typed_body(db);
    let mut explicit_bindings = Vec::new();
    let mut idx = 0;
    while let Some(binding) = typed_body.param_binding(idx) {
        explicit_bindings.push(binding);
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
        for arg_binding in arm.arg_bindings(db) {
            let Some(binding) = typed_body.pat_binding(arg_binding.pat) else {
                continue;
            };
            explicit_bindings.push(binding);
        }
    }
    explicit_bindings.extend(owner_effect_bindings(db, owner));
    if explicit_bindings.len() < params.len() {
        panic!(
            "failed to map runtime params to semantic locals for {:?}: mapped {} of {}; params={params:?}",
            semantic.key(db),
            explicit_bindings.len(),
            params.len(),
        );
    }
    explicit_bindings.truncate(params.len());
    explicit_bindings
        .into_iter()
        .map(|binding| runtime_visible_binding_local(db, owner, &typed_body, binding))
        .collect()
}

pub(crate) fn runtime_class_for_provider_binding<'db>(
    db: &'db dyn MirDb,
    provider: &ProviderBinding<'db>,
    scope: Option<hir::hir_def::scope_graph::ScopeId<'db>>,
    assumptions: PredicateListId<'db>,
) -> Option<RuntimeClass<'db>> {
    match provider.semantics.kind {
        ProviderKind::RootObject => top_level_class_for_ty_in_context(
            db,
            provider.provider_ty,
            AddressSpaceKind::Memory,
            scope,
            assumptions,
        ),
        ProviderKind::Handle | ProviderKind::RawAddress => {
            Some(provider_class_for_target_in_context(
                db,
                provider.semantics.target_ty,
                provider_address_space_to_runtime(provider.semantics.address_space?),
                scope,
                assumptions,
            ))
        }
    }
}

pub(crate) fn runtime_class_for_effect_binding_provider_in_context<'db>(
    db: &'db dyn MirDb,
    provider: &ProviderBinding<'db>,
    scope: Option<hir::hir_def::scope_graph::ScopeId<'db>>,
    assumptions: PredicateListId<'db>,
) -> Option<RuntimeClass<'db>> {
    match provider.semantics.kind {
        ProviderKind::RootObject => Some(provider_class_for_target_in_context(
            db,
            Some(provider.provider_ty),
            AddressSpaceKind::Memory,
            scope,
            assumptions,
        )),
        ProviderKind::Handle | ProviderKind::RawAddress => {
            runtime_class_for_provider_binding(db, provider, scope, assumptions)
        }
    }
}

pub(crate) fn runtime_class_for_effect_binding_provider_in_env<'db>(
    db: &'db dyn MirDb,
    env: RuntimeTypeEnv<'db>,
    provider: &ProviderBinding<'db>,
) -> Option<RuntimeClass<'db>> {
    runtime_class_for_effect_binding_provider_in_context(db, provider, env.scope, env.assumptions)
}

pub(crate) fn runtime_class_for_direct_value_provider_in_context<'db>(
    db: &'db dyn MirDb,
    provider: &ProviderBinding<'db>,
    scope: Option<hir::hir_def::scope_graph::ScopeId<'db>>,
    assumptions: PredicateListId<'db>,
) -> Option<RuntimeClass<'db>> {
    runtime_class_for_effect_binding_provider_in_context(db, provider, scope, assumptions)
}

pub(crate) fn runtime_class_for_direct_value_provider_in_env<'db>(
    db: &'db dyn MirDb,
    env: RuntimeTypeEnv<'db>,
    provider: &ProviderBinding<'db>,
) -> Option<RuntimeClass<'db>> {
    runtime_class_for_direct_value_provider_in_context(db, provider, env.scope, env.assumptions)
}

pub(crate) fn runtime_visible_binding_class<'db>(
    db: &'db dyn MirDb,
    semantic: SemanticInstance<'db>,
    binding: LocalBinding<'db>,
) -> Option<RuntimeClass<'db>> {
    let owner = semantic.key(db).owner(db);
    let typed_body = semantic.key(db).instantiate_typed_body(db);
    let env = RuntimeTypeEnv::new(Some(owner.scope()), typed_body.assumptions());
    let binding_ty = semantic_binding_ty(db, semantic, binding);
    match semantic_binding_lowering(db, semantic, binding) {
        hir::analysis::semantic::SemanticBindingLowering::Erased => None,
        hir::analysis::semantic::SemanticBindingLowering::DirectValue {
            provenance: ValueProvenance::RootProvider(provider),
        } => runtime_class_for_direct_value_provider_in_env(db, env, &provider),
        hir::analysis::semantic::SemanticBindingLowering::DirectValue { .. } => {
            runtime_class_for_explicit_root_provider_param(db, env, binding, binding_ty).or_else(
                || top_level_class_for_ty_in_env(db, env, binding_ty, AddressSpaceKind::Memory),
            )
        }
        hir::analysis::semantic::SemanticBindingLowering::DirectCarrier {
            provider: Some(provider),
            ..
        } => runtime_class_for_provider_binding(db, &provider, env.scope, env.assumptions),
        hir::analysis::semantic::SemanticBindingLowering::DirectCarrier {
            provider: None,
            target_ty,
        } => top_level_class_for_ty_in_env(db, env, binding_ty, AddressSpaceKind::Memory).or_else(
            || {
                Some(provider_class_for_target_in_env(
                    db,
                    env,
                    Some(target_ty),
                    AddressSpaceKind::Memory,
                ))
            },
        ),
        hir::analysis::semantic::SemanticBindingLowering::PlaceCarrier { value_ty } => Some(
            provider_class_for_target_in_env(db, env, Some(value_ty), AddressSpaceKind::Memory),
        ),
        hir::analysis::semantic::SemanticBindingLowering::PlaceBoundValue { provider, .. } => {
            runtime_class_for_effect_binding_provider_in_env(db, env, &provider)
        }
    }
}

fn runtime_visible_binding_local<'db>(
    db: &'db dyn MirDb,
    owner: BodyOwner<'db>,
    typed_body: &hir::analysis::ty::ty_check::TypedBody<'db>,
    binding: LocalBinding<'db>,
) -> SLocalId {
    let mut next = 0u32;
    let mut param_idx = 0;
    while let Some(param_binding) = typed_body.param_binding(param_idx) {
        if param_binding == binding {
            return SLocalId::from_u32(next);
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
                return SLocalId::from_u32(next);
            }
            next += 1;
        }
    }
    for effect_binding in owner_effect_bindings(db, owner) {
        if same_owner_effect_binding(effect_binding, binding) {
            return SLocalId::from_u32(next);
        }
        next += 1;
    }
    panic!("missing semantic local for runtime-visible binding {binding:?}")
}

fn runtime_class_for_explicit_root_provider_param<'db>(
    db: &'db dyn MirDb,
    env: RuntimeTypeEnv<'db>,
    binding: LocalBinding<'db>,
    binding_ty: TyId<'db>,
) -> Option<RuntimeClass<'db>> {
    let LocalBinding::Param {
        site: ParamSite::Func(func),
        idx,
        ..
    } = binding
    else {
        return None;
    };
    if !func
        .params(db)
        .nth(idx)
        .is_some_and(|param| param.is_self_param(db))
    {
        return None;
    }
    let canonical = |ty| {
        strip_derived_adt_layout_args(
            db,
            runtime_repr_ty_in_context(
                db,
                env.scope
                    .map_or(ty, |scope| normalize_ty(db, ty, scope, env.assumptions)),
                env.scope,
                env.assumptions,
            ),
        )
    };
    let binding_ty = canonical(binding_ty);
    registered_root_providers(db, EffectParamSite::Func(func))
        .into_iter()
        .find(|provider| canonical(provider.provider_ty) == binding_ty)
        .map(|provider| {
            provider_class_for_target_in_env(
                db,
                env,
                Some(provider.provider_ty),
                AddressSpaceKind::Memory,
            )
        })
}

fn expr_direct_class<'db>(
    db: &'db dyn MirDb,
    body: &NormalizedSemanticBody<'db>,
    raw_body: &SemanticBody<'db>,
    expr: &NExpr<'db>,
    result_ty: TyId<'db>,
    carriers: &[RuntimeCarrier<'db>],
) -> Option<RuntimeClass<'db>> {
    let owner = body.owner.key(db).owner(db);
    let typed_body = body.owner.key(db).instantiate_typed_body(db);
    let env = RuntimeTypeEnv::new(Some(owner.scope()), typed_body.assumptions());
    Some(match expr {
        NExpr::Use(value) => semantic_value_class(db, body, raw_body, value.local, carriers)?,
        NExpr::Const(const_) => match const_ {
            SConst::Value(value) => {
                let ty = sem_const_ty(db, *value);
                if ty == TyId::unit(db) {
                    return None;
                }
                top_level_class_for_ty_in_env(db, env, ty, AddressSpaceKind::Memory)?
            }
            SConst::Ref(cref) => {
                panic!("unresolved const ref reached runtime class inference: {cref:?}")
            }
        },
        NExpr::Unary { .. }
        | NExpr::Binary { .. }
        | NExpr::Cast { .. }
        | NExpr::CodeRegionOffset { .. }
        | NExpr::CodeRegionLen { .. }
        | NExpr::GetEnumTag { .. } => {
            RuntimeClass::Scalar(scalar_class_for_ty_in_env(db, env, result_ty)?)
        }
        NExpr::AggregateMake { ty, .. } => {
            top_level_class_for_ty_in_env(db, env, *ty, AddressSpaceKind::Memory)?
        }
        NExpr::EnumMake { enum_ty, .. } => RuntimeClass::AggregateValue {
            layout: layout_for_ty_in_env(db, env, *enum_ty),
        },
        NExpr::ReadPlace { place, .. } => {
            normalized_place_class(db, body, raw_body, place, carriers).or_else(|| {
                top_level_class_for_ty_in_env(db, env, result_ty, AddressSpaceKind::Memory)
            })?
        }
        NExpr::ExtractEnumField { .. } => {
            top_level_class_for_ty_in_env(db, env, result_ty, AddressSpaceKind::Memory)?
        }
        NExpr::Borrow { provider, .. } => provider_class_for_target_in_env(
            db,
            env,
            Some(
                result_ty
                    .as_borrow(db)
                    .map_or(result_ty, |(_, inner)| inner),
            ),
            provider.map_or(AddressSpaceKind::Memory, address_space_from_provider),
        ),
        NExpr::IsEnumVariant { .. } => RuntimeClass::Scalar(ScalarClass {
            repr: ScalarRepr::Bool,
            role: ScalarRole::Plain,
        }),
        NExpr::Call {
            callee,
            args,
            effect_args,
        } => {
            let callee_key = resolve_runtime_call_key(
                db,
                body.owner.key(db),
                &typed_body,
                raw_body,
                *callee,
                args,
            );
            let semantic = get_or_build_semantic_instance(db, callee_key);
            if let Some(class) = extern_builtin_return_class(db, semantic, result_ty) {
                return class;
            }
            let typed_body = semantic.key(db).instantiate_typed_body(db);
            let env = RuntimeTypeEnv::new(
                typed_body.body().map(|body| body.scope()),
                typed_body.assumptions(),
            );
            let mut param_classes = Vec::new();
            for (idx, arg) in args.iter().enumerate() {
                let desired = desired_runtime_param_class(db, &typed_body, idx);
                let actual = runtime_visible_arg_class(
                    db,
                    body,
                    raw_body,
                    arg.local,
                    desired.as_ref(),
                    carriers,
                );
                if let Some(actual) = actual {
                    param_classes.push(desired.map_or(actual.clone(), |desired| {
                        preserve_provider_space(&actual, &desired)
                    }));
                }
            }
            for arg in effect_args {
                if let Some(class) = effect_arg_class(db, env, arg, carriers) {
                    param_classes.push(class);
                }
            }
            for binding in owner_effect_bindings(db, semantic.key(db).owner(db)) {
                if let Some(class) = owner_effect_arg_class(body, carriers, binding) {
                    param_classes.push(class);
                }
            }
            runtime_return_class_for_key(
                db,
                RuntimeInstanceKey::new(
                    db,
                    RuntimeInstanceSource::Semantic(semantic),
                    param_classes,
                ),
            )?
        }
    })
}

fn runtime_visible_arg_class<'db>(
    db: &'db dyn MirDb,
    body: &NormalizedSemanticBody<'db>,
    raw_body: &SemanticBody<'db>,
    local: SLocalId,
    desired: Option<&RuntimeClass<'db>>,
    carriers: &[RuntimeCarrier<'db>],
) -> Option<RuntimeClass<'db>> {
    if matches!(
        desired,
        Some(
            RuntimeClass::Handle {
                kind: HandleKind::Provider { .. },
                ..
            } | RuntimeClass::RawAddr { .. }
        )
    ) {
        return carrier_value_class(local, carriers);
    }
    semantic_value_class(db, body, raw_body, local, carriers)
}

pub(crate) fn desired_runtime_param_class<'db>(
    db: &'db dyn MirDb,
    typed_body: &hir::analysis::ty::ty_check::TypedBody<'db>,
    idx: usize,
) -> Option<RuntimeClass<'db>> {
    let binding = typed_body.param_binding(idx)?;
    let binding_ty = typed_body.binding_ty(db, binding);
    let scope = typed_body.body().map(|body| body.scope());
    let assumptions = typed_body.assumptions();
    let repr_ty = runtime_repr_ty_in_context(db, binding_ty, scope, assumptions);
    if runtime_abstract_param_ty(db, binding_ty, scope, assumptions)
        || matches!(
            repr_ty.base_ty(db).data(db),
            TyData::TyParam(param) if param.is_effect() || param.is_effect_provider()
        )
    {
        return None;
    }
    top_level_class_for_ty_in_context(db, binding_ty, AddressSpaceKind::Memory, scope, assumptions)
        .map(|class| runtime_param_class(db, typed_body, binding, class))
}

pub(crate) fn resolve_runtime_call_key<'db>(
    db: &'db dyn MirDb,
    caller_key: SemanticInstanceKey<'db>,
    caller_typed_body: &hir::analysis::ty::ty_check::TypedBody<'db>,
    raw_body: &SemanticBody<'db>,
    callee: SemanticCalleeRef<'db>,
    args: &[NOperand],
) -> SemanticInstanceKey<'db> {
    let callee_key = callee.key;
    let BodyOwner::Func(func) = callee_key.owner(db) else {
        return callee_key;
    };
    let Some(trait_) = func.containing_trait(db) else {
        return callee_key;
    };
    if func.body(db).is_some() {
        return callee_key;
    }
    let Some(method_name) = func.name(db).to_opt() else {
        return callee_key;
    };
    let impl_env = callee_key.impl_env(db);
    let original_inst: Option<TraitInstId<'db>> = impl_env
        .witnesses(db)
        .iter()
        .find(|inst| inst.def(db) == trait_)
        .copied();
    let concrete_inst = if func
        .params(db)
        .next()
        .is_some_and(|param| param.is_self_param(db))
    {
        let Some(arg) = args.first() else {
            return callee_key;
        };
        let Some(self_ty) =
            concrete_runtime_self_ty_for_call_arg(db, caller_typed_body, raw_body, arg.local)
        else {
            return callee_key;
        };
        let mut inst_args = original_inst
            .map(|inst| inst.args(db).to_vec())
            .unwrap_or_else(|| vec![self_ty]);
        let Some(first) = inst_args.first_mut() else {
            return callee_key;
        };
        *first = self_ty;
        TraitInstId::new(
            db,
            trait_,
            inst_args,
            original_inst
                .map(|inst| inst.assoc_type_bindings(db).clone())
                .unwrap_or_default(),
        )
    } else {
        let Some(original_inst) = original_inst else {
            return callee_key;
        };
        original_inst
    };
    let assumptions = runtime_callee_assumptions(db, caller_key, caller_typed_body);
    let Some((impl_func, mut impl_args)) = resolve_trait_method_instance(
        db,
        TraitSolveCx::new(db, caller_key.impl_env(db).normalization_scope(db))
            .with_assumptions(assumptions),
        concrete_inst,
        method_name,
    ) else {
        return callee_key;
    };
    let trait_arg_len = concrete_inst.args(db).len();
    let tail = callee_key
        .subst(db)
        .generic_args(db)
        .get(trait_arg_len..)
        .unwrap_or(callee_key.subst(db).generic_args(db).as_slice());
    impl_args.extend_from_slice(tail);
    let mut witnesses = IndexSet::new();
    witnesses.extend(caller_key.impl_env(db).witnesses(db).iter().copied());
    witnesses.extend(impl_env.witnesses(db).iter().copied());
    witnesses.insert(concrete_inst);
    SemanticInstanceKey::new(
        db,
        BodyOwner::Func(impl_func),
        GenericSubst::new(db, impl_args),
        ImplEnv::new(
            db,
            caller_key.impl_env(db).normalization_scope(db),
            assumptions,
            witnesses.into_iter().collect::<Vec<_>>(),
        ),
    )
}

pub(crate) fn runtime_callee_for_semantic_call<'db>(
    db: &'db dyn MirDb,
    cx: RuntimeSemanticCallContext<'_, 'db>,
    callee: SemanticCalleeRef<'db>,
    args: &[NOperand],
    effect_args: &[NEffectArg<'db>],
) -> Option<crate::instance::RuntimeInstance<'db>> {
    let caller_key = cx.caller.key(db);
    let caller_typed_body = caller_key.instantiate_typed_body(db);
    let callee_key = resolve_runtime_call_key(
        db,
        caller_key,
        &caller_typed_body,
        cx.raw_body,
        callee,
        args,
    );
    let semantic = get_or_build_semantic_instance(db, callee_key);
    if extern_builtin_return_class(db, semantic, cx.result_ty).is_some() {
        return None;
    }

    let typed_body = semantic.key(db).instantiate_typed_body(db);
    let env = RuntimeTypeEnv::new(
        typed_body.body().map(|body| body.scope()),
        typed_body.assumptions(),
    );
    let mut param_classes = Vec::new();
    for (idx, arg) in args.iter().enumerate() {
        let desired = desired_runtime_param_class(db, &typed_body, idx);
        let actual = runtime_visible_arg_class(
            db,
            cx.body,
            cx.raw_body,
            arg.local,
            desired.as_ref(),
            cx.carriers,
        );
        if let Some(actual) = actual {
            param_classes.push(desired.map_or(actual.clone(), |desired| {
                preserve_provider_space(&actual, &desired)
            }));
        }
    }
    for arg in effect_args {
        if let Some(class) = effect_arg_class(db, env, arg, cx.carriers) {
            param_classes.push(class);
        }
    }
    for binding in owner_effect_bindings(db, semantic.key(db).owner(db)) {
        if let Some(class) = owner_effect_arg_class(cx.body, cx.carriers, binding) {
            param_classes.push(class);
        }
    }

    Some(get_or_build_runtime_instance(
        db,
        RuntimeInstanceKey::new(db, RuntimeInstanceSource::Semantic(semantic), param_classes),
    ))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum GenericNumericIntrinsicKind {
    Bitcast,
    Saturating(SaturatingBinOp),
    CheckedBinary(ArithBinOp),
    CheckedNeg,
}

fn runtime_callee_assumptions<'db>(
    db: &'db dyn MirDb,
    caller_key: SemanticInstanceKey<'db>,
    caller_typed_body: &hir::analysis::ty::ty_check::TypedBody<'db>,
) -> PredicateListId<'db> {
    let impl_env = caller_key.impl_env(db);
    let mut predicates: IndexSet<_> = caller_typed_body
        .assumptions()
        .list(db)
        .iter()
        .copied()
        .collect();
    predicates.extend(impl_env.assumptions(db).list(db).iter().copied());
    predicates.extend(impl_env.witnesses(db).iter().copied());
    PredicateListId::new(db, predicates.into_iter().collect::<Vec<_>>())
}

fn concrete_runtime_self_ty_for_call_arg<'db>(
    db: &'db dyn MirDb,
    caller_typed_body: &hir::analysis::ty::ty_check::TypedBody<'db>,
    raw_body: &SemanticBody<'db>,
    local: SLocalId,
) -> Option<TyId<'db>> {
    let scope = caller_typed_body.body().map(|body| body.scope());
    let assumptions = caller_typed_body.assumptions();
    let normalized = |ty| normalize_runtime_self_ty(db, ty, scope, assumptions);
    match &raw_body.locals.get(local.index())?.role {
        SemanticLocalRole::Erased => None,
        SemanticLocalRole::DirectValue {
            provenance: ValueProvenance::RootProvider(provider),
        }
        | SemanticLocalRole::PlaceBoundValue { provider, .. }
        | SemanticLocalRole::DirectCarrier {
            provider: Some(provider),
            ..
        } => Some(normalized(provider.provider_ty)),
        SemanticLocalRole::DirectValue { .. } => {
            Some(normalized(raw_body.locals[local.index()].ty))
        }
        SemanticLocalRole::PlaceCarrier { value_ty } => Some(normalized(*value_ty)),
        SemanticLocalRole::DirectCarrier {
            provider: None,
            target_ty,
        } => Some(normalized(*target_ty)),
    }
}

fn normalize_runtime_self_ty<'db>(
    db: &'db dyn MirDb,
    ty: TyId<'db>,
    scope: Option<hir::hir_def::scope_graph::ScopeId<'db>>,
    assumptions: PredicateListId<'db>,
) -> TyId<'db> {
    let ty = runtime_repr_ty_in_context(db, ty, scope, assumptions);
    if let Some((_, inner)) = ty.as_borrow(db) {
        return scope.map_or(inner, |scope| normalize_ty(db, inner, scope, assumptions));
    }
    scope.map_or(ty, |scope| normalize_ty(db, ty, scope, assumptions))
}

fn preserve_provider_space<'db>(
    actual: &RuntimeClass<'db>,
    desired: &RuntimeClass<'db>,
) -> RuntimeClass<'db> {
    match (actual, desired) {
        (
            RuntimeClass::Handle {
                layout: actual_layout,
                kind:
                    HandleKind::Provider {
                        provider_ty: actual_provider_ty,
                        space: actual_space,
                    },
                view: HandleView::Whole,
            },
            RuntimeClass::Handle {
                layout: desired_layout,
                kind:
                    HandleKind::Provider {
                        provider_ty: desired_provider_ty,
                        ..
                    },
                view: HandleView::Whole,
            },
        ) if actual_layout == desired_layout && actual_provider_ty == desired_provider_ty => {
            RuntimeClass::Handle {
                layout: *actual_layout,
                kind: HandleKind::Provider {
                    provider_ty: *actual_provider_ty,
                    space: *actual_space,
                },
                view: HandleView::Whole,
            }
        }
        (
            RuntimeClass::RawAddr {
                space: actual_space,
                target: actual_target,
            },
            RuntimeClass::RawAddr {
                target: desired_target,
                ..
            },
        ) if actual_target == desired_target => RuntimeClass::RawAddr {
            space: *actual_space,
            target: *actual_target,
        },
        _ => desired.clone(),
    }
}

fn semantic_value_class<'db>(
    db: &'db dyn MirDb,
    body: &NormalizedSemanticBody<'db>,
    raw_body: &SemanticBody<'db>,
    local: SLocalId,
    carriers: &[RuntimeCarrier<'db>],
) -> Option<RuntimeClass<'db>> {
    let owner = body.owner.key(db).owner(db);
    let typed_body = body.owner.key(db).instantiate_typed_body(db);
    let scope = Some(owner.scope());
    let assumptions = typed_body.assumptions();
    match &raw_body.locals.get(local.index())?.role {
        SemanticLocalRole::Erased => None,
        SemanticLocalRole::DirectValue { .. } | SemanticLocalRole::DirectCarrier { .. } => {
            carrier_value_class(local, carriers)
        }
        SemanticLocalRole::PlaceCarrier { value_ty } => Some(stored_class_for_ty_in_context(
            db,
            *value_ty,
            scope,
            assumptions,
        )),
        SemanticLocalRole::PlaceBoundValue { provider, value_ty } => {
            Some(semantic_place_bound_class(
                db,
                local,
                provider,
                *value_ty,
                carriers,
                scope,
                assumptions,
            ))
        }
    }
}

fn carrier_value_class<'db>(
    local: SLocalId,
    carriers: &[RuntimeCarrier<'db>],
) -> Option<RuntimeClass<'db>> {
    match carriers.get(local.index())? {
        RuntimeCarrier::Erased => None,
        RuntimeCarrier::Value(class) => Some(class.clone()),
    }
}

fn semantic_place_bound_class<'db>(
    db: &'db dyn MirDb,
    local: SLocalId,
    provider: &ProviderBinding<'db>,
    value_ty: TyId<'db>,
    carriers: &[RuntimeCarrier<'db>],
    scope: Option<hir::hir_def::scope_graph::ScopeId<'db>>,
    assumptions: PredicateListId<'db>,
) -> RuntimeClass<'db> {
    carrier_value_class(local, carriers)
        .or_else(|| {
            runtime_class_for_effect_binding_provider_in_context(db, provider, scope, assumptions)
        })
        .map(|provider_class| {
            place_bound_value_class(db, value_ty, &provider_class, scope, assumptions)
        })
        .unwrap_or_else(|| stored_class_for_ty_in_context(db, value_ty, scope, assumptions))
}

fn normalized_place_class<'db>(
    db: &'db dyn MirDb,
    body: &NormalizedSemanticBody<'db>,
    raw_body: &SemanticBody<'db>,
    place: &NSPlace<'db>,
    carriers: &[RuntimeCarrier<'db>],
) -> Option<RuntimeClass<'db>> {
    let owner = body.owner.key(db).owner(db);
    let typed_body = body.owner.key(db).instantiate_typed_body(db);
    let scope = Some(owner.scope());
    let assumptions = typed_body.assumptions();
    let mut current = normalized_place_root_class(
        db,
        body,
        raw_body,
        place.root.clone(),
        carriers,
        scope,
        assumptions,
    )?;
    for projection in place.path.iter() {
        current = match projection {
            Projection::Field(field) => project_field_class(
                db,
                current,
                FieldIndex((*field).try_into().expect("field index fits")),
            ),
            Projection::Index(_) => project_index_class(db, current),
            Projection::Deref => match current {
                RuntimeClass::Handle { layout, .. }
                | RuntimeClass::RawAddr {
                    target: Some(layout),
                    ..
                } => RuntimeClass::AggregateValue { layout },
                RuntimeClass::AggregateValue { .. }
                | RuntimeClass::Scalar(_)
                | RuntimeClass::RawAddr { target: None, .. } => {
                    panic!("invalid deref projection class")
                }
            },
            Projection::VariantField {
                variant, field_idx, ..
            } => project_variant_field_place_class(
                db,
                current,
                *variant,
                FieldIndex((*field_idx).try_into().expect("field index fits")),
            ),
            Projection::Discriminant => match current {
                RuntimeClass::AggregateValue { layout }
                | RuntimeClass::Handle { layout, .. }
                | RuntimeClass::RawAddr {
                    target: Some(layout),
                    ..
                } => match layout.data(db) {
                    Layout::Enum(layout) => RuntimeClass::Scalar(layout.tag),
                    Layout::Struct(_) | Layout::Array(_) => {
                        panic!("invalid discriminant projection class")
                    }
                },
                RuntimeClass::Scalar(_) | RuntimeClass::RawAddr { target: None, .. } => {
                    panic!("invalid discriminant projection class")
                }
            },
        };
    }
    Some(current)
}

fn normalized_place_root_class<'db>(
    db: &'db dyn MirDb,
    body: &NormalizedSemanticBody<'db>,
    raw_body: &SemanticBody<'db>,
    root: NSPlaceRoot,
    carriers: &[RuntimeCarrier<'db>],
    scope: Option<hir::hir_def::scope_graph::ScopeId<'db>>,
    assumptions: PredicateListId<'db>,
) -> Option<RuntimeClass<'db>> {
    match root {
        NSPlaceRoot::CarrierDerefLocal(local) => {
            Some(match &body.locals.get(local.index())?.lowering {
                hir::analysis::semantic::borrowck::NormalizedBindingLowering::CarrierLocal {
                    target_ty,
                    ..
                } => stored_class_for_ty_in_context(db, *target_ty, scope, assumptions),
                hir::analysis::semantic::borrowck::NormalizedBindingLowering::Erased
                | hir::analysis::semantic::borrowck::NormalizedBindingLowering::ValueLocal {
                    ..
                }
                | hir::analysis::semantic::borrowck::NormalizedBindingLowering::PlaceBoundValue {
                    ..
                } => local_place_root_class(
                    db,
                    &raw_body.locals.get(local.index())?.role,
                    body.locals.get(local.index())?.ty,
                    carriers.get(local.index())?,
                    scope,
                    assumptions,
                )?,
            })
        }
        NSPlaceRoot::Root(root) => match body.root(root)? {
            NBorrowRoot::Param { local, .. } | NBorrowRoot::LocalSlot { local } => {
                local_place_root_class(
                    db,
                    &raw_body.locals.get(local.index())?.role,
                    body.locals.get(local.index())?.ty,
                    carriers.get(local.index())?,
                    scope,
                    assumptions,
                )
            }
            NBorrowRoot::Provider { binding } => {
                let provider_class = runtime_class_for_effect_binding_provider_in_context(
                    db,
                    binding,
                    scope,
                    assumptions,
                )
                .or_else(|| {
                    runtime_class_for_direct_value_provider_in_context(
                        db,
                        binding,
                        scope,
                        assumptions,
                    )
                })?;
                Some(place_bound_value_class(
                    db,
                    binding.provider_ty,
                    &provider_class,
                    scope,
                    assumptions,
                ))
            }
        },
    }
}

fn project_variant_field_place_class<'db>(
    db: &'db dyn MirDb,
    class: RuntimeClass<'db>,
    variant: VariantIndex,
    field: FieldIndex,
) -> RuntimeClass<'db> {
    let layout = match class {
        RuntimeClass::AggregateValue { layout } | RuntimeClass::Handle { layout, .. } => layout,
        RuntimeClass::RawAddr {
            target: Some(layout),
            ..
        } => layout,
        RuntimeClass::Scalar(_) | RuntimeClass::RawAddr { target: None, .. } => {
            panic!("invalid variant-field projection class")
        }
    };
    match layout.data(db) {
        Layout::Enum(layout) => {
            layout.variants[variant.0 as usize].fields[field.0 as usize].clone()
        }
        Layout::Struct(_) | Layout::Array(_) => panic!("invalid variant-field projection layout"),
    }
}

fn push_runtime_provider_binding<'db>(
    provider_bindings: &mut Vec<RuntimeProviderBinding<'db>>,
    provider: ProviderBinding<'db>,
    local: SLocalId,
    provider_class: RuntimeClass<'db>,
    place_class: RuntimeClass<'db>,
) -> RuntimeProviderBindingId {
    let id = RuntimeProviderBindingId::from_u32(provider_bindings.len() as u32);
    provider_bindings.push(RuntimeProviderBinding {
        provider,
        value: crate::runtime::RLocalId::from_u32(local.index() as u32),
        provider_class,
        place_class,
    });
    id
}

fn owner_effect_arg_class<'db>(
    body: &NormalizedSemanticBody<'db>,
    carriers: &[RuntimeCarrier<'db>],
    binding: LocalBinding<'db>,
) -> Option<RuntimeClass<'db>> {
    let idx = body.locals.iter().position(|local| {
        local
            .source
            .is_some_and(|source| same_owner_effect_binding(source, binding))
    })?;
    carrier_value_class(SLocalId::from_u32(idx as u32), carriers)
}

fn effect_arg_class<'db>(
    db: &'db dyn MirDb,
    env: RuntimeTypeEnv<'db>,
    arg: &NEffectArg<'db>,
    carriers: &[RuntimeCarrier<'db>],
) -> Option<RuntimeClass<'db>> {
    match arg.pass_mode {
        EffectPassMode::ByValue | EffectPassMode::Unknown => match arg.arg {
            NEffectArgValue::Place(_) => Some(provider_class_for_target_in_context(
                db,
                arg.target_ty,
                resolved_address_space(arg.provider),
                env.scope,
                env.assumptions,
            )),
            NEffectArgValue::Value(value) => match carriers.get(value.local.index())? {
                RuntimeCarrier::Value(class) => Some(class.clone()),
                RuntimeCarrier::Erased if arg.provider.is_none() && arg.target_ty.is_none() => None,
                RuntimeCarrier::Erased => Some(provider_class_for_target_in_context(
                    db,
                    arg.target_ty,
                    resolved_address_space(arg.provider),
                    env.scope,
                    env.assumptions,
                )),
            },
        },
        EffectPassMode::ByPlace | EffectPassMode::ByTempPlace => {
            Some(provider_class_for_target_in_context(
                db,
                arg.target_ty,
                resolved_address_space(arg.provider),
                env.scope,
                env.assumptions,
            ))
        }
    }
}

pub(crate) enum ContractMetadataBuiltin<'db> {
    InitCodeOffset(RuntimeCodeRegion<'db>),
    InitCodeLen(RuntimeCodeRegion<'db>),
}

pub(crate) fn contract_metadata_builtin<'db>(
    db: &'db dyn MirDb,
    semantic: SemanticInstance<'db>,
) -> Option<ContractMetadataBuiltin<'db>> {
    let BodyOwner::Func(func) = semantic.key(db).owner(db) else {
        return None;
    };
    let name = func.name(db).to_opt()?.data(db);
    let trait_ = func.containing_trait(db)?;
    if trait_.name(db).to_opt()?.data(db) != "Contract" {
        return None;
    }
    let contract = semantic
        .key(db)
        .subst(db)
        .generic_args(db)
        .iter()
        .find_map(|ty| ty.as_contract(db))?;
    let region = RuntimeCodeRegion::new(db, RuntimeCodeRegionKey::ContractInit { contract });
    match name.as_str() {
        "init_code_offset" => Some(ContractMetadataBuiltin::InitCodeOffset(region)),
        "init_code_len" => Some(ContractMetadataBuiltin::InitCodeLen(region)),
        _ => None,
    }
}

fn extern_builtin_return_class<'db>(
    db: &'db dyn MirDb,
    semantic: SemanticInstance<'db>,
    result_ty: TyId<'db>,
) -> Option<Option<RuntimeClass<'db>>> {
    let typed_body = semantic.key(db).instantiate_typed_body(db);
    let env = RuntimeTypeEnv::new(
        typed_body.body().map(|body| body.scope()),
        typed_body.assumptions(),
    );
    if contract_metadata_builtin(db, semantic).is_some() {
        return Some(top_level_class_for_ty_in_env(
            db,
            env,
            result_ty,
            AddressSpaceKind::Memory,
        ));
    }
    let hir::analysis::ty::ty_check::BodyOwner::Func(func) = semantic.key(db).owner(db) else {
        return None;
    };
    if func.body(db).is_none()
        && func
            .name(db)
            .to_opt()
            .is_some_and(|name| is_runtime_intrinsic_name(name.data(db).as_str()))
    {
        return Some(top_level_class_for_ty_in_env(
            db,
            env,
            result_ty,
            AddressSpaceKind::Memory,
        ));
    }
    let matches = |path: &str| resolve_lib_func_path(db, func.scope(), path) == Some(func);
    if matches("std::evm::mem::alloc") {
        return Some(top_level_class_for_ty_in_env(
            db,
            env,
            result_ty,
            AddressSpaceKind::Memory,
        ));
    }
    if matches("core::intrinsic::__keccak256") {
        return Some(top_level_class_for_ty_in_env(
            db,
            env,
            result_ty,
            AddressSpaceKind::Memory,
        ));
    }

    let known = [
        "std::evm::ops::mload",
        "std::evm::ops::mstore",
        "std::evm::ops::mstore8",
        "std::evm::ops::msize",
        "std::evm::ops::sload",
        "std::evm::ops::sstore",
        "std::evm::ops::calldataload",
        "std::evm::ops::calldatacopy",
        "std::evm::ops::calldatasize",
        "std::evm::ops::returndatacopy",
        "std::evm::ops::returndatasize",
        "std::evm::ops::codecopy",
        "std::evm::ops::codesize",
        "std::evm::ops::keccak256",
        "std::evm::ops::addmod",
        "std::evm::ops::mulmod",
        "std::evm::ops::address",
        "std::evm::ops::caller",
        "std::evm::ops::callvalue",
        "std::evm::ops::origin",
        "std::evm::ops::gasprice",
        "std::evm::ops::coinbase",
        "std::evm::ops::timestamp",
        "std::evm::ops::number",
        "std::evm::ops::prevrandao",
        "std::evm::ops::gaslimit",
        "std::evm::ops::chainid",
        "std::evm::ops::basefee",
        "std::evm::ops::selfbalance",
        "std::evm::ops::blockhash",
        "std::evm::ops::gas",
        "std::evm::ops::call",
        "std::evm::ops::staticcall",
        "std::evm::ops::delegatecall",
        "std::evm::ops::create",
        "std::evm::ops::create2",
        "std::evm::ops::log0",
        "std::evm::ops::log1",
        "std::evm::ops::log2",
        "std::evm::ops::log3",
        "std::evm::ops::log4",
        "std::evm::ops::revert",
        "std::evm::ops::return_data",
        "std::evm::ops::selfdestruct",
        "std::evm::ops::stop",
        "core::panic",
        "core::panic_with_value",
        "core::todo",
    ]
    .iter()
    .any(|path| matches(path));
    known.then(|| top_level_class_for_ty_in_env(db, env, result_ty, AddressSpaceKind::Memory))
}

fn is_runtime_intrinsic_name(name: &str) -> bool {
    if matches!(name, "alloc") || generic_numeric_intrinsic_kind(name).is_some() {
        return true;
    }
    intrinsic_numeric_name_parts(name).is_some()
}

pub(super) fn generic_numeric_intrinsic_kind(name: &str) -> Option<GenericNumericIntrinsicKind> {
    Some(match name {
        "__bitcast" => GenericNumericIntrinsicKind::Bitcast,
        "__saturating_add" => GenericNumericIntrinsicKind::Saturating(SaturatingBinOp::Add),
        "__saturating_sub" => GenericNumericIntrinsicKind::Saturating(SaturatingBinOp::Sub),
        "__saturating_mul" => GenericNumericIntrinsicKind::Saturating(SaturatingBinOp::Mul),
        "__checked_add" => GenericNumericIntrinsicKind::CheckedBinary(ArithBinOp::Add),
        "__checked_sub" => GenericNumericIntrinsicKind::CheckedBinary(ArithBinOp::Sub),
        "__checked_mul" => GenericNumericIntrinsicKind::CheckedBinary(ArithBinOp::Mul),
        "__checked_div" => GenericNumericIntrinsicKind::CheckedBinary(ArithBinOp::Div),
        "__checked_rem" => GenericNumericIntrinsicKind::CheckedBinary(ArithBinOp::Rem),
        "__checked_neg" => GenericNumericIntrinsicKind::CheckedNeg,
        _ => return None,
    })
}

fn intrinsic_numeric_name_parts(name: &str) -> Option<(&str, &str)> {
    let op = name.strip_prefix("__")?;
    [
        "_u8", "_u16", "_u32", "_u64", "_u128", "_u256", "_usize", "_i8", "_i16", "_i32", "_i64",
        "_i128", "_i256", "_isize", "_bool",
    ]
    .iter()
    .find_map(|suffix| op.strip_suffix(suffix).map(|prefix| (prefix, *suffix)))
}

fn runtime_return_class_cycle_initial<'db>(
    db: &'db dyn MirDb,
    key: RuntimeInstanceKey<'db>,
) -> Option<RuntimeClass<'db>> {
    let typed_body = key
        .semantic(db)
        .expect("cycle handling only applies to semantic runtime instances")
        .key(db)
        .instantiate_typed_body(db);
    default_return_class(db, &typed_body)
}

fn runtime_return_class_cycle_recover<'db>(
    _db: &'db dyn MirDb,
    _value: &Option<RuntimeClass<'db>>,
    _count: u32,
    _key: RuntimeInstanceKey<'db>,
) -> salsa::CycleRecoveryAction<Option<RuntimeClass<'db>>> {
    salsa::CycleRecoveryAction::Iterate
}

pub(crate) fn runtime_param_class<'db>(
    db: &'db dyn MirDb,
    typed_body: &hir::analysis::ty::ty_check::TypedBody<'db>,
    binding: hir::analysis::ty::ty_check::LocalBinding<'db>,
    actual: RuntimeClass<'db>,
) -> RuntimeClass<'db> {
    let ty = runtime_repr_ty_in_context(
        db,
        typed_body.binding_ty(db, binding),
        typed_body.body().map(|body| body.scope()),
        typed_body.assumptions(),
    );
    if runtime_abstract_param_ty(
        db,
        typed_body.binding_ty(db, binding),
        typed_body.body().map(|body| body.scope()),
        typed_body.assumptions(),
    ) || matches!(
        ty.base_ty(db).data(db),
        TyData::TyParam(param) if param.is_effect() || param.is_effect_provider()
    ) {
        return actual;
    }
    if binding.is_mut() && ty.as_enum(db).is_some() {
        return RuntimeClass::Handle {
            layout: layout_for_ty_in_context(
                db,
                ty,
                typed_body.body().map(|body| body.scope()),
                typed_body.assumptions(),
            ),
            kind: HandleKind::ObjectValue,
            view: HandleView::Whole,
        };
    }
    actual
}

pub(crate) fn semantic_return_ty<'db>(
    db: &'db dyn MirDb,
    semantic: SemanticInstance<'db>,
) -> TyId<'db> {
    semantic.key(db).instantiate_typed_body(db).result_ty()
}

fn default_return_class<'db>(
    db: &'db dyn MirDb,
    typed_body: &hir::analysis::ty::ty_check::TypedBody<'db>,
) -> Option<RuntimeClass<'db>> {
    let env = RuntimeTypeEnv::new(
        typed_body.body().map(|body| body.scope()),
        typed_body.assumptions(),
    );
    let default_space = typed_body
        .return_borrow_provider()
        .map_or(AddressSpaceKind::Memory, address_space_from_provider);
    top_level_class_for_ty_in_env(db, env, typed_body.result_ty(), default_space)
}

fn return_ty_requires_runtime_body_inference<'db>(
    db: &'db dyn MirDb,
    ty: TyId<'db>,
    scope: Option<hir::hir_def::scope_graph::ScopeId<'db>>,
    assumptions: PredicateListId<'db>,
) -> bool {
    runtime_abstract_param_ty(db, ty, scope, assumptions)
}

pub(crate) fn top_level_class_for_ty_in_env<'db>(
    db: &'db dyn MirDb,
    env: RuntimeTypeEnv<'db>,
    ty: TyId<'db>,
    default_space: AddressSpaceKind,
) -> Option<RuntimeClass<'db>> {
    top_level_class_for_ty_in_context(db, ty, default_space, env.scope, env.assumptions)
}

pub(crate) fn provider_class_for_target_in_env<'db>(
    db: &'db dyn MirDb,
    env: RuntimeTypeEnv<'db>,
    target_ty: Option<TyId<'db>>,
    space: AddressSpaceKind,
) -> RuntimeClass<'db> {
    provider_class_for_target_in_context(db, target_ty, space, env.scope, env.assumptions)
}

pub(crate) fn scalar_class_for_ty_in_env<'db>(
    db: &'db dyn MirDb,
    env: RuntimeTypeEnv<'db>,
    ty: TyId<'db>,
) -> Option<ScalarClass<'db>> {
    scalar_class_for_ty_in_context(db, ty, env.scope, env.assumptions)
}

pub(crate) fn top_level_class_for_ty_in_context<'db>(
    db: &'db dyn MirDb,
    ty: TyId<'db>,
    default_space: AddressSpaceKind,
    scope: Option<hir::hir_def::scope_graph::ScopeId<'db>>,
    assumptions: PredicateListId<'db>,
) -> Option<RuntimeClass<'db>> {
    let ty = runtime_repr_ty_in_context(db, ty, scope, assumptions);
    if ty == TyId::unit(db) || is_zero_sized_in_context(db, ty, scope, assumptions) {
        return None;
    }
    if let Some((_, inner)) = ty.as_borrow(db) {
        if is_zero_sized_in_context(db, inner, scope, assumptions) {
            return None;
        }
        return Some(provider_class_for_target_in_context(
            db,
            Some(inner),
            default_space,
            scope,
            assumptions,
        ));
    }
    if let Some((_, inner)) = ty.as_capability(db) {
        if is_zero_sized_in_context(db, inner, scope, assumptions) {
            return None;
        }
        return Some(provider_class_for_target_in_context(
            db,
            Some(inner),
            default_space,
            scope,
            assumptions,
        ));
    }
    if let Some(class) = effect_handle_class_for_ty(db, ty, scope, assumptions) {
        return Some(class);
    }
    if let Some(scalar) = scalar_class_for_ty_in_context(db, ty, scope, assumptions) {
        return Some(RuntimeClass::Scalar(scalar));
    }
    if ty.as_enum(db).is_some() {
        return Some(RuntimeClass::AggregateValue {
            layout: layout_for_ty_in_context(db, ty, scope, assumptions),
        });
    }
    if ty.is_struct(db) || ty.is_array(db) || ty.is_tuple(db) {
        return Some(RuntimeClass::AggregateValue {
            layout: layout_for_ty_in_context(db, ty, scope, assumptions),
        });
    }
    None
}

pub(crate) fn stored_class_for_ty_in_context<'db>(
    db: &'db dyn MirDb,
    ty: TyId<'db>,
    scope: Option<hir::hir_def::scope_graph::ScopeId<'db>>,
    assumptions: PredicateListId<'db>,
) -> RuntimeClass<'db> {
    let ty = runtime_repr_ty_in_context(db, ty, scope, assumptions);
    if let Some((_, inner)) = ty.as_capability(db) {
        return provider_class_for_target_in_context(
            db,
            Some(inner),
            AddressSpaceKind::Memory,
            scope,
            assumptions,
        );
    }
    if let Some(class) = effect_handle_class_for_ty(db, ty, scope, assumptions) {
        return class;
    }
    if let Some(scalar) = scalar_class_for_ty_in_context(db, ty, scope, assumptions) {
        return RuntimeClass::Scalar(scalar);
    }
    RuntimeClass::AggregateValue {
        layout: layout_for_ty_in_context(db, ty, scope, assumptions),
    }
}

pub(crate) fn provider_class_for_target_in_context<'db>(
    db: &'db dyn MirDb,
    target_ty: Option<TyId<'db>>,
    space: AddressSpaceKind,
    scope: Option<hir::hir_def::scope_graph::ScopeId<'db>>,
    assumptions: PredicateListId<'db>,
) -> RuntimeClass<'db> {
    match target_ty.map(|ty| runtime_repr_ty_in_context(db, ty, scope, assumptions)) {
        Some(target_ty)
            if target_ty.is_struct(db)
                || target_ty.is_array(db)
                || target_ty.is_tuple(db)
                || target_ty.as_enum(db).is_some() =>
        {
            RuntimeClass::Handle {
                layout: layout_for_ty_in_context(db, target_ty, scope, assumptions),
                kind: HandleKind::Provider {
                    provider_ty: TyId::borrow_ref_of(db, target_ty),
                    space,
                },
                view: HandleView::Whole,
            }
        }
        Some(target_ty)
            if scalar_class_for_ty_in_context(db, target_ty, scope, assumptions).is_some() =>
        {
            RuntimeClass::RawAddr {
                space,
                target: None,
            }
        }
        Some(target_ty) => RuntimeClass::RawAddr {
            space,
            target: layout_for_ty_in_context(db, target_ty, scope, assumptions).into(),
        },
        None => RuntimeClass::RawAddr {
            space,
            target: None,
        },
    }
}

pub(crate) fn scalar_class_for_ty_in_context<'db>(
    db: &'db dyn MirDb,
    ty: TyId<'db>,
    scope: Option<hir::hir_def::scope_graph::ScopeId<'db>>,
    assumptions: PredicateListId<'db>,
) -> Option<ScalarClass<'db>> {
    let ty = runtime_repr_ty_in_context(db, ty, scope, assumptions);
    scalar_class_from_repr_ty(db, ty)
}

fn scalar_class_from_repr_ty<'db>(db: &'db dyn MirDb, ty: TyId<'db>) -> Option<ScalarClass<'db>> {
    let repr = match ty.base_ty(db).data(db) {
        TyData::TyBase(TyBase::Prim(prim)) => match prim {
            PrimTy::Bool => ScalarRepr::Bool,
            PrimTy::U8 => ScalarRepr::Int {
                bits: 8,
                signed: false,
            },
            PrimTy::U16 => ScalarRepr::Int {
                bits: 16,
                signed: false,
            },
            PrimTy::U32 => ScalarRepr::Int {
                bits: 32,
                signed: false,
            },
            PrimTy::U64 => ScalarRepr::Int {
                bits: 64,
                signed: false,
            },
            PrimTy::U128 => ScalarRepr::Int {
                bits: 128,
                signed: false,
            },
            PrimTy::U256 | PrimTy::Usize => ScalarRepr::Int {
                bits: 256,
                signed: false,
            },
            PrimTy::I8 => ScalarRepr::Int {
                bits: 8,
                signed: true,
            },
            PrimTy::I16 => ScalarRepr::Int {
                bits: 16,
                signed: true,
            },
            PrimTy::I32 => ScalarRepr::Int {
                bits: 32,
                signed: true,
            },
            PrimTy::I64 => ScalarRepr::Int {
                bits: 64,
                signed: true,
            },
            PrimTy::I128 => ScalarRepr::Int {
                bits: 128,
                signed: true,
            },
            PrimTy::I256 | PrimTy::Isize => ScalarRepr::Int {
                bits: 256,
                signed: true,
            },
            PrimTy::String => ScalarRepr::FixedBytes {
                len: MAX_INLINE_STRING_BYTES as u16,
            },
            PrimTy::Array
            | PrimTy::Tuple(_)
            | PrimTy::Ptr
            | PrimTy::View
            | PrimTy::BorrowMut
            | PrimTy::BorrowRef => return None,
        },
        TyData::TyBase(TyBase::Contract(_)) => ScalarRepr::Address { bits: 256 },
        _ => return None,
    };

    Some(ScalarClass {
        repr,
        role: ScalarRole::Plain,
    })
}

fn effect_handle_class_for_ty<'db>(
    db: &'db dyn MirDb,
    ty: TyId<'db>,
    scope: Option<hir::hir_def::scope_graph::ScopeId<'db>>,
    assumptions: PredicateListId<'db>,
) -> Option<RuntimeClass<'db>> {
    let scope = scope.or_else(|| ty.as_scope(db))?;
    let semantics = provider_semantics(db, scope, assumptions, ty);
    if matches!(semantics.kind, ProviderKind::RootObject) {
        return None;
    }
    if let Some(target_ty) = semantics.target_ty {
        if is_zero_sized_in_context(db, target_ty, Some(scope), assumptions) {
            return None;
        }
        return Some(provider_class_for_target_in_context(
            db,
            Some(target_ty),
            provider_address_space_to_runtime(semantics.address_space?),
            Some(scope),
            assumptions,
        ));
    }
    None
}

fn runtime_abstract_param_ty<'db>(
    db: &'db dyn MirDb,
    ty: TyId<'db>,
    scope: Option<hir::hir_def::scope_graph::ScopeId<'db>>,
    assumptions: PredicateListId<'db>,
) -> bool {
    let ty = runtime_repr_ty_in_context(db, ty, scope, assumptions);
    ty.has_param(db) || ty.contains_assoc_ty_of_param(db)
}

fn provider_address_space_to_runtime(space: ProviderAddressSpace) -> AddressSpaceKind {
    match space {
        ProviderAddressSpace::Memory => AddressSpaceKind::Memory,
        ProviderAddressSpace::Storage => AddressSpaceKind::Storage,
        ProviderAddressSpace::Transient => AddressSpaceKind::Transient,
        ProviderAddressSpace::Calldata => AddressSpaceKind::Calldata,
    }
}
