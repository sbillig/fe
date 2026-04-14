use common::indexmap::IndexSet;
use cranelift_entity::EntityRef;
use hir::analysis::{
    semantic::{
        FieldIndex, GenericSubst, ImplEnv, Mutability, NEffectArg, NEffectArgValue, NSStmtKind,
        SConst, SLocalId, SemanticCalleeRef, SemanticInstance, SemanticInstanceKey,
        ValueProvenance, VariantIndex,
        borrowck::{
            NBorrowRoot, NExpr, NLocalInterface, NLocalOrigin, NOperand, NSLocal, NSPlace,
            NSPlaceRoot, NSTerminatorKind, NormalizedBindingLowering, NormalizedSemanticBody,
            normalize_semantic_body,
        },
        get_or_build_semantic_instance, owner_effect_bindings, same_owner_effect_binding,
        sem_const_ty, semantic_binding_lowering, semantic_binding_ty,
        semantic_instance_assumptions,
    },
    ty::{
        ProviderAddressSpace, ProviderKind,
        corelib::lib_func_matches,
        normalize::normalize_ty,
        provider::{provider_semantics, registered_root_providers},
        trait_def::{TraitInstId, resolve_trait_method_instance},
        trait_resolution::{PredicateListId, TraitSolveCx},
        ty_check::{BodyOwner, EffectParamSite, EffectPassMode, LocalBinding, ParamSite},
        ty_def::{
            BorrowKind, MAX_INLINE_STRING_BYTES, PrimTy, TyBase, TyData, TyId,
            strip_derived_adt_layout_args,
        },
    },
};
use hir::hir_def::ArithBinOp;
use hir::projection::Projection;
use hir::semantic::ProviderBinding;
use rustc_hash::FxHashSet;

use crate::{
    db::MirDb,
    instance::{RuntimeInstanceKey, RuntimeInstanceSource},
    runtime::{
        AddressSpaceKind, BorrowAccess, BorrowTransportSet, Layout, RefKind, RefView,
        RuntimeBoundarySpec, RuntimeCarrier, RuntimeClass, RuntimeCodeRegion, RuntimeCodeRegionKey,
        RuntimeLocalLowering, RuntimeLocalRoot, RuntimeParam, RuntimeParamPlan,
        RuntimeProviderBinding, RuntimeProviderBindingId, RuntimeSignature, SaturatingBinOp,
        ScalarClass, ScalarRepr, ScalarRole,
    },
};

use super::{
    layout::{
        RuntimeTypeEnv, is_zero_sized_in_context, layout_for_aggregate_instance_in_context,
        layout_for_ty_in_context, layout_for_ty_in_env, runtime_repr_ty_in_context,
    },
    place::{
        address_space_from_provider, project_field_class, project_index_class,
        resolved_effect_arg_address_space,
    },
};

#[derive(Clone, Debug)]
pub(super) struct InferredRuntimeLocal<'db> {
    pub(super) carrier: RuntimeCarrier<'db>,
    pub(super) root: RuntimeLocalRoot<'db>,
}

struct LocalRuntimeClassCtxt<'a, 'db> {
    body: &'a NormalizedSemanticBody<'db>,
    carriers: &'a [RuntimeCarrier<'db>],
    scope: Option<hir::hir_def::scope_graph::ScopeId<'db>>,
    assumptions: PredicateListId<'db>,
}

#[derive(Clone, Debug)]
pub(crate) struct RuntimeVisibleBindingPlan<'db> {
    pub(crate) binding: LocalBinding<'db>,
    pub(crate) local: SLocalId,
    pub(crate) plan: RuntimeParamPlan<'db>,
}

#[derive(Clone, Debug)]
pub(crate) struct RuntimeEffectBindingPlan<'db> {
    pub(crate) class: RuntimeClass<'db>,
    pub(crate) boundary: RuntimeBoundarySpec<'db>,
}

pub(crate) fn runtime_address_space(class: &RuntimeClass<'_>) -> Option<AddressSpaceKind> {
    match class {
        RuntimeClass::Ref {
            kind: RefKind::Provider { space, .. },
            ..
        }
        | RuntimeClass::RawAddr { space, .. } => Some(*space),
        RuntimeClass::Scalar(_)
        | RuntimeClass::AggregateValue { .. }
        | RuntimeClass::Ref {
            kind: RefKind::Const | RefKind::Object,
            ..
        } => None,
    }
}

fn provider_root_space<'db>(
    binding: &ProviderBinding<'db>,
    root_class: &RuntimeClass<'db>,
) -> AddressSpaceKind {
    runtime_address_space(root_class).unwrap_or_else(|| match binding.semantics.kind {
        ProviderKind::RootObject => AddressSpaceKind::Memory,
        ProviderKind::Handle | ProviderKind::RawAddress => address_space_from_provider(
            binding
                .semantics
                .address_space
                .unwrap_or_else(|| panic!("provider binding missing resolved space")),
        ),
    })
}

pub(crate) fn ref_class_for_place_result<'db>(
    root_class: &RuntimeClass<'db>,
    value_class: &RuntimeClass<'db>,
    root_space: AddressSpaceKind,
    force_raw: bool,
) -> RuntimeClass<'db> {
    if !force_raw {
        match root_class {
            RuntimeClass::Ref { kind, .. } => {
                return RuntimeClass::Ref {
                    pointee: Box::new(value_class.clone()),
                    kind: kind.clone(),
                    view: RefView::Whole,
                };
            }
            RuntimeClass::AggregateValue { .. } => {
                return RuntimeClass::Ref {
                    pointee: Box::new(value_class.clone()),
                    kind: RefKind::Object,
                    view: RefView::Whole,
                };
            }
            RuntimeClass::Scalar(_) | RuntimeClass::RawAddr { .. } => {}
        }
    }
    RuntimeClass::RawAddr {
        space: runtime_address_space(root_class).unwrap_or(root_space),
        target: value_class.aggregate_layout(),
    }
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
    let semantic_body = normalize_semantic_body(db, semantic)
        .unwrap_or_else(|err| panic!("semantic normalization failed for {:?}: {err:?}", key));
    let states = infer_local_runtime_state(
        db,
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
    for (idx, local) in body.locals.iter().enumerate() {
        if !matches!(carriers[idx], RuntimeCarrier::Erased) {
            continue;
        }
        let class = match (&local.facts.interface, &local.facts.origin) {
            (NLocalInterface::DirectValue, NLocalOrigin::RootProvider(provider)) => {
                actual_runtime_visible_root_provider_class(db, body, &carriers, provider)
                    .map(|(_, class)| class)
                    .or_else(|| {
                        runtime_class_for_direct_value_provider_in_context(
                            db,
                            provider,
                            scope,
                            assumptions,
                        )
                    })
            }
            (NLocalInterface::DirectCarrier, NLocalOrigin::RootProvider(provider)) => {
                actual_runtime_visible_root_provider_class(db, body, &carriers, provider)
                    .map(|(_, class)| class)
                    .or_else(|| {
                        runtime_class_for_provider_binding(db, provider, scope, assumptions)
                    })
            }
            _ => None,
        };
        if let Some(class) = class {
            carriers[idx] = RuntimeCarrier::Value(class);
        }
    }

    loop {
        let mut changed = false;
        for block in &body.blocks {
            for stmt in &block.stmts {
                let NSStmtKind::Assign { dst, expr } = &stmt.kind else {
                    continue;
                };
                let desired = match &body.locals[dst.index()] {
                    NSLocal {
                        mutability: Mutability::Mutable,
                        source: Some(_),
                        ..
                    } => {
                        match expr_direct_class(
                            db,
                            body,
                            expr,
                            body.locals[dst.index()].ty,
                            &carriers,
                        ) {
                            Some(RuntimeClass::AggregateValue { layout }) => {
                                RuntimeCarrier::Value(RuntimeClass::object_ref(layout))
                            }
                            Some(class) => RuntimeCarrier::Value(class),
                            None => continue,
                        }
                    }
                    local => match expr_direct_class(db, body, expr, local.ty, &carriers) {
                        Some(class) => RuntimeCarrier::Value(class),
                        None => continue,
                    },
                };
                if carriers[dst.index()] != desired {
                    carriers[dst.index()] = desired;
                    changed = true;
                }
            }
        }
        if !changed {
            break;
        }
    }

    let class_ctxt = LocalRuntimeClassCtxt {
        body,
        carriers: &carriers,
        scope,
        assumptions,
    };

    body.locals
        .iter()
        .enumerate()
        .map(|(idx, local)| {
            let local_id = SLocalId::from_u32(idx as u32);
            let mut carrier = carriers[idx].clone();
            let root = if !local.facts.root_demand.needs_runtime_root() {
                RuntimeLocalRoot::None
            } else {
                infer_runtime_local_root(db, &class_ctxt, local_id, &mut carrier)
            };
            InferredRuntimeLocal { carrier, root }
        })
        .collect()
}

pub(super) fn lower_semantic_locals<'db>(
    db: &'db dyn MirDb,
    body: &NormalizedSemanticBody<'db>,
    states: &[InferredRuntimeLocal<'db>],
    scope: Option<hir::hir_def::scope_graph::ScopeId<'db>>,
    assumptions: PredicateListId<'db>,
) -> (
    Vec<RuntimeLocalLowering<'db>>,
    Vec<RuntimeProviderBinding<'db>>,
) {
    let carriers = states
        .iter()
        .map(|state| state.carrier.clone())
        .collect::<Vec<_>>();
    let mut provider_bindings = Vec::new();
    for (idx, local) in body.locals.iter().enumerate() {
        let local_id = SLocalId::from_u32(idx as u32);
        let binding = match (&local.facts.interface, &local.facts.origin) {
            (NLocalInterface::DirectValue, NLocalOrigin::RootProvider(provider)) => {
                let (provider_local, provider_class) = actual_runtime_visible_root_provider_class(
                    db,
                    body,
                    &carriers,
                    provider,
                )
                .or_else(|| {
                    runtime_class_for_direct_value_provider_in_context(
                        db,
                        provider,
                        scope,
                        assumptions,
                    )
                    .map(|class| (local_id, class))
                })
                .unwrap_or_else(|| {
                    panic!(
                        "missing runtime provider class for root-provider direct value local {:?}: {provider:?}",
                        local.source
                    )
                });
                Some((
                    provider.clone(),
                    provider_class,
                    normalized_local_place_class(db, body, local_id, &carriers)
                        .unwrap_or_else(|| {
                            panic!(
                                "missing normalized place class for root-provider direct value local {idx}"
                            )
                        }),
                    provider_local,
                ))
            }
            (NLocalInterface::PlaceBoundValue, NLocalOrigin::RootProvider(provider)) => {
                let (provider_local, provider_class) = actual_runtime_visible_root_provider_class(
                    db,
                    body,
                    &carriers,
                    provider,
                )
                .or_else(|| {
                    runtime_class_for_effect_binding_provider_in_context(
                        db,
                        provider,
                        scope,
                        assumptions,
                    )
                    .or_else(|| {
                        runtime_class_for_direct_value_provider_in_context(
                            db,
                            provider,
                            scope,
                            assumptions,
                        )
                    })
                    .map(|class| (local_id, class))
                })
                .unwrap_or_else(|| {
                    panic!(
                        "missing runtime provider class for root-provider place-bound local {idx}: {provider:?}"
                    )
                });
                Some((
                    provider.clone(),
                    provider_class,
                    normalized_local_place_class(db, body, local_id, &carriers)
                        .unwrap_or_else(|| {
                            panic!(
                                "missing normalized place class for root-provider place-bound local {idx}"
                            )
                        }),
                    provider_local,
                ))
            }
            (NLocalInterface::DirectCarrier, NLocalOrigin::RootProvider(provider)) => {
                let NormalizedBindingLowering::CarrierLocal { .. } = &local.lowering else {
                    panic!("direct-carrier local missing carrier lowering: {idx}");
                };
                let (provider_local, provider_class) =
                    actual_runtime_visible_root_provider_class(db, body, &carriers, provider)
                        .or_else(|| {
                            runtime_class_for_provider_binding(db, provider, scope, assumptions)
                                .map(|class| (local_id, class))
                        })
                        .unwrap_or_else(|| {
                            panic!(
                                "missing direct-carrier runtime class for semantic local {idx}: {}",
                                local.ty.pretty_print(db),
                            )
                        });
                Some((
                    provider.clone(),
                    provider_class,
                    carrier_local_place_class(db, local, local_id, &carriers, scope, assumptions),
                    provider_local,
                ))
            }
            _ => None,
        };
        let Some((provider, provider_class, place_class, provider_local)) = binding else {
            continue;
        };
        if runtime_provider_binding_id(&provider_bindings, &provider).is_some() {
            continue;
        }
        push_runtime_provider_binding(
            &mut provider_bindings,
            provider,
            provider_local,
            provider_class,
            place_class,
        );
    }
    let lowerings = body
        .locals
        .iter()
        .enumerate()
        .map(|(idx, local)| match (&local.facts.interface, &local.facts.origin) {
            (NLocalInterface::Erased, _) => RuntimeLocalLowering::Erased,
            (NLocalInterface::DirectValue, NLocalOrigin::RootProvider(provider)) => {
                    let provider =
                        runtime_provider_binding_id(&provider_bindings, provider).unwrap_or_else(
                            || {
                            panic!(
                                "missing runtime provider binding for root-provider direct value local {idx}: {provider:?}"
                            )
                        });
                    RuntimeLocalLowering::PlaceBoundValue {
                        provider: Some(provider),
                        place_class: provider_bindings[provider.index()].place_class.clone(),
                    }
            }
            (NLocalInterface::DirectValue, _) => RuntimeLocalLowering::DirectValue,
            (NLocalInterface::PlaceCarrier, _) => {
                RuntimeLocalLowering::PlaceCarrier {
                    place_class: carrier_local_place_class(
                        db,
                        local,
                        SLocalId::from_u32(idx as u32),
                        &carriers,
                        scope,
                        assumptions,
                    ),
                }
            }
            (NLocalInterface::PlaceBoundValue, origin) => {
                let place_class = normalized_local_place_class(
                    db,
                    body,
                    SLocalId::from_u32(idx as u32),
                    &carriers,
                )
                .unwrap_or_else(|| {
                    panic!("missing normalized place class for place-bound semantic local {idx}")
                });
                let provider = origin.root_provider().map(|provider| {
                    runtime_provider_binding_id(&provider_bindings, provider).unwrap_or_else(|| {
                        panic!(
                            "missing runtime provider binding for place-bound semantic local {idx}: {origin:?}"
                        )
                    })
                });
                RuntimeLocalLowering::PlaceBoundValue {
                    provider,
                    place_class,
                }
            }
            (NLocalInterface::DirectCarrier, origin) => {
                let place_class = carrier_local_place_class(
                    db,
                    local,
                    SLocalId::from_u32(idx as u32),
                    &carriers,
                    scope,
                    assumptions,
                );
                let provider = origin.root_provider().map(|provider| {
                    let provider_class = runtime_class_for_provider_binding(
                        db,
                        provider,
                        scope,
                        assumptions,
                    )
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
        })
        .collect();
    (lowerings, provider_bindings)
}

fn runtime_provider_binding_id<'db>(
    provider_bindings: &[RuntimeProviderBinding<'db>],
    provider: &ProviderBinding<'db>,
) -> Option<RuntimeProviderBindingId> {
    provider_bindings
        .iter()
        .enumerate()
        .find_map(|(idx, binding)| {
            (binding.provider == *provider).then(|| RuntimeProviderBindingId::from_u32(idx as u32))
        })
}

fn root_provider_for_runtime_visible_binding<'db>(
    db: &'db dyn MirDb,
    semantic: SemanticInstance<'db>,
    binding: LocalBinding<'db>,
) -> Option<ProviderBinding<'db>> {
    match semantic_binding_lowering(db, semantic, binding) {
        hir::analysis::semantic::SemanticBindingLowering::DirectValue {
            provenance: ValueProvenance::RootProvider(provider),
        }
        | hir::analysis::semantic::SemanticBindingLowering::DirectCarrier {
            provider: Some(provider),
            ..
        }
        | hir::analysis::semantic::SemanticBindingLowering::PlaceBoundValue {
            provenance: hir::analysis::semantic::PlaceProvenance::RootProvider(provider),
            ..
        } => Some(provider),
        hir::analysis::semantic::SemanticBindingLowering::Erased
        | hir::analysis::semantic::SemanticBindingLowering::DirectValue { .. }
        | hir::analysis::semantic::SemanticBindingLowering::DirectCarrier {
            provider: None, ..
        }
        | hir::analysis::semantic::SemanticBindingLowering::PlaceCarrier { .. }
        | hir::analysis::semantic::SemanticBindingLowering::PlaceBoundValue {
            provenance: hir::analysis::semantic::PlaceProvenance::Derived { .. },
            ..
        } => None,
    }
}

fn actual_runtime_visible_root_provider_local<'db>(
    db: &'db dyn MirDb,
    semantic: SemanticInstance<'db>,
    provider: &ProviderBinding<'db>,
) -> Option<SLocalId> {
    runtime_visible_binding_plans(db, semantic)
        .into_iter()
        .find(|entry| {
            root_provider_for_runtime_visible_binding(db, semantic, entry.binding)
                .as_ref()
                .is_some_and(|candidate| candidate == provider)
        })
        .map(|entry| entry.local)
}

fn actual_runtime_visible_root_provider_class<'db>(
    db: &'db dyn MirDb,
    body: &NormalizedSemanticBody<'db>,
    carriers: &[RuntimeCarrier<'db>],
    provider: &ProviderBinding<'db>,
) -> Option<(SLocalId, RuntimeClass<'db>)> {
    let local = actual_runtime_visible_root_provider_local(db, body.owner, provider)?;
    carrier_value_class(local, carriers).map(|class| (local, class))
}

fn carrier_local_place_class<'db>(
    db: &'db dyn MirDb,
    local: &NSLocal<'db>,
    local_id: SLocalId,
    carriers: &[RuntimeCarrier<'db>],
    scope: Option<hir::hir_def::scope_graph::ScopeId<'db>>,
    assumptions: PredicateListId<'db>,
) -> RuntimeClass<'db> {
    let NormalizedBindingLowering::CarrierLocal { target_ty, .. } = &local.lowering else {
        panic!("carrier local missing carrier lowering: {local_id:?}");
    };
    carrier_value_class(local_id, carriers)
        .and_then(|class| actual_aggregate_class_from_runtime_source(&class))
        .unwrap_or_else(|| stored_class_for_ty_in_context(db, *target_ty, scope, assumptions))
}

fn normalized_local_place_class<'db>(
    db: &'db dyn MirDb,
    body: &NormalizedSemanticBody<'db>,
    local: SLocalId,
    carriers: &[RuntimeCarrier<'db>],
) -> Option<RuntimeClass<'db>> {
    normalized_place_class(
        db,
        body,
        body.locals.get(local.index())?.backing_place()?,
        carriers,
    )
}

fn infer_runtime_local_root<'db>(
    db: &'db dyn MirDb,
    ctxt: &LocalRuntimeClassCtxt<'_, 'db>,
    local: SLocalId,
    carrier: &mut RuntimeCarrier<'db>,
) -> RuntimeLocalRoot<'db> {
    let local_data = ctxt
        .body
        .locals
        .get(local.index())
        .expect("normalized local exists");
    let place_class = local_place_root_class(db, ctxt, local, local_data, carrier);
    let transport_class = match carrier {
        RuntimeCarrier::Value(class) => Some(class.clone()),
        RuntimeCarrier::Erased => {
            fallback_root_transport_class(db, local_data, ctxt.scope, ctxt.assumptions)
        }
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
            RuntimeClass::RawAddr { .. } | RuntimeClass::Ref { .. }
        )
    ) {
        *carrier = RuntimeCarrier::Value(transport_class.clone());
    }
    match transport_class {
        RuntimeClass::RawAddr { space, .. } => RuntimeLocalRoot::Ptr {
            space,
            class: place_class,
        },
        RuntimeClass::Ref {
            kind: RefKind::Provider { space, .. },
            ..
        } if space != AddressSpaceKind::Memory => RuntimeLocalRoot::Ptr {
            space,
            class: place_class,
        },
        RuntimeClass::Ref { .. } => RuntimeLocalRoot::Ref(transport_class),
        RuntimeClass::Scalar(_) | RuntimeClass::AggregateValue { .. } => {
            RuntimeLocalRoot::Slot(place_class)
        }
    }
}

fn local_place_root_class<'db>(
    db: &'db dyn MirDb,
    ctxt: &LocalRuntimeClassCtxt<'_, 'db>,
    local: SLocalId,
    local_data: &NSLocal<'db>,
    carrier: &RuntimeCarrier<'db>,
) -> Option<RuntimeClass<'db>> {
    match local_data.facts.interface {
        NLocalInterface::Erased => None,
        NLocalInterface::DirectValue => {
            if let Some(carrier_class) = carrier_value_class_for_runtime(carrier) {
                if let Some(actual) = actual_aggregate_class_from_runtime_source(&carrier_class) {
                    return Some(actual);
                }
                if !matches!(carrier_class, RuntimeClass::RawAddr { .. }) {
                    return Some(carrier_class);
                }
            }
            Some(stored_class_for_ty_in_context(
                db,
                local_data.ty,
                ctxt.scope,
                ctxt.assumptions,
            ))
        }
        NLocalInterface::PlaceCarrier => {
            if let Some(carrier_class) = carrier_value_class_for_runtime(carrier)
                && let Some(actual) = actual_aggregate_class_from_runtime_source(&carrier_class)
            {
                return Some(actual);
            }
            let NormalizedBindingLowering::CarrierLocal { target_ty, .. } = &local_data.lowering
            else {
                panic!("place-carrier local missing carrier lowering: {local:?}");
            };
            Some(stored_class_for_ty_in_context(
                db,
                *target_ty,
                ctxt.scope,
                ctxt.assumptions,
            ))
        }
        NLocalInterface::PlaceBoundValue => {
            normalized_local_place_class(db, ctxt.body, local, ctxt.carriers).or_else(|| {
                let NormalizedBindingLowering::PlaceBoundValue { value_ty, .. } =
                    &local_data.lowering
                else {
                    panic!("place-bound local missing place-bound lowering: {local:?}");
                };
                Some(stored_class_for_ty_in_context(
                    db,
                    *value_ty,
                    ctxt.scope,
                    ctxt.assumptions,
                ))
            })
        }
        NLocalInterface::DirectCarrier => {
            if let Some(carrier_class) = carrier_value_class_for_runtime(carrier)
                && let Some(actual) = actual_aggregate_class_from_runtime_source(&carrier_class)
            {
                return Some(actual);
            }
            let NormalizedBindingLowering::CarrierLocal { target_ty, .. } = &local_data.lowering
            else {
                panic!("direct-carrier local missing carrier lowering: {local:?}");
            };
            Some(stored_class_for_ty_in_context(
                db,
                *target_ty,
                ctxt.scope,
                ctxt.assumptions,
            ))
        }
    }
}

fn fallback_root_transport_class<'db>(
    db: &'db dyn MirDb,
    local: &NSLocal<'db>,
    scope: Option<hir::hir_def::scope_graph::ScopeId<'db>>,
    assumptions: PredicateListId<'db>,
) -> Option<RuntimeClass<'db>> {
    match local.facts.interface {
        NLocalInterface::Erased
        | NLocalInterface::DirectValue
        | NLocalInterface::PlaceBoundValue => None,
        NLocalInterface::PlaceCarrier => {
            let NormalizedBindingLowering::CarrierLocal { target_ty, .. } = &local.lowering else {
                panic!("place-carrier local missing carrier lowering");
            };
            top_level_class_for_ty_in_context(
                db,
                local.ty,
                AddressSpaceKind::Memory,
                scope,
                assumptions,
            )
            .or_else(|| {
                Some(provider_class_for_target_in_context(
                    db,
                    Some(*target_ty),
                    AddressSpaceKind::Memory,
                    scope,
                    assumptions,
                ))
            })
        }
        NLocalInterface::DirectCarrier => {
            let provider = local.facts.origin.root_provider();
            let NormalizedBindingLowering::CarrierLocal { target_ty, .. } = &local.lowering else {
                panic!("direct-carrier local missing carrier lowering");
            };
            provider
                .and_then(|provider| {
                    runtime_class_for_provider_binding(db, provider, scope, assumptions)
                })
                .or_else(|| {
                    top_level_class_for_ty_in_context(
                        db,
                        local.ty,
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
                })
        }
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

fn provider_root_place_class<'db>(
    db: &'db dyn MirDb,
    value_ty: TyId<'db>,
    provider_class: &RuntimeClass<'db>,
    scope: Option<hir::hir_def::scope_graph::ScopeId<'db>>,
    assumptions: PredicateListId<'db>,
) -> RuntimeClass<'db> {
    match provider_class {
        RuntimeClass::Ref { pointee, .. } => *pointee.clone(),
        RuntimeClass::RawAddr {
            target: Some(layout),
            ..
        } => RuntimeClass::AggregateValue { layout: *layout },
        RuntimeClass::Scalar(_)
        | RuntimeClass::AggregateValue { .. }
        | RuntimeClass::RawAddr { target: None, .. } => {
            stored_class_for_ty_in_context(db, value_ty, scope, assumptions)
        }
    }
}

pub(crate) fn runtime_param_locals<'db>(
    db: &'db dyn MirDb,
    semantic: SemanticInstance<'db>,
    params: &[RuntimeClass<'db>],
) -> Vec<SLocalId> {
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
    entries.into_iter().map(|entry| entry.local).collect()
}

pub(crate) fn runtime_visible_binding_plans<'db>(
    db: &'db dyn MirDb,
    semantic: SemanticInstance<'db>,
) -> Vec<RuntimeVisibleBindingPlan<'db>> {
    let owner = semantic.key(db).owner(db);
    let typed_body = semantic.key(db).instantiate_typed_body(db);
    let mut entries = Vec::new();
    let mut push = |binding, plan| {
        if !matches!(plan, RuntimeParamPlan::Erased) {
            entries.push(RuntimeVisibleBindingPlan {
                binding,
                local: runtime_visible_binding_local(db, owner, &typed_body, binding),
                plan,
            });
        }
    };

    let mut idx = 0;
    while let Some(binding) = typed_body.param_binding(idx) {
        push(binding, desired_runtime_param_plan(db, &typed_body, idx));
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
                .map(RuntimeBoundarySpec::Exact)
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
            Some(provider.semantics.target_ty.unwrap_or(provider.provider_ty)),
            provider
                .semantics
                .address_space
                .map_or(AddressSpaceKind::Memory, provider_address_space_to_runtime),
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

fn effect_binding_borrow_boundary<'db>(
    db: &'db dyn MirDb,
    binding: LocalBinding<'db>,
    pointee_ty: TyId<'db>,
    scope: Option<hir::hir_def::scope_graph::ScopeId<'db>>,
    assumptions: PredicateListId<'db>,
) -> RuntimeBoundarySpec<'db> {
    let access = if binding.is_mut() {
        BorrowAccess::ReadWrite
    } else {
        BorrowAccess::ReadOnly
    };
    RuntimeBoundarySpec::BorrowLike {
        pointee: stored_class_for_ty_in_context(db, pointee_ty, scope, assumptions),
        access,
        allow: default_borrow_transport_set(access, AddressSpaceKind::Memory),
    }
}

pub(crate) fn runtime_effect_binding_plan<'db>(
    db: &'db dyn MirDb,
    semantic: SemanticInstance<'db>,
    binding: LocalBinding<'db>,
) -> Option<RuntimeEffectBindingPlan<'db>> {
    if !matches!(binding, LocalBinding::EffectParam { .. }) {
        return None;
    }
    let owner = semantic.key(db).owner(db);
    let env = RuntimeTypeEnv::new(
        Some(owner.scope()),
        semantic_instance_assumptions(db, semantic),
    );
    let binding_ty = semantic_binding_ty(db, semantic, binding);
    match semantic_binding_lowering(db, semantic, binding) {
        hir::analysis::semantic::SemanticBindingLowering::Erased => None,
        hir::analysis::semantic::SemanticBindingLowering::DirectValue {
            provenance: ValueProvenance::RootProvider(provider),
        } => {
            let class = runtime_class_for_direct_value_provider_in_env(db, env, &provider)?;
            Some(RuntimeEffectBindingPlan {
                class,
                boundary: effect_binding_borrow_boundary(
                    db,
                    binding,
                    binding_ty,
                    env.scope,
                    env.assumptions,
                ),
            })
        }
        hir::analysis::semantic::SemanticBindingLowering::DirectValue { .. } => {
            let class =
                runtime_class_for_explicit_root_provider_param(db, env, binding, binding_ty)
                    .or_else(|| {
                        top_level_class_for_ty_in_env(db, env, binding_ty, AddressSpaceKind::Memory)
                    })?;
            Some(RuntimeEffectBindingPlan {
                class: class.clone(),
                boundary: RuntimeBoundarySpec::Exact(class),
            })
        }
        hir::analysis::semantic::SemanticBindingLowering::DirectCarrier {
            provider: Some(provider),
            ..
        } => {
            let class =
                runtime_class_for_provider_binding(db, &provider, env.scope, env.assumptions)?;
            Some(RuntimeEffectBindingPlan {
                class: class.clone(),
                boundary: RuntimeBoundarySpec::Exact(class),
            })
        }
        hir::analysis::semantic::SemanticBindingLowering::DirectCarrier {
            provider: None,
            target_ty,
        } => {
            let class =
                top_level_class_for_ty_in_env(db, env, binding_ty, AddressSpaceKind::Memory)
                    .or_else(|| {
                        Some(provider_class_for_target_in_env(
                            db,
                            env,
                            Some(target_ty),
                            AddressSpaceKind::Memory,
                        ))
                    })?;
            Some(RuntimeEffectBindingPlan {
                class: class.clone(),
                boundary: RuntimeBoundarySpec::Exact(class),
            })
        }
        hir::analysis::semantic::SemanticBindingLowering::PlaceCarrier { value_ty } => {
            let class =
                provider_class_for_target_in_env(db, env, Some(value_ty), AddressSpaceKind::Memory);
            Some(RuntimeEffectBindingPlan {
                class: class.clone(),
                boundary: RuntimeBoundarySpec::Exact(class),
            })
        }
        hir::analysis::semantic::SemanticBindingLowering::PlaceBoundValue {
            provenance: hir::analysis::semantic::PlaceProvenance::RootProvider(provider),
            value_ty,
        } => {
            let class = runtime_class_for_effect_binding_provider_in_env(db, env, &provider)?;
            Some(RuntimeEffectBindingPlan {
                class,
                boundary: effect_binding_borrow_boundary(
                    db,
                    binding,
                    value_ty,
                    env.scope,
                    env.assumptions,
                ),
            })
        }
        hir::analysis::semantic::SemanticBindingLowering::PlaceBoundValue {
            provenance: hir::analysis::semantic::PlaceProvenance::Derived { .. },
            ..
        } => None,
    }
}

pub(crate) fn runtime_effect_binding_plan_for_binding_idx<'db>(
    db: &'db dyn MirDb,
    semantic: SemanticInstance<'db>,
    binding_idx: u32,
) -> Option<RuntimeEffectBindingPlan<'db>> {
    let BodyOwner::Func(func) = semantic.key(db).owner(db) else {
        return None;
    };
    let resolved = hir::semantic::EffectEnvView::new(EffectParamSite::Func(func))
        .resolved_binding(db, binding_idx as usize)?;
    runtime_effect_binding_plan(
        db,
        semantic,
        LocalBinding::EffectParam {
            site: resolved.requirement.binding_site,
            idx: resolved.requirement.binding_idx as usize,
            binding_name: resolved.requirement.binding_name,
            provider_idx: resolved.provider.provider_idx,
            key_path: resolved.requirement.binding_path,
            is_mut: resolved.requirement.is_mut,
        },
    )
}

pub(crate) fn runtime_visible_binding_class<'db>(
    db: &'db dyn MirDb,
    semantic: SemanticInstance<'db>,
    binding: LocalBinding<'db>,
) -> Option<RuntimeClass<'db>> {
    if let Some(plan) = runtime_effect_binding_plan(db, semantic, binding) {
        return Some(plan.class);
    }
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
        hir::analysis::semantic::SemanticBindingLowering::PlaceBoundValue {
            provenance: hir::analysis::semantic::PlaceProvenance::RootProvider(provider),
            ..
        } => runtime_class_for_effect_binding_provider_in_env(db, env, &provider),
        hir::analysis::semantic::SemanticBindingLowering::PlaceBoundValue {
            provenance: hir::analysis::semantic::PlaceProvenance::Derived { .. },
            ..
        } => None,
    }
}

pub(crate) fn owner_effect_binding_boundary<'db>(
    db: &'db dyn MirDb,
    semantic: SemanticInstance<'db>,
    binding: LocalBinding<'db>,
) -> Option<RuntimeBoundarySpec<'db>> {
    runtime_effect_binding_plan(db, semantic, binding).map(|plan| plan.boundary)
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

pub(crate) fn expr_direct_class<'db>(
    db: &'db dyn MirDb,
    body: &NormalizedSemanticBody<'db>,
    expr: &NExpr<'db>,
    result_ty: TyId<'db>,
    carriers: &[RuntimeCarrier<'db>],
) -> Option<RuntimeClass<'db>> {
    let owner = body.owner.key(db).owner(db);
    let typed_body = body.owner.key(db).instantiate_typed_body(db);
    let env = RuntimeTypeEnv::new(Some(owner.scope()), typed_body.assumptions());
    Some(match expr {
        NExpr::Use(value) => materialized_value_class(db, body, value.local, carriers)?,
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
        NExpr::CodeRegionRef { .. } => return None,
        NExpr::AggregateMake { ty, fields } => {
            aggregate_make_class(db, body, *ty, fields, carriers, env)?
        }
        NExpr::EnumMake { enum_ty, .. } => RuntimeClass::AggregateValue {
            layout: layout_for_ty_in_env(db, env, *enum_ty),
        },
        NExpr::ReadPlace { place, .. } => normalized_place_class(db, body, place, carriers)
            .or_else(|| {
                top_level_class_for_ty_in_env(db, env, result_ty, AddressSpaceKind::Memory)
            })?,
        NExpr::ExtractEnumField { .. } => {
            top_level_class_for_ty_in_env(db, env, result_ty, AddressSpaceKind::Memory)?
        }
        NExpr::Borrow {
            place, provider, ..
        } => normalized_place_address_class(
            db,
            body,
            place,
            carriers,
            Some(owner.scope()),
            typed_body.assumptions(),
        )
        .or_else(|| {
            provider.map(|provider| RuntimeClass::RawAddr {
                space: address_space_from_provider(provider),
                target: None,
            })
        })?,
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
                body,
                *callee,
                args,
            )
            .unwrap_or_else(|err| {
                panic!(
                    "runtime call resolution failed during return-class inference for {:?}: {err}",
                    body.owner.key(db),
                )
            });
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
                match desired_runtime_param_plan(db, &typed_body, idx) {
                    RuntimeParamPlan::Erased => {}
                    RuntimeParamPlan::Boundary(desired) => {
                        let actual = runtime_visible_arg_class(
                            db,
                            body,
                            arg.local,
                            Some(&desired),
                            carriers,
                        );
                        if let Some(actual) = actual {
                            param_classes.push(match desired {
                                RuntimeBoundarySpec::Exact(desired) => {
                                    preserve_provider_space(&actual, &desired)
                                }
                                RuntimeBoundarySpec::BorrowLike { .. } => actual,
                            });
                        }
                    }
                    RuntimeParamPlan::PassActual => {
                        if let Some(actual) =
                            runtime_visible_arg_class(db, body, arg.local, None, carriers)
                        {
                            param_classes.push(actual);
                        }
                    }
                }
            }
            for arg in effect_args {
                let plan =
                    runtime_effect_binding_plan_for_binding_idx(db, semantic, arg.binding_idx);
                if let Some(class) = effect_arg_class(db, body, env, arg, plan.as_ref(), carriers) {
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

fn aggregate_make_class<'db>(
    db: &'db dyn MirDb,
    body: &NormalizedSemanticBody<'db>,
    ty: TyId<'db>,
    fields: &[NOperand],
    carriers: &[RuntimeCarrier<'db>],
    env: RuntimeTypeEnv<'db>,
) -> Option<RuntimeClass<'db>> {
    if let Some(class) = top_level_class_for_ty_in_env(db, env, ty, AddressSpaceKind::Memory)
        && !matches!(class, RuntimeClass::AggregateValue { .. })
    {
        return Some(class);
    }
    let field_tys = if ty.is_array(db) {
        let (_, args) = ty.decompose_ty_app(db);
        let elem_ty = args.first().copied().expect("array element type");
        vec![elem_ty; fields.len()]
    } else {
        ty.field_types(db)
    };
    if field_tys.len() != fields.len() {
        return None;
    }
    let mut field_classes = Vec::with_capacity(fields.len());
    for (field, field_ty) in fields.iter().copied().zip(field_tys.iter().copied()) {
        let class =
            match boundary_spec_for_ty_in_env(db, env, field_ty, AddressSpaceKind::Memory) {
                Some(boundary) => {
                    runtime_visible_arg_class(db, body, field.local, Some(&boundary), carriers).map(
                        |actual| match &boundary {
                            RuntimeBoundarySpec::Exact(desired) => {
                                preserve_provider_space(&actual, desired)
                            }
                            RuntimeBoundarySpec::BorrowLike { .. } => actual,
                        },
                    )
                }
                None => materialized_value_class(db, body, field.local, carriers),
            }
            .unwrap_or_else(|| {
                stored_class_for_ty_in_context(db, field_ty, env.scope, env.assumptions)
            });
        field_classes.push(class);
    }
    Some(RuntimeClass::AggregateValue {
        layout: layout_for_aggregate_instance_in_context(
            db,
            ty,
            &field_classes,
            env.scope,
            env.assumptions,
        ),
    })
}

fn runtime_visible_arg_class<'db>(
    db: &'db dyn MirDb,
    body: &NormalizedSemanticBody<'db>,
    local: SLocalId,
    desired: Option<&RuntimeBoundarySpec<'db>>,
    carriers: &[RuntimeCarrier<'db>],
) -> Option<RuntimeClass<'db>> {
    let desired = desired.map(|boundary| {
        specialize_boundary_for_runtime_source(db, body, local, boundary, carriers)
    });
    let owner = body.owner.key(db).owner(db);
    let typed_body = body.owner.key(db).instantiate_typed_body(db);
    let scope = Some(owner.scope());
    let assumptions = typed_body.assumptions();
    match desired.as_ref() {
        Some(RuntimeBoundarySpec::Exact(RuntimeClass::AggregateValue { .. }))
            if boundary_source_uses_transport_sensitive_aggregate(
                db,
                body.locals.get(local.index())?.ty,
                scope,
                assumptions,
            ) =>
        {
            actual_aggregate_class_for_semantic_source(db, body, local, carriers)
        }
        Some(RuntimeBoundarySpec::Exact(
            RuntimeClass::Ref { .. } | RuntimeClass::RawAddr { .. },
        )) => carrier_value_class(local, carriers),
        Some(RuntimeBoundarySpec::BorrowLike { .. }) => {
            borrow_like_runtime_visible_arg_class(db, body, local, &desired?, carriers)
        }
        Some(RuntimeBoundarySpec::Exact(_)) | None => {
            materialized_value_class(db, body, local, carriers)
        }
    }
}

pub(crate) fn specialize_boundary_for_runtime_source<'db>(
    db: &'db dyn MirDb,
    body: &NormalizedSemanticBody<'db>,
    local: SLocalId,
    boundary: &RuntimeBoundarySpec<'db>,
    carriers: &[RuntimeCarrier<'db>],
) -> RuntimeBoundarySpec<'db> {
    let actual = || {
        carrier_value_class(local, carriers)
            .or_else(|| semantic_value_class(db, body, local, carriers))
    };
    match boundary {
        RuntimeBoundarySpec::Exact(desired) => {
            if let Some(actual) = actual() {
                let preserved = preserve_provider_space(&actual, desired);
                if preserved == actual {
                    return RuntimeBoundarySpec::Exact(preserved);
                }
                if matches!(desired, RuntimeClass::AggregateValue { .. })
                    && let Some(actual) =
                        actual_aggregate_class_for_semantic_source(db, body, local, carriers)
                {
                    return RuntimeBoundarySpec::Exact(actual);
                }
            }
            boundary.clone()
        }
        RuntimeBoundarySpec::BorrowLike {
            pointee: RuntimeClass::AggregateValue { .. },
            access,
            allow,
        } => actual_aggregate_class_for_semantic_source(db, body, local, carriers)
            .map(|pointee| RuntimeBoundarySpec::BorrowLike {
                pointee,
                access: *access,
                allow: allow.clone(),
            })
            .unwrap_or_else(|| boundary.clone()),
        _ => boundary.clone(),
    }
}

fn borrow_like_runtime_visible_arg_class<'db>(
    db: &'db dyn MirDb,
    body: &NormalizedSemanticBody<'db>,
    local: SLocalId,
    boundary: &RuntimeBoundarySpec<'db>,
    carriers: &[RuntimeCarrier<'db>],
) -> Option<RuntimeClass<'db>> {
    let owner = body.owner.key(db).owner(db);
    let typed_body = body.owner.key(db).instantiate_typed_body(db);
    let scope = Some(owner.scope());
    let assumptions = typed_body.assumptions();
    let class_ctxt = LocalRuntimeClassCtxt {
        body,
        carriers,
        scope,
        assumptions,
    };
    let local_data = body.locals.get(local.index())?;

    if let Some(place) = nonself_backing_value_place(body, local) {
        let class = normalized_place_address_class(db, body, place, carriers, scope, assumptions)?;
        if runtime_class_satisfies_boundary(&class, boundary) {
            return Some(class);
        }
    }

    let value_class = semantic_value_class(db, body, local, carriers)?;
    let root_class = local_place_root_class(
        db,
        &class_ctxt,
        local,
        local_data,
        carriers.get(local.index())?,
    )?;
    let class =
        ref_class_for_place_result(&root_class, &value_class, AddressSpaceKind::Memory, false);
    if runtime_class_satisfies_boundary(&class, boundary) {
        return Some(class);
    }

    if let Some(class) = carrier_value_class(local, carriers)
        && runtime_class_satisfies_boundary(&class, boundary)
    {
        return Some(class);
    }

    None
}

fn nonself_backing_value_place<'a, 'db>(
    body: &'a NormalizedSemanticBody<'db>,
    local: SLocalId,
) -> Option<&'a NSPlace<'db>> {
    let place = body.local(local)?.backing_place()?;
    (!is_self_rooted_value_place(body, local, place)).then_some(place)
}

fn is_self_rooted_value_place<'db>(
    body: &NormalizedSemanticBody<'db>,
    local: SLocalId,
    place: &NSPlace<'db>,
) -> bool {
    if !place.path.is_empty() {
        return false;
    }
    match place.root {
        NSPlaceRoot::CarrierDerefLocal(root_local) => root_local == local,
        NSPlaceRoot::Root(root) => matches!(
            body.root(root),
            Some(NBorrowRoot::Param { local: root_local, .. } | NBorrowRoot::LocalSlot { local: root_local })
                if *root_local == local
        ),
    }
}

fn snapshot_source_place<'a, 'db>(
    body: &'a NormalizedSemanticBody<'db>,
    local: SLocalId,
) -> Option<&'a NSPlace<'db>> {
    body.local(local)?.snapshot_source_place()
}

fn snapshot_source_value_class<'db>(
    db: &'db dyn MirDb,
    body: &NormalizedSemanticBody<'db>,
    local: SLocalId,
    carriers: &[RuntimeCarrier<'db>],
) -> Option<RuntimeClass<'db>> {
    normalized_place_class(db, body, snapshot_source_place(body, local)?, carriers)
}

pub(crate) fn actual_aggregate_class_for_semantic_source<'db>(
    db: &'db dyn MirDb,
    body: &NormalizedSemanticBody<'db>,
    local: SLocalId,
    carriers: &[RuntimeCarrier<'db>],
) -> Option<RuntimeClass<'db>> {
    snapshot_source_value_class(db, body, local, carriers)
        .and_then(|class| actual_aggregate_class_from_runtime_source(&class))
        .or_else(|| {
            semantic_value_class(db, body, local, carriers)
                .and_then(|class| actual_aggregate_class_from_runtime_source(&class))
        })
}

pub(crate) fn materialized_value_class<'db>(
    db: &'db dyn MirDb,
    body: &NormalizedSemanticBody<'db>,
    local: SLocalId,
    carriers: &[RuntimeCarrier<'db>],
) -> Option<RuntimeClass<'db>> {
    let owner = body.owner.key(db).owner(db);
    let typed_body = body.owner.key(db).instantiate_typed_body(db);
    let scope = Some(owner.scope());
    let assumptions = typed_body.assumptions();
    let local_data = body.locals.get(local.index())?;
    match (&local_data.facts.interface, &local_data.facts.origin) {
        (NLocalInterface::Erased, _) => None,
        (NLocalInterface::DirectValue, NLocalOrigin::RootProvider(_)) => {
            semantic_value_class(db, body, local, carriers)
        }
        (NLocalInterface::DirectValue, _) => {
            actual_aggregate_class_for_semantic_source(db, body, local, carriers).or_else(|| {
                top_level_class_for_ty_in_context(
                    db,
                    local_data.ty,
                    AddressSpaceKind::Memory,
                    scope,
                    assumptions,
                )
            })
        }
        (
            NLocalInterface::PlaceBoundValue
            | NLocalInterface::PlaceCarrier
            | NLocalInterface::DirectCarrier,
            _,
        ) => semantic_value_class(db, body, local, carriers),
    }
}

pub(crate) fn desired_runtime_param_plan<'db>(
    db: &'db dyn MirDb,
    typed_body: &hir::analysis::ty::ty_check::TypedBody<'db>,
    idx: usize,
) -> RuntimeParamPlan<'db> {
    let Some(binding) = typed_body.param_binding(idx) else {
        return RuntimeParamPlan::Erased;
    };
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
        return RuntimeParamPlan::PassActual;
    }
    let Some(boundary) = boundary_spec_for_ty_in_context(
        db,
        binding_ty,
        AddressSpaceKind::Memory,
        scope,
        assumptions,
    ) else {
        return RuntimeParamPlan::Erased;
    };
    if matches!(
        boundary,
        RuntimeBoundarySpec::Exact(RuntimeClass::AggregateValue { .. })
    ) && aggregate_transport_depends_on_runtime_source(db, binding_ty, scope, assumptions)
    {
        return RuntimeParamPlan::PassActual;
    }
    RuntimeParamPlan::Boundary(runtime_param_boundary(db, typed_body, binding, boundary))
}

pub(crate) fn resolve_runtime_call_key<'db>(
    db: &'db dyn MirDb,
    caller_key: SemanticInstanceKey<'db>,
    caller_typed_body: &hir::analysis::ty::ty_check::TypedBody<'db>,
    body: &NormalizedSemanticBody<'db>,
    callee: SemanticCalleeRef<'db>,
    args: &[NOperand],
) -> Result<SemanticInstanceKey<'db>, crate::runtime::LowerError> {
    let callee_key = callee.key;
    let callee_semantic = get_or_build_semantic_instance(db, callee_key);
    if contract_metadata_builtin(db, callee_semantic).is_some() {
        return Ok(callee_key);
    }
    let BodyOwner::Func(func) = callee_key.owner(db) else {
        return Ok(callee_key);
    };
    let Some(trait_) = func.containing_trait(db) else {
        return Ok(callee_key);
    };
    if func.body(db).is_some() {
        return Ok(callee_key);
    }
    let Some(method_name) = func.name(db).to_opt() else {
        return Err(crate::runtime::LowerError::Unsupported(format!(
            "runtime trait-call resolution reached an unnamed declaration-only method: caller={caller_key:?} callee={callee_key:?}"
        )));
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
            return Err(crate::runtime::LowerError::Unsupported(format!(
                "runtime trait-call resolution is missing a self argument: caller={caller_key:?} callee={callee_key:?}"
            )));
        };
        let Some(self_ty) =
            concrete_runtime_self_ty_for_call_arg(db, caller_typed_body, body, arg.local)
        else {
            return Err(crate::runtime::LowerError::Unsupported(format!(
                "runtime trait-call resolution could not infer the concrete self type: caller={caller_key:?} callee={callee_key:?} local={:?}",
                arg.local,
            )));
        };
        let mut inst_args = original_inst
            .map(|inst| inst.args(db).to_vec())
            .unwrap_or_else(|| vec![self_ty]);
        let Some(first) = inst_args.first_mut() else {
            return Err(crate::runtime::LowerError::Unsupported(format!(
                "runtime trait-call resolution produced an empty trait-inst arg list: caller={caller_key:?} callee={callee_key:?}"
            )));
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
            return Err(crate::runtime::LowerError::Unsupported(format!(
                "runtime trait-call resolution is missing a trait witness for a declaration-only method: caller={caller_key:?} callee={callee_key:?}"
            )));
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
        return Err(crate::runtime::LowerError::Unsupported(format!(
            "runtime trait-call resolution failed to resolve a concrete impl body: caller={caller_key:?} decl={callee_key:?} method={} concrete_inst={} original_inst={}",
            method_name.data(db),
            concrete_inst.pretty_print(db, false),
            original_inst
                .map(|inst| inst.pretty_print(db, false))
                .unwrap_or_else(|| "<none>".to_string()),
        )));
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
    Ok(SemanticInstanceKey::new(
        db,
        BodyOwner::Func(impl_func),
        GenericSubst::new(db, impl_args),
        hir::analysis::semantic::EffectProviderSubst::empty(db),
        ImplEnv::new(
            db,
            caller_key.impl_env(db).normalization_scope(db),
            assumptions,
            witnesses.into_iter().collect::<Vec<_>>(),
        ),
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
    body: &NormalizedSemanticBody<'db>,
    local: SLocalId,
) -> Option<TyId<'db>> {
    let scope = caller_typed_body.body().map(|body| body.scope());
    let assumptions = caller_typed_body.assumptions();
    let normalized = |ty| normalize_runtime_self_ty(db, ty, scope, assumptions);
    let local_data = body.locals.get(local.index())?;
    match (
        &local_data.facts.interface,
        &local_data.facts.origin,
        &local_data.lowering,
    ) {
        (NLocalInterface::Erased, _, _) => None,
        (
            NLocalInterface::DirectValue | NLocalInterface::DirectCarrier,
            NLocalOrigin::RootProvider(provider),
            _,
        ) => Some(normalized(provider.provider_ty)),
        (
            NLocalInterface::PlaceBoundValue,
            NLocalOrigin::RootProvider(provider),
            NormalizedBindingLowering::PlaceBoundValue { value_ty, .. },
        ) => Some(normalized(
            provider.semantics.target_ty.unwrap_or(*value_ty),
        )),
        (
            NLocalInterface::PlaceBoundValue,
            NLocalOrigin::SelfRooted | NLocalOrigin::AliasedPlace,
            NormalizedBindingLowering::PlaceBoundValue { value_ty, .. },
        ) => Some(normalized(*value_ty)),
        (NLocalInterface::DirectValue, _, _) => Some(normalized(local_data.ty)),
        (
            NLocalInterface::PlaceCarrier | NLocalInterface::DirectCarrier,
            _,
            NormalizedBindingLowering::CarrierLocal { target_ty, .. },
        ) => Some(normalized(*target_ty)),
        _ => None,
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
            RuntimeClass::Ref {
                pointee: actual_pointee,
                kind: actual_kind,
                view: actual_view,
            },
            RuntimeClass::Ref {
                pointee: desired_pointee,
                view: desired_view,
                ..
            },
        ) if actual_pointee == desired_pointee && actual_view == desired_view => actual.clone(),
        (
            RuntimeClass::RawAddr {
                space: actual_space,
                target: actual_target,
            },
            RuntimeClass::Ref { pointee, .. },
        ) if actual_target == &pointee.aggregate_layout() => RuntimeClass::RawAddr {
            space: *actual_space,
            target: *actual_target,
        },
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
    local: SLocalId,
    carriers: &[RuntimeCarrier<'db>],
) -> Option<RuntimeClass<'db>> {
    let owner = body.owner.key(db).owner(db);
    let typed_body = body.owner.key(db).instantiate_typed_body(db);
    let scope = Some(owner.scope());
    let assumptions = typed_body.assumptions();
    let local_data = body.locals.get(local.index())?;
    match local_data.facts.interface {
        NLocalInterface::Erased => None,
        NLocalInterface::DirectValue | NLocalInterface::DirectCarrier => {
            carrier_value_class(local, carriers)
        }
        NLocalInterface::PlaceCarrier => carrier_value_class(local, carriers).or_else(|| {
            let NormalizedBindingLowering::CarrierLocal { target_ty, .. } = &local_data.lowering
            else {
                panic!("place-carrier local missing carrier lowering: {local:?}");
            };
            Some(stored_class_for_ty_in_context(
                db,
                *target_ty,
                scope,
                assumptions,
            ))
        }),
        NLocalInterface::PlaceBoundValue => normalized_local_place_class(db, body, local, carriers)
            .or_else(|| {
                let NormalizedBindingLowering::PlaceBoundValue { value_ty, .. } =
                    &local_data.lowering
                else {
                    panic!("place-bound local missing place-bound lowering: {local:?}");
                };
                Some(stored_class_for_ty_in_context(
                    db,
                    *value_ty,
                    scope,
                    assumptions,
                ))
            }),
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

pub(crate) fn normalized_place_class<'db>(
    db: &'db dyn MirDb,
    body: &NormalizedSemanticBody<'db>,
    place: &NSPlace<'db>,
    carriers: &[RuntimeCarrier<'db>],
) -> Option<RuntimeClass<'db>> {
    let owner = body.owner.key(db).owner(db);
    let typed_body = body.owner.key(db).instantiate_typed_body(db);
    let scope = Some(owner.scope());
    let assumptions = typed_body.assumptions();
    let mut current =
        normalized_place_root_class(db, body, place.root.clone(), carriers, scope, assumptions)?;
    for projection in place.path.iter() {
        current = match projection {
            Projection::Field(field) => project_field_class(
                db,
                current,
                FieldIndex((*field).try_into().expect("field index fits")),
            ),
            Projection::Index(_) => project_index_class(db, current),
            Projection::Deref => match current {
                RuntimeClass::Ref { pointee, .. } => *pointee,
                RuntimeClass::RawAddr {
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
                RuntimeClass::Ref { pointee, .. } => match pointee.aggregate_layout() {
                    Some(layout) => match layout.data(db) {
                        Layout::Enum(layout) => RuntimeClass::Scalar(layout.tag),
                        Layout::Struct(_) | Layout::Array(_) => {
                            panic!("invalid discriminant projection class")
                        }
                    },
                    None => panic!("invalid discriminant projection class"),
                },
                RuntimeClass::AggregateValue { layout }
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

pub(crate) fn normalized_place_address_class<'db>(
    db: &'db dyn MirDb,
    body: &NormalizedSemanticBody<'db>,
    place: &NSPlace<'db>,
    carriers: &[RuntimeCarrier<'db>],
    scope: Option<hir::hir_def::scope_graph::ScopeId<'db>>,
    assumptions: PredicateListId<'db>,
) -> Option<RuntimeClass<'db>> {
    let value_class = normalized_place_class(db, body, place, carriers)?;
    let root_class = normalized_place_root_transport_class(
        db,
        body,
        place.root.clone(),
        carriers,
        scope,
        assumptions,
    )?;
    let (root_space, force_raw) = match place.root {
        NSPlaceRoot::CarrierDerefLocal(_) => (AddressSpaceKind::Memory, false),
        NSPlaceRoot::Root(root) => match body.root(root)? {
            NBorrowRoot::Param { .. } | NBorrowRoot::LocalSlot { .. } => {
                (AddressSpaceKind::Memory, false)
            }
            NBorrowRoot::Provider { binding } => (provider_root_space(binding, &root_class), false),
        },
    };
    Some(ref_class_for_place_result(
        &root_class,
        &value_class,
        root_space,
        force_raw,
    ))
}

fn normalized_place_root_transport_class<'db>(
    db: &'db dyn MirDb,
    body: &NormalizedSemanticBody<'db>,
    root: NSPlaceRoot,
    carriers: &[RuntimeCarrier<'db>],
    scope: Option<hir::hir_def::scope_graph::ScopeId<'db>>,
    assumptions: PredicateListId<'db>,
) -> Option<RuntimeClass<'db>> {
    match root {
        NSPlaceRoot::CarrierDerefLocal(local) => {
            carrier_value_class(local, carriers).or_else(|| {
                fallback_root_transport_class(
                    db,
                    body.locals.get(local.index())?,
                    scope,
                    assumptions,
                )
            })
        }
        NSPlaceRoot::Root(root) => match body.root(root)? {
            NBorrowRoot::Param { local, .. } | NBorrowRoot::LocalSlot { local } => {
                carrier_value_class(*local, carriers).or_else(|| {
                    fallback_root_transport_class(
                        db,
                        body.locals.get(local.index())?,
                        scope,
                        assumptions,
                    )
                })
            }
            NBorrowRoot::Provider { binding } => {
                actual_runtime_visible_root_provider_class(db, body, carriers, binding)
                    .map(|(_, class)| class)
                    .or_else(|| {
                        runtime_class_for_effect_binding_provider_in_context(
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
                        })
                    })
            }
        },
    }
}

fn normalized_place_root_class<'db>(
    db: &'db dyn MirDb,
    body: &NormalizedSemanticBody<'db>,
    root: NSPlaceRoot,
    carriers: &[RuntimeCarrier<'db>],
    scope: Option<hir::hir_def::scope_graph::ScopeId<'db>>,
    assumptions: PredicateListId<'db>,
) -> Option<RuntimeClass<'db>> {
    let class_ctxt = LocalRuntimeClassCtxt {
        body,
        carriers,
        scope,
        assumptions,
    };
    match root {
        NSPlaceRoot::CarrierDerefLocal(local) => {
            let local_data = body.locals.get(local.index())?;
            local_place_root_class(
                db,
                &class_ctxt,
                local,
                local_data,
                carriers.get(local.index())?,
            )
        }
        NSPlaceRoot::Root(root) => match body.root(root)? {
            NBorrowRoot::Param { local, .. } | NBorrowRoot::LocalSlot { local } => {
                local_place_root_class(
                    db,
                    &class_ctxt,
                    *local,
                    body.locals.get(local.index())?,
                    carriers.get(local.index())?,
                )
            }
            NBorrowRoot::Provider { binding } => {
                let provider_class =
                    actual_runtime_visible_root_provider_class(db, body, carriers, binding)
                        .map(|(_, class)| class)
                        .or_else(|| {
                            runtime_class_for_effect_binding_provider_in_context(
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
                            })
                        })?;
                Some(provider_root_place_class(
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
    let layout = class
        .aggregate_layout()
        .unwrap_or_else(|| panic!("invalid variant-field projection class"));
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

fn effect_arg_class<'db>(
    db: &'db dyn MirDb,
    body: &NormalizedSemanticBody<'db>,
    env: RuntimeTypeEnv<'db>,
    arg: &NEffectArg<'db>,
    plan: Option<&RuntimeEffectBindingPlan<'db>>,
    carriers: &[RuntimeCarrier<'db>],
) -> Option<RuntimeClass<'db>> {
    if plan.is_none() && arg.provider.is_none() && arg.target_ty.is_none() {
        return match (&arg.pass_mode, &arg.arg) {
            (EffectPassMode::ByValue | EffectPassMode::Unknown, NEffectArgValue::Value(value)) => {
                materialized_value_class(db, body, value.local, carriers)
            }
            (EffectPassMode::ByValue | EffectPassMode::Unknown, NEffectArgValue::Place(_))
            | (
                EffectPassMode::ByPlace | EffectPassMode::ByTempPlace,
                NEffectArgValue::Value(_) | NEffectArgValue::Place(_),
            ) => panic!(
                "effect arg without provider/target should infer as a plain value: owner={:?}; arg={arg:?}",
                body.owner.key(db).owner(db),
            ),
        };
    }
    let effect_space = || resolved_effect_arg_address_space(db, body, arg);
    let boundary = desired_runtime_effect_arg_boundary(db, env, arg, plan, effect_space());
    match arg.pass_mode {
        EffectPassMode::ByValue | EffectPassMode::Unknown => {
            if let Some(boundary) = boundary.as_ref() {
                return match &arg.arg {
                    NEffectArgValue::Place(place) => runtime_visible_place_arg_class_for_boundary(
                        db,
                        body,
                        place,
                        boundary,
                        carriers,
                        env.scope,
                        env.assumptions,
                    ),
                    NEffectArgValue::Value(value) => {
                        runtime_visible_arg_class(db, body, value.local, Some(boundary), carriers)
                    }
                };
            }
            match arg.arg {
                NEffectArgValue::Place(_) => Some(provider_class_for_target_in_context(
                    db,
                    arg.target_ty,
                    effect_space(),
                    env.scope,
                    env.assumptions,
                )),
                NEffectArgValue::Value(value) => match carriers.get(value.local.index())? {
                    RuntimeCarrier::Value(class) => Some(class.clone()),
                    RuntimeCarrier::Erased if arg.provider.is_none() && arg.target_ty.is_none() => {
                        None
                    }
                    RuntimeCarrier::Erased => Some(provider_class_for_target_in_context(
                        db,
                        arg.target_ty,
                        effect_space(),
                        env.scope,
                        env.assumptions,
                    )),
                },
            }
        }
        EffectPassMode::ByPlace | EffectPassMode::ByTempPlace => {
            let boundary = boundary.unwrap_or_else(|| {
                let Some(target_ty) = arg.target_ty else {
                    return RuntimeBoundarySpec::Exact(provider_class_for_target_in_context(
                        db,
                        None,
                        effect_space(),
                        env.scope,
                        env.assumptions,
                    ));
                };
                RuntimeBoundarySpec::BorrowLike {
                    pointee: stored_class_for_ty_in_context(
                        db,
                        target_ty,
                        env.scope,
                        env.assumptions,
                    ),
                    access: BorrowAccess::ReadWrite,
                    allow: default_borrow_transport_set(BorrowAccess::ReadWrite, effect_space()),
                }
            });
            match &arg.arg {
                NEffectArgValue::Place(place) => runtime_visible_place_arg_class_for_boundary(
                    db,
                    body,
                    place,
                    &boundary,
                    carriers,
                    env.scope,
                    env.assumptions,
                ),
                NEffectArgValue::Value(value) => borrow_like_runtime_visible_arg_class(
                    db,
                    body,
                    value.local,
                    &boundary,
                    carriers,
                ),
            }
        }
    }
}

pub(crate) fn desired_runtime_effect_arg_boundary<'db>(
    db: &'db dyn MirDb,
    env: RuntimeTypeEnv<'db>,
    arg: &NEffectArg<'db>,
    plan: Option<&RuntimeEffectBindingPlan<'db>>,
    effect_space: AddressSpaceKind,
) -> Option<RuntimeBoundarySpec<'db>> {
    if let Some(plan) = plan {
        return Some(plan.boundary.clone());
    }
    arg.target_ty.map(|target_ty| match arg.pass_mode {
        EffectPassMode::ByPlace | EffectPassMode::ByTempPlace => RuntimeBoundarySpec::BorrowLike {
            pointee: stored_class_for_ty_in_context(db, target_ty, env.scope, env.assumptions),
            access: BorrowAccess::ReadWrite,
            allow: default_borrow_transport_set(BorrowAccess::ReadWrite, effect_space),
        },
        EffectPassMode::ByValue | EffectPassMode::Unknown => boundary_spec_for_ty_in_env(
            db,
            env,
            target_ty,
            effect_space,
        )
        .unwrap_or(RuntimeBoundarySpec::Exact(
            provider_class_for_target_in_env(db, env, Some(target_ty), effect_space),
        )),
    })
}

fn runtime_visible_place_arg_class_for_boundary<'db>(
    db: &'db dyn MirDb,
    body: &NormalizedSemanticBody<'db>,
    place: &NSPlace<'db>,
    boundary: &RuntimeBoundarySpec<'db>,
    carriers: &[RuntimeCarrier<'db>],
    scope: Option<hir::hir_def::scope_graph::ScopeId<'db>>,
    assumptions: PredicateListId<'db>,
) -> Option<RuntimeClass<'db>> {
    match boundary {
        RuntimeBoundarySpec::Exact(target) => Some(target.clone()),
        RuntimeBoundarySpec::BorrowLike { pointee, allow, .. } => {
            let class =
                normalized_place_address_class(db, body, place, carriers, scope, assumptions)?;
            if runtime_class_satisfies_boundary(&class, boundary) {
                Some(class)
            } else if let Some(layout) = pointee.aggregate_layout()
                && allow.allow_object
            {
                Some(RuntimeClass::object_ref(layout))
            } else if allow.allow_raw_addr {
                Some(RuntimeClass::RawAddr {
                    space: AddressSpaceKind::Memory,
                    target: None,
                })
            } else {
                None
            }
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
    let matches = |path: &str| lib_func_matches(db, func, path);
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
        "__checked_pow" => GenericNumericIntrinsicKind::CheckedBinary(ArithBinOp::Pow),
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
        return RuntimeClass::object_ref(layout_for_ty_in_context(
            db,
            ty,
            typed_body.body().map(|body| body.scope()),
            typed_body.assumptions(),
        ));
    }
    actual
}

pub(crate) fn runtime_param_boundary<'db>(
    db: &'db dyn MirDb,
    typed_body: &hir::analysis::ty::ty_check::TypedBody<'db>,
    binding: hir::analysis::ty::ty_check::LocalBinding<'db>,
    boundary: RuntimeBoundarySpec<'db>,
) -> RuntimeBoundarySpec<'db> {
    match boundary {
        RuntimeBoundarySpec::Exact(actual) => {
            RuntimeBoundarySpec::Exact(runtime_param_class(db, typed_body, binding, actual))
        }
        RuntimeBoundarySpec::BorrowLike {
            pointee,
            access,
            allow,
        } => {
            let ty = runtime_repr_ty_in_context(
                db,
                typed_body.binding_ty(db, binding),
                typed_body.body().map(|body| body.scope()),
                typed_body.assumptions(),
            );
            if binding.is_mut() && ty.as_enum(db).is_some() {
                return RuntimeBoundarySpec::Exact(RuntimeClass::object_ref(
                    layout_for_ty_in_context(
                        db,
                        ty,
                        typed_body.body().map(|body| body.scope()),
                        typed_body.assumptions(),
                    ),
                ));
            }
            RuntimeBoundarySpec::BorrowLike {
                pointee,
                access,
                allow,
            }
        }
    }
}

pub(crate) fn runtime_class_satisfies_boundary<'db>(
    class: &RuntimeClass<'db>,
    boundary: &RuntimeBoundarySpec<'db>,
) -> bool {
    match boundary {
        RuntimeBoundarySpec::Exact(expected) => preserve_provider_space(class, expected) == *class,
        RuntimeBoundarySpec::BorrowLike { pointee, allow, .. } => match class {
            RuntimeClass::Ref {
                pointee: actual_pointee,
                kind: RefKind::Object,
                view: RefView::Whole,
            } => allow.allow_object && **actual_pointee == *pointee,
            RuntimeClass::Ref {
                pointee: actual_pointee,
                kind: RefKind::Const,
                view: RefView::Whole,
            } => allow.allow_const && **actual_pointee == *pointee,
            RuntimeClass::Ref {
                pointee: actual_pointee,
                kind: RefKind::Provider { space, .. },
                view: RefView::Whole,
            } => allow.provider_spaces.contains(space) && **actual_pointee == *pointee,
            RuntimeClass::Ref {
                view: RefView::EnumVariant(_),
                ..
            } => false,
            RuntimeClass::RawAddr { .. } => allow.allow_raw_addr,
            RuntimeClass::Scalar(_) | RuntimeClass::AggregateValue { .. } => false,
        },
    }
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
        || !matches!(
            top_level_class_for_ty_in_context(db, ty, AddressSpaceKind::Memory, scope, assumptions),
            None | Some(RuntimeClass::Scalar(_))
        )
}

pub(crate) fn top_level_class_for_ty_in_env<'db>(
    db: &'db dyn MirDb,
    env: RuntimeTypeEnv<'db>,
    ty: TyId<'db>,
    default_space: AddressSpaceKind,
) -> Option<RuntimeClass<'db>> {
    top_level_class_for_ty_in_context(db, ty, default_space, env.scope, env.assumptions)
}

pub(crate) fn boundary_spec_for_ty_in_env<'db>(
    db: &'db dyn MirDb,
    env: RuntimeTypeEnv<'db>,
    ty: TyId<'db>,
    default_space: AddressSpaceKind,
) -> Option<RuntimeBoundarySpec<'db>> {
    boundary_spec_for_ty_in_context(db, ty, default_space, env.scope, env.assumptions)
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

pub(crate) fn boundary_spec_for_ty_in_context<'db>(
    db: &'db dyn MirDb,
    ty: TyId<'db>,
    default_space: AddressSpaceKind,
    scope: Option<hir::hir_def::scope_graph::ScopeId<'db>>,
    assumptions: PredicateListId<'db>,
) -> Option<RuntimeBoundarySpec<'db>> {
    let ty = runtime_repr_ty_in_context(db, ty, scope, assumptions);
    if ty == TyId::unit(db) || is_zero_sized_in_context(db, ty, scope, assumptions) {
        return None;
    }
    if let Some((kind, inner)) = ty.as_borrow(db) {
        if is_zero_sized_in_context(db, inner, scope, assumptions) {
            return None;
        }
        let access = match kind {
            BorrowKind::Ref => BorrowAccess::ReadOnly,
            BorrowKind::Mut => BorrowAccess::ReadWrite,
        };
        return Some(RuntimeBoundarySpec::BorrowLike {
            pointee: stored_class_for_ty_in_context(db, inner, scope, assumptions),
            access,
            allow: default_borrow_transport_set(access, default_space),
        });
    }
    top_level_class_for_ty_in_context(db, ty, default_space, scope, assumptions)
        .map(RuntimeBoundarySpec::Exact)
}

pub(crate) fn default_borrow_transport_set(
    access: BorrowAccess,
    default_space: AddressSpaceKind,
) -> BorrowTransportSet {
    let mut provider_spaces = IndexSet::new();
    provider_spaces.insert(default_space);
    provider_spaces.insert(AddressSpaceKind::Memory);
    provider_spaces.insert(AddressSpaceKind::Storage);
    provider_spaces.insert(AddressSpaceKind::Transient);
    if matches!(access, BorrowAccess::ReadOnly) {
        provider_spaces.insert(AddressSpaceKind::Calldata);
    }
    BorrowTransportSet {
        allow_object: true,
        allow_const: matches!(access, BorrowAccess::ReadOnly),
        provider_spaces: provider_spaces.into_iter().collect(),
        allow_raw_addr: true,
    }
}

fn aggregate_transport_depends_on_runtime_source<'db>(
    db: &'db dyn MirDb,
    ty: TyId<'db>,
    scope: Option<hir::hir_def::scope_graph::ScopeId<'db>>,
    assumptions: PredicateListId<'db>,
) -> bool {
    fn inner<'db>(
        db: &'db dyn MirDb,
        ty: TyId<'db>,
        scope: Option<hir::hir_def::scope_graph::ScopeId<'db>>,
        assumptions: PredicateListId<'db>,
        visiting: &mut FxHashSet<TyId<'db>>,
    ) -> bool {
        let ty = runtime_repr_ty_in_context(db, ty, scope, assumptions);
        if !visiting.insert(ty) {
            return false;
        }

        let result = if ty.as_borrow(db).is_some() {
            true
        } else if ty.as_capability(db).is_some()
            || effect_handle_class_for_ty(db, ty, scope, assumptions).is_some()
            || scalar_class_for_ty_in_context(db, ty, scope, assumptions).is_some()
        {
            false
        } else if ty.is_array(db) {
            let (_, args) = ty.decompose_ty_app(db);
            args.first()
                .copied()
                .is_some_and(|elem| inner(db, elem, scope, assumptions, visiting))
        } else if ty.is_tuple(db) || ty.is_struct(db) {
            ty.field_types(db)
                .into_iter()
                .any(|field| inner(db, field, scope, assumptions, visiting))
        } else if let Some(enum_) = ty.as_enum(db) {
            let adt = enum_.as_adt(db);
            let args = ty.generic_args(db);
            adt.fields(db)
                .iter()
                .enumerate()
                .any(|(variant_idx, variant)| {
                    (0..variant.num_types()).any(|field_idx| {
                        inner(
                            db,
                            adt.fields(db)[variant_idx]
                                .ty(db, field_idx)
                                .instantiate(db, args),
                            scope,
                            assumptions,
                            visiting,
                        )
                    })
                })
        } else {
            false
        };

        visiting.remove(&ty);
        result
    }

    inner(db, ty, scope, assumptions, &mut FxHashSet::default())
}

pub(crate) fn boundary_source_uses_transport_sensitive_aggregate<'db>(
    db: &'db dyn MirDb,
    ty: TyId<'db>,
    scope: Option<hir::hir_def::scope_graph::ScopeId<'db>>,
    assumptions: PredicateListId<'db>,
) -> bool {
    if let Some((_, inner)) = ty.as_borrow(db) {
        return aggregate_transport_depends_on_runtime_source(db, inner, scope, assumptions);
    }
    let repr = runtime_repr_ty_in_context(db, ty, scope, assumptions);
    if let Some((_, inner)) = repr.as_borrow(db) {
        return aggregate_transport_depends_on_runtime_source(db, inner, scope, assumptions);
    }
    aggregate_transport_depends_on_runtime_source(db, repr, scope, assumptions)
}

pub(crate) fn actual_aggregate_class_from_runtime_source<'db>(
    class: &RuntimeClass<'db>,
) -> Option<RuntimeClass<'db>> {
    match class {
        RuntimeClass::AggregateValue { .. } => Some(class.clone()),
        RuntimeClass::Ref { pointee, .. } => pointee
            .aggregate_layout()
            .map(|layout| RuntimeClass::AggregateValue { layout }),
        RuntimeClass::RawAddr {
            target: Some(layout),
            ..
        } => Some(RuntimeClass::AggregateValue { layout: *layout }),
        RuntimeClass::Scalar(_) | RuntimeClass::RawAddr { target: None, .. } => None,
    }
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
        return Some(object_ref_class_for_target_in_context(
            db,
            inner,
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

pub(crate) fn object_ref_class_for_target_in_context<'db>(
    db: &'db dyn MirDb,
    target_ty: TyId<'db>,
    scope: Option<hir::hir_def::scope_graph::ScopeId<'db>>,
    assumptions: PredicateListId<'db>,
) -> RuntimeClass<'db> {
    let target_ty = runtime_repr_ty_in_context(db, target_ty, scope, assumptions);
    RuntimeClass::Ref {
        pointee: Box::new(stored_class_for_ty_in_context(
            db,
            target_ty,
            scope,
            assumptions,
        )),
        kind: RefKind::Object,
        view: RefView::Whole,
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
        Some(target_ty) => RuntimeClass::Ref {
            pointee: Box::new(stored_class_for_ty_in_context(
                db,
                target_ty,
                scope,
                assumptions,
            )),
            kind: RefKind::Provider {
                provider_ty: TyId::borrow_ref_of(db, target_ty),
                space,
            },
            view: RefView::Whole,
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
