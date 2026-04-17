use cranelift_entity::EntityRef;
use hir::{
    analysis::{
        semantic::{
            SLocalId, SemanticInstance, ValueProvenance,
            borrowck::{
                NLocalInterface, NLocalOrigin, NSLocal, NormalizedBindingLowering,
                NormalizedSemanticBody,
            },
            semantic_binding_lowering,
        },
        ty::{trait_resolution::PredicateListId, ty_check::LocalBinding},
    },
    semantic::ProviderBinding,
};
use rustc_hash::FxHashMap;

use crate::{
    db::MirDb,
    runtime::{
        AddressSpaceKind, RefKind, RuntimeCarrier, RuntimeClass, RuntimeLocalLowering,
        RuntimeLocalRoot, RuntimeProviderBinding, RuntimeProviderBindingId,
    },
};

use super::{
    classify::{
        BodyEnv, RuntimeBodyCx, actual_aggregate_class_from_runtime_source, carrier_value_class,
        normalized_place_class_in_context, runtime_class_for_direct_value_provider_in_context,
        runtime_class_for_effect_binding_provider_in_context, runtime_class_for_provider_binding,
    },
    interface::runtime_visible_binding_plans,
    type_info::{
        provider_class_for_target_in_context, stored_class_for_ty_in_context,
        top_level_class_for_ty_in_context,
    },
};

#[derive(Clone, Debug)]
pub(super) struct InferenceResult<'db> {
    pub(super) carriers: Vec<RuntimeCarrier<'db>>,
    pub(super) roots: Vec<RuntimeLocalRoot<'db>>,
    pub(super) semantic_locals: Vec<RuntimeLocalLowering<'db>>,
    pub(super) provider_bindings: Vec<RuntimeProviderBinding<'db>>,
}

pub(super) struct LocalStateInferer<'a, 'db> {
    env: BodyEnv<'a, 'db>,
    carriers: Vec<RuntimeCarrier<'db>>,
}

impl<'a, 'db> LocalStateInferer<'a, 'db> {
    pub(super) fn new(
        env: BodyEnv<'a, 'db>,
        params: &[RuntimeClass<'db>],
        param_locals: &[SLocalId],
    ) -> Self {
        let mut carriers = vec![RuntimeCarrier::Erased; env.body().locals.len()];
        for (class, local) in params.iter().zip(param_locals.iter().copied()) {
            carriers[local.index()] = RuntimeCarrier::Value(class.clone());
        }
        Self { env, carriers }
    }

    pub(super) fn run(mut self) -> InferenceResult<'db> {
        self.seed_root_provider_carriers();
        self.infer_carriers();
        let roots = self.infer_roots();
        let (semantic_locals, provider_bindings) =
            lower_semantic_locals(self.env.with_carriers(&self.carriers));
        InferenceResult {
            carriers: self.carriers,
            roots,
            semantic_locals,
            provider_bindings,
        }
    }

    fn seed_root_provider_carriers(&mut self) {
        for (idx, local) in self.env.body().locals.iter().enumerate() {
            if !matches!(self.carriers[idx], RuntimeCarrier::Erased) {
                continue;
            }
            let class = match (&local.facts.interface, &local.facts.origin) {
                (NLocalInterface::DirectValue, NLocalOrigin::RootProvider(provider)) => {
                    actual_runtime_visible_root_provider_class(
                        self.env.db(),
                        self.env.body(),
                        &self.carriers,
                        provider,
                    )
                    .map(|(_, class)| class)
                    .or_else(|| {
                        runtime_class_for_direct_value_provider_in_context(
                            self.env.db(),
                            provider,
                            self.env.scope(),
                            self.env.assumptions(),
                        )
                    })
                }
                (NLocalInterface::DirectCarrier, NLocalOrigin::RootProvider(provider)) => {
                    actual_runtime_visible_root_provider_class(
                        self.env.db(),
                        self.env.body(),
                        &self.carriers,
                        provider,
                    )
                    .map(|(_, class)| class)
                    .or_else(|| {
                        runtime_class_for_provider_binding(
                            self.env.db(),
                            provider,
                            self.env.scope(),
                            self.env.assumptions(),
                        )
                    })
                }
                _ => None,
            };
            if let Some(class) = class {
                self.carriers[idx] = RuntimeCarrier::Value(class);
            }
        }
    }

    fn infer_carriers(&mut self) {
        loop {
            let mut changed = false;
            for block in &self.env.body().blocks {
                for stmt in &block.stmts {
                    let hir::analysis::semantic::NSStmtKind::Assign { dst, expr } = &stmt.kind
                    else {
                        continue;
                    };
                    let local = &self.env.body().locals[dst.index()];
                    let desired = match self.env.expr_direct_class(&self.carriers, expr, local.ty) {
                        Some(RuntimeClass::AggregateValue { layout })
                            if matches!(local.facts.interface, NLocalInterface::DirectValue)
                                && local.facts.root_demand.needs_projectable_owned_storage() =>
                        {
                            RuntimeCarrier::Value(RuntimeClass::object_ref(layout))
                        }
                        Some(class) => RuntimeCarrier::Value(class),
                        None => continue,
                    };
                    if self.carriers[dst.index()] != desired {
                        self.carriers[dst.index()] = desired;
                        changed = true;
                    }
                }
            }
            if !changed {
                break;
            }
        }
    }

    fn infer_roots(&mut self) -> Vec<RuntimeLocalRoot<'db>> {
        let carriers = self.carriers.clone();
        let cx = self.env.with_carriers(&carriers);
        let mut roots = Vec::with_capacity(cx.body().locals.len());
        for (idx, local) in cx.body().locals.iter().enumerate() {
            let local_id = SLocalId::from_u32(idx as u32);
            let mut carrier = carriers[idx].clone();
            let root = if !local.facts.root_demand.needs_runtime_root() {
                RuntimeLocalRoot::None
            } else {
                infer_runtime_local_root(cx, local_id, &mut carrier)
            };
            self.carriers[idx] = carrier;
            roots.push(root);
        }
        roots
    }
}

fn lower_semantic_locals<'db>(
    cx: RuntimeBodyCx<'_, '_, 'db>,
) -> (
    Vec<RuntimeLocalLowering<'db>>,
    Vec<RuntimeProviderBinding<'db>>,
) {
    let mut provider_bindings = Vec::new();
    for (idx, local) in cx.body().locals.iter().enumerate() {
        let local_id = SLocalId::from_u32(idx as u32);
        let binding = match (&local.facts.interface, &local.facts.origin) {
            (NLocalInterface::DirectValue, NLocalOrigin::RootProvider(provider)) => {
                let (provider_local, provider_class) = actual_runtime_visible_root_provider_class(
                    cx.db(),
                    cx.body(),
                    cx.carriers(),
                    provider,
                )
                .or_else(|| {
                    runtime_class_for_direct_value_provider_in_context(
                        cx.db(),
                        provider,
                        cx.scope(),
                        cx.assumptions(),
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
                    normalized_local_place_class(cx.db(), cx.body(), local_id, cx.carriers())
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
                    cx.db(),
                    cx.body(),
                    cx.carriers(),
                    provider,
                )
                .or_else(|| {
                    runtime_class_for_effect_binding_provider_in_context(
                        cx.db(),
                        provider,
                        cx.scope(),
                        cx.assumptions(),
                    )
                    .or_else(|| {
                        runtime_class_for_direct_value_provider_in_context(
                            cx.db(),
                            provider,
                            cx.scope(),
                            cx.assumptions(),
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
                    normalized_local_place_class(cx.db(), cx.body(), local_id, cx.carriers())
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
                let (provider_local, provider_class) = actual_runtime_visible_root_provider_class(
                    cx.db(),
                    cx.body(),
                    cx.carriers(),
                    provider,
                )
                .or_else(|| {
                    runtime_class_for_provider_binding(
                        cx.db(),
                        provider,
                        cx.scope(),
                        cx.assumptions(),
                    )
                    .map(|class| (local_id, class))
                })
                .unwrap_or_else(|| {
                    panic!(
                        "missing direct-carrier runtime class for semantic local {idx}: {}",
                        local.ty.pretty_print(cx.db()),
                    )
                });
                Some((
                    provider.clone(),
                    provider_class,
                    carrier_local_place_class(
                        cx.db(),
                        local,
                        local_id,
                        cx.carriers(),
                        cx.scope(),
                        cx.assumptions(),
                    ),
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
    let lowerings = cx
        .body()
        .locals
        .iter()
        .enumerate()
        .map(|(idx, local)| match (&local.facts.interface, &local.facts.origin) {
            (NLocalInterface::Erased, _) => RuntimeLocalLowering::Erased,
            (NLocalInterface::DirectValue, NLocalOrigin::RootProvider(provider)) => {
                let provider = runtime_provider_binding_id(&provider_bindings, provider)
                    .unwrap_or_else(|| {
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
            (NLocalInterface::PlaceCarrier, _) => RuntimeLocalLowering::PlaceCarrier {
                place_class: carrier_local_place_class(
                    cx.db(),
                    local,
                    SLocalId::from_u32(idx as u32),
                    cx.carriers(),
                    cx.scope(),
                    cx.assumptions(),
                ),
            },
            (NLocalInterface::PlaceBoundValue, origin) => {
                let place_class = normalized_local_place_class(
                    cx.db(),
                    cx.body(),
                    SLocalId::from_u32(idx as u32),
                    cx.carriers(),
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
                    cx.db(),
                    local,
                    SLocalId::from_u32(idx as u32),
                    cx.carriers(),
                    cx.scope(),
                    cx.assumptions(),
                );
                let provider = origin.root_provider().map(|provider| {
                    let provider_class =
                        runtime_class_for_provider_binding(cx.db(), provider, cx.scope(), cx.assumptions())
                            .unwrap_or_else(|| {
                                panic!(
                                    "missing direct-carrier runtime class for semantic local {idx}: {}",
                                    local.ty.pretty_print(cx.db()),
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

#[salsa::tracked(return_ref)]
fn runtime_visible_root_provider_locals<'db>(
    db: &'db dyn MirDb,
    semantic: SemanticInstance<'db>,
) -> FxHashMap<ProviderBinding<'db>, SLocalId> {
    let mut locals = FxHashMap::default();
    for entry in runtime_visible_binding_plans(db, semantic) {
        let Some(provider) = root_provider_for_runtime_visible_binding(db, semantic, entry.binding)
        else {
            continue;
        };
        locals.entry(provider).or_insert(entry.local);
    }
    locals
}

fn actual_runtime_visible_root_provider_local<'db>(
    db: &'db dyn MirDb,
    semantic: SemanticInstance<'db>,
    provider: &ProviderBinding<'db>,
) -> Option<SLocalId> {
    runtime_visible_root_provider_locals(db, semantic)
        .get(provider)
        .copied()
}

pub(super) fn actual_runtime_visible_root_provider_class<'db>(
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
    let typed_body = body.owner.key(db).typed_body(db);
    normalized_local_place_class_in_context(
        db,
        body,
        local,
        carriers,
        typed_body.body().map(|body| body.scope()),
        typed_body.assumptions(),
    )
}

pub(super) fn normalized_local_place_class_in_context<'db>(
    db: &'db dyn MirDb,
    body: &NormalizedSemanticBody<'db>,
    local: SLocalId,
    carriers: &[RuntimeCarrier<'db>],
    scope: Option<hir::hir_def::scope_graph::ScopeId<'db>>,
    assumptions: PredicateListId<'db>,
) -> Option<RuntimeClass<'db>> {
    normalized_place_class_in_context(
        db,
        body,
        body.locals.get(local.index())?.backing_place()?,
        carriers,
        scope,
        assumptions,
    )
}

fn infer_runtime_local_root<'db>(
    cx: RuntimeBodyCx<'_, '_, 'db>,
    local: SLocalId,
    carrier: &mut RuntimeCarrier<'db>,
) -> RuntimeLocalRoot<'db> {
    let local_data = cx
        .body()
        .locals
        .get(local.index())
        .expect("normalized local exists");
    let place_class = local_place_root_class(cx, local, local_data, carrier);
    let transport_class = match carrier {
        RuntimeCarrier::Value(class) => Some(class.clone()),
        RuntimeCarrier::Erased => {
            fallback_root_transport_class(cx.db(), local_data, cx.scope(), cx.assumptions())
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

pub(super) fn local_place_root_class<'db>(
    cx: RuntimeBodyCx<'_, '_, 'db>,
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
                cx.db(),
                local_data.ty,
                cx.scope(),
                cx.assumptions(),
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
                cx.db(),
                *target_ty,
                cx.scope(),
                cx.assumptions(),
            ))
        }
        NLocalInterface::PlaceBoundValue => normalized_local_place_class_in_context(
            cx.db(),
            cx.body(),
            local,
            cx.carriers(),
            cx.scope(),
            cx.assumptions(),
        )
        .or_else(|| {
            let NormalizedBindingLowering::PlaceBoundValue { value_ty, .. } = &local_data.lowering
            else {
                panic!("place-bound local missing place-bound lowering: {local:?}");
            };
            Some(stored_class_for_ty_in_context(
                cx.db(),
                *value_ty,
                cx.scope(),
                cx.assumptions(),
            ))
        }),
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
                cx.db(),
                *target_ty,
                cx.scope(),
                cx.assumptions(),
            ))
        }
    }
}

pub(super) fn fallback_root_transport_class<'db>(
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
