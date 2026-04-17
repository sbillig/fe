use std::collections::VecDeque;

use cranelift_entity::EntityRef;
use hir::{
    analysis::{
        semantic::{
            SLocalId,
            borrowck::{
                NLocalInterface, NLocalOrigin, NSLocal, NormalizedBindingLowering,
                NormalizedSemanticBody,
            },
        },
        ty::trait_resolution::PredicateListId,
    },
    semantic::ProviderBinding,
};

use crate::{
    db::MirDb,
    runtime::{
        AddressSpaceKind, ArrayLayout, EnumLayoutKey, EnumVariantLayout, Layout, LayoutId,
        LayoutKey, RefKind, RefView, RuntimeCarrier, RuntimeClass, RuntimeLocalLowering,
        RuntimeLocalRoot, RuntimeProviderBinding, RuntimeProviderBindingId, StructLayout,
    },
};

use super::{
    classify::{
        BodyEnv, BodyStaticFacts, InferClassCache, RuntimeBodyCx,
        actual_aggregate_class_from_runtime_source, carrier_value_class,
        runtime_class_for_direct_value_provider_in_context,
        runtime_class_for_effect_binding_provider_in_context, runtime_class_for_provider_binding,
    },
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
    class_cache: InferClassCache<'db>,
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
        Self {
            env,
            carriers,
            class_cache: InferClassCache::new(env.body().locals.len()),
        }
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
                (NLocalInterface::DirectValue, NLocalOrigin::RootProvider(provider)) => self
                    .env
                    .actual_runtime_visible_root_provider_class(&self.carriers, provider)
                    .map(|(_, class)| class)
                    .or_else(|| {
                        runtime_class_for_direct_value_provider_in_context(
                            self.env.db(),
                            provider,
                            self.env.scope(),
                            self.env.assumptions(),
                        )
                    }),
                (NLocalInterface::DirectCarrier, NLocalOrigin::RootProvider(provider)) => self
                    .env
                    .actual_runtime_visible_root_provider_class(&self.carriers, provider)
                    .map(|(_, class)| class)
                    .or_else(|| {
                        runtime_class_for_provider_binding(
                            self.env.db(),
                            provider,
                            self.env.scope(),
                            self.env.assumptions(),
                        )
                    }),
                _ => None,
            };
            if let Some(class) = class {
                self.set_carrier(SLocalId::from_u32(idx as u32), RuntimeCarrier::Value(class));
            }
        }
    }

    fn infer_carriers(&mut self) {
        let mut queued = vec![true; self.env.assignment_count()];
        let mut worklist = VecDeque::from_iter(0..self.env.assignment_count());
        while let Some(assign_id) = worklist.pop_front() {
            queued[assign_id] = false;
            let assign = self
                .env
                .assignment(assign_id)
                .unwrap_or_else(|| panic!("missing assignment facts for statement {assign_id}"));
            let stmt = &self.env.body().blocks[assign.block_idx].stmts[assign.stmt_idx];
            let expr = match &stmt.kind {
                hir::analysis::semantic::NSStmtKind::Assign { expr, .. } => expr,
                hir::analysis::semantic::NSStmtKind::Store { .. } => {
                    panic!(
                        "assignment facts point to non-assignment statement: block={} stmt={}",
                        assign.block_idx, assign.stmt_idx
                    )
                }
            };
            let local = &self.env.body().locals[assign.dst.index()];
            let desired = match self.env.expr_direct_class_with_cache(
                &self.carriers,
                assign.block_idx,
                assign.stmt_idx,
                expr,
                local.ty,
                &mut self.class_cache,
            ) {
                Some(RuntimeClass::AggregateValue { layout })
                    if matches!(local.facts.interface, NLocalInterface::DirectValue)
                        && local.facts.root_demand.needs_projectable_owned_storage() =>
                {
                    RuntimeCarrier::Value(RuntimeClass::object_ref(layout))
                }
                Some(class) => RuntimeCarrier::Value(class),
                None => continue,
            };
            if self.set_carrier(assign.dst, desired) {
                self.propagate_local_change(assign.dst, &mut worklist, &mut queued);
            }
        }
    }

    fn set_carrier(&mut self, local: SLocalId, desired: RuntimeCarrier<'db>) -> bool {
        let current = self
            .carriers
            .get(local.index())
            .cloned()
            .unwrap_or(RuntimeCarrier::Erased);
        let desired = merge_runtime_carrier(self.env.db(), current, desired);
        if self.carriers[local.index()] == desired {
            return false;
        }
        self.carriers[local.index()] = desired;
        self.class_cache.note_carrier_changed(local);
        true
    }

    fn propagate_local_change(
        &mut self,
        changed_local: SLocalId,
        worklist: &mut VecDeque<usize>,
        queued: &mut [bool],
    ) {
        let mut pending = VecDeque::from([changed_local]);
        let mut seen = vec![false; self.env.body().locals.len()];
        while let Some(local) = pending.pop_front() {
            if std::mem::replace(&mut seen[local.index()], true) {
                continue;
            }
            self.class_cache.invalidate_local_dynamic_facts(local);
            for &assign_id in self.env.assignments_using_local(local) {
                if !queued[assign_id] {
                    queued[assign_id] = true;
                    worklist.push_back(assign_id);
                }
            }
            for dependent in self.env.dynamic_dependents(local).iter().copied() {
                pending.push_back(dependent);
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
                let (provider_local, provider_class) = cx
                    .env
                    .actual_runtime_visible_root_provider_class(cx.carriers(), provider)
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
                let (provider_local, provider_class) = cx
                    .env
                    .actual_runtime_visible_root_provider_class(cx.carriers(), provider)
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
                let (provider_local, provider_class) = cx
                    .env
                    .actual_runtime_visible_root_provider_class(cx.carriers(), provider)
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
    let typed_body = body.owner.key(db).typed_body(db);
    let type_env = super::type_info::RuntimeTypeEnv::new(scope, assumptions);
    let facts = BodyStaticFacts::new_in_context(db, body, typed_body, type_env);
    BodyEnv::from_parts(db, body, type_env, &facts)
        .normalized_place_class(carriers, body.locals.get(local.index())?.backing_place()?)
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
        RuntimeCarrier::Erased => cx.env.root_transport_fallback_class(local),
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
            if let Some(carrier_class) = carrier_value_class_for_runtime(carrier)
                && let Some(place_class) =
                    materialized_place_class_from_runtime_source(&carrier_class)
            {
                return Some(place_class);
            }
            cx.env.root_place_fallback_class(local)
        }
        NLocalInterface::PlaceCarrier => {
            if let Some(carrier_class) = carrier_value_class_for_runtime(carrier) {
                return actual_aggregate_class_from_runtime_source(&carrier_class)
                    .or(Some(carrier_class));
            }
            cx.env.root_place_fallback_class(local)
        }
        NLocalInterface::PlaceBoundValue => cx
            .env
            .normalized_place_class(
                cx.carriers(),
                cx.body().locals.get(local.index())?.backing_place()?,
            )
            .or_else(|| cx.env.root_place_fallback_class(local)),
        NLocalInterface::DirectCarrier => {
            if let Some(carrier_class) = carrier_value_class_for_runtime(carrier) {
                return actual_aggregate_class_from_runtime_source(&carrier_class)
                    .or(Some(carrier_class));
            }
            cx.env.root_place_fallback_class(local)
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

fn materialized_place_class_from_runtime_source<'db>(
    class: &RuntimeClass<'db>,
) -> Option<RuntimeClass<'db>> {
    match class {
        RuntimeClass::Scalar(_) | RuntimeClass::AggregateValue { .. } => Some(class.clone()),
        RuntimeClass::Ref { pointee, .. } => Some((**pointee).clone()),
        RuntimeClass::RawAddr {
            target: Some(layout),
            ..
        } => Some(RuntimeClass::AggregateValue { layout: *layout }),
        RuntimeClass::RawAddr { target: None, .. } => None,
    }
}

fn merge_runtime_carrier<'db>(
    db: &'db dyn MirDb,
    current: RuntimeCarrier<'db>,
    desired: RuntimeCarrier<'db>,
) -> RuntimeCarrier<'db> {
    match (current, desired) {
        (RuntimeCarrier::Erased, desired) | (desired, RuntimeCarrier::Erased) => desired,
        (RuntimeCarrier::Value(current), RuntimeCarrier::Value(desired)) => {
            RuntimeCarrier::Value(merge_runtime_class(db, &current, &desired).unwrap_or(desired))
        }
    }
}

pub(super) fn merge_runtime_class<'db>(
    db: &'db dyn MirDb,
    current: &RuntimeClass<'db>,
    desired: &RuntimeClass<'db>,
) -> Option<RuntimeClass<'db>> {
    if current == desired {
        return Some(current.clone());
    }
    match (current, desired) {
        (
            RuntimeClass::AggregateValue {
                layout: current_layout,
            },
            RuntimeClass::AggregateValue {
                layout: desired_layout,
            },
        ) => merge_layouts(db, *current_layout, *desired_layout)
            .map(|layout| RuntimeClass::AggregateValue { layout }),
        (
            RuntimeClass::Ref {
                pointee: current_pointee,
                kind: current_kind,
                view: current_view,
            },
            RuntimeClass::Ref {
                pointee: desired_pointee,
                kind: desired_kind,
                view: desired_view,
            },
        ) if current_view == desired_view => Some(RuntimeClass::Ref {
            pointee: Box::new(merge_runtime_class(db, current_pointee, desired_pointee)?),
            kind: merge_ref_kind(current_kind, desired_kind)?,
            view: current_view.clone(),
        }),
        (
            RuntimeClass::RawAddr {
                space: current_space,
                target: current_target,
            },
            RuntimeClass::RawAddr {
                space: desired_space,
                target: desired_target,
            },
        ) if current_target == desired_target => Some(RuntimeClass::RawAddr {
            space: preferred_address_space(*current_space, *desired_space),
            target: *current_target,
        }),
        (
            RuntimeClass::Ref {
                pointee,
                kind,
                view: RefView::Whole,
            },
            RuntimeClass::RawAddr { space, target },
        )
        | (
            RuntimeClass::RawAddr { space, target },
            RuntimeClass::Ref {
                pointee,
                kind,
                view: RefView::Whole,
            },
        ) if pointee.aggregate_layout() == *target => {
            let ref_space = ref_kind_address_space(kind);
            if ref_space == *space {
                Some(RuntimeClass::Ref {
                    pointee: pointee.clone(),
                    kind: kind.clone(),
                    view: RefView::Whole,
                })
            } else {
                Some(RuntimeClass::RawAddr {
                    space: preferred_address_space(ref_space, *space),
                    target: *target,
                })
            }
        }
        _ => None,
    }
}

fn merge_layouts<'db>(
    db: &'db dyn MirDb,
    current: LayoutId<'db>,
    desired: LayoutId<'db>,
) -> Option<LayoutId<'db>> {
    if current == desired {
        return Some(current);
    }
    match (current.data(db), desired.data(db)) {
        (Layout::Array(current), Layout::Array(desired))
            if current.source_ty == desired.source_ty && current.len == desired.len =>
        {
            Some(LayoutId::new(
                db,
                LayoutKey::Array(ArrayLayout {
                    source_ty: current.source_ty,
                    elem: merge_runtime_class(db, &current.elem, &desired.elem)?,
                    len: current.len,
                }),
            ))
        }
        (Layout::Struct(current), Layout::Struct(desired))
            if current.source_ty == desired.source_ty
                && current.fields.len() == desired.fields.len() =>
        {
            Some(LayoutId::new(
                db,
                LayoutKey::Struct(StructLayout {
                    source_ty: current.source_ty,
                    fields: current
                        .fields
                        .iter()
                        .zip(desired.fields.iter())
                        .map(|(current, desired)| merge_runtime_class(db, current, desired))
                        .collect::<Option<Vec<_>>>()?
                        .into(),
                }),
            ))
        }
        (Layout::Enum(current), Layout::Enum(desired))
            if current.source_ty == desired.source_ty
                && current.variants.len() == desired.variants.len() =>
        {
            Some(LayoutId::new(
                db,
                LayoutKey::Enum(EnumLayoutKey {
                    source_ty: current.source_ty,
                    variants: current
                        .variants
                        .iter()
                        .zip(desired.variants.iter())
                        .map(|(current, desired)| {
                            (current.fields.len() == desired.fields.len()).then_some(
                                EnumVariantLayout {
                                    name: current.name.clone(),
                                    fields: current
                                        .fields
                                        .iter()
                                        .zip(desired.fields.iter())
                                        .map(|(current, desired)| {
                                            merge_runtime_class(db, current, desired)
                                        })
                                        .collect::<Option<Vec<_>>>()?
                                        .into(),
                                },
                            )
                        })
                        .collect::<Option<Vec<_>>>()?
                        .into(),
                }),
            ))
        }
        (Layout::Struct(_) | Layout::Array(_) | Layout::Enum(_), _) => None,
    }
}

fn merge_ref_kind<'db>(current: &RefKind<'db>, desired: &RefKind<'db>) -> Option<RefKind<'db>> {
    match (current, desired) {
        (RefKind::Object, RefKind::Object) => Some(RefKind::Object),
        (RefKind::Const, RefKind::Const) => Some(RefKind::Const),
        (
            RefKind::Provider {
                provider_ty: current_provider_ty,
                space: current_space,
            },
            RefKind::Provider {
                provider_ty: desired_provider_ty,
                space: desired_space,
            },
        ) => {
            let space = preferred_address_space(*current_space, *desired_space);
            let provider_ty = if current_provider_ty == desired_provider_ty {
                *current_provider_ty
            } else if *current_space == AddressSpaceKind::Memory
                && *desired_space != AddressSpaceKind::Memory
            {
                *desired_provider_ty
            } else if *desired_space == AddressSpaceKind::Memory
                && *current_space != AddressSpaceKind::Memory
            {
                *current_provider_ty
            } else {
                return None;
            };
            Some(RefKind::Provider { provider_ty, space })
        }
        _ => None,
    }
}

fn preferred_address_space(
    current: AddressSpaceKind,
    desired: AddressSpaceKind,
) -> AddressSpaceKind {
    match (current, desired) {
        (AddressSpaceKind::Memory, desired) => desired,
        (current, AddressSpaceKind::Memory) => current,
        (current, _) => current,
    }
}

fn ref_kind_address_space(kind: &RefKind<'_>) -> AddressSpaceKind {
    match kind {
        RefKind::Provider { space, .. } => *space,
        RefKind::Object | RefKind::Const => AddressSpaceKind::Memory,
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

#[cfg(test)]
mod tests {
    use driver::DriverDataBase;
    use hir::analysis::ty::ty_def::TyId;

    use super::*;
    use crate::runtime::{ScalarClass, ScalarRepr, ScalarRole};

    fn test_enum_layout<'db>(
        db: &'db dyn MirDb,
        source_ty: TyId<'db>,
        payload: RuntimeClass<'db>,
    ) -> LayoutId<'db> {
        LayoutId::new(
            db,
            LayoutKey::Enum(EnumLayoutKey {
                source_ty,
                variants: vec![
                    EnumVariantLayout {
                        name: "Some".to_string(),
                        fields: vec![payload].into(),
                    },
                    EnumVariantLayout {
                        name: "None".to_string(),
                        fields: vec![].into(),
                    },
                ]
                .into(),
            }),
        )
    }

    #[test]
    fn merge_runtime_class_prefers_non_memory_provider_enum_layouts() {
        let db = DriverDataBase::default();
        let source_ty = TyId::unit(&db);
        let pointee = RuntimeClass::Scalar(ScalarClass {
            repr: ScalarRepr::Int {
                bits: 256,
                signed: false,
            },
            role: ScalarRole::Plain,
        });
        let storage_class = RuntimeClass::AggregateValue {
            layout: test_enum_layout(
                &db,
                source_ty,
                RuntimeClass::Ref {
                    pointee: Box::new(pointee.clone()),
                    kind: RefKind::Provider {
                        provider_ty: TyId::bool(&db),
                        space: AddressSpaceKind::Storage,
                    },
                    view: RefView::Whole,
                },
            ),
        };
        let memory_class = RuntimeClass::AggregateValue {
            layout: test_enum_layout(
                &db,
                source_ty,
                RuntimeClass::Ref {
                    pointee: Box::new(pointee),
                    kind: RefKind::Provider {
                        provider_ty: TyId::u256(&db),
                        space: AddressSpaceKind::Memory,
                    },
                    view: RefView::Whole,
                },
            ),
        };

        assert_eq!(
            merge_runtime_class(&db, &storage_class, &memory_class),
            Some(storage_class.clone())
        );
        assert_eq!(
            merge_runtime_class(&db, &memory_class, &storage_class),
            Some(storage_class)
        );
    }
}
