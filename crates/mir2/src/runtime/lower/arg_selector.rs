use cranelift_entity::EntityRef;
use hir::analysis::{
    semantic::{
        NBorrowRoot, NEffectArg, NEffectArgValue, NLocalInterface, NOperand, NSPlace, NSPlaceRoot,
        ReadMode, SLocalId,
    },
    ty::ty_def::TyId,
};

use crate::runtime::{AddressSpaceKind, RuntimeBoundarySpec, RuntimeCarrier, RuntimeClass};

use super::{
    call_input::{
        CompiledCallInputPlan, CompiledEffectArgPlan, CompiledMaterializationPlan,
        CompiledValuePassPlan,
    },
    classify::{
        BodyEnv, BoundaryRef, InferClassCache, StagedBoundary, carrier_value_class,
        provider_root_space, ref_class_for_place_result,
        runtime_class_for_direct_value_provider_in_context,
        runtime_class_for_effect_binding_provider_in_context, snapshot_source_place,
        specialize_boundary_for_runtime_source_in_context,
    },
    realize::{
        RuntimeArgRealization, RuntimeBoundaryMatcher, RuntimeBoundaryMaterialization,
        SelectedRuntimeArg,
    },
};

pub(super) struct RuntimeArgSelector<'a, 'carriers, 'cache, 'db> {
    env: BodyEnv<'a, 'db>,
    carriers: &'carriers [RuntimeCarrier<'db>],
    class_cache: Option<&'cache mut InferClassCache<'db>>,
}

impl<'a, 'carriers, 'cache, 'db> RuntimeArgSelector<'a, 'carriers, 'cache, 'db> {
    pub(super) fn new(
        env: BodyEnv<'a, 'db>,
        carriers: &'carriers [RuntimeCarrier<'db>],
        class_cache: Option<&'cache mut InferClassCache<'db>>,
    ) -> Self {
        Self {
            env,
            carriers,
            class_cache,
        }
    }

    pub(super) fn actual_value(&mut self, local: SLocalId) -> Option<RuntimeClass<'db>> {
        carrier_value_class(local, self.carriers)
            .or_else(|| self.env.semantic_value_class(self.carriers, local))
    }

    pub(super) fn materialize(&mut self, local: SLocalId) -> Option<RuntimeClass<'db>> {
        match self.env.materialization_plan(local)? {
            CompiledMaterializationPlan::Erased => None,
            CompiledMaterializationPlan::SemanticValue => {
                self.env.semantic_value_class(self.carriers, local)
            }
            CompiledMaterializationPlan::AggregateFromSource => self
                .env
                .actual_aggregate_class_for_source(self.carriers, local),
            CompiledMaterializationPlan::AggregateFromSourceOrFallback { fallback } => self
                .env
                .actual_aggregate_class_for_source(self.carriers, local)
                .or_else(|| Some(fallback.clone())),
        }
    }

    pub(super) fn selected_call_inputs(
        &mut self,
        args: &[NOperand],
        effect_args: &[NEffectArg<'db>],
        plan: &CompiledCallInputPlan<'db>,
    ) -> Vec<SelectedRuntimeArg<'db>> {
        assert_eq!(
            args.len(),
            plan.param_plans.len(),
            "runtime call arg count mismatch during value evaluation: caller={:?} args={args:?} plans={:?}",
            self.env.body().owner.key(self.env.db()),
            plan.param_plans,
        );
        assert_eq!(
            effect_args.len(),
            plan.effect_plans.len(),
            "runtime effect arg count mismatch during value evaluation: caller={:?} effect_args={effect_args:?} plans={:?}",
            self.env.body().owner.key(self.env.db()),
            plan.effect_plans,
        );
        let mut selected = self.selected_param_inputs(args, &plan.param_plans);
        for (arg, effect_plan) in effect_args.iter().zip(plan.effect_plans.iter()) {
            if let Some(arg) = self.select_effect_arg(arg, effect_plan) {
                selected.push(arg);
            }
        }
        selected
    }

    pub(super) fn selected_param_inputs(
        &mut self,
        args: &[NOperand],
        plans: &[CompiledValuePassPlan<'db>],
    ) -> Vec<SelectedRuntimeArg<'db>> {
        assert_eq!(
            args.len(),
            plans.len(),
            "runtime call arg count mismatch during param evaluation: caller={:?} args={args:?} plans={plans:?}",
            self.env.body().owner.key(self.env.db()),
        );
        let mut selected = Vec::new();
        for (arg, plan) in args.iter().zip(plans.iter()) {
            if let Some(arg) = self.selected_value_pass_plan(*arg, plan) {
                selected.push(arg);
            }
        }
        selected
    }

    pub(super) fn selected_semantic_operand_for_boundary(
        &mut self,
        arg: NOperand,
        boundary: &RuntimeBoundarySpec<'db>,
    ) -> SelectedRuntimeArg<'db> {
        let local = arg.local;
        let semantic_ty = self.env.body().locals[local.index()].ty;
        let boundary = self
            .env
            .specialize_boundary_for_source(self.carriers, local, boundary);
        match &boundary {
            RuntimeBoundarySpec::ExactTransport(target) => SelectedRuntimeArg {
                class: target.clone(),
                realization: RuntimeArgRealization::LowerSemanticOperand(arg),
            },
            RuntimeBoundarySpec::ExactShape(target) => self
                .select_runtime_boundary_compatible_value(local, &boundary)
                .unwrap_or_else(|| SelectedRuntimeArg {
                    class: target.clone(),
                    realization: RuntimeArgRealization::LowerSemanticOperand(arg),
                }),
            RuntimeBoundarySpec::BorrowLike { .. } => self
                .select_runtime_boundary_compatible_value(local, &boundary)
                .or_else(|| {
                    RuntimeBoundaryMaterialization::for_boundary(&boundary).map(
                        |materialization| SelectedRuntimeArg {
                            class: materialization.class(),
                            realization: RuntimeArgRealization::MaterializeSemanticValue {
                                operand: arg,
                                materialization,
                                semantic_ty,
                            },
                        },
                    )
                })
                .unwrap_or_else(|| {
                    panic!(
                        "semantic operand boundary has no runtime realization: owner={:?}; arg={arg:?}; boundary={boundary:?}",
                        self.env.body().owner.key(self.env.db()).owner(self.env.db()),
                    )
                }),
        }
    }

    pub(super) fn selected_value_pass_plan(
        &mut self,
        arg: NOperand,
        plan: &CompiledValuePassPlan<'db>,
    ) -> Option<SelectedRuntimeArg<'db>> {
        let local = arg.local;
        match plan {
            CompiledValuePassPlan::Erased => None,
            CompiledValuePassPlan::VisibleValue => self.select_materialized_value(local, arg),
            CompiledValuePassPlan::ActualValue => self.select_actual_value(local, arg),
            CompiledValuePassPlan::ExactTransport { exact } => Some(SelectedRuntimeArg {
                class: exact.clone(),
                realization: RuntimeArgRealization::LowerSemanticOperand(arg),
            }),
            CompiledValuePassPlan::ExactShapeAggregate { exact } => {
                self.select_actual_aggregate_value(local).or_else(|| {
                    Some(SelectedRuntimeArg {
                        class: exact.clone(),
                        realization: RuntimeArgRealization::LowerSemanticOperand(arg),
                    })
                })
            }
            CompiledValuePassPlan::ExactShapeRefLike { boundary } => self
                .select_exact_shape_ref_like_value(local, boundary)
                .or_else(|| {
                    let RuntimeBoundarySpec::ExactShape(class) =
                        self.specialized_boundary(local, boundary)
                    else {
                        unreachable!();
                    };
                    Some(SelectedRuntimeArg {
                        class,
                        realization: RuntimeArgRealization::Placeholder {
                            semantic_ty: self.env.body().locals[local.index()].ty,
                        },
                    })
                }),
            CompiledValuePassPlan::BorrowLike { boundary } => {
                self.select_boundary_compatible_value(local, boundary)
            }
        }
    }

    pub(super) fn selected_value_class_for_local(
        &mut self,
        local: SLocalId,
        plan: &CompiledValuePassPlan<'db>,
    ) -> Option<RuntimeClass<'db>> {
        self.selected_value_pass_plan(
            NOperand {
                local,
                mode: ReadMode::Copy,
            },
            plan,
        )
        .map(|arg| arg.class)
    }

    fn select_materialized_value(
        &mut self,
        local: SLocalId,
        arg: NOperand,
    ) -> Option<SelectedRuntimeArg<'db>> {
        match self.env.materialization_plan(local)? {
            CompiledMaterializationPlan::Erased => None,
            CompiledMaterializationPlan::SemanticValue => self
                .env
                .semantic_value_class(self.carriers, local)
                .map(|class| SelectedRuntimeArg {
                    class,
                    realization: RuntimeArgRealization::LowerSemanticOperand(arg),
                }),
            CompiledMaterializationPlan::AggregateFromSource => {
                self.select_actual_aggregate_value(local)
            }
            CompiledMaterializationPlan::AggregateFromSourceOrFallback { fallback } => {
                self.select_actual_aggregate_value(local).or_else(|| {
                    Some(SelectedRuntimeArg {
                        class: fallback.clone(),
                        realization: RuntimeArgRealization::LowerSemanticOperand(arg),
                    })
                })
            }
        }
    }

    fn select_actual_value(
        &mut self,
        local: SLocalId,
        arg: NOperand,
    ) -> Option<SelectedRuntimeArg<'db>> {
        carrier_value_class(local, self.carriers)
            .map(|class| {
                let realization = if class.is_transport() {
                    RuntimeArgRealization::UseHandleLikeValue { local }
                } else {
                    RuntimeArgRealization::UseRuntimeValue { local }
                };
                SelectedRuntimeArg { class, realization }
            })
            .or_else(|| {
                self.env
                    .semantic_value_class(self.carriers, local)
                    .map(|class| SelectedRuntimeArg {
                        class,
                        realization: RuntimeArgRealization::LowerSemanticOperand(arg),
                    })
            })
    }

    fn select_actual_aggregate_value(
        &mut self,
        local: SLocalId,
    ) -> Option<SelectedRuntimeArg<'db>> {
        self.env
            .actual_aggregate_class_for_source(self.carriers, local)
            .map(|class| SelectedRuntimeArg {
                class,
                realization: RuntimeArgRealization::AggregateFromRuntimeSource { local },
            })
    }

    fn select_boundary_compatible_value(
        &mut self,
        local: SLocalId,
        boundary: &StagedBoundary<'db>,
    ) -> Option<SelectedRuntimeArg<'db>> {
        let boundary = self.specialized_boundary(local, boundary);
        self.select_runtime_boundary_compatible_value(local, &boundary)
    }

    fn select_runtime_boundary_compatible_value(
        &mut self,
        local: SLocalId,
        boundary: &RuntimeBoundarySpec<'db>,
    ) -> Option<SelectedRuntimeArg<'db>> {
        let local_data = self.env.body().locals.get(local.index())?;
        let semantic_ty = local_data.ty;
        if let Some(class) = carrier_value_class(local, self.carriers)
            && RuntimeBoundaryMatcher::class_satisfies_boundary(&class, boundary)
        {
            let realization = if class.is_transport() {
                RuntimeArgRealization::UseHandleLikeValue { local }
            } else {
                RuntimeArgRealization::UseRuntimeValue { local }
            };
            return Some(SelectedRuntimeArg { class, realization });
        }
        if let Some(value_class) = self.env.semantic_value_class(self.carriers, local) {
            if let Some(provider) = local_data.facts.origin.root_provider()
                && let Some(root_class) = self
                    .env
                    .actual_runtime_visible_root_provider_class(self.carriers, provider)
                    .map(|(_, class)| class)
                    .or_else(|| {
                        runtime_class_for_direct_value_provider_in_context(
                            self.env.db(),
                            provider,
                            self.env.scope(),
                            self.env.assumptions(),
                        )
                    })
            {
                let class = ref_class_for_place_result(
                    &root_class,
                    &value_class,
                    provider_root_space(provider, &root_class),
                    false,
                );
                if RuntimeBoundaryMatcher::class_satisfies_boundary(&class, boundary) {
                    return Some(SelectedRuntimeArg {
                        class,
                        realization: RuntimeArgRealization::UseHandleLikeValue { local },
                    });
                }
            }
            let cx = self.env.with_carriers(self.carriers);
            if let Some(root_class) = super::infer::local_place_root_class(
                cx,
                local,
                local_data,
                self.carriers.get(local.index())?,
            ) {
                let class = ref_class_for_place_result(
                    &root_class,
                    &value_class,
                    AddressSpaceKind::Memory,
                    false,
                );
                if RuntimeBoundaryMatcher::class_satisfies_boundary(&class, boundary) {
                    return Some(SelectedRuntimeArg {
                        class,
                        realization: RuntimeArgRealization::UseHandleLikeValue { local },
                    });
                }
            }
        }

        if let Some(place) = local_data.backing_place()
            && let Some(arg) =
                self.select_place_address_if_satisfies(place.clone(), semantic_ty, boundary)
        {
            return Some(arg);
        }
        if let Some(place) = snapshot_source_place(self.env.body(), local)
            && let Some(arg) =
                self.select_place_address_if_satisfies(place.clone(), semantic_ty, boundary)
        {
            return Some(arg);
        }
        None
    }

    fn select_place_address_if_satisfies(
        &self,
        place: NSPlace<'db>,
        semantic_ty: TyId<'db>,
        boundary: &RuntimeBoundarySpec<'db>,
    ) -> Option<SelectedRuntimeArg<'db>> {
        if !self.place_is_lowerable(&place) {
            return None;
        }
        let class = self
            .env
            .normalized_place_address_class(self.carriers, &place)?;
        RuntimeBoundaryMatcher::class_satisfies_boundary(&class, boundary).then(|| {
            SelectedRuntimeArg {
                class,
                realization: RuntimeArgRealization::AddrOfPlace { place, semantic_ty },
            }
        })
    }

    fn select_exact_shape_ref_like_value(
        &mut self,
        local: SLocalId,
        boundary: &StagedBoundary<'db>,
    ) -> Option<SelectedRuntimeArg<'db>> {
        let boundary = self.specialized_boundary(local, boundary);
        match &boundary {
            RuntimeBoundarySpec::ExactShape(
                RuntimeClass::Ref { .. } | RuntimeClass::RawAddr { .. },
            ) => self.select_runtime_boundary_compatible_value(local, &boundary),
            RuntimeBoundarySpec::ExactShape(_) | RuntimeBoundarySpec::ExactTransport(_) => None,
            RuntimeBoundarySpec::BorrowLike { .. } => unreachable!(),
        }
    }

    fn select_place_for_boundary(
        &self,
        place: NSPlace<'db>,
        semantic_ty: TyId<'db>,
        boundary: RuntimeBoundarySpec<'db>,
    ) -> Option<SelectedRuntimeArg<'db>> {
        if !self.place_is_lowerable(&place) {
            return self.placeholder_arg_for_unlowerable_place(place, semantic_ty, &boundary);
        }
        if let Some(class) = self
            .env
            .normalized_place_address_class(self.carriers, &place)
            && RuntimeBoundaryMatcher::class_satisfies_boundary(&class, &boundary)
        {
            return Some(SelectedRuntimeArg {
                class,
                realization: RuntimeArgRealization::AddrOfPlace { place, semantic_ty },
            });
        }
        let (class, realization) = match &boundary {
            RuntimeBoundarySpec::ExactTransport(target) => {
                if target.is_transport() {
                    (
                        target.clone(),
                        RuntimeArgRealization::AddrOfPlace { place, semantic_ty },
                    )
                } else {
                    (
                        target.clone(),
                        RuntimeArgRealization::LoadPlaceValue { place, semantic_ty },
                    )
                }
            }
            RuntimeBoundarySpec::ExactShape(target) if !target.is_transport() => (
                target.clone(),
                RuntimeArgRealization::LoadPlaceValue { place, semantic_ty },
            ),
            RuntimeBoundarySpec::BorrowLike { .. } => {
                let Some(materialization) = RuntimeBoundaryMaterialization::for_boundary(&boundary)
                else {
                    return self.placeholder_arg_for_unlowerable_place(
                        place,
                        semantic_ty,
                        &boundary,
                    );
                };
                (
                    materialization.class(),
                    RuntimeArgRealization::MaterializePlaceValue {
                        place,
                        materialization,
                        semantic_ty,
                    },
                )
            }
            RuntimeBoundarySpec::ExactShape(_) => {
                return self.placeholder_arg_for_unlowerable_place(place, semantic_ty, &boundary);
            }
        };
        Some(SelectedRuntimeArg { class, realization })
    }

    fn placeholder_arg_for_unlowerable_place(
        &self,
        place: NSPlace<'db>,
        semantic_ty: TyId<'db>,
        boundary: &RuntimeBoundarySpec<'db>,
    ) -> Option<SelectedRuntimeArg<'db>> {
        if !place.path.is_empty() {
            return None;
        }
        RuntimeBoundaryMatcher::placeholder_class(boundary).map(|class| SelectedRuntimeArg {
            class,
            realization: RuntimeArgRealization::Placeholder { semantic_ty },
        })
    }

    fn place_is_lowerable(&self, place: &NSPlace<'db>) -> bool {
        let mut visiting = vec![false; self.env.body().locals.len()];
        self.place_is_lowerable_with_seen(place, &mut visiting)
    }

    fn place_is_lowerable_with_seen(&self, place: &NSPlace<'db>, visiting: &mut [bool]) -> bool {
        match place.root {
            NSPlaceRoot::CarrierDerefLocal(local) => {
                carrier_value_class(local, self.carriers).is_some_and(|class| class.is_transport())
                    || self.semantic_place_root_is_lowerable(local, visiting)
            }
            NSPlaceRoot::Root(root) => match self.env.body().root(root) {
                Some(NBorrowRoot::Param { local, .. } | NBorrowRoot::LocalSlot { local }) => {
                    self.semantic_place_root_is_lowerable(*local, visiting)
                }
                Some(NBorrowRoot::Provider { binding }) => {
                    self.provider_place_root_is_lowerable(binding)
                }
                None => false,
            },
        }
    }

    fn semantic_place_root_is_lowerable(&self, local: SLocalId, visiting: &mut [bool]) -> bool {
        let Some(local_data) = self.env.body().locals.get(local.index()) else {
            return false;
        };
        if std::mem::replace(&mut visiting[local.index()], true) {
            return false;
        }
        let lowerable = snapshot_source_place(self.env.body(), local)
            .is_some_and(|place| self.place_is_lowerable_with_seen(place, visiting))
            || local_data
                .backing_place()
                .is_some_and(|place| self.place_is_lowerable_with_seen(place, visiting))
            || (matches!(local_data.facts.interface, NLocalInterface::PlaceCarrier)
                && carrier_value_class(local, self.carriers)
                    .is_some_and(|class| class.is_transport()))
            || local_data
                .facts
                .origin
                .root_provider()
                .is_some_and(|provider| self.provider_place_root_is_lowerable(provider))
            || local_data.facts.root_demand.needs_runtime_root();
        visiting[local.index()] = false;
        lowerable
    }

    fn provider_place_root_is_lowerable(
        &self,
        provider: &hir::semantic::ProviderBinding<'db>,
    ) -> bool {
        self.env
            .actual_runtime_visible_root_provider_class(self.carriers, provider)
            .is_some()
            || runtime_class_for_effect_binding_provider_in_context(
                self.env.db(),
                provider,
                self.env.scope(),
                self.env.assumptions(),
            )
            .is_some()
            || runtime_class_for_direct_value_provider_in_context(
                self.env.db(),
                provider,
                self.env.scope(),
                self.env.assumptions(),
            )
            .is_some()
    }

    fn select_effect_arg(
        &mut self,
        arg: &NEffectArg<'db>,
        plan: &CompiledEffectArgPlan<'db>,
    ) -> Option<SelectedRuntimeArg<'db>> {
        match plan {
            CompiledEffectArgPlan::ErasedPlainValue => match &arg.arg {
                NEffectArgValue::Value(value) => {
                    self.select_materialized_value(value.local, *value)
                }
                NEffectArgValue::Place(_) => panic!(
                    "effect arg without provider/target should evaluate as a plain value: owner={:?}; arg={arg:?}",
                    self.env
                        .body()
                        .owner
                        .key(self.env.db())
                        .owner(self.env.db()),
                ),
            },
            CompiledEffectArgPlan::ByValueValue { plan } => {
                let NEffectArgValue::Value(value) = &arg.arg else {
                    panic!("compiled value effect arg plan reached place arg: {arg:?}");
                };
                self.selected_value_pass_plan(*value, plan)
            }
            CompiledEffectArgPlan::ByValueValueFallback { fallback } => {
                let NEffectArgValue::Value(value) = &arg.arg else {
                    panic!("compiled value effect arg fallback reached place arg: {arg:?}");
                };
                self.select_actual_value(value.local, *value).or_else(|| {
                    Some(SelectedRuntimeArg {
                        class: fallback.clone(),
                        realization: RuntimeArgRealization::Placeholder {
                            semantic_ty: self.env.body().locals[value.local.index()].ty,
                        },
                    })
                })
            }
            CompiledEffectArgPlan::ByValuePlaceBoundary { boundary } => {
                let NEffectArgValue::Place(place) = &arg.arg else {
                    panic!("compiled place effect arg plan reached value arg: {arg:?}");
                };
                Some(self.select_place_for_boundary(
                    place.clone(),
                    arg.target_ty.unwrap_or_else(|| TyId::unit(self.env.db())),
                    boundary.boundary.clone(),
                )
                .unwrap_or_else(|| {
                    panic!(
                        "compiled place effect arg boundary has no runtime realization: owner={:?}; arg={arg:?}; boundary={:?}",
                        self.env.body().owner.key(self.env.db()).owner(self.env.db()),
                        boundary.boundary,
                    )
                }))
            }
            CompiledEffectArgPlan::ByValuePlaceFallback { fallback } => {
                let NEffectArgValue::Place(_) = &arg.arg else {
                    panic!("compiled place effect arg fallback reached value arg: {arg:?}");
                };
                let NEffectArgValue::Place(place) = &arg.arg else {
                    unreachable!();
                };
                Some(self.select_place_for_boundary(
                    place.clone(),
                    arg.target_ty.unwrap_or_else(|| TyId::unit(self.env.db())),
                    RuntimeBoundarySpec::ExactTransport(fallback.clone()),
                )
                .unwrap_or_else(|| {
                    panic!(
                        "compiled place effect arg fallback has no runtime realization: owner={:?}; arg={arg:?}; fallback={fallback:?}",
                        self.env.body().owner.key(self.env.db()).owner(self.env.db()),
                    )
                }))
            }
            CompiledEffectArgPlan::ByPlaceValue { boundary } => {
                let NEffectArgValue::Value(value) = &arg.arg else {
                    panic!("compiled by-place effect arg plan reached place arg: {arg:?}");
                };
                self.select_boundary_compatible_value(value.local, boundary)
            }
            CompiledEffectArgPlan::ByPlacePlace { boundary } => {
                let NEffectArgValue::Place(place) = &arg.arg else {
                    panic!("compiled by-place effect arg plan reached value arg: {arg:?}");
                };
                Some(self.select_place_for_boundary(
                    place.clone(),
                    arg.target_ty.unwrap_or_else(|| TyId::unit(self.env.db())),
                    boundary.boundary.clone(),
                )
                .unwrap_or_else(|| {
                    panic!(
                        "compiled by-place effect arg boundary has no runtime realization: owner={:?}; arg={arg:?}; boundary={:?}",
                        self.env.body().owner.key(self.env.db()).owner(self.env.db()),
                        boundary.boundary,
                    )
                }))
            }
        }
    }

    fn specialized_boundary(
        &mut self,
        local: SLocalId,
        boundary: &StagedBoundary<'db>,
    ) -> RuntimeBoundarySpec<'db> {
        specialize_boundary_for_runtime_source_in_context(
            self.env,
            local,
            BoundaryRef::staged(boundary),
            self.carriers,
            self.class_cache.as_deref_mut(),
        )
        .boundary
        .into_owned()
    }
}
