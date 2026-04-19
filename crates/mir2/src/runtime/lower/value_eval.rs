use cranelift_entity::EntityRef;
use hir::analysis::{
    semantic::{
        NEffectArg, NEffectArgValue, NOperand, NSPlace, NSPlaceRoot, SLocalId, SemanticInstance,
    },
    ty::ty_check::EffectPassMode,
};

use crate::{
    db::MirDb,
    runtime::{AddressSpaceKind, RuntimeBoundarySpec, RuntimeCarrier, RuntimeClass},
};

use super::{
    classify::{
        BodyEnv, BoundaryRef, BoundarySiteAllocator, InferClassCache, StagedBoundary,
        carrier_value_class, desired_runtime_effect_arg_boundary, provider_root_space,
        ref_class_for_place_result, runtime_class_for_direct_value_provider_in_context,
        runtime_effect_binding_plan_for_binding_idx, runtime_visible_place_arg_class_for_boundary,
        specialize_boundary_for_runtime_source_in_context,
    },
    coerce::CoercionPlanner,
    place::resolved_effect_arg_address_space,
    type_info::provider_class_for_target_in_env,
};

#[derive(Clone, Debug)]
pub(super) enum CompiledMaterializationPlan<'db> {
    Erased,
    SemanticValue,
    AggregateFromSource,
    AggregateFromSourceOrFallback { fallback: RuntimeClass<'db> },
}

#[derive(Clone, Debug)]
pub(super) enum CompiledValuePassPlan<'db> {
    Erased,
    VisibleValue,
    ActualValue,
    ExactValue { exact: RuntimeClass<'db> },
    ExactAggregate { exact: RuntimeClass<'db> },
    ExactRefLike { boundary: StagedBoundary<'db> },
    BorrowLike { boundary: StagedBoundary<'db> },
}

#[derive(Clone, Debug)]
pub(super) enum CompiledEffectArgPlan<'db> {
    ErasedPlainValue,
    ByValueValue { plan: CompiledValuePassPlan<'db> },
    ByValueValueFallback { fallback: RuntimeClass<'db> },
    ByValuePlaceBoundary { boundary: StagedBoundary<'db> },
    ByValuePlaceFallback { fallback: RuntimeClass<'db> },
    ByPlaceValue { boundary: StagedBoundary<'db> },
    ByPlacePlace { boundary: StagedBoundary<'db> },
}

#[derive(Clone, Debug)]
pub(super) struct CompiledCallInputPlan<'db> {
    pub(super) param_plans: Box<[CompiledValuePassPlan<'db>]>,
    pub(super) effect_plans: Box<[CompiledEffectArgPlan<'db>]>,
}

pub(super) struct RuntimeValueEvaluator<'a, 'carriers, 'cache, 'db> {
    env: BodyEnv<'a, 'db>,
    carriers: &'carriers [RuntimeCarrier<'db>],
    class_cache: Option<&'cache mut InferClassCache<'db>>,
}

impl<'a, 'carriers, 'cache, 'db> RuntimeValueEvaluator<'a, 'carriers, 'cache, 'db> {
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

    pub(super) fn evaluate_value_pass_plan(
        &mut self,
        local: SLocalId,
        plan: &CompiledValuePassPlan<'db>,
    ) -> Option<RuntimeClass<'db>> {
        match plan {
            CompiledValuePassPlan::Erased => None,
            CompiledValuePassPlan::VisibleValue => self.materialize(local),
            CompiledValuePassPlan::ActualValue => self.actual_value(local),
            CompiledValuePassPlan::ExactValue { exact } => Some(exact.clone()),
            CompiledValuePassPlan::ExactAggregate { exact } => self
                .env
                .actual_aggregate_class_for_source(self.carriers, local)
                .or_else(|| Some(exact.clone())),
            CompiledValuePassPlan::ExactRefLike { boundary } => {
                self.exact_ref_like_value(local, boundary)
            }
            CompiledValuePassPlan::BorrowLike { boundary } => {
                self.boundary_compatible_value(local, boundary)
            }
        }
    }

    pub(super) fn call_inputs(
        &mut self,
        args: &[NOperand],
        effect_args: &[NEffectArg<'db>],
        plan: &CompiledCallInputPlan<'db>,
    ) -> Vec<RuntimeClass<'db>> {
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
        let mut classes = self.param_inputs(args, &plan.param_plans);
        for (arg, effect_plan) in effect_args.iter().zip(plan.effect_plans.iter()) {
            if let Some(class) = self.effect_arg_class(arg, effect_plan) {
                classes.push(class);
            }
        }
        classes
    }

    pub(super) fn param_inputs(
        &mut self,
        args: &[NOperand],
        plans: &[CompiledValuePassPlan<'db>],
    ) -> Vec<RuntimeClass<'db>> {
        assert_eq!(
            args.len(),
            plans.len(),
            "runtime call arg count mismatch during param evaluation: caller={:?} args={args:?} plans={plans:?}",
            self.env.body().owner.key(self.env.db()),
        );
        let mut classes = Vec::new();
        for (arg, plan) in args.iter().zip(plans.iter()) {
            if let Some(class) = self.evaluate_value_pass_plan(arg.local, plan) {
                classes.push(class);
            }
        }
        classes
    }

    pub(super) fn boundary_compatible_value(
        &mut self,
        local: SLocalId,
        boundary: &StagedBoundary<'db>,
    ) -> Option<RuntimeClass<'db>> {
        let boundary = self.specialized_boundary(local, boundary);
        let local_data = self.env.body().locals.get(local.index())?;
        if let Some(root) = local_data.lowering.root() {
            let class = self.env.normalized_place_address_class(
                self.carriers,
                &NSPlace {
                    root: NSPlaceRoot::Root(root),
                    path: Default::default(),
                },
            )?;
            if CoercionPlanner::class_satisfies_boundary(&class, &boundary) {
                return Some(class);
            }
        }
        if let Some(place) = local_data.backing_place() {
            let class = self
                .env
                .normalized_place_address_class(self.carriers, place)?;
            if CoercionPlanner::class_satisfies_boundary(&class, &boundary) {
                return Some(class);
            }
        }
        if let Some(place) = super::classify::snapshot_source_place(self.env.body(), local) {
            let class = self
                .env
                .normalized_place_address_class(self.carriers, place)?;
            if CoercionPlanner::class_satisfies_boundary(&class, &boundary) {
                return Some(class);
            }
        }

        let value_class = self.env.semantic_value_class(self.carriers, local)?;
        if let Some(provider) = local_data.facts.origin.root_provider() {
            let root_class = self
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
                })?;
            let class = ref_class_for_place_result(
                &root_class,
                &value_class,
                provider_root_space(provider, &root_class),
                false,
            );
            if CoercionPlanner::class_satisfies_boundary(&class, &boundary) {
                return Some(class);
            }
        }
        let cx = self.env.with_carriers(self.carriers);
        let root_class = super::infer::local_place_root_class(
            cx,
            local,
            local_data,
            self.carriers.get(local.index())?,
        )?;
        let class =
            ref_class_for_place_result(&root_class, &value_class, AddressSpaceKind::Memory, false);
        if CoercionPlanner::class_satisfies_boundary(&class, &boundary) {
            return Some(class);
        }

        carrier_value_class(local, self.carriers)
            .filter(|class| CoercionPlanner::class_satisfies_boundary(class, &boundary))
    }

    fn exact_ref_like_value(
        &mut self,
        local: SLocalId,
        boundary: &StagedBoundary<'db>,
    ) -> Option<RuntimeClass<'db>> {
        let boundary = self.specialized_boundary(local, boundary);
        match &boundary {
            RuntimeBoundarySpec::Exact(target)
                if matches!(
                    target,
                    RuntimeClass::Ref { .. } | RuntimeClass::RawAddr { .. }
                ) =>
            {
                carrier_value_class(local, self.carriers)
                    .filter(|class| CoercionPlanner::class_satisfies_boundary(class, &boundary))
                    .or_else(|| Some(target.clone()))
            }
            RuntimeBoundarySpec::Exact(class) => Some(class.clone()),
            RuntimeBoundarySpec::BorrowLike { .. } => unreachable!(),
        }
    }

    fn effect_arg_class(
        &mut self,
        arg: &NEffectArg<'db>,
        plan: &CompiledEffectArgPlan<'db>,
    ) -> Option<RuntimeClass<'db>> {
        match plan {
            CompiledEffectArgPlan::ErasedPlainValue => match &arg.arg {
                NEffectArgValue::Value(value) => self.materialize(value.local),
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
                self.evaluate_value_pass_plan(value.local, plan)
            }
            CompiledEffectArgPlan::ByValueValueFallback { fallback } => {
                let NEffectArgValue::Value(value) = &arg.arg else {
                    panic!("compiled value effect arg fallback reached place arg: {arg:?}");
                };
                self.actual_value(value.local)
                    .or_else(|| Some(fallback.clone()))
            }
            CompiledEffectArgPlan::ByValuePlaceBoundary { boundary } => {
                let NEffectArgValue::Place(place) = &arg.arg else {
                    panic!("compiled place effect arg plan reached value arg: {arg:?}");
                };
                runtime_visible_place_arg_class_for_boundary(
                    self.env.db(),
                    self.env.body(),
                    place,
                    &boundary.boundary,
                    self.carriers,
                    self.env.scope(),
                    self.env.assumptions(),
                )
            }
            CompiledEffectArgPlan::ByValuePlaceFallback { fallback } => {
                let NEffectArgValue::Place(_) = &arg.arg else {
                    panic!("compiled place effect arg fallback reached value arg: {arg:?}");
                };
                Some(fallback.clone())
            }
            CompiledEffectArgPlan::ByPlaceValue { boundary } => {
                let NEffectArgValue::Value(value) = &arg.arg else {
                    panic!("compiled by-place effect arg plan reached place arg: {arg:?}");
                };
                self.boundary_compatible_value(value.local, boundary)
            }
            CompiledEffectArgPlan::ByPlacePlace { boundary } => {
                let NEffectArgValue::Place(place) = &arg.arg else {
                    panic!("compiled by-place effect arg plan reached value arg: {arg:?}");
                };
                runtime_visible_place_arg_class_for_boundary(
                    self.env.db(),
                    self.env.body(),
                    place,
                    &boundary.boundary,
                    self.carriers,
                    self.env.scope(),
                    self.env.assumptions(),
                )
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

pub(super) fn compile_value_pass_plan<'db>(
    plan: crate::runtime::RuntimeParamPlan<'db>,
    boundary_sites: &mut BoundarySiteAllocator,
) -> CompiledValuePassPlan<'db> {
    match plan {
        crate::runtime::RuntimeParamPlan::Erased => CompiledValuePassPlan::Erased,
        crate::runtime::RuntimeParamPlan::PassActual => CompiledValuePassPlan::VisibleValue,
        crate::runtime::RuntimeParamPlan::Boundary(
            boundary @ RuntimeBoundarySpec::Exact(RuntimeClass::AggregateValue { .. }),
        ) => {
            let RuntimeBoundarySpec::Exact(exact) = boundary else {
                unreachable!();
            };
            CompiledValuePassPlan::ExactAggregate { exact }
        }
        crate::runtime::RuntimeParamPlan::Boundary(
            boundary @ RuntimeBoundarySpec::Exact(
                RuntimeClass::Ref { .. } | RuntimeClass::RawAddr { .. },
            ),
        ) => CompiledValuePassPlan::ExactRefLike {
            boundary: boundary_sites.stage(boundary),
        },
        crate::runtime::RuntimeParamPlan::Boundary(RuntimeBoundarySpec::Exact(exact)) => {
            CompiledValuePassPlan::ExactValue { exact }
        }
        crate::runtime::RuntimeParamPlan::Boundary(
            boundary @ RuntimeBoundarySpec::BorrowLike { .. },
        ) => CompiledValuePassPlan::BorrowLike {
            boundary: boundary_sites.stage(boundary),
        },
    }
}

pub(super) fn compile_call_input_plan_for_semantic<'db>(
    db: &'db dyn MirDb,
    body: &hir::analysis::semantic::borrowck::NormalizedSemanticBody<'db>,
    semantic: SemanticInstance<'db>,
    type_env: super::type_info::RuntimeTypeEnv<'db>,
    effect_args: &[NEffectArg<'db>],
    boundary_sites: &mut BoundarySiteAllocator,
) -> CompiledCallInputPlan<'db> {
    let param_plans = super::interface::runtime_param_plans(db, semantic)
        .iter()
        .cloned()
        .map(|plan| compile_value_pass_plan(plan, boundary_sites))
        .collect::<Vec<_>>()
        .into_boxed_slice();
    let effect_plans = effect_args
        .iter()
        .map(|arg| compile_effect_arg_plan(db, body, semantic, type_env, arg, boundary_sites))
        .collect::<Vec<_>>()
        .into_boxed_slice();
    CompiledCallInputPlan {
        param_plans,
        effect_plans,
    }
}

fn compile_effect_arg_plan<'db>(
    db: &'db dyn MirDb,
    body: &hir::analysis::semantic::borrowck::NormalizedSemanticBody<'db>,
    semantic: SemanticInstance<'db>,
    type_env: super::type_info::RuntimeTypeEnv<'db>,
    arg: &NEffectArg<'db>,
    boundary_sites: &mut BoundarySiteAllocator,
) -> CompiledEffectArgPlan<'db> {
    let space = resolved_effect_arg_address_space(db, body, arg);
    let boundary = desired_runtime_effect_arg_boundary(
        db,
        type_env,
        arg,
        runtime_effect_binding_plan_for_binding_idx(db, semantic, arg.binding_idx).as_ref(),
        space,
    );
    if boundary.is_none() && arg.provider.is_none() && arg.target_ty.is_none() {
        return match (&arg.pass_mode, &arg.arg) {
            (EffectPassMode::ByValue | EffectPassMode::Unknown, NEffectArgValue::Value(_)) => {
                CompiledEffectArgPlan::ErasedPlainValue
            }
            (EffectPassMode::ByValue | EffectPassMode::Unknown, NEffectArgValue::Place(_))
            | (
                EffectPassMode::ByPlace | EffectPassMode::ByTempPlace,
                NEffectArgValue::Value(_) | NEffectArgValue::Place(_),
            ) => panic!(
                "effect arg without provider/target should compile as a plain value: owner={:?}; arg={arg:?}",
                body.owner.key(db).owner(db),
            ),
        };
    }
    match (&arg.pass_mode, &arg.arg) {
        (EffectPassMode::ByValue | EffectPassMode::Unknown, NEffectArgValue::Value(_)) => {
            let plan = boundary.map_or(CompiledValuePassPlan::ActualValue, |boundary| {
                compile_value_pass_plan(
                    crate::runtime::RuntimeParamPlan::Boundary(boundary),
                    boundary_sites,
                )
            });
            if matches!(plan, CompiledValuePassPlan::ActualValue)
                && (arg.provider.is_some() || arg.target_ty.is_some())
            {
                CompiledEffectArgPlan::ByValueValueFallback {
                    fallback: provider_class_for_target_in_env(db, type_env, arg.target_ty, space),
                }
            } else {
                CompiledEffectArgPlan::ByValueValue { plan }
            }
        }
        (EffectPassMode::ByValue | EffectPassMode::Unknown, NEffectArgValue::Place(_)) => boundary
            .map_or_else(
                || CompiledEffectArgPlan::ByValuePlaceFallback {
                    fallback: provider_class_for_target_in_env(db, type_env, arg.target_ty, space),
                },
                |boundary| CompiledEffectArgPlan::ByValuePlaceBoundary {
                    boundary: boundary_sites.stage(boundary),
                },
            ),
        (EffectPassMode::ByPlace | EffectPassMode::ByTempPlace, NEffectArgValue::Value(_)) => {
            let boundary = boundary.unwrap_or_else(|| {
                let Some(target_ty) = arg.target_ty else {
                    return RuntimeBoundarySpec::Exact(provider_class_for_target_in_env(
                        db, type_env, None, space,
                    ));
                };
                RuntimeBoundarySpec::BorrowLike {
                    pointee: super::type_info::stored_class_for_ty_in_context(
                        db,
                        target_ty,
                        type_env.scope,
                        type_env.assumptions,
                    ),
                    access: crate::runtime::BorrowAccess::ReadWrite,
                    allow: super::type_info::default_borrow_transport_set(
                        crate::runtime::BorrowAccess::ReadWrite,
                        space,
                    ),
                }
            });
            CompiledEffectArgPlan::ByPlaceValue {
                boundary: boundary_sites.stage(boundary),
            }
        }
        (EffectPassMode::ByPlace | EffectPassMode::ByTempPlace, NEffectArgValue::Place(_)) => {
            let boundary = boundary.unwrap_or_else(|| {
                let Some(target_ty) = arg.target_ty else {
                    return RuntimeBoundarySpec::Exact(provider_class_for_target_in_env(
                        db, type_env, None, space,
                    ));
                };
                RuntimeBoundarySpec::BorrowLike {
                    pointee: super::type_info::stored_class_for_ty_in_context(
                        db,
                        target_ty,
                        type_env.scope,
                        type_env.assumptions,
                    ),
                    access: crate::runtime::BorrowAccess::ReadWrite,
                    allow: super::type_info::default_borrow_transport_set(
                        crate::runtime::BorrowAccess::ReadWrite,
                        space,
                    ),
                }
            });
            CompiledEffectArgPlan::ByPlacePlace {
                boundary: boundary_sites.stage(boundary),
            }
        }
    }
}
