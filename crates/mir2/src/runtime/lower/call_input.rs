use hir::analysis::{
    semantic::{NEffectArg, NEffectArgValue, SemanticInstance},
    ty::ty_check::EffectPassMode,
};

use crate::{
    db::MirDb,
    runtime::{RuntimeBoundarySpec, RuntimeClass},
};

use super::{
    classify::{
        BoundarySiteAllocator, StagedBoundary, desired_runtime_effect_arg_boundary,
        runtime_effect_binding_plan_for_binding_idx,
    },
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
    ExactTransport { exact: RuntimeClass<'db> },
    ExactShapeAggregate { exact: RuntimeClass<'db> },
    ExactShapeRefLike { boundary: StagedBoundary<'db> },
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

pub(super) fn compile_value_pass_plan<'db>(
    plan: crate::runtime::RuntimeParamPlan<'db>,
    boundary_sites: &mut BoundarySiteAllocator,
) -> CompiledValuePassPlan<'db> {
    match plan {
        crate::runtime::RuntimeParamPlan::Erased => CompiledValuePassPlan::Erased,
        crate::runtime::RuntimeParamPlan::PassActual => CompiledValuePassPlan::VisibleValue,
        crate::runtime::RuntimeParamPlan::Boundary(RuntimeBoundarySpec::ExactTransport(exact)) => {
            CompiledValuePassPlan::ExactTransport { exact }
        }
        crate::runtime::RuntimeParamPlan::Boundary(
            boundary @ RuntimeBoundarySpec::ExactShape(RuntimeClass::AggregateValue { .. }),
        ) => {
            let RuntimeBoundarySpec::ExactShape(exact) = boundary else {
                unreachable!();
            };
            CompiledValuePassPlan::ExactShapeAggregate { exact }
        }
        crate::runtime::RuntimeParamPlan::Boundary(
            boundary @ RuntimeBoundarySpec::ExactShape(
                RuntimeClass::Ref { .. } | RuntimeClass::RawAddr { .. },
            ),
        ) => CompiledValuePassPlan::ExactShapeRefLike {
            boundary: boundary_sites.stage(boundary),
        },
        crate::runtime::RuntimeParamPlan::Boundary(RuntimeBoundarySpec::ExactShape(exact)) => {
            CompiledValuePassPlan::ExactTransport { exact }
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
                    return RuntimeBoundarySpec::ExactShape(provider_class_for_target_in_env(
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
                    return RuntimeBoundarySpec::ExactShape(provider_class_for_target_in_env(
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
