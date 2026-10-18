use hir::analysis::{
    semantic::{
        SemanticInstance,
        normalized::{NEffectArg, NEffectArgValue},
    },
    ty::ty_check::EffectPassMode,
};

use crate::{
    db::MirDb,
    runtime::{RuntimeBoundarySpec, RuntimeClass, RuntimeParamPlan},
};

use super::{boundary::StagedBoundary, classify::runtime_effect_binding_plan_for_binding_idx};

#[derive(Clone, Debug)]
pub(super) enum CompiledMaterializationPlan<'db> {
    Erased,
    SemanticValue,
    AggregateFromSource,
    AggregateFromSourceOrFallback(RuntimeClass<'db>),
}

#[derive(Clone, Debug)]
pub(super) enum CompiledValuePassPlan<'db> {
    Erased,
    VisibleValue,
    ExactTransport(RuntimeClass<'db>),
    ExactShapeAggregate(RuntimeClass<'db>),
    ExactShapeRefLike(StagedBoundary<'db>),
    ReadOnlyView {
        value: RuntimeClass<'db>,
        borrow: StagedBoundary<'db>,
    },
    BorrowLike(StagedBoundary<'db>),
}

#[derive(Clone, Debug)]
pub(super) enum CompiledEffectArgPlan<'db> {
    Erased,
    Value(CompiledValuePassPlan<'db>),
    Place(StagedBoundary<'db>),
}

#[derive(Clone, Debug)]
pub(super) struct CompiledCallInputPlan<'db> {
    pub(super) param_plans: Box<[CompiledValuePassPlan<'db>]>,
    pub(super) effect_plans: Box<[CompiledEffectArgPlan<'db>]>,
}

pub(super) fn compile_value_pass_plan<'db>(
    plan: RuntimeParamPlan<'db>,
) -> CompiledValuePassPlan<'db> {
    match plan {
        RuntimeParamPlan::Erased => CompiledValuePassPlan::Erased,
        RuntimeParamPlan::PassActual => CompiledValuePassPlan::VisibleValue,
        RuntimeParamPlan::ReadOnlyView { value, borrow } => CompiledValuePassPlan::ReadOnlyView {
            value,
            borrow: StagedBoundary::new(borrow),
        },
        RuntimeParamPlan::Boundary(RuntimeBoundarySpec::ExactTransport(exact)) => {
            CompiledValuePassPlan::ExactTransport(exact)
        }
        RuntimeParamPlan::Boundary(RuntimeBoundarySpec::ExactShape(
            exact @ RuntimeClass::AggregateValue { .. },
        )) => CompiledValuePassPlan::ExactShapeAggregate(exact),
        RuntimeParamPlan::Boundary(
            boundary @ RuntimeBoundarySpec::ExactShape(
                RuntimeClass::Ref { .. } | RuntimeClass::RawAddr { .. },
            ),
        ) => CompiledValuePassPlan::ExactShapeRefLike(StagedBoundary::new(boundary)),
        RuntimeParamPlan::Boundary(RuntimeBoundarySpec::ExactShape(exact)) => {
            CompiledValuePassPlan::ExactTransport(exact)
        }
        RuntimeParamPlan::Boundary(boundary @ RuntimeBoundarySpec::BorrowLike { .. }) => {
            CompiledValuePassPlan::BorrowLike(StagedBoundary::new(boundary))
        }
    }
}

pub(super) fn compile_call_input_plan_for_semantic<'db>(
    db: &'db dyn MirDb,
    semantic: SemanticInstance<'db>,
    effect_args: &[NEffectArg<'db>],
) -> CompiledCallInputPlan<'db> {
    let param_plans = super::interface::runtime_param_plans(db, semantic)
        .iter()
        .cloned()
        .map(compile_value_pass_plan)
        .collect::<Vec<_>>()
        .into_boxed_slice();
    let effect_plans = effect_args
        .iter()
        .map(|arg| compile_effect_arg_plan(db, semantic, arg))
        .collect::<Vec<_>>()
        .into_boxed_slice();
    CompiledCallInputPlan {
        param_plans,
        effect_plans,
    }
}

fn compile_effect_arg_plan<'db>(
    db: &'db dyn MirDb,
    semantic: SemanticInstance<'db>,
    arg: &NEffectArg<'db>,
) -> CompiledEffectArgPlan<'db> {
    let Some(binding_plan) =
        runtime_effect_binding_plan_for_binding_idx(db, semantic, arg.binding_idx)
    else {
        return CompiledEffectArgPlan::Erased;
    };
    match (&arg.pass_mode, &arg.arg) {
        (EffectPassMode::ByValue | EffectPassMode::Unknown, NEffectArgValue::Value(_)) => {
            CompiledEffectArgPlan::Value(compile_value_pass_plan(RuntimeParamPlan::Boundary(
                binding_plan.boundary,
            )))
        }
        (EffectPassMode::ByPlace | EffectPassMode::ByTempPlace, NEffectArgValue::Value(_)) => {
            panic!("borrowed effect argument must already have a stable provider place: {arg:?}")
        }
        (_, NEffectArgValue::Place(_)) => {
            CompiledEffectArgPlan::Place(StagedBoundary::new(binding_plan.boundary))
        }
    }
}
