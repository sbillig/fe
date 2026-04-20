use hir::analysis::ty::ty_def::TyId;

use crate::{
    db::MirDb,
    runtime::{
        AddressSpaceKind, LayoutId, RefKind, RefView, RuntimeClass, ScalarClass, ScalarRepr,
        ScalarRole, runtime_classes_share_runtime_rep,
    },
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct RuntimeConversionPlan<'db> {
    pub(crate) target: RuntimeClass<'db>,
    pub(crate) steps: Box<[RuntimeConversionStep<'db>]>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum RuntimeConversionStep<'db> {
    UseAs {
        class: RuntimeClass<'db>,
    },
    RetagRef {
        class: RuntimeClass<'db>,
    },
    LoadRef {
        class: RuntimeClass<'db>,
    },
    AddrOfRef {
        class: RuntimeClass<'db>,
    },
    LoadRawAddr {
        class: RuntimeClass<'db>,
        space: AddressSpaceKind,
        layout: LayoutId<'db>,
    },
    MaterializeToObject {
        class: RuntimeClass<'db>,
    },
    AllocObjectCopy {
        class: RuntimeClass<'db>,
        layout: LayoutId<'db>,
    },
    ProviderFromRaw {
        class: RuntimeClass<'db>,
        provider_ty: TyId<'db>,
        space: AddressSpaceKind,
        target: Option<LayoutId<'db>>,
    },
    ProviderToRaw {
        class: RuntimeClass<'db>,
    },
    WordToRawAddr {
        class: RuntimeClass<'db>,
        space: AddressSpaceKind,
        target: Option<LayoutId<'db>>,
    },
    RawAddrToWord {
        class: RuntimeClass<'db>,
        scalar: ScalarClass<'db>,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum RuntimeConversionError<'db> {
    Unsupported {
        source: RuntimeClass<'db>,
        target: RuntimeClass<'db>,
    },
    Cycle {
        source: RuntimeClass<'db>,
        target: RuntimeClass<'db>,
    },
}

pub(crate) struct RuntimeConversionPlanner<'db> {
    db: &'db dyn MirDb,
    stack: Vec<(RuntimeClass<'db>, RuntimeClass<'db>)>,
}

impl<'db> RuntimeConversionPlanner<'db> {
    pub(crate) fn plan(
        db: &'db dyn MirDb,
        source: RuntimeClass<'db>,
        target: RuntimeClass<'db>,
    ) -> Result<RuntimeConversionPlan<'db>, RuntimeConversionError<'db>> {
        let mut planner = Self {
            db,
            stack: Vec::new(),
        };
        let mut steps = Vec::new();
        planner.convert(source, target.clone(), &mut steps)?;
        Ok(RuntimeConversionPlan {
            target,
            steps: steps.into_boxed_slice(),
        })
    }

    fn convert(
        &mut self,
        source: RuntimeClass<'db>,
        target: RuntimeClass<'db>,
        steps: &mut Vec<RuntimeConversionStep<'db>>,
    ) -> Result<(), RuntimeConversionError<'db>> {
        if source == target {
            return Ok(());
        }
        if self.stack.iter().any(|(active_source, active_target)| {
            active_source == &source && active_target == &target
        }) {
            return Err(RuntimeConversionError::Cycle { source, target });
        }

        self.stack.push((source.clone(), target.clone()));
        let result = self.convert_inner(source, target, steps);
        self.stack.pop();
        result
    }

    fn convert_inner(
        &mut self,
        source: RuntimeClass<'db>,
        target: RuntimeClass<'db>,
        steps: &mut Vec<RuntimeConversionStep<'db>>,
    ) -> Result<(), RuntimeConversionError<'db>> {
        match (&source, &target) {
            (RuntimeClass::AggregateValue { .. }, RuntimeClass::AggregateValue { .. })
                if runtime_classes_share_runtime_rep(self.db, &source, &target) =>
            {
                steps.push(RuntimeConversionStep::UseAs { class: target });
                Ok(())
            }
            (
                RuntimeClass::Ref {
                    view: actual_view, ..
                },
                RuntimeClass::Ref {
                    view: desired_view, ..
                },
            ) if actual_view == desired_view
                && runtime_classes_share_runtime_rep(self.db, &source, &target) =>
            {
                steps.push(RuntimeConversionStep::RetagRef { class: target });
                Ok(())
            }
            (RuntimeClass::Ref { pointee, .. }, _) if !target.is_transport() => {
                let loaded = pointee.as_ref().clone();
                steps.push(RuntimeConversionStep::LoadRef {
                    class: loaded.clone(),
                });
                self.convert(loaded, target, steps)
            }
            (
                RuntimeClass::RawAddr {
                    space,
                    target: Some(layout),
                },
                RuntimeClass::Ref {
                    pointee,
                    kind:
                        RefKind::Provider {
                            provider_ty,
                            space: provider_space,
                        },
                    view: RefView::Whole,
                },
            ) if space == provider_space
                && pointee.as_ref() == &(RuntimeClass::AggregateValue { layout: *layout }) =>
            {
                steps.push(RuntimeConversionStep::ProviderFromRaw {
                    class: target.clone(),
                    provider_ty: *provider_ty,
                    space: *space,
                    target: Some(*layout),
                });
                Ok(())
            }
            (
                RuntimeClass::Ref {
                    pointee,
                    kind: RefKind::Const,
                    view: RefView::Whole,
                },
                RuntimeClass::Ref {
                    pointee: target_pointee,
                    kind: RefKind::Object,
                    view: RefView::Whole,
                },
            ) if pointee == target_pointee => {
                let layout = target_pointee
                    .aggregate_layout()
                    .expect("aggregate ref layout");
                steps.push(RuntimeConversionStep::MaterializeToObject {
                    class: RuntimeClass::object_ref(layout),
                });
                Ok(())
            }
            (
                RuntimeClass::Ref {
                    pointee,
                    kind: RefKind::Object | RefKind::Const,
                    view: RefView::Whole,
                },
                RuntimeClass::Ref {
                    pointee: target_pointee,
                    kind: RefKind::Provider { .. },
                    view: RefView::Whole,
                },
            ) if pointee == target_pointee => {
                steps.push(RuntimeConversionStep::AddrOfRef { class: target });
                Ok(())
            }
            (
                RuntimeClass::Ref {
                    pointee,
                    kind: RefKind::Object | RefKind::Const,
                    view: RefView::Whole,
                },
                RuntimeClass::RawAddr {
                    space,
                    target: target_layout,
                },
            ) if target_layout
                .is_none_or(|target_layout| Some(target_layout) == pointee.aggregate_layout()) =>
            {
                let layout = pointee.aggregate_layout().expect("aggregate ref layout");
                steps.push(RuntimeConversionStep::AddrOfRef {
                    class: RuntimeClass::RawAddr {
                        space: *space,
                        target: Some(layout),
                    },
                });
                Ok(())
            }
            (
                RuntimeClass::RawAddr {
                    space,
                    target: Some(layout),
                },
                RuntimeClass::AggregateValue {
                    layout: target_layout,
                },
            ) if layout == target_layout => {
                steps.push(RuntimeConversionStep::LoadRawAddr {
                    class: RuntimeClass::AggregateValue {
                        layout: *target_layout,
                    },
                    space: *space,
                    layout: *layout,
                });
                Ok(())
            }
            (
                RuntimeClass::AggregateValue { layout },
                RuntimeClass::Ref {
                    pointee,
                    kind: RefKind::Object,
                    view: RefView::Whole,
                },
            ) if pointee.as_ref() == &(RuntimeClass::AggregateValue { layout: *layout }) => {
                steps.push(RuntimeConversionStep::MaterializeToObject {
                    class: RuntimeClass::object_ref(*layout),
                });
                Ok(())
            }
            (
                RuntimeClass::RawAddr { space, .. },
                RuntimeClass::Ref {
                    pointee,
                    kind:
                        RefKind::Provider {
                            provider_ty,
                            space: provider_space,
                        },
                    view: RefView::Whole,
                },
            ) if space == provider_space => {
                let target_layout = pointee.aggregate_layout();
                steps.push(RuntimeConversionStep::ProviderFromRaw {
                    class: target.clone(),
                    provider_ty: *provider_ty,
                    space: *space,
                    target: target_layout,
                });
                Ok(())
            }
            (
                RuntimeClass::Ref {
                    kind:
                        RefKind::Provider {
                            space: source_space,
                            ..
                        },
                    ..
                },
                RuntimeClass::Ref {
                    pointee,
                    kind:
                        RefKind::Provider {
                            provider_ty,
                            space: target_space,
                        },
                    view: RefView::Whole,
                },
            ) if source_space == target_space && pointee.aggregate_layout().is_some() => {
                let target_layout = pointee.aggregate_layout();
                let raw = RuntimeClass::RawAddr {
                    space: *target_space,
                    target: target_layout,
                };
                self.convert(source, raw, steps)?;
                self.convert(
                    RuntimeClass::RawAddr {
                        space: *target_space,
                        target: target_layout,
                    },
                    RuntimeClass::Ref {
                        pointee: pointee.clone(),
                        kind: RefKind::Provider {
                            provider_ty: *provider_ty,
                            space: *target_space,
                        },
                        view: RefView::Whole,
                    },
                    steps,
                )
            }
            (
                RuntimeClass::AggregateValue { layout },
                RuntimeClass::Ref {
                    pointee,
                    kind: RefKind::Provider { provider_ty, space },
                    view: RefView::Whole,
                },
            ) if runtime_classes_share_runtime_rep(
                self.db,
                &(RuntimeClass::AggregateValue { layout: *layout }),
                pointee,
            ) =>
            {
                let target_layout = pointee
                    .aggregate_layout()
                    .expect("aggregate provider ref layout");
                self.convert(source, pointee.as_ref().clone(), steps)?;
                self.convert(
                    pointee.as_ref().clone(),
                    RuntimeClass::object_ref(target_layout),
                    steps,
                )?;
                self.convert(
                    RuntimeClass::object_ref(target_layout),
                    RuntimeClass::provider_ref(target_layout, *provider_ty, *space),
                    steps,
                )
            }
            (
                RuntimeClass::Scalar(scalar),
                RuntimeClass::Ref {
                    pointee,
                    kind: RefKind::Provider { space, .. },
                    view: RefView::Whole,
                },
            ) if is_plain_word_scalar(scalar) => {
                let target_layout = pointee.aggregate_layout();
                let raw = RuntimeClass::RawAddr {
                    space: *space,
                    target: target_layout,
                };
                steps.push(RuntimeConversionStep::WordToRawAddr {
                    class: raw.clone(),
                    space: *space,
                    target: target_layout,
                });
                self.convert(raw, target, steps)
            }
            (
                RuntimeClass::Ref {
                    kind: RefKind::Provider { .. },
                    ..
                },
                RuntimeClass::RawAddr { .. },
            ) => {
                steps.push(RuntimeConversionStep::ProviderToRaw { class: target });
                Ok(())
            }
            (RuntimeClass::RawAddr { .. }, RuntimeClass::Scalar(scalar))
                if matches!(
                    scalar.repr,
                    ScalarRepr::Int {
                        bits: 256,
                        signed: false
                    }
                ) =>
            {
                steps.push(RuntimeConversionStep::RawAddrToWord {
                    class: target.clone(),
                    scalar: scalar.clone(),
                });
                Ok(())
            }
            (
                RuntimeClass::Scalar(scalar),
                RuntimeClass::RawAddr {
                    space,
                    target: target_layout,
                },
            ) if is_plain_word_scalar(scalar) => {
                steps.push(RuntimeConversionStep::WordToRawAddr {
                    class: target.clone(),
                    space: *space,
                    target: *target_layout,
                });
                Ok(())
            }
            (
                _,
                RuntimeClass::Ref {
                    pointee,
                    kind: RefKind::Object,
                    view: RefView::Whole,
                },
            ) if pointee.aggregate_layout().is_some() => {
                let layout = pointee
                    .aggregate_layout()
                    .expect("aggregate object ref layout");
                self.convert(source, pointee.as_ref().clone(), steps)?;
                steps.push(RuntimeConversionStep::AllocObjectCopy {
                    class: RuntimeClass::object_ref(layout),
                    layout,
                });
                Ok(())
            }
            _ => Err(RuntimeConversionError::Unsupported { source, target }),
        }
    }
}

fn is_plain_word_scalar(scalar: &ScalarClass<'_>) -> bool {
    matches!(
        scalar,
        ScalarClass {
            repr: ScalarRepr::Int {
                bits: 256,
                signed: false
            },
            role: ScalarRole::Plain
        }
    )
}

#[cfg(test)]
mod tests {
    use driver::DriverDataBase;
    use hir::analysis::ty::ty_def::TyId;

    use super::*;

    fn word_class<'db>() -> RuntimeClass<'db> {
        RuntimeClass::Scalar(ScalarClass {
            repr: ScalarRepr::Int {
                bits: 256,
                signed: false,
            },
            role: ScalarRole::Plain,
        })
    }

    #[test]
    fn identity_conversion_has_no_steps() {
        let db = DriverDataBase::default();
        let source = RuntimeClass::RawAddr {
            space: AddressSpaceKind::Memory,
            target: None,
        };

        let plan = RuntimeConversionPlanner::plan(&db, source.clone(), source).unwrap();

        assert!(plan.steps.is_empty());
    }

    #[test]
    fn word_and_raw_address_conversions_are_explicit_steps() {
        let db = DriverDataBase::default();
        let raw = RuntimeClass::RawAddr {
            space: AddressSpaceKind::Storage,
            target: None,
        };

        let to_raw = RuntimeConversionPlanner::plan(&db, word_class(), raw.clone()).unwrap();
        assert_eq!(
            to_raw.steps.as_ref(),
            &[RuntimeConversionStep::WordToRawAddr {
                class: raw.clone(),
                space: AddressSpaceKind::Storage,
                target: None,
            }]
        );

        let to_word = RuntimeConversionPlanner::plan(&db, raw, word_class()).unwrap();
        assert_eq!(
            to_word.steps.as_ref(),
            &[RuntimeConversionStep::RawAddrToWord {
                class: word_class(),
                scalar: match word_class() {
                    RuntimeClass::Scalar(scalar) => scalar,
                    _ => unreachable!(),
                },
            }]
        );
    }

    #[test]
    fn raw_address_to_provider_requires_matching_space() {
        let db = DriverDataBase::default();
        let provider_ty = TyId::unit(&db);
        let pointee = Box::new(word_class());
        let storage_provider = RuntimeClass::Ref {
            pointee: pointee.clone(),
            kind: RefKind::Provider {
                provider_ty,
                space: AddressSpaceKind::Storage,
            },
            view: RefView::Whole,
        };
        let storage_raw = RuntimeClass::RawAddr {
            space: AddressSpaceKind::Storage,
            target: None,
        };

        let plan =
            RuntimeConversionPlanner::plan(&db, storage_raw.clone(), storage_provider.clone())
                .unwrap();
        assert_eq!(
            plan.steps.as_ref(),
            &[RuntimeConversionStep::ProviderFromRaw {
                class: storage_provider,
                provider_ty,
                space: AddressSpaceKind::Storage,
                target: None,
            }]
        );

        let memory_provider = RuntimeClass::Ref {
            pointee,
            kind: RefKind::Provider {
                provider_ty,
                space: AddressSpaceKind::Memory,
            },
            view: RefView::Whole,
        };
        assert!(matches!(
            RuntimeConversionPlanner::plan(&db, storage_raw, memory_provider),
            Err(RuntimeConversionError::Unsupported { .. })
        ));
    }
}
