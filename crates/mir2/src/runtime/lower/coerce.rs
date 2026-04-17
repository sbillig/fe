use crate::runtime::{RefKind, RefView, RuntimeBoundarySpec, RuntimeClass, RuntimeParamPlan};

#[derive(Clone, Debug, Default)]
pub(crate) struct BoundarySourceClass<'db> {
    pub(crate) value: Option<RuntimeClass<'db>>,
    pub(crate) addr: Option<RuntimeClass<'db>>,
}

pub(crate) struct CoercionPlanner;

impl CoercionPlanner {
    pub(crate) fn target_prefers_transport(class: &RuntimeClass<'_>) -> bool {
        matches!(
            class,
            RuntimeClass::Ref { .. } | RuntimeClass::RawAddr { .. }
        )
    }

    pub(crate) fn preserve_provider_space<'db>(
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

    pub(crate) fn class_satisfies_boundary<'db>(
        class: &RuntimeClass<'db>,
        boundary: &RuntimeBoundarySpec<'db>,
    ) -> bool {
        match boundary {
            RuntimeBoundarySpec::Exact(expected) => {
                Self::preserve_provider_space(class, expected) == *class
            }
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

    pub(crate) fn placeholder_class<'db>(
        boundary: &RuntimeBoundarySpec<'db>,
    ) -> Option<RuntimeClass<'db>> {
        match boundary {
            RuntimeBoundarySpec::Exact(class) => Some(class.clone()),
            RuntimeBoundarySpec::BorrowLike { pointee, allow, .. }
                if pointee.aggregate_layout().is_some() && allow.allow_object =>
            {
                Some(RuntimeClass::Ref {
                    pointee: Box::new(pointee.clone()),
                    kind: RefKind::Object,
                    view: RefView::Whole,
                })
            }
            RuntimeBoundarySpec::BorrowLike { pointee, allow, .. }
                if pointee.aggregate_layout().is_some() && allow.allow_const =>
            {
                Some(RuntimeClass::Ref {
                    pointee: Box::new(pointee.clone()),
                    kind: RefKind::Const,
                    view: RefView::Whole,
                })
            }
            RuntimeBoundarySpec::BorrowLike { pointee, allow, .. } if allow.allow_raw_addr => {
                Some(RuntimeClass::RawAddr {
                    space: crate::runtime::AddressSpaceKind::Memory,
                    target: pointee.aggregate_layout(),
                })
            }
            RuntimeBoundarySpec::BorrowLike { .. } => None,
        }
    }

    pub(crate) fn realize_boundary_class<'db>(
        source: Option<&BoundarySourceClass<'db>>,
        boundary: &RuntimeBoundarySpec<'db>,
    ) -> Option<RuntimeClass<'db>> {
        match boundary {
            RuntimeBoundarySpec::Exact(class) => Some(class.clone()),
            RuntimeBoundarySpec::BorrowLike { .. } => source
                .and_then(|source| {
                    source
                        .value
                        .as_ref()
                        .filter(|class| Self::class_satisfies_boundary(class, boundary))
                        .cloned()
                        .or_else(|| {
                            source
                                .addr
                                .as_ref()
                                .filter(|class| Self::class_satisfies_boundary(class, boundary))
                                .cloned()
                        })
                })
                .or_else(|| Self::placeholder_class(boundary)),
        }
    }

    pub(crate) fn param_arg_class<'db>(
        source: &BoundarySourceClass<'db>,
        plan: &RuntimeParamPlan<'db>,
    ) -> Option<RuntimeClass<'db>> {
        match plan {
            RuntimeParamPlan::Erased => None,
            RuntimeParamPlan::Boundary(boundary) => {
                Self::realize_boundary_class(Some(source), boundary)
            }
            RuntimeParamPlan::PassActual => source.value.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::{
        AddressSpaceKind, BorrowAccess, BorrowTransportSet, ScalarClass, ScalarRepr, ScalarRole,
    };

    fn word_class<'db>() -> RuntimeClass<'db> {
        RuntimeClass::Scalar(ScalarClass {
            repr: ScalarRepr::Int {
                bits: 256,
                signed: false,
            },
            role: ScalarRole::Plain,
        })
    }

    fn raw_boundary<'db>() -> RuntimeBoundarySpec<'db> {
        RuntimeBoundarySpec::BorrowLike {
            pointee: word_class(),
            access: BorrowAccess::ReadWrite,
            allow: BorrowTransportSet {
                allow_object: false,
                allow_const: false,
                provider_spaces: Vec::new().into_boxed_slice(),
                allow_raw_addr: true,
            },
        }
    }

    #[test]
    fn exact_boundary_preserves_raw_addr_space() {
        let actual = RuntimeClass::RawAddr {
            space: AddressSpaceKind::Storage,
            target: None,
        };
        let desired = RuntimeBoundarySpec::Exact(RuntimeClass::RawAddr {
            space: AddressSpaceKind::Memory,
            target: None,
        });
        assert!(CoercionPlanner::class_satisfies_boundary(&actual, &desired));
    }

    #[test]
    fn realize_boundary_class_uses_source_before_placeholder() {
        let source = BoundarySourceClass {
            value: Some(RuntimeClass::RawAddr {
                space: AddressSpaceKind::Storage,
                target: None,
            }),
            addr: None,
        };
        assert_eq!(
            CoercionPlanner::realize_boundary_class(Some(&source), &raw_boundary()),
            source.value
        );
    }

    #[test]
    fn realize_boundary_class_falls_back_to_placeholder() {
        assert_eq!(
            CoercionPlanner::realize_boundary_class(None, &raw_boundary()),
            Some(RuntimeClass::RawAddr {
                space: AddressSpaceKind::Memory,
                target: None,
            })
        );
    }
}
