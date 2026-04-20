use crate::runtime::{RefKind, RefView, RuntimeBoundarySpec, RuntimeClass};

pub(crate) struct CoercionPlanner;

impl CoercionPlanner {
    pub(crate) fn target_prefers_transport(class: &RuntimeClass<'_>) -> bool {
        class.is_transport()
    }

    pub(crate) fn class_satisfies_boundary<'db>(
        class: &RuntimeClass<'db>,
        boundary: &RuntimeBoundarySpec<'db>,
    ) -> bool {
        match boundary {
            RuntimeBoundarySpec::ExactTransport(expected) => class == expected,
            RuntimeBoundarySpec::ExactShape(expected) => {
                Self::class_matches_shape_boundary(class, expected)
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

    pub(crate) fn class_matches_shape_boundary<'db>(
        actual: &RuntimeClass<'db>,
        expected: &RuntimeClass<'db>,
    ) -> bool {
        match (actual, expected) {
            (
                RuntimeClass::Ref {
                    pointee: actual_pointee,
                    view: actual_view,
                    ..
                },
                RuntimeClass::Ref {
                    pointee: expected_pointee,
                    view: expected_view,
                    ..
                },
            ) => actual_pointee == expected_pointee && actual_view == expected_view,
            (
                RuntimeClass::RawAddr {
                    target: actual_target,
                    ..
                },
                RuntimeClass::Ref { pointee, .. },
            ) => actual_target == &pointee.aggregate_layout(),
            (
                RuntimeClass::RawAddr {
                    target: actual_target,
                    ..
                },
                RuntimeClass::RawAddr {
                    target: expected_target,
                    ..
                },
            ) => actual_target == expected_target,
            _ => actual == expected,
        }
    }

    pub(crate) fn placeholder_class<'db>(
        boundary: &RuntimeBoundarySpec<'db>,
    ) -> Option<RuntimeClass<'db>> {
        match boundary {
            RuntimeBoundarySpec::ExactTransport(class) | RuntimeBoundarySpec::ExactShape(class) => {
                Some(class.clone())
            }
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
        let desired = RuntimeBoundarySpec::ExactShape(RuntimeClass::RawAddr {
            space: AddressSpaceKind::Memory,
            target: None,
        });
        assert!(CoercionPlanner::class_satisfies_boundary(&actual, &desired));
    }

    #[test]
    fn exact_transport_rejects_raw_addr_space_mismatch() {
        let actual = RuntimeClass::RawAddr {
            space: AddressSpaceKind::Storage,
            target: None,
        };
        let desired = RuntimeBoundarySpec::ExactTransport(RuntimeClass::RawAddr {
            space: AddressSpaceKind::Memory,
            target: None,
        });
        assert!(!CoercionPlanner::class_satisfies_boundary(
            &actual, &desired
        ));
    }

    #[test]
    fn placeholder_class_uses_memory_raw_addr_for_scalar_borrow_boundary() {
        assert_eq!(
            CoercionPlanner::placeholder_class(&raw_boundary()),
            Some(RuntimeClass::RawAddr {
                space: AddressSpaceKind::Memory,
                target: None,
            })
        );
    }
}
