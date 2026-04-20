use hir::analysis::{
    semantic::{NOperand, NSPlace, SLocalId},
    ty::ty_def::TyId,
};

use crate::runtime::{
    AddressSpaceKind, LayoutId, RefKind, RefView, RuntimeBoundarySpec, RuntimeClass, RuntimePlace,
};

#[derive(Clone, Debug)]
pub(super) struct SelectedRuntimeArg<'db> {
    pub(super) class: RuntimeClass<'db>,
    pub(super) realization: RuntimeArgRealization<'db>,
}

#[derive(Clone, Debug)]
pub(super) enum RuntimeArgRealization<'db> {
    LowerSemanticOperand(NOperand),
    UseRuntimeValue {
        local: SLocalId,
    },
    UseHandleLikeValue {
        local: SLocalId,
    },
    AddrOfPlace {
        place: NSPlace<'db>,
        semantic_ty: TyId<'db>,
    },
    LoadPlaceValue {
        place: NSPlace<'db>,
        semantic_ty: TyId<'db>,
    },
    MaterializePlaceValue {
        place: NSPlace<'db>,
        materialization: RuntimeBoundaryMaterialization<'db>,
        semantic_ty: TyId<'db>,
    },
    MaterializeSemanticValue {
        operand: NOperand,
        materialization: RuntimeBoundaryMaterialization<'db>,
        semantic_ty: TyId<'db>,
    },
    AggregateFromRuntimeSource {
        local: SLocalId,
    },
    Placeholder {
        semantic_ty: TyId<'db>,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum RuntimeBoundaryMaterialization<'db> {
    ObjectRef { layout: LayoutId<'db> },
    RawAddrSlot { pointee: RuntimeClass<'db> },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum RuntimeBoundaryValueRealization<'db> {
    UseValue,
    AddrOfRuntimePlace {
        place: RuntimePlace<'db>,
        class: RuntimeClass<'db>,
    },
    CoerceValue {
        target: RuntimeClass<'db>,
    },
    MaterializeValue {
        materialization: RuntimeBoundaryMaterialization<'db>,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct RuntimeBoundaryAddress<'db> {
    pub(crate) place: RuntimePlace<'db>,
    pub(crate) class: RuntimeClass<'db>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct RuntimeBoundaryValueSource<'db> {
    pub(crate) value: RuntimeClass<'db>,
    pub(crate) address: Option<RuntimeBoundaryAddress<'db>>,
}

pub(crate) struct RuntimeBoundaryValueSelector;

impl RuntimeBoundaryValueSelector {
    pub(crate) fn select<'db>(
        source: RuntimeBoundaryValueSource<'db>,
        boundary: &RuntimeBoundarySpec<'db>,
    ) -> Option<RuntimeBoundaryValueRealization<'db>> {
        match boundary {
            RuntimeBoundarySpec::ExactTransport(target) => {
                Some(RuntimeBoundaryValueRealization::CoerceValue {
                    target: target.clone(),
                })
            }
            RuntimeBoundarySpec::ExactShape(target) => {
                if source.value_satisfies(boundary) {
                    return Some(RuntimeBoundaryValueRealization::UseValue);
                }
                if let Some(address) = source.compatible_address(boundary) {
                    return Some(RuntimeBoundaryValueRealization::AddrOfRuntimePlace {
                        place: address.place,
                        class: address.class,
                    });
                }
                Some(RuntimeBoundaryValueRealization::CoerceValue {
                    target: target.clone(),
                })
            }
            RuntimeBoundarySpec::BorrowLike { .. } if source.value_satisfies(boundary) => {
                Some(RuntimeBoundaryValueRealization::UseValue)
            }
            RuntimeBoundarySpec::BorrowLike { .. } => {
                if let Some(address) = source.compatible_address(boundary) {
                    return Some(RuntimeBoundaryValueRealization::AddrOfRuntimePlace {
                        place: address.place,
                        class: address.class,
                    });
                }
                RuntimeBoundaryMaterialization::for_boundary(boundary).map(|materialization| {
                    RuntimeBoundaryValueRealization::MaterializeValue { materialization }
                })
            }
        }
    }
}

pub(crate) struct RuntimeBoundaryMatcher;

impl RuntimeBoundaryMatcher {
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
                    space: AddressSpaceKind::Memory,
                    target: pointee.aggregate_layout(),
                })
            }
            RuntimeBoundarySpec::BorrowLike { .. } => None,
        }
    }

    fn class_matches_shape_boundary<'db>(
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
}

impl<'db> RuntimeBoundaryMaterialization<'db> {
    pub(crate) fn for_boundary(boundary: &RuntimeBoundarySpec<'db>) -> Option<Self> {
        match boundary {
            RuntimeBoundarySpec::BorrowLike { pointee, allow, .. }
                if pointee.aggregate_layout().is_some() && allow.allow_object =>
            {
                Some(Self::ObjectRef {
                    layout: pointee.aggregate_layout().expect("aggregate layout"),
                })
            }
            RuntimeBoundarySpec::BorrowLike { pointee, allow, .. }
                if pointee.aggregate_layout().is_none() && allow.allow_raw_addr =>
            {
                Some(Self::RawAddrSlot {
                    pointee: pointee.clone(),
                })
            }
            RuntimeBoundarySpec::ExactTransport(_)
            | RuntimeBoundarySpec::ExactShape(_)
            | RuntimeBoundarySpec::BorrowLike { .. } => None,
        }
    }

    pub(crate) fn class(&self) -> RuntimeClass<'db> {
        match self {
            Self::ObjectRef { layout } => RuntimeClass::object_ref(*layout),
            Self::RawAddrSlot { pointee } => RuntimeClass::RawAddr {
                space: AddressSpaceKind::Memory,
                target: pointee.aggregate_layout(),
            },
        }
    }
}

impl<'db> RuntimeBoundaryValueSource<'db> {
    pub(crate) fn realized_boundary_class(
        &self,
        boundary: &RuntimeBoundarySpec<'db>,
    ) -> Option<RuntimeClass<'db>> {
        match boundary {
            RuntimeBoundarySpec::ExactTransport(target) => Some(target.clone()),
            RuntimeBoundarySpec::ExactShape(_) | RuntimeBoundarySpec::BorrowLike { .. } => self
                .compatible_class(boundary)
                .or_else(|| RuntimeBoundaryMatcher::placeholder_class(boundary)),
        }
    }

    pub(crate) fn compatible_class(
        &self,
        boundary: &RuntimeBoundarySpec<'db>,
    ) -> Option<RuntimeClass<'db>> {
        if self.value_satisfies(boundary) {
            return Some(self.value.clone());
        }
        self.compatible_address(boundary)
            .map(|address| address.class)
    }

    fn value_satisfies(&self, boundary: &RuntimeBoundarySpec<'db>) -> bool {
        RuntimeBoundaryMatcher::class_satisfies_boundary(&self.value, boundary)
    }

    fn compatible_address(
        &self,
        boundary: &RuntimeBoundarySpec<'db>,
    ) -> Option<RuntimeBoundaryAddress<'db>> {
        self.address
            .as_ref()
            .filter(|address| {
                RuntimeBoundaryMatcher::class_satisfies_boundary(&address.class, boundary)
            })
            .cloned()
    }
}

#[cfg(test)]
mod tests {
    use cranelift_entity::EntityRef;

    use crate::runtime::{
        AddressSpaceKind, BorrowAccess, BorrowTransportSet, PlaceRoot, RLocalId,
        RuntimeBoundarySpec, RuntimeClass, RuntimePlace, ScalarClass, ScalarRepr, ScalarRole,
    };

    use super::{
        RuntimeBoundaryAddress, RuntimeBoundaryMatcher, RuntimeBoundaryMaterialization,
        RuntimeBoundaryValueRealization, RuntimeBoundaryValueSelector, RuntimeBoundaryValueSource,
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

    fn raw_addr_class<'db>(space: AddressSpaceKind) -> RuntimeClass<'db> {
        RuntimeClass::RawAddr {
            space,
            target: None,
        }
    }

    fn source_with_value<'db>(value: RuntimeClass<'db>) -> RuntimeBoundaryValueSource<'db> {
        RuntimeBoundaryValueSource {
            value,
            address: None,
        }
    }

    fn source_with_address<'db>(
        value: RuntimeClass<'db>,
        address: RuntimeClass<'db>,
    ) -> RuntimeBoundaryValueSource<'db> {
        RuntimeBoundaryValueSource {
            value,
            address: Some(RuntimeBoundaryAddress {
                place: RuntimePlace {
                    root: PlaceRoot::Slot(RLocalId::new(0)),
                    path: Box::default(),
                },
                class: address,
            }),
        }
    }

    #[test]
    fn raw_addr_materialization_has_explicit_class() {
        let materialization = RuntimeBoundaryMaterialization::for_boundary(&raw_boundary())
            .expect("raw boundary should materialize through a slot");
        assert_eq!(
            materialization.class(),
            RuntimeClass::RawAddr {
                space: AddressSpaceKind::Memory,
                target: None,
            }
        );
    }

    #[test]
    fn exact_shape_realization_preserves_source_transport() {
        let source = raw_addr_class(AddressSpaceKind::Storage);
        let boundary = RuntimeBoundarySpec::ExactShape(RuntimeClass::RawAddr {
            space: AddressSpaceKind::Memory,
            target: None,
        });
        let source = source_with_value(source.clone());
        assert_eq!(
            source.realized_boundary_class(&boundary),
            Some(source.value)
        );
    }

    #[test]
    fn exact_shape_selector_uses_compatible_source_value() {
        let source = raw_addr_class(AddressSpaceKind::Storage);
        let boundary = RuntimeBoundarySpec::ExactShape(raw_addr_class(AddressSpaceKind::Memory));

        assert_eq!(
            RuntimeBoundaryValueSelector::select(source_with_value(source), &boundary),
            Some(RuntimeBoundaryValueRealization::UseValue)
        );
    }

    #[test]
    fn exact_transport_selector_coerces_to_target() {
        let target = raw_addr_class(AddressSpaceKind::Memory);
        let boundary = RuntimeBoundarySpec::ExactTransport(target.clone());

        assert_eq!(
            RuntimeBoundaryValueSelector::select(
                source_with_value(raw_addr_class(AddressSpaceKind::Storage)),
                &boundary
            ),
            Some(RuntimeBoundaryValueRealization::CoerceValue { target })
        );
    }

    #[test]
    fn borrow_like_selector_uses_compatible_address() {
        let address = raw_addr_class(AddressSpaceKind::Storage);
        let source = source_with_address(word_class(), address.clone());

        assert_eq!(
            RuntimeBoundaryValueSelector::select(source, &raw_boundary()),
            Some(RuntimeBoundaryValueRealization::AddrOfRuntimePlace {
                place: RuntimePlace {
                    root: PlaceRoot::Slot(RLocalId::new(0)),
                    path: Box::default(),
                },
                class: address,
            })
        );
    }

    #[test]
    fn borrow_like_selector_materializes_when_no_source_address_matches() {
        assert_eq!(
            RuntimeBoundaryValueSelector::select(source_with_value(word_class()), &raw_boundary()),
            Some(RuntimeBoundaryValueRealization::MaterializeValue {
                materialization: RuntimeBoundaryMaterialization::RawAddrSlot {
                    pointee: word_class(),
                },
            })
        );
    }

    #[test]
    fn exact_shape_boundary_preserves_raw_addr_space() {
        let actual = RuntimeClass::RawAddr {
            space: AddressSpaceKind::Storage,
            target: None,
        };
        let desired = RuntimeBoundarySpec::ExactShape(RuntimeClass::RawAddr {
            space: AddressSpaceKind::Memory,
            target: None,
        });
        assert!(RuntimeBoundaryMatcher::class_satisfies_boundary(
            &actual, &desired
        ));
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
        assert!(!RuntimeBoundaryMatcher::class_satisfies_boundary(
            &actual, &desired
        ));
    }

    #[test]
    fn placeholder_class_uses_memory_raw_addr_for_scalar_borrow_boundary() {
        assert_eq!(
            RuntimeBoundaryMatcher::placeholder_class(&raw_boundary()),
            Some(RuntimeClass::RawAddr {
                space: AddressSpaceKind::Memory,
                target: None,
            })
        );
    }
}
