use hir::analysis::{
    semantic::{NOperand, NSPlace, SLocalId},
    ty::ty_def::TyId,
};

use crate::runtime::{AddressSpaceKind, LayoutId, RuntimeBoundarySpec, RuntimeClass, RuntimePlace};

use super::coerce::CoercionPlanner;

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

#[derive(Clone, Debug)]
pub(crate) enum RuntimeBoundaryMaterialization<'db> {
    ObjectRef { layout: LayoutId<'db> },
    RawAddrSlot { pointee: RuntimeClass<'db> },
}

#[derive(Clone, Debug)]
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

pub(crate) struct RuntimeBoundarySourceClasses<'db> {
    pub(crate) value: Option<RuntimeClass<'db>>,
    pub(crate) address: Option<RuntimeClass<'db>>,
}

impl<'db> RuntimeBoundarySourceClasses<'db> {
    pub(crate) fn realized_boundary_class(
        &self,
        boundary: &RuntimeBoundarySpec<'db>,
    ) -> Option<RuntimeClass<'db>> {
        match boundary {
            RuntimeBoundarySpec::ExactTransport(target) => Some(target.clone()),
            RuntimeBoundarySpec::ExactShape(_) | RuntimeBoundarySpec::BorrowLike { .. } => self
                .compatible_class(boundary)
                .or_else(|| CoercionPlanner::placeholder_class(boundary)),
        }
    }

    pub(crate) fn compatible_class(
        &self,
        boundary: &RuntimeBoundarySpec<'db>,
    ) -> Option<RuntimeClass<'db>> {
        self.compatible_value_class(boundary)
            .or_else(|| self.compatible_address_class(boundary))
    }

    pub(crate) fn compatible_value_class(
        &self,
        boundary: &RuntimeBoundarySpec<'db>,
    ) -> Option<RuntimeClass<'db>> {
        self.value
            .as_ref()
            .filter(|class| CoercionPlanner::class_satisfies_boundary(class, boundary))
            .cloned()
    }

    pub(crate) fn compatible_address_class(
        &self,
        boundary: &RuntimeBoundarySpec<'db>,
    ) -> Option<RuntimeClass<'db>> {
        self.address
            .as_ref()
            .filter(|class| CoercionPlanner::class_satisfies_boundary(class, boundary))
            .cloned()
    }
}

#[cfg(test)]
mod tests {
    use crate::runtime::{
        AddressSpaceKind, BorrowAccess, BorrowTransportSet, RuntimeBoundarySpec, RuntimeClass,
        ScalarClass, ScalarRepr, ScalarRole,
    };

    use super::{RuntimeBoundaryMaterialization, RuntimeBoundarySourceClasses};

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
        let source = RuntimeClass::RawAddr {
            space: AddressSpaceKind::Storage,
            target: None,
        };
        let boundary = RuntimeBoundarySpec::ExactShape(RuntimeClass::RawAddr {
            space: AddressSpaceKind::Memory,
            target: None,
        });
        let classes = RuntimeBoundarySourceClasses {
            value: Some(source.clone()),
            address: None,
        };
        assert_eq!(classes.realized_boundary_class(&boundary), Some(source));
    }
}
