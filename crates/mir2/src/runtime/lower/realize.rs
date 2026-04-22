use hir::analysis::{
    semantic::{NOperand, NSPlace, SLocalId},
    ty::ty_def::TyId,
};

use crate::runtime::{
    AddressSpaceKind, LayoutId, PlaceRoot, RBlockId, RLocalId, RefKind, RefView,
    RuntimeBoundarySpec, RuntimeClass, RuntimeParamPlan, RuntimePlace,
};

#[derive(Clone, Debug)]
pub(super) struct SelectedRuntimeArg<'db> {
    pub(super) class: RuntimeClass<'db>,
    pub(super) source: RuntimeArgSource<'db>,
    pub(super) use_plan: RuntimeValueUsePlan<'db>,
}

#[derive(Clone, Debug)]
pub(super) enum RuntimeArgSource<'db> {
    SemanticOperand(NOperand),
    RuntimeValue {
        local: SLocalId,
    },
    HandleLikeValue {
        local: SLocalId,
    },
    PlaceAddress {
        place: NSPlace<'db>,
        semantic_ty: TyId<'db>,
    },
    PlaceValue {
        place: NSPlace<'db>,
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
pub(crate) struct SelectedRuntimeValueArg<'db> {
    pub(crate) source: RLocalId,
    pub(crate) semantic_ty: TyId<'db>,
    pub(crate) class: RuntimeClass<'db>,
    pub(crate) use_plan: RuntimeValueUsePlan<'db>,
}

impl<'db> SelectedRuntimeValueArg<'db> {
    fn use_value(source: RLocalId, semantic_ty: TyId<'db>, class: RuntimeClass<'db>) -> Self {
        Self {
            source,
            semantic_ty,
            class,
            use_plan: RuntimeValueUsePlan::UseValue,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum RuntimeValueMaterialization<'db> {
    ObjectRef { layout: LayoutId<'db> },
    RawAddrSlot { pointee: RuntimeClass<'db> },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum RuntimeValueUsePlan<'db> {
    UseValue,
    AddrOfRuntimePlace {
        place: RuntimePlace<'db>,
        class: RuntimeClass<'db>,
    },
    CoerceValue {
        target: RuntimeClass<'db>,
    },
    MaterializeValue {
        materialization: RuntimeValueMaterialization<'db>,
    },
}

impl<'db> RuntimeValueUsePlan<'db> {
    pub(crate) fn class(&self, source: &RuntimeClass<'db>) -> RuntimeClass<'db> {
        match self {
            Self::UseValue => source.clone(),
            Self::AddrOfRuntimePlace { class, .. } => class.clone(),
            Self::CoerceValue { target } => target.clone(),
            Self::MaterializeValue { materialization } => materialization.class(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct RuntimeValueAddress<'db> {
    pub(crate) place: RuntimePlace<'db>,
    pub(crate) class: RuntimeClass<'db>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct RuntimeValueSource<'db> {
    pub(crate) value: RuntimeClass<'db>,
    pub(crate) address: Option<RuntimeValueAddress<'db>>,
}

pub(crate) struct RuntimeValueUsePlanner;

impl RuntimeValueUsePlanner {
    pub(crate) fn select<'db>(
        source: RuntimeValueSource<'db>,
        boundary: &RuntimeBoundarySpec<'db>,
    ) -> Option<RuntimeValueUsePlan<'db>> {
        match boundary {
            RuntimeBoundarySpec::ExactTransport(target) => Some(RuntimeValueUsePlan::CoerceValue {
                target: target.clone(),
            }),
            RuntimeBoundarySpec::ExactShape(target) => {
                if source.value_satisfies(boundary) {
                    return Some(RuntimeValueUsePlan::UseValue);
                }
                if let Some(address) = source.compatible_address(boundary) {
                    return Some(RuntimeValueUsePlan::AddrOfRuntimePlace {
                        place: address.place,
                        class: address.class,
                    });
                }
                Some(RuntimeValueUsePlan::CoerceValue {
                    target: target.clone(),
                })
            }
            RuntimeBoundarySpec::BorrowLike { .. } if source.value_satisfies(boundary) => {
                Some(RuntimeValueUsePlan::UseValue)
            }
            RuntimeBoundarySpec::BorrowLike { .. } => {
                if let Some(address) = source.compatible_address(boundary) {
                    return Some(RuntimeValueUsePlan::AddrOfRuntimePlace {
                        place: address.place,
                        class: address.class,
                    });
                }
                RuntimeValueMaterialization::for_boundary(boundary).map(|materialization| {
                    RuntimeValueUsePlan::MaterializeValue { materialization }
                })
            }
        }
    }
}

pub(crate) trait RuntimeValueArgSelectionCx<'db> {
    fn runtime_value_class(&self, value: RLocalId) -> Option<RuntimeClass<'db>>;

    fn runtime_value_source(&self, value: RLocalId) -> Option<RuntimeValueSource<'db>>;

    fn promote_runtime_value_address(
        &mut self,
        value: RLocalId,
        boundary: &RuntimeBoundarySpec<'db>,
    ) -> Option<RuntimeValueAddress<'db>>;
}

pub(crate) struct RuntimeValueArgSelector<'cx, Cx> {
    cx: &'cx mut Cx,
}

impl<'cx, Cx> RuntimeValueArgSelector<'cx, Cx> {
    pub(crate) fn new(cx: &'cx mut Cx) -> Self {
        Self { cx }
    }
}

impl<'cx, 'db, Cx> RuntimeValueArgSelector<'cx, Cx>
where
    Cx: RuntimeValueArgSelectionCx<'db>,
{
    pub(crate) fn selected_arg_for_param_plan(
        &mut self,
        source: RLocalId,
        plan: &RuntimeParamPlan<'db>,
        semantic_ty: TyId<'db>,
    ) -> SelectedRuntimeValueArg<'db> {
        match plan {
            RuntimeParamPlan::Erased => {
                panic!("erased runtime param should not have a runtime arg")
            }
            RuntimeParamPlan::PassActual => {
                SelectedRuntimeValueArg::use_value(source, semantic_ty, self.source_class(source))
            }
            RuntimeParamPlan::Boundary(boundary) => {
                let source_class = self.source_class(source);
                let use_plan = self.select_boundary_use_plan(source, boundary);
                SelectedRuntimeValueArg {
                    source,
                    semantic_ty,
                    class: use_plan.class(&source_class),
                    use_plan,
                }
            }
        }
    }

    fn select_boundary_use_plan(
        &mut self,
        source: RLocalId,
        boundary: &RuntimeBoundarySpec<'db>,
    ) -> RuntimeValueUsePlan<'db> {
        let mut value_source = self.cx.runtime_value_source(source).unwrap_or_else(|| {
            panic!("cannot realize erased runtime value {source:?} for boundary {boundary:?}")
        });
        if let Some(address) = self.cx.promote_runtime_value_address(source, boundary) {
            value_source.address = Some(address);
        }
        RuntimeValueUsePlanner::select(value_source, boundary).unwrap_or_else(|| {
            let source = self.source_class(source);
            panic!(
                "runtime boundary has no realizable materialization: source={source:?} boundary={boundary:?}"
            )
        })
    }

    fn source_class(&self, source: RLocalId) -> RuntimeClass<'db> {
        self.cx
            .runtime_value_class(source)
            .unwrap_or_else(|| panic!("cannot pass erased runtime arg {source:?}"))
    }
}

pub(crate) trait RuntimeValueUseEmitter<'db> {
    fn value_class_for_use(&self, value: RLocalId) -> Option<RuntimeClass<'db>>;

    fn coerce_value_for_use(
        &mut self,
        bb: RBlockId,
        src: RLocalId,
        target: &RuntimeClass<'db>,
        semantic_ty: TyId<'db>,
    ) -> RLocalId;

    fn emit_addr_of_place_for_use(
        &mut self,
        bb: RBlockId,
        place: RuntimePlace<'db>,
        class: RuntimeClass<'db>,
        semantic_ty: TyId<'db>,
    ) -> RLocalId;

    fn alloc_value_slot(&mut self, semantic_ty: TyId<'db>, class: RuntimeClass<'db>) -> RLocalId;

    fn push_value_use(&mut self, bb: RBlockId, dst: RLocalId, src: RLocalId);
}

pub(crate) fn emit_runtime_value_use_plan<'db>(
    emitter: &mut impl RuntimeValueUseEmitter<'db>,
    bb: RBlockId,
    src: RLocalId,
    use_plan: RuntimeValueUsePlan<'db>,
    semantic_ty: TyId<'db>,
) -> RLocalId {
    match use_plan {
        RuntimeValueUsePlan::UseValue => src,
        RuntimeValueUsePlan::AddrOfRuntimePlace { place, class } => {
            emitter.emit_addr_of_place_for_use(bb, place, class, semantic_ty)
        }
        RuntimeValueUsePlan::CoerceValue { target } => {
            emitter.coerce_value_for_use(bb, src, &target, semantic_ty)
        }
        RuntimeValueUsePlan::MaterializeValue { materialization } => {
            emit_runtime_value_materialization(emitter, bb, src, materialization, semantic_ty)
        }
    }
}

pub(crate) fn emit_selected_runtime_value_arg<'db>(
    emitter: &mut impl RuntimeValueUseEmitter<'db>,
    bb: RBlockId,
    arg: &SelectedRuntimeValueArg<'db>,
) -> RLocalId {
    let value = emit_runtime_value_use_plan(
        emitter,
        bb,
        arg.source,
        arg.use_plan.clone(),
        arg.semantic_ty,
    );
    let Some(class) = emitter.value_class_for_use(value) else {
        panic!(
            "selected runtime value arg lowered without a runtime class: arg={arg:?}; value={value:?}"
        );
    };
    assert_eq!(
        class, arg.class,
        "selected runtime value arg class mismatch: arg={arg:?}; value={value:?}",
    );
    value
}

pub(crate) fn emit_selected_runtime_value_args<'db>(
    emitter: &mut impl RuntimeValueUseEmitter<'db>,
    bb: RBlockId,
    args: &[SelectedRuntimeValueArg<'db>],
) -> Vec<RLocalId> {
    args.iter()
        .map(|arg| emit_selected_runtime_value_arg(emitter, bb, arg))
        .collect()
}

fn emit_runtime_value_materialization<'db>(
    emitter: &mut impl RuntimeValueUseEmitter<'db>,
    bb: RBlockId,
    src: RLocalId,
    materialization: RuntimeValueMaterialization<'db>,
    semantic_ty: TyId<'db>,
) -> RLocalId {
    match materialization {
        RuntimeValueMaterialization::ObjectRef { layout } => {
            emitter.coerce_value_for_use(bb, src, &RuntimeClass::object_ref(layout), semantic_ty)
        }
        RuntimeValueMaterialization::RawAddrSlot { pointee } => {
            let source = emitter
                .value_class_for_use(src)
                .unwrap_or_else(|| panic!("cannot materialize erased runtime value {src:?}"));
            let stored = if source == pointee {
                src
            } else {
                emitter.coerce_value_for_use(bb, src, &pointee, semantic_ty)
            };
            let slot = emitter.alloc_value_slot(semantic_ty, pointee.clone());
            emitter.push_value_use(bb, slot, stored);
            emitter.emit_addr_of_place_for_use(
                bb,
                RuntimePlace {
                    root: PlaceRoot::Slot(slot),
                    path: Box::default(),
                },
                RuntimeValueMaterialization::RawAddrSlot { pointee }.class(),
                semantic_ty,
            )
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

impl<'db> RuntimeValueMaterialization<'db> {
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

impl<'db> SelectedRuntimeArg<'db> {
    pub(super) fn local_value(local: SLocalId, class: RuntimeClass<'db>) -> Self {
        let source = if class.is_transport() {
            RuntimeArgSource::HandleLikeValue { local }
        } else {
            RuntimeArgSource::RuntimeValue { local }
        };
        let use_plan = RuntimeValueUsePlan::CoerceValue {
            target: class.clone(),
        };
        Self {
            class,
            source,
            use_plan,
        }
    }

    pub(super) fn handle_like_value(local: SLocalId, class: RuntimeClass<'db>) -> Self {
        Self {
            use_plan: RuntimeValueUsePlan::CoerceValue {
                target: class.clone(),
            },
            class,
            source: RuntimeArgSource::HandleLikeValue { local },
        }
    }

    pub(super) fn semantic_operand(arg: NOperand, class: RuntimeClass<'db>) -> Self {
        Self {
            class,
            source: RuntimeArgSource::SemanticOperand(arg),
            use_plan: RuntimeValueUsePlan::UseValue,
        }
    }

    pub(super) fn placeholder(semantic_ty: TyId<'db>, class: RuntimeClass<'db>) -> Self {
        Self {
            class,
            source: RuntimeArgSource::Placeholder { semantic_ty },
            use_plan: RuntimeValueUsePlan::UseValue,
        }
    }

    pub(super) fn place_addr(
        place: NSPlace<'db>,
        semantic_ty: TyId<'db>,
        class: RuntimeClass<'db>,
    ) -> Self {
        Self {
            class,
            source: RuntimeArgSource::PlaceAddress { place, semantic_ty },
            use_plan: RuntimeValueUsePlan::UseValue,
        }
    }

    pub(super) fn place_load(
        place: NSPlace<'db>,
        semantic_ty: TyId<'db>,
        class: RuntimeClass<'db>,
    ) -> Self {
        Self {
            use_plan: RuntimeValueUsePlan::CoerceValue {
                target: class.clone(),
            },
            class,
            source: RuntimeArgSource::PlaceValue { place, semantic_ty },
        }
    }

    pub(super) fn materialized_place(
        place: NSPlace<'db>,
        semantic_ty: TyId<'db>,
        materialization: RuntimeValueMaterialization<'db>,
    ) -> Self {
        Self {
            class: materialization.class(),
            source: RuntimeArgSource::PlaceValue { place, semantic_ty },
            use_plan: RuntimeValueUsePlan::MaterializeValue { materialization },
        }
    }

    pub(super) fn materialized_semantic_operand(
        arg: NOperand,
        boundary: &RuntimeBoundarySpec<'db>,
    ) -> Option<Self> {
        RuntimeValueMaterialization::for_boundary(boundary).map(|materialization| Self {
            class: materialization.class(),
            source: RuntimeArgSource::SemanticOperand(arg),
            use_plan: RuntimeValueUsePlan::MaterializeValue { materialization },
        })
    }
}

impl<'db> RuntimeValueSource<'db> {
    fn value_satisfies(&self, boundary: &RuntimeBoundarySpec<'db>) -> bool {
        RuntimeBoundaryMatcher::class_satisfies_boundary(&self.value, boundary)
    }

    fn compatible_address(
        &self,
        boundary: &RuntimeBoundarySpec<'db>,
    ) -> Option<RuntimeValueAddress<'db>> {
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
    use driver::DriverDataBase;
    use hir::analysis::ty::ty_def::TyId;

    use crate::runtime::{
        AddressSpaceKind, BorrowAccess, BorrowTransportSet, PlaceRoot, RBlockId, RLocalId,
        RuntimeBoundarySpec, RuntimeClass, RuntimePlace, ScalarClass, ScalarRepr, ScalarRole,
    };

    use super::{
        RuntimeBoundaryMatcher, RuntimeValueAddress, RuntimeValueMaterialization,
        RuntimeValueSource, RuntimeValueUseEmitter, RuntimeValueUsePlan, RuntimeValueUsePlanner,
        SelectedRuntimeValueArg, emit_runtime_value_use_plan, emit_selected_runtime_value_args,
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

    fn bool_class<'db>() -> RuntimeClass<'db> {
        RuntimeClass::Scalar(ScalarClass {
            repr: ScalarRepr::Bool,
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

    fn source_with_value<'db>(value: RuntimeClass<'db>) -> RuntimeValueSource<'db> {
        RuntimeValueSource {
            value,
            address: None,
        }
    }

    fn source_with_address<'db>(
        value: RuntimeClass<'db>,
        address: RuntimeClass<'db>,
    ) -> RuntimeValueSource<'db> {
        RuntimeValueSource {
            value,
            address: Some(RuntimeValueAddress {
                place: RuntimePlace {
                    root: PlaceRoot::Slot(RLocalId::new(0)),
                    path: Box::default(),
                },
                class: address,
            }),
        }
    }

    #[derive(Debug, PartialEq, Eq)]
    enum FakeEmitOp<'db> {
        Coerce {
            src: RLocalId,
            dst: RLocalId,
            target: RuntimeClass<'db>,
        },
        AddrOf {
            dst: RLocalId,
            class: RuntimeClass<'db>,
        },
        Assign {
            dst: RLocalId,
            src: RLocalId,
        },
    }

    struct FakeBoundaryEmitter<'db> {
        classes: Vec<Option<RuntimeClass<'db>>>,
        roots: Vec<Option<RuntimeClass<'db>>>,
        ops: Vec<FakeEmitOp<'db>>,
    }

    impl<'db> FakeBoundaryEmitter<'db> {
        fn new(classes: Vec<Option<RuntimeClass<'db>>>) -> Self {
            let roots = vec![None; classes.len()];
            Self {
                classes,
                roots,
                ops: Vec::new(),
            }
        }

        fn push_value(&mut self, class: RuntimeClass<'db>) -> RLocalId {
            let id = RLocalId::new(self.classes.len());
            self.classes.push(Some(class));
            self.roots.push(None);
            id
        }
    }

    impl<'db> RuntimeValueUseEmitter<'db> for FakeBoundaryEmitter<'db> {
        fn value_class_for_use(&self, value: RLocalId) -> Option<RuntimeClass<'db>> {
            self.classes.get(value.index())?.clone()
        }

        fn coerce_value_for_use(
            &mut self,
            bb: RBlockId,
            src: RLocalId,
            target: &RuntimeClass<'db>,
            semantic_ty: TyId<'db>,
        ) -> RLocalId {
            let _ = (bb, semantic_ty);
            let dst = self.push_value(target.clone());
            self.ops.push(FakeEmitOp::Coerce {
                src,
                dst,
                target: target.clone(),
            });
            dst
        }

        fn emit_addr_of_place_for_use(
            &mut self,
            bb: RBlockId,
            place: RuntimePlace<'db>,
            class: RuntimeClass<'db>,
            semantic_ty: TyId<'db>,
        ) -> RLocalId {
            let _ = (bb, place, semantic_ty);
            let dst = self.push_value(class.clone());
            self.ops.push(FakeEmitOp::AddrOf { dst, class });
            dst
        }

        fn alloc_value_slot(
            &mut self,
            semantic_ty: TyId<'db>,
            class: RuntimeClass<'db>,
        ) -> RLocalId {
            let _ = semantic_ty;
            let dst = self.push_value(class.clone());
            self.roots[dst.index()] = Some(class);
            dst
        }

        fn push_value_use(&mut self, bb: RBlockId, dst: RLocalId, src: RLocalId) {
            let _ = bb;
            self.ops.push(FakeEmitOp::Assign { dst, src });
        }
    }

    #[test]
    fn raw_addr_materialization_has_explicit_class() {
        let materialization = RuntimeValueMaterialization::for_boundary(&raw_boundary())
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
    fn exact_shape_use_plan_preserves_source_transport() {
        let source = raw_addr_class(AddressSpaceKind::Storage);
        let boundary = RuntimeBoundarySpec::ExactShape(RuntimeClass::RawAddr {
            space: AddressSpaceKind::Memory,
            target: None,
        });
        let use_plan = RuntimeValueUsePlanner::select(source_with_value(source.clone()), &boundary)
            .expect("exact-shape source should select a use plan");
        assert_eq!(use_plan.class(&source), source,);
    }

    #[test]
    fn exact_shape_selector_uses_compatible_source_value() {
        let source = raw_addr_class(AddressSpaceKind::Storage);
        let boundary = RuntimeBoundarySpec::ExactShape(raw_addr_class(AddressSpaceKind::Memory));

        assert_eq!(
            RuntimeValueUsePlanner::select(source_with_value(source), &boundary),
            Some(RuntimeValueUsePlan::UseValue)
        );
    }

    #[test]
    fn exact_transport_selector_coerces_to_target() {
        let target = raw_addr_class(AddressSpaceKind::Memory);
        let boundary = RuntimeBoundarySpec::ExactTransport(target.clone());

        assert_eq!(
            RuntimeValueUsePlanner::select(
                source_with_value(raw_addr_class(AddressSpaceKind::Storage)),
                &boundary
            ),
            Some(RuntimeValueUsePlan::CoerceValue { target })
        );
    }

    #[test]
    fn borrow_like_selector_uses_compatible_address() {
        let address = raw_addr_class(AddressSpaceKind::Storage);
        let source = source_with_address(word_class(), address.clone());

        assert_eq!(
            RuntimeValueUsePlanner::select(source, &raw_boundary()),
            Some(RuntimeValueUsePlan::AddrOfRuntimePlace {
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
            RuntimeValueUsePlanner::select(source_with_value(word_class()), &raw_boundary()),
            Some(RuntimeValueUsePlan::MaterializeValue {
                materialization: RuntimeValueMaterialization::RawAddrSlot {
                    pointee: word_class(),
                },
            })
        );
    }

    #[test]
    fn raw_addr_slot_materialization_coerces_stores_and_addresses_slot() {
        let db = DriverDataBase::default();
        let semantic_ty = TyId::unit(&db);
        let mut emitter = FakeBoundaryEmitter::new(vec![Some(bool_class())]);

        let result = emit_runtime_value_use_plan(
            &mut emitter,
            RBlockId::new(0),
            RLocalId::new(0),
            RuntimeValueUsePlan::MaterializeValue {
                materialization: RuntimeValueMaterialization::RawAddrSlot {
                    pointee: word_class(),
                },
            },
            semantic_ty,
        );

        assert_eq!(result, RLocalId::new(3));
        assert_eq!(
            emitter.classes[result.index()],
            Some(raw_addr_class(AddressSpaceKind::Memory))
        );
        assert_eq!(emitter.roots[RLocalId::new(2).index()], Some(word_class()));
        assert_eq!(
            emitter.ops,
            vec![
                FakeEmitOp::Coerce {
                    src: RLocalId::new(0),
                    dst: RLocalId::new(1),
                    target: word_class(),
                },
                FakeEmitOp::Assign {
                    dst: RLocalId::new(2),
                    src: RLocalId::new(1),
                },
                FakeEmitOp::AddrOf {
                    dst: RLocalId::new(3),
                    class: raw_addr_class(AddressSpaceKind::Memory),
                },
            ]
        );
    }

    #[test]
    fn selected_runtime_value_args_share_use_plan_lowering() {
        let db = DriverDataBase::default();
        let semantic_ty = TyId::unit(&db);
        let mut emitter = FakeBoundaryEmitter::new(vec![Some(word_class())]);

        let values = emit_selected_runtime_value_args(
            &mut emitter,
            RBlockId::new(0),
            &[SelectedRuntimeValueArg {
                source: RLocalId::new(0),
                semantic_ty,
                class: word_class(),
                use_plan: RuntimeValueUsePlan::UseValue,
            }],
        );

        assert_eq!(values, vec![RLocalId::new(0)]);
        assert!(emitter.ops.is_empty());
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
