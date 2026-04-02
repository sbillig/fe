use cranelift_entity::EntityRef;
use hir::analysis::{
    semantic::{
        FieldIndex, GenericSubst, ImplEnv, SBlockId, SConst, SEffectArg, SEffectArgValue, SExpr,
        SLocalId, SPlace, SPlaceElem, SStmt, STerminator, SemConstId, SemanticBody,
        SemanticCalleeRef, SemanticCodeRegionRef, SemanticInstance, SemanticInstanceKey,
        VariantIndex, ctfe::canonicalize_semantic_consts, get_or_build_semantic_instance,
        owner_effect_bindings, sem_const_ty, semantic_may_return_normally,
    },
    ty::{
        corelib::resolve_lib_func_path,
        trait_resolution::PredicateListId,
        ty_check::{BodyOwner, EffectPassMode},
        ty_def::TyId,
    },
};
use hir::hir_def::{ArithBinOp, BinOp, CompBinOp, Func, UnOp};

use crate::{
    db::MirDb,
    instance::{RuntimeInstance, RuntimeInstanceKey},
    runtime::{
        AddressSpaceKind, ConstScalar, HandleKind, HandleView, IntrinsicArithBinOp, LayoutId,
        LocalSlotKind, PlaceElem, PlaceRoot, RBlock, RBlockId, RExpr, RLocal, RLocalId, RStmt,
        RTerminator, RuntimeBody, RuntimeCarrier, RuntimeClass, RuntimeCodeRegion,
        RuntimeCodeRegionKey, RuntimePlace, RuntimeSignature, SaturatingBinOp, ScalarClass,
        ScalarRepr, ScalarRole, VariantId,
    },
};

use super::{
    class::{
        ContractMetadataBuiltin, contract_metadata_builtin, infer_local_carriers,
        is_effect_binding, provider_class_for_target_in_context, runtime_param_class,
        runtime_param_locals, runtime_signature_for_key, same_owner_effect_binding,
        semantic_return_ty, stored_class_for_ty_in_context, top_level_class_for_ty_in_context,
    },
    consts::{const_scalar_from_value, enum_tag_scalar, lower_const_region},
    layout::layout_for_ty_in_context,
    place::{
        effect_arg_address_space, project_field_class, project_index_class,
        project_variant_field_class,
    },
};

pub fn lower_to_rmir<'db>(db: &'db dyn MirDb, instance: RuntimeInstance<'db>) -> RuntimeBody<'db> {
    let key = instance.key(db);
    let semantic = key
        .semantic(db)
        .expect("semantic lowering only applies to semantic runtime instances");
    let semantic_body = canonicalize_semantic_consts(db, semantic);
    let local_carriers = infer_local_carriers(
        db,
        &semantic_body,
        key.params(db),
        &runtime_param_locals(db, semantic, key.params(db)),
        semantic.key(db).owner(db).scope().into(),
        semantic.key(db).instantiate_typed_body(db).assumptions(),
    );
    let signature = runtime_signature_for_key(db, semantic, key.params(db));
    let mut cx = RmirLowerCtxt::new(
        db,
        instance,
        key,
        semantic_body,
        local_carriers,
        signature.clone(),
    );
    cx.lower_blocks();
    cx.finish(signature)
}

pub(super) struct RmirLowerCtxt<'db> {
    pub(super) db: &'db dyn MirDb,
    pub(super) instance: RuntimeInstance<'db>,
    pub(super) key: RuntimeInstanceKey<'db>,
    pub(super) semantic_body: SemanticBody<'db>,
    pub(super) ret_class: Option<RuntimeClass<'db>>,
    pub(super) scope: Option<hir::hir_def::scope_graph::ScopeId<'db>>,
    pub(super) assumptions: PredicateListId<'db>,
    pub(super) locals: Vec<RLocal<'db>>,
    pub(super) blocks: Vec<RBlock<'db>>,
    pub(super) terminated_blocks: Vec<bool>,
}

enum LoweredBuiltinCall<'db> {
    Expr {
        builtin: crate::runtime::RuntimeBuiltin<'db>,
        class: Option<RuntimeClass<'db>>,
    },
    Terminator(RTerminator<'db>),
}

impl<'db> RmirLowerCtxt<'db> {
    fn new(
        db: &'db dyn MirDb,
        instance: RuntimeInstance<'db>,
        key: RuntimeInstanceKey<'db>,
        semantic_body: SemanticBody<'db>,
        local_carriers: Vec<RuntimeCarrier<'db>>,
        signature: RuntimeSignature<'db>,
    ) -> Self {
        let typed_body = key
            .semantic(db)
            .expect("runtime lowering requires a semantic instance")
            .key(db)
            .instantiate_typed_body(db);
        let terminated_blocks = vec![false; semantic_body.blocks.len()];
        let locals = semantic_body
            .locals
            .iter()
            .enumerate()
            .map(|(idx, local)| {
                let carrier = local_carriers
                    .get(idx)
                    .cloned()
                    .unwrap_or(RuntimeCarrier::Erased);
                let slot = match &carrier {
                    RuntimeCarrier::Erased | RuntimeCarrier::Value(RuntimeClass::Handle { .. }) => {
                        LocalSlotKind::None
                    }
                    RuntimeCarrier::Value(class) => LocalSlotKind::Slot(class.clone()),
                };
                RLocal {
                    semantic_ty: local.ty,
                    carrier,
                    slot,
                }
            })
            .collect::<Vec<_>>();
        let blocks = Vec::with_capacity(semantic_body.blocks.len());
        Self {
            db,
            instance,
            key,
            semantic_body,
            ret_class: signature.ret.clone(),
            scope: typed_body.body().map(|body| body.scope()),
            assumptions: typed_body.assumptions(),
            locals,
            blocks,
            terminated_blocks,
        }
    }

    fn finish(mut self, signature: RuntimeSignature<'db>) -> RuntimeBody<'db> {
        while self.blocks.len() < self.semantic_body.blocks.len() {
            self.blocks.push(RBlock {
                stmts: Vec::new(),
                terminator: RTerminator::Return(None),
            });
        }
        RuntimeBody {
            owner: self.instance,
            key: self.key,
            signature,
            locals: self.locals,
            blocks: self.blocks,
        }
    }

    fn layout_for_ty(&self, ty: TyId<'db>) -> LayoutId<'db> {
        layout_for_ty_in_context(self.db, ty, self.scope, self.assumptions)
    }

    fn stored_class_for_ty(&self, ty: TyId<'db>) -> RuntimeClass<'db> {
        stored_class_for_ty_in_context(self.db, ty, self.scope, self.assumptions)
    }

    fn top_level_class_for_ty(
        &self,
        ty: TyId<'db>,
        default_space: crate::runtime::AddressSpaceKind,
    ) -> Option<RuntimeClass<'db>> {
        top_level_class_for_ty_in_context(self.db, ty, default_space, self.scope, self.assumptions)
    }

    fn lower_blocks(&mut self) {
        self.blocks = (0..self.semantic_body.blocks.len())
            .map(|_| RBlock {
                stmts: Vec::new(),
                terminator: RTerminator::Return(None),
            })
            .collect();
        self.terminated_blocks = vec![false; self.semantic_body.blocks.len()];
        let blocks = self.semantic_body.blocks.clone();
        for (idx, block) in blocks.iter().enumerate() {
            let bb = RBlockId::from_u32(idx as u32);
            for stmt in &block.stmts {
                self.lower_stmt(bb, stmt);
                if self.terminated_blocks[bb.index()] {
                    break;
                }
            }
            if !self.terminated_blocks[bb.index()] {
                self.blocks[bb.index()].terminator = self.lower_terminator(bb, &block.terminator);
            }
        }
    }

    fn set_terminator(&mut self, bb: RBlockId, terminator: RTerminator<'db>) {
        self.blocks[bb.index()].terminator = terminator;
        self.terminated_blocks[bb.index()] = true;
    }

    fn lower_stmt(&mut self, bb: RBlockId, stmt: &SStmt<'db>) {
        match stmt {
            SStmt::Assign { dst, expr } => self.lower_assign(bb, *dst, expr),
            SStmt::Store { dst, src } => {
                let place = self.lower_place(bb, dst);
                let target = self.project_place_class(&place);
                let value = self.runtime_value_use(bb, *src);
                self.write_value_to_place(bb, place, value, &target);
            }
        }
    }

    fn lower_assign(&mut self, bb: RBlockId, dst: SLocalId, expr: &SExpr<'db>) {
        let desired = self.local_class(dst);
        match desired {
            None => {
                let sink = self.alloc_runtime_temp(TyId::unit(self.db), RuntimeCarrier::Erased);
                self.lower_expr_into(bb, sink, expr);
            }
            Some(_) => self.lower_expr_into(bb, self.runtime_value(dst), expr),
        }
    }

    fn lower_expr_into(&mut self, bb: RBlockId, dst: RLocalId, expr: &SExpr<'db>) {
        let Some(dst_class) = self.value_class(dst).cloned() else {
            if let SExpr::Call {
                callee,
                args,
                effect_args,
            } = expr
            {
                let _ = self.lower_call(bb, *callee, args, effect_args);
            }
            return;
        };

        match expr {
            SExpr::Use(src) => {
                let src = self.runtime_value_use(bb, *src);
                let value = self.coerce_value(bb, src, &dst_class);
                self.push_stmt(
                    bb,
                    RStmt::Assign {
                        dst,
                        expr: RExpr::Use(value),
                    },
                );
            }
            SExpr::Const(const_) => self.lower_const_into(bb, dst, const_),
            SExpr::Unary { op, value } => {
                let value = self.runtime_value_use(bb, *value);
                self.push_stmt(
                    bb,
                    RStmt::Assign {
                        dst,
                        expr: RExpr::Unary { op: *op, value },
                    },
                );
            }
            SExpr::Binary { op, lhs, rhs } => {
                let lhs = self.runtime_value_use(bb, *lhs);
                let rhs = self.runtime_value_use(bb, *rhs);
                self.push_stmt(
                    bb,
                    RStmt::Assign {
                        dst,
                        expr: RExpr::Binary { op: *op, lhs, rhs },
                    },
                );
            }
            SExpr::Cast { value, .. } => {
                let RuntimeClass::Scalar(to) = dst_class else {
                    panic!("casts must lower to scalar carriers");
                };
                let value = self.runtime_value_use(bb, *value);
                self.push_stmt(
                    bb,
                    RStmt::Assign {
                        dst,
                        expr: RExpr::Cast { value, to },
                    },
                );
            }
            SExpr::AggregateMake { ty, fields } => self.lower_aggregate_make(bb, dst, *ty, fields),
            SExpr::EnumMake {
                enum_ty,
                variant,
                fields,
            } => self.lower_enum_make(bb, dst, *enum_ty, *variant, fields),
            SExpr::Field { base, field } => {
                self.lower_field_like(bb, dst, *base, PlaceElem::Field(*field))
            }
            SExpr::Index { base, index } => {
                let index = self.runtime_value_use(bb, *index);
                self.lower_field_like(bb, dst, *base, PlaceElem::Index(index));
            }
            SExpr::Borrow { place, .. } => {
                let place = self.lower_place(bb, place);
                self.push_stmt(
                    bb,
                    RStmt::Assign {
                        dst,
                        expr: RExpr::AddrOf { place },
                    },
                );
            }
            SExpr::GetEnumTag { value } => self.lower_enum_tag(bb, dst, *value),
            SExpr::IsEnumVariant { value, variant } => {
                self.lower_is_enum_variant(bb, dst, *value, *variant);
            }
            SExpr::ExtractEnumField {
                value,
                variant,
                field,
            } => {
                self.lower_extract_enum_field(bb, dst, *value, *variant, *field);
            }
            SExpr::CodeRegionOffset { region } => self.push_stmt(
                bb,
                RStmt::Assign {
                    dst,
                    expr: RExpr::Builtin(crate::runtime::RuntimeBuiltin::CodeRegionOffset {
                        region: self.lower_code_region_ref(region),
                    }),
                },
            ),
            SExpr::CodeRegionLen { region } => self.push_stmt(
                bb,
                RStmt::Assign {
                    dst,
                    expr: RExpr::Builtin(crate::runtime::RuntimeBuiltin::CodeRegionLen {
                        region: self.lower_code_region_ref(region),
                    }),
                },
            ),
            SExpr::Call {
                callee,
                args,
                effect_args,
            } => {
                let value = self.lower_call(bb, *callee, args, effect_args);
                if self.terminated_blocks[bb.index()] {
                    return;
                }
                let value = self.coerce_value(bb, value, &dst_class);
                self.push_stmt(
                    bb,
                    RStmt::Assign {
                        dst,
                        expr: RExpr::Use(value),
                    },
                );
            }
        }
    }

    fn lower_code_region_ref(&self, region: &SemanticCodeRegionRef<'db>) -> RuntimeCodeRegion<'db> {
        match region {
            SemanticCodeRegionRef::ManualContractRoot {
                func,
                contract_name,
                section,
            } => {
                let semantic = get_or_build_semantic_instance(
                    self.db,
                    SemanticInstanceKey::new(
                        self.db,
                        BodyOwner::Func(*func),
                        GenericSubst::empty(self.db),
                        ImplEnv::empty(self.db, func.scope()),
                    ),
                );
                let callee = crate::instance::get_or_build_runtime_instance(
                    self.db,
                    RuntimeInstanceKey::new(
                        self.db,
                        crate::instance::RuntimeInstanceSource::Semantic(semantic),
                        Vec::new(),
                    ),
                );
                RuntimeCodeRegion::new(
                    self.db,
                    RuntimeCodeRegionKey::ManualContractSection {
                        contract_name: contract_name.clone(),
                        section: *section,
                        callee,
                    },
                )
            }
        }
    }

    fn lower_const_into(&mut self, bb: RBlockId, dst: RLocalId, const_: &SConst<'db>) {
        match const_ {
            SConst::Value(value) => self.lower_sem_const_into(bb, dst, *value),
            SConst::Ref(cref) => panic!("unresolved const ref reached rMIR lowering: {cref:?}"),
        }
    }

    fn lower_sem_const_into(&mut self, bb: RBlockId, dst: RLocalId, value: SemConstId<'db>) {
        if let Some(scalar) = const_scalar_from_value(self.db, value) {
            self.push_stmt(
                bb,
                RStmt::Assign {
                    dst,
                    expr: RExpr::ConstScalar(scalar),
                },
            );
            return;
        }

        let ty = sem_const_ty(self.db, value);
        if ty == TyId::unit(self.db) {
            return;
        }

        let region = lower_const_region(self.db, value)
            .expect("aggregate constants should lower to const regions");
        let layout = region.layout(self.db);
        let const_local = self.alloc_runtime_temp(
            ty,
            RuntimeCarrier::Value(RuntimeClass::Handle {
                layout,
                kind: HandleKind::ConstValue,
                view: HandleView::Whole,
            }),
        );
        self.push_stmt(
            bb,
            RStmt::Assign {
                dst: const_local,
                expr: RExpr::ConstHandle { region, layout },
            },
        );
        let target = self
            .value_class(dst)
            .cloned()
            .expect("const destination should have a class");
        let value = self.coerce_value(bb, const_local, &target);
        self.push_stmt(
            bb,
            RStmt::Assign {
                dst,
                expr: RExpr::Use(value),
            },
        );
    }

    fn lower_aggregate_make(
        &mut self,
        bb: RBlockId,
        dst: RLocalId,
        ty: TyId<'db>,
        fields: &[SLocalId],
    ) {
        let layout = self.layout_for_ty(ty);
        let dst_class = self
            .value_class(dst)
            .cloned()
            .expect("aggregate destination must have a runtime class");
        match dst_class {
            RuntimeClass::AggregateValue { .. } => {
                let temp = self.alloc_runtime_temp(
                    ty,
                    RuntimeCarrier::Value(RuntimeClass::Handle {
                        layout,
                        kind: HandleKind::ObjectValue,
                        view: HandleView::Whole,
                    }),
                );
                self.push_stmt(
                    bb,
                    RStmt::Assign {
                        dst: temp,
                        expr: RExpr::AllocObject { layout },
                    },
                );
                for (idx, field_ty) in ty.field_types(self.db).into_iter().enumerate() {
                    let value = self.runtime_value_use(bb, fields[idx]);
                    if self.value_class(value).is_none() {
                        continue;
                    }
                    let place = RuntimePlace {
                        root: PlaceRoot::Handle(temp),
                        path: vec![PlaceElem::Field(FieldIndex(idx as u16))].into_boxed_slice(),
                    };
                    self.write_value_to_place(
                        bb,
                        place,
                        value,
                        &self.stored_class_for_ty(field_ty),
                    );
                }
                self.push_stmt(
                    bb,
                    RStmt::Assign {
                        dst,
                        expr: RExpr::Load {
                            place: RuntimePlace {
                                root: PlaceRoot::Handle(temp),
                                path: Box::default(),
                            },
                        },
                    },
                );
            }
            RuntimeClass::Handle {
                kind: HandleKind::ObjectValue,
                ..
            } => {
                self.push_stmt(
                    bb,
                    RStmt::Assign {
                        dst,
                        expr: RExpr::AllocObject { layout },
                    },
                );
                for (idx, field_ty) in ty.field_types(self.db).into_iter().enumerate() {
                    let value = self.runtime_value_use(bb, fields[idx]);
                    if self.value_class(value).is_none() {
                        continue;
                    }
                    let place = RuntimePlace {
                        root: PlaceRoot::Handle(dst),
                        path: vec![PlaceElem::Field(FieldIndex(idx as u16))].into_boxed_slice(),
                    };
                    self.write_value_to_place(
                        bb,
                        place,
                        value,
                        &self.stored_class_for_ty(field_ty),
                    );
                }
            }
            RuntimeClass::RawAddr { space, target } => {
                let [field] = fields else {
                    panic!("raw-address aggregate construction requires exactly one field");
                };
                let field = self.runtime_value_use(bb, *field);
                let value = self.coerce_scalar_word_to_raw(bb, field, space, target);
                self.push_stmt(
                    bb,
                    RStmt::Assign {
                        dst,
                        expr: RExpr::Use(value),
                    },
                );
            }
            RuntimeClass::Handle {
                layout: target_layout,
                kind: HandleKind::Provider { provider_ty, space },
                ..
            } => {
                let [field] = fields else {
                    panic!("provider aggregate construction requires exactly one field");
                };
                let field = self.runtime_value_use(bb, *field);
                let raw = self.coerce_scalar_word_to_raw(bb, field, space, None);
                let value = self.coerce_value(
                    bb,
                    raw,
                    &RuntimeClass::Handle {
                        layout: target_layout,
                        kind: HandleKind::Provider { provider_ty, space },
                        view: HandleView::Whole,
                    },
                );
                self.push_stmt(
                    bb,
                    RStmt::Assign {
                        dst,
                        expr: RExpr::Use(value),
                    },
                );
            }
            class => {
                panic!(
                    "aggregate construction requires object/provider/raw destination, found {class:?}"
                )
            }
        }
    }

    fn lower_enum_make(
        &mut self,
        bb: RBlockId,
        dst: RLocalId,
        enum_ty: TyId<'db>,
        variant: VariantIndex,
        fields: &[SLocalId],
    ) {
        let layout = self.layout_for_ty(enum_ty);
        let variant = VariantId {
            enum_layout: layout,
            index: variant.0,
        };
        let field_values = fields
            .iter()
            .map(|field| self.runtime_value_use(bb, *field))
            .collect::<Vec<_>>();
        match self
            .value_class(dst)
            .cloned()
            .expect("enum destination must have a runtime class")
        {
            RuntimeClass::AggregateValue { .. } => {
                self.push_stmt(
                    bb,
                    RStmt::Assign {
                        dst,
                        expr: RExpr::EnumMake {
                            layout,
                            variant,
                            fields: field_values.into_boxed_slice(),
                        },
                    },
                );
            }
            RuntimeClass::Handle {
                kind: HandleKind::ObjectValue,
                ..
            } => {
                self.push_stmt(
                    bb,
                    RStmt::Assign {
                        dst,
                        expr: RExpr::AllocObject { layout },
                    },
                );
                self.push_stmt(
                    bb,
                    if field_values.is_empty() {
                        RStmt::EnumSetTag { root: dst, variant }
                    } else {
                        RStmt::EnumWriteVariant {
                            root: dst,
                            variant,
                            fields: field_values.into_boxed_slice(),
                        }
                    },
                );
            }
            class => panic!(
                "enum construction requires aggregate or object-handle destination, found {class:?}"
            ),
        }
    }

    fn lower_field_like(&mut self, bb: RBlockId, dst: RLocalId, base: SLocalId, elem: PlaceElem) {
        let mut place = self.place_for_local(base);
        place.path = vec![elem].into_boxed_slice();
        let projected = self.project_place_class(&place);
        let target = self
            .value_class(dst)
            .cloned()
            .expect("field result must have class");
        if projected == target {
            self.push_stmt(
                bb,
                RStmt::Assign {
                    dst,
                    expr: RExpr::Load { place },
                },
            );
            return;
        }

        let source = self.alloc_runtime_temp(
            self.locals[dst.index()].semantic_ty,
            RuntimeCarrier::Value(projected),
        );
        self.push_stmt(
            bb,
            RStmt::Assign {
                dst: source,
                expr: RExpr::Load { place },
            },
        );
        let copied = self.coerce_value(bb, source, &target);
        self.push_stmt(
            bb,
            RStmt::Assign {
                dst,
                expr: RExpr::Use(copied),
            },
        );
    }

    fn lower_enum_tag(&mut self, bb: RBlockId, dst: RLocalId, value: SLocalId) {
        if self.effect_binding_place(value).is_some() {
            let value = self.runtime_value_use(bb, value);
            self.push_stmt(
                bb,
                RStmt::Assign {
                    dst,
                    expr: RExpr::EnumTagOfValue { value },
                },
            );
            return;
        }
        match self.local_class(value) {
            Some(RuntimeClass::Handle { .. }) => {
                self.push_stmt(
                    bb,
                    RStmt::Assign {
                        dst,
                        expr: RExpr::EnumGetTag {
                            root: self.runtime_value(value),
                        },
                    },
                );
            }
            _ => {
                self.push_stmt(
                    bb,
                    RStmt::Assign {
                        dst,
                        expr: RExpr::EnumTagOfValue {
                            value: self.runtime_value(value),
                        },
                    },
                );
            }
        }
    }

    fn lower_is_enum_variant(
        &mut self,
        bb: RBlockId,
        dst: RLocalId,
        value: SLocalId,
        variant: VariantIndex,
    ) {
        if self.effect_binding_place(value).is_some() {
            let variant = self.enum_variant_for_local(value, variant);
            let value = self.runtime_value_use(bb, value);
            self.push_stmt(
                bb,
                RStmt::Assign {
                    dst,
                    expr: RExpr::EnumIsVariant { value, variant },
                },
            );
            return;
        }
        match self.local_class(value) {
            Some(RuntimeClass::Handle { .. }) => {
                let enum_layout = self.enum_layout_for_local(value);
                let tag_class = RuntimeClass::Scalar(ScalarClass {
                    repr: match enum_layout.data(self.db) {
                        crate::runtime::Layout::Enum(layout) => layout.tag.repr,
                        _ => unreachable!(),
                    },
                    role: ScalarRole::EnumTag { enum_layout },
                });
                let tag = self.alloc_runtime_temp(
                    self.locals[value.index()].semantic_ty,
                    RuntimeCarrier::Value(tag_class),
                );
                self.lower_enum_tag(bb, tag, value);
                let expected = self.alloc_runtime_temp(
                    self.locals[value.index()].semantic_ty,
                    RuntimeCarrier::Value(
                        self.value_class(tag)
                            .cloned()
                            .expect("enum tag temp should have a class"),
                    ),
                );
                self.push_stmt(
                    bb,
                    RStmt::Assign {
                        dst: expected,
                        expr: RExpr::ConstScalar(
                            enum_tag_scalar(self.db, enum_layout, variant)
                                .expect("enum variant should lower to a tag scalar"),
                        ),
                    },
                );
                self.push_stmt(
                    bb,
                    RStmt::Assign {
                        dst,
                        expr: RExpr::Binary {
                            op: hir::hir_def::BinOp::Comp(hir::hir_def::CompBinOp::Eq),
                            lhs: tag,
                            rhs: expected,
                        },
                    },
                );
            }
            _ => {
                let variant = self.enum_variant_for_local(value, variant);
                self.push_stmt(
                    bb,
                    RStmt::Assign {
                        dst,
                        expr: RExpr::EnumIsVariant {
                            value: self.runtime_value(value),
                            variant,
                        },
                    },
                );
            }
        }
    }

    fn lower_extract_enum_field(
        &mut self,
        bb: RBlockId,
        dst: RLocalId,
        value: SLocalId,
        variant: VariantIndex,
        field: FieldIndex,
    ) {
        if self.effect_binding_place(value).is_some() {
            let variant = self.enum_variant_for_local(value, variant);
            let value = self.runtime_value_use(bb, value);
            self.push_stmt(
                bb,
                RStmt::Assign {
                    dst,
                    expr: RExpr::EnumExtract {
                        value,
                        variant,
                        field,
                    },
                },
            );
            return;
        }
        match self.local_class(value) {
            Some(RuntimeClass::Handle { .. }) => {
                let refined = self.enum_variant_ref(bb, value, variant);
                self.lower_field_like(bb, dst, refined, PlaceElem::VariantField(field));
            }
            _ => {
                let variant = self.enum_variant_for_local(value, variant);
                self.push_stmt(
                    bb,
                    RStmt::Assign {
                        dst,
                        expr: RExpr::EnumExtract {
                            value: self.runtime_value(value),
                            variant,
                            field,
                        },
                    },
                );
            }
        }
    }

    fn lower_call(
        &mut self,
        bb: RBlockId,
        callee: SemanticCalleeRef<'db>,
        args: &[SLocalId],
        effect_args: &[SEffectArg<'db>],
    ) -> RLocalId {
        let semantic = SemanticInstance::new(self.db, callee.key);
        let typed_body = semantic.key(self.db).instantiate_typed_body(self.db);
        if let Some(ret) = self.lower_extern_builtin_call(bb, semantic, args, effect_args) {
            return ret;
        }
        let (mut runtime_args, mut runtime_classes) =
            self.lower_visible_call_args(bb, &typed_body, args);
        for effect_arg in effect_args {
            if let Some((value, class)) = self.lower_effect_arg(bb, effect_arg) {
                runtime_args.push(value);
                runtime_classes.push(class);
            }
        }
        for (value, class) in self.lower_owner_effect_args(semantic) {
            runtime_args.push(value);
            runtime_classes.push(class);
        }
        let callee_key = RuntimeInstanceKey::new(
            self.db,
            crate::instance::RuntimeInstanceSource::Semantic(semantic),
            runtime_classes,
        );
        let callee = RuntimeInstance::new(self.db, callee_key);
        let ret_ty = semantic_return_ty(self.db, semantic);
        let ret_class =
            runtime_signature_for_key(self.db, semantic, callee_key.params(self.db)).ret;
        let Some(ret_class) = ret_class else {
            if !semantic_may_return_normally(self.db, semantic) {
                self.set_terminator(
                    bb,
                    RTerminator::TerminalCall {
                        callee,
                        args: runtime_args.into_boxed_slice(),
                    },
                );
                return self.alloc_runtime_temp(ret_ty, RuntimeCarrier::Erased);
            }

            let ret = self.alloc_runtime_temp(TyId::unit(self.db), RuntimeCarrier::Erased);
            self.push_stmt(
                bb,
                RStmt::Assign {
                    dst: ret,
                    expr: RExpr::Call {
                        callee,
                        args: runtime_args.into_boxed_slice(),
                    },
                },
            );
            return ret;
        };
        let ret = self.alloc_runtime_temp(ret_ty, RuntimeCarrier::Value(ret_class));
        self.push_stmt(
            bb,
            RStmt::Assign {
                dst: ret,
                expr: RExpr::Call {
                    callee,
                    args: runtime_args.into_boxed_slice(),
                },
            },
        );
        ret
    }

    fn lower_extern_builtin_call(
        &mut self,
        bb: RBlockId,
        semantic: SemanticInstance<'db>,
        args: &[SLocalId],
        effect_args: &[SEffectArg<'db>],
    ) -> Option<RLocalId> {
        let BodyOwner::Func(func) = semantic.key(self.db).owner(self.db) else {
            return None;
        };

        if let Some(ret) = self.lower_intrinsic_keccak256_call(bb, func, args) {
            return Some(ret);
        }
        if let Some(builtin) = contract_metadata_builtin(self.db, semantic) {
            let ret_ty = semantic_return_ty(self.db, semantic);
            let class = RuntimeClass::Scalar(ScalarClass {
                repr: ScalarRepr::Int {
                    bits: 256,
                    signed: false,
                },
                role: ScalarRole::Plain,
            });
            let ret = self.alloc_runtime_temp(ret_ty, RuntimeCarrier::Value(class.clone()));
            let builtin = match builtin {
                ContractMetadataBuiltin::InitCodeOffset(region) => {
                    crate::runtime::RuntimeBuiltin::CodeRegionOffset { region }
                }
                ContractMetadataBuiltin::InitCodeLen(region) => {
                    crate::runtime::RuntimeBuiltin::CodeRegionLen { region }
                }
            };
            self.push_stmt(
                bb,
                RStmt::Assign {
                    dst: ret,
                    expr: RExpr::Builtin(builtin),
                },
            );
            return Some(ret);
        }
        if let Some(ret) = self.lower_numeric_intrinsic_call(bb, semantic, args) {
            return Some(ret);
        }

        let typed_body = semantic.key(self.db).instantiate_typed_body(self.db);
        let (args, _) = self.lower_visible_call_args(bb, &typed_body, args);
        let lowered = self.lower_extern_builtin(func, &args)?;
        let ret_ty = semantic_return_ty(self.db, semantic);
        let _ = effect_args;
        Some(match lowered {
            LoweredBuiltinCall::Expr { builtin, class } => {
                let ret = class
                    .clone()
                    .map(|class| self.alloc_runtime_temp(ret_ty, RuntimeCarrier::Value(class)))
                    .unwrap_or_else(|| {
                        self.alloc_runtime_temp(TyId::unit(self.db), RuntimeCarrier::Erased)
                    });
                self.push_stmt(
                    bb,
                    RStmt::Assign {
                        dst: ret,
                        expr: RExpr::Builtin(builtin),
                    },
                );
                ret
            }
            LoweredBuiltinCall::Terminator(terminator) => {
                self.set_terminator(bb, terminator);
                self.alloc_runtime_temp(TyId::unit(self.db), RuntimeCarrier::Erased)
            }
        })
    }

    fn lower_numeric_intrinsic_call(
        &mut self,
        bb: RBlockId,
        semantic: SemanticInstance<'db>,
        args: &[SLocalId],
    ) -> Option<RLocalId> {
        let BodyOwner::Func(func) = semantic.key(self.db).owner(self.db) else {
            return None;
        };
        if func.body(self.db).is_some() {
            return None;
        }
        let name = func.name(self.db).to_opt()?.data(self.db);
        let typed_body = semantic.key(self.db).instantiate_typed_body(self.db);
        let (args, _) = self.lower_visible_call_args(bb, &typed_body, args);
        let ret_ty = semantic_return_ty(self.db, semantic);
        let ret_class = self.top_level_class_for_ty(ret_ty, AddressSpaceKind::Memory)?;
        let ret = self.alloc_runtime_temp(ret_ty, RuntimeCarrier::Value(ret_class.clone()));
        let scalar = match &ret_class {
            RuntimeClass::Scalar(scalar) => scalar.clone(),
            RuntimeClass::AggregateValue { .. }
            | RuntimeClass::Handle { .. }
            | RuntimeClass::RawAddr { .. } => return None,
        };
        let expr = match name.as_str() {
            "__saturating_add" | "__saturating_sub" | "__saturating_mul" => {
                let [lhs, rhs] = args.as_slice() else {
                    return None;
                };
                let op = match name.as_str() {
                    "__saturating_add" => SaturatingBinOp::Add,
                    "__saturating_sub" => SaturatingBinOp::Sub,
                    "__saturating_mul" => SaturatingBinOp::Mul,
                    _ => unreachable!(),
                };
                RExpr::Builtin(crate::runtime::RuntimeBuiltin::Saturating {
                    op,
                    lhs: *lhs,
                    rhs: *rhs,
                    class: scalar,
                })
            }
            "__bitcast" => {
                let [value] = args.as_slice() else {
                    return None;
                };
                RExpr::Cast {
                    value: *value,
                    to: scalar,
                }
            }
            _ => self.lower_numeric_intrinsic_expr(name.as_str(), &args, &scalar)?,
        };
        self.push_stmt(bb, RStmt::Assign { dst: ret, expr });
        Some(ret)
    }

    fn lower_numeric_intrinsic_expr(
        &self,
        name: &str,
        args: &[RLocalId],
        scalar: &ScalarClass<'db>,
    ) -> Option<RExpr<'db>> {
        let (op, _) = intrinsic_numeric_name_parts(name)?;
        Some(match op {
            "add" => {
                let [lhs, rhs] = args else { return None };
                RExpr::Builtin(crate::runtime::RuntimeBuiltin::IntrinsicArith {
                    op: IntrinsicArithBinOp::Add,
                    lhs: *lhs,
                    rhs: *rhs,
                    class: scalar.clone(),
                })
            }
            "sub" => {
                let [lhs, rhs] = args else { return None };
                RExpr::Builtin(crate::runtime::RuntimeBuiltin::IntrinsicArith {
                    op: IntrinsicArithBinOp::Sub,
                    lhs: *lhs,
                    rhs: *rhs,
                    class: scalar.clone(),
                })
            }
            "mul" => {
                let [lhs, rhs] = args else { return None };
                RExpr::Builtin(crate::runtime::RuntimeBuiltin::IntrinsicArith {
                    op: IntrinsicArithBinOp::Mul,
                    lhs: *lhs,
                    rhs: *rhs,
                    class: scalar.clone(),
                })
            }
            "div" => {
                let [lhs, rhs] = args else { return None };
                RExpr::Builtin(crate::runtime::RuntimeBuiltin::IntrinsicArith {
                    op: IntrinsicArithBinOp::Div,
                    lhs: *lhs,
                    rhs: *rhs,
                    class: scalar.clone(),
                })
            }
            "rem" => {
                let [lhs, rhs] = args else { return None };
                RExpr::Builtin(crate::runtime::RuntimeBuiltin::IntrinsicArith {
                    op: IntrinsicArithBinOp::Rem,
                    lhs: *lhs,
                    rhs: *rhs,
                    class: scalar.clone(),
                })
            }
            "pow" => {
                let [lhs, rhs] = args else { return None };
                RExpr::Binary {
                    op: BinOp::Arith(ArithBinOp::Pow),
                    lhs: *lhs,
                    rhs: *rhs,
                }
            }
            "shl" => {
                let [lhs, rhs] = args else { return None };
                RExpr::Binary {
                    op: BinOp::Arith(ArithBinOp::LShift),
                    lhs: *lhs,
                    rhs: *rhs,
                }
            }
            "shr" => {
                let [lhs, rhs] = args else { return None };
                RExpr::Binary {
                    op: BinOp::Arith(ArithBinOp::RShift),
                    lhs: *lhs,
                    rhs: *rhs,
                }
            }
            "bitand" => {
                let [lhs, rhs] = args else { return None };
                RExpr::Binary {
                    op: BinOp::Arith(ArithBinOp::BitAnd),
                    lhs: *lhs,
                    rhs: *rhs,
                }
            }
            "bitor" => {
                let [lhs, rhs] = args else { return None };
                RExpr::Binary {
                    op: BinOp::Arith(ArithBinOp::BitOr),
                    lhs: *lhs,
                    rhs: *rhs,
                }
            }
            "bitxor" => {
                let [lhs, rhs] = args else { return None };
                RExpr::Binary {
                    op: BinOp::Arith(ArithBinOp::BitXor),
                    lhs: *lhs,
                    rhs: *rhs,
                }
            }
            "eq" => {
                let [lhs, rhs] = args else { return None };
                RExpr::Binary {
                    op: BinOp::Comp(CompBinOp::Eq),
                    lhs: *lhs,
                    rhs: *rhs,
                }
            }
            "ne" => {
                let [lhs, rhs] = args else { return None };
                RExpr::Binary {
                    op: BinOp::Comp(CompBinOp::NotEq),
                    lhs: *lhs,
                    rhs: *rhs,
                }
            }
            "lt" => {
                let [lhs, rhs] = args else { return None };
                RExpr::Binary {
                    op: BinOp::Comp(CompBinOp::Lt),
                    lhs: *lhs,
                    rhs: *rhs,
                }
            }
            "le" => {
                let [lhs, rhs] = args else { return None };
                RExpr::Binary {
                    op: BinOp::Comp(CompBinOp::LtEq),
                    lhs: *lhs,
                    rhs: *rhs,
                }
            }
            "gt" => {
                let [lhs, rhs] = args else { return None };
                RExpr::Binary {
                    op: BinOp::Comp(CompBinOp::Gt),
                    lhs: *lhs,
                    rhs: *rhs,
                }
            }
            "ge" => {
                let [lhs, rhs] = args else { return None };
                RExpr::Binary {
                    op: BinOp::Comp(CompBinOp::GtEq),
                    lhs: *lhs,
                    rhs: *rhs,
                }
            }
            "bitnot" => {
                let [value] = args else { return None };
                RExpr::Unary {
                    op: UnOp::BitNot,
                    value: *value,
                }
            }
            "not" => {
                let [value] = args else { return None };
                RExpr::Unary {
                    op: UnOp::Not,
                    value: *value,
                }
            }
            "neg" => {
                let [value] = args else { return None };
                RExpr::Unary {
                    op: UnOp::Minus,
                    value: *value,
                }
            }
            _ => return None,
        })
    }

    fn lower_visible_call_args(
        &mut self,
        bb: RBlockId,
        typed_body: &hir::analysis::ty::ty_check::TypedBody<'db>,
        args: &[SLocalId],
    ) -> (Vec<RLocalId>, Vec<RuntimeClass<'db>>) {
        let mut runtime_args = Vec::with_capacity(args.len());
        let mut runtime_classes = Vec::with_capacity(args.len());
        let scope = typed_body.body().map(|body| body.scope());
        let assumptions = typed_body.assumptions();
        for (idx, arg) in args.iter().enumerate() {
            let desired = typed_body.param_binding(idx).and_then(|binding| {
                top_level_class_for_ty_in_context(
                    self.db,
                    typed_body.binding_ty(self.db, binding),
                    crate::runtime::AddressSpaceKind::Memory,
                    scope,
                    assumptions,
                )
                .map(|class| runtime_param_class(self.db, typed_body, binding, class))
            });
            let value = self.runtime_visible_arg_value(bb, *arg, desired.as_ref());
            let actual = self.value_class(value).cloned();
            let desired = desired.map(|class| {
                actual.as_ref().map_or(class.clone(), |actual| {
                    preserve_provider_space(actual, &class)
                })
            });
            let Some(value) = (match (actual, desired) {
                (None, None) => None,
                (Some(_), None) => None,
                (Some(_), Some(class)) => Some(self.coerce_value(bb, value, &class)),
                (None, Some(_)) => None,
            }) else {
                continue;
            };
            runtime_classes.push(
                self.value_class(value)
                    .cloned()
                    .expect("coerced call args should have classes"),
            );
            runtime_args.push(value);
        }
        (runtime_args, runtime_classes)
    }

    fn runtime_visible_arg_value(
        &mut self,
        bb: RBlockId,
        arg: SLocalId,
        desired: Option<&RuntimeClass<'db>>,
    ) -> RLocalId {
        let Some(desired) = desired else {
            return self.runtime_value_use(bb, arg);
        };
        if !matches!(
            desired,
            RuntimeClass::Handle {
                kind: HandleKind::Provider { .. },
                ..
            } | RuntimeClass::RawAddr { .. }
        ) {
            return self.runtime_value_use(bb, arg);
        }
        if self.effect_binding_place(arg).is_some() {
            return self.runtime_value(arg);
        }
        if let Some(source) = self.use_alias_source(arg)
            && self.effect_binding_place(source).is_some()
        {
            return self.runtime_value(source);
        }
        self.runtime_value_use(bb, arg)
    }

    fn lower_owner_effect_args(
        &self,
        semantic: SemanticInstance<'db>,
    ) -> Vec<(RLocalId, RuntimeClass<'db>)> {
        owner_effect_bindings(self.db, semantic.key(self.db).owner(self.db))
            .into_iter()
            .filter_map(|binding| {
                self.semantic_body
                    .locals
                    .iter()
                    .position(|local| {
                        local
                            .source
                            .is_some_and(|source| same_owner_effect_binding(source, binding))
                    })
                    .and_then(|idx| {
                        let local = RLocalId::from_u32(idx as u32);
                        self.value_class(local).cloned().map(|class| (local, class))
                    })
            })
            .collect()
    }

    fn lower_extern_builtin(
        &self,
        func: Func<'db>,
        args: &[RLocalId],
    ) -> Option<LoweredBuiltinCall<'db>> {
        let word = RuntimeClass::Scalar(ScalarClass {
            repr: ScalarRepr::Int {
                bits: 256,
                signed: false,
            },
            role: ScalarRole::Plain,
        });
        let builtin = |builtin, class| LoweredBuiltinCall::Expr { builtin, class };
        let matches = |path: &str| resolve_lib_func_path(self.db, func.scope(), path) == Some(func);

        Some(if matches("std::evm::mem::alloc") {
            let [size] = args else { return None };
            builtin(
                crate::runtime::RuntimeBuiltin::Malloc { size: *size },
                Some(word.clone()),
            )
        } else if matches("std::evm::ops::mload") {
            let [addr] = args else { return None };
            builtin(
                crate::runtime::RuntimeBuiltin::Mload { addr: *addr },
                Some(word.clone()),
            )
        } else if matches("std::evm::ops::mstore") {
            let [addr, value] = args else { return None };
            builtin(
                crate::runtime::RuntimeBuiltin::Mstore {
                    addr: *addr,
                    value: *value,
                },
                None,
            )
        } else if matches("std::evm::ops::mstore8") {
            let [addr, value] = args else { return None };
            builtin(
                crate::runtime::RuntimeBuiltin::Mstore8 {
                    addr: *addr,
                    value: *value,
                },
                None,
            )
        } else if matches("std::evm::ops::msize") {
            let [] = args else { return None };
            builtin(crate::runtime::RuntimeBuiltin::Msize, Some(word.clone()))
        } else if matches("std::evm::ops::sload") {
            let [slot] = args else { return None };
            builtin(
                crate::runtime::RuntimeBuiltin::Sload { slot: *slot },
                Some(word.clone()),
            )
        } else if matches("std::evm::ops::sstore") {
            let [slot, value] = args else { return None };
            builtin(
                crate::runtime::RuntimeBuiltin::Sstore {
                    slot: *slot,
                    value: *value,
                },
                None,
            )
        } else if matches("std::evm::ops::calldataload") {
            let [offset] = args else { return None };
            builtin(
                crate::runtime::RuntimeBuiltin::CallDataLoad { offset: *offset },
                Some(word.clone()),
            )
        } else if matches("std::evm::ops::calldatacopy") {
            let [dst, offset, len] = args else {
                return None;
            };
            builtin(
                crate::runtime::RuntimeBuiltin::CallDataCopy {
                    dst: *dst,
                    offset: *offset,
                    len: *len,
                },
                None,
            )
        } else if matches("std::evm::ops::calldatasize") {
            let [] = args else { return None };
            builtin(
                crate::runtime::RuntimeBuiltin::CallDataSize,
                Some(word.clone()),
            )
        } else if matches("std::evm::ops::returndatacopy") {
            let [dst, offset, len] = args else {
                return None;
            };
            builtin(
                crate::runtime::RuntimeBuiltin::ReturnDataCopy {
                    dst: *dst,
                    offset: *offset,
                    len: *len,
                },
                None,
            )
        } else if matches("std::evm::ops::returndatasize") {
            let [] = args else { return None };
            builtin(
                crate::runtime::RuntimeBuiltin::ReturnDataSize,
                Some(word.clone()),
            )
        } else if matches("std::evm::ops::codecopy") {
            let [dst, offset, len] = args else {
                return None;
            };
            builtin(
                crate::runtime::RuntimeBuiltin::CodeCopy {
                    dst: *dst,
                    offset: *offset,
                    len: *len,
                },
                None,
            )
        } else if matches("std::evm::ops::codesize") {
            let [] = args else { return None };
            builtin(crate::runtime::RuntimeBuiltin::CodeSize, Some(word.clone()))
        } else if matches("std::evm::ops::keccak256") {
            let [offset, len] = args else { return None };
            builtin(
                crate::runtime::RuntimeBuiltin::Keccak256 {
                    offset: *offset,
                    len: *len,
                },
                Some(word.clone()),
            )
        } else if matches("std::evm::ops::addmod") {
            let [lhs, rhs, modulus] = args else {
                return None;
            };
            builtin(
                crate::runtime::RuntimeBuiltin::AddMod {
                    lhs: *lhs,
                    rhs: *rhs,
                    modulus: *modulus,
                },
                Some(word.clone()),
            )
        } else if matches("std::evm::ops::mulmod") {
            let [lhs, rhs, modulus] = args else {
                return None;
            };
            builtin(
                crate::runtime::RuntimeBuiltin::MulMod {
                    lhs: *lhs,
                    rhs: *rhs,
                    modulus: *modulus,
                },
                Some(word.clone()),
            )
        } else if matches("std::evm::ops::address") {
            let [] = args else { return None };
            builtin(crate::runtime::RuntimeBuiltin::Address, Some(word.clone()))
        } else if matches("std::evm::ops::caller") {
            let [] = args else { return None };
            builtin(crate::runtime::RuntimeBuiltin::Caller, Some(word.clone()))
        } else if matches("std::evm::ops::callvalue") {
            let [] = args else { return None };
            builtin(
                crate::runtime::RuntimeBuiltin::CallValue,
                Some(word.clone()),
            )
        } else if matches("std::evm::ops::origin") {
            let [] = args else { return None };
            builtin(crate::runtime::RuntimeBuiltin::Origin, Some(word.clone()))
        } else if matches("std::evm::ops::gasprice") {
            let [] = args else { return None };
            builtin(crate::runtime::RuntimeBuiltin::GasPrice, Some(word.clone()))
        } else if matches("std::evm::ops::coinbase") {
            let [] = args else { return None };
            builtin(crate::runtime::RuntimeBuiltin::CoinBase, Some(word.clone()))
        } else if matches("std::evm::ops::timestamp") {
            let [] = args else { return None };
            builtin(
                crate::runtime::RuntimeBuiltin::Timestamp,
                Some(word.clone()),
            )
        } else if matches("std::evm::ops::number") {
            let [] = args else { return None };
            builtin(crate::runtime::RuntimeBuiltin::Number, Some(word.clone()))
        } else if matches("std::evm::ops::prevrandao") {
            let [] = args else { return None };
            builtin(
                crate::runtime::RuntimeBuiltin::PrevRandao,
                Some(word.clone()),
            )
        } else if matches("std::evm::ops::gaslimit") {
            let [] = args else { return None };
            builtin(crate::runtime::RuntimeBuiltin::GasLimit, Some(word.clone()))
        } else if matches("std::evm::ops::chainid") {
            let [] = args else { return None };
            builtin(crate::runtime::RuntimeBuiltin::ChainId, Some(word.clone()))
        } else if matches("std::evm::ops::basefee") {
            let [] = args else { return None };
            builtin(crate::runtime::RuntimeBuiltin::BaseFee, Some(word.clone()))
        } else if matches("std::evm::ops::selfbalance") {
            let [] = args else { return None };
            builtin(
                crate::runtime::RuntimeBuiltin::SelfBalance,
                Some(word.clone()),
            )
        } else if matches("std::evm::ops::blockhash") {
            let [block] = args else { return None };
            builtin(
                crate::runtime::RuntimeBuiltin::BlockHash { block: *block },
                Some(word.clone()),
            )
        } else if matches("std::evm::ops::gas") {
            let [] = args else { return None };
            builtin(crate::runtime::RuntimeBuiltin::Gas, Some(word.clone()))
        } else if matches("std::evm::ops::call") {
            let [gas, addr, value, args_offset, args_len, ret_offset, ret_len] = args else {
                return None;
            };
            builtin(
                crate::runtime::RuntimeBuiltin::Call {
                    gas: *gas,
                    addr: *addr,
                    value: *value,
                    args_offset: *args_offset,
                    args_len: *args_len,
                    ret_offset: *ret_offset,
                    ret_len: *ret_len,
                },
                Some(word.clone()),
            )
        } else if matches("std::evm::ops::staticcall") {
            let [gas, addr, args_offset, args_len, ret_offset, ret_len] = args else {
                return None;
            };
            builtin(
                crate::runtime::RuntimeBuiltin::StaticCall {
                    gas: *gas,
                    addr: *addr,
                    args_offset: *args_offset,
                    args_len: *args_len,
                    ret_offset: *ret_offset,
                    ret_len: *ret_len,
                },
                Some(word.clone()),
            )
        } else if matches("std::evm::ops::delegatecall") {
            let [gas, addr, args_offset, args_len, ret_offset, ret_len] = args else {
                return None;
            };
            builtin(
                crate::runtime::RuntimeBuiltin::DelegateCall {
                    gas: *gas,
                    addr: *addr,
                    args_offset: *args_offset,
                    args_len: *args_len,
                    ret_offset: *ret_offset,
                    ret_len: *ret_len,
                },
                Some(word.clone()),
            )
        } else if matches("std::evm::ops::create") {
            let [value, offset, len] = args else {
                return None;
            };
            builtin(
                crate::runtime::RuntimeBuiltin::Create {
                    value: *value,
                    offset: *offset,
                    len: *len,
                },
                Some(word.clone()),
            )
        } else if matches("std::evm::ops::create2") {
            let [value, offset, len, salt] = args else {
                return None;
            };
            builtin(
                crate::runtime::RuntimeBuiltin::Create2 {
                    value: *value,
                    offset: *offset,
                    len: *len,
                    salt: *salt,
                },
                Some(word.clone()),
            )
        } else if matches("std::evm::ops::log0") {
            let [offset, len] = args else { return None };
            builtin(
                crate::runtime::RuntimeBuiltin::Log0 {
                    offset: *offset,
                    len: *len,
                },
                None,
            )
        } else if matches("std::evm::ops::log1") {
            let [offset, len, topic0] = args else {
                return None;
            };
            builtin(
                crate::runtime::RuntimeBuiltin::Log1 {
                    offset: *offset,
                    len: *len,
                    topic0: *topic0,
                },
                None,
            )
        } else if matches("std::evm::ops::log2") {
            let [offset, len, topic0, topic1] = args else {
                return None;
            };
            builtin(
                crate::runtime::RuntimeBuiltin::Log2 {
                    offset: *offset,
                    len: *len,
                    topic0: *topic0,
                    topic1: *topic1,
                },
                None,
            )
        } else if matches("std::evm::ops::log3") {
            let [offset, len, topic0, topic1, topic2] = args else {
                return None;
            };
            builtin(
                crate::runtime::RuntimeBuiltin::Log3 {
                    offset: *offset,
                    len: *len,
                    topic0: *topic0,
                    topic1: *topic1,
                    topic2: *topic2,
                },
                None,
            )
        } else if matches("std::evm::ops::log4") {
            let [offset, len, topic0, topic1, topic2, topic3] = args else {
                return None;
            };
            builtin(
                crate::runtime::RuntimeBuiltin::Log4 {
                    offset: *offset,
                    len: *len,
                    topic0: *topic0,
                    topic1: *topic1,
                    topic2: *topic2,
                    topic3: *topic3,
                },
                None,
            )
        } else if matches("std::evm::ops::revert") {
            let [offset, len] = args else { return None };
            LoweredBuiltinCall::Terminator(RTerminator::Revert {
                offset: *offset,
                len: *len,
            })
        } else if matches("std::evm::ops::return_data") {
            let [offset, len] = args else { return None };
            LoweredBuiltinCall::Terminator(RTerminator::ReturnData {
                offset: *offset,
                len: *len,
            })
        } else if matches("std::evm::ops::selfdestruct") {
            let [beneficiary] = args else { return None };
            LoweredBuiltinCall::Terminator(RTerminator::SelfDestruct {
                beneficiary: *beneficiary,
            })
        } else if matches("std::evm::ops::stop") {
            let [] = args else { return None };
            LoweredBuiltinCall::Terminator(RTerminator::Stop)
        } else if matches("core::panic") || matches("core::todo") {
            let [] = args else { return None };
            LoweredBuiltinCall::Terminator(RTerminator::Trap)
        } else if matches("core::panic_with_value") {
            let [_value] = args else { return None };
            LoweredBuiltinCall::Terminator(RTerminator::Trap)
        } else {
            return None;
        })
    }

    fn lower_intrinsic_keccak256_call(
        &mut self,
        bb: RBlockId,
        func: Func<'db>,
        args: &[SLocalId],
    ) -> Option<RLocalId> {
        if resolve_lib_func_path(self.db, func.scope(), "core::intrinsic::__keccak256")
            != Some(func)
        {
            return None;
        }

        let [bytes] = args else {
            return None;
        };
        let bytes_ty = self.semantic_body.local(*bytes)?.ty;
        let layout = self.layout_for_ty(bytes_ty);
        let crate::runtime::Layout::Array(array_layout) = layout.data(self.db) else {
            panic!(
                "__keccak256 expects a byte-array argument, found {}",
                bytes_ty.pretty_print(self.db)
            );
        };

        let word_class = RuntimeClass::Scalar(ScalarClass {
            repr: ScalarRepr::Int {
                bits: 256,
                signed: false,
            },
            role: ScalarRole::Plain,
        });
        let len = self.alloc_runtime_temp(
            TyId::u256(self.db),
            RuntimeCarrier::Value(word_class.clone()),
        );
        self.push_stmt(
            bb,
            RStmt::Assign {
                dst: len,
                expr: RExpr::ConstScalar(ConstScalar::Int {
                    bits: 256,
                    signed: false,
                    words: if array_layout.len == 0 {
                        Vec::new()
                    } else {
                        let bytes = array_layout.len.to_be_bytes();
                        bytes
                            .into_iter()
                            .skip_while(|byte| *byte == 0)
                            .collect::<Vec<_>>()
                    },
                }),
            },
        );

        let value = self.runtime_value_use(bb, *bytes);
        let provider_class = provider_class_for_target_in_context(
            self.db,
            Some(bytes_ty),
            AddressSpaceKind::Memory,
            self.scope,
            self.assumptions,
        );
        let provider = match self.value_class(value) {
            Some(
                RuntimeClass::Handle {
                    kind:
                        HandleKind::Provider {
                            space: AddressSpaceKind::Memory,
                            ..
                        },
                    ..
                }
                | RuntimeClass::RawAddr {
                    space: AddressSpaceKind::Memory,
                    ..
                },
            ) => value,
            Some(_) => self.coerce_value(bb, value, &provider_class),
            None => panic!(
                "__keccak256 argument should have a runtime class: key={:?}; local={bytes:?}",
                self.key
            ),
        };
        let offset = self.coerce_value(
            bb,
            provider,
            &RuntimeClass::RawAddr {
                space: AddressSpaceKind::Memory,
                target: Some(layout),
            },
        );
        let ret = self.alloc_runtime_temp(TyId::u256(self.db), RuntimeCarrier::Value(word_class));
        self.push_stmt(
            bb,
            RStmt::Assign {
                dst: ret,
                expr: RExpr::Builtin(crate::runtime::RuntimeBuiltin::Keccak256 { offset, len }),
            },
        );
        Some(ret)
    }

    fn lower_effect_arg(
        &mut self,
        bb: RBlockId,
        arg: &SEffectArg<'db>,
    ) -> Option<(RLocalId, RuntimeClass<'db>)> {
        match arg.pass_mode {
            EffectPassMode::ByValue | EffectPassMode::Unknown => match &arg.arg {
                SEffectArgValue::Value(value) => {
                    let value = if arg.provider.is_none() && arg.target_ty.is_none() {
                        self.runtime_value_use(bb, *value)
                    } else {
                        self.runtime_value(*value)
                    };
                    if let Some(class) = self.value_class(value).cloned() {
                        return Some((value, class));
                    }
                    if arg.provider.is_none() && arg.target_ty.is_none() {
                        return None;
                    }
                    let class = provider_class_for_target_in_context(
                        self.db,
                        arg.target_ty,
                        effect_arg_address_space(arg),
                        self.scope,
                        self.assumptions,
                    );
                    let placeholder = self.alloc_runtime_temp(
                        self.locals[value.index()].semantic_ty,
                        RuntimeCarrier::Value(class.clone()),
                    );
                    self.push_stmt(
                        bb,
                        RStmt::Assign {
                            dst: placeholder,
                            expr: RExpr::Placeholder {
                                class: class.clone(),
                            },
                        },
                    );
                    Some((placeholder, class))
                }
                SEffectArgValue::Place(place) => {
                    let class = provider_class_for_target_in_context(
                        self.db,
                        arg.target_ty,
                        effect_arg_address_space(arg),
                        self.scope,
                        self.assumptions,
                    );
                    let temp = self.alloc_runtime_temp(
                        arg.target_ty.unwrap_or_else(|| TyId::unit(self.db)),
                        RuntimeCarrier::Value(class.clone()),
                    );
                    if let Some(place) = self.try_lower_place(bb, place) {
                        self.push_stmt(
                            bb,
                            RStmt::Assign {
                                dst: temp,
                                expr: RExpr::AddrOf { place },
                            },
                        );
                    } else {
                        assert!(
                            place.path.is_empty(),
                            "erased capability place effect args cannot have projections"
                        );
                        self.push_stmt(
                            bb,
                            RStmt::Assign {
                                dst: temp,
                                expr: RExpr::Placeholder {
                                    class: class.clone(),
                                },
                            },
                        );
                    }
                    Some((temp, class))
                }
            },
            EffectPassMode::ByPlace | EffectPassMode::ByTempPlace => {
                let target_ty = arg.target_ty.unwrap_or_else(|| TyId::unit(self.db));
                let space = effect_arg_address_space(arg);
                let source = match (&arg.arg, arg.pass_mode) {
                    (SEffectArgValue::Place(place), _) => self.try_lower_place(bb, place),
                    (SEffectArgValue::Value(value), EffectPassMode::ByTempPlace) => {
                        let storage_class = self
                            .top_level_class_for_ty(target_ty, space)
                            .unwrap_or(RuntimeClass::RawAddr {
                                space,
                                target: None,
                            });
                        let temp = self.alloc_runtime_temp(
                            target_ty,
                            RuntimeCarrier::Value(storage_class.clone()),
                        );
                        let value = self.runtime_value_use(bb, *value);
                        let coerced = self.coerce_value(bb, value, &storage_class);
                        self.push_stmt(
                            bb,
                            RStmt::Assign {
                                dst: temp,
                                expr: RExpr::Use(coerced),
                            },
                        );
                        RuntimePlace {
                            root: match storage_class {
                                RuntimeClass::Handle { .. } => PlaceRoot::Handle(temp),
                                _ => PlaceRoot::Slot(temp),
                            },
                            path: Box::default(),
                        }
                        .into()
                    }
                    _ => panic!("invalid effect arg lowering mode"),
                };
                let class = provider_class_for_target_in_context(
                    self.db,
                    Some(target_ty),
                    space,
                    self.scope,
                    self.assumptions,
                );
                let temp = self.alloc_runtime_temp(target_ty, RuntimeCarrier::Value(class.clone()));
                if let Some(source) = source {
                    self.push_stmt(
                        bb,
                        RStmt::Assign {
                            dst: temp,
                            expr: RExpr::AddrOf { place: source },
                        },
                    );
                } else {
                    let SEffectArgValue::Place(place) = &arg.arg else {
                        unreachable!();
                    };
                    assert!(
                        place.path.is_empty(),
                        "erased capability place effect args cannot have projections"
                    );
                    self.push_stmt(
                        bb,
                        RStmt::Assign {
                            dst: temp,
                            expr: RExpr::Placeholder {
                                class: class.clone(),
                            },
                        },
                    );
                }
                Some((temp, class))
            }
        }
    }

    fn lower_terminator(
        &mut self,
        bb: RBlockId,
        terminator: &STerminator<'db>,
    ) -> RTerminator<'db> {
        match terminator {
            STerminator::Goto(block) => RTerminator::Goto(self.runtime_block(*block)),
            STerminator::Branch {
                cond,
                then_bb,
                else_bb,
            } => RTerminator::Branch {
                cond: self.runtime_value_use(bb, *cond),
                then_bb: self.runtime_block(*then_bb),
                else_bb: self.runtime_block(*else_bb),
            },
            STerminator::MatchEnum {
                value,
                enum_ty,
                cases,
                default,
            } => {
                let enum_layout = self.layout_for_ty(*enum_ty);
                let tag_class = RuntimeClass::Scalar(ScalarClass {
                    repr: match enum_layout.data(self.db) {
                        crate::runtime::Layout::Enum(layout) => layout.tag.repr,
                        _ => unreachable!(),
                    },
                    role: ScalarRole::EnumTag { enum_layout },
                });
                let tag = self.alloc_runtime_temp(*enum_ty, RuntimeCarrier::Value(tag_class));
                self.lower_enum_tag(bb, tag, *value);
                RTerminator::MatchEnumTag {
                    tag,
                    enum_layout,
                    cases: cases
                        .iter()
                        .map(|(variant, block)| {
                            (
                                VariantId {
                                    enum_layout,
                                    index: variant.0,
                                },
                                self.runtime_block(*block),
                            )
                        })
                        .collect(),
                    default: default.map(|block| self.runtime_block(block)),
                }
            }
            STerminator::Return(value) => {
                let ret_class = self.ret_class.clone();
                RTerminator::Return(match ret_class {
                    Some(class) => value.map(|value| {
                        let value = self.runtime_value_use(bb, value);
                        self.coerce_value(bb, value, &class)
                    }),
                    None => None,
                })
            }
        }
    }

    fn write_value_to_place(
        &mut self,
        bb: RBlockId,
        dst: RuntimePlace<'db>,
        src: RLocalId,
        target: &RuntimeClass<'db>,
    ) {
        if self.class_is_runtime_zst(target) {
            return;
        }
        match target {
            RuntimeClass::Scalar(_) | RuntimeClass::RawAddr { .. } => {
                let src = self.coerce_value(bb, src, target);
                self.push_stmt(bb, RStmt::Store { dst, src });
            }
            RuntimeClass::AggregateValue { .. } | RuntimeClass::Handle { .. } => {
                let src = self.coerce_value(bb, src, target);
                self.push_stmt(bb, RStmt::CopyInto { dst, src });
            }
        }
    }

    fn coerce_value(
        &mut self,
        bb: RBlockId,
        src: RLocalId,
        target: &RuntimeClass<'db>,
    ) -> RLocalId {
        let source = self
            .value_class(src)
            .cloned()
            .unwrap_or_else(|| {
                panic!(
                    "cannot coerce erased value {src:?} to {target:?}; owner={:?}; src_ty={}; locals={:?}",
                    self.key
                        .semantic(self.db)
                        .map(|semantic| semantic.key(self.db).owner(self.db)),
                    self.locals[src.index()].semantic_ty.pretty_print(self.db),
                    self.locals,
                )
            });
        if &source == target {
            return src;
        }

        match (source, target.clone()) {
            (
                RuntimeClass::Handle {
                    layout,
                    kind: HandleKind::ObjectValue,
                    view: HandleView::Whole,
                },
                RuntimeClass::Handle {
                    layout: target_layout,
                    kind: HandleKind::Provider { provider_ty, space },
                    view: HandleView::Whole,
                },
            ) if layout == target_layout => {
                let temp = self.alloc_runtime_temp(
                    self.locals[src.index()].semantic_ty,
                    RuntimeCarrier::Value(RuntimeClass::Handle {
                        layout,
                        kind: HandleKind::Provider { provider_ty, space },
                        view: HandleView::Whole,
                    }),
                );
                self.push_stmt(
                    bb,
                    RStmt::Assign {
                        dst: temp,
                        expr: RExpr::AddrOf {
                            place: RuntimePlace {
                                root: PlaceRoot::Handle(src),
                                path: Box::default(),
                            },
                        },
                    },
                );
                temp
            }
            (
                RuntimeClass::AggregateValue { layout },
                RuntimeClass::Handle {
                    layout: target_layout,
                    kind: HandleKind::Provider { provider_ty, space },
                    view: HandleView::Whole,
                },
            ) if layout == target_layout && space == AddressSpaceKind::Memory => {
                let temp = self.alloc_runtime_temp(
                    self.locals[src.index()].semantic_ty,
                    RuntimeCarrier::Value(RuntimeClass::Handle {
                        layout,
                        kind: HandleKind::Provider { provider_ty, space },
                        view: HandleView::Whole,
                    }),
                );
                self.push_stmt(
                    bb,
                    RStmt::Assign {
                        dst: temp,
                        expr: RExpr::AddrOf {
                            place: RuntimePlace {
                                root: PlaceRoot::Slot(src),
                                path: Box::default(),
                            },
                        },
                    },
                );
                temp
            }
            (
                RuntimeClass::Handle {
                    layout,
                    kind: _,
                    view: _,
                },
                RuntimeClass::AggregateValue {
                    layout: target_layout,
                },
            ) if layout == target_layout => {
                let temp = self.alloc_runtime_temp(
                    self.locals[src.index()].semantic_ty,
                    RuntimeCarrier::Value(RuntimeClass::AggregateValue { layout }),
                );
                self.push_stmt(
                    bb,
                    RStmt::Assign {
                        dst: temp,
                        expr: RExpr::Load {
                            place: RuntimePlace {
                                root: PlaceRoot::Handle(src),
                                path: Box::default(),
                            },
                        },
                    },
                );
                temp
            }
            (
                RuntimeClass::AggregateValue { layout },
                RuntimeClass::Handle {
                    layout: target_layout,
                    kind: HandleKind::ObjectValue,
                    view: HandleView::Whole,
                },
            ) if layout == target_layout => {
                let temp = self.alloc_runtime_temp(
                    self.locals[src.index()].semantic_ty,
                    RuntimeCarrier::Value(RuntimeClass::Handle {
                        layout,
                        kind: HandleKind::ObjectValue,
                        view: HandleView::Whole,
                    }),
                );
                self.push_stmt(
                    bb,
                    RStmt::Assign {
                        dst: temp,
                        expr: RExpr::MaterializeToObject { src },
                    },
                );
                temp
            }
            (
                RuntimeClass::Handle {
                    layout,
                    kind: HandleKind::ConstValue,
                    view: HandleView::Whole,
                },
                RuntimeClass::Handle {
                    layout: target_layout,
                    kind: HandleKind::ObjectValue,
                    view: HandleView::Whole,
                },
            ) if layout == target_layout => {
                let temp = self.alloc_runtime_temp(
                    self.locals[src.index()].semantic_ty,
                    RuntimeCarrier::Value(RuntimeClass::Handle {
                        layout,
                        kind: HandleKind::ObjectValue,
                        view: HandleView::Whole,
                    }),
                );
                self.push_stmt(
                    bb,
                    RStmt::Assign {
                        dst: temp,
                        expr: RExpr::MaterializeToObject { src },
                    },
                );
                temp
            }
            (
                RuntimeClass::RawAddr { space, .. },
                RuntimeClass::Handle {
                    layout,
                    kind:
                        HandleKind::Provider {
                            provider_ty,
                            space: provider_space,
                        },
                    ..
                },
            ) if space == provider_space => {
                let temp = self.alloc_runtime_temp(
                    self.locals[src.index()].semantic_ty,
                    RuntimeCarrier::Value(RuntimeClass::Handle {
                        layout,
                        kind: HandleKind::Provider { provider_ty, space },
                        view: HandleView::Whole,
                    }),
                );
                self.push_stmt(
                    bb,
                    RStmt::Assign {
                        dst: temp,
                        expr: RExpr::ProviderFromRaw {
                            raw: src,
                            provider_ty,
                            space,
                            layout,
                        },
                    },
                );
                temp
            }
            (
                RuntimeClass::AggregateValue { layout },
                RuntimeClass::Handle {
                    layout: target_layout,
                    kind: HandleKind::Provider { provider_ty, space },
                    view: HandleView::Whole,
                },
            ) if layout == target_layout => {
                let object = self.coerce_value(
                    bb,
                    src,
                    &RuntimeClass::Handle {
                        layout,
                        kind: HandleKind::ObjectValue,
                        view: HandleView::Whole,
                    },
                );
                self.coerce_value(
                    bb,
                    object,
                    &RuntimeClass::Handle {
                        layout,
                        kind: HandleKind::Provider { provider_ty, space },
                        view: HandleView::Whole,
                    },
                )
            }
            (
                RuntimeClass::Handle {
                    kind: HandleKind::Provider { .. },
                    ..
                },
                RuntimeClass::RawAddr { .. },
            ) => {
                let temp = self.alloc_runtime_temp(
                    self.locals[src.index()].semantic_ty,
                    RuntimeCarrier::Value(target.clone()),
                );
                self.push_stmt(
                    bb,
                    RStmt::Assign {
                        dst: temp,
                        expr: RExpr::ProviderToRaw { value: src },
                    },
                );
                temp
            }
            (
                RuntimeClass::RawAddr { .. },
                RuntimeClass::Scalar(ScalarClass {
                    repr:
                        ScalarRepr::Int {
                            bits: 256,
                            signed: false,
                        },
                    ..
                }),
            ) => {
                let temp = self.alloc_runtime_temp(
                    self.locals[src.index()].semantic_ty,
                    RuntimeCarrier::Value(target.clone()),
                );
                self.push_stmt(
                    bb,
                    RStmt::Assign {
                        dst: temp,
                        expr: RExpr::Cast {
                            value: src,
                            to: match target {
                                RuntimeClass::Scalar(scalar) => scalar.clone(),
                                _ => unreachable!(),
                            },
                        },
                    },
                );
                temp
            }
            (
                RuntimeClass::Scalar(ScalarClass {
                    repr:
                        ScalarRepr::Int {
                            bits: 256,
                            signed: false,
                        },
                    role: ScalarRole::Plain,
                }),
                RuntimeClass::RawAddr { space, target },
            ) => self.coerce_scalar_word_to_raw(bb, src, space, target),
            (_, RuntimeClass::Handle { layout, kind, .. })
                if matches!(kind, HandleKind::ObjectValue) =>
            {
                let temp = self.alloc_runtime_temp(
                    self.locals[src.index()].semantic_ty,
                    RuntimeCarrier::Value(RuntimeClass::Handle {
                        layout,
                        kind: kind.clone(),
                        view: HandleView::Whole,
                    }),
                );
                self.push_stmt(
                    bb,
                    RStmt::Assign {
                        dst: temp,
                        expr: RExpr::AllocObject { layout },
                    },
                );
                self.push_stmt(
                    bb,
                    RStmt::CopyInto {
                        dst: RuntimePlace {
                            root: PlaceRoot::Handle(temp),
                            path: Box::default(),
                        },
                        src,
                    },
                );
                temp
            }
            (source, target) => {
                panic!(
                    "unsupported runtime class coercion in {:?} owner={:?} from {source:?} to {target:?}; src={src:?}; src_ty={}; locals={:?}",
                    self.key.source(self.db),
                    self.key
                        .semantic(self.db)
                        .map(|semantic| semantic.key(self.db).owner(self.db)),
                    self.locals[src.index()].semantic_ty.pretty_print(self.db),
                    self.locals,
                )
            }
        }
    }

    fn coerce_scalar_word_to_raw(
        &mut self,
        bb: RBlockId,
        src: RLocalId,
        space: crate::runtime::AddressSpaceKind,
        target: Option<LayoutId<'db>>,
    ) -> RLocalId {
        let temp = self.alloc_runtime_temp(
            self.locals[src.index()].semantic_ty,
            RuntimeCarrier::Value(RuntimeClass::RawAddr { space, target }),
        );
        self.push_stmt(
            bb,
            RStmt::Assign {
                dst: temp,
                expr: RExpr::WordToRawAddr {
                    value: src,
                    space,
                    target,
                },
            },
        );
        temp
    }

    fn effect_binding_value_class(&self, local: SLocalId) -> Option<RuntimeClass<'db>> {
        let value_ty = self.place_like_local_value_ty(local)?;
        Some(self.stored_class_for_ty(value_ty))
    }

    fn binding_place_value_class(&self, local: SLocalId) -> Option<RuntimeClass<'db>> {
        let value_ty = self.binding_place_value_ty(local)?;
        Some(self.stored_class_for_ty(value_ty))
    }

    fn place_like_local_value_ty(&self, local: SLocalId) -> Option<TyId<'db>> {
        let local_data = self.semantic_body.local(local)?;
        if let Some(binding) = local_data.source
            && is_effect_binding(binding)
        {
            let carrier_class = self.local_class(local)?;
            let top_level = self
                .top_level_class_for_ty(local_data.ty, crate::runtime::AddressSpaceKind::Memory)?;
            if &top_level != carrier_class {
                return Some(local_data.ty);
            }
        }
        if let Some((_, inner)) = local_data.ty.as_capability(self.db) {
            return Some(inner);
        }
        None
    }

    fn binding_place_value_ty(&self, local: SLocalId) -> Option<TyId<'db>> {
        let local_data = self.semantic_body.local(local)?;
        let binding = local_data.source?;
        if is_effect_binding(binding) {
            let carrier_class = self.local_class(local)?;
            let top_level = self
                .top_level_class_for_ty(local_data.ty, crate::runtime::AddressSpaceKind::Memory)?;
            if &top_level != carrier_class {
                return Some(local_data.ty);
            }
        }
        if let Some((_, inner)) = local_data.ty.as_capability(self.db) {
            return Some(inner);
        }
        None
    }

    fn effect_binding_place(&self, local: SLocalId) -> Option<RuntimePlace<'db>> {
        let class = self.effect_binding_value_class(local)?;
        self.place_like_root(local, class)
    }

    fn binding_place(&self, local: SLocalId) -> Option<RuntimePlace<'db>> {
        let class = self.binding_place_value_class(local)?;
        self.place_like_root(local, class)
    }

    fn place_like_root(
        &self,
        local: SLocalId,
        class: RuntimeClass<'db>,
    ) -> Option<RuntimePlace<'db>> {
        match self.local_class(local)?.clone() {
            RuntimeClass::RawAddr { space, .. } => RuntimePlace {
                root: PlaceRoot::Ptr {
                    addr: self.runtime_value(local),
                    space,
                    class,
                },
                path: Box::default(),
            },
            RuntimeClass::Handle {
                kind: HandleKind::Provider { space, .. },
                ..
            } if space != crate::runtime::AddressSpaceKind::Memory => RuntimePlace {
                root: PlaceRoot::Ptr {
                    addr: self.runtime_value(local),
                    space,
                    class,
                },
                path: Box::default(),
            },
            RuntimeClass::Handle { .. } => RuntimePlace {
                root: PlaceRoot::Handle(self.runtime_value(local)),
                path: Box::default(),
            },
            RuntimeClass::Scalar(_) | RuntimeClass::AggregateValue { .. } => return None,
        }
        .into()
    }

    fn try_place_for_local(&self, local: SLocalId) -> Option<RuntimePlace<'db>> {
        self.binding_place(local).or_else(|| {
            let root = match self.local_class(local)? {
                RuntimeClass::Handle { .. } => PlaceRoot::Handle(self.runtime_value(local)),
                RuntimeClass::Scalar(_)
                | RuntimeClass::AggregateValue { .. }
                | RuntimeClass::RawAddr { .. } => PlaceRoot::Slot(self.runtime_value(local)),
            };
            Some(RuntimePlace {
                root,
                path: Box::default(),
            })
        })
    }

    fn place_for_local(&self, local: SLocalId) -> RuntimePlace<'db> {
        self.try_place_for_local(local).unwrap_or_else(|| {
            panic!(
                "cannot lower erased local as a runtime place root: local={local:?}; ty={}",
                self.locals[local.index()].semantic_ty.pretty_print(self.db),
            )
        })
    }

    fn try_lower_place(&mut self, bb: RBlockId, place: &SPlace) -> Option<RuntimePlace<'db>> {
        let mut runtime_place = self.try_place_for_local(place.local)?;
        runtime_place.path = place
            .path
            .iter()
            .map(|elem| match elem {
                SPlaceElem::Field(field) => PlaceElem::Field(*field),
                SPlaceElem::Index(index) => PlaceElem::Index(self.runtime_value_use(bb, *index)),
            })
            .collect();
        Some(runtime_place)
    }

    fn runtime_value_use(&mut self, bb: RBlockId, local: SLocalId) -> RLocalId {
        let Some(place) = self.effect_binding_place(local) else {
            return self.runtime_value(local);
        };
        let class = self
            .effect_binding_value_class(local)
            .expect("effect-bound place should have a value class");
        let temp = self.alloc_runtime_temp(
            self.locals[local.index()].semantic_ty,
            RuntimeCarrier::Value(class),
        );
        self.push_stmt(
            bb,
            RStmt::Assign {
                dst: temp,
                expr: RExpr::Load { place },
            },
        );
        temp
    }

    fn use_alias_source(&self, local: SLocalId) -> Option<SLocalId> {
        self.semantic_body.blocks.iter().find_map(|block| {
            block.stmts.iter().find_map(|stmt| match stmt {
                SStmt::Assign {
                    dst,
                    expr: SExpr::Use(src),
                } if *dst == local => Some(*src),
                SStmt::Assign { .. } | SStmt::Store { .. } => None,
            })
        })
    }

    fn lower_place(&mut self, bb: RBlockId, place: &SPlace) -> RuntimePlace<'db> {
        self.try_lower_place(bb, place).unwrap_or_else(|| {
            panic!(
                "cannot lower erased place root: local={:?}; ty={}",
                place.local,
                self.locals[place.local.index()]
                    .semantic_ty
                    .pretty_print(self.db),
            )
        })
    }

    fn project_place_class(&self, place: &RuntimePlace<'db>) -> RuntimeClass<'db> {
        let mut current = match &place.root {
            PlaceRoot::Slot(local) | PlaceRoot::Handle(local) => self
                .value_class(*local)
                .cloned()
                .expect("projected places should have runtime classes"),
            PlaceRoot::Ptr { class, .. } => class.clone(),
        };
        for elem in place.path.iter() {
            current = match elem {
                PlaceElem::Field(field) => project_field_class(self.db, current, *field),
                PlaceElem::Index(_) => project_index_class(self.db, current),
                PlaceElem::VariantField(field) => {
                    project_variant_field_class(self.db, current, *field)
                }
            };
        }
        current
    }

    fn class_is_runtime_zst(&self, class: &RuntimeClass<'db>) -> bool {
        match class {
            RuntimeClass::Scalar(_)
            | RuntimeClass::Handle { .. }
            | RuntimeClass::RawAddr { .. } => false,
            RuntimeClass::AggregateValue { layout } => self.layout_is_runtime_zst(*layout),
        }
    }

    fn layout_is_runtime_zst(&self, layout: LayoutId<'db>) -> bool {
        match layout.data(self.db) {
            crate::runtime::Layout::Struct(layout) => layout
                .fields
                .iter()
                .all(|field| self.class_is_runtime_zst(field)),
            crate::runtime::Layout::Array(layout) => {
                layout.len == 0 || self.class_is_runtime_zst(&layout.elem)
            }
            crate::runtime::Layout::Enum(_) => false,
        }
    }

    fn enum_variant_for_local(&self, value: SLocalId, variant: VariantIndex) -> VariantId<'db> {
        VariantId {
            enum_layout: self.enum_layout_for_local(value),
            index: variant.0,
        }
    }

    fn enum_layout_for_local(&self, value: SLocalId) -> LayoutId<'db> {
        let class = self
            .effect_binding_value_class(value)
            .or_else(|| self.local_class(value).cloned())
            .expect("enum value should have a runtime class");
        match class {
            RuntimeClass::AggregateValue { layout } | RuntimeClass::Handle { layout, .. } => layout,
            class => {
                panic!("enum values should lower as aggregate values or handles, found {class:?}")
            }
        }
    }

    fn enum_variant_ref(
        &mut self,
        bb: RBlockId,
        value: SLocalId,
        variant: VariantIndex,
    ) -> SLocalId {
        let class = self
            .local_class(value)
            .cloned()
            .expect("enum handle should have a class");
        let RuntimeClass::Handle { layout, kind, .. } = class else {
            panic!("enum variant refs require handle-form enums");
        };
        let variant_id = VariantId {
            enum_layout: layout,
            index: variant.0,
        };
        let temp = self.alloc_runtime_temp(
            self.locals[value.index()].semantic_ty,
            RuntimeCarrier::Value(RuntimeClass::Handle {
                layout,
                kind,
                view: HandleView::EnumVariant(variant_id),
            }),
        );
        self.push_stmt(
            bb,
            RStmt::Assign {
                dst: temp,
                expr: RExpr::EnumAssertVariantRef {
                    root: self.runtime_value(value),
                    variant: variant_id,
                },
            },
        );
        SLocalId::from_u32(temp.as_u32())
    }

    fn alloc_runtime_temp(
        &mut self,
        semantic_ty: TyId<'db>,
        carrier: RuntimeCarrier<'db>,
    ) -> RLocalId {
        let id = RLocalId::from_u32(self.locals.len() as u32);
        let slot = match &carrier {
            RuntimeCarrier::Erased | RuntimeCarrier::Value(RuntimeClass::Handle { .. }) => {
                LocalSlotKind::None
            }
            RuntimeCarrier::Value(class) => LocalSlotKind::Slot(class.clone()),
        };
        self.locals.push(RLocal {
            semantic_ty,
            carrier,
            slot,
        });
        id
    }

    fn push_stmt(&mut self, bb: RBlockId, stmt: RStmt<'db>) {
        if !self.terminated_blocks[bb.index()] {
            self.blocks[bb.index()].stmts.push(stmt);
        }
    }

    fn runtime_value(&self, local: SLocalId) -> RLocalId {
        RLocalId::from_u32(local.as_u32())
    }

    fn runtime_block(&self, block: SBlockId) -> RBlockId {
        RBlockId::from_u32(block.as_u32())
    }

    fn local_class(&self, local: SLocalId) -> Option<&RuntimeClass<'db>> {
        self.value_class(self.runtime_value(local))
    }

    fn value_class(&self, local: RLocalId) -> Option<&RuntimeClass<'db>> {
        match self.locals.get(local.index())?.carrier {
            RuntimeCarrier::Erased => None,
            RuntimeCarrier::Value(ref class) => Some(class),
        }
    }
}

fn preserve_provider_space<'db>(
    actual: &RuntimeClass<'db>,
    desired: &RuntimeClass<'db>,
) -> RuntimeClass<'db> {
    match (actual, desired) {
        (
            RuntimeClass::Handle {
                layout: actual_layout,
                kind:
                    HandleKind::Provider {
                        provider_ty: actual_provider_ty,
                        space: actual_space,
                    },
                view: HandleView::Whole,
            },
            RuntimeClass::Handle {
                layout: desired_layout,
                kind:
                    HandleKind::Provider {
                        provider_ty: desired_provider_ty,
                        ..
                    },
                view: HandleView::Whole,
            },
        ) if actual_layout == desired_layout && actual_provider_ty == desired_provider_ty => {
            RuntimeClass::Handle {
                layout: *actual_layout,
                kind: HandleKind::Provider {
                    provider_ty: *actual_provider_ty,
                    space: *actual_space,
                },
                view: HandleView::Whole,
            }
        }
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

fn intrinsic_numeric_name_parts(name: &str) -> Option<(&str, &str)> {
    let op = name.strip_prefix("__")?;
    [
        "_u8", "_u16", "_u32", "_u64", "_u128", "_u256", "_usize", "_i8", "_i16", "_i32", "_i64",
        "_i128", "_i256", "_isize", "_bool",
    ]
    .iter()
    .find_map(|suffix| op.strip_suffix(suffix).map(|prefix| (prefix, *suffix)))
}
