use cranelift_entity::EntityRef;
use hir::analysis::{
    semantic::{
        FieldIndex, SBlockId, SConst, SEffectArg, SEffectArgValue, SExpr, SLocalId, SPlace,
        SPlaceElem, SStmt, STerminator, SemConstId, SemanticBody, SemanticCalleeRef,
        SemanticInstance, VariantIndex, ctfe::canonicalize_semantic_consts, sem_const_ty,
    },
    ty::{ty_check::EffectPassMode, ty_def::TyId},
};

use crate::{
    db::MirDb,
    instance::{RuntimeInstance, RuntimeInstanceKey},
    runtime::{
        HandleKind, HandleView, LayoutId, LocalSlotKind, PlaceElem, PlaceRoot, RBlock, RBlockId,
        RExpr, RLocal, RLocalId, RStmt, RTerminator, RuntimeBody, RuntimeCarrier, RuntimeClass,
        RuntimePlace, RuntimeSignature, ScalarClass, ScalarRole, VariantId,
    },
};

use super::{
    class::{
        infer_local_carriers, provider_class_for_target, runtime_param_class,
        runtime_signature_for_key, semantic_return_ty, stored_class_for_ty, top_level_class_for_ty,
    },
    consts::{const_scalar_from_value, enum_tag_scalar, lower_const_region},
    layout::layout_for_ty,
    place::{
        effect_arg_address_space, project_field_class, project_index_class,
        project_variant_field_class,
    },
};

pub fn lower_to_rmir<'db>(db: &'db dyn MirDb, instance: RuntimeInstance<'db>) -> RuntimeBody<'db> {
    let key = instance.key(db);
    let semantic = key.semantic(db);
    let semantic_body = canonicalize_semantic_consts(db, semantic);
    let local_carriers = infer_local_carriers(db, &semantic_body, key.params(db));
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
    pub(super) locals: Vec<RLocal<'db>>,
    pub(super) blocks: Vec<RBlock<'db>>,
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
            locals,
            blocks,
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

    fn lower_blocks(&mut self) {
        self.blocks = (0..self.semantic_body.blocks.len())
            .map(|_| RBlock {
                stmts: Vec::new(),
                terminator: RTerminator::Return(None),
            })
            .collect();
        let blocks = self.semantic_body.blocks.clone();
        for (idx, block) in blocks.iter().enumerate() {
            let bb = RBlockId::from_u32(idx as u32);
            for stmt in &block.stmts {
                self.lower_stmt(bb, stmt);
            }
            self.blocks[bb.index()].terminator = self.lower_terminator(bb, &block.terminator);
        }
    }

    fn lower_stmt(&mut self, bb: RBlockId, stmt: &SStmt<'db>) {
        match stmt {
            SStmt::Assign { dst, expr } => self.lower_assign(bb, *dst, expr),
            SStmt::Store { dst, src } => {
                let place = self.lower_place(dst);
                let target = self.project_place_class(&place);
                self.write_value_to_place(bb, place, self.runtime_value(*src), &target);
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
                let _ = self.lower_call(bb, *callee, args, effect_args, None);
            }
            return;
        };

        match expr {
            SExpr::Use(src) => {
                let value = self.coerce_value(bb, self.runtime_value(*src), &dst_class);
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
                self.push_stmt(
                    bb,
                    RStmt::Assign {
                        dst,
                        expr: RExpr::Unary {
                            op: *op,
                            value: self.runtime_value(*value),
                        },
                    },
                );
            }
            SExpr::Binary { op, lhs, rhs } => {
                self.push_stmt(
                    bb,
                    RStmt::Assign {
                        dst,
                        expr: RExpr::Binary {
                            op: *op,
                            lhs: self.runtime_value(*lhs),
                            rhs: self.runtime_value(*rhs),
                        },
                    },
                );
            }
            SExpr::Cast { value, .. } => {
                let RuntimeClass::Scalar(to) = dst_class else {
                    panic!("casts must lower to scalar carriers");
                };
                self.push_stmt(
                    bb,
                    RStmt::Assign {
                        dst,
                        expr: RExpr::Cast {
                            value: self.runtime_value(*value),
                            to,
                        },
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
                self.lower_field_like(bb, dst, *base, PlaceElem::Index(self.runtime_value(*index)))
            }
            SExpr::Borrow { place, .. } => {
                self.push_stmt(
                    bb,
                    RStmt::Assign {
                        dst,
                        expr: RExpr::AddrOf {
                            place: self.lower_place(place),
                        },
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
            SExpr::Call {
                callee,
                args,
                effect_args,
            } => {
                let value = self.lower_call(bb, *callee, args, effect_args, Some(&dst_class));
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
        let layout = layout_for_ty(self.db, ty);
        self.push_stmt(
            bb,
            RStmt::Assign {
                dst,
                expr: RExpr::AllocObject { layout },
            },
        );
        for (idx, field_ty) in ty.field_types(self.db).into_iter().enumerate() {
            let place = RuntimePlace {
                root: PlaceRoot::Handle(dst),
                path: vec![PlaceElem::Field(FieldIndex(idx as u16))].into_boxed_slice(),
            };
            self.write_value_to_place(
                bb,
                place,
                self.runtime_value(fields[idx]),
                &stored_class_for_ty(self.db, field_ty),
            );
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
        let layout = layout_for_ty(self.db, enum_ty);
        let variant = VariantId {
            enum_layout: layout,
            index: variant.0,
        };
        let field_values = fields
            .iter()
            .map(|field| self.runtime_value(*field))
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
        let place = RuntimePlace {
            root: match self.local_class(base) {
                Some(RuntimeClass::Handle { .. }) => PlaceRoot::Handle(self.runtime_value(base)),
                _ => PlaceRoot::Slot(self.runtime_value(base)),
            },
            path: vec![elem].into_boxed_slice(),
        };
        let target = self
            .value_class(dst)
            .cloned()
            .expect("field result must have class");
        match target {
            RuntimeClass::Scalar(_)
            | RuntimeClass::AggregateValue { .. }
            | RuntimeClass::RawAddr { .. } => {
                self.push_stmt(
                    bb,
                    RStmt::Assign {
                        dst,
                        expr: RExpr::Load { place },
                    },
                );
            }
            RuntimeClass::Handle { layout, kind, .. } => {
                let source = self.alloc_runtime_temp(
                    self.locals[dst.index()].semantic_ty,
                    RuntimeCarrier::Value(RuntimeClass::Handle {
                        layout,
                        kind: kind.clone(),
                        view: HandleView::Whole,
                    }),
                );
                self.push_stmt(
                    bb,
                    RStmt::Assign {
                        dst: source,
                        expr: RExpr::AddrOf { place },
                    },
                );
                let target = self
                    .value_class(dst)
                    .cloned()
                    .expect("field destination should have a class");
                let copied = self.coerce_value(bb, source, &target);
                self.push_stmt(
                    bb,
                    RStmt::Assign {
                        dst,
                        expr: RExpr::Use(copied),
                    },
                );
            }
        }
    }

    fn lower_enum_tag(&mut self, bb: RBlockId, dst: RLocalId, value: SLocalId) {
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
        ret_class: Option<&RuntimeClass<'db>>,
    ) -> RLocalId {
        let semantic = SemanticInstance::new(self.db, callee.key);
        let typed_body = semantic.key(self.db).instantiate_typed_body(self.db);
        let mut runtime_args = Vec::with_capacity(args.len() + effect_args.len());
        let mut runtime_classes = Vec::with_capacity(args.len() + effect_args.len());
        for (idx, arg) in args.iter().enumerate() {
            let value = self.runtime_value(*arg);
            let actual = self
                .value_class(value)
                .cloned()
                .expect("call args should have classes");
            let desired = typed_body
                .param_binding(idx)
                .map(|binding| runtime_param_class(self.db, &typed_body, binding, actual.clone()));
            let value = match desired {
                Some(class) => self.coerce_value(bb, value, &class),
                None => value,
            };
            runtime_classes.push(
                self.value_class(value)
                    .cloned()
                    .expect("coerced call args should have classes"),
            );
            runtime_args.push(value);
        }
        for effect_arg in effect_args {
            let (value, class) = self.lower_effect_arg(bb, effect_arg);
            runtime_args.push(value);
            runtime_classes.push(class);
        }
        let callee_key = RuntimeInstanceKey::new(self.db, semantic, runtime_classes);
        let callee = RuntimeInstance::new(self.db, callee_key);
        let ret_ty = semantic_return_ty(self.db, semantic);
        let ret = ret_class
            .cloned()
            .or_else(|| {
                runtime_signature_for_key(self.db, semantic, callee_key.params(self.db)).ret
            })
            .map(|class| self.alloc_runtime_temp(ret_ty, RuntimeCarrier::Value(class)))
            .unwrap_or_else(|| {
                self.alloc_runtime_temp(TyId::unit(self.db), RuntimeCarrier::Erased)
            });
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

    fn lower_effect_arg(
        &mut self,
        bb: RBlockId,
        arg: &SEffectArg<'db>,
    ) -> (RLocalId, RuntimeClass<'db>) {
        match arg.pass_mode {
            EffectPassMode::ByValue | EffectPassMode::Unknown => match &arg.arg {
                SEffectArgValue::Value(value) => {
                    let value = self.runtime_value(*value);
                    (
                        value,
                        self.value_class(value)
                            .cloned()
                            .expect("effect value args should not be erased"),
                    )
                }
                SEffectArgValue::Place(place) => {
                    let class = provider_class_for_target(
                        self.db,
                        arg.target_ty,
                        effect_arg_address_space(arg),
                    );
                    let temp = self.alloc_runtime_temp(
                        arg.target_ty.unwrap_or_else(|| TyId::unit(self.db)),
                        RuntimeCarrier::Value(class.clone()),
                    );
                    self.push_stmt(
                        bb,
                        RStmt::Assign {
                            dst: temp,
                            expr: RExpr::AddrOf {
                                place: self.lower_place(place),
                            },
                        },
                    );
                    (temp, class)
                }
            },
            EffectPassMode::ByPlace | EffectPassMode::ByTempPlace => {
                let target_ty = arg.target_ty.unwrap_or_else(|| TyId::unit(self.db));
                let space = effect_arg_address_space(arg);
                let source = match (&arg.arg, arg.pass_mode) {
                    (SEffectArgValue::Place(place), _) => self.lower_place(place),
                    (SEffectArgValue::Value(value), EffectPassMode::ByTempPlace) => {
                        let storage_class = top_level_class_for_ty(self.db, target_ty, space)
                            .unwrap_or(RuntimeClass::RawAddr {
                                space,
                                target: None,
                            });
                        let temp = self.alloc_runtime_temp(
                            target_ty,
                            RuntimeCarrier::Value(storage_class.clone()),
                        );
                        let coerced =
                            self.coerce_value(bb, self.runtime_value(*value), &storage_class);
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
                    }
                    _ => panic!("invalid effect arg lowering mode"),
                };
                let class = provider_class_for_target(self.db, Some(target_ty), space);
                let temp = self.alloc_runtime_temp(target_ty, RuntimeCarrier::Value(class.clone()));
                self.push_stmt(
                    bb,
                    RStmt::Assign {
                        dst: temp,
                        expr: RExpr::AddrOf { place: source },
                    },
                );
                (temp, class)
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
                cond: self.runtime_value(*cond),
                then_bb: self.runtime_block(*then_bb),
                else_bb: self.runtime_block(*else_bb),
            },
            STerminator::MatchEnum {
                value,
                enum_ty,
                cases,
                default,
            } => {
                let enum_layout = layout_for_ty(self.db, *enum_ty);
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
                RTerminator::Return(value.map(|value| match &ret_class {
                    Some(class) => self.coerce_value(bb, self.runtime_value(value), class),
                    None => self.runtime_value(value),
                }))
            }
        }
    }

    fn write_value_to_place(
        &mut self,
        bb: RBlockId,
        dst: RuntimePlace,
        src: RLocalId,
        target: &RuntimeClass<'db>,
    ) {
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
            .expect("cannot coerce erased value");
        if &source == target {
            return src;
        }

        match (source, target.clone()) {
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
            ) if layout == target_layout
                && matches!(layout.data(self.db), crate::runtime::Layout::Enum(_)) =>
            {
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
                panic!("unsupported runtime class coercion from {source:?} to {target:?}")
            }
        }
    }

    fn lower_place(&self, place: &SPlace) -> RuntimePlace {
        RuntimePlace {
            root: match self.local_class(place.local) {
                Some(RuntimeClass::Handle { .. }) => {
                    PlaceRoot::Handle(self.runtime_value(place.local))
                }
                _ => PlaceRoot::Slot(self.runtime_value(place.local)),
            },
            path: place
                .path
                .iter()
                .map(|elem| match elem {
                    SPlaceElem::Field(field) => PlaceElem::Field(*field),
                    SPlaceElem::Index(index) => PlaceElem::Index(self.runtime_value(*index)),
                })
                .collect(),
        }
    }

    fn project_place_class(&self, place: &RuntimePlace) -> RuntimeClass<'db> {
        let mut current = match place.root {
            PlaceRoot::Slot(local) | PlaceRoot::Handle(local) => self
                .value_class(local)
                .cloned()
                .expect("projected places should have runtime classes"),
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

    fn enum_variant_for_local(&self, value: SLocalId, variant: VariantIndex) -> VariantId<'db> {
        VariantId {
            enum_layout: self.enum_layout_for_local(value),
            index: variant.0,
        }
    }

    fn enum_layout_for_local(&self, value: SLocalId) -> LayoutId<'db> {
        match self
            .local_class(value)
            .cloned()
            .expect("enum value should have a runtime class")
        {
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
        self.blocks[bb.index()].stmts.push(stmt);
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
