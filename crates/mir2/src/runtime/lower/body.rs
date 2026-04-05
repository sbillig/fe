use cranelift_entity::EntityRef;
use hir::analysis::{
    semantic::{
        FieldIndex, SBlockId, SConst, SLocalId, SemConstId, SemConstValue, SemanticCalleeRef,
        SemanticCodeRegionRef, SemanticInstance, SemanticInstanceKey, VariantIndex,
        borrowck::{
            NBorrowRoot, NEffectArg, NEffectArgValue, NExpr, NOperand, NSPlace, NSPlaceRoot,
            NSStmt, NSStmtKind, NSTerminator, NSTerminatorKind, NormalizedSemanticBody,
            normalize_semantic_body,
        },
        ctfe::canonicalize_semantic_consts,
        get_or_build_semantic_instance, owner_effect_bindings, same_owner_effect_binding,
        sem_const_ty, semantic_may_return_normally,
    },
    ty::{
        corelib::resolve_lib_func_path,
        ty_check::{BodyOwner, EffectPassMode},
        ty_def::TyId,
    },
};
use hir::hir_def::{ArithBinOp, BinOp, CompBinOp, Func, UnOp};
use hir::projection::{IndexSource, Projection};

use crate::{
    db::MirDb,
    instance::{RuntimeInstance, RuntimeInstanceKey, get_or_build_runtime_instance},
    runtime::{
        AddressSpaceKind, ConstScalar, HandleKind, HandleView, IntrinsicArithBinOp, LayoutId,
        PlaceElem, PlaceRoot, RBlock, RBlockId, RExpr, RLocal, RLocalId, RStmt, RTerminator,
        RuntimeBody, RuntimeCarrier, RuntimeClass, RuntimeCodeRegion, RuntimeLocalLowering,
        RuntimeLocalRoot, RuntimePlace, RuntimeProviderBinding, RuntimeProviderBindingId,
        RuntimeSignature, ScalarClass, ScalarRepr, ScalarRole, VariantId,
        code_region::runtime_code_region_for_semantic_ref,
    },
};

use super::{
    class::{
        ContractMetadataBuiltin, GenericNumericIntrinsicKind, InferredRuntimeLocal,
        contract_metadata_builtin, desired_runtime_param_class, generic_numeric_intrinsic_kind,
        infer_local_runtime_state, lower_semantic_locals, provider_class_for_target_in_env,
        resolve_runtime_call_key, runtime_param_locals, runtime_signature_for_key,
        semantic_return_ty, top_level_class_for_ty_in_env,
    },
    consts::{const_scalar_from_value, enum_tag_scalar, lower_const_region},
    layout::{
        AggregateCtorElem, RuntimeTypeEnv, aggregate_ctor_elems_for_layout, layout_for_ty_in_env,
    },
    place::{
        project_field_class, project_index_class, project_variant_field_class,
        resolved_address_space,
    },
};

pub fn lower_to_rmir<'db>(db: &'db dyn MirDb, instance: RuntimeInstance<'db>) -> RuntimeBody<'db> {
    let key = instance.key(db);
    let semantic = key
        .semantic(db)
        .expect("semantic lowering only applies to semantic runtime instances");
    let semantic_body = canonicalize_semantic_consts(db, semantic);
    let normalized_body = normalize_semantic_body(db, semantic).unwrap_or_else(|err| {
        panic!(
            "semantic normalization failed for {:?}: {err:?}",
            semantic.key(db)
        )
    });
    let local_state = infer_local_runtime_state(
        db,
        &semantic_body,
        &normalized_body,
        key.params(db),
        &runtime_param_locals(db, semantic, key.params(db)),
        semantic.key(db).owner(db).scope().into(),
        semantic.key(db).instantiate_typed_body(db).assumptions(),
    );
    let (semantic_locals, provider_bindings) = lower_semantic_locals(
        db,
        &semantic_body,
        &normalized_body,
        &local_state,
        semantic.key(db).owner(db).scope().into(),
        semantic.key(db).instantiate_typed_body(db).assumptions(),
    );
    let signature = runtime_signature_for_key(db, semantic, key.params(db));
    let mut cx = RmirLowerCtxt::new(
        db,
        instance,
        key,
        semantic_body,
        normalized_body,
        LoweredSemanticLocals {
            semantic_locals,
            provider_bindings,
            local_state,
        },
        signature.clone(),
    );
    cx.lower_blocks();
    cx.finish(signature)
}

pub(super) struct RmirLowerCtxt<'db> {
    pub(super) db: &'db dyn MirDb,
    pub(super) instance: RuntimeInstance<'db>,
    pub(super) key: RuntimeInstanceKey<'db>,
    pub(super) raw_semantic_body: hir::analysis::semantic::SemanticBody<'db>,
    pub(super) semantic_body: NormalizedSemanticBody<'db>,
    pub(super) ret_class: Option<RuntimeClass<'db>>,
    pub(super) env: RuntimeTypeEnv<'db>,
    pub(super) semantic_locals: Vec<RuntimeLocalLowering<'db>>,
    pub(super) provider_bindings: Vec<RuntimeProviderBinding<'db>>,
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

struct LoweredSemanticLocals<'db> {
    semantic_locals: Vec<RuntimeLocalLowering<'db>>,
    provider_bindings: Vec<RuntimeProviderBinding<'db>>,
    local_state: Vec<InferredRuntimeLocal<'db>>,
}

impl<'db> RmirLowerCtxt<'db> {
    fn new(
        db: &'db dyn MirDb,
        instance: RuntimeInstance<'db>,
        key: RuntimeInstanceKey<'db>,
        raw_semantic_body: hir::analysis::semantic::SemanticBody<'db>,
        semantic_body: NormalizedSemanticBody<'db>,
        lowered_locals: LoweredSemanticLocals<'db>,
        signature: RuntimeSignature<'db>,
    ) -> Self {
        let LoweredSemanticLocals {
            semantic_locals,
            provider_bindings,
            local_state,
        } = lowered_locals;
        let typed_body = key
            .semantic(db)
            .expect("runtime lowering requires a semantic instance")
            .key(db)
            .instantiate_typed_body(db);
        let env = RuntimeTypeEnv::new(
            typed_body.body().map(|body| body.scope()),
            typed_body.assumptions(),
        );
        let terminated_blocks = vec![false; semantic_body.blocks.len()];
        let locals = semantic_body
            .locals
            .iter()
            .enumerate()
            .map(|(idx, local)| {
                let inferred = local_state
                    .get(idx)
                    .cloned()
                    .unwrap_or(InferredRuntimeLocal {
                        carrier: RuntimeCarrier::Erased,
                        root: RuntimeLocalRoot::None,
                    });
                RLocal {
                    semantic_ty: local.ty,
                    carrier: inferred.carrier,
                    root: inferred.root,
                }
            })
            .collect::<Vec<_>>();
        let blocks = Vec::with_capacity(semantic_body.blocks.len());
        Self {
            db,
            instance,
            key,
            raw_semantic_body,
            semantic_body,
            ret_class: signature.ret.clone(),
            env,
            semantic_locals,
            provider_bindings,
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
            semantic_locals: self.semantic_locals,
            provider_bindings: self.provider_bindings,
            locals: self.locals,
            blocks: self.blocks,
        }
    }

    fn layout_for_ty(&self, ty: TyId<'db>) -> LayoutId<'db> {
        layout_for_ty_in_env(self.db, self.env, ty)
    }

    fn current_semantic_key(&self) -> SemanticInstanceKey<'db> {
        self.key
            .semantic(self.db)
            .expect("runtime lowering requires a semantic instance")
            .key(self.db)
    }

    fn top_level_class_for_ty(
        &self,
        ty: TyId<'db>,
        default_space: crate::runtime::AddressSpaceKind,
    ) -> Option<RuntimeClass<'db>> {
        top_level_class_for_ty_in_env(self.db, self.env, ty, default_space)
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

    fn lower_stmt(&mut self, bb: RBlockId, stmt: &NSStmt<'db>) {
        match &stmt.kind {
            NSStmtKind::Assign { dst, expr } => self.lower_assign(bb, *dst, expr),
            NSStmtKind::Store { dst, src } => {
                let place = self.lower_place(bb, dst);
                let target = self.project_place_class(&place);
                let value = self.read_semantic_value(bb, *src);
                self.write_value_to_place(bb, place, value, &target);
            }
        }
    }

    fn lower_assign(&mut self, bb: RBlockId, dst: SLocalId, expr: &NExpr<'db>) {
        let desired = self.semantic_value_class(dst);
        match desired {
            None => {
                let sink = self.alloc_runtime_temp(TyId::unit(self.db), RuntimeCarrier::Erased);
                self.lower_expr_into(bb, sink, expr);
            }
            Some(desired) => {
                if self.semantic_local_is_direct(dst) {
                    self.lower_expr_into(bb, self.runtime_value(dst), expr);
                    return;
                }
                let temp = self.alloc_runtime_temp(
                    self.locals[dst.index()].semantic_ty,
                    RuntimeCarrier::Value(desired),
                );
                self.lower_expr_into(bb, temp, expr);
                self.write_semantic_value(bb, dst, temp);
            }
        }
    }

    fn lower_expr_into(&mut self, bb: RBlockId, dst: RLocalId, expr: &NExpr<'db>) {
        let Some(dst_class) = self.value_class(dst).cloned() else {
            if let NExpr::Call {
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
            NExpr::Use(src) => {
                let value = self.lower_semantic_operand_for_class(bb, *src, &dst_class);
                self.push_stmt(
                    bb,
                    RStmt::Assign {
                        dst,
                        expr: RExpr::Use(value),
                    },
                );
            }
            NExpr::ReadPlace { place, .. } => {
                let place = self.lower_place(bb, place);
                let projected = self.project_place_class(&place);
                if projected == dst_class {
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
                let copied = self.coerce_value(bb, source, &dst_class);
                self.push_stmt(
                    bb,
                    RStmt::Assign {
                        dst,
                        expr: RExpr::Use(copied),
                    },
                );
            }
            NExpr::Const(const_) => self.lower_const_into(bb, dst, const_),
            NExpr::Unary { op, value } => {
                let value = self.read_semantic_operand(bb, *value);
                self.push_stmt(
                    bb,
                    RStmt::Assign {
                        dst,
                        expr: RExpr::Unary { op: *op, value },
                    },
                );
            }
            NExpr::Binary { op, lhs, rhs } => {
                let lhs = self.read_semantic_operand(bb, *lhs);
                let rhs = self.read_semantic_operand(bb, *rhs);
                self.push_stmt(
                    bb,
                    RStmt::Assign {
                        dst,
                        expr: RExpr::Binary { op: *op, lhs, rhs },
                    },
                );
            }
            NExpr::Cast { value, .. } => {
                let RuntimeClass::Scalar(to) = dst_class else {
                    panic!("casts must lower to scalar carriers");
                };
                let value = self.read_semantic_operand(bb, *value);
                self.push_stmt(
                    bb,
                    RStmt::Assign {
                        dst,
                        expr: RExpr::Cast { value, to },
                    },
                );
            }
            NExpr::AggregateMake { ty, fields } => self.lower_aggregate_make(bb, dst, *ty, fields),
            NExpr::EnumMake {
                enum_ty,
                variant,
                fields,
            } => self.lower_enum_make(bb, dst, *enum_ty, *variant, fields),
            NExpr::Borrow { place, .. } => {
                let place = self.lower_place(bb, place);
                self.push_stmt(
                    bb,
                    RStmt::Assign {
                        dst,
                        expr: RExpr::AddrOf { place },
                    },
                );
            }
            NExpr::GetEnumTag { value } => self.lower_enum_tag(bb, dst, *value),
            NExpr::IsEnumVariant { value, variant } => {
                self.lower_is_enum_variant(bb, dst, *value, *variant);
            }
            NExpr::ExtractEnumField {
                value,
                variant,
                field,
            } => {
                self.lower_extract_enum_field(bb, dst, *value, *variant, *field);
            }
            NExpr::CodeRegionOffset { region } => self.push_stmt(
                bb,
                RStmt::Assign {
                    dst,
                    expr: RExpr::Builtin(crate::runtime::RuntimeBuiltin::CodeRegionOffset {
                        region: self.lower_code_region_ref(region),
                    }),
                },
            ),
            NExpr::CodeRegionLen { region } => self.push_stmt(
                bb,
                RStmt::Assign {
                    dst,
                    expr: RExpr::Builtin(crate::runtime::RuntimeBuiltin::CodeRegionLen {
                        region: self.lower_code_region_ref(region),
                    }),
                },
            ),
            NExpr::Call {
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
        runtime_code_region_for_semantic_ref(self.db, region)
    }

    fn lower_const_into(&mut self, bb: RBlockId, dst: RLocalId, const_: &SConst<'db>) {
        match const_ {
            SConst::Value(value) => {
                if sem_const_ty(self.db, *value) == TyId::unit(self.db) {
                    return;
                }
                let target = self
                    .value_class(dst)
                    .cloned()
                    .expect("const destination should have a runtime class");
                self.lower_sem_const_for_class(bb, dst, *value, &target);
            }
            SConst::Ref(cref) => panic!("unresolved const ref reached rMIR lowering: {cref:?}"),
        }
    }

    fn lower_sem_const_for_class(
        &mut self,
        bb: RBlockId,
        dst: RLocalId,
        value: SemConstId<'db>,
        target: &RuntimeClass<'db>,
    ) {
        let src = self.lower_sem_const_as_class(bb, value, target);
        let value = if self.value_class(src) == Some(target) {
            src
        } else {
            self.coerce_value(bb, src, target)
        };
        self.push_stmt(
            bb,
            RStmt::Assign {
                dst,
                expr: RExpr::Use(value),
            },
        );
    }

    fn lower_sem_const_as_class(
        &mut self,
        bb: RBlockId,
        value: SemConstId<'db>,
        target: &RuntimeClass<'db>,
    ) -> RLocalId {
        let ty = sem_const_ty(self.db, value);
        if matches!(
            target,
            RuntimeClass::Handle {
                kind: HandleKind::ConstValue,
                ..
            }
        ) {
            return self.lower_sem_const_as_const_handle(bb, value, ty);
        }
        if let Some(scalar) = const_scalar_from_value(self.db, self.env, value) {
            return self.lower_sem_const_scalar(bb, ty, scalar);
        }
        match target {
            RuntimeClass::Scalar(_) => {
                panic!(
                    "non-scalar semantic const {value:?} cannot lower to scalar class {target:?}"
                )
            }
            RuntimeClass::AggregateValue { layout } => {
                self.lower_non_scalar_const_as_aggregate_value(bb, value, ty, *layout)
            }
            RuntimeClass::Handle {
                layout,
                kind: HandleKind::ObjectValue,
                view: HandleView::Whole,
            } => self.lower_non_scalar_const_as_object(bb, value, ty, *layout),
            RuntimeClass::Handle {
                kind: HandleKind::Provider { .. },
                ..
            }
            | RuntimeClass::RawAddr { .. } => {
                let layout = self.layout_for_ty(ty);
                self.lower_non_scalar_const_as_object(bb, value, ty, layout)
            }
            RuntimeClass::Handle {
                kind: HandleKind::ObjectValue,
                view,
                ..
            } => {
                panic!("non-whole object-handle const lowering is invalid for {view:?}")
            }
            RuntimeClass::Handle {
                kind: HandleKind::ConstValue,
                ..
            } => unreachable!(),
        }
    }

    fn lower_sem_const_as_const_handle(
        &mut self,
        bb: RBlockId,
        value: SemConstId<'db>,
        ty: TyId<'db>,
    ) -> RLocalId {
        let region = lower_const_region(self.db, self.env, value).unwrap_or_else(|| {
            panic!("const-backed handle should lower to a const region: {value:?}")
        });
        let layout = region.layout(self.db);
        let local = self.alloc_runtime_temp(
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
                dst: local,
                expr: RExpr::ConstHandle { region, layout },
            },
        );
        local
    }

    fn lower_sem_const_scalar(
        &mut self,
        bb: RBlockId,
        ty: TyId<'db>,
        scalar: ConstScalar,
    ) -> RLocalId {
        let class = self
            .top_level_class_for_ty(ty, AddressSpaceKind::Memory)
            .unwrap_or_else(|| panic!("scalar const should have a runtime class: {ty:?}"));
        let local = self.alloc_runtime_temp(ty, RuntimeCarrier::Value(class));
        self.push_stmt(
            bb,
            RStmt::Assign {
                dst: local,
                expr: RExpr::ConstScalar(scalar),
            },
        );
        local
    }

    fn lower_non_scalar_const_as_aggregate_value(
        &mut self,
        bb: RBlockId,
        value: SemConstId<'db>,
        ty: TyId<'db>,
        layout: LayoutId<'db>,
    ) -> RLocalId {
        match value.value(self.db) {
            SemConstValue::Tuple { elems, .. }
            | SemConstValue::Struct { fields: elems, .. }
            | SemConstValue::Array { elems, .. } => {
                let ctor_elems = aggregate_ctor_elems_for_layout(self.db, layout, elems.len());
                let field_values = elems
                    .iter()
                    .copied()
                    .zip(ctor_elems.iter())
                    .map(|(field, elem)| self.lower_sem_const_as_class(bb, field, &elem.class))
                    .collect::<Vec<_>>();
                let dst = self.alloc_runtime_temp(
                    ty,
                    RuntimeCarrier::Value(RuntimeClass::AggregateValue { layout }),
                );
                self.lower_aggregate_values(bb, dst, ty, layout, &ctor_elems, &field_values);
                dst
            }
            SemConstValue::Enum {
                variant, fields, ..
            } => {
                let crate::runtime::Layout::Enum(layout_data) = layout.data(self.db) else {
                    panic!("enum constant requires an enum layout");
                };
                let field_classes = layout_data
                    .variants
                    .get(variant.0 as usize)
                    .unwrap_or_else(|| panic!("missing enum variant layout for {variant:?}"))
                    .fields
                    .to_vec();
                let field_values = fields
                    .iter()
                    .copied()
                    .zip(field_classes.iter())
                    .map(|(field, class)| self.lower_sem_const_as_class(bb, field, class))
                    .collect::<Vec<_>>();
                let dst = self.alloc_runtime_temp(
                    ty,
                    RuntimeCarrier::Value(RuntimeClass::AggregateValue { layout }),
                );
                self.lower_enum_values(bb, dst, layout, variant, &field_values);
                dst
            }
            SemConstValue::Unit
            | SemConstValue::Scalar { .. }
            | SemConstValue::TypeLevel { .. } => {
                panic!("expected non-scalar semantic const, found {value:?}")
            }
        }
    }

    fn lower_non_scalar_const_as_object(
        &mut self,
        bb: RBlockId,
        value: SemConstId<'db>,
        ty: TyId<'db>,
        layout: LayoutId<'db>,
    ) -> RLocalId {
        match value.value(self.db) {
            SemConstValue::Tuple { elems, .. }
            | SemConstValue::Struct { fields: elems, .. }
            | SemConstValue::Array { elems, .. } => {
                let ctor_elems = aggregate_ctor_elems_for_layout(self.db, layout, elems.len());
                let field_values = elems
                    .iter()
                    .copied()
                    .zip(ctor_elems.iter())
                    .map(|(field, elem)| self.lower_sem_const_as_class(bb, field, &elem.class))
                    .collect::<Vec<_>>();
                let dst = self.alloc_runtime_temp(
                    ty,
                    RuntimeCarrier::Value(RuntimeClass::Handle {
                        layout,
                        kind: HandleKind::ObjectValue,
                        view: HandleView::Whole,
                    }),
                );
                self.lower_aggregate_values(bb, dst, ty, layout, &ctor_elems, &field_values);
                dst
            }
            SemConstValue::Enum {
                variant, fields, ..
            } => {
                let crate::runtime::Layout::Enum(layout_data) = layout.data(self.db) else {
                    panic!("enum constant requires an enum layout");
                };
                let field_classes = layout_data
                    .variants
                    .get(variant.0 as usize)
                    .unwrap_or_else(|| panic!("missing enum variant layout for {variant:?}"))
                    .fields
                    .to_vec();
                let field_values = fields
                    .iter()
                    .copied()
                    .zip(field_classes.iter())
                    .map(|(field, class)| self.lower_sem_const_as_class(bb, field, class))
                    .collect::<Vec<_>>();
                let dst = self.alloc_runtime_temp(
                    ty,
                    RuntimeCarrier::Value(RuntimeClass::Handle {
                        layout,
                        kind: HandleKind::ObjectValue,
                        view: HandleView::Whole,
                    }),
                );
                self.lower_enum_values(bb, dst, layout, variant, &field_values);
                dst
            }
            SemConstValue::Unit
            | SemConstValue::Scalar { .. }
            | SemConstValue::TypeLevel { .. } => {
                panic!("expected non-scalar semantic const, found {value:?}")
            }
        }
    }

    fn lower_aggregate_make(
        &mut self,
        bb: RBlockId,
        dst: RLocalId,
        ty: TyId<'db>,
        fields: &[NOperand],
    ) {
        let layout = self.layout_for_ty(ty);
        let ctor_elems = aggregate_ctor_elems_for_layout(self.db, layout, fields.len());
        let field_values = fields
            .iter()
            .copied()
            .zip(ctor_elems.iter())
            .map(|(field, elem)| self.lower_semantic_operand_for_class(bb, field, &elem.class))
            .collect::<Vec<_>>();
        self.lower_aggregate_values(bb, dst, ty, layout, &ctor_elems, &field_values);
    }

    fn lower_aggregate_values(
        &mut self,
        bb: RBlockId,
        dst: RLocalId,
        ty: TyId<'db>,
        layout: LayoutId<'db>,
        ctor_elems: &[AggregateCtorElem<'db>],
        field_values: &[RLocalId],
    ) {
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
                for (value, elem) in field_values.iter().copied().zip(ctor_elems.iter()) {
                    if self.value_class(value).is_none() {
                        continue;
                    }
                    let place = RuntimePlace {
                        root: PlaceRoot::Handle(temp),
                        path: vec![elem.elem.clone()].into_boxed_slice(),
                    };
                    self.write_value_to_place(bb, place, value, &elem.class);
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
                for (value, elem) in field_values.iter().copied().zip(ctor_elems.iter()) {
                    if self.value_class(value).is_none() {
                        continue;
                    }
                    let place = RuntimePlace {
                        root: PlaceRoot::Handle(dst),
                        path: vec![elem.elem.clone()].into_boxed_slice(),
                    };
                    self.write_value_to_place(bb, place, value, &elem.class);
                }
            }
            RuntimeClass::RawAddr { space, target } => {
                let [value] = field_values else {
                    panic!("raw-address aggregate construction requires exactly one field");
                };
                let value = self.coerce_value(bb, *value, &RuntimeClass::RawAddr { space, target });
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
                let [value] = field_values else {
                    panic!("provider aggregate construction requires exactly one field");
                };
                let value = self.coerce_value(
                    bb,
                    *value,
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
        fields: &[NOperand],
    ) {
        let layout = self.layout_for_ty(enum_ty);
        let crate::runtime::Layout::Enum(layout_data) = layout.data(self.db) else {
            panic!("enum construction requires an enum layout");
        };
        let field_classes = layout_data
            .variants
            .get(variant.0 as usize)
            .unwrap_or_else(|| panic!("missing enum variant layout for {variant:?}"))
            .fields
            .to_vec();
        let field_values = fields
            .iter()
            .zip(field_classes.iter())
            .map(|(field, class)| self.lower_semantic_operand_for_class(bb, *field, class))
            .collect::<Vec<_>>();
        self.lower_enum_values(bb, dst, layout, variant, &field_values);
    }

    fn lower_enum_values(
        &mut self,
        bb: RBlockId,
        dst: RLocalId,
        layout: LayoutId<'db>,
        variant: VariantIndex,
        field_values: &[RLocalId],
    ) {
        let variant = VariantId {
            enum_layout: layout,
            index: variant.0,
        };
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
                            fields: field_values.to_vec().into_boxed_slice(),
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
                            fields: field_values.to_vec().into_boxed_slice(),
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
        let mut place = self.semantic_place(bb, base);
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

    fn lower_enum_tag(&mut self, bb: RBlockId, dst: RLocalId, value: NOperand) {
        if self.semantic_local_is_place_bound(value.local) {
            let value = self.read_semantic_operand(bb, value);
            self.push_stmt(
                bb,
                RStmt::Assign {
                    dst,
                    expr: RExpr::EnumTagOfValue { value },
                },
            );
            return;
        }
        match self.local_class(value.local) {
            Some(RuntimeClass::Handle { .. }) => {
                self.push_stmt(
                    bb,
                    RStmt::Assign {
                        dst,
                        expr: RExpr::EnumGetTag {
                            root: self.runtime_value(value.local),
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
                            value: self.runtime_value(value.local),
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
        value: NOperand,
        variant: VariantIndex,
    ) {
        if self.semantic_local_is_place_bound(value.local) {
            let variant = self.enum_variant_for_local(value.local, variant);
            let value = self.read_semantic_operand(bb, value);
            self.push_stmt(
                bb,
                RStmt::Assign {
                    dst,
                    expr: RExpr::EnumIsVariant { value, variant },
                },
            );
            return;
        }
        match self.local_class(value.local) {
            Some(RuntimeClass::Handle { .. }) => {
                let enum_layout = self.enum_layout_for_local(value.local);
                let tag_class = RuntimeClass::Scalar(ScalarClass {
                    repr: match enum_layout.data(self.db) {
                        crate::runtime::Layout::Enum(layout) => layout.tag.repr,
                        _ => unreachable!(),
                    },
                    role: ScalarRole::EnumTag { enum_layout },
                });
                let tag = self.alloc_runtime_temp(
                    self.locals[value.local.index()].semantic_ty,
                    RuntimeCarrier::Value(tag_class),
                );
                self.lower_enum_tag(bb, tag, value);
                let expected = self.alloc_runtime_temp(
                    self.locals[value.local.index()].semantic_ty,
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
                let variant = self.enum_variant_for_local(value.local, variant);
                self.push_stmt(
                    bb,
                    RStmt::Assign {
                        dst,
                        expr: RExpr::EnumIsVariant {
                            value: self.runtime_value(value.local),
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
        value: NOperand,
        variant: VariantIndex,
        field: FieldIndex,
    ) {
        if self.semantic_local_is_place_bound(value.local) {
            let variant = self.enum_variant_for_local(value.local, variant);
            let value = self.read_semantic_operand(bb, value);
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
        match self.local_class(value.local) {
            Some(RuntimeClass::Handle { .. }) => {
                let refined = self.enum_variant_ref(bb, value.local, variant);
                self.lower_field_like(bb, dst, refined, PlaceElem::VariantField(field));
            }
            _ => {
                let variant = self.enum_variant_for_local(value.local, variant);
                self.push_stmt(
                    bb,
                    RStmt::Assign {
                        dst,
                        expr: RExpr::EnumExtract {
                            value: self.runtime_value(value.local),
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
        args: &[NOperand],
        effect_args: &[NEffectArg<'db>],
    ) -> RLocalId {
        let caller_key = self.current_semantic_key();
        let caller_typed_body = caller_key.instantiate_typed_body(self.db);
        let callee_key = resolve_runtime_call_key(
            self.db,
            caller_key,
            &caller_typed_body,
            &self.raw_semantic_body,
            callee,
            args,
        );
        let semantic = get_or_build_semantic_instance(self.db, callee_key);
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
        let callee = get_or_build_runtime_instance(self.db, callee_key);
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
        args: &[NOperand],
        effect_args: &[NEffectArg<'db>],
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
        args: &[NOperand],
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
        let scalar = match &ret_class {
            RuntimeClass::Scalar(scalar) => scalar.clone(),
            RuntimeClass::AggregateValue { .. }
            | RuntimeClass::Handle { .. }
            | RuntimeClass::RawAddr { .. } => return None,
        };
        let ret = self.alloc_runtime_temp(ret_ty, RuntimeCarrier::Value(ret_class.clone()));
        let expr = match generic_numeric_intrinsic_kind(name.as_str()) {
            Some(GenericNumericIntrinsicKind::Saturating(op)) => {
                let [lhs, rhs] = args.as_slice() else {
                    return None;
                };
                RExpr::Builtin(crate::runtime::RuntimeBuiltin::Saturating {
                    op,
                    lhs: *lhs,
                    rhs: *rhs,
                    class: scalar,
                })
            }
            Some(GenericNumericIntrinsicKind::Bitcast) => {
                let [value] = args.as_slice() else {
                    return None;
                };
                RExpr::Cast {
                    value: *value,
                    to: scalar,
                }
            }
            Some(GenericNumericIntrinsicKind::CheckedBinary(op)) => {
                let [lhs, rhs] = args.as_slice() else {
                    return None;
                };
                RExpr::Binary {
                    op: BinOp::Arith(op),
                    lhs: *lhs,
                    rhs: *rhs,
                }
            }
            Some(GenericNumericIntrinsicKind::CheckedNeg) => {
                let [value] = args.as_slice() else {
                    return None;
                };
                let ScalarRepr::Int { bits, signed } = scalar.repr else {
                    return None;
                };
                let zero = self.alloc_runtime_temp(ret_ty, RuntimeCarrier::Value(ret_class));
                self.push_stmt(
                    bb,
                    RStmt::Assign {
                        dst: zero,
                        expr: RExpr::ConstScalar(ConstScalar::Int {
                            bits,
                            signed,
                            words: Vec::new(),
                        }),
                    },
                );
                RExpr::Binary {
                    op: BinOp::Arith(ArithBinOp::Sub),
                    lhs: zero,
                    rhs: *value,
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
        args: &[NOperand],
    ) -> (Vec<RLocalId>, Vec<RuntimeClass<'db>>) {
        let mut runtime_args = Vec::with_capacity(args.len());
        let mut runtime_classes = Vec::with_capacity(args.len());
        for (idx, arg) in args.iter().enumerate() {
            let desired = desired_runtime_param_class(self.db, typed_body, idx);
            let value = self.runtime_visible_arg_value(bb, *arg, desired.as_ref());
            let actual = self.value_class(value).cloned();
            let desired = desired.map(|class| {
                actual.as_ref().map_or(class.clone(), |actual| {
                    preserve_provider_space(actual, &class)
                })
            });
            let Some(value) = (match (actual, desired) {
                (None, None) => None,
                (Some(_), None) => Some(value),
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
        arg: NOperand,
        desired: Option<&RuntimeClass<'db>>,
    ) -> RLocalId {
        if let Some(desired) = desired {
            self.lower_semantic_operand_for_class(bb, arg, desired)
        } else {
            self.read_semantic_operand(bb, arg)
        }
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
                        let local = SLocalId::from_u32(idx as u32);
                        if let Some(value) = self.semantic_provider_value(local) {
                            return match self.semantic_local_lowering(local) {
                                RuntimeLocalLowering::PlaceBoundValue { provider, .. }
                                | RuntimeLocalLowering::DirectCarrier {
                                    provider: Some(provider),
                                    ..
                                } => Some((
                                    value,
                                    self.provider_binding(*provider).provider_class.clone(),
                                )),
                                RuntimeLocalLowering::Erased
                                | RuntimeLocalLowering::DirectValue
                                | RuntimeLocalLowering::PlaceCarrier { .. }
                                | RuntimeLocalLowering::DirectCarrier { provider: None, .. } => {
                                    None
                                }
                            };
                        }
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
        args: &[NOperand],
    ) -> Option<RLocalId> {
        if resolve_lib_func_path(self.db, func.scope(), "core::intrinsic::__keccak256")
            != Some(func)
        {
            return None;
        }

        let [bytes] = args else {
            return None;
        };
        let bytes_ty = self.semantic_body.local(bytes.local)?.ty;
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

        let value = self.read_semantic_operand(bb, *bytes);
        let provider_class = provider_class_for_target_in_env(
            self.db,
            self.env,
            Some(bytes_ty),
            AddressSpaceKind::Memory,
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
        arg: &NEffectArg<'db>,
    ) -> Option<(RLocalId, RuntimeClass<'db>)> {
        match arg.pass_mode {
            EffectPassMode::ByValue | EffectPassMode::Unknown => match &arg.arg {
                NEffectArgValue::Value(value) => {
                    let value = if arg.provider.is_none() && arg.target_ty.is_none() {
                        self.read_semantic_operand(bb, *value)
                    } else {
                        self.runtime_value(value.local)
                    };
                    if let Some(class) = self.value_class(value).cloned() {
                        return Some((value, class));
                    }
                    if arg.provider.is_none() && arg.target_ty.is_none() {
                        return None;
                    }
                    let class = provider_class_for_target_in_env(
                        self.db,
                        self.env,
                        arg.target_ty,
                        resolved_address_space(arg.provider),
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
                NEffectArgValue::Place(place) => {
                    let class = provider_class_for_target_in_env(
                        self.db,
                        self.env,
                        arg.target_ty,
                        resolved_address_space(arg.provider),
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
                let space = resolved_address_space(arg.provider);
                let source = match (&arg.arg, arg.pass_mode) {
                    (NEffectArgValue::Place(place), _) => self.try_lower_place(bb, place),
                    (NEffectArgValue::Value(value), EffectPassMode::ByTempPlace) => {
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
                        let value = self.read_semantic_operand(bb, *value);
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
                let class =
                    provider_class_for_target_in_env(self.db, self.env, Some(target_ty), space);
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
                    let NEffectArgValue::Place(place) = &arg.arg else {
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
        terminator: &NSTerminator<'db>,
    ) -> RTerminator<'db> {
        match &terminator.kind {
            NSTerminatorKind::Goto(block) => RTerminator::Goto(self.runtime_block(*block)),
            NSTerminatorKind::Branch {
                cond,
                then_bb,
                else_bb,
            } => RTerminator::Branch {
                cond: self.read_semantic_operand(bb, *cond),
                then_bb: self.runtime_block(*then_bb),
                else_bb: self.runtime_block(*else_bb),
            },
            NSTerminatorKind::MatchEnum {
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
            NSTerminatorKind::Return(value) => {
                let ret_class = self.ret_class.clone();
                RTerminator::Return(match ret_class {
                    Some(class) => {
                        value.map(|value| self.lower_semantic_operand_for_class(bb, value, &class))
                    }
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
                    kind: HandleKind::ObjectValue | HandleKind::ConstValue,
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
                RuntimeClass::Handle {
                    layout,
                    kind: HandleKind::ObjectValue | HandleKind::ConstValue,
                    view: HandleView::Whole,
                },
                RuntimeClass::RawAddr {
                    space,
                    target: target_layout,
                },
            ) if target_layout.is_none_or(|target_layout| target_layout == layout) => {
                let temp = self.alloc_runtime_temp(
                    self.locals[src.index()].semantic_ty,
                    RuntimeCarrier::Value(RuntimeClass::RawAddr {
                        space,
                        target: Some(layout),
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
                RuntimeClass::Scalar(ScalarClass {
                    repr:
                        ScalarRepr::Int {
                            bits: 256,
                            signed: false,
                        },
                    role: ScalarRole::Plain,
                }),
                RuntimeClass::Handle {
                    layout,
                    kind: HandleKind::Provider { provider_ty, space },
                    view: HandleView::Whole,
                },
            ) => {
                let raw = self.coerce_scalar_word_to_raw(bb, src, space, Some(layout));
                self.coerce_value(
                    bb,
                    raw,
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
                let layout_source_ty = |layout: LayoutId<'db>| match layout.data(self.db) {
                    crate::runtime::Layout::Struct(data) => {
                        data.source_ty.pretty_print(self.db).to_string()
                    }
                    crate::runtime::Layout::Array(data) => {
                        data.source_ty.pretty_print(self.db).to_string()
                    }
                    crate::runtime::Layout::Enum(data) => {
                        data.source_ty.pretty_print(self.db).to_string()
                    }
                };
                let source_layout = match source {
                    RuntimeClass::AggregateValue { layout }
                    | RuntimeClass::Handle { layout, .. }
                    | RuntimeClass::RawAddr {
                        target: Some(layout),
                        ..
                    } => Some((layout, layout.data(self.db), layout_source_ty(layout))),
                    RuntimeClass::Scalar(_) | RuntimeClass::RawAddr { target: None, .. } => None,
                };
                let target_layout = match target {
                    RuntimeClass::AggregateValue { layout }
                    | RuntimeClass::Handle { layout, .. }
                    | RuntimeClass::RawAddr {
                        target: Some(layout),
                        ..
                    } => Some((layout, layout.data(self.db), layout_source_ty(layout))),
                    RuntimeClass::Scalar(_) | RuntimeClass::RawAddr { target: None, .. } => None,
                };
                panic!(
                    "unsupported runtime class coercion in {:?} owner={:?} from {source:?} to {target:?}; source_layout={source_layout:?}; target_layout={target_layout:?}; src={src:?}; src_ty={}; locals={:?}",
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

    fn semantic_local_lowering(&self, local: SLocalId) -> &RuntimeLocalLowering<'db> {
        self.semantic_locals.get(local.index()).unwrap_or_else(|| {
            panic!(
                "missing semantic local lowering for {local:?}; semantic_locals={:?}",
                self.semantic_locals,
            )
        })
    }

    fn semantic_local_is_direct(&self, local: SLocalId) -> bool {
        matches!(
            self.semantic_local_lowering(local),
            RuntimeLocalLowering::DirectValue
                | RuntimeLocalLowering::PlaceCarrier { .. }
                | RuntimeLocalLowering::DirectCarrier { .. }
        )
    }

    fn semantic_local_is_place_bound(&self, local: SLocalId) -> bool {
        matches!(
            self.semantic_local_lowering(local),
            RuntimeLocalLowering::PlaceCarrier { .. }
                | RuntimeLocalLowering::PlaceBoundValue { .. }
        )
    }

    fn semantic_value_class(&self, local: SLocalId) -> Option<RuntimeClass<'db>> {
        match self.semantic_local_lowering(local) {
            RuntimeLocalLowering::Erased => None,
            RuntimeLocalLowering::DirectValue | RuntimeLocalLowering::DirectCarrier { .. } => {
                self.local_class(local).cloned()
            }
            RuntimeLocalLowering::PlaceCarrier { place_class }
            | RuntimeLocalLowering::PlaceBoundValue { place_class, .. } => {
                Some(place_class.clone())
            }
        }
    }

    fn provider_binding(&self, id: RuntimeProviderBindingId) -> &RuntimeProviderBinding<'db> {
        self.provider_bindings
            .get(id.index())
            .unwrap_or_else(|| panic!("missing runtime provider binding for {id:?}"))
    }

    fn provider_binding_value(&self, id: RuntimeProviderBindingId) -> RLocalId {
        self.provider_binding(id).value
    }

    fn provider_binding_id_for_semantic(
        &self,
        provider: &hir::semantic::ProviderBinding<'db>,
    ) -> Option<RuntimeProviderBindingId> {
        self.provider_bindings
            .iter()
            .enumerate()
            .find_map(|(idx, binding)| {
                (binding.provider == *provider)
                    .then(|| RuntimeProviderBindingId::from_u32(idx as u32))
            })
    }

    fn provider_place_root(
        &self,
        provider: &hir::semantic::ProviderBinding<'db>,
    ) -> Option<PlaceRoot<'db>> {
        self.provider_binding_id_for_semantic(provider)
            .map(PlaceRoot::Provider)
    }

    fn semantic_provider_value(&self, local: SLocalId) -> Option<RLocalId> {
        match self.semantic_local_lowering(local) {
            RuntimeLocalLowering::PlaceBoundValue { provider, .. } => {
                Some(self.provider_binding_value(*provider))
            }
            RuntimeLocalLowering::PlaceCarrier { .. } => Some(self.runtime_value(local)),
            RuntimeLocalLowering::DirectCarrier {
                provider: Some(provider),
                ..
            } => Some(self.provider_binding_value(*provider)),
            RuntimeLocalLowering::Erased
            | RuntimeLocalLowering::DirectValue
            | RuntimeLocalLowering::DirectCarrier { provider: None, .. } => None,
        }
    }

    fn semantic_place_root(&self, local: SLocalId) -> Option<PlaceRoot<'db>> {
        if let RuntimeLocalLowering::PlaceBoundValue { provider, .. } =
            self.semantic_local_lowering(local)
        {
            return Some(PlaceRoot::Provider(*provider));
        }
        if let RuntimeLocalLowering::DirectCarrier {
            provider: Some(provider),
            ..
        } = self.semantic_local_lowering(local)
        {
            return Some(PlaceRoot::Provider(*provider));
        }
        self.local_root(local)
    }

    fn normalized_value_place(&self, local: SLocalId) -> Option<&NSPlace<'db>> {
        self.semantic_body.local(local)?.lowering.place()
    }

    fn is_self_rooted_value_place(&self, local: SLocalId, place: &NSPlace<'db>) -> bool {
        if !place.path.is_empty() {
            return false;
        }
        match place.root {
            NSPlaceRoot::CarrierDerefLocal(root_local) => root_local == local,
            NSPlaceRoot::Root(root) => matches!(
                self.semantic_body.root(root),
                Some(NBorrowRoot::Param { local: root_local, .. } | NBorrowRoot::LocalSlot { local: root_local })
                    if *root_local == local
            ),
        }
    }

    fn try_semantic_place(&mut self, bb: RBlockId, local: SLocalId) -> Option<RuntimePlace<'db>> {
        if let Some(place) = self.normalized_value_place(local).cloned()
            && !self.is_self_rooted_value_place(local, &place)
        {
            return self.try_lower_place(bb, &place);
        }
        let root = self.semantic_place_root(local)?;
        Some(RuntimePlace {
            root,
            path: Box::default(),
        })
    }

    fn semantic_place(&mut self, bb: RBlockId, local: SLocalId) -> RuntimePlace<'db> {
        self.try_semantic_place(bb, local).unwrap_or_else(|| {
            panic!(
                "cannot lower erased local as a runtime place root: source={:?}; owner={:?}; local={local:?}; ty={}; source_binding={:?}; lowering={:?}; root={:?}; carrier={:?}",
                self.key.source(self.db),
                self.key
                    .semantic(self.db)
                    .map(|semantic| semantic.key(self.db).owner(self.db)),
                self.locals[local.index()].semantic_ty.pretty_print(self.db),
                self.semantic_body.locals[local.index()].source,
                self.semantic_local_lowering(local),
                self.locals[self.runtime_value(local).index()].root,
                self.locals[self.runtime_value(local).index()].carrier,
            )
        })
    }

    fn try_lower_place(&mut self, bb: RBlockId, place: &NSPlace<'db>) -> Option<RuntimePlace<'db>> {
        let mut runtime_place = match place.root {
            NSPlaceRoot::CarrierDerefLocal(local) => self.try_semantic_place(bb, local)?,
            NSPlaceRoot::Root(root) => match self.semantic_body.root(root)? {
                NBorrowRoot::Param { local, .. } | NBorrowRoot::LocalSlot { local } => {
                    self.try_semantic_place(bb, *local)?
                }
                NBorrowRoot::Provider { binding } => RuntimePlace {
                    root: self.provider_place_root(binding)?,
                    path: Box::default(),
                },
            },
        };
        let mut projected = Vec::new();
        for elem in place.path.iter() {
            match elem {
                Projection::Deref => {
                    panic!("unexpected deref in normalized runtime place: {place:?}")
                }
                Projection::Field(field) => projected.push(PlaceElem::Field(FieldIndex(
                    (*field).try_into().expect("field index fits in u16"),
                ))),
                Projection::VariantField { field_idx, .. } => {
                    projected.push(PlaceElem::VariantField(FieldIndex(
                        (*field_idx).try_into().expect("field index fits in u16"),
                    )));
                }
                Projection::Index(IndexSource::Dynamic(index)) => projected.push(PlaceElem::Index(
                    IndexSource::Dynamic(self.read_semantic_value(bb, *index)),
                )),
                Projection::Index(IndexSource::Constant(index)) => {
                    projected.push(PlaceElem::Index(IndexSource::Constant(*index)));
                }
                Projection::Discriminant => {
                    panic!("discriminant projections are not valid runtime places: {place:?}");
                }
            }
        }
        runtime_place.path = projected.into_boxed_slice();
        Some(runtime_place)
    }

    fn read_semantic_value(&mut self, bb: RBlockId, local: SLocalId) -> RLocalId {
        match self.semantic_local_lowering(local) {
            RuntimeLocalLowering::Erased => self.runtime_value(local),
            RuntimeLocalLowering::DirectValue => self.runtime_value(local),
            RuntimeLocalLowering::DirectCarrier { provider, .. } => provider.map_or_else(
                || self.runtime_value(local),
                |provider| self.provider_binding_value(provider),
            ),
            RuntimeLocalLowering::PlaceCarrier { place_class }
            | RuntimeLocalLowering::PlaceBoundValue { place_class, .. } => {
                let temp = self.alloc_runtime_temp(
                    self.locals[local.index()].semantic_ty,
                    RuntimeCarrier::Value(place_class.clone()),
                );
                let place = self.semantic_place(bb, local);
                self.push_stmt(
                    bb,
                    RStmt::Assign {
                        dst: temp,
                        expr: RExpr::Load { place },
                    },
                );
                temp
            }
        }
    }

    fn read_semantic_operand(&mut self, bb: RBlockId, operand: NOperand) -> RLocalId {
        self.read_semantic_value(bb, operand.local)
    }

    fn lower_semantic_operand_for_class(
        &mut self,
        bb: RBlockId,
        operand: NOperand,
        target: &RuntimeClass<'db>,
    ) -> RLocalId {
        if let Some(value) = self.handle_like_semantic_value(operand.local) {
            return if self.value_class(value) == Some(target) {
                value
            } else {
                self.coerce_value(bb, value, target)
            };
        }
        if runtime_target_prefers_transport(target)
            && let Some(value) = self.addr_of_semantic_operand_for_class(bb, operand, target)
        {
            return value;
        }
        let value = self.read_semantic_operand(bb, operand);
        if self.value_class(value).is_none() || self.value_class(value) == Some(target) {
            value
        } else {
            self.coerce_value(bb, value, target)
        }
    }

    fn addr_of_semantic_operand_for_class(
        &mut self,
        bb: RBlockId,
        operand: NOperand,
        target: &RuntimeClass<'db>,
    ) -> Option<RLocalId> {
        if !runtime_target_prefers_transport(target) {
            return None;
        }
        let local = operand.local;
        if !self.semantic_local_allows_transport_addr_of(local) {
            return None;
        }
        if let Some(place) = self.normalized_value_place(local).cloned()
            && !self.is_self_rooted_value_place(local, &place)
            && let Some(place) = self.try_lower_place(bb, &place)
        {
            return Some(self.lower_place_addr_of_for_class(local, bb, place, target.clone()));
        }
        if let Some(value) = self.handle_like_semantic_value(local) {
            return Some(if self.value_class(value) == Some(target) {
                value
            } else {
                self.coerce_value(bb, value, target)
            });
        }
        if matches!(
            self.semantic_local_lowering(local),
            RuntimeLocalLowering::DirectValue
                | RuntimeLocalLowering::PlaceCarrier { .. }
                | RuntimeLocalLowering::PlaceBoundValue { .. }
        ) && let Some(place) = self.try_semantic_place(bb, local)
        {
            return Some(self.lower_place_addr_of_for_class(local, bb, place, target.clone()));
        }
        None
    }

    fn semantic_local_allows_transport_addr_of(&self, local: SLocalId) -> bool {
        match &self.raw_semantic_body.locals[local.index()].role {
            hir::analysis::semantic::SemanticLocalRole::DirectValue {
                provenance: hir::analysis::semantic::ValueProvenance::Ordinary,
            }
            | hir::analysis::semantic::SemanticLocalRole::Erased => false,
            hir::analysis::semantic::SemanticLocalRole::DirectValue { .. }
            | hir::analysis::semantic::SemanticLocalRole::PlaceCarrier { .. }
            | hir::analysis::semantic::SemanticLocalRole::PlaceBoundValue { .. }
            | hir::analysis::semantic::SemanticLocalRole::DirectCarrier { .. } => true,
        }
    }

    fn handle_like_semantic_value(&self, local: SLocalId) -> Option<RLocalId> {
        let value = match self.semantic_local_lowering(local) {
            RuntimeLocalLowering::Erased => None,
            RuntimeLocalLowering::DirectValue | RuntimeLocalLowering::PlaceCarrier { .. } => {
                Some(self.runtime_value(local))
            }
            RuntimeLocalLowering::PlaceBoundValue { provider, .. } => {
                Some(self.provider_binding_value(*provider))
            }
            RuntimeLocalLowering::DirectCarrier { provider, .. } => Some(provider.map_or_else(
                || self.runtime_value(local),
                |provider| self.provider_binding_value(provider),
            )),
        }?;
        runtime_class_is_handle_like(self.value_class(value)?).then_some(value)
    }

    fn lower_place_addr_of_for_class(
        &mut self,
        local: SLocalId,
        bb: RBlockId,
        place: RuntimePlace<'db>,
        target: RuntimeClass<'db>,
    ) -> RLocalId {
        let temp = self.alloc_runtime_temp(
            self.locals[local.index()].semantic_ty,
            RuntimeCarrier::Value(target),
        );
        self.push_stmt(
            bb,
            RStmt::Assign {
                dst: temp,
                expr: RExpr::AddrOf { place },
            },
        );
        temp
    }

    fn write_semantic_value(&mut self, bb: RBlockId, local: SLocalId, src: RLocalId) {
        match self.semantic_local_lowering(local).clone() {
            RuntimeLocalLowering::Erased => {}
            RuntimeLocalLowering::DirectValue | RuntimeLocalLowering::PlaceCarrier { .. } => {
                let dst = self.runtime_value(local);
                let Some(target) = self.value_class(dst).cloned() else {
                    return;
                };
                let src = self.coerce_value(bb, src, &target);
                self.push_stmt(
                    bb,
                    RStmt::Assign {
                        dst,
                        expr: RExpr::Use(src),
                    },
                );
            }
            RuntimeLocalLowering::DirectCarrier { provider, .. } => {
                let dst = provider
                    .map(|provider| self.provider_binding_value(provider))
                    .unwrap_or_else(|| self.runtime_value(local));
                let Some(target) = self.value_class(dst).cloned() else {
                    return;
                };
                let src = self.coerce_value(bb, src, &target);
                self.push_stmt(
                    bb,
                    RStmt::Assign {
                        dst,
                        expr: RExpr::Use(src),
                    },
                );
            }
            RuntimeLocalLowering::PlaceBoundValue { place_class, .. } => {
                let place = self.semantic_place(bb, local);
                self.write_value_to_place(bb, place, src, &place_class);
            }
        }
    }

    fn lower_place(&mut self, bb: RBlockId, place: &NSPlace<'db>) -> RuntimePlace<'db> {
        self.try_lower_place(bb, place).unwrap_or_else(|| {
            let root_info = match place.root {
                NSPlaceRoot::CarrierDerefLocal(local) => format!(
                    "carrier_deref local={local:?} lowering={:?} root={:?}",
                    self.semantic_local_lowering(local),
                    self.locals[self.runtime_value(local).index()].root,
                ),
                NSPlaceRoot::Root(root) => match self.semantic_body.root(root) {
                    Some(NBorrowRoot::Param { local, param_idx }) => format!(
                        "param root={root:?} local={local:?} param_idx={param_idx} lowering={:?} runtime_root={:?}",
                        self.semantic_local_lowering(*local),
                        self.locals[self.runtime_value(*local).index()].root,
                    ),
                    Some(NBorrowRoot::LocalSlot { local }) => format!(
                        "local root={root:?} local={local:?} lowering={:?} runtime_root={:?}",
                        self.semantic_local_lowering(*local),
                        self.locals[self.runtime_value(*local).index()].root,
                    ),
                    Some(NBorrowRoot::Provider { binding }) => {
                        format!("provider root={root:?} binding={binding:?} provider_place_root={:?}", self.provider_place_root(binding))
                    }
                    None => format!("missing root {root:?}"),
                },
            };
            panic!(
                "cannot lower erased place root: place={place:?}; root_info={root_info}; locals={:?}",
                self.locals,
            )
        })
    }

    fn project_place_class(&self, place: &RuntimePlace<'db>) -> RuntimeClass<'db> {
        let mut current = match &place.root {
            PlaceRoot::Slot(local) => self
                .local_root_class_r(*local)
                .expect("projected places should have runtime root classes"),
            PlaceRoot::Handle(local) => self
                .value_class(*local)
                .cloned()
                .expect("projected handle places should have runtime classes"),
            PlaceRoot::Provider(binding) => self.provider_binding(*binding).place_class.clone(),
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
        if place.path.is_empty()
            && matches!(place.root, PlaceRoot::Handle(_) | PlaceRoot::Provider(_))
            && let RuntimeClass::Handle { layout, .. } = current
        {
            return RuntimeClass::AggregateValue { layout };
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
            .semantic_value_class(value)
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
                kind: kind.clone(),
                view: HandleView::EnumVariant(variant_id),
            }),
        );
        self.locals[temp.index()].root = RuntimeLocalRoot::Handle(RuntimeClass::Handle {
            layout,
            kind,
            view: HandleView::EnumVariant(variant_id),
        });
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
        self.locals.push(RLocal {
            semantic_ty,
            carrier,
            root: RuntimeLocalRoot::None,
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

    fn local_root(&self, local: SLocalId) -> Option<PlaceRoot<'db>> {
        self.local_root_r(self.runtime_value(local))
    }

    fn local_root_r(&self, local: RLocalId) -> Option<PlaceRoot<'db>> {
        match self.locals.get(local.index())?.root.clone() {
            RuntimeLocalRoot::None => None,
            RuntimeLocalRoot::Slot(_) => Some(PlaceRoot::Slot(local)),
            RuntimeLocalRoot::Handle(_) => Some(PlaceRoot::Handle(local)),
            RuntimeLocalRoot::Ptr { space, class } => Some(PlaceRoot::Ptr {
                addr: local,
                space,
                class,
            }),
        }
    }

    fn local_root_class_r(&self, local: RLocalId) -> Option<RuntimeClass<'db>> {
        match &self.locals.get(local.index())?.root {
            RuntimeLocalRoot::None => None,
            RuntimeLocalRoot::Slot(class)
            | RuntimeLocalRoot::Handle(class)
            | RuntimeLocalRoot::Ptr { class, .. } => Some(class.clone()),
        }
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

fn runtime_target_prefers_transport(class: &RuntimeClass<'_>) -> bool {
    matches!(
        class,
        RuntimeClass::Handle {
            kind: HandleKind::Provider { .. },
            ..
        } | RuntimeClass::RawAddr { .. }
    )
}

fn runtime_class_is_handle_like(class: &RuntimeClass<'_>) -> bool {
    matches!(
        class,
        RuntimeClass::Handle { .. } | RuntimeClass::RawAddr { .. }
    )
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
