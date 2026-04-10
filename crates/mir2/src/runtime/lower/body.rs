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
        get_or_build_semantic_instance, owner_effect_bindings, reify_runtime_const_for_ty,
        same_owner_effect_binding, sem_const_ty, semantic_may_return_normally,
    },
    ty::{
        corelib::lib_func_matches,
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
        AddressSpaceKind, BorrowAccess, ConstScalar, IntrinsicArithBinOp, LayoutId, PlaceElem,
        PlaceRoot, RBlock, RBlockId, RExpr, RLocal, RLocalId, RStmt, RTerminator, RefKind, RefView,
        RuntimeBody, RuntimeCarrier, RuntimeClass, RuntimeCodeRegion, RuntimeLocalLowering,
        RuntimeLocalRoot, RuntimePlace, RuntimeProviderBinding, RuntimeProviderBindingId,
        RuntimeSignature, ScalarClass, ScalarRepr, ScalarRole, VariantId,
        code_region::runtime_code_region_for_semantic_ref, runtime_classes_share_runtime_rep,
    },
};

use super::{
    class::{
        ContractMetadataBuiltin, GenericNumericIntrinsicKind, InferredRuntimeLocal,
        actual_aggregate_class_from_runtime_source,
        boundary_source_uses_transport_sensitive_aggregate, boundary_spec_for_ty_in_env,
        contract_metadata_builtin, default_borrow_transport_set, desired_runtime_param_plan,
        expr_direct_class, generic_numeric_intrinsic_kind, infer_local_runtime_state,
        lower_semantic_locals, normalized_place_address_class, normalized_place_class,
        provider_class_for_target_in_env, ref_class_for_place_result, resolve_runtime_call_key,
        runtime_address_space, runtime_class_satisfies_boundary, runtime_param_locals,
        runtime_signature_for_key, semantic_return_ty, specialize_boundary_for_runtime_source,
        stored_class_for_ty_in_context, top_level_class_for_ty_in_env,
    },
    consts::{
        const_scalar_for_class, const_scalar_from_value, enum_tag_scalar, lower_const_region,
    },
    layout::{
        AggregateCtorElem, RuntimeTypeEnv, aggregate_ctor_elems_for_layout,
        layout_for_aggregate_instance_in_env, layout_for_ty_in_env,
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
    let normalized_body = normalize_semantic_body(db, semantic).unwrap_or_else(|err| {
        panic!(
            "semantic normalization failed for {:?}: {err:?}",
            semantic.key(db)
        )
    });
    let local_state = infer_local_runtime_state(
        db,
        &normalized_body,
        key.params(db),
        &runtime_param_locals(db, semantic, key.params(db)),
        semantic.key(db).owner(db).scope().into(),
        semantic.key(db).instantiate_typed_body(db).assumptions(),
    );
    let (semantic_locals, provider_bindings) = lower_semantic_locals(
        db,
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

fn expr_requires_runtime_eval_when_erased(expr: &NExpr<'_>) -> bool {
    match expr {
        NExpr::Unary {
            op: UnOp::Minus, ..
        }
        | NExpr::Binary {
            op:
                BinOp::Arith(
                    ArithBinOp::Add
                    | ArithBinOp::Sub
                    | ArithBinOp::Mul
                    | ArithBinOp::Div
                    | ArithBinOp::Rem
                    | ArithBinOp::Pow,
                ),
            ..
        }
        | NExpr::Call { .. } => true,
        NExpr::Use(_)
        | NExpr::ReadPlace { .. }
        | NExpr::Const(_)
        | NExpr::Unary { .. }
        | NExpr::Binary { .. }
        | NExpr::Cast { .. }
        | NExpr::AggregateMake { .. }
        | NExpr::EnumMake { .. }
        | NExpr::Borrow { .. }
        | NExpr::GetEnumTag { .. }
        | NExpr::IsEnumVariant { .. }
        | NExpr::ExtractEnumField { .. }
        | NExpr::CodeRegionOffset { .. }
        | NExpr::CodeRegionLen { .. } => false,
    }
}

pub(super) struct RmirLowerCtxt<'db> {
    pub(super) db: &'db dyn MirDb,
    pub(super) instance: RuntimeInstance<'db>,
    pub(super) key: RuntimeInstanceKey<'db>,
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

enum BoundarySource<'db> {
    SemanticOperand(NOperand),
    RuntimePlace(RuntimePlace<'db>),
}

impl<'db> RmirLowerCtxt<'db> {
    fn new(
        db: &'db dyn MirDb,
        instance: RuntimeInstance<'db>,
        key: RuntimeInstanceKey<'db>,
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
                if !expr_requires_runtime_eval_when_erased(expr) {
                    return;
                }
                let carrier = expr_direct_class(
                    self.db,
                    &self.semantic_body,
                    expr,
                    self.semantic_body.locals[dst.index()].ty,
                    &self.current_runtime_carriers(),
                )
                .map(RuntimeCarrier::Value)
                .unwrap_or(RuntimeCarrier::Erased);
                let sink = self.alloc_runtime_temp(
                    self.semantic_body.locals[dst.index()].ty,
                    RuntimeCarrier::Erased,
                );
                self.locals[sink.index()].carrier = carrier;
                self.lower_expr_into(bb, sink, expr);
            }
            Some(desired) => {
                if self.semantic_local_is_direct(dst) {
                    self.specialize_direct_assign_target_from_expr(dst, expr);
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

    fn current_runtime_carriers(&self) -> Vec<RuntimeCarrier<'db>> {
        self.locals
            .iter()
            .map(|local| local.carrier.clone())
            .collect()
    }

    fn direct_assign_source_class(&self, expr: &NExpr<'db>) -> Option<RuntimeClass<'db>> {
        match expr {
            NExpr::Use(value) => self
                .value_class(self.runtime_value(value.local))
                .cloned()
                .or_else(|| self.semantic_value_class(value.local)),
            NExpr::Borrow { place, .. } => normalized_place_address_class(
                self.db,
                &self.semantic_body,
                place,
                &self.current_runtime_carriers(),
                self.env.scope,
                self.env.assumptions,
            ),
            NExpr::ReadPlace { place, .. } => normalized_place_class(
                self.db,
                &self.semantic_body,
                place,
                &self.current_runtime_carriers(),
            ),
            NExpr::Const(_)
            | NExpr::Unary { .. }
            | NExpr::Binary { .. }
            | NExpr::Cast { .. }
            | NExpr::AggregateMake { .. }
            | NExpr::EnumMake { .. }
            | NExpr::GetEnumTag { .. }
            | NExpr::IsEnumVariant { .. }
            | NExpr::ExtractEnumField { .. }
            | NExpr::CodeRegionOffset { .. }
            | NExpr::CodeRegionLen { .. }
            | NExpr::Call { .. } => None,
        }
    }

    fn specialize_direct_assign_target_from_expr(&mut self, dst: SLocalId, expr: &NExpr<'db>) {
        let dst_value = self.runtime_value(dst);
        let Some(target) = self.value_class(dst_value).cloned() else {
            return;
        };
        if matches!(
            self.semantic_local_lowering(dst),
            RuntimeLocalLowering::PlaceCarrier { .. }
        ) {
            let actual = self.direct_assign_source_class(expr);
            if let Some(actual) = actual
                && runtime_class_is_handle_like(&actual)
            {
                self.refine_local_runtime_class(dst_value, actual);
                return;
            }
        }
        if !matches!(
            target,
            RuntimeClass::AggregateValue { .. }
                | RuntimeClass::Ref {
                    kind: RefKind::Object,
                    ..
                }
        ) {
            return;
        }
        let source = match expr {
            NExpr::Use(value) => self.semantic_value_class(value.local),
            _ => None,
        };
        let Some(actual) = source
            .as_ref()
            .and_then(actual_aggregate_class_from_runtime_source)
        else {
            return;
        };
        let target = match target {
            RuntimeClass::AggregateValue { .. } => actual,
            RuntimeClass::Ref {
                kind: RefKind::Object,
                ..
            } => RuntimeClass::object_ref(actual.aggregate_layout().expect("aggregate ref layout")),
            RuntimeClass::Ref { .. } | RuntimeClass::Scalar(_) | RuntimeClass::RawAddr { .. } => {
                unreachable!()
            }
        };
        self.refine_local_runtime_class(dst_value, target);
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
                let dst_class = self.specialize_runtime_target_from_operand(dst, *src, &dst_class);
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
                let dst_class = if self.runtime_local_uses_source_transport(dst)
                    && runtime_class_is_handle_like(&projected)
                {
                    self.refine_local_runtime_class(dst, projected.clone());
                    projected.clone()
                } else {
                    dst_class
                };
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
                let actual = self.place_addr_class(&place);
                let dst_class = if self.runtime_local_uses_source_transport(dst)
                    && runtime_class_is_handle_like(&actual)
                {
                    self.refine_local_runtime_class(dst, actual.clone());
                    actual
                } else {
                    dst_class
                };
                let value = self.lower_place_addr_of_for_class(
                    self.locals[dst.index()].semantic_ty,
                    bb,
                    place,
                    dst_class,
                );
                self.push_stmt(
                    bb,
                    RStmt::Assign {
                        dst,
                        expr: RExpr::Use(value),
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
                let expected_ty =
                    self.const_lowering_ty(self.locals[dst.index()].semantic_ty, &target);
                self.lower_sem_const_for_class(bb, dst, *value, expected_ty, &target);
            }
            SConst::Ref(cref) => panic!("unresolved const ref reached rMIR lowering: {cref:?}"),
        }
    }

    fn lower_sem_const_for_class(
        &mut self,
        bb: RBlockId,
        dst: RLocalId,
        value: SemConstId<'db>,
        expected_ty: TyId<'db>,
        target: &RuntimeClass<'db>,
    ) {
        let value = self.reify_runtime_const(expected_ty, value);
        let src = self.lower_sem_const_as_class(bb, value, expected_ty, target);
        let actual = self.value_class(src).cloned();
        let value = if self.value_class(src) == Some(target) {
            src
        } else if actual
            .as_ref()
            .is_some_and(|actual| self.should_preserve_const_source_class(actual, target))
        {
            self.refine_local_runtime_class(
                dst,
                actual.expect("actual runtime class should be present"),
            );
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
        expected_ty: TyId<'db>,
        target: &RuntimeClass<'db>,
    ) -> RLocalId {
        let expected_ty = self.const_lowering_ty(expected_ty, target);
        let value = self.reify_runtime_const(expected_ty, value);
        let ty = expected_ty;
        if matches!(
            target,
            RuntimeClass::Ref {
                kind: RefKind::Const,
                ..
            }
        ) {
            return self.lower_sem_const_as_const_handle(bb, value, ty);
        }
        if let Some(scalar) = const_scalar_from_value(self.db, self.env, value) {
            return self.lower_sem_const_scalar(bb, ty, scalar);
        }
        if let RuntimeClass::Scalar(class) = target
            && let SemConstValue::Scalar { value, .. } = value.value(self.db)
            && let Some(scalar) = const_scalar_for_class(&value, class)
        {
            return self.lower_sem_const_scalar(bb, ty, scalar);
        }
        match target {
            RuntimeClass::Scalar(_) => {
                panic!(
                    "non-scalar semantic const {value:?} cannot lower to scalar class {target:?}"
                )
            }
            RuntimeClass::Ref {
                pointee,
                kind: RefKind::Object,
                view: RefView::Whole,
            } => {
                let layout = pointee
                    .aggregate_layout()
                    .expect("object ref const target should have aggregate layout");
                self.lower_non_scalar_const_as_object(bb, value, ty, layout)
            }
            RuntimeClass::Ref { .. } => {
                panic!(
                    "non-scalar semantic const {value:?} cannot lower directly to ref class {target:?}"
                )
            }
            RuntimeClass::AggregateValue { layout: _ } => {
                let src = self.lower_sem_const_as_value(bb, value, expected_ty);
                let actual = self.value_class(src).cloned();
                if self.value_class(src) == Some(target)
                    || actual.as_ref().is_some_and(|actual| {
                        self.should_preserve_const_source_class(actual, target)
                    })
                {
                    src
                } else {
                    self.coerce_value(bb, src, target)
                }
            }
            RuntimeClass::RawAddr { .. } => {
                let layout = self.layout_for_ty(ty);
                self.lower_non_scalar_const_as_object(bb, value, ty, layout)
            }
        }
    }

    fn lower_sem_const_as_value(
        &mut self,
        bb: RBlockId,
        value: SemConstId<'db>,
        expected_ty: TyId<'db>,
    ) -> RLocalId {
        let value = self.reify_runtime_const(expected_ty, value);
        let ty = expected_ty;
        if let Some(scalar) = const_scalar_from_value(self.db, self.env, value) {
            return self.lower_sem_const_scalar(bb, ty, scalar);
        }
        if let SemConstValue::Scalar { value, .. } = value.value(self.db)
            && let Some(scalar) = const_scalar_for_class(&value, &word_scalar_class())
        {
            return self.lower_sem_const_scalar_with_class(
                bb,
                ty,
                RuntimeClass::Scalar(word_scalar_class()),
                scalar,
            );
        }
        let layout = self.layout_for_ty(ty);
        match value.value(self.db) {
            SemConstValue::Tuple { .. }
            | SemConstValue::Struct { .. }
            | SemConstValue::Array { .. } => {
                self.lower_non_scalar_const_as_aggregate_value(bb, value, ty, layout)
            }
            SemConstValue::Enum { .. } => {
                self.lower_non_scalar_const_as_aggregate_value(bb, value, ty, layout)
            }
            SemConstValue::Unit
            | SemConstValue::Scalar { .. }
            | SemConstValue::TypeLevel { .. } => {
                panic!("semantic const should lower as a natural runtime value: {value:?}")
            }
        }
    }

    fn lower_sem_const_as_const_handle(
        &mut self,
        bb: RBlockId,
        value: SemConstId<'db>,
        ty: TyId<'db>,
    ) -> RLocalId {
        let value = self.reify_runtime_const(ty, value);
        let region = lower_const_region(self.db, self.env, value).unwrap_or_else(|| {
            panic!("const-backed handle should lower to a const region: {value:?}")
        });
        let layout = region.layout(self.db);
        let local =
            self.alloc_runtime_temp(ty, RuntimeCarrier::Value(RuntimeClass::const_ref(layout)));
        self.push_stmt(
            bb,
            RStmt::Assign {
                dst: local,
                expr: RExpr::ConstRef { region, layout },
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
        self.lower_sem_const_scalar_with_class(bb, ty, class, scalar)
    }

    fn lower_sem_const_scalar_with_class(
        &mut self,
        bb: RBlockId,
        ty: TyId<'db>,
        class: RuntimeClass<'db>,
        scalar: ConstScalar,
    ) -> RLocalId {
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
        debug_assert!(!matches!(
            value.value(self.db),
            SemConstValue::Scalar { .. }
        ));
        let ty = self.const_lowering_ty(ty, &RuntimeClass::AggregateValue { layout });
        match value.value(self.db) {
            SemConstValue::Tuple { elems, .. }
            | SemConstValue::Struct { fields: elems, .. }
            | SemConstValue::Array { elems, .. } => {
                let field_tys = if ty.is_array(self.db) {
                    let (_, args) = ty.decompose_ty_app(self.db);
                    let elem_ty = args.first().copied().expect("array element type");
                    vec![elem_ty; elems.len()]
                } else {
                    ty.field_types(self.db)
                };
                let field_values = elems
                    .iter()
                    .copied()
                    .zip(field_tys)
                    .map(|(field, field_ty)| self.lower_sem_const_as_value(bb, field, field_ty))
                    .collect::<Vec<_>>();
                let field_classes = field_values
                    .iter()
                    .map(|value| {
                        self.value_class(*value)
                            .cloned()
                            .expect("aggregate const field should have a runtime class")
                    })
                    .collect::<Vec<_>>();
                let layout =
                    layout_for_aggregate_instance_in_env(self.db, self.env, ty, &field_classes);
                let ctor_elems = aggregate_ctor_elems_for_layout(self.db, layout, elems.len());
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
                let field_tys = ty
                    .as_enum(self.db)
                    .expect("enum constant should have an enum type")
                    .variants(self.db)
                    .nth(variant.0 as usize)
                    .expect("enum variant index should resolve")
                    .field_tys(self.db)
                    .into_iter()
                    .map(|field| field.instantiate(self.db, ty.generic_args(self.db)))
                    .collect::<Vec<_>>();
                let field_values = fields
                    .iter()
                    .copied()
                    .zip(field_tys)
                    .zip(layout_data.variants[variant.0 as usize].fields.iter())
                    .map(|((field, field_ty), field_class)| {
                        self.lower_sem_const_as_class(bb, field, field_ty, field_class)
                    })
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
        debug_assert!(!matches!(
            value.value(self.db),
            SemConstValue::Scalar { .. }
        ));
        let ty = self.const_lowering_ty(ty, &RuntimeClass::AggregateValue { layout });
        match value.value(self.db) {
            SemConstValue::Tuple { elems, .. }
            | SemConstValue::Struct { fields: elems, .. }
            | SemConstValue::Array { elems, .. } => {
                let field_tys = if ty.is_array(self.db) {
                    let (_, args) = ty.decompose_ty_app(self.db);
                    let elem_ty = args.first().copied().expect("array element type");
                    vec![elem_ty; elems.len()]
                } else {
                    ty.field_types(self.db)
                };
                let field_values = elems
                    .iter()
                    .copied()
                    .zip(field_tys)
                    .map(|(field, field_ty)| self.lower_sem_const_as_value(bb, field, field_ty))
                    .collect::<Vec<_>>();
                let field_classes = field_values
                    .iter()
                    .map(|value| {
                        self.value_class(*value)
                            .cloned()
                            .expect("aggregate const field should have a runtime class")
                    })
                    .collect::<Vec<_>>();
                let layout =
                    layout_for_aggregate_instance_in_env(self.db, self.env, ty, &field_classes);
                let ctor_elems = aggregate_ctor_elems_for_layout(self.db, layout, elems.len());
                let dst = self.alloc_runtime_temp(
                    ty,
                    RuntimeCarrier::Value(RuntimeClass::object_ref(layout)),
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
                let field_tys = ty
                    .as_enum(self.db)
                    .expect("enum constant should have an enum type")
                    .variants(self.db)
                    .nth(variant.0 as usize)
                    .expect("enum variant index should resolve")
                    .field_tys(self.db)
                    .into_iter()
                    .map(|field| field.instantiate(self.db, ty.generic_args(self.db)))
                    .collect::<Vec<_>>();
                let field_values = fields
                    .iter()
                    .copied()
                    .zip(field_tys)
                    .zip(layout_data.variants[variant.0 as usize].fields.iter())
                    .map(|((field, field_ty), field_class)| {
                        self.lower_sem_const_as_class(bb, field, field_ty, field_class)
                    })
                    .collect::<Vec<_>>();
                let dst = self.alloc_runtime_temp(
                    ty,
                    RuntimeCarrier::Value(RuntimeClass::object_ref(layout)),
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
        let field_tys = if ty.is_array(self.db) {
            let (_, args) = ty.decompose_ty_app(self.db);
            let elem_ty = args.first().copied().expect("array element type");
            vec![elem_ty; fields.len()]
        } else {
            ty.field_types(self.db)
        };
        assert_eq!(
            field_tys.len(),
            fields.len(),
            "aggregate constructor arity mismatch for {}",
            ty.pretty_print(self.db),
        );
        let mut field_values = Vec::with_capacity(fields.len());
        let mut field_classes = Vec::with_capacity(fields.len());
        for (field, field_ty) in fields.iter().copied().zip(field_tys.iter().copied()) {
            let value = match boundary_spec_for_ty_in_env(
                self.db,
                self.env,
                field_ty,
                AddressSpaceKind::Memory,
            ) {
                Some(boundary) => self.lower_semantic_operand_for_boundary(bb, field, &boundary),
                None => {
                    let class = self
                        .top_level_class_for_ty(field_ty, AddressSpaceKind::Memory)
                        .expect("non-zst aggregate field should have a runtime class");
                    self.lower_semantic_operand_for_class(bb, field, &class)
                }
            };
            let class = self.value_class(value).cloned().unwrap_or_else(|| {
                stored_class_for_ty_in_context(
                    self.db,
                    field_ty,
                    self.env.scope,
                    self.env.assumptions,
                )
            });
            field_values.push(value);
            field_classes.push(class);
        }
        let layout = layout_for_aggregate_instance_in_env(self.db, self.env, ty, &field_classes);
        let ctor_elems = aggregate_ctor_elems_for_layout(self.db, layout, fields.len());
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
        let dst_class = match dst_class {
            RuntimeClass::AggregateValue { .. } => RuntimeClass::AggregateValue { layout },
            RuntimeClass::Ref {
                kind: RefKind::Object,
                ..
            } => RuntimeClass::object_ref(layout),
            RuntimeClass::RawAddr { space, target } => RuntimeClass::RawAddr { space, target },
            RuntimeClass::Ref {
                kind: RefKind::Provider { provider_ty, space },
                pointee,
                ..
            } => {
                let target_layout = pointee.aggregate_layout().expect("provider ref layout");
                RuntimeClass::provider_ref(target_layout, provider_ty, space)
            }
            RuntimeClass::Ref {
                kind: RefKind::Const,
                pointee,
                ..
            } => RuntimeClass::const_ref(pointee.aggregate_layout().expect("const ref layout")),
            RuntimeClass::Scalar(_) => panic!("aggregate destination must not be scalar"),
        };
        self.refine_local_runtime_class(dst, dst_class.clone());
        match dst_class {
            RuntimeClass::AggregateValue { .. } => {
                let temp = self.alloc_runtime_temp(
                    ty,
                    RuntimeCarrier::Value(RuntimeClass::object_ref(layout)),
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
                        root: PlaceRoot::Ref(temp),
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
                                root: PlaceRoot::Ref(temp),
                                path: Box::default(),
                            },
                        },
                    },
                );
            }
            RuntimeClass::Ref {
                kind: RefKind::Object,
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
                        root: PlaceRoot::Ref(dst),
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
            RuntimeClass::Ref {
                pointee,
                kind: RefKind::Provider { provider_ty, space },
                ..
            } => {
                let target_layout = pointee.aggregate_layout().expect("provider ref layout");
                let [value] = field_values else {
                    panic!("provider aggregate construction requires exactly one field");
                };
                let value = self.coerce_value(
                    bb,
                    *value,
                    &RuntimeClass::provider_ref(target_layout, provider_ty, space),
                );
                self.push_stmt(
                    bb,
                    RStmt::Assign {
                        dst,
                        expr: RExpr::Use(value),
                    },
                );
            }
            RuntimeClass::Ref {
                kind: RefKind::Const,
                ..
            } => {
                panic!("aggregate construction must not target const refs");
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
            RuntimeClass::Ref {
                kind: RefKind::Object,
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
                "enum construction requires aggregate or object-ref destination, found {class:?}"
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
            Some(RuntimeClass::Ref { .. }) => {
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
            Some(RuntimeClass::Ref { .. }) => {
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
            Some(RuntimeClass::Ref { .. }) => {
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
            &self.semantic_body,
            callee,
            args,
        )
        .unwrap_or_else(|err| {
            panic!(
                "runtime call resolution failed while lowering {:?}: {err}",
                self.key
                    .semantic(self.db)
                    .map(|semantic| semantic.key(self.db)),
            )
        });
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
            RuntimeClass::Ref { .. } => return None,
            RuntimeClass::AggregateValue { .. } | RuntimeClass::RawAddr { .. } => return None,
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
            match desired_runtime_param_plan(self.db, typed_body, idx) {
                crate::runtime::RuntimeParamPlan::Erased => {}
                crate::runtime::RuntimeParamPlan::Boundary(desired) => {
                    let value = self.runtime_visible_arg_value(bb, *arg, Some(&desired));
                    let Some(class) = self.value_class(value).cloned() else {
                        continue;
                    };
                    runtime_classes.push(class);
                    runtime_args.push(value);
                }
                crate::runtime::RuntimeParamPlan::PassActual => {
                    let value = self.runtime_visible_arg_value(bb, *arg, None);
                    let Some(class) = self.value_class(value).cloned() else {
                        continue;
                    };
                    runtime_classes.push(class);
                    runtime_args.push(value);
                }
            }
        }
        (runtime_args, runtime_classes)
    }

    fn runtime_visible_arg_value(
        &mut self,
        bb: RBlockId,
        arg: NOperand,
        desired: Option<&crate::runtime::RuntimeBoundarySpec<'db>>,
    ) -> RLocalId {
        if let Some(desired) = desired {
            self.lower_semantic_operand_for_boundary(bb, arg, desired)
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
                                RuntimeLocalLowering::PlaceBoundValue {
                                    provider: Some(provider),
                                    ..
                                }
                                | RuntimeLocalLowering::DirectCarrier {
                                    provider: Some(provider),
                                    ..
                                } => Some((
                                    value,
                                    self.provider_binding(*provider).provider_class.clone(),
                                )),
                                RuntimeLocalLowering::Erased
                                | RuntimeLocalLowering::DirectValue
                                | RuntimeLocalLowering::PlaceBoundValue {
                                    provider: None, ..
                                }
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
        let matches = |path: &str| lib_func_matches(self.db, func, path);

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
        if !lib_func_matches(self.db, func, "core::intrinsic::__keccak256") {
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
                RuntimeClass::Ref {
                    kind:
                        RefKind::Provider {
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
        if arg.provider.is_none() && arg.target_ty.is_none() {
            return match (&arg.pass_mode, &arg.arg) {
                (
                    EffectPassMode::ByValue | EffectPassMode::Unknown,
                    NEffectArgValue::Value(value),
                ) => {
                    let value = self.read_semantic_operand(bb, *value);
                    self.value_class(value).cloned().map(|class| (value, class))
                }
                (EffectPassMode::ByValue | EffectPassMode::Unknown, NEffectArgValue::Place(_))
                | (
                    EffectPassMode::ByPlace | EffectPassMode::ByTempPlace,
                    NEffectArgValue::Value(_) | NEffectArgValue::Place(_),
                ) => panic!(
                    "effect arg without provider/target should lower as a plain value: owner={:?}; arg={arg:?}",
                    self.key
                        .semantic(self.db)
                        .map(|semantic| semantic.key(self.db).owner(self.db)),
                ),
            };
        }
        let space = resolved_address_space(arg.provider);
        let boundary = arg.target_ty.map(|target_ty| match arg.pass_mode {
            EffectPassMode::ByPlace | EffectPassMode::ByTempPlace => {
                crate::runtime::RuntimeBoundarySpec::BorrowLike {
                    pointee: stored_class_for_ty_in_context(
                        self.db,
                        target_ty,
                        self.env.scope,
                        self.env.assumptions,
                    ),
                    access: BorrowAccess::ReadWrite,
                    allow: default_borrow_transport_set(BorrowAccess::ReadWrite, space),
                }
            }
            EffectPassMode::ByValue | EffectPassMode::Unknown => boundary_spec_for_ty_in_env(
                self.db, self.env, target_ty, space,
            )
            .unwrap_or(crate::runtime::RuntimeBoundarySpec::Exact(
                provider_class_for_target_in_env(self.db, self.env, Some(target_ty), space),
            )),
        });
        match arg.pass_mode {
            EffectPassMode::ByValue | EffectPassMode::Unknown => match &arg.arg {
                NEffectArgValue::Value(value) => {
                    let value = if let Some(boundary) = boundary.as_ref() {
                        self.lower_semantic_operand_for_boundary(bb, *value, boundary)
                    } else {
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
                            space,
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
                        return Some((placeholder, class));
                    };
                    self.value_class(value).cloned().map(|class| (value, class))
                }
                NEffectArgValue::Place(place) => {
                    if let Some(place) = self.try_lower_place(bb, place) {
                        let boundary = boundary.as_ref().cloned().unwrap_or_else(|| {
                            crate::runtime::RuntimeBoundarySpec::Exact(
                                provider_class_for_target_in_env(
                                    self.db,
                                    self.env,
                                    arg.target_ty,
                                    space,
                                ),
                            )
                        });
                        let value = self.lower_for_boundary(
                            bb,
                            BoundarySource::RuntimePlace(place),
                            &boundary,
                            arg.target_ty.unwrap_or_else(|| TyId::unit(self.db)),
                        );
                        self.value_class(value).cloned().map(|class| (value, class))
                    } else {
                        assert!(
                            place.path.is_empty(),
                            "erased capability place effect args cannot have projections"
                        );
                        let class = boundary
                            .as_ref()
                            .map_or_else(
                                || {
                                    provider_class_for_target_in_env(
                                        self.db,
                                        self.env,
                                        arg.target_ty,
                                        space,
                                    )
                                },
                                |boundary| match boundary {
                                    crate::runtime::RuntimeBoundarySpec::Exact(class) => {
                                        class.clone()
                                    }
                                    crate::runtime::RuntimeBoundarySpec::BorrowLike {
                                        pointee,
                                        allow,
                                        ..
                                    } if pointee.aggregate_layout().is_some()
                                        && allow.allow_object =>
                                    {
                                        RuntimeClass::object_ref(
                                            pointee.aggregate_layout().expect("aggregate layout"),
                                        )
                                    }
                                    crate::runtime::RuntimeBoundarySpec::BorrowLike {
                                        allow, ..
                                    } if allow.allow_raw_addr => RuntimeClass::RawAddr {
                                        space: AddressSpaceKind::Memory,
                                        target: None,
                                    },
                                    crate::runtime::RuntimeBoundarySpec::BorrowLike { .. } => {
                                        panic!(
                                            "erased effect arg place has no realizable borrow-like transport: owner={:?}; arg={arg:?}; boundary={boundary:?}",
                                            self.key
                                                .semantic(self.db)
                                                .map(|semantic| semantic.key(self.db).owner(self.db)),
                                        )
                                    }
                                },
                            );
                        let temp = self.alloc_runtime_temp(
                            arg.target_ty.unwrap_or_else(|| TyId::unit(self.db)),
                            RuntimeCarrier::Value(class.clone()),
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
                        Some((temp, class))
                    }
                }
            },
            EffectPassMode::ByPlace | EffectPassMode::ByTempPlace => {
                let target_ty = arg.target_ty.unwrap_or_else(|| TyId::unit(self.db));
                let boundary =
                    boundary
                        .clone()
                        .unwrap_or(crate::runtime::RuntimeBoundarySpec::Exact(
                            provider_class_for_target_in_env(
                                self.db,
                                self.env,
                                Some(target_ty),
                                space,
                            ),
                        ));
                let value = match (&arg.arg, arg.pass_mode) {
                    (NEffectArgValue::Place(place), _) => {
                        if let Some(place) = self.try_lower_place(bb, place) {
                            self.lower_for_boundary(
                                bb,
                                BoundarySource::RuntimePlace(place),
                                &boundary,
                                target_ty,
                            )
                        } else {
                            let NEffectArgValue::Place(place) = &arg.arg else {
                                unreachable!();
                            };
                            assert!(
                                place.path.is_empty(),
                                "erased capability place effect args cannot have projections"
                            );
                            let class = match &boundary {
                                crate::runtime::RuntimeBoundarySpec::Exact(class) => class.clone(),
                                crate::runtime::RuntimeBoundarySpec::BorrowLike {
                                    pointee,
                                    allow,
                                    ..
                                } if pointee.aggregate_layout().is_some() && allow.allow_object => {
                                    RuntimeClass::object_ref(
                                        pointee.aggregate_layout().expect("aggregate layout"),
                                    )
                                }
                                crate::runtime::RuntimeBoundarySpec::BorrowLike {
                                    allow, ..
                                } if allow.allow_raw_addr => RuntimeClass::RawAddr {
                                    space: AddressSpaceKind::Memory,
                                    target: None,
                                },
                                crate::runtime::RuntimeBoundarySpec::BorrowLike { .. } => {
                                    panic!(
                                        "erased effect arg place has no realizable borrow-like transport: owner={:?}; arg={arg:?}; boundary={boundary:?}",
                                        self.key
                                            .semantic(self.db)
                                            .map(|semantic| semantic.key(self.db).owner(self.db)),
                                    )
                                }
                            };
                            let temp = self.alloc_runtime_temp(
                                target_ty,
                                RuntimeCarrier::Value(class.clone()),
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
                            temp
                        }
                    }
                    (NEffectArgValue::Value(value), EffectPassMode::ByTempPlace) => {
                        self.lower_semantic_operand_for_boundary(bb, *value, &boundary)
                    }
                    _ => panic!("invalid effect arg lowering mode"),
                };
                self.value_class(value).cloned().map(|class| (value, class))
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
            RuntimeClass::Scalar(_) | RuntimeClass::Ref { .. } | RuntimeClass::RawAddr { .. } => {
                let src = self.coerce_value(bb, src, target);
                self.push_stmt(bb, RStmt::Store { dst, src });
            }
            RuntimeClass::AggregateValue { .. } => {
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
                RuntimeClass::Ref {
                    pointee: actual_pointee,
                    kind: actual_kind,
                    view: actual_view,
                },
                RuntimeClass::Ref {
                    pointee: desired_pointee,
                    kind: desired_kind,
                    view: desired_view,
                },
            ) if actual_view == desired_view
                && runtime_classes_share_runtime_rep(
                    self.db,
                    &RuntimeClass::Ref {
                        pointee: actual_pointee.clone(),
                        kind: actual_kind.clone(),
                        view: actual_view.clone(),
                    },
                    &RuntimeClass::Ref {
                        pointee: desired_pointee.clone(),
                        kind: desired_kind.clone(),
                        view: desired_view.clone(),
                    },
                ) =>
            {
                let temp = self.alloc_runtime_temp(
                    self.locals[src.index()].semantic_ty,
                    RuntimeCarrier::Value(target.clone()),
                );
                self.push_stmt(
                    bb,
                    RStmt::Assign {
                        dst: temp,
                        expr: RExpr::RetagRef { value: src },
                    },
                );
                temp
            }
            (RuntimeClass::Ref { pointee, .. }, target)
                if !runtime_target_prefers_transport(&target) =>
            {
                let loaded = self.alloc_runtime_temp(
                    self.locals[src.index()].semantic_ty,
                    RuntimeCarrier::Value((*pointee).clone()),
                );
                self.push_stmt(
                    bb,
                    RStmt::Assign {
                        dst: loaded,
                        expr: RExpr::Load {
                            place: RuntimePlace {
                                root: PlaceRoot::Ref(src),
                                path: Box::default(),
                            },
                        },
                    },
                );
                if *pointee == target {
                    loaded
                } else {
                    self.coerce_value(bb, loaded, &target)
                }
            }
            (
                RuntimeClass::Ref {
                    pointee,
                    kind: RefKind::Object,
                    view: RefView::Whole,
                },
                RuntimeClass::Ref {
                    pointee: desired_pointee,
                    view: RefView::Whole,
                    ..
                },
            ) if pointee == desired_pointee => {
                let actual = RuntimeClass::Ref {
                    pointee,
                    kind: RefKind::Object,
                    view: RefView::Whole,
                };
                let temp = self.alloc_runtime_temp(
                    self.locals[src.index()].semantic_ty,
                    RuntimeCarrier::Value(actual.clone()),
                );
                self.push_stmt(
                    bb,
                    RStmt::Assign {
                        dst: temp,
                        expr: RExpr::AddrOf {
                            place: RuntimePlace {
                                root: PlaceRoot::Ref(src),
                                path: Box::default(),
                            },
                        },
                    },
                );
                if actual == *target {
                    temp
                } else {
                    self.coerce_value(bb, temp, target)
                }
            }
            (
                RuntimeClass::Ref {
                    pointee,
                    kind: RefKind::Const,
                    view: RefView::Whole,
                },
                RuntimeClass::Ref {
                    pointee: desired_pointee,
                    view: RefView::Whole,
                    ..
                },
            ) if pointee == desired_pointee => {
                let actual = RuntimeClass::Ref {
                    pointee,
                    kind: RefKind::Const,
                    view: RefView::Whole,
                };
                let temp = self.alloc_runtime_temp(
                    self.locals[src.index()].semantic_ty,
                    RuntimeCarrier::Value(actual.clone()),
                );
                self.push_stmt(
                    bb,
                    RStmt::Assign {
                        dst: temp,
                        expr: RExpr::AddrOf {
                            place: RuntimePlace {
                                root: PlaceRoot::Ref(src),
                                path: Box::default(),
                            },
                        },
                    },
                );
                if actual == *target {
                    temp
                } else {
                    self.coerce_value(bb, temp, target)
                }
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
                    view: crate::runtime::RefView::Whole,
                },
            ) if space == provider_space && *pointee == RuntimeClass::AggregateValue { layout } => {
                let temp = self.alloc_runtime_temp(
                    self.locals[src.index()].semantic_ty,
                    RuntimeCarrier::Value(target.clone()),
                );
                self.push_stmt(
                    bb,
                    RStmt::Assign {
                        dst: temp,
                        expr: RExpr::ProviderFromRaw {
                            raw: src,
                            provider_ty,
                            space,
                            target: Some(layout),
                        },
                    },
                );
                temp
            }
            (
                RuntimeClass::Ref {
                    pointee,
                    kind: RefKind::Object | RefKind::Const,
                    view: RefView::Whole,
                },
                RuntimeClass::Ref {
                    pointee: target_pointee,
                    kind: RefKind::Provider { provider_ty, space },
                    view: RefView::Whole,
                },
            ) if pointee == target_pointee => {
                let temp = self.alloc_runtime_temp(
                    self.locals[src.index()].semantic_ty,
                    RuntimeCarrier::Value(RuntimeClass::Ref {
                        pointee: target_pointee,
                        kind: RefKind::Provider { provider_ty, space },
                        view: RefView::Whole,
                    }),
                );
                self.push_stmt(
                    bb,
                    RStmt::Assign {
                        dst: temp,
                        expr: RExpr::AddrOf {
                            place: RuntimePlace {
                                root: PlaceRoot::Ref(src),
                                path: Box::default(),
                            },
                        },
                    },
                );
                temp
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
                                root: PlaceRoot::Ref(src),
                                path: Box::default(),
                            },
                        },
                    },
                );
                temp
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
                let temp = self.alloc_runtime_temp(
                    self.locals[src.index()].semantic_ty,
                    RuntimeCarrier::Value(RuntimeClass::AggregateValue {
                        layout: target_layout,
                    }),
                );
                self.push_stmt(
                    bb,
                    RStmt::Assign {
                        dst: temp,
                        expr: RExpr::Load {
                            place: RuntimePlace {
                                root: PlaceRoot::Ptr {
                                    addr: src,
                                    space,
                                    class: RuntimeClass::AggregateValue { layout },
                                },
                                path: Box::default(),
                            },
                        },
                    },
                );
                temp
            }
            (
                RuntimeClass::Ref { pointee, .. },
                RuntimeClass::AggregateValue {
                    layout: target_layout,
                },
            ) if *pointee
                == RuntimeClass::AggregateValue {
                    layout: target_layout,
                } =>
            {
                let temp = self.alloc_runtime_temp(
                    self.locals[src.index()].semantic_ty,
                    RuntimeCarrier::Value(RuntimeClass::AggregateValue {
                        layout: target_layout,
                    }),
                );
                self.push_stmt(
                    bb,
                    RStmt::Assign {
                        dst: temp,
                        expr: RExpr::Load {
                            place: RuntimePlace {
                                root: PlaceRoot::Ref(src),
                                path: Box::default(),
                            },
                        },
                    },
                );
                temp
            }
            (
                RuntimeClass::AggregateValue { layout },
                RuntimeClass::Ref {
                    pointee,
                    kind: RefKind::Object,
                    view: RefView::Whole,
                },
            ) if *pointee == RuntimeClass::AggregateValue { layout } => {
                let temp = self.alloc_runtime_temp(
                    self.locals[src.index()].semantic_ty,
                    RuntimeCarrier::Value(RuntimeClass::object_ref(layout)),
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
                let temp = self.alloc_runtime_temp(
                    self.locals[src.index()].semantic_ty,
                    RuntimeCarrier::Value(RuntimeClass::object_ref(layout)),
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
                RuntimeClass::Ref {
                    pointee,
                    kind:
                        RefKind::Provider {
                            provider_ty,
                            space: provider_space,
                        },
                    ..
                },
            ) if space == provider_space => {
                let target = pointee.aggregate_layout();
                let temp = self.alloc_runtime_temp(
                    self.locals[src.index()].semantic_ty,
                    RuntimeCarrier::Value(target.map_or(
                        RuntimeClass::Ref {
                            pointee,
                            kind: RefKind::Provider { provider_ty, space },
                            view: RefView::Whole,
                        },
                        |layout| RuntimeClass::provider_ref(layout, provider_ty, space),
                    )),
                );
                self.push_stmt(
                    bb,
                    RStmt::Assign {
                        dst: temp,
                        expr: RExpr::ProviderFromRaw {
                            raw: src,
                            provider_ty,
                            space,
                            target,
                        },
                    },
                );
                temp
            }
            (
                RuntimeClass::AggregateValue { layout },
                RuntimeClass::Ref {
                    pointee,
                    kind: RefKind::Provider { provider_ty, space },
                    view: RefView::Whole,
                },
            ) if *pointee == RuntimeClass::AggregateValue { layout } => {
                let object = self.coerce_value(bb, src, &RuntimeClass::object_ref(layout));
                self.coerce_value(
                    bb,
                    object,
                    &RuntimeClass::provider_ref(layout, provider_ty, space),
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
                RuntimeClass::Ref {
                    pointee,
                    kind:
                        RefKind::Provider {
                            provider_ty: _,
                            space,
                        },
                    view: RefView::Whole,
                },
            ) => {
                let target_layout = pointee.aggregate_layout();
                let raw = self.coerce_scalar_word_to_raw(bb, src, space, target_layout);
                self.coerce_value(bb, raw, &target.clone())
            }
            (
                RuntimeClass::Ref {
                    kind: RefKind::Provider { .. },
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
                let value = self.coerce_value(bb, src, &pointee);
                let temp = self.alloc_runtime_temp(
                    self.locals[src.index()].semantic_ty,
                    RuntimeCarrier::Value(RuntimeClass::object_ref(layout)),
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
                            root: PlaceRoot::Ref(temp),
                            path: Box::default(),
                        },
                        src: value,
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
                    RuntimeClass::Ref { .. } => source
                        .aggregate_layout()
                        .map(|layout| (layout, layout.data(self.db), layout_source_ty(layout))),
                    RuntimeClass::AggregateValue { layout }
                    | RuntimeClass::RawAddr {
                        target: Some(layout),
                        ..
                    } => Some((layout, layout.data(self.db), layout_source_ty(layout))),
                    RuntimeClass::Scalar(_) | RuntimeClass::RawAddr { target: None, .. } => None,
                };
                let target_layout = match target {
                    RuntimeClass::Ref { .. } => target
                        .aggregate_layout()
                        .map(|layout| (layout, layout.data(self.db), layout_source_ty(layout))),
                    RuntimeClass::AggregateValue { layout }
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

    fn runtime_local_uses_source_transport(&self, local: RLocalId) -> bool {
        local.index() < self.semantic_locals.len()
            && matches!(
                self.semantic_locals[local.index()],
                RuntimeLocalLowering::PlaceCarrier { .. }
            )
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
            RuntimeLocalLowering::PlaceBoundValue {
                provider: Some(provider),
                ..
            } => Some(self.provider_binding_value(*provider)),
            RuntimeLocalLowering::PlaceCarrier { .. } => Some(self.runtime_value(local)),
            RuntimeLocalLowering::DirectCarrier {
                provider: Some(provider),
                ..
            } => Some(self.provider_binding_value(*provider)),
            RuntimeLocalLowering::Erased
            | RuntimeLocalLowering::DirectValue
            | RuntimeLocalLowering::PlaceBoundValue { provider: None, .. }
            | RuntimeLocalLowering::DirectCarrier { provider: None, .. } => None,
        }
    }

    fn semantic_place_root(&self, local: SLocalId) -> Option<PlaceRoot<'db>> {
        if let RuntimeLocalLowering::PlaceBoundValue {
            provider: Some(provider),
            ..
        } = self.semantic_local_lowering(local)
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
        self.semantic_body.local(local)?.transport_place()
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
            RuntimeLocalLowering::PlaceCarrier { .. }
            | RuntimeLocalLowering::PlaceBoundValue { .. } => {
                let place = self.semantic_place(bb, local);
                let place_class = self.project_place_class(&place);
                let temp = self.alloc_runtime_temp(
                    self.locals[local.index()].semantic_ty,
                    RuntimeCarrier::Value(place_class),
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
        if matches!(target, RuntimeClass::AggregateValue { .. })
            && boundary_source_uses_transport_sensitive_aggregate(
                self.db,
                self.locals[operand.local.index()].semantic_ty,
                self.env.scope,
                self.env.assumptions,
            )
            && let Some(value) = self.actual_aggregate_value_from_runtime_source(bb, operand.local)
        {
            return value;
        }
        if runtime_target_prefers_transport(target) {
            if let Some(value) = self.handle_like_semantic_value(operand.local) {
                return if self.value_class(value) == Some(target) {
                    value
                } else {
                    self.coerce_value(bb, value, target)
                };
            }
            if let Some(value) = self.addr_of_semantic_operand_for_class(bb, operand, target) {
                return value;
            }
        }
        let value = self.read_semantic_operand(bb, operand);
        if self.value_class(value).is_none() || self.value_class(value) == Some(target) {
            value
        } else {
            self.coerce_value(bb, value, target)
        }
    }

    fn actual_aggregate_value_from_runtime_source(
        &mut self,
        bb: RBlockId,
        local: SLocalId,
    ) -> Option<RLocalId> {
        let value = self.read_semantic_value(bb, local);
        let class = self.value_class(value).cloned()?;
        let actual = actual_aggregate_class_from_runtime_source(&class)?;
        Some(if class == actual {
            value
        } else {
            self.coerce_value(bb, value, &actual)
        })
    }

    fn lower_semantic_operand_for_boundary(
        &mut self,
        bb: RBlockId,
        operand: NOperand,
        boundary: &crate::runtime::RuntimeBoundarySpec<'db>,
    ) -> RLocalId {
        let boundary = self.specialize_runtime_boundary_for_source(operand.local, boundary);
        self.lower_for_boundary(
            bb,
            BoundarySource::SemanticOperand(operand),
            &boundary,
            self.locals[operand.local.index()].semantic_ty,
        )
    }

    fn specialize_runtime_boundary_for_source(
        &self,
        local: SLocalId,
        boundary: &crate::runtime::RuntimeBoundarySpec<'db>,
    ) -> crate::runtime::RuntimeBoundarySpec<'db> {
        specialize_boundary_for_runtime_source(
            self.db,
            &self.semantic_body,
            local,
            boundary,
            &self
                .locals
                .iter()
                .take(self.semantic_body.locals.len())
                .map(|local| local.carrier.clone())
                .collect::<Vec<_>>(),
        )
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
            return Some(self.lower_place_addr_of_for_class(
                self.locals[local.index()].semantic_ty,
                bb,
                place,
                target.clone(),
            ));
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
            return Some(self.lower_place_addr_of_for_class(
                self.locals[local.index()].semantic_ty,
                bb,
                place,
                target.clone(),
            ));
        }
        None
    }

    fn semantic_local_allows_transport_addr_of(&self, local: SLocalId) -> bool {
        self.semantic_body
            .local(local)
            .is_some_and(|local| local.transport_place().is_some())
    }

    fn handle_like_semantic_value(&self, local: SLocalId) -> Option<RLocalId> {
        let value = match self.semantic_local_lowering(local) {
            RuntimeLocalLowering::Erased => None,
            RuntimeLocalLowering::DirectValue | RuntimeLocalLowering::PlaceCarrier { .. } => {
                Some(self.runtime_value(local))
            }
            RuntimeLocalLowering::PlaceBoundValue {
                provider: Some(provider),
                ..
            } => Some(self.provider_binding_value(*provider)),
            RuntimeLocalLowering::PlaceBoundValue { provider: None, .. } => None,
            RuntimeLocalLowering::DirectCarrier { provider, .. } => Some(provider.map_or_else(
                || self.runtime_value(local),
                |provider| self.provider_binding_value(provider),
            )),
        }?;
        runtime_class_is_handle_like(self.value_class(value)?).then_some(value)
    }

    fn lower_for_boundary(
        &mut self,
        bb: RBlockId,
        source: BoundarySource<'db>,
        boundary: &crate::runtime::RuntimeBoundarySpec<'db>,
        semantic_ty: TyId<'db>,
    ) -> RLocalId {
        match boundary {
            crate::runtime::RuntimeBoundarySpec::Exact(target) => {
                self.lower_boundary_source_for_class(bb, source, target, semantic_ty)
            }
            crate::runtime::RuntimeBoundarySpec::BorrowLike { .. } => {
                self.lower_borrow_like_source(bb, source, boundary, semantic_ty)
            }
        }
    }

    fn lower_boundary_source_for_class(
        &mut self,
        bb: RBlockId,
        source: BoundarySource<'db>,
        target: &RuntimeClass<'db>,
        semantic_ty: TyId<'db>,
    ) -> RLocalId {
        match source {
            BoundarySource::SemanticOperand(operand) => {
                self.lower_semantic_operand_for_class(bb, operand, target)
            }
            BoundarySource::RuntimePlace(place) if runtime_target_prefers_transport(target) => {
                self.lower_place_addr_of_for_class(semantic_ty, bb, place, target.clone())
            }
            BoundarySource::RuntimePlace(place) => {
                let place_class = self.project_place_class(&place);
                let value = self
                    .alloc_runtime_temp(semantic_ty, RuntimeCarrier::Value(place_class.clone()));
                self.push_stmt(
                    bb,
                    RStmt::Assign {
                        dst: value,
                        expr: RExpr::Load { place },
                    },
                );
                self.coerce_value(bb, value, target)
            }
        }
    }

    fn lower_borrow_like_source(
        &mut self,
        bb: RBlockId,
        source: BoundarySource<'db>,
        boundary: &crate::runtime::RuntimeBoundarySpec<'db>,
        semantic_ty: TyId<'db>,
    ) -> RLocalId {
        if let Some(value) = self.compatible_boundary_value(&source, boundary) {
            return value;
        }
        if let Some(value) = self.addr_of_boundary_source(bb, &source, boundary, semantic_ty) {
            return value;
        }
        match source {
            BoundarySource::SemanticOperand(operand) => {
                let value = self.read_semantic_operand(bb, operand);
                self.materialize_value_for_boundary(bb, value, boundary, semantic_ty)
            }
            BoundarySource::RuntimePlace(place) => {
                let place_class = self.project_place_class(&place);
                let value = self
                    .alloc_runtime_temp(semantic_ty, RuntimeCarrier::Value(place_class.clone()));
                self.push_stmt(
                    bb,
                    RStmt::Assign {
                        dst: value,
                        expr: RExpr::Load { place },
                    },
                );
                self.materialize_value_for_boundary(bb, value, boundary, semantic_ty)
            }
        }
    }

    fn compatible_boundary_value(
        &self,
        source: &BoundarySource<'db>,
        boundary: &crate::runtime::RuntimeBoundarySpec<'db>,
    ) -> Option<RLocalId> {
        let value = match source {
            BoundarySource::SemanticOperand(operand) => {
                if let Some(value) = self.handle_like_semantic_value(operand.local)
                    && self
                        .value_class(value)
                        .is_some_and(|class| runtime_class_satisfies_boundary(class, boundary))
                {
                    return Some(value);
                }
                let value = self.runtime_value(operand.local);
                self.value_class(value)
                    .is_some_and(|class| runtime_class_satisfies_boundary(class, boundary))
                    .then_some(value)
            }
            BoundarySource::RuntimePlace(_) => None,
        }?;
        self.value_class(value)
            .is_some_and(|class| runtime_class_satisfies_boundary(class, boundary))
            .then_some(value)
    }

    fn addr_of_boundary_source(
        &mut self,
        bb: RBlockId,
        source: &BoundarySource<'db>,
        boundary: &crate::runtime::RuntimeBoundarySpec<'db>,
        semantic_ty: TyId<'db>,
    ) -> Option<RLocalId> {
        let place = match source {
            BoundarySource::RuntimePlace(place) => Some(place.clone()),
            BoundarySource::SemanticOperand(operand) => {
                let local = operand.local;
                if !self.semantic_local_allows_transport_addr_of(local) {
                    return None;
                }
                if let Some(place) = self.normalized_value_place(local).cloned()
                    && !self.is_self_rooted_value_place(local, &place)
                    && let Some(place) = self.try_lower_place(bb, &place)
                {
                    Some(place)
                } else {
                    self.try_semantic_place(bb, local)
                }
            }
        }?;
        let actual = self.place_addr_class(&place);
        runtime_class_satisfies_boundary(&actual, boundary)
            .then(|| self.lower_place_addr_of_for_class(semantic_ty, bb, place, actual))
    }

    fn materialize_value_for_boundary(
        &mut self,
        bb: RBlockId,
        value: RLocalId,
        boundary: &crate::runtime::RuntimeBoundarySpec<'db>,
        semantic_ty: TyId<'db>,
    ) -> RLocalId {
        let Some(source_class) = self.value_class(value).cloned() else {
            panic!(
                "borrow-like boundary value has no runtime class: owner={:?}; value={value:?}; boundary={boundary:?}",
                self.key
                    .semantic(self.db)
                    .map(|semantic| semantic.key(self.db).owner(self.db)),
            );
        };
        if runtime_class_satisfies_boundary(&source_class, boundary) {
            return value;
        }
        let crate::runtime::RuntimeBoundarySpec::BorrowLike { pointee, allow, .. } = boundary
        else {
            unreachable!();
        };
        if let Some(layout) = pointee.aggregate_layout()
            && allow.allow_object
        {
            let object = self.coerce_value(bb, value, &RuntimeClass::object_ref(layout));
            if self
                .value_class(object)
                .is_some_and(|class| runtime_class_satisfies_boundary(class, boundary))
            {
                return object;
            }
        }
        if pointee.aggregate_layout().is_none() && allow.allow_raw_addr {
            let slot = self.alloc_runtime_temp(semantic_ty, RuntimeCarrier::Value(pointee.clone()));
            self.locals[slot.index()].root = RuntimeLocalRoot::Slot(pointee.clone());
            let src = if self.value_class(value) == Some(pointee) {
                value
            } else {
                self.coerce_value(bb, value, pointee)
            };
            self.push_stmt(
                bb,
                RStmt::Assign {
                    dst: slot,
                    expr: RExpr::Use(src),
                },
            );
            let place = RuntimePlace {
                root: PlaceRoot::Slot(slot),
                path: Box::default(),
            };
            let raw = self.lower_place_addr_of_for_class(
                semantic_ty,
                bb,
                place,
                RuntimeClass::RawAddr {
                    space: AddressSpaceKind::Memory,
                    target: None,
                },
            );
            if self
                .value_class(raw)
                .is_some_and(|class| runtime_class_satisfies_boundary(class, boundary))
            {
                return raw;
            }
        }
        panic!(
            "borrow-like boundary source has no realizable runtime transport: owner={:?}; source={source_class:?}; boundary={boundary:?}; semantic_ty={}",
            self.key
                .semantic(self.db)
                .map(|semantic| semantic.key(self.db).owner(self.db)),
            semantic_ty.pretty_print(self.db),
        );
    }

    fn lower_place_addr_of_for_class(
        &mut self,
        semantic_ty: TyId<'db>,
        bb: RBlockId,
        place: RuntimePlace<'db>,
        target: RuntimeClass<'db>,
    ) -> RLocalId {
        let actual = self.place_addr_class(&place);
        let temp = self.alloc_runtime_temp(semantic_ty, RuntimeCarrier::Value(actual.clone()));
        self.push_stmt(
            bb,
            RStmt::Assign {
                dst: temp,
                expr: RExpr::AddrOf { place },
            },
        );
        if actual == target {
            temp
        } else {
            self.coerce_value(bb, temp, &target)
        }
    }

    fn place_addr_class(&self, place: &RuntimePlace<'db>) -> RuntimeClass<'db> {
        let place_class = self.project_place_class(place);
        let root_class = match &place.root {
            PlaceRoot::Slot(local) => self
                .local_root_class_r(*local)
                .expect("slot place root should have a runtime class"),
            PlaceRoot::Ref(local) => self
                .value_class(*local)
                .cloned()
                .expect("ref place root should have a runtime class"),
            PlaceRoot::Provider(binding) => self.provider_binding(*binding).provider_class.clone(),
            PlaceRoot::Ptr { space, class, .. } => RuntimeClass::RawAddr {
                space: *space,
                target: class.aggregate_layout(),
            },
        };
        ref_class_for_place_result(
            &root_class,
            &place_class,
            match &place.root {
                PlaceRoot::Slot(_) | PlaceRoot::Ref(_) => AddressSpaceKind::Memory,
                PlaceRoot::Provider(_) => {
                    runtime_address_space(&root_class).unwrap_or(AddressSpaceKind::Memory)
                }
                PlaceRoot::Ptr { space, .. } => *space,
            },
            matches!(place.root, PlaceRoot::Ptr { .. }),
        )
    }

    fn write_semantic_value(&mut self, bb: RBlockId, local: SLocalId, src: RLocalId) {
        match self.semantic_local_lowering(local).clone() {
            RuntimeLocalLowering::Erased => {}
            RuntimeLocalLowering::DirectValue | RuntimeLocalLowering::PlaceCarrier { .. } => {
                let dst = self.runtime_value(local);
                let Some(mut target) = self.value_class(dst).cloned() else {
                    return;
                };
                if matches!(
                    target,
                    RuntimeClass::AggregateValue { .. }
                        | RuntimeClass::Ref {
                            kind: RefKind::Object,
                            ..
                        }
                ) && let Some(actual) = self
                    .value_class(src)
                    .and_then(actual_aggregate_class_from_runtime_source)
                {
                    target = match target {
                        RuntimeClass::AggregateValue { .. } => actual,
                        RuntimeClass::Ref {
                            kind: RefKind::Object,
                            ..
                        } => RuntimeClass::object_ref(
                            actual.aggregate_layout().expect("aggregate ref layout"),
                        ),
                        RuntimeClass::Ref { .. }
                        | RuntimeClass::Scalar(_)
                        | RuntimeClass::RawAddr { .. } => unreachable!(),
                    };
                    self.refine_local_runtime_class(dst, target.clone());
                }
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
            RuntimeLocalLowering::PlaceBoundValue { .. } => {
                let place = self.semantic_place(bb, local);
                let target = self.project_place_class(&place);
                self.write_value_to_place(bb, place, src, &target);
            }
        }
    }

    fn specialize_runtime_target_from_operand(
        &mut self,
        dst: RLocalId,
        src: NOperand,
        target: &RuntimeClass<'db>,
    ) -> RuntimeClass<'db> {
        if !matches!(
            target,
            RuntimeClass::AggregateValue { .. }
                | RuntimeClass::Ref {
                    kind: RefKind::Object,
                    ..
                }
        ) {
            return target.clone();
        }
        let Some(actual) = self
            .semantic_value_class(src.local)
            .as_ref()
            .and_then(actual_aggregate_class_from_runtime_source)
        else {
            return target.clone();
        };
        let target = match target {
            RuntimeClass::AggregateValue { .. } => actual,
            RuntimeClass::Ref {
                kind: RefKind::Object,
                ..
            } => RuntimeClass::object_ref(actual.aggregate_layout().expect("aggregate ref layout")),
            RuntimeClass::Ref { .. } | RuntimeClass::Scalar(_) | RuntimeClass::RawAddr { .. } => {
                unreachable!()
            }
        };
        self.refine_local_runtime_class(dst, target.clone());
        target
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
            PlaceRoot::Ref(local) => match self
                .value_class(*local)
                .cloned()
                .expect("projected ref places should have runtime classes")
            {
                class @ RuntimeClass::Ref {
                    view: RefView::EnumVariant(_),
                    ..
                } if matches!(place.path.first(), Some(PlaceElem::VariantField(_))) => class,
                RuntimeClass::Ref { pointee, .. } => *pointee,
                class => class,
            },
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
        current
    }

    fn class_is_runtime_zst(&self, class: &RuntimeClass<'db>) -> bool {
        match class {
            RuntimeClass::Scalar(_) | RuntimeClass::Ref { .. } | RuntimeClass::RawAddr { .. } => {
                false
            }
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
            RuntimeClass::AggregateValue { layout } => layout,
            RuntimeClass::Ref { ref pointee, .. } => {
                pointee.aggregate_layout().unwrap_or_else(|| {
                    panic!("enum values should lower as aggregate values or refs, found {class:?}")
                })
            }
            class => {
                panic!("enum values should lower as aggregate values or refs, found {class:?}")
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
            .expect("enum ref should have a class");
        let RuntimeClass::Ref { pointee, kind, .. } = class else {
            panic!("enum variant refs require ref-form enums");
        };
        let layout = pointee.aggregate_layout().expect("enum ref pointee layout");
        let variant_id = VariantId {
            enum_layout: layout,
            index: variant.0,
        };
        let temp = self.alloc_runtime_temp(
            self.locals[value.index()].semantic_ty,
            RuntimeCarrier::Value(RuntimeClass::Ref {
                pointee: Box::new(RuntimeClass::AggregateValue { layout }),
                kind: kind.clone(),
                view: RefView::EnumVariant(variant_id),
            }),
        );
        self.locals[temp.index()].root = RuntimeLocalRoot::Ref(RuntimeClass::Ref {
            pointee: Box::new(RuntimeClass::AggregateValue { layout }),
            kind,
            view: RefView::EnumVariant(variant_id),
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
            RuntimeLocalRoot::Ref(_) => Some(PlaceRoot::Ref(local)),
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
            | RuntimeLocalRoot::Ref(class)
            | RuntimeLocalRoot::Ptr { class, .. } => Some(class.clone()),
        }
    }

    fn value_class(&self, local: RLocalId) -> Option<&RuntimeClass<'db>> {
        match self.locals.get(local.index())?.carrier {
            RuntimeCarrier::Erased => None,
            RuntimeCarrier::Value(ref class) => Some(class),
        }
    }

    fn refine_local_runtime_class(&mut self, local: RLocalId, class: RuntimeClass<'db>) {
        self.locals[local.index()].carrier = RuntimeCarrier::Value(class.clone());
        match &mut self.locals[local.index()].root {
            RuntimeLocalRoot::Slot(root) => *root = class,
            RuntimeLocalRoot::Ref(root) => *root = class,
            RuntimeLocalRoot::Ptr { .. } | RuntimeLocalRoot::None => {}
        }
    }

    fn reify_runtime_const(
        &self,
        expected_ty: TyId<'db>,
        value: SemConstId<'db>,
    ) -> SemConstId<'db> {
        let semantic = self
            .key
            .semantic(self.db)
            .expect("runtime const reification requires a semantic instance");
        reify_runtime_const_for_ty(self.db, semantic, expected_ty, value).unwrap_or_else(|| {
            panic!("semantic const should reify for runtime lowering: {value:?}")
        })
    }

    fn const_lowering_ty(&self, fallback: TyId<'db>, target: &RuntimeClass<'db>) -> TyId<'db> {
        target
            .aggregate_layout()
            .map_or(fallback, |layout| match layout.data(self.db) {
                crate::runtime::Layout::Struct(layout) => layout.source_ty,
                crate::runtime::Layout::Array(layout) => layout.source_ty,
                crate::runtime::Layout::Enum(layout) => layout.source_ty,
            })
    }

    fn should_preserve_const_source_class(
        &self,
        actual: &RuntimeClass<'db>,
        target: &RuntimeClass<'db>,
    ) -> bool {
        match (actual.aggregate_layout(), target.aggregate_layout()) {
            (Some(actual), Some(target)) => {
                self.layout_source_ty(actual) == self.layout_source_ty(target)
            }
            _ => false,
        }
    }

    fn layout_source_ty(&self, layout: LayoutId<'db>) -> TyId<'db> {
        match layout.data(self.db) {
            crate::runtime::Layout::Struct(layout) => layout.source_ty,
            crate::runtime::Layout::Array(layout) => layout.source_ty,
            crate::runtime::Layout::Enum(layout) => layout.source_ty,
        }
    }
}

fn runtime_target_prefers_transport(class: &RuntimeClass<'_>) -> bool {
    matches!(
        class,
        RuntimeClass::Ref { .. } | RuntimeClass::RawAddr { .. }
    )
}

fn runtime_class_is_handle_like(class: &RuntimeClass<'_>) -> bool {
    matches!(
        class,
        RuntimeClass::Ref { .. } | RuntimeClass::RawAddr { .. }
    )
}

fn word_scalar_class<'db>() -> ScalarClass<'db> {
    ScalarClass {
        repr: ScalarRepr::Int {
            bits: 256,
            signed: false,
        },
        role: ScalarRole::Plain,
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
