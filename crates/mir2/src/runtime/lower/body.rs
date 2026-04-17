use cranelift_entity::EntityRef;
use hir::analysis::{
    semantic::{
        FieldIndex, SBlockId, SConst, SLocalId, SemConstId, SemConstScalar, SemConstValue,
        SemanticCalleeRef, SemanticCodeRegionRef, SemanticInstance, SemanticInstanceKey,
        VariantIndex,
        borrowck::{
            NBorrowRoot, NEffectArg, NEffectArgValue, NExpr, NLocalInterface, NLocalOrigin,
            NOperand, NSPlace, NSPlaceRoot, NSStmt, NSStmtKind, NSTerminator, NSTerminatorKind,
            NormalizedSemanticBody, normalize_semantic_body,
        },
        get_or_build_semantic_instance, reify_runtime_const_for_ty, sem_const_ty,
        semantic_may_return_normally,
    },
    ty::{
        corelib::{
            PrimitiveWrapperCallKind, RuntimeBuiltinFuncKind, core_primitive_wrapper_call_kind,
            runtime_builtin_func_kind,
        },
        ty_check::{BodyOwner, EffectPassMode},
        ty_def::TyId,
    },
};
use hir::hir_def::{ArithBinOp, BinOp, CompBinOp, Func, UnOp, attr::ArithmeticMode};
use hir::projection::{IndexSource, Projection};

use crate::{
    db::MirDb,
    instance::{RuntimeInstance, RuntimeInstanceKey, get_or_build_runtime_instance},
    resolve_runtime_place_address_class,
    runtime::{
        AddressSpaceKind, ConstScalar, IntrinsicArithBinOp, LayoutId, PlaceElem, PlaceRoot, RBlock,
        RBlockId, RExpr, RLocal, RLocalId, RStmt, RTerminator, RefKind, RefView, RuntimeBody,
        RuntimeCarrier, RuntimeClass, RuntimeCodeRegion, RuntimeLocalLowering, RuntimeLocalRoot,
        RuntimePlace, RuntimeProviderBinding, RuntimeProviderBindingId, RuntimeSignature,
        ScalarClass, ScalarRepr, ScalarRole, VariantId,
        code_region::runtime_code_region_for_semantic_ref, runtime_classes_share_runtime_rep,
    },
};

use super::{
    classify::{
        BodyEnv, BodyStaticFacts, ContractMetadataBuiltin, GenericNumericIntrinsicKind,
        RuntimeBodyCx, RuntimeEffectBindingPlan, actual_aggregate_class_from_runtime_source,
        contract_metadata_builtin, desired_runtime_effect_arg_boundary, desired_runtime_param_plan,
        generic_numeric_intrinsic_kind, nonself_backing_value_place, resolve_runtime_call_key,
        runtime_effect_binding_plan_for_binding_idx, runtime_signature_for_key, semantic_return_ty,
        snapshot_source_place,
    },
    coerce::CoercionPlanner,
    consts::{
        const_scalar_for_class, const_scalar_from_value, enum_tag_scalar, lower_const_region,
    },
    infer::{InferenceResult, LocalStateInferer, merge_runtime_class},
    interface::runtime_param_locals,
    layout::{
        AggregateCtorElem, aggregate_ctor_elems_for_layout, layout_for_aggregate_instance_in_env,
        layout_for_enum_variant_instance_in_env, layout_for_ty_in_env,
    },
    place::{
        project_field_class, project_index_class, project_variant_field_class,
        resolved_effect_arg_address_space,
    },
    type_info::{
        RuntimeTypeEnv, boundary_spec_for_ty_in_env, provider_class_for_target_in_env,
        stored_class_for_ty_in_context, top_level_class_for_ty_in_env,
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
    let typed_body = semantic.key(db).typed_body(db);
    let facts = BodyStaticFacts::new(db, &normalized_body);
    let inferred = LocalStateInferer::new(
        BodyEnv::new(db, &normalized_body, typed_body, &facts),
        key.params(db),
        &runtime_param_locals(db, semantic, key.params(db)),
    )
    .run();
    let signature = runtime_signature_for_key(db, semantic, key.params(db));
    let mut emitter = RmirEmitter::new(
        db,
        instance,
        normalized_body,
        typed_body,
        facts,
        inferred,
        signature.clone(),
    );
    emitter.lower_blocks();
    emitter.finish(signature)
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
        | NExpr::CodeRegionRef { .. }
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

pub(super) struct RmirEmitter<'db> {
    pub(super) db: &'db dyn MirDb,
    pub(super) instance: RuntimeInstance<'db>,
    pub(super) key: RuntimeInstanceKey<'db>,
    pub(super) semantic_body: NormalizedSemanticBody<'db>,
    pub(super) typed_body: &'db hir::analysis::ty::ty_check::TypedBody<'db>,
    pub(super) facts: BodyStaticFacts<'db>,
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
    carriers: Vec<RuntimeCarrier<'db>>,
    roots: Vec<RuntimeLocalRoot<'db>>,
    semantic_locals: Vec<RuntimeLocalLowering<'db>>,
    provider_bindings: Vec<RuntimeProviderBinding<'db>>,
}

enum BoundarySource<'db> {
    SemanticOperand(NOperand),
    RuntimePlace(RuntimePlace<'db>),
}

impl<'db> RmirEmitter<'db> {
    fn new(
        db: &'db dyn MirDb,
        instance: RuntimeInstance<'db>,
        semantic_body: NormalizedSemanticBody<'db>,
        typed_body: &'db hir::analysis::ty::ty_check::TypedBody<'db>,
        facts: BodyStaticFacts<'db>,
        inferred: InferenceResult<'db>,
        signature: RuntimeSignature<'db>,
    ) -> Self {
        let key = instance.key(db);
        let LoweredSemanticLocals {
            carriers,
            roots,
            semantic_locals,
            provider_bindings,
        } = LoweredSemanticLocals {
            carriers: inferred.carriers,
            roots: inferred.roots,
            semantic_locals: inferred.semantic_locals,
            provider_bindings: inferred.provider_bindings,
        };
        let env = RuntimeTypeEnv::new(
            typed_body.body().map(|body| body.scope()),
            typed_body.assumptions(),
        );
        let terminated_blocks = vec![false; semantic_body.blocks.len()];
        let locals = semantic_body
            .locals
            .iter()
            .enumerate()
            .map(|(idx, local)| RLocal {
                semantic_ty: local.ty,
                carrier: carriers.get(idx).cloned().unwrap_or(RuntimeCarrier::Erased),
                root: roots.get(idx).cloned().unwrap_or(RuntimeLocalRoot::None),
            })
            .collect::<Vec<_>>();
        let blocks = Vec::with_capacity(semantic_body.blocks.len());
        Self {
            db,
            instance,
            key,
            semantic_body,
            typed_body,
            facts,
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
            for (stmt_idx, stmt) in block.stmts.iter().enumerate() {
                self.lower_stmt(bb, stmt_idx, stmt);
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

    fn lower_stmt(&mut self, bb: RBlockId, stmt_idx: usize, stmt: &NSStmt<'db>) {
        match &stmt.kind {
            NSStmtKind::Assign { dst, expr } => self.lower_assign(bb, stmt_idx, *dst, expr),
            NSStmtKind::Store { dst, src } => {
                let place = self.lower_place(bb, dst);
                let target = self.project_place_class(&place);
                let value = self.read_semantic_value(bb, *src);
                self.write_value_to_place(bb, place, value, &target);
            }
        }
    }

    fn lower_assign(&mut self, bb: RBlockId, stmt_idx: usize, dst: SLocalId, expr: &NExpr<'db>) {
        let desired = self.semantic_value_class(dst);
        match desired {
            None => {
                if !expr_requires_runtime_eval_when_erased(expr) {
                    return;
                }
                let carrier = self
                    .with_current_body_cx(|cx| {
                        cx.expr_direct_class(
                            bb.index(),
                            stmt_idx,
                            expr,
                            self.semantic_body.locals[dst.index()].ty,
                        )
                    })
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

    fn with_current_body_cx<T>(&self, f: impl FnOnce(RuntimeBodyCx<'_, '_, 'db>) -> T) -> T {
        let carriers = self
            .locals
            .iter()
            .take(self.semantic_body.locals.len())
            .map(|local| local.carrier.clone())
            .collect::<Vec<_>>();
        f(
            BodyEnv::new(self.db, &self.semantic_body, self.typed_body, &self.facts)
                .with_carriers(&carriers),
        )
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
            let actual = self.with_current_body_cx(|cx| match expr {
                NExpr::Use(value) => cx.materialized_value_class(value.local),
                NExpr::Borrow { place, .. } => cx.normalized_place_address_class(place),
                NExpr::ReadPlace { place, .. } => cx.normalized_place_class(place),
                NExpr::Const(_)
                | NExpr::CodeRegionRef { .. }
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
            });
            if let Some(actual) = actual
                && CoercionPlanner::target_prefers_transport(&actual)
            {
                self.refine_local_runtime_class(dst_value, actual);
                return;
            }
        }
        if !self.runtime_local_uses_source_transport(dst_value) {
            return;
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
            NExpr::Use(value) => {
                self.with_current_body_cx(|cx| cx.actual_aggregate_class_for_source(value.local))
            }
            _ => None,
        };
        let Some(actual) = source else {
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
                    && CoercionPlanner::target_prefers_transport(&projected)
                {
                    self.refine_local_runtime_class(dst, projected.clone());
                    projected.clone()
                } else {
                    dst_class
                };
                if let (
                    RuntimeClass::AggregateValue { layout },
                    RuntimeClass::Ref {
                        pointee,
                        kind: RefKind::Object,
                        view: RefView::Whole,
                    },
                ) = (&projected, &dst_class)
                    && **pointee == (RuntimeClass::AggregateValue { layout: *layout })
                {
                    self.push_stmt(
                        bb,
                        RStmt::Assign {
                            dst,
                            expr: RExpr::MaterializePlaceToObject { place },
                        },
                    );
                    return;
                }
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
                    && CoercionPlanner::target_prefers_transport(&actual)
                {
                    let target =
                        merge_runtime_class(self.db, &dst_class, &actual).unwrap_or(actual);
                    self.refine_local_runtime_class(dst, target.clone());
                    target
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
            NExpr::CodeRegionRef { .. } => {
                panic!(
                    "code-region ref reached runtime lowering as a runtime value: owner={:?}; dst={dst:?}; expr={expr:?}",
                    self.current_semantic_key(),
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
        if let Some(value) = self.try_lower_dyn_string_literal(bb, ty, value) {
            return value;
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
                let RuntimeClass::AggregateValue { layout } = target else {
                    unreachable!();
                };
                let src =
                    self.lower_non_scalar_const_as_aggregate_value(bb, value, expected_ty, *layout);
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
        if let Some(value) = self.try_lower_dyn_string_literal(bb, ty, value) {
            return value;
        }
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

    fn try_lower_dyn_string_literal(
        &mut self,
        bb: RBlockId,
        ty: TyId<'db>,
        value: SemConstId<'db>,
    ) -> Option<RLocalId> {
        let SemConstValue::Scalar {
            value: SemConstScalar::Bytes(bytes),
            ..
        } = value.value(self.db)
        else {
            return None;
        };
        ty.is_core_dyn_string(self.db)
            .then(|| self.lower_dyn_string_literal(bb, ty, &bytes))
    }

    fn lower_dyn_string_literal(&mut self, bb: RBlockId, ty: TyId<'db>, bytes: &[u8]) -> RLocalId {
        let len = self.alloc_u256_const(bb, bytes.len());
        let payload_size = 32 + bytes.len().next_multiple_of(32);
        let size = self.alloc_u256_const(bb, payload_size);
        let ptr = self.alloc_runtime_temp(
            TyId::u256(self.db),
            RuntimeCarrier::Value(RuntimeClass::Scalar(word_scalar_class())),
        );
        self.push_stmt(
            bb,
            RStmt::Assign {
                dst: ptr,
                expr: RExpr::Builtin(crate::runtime::RuntimeBuiltin::Malloc { size }),
            },
        );
        self.push_ignored_builtin(
            bb,
            crate::runtime::RuntimeBuiltin::Mstore {
                addr: ptr,
                value: len,
            },
        );
        for (idx, chunk) in bytes.chunks(32).enumerate() {
            let addr = self.alloc_runtime_temp(
                TyId::u256(self.db),
                RuntimeCarrier::Value(RuntimeClass::Scalar(word_scalar_class())),
            );
            let offset = self.alloc_u256_const(bb, 32 * (idx + 1));
            self.push_stmt(
                bb,
                RStmt::Assign {
                    dst: addr,
                    expr: RExpr::Binary {
                        op: BinOp::Arith(ArithBinOp::Add),
                        lhs: ptr,
                        rhs: offset,
                    },
                },
            );
            let word = self.alloc_runtime_temp(
                TyId::u256(self.db),
                RuntimeCarrier::Value(RuntimeClass::Scalar(word_scalar_class())),
            );
            self.push_stmt(
                bb,
                RStmt::Assign {
                    dst: word,
                    expr: RExpr::ConstScalar(ConstScalar::Int {
                        bits: 256,
                        signed: false,
                        words: padded_word_bytes(chunk),
                    }),
                },
            );
            self.push_ignored_builtin(
                bb,
                crate::runtime::RuntimeBuiltin::Mstore { addr, value: word },
            );
        }

        let layout = self.layout_for_ty(ty);
        let dst = self.alloc_runtime_temp(
            ty,
            RuntimeCarrier::Value(RuntimeClass::AggregateValue { layout }),
        );
        let ctor_elems = aggregate_ctor_elems_for_layout(self.db, layout, 3);
        self.lower_aggregate_values(bb, dst, ty, layout, &ctor_elems, &[ptr, len, size]);
        dst
    }

    fn alloc_u256_const(&mut self, bb: RBlockId, value: usize) -> RLocalId {
        let local = self.alloc_runtime_temp(
            TyId::u256(self.db),
            RuntimeCarrier::Value(RuntimeClass::Scalar(word_scalar_class())),
        );
        self.push_stmt(
            bb,
            RStmt::Assign {
                dst: local,
                expr: RExpr::ConstScalar(ConstScalar::Int {
                    bits: 256,
                    signed: false,
                    words: usize_word_bytes(value),
                }),
            },
        );
        local
    }

    fn push_ignored_builtin(&mut self, bb: RBlockId, builtin: crate::runtime::RuntimeBuiltin<'db>) {
        let sink = self.alloc_runtime_temp(TyId::unit(self.db), RuntimeCarrier::Erased);
        self.push_stmt(
            bb,
            RStmt::Assign {
                dst: sink,
                expr: RExpr::Builtin(builtin),
            },
        );
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
            class @ (RuntimeClass::Ref {
                kind: RefKind::Provider { .. } | RefKind::Const,
                ..
            }
            | RuntimeClass::RawAddr { .. }) => class,
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
            provider @ RuntimeClass::Ref {
                kind: RefKind::Provider { .. },
                ..
            } => {
                self.lower_single_field_transport_ctor(bb, dst, provider, field_values, "provider")
            }
            raw_addr @ RuntimeClass::RawAddr { .. } => self.lower_single_field_transport_ctor(
                bb,
                dst,
                raw_addr,
                field_values,
                "raw-address",
            ),
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

    fn lower_single_field_transport_ctor(
        &mut self,
        bb: RBlockId,
        dst: RLocalId,
        dst_class: RuntimeClass<'db>,
        field_values: &[RLocalId],
        kind_name: &str,
    ) {
        let [value] = field_values else {
            panic!("{kind_name} aggregate construction requires exactly one field");
        };
        let value = self.coerce_value(bb, *value, &dst_class);
        self.push_stmt(
            bb,
            RStmt::Assign {
                dst,
                expr: RExpr::Use(value),
            },
        );
    }

    fn lower_enum_make(
        &mut self,
        bb: RBlockId,
        dst: RLocalId,
        enum_ty: TyId<'db>,
        variant: VariantIndex,
        fields: &[NOperand],
    ) {
        let enum_ = enum_ty
            .as_enum(self.db)
            .unwrap_or_else(|| panic!("enum construction reached non-enum type"));
        let args = enum_ty.generic_args(self.db);
        let Some(enum_variant) = enum_.variants(self.db).nth(variant.0 as usize) else {
            panic!("missing enum variant for {variant:?}");
        };
        let field_tys = enum_variant
            .field_tys(self.db)
            .into_iter()
            .map(|field| field.instantiate(self.db, args))
            .collect::<Vec<_>>();
        assert_eq!(
            field_tys.len(),
            fields.len(),
            "enum constructor arity mismatch for {}::{variant:?}",
            enum_ty.pretty_print(self.db),
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
                        .expect("non-zst enum field should have a runtime class");
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
        let layout = layout_for_enum_variant_instance_in_env(
            self.db,
            self.env,
            enum_ty,
            variant.0 as usize,
            &field_classes,
        );
        self.lower_enum_values(bb, dst, layout, variant, &field_values);
    }

    fn lower_enum_values(
        &mut self,
        bb: RBlockId,
        dst: RLocalId,
        ctor_layout: LayoutId<'db>,
        variant: VariantIndex,
        field_values: &[RLocalId],
    ) {
        let dst_class = self
            .value_class(dst)
            .cloned()
            .expect("enum destination must have a runtime class");
        let layout = match &dst_class {
            RuntimeClass::AggregateValue { layout } => *layout,
            RuntimeClass::Ref {
                pointee,
                kind: RefKind::Object,
                ..
            } => pointee
                .aggregate_layout()
                .expect("object enum destination must have aggregate layout"),
            _ => ctor_layout,
        };
        let variant = VariantId {
            enum_layout: layout,
            index: variant.0,
        };
        let field_values = match layout.data(self.db) {
            crate::runtime::Layout::Enum(layout_data) => field_values
                .iter()
                .copied()
                .zip(layout_data.variants[variant.index as usize].fields.iter())
                .map(|(value, class)| {
                    if self.value_class(value) == Some(class) {
                        value
                    } else {
                        self.coerce_value(bb, value, class)
                    }
                })
                .collect::<Vec<_>>(),
            crate::runtime::Layout::Struct(_) | crate::runtime::Layout::Array(_) => {
                panic!("enum destination must use an enum layout")
            }
        };
        match dst_class {
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
                            fields: field_values.into_boxed_slice(),
                        }
                    },
                );
            }
            class => panic!(
                "enum construction requires aggregate or object-ref destination, found {class:?}"
            ),
        }
    }

    fn lower_field_like(
        &mut self,
        bb: RBlockId,
        dst: RLocalId,
        base: SLocalId,
        elem: PlaceElem<'db>,
    ) {
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
        let value_class = self
            .semantic_value_class(value.local)
            .expect("enum value should have a runtime class");
        let variant_id = self.enum_variant_for_local(value.local, variant);
        if self.semantic_local_is_place_bound(value.local) {
            let value = self.read_semantic_operand(bb, value);
            self.push_stmt(
                bb,
                RStmt::EnumAssertVariant {
                    value,
                    variant: variant_id,
                },
            );
            self.lower_enum_extract_value(
                bb,
                dst,
                value,
                variant_id,
                field,
                project_variant_field_class(self.db, value_class.clone(), variant_id, field),
            );
            return;
        }
        match self.local_class(value.local) {
            Some(RuntimeClass::Ref { .. }) => {
                let refined = self.enum_variant_ref(bb, value.local, variant);
                self.lower_field_like(
                    bb,
                    dst,
                    refined,
                    PlaceElem::VariantField {
                        variant: variant_id,
                        field,
                    },
                );
            }
            _ => {
                self.push_stmt(
                    bb,
                    RStmt::EnumAssertVariant {
                        value: self.runtime_value(value.local),
                        variant: variant_id,
                    },
                );
                self.lower_enum_extract_value(
                    bb,
                    dst,
                    self.runtime_value(value.local),
                    variant_id,
                    field,
                    project_variant_field_class(self.db, value_class, variant_id, field),
                );
            }
        }
    }

    fn lower_enum_extract_value(
        &mut self,
        bb: RBlockId,
        dst: RLocalId,
        value: RLocalId,
        variant: VariantId<'db>,
        field: FieldIndex,
        field_class: RuntimeClass<'db>,
    ) {
        let target = self
            .value_class(dst)
            .cloned()
            .expect("enum extract result must have a runtime class");
        if field_class == target {
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
        let source = self.alloc_runtime_temp(
            self.locals[dst.index()].semantic_ty,
            RuntimeCarrier::Value(field_class),
        );
        self.push_stmt(
            bb,
            RStmt::Assign {
                dst: source,
                expr: RExpr::EnumExtract {
                    value,
                    variant,
                    field,
                },
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

    fn current_arithmetic_mode(&self) -> ArithmeticMode {
        self.current_semantic_key()
            .owner(self.db)
            .arithmetic_mode(self.db)
    }

    fn alloc_zero_scalar(
        &mut self,
        bb: RBlockId,
        semantic_ty: TyId<'db>,
        class: &ScalarClass<'db>,
    ) -> RLocalId {
        let zero = self.alloc_runtime_temp(
            semantic_ty,
            RuntimeCarrier::Value(RuntimeClass::Scalar(class.clone())),
        );
        let ScalarRepr::Int { bits, signed } = class.repr else {
            panic!("checked neg requires an integer scalar class");
        };
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
        zero
    }

    fn lower_intrinsic_arith_expr(
        &mut self,
        op: IntrinsicArithBinOp,
        checked: bool,
        lhs: RLocalId,
        rhs: RLocalId,
        class: &ScalarClass<'db>,
    ) -> RExpr<'db> {
        RExpr::Builtin(crate::runtime::RuntimeBuiltin::IntrinsicArith {
            op,
            checked,
            lhs,
            rhs,
            class: class.clone(),
        })
    }

    fn lower_arith_expr_for_mode(
        &mut self,
        bb: RBlockId,
        op: ArithBinOp,
        checked: bool,
        lhs: RLocalId,
        rhs: RLocalId,
        class: &ScalarClass<'db>,
    ) -> Option<RExpr<'db>> {
        let _ = bb;
        Some(match op {
            ArithBinOp::Add => {
                self.lower_intrinsic_arith_expr(IntrinsicArithBinOp::Add, checked, lhs, rhs, class)
            }
            ArithBinOp::Sub => {
                self.lower_intrinsic_arith_expr(IntrinsicArithBinOp::Sub, checked, lhs, rhs, class)
            }
            ArithBinOp::Mul => {
                self.lower_intrinsic_arith_expr(IntrinsicArithBinOp::Mul, checked, lhs, rhs, class)
            }
            ArithBinOp::Div => {
                self.lower_intrinsic_arith_expr(IntrinsicArithBinOp::Div, checked, lhs, rhs, class)
            }
            ArithBinOp::Rem => {
                self.lower_intrinsic_arith_expr(IntrinsicArithBinOp::Rem, checked, lhs, rhs, class)
            }
            ArithBinOp::Pow => {
                self.lower_intrinsic_arith_expr(IntrinsicArithBinOp::Pow, checked, lhs, rhs, class)
            }
            ArithBinOp::BitAnd
            | ArithBinOp::BitOr
            | ArithBinOp::BitXor
            | ArithBinOp::LShift
            | ArithBinOp::RShift => RExpr::Binary {
                op: BinOp::Arith(op),
                lhs,
                rhs,
            },
            ArithBinOp::Range => return None,
        })
    }

    fn lower_unary_expr_for_mode(
        &mut self,
        bb: RBlockId,
        op: UnOp,
        checked: bool,
        value: RLocalId,
        semantic_ty: TyId<'db>,
        class: &ScalarClass<'db>,
    ) -> Option<RExpr<'db>> {
        Some(match op {
            UnOp::Minus if checked => {
                let zero = self.alloc_zero_scalar(bb, semantic_ty, class);
                self.lower_intrinsic_arith_expr(IntrinsicArithBinOp::Sub, true, zero, value, class)
            }
            UnOp::Minus | UnOp::BitNot | UnOp::Not => RExpr::Unary { op, value },
            UnOp::Plus | UnOp::Mut | UnOp::Ref => return None,
        })
    }

    fn runtime_place_from_addr_value(&self, value: RLocalId) -> Option<RuntimePlace<'db>> {
        match self.value_class(value)? {
            RuntimeClass::Ref { .. } => Some(RuntimePlace {
                root: PlaceRoot::Ref(value),
                path: Box::default(),
            }),
            RuntimeClass::RawAddr { space, target } => {
                let pointee = if let Some(layout) = target {
                    RuntimeClass::AggregateValue { layout: *layout }
                } else {
                    let (_, inner) = self.locals[value.index()].semantic_ty.as_borrow(self.db)?;
                    self.top_level_class_for_ty(inner, *space)?
                };
                Some(RuntimePlace {
                    root: PlaceRoot::Ptr {
                        addr: value,
                        space: *space,
                        class: pointee,
                    },
                    path: Box::default(),
                })
            }
            RuntimeClass::Scalar(_) | RuntimeClass::AggregateValue { .. } => None,
        }
    }

    fn lower_core_primitive_wrapper_call(
        &mut self,
        bb: RBlockId,
        semantic: SemanticInstance<'db>,
        typed_body: &hir::analysis::ty::ty_check::TypedBody<'db>,
        args: &[NOperand],
    ) -> Option<RLocalId> {
        let BodyOwner::Func(func) = semantic.key(self.db).owner(self.db) else {
            return None;
        };
        let ret_ty = semantic_return_ty(self.db, semantic);
        let kind = core_primitive_wrapper_call_kind(self.db, func, ret_ty)?;
        let checked = self.current_arithmetic_mode() == ArithmeticMode::Checked;
        let (runtime_args, _) = self.lower_visible_call_args(bb, semantic, typed_body, args);

        match kind {
            PrimitiveWrapperCallKind::Unary(op) => {
                let [value] = runtime_args.as_slice() else {
                    return None;
                };
                let RuntimeClass::Scalar(class) =
                    self.top_level_class_for_ty(ret_ty, AddressSpaceKind::Memory)?
                else {
                    return None;
                };
                let ret = self.alloc_runtime_temp(
                    ret_ty,
                    RuntimeCarrier::Value(RuntimeClass::Scalar(class.clone())),
                );
                let expr =
                    self.lower_unary_expr_for_mode(bb, op, checked, *value, ret_ty, &class)?;
                self.push_stmt(bb, RStmt::Assign { dst: ret, expr });
                Some(ret)
            }
            PrimitiveWrapperCallKind::Binary(op) => {
                let [lhs, rhs] = runtime_args.as_slice() else {
                    return None;
                };
                let ret_class = self.top_level_class_for_ty(ret_ty, AddressSpaceKind::Memory)?;
                let ret = self.alloc_runtime_temp(ret_ty, RuntimeCarrier::Value(ret_class.clone()));
                let expr = match op {
                    BinOp::Arith(op) => {
                        let RuntimeClass::Scalar(class) = &ret_class else {
                            return None;
                        };
                        self.lower_arith_expr_for_mode(bb, op, checked, *lhs, *rhs, class)?
                    }
                    BinOp::Comp(_) | BinOp::Logical(_) => RExpr::Binary {
                        op,
                        lhs: *lhs,
                        rhs: *rhs,
                    },
                    BinOp::Index => return None,
                };
                self.push_stmt(bb, RStmt::Assign { dst: ret, expr });
                Some(ret)
            }
            PrimitiveWrapperCallKind::Assign(op) => {
                let [dst_addr, rhs] = runtime_args.as_slice() else {
                    return None;
                };
                let place = self.runtime_place_from_addr_value(*dst_addr)?;
                let target = self.project_place_class(&place);
                let RuntimeClass::Scalar(class) = target.clone() else {
                    return None;
                };
                let lhs = self.alloc_runtime_temp(
                    self.locals[args[0].local.index()].semantic_ty,
                    RuntimeCarrier::Value(target.clone()),
                );
                self.push_stmt(
                    bb,
                    RStmt::Assign {
                        dst: lhs,
                        expr: RExpr::Load {
                            place: place.clone(),
                        },
                    },
                );
                let result = self.alloc_runtime_temp(
                    self.locals[args[0].local.index()].semantic_ty,
                    RuntimeCarrier::Value(target.clone()),
                );
                let expr = match op {
                    BinOp::Arith(op) => {
                        self.lower_arith_expr_for_mode(bb, op, checked, lhs, *rhs, &class)?
                    }
                    BinOp::Comp(_) | BinOp::Logical(_) | BinOp::Index => return None,
                };
                self.push_stmt(bb, RStmt::Assign { dst: result, expr });
                self.write_value_to_place(bb, place, result, &target);
                Some(self.alloc_runtime_temp(TyId::unit(self.db), RuntimeCarrier::Erased))
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
        if let Some(ret) = self.lower_core_primitive_wrapper_call(bb, semantic, &typed_body, args) {
            return ret;
        }
        if let Some(ret) = self.lower_extern_builtin_call(bb, semantic, args, effect_args) {
            return ret;
        }
        let (mut runtime_args, mut runtime_classes) =
            self.lower_visible_call_args(bb, semantic, &typed_body, args);
        for effect_arg in effect_args {
            let plan = runtime_effect_binding_plan_for_binding_idx(
                self.db,
                semantic,
                effect_arg.binding_idx,
            );
            if let Some((value, class)) = self.lower_effect_arg(bb, effect_arg, plan.as_ref()) {
                runtime_args.push(value);
                runtime_classes.push(class);
            }
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
        let (args, _) = self.lower_visible_call_args(bb, semantic, &typed_body, args);
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
        let (args, _) = self.lower_visible_call_args(bb, semantic, &typed_body, args);
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
                self.lower_arith_expr_for_mode(bb, op, true, *lhs, *rhs, &scalar)?
            }
            Some(GenericNumericIntrinsicKind::CheckedNeg) => {
                let [value] = args.as_slice() else {
                    return None;
                };
                self.lower_unary_expr_for_mode(bb, UnOp::Minus, true, *value, ret_ty, &scalar)?
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
                    checked: false,
                    lhs: *lhs,
                    rhs: *rhs,
                    class: scalar.clone(),
                })
            }
            "sub" => {
                let [lhs, rhs] = args else { return None };
                RExpr::Builtin(crate::runtime::RuntimeBuiltin::IntrinsicArith {
                    op: IntrinsicArithBinOp::Sub,
                    checked: false,
                    lhs: *lhs,
                    rhs: *rhs,
                    class: scalar.clone(),
                })
            }
            "mul" => {
                let [lhs, rhs] = args else { return None };
                RExpr::Builtin(crate::runtime::RuntimeBuiltin::IntrinsicArith {
                    op: IntrinsicArithBinOp::Mul,
                    checked: false,
                    lhs: *lhs,
                    rhs: *rhs,
                    class: scalar.clone(),
                })
            }
            "div" => {
                let [lhs, rhs] = args else { return None };
                RExpr::Builtin(crate::runtime::RuntimeBuiltin::IntrinsicArith {
                    op: IntrinsicArithBinOp::Div,
                    checked: false,
                    lhs: *lhs,
                    rhs: *rhs,
                    class: scalar.clone(),
                })
            }
            "rem" => {
                let [lhs, rhs] = args else { return None };
                RExpr::Builtin(crate::runtime::RuntimeBuiltin::IntrinsicArith {
                    op: IntrinsicArithBinOp::Rem,
                    checked: false,
                    lhs: *lhs,
                    rhs: *rhs,
                    class: scalar.clone(),
                })
            }
            "pow" => {
                let [lhs, rhs] = args else { return None };
                RExpr::Builtin(crate::runtime::RuntimeBuiltin::IntrinsicArith {
                    op: IntrinsicArithBinOp::Pow,
                    checked: false,
                    lhs: *lhs,
                    rhs: *rhs,
                    class: scalar.clone(),
                })
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
        semantic: SemanticInstance<'db>,
        typed_body: &hir::analysis::ty::ty_check::TypedBody<'db>,
        args: &[NOperand],
    ) -> (Vec<RLocalId>, Vec<RuntimeClass<'db>>) {
        let mut runtime_args = Vec::with_capacity(args.len());
        let mut runtime_classes = Vec::with_capacity(args.len());
        for (idx, arg) in args.iter().enumerate() {
            match desired_runtime_param_plan(self.db, semantic, typed_body, idx) {
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

    fn lower_extern_builtin(
        &self,
        func: Func<'db>,
        args: &[RLocalId],
    ) -> Option<LoweredBuiltinCall<'db>> {
        let kind = runtime_builtin_func_kind(self.db, func)?;
        let word = RuntimeClass::Scalar(ScalarClass {
            repr: ScalarRepr::Int {
                bits: 256,
                signed: false,
            },
            role: ScalarRole::Plain,
        });
        let builtin = |builtin, class| LoweredBuiltinCall::Expr { builtin, class };
        Some(match kind {
            RuntimeBuiltinFuncKind::Malloc => {
                let [size] = args else { return None };
                builtin(
                    crate::runtime::RuntimeBuiltin::Malloc { size: *size },
                    Some(word.clone()),
                )
            }
            RuntimeBuiltinFuncKind::Mload => {
                let [addr] = args else { return None };
                builtin(
                    crate::runtime::RuntimeBuiltin::Mload { addr: *addr },
                    Some(word.clone()),
                )
            }
            RuntimeBuiltinFuncKind::Mstore => {
                let [addr, value] = args else { return None };
                builtin(
                    crate::runtime::RuntimeBuiltin::Mstore {
                        addr: *addr,
                        value: *value,
                    },
                    None,
                )
            }
            RuntimeBuiltinFuncKind::Mstore8 => {
                let [addr, value] = args else { return None };
                builtin(
                    crate::runtime::RuntimeBuiltin::Mstore8 {
                        addr: *addr,
                        value: *value,
                    },
                    None,
                )
            }
            RuntimeBuiltinFuncKind::Msize => {
                let [] = args else { return None };
                builtin(crate::runtime::RuntimeBuiltin::Msize, Some(word.clone()))
            }
            RuntimeBuiltinFuncKind::Sload => {
                let [slot] = args else { return None };
                builtin(
                    crate::runtime::RuntimeBuiltin::Sload { slot: *slot },
                    Some(word.clone()),
                )
            }
            RuntimeBuiltinFuncKind::Sstore => {
                let [slot, value] = args else { return None };
                builtin(
                    crate::runtime::RuntimeBuiltin::Sstore {
                        slot: *slot,
                        value: *value,
                    },
                    None,
                )
            }
            RuntimeBuiltinFuncKind::CallDataLoad => {
                let [offset] = args else { return None };
                builtin(
                    crate::runtime::RuntimeBuiltin::CallDataLoad { offset: *offset },
                    Some(word.clone()),
                )
            }
            RuntimeBuiltinFuncKind::CallDataCopy => {
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
            }
            RuntimeBuiltinFuncKind::CallDataSize => {
                let [] = args else { return None };
                builtin(
                    crate::runtime::RuntimeBuiltin::CallDataSize,
                    Some(word.clone()),
                )
            }
            RuntimeBuiltinFuncKind::ReturnDataCopy => {
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
            }
            RuntimeBuiltinFuncKind::ReturnDataSize => {
                let [] = args else { return None };
                builtin(
                    crate::runtime::RuntimeBuiltin::ReturnDataSize,
                    Some(word.clone()),
                )
            }
            RuntimeBuiltinFuncKind::CodeCopy => {
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
            }
            RuntimeBuiltinFuncKind::CodeSize => {
                let [] = args else { return None };
                builtin(crate::runtime::RuntimeBuiltin::CodeSize, Some(word.clone()))
            }
            RuntimeBuiltinFuncKind::Keccak256 => {
                let [offset, len] = args else { return None };
                builtin(
                    crate::runtime::RuntimeBuiltin::Keccak256 {
                        offset: *offset,
                        len: *len,
                    },
                    Some(word.clone()),
                )
            }
            RuntimeBuiltinFuncKind::AddMod => {
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
            }
            RuntimeBuiltinFuncKind::MulMod => {
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
            }
            RuntimeBuiltinFuncKind::Address => {
                let [] = args else { return None };
                builtin(crate::runtime::RuntimeBuiltin::Address, Some(word.clone()))
            }
            RuntimeBuiltinFuncKind::Caller => {
                let [] = args else { return None };
                builtin(crate::runtime::RuntimeBuiltin::Caller, Some(word.clone()))
            }
            RuntimeBuiltinFuncKind::CallValue => {
                let [] = args else { return None };
                builtin(
                    crate::runtime::RuntimeBuiltin::CallValue,
                    Some(word.clone()),
                )
            }
            RuntimeBuiltinFuncKind::Origin => {
                let [] = args else { return None };
                builtin(crate::runtime::RuntimeBuiltin::Origin, Some(word.clone()))
            }
            RuntimeBuiltinFuncKind::GasPrice => {
                let [] = args else { return None };
                builtin(crate::runtime::RuntimeBuiltin::GasPrice, Some(word.clone()))
            }
            RuntimeBuiltinFuncKind::CoinBase => {
                let [] = args else { return None };
                builtin(crate::runtime::RuntimeBuiltin::CoinBase, Some(word.clone()))
            }
            RuntimeBuiltinFuncKind::Timestamp => {
                let [] = args else { return None };
                builtin(
                    crate::runtime::RuntimeBuiltin::Timestamp,
                    Some(word.clone()),
                )
            }
            RuntimeBuiltinFuncKind::Number => {
                let [] = args else { return None };
                builtin(crate::runtime::RuntimeBuiltin::Number, Some(word.clone()))
            }
            RuntimeBuiltinFuncKind::PrevRandao => {
                let [] = args else { return None };
                builtin(
                    crate::runtime::RuntimeBuiltin::PrevRandao,
                    Some(word.clone()),
                )
            }
            RuntimeBuiltinFuncKind::GasLimit => {
                let [] = args else { return None };
                builtin(crate::runtime::RuntimeBuiltin::GasLimit, Some(word.clone()))
            }
            RuntimeBuiltinFuncKind::ChainId => {
                let [] = args else { return None };
                builtin(crate::runtime::RuntimeBuiltin::ChainId, Some(word.clone()))
            }
            RuntimeBuiltinFuncKind::BaseFee => {
                let [] = args else { return None };
                builtin(crate::runtime::RuntimeBuiltin::BaseFee, Some(word.clone()))
            }
            RuntimeBuiltinFuncKind::SelfBalance => {
                let [] = args else { return None };
                builtin(
                    crate::runtime::RuntimeBuiltin::SelfBalance,
                    Some(word.clone()),
                )
            }
            RuntimeBuiltinFuncKind::BlockHash => {
                let [block] = args else { return None };
                builtin(
                    crate::runtime::RuntimeBuiltin::BlockHash { block: *block },
                    Some(word.clone()),
                )
            }
            RuntimeBuiltinFuncKind::Gas => {
                let [] = args else { return None };
                builtin(crate::runtime::RuntimeBuiltin::Gas, Some(word.clone()))
            }
            RuntimeBuiltinFuncKind::Call => {
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
            }
            RuntimeBuiltinFuncKind::StaticCall => {
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
            }
            RuntimeBuiltinFuncKind::DelegateCall => {
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
            }
            RuntimeBuiltinFuncKind::Create => {
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
            }
            RuntimeBuiltinFuncKind::Create2 => {
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
            }
            RuntimeBuiltinFuncKind::Log0 => {
                let [offset, len] = args else { return None };
                builtin(
                    crate::runtime::RuntimeBuiltin::Log0 {
                        offset: *offset,
                        len: *len,
                    },
                    None,
                )
            }
            RuntimeBuiltinFuncKind::Log1 => {
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
            }
            RuntimeBuiltinFuncKind::Log2 => {
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
            }
            RuntimeBuiltinFuncKind::Log3 => {
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
            }
            RuntimeBuiltinFuncKind::Log4 => {
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
            }
            RuntimeBuiltinFuncKind::Revert => {
                let [offset, len] = args else { return None };
                LoweredBuiltinCall::Terminator(RTerminator::Revert {
                    offset: *offset,
                    len: *len,
                })
            }
            RuntimeBuiltinFuncKind::ReturnData => {
                let [offset, len] = args else { return None };
                LoweredBuiltinCall::Terminator(RTerminator::ReturnData {
                    offset: *offset,
                    len: *len,
                })
            }
            RuntimeBuiltinFuncKind::SelfDestruct => {
                let [beneficiary] = args else { return None };
                LoweredBuiltinCall::Terminator(RTerminator::SelfDestruct {
                    beneficiary: *beneficiary,
                })
            }
            RuntimeBuiltinFuncKind::Stop => {
                let [] = args else { return None };
                LoweredBuiltinCall::Terminator(RTerminator::Stop)
            }
            RuntimeBuiltinFuncKind::Panic | RuntimeBuiltinFuncKind::Todo => {
                let [] = args else { return None };
                LoweredBuiltinCall::Terminator(RTerminator::Trap)
            }
            RuntimeBuiltinFuncKind::PanicWithValue => {
                let [_value] = args else { return None };
                LoweredBuiltinCall::Terminator(RTerminator::Trap)
            }
            RuntimeBuiltinFuncKind::IntrinsicKeccak256 => return None,
        })
    }

    fn lower_intrinsic_keccak256_call(
        &mut self,
        bb: RBlockId,
        func: Func<'db>,
        args: &[NOperand],
    ) -> Option<RLocalId> {
        if runtime_builtin_func_kind(self.db, func)
            != Some(RuntimeBuiltinFuncKind::IntrinsicKeccak256)
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
        plan: Option<&RuntimeEffectBindingPlan<'db>>,
    ) -> Option<(RLocalId, RuntimeClass<'db>)> {
        if plan.is_none() && arg.provider.is_none() && arg.target_ty.is_none() {
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
        let space = resolved_effect_arg_address_space(self.db, &self.semantic_body, arg);
        let boundary = desired_runtime_effect_arg_boundary(self.db, self.env, arg, plan, space);
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
                let enum_layout = self.enum_layout_for_local(value.local);
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
                if !CoercionPlanner::target_prefers_transport(&target) =>
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

    fn try_semantic_place(&mut self, bb: RBlockId, local: SLocalId) -> Option<RuntimePlace<'db>> {
        if let Some(place) = nonself_backing_value_place(&self.semantic_body, local).cloned() {
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
        let mut current = self.project_place_class(&runtime_place);
        let mut projected = Vec::new();
        for elem in place.path.iter() {
            match elem {
                Projection::Deref => {
                    panic!("unexpected deref in normalized runtime place: {place:?}")
                }
                Projection::Field(field) => {
                    let field = FieldIndex((*field).try_into().expect("field index fits in u16"));
                    projected.push(PlaceElem::Field(field));
                    current = project_field_class(self.db, current, field);
                }
                Projection::VariantField {
                    variant, field_idx, ..
                } => {
                    let field =
                        FieldIndex((*field_idx).try_into().expect("field index fits in u16"));
                    let variant = VariantId {
                        enum_layout: current
                            .aggregate_layout()
                            .expect("variant-field places should project from enum layouts"),
                        index: variant.0,
                    };
                    projected.push(PlaceElem::VariantField { variant, field });
                    current = project_variant_field_class(self.db, current, variant, field);
                }
                Projection::Index(IndexSource::Dynamic(index)) => {
                    projected.push(PlaceElem::Index(IndexSource::Dynamic(
                        self.read_semantic_value(bb, *index),
                    )));
                    current = project_index_class(self.db, current);
                }
                Projection::Index(IndexSource::Constant(index)) => {
                    projected.push(PlaceElem::Index(IndexSource::Constant(*index)));
                    current = project_index_class(self.db, current);
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

    fn coerce_value_if_needed(
        &mut self,
        bb: RBlockId,
        value: RLocalId,
        target: &RuntimeClass<'db>,
    ) -> RLocalId {
        if self.value_class(value).is_none() || self.value_class(value) == Some(target) {
            value
        } else {
            self.coerce_value(bb, value, target)
        }
    }

    fn lower_semantic_operand_for_class(
        &mut self,
        bb: RBlockId,
        operand: NOperand,
        target: &RuntimeClass<'db>,
    ) -> RLocalId {
        if matches!(target, RuntimeClass::AggregateValue { .. })
            && self.with_current_body_cx(|cx| {
                cx.env.boundary_source_transport_sensitive(operand.local)
            })
            && let Some(value) = self.actual_aggregate_value_from_runtime_source(bb, operand.local)
        {
            return self.coerce_value_if_needed(bb, value, target);
        }
        if !CoercionPlanner::target_prefers_transport(target)
            && let Some(value) = self.materialize_ordinary_direct_value(bb, operand.local)
        {
            return self.coerce_value_if_needed(bb, value, target);
        }
        if CoercionPlanner::target_prefers_transport(target) {
            if let Some(value) = self.handle_like_semantic_value(operand.local) {
                return self.coerce_value_if_needed(bb, value, target);
            }
            if let Some(value) = self.addr_of_semantic_operand_for_class(bb, operand, target) {
                return value;
            }
        }
        let value = self.read_semantic_operand(bb, operand);
        self.coerce_value_if_needed(bb, value, target)
    }

    fn actual_aggregate_value_from_runtime_source(
        &mut self,
        bb: RBlockId,
        local: SLocalId,
    ) -> Option<RLocalId> {
        if let Some(place) = snapshot_source_place(&self.semantic_body, local).cloned()
            && let Some(place) = self.try_lower_place(bb, &place)
        {
            let class = self.project_place_class(&place);
            let actual = actual_aggregate_class_from_runtime_source(&class)?;
            let value = self.alloc_runtime_temp(
                self.locals[local.index()].semantic_ty,
                RuntimeCarrier::Value(class.clone()),
            );
            self.push_stmt(
                bb,
                RStmt::Assign {
                    dst: value,
                    expr: RExpr::Load { place },
                },
            );
            return Some(self.coerce_value_if_needed(bb, value, &actual));
        }
        let value = self.read_semantic_value(bb, local);
        let class = self.value_class(value).cloned()?;
        let actual = actual_aggregate_class_from_runtime_source(&class)?;
        Some(self.coerce_value_if_needed(bb, value, &actual))
    }

    fn materialize_ordinary_direct_value(
        &mut self,
        bb: RBlockId,
        local: SLocalId,
    ) -> Option<RLocalId> {
        let local_data = self.semantic_body.local(local)?;
        if !matches!(
            (&local_data.facts.interface, &local_data.facts.origin),
            (
                NLocalInterface::DirectValue,
                NLocalOrigin::SelfRooted | NLocalOrigin::AliasedPlace
            )
        ) {
            return None;
        }
        let current = self.semantic_value_class(local)?;
        let target = self.with_current_body_cx(|cx| cx.materialized_value_class(local))?;
        if current == target {
            return None;
        }
        let place = self
            .try_semantic_place(bb, local)
            .or_else(|| self.place_from_direct_value_transport(local, &target))?;
        let place_class = self.project_place_class(&place);
        let temp = self.alloc_runtime_temp(
            self.locals[local.index()].semantic_ty,
            RuntimeCarrier::Value(place_class.clone()),
        );
        self.push_stmt(
            bb,
            RStmt::Assign {
                dst: temp,
                expr: RExpr::Load { place },
            },
        );
        Some(self.coerce_value_if_needed(bb, temp, &target))
    }

    fn place_from_direct_value_transport(
        &self,
        local: SLocalId,
        target: &RuntimeClass<'db>,
    ) -> Option<RuntimePlace<'db>> {
        let value = self.runtime_value(local);
        match self.value_class(value)? {
            RuntimeClass::Ref { .. } => Some(RuntimePlace {
                root: PlaceRoot::Ref(value),
                path: Box::default(),
            }),
            RuntimeClass::RawAddr {
                target: Some(layout),
                space,
            } => Some(RuntimePlace {
                root: PlaceRoot::Ptr {
                    addr: value,
                    space: *space,
                    class: RuntimeClass::AggregateValue { layout: *layout },
                },
                path: Box::default(),
            }),
            RuntimeClass::RawAddr {
                target: None,
                space,
            } if matches!(target, RuntimeClass::Scalar(_)) => Some(RuntimePlace {
                root: PlaceRoot::Ptr {
                    addr: value,
                    space: *space,
                    class: target.clone(),
                },
                path: Box::default(),
            }),
            RuntimeClass::Scalar(_) | RuntimeClass::AggregateValue { .. } => None,
            RuntimeClass::RawAddr { target: None, .. } => None,
        }
    }

    fn lower_semantic_operand_for_boundary(
        &mut self,
        bb: RBlockId,
        operand: NOperand,
        boundary: &crate::runtime::RuntimeBoundarySpec<'db>,
    ) -> RLocalId {
        let boundary = self
            .with_current_body_cx(|cx| cx.specialize_boundary_for_source(operand.local, boundary));
        self.lower_for_boundary(
            bb,
            BoundarySource::SemanticOperand(operand),
            &boundary,
            self.locals[operand.local.index()].semantic_ty,
        )
    }

    fn addr_of_semantic_operand_for_class(
        &mut self,
        bb: RBlockId,
        operand: NOperand,
        target: &RuntimeClass<'db>,
    ) -> Option<RLocalId> {
        if !CoercionPlanner::target_prefers_transport(target) {
            return None;
        }
        let local = operand.local;
        if let Some(place) = nonself_backing_value_place(&self.semantic_body, local).cloned()
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
            return Some(self.coerce_value_if_needed(bb, value, target));
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
        CoercionPlanner::target_prefers_transport(self.value_class(value)?).then_some(value)
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
            BoundarySource::RuntimePlace(place)
                if CoercionPlanner::target_prefers_transport(target) =>
            {
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
                    && self.value_class(value).is_some_and(|class| {
                        CoercionPlanner::class_satisfies_boundary(class, boundary)
                    })
                {
                    return Some(value);
                }
                let value = self.runtime_value(operand.local);
                self.value_class(value)
                    .is_some_and(|class| CoercionPlanner::class_satisfies_boundary(class, boundary))
                    .then_some(value)
            }
            BoundarySource::RuntimePlace(_) => None,
        }?;
        self.value_class(value)
            .is_some_and(|class| CoercionPlanner::class_satisfies_boundary(class, boundary))
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
                if let Some(place) =
                    nonself_backing_value_place(&self.semantic_body, local).cloned()
                    && let Some(place) = self.try_lower_place(bb, &place)
                {
                    Some(place)
                } else {
                    self.try_semantic_place(bb, local)
                }
            }
        }?;
        let actual = self.place_addr_class(&place);
        CoercionPlanner::class_satisfies_boundary(&actual, boundary)
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
        if CoercionPlanner::class_satisfies_boundary(&source_class, boundary) {
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
                .is_some_and(|class| CoercionPlanner::class_satisfies_boundary(class, boundary))
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
                .is_some_and(|class| CoercionPlanner::class_satisfies_boundary(class, boundary))
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
        let program = self.db as &dyn MirDb;
        let body = RuntimeBody {
            owner: self.instance,
            key: self.key,
            signature: RuntimeSignature {
                params: Vec::new(),
                ret: None,
            },
            semantic_locals: self.semantic_locals.clone(),
            provider_bindings: self.provider_bindings.clone(),
            locals: self.locals.clone(),
            blocks: Vec::new(),
        };
        resolve_runtime_place_address_class(self.db, &program, &body, place)
            .unwrap_or_else(|err| panic!("invalid runtime place address class: {err:?}"))
    }

    fn write_semantic_value(&mut self, bb: RBlockId, local: SLocalId, src: RLocalId) {
        match self.semantic_local_lowering(local).clone() {
            RuntimeLocalLowering::Erased => {}
            RuntimeLocalLowering::DirectValue | RuntimeLocalLowering::PlaceCarrier { .. } => {
                let dst = self.runtime_value(local);
                let Some(mut target) = self.value_class(dst).cloned() else {
                    return;
                };
                if self.runtime_local_uses_source_transport(dst)
                    && matches!(
                        target,
                        RuntimeClass::AggregateValue { .. }
                            | RuntimeClass::Ref {
                                kind: RefKind::Object,
                                ..
                            }
                    )
                    && let Some(actual) =
                        self.with_current_body_cx(|cx| cx.actual_aggregate_class_for_source(local))
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
        if !self.runtime_local_uses_source_transport(dst)
            || !matches!(
                target,
                RuntimeClass::AggregateValue { .. }
                    | RuntimeClass::Ref {
                        kind: RefKind::Object,
                        ..
                    }
            )
        {
            return target.clone();
        }
        let Some(actual) =
            self.with_current_body_cx(|cx| cx.actual_aggregate_class_for_source(src.local))
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
                PlaceElem::VariantField { variant, field } => {
                    project_variant_field_class(self.db, current, *variant, *field)
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
        let class = self
            .value_class(local)
            .and_then(|current| merge_runtime_class(self.db, current, &class))
            .unwrap_or(class);
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
        merge_runtime_class(self.db, actual, target).is_some_and(|merged| &merged == actual)
    }
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

fn padded_word_bytes(bytes: &[u8]) -> Vec<u8> {
    let mut word = [0; 32];
    word[..bytes.len()].copy_from_slice(bytes);
    trim_leading_zero_bytes(&word)
}

fn usize_word_bytes(value: usize) -> Vec<u8> {
    trim_leading_zero_bytes(&value.to_be_bytes())
}

fn trim_leading_zero_bytes(bytes: &[u8]) -> Vec<u8> {
    bytes
        .iter()
        .copied()
        .skip_while(|byte| *byte == 0)
        .collect()
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
