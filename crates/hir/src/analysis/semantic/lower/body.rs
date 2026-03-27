use cranelift_entity::EntityRef;
use num_bigint::BigInt;
use rustc_hash::FxHashMap;

use crate::{
    analysis::semantic::instance::{
        SemanticInstance, SemanticInstanceKey, resolve_semantic_const_ref, semantic_callee_key,
    },
    analysis::{
        HirAnalysisDb,
        name_resolution::{PathRes, resolve_path},
        semantic::{
            Mutability, SBlock, SBlockId, SConst, SExpr, SLocal, SLocalId, SStmt, STerminator,
            SValueId, SemOrigin, SemanticBody, SemanticCalleeRef, VariantIndex, bool_const,
            bytes_const, int_const, sem_const_from_ty, unit_const,
        },
        ty::{
            const_ty::const_ty_or_abstract_from_assoc_const_use,
            ty_check::{BodyOwner, ConstRef, LocalBinding, RecordLike, TypedBody},
            ty_def::{BorrowKind, TyId},
        },
    },
    hir_def::{
        ArithBinOp, Body, CallArg, CallableDef, Cond, CondId, Expr, ExprId, Field as HirField,
        LitKind, MatchArm, Partial, PatId, PathId, Stmt, StmtId, VariantKind,
        expr::{BinOp, CompBinOp, LogicalBinOp, UnOp},
    },
};

use super::{
    effects::{lower_seq_effect_args, owner_effect_bindings},
    pattern::ArmVariants,
};

pub fn lower_to_smir<'db>(
    db: &'db dyn HirAnalysisDb,
    instance: SemanticInstance<'db>,
    template_owner: BodyOwner<'db>,
    typed_body: TypedBody<'db>,
) -> SemanticBody<'db> {
    let Some(body) = typed_body.body() else {
        return SemanticBody {
            owner: instance,
            template_owner,
            locals: Vec::new(),
            blocks: vec![SBlock {
                stmts: Vec::new(),
                terminator: STerminator::Return(None),
            }],
        };
    };

    let mut cx = SmirLowerCtxt::new(db, instance, template_owner, typed_body, body);
    let result = cx.lower_expr(body.expr(db));
    if !cx.is_terminated(cx.current) {
        cx.set_terminator(
            cx.current,
            if cx.expr_ty(body.expr(db)) == TyId::unit(db) {
                STerminator::Return(None)
            } else {
                STerminator::Return(Some(result))
            },
        );
    }
    cx.finish()
}

pub(super) struct SmirLowerCtxt<'db> {
    pub(super) db: &'db dyn HirAnalysisDb,
    pub(super) instance: SemanticInstance<'db>,
    pub(super) template_owner: BodyOwner<'db>,
    pub(super) typed_body: TypedBody<'db>,
    pub(super) body: Body<'db>,
    pub(super) owner_key: SemanticInstanceKey<'db>,
    pub(super) locals: Vec<SLocal<'db>>,
    pub(super) blocks: Vec<BlockState<'db>>,
    pub(super) binding_locals: FxHashMap<LocalBinding<'db>, SLocalId>,
    pub(super) with_binding_values: FxHashMap<ExprId, SValueId>,
    pub(super) current: SBlockId,
    pub(super) loop_stack: Vec<LoopScope>,
}

pub(super) struct BlockState<'db> {
    pub(super) stmts: Vec<SStmt<'db>>,
    pub(super) terminator: Option<STerminator<'db>>,
}

#[derive(Clone, Copy)]
pub(super) struct LoopScope {
    pub(super) continue_bb: SBlockId,
    pub(super) break_bb: SBlockId,
}

impl<'db> SmirLowerCtxt<'db> {
    fn new(
        db: &'db dyn HirAnalysisDb,
        instance: SemanticInstance<'db>,
        template_owner: BodyOwner<'db>,
        typed_body: TypedBody<'db>,
        body: Body<'db>,
    ) -> Self {
        let owner_key = instance.key(db);
        let mut cx = Self {
            db,
            instance,
            template_owner,
            typed_body,
            body,
            owner_key,
            locals: Vec::new(),
            blocks: Vec::new(),
            binding_locals: FxHashMap::default(),
            with_binding_values: FxHashMap::default(),
            current: SBlockId::from_u32(0),
            loop_stack: Vec::new(),
        };
        cx.collect_binding_locals();
        cx.current = cx.new_block();
        cx
    }

    fn finish(self) -> SemanticBody<'db> {
        let blocks = self
            .blocks
            .into_iter()
            .map(|block| SBlock {
                stmts: block.stmts,
                terminator: block.terminator.unwrap_or(STerminator::Return(None)),
            })
            .collect();

        SemanticBody {
            owner: self.instance,
            template_owner: self.template_owner,
            locals: self.locals,
            blocks,
        }
    }

    fn collect_binding_locals(&mut self) {
        let mut param_idx = 0;
        while let Some(binding) = self.typed_body.param_binding(param_idx) {
            self.alloc_binding_local(binding);
            param_idx += 1;
        }
        for binding in owner_effect_bindings(self.db, self.template_owner) {
            self.alloc_binding_local(binding);
        }

        for (pat, _) in self.body.pats(self.db).iter() {
            if let Some(binding) = self.typed_body.pat_binding(pat) {
                self.alloc_binding_local(binding);
            }
        }
    }

    pub(super) fn alloc_binding_local(&mut self, binding: LocalBinding<'db>) -> SLocalId {
        if let Some(&local) = self.binding_locals.get(&binding) {
            return local;
        }
        let local = self.alloc_local(
            self.typed_body.binding_ty(self.db, binding),
            if binding.is_mut() {
                Mutability::Mutable
            } else {
                Mutability::Immutable
            },
            Some(binding),
        );
        self.binding_locals.insert(binding, local);
        local
    }

    fn alloc_local(
        &mut self,
        ty: TyId<'db>,
        mutability: Mutability,
        source: Option<LocalBinding<'db>>,
    ) -> SLocalId {
        let id = SLocalId::from_u32(self.locals.len() as u32);
        self.locals.push(SLocal {
            ty,
            mutability,
            source,
        });
        id
    }

    fn alloc_temp(&mut self, ty: TyId<'db>) -> SLocalId {
        self.alloc_local(ty, Mutability::Immutable, None)
    }

    pub(super) fn new_block(&mut self) -> SBlockId {
        let id = SBlockId::from_u32(self.blocks.len() as u32);
        self.blocks.push(BlockState {
            stmts: Vec::new(),
            terminator: None,
        });
        id
    }

    pub(super) fn switch_to(&mut self, block: SBlockId) {
        self.current = block;
    }

    fn is_terminated(&self, block: SBlockId) -> bool {
        self.blocks[block.index()].terminator.is_some()
    }

    pub(super) fn push_stmt(&mut self, stmt: SStmt<'db>) {
        if !self.is_terminated(self.current) {
            self.blocks[self.current.index()].stmts.push(stmt);
        }
    }

    pub(super) fn set_terminator(&mut self, block: SBlockId, terminator: STerminator<'db>) {
        if self.blocks[block.index()].terminator.is_none() {
            self.blocks[block.index()].terminator = Some(terminator);
        }
    }

    pub(super) fn emit_expr(&mut self, ty: TyId<'db>, expr: SExpr<'db>) -> SValueId {
        let dst = self.alloc_temp(ty);
        self.push_stmt(SStmt::Assign { dst, expr });
        dst
    }

    pub(super) fn expr_ty(&self, expr: ExprId) -> TyId<'db> {
        self.typed_body.expr_ty(self.db, expr)
    }

    pub(super) fn unit_value(&mut self) -> SValueId {
        self.emit_expr(
            TyId::unit(self.db),
            SExpr::Const(SConst::Value(unit_const(self.db))),
        )
    }

    pub(super) fn lower_expr(&mut self, expr: ExprId) -> SValueId {
        let Partial::Present(expr_data) = expr.data(self.db, self.body) else {
            panic!("cannot lower absent expression")
        };

        match expr_data {
            Expr::Lit(lit) => self.lower_leaf_literal(expr, lit),
            Expr::Path(path) => self.lower_path_expr(expr, path),
            Expr::Tuple(elems) | Expr::Array(elems) => {
                let fields = elems.iter().map(|expr| self.lower_expr(*expr)).collect();
                self.emit_expr(
                    self.expr_ty(expr),
                    SExpr::AggregateMake {
                        ty: self.expr_ty(expr),
                        fields,
                    },
                )
            }
            Expr::ArrayRep(elem, _) => {
                let len = self.expr_ty(expr).field_count(self.db);
                let value = self.lower_expr(*elem);
                self.emit_expr(
                    self.expr_ty(expr),
                    SExpr::AggregateMake {
                        ty: self.expr_ty(expr),
                        fields: vec![value; len].into_boxed_slice(),
                    },
                )
            }
            Expr::RecordInit(path, fields) => self.lower_record_init(expr, *path, fields),
            Expr::Field(base, field) => {
                let base_expr = *base;
                let base = self.lower_expr(base_expr);
                let field = self.lower_field_index(self.expr_ty(base_expr), field.to_opt());
                self.emit_expr(self.expr_ty(expr), SExpr::Field { base, field })
            }
            Expr::Bin(base, index, BinOp::Index) => {
                let base = self.lower_expr(*base);
                let index = self.lower_expr(*index);
                self.emit_expr(self.expr_ty(expr), SExpr::Index { base, index })
            }
            Expr::Un(inner, UnOp::Mut | UnOp::Ref) => {
                let kind = match expr_data {
                    Expr::Un(_, UnOp::Mut) => BorrowKind::Mut,
                    Expr::Un(_, UnOp::Ref) => BorrowKind::Ref,
                    _ => unreachable!(),
                };
                let place = self.lower_place(*inner);
                self.emit_expr(
                    self.expr_ty(expr),
                    SExpr::Borrow {
                        place,
                        kind,
                        provider: self.typed_body.expr_prop(self.db, expr).borrow_provider,
                    },
                )
            }
            Expr::Un(inner, op) => {
                let value = self.lower_expr(*inner);
                self.emit_expr(self.expr_ty(expr), SExpr::Unary { op: *op, value })
            }
            Expr::Bin(lhs, rhs, op) => {
                let lhs = self.lower_expr(*lhs);
                let rhs = self.lower_expr(*rhs);
                self.emit_expr(self.expr_ty(expr), SExpr::Binary { op: *op, lhs, rhs })
            }
            Expr::Cast(value, to) => {
                let value = self.lower_expr(*value);
                self.emit_expr(
                    self.expr_ty(expr),
                    SExpr::Cast {
                        value,
                        to: to
                            .to_opt()
                            .map_or_else(|| self.expr_ty(expr), |_| self.expr_ty(expr)),
                    },
                )
            }
            Expr::Call(_, args) => self.lower_call(expr, None, args),
            Expr::MethodCall(receiver, _, _, args) => self.lower_call(expr, Some(*receiver), args),
            Expr::Assign(dst, src) => {
                let src = self.lower_expr(*src);
                let dst_place = self.lower_place(*dst);
                if dst_place.path.is_empty() {
                    self.push_stmt(SStmt::Assign {
                        dst: dst_place.local,
                        expr: SExpr::Use(src),
                    });
                } else {
                    self.push_stmt(SStmt::Store {
                        dst: dst_place,
                        src,
                    });
                }
                self.unit_value()
            }
            Expr::AugAssign(dst, src, op) => {
                let lhs = self.lower_expr(*dst);
                let rhs = self.lower_expr(*src);
                let sum = self.emit_expr(
                    self.expr_ty(*dst),
                    SExpr::Binary {
                        op: BinOp::Arith(*op),
                        lhs,
                        rhs,
                    },
                );
                let dst_place = self.lower_place(*dst);
                if dst_place.path.is_empty() {
                    self.push_stmt(SStmt::Assign {
                        dst: dst_place.local,
                        expr: SExpr::Use(sum),
                    });
                } else {
                    self.push_stmt(SStmt::Store {
                        dst: dst_place,
                        src: sum,
                    });
                }
                self.unit_value()
            }
            Expr::Block(stmts) => self.lower_block_expr(stmts),
            Expr::If(cond, then_expr, else_expr) => {
                self.lower_if_expr(*cond, *then_expr, *else_expr)
            }
            Expr::Match(scrutinee, arms) => self.lower_match_expr(expr, *scrutinee, arms),
            Expr::With(bindings, body) => self.lower_with_expr(bindings, *body),
        }
    }

    fn lower_leaf_literal(&mut self, expr: ExprId, lit: &LitKind<'db>) -> SValueId {
        let ty = self.expr_ty(expr);
        let value = match lit {
            LitKind::Int(int_id) => int_const(self.db, ty, int_id.data(self.db).clone().into()),
            LitKind::String(string_id) => {
                bytes_const(self.db, ty, string_id.data(self.db).as_bytes().to_vec())
            }
            LitKind::Bool(value) => bool_const(self.db, *value),
        };
        self.emit_expr(ty, SExpr::Const(SConst::Value(value)))
    }

    fn lower_const_ref(&mut self, expr: ExprId, const_ref: ConstRef<'db>) -> SValueId {
        let ty = self.expr_ty(expr);
        if let Some(const_ref) =
            resolve_semantic_const_ref(self.db, const_ref, ty, SemOrigin::Expr(expr))
        {
            return self.emit_expr(ty, SExpr::Const(SConst::Ref(const_ref)));
        }

        let symbolic = match const_ref {
            ConstRef::TraitConst(assoc) => {
                const_ty_or_abstract_from_assoc_const_use(self.db, assoc, ty)
                    .map(|const_ty| TyId::const_ty(self.db, const_ty))
                    .and_then(|const_ty| sem_const_from_ty(self.db, const_ty))
            }
            ConstRef::Const(_) => None,
        };
        let Some(symbolic) = symbolic else {
            panic!("const ref should resolve to a semantic instance: {const_ref:?}");
        };
        self.emit_expr(ty, SExpr::Const(SConst::Value(symbolic)))
    }

    fn lower_path_expr(&mut self, expr: ExprId, path: &Partial<PathId<'db>>) -> SValueId {
        if let Some(binding) = self.typed_body.expr_binding(expr) {
            return *self
                .binding_locals
                .get(&binding)
                .expect("binding local should be allocated");
        }
        if let Some(const_ref) = self.typed_body.expr_const_ref(expr) {
            return self.lower_const_ref(expr, const_ref);
        }

        let Some(path) = path.to_opt() else {
            panic!("missing path for expression {expr:?}");
        };
        let resolved = resolve_path(
            self.db,
            path,
            self.body.scope(),
            self.typed_body.assumptions(),
            false,
        )
        .ok();
        match resolved {
            Some(PathRes::EnumVariant(variant))
                if matches!(variant.kind(self.db), VariantKind::Unit) =>
            {
                self.emit_expr(
                    self.expr_ty(expr),
                    SExpr::EnumMake {
                        enum_ty: variant.ty,
                        variant: VariantIndex(variant.variant.idx),
                        fields: Box::new([]),
                    },
                )
            }
            Some(PathRes::Ty(ty)) | Some(PathRes::TyAlias(_, ty)) => {
                let generic_args = self.owner_key.subst(self.db);
                let ty = crate::analysis::semantic::instantiate_with_generic_args(
                    self.db,
                    ty,
                    generic_args.generic_args(self.db),
                );
                if let Some(value) = sem_const_from_ty(self.db, ty) {
                    self.emit_expr(self.expr_ty(expr), SExpr::Const(SConst::Value(value)))
                } else {
                    panic!(
                        "non-binding path expression is not lowerable without const/callee info: path={} ty={} data={:?}",
                        path.pretty_print(self.db),
                        ty.pretty_print(self.db),
                        ty.data(self.db),
                    )
                }
            }
            _ => {
                panic!(
                    "non-binding path expression is not lowerable without const/callee info: path={} resolved={resolved:?}",
                    path.pretty_print(self.db),
                )
            }
        }
    }

    fn lower_record_init(
        &mut self,
        expr: ExprId,
        path: Partial<PathId<'db>>,
        fields: &[HirField<'db>],
    ) -> SValueId {
        let Some(path) = path.to_opt() else {
            panic!("record init path missing")
        };
        let path_res = resolve_path(
            self.db,
            path,
            self.body.scope(),
            self.typed_body.assumptions(),
            false,
        )
        .ok();
        match path_res {
            Some(PathRes::EnumVariant(variant)) => {
                let mut values = vec![None; fields.len()];
                for field in fields {
                    let Some(label) = field.label_eagerly(self.db, self.body) else {
                        panic!("record variant init field label missing")
                    };
                    let idx = RecordLike::from_variant(variant)
                        .record_field_idx(self.db, label)
                        .expect("record variant field should resolve");
                    values[idx] = Some(self.lower_expr(field.expr));
                }
                self.emit_expr(
                    self.expr_ty(expr),
                    SExpr::EnumMake {
                        enum_ty: variant.ty,
                        variant: VariantIndex(variant.variant.idx),
                        fields: values
                            .into_iter()
                            .map(|value| value.expect("missing enum field"))
                            .collect(),
                    },
                )
            }
            _ => {
                let ty = self.expr_ty(expr);
                let mut values = vec![None; fields.len()];
                for field in fields {
                    let Some(label) = field.label_eagerly(self.db, self.body) else {
                        panic!("record init field label missing")
                    };
                    let idx = RecordLike::Type(ty)
                        .record_field_idx(self.db, label)
                        .expect("record field should resolve");
                    values[idx] = Some(self.lower_expr(field.expr));
                }
                self.emit_expr(
                    ty,
                    SExpr::AggregateMake {
                        ty,
                        fields: values
                            .into_iter()
                            .map(|value| value.expect("missing record field"))
                            .collect(),
                    },
                )
            }
        }
    }

    fn lower_call(
        &mut self,
        expr: ExprId,
        receiver: Option<ExprId>,
        args: &[CallArg<'db>],
    ) -> SValueId {
        let callable = self
            .typed_body
            .callable_expr(expr)
            .cloned()
            .unwrap_or_else(|| panic!("callable missing for call expression {expr:?}"));
        let mut values = Vec::with_capacity(args.len() + usize::from(receiver.is_some()));
        if let Some(receiver) = receiver {
            values.push(self.lower_expr(receiver));
        }
        for arg in args {
            values.push(self.lower_expr(arg.expr));
        }

        match callable.callable_def() {
            CallableDef::VariantCtor(variant) => self.emit_expr(
                self.expr_ty(expr),
                SExpr::EnumMake {
                    enum_ty: self.expr_ty(expr),
                    variant: VariantIndex(variant.idx),
                    fields: values.into_boxed_slice(),
                },
            ),
            CallableDef::Func(_) => {
                let callee_key =
                    semantic_callee_key(self.db, self.owner_key, &self.typed_body, &callable)
                        .expect("callable function should produce a semantic callee");
                let effect_args = self.lower_effect_args(expr);
                self.emit_expr(
                    self.expr_ty(expr),
                    SExpr::Call {
                        callee: SemanticCalleeRef { key: callee_key },
                        args: values.into_boxed_slice(),
                        effect_args,
                    },
                )
            }
        }
    }

    fn lower_block_expr(&mut self, stmts: &[StmtId]) -> SValueId {
        let Some((tail, head)) = stmts.split_last() else {
            return self.unit_value();
        };

        for stmt in head {
            self.lower_stmt(*stmt);
            if self.is_terminated(self.current) {
                return self.unit_value();
            }
        }

        match tail.data(self.db, self.body) {
            Partial::Present(Stmt::Expr(expr)) => self.lower_expr(*expr),
            _ => {
                self.lower_stmt(*tail);
                self.unit_value()
            }
        }
    }

    fn lower_stmt(&mut self, stmt: StmtId) {
        let Partial::Present(stmt_data) = stmt.data(self.db, self.body) else {
            panic!("cannot lower absent statement")
        };

        match stmt_data {
            Stmt::Let(pat, _, init) => {
                if let Some(init) = init {
                    let value = self.lower_expr(*init);
                    self.bind_pattern(*pat, value);
                }
            }
            Stmt::While(cond, body_expr) => self.lower_while(*cond, *body_expr),
            Stmt::For(pat, iter, body_expr, _) => self.lower_for(stmt, *pat, *iter, *body_expr),
            Stmt::Continue => {
                let scope = self
                    .loop_stack
                    .last()
                    .copied()
                    .expect("continue outside loop");
                self.set_terminator(self.current, STerminator::Goto(scope.continue_bb));
            }
            Stmt::Break => {
                let scope = self.loop_stack.last().copied().expect("break outside loop");
                self.set_terminator(self.current, STerminator::Goto(scope.break_bb));
            }
            Stmt::Return(expr) => {
                let value = expr.map(|expr| self.lower_expr(expr));
                self.set_terminator(
                    self.current,
                    if expr.is_some_and(|expr| self.expr_ty(expr) == TyId::unit(self.db)) {
                        STerminator::Return(None)
                    } else {
                        STerminator::Return(value)
                    },
                );
            }
            Stmt::Expr(expr) => {
                let _ = self.lower_expr(*expr);
            }
        }
    }

    fn lower_while(&mut self, cond: CondId, body_expr: ExprId) {
        let cond_bb = self.new_block();
        let body_bb = self.new_block();
        let exit_bb = self.new_block();
        self.set_terminator(self.current, STerminator::Goto(cond_bb));

        self.switch_to(cond_bb);
        self.lower_cond_branch(cond, body_bb, exit_bb);

        self.loop_stack.push(LoopScope {
            continue_bb: cond_bb,
            break_bb: exit_bb,
        });
        self.switch_to(body_bb);
        let _ = self.lower_expr(body_expr);
        if !self.is_terminated(self.current) {
            self.set_terminator(self.current, STerminator::Goto(cond_bb));
        }
        self.loop_stack.pop();

        self.switch_to(exit_bb);
    }

    fn lower_for(&mut self, stmt: StmtId, pat: PatId, iter: ExprId, body_expr: ExprId) {
        let seq = self
            .typed_body
            .for_loop_seq(stmt)
            .cloned()
            .unwrap_or_else(|| panic!("missing Seq resolution for for-loop {stmt:?}"));
        let iter_value = self.lower_expr(iter);
        let usize_ty = seq.len_callable.ret_ty(self.db);
        let len_callee =
            semantic_callee_key(self.db, self.owner_key, &self.typed_body, &seq.len_callable)
                .expect("Seq::len should lower to a semantic callee");
        let get_callee =
            semantic_callee_key(self.db, self.owner_key, &self.typed_body, &seq.get_callable)
                .expect("Seq::get should lower to a semantic callee");
        let idx_local = self.alloc_temp(usize_ty);
        self.push_stmt(SStmt::Assign {
            dst: idx_local,
            expr: SExpr::Const(SConst::Value(int_const(
                self.db,
                usize_ty,
                BigInt::default(),
            ))),
        });
        let len_effect_args = lower_seq_effect_args(self, &seq.len_effect_args);
        let len_value = self.emit_expr(
            usize_ty,
            SExpr::Call {
                callee: SemanticCalleeRef { key: len_callee },
                args: vec![iter_value].into_boxed_slice(),
                effect_args: len_effect_args,
            },
        );

        let cond_bb = self.new_block();
        let body_bb = self.new_block();
        let exit_bb = self.new_block();
        self.set_terminator(self.current, STerminator::Goto(cond_bb));

        self.switch_to(cond_bb);
        let cond = self.emit_expr(
            TyId::bool(self.db),
            SExpr::Binary {
                op: BinOp::Comp(CompBinOp::Lt),
                lhs: idx_local,
                rhs: len_value,
            },
        );
        self.set_terminator(
            self.current,
            STerminator::Branch {
                cond,
                then_bb: body_bb,
                else_bb: exit_bb,
            },
        );

        self.loop_stack.push(LoopScope {
            continue_bb: cond_bb,
            break_bb: exit_bb,
        });
        self.switch_to(body_bb);
        let get_effect_args = lower_seq_effect_args(self, &seq.get_effect_args);
        let elem = self.emit_expr(
            seq.elem_ty,
            SExpr::Call {
                callee: SemanticCalleeRef { key: get_callee },
                args: vec![iter_value, idx_local].into_boxed_slice(),
                effect_args: get_effect_args,
            },
        );
        self.bind_pattern(pat, elem);
        let _ = self.lower_expr(body_expr);
        if !self.is_terminated(self.current) {
            let one = self.emit_expr(
                usize_ty,
                SExpr::Const(SConst::Value(int_const(
                    self.db,
                    usize_ty,
                    BigInt::from(1u8),
                ))),
            );
            let next = self.emit_expr(
                usize_ty,
                SExpr::Binary {
                    op: BinOp::Arith(ArithBinOp::Add),
                    lhs: idx_local,
                    rhs: one,
                },
            );
            self.push_stmt(SStmt::Assign {
                dst: idx_local,
                expr: SExpr::Use(next),
            });
            self.set_terminator(self.current, STerminator::Goto(cond_bb));
        }
        self.loop_stack.pop();
        self.switch_to(exit_bb);
    }

    fn lower_if_expr(
        &mut self,
        cond: CondId,
        then_expr: ExprId,
        else_expr: Option<ExprId>,
    ) -> SValueId {
        let result_ty = self.expr_ty(then_expr);
        let result = self.alloc_temp(result_ty);
        let then_bb = self.new_block();
        let else_bb = self.new_block();
        let join_bb = self.new_block();

        self.lower_cond_branch(cond, then_bb, else_bb);

        self.switch_to(then_bb);
        let then_value = self.lower_expr(then_expr);
        if !self.is_terminated(self.current) {
            self.push_stmt(SStmt::Assign {
                dst: result,
                expr: SExpr::Use(then_value),
            });
            self.set_terminator(self.current, STerminator::Goto(join_bb));
        }

        self.switch_to(else_bb);
        let else_value = if let Some(expr) = else_expr {
            self.lower_expr(expr)
        } else {
            self.unit_value()
        };
        if !self.is_terminated(self.current) {
            self.push_stmt(SStmt::Assign {
                dst: result,
                expr: SExpr::Use(else_value),
            });
            self.set_terminator(self.current, STerminator::Goto(join_bb));
        }

        self.switch_to(join_bb);
        result
    }

    fn lower_match_expr(
        &mut self,
        expr: ExprId,
        scrutinee: ExprId,
        arms: &Partial<Vec<MatchArm>>,
    ) -> SValueId {
        let Partial::Present(arms) = arms else {
            panic!("match arms missing")
        };
        let value = self.lower_expr(scrutinee);
        let enum_ty = self.expr_ty(scrutinee);
        let result = self.alloc_temp(self.expr_ty(expr));
        let join_bb = self.new_block();

        if self.expr_ty(scrutinee).as_enum(self.db).is_none()
            || !arms
                .iter()
                .all(|arm| self.pattern_is_enum_dispatchable(arm.pat))
        {
            return self.lower_match_expr_with_branches(value, result, join_bb, arms);
        }

        let mut cases = Vec::new();
        let mut default = None;
        let mut arm_blocks = Vec::with_capacity(arms.len());

        for arm in arms {
            let block = self.new_block();
            arm_blocks.push((arm, block));
            match self.arm_variants(arm.pat) {
                ArmVariants::Variants(variants) => {
                    cases.extend(variants.into_iter().map(|variant| (variant, block)));
                }
                ArmVariants::Default => default = Some(block),
            }
        }

        self.set_terminator(
            self.current,
            STerminator::MatchEnum {
                value,
                enum_ty,
                cases: cases.into_boxed_slice(),
                default,
            },
        );

        for (arm, block) in arm_blocks {
            self.switch_to(block);
            self.bind_pattern(arm.pat, value);
            let arm_value = self.lower_expr(arm.body);
            if !self.is_terminated(self.current) {
                self.push_stmt(SStmt::Assign {
                    dst: result,
                    expr: SExpr::Use(arm_value),
                });
                self.set_terminator(self.current, STerminator::Goto(join_bb));
            }
        }

        self.switch_to(join_bb);
        result
    }

    fn lower_match_expr_with_branches(
        &mut self,
        value: SValueId,
        result: SLocalId,
        join_bb: SBlockId,
        arms: &[MatchArm],
    ) -> SValueId {
        let mut arm_blocks = Vec::with_capacity(arms.len());
        for arm in arms {
            arm_blocks.push((arm, self.new_block()));
        }

        let mut dispatch_bb = self.current;
        let mut exhausted = false;
        for (idx, (arm, arm_bb)) in arm_blocks.iter().enumerate() {
            if idx > 0 {
                self.switch_to(dispatch_bb);
            }

            if self.pattern_is_irrefutable(arm.pat) {
                self.set_terminator(self.current, STerminator::Goto(*arm_bb));
                exhausted = true;
                break;
            }

            let next_bb = self.new_block();
            self.lower_pattern_branch(arm.pat, value, *arm_bb, next_bb);
            dispatch_bb = next_bb;
        }

        if !exhausted {
            panic!("non-enum match lowering requires an irrefutable fallback arm");
        }

        for (arm, block) in arm_blocks {
            self.switch_to(block);
            self.bind_pattern(arm.pat, value);
            let arm_value = self.lower_expr(arm.body);
            if !self.is_terminated(self.current) {
                self.push_stmt(SStmt::Assign {
                    dst: result,
                    expr: SExpr::Use(arm_value),
                });
                self.set_terminator(self.current, STerminator::Goto(join_bb));
            }
        }

        self.switch_to(join_bb);
        result
    }

    fn lower_cond_branch(&mut self, cond: CondId, then_bb: SBlockId, else_bb: SBlockId) {
        let Partial::Present(cond_data) = cond.data(self.db, self.body) else {
            panic!("cannot lower absent condition")
        };

        match cond_data {
            Cond::Expr(expr) => {
                let cond = self.lower_expr(*expr);
                self.set_terminator(
                    self.current,
                    STerminator::Branch {
                        cond,
                        then_bb,
                        else_bb,
                    },
                );
            }
            Cond::Bin(lhs, rhs, LogicalBinOp::And) => {
                let rhs_bb = self.new_block();
                self.lower_cond_branch(*lhs, rhs_bb, else_bb);
                self.switch_to(rhs_bb);
                self.lower_cond_branch(*rhs, then_bb, else_bb);
            }
            Cond::Bin(lhs, rhs, LogicalBinOp::Or) => {
                let rhs_bb = self.new_block();
                self.lower_cond_branch(*lhs, then_bb, rhs_bb);
                self.switch_to(rhs_bb);
                self.lower_cond_branch(*rhs, then_bb, else_bb);
            }
            Cond::Let(pat, expr) => {
                let value = self.lower_expr(*expr);
                if self.pattern_is_irrefutable(*pat) {
                    self.bind_pattern(*pat, value);
                    self.set_terminator(self.current, STerminator::Goto(then_bb));
                } else {
                    self.lower_pattern_branch(*pat, value, then_bb, else_bb);
                    self.switch_to(then_bb);
                    self.bind_pattern(*pat, value);
                }
            }
        }
    }
}
