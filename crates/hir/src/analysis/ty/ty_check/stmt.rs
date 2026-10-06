use salsa::Update;

use crate::analysis::{
    HirAnalysisDb,
    name_resolution::method_selection::{MethodCandidate, select_method_candidate},
};
use crate::core::hir_def::{Expr, ExprId, IdentId, Partial, Pat, PatId, Stmt, StmtId, Trait, UnOp};
use crate::span::DynLazySpan;

use super::{
    Callable, LocalBinding, TyChecker,
    env::{TraitObligation, TraitObligationOrigin},
};
use crate::analysis::ty::{
    LayoutBundlePathStep,
    canonical::Canonicalized,
    corelib::resolve_core_trait,
    diagnostics::BodyDiag,
    fold::{TyFoldable, TyFolder},
    shape::Shape,
    trait_def::TraitInstId,
    trait_resolution::TraitSolveCx,
    ty_def::{BorrowKind, InvalidCause, TyId},
    visitor::TyVisitable,
};

/// A step of the collection protocol a `for` loop runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Update)]
pub enum ForLoopStep {
    /// `start(base)`: the first cursor.
    Start,
    /// `next(base, cursor)`: the cursor after.
    Next,
    /// `at(base, cursor)`: the element.
    At,
}

impl ForLoopStep {
    pub const ALL: [Self; 3] = [Self::Start, Self::Next, Self::At];
}

/// What a loop's body receives from each element.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Update)]
pub enum ForLoopItem {
    /// A copy, taken before the body runs, so the element's session ends
    /// first.
    Copy,
    /// The element's access, open while the body uses it.
    Access(BorrowKind),
}

/// A call of the collection protocol, resolved.
#[derive(Debug, Clone, PartialEq, Eq, Update)]
pub struct ForLoopCall<'db> {
    pub callable: Callable<'db>,
    pub effect_args: Vec<super::ResolvedEffectArg<'db>>,
}

/// How a `for` loop reaches its elements: the `Collection` protocol calls it
/// selects, and what its body receives.
#[derive(Debug, Clone, PartialEq, Eq, Update)]
pub struct ForLoopPlan<'db> {
    pub cursor_ty: TyId<'db>,
    pub item_ty: TyId<'db>,
    pub item: ForLoopItem,
    /// The calls, in `ForLoopStep` order.
    pub calls: [ForLoopCall<'db>; 3],
    /// The element is the indexed layout projection of the base. Semantic
    /// lowering uses this explicit desugaring fact to retain the dynamic
    /// array-index source on the element.
    pub element_layout_backing_source: bool,
}

impl<'db> ForLoopPlan<'db> {
    pub fn call(&self, step: ForLoopStep) -> &ForLoopCall<'db> {
        &self.calls[step as usize]
    }
}

impl<'db> TyVisitable<'db> for ForLoopPlan<'db> {
    fn visit_with<V>(&self, visitor: &mut V)
    where
        V: crate::analysis::ty::visitor::TyVisitor<'db> + ?Sized,
    {
        self.cursor_ty.visit_with(visitor);
        self.item_ty.visit_with(visitor);
        for call in &self.calls {
            call.callable.visit_with(visitor);
        }
    }
}

impl<'db> TyFoldable<'db> for ForLoopPlan<'db> {
    fn super_fold_with<F>(self, db: &'db dyn HirAnalysisDb, folder: &mut F) -> Self
    where
        F: TyFolder<'db>,
    {
        ForLoopPlan {
            cursor_ty: self.cursor_ty.fold_with(db, folder),
            item_ty: self.item_ty.fold_with(db, folder),
            calls: self.calls.map(|call| ForLoopCall {
                callable: call.callable.fold_with(db, folder),
                effect_args: call.effect_args,
            }),
            ..self
        }
    }
}

impl<'db> TyChecker<'db> {
    pub(super) fn check_stmt(&mut self, stmt: StmtId, expected: TyId<'db>) -> TyId<'db> {
        let Partial::Present(stmt_data) = self.env.stmt_data(stmt) else {
            return TyId::invalid(self.db, InvalidCause::ParseError);
        };

        match stmt_data {
            Stmt::Let(..) => self.check_let(stmt, stmt_data),
            Stmt::For(..) => self.check_for(stmt, stmt_data),
            Stmt::While(..) => self.check_while(stmt, stmt_data),
            Stmt::Continue => self.check_continue(stmt, stmt_data),
            Stmt::Break => self.check_break(stmt, stmt_data),
            Stmt::Return(..) => self.check_return(stmt, stmt_data),
            Stmt::Yield(expr) => self.check_yield(stmt, *expr),
            Stmt::Expr(expr) => self.check_expr(*expr, expected).ty,
        }
    }

    fn check_let(&mut self, stmt: StmtId, stmt_data: &Stmt<'db>) -> TyId<'db> {
        let Stmt::Let(pat, ascription, expr, else_) = stmt_data else {
            unreachable!()
        };

        let span = stmt.span(self.env.body()).into_let_stmt();

        let ascription = ascription.map(|ty| self.lower_ty(ty, span.clone().ty(), true));

        if let Some(expr) = expr {
            let prop = if let Some(ascription) = ascription {
                self.check_expr(*expr, ascription)
            } else {
                self.check_expr_unknown(*expr)
            };
            let layout = self.pattern_layout_context(*expr);
            self.check_pat_with_layout(*pat, prop.ty, layout.as_ref());
            if let Some(LocalBinding::Local { pat, .. }) = self.env.pat_binding(*pat) {
                self.env
                    .set_local_borrow_provider(pat, prop.borrow_provider);
            }

            // `let mut x = p.get()` binds a mutable copy of a projection's
            // `Copy` grant (an explicit `mut p` stays an access).
            let copies_grant = matches!(
                pat.data(self.db, self.body()),
                Partial::Present(Pat::Path(_, true))
            ) && !matches!(
                self.env.expr_data(*expr),
                Partial::Present(Expr::Un(_, UnOp::Ref | UnOp::Mut))
            ) && matches!(prop.shape, Some(Shape::Access(_, ty)) if self.ty_is_copy(ty));
            if copies_grant {
                self.consume_access(*expr);
            } else if prop.shape.is_some() {
                self.bind_pattern_source(*pat, *expr, &prop);
            } else if self.pattern_binds_any(*pat) {
                self.record_implicit_move_for_owned_expr(*expr, prop.ty);
            }
            if let Some(else_) = else_ {
                let else_ty = self
                    .check_expr_unknown(*else_)
                    .ty
                    .fold_with(self.db, &mut self.table);
                if !else_ty.is_never(self.db) && !else_ty.has_invalid(self.db) {
                    self.push_diag(BodyDiag::LetElseMustDiverge {
                        primary: else_.span(self.body()).into(),
                    });
                }
            } else if let Some(root) = self.env.pattern_store().root(*pat)
                && !self.env.pattern_store().is_irrefutable(self.db, root)
            {
                self.push_diag(BodyDiag::RefutableLetPattern {
                    primary: pat.span(self.body()).into(),
                });
            }
        } else {
            let ascription = ascription.unwrap_or_else(|| self.fresh_ty());
            if let Some(diag) = ascription.emit_wf_diag(
                self.db,
                TraitSolveCx::new(self.db, self.env.scope()),
                self.env.assumptions(),
                span.ty().into(),
            ) {
                self.push_diag(diag);
            }
            self.check_pat(*pat, ascription);
        }
        self.check_mutable_pattern_bindings(*pat);
        self.env.flush_pending_bindings();
        TyId::unit(self.db)
    }

    fn check_mutable_pattern_bindings(&mut self, pat: PatId) {
        let Partial::Present(pat_data) = pat.data(self.db, self.body()) else {
            return;
        };

        match pat_data {
            Pat::Path(_, is_mut) => {
                if !*is_mut {
                    return;
                }

                let Some(binding) = self.env.pat_binding(pat) else {
                    return;
                };
                let ty = self.env.lookup_binding_ty(&binding);
                if ty.has_invalid(self.db) || self.env.binding_access(&binding).is_none() {
                    return;
                }

                self.push_diag(BodyDiag::MutableBindingCannotBeCapability {
                    primary: pat.span(self.body()).into_path_pat().mut_token().into(),
                    ty,
                });
            }
            Pat::Tuple(pats) | Pat::PathTuple(_, pats) => {
                for &pat in pats {
                    self.check_mutable_pattern_bindings(pat);
                }
            }
            Pat::Record(_, fields) => {
                for field in fields {
                    self.check_mutable_pattern_bindings(field.pat);
                }
            }
            Pat::Or(lhs, rhs) => {
                self.check_mutable_pattern_bindings(*lhs);
                self.check_mutable_pattern_bindings(*rhs);
            }
            Pat::WildCard | Pat::Rest | Pat::Lit(..) => {}
        }
    }

    fn check_for(&mut self, stmt: StmtId, stmt_data: &Stmt<'db>) -> TyId<'db> {
        let Stmt::For(pat, expr, body, _unroll) = stmt_data else {
            unreachable!()
        };

        let expected = self.fresh_ty();
        let prop = self
            .check_expr(*expr, expected)
            .fold_with(self.db, &mut self.table);
        // The loop holds its base; `for pat in mut e` mutates the elements.
        if prop.shape.is_some() {
            self.consume_access(*expr);
        }
        let mutates = matches!(prop.shape, Some(Shape::Access(BorrowKind::Mut, _)));
        match self.plan_for_loop(*expr, prop.ty, mutates) {
            Some(mut plan) => {
                let layout = self
                    .pattern_layout_context_for_projection(*expr, &[LayoutBundlePathStep::Index])
                    .filter(|layout| {
                        self.projected_pattern_layout_ty(layout, &[])
                            .is_some_and(|projected| {
                                crate::analysis::ty::layout_shape_key(self.db, projected)
                                    == crate::analysis::ty::layout_shape_key(self.db, plan.item_ty)
                            })
                    });
                plan.element_layout_backing_source = layout.is_some();
                self.check_pat_with_layout(*pat, plan.item_ty, layout.as_ref());
                if let ForLoopItem::Access(kind) = plan.item {
                    self.bind_pattern_accesses(*pat, &Shape::Access(kind, plan.item_ty));
                }
                self.env.register_for_loop_plan(stmt, plan);
            }
            None => {
                self.check_pat(*pat, TyId::invalid(self.db, InvalidCause::Other));
            }
        }

        self.env.enter_loop(stmt);
        self.env.enter_scope(*body);
        self.env.flush_pending_bindings();

        let body_ty = self.fresh_ty();
        self.check_expr_with_discarded_result(*body, body_ty);

        self.env.leave_scope();
        self.env.leave_loop();

        TyId::unit(self.db)
    }

    /// The `Collection` protocol a loop over a base of type `base_ty` runs:
    /// `CollectionMut::at` for a loop that mutates its elements, and otherwise
    /// a copy of each `Copy` element or its `ref` access.
    fn plan_for_loop(
        &mut self,
        expr: ExprId,
        base_ty: TyId<'db>,
        mutates: bool,
    ) -> Option<ForLoopPlan<'db>> {
        if base_ty.has_invalid(self.db) {
            return None;
        }
        if base_ty.is_never(self.db) || base_ty.base_ty(self.db).is_ty_var(self.db) {
            self.push_diag(BodyDiag::TypeMustBeKnown(expr.span(self.body()).into()));
            return None;
        }
        let span: DynLazySpan<'db> = expr.span(self.body()).into();
        let collection = resolve_core_trait(self.db, self.env.scope(), &["iter", "Collection"])?;
        let collection_mut =
            resolve_core_trait(self.db, self.env.scope(), &["iter", "CollectionMut"])?;
        let (collection, collection_inst) = (
            collection,
            self.select_for_loop_trait(expr, base_ty, collection, "start")?,
        );
        let at_inst = if mutates {
            self.select_for_loop_trait(expr, base_ty, collection_mut, "at")?
        } else {
            collection_inst
        };
        let call = |this: &mut Self, trait_def: Trait<'db>, inst, name: &str, inputs| {
            let method = *trait_def
                .method_defs(this.db)
                .get(&IdentId::new(this.db, name.to_string()))?;
            let func_ty = this.instantiate_trait_method_to_term(method, base_ty, inst);
            let mut callable = Callable::new(this.db, func_ty, span.clone(), Some(inst)).ok()?;
            callable.set_checked_input_tys(inputs);
            let effect_args = this.resolve_callable_effects(span.clone(), &mut callable);
            Some(ForLoopCall {
                callable,
                effect_args,
            })
        };
        let start = call(self, collection, collection_inst, "start", vec![base_ty])?;
        let cursor_ty = *self
            .normalize_ty(start.callable.ret_ty(self.db))
            .generic_args(self.db)
            .first()?;
        let next = call(
            self,
            collection,
            collection_inst,
            "next",
            vec![base_ty, cursor_ty],
        )?;
        let at_trait = if mutates { collection_mut } else { collection };
        let at = call(self, at_trait, at_inst, "at", vec![base_ty, cursor_ty])?;
        let Some(Shape::Access(_, item_ty)) = at.callable.ret_shape(self.db) else {
            return None;
        };
        let item_ty = self.normalize_ty(item_ty);
        let item = if mutates {
            ForLoopItem::Access(BorrowKind::Mut)
        } else if self.ty_is_copy(item_ty) {
            ForLoopItem::Copy
        } else {
            ForLoopItem::Access(BorrowKind::Ref)
        };
        Some(ForLoopPlan {
            cursor_ty,
            item_ty,
            item,
            calls: [start, next, at],
            element_layout_backing_source: false,
        })
    }

    /// The instance of `trait_def`, selected through its method `method`, that
    /// a loop base of type `base_ty` implements.
    fn select_for_loop_trait(
        &mut self,
        expr: ExprId,
        base_ty: TyId<'db>,
        trait_def: Trait<'db>,
        method: &str,
    ) -> Option<TraitInstId<'db>> {
        let canonical = Canonicalized::new(self.db, base_ty);
        let candidate = select_method_candidate(
            self.db,
            &canonical,
            IdentId::new(self.db, method.to_string()),
            self.env.scope(),
            self.env.assumptions(),
            Some(trait_def),
        );
        let (cand, confirm) = match candidate {
            Ok(MethodCandidate::TraitMethod(cand)) => (cand, false),
            Ok(MethodCandidate::NeedsConfirmation(cand)) => (cand, true),
            _ => {
                self.push_diag(BodyDiag::TraitNotImplemented {
                    primary: expr.span(self.body()).into(),
                    ty: base_ty.pretty_print(self.db).to_string(),
                    trait_name: trait_def.name(self.db).to_opt()?,
                });
                return None;
            }
        };
        let inst = canonical.extract_solution(&mut self.table, cand.inst);
        if confirm {
            self.env.register_trait_obligation(TraitObligation {
                goal: inst,
                origin: TraitObligationOrigin::GenericConfirmation,
                span: expr.span(self.body()).into(),
            });
        }
        Some(inst)
    }

    fn check_while(&mut self, stmt: StmtId, stmt_data: &Stmt<'db>) -> TyId<'db> {
        let Stmt::While(cond, body) = stmt_data else {
            unreachable!()
        };

        // Keep let-chain bindings local to the loop condition/body.
        self.env.enter_lexical_scope();
        self.check_cond(*cond);

        self.env.enter_loop(stmt);
        self.env.enter_scope(*body);
        self.env.flush_pending_bindings();
        let body_ty = self.fresh_ty();
        self.check_expr_with_discarded_result(*body, body_ty);
        self.env.leave_scope();
        self.env.clear_pending_bindings();
        self.env.leave_loop();
        self.env.leave_scope();

        TyId::unit(self.db)
    }

    fn check_continue(&mut self, stmt: StmtId, stmt_data: &Stmt<'db>) -> TyId<'db> {
        assert!(matches!(stmt_data, Stmt::Continue));

        if self.env.current_loop().is_none() {
            let span = stmt.span(self.env.body());
            let diag = BodyDiag::LoopControlOutsideOfLoop {
                primary: span.into(),
                is_break: false,
            };
            self.push_diag(diag);
        }

        TyId::never(self.db)
    }

    fn check_break(&mut self, stmt: StmtId, stmt_data: &Stmt<'db>) -> TyId<'db> {
        assert!(matches!(stmt_data, Stmt::Break));

        if self.env.current_loop().is_none() {
            let span = stmt.span(self.env.body());
            let diag = BodyDiag::LoopControlOutsideOfLoop {
                primary: span.into(),
                is_break: true,
            };
            self.push_diag(diag);
        }

        TyId::never(self.db)
    }

    fn check_return(&mut self, stmt: StmtId, stmt_data: &Stmt<'db>) -> TyId<'db> {
        let Stmt::Return(expr) = stmt_data else {
            unreachable!()
        };
        // In a body with `yield` statements, a bare `return` ends the slide.
        if expr.is_some() || !self.explicit_yields || self.projection_shape.is_none() {
            self.check_exit_value(stmt, *expr);
        }
        TyId::never(self.db)
    }

    fn check_yield(&mut self, stmt: StmtId, expr: ExprId) -> TyId<'db> {
        if self.projection_shape.is_some() {
            self.check_exit_value(stmt, Some(expr));
        } else {
            self.check_expr(expr, self.expected);
            self.push_diag(BodyDiag::YieldOutsideProjection {
                primary: stmt.span(self.env.body()).into(),
            });
        }
        TyId::unit(self.db)
    }

    /// Checks the value a `return` or `yield` ends its path with.
    fn check_exit_value(&mut self, stmt: StmtId, expr: Option<ExprId>) {
        let (returned_expr, returned_prop, returned_ty, had_child_err) = if let Some(expr) = expr {
            let before = self.diags.len();
            let expected = self.fresh_ty();
            let prop = self.check_expr(expr, expected);
            let ty = expected.fold_with(self.db, &mut self.table);
            (Some(expr), Some(prop), ty, self.diags.len() > before)
        } else {
            (None, None, TyId::unit(self.db), false)
        };

        let ret_ty_ok = !had_child_err
            && !returned_ty.has_invalid(self.db)
            && self.table.unify(returned_ty, self.expected).is_ok();

        if !had_child_err && !returned_ty.has_invalid(self.db) && !ret_ty_ok {
            let func = self.env.func();
            let span = stmt.span(self.env.body());
            let diag = BodyDiag::ReturnedTypeMismatch {
                primary: span.into(),
                actual: returned_ty,
                expected: self.expected,
                func,
            };

            self.push_diag(diag);
        } else if ret_ty_ok && let Some(expr) = returned_expr {
            match self.projection_shape.clone() {
                Some(shape) => self.check_yield_site(expr, &shape),
                None => self.record_implicit_move_for_owned_expr(expr, self.expected),
            }
        }

        if ret_ty_ok
            && returned_expr.is_some()
            && let Some(prop) = returned_prop
            && let Some(provider) = prop.borrow_provider
        {
            self.first_return_borrow_provider.get_or_insert(provider);
        }
    }
}
