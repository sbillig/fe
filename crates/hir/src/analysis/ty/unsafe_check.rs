use crate::analysis::HirAnalysisDb;
use crate::analysis::place::resolve_place_field;
use crate::analysis::ty::diagnostics::{BodyDiag, FuncBodyDiag};
use crate::analysis::ty::ty_check::{BodyOwner, TypedBody};
use crate::hir_def::{
    BlockKind, Body, CallableDef, Cond, CondId, Expr, ExprId, Partial, Stmt, StmtId, UnOp,
};

/// Reports the unsafe operations of a body that are outside an unsafe
/// context: an `unsafe { .. }` block, or the body of an `unsafe fn`. These are
/// dereferencing a raw pointer, explicitly or by selecting a field through
/// it, and calling an `unsafe fn`.
///
/// Nested items and anonymous const bodies are checked as their own owners,
/// and never inherit an enclosing unsafe context.
pub(crate) fn check_unsafe_ops<'db>(
    db: &'db dyn HirAnalysisDb,
    owner: BodyOwner<'db>,
    typed_body: &TypedBody<'db>,
) -> Vec<FuncBodyDiag<'db>> {
    let Some(body) = owner.body(db) else {
        return Vec::new();
    };
    if let BodyOwner::Func(func) = owner
        && func.is_unsafe(db)
    {
        return Vec::new();
    }

    let mut checker = UnsafeChecker {
        db,
        body,
        typed_body,
        diags: Vec::new(),
    };
    checker.check_expr(body.expr(db));
    checker.diags
}

/// Walks the safe code of a body; it never enters an `unsafe` block, since
/// everything there is allowed.
struct UnsafeChecker<'db, 'a> {
    db: &'db dyn HirAnalysisDb,
    body: Body<'db>,
    typed_body: &'a TypedBody<'db>,
    diags: Vec<FuncBodyDiag<'db>>,
}

impl<'db> UnsafeChecker<'db, '_> {
    fn report_deref(&mut self, expr: ExprId) {
        self.diags.push(
            BodyDiag::UnsafeDerefRequiresUnsafe {
                primary: expr.span(self.body).into(),
            }
            .into(),
        );
    }

    /// Reports a call, method call or operator whose callee is an
    /// `unsafe fn`. A trait method call resolves to the trait's declaration,
    /// whose unsafety every implementation shares.
    fn check_call_target(&mut self, expr: ExprId) {
        if let Some(callable) = self.typed_body.callable_expr(expr)
            && let callee @ CallableDef::Func(func) = callable.callable_def()
            && func.is_unsafe(self.db)
        {
            self.diags.push(
                BodyDiag::UnsafeCallRequiresUnsafe {
                    primary: expr.span(self.body).into(),
                    callee,
                }
                .into(),
            );
        }
    }

    fn check_stmt(&mut self, stmt: StmtId) {
        let Partial::Present(stmt_data) = stmt.data(self.db, self.body) else {
            return;
        };

        match stmt_data {
            Stmt::Let(_, _, init) => {
                if let Some(init) = init {
                    self.check_expr(*init);
                }
            }
            Stmt::For(_, iter, body, _) => {
                self.check_expr(*iter);
                self.check_expr(*body);
            }
            Stmt::While(cond, body) => {
                self.check_cond(*cond);
                self.check_expr(*body);
            }
            Stmt::Continue | Stmt::Break => {}
            Stmt::Return(expr) => {
                if let Some(expr) = expr {
                    self.check_expr(*expr);
                }
            }
            Stmt::Expr(expr) => self.check_expr(*expr),
        }
    }

    fn check_expr(&mut self, expr: ExprId) {
        let Partial::Present(expr_data) = expr.data(self.db, self.body) else {
            return;
        };

        match expr_data {
            Expr::Lit(_) | Expr::Path(_) | Expr::UnsupportedMacroCall => {}
            Expr::Block(_, BlockKind::Unsafe) => {}
            Expr::Block(stmts, BlockKind::Normal) => {
                stmts.iter().for_each(|stmt| self.check_stmt(*stmt));
            }

            Expr::Un(inner, op) => {
                self.check_expr(*inner);
                if *op == UnOp::Deref {
                    let ty = self.typed_body.expr_ty(self.db, *inner);
                    if ty
                        .as_capability(self.db)
                        .map_or(ty, |(_, inner)| inner)
                        .as_ptr(self.db)
                        .is_some()
                    {
                        self.report_deref(expr);
                    }
                }
                self.check_call_target(expr);
            }

            Expr::Field(inner, field) => {
                self.check_expr(*inner);
                if let Some(field) = field.to_opt()
                    && resolve_place_field(self.db, self.typed_body.expr_ty(self.db, *inner), field)
                        .is_some_and(|resolved| resolved.implicit_deref_ty.is_some())
                {
                    self.report_deref(expr);
                }
            }

            Expr::Bin(lhs, rhs, _) | Expr::Assign(lhs, rhs) | Expr::AugAssign(lhs, rhs, _) => {
                self.check_expr(*lhs);
                self.check_expr(*rhs);
                self.check_call_target(expr);
            }

            Expr::Cast(inner, _) | Expr::ArrayRep(inner, _) => self.check_expr(*inner),

            Expr::Call(callee, args) => {
                self.check_expr(*callee);
                args.iter().for_each(|arg| self.check_expr(arg.expr));
                self.check_call_target(expr);
            }
            Expr::MethodCall(receiver, _, _, args) => {
                self.check_expr(*receiver);
                args.iter().for_each(|arg| self.check_expr(arg.expr));
                self.check_call_target(expr);
            }
            Expr::Assert(args) => args.iter().for_each(|arg| self.check_expr(arg.expr)),

            Expr::If(cond, then, else_) => {
                self.check_cond(*cond);
                self.check_expr(*then);
                if let Some(else_) = else_ {
                    self.check_expr(*else_);
                }
            }
            Expr::Match(scrutinee, arms) => {
                self.check_expr(*scrutinee);
                if let Partial::Present(arms) = arms {
                    arms.iter().for_each(|arm| self.check_expr(arm.body));
                }
            }
            Expr::With(bindings, body) => {
                bindings
                    .iter()
                    .for_each(|binding| self.check_expr(binding.value));
                self.check_expr(*body);
            }
            Expr::RecordInit(_, fields) => {
                fields.iter().for_each(|field| self.check_expr(field.expr));
            }
            Expr::Tuple(elems) | Expr::Array(elems) => {
                elems.iter().for_each(|elem| self.check_expr(*elem));
            }
        }
    }

    fn check_cond(&mut self, cond: CondId) {
        let Partial::Present(cond_data) = cond.data(self.db, self.body) else {
            return;
        };

        match cond_data {
            Cond::Expr(expr) | Cond::Let(_, expr) => self.check_expr(*expr),
            Cond::Bin(lhs, rhs, _) => {
                self.check_cond(*lhs);
                self.check_cond(*rhs);
            }
        }
    }
}
