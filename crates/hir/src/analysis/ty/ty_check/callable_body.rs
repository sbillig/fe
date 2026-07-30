use crate::{
    analysis::{
        HirAnalysisDb,
        ty::{
            ProviderAddressSpace,
            ty_check::{ClosureCapture, ClosureInfo, LocalBinding},
            ty_def::{ClosureTy, TyId},
        },
    },
    hir_def::{EffectParamListId, ExprId, attr::ArithmeticMode},
};

use super::{BodyOwner, ReturnProvenance, ReturnSource, TypedBody};

/// The complete typed view of a callable body.
///
/// A closure reuses its enclosing HIR [`Body`] and [`TypedBody`], but has its
/// own root expression, parameters, result type, and return-borrow provider.
/// Keeping the owner and typed body paired prevents consumers from
/// accidentally reading enclosing-callable metadata for a closure.
#[derive(Clone, Copy, Debug)]
pub struct TypedCallableBody<'db> {
    owner: BodyOwner<'db>,
    typed_body: &'db TypedBody<'db>,
}

impl<'db> TypedCallableBody<'db> {
    pub fn new(owner: BodyOwner<'db>, typed_body: &'db TypedBody<'db>) -> Self {
        Self { owner, typed_body }
    }

    pub fn root_expr(self, db: &'db dyn HirAnalysisDb) -> Option<ExprId> {
        self.owner.root_expr(db)
    }

    pub fn param_bindings(self, db: &'db dyn HirAnalysisDb) -> Vec<LocalBinding<'db>> {
        if let BodyOwner::Closure {
            ty, receiver_mode, ..
        } = self.owner
        {
            let Some(_) = self.closure_info(db, ty) else {
                return Vec::new();
            };
            return vec![
                LocalBinding::closure_env(db, ty, receiver_mode),
                LocalBinding::closure_args(db, ty),
            ];
        }

        self.typed_body.param_bindings.clone()
    }

    pub fn param_binding(
        self,
        db: &'db dyn HirAnalysisDb,
        idx: usize,
    ) -> Option<LocalBinding<'db>> {
        if let BodyOwner::Closure {
            ty, receiver_mode, ..
        } = self.owner
        {
            self.closure_info(db, ty)?;
            return if idx == 0 {
                Some(LocalBinding::closure_env(db, ty, receiver_mode))
            } else if idx == 1 {
                Some(LocalBinding::closure_args(db, ty))
            } else {
                None
            };
        }

        self.typed_body.param_binding(idx)
    }

    pub fn result_ty(self, db: &'db dyn HirAnalysisDb) -> TyId<'db> {
        match self.owner {
            BodyOwner::Closure { ty, .. } => ty.ret_ty(db),
            _ => self.typed_body.result_ty(),
        }
    }

    pub fn return_borrow_provider(
        self,
        db: &'db dyn HirAnalysisDb,
    ) -> Option<ProviderAddressSpace> {
        match self.owner {
            BodyOwner::Closure { ty, .. } => self
                .closure_info(db, ty)
                .and_then(|info| info.return_borrow_provider),
            _ => self.typed_body.return_borrow_provider(),
        }
    }

    pub fn return_provenance(self, db: &'db dyn HirAnalysisDb) -> ReturnProvenance {
        self.typed_body.callable_return_provenance(db, self.owner)
    }

    pub fn forwarded_return_sources(self, db: &'db dyn HirAnalysisDb) -> Vec<ReturnSource> {
        self.typed_body
            .callable_forwarded_return_sources(db, self.owner)
    }

    pub fn closure_capture_plan(
        self,
        db: &'db dyn HirAnalysisDb,
        closure: ClosureTy<'db>,
    ) -> Option<Vec<ClosureCapture<'db>>> {
        let info = self.closure_info(db, closure)?;
        Some(
            info.captures
                .iter()
                .zip(closure.captures(db))
                .map(|(capture, ty)| ClosureCapture {
                    ty: *ty,
                    ..*capture
                })
                .collect(),
        )
    }

    pub fn arithmetic_mode(self, db: &'db dyn HirAnalysisDb) -> ArithmeticMode {
        self.owner.arithmetic_mode(db)
    }

    pub fn effects(self, db: &'db dyn HirAnalysisDb) -> EffectParamListId<'db> {
        self.owner.effects(db)
    }

    fn closure_info(
        self,
        db: &'db dyn HirAnalysisDb,
        closure: ClosureTy<'db>,
    ) -> Option<&'db ClosureInfo<'db>> {
        let def = closure.def(db);
        let info = self
            .typed_body
            .closure_info(def.expr)
            .filter(|info| info.def == def)?;
        (info.params.len() == closure.params(db).len()
            && info.captures.len() == closure.captures(db).len()
            && info.ty.call_mode(db) == closure.call_mode(db))
        .then_some(info)
    }
}
