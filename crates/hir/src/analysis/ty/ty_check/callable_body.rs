use crate::{
    analysis::{
        HirAnalysisDb,
        ty::{
            ProviderAddressSpace,
            trait_resolution::PredicateListId,
            ty_check::{
                ClosureCapture, ClosureCaptureConstruction, ClosureInfo, LocalBinding, ParamSite,
            },
            ty_def::{ClosureCaptureAccess, ClosureTy, TyId},
            ty_is_copy,
        },
    },
    hir_def::{EffectParamListId, ExprId, attr::ArithmeticMode},
};

use super::{BodyOwner, ClosureReceiverMode, ReturnProvenance, ReturnSource, TypedBody};

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

/// A validated join between a closure's body-local binding metadata and its
/// specialized type-level signature.
///
/// Capture and parameter order is ABI-significant. Consumers should use this
/// view instead of independently indexing [`ClosureInfo`] and [`ClosureTy`].
/// Body-local types may still contain parent or layout-template arguments that
/// differ from the normalized callee ABI, so coherence is established through
/// definition, cardinality, order, modes, and capture accesses.
#[derive(Clone, Copy, Debug)]
pub struct TypedClosureBody<'db> {
    ty: ClosureTy<'db>,
    info: &'db ClosureInfo<'db>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TypedClosureCapture<'db> {
    pub binding: LocalBinding<'db>,
    pub ty: TyId<'db>,
    pub access: ClosureCaptureAccess,
}

impl<'db> TypedClosureBody<'db> {
    pub fn ty(self) -> ClosureTy<'db> {
        self.ty
    }

    pub fn physical_param_bindings(
        self,
        db: &'db dyn HirAnalysisDb,
        receiver_mode: ClosureReceiverMode,
    ) -> [LocalBinding<'db>; 2] {
        [
            LocalBinding::closure_env(db, self.ty, receiver_mode),
            LocalBinding::closure_args(db, self.ty),
        ]
    }

    pub fn param_bindings(self) -> &'db [LocalBinding<'db>] {
        &self.info.params
    }

    pub fn captures(
        self,
        db: &'db dyn HirAnalysisDb,
    ) -> impl ExactSizeIterator<Item = TypedClosureCapture<'db>> + 'db {
        self.info
            .captures
            .iter()
            .zip(self.ty.captures_with_accesses(db))
            .map(|(capture, (ty, access))| TypedClosureCapture {
                binding: capture.binding,
                ty,
                access,
            })
    }

    fn capture_plan(
        self,
        db: &'db dyn HirAnalysisDb,
        assumptions: PredicateListId<'db>,
    ) -> Vec<ClosureCapture<'db>> {
        let scope = self.ty.def(db).body.scope();
        self.captures(db)
            .map(|capture| ClosureCapture {
                binding: capture.binding,
                ty: capture.ty,
                construction: if ty_is_copy(db, scope, capture.ty, assumptions) {
                    ClosureCaptureConstruction::Copy
                } else {
                    ClosureCaptureConstruction::Move
                },
                access: capture.access,
            })
            .collect()
    }

    fn return_borrow_provider(self) -> Option<ProviderAddressSpace> {
        self.info.return_borrow_provider
    }
}

impl<'db> TypedCallableBody<'db> {
    pub fn new(owner: BodyOwner<'db>, typed_body: &'db TypedBody<'db>) -> Self {
        Self { owner, typed_body }
    }

    pub fn root_expr(self, db: &'db dyn HirAnalysisDb) -> Option<ExprId> {
        self.owner.root_expr(db)
    }

    pub fn param_bindings(self, db: &'db dyn HirAnalysisDb) -> Vec<LocalBinding<'db>> {
        if matches!(self.owner, BodyOwner::Closure { .. }) {
            let Some((closure, receiver_mode)) = self.owner_closure_body(db) else {
                return Vec::new();
            };
            return closure.physical_param_bindings(db, receiver_mode).to_vec();
        }

        self.typed_body.param_bindings.clone()
    }

    pub fn param_binding(
        self,
        db: &'db dyn HirAnalysisDb,
        idx: usize,
    ) -> Option<LocalBinding<'db>> {
        if matches!(self.owner, BodyOwner::Closure { .. }) {
            let (closure, receiver_mode) = self.owner_closure_body(db)?;
            return closure
                .physical_param_bindings(db, receiver_mode)
                .get(idx)
                .copied();
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
            BodyOwner::Closure { .. } => self
                .owner_closure_body(db)
                .and_then(|(closure, _)| closure.return_borrow_provider()),
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
        Some(
            self.closure_body(db, closure)?
                .capture_plan(db, self.typed_body.assumptions()),
        )
    }

    pub fn arithmetic_mode(self, db: &'db dyn HirAnalysisDb) -> ArithmeticMode {
        self.owner.arithmetic_mode(db)
    }

    pub fn effects(self, db: &'db dyn HirAnalysisDb) -> EffectParamListId<'db> {
        self.owner.effects(db)
    }

    /// Returns the descriptor for `closure` only when its body metadata and
    /// specialized type agree on every ABI-significant ordering invariant.
    pub fn closure_body(
        self,
        db: &'db dyn HirAnalysisDb,
        closure: ClosureTy<'db>,
    ) -> Option<TypedClosureBody<'db>> {
        let def = closure.def(db);
        let info = self
            .typed_body
            .closure_info(def.expr)
            .filter(|info| info.def == def)?;
        let stored_captures_match_type = info
            .captures
            .iter()
            .map(|capture| (capture.ty, capture.access))
            .eq(info.ty.captures_with_accesses(db));
        let stored_params_match_type =
            info.params
                .iter()
                .zip(info.ty.params(db))
                .enumerate()
                .all(|(idx, (binding, ty))| {
                    matches!(
                        binding,
                        LocalBinding::Param {
                            site: ParamSite::Closure(site),
                            idx: binding_idx,
                            ty: binding_ty,
                            ..
                        } if *site == def && *binding_idx == idx && binding_ty == ty
                    )
                });
        (info.ty.def(db) == def
            && info.params.len() == info.ty.params(db).len()
            && info.params.len() == closure.params(db).len()
            && stored_params_match_type
            && info.captures.len() == info.ty.captures_with_accesses(db).len()
            && info.captures.len() == closure.captures_with_accesses(db).len()
            && stored_captures_match_type
            && info.ty.param_modes(db) == closure.param_modes(db)
            && info.ty.capture_accesses(db) == closure.capture_accesses(db))
        .then_some(TypedClosureBody { ty: closure, info })
    }

    pub fn owner_closure_body(
        self,
        db: &'db dyn HirAnalysisDb,
    ) -> Option<(TypedClosureBody<'db>, ClosureReceiverMode)> {
        let BodyOwner::Closure {
            ty,
            def,
            receiver_mode,
        } = self.owner
        else {
            return None;
        };
        if ty.def(db) != def {
            return None;
        }
        self.closure_body(db, ty)
            .map(|closure| (closure, receiver_mode))
    }
}
