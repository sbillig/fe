use salsa::Update;

use crate::analysis::ty::{
    trait_def::{ImplementorId, TraitInstId},
    trait_resolution::TraitSolveCx,
};
use crate::analysis::{HirAnalysisDb, ty::trait_resolution::PredicateListId};
use crate::core::hir_def::scope_graph::ScopeId;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Update)]
pub(crate) struct ImplOverlay<'db> {
    current_impl: Option<ImplementorId<'db>>,
}

impl<'db> ImplOverlay<'db> {
    pub(crate) fn none() -> Self {
        Self { current_impl: None }
    }

    pub(crate) fn with_current_impl(current_impl: ImplementorId<'db>) -> Self {
        Self {
            current_impl: Some(current_impl),
        }
    }

    pub(crate) fn current_impl(self) -> Option<ImplementorId<'db>> {
        self.current_impl
    }
}

impl<'db> Default for ImplOverlay<'db> {
    fn default() -> Self {
        Self::none()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Update)]
pub(crate) enum LoweringMode<'db> {
    Normal,
    ImplTraitSignature {
        trait_inst: TraitInstId<'db>,
        self_ty: crate::analysis::ty::ty_def::TyId<'db>,
        current_impl: Option<ImplementorId<'db>>,
    },
    SelectedTraitBody {
        trait_inst: TraitInstId<'db>,
        self_ty: crate::analysis::ty::ty_def::TyId<'db>,
        current_impl: Option<ImplementorId<'db>>,
    },
}

impl<'db> LoweringMode<'db> {
    pub(crate) fn trait_inst(self) -> Option<TraitInstId<'db>> {
        match self {
            Self::Normal => None,
            Self::ImplTraitSignature { trait_inst, .. }
            | Self::SelectedTraitBody { trait_inst, .. } => Some(trait_inst),
        }
    }

    pub(crate) fn self_ty(self) -> Option<crate::analysis::ty::ty_def::TyId<'db>> {
        match self {
            Self::Normal => None,
            Self::ImplTraitSignature { self_ty, .. } | Self::SelectedTraitBody { self_ty, .. } => {
                Some(self_ty)
            }
        }
    }

    pub(crate) fn current_impl(self) -> Option<ImplementorId<'db>> {
        match self {
            Self::Normal => None,
            Self::ImplTraitSignature { current_impl, .. }
            | Self::SelectedTraitBody { current_impl, .. } => current_impl,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Update)]
pub(crate) struct AnalysisCx<'db> {
    pub(crate) proof: TraitSolveCx<'db>,
    pub(crate) overlay: ImplOverlay<'db>,
    pub(crate) mode: LoweringMode<'db>,
}

impl<'db> AnalysisCx<'db> {
    pub(crate) fn new(proof: TraitSolveCx<'db>) -> Self {
        Self {
            proof,
            overlay: ImplOverlay::none(),
            mode: LoweringMode::Normal,
        }
    }

    pub(crate) fn minimal(
        db: &'db dyn HirAnalysisDb,
        scope: ScopeId<'db>,
        assumptions: PredicateListId<'db>,
    ) -> Self {
        Self::new(TraitSolveCx::new(db, scope).with_assumptions(assumptions))
    }

    pub(crate) fn for_mode(
        db: &'db dyn HirAnalysisDb,
        scope: ScopeId<'db>,
        assumptions: PredicateListId<'db>,
        mode: LoweringMode<'db>,
    ) -> Self {
        Self::minimal(db, scope, assumptions)
            .with_overlay(
                mode.current_impl()
                    .map(ImplOverlay::with_current_impl)
                    .unwrap_or_default(),
            )
            .with_mode(mode)
    }

    pub(crate) fn from_solve_cx(solve_cx: TraitSolveCx<'db>) -> Self {
        Self::new(solve_cx)
    }

    pub(crate) fn assumptions(self) -> PredicateListId<'db> {
        self.proof.assumptions()
    }

    pub(crate) fn with_assumptions(mut self, assumptions: PredicateListId<'db>) -> Self {
        self.proof = self.proof.with_assumptions(assumptions);
        self
    }

    pub(crate) fn with_proof(mut self, proof: TraitSolveCx<'db>) -> Self {
        self.proof = proof;
        self
    }

    pub(crate) fn rebased(
        self,
        db: &'db dyn HirAnalysisDb,
        scope: ScopeId<'db>,
        assumptions: PredicateListId<'db>,
    ) -> Self {
        let mut proof = TraitSolveCx::new(db, scope).with_assumptions(assumptions);
        if let Some(local_implementors) = self.proof.local_implementors() {
            proof = proof.with_local_implementors(local_implementors);
        }
        self.with_proof(proof)
    }

    pub(crate) fn with_overlay(mut self, overlay: ImplOverlay<'db>) -> Self {
        self.overlay = overlay;
        self
    }

    pub(crate) fn with_mode(mut self, mode: LoweringMode<'db>) -> Self {
        self.mode = mode;
        self
    }
}
