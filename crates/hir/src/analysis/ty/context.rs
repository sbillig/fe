use salsa::Update;

use crate::analysis::{
    HirAnalysisDb,
    ty::{
        trait_def::{ImplementorId, TraitInstId},
        trait_resolution::{PredicateListId, Selection, TraitSolveCx},
    },
};
use crate::hir_def::scope_graph::ScopeId;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Update)]
pub(crate) struct ProofCx<'db> {
    solve_cx: TraitSolveCx<'db>,
}

impl<'db> ProofCx<'db> {
    pub(crate) fn new(db: &'db dyn HirAnalysisDb, scope: ScopeId<'db>) -> Self {
        Self {
            solve_cx: TraitSolveCx::new(db, scope),
        }
    }

    pub(crate) fn from_solve_cx(solve_cx: TraitSolveCx<'db>) -> Self {
        Self { solve_cx }
    }

    pub(crate) fn with_assumptions(self, assumptions: PredicateListId<'db>) -> Self {
        Self {
            solve_cx: self.solve_cx.with_assumptions(assumptions),
        }
    }

    pub(crate) fn assumptions(self) -> PredicateListId<'db> {
        self.solve_cx.assumptions()
    }

    pub(crate) fn solve_cx(self) -> TraitSolveCx<'db> {
        self.solve_cx
    }

    pub(crate) fn normalization_scope_for_trait_inst(
        self,
        db: &'db dyn HirAnalysisDb,
        inst: TraitInstId<'db>,
    ) -> ScopeId<'db> {
        self.solve_cx.normalization_scope_for_trait_inst(db, inst)
    }

    pub(crate) fn search_ingots_for_trait_inst(
        self,
        db: &'db dyn HirAnalysisDb,
        inst: TraitInstId<'db>,
    ) -> (crate::Ingot<'db>, Option<crate::Ingot<'db>>) {
        self.solve_cx.search_ingots_for_trait_inst(db, inst)
    }

    pub(crate) fn select_impl(
        self,
        db: &'db dyn HirAnalysisDb,
        inst: TraitInstId<'db>,
    ) -> Selection<ImplementorId<'db>> {
        self.solve_cx.select_impl(db, inst)
    }
}

impl<'db> From<TraitSolveCx<'db>> for ProofCx<'db> {
    fn from(value: TraitSolveCx<'db>) -> Self {
        Self::from_solve_cx(value)
    }
}

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
    pub(crate) proof: ProofCx<'db>,
    pub(crate) overlay: ImplOverlay<'db>,
    pub(crate) mode: LoweringMode<'db>,
}

impl<'db> AnalysisCx<'db> {
    pub(crate) fn new(proof: ProofCx<'db>) -> Self {
        Self {
            proof,
            overlay: ImplOverlay::none(),
            mode: LoweringMode::Normal,
        }
    }

    pub(crate) fn from_solve_cx(solve_cx: TraitSolveCx<'db>) -> Self {
        Self::new(ProofCx::from_solve_cx(solve_cx))
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
