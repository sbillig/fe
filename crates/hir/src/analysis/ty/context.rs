use salsa::Update;

use crate::analysis::{
    HirAnalysisDb,
    ty::{
        trait_def::{ImplementorId, TraitInstId},
        trait_resolution::{
            GoalSatisfiability, PredicateListId, Selection, TraitSolveCx, is_goal_satisfiable,
        },
    },
};
use crate::hir_def::scope_graph::ScopeId;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Update)]
pub struct ProofCx<'db> {
    solve_cx: TraitSolveCx<'db>,
}

impl<'db> ProofCx<'db> {
    pub fn new(db: &'db dyn HirAnalysisDb, scope: ScopeId<'db>) -> Self {
        Self {
            solve_cx: TraitSolveCx::new(db, scope),
        }
    }

    pub fn from_solve_cx(solve_cx: TraitSolveCx<'db>) -> Self {
        Self { solve_cx }
    }

    pub fn with_assumptions(self, assumptions: PredicateListId<'db>) -> Self {
        Self {
            solve_cx: self.solve_cx.with_assumptions(assumptions),
        }
    }

    pub fn assumptions(self) -> PredicateListId<'db> {
        self.solve_cx.assumptions()
    }

    pub fn solve_cx(self) -> TraitSolveCx<'db> {
        self.solve_cx
    }

    pub fn origin_scope(self, db: &'db dyn HirAnalysisDb) -> ScopeId<'db> {
        self.solve_cx.origin_scope(db)
    }

    pub fn normalization_scope_for_trait_inst(
        self,
        db: &'db dyn HirAnalysisDb,
        inst: TraitInstId<'db>,
    ) -> ScopeId<'db> {
        self.solve_cx.normalization_scope_for_trait_inst(db, inst)
    }

    pub fn search_ingots_for_trait_inst(
        self,
        db: &'db dyn HirAnalysisDb,
        inst: TraitInstId<'db>,
    ) -> (Option<crate::Ingot<'db>>, Option<crate::Ingot<'db>>) {
        self.solve_cx.search_ingots_for_trait_inst(db, inst)
    }

    pub(crate) fn select_impl(
        self,
        db: &'db dyn HirAnalysisDb,
        inst: TraitInstId<'db>,
    ) -> Selection<ImplementorId<'db>> {
        self.solve_cx.select_impl(db, inst)
    }

    pub fn is_goal_satisfiable(
        self,
        db: &'db dyn HirAnalysisDb,
        goal: crate::analysis::ty::canonical::Canonical<TraitInstId<'db>>,
    ) -> GoalSatisfiability<'db> {
        is_goal_satisfiable(db, self.solve_cx, goal).clone()
    }
}

impl<'db> From<TraitSolveCx<'db>> for ProofCx<'db> {
    fn from(value: TraitSolveCx<'db>) -> Self {
        Self::from_solve_cx(value)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Update)]
pub struct ImplOverlay<'db> {
    current_impl: Option<ImplementorId<'db>>,
}

impl<'db> ImplOverlay<'db> {
    pub fn none() -> Self {
        Self { current_impl: None }
    }

    pub fn with_current_impl(current_impl: ImplementorId<'db>) -> Self {
        Self {
            current_impl: Some(current_impl),
        }
    }

    pub fn current_impl(self) -> Option<ImplementorId<'db>> {
        self.current_impl
    }
}

impl<'db> Default for ImplOverlay<'db> {
    fn default() -> Self {
        Self::none()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Update)]
pub enum LoweringMode<'db> {
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
    pub fn trait_inst(self) -> Option<TraitInstId<'db>> {
        match self {
            Self::Normal => None,
            Self::ImplTraitSignature { trait_inst, .. }
            | Self::SelectedTraitBody { trait_inst, .. } => Some(trait_inst),
        }
    }

    pub fn self_ty(self) -> Option<crate::analysis::ty::ty_def::TyId<'db>> {
        match self {
            Self::Normal => None,
            Self::ImplTraitSignature { self_ty, .. } | Self::SelectedTraitBody { self_ty, .. } => {
                Some(self_ty)
            }
        }
    }

    pub fn current_impl(self) -> Option<ImplementorId<'db>> {
        match self {
            Self::Normal => None,
            Self::ImplTraitSignature { current_impl, .. }
            | Self::SelectedTraitBody { current_impl, .. } => current_impl,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Update)]
pub struct AnalysisCx<'db> {
    pub proof: ProofCx<'db>,
    pub overlay: ImplOverlay<'db>,
    pub mode: LoweringMode<'db>,
}

impl<'db> AnalysisCx<'db> {
    pub fn new(proof: ProofCx<'db>) -> Self {
        Self {
            proof,
            overlay: ImplOverlay::none(),
            mode: LoweringMode::Normal,
        }
    }

    pub fn from_solve_cx(solve_cx: TraitSolveCx<'db>) -> Self {
        Self::new(ProofCx::from_solve_cx(solve_cx))
    }

    pub fn with_overlay(mut self, overlay: ImplOverlay<'db>) -> Self {
        self.overlay = overlay;
        self
    }

    pub fn with_mode(mut self, mode: LoweringMode<'db>) -> Self {
        self.mode = mode;
        self
    }
}
