use salsa::Update;

use super::{
    context::{AnalysisCx, ProofCx},
    fold::{TyFoldable, TyFolder},
    trait_def::TraitInstId,
    trait_resolution::{PredicateListId, TraitSolveCx},
    visitor::{TyVisitable, TyVisitor},
};
use crate::{
    analysis::HirAnalysisDb,
    hir_def::{IdentId, scope_graph::ScopeId},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Update)]
pub struct AssocConstUse<'db> {
    origin_scope: ScopeId<'db>,
    assumptions: PredicateListId<'db>,
    inst: TraitInstId<'db>,
    name: IdentId<'db>,
    analysis_cx: Option<AnalysisCx<'db>>,
}

impl<'db> AssocConstUse<'db> {
    pub fn new(
        origin_scope: ScopeId<'db>,
        assumptions: PredicateListId<'db>,
        inst: TraitInstId<'db>,
        name: IdentId<'db>,
    ) -> Self {
        Self {
            origin_scope,
            assumptions,
            inst,
            name,
            analysis_cx: None,
        }
    }

    pub fn origin_scope(self) -> ScopeId<'db> {
        self.origin_scope
    }

    pub fn assumptions(self) -> PredicateListId<'db> {
        self.assumptions
    }

    pub fn inst(self) -> TraitInstId<'db> {
        self.inst
    }

    pub fn with_inst(self, inst: TraitInstId<'db>) -> Self {
        Self { inst, ..self }
    }

    pub(crate) fn with_analysis_cx(self, analysis_cx: AnalysisCx<'db>) -> Self {
        Self {
            analysis_cx: Some(analysis_cx),
            ..self
        }
    }

    pub(crate) fn analysis_cx(
        self,
        db: &'db dyn HirAnalysisDb,
        solve_cx: Option<TraitSolveCx<'db>>,
    ) -> Option<AnalysisCx<'db>> {
        let cx = self.analysis_cx?;
        let proof = ProofCx::from_solve_cx(
            solve_cx
                .unwrap_or_else(|| TraitSolveCx::new(db, self.origin_scope))
                .with_assumptions(self.assumptions),
        );
        Some(
            AnalysisCx::new(proof)
                .with_overlay(cx.overlay)
                .with_mode(cx.mode),
        )
    }

    pub fn with_env(self, origin_scope: ScopeId<'db>, assumptions: PredicateListId<'db>) -> Self {
        Self {
            origin_scope,
            assumptions,
            analysis_cx: self.analysis_cx.map(|cx| {
                AnalysisCx::new(ProofCx::from_solve_cx(
                    cx.proof.solve_cx().with_assumptions(assumptions),
                ))
                .with_overlay(cx.overlay)
                .with_mode(cx.mode)
            }),
            ..self
        }
    }

    pub fn name(self) -> IdentId<'db> {
        self.name
    }

    pub fn solve_cx(self, db: &'db dyn HirAnalysisDb) -> TraitSolveCx<'db> {
        TraitSolveCx::new(db, self.origin_scope).with_assumptions(self.assumptions)
    }
}

impl<'db> TyVisitable<'db> for AssocConstUse<'db> {
    fn visit_with<V>(&self, visitor: &mut V)
    where
        V: TyVisitor<'db> + ?Sized,
    {
        self.assumptions.visit_with(visitor);
        self.inst.visit_with(visitor);
    }
}

impl<'db> TyFoldable<'db> for AssocConstUse<'db> {
    fn super_fold_with<F>(self, db: &'db dyn HirAnalysisDb, folder: &mut F) -> Self
    where
        F: TyFolder<'db>,
    {
        let assumptions = self.assumptions.fold_with(db, folder);
        let analysis_cx = self.analysis_cx.map(|cx| {
            let overlay = cx
                .overlay
                .current_impl()
                .map(|current_impl| {
                    super::context::ImplOverlay::with_current_impl(
                        current_impl.fold_with(db, folder),
                    )
                })
                .unwrap_or_default();
            let mode = match cx.mode {
                super::context::LoweringMode::Normal => super::context::LoweringMode::Normal,
                super::context::LoweringMode::ImplTraitSignature {
                    trait_inst,
                    self_ty,
                    current_impl,
                } => super::context::LoweringMode::ImplTraitSignature {
                    trait_inst: trait_inst.fold_with(db, folder),
                    self_ty: self_ty.fold_with(db, folder),
                    current_impl: current_impl.map(|impl_| impl_.fold_with(db, folder)),
                },
                super::context::LoweringMode::SelectedTraitBody {
                    trait_inst,
                    self_ty,
                    current_impl,
                } => super::context::LoweringMode::SelectedTraitBody {
                    trait_inst: trait_inst.fold_with(db, folder),
                    self_ty: self_ty.fold_with(db, folder),
                    current_impl: current_impl.map(|impl_| impl_.fold_with(db, folder)),
                },
            };
            AnalysisCx::new(ProofCx::from_solve_cx(
                cx.proof.solve_cx().with_assumptions(assumptions),
            ))
            .with_overlay(overlay)
            .with_mode(mode)
        });
        Self {
            origin_scope: self.origin_scope,
            assumptions,
            inst: self.inst.fold_with(db, folder),
            name: self.name,
            analysis_cx,
        }
    }
}
