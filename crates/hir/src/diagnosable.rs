//! Semantic diagnostics helpers.
//!
//! This module is the home for traversal API helpers that produce
//! `TyDiagCollection` / diagnostics. Over time, diagnostic-focused
//! logic from `core::semantic` is being migrated here to keep the main
//! traversal surface free of diagnostic concerns.

use rustc_hash::FxHashMap;
use smallvec1::SmallVec;

use crate::analysis::HirAnalysisDb;
use crate::analysis::name_resolution;
use crate::analysis::ty;
use crate::analysis::ty::diagnostics::{TraitConstraintDiag, TyDiagCollection, TyLowerDiag};
use crate::analysis::ty::ty_def::{InvalidCause, TyId};
use crate::hir_def::scope_graph::ScopeId;
use crate::hir_def::{
    Contract, Enum, EnumVariant, FieldParent, Func, GenericParam, GenericParamOwner,
    GenericParamView, IdentId, Impl, ImplTrait, ItemKind, Partial, PathId, Struct, Trait,
    TypeAlias, TypeBound, VariantKind, WhereClauseOwner,
};
use crate::span::DynLazySpan;

use crate::analysis::ty::adt_def::AdtRef;
use crate::analysis::ty::binder::Binder;
use crate::analysis::ty::trait_def::ImplementorId;
use crate::semantic::{
    FieldView, FuncParamView, ImplAssocTypeView, SuperTraitRefView, VariantView,
    WherePredicateBoundView, WherePredicateView, constraints_for, lower_hir_kind_local,
};

/// Unified "pull" diagnostics surface for HIR items and views.
pub trait Diagnosable<'db> {
    type Diagnostic;
    fn diags(self, db: &'db dyn HirAnalysisDb) -> Vec<Self::Diagnostic>;
}

/// Shared helper for duplicate name diagnostics.
pub(crate) fn check_duplicate_names<'db, F>(
    names: impl Iterator<Item = Option<IdentId<'db>>>,
    create_diag: F,
) -> SmallVec<[TyDiagCollection<'db>; 2]>
where
    F: Fn(SmallVec<[u16; 4]>) -> TyDiagCollection<'db>,
{
    let mut defs = FxHashMap::<IdentId<'db>, SmallVec<[u16; 4]>>::default();
    for (i, name) in names.enumerate() {
        if let Some(name) = name {
            defs.entry(name).or_default().push(i as u16);
        }
    }
    defs.into_iter()
        .filter_map(|(_name, idxs)| (idxs.len() > 1).then_some(create_diag(idxs)))
        .collect()
}

fn const_ty_mismatch_diag<'db>(
    span: DynLazySpan<'db>,
    expected: TyId<'db>,
    given: TyId<'db>,
) -> TyDiagCollection<'db> {
    TyLowerDiag::ConstTyMismatch {
        span,
        expected,
        given,
    }
    .into()
}

impl<'db> SuperTraitRefView<'db> {
    /// Diagnostics for this super-trait reference in its owner's context.
    /// Uses the trait's `Self` as subject and checks WF; kind mismatch is emitted
    /// elsewhere via `Trait::diags_super_traits`.
    pub fn diags(self, db: &'db dyn HirAnalysisDb) -> Option<TyDiagCollection<'db>> {
        use name_resolution::{ExpectedPathKind, diagnostics::PathResDiag};
        use ty::trait_lower::{self, TraitRefLowerError};
        use ty::trait_resolution::{WellFormedness, check_trait_inst_wf};

        let span = self.span();
        let subject = self.subject_self(db);
        let scope = self.owner.scope();
        let assumptions = self.assumptions(db);
        let tr = self.trait_ref(db);

        let inst = match trait_lower::lower_trait_ref(db, subject, tr, scope, assumptions, None) {
            Ok(i) => i,
            Err(TraitRefLowerError::PathResError(err)) => {
                let path = tr.path(db).unwrap();
                let diag = err.into_diag(db, path, span.path(), ExpectedPathKind::Trait)?;
                return Some(diag.into());
            }
            Err(TraitRefLowerError::InvalidDomain(res)) => {
                let path = tr.path(db).unwrap();
                let ident = path.ident(db).unwrap();
                return Some(
                    PathResDiag::ExpectedTrait(span.path().into(), ident, res.kind_name()).into(),
                );
            }
            Err(TraitRefLowerError::Ignored) => return None,
        };

        // Do not emit when subject contains assoc types of params
        if inst.self_ty(db).contains_assoc_ty_of_param(db) {
            return None;
        }

        match check_trait_inst_wf(
            db,
            ty::trait_resolution::TraitSolveCx::new(db, scope).with_assumptions(assumptions),
            inst,
        ) {
            WellFormedness::WellFormed => None,
            WellFormedness::IllFormed { goal, subgoal } => Some(
                TraitConstraintDiag::TraitBoundNotSat {
                    span: span.into(),
                    primary_goal: goal,
                    unsat_subgoal: subgoal,
                }
                .into(),
            ),
        }
    }
}

impl<'db> WherePredicateView<'db> {
    /// Aggregate diagnostics for this where-predicate:
    /// - Subject-level errors (const/concrete or path-domain remapped)
    /// - Per-bound trait diagnostics
    /// - Per-bound kind consistency
    pub fn diags(self, db: &'db dyn HirAnalysisDb) -> Vec<TyDiagCollection<'db>> {
        let Some(subject) = self.subject_ty(db) else {
            return Vec::new();
        };

        if let Some(diag) = self.diag_subject_ty(db, subject) {
            return vec![diag];
        }

        self.bound_diags(db, subject)
    }

    /// Diagnostic for this predicate's subject type, if any:
    /// - Path-resolution domain errors are remapped to precise diagnostics.
    /// - Const subjects are rejected.
    /// - Fully concrete, non-generic subjects are rejected.
    fn diag_subject_ty(
        self,
        db: &'db dyn HirAnalysisDb,
        subject: TyId<'db>,
    ) -> Option<TyDiagCollection<'db>> {
        use crate::analysis::name_resolution::diagnostics::PathResDiag;
        use crate::analysis::name_resolution::{ExpectedPathKind, resolve_path};

        // Path-resolution failures are carried via the subject's InvalidCause.
        let owner_item = ItemKind::from(self.clause.owner);
        let assumptions = constraints_for(db, owner_item);
        if let Some(InvalidCause::PathResolutionFailed { path }) = subject.invalid_cause(db) {
            // Re-run name resolution on the failed path and surface a precise diagnostic
            // at the type path span within the where-predicate.
            let ty_span = self.span().ty().into_path_type().path();
            match resolve_path(db, path, owner_item.scope(), assumptions, false) {
                Ok(res) => {
                    // Resolved to a non-type domain
                    if let Some(ident) = path.ident(db).to_opt() {
                        let diag =
                            PathResDiag::ExpectedType(ty_span.into(), ident, res.kind_name());
                        return Some(diag.into());
                    }
                }
                Err(inner) => {
                    if let Some(diag) = inner.into_diag(db, path, ty_span, ExpectedPathKind::Type) {
                        return Some(diag.into());
                    }
                }
            }
        }
        let span = self.span().ty().into();

        if subject.is_const_ty(db) {
            return Some(TraitConstraintDiag::ConstTyBound(span, subject).into());
        }

        if !subject.has_invalid(db) && !subject.has_param(db) {
            return Some(TraitConstraintDiag::ConcreteTypeBound(span, subject).into());
        }

        None
    }
}

impl<'db> WherePredicateBoundView<'db> {
    /// Diagnostics for this trait bound, given an explicit subject type.
    /// Mirrors legacy visitor behavior for path errors, kind mismatch, and satisfiability.
    pub(crate) fn diags_for_subject(
        self,
        db: &'db dyn HirAnalysisDb,
        subject: ty::ty_def::TyId<'db>,
    ) -> Vec<TyDiagCollection<'db>> {
        use name_resolution::{ExpectedPathKind, diagnostics::PathResDiag};
        use ty::trait_lower::{self, TraitRefLowerError};
        use ty::trait_resolution::{WellFormedness, check_trait_inst_wf};

        let mut out = Vec::new();
        let owner_item = ItemKind::from(self.pred.clause.owner);
        let scope = owner_item.scope();
        let assumptions = constraints_for(db, owner_item);
        let is_trait_self_subject =
            matches!(owner_item, ItemKind::Trait(_)) && self.pred.is_self_subject(db);
        let tr = self.trait_ref(db);
        let span = self.trait_ref_span();

        match trait_lower::lower_trait_ref(db, subject, tr, scope, assumptions, None) {
            Ok(inst) => {
                let expected = inst.def(db).self_param(db).kind(db);
                if !expected.does_match(subject.kind(db)) {
                    out.push(
                        TraitConstraintDiag::TraitArgKindMismatch {
                            span: span.clone(),
                            expected: expected.clone(),
                            actual: subject,
                        }
                        .into(),
                    );
                }

                if inst.self_ty(db).contains_assoc_ty_of_param(db) {
                    return out;
                }

                // For trait-level `Self: Bound` constraints, treat as preconditions;
                // do not emit unsatisfied bound diagnostics here.
                if !is_trait_self_subject {
                    match check_trait_inst_wf(
                        db,
                        ty::trait_resolution::TraitSolveCx::new(db, scope)
                            .with_assumptions(assumptions),
                        inst,
                    ) {
                        WellFormedness::WellFormed => {}
                        WellFormedness::IllFormed { goal, .. } => {
                            out.push(
                                TraitConstraintDiag::TraitBoundNotSat {
                                    span: span.into(),
                                    primary_goal: goal,
                                    unsat_subgoal: None,
                                }
                                .into(),
                            );
                        }
                    }
                }
            }
            Err(TraitRefLowerError::PathResError(err)) => {
                if let Some(path) = tr.path(db).to_opt()
                    && let Some(diag) =
                        err.into_diag(db, path, span.path(), ExpectedPathKind::Trait)
                {
                    out.push(diag.into());
                }
            }
            Err(TraitRefLowerError::InvalidDomain(res)) => {
                if let Some(path) = tr.path(db).to_opt()
                    && let Some(ident) = path.ident(db).to_opt()
                {
                    out.push(
                        PathResDiag::ExpectedTrait(span.path().into(), ident, res.kind_name())
                            .into(),
                    );
                }
            }
            Err(TraitRefLowerError::Ignored) => {}
        }

        out
    }

    /// Diagnostics for this trait bound, deriving the subject from the predicate's LHS.
    /// Returns a single-element vec with the subject error if subject lowering fails.
    pub fn diags(self, db: &'db dyn HirAnalysisDb) -> Vec<TyDiagCollection<'db>> {
        let subject = match self.pred.subject_ty(db) {
            Some(s) => s,
            None => return Vec::new(),
        };
        self.diags_for_subject(db, subject)
    }
}

impl<'db> Func<'db> {
    pub fn diags_const_fn(self, db: &'db dyn HirAnalysisDb) -> Vec<TyDiagCollection<'db>> {
        // Const-safety diagnostics are handled by the const-check pass on the body.
        let _ = db;
        Vec::new()
    }

    /// Diagnostics related to parameters (duplicate names/labels).
    pub fn diags_parameters(self, db: &'db dyn HirAnalysisDb) -> Vec<TyDiagCollection<'db>> {
        check_duplicate_names(self.params(db).map(|v| v.name(db)), |idxs| {
            TyLowerDiag::DuplicateArgName(self, idxs).into()
        })
        .into_iter()
        .collect()
    }

    /// Diagnostics related to the explicit return type (kind/const checks).
    pub fn diags_return(self, db: &'db dyn HirAnalysisDb) -> Vec<TyDiagCollection<'db>> {
        let mut diags = Vec::new();
        if self.has_explicit_return_ty(db) {
            // First, surface name-resolution/path-domain errors on the return type itself
            let errs = self.ret_ty_errors(db);
            if !errs.is_empty() {
                return errs;
            }

            // Then run kind/const checks on the lowered semantic type
            let ret = self.return_ty(db);
            let span = self.span().ret_ty().into();
            if !ret.has_star_kind(db) {
                diags.push(TyLowerDiag::ExpectedStarKind(span).into());
            } else if ret.is_const_ty(db) {
                diags.push(TyLowerDiag::NormalTypeExpected { span, given: ret }.into());
            } else if ty::ty_contains_const_hole(db, ret) {
                diags.push(TyLowerDiag::ConstHoleInValuePosition { span }.into());
            }
        }
        diags
    }

    /// Diagnostics for function parameter types:
    /// - For all params: star kind required and reject const types
    /// - For self param: enforce exact `Self` type shape
    ///   Note: WF/invalid errors are still surfaced via the general type walker.
    pub fn diags_param_types(self, db: &'db dyn HirAnalysisDb) -> Vec<TyDiagCollection<'db>> {
        self.params(db).flat_map(|v| v.diags(db)).collect()
    }
}

impl<'db> Diagnosable<'db> for FuncParamView<'db> {
    type Diagnostic = TyDiagCollection<'db>;

    fn diags(self, db: &'db dyn HirAnalysisDb) -> Vec<Self::Diagnostic> {
        self.ty_diags(db)
    }
}

impl<'db> Diagnosable<'db> for TypeAlias<'db> {
    type Diagnostic = TyDiagCollection<'db>;

    fn diags(self, db: &'db dyn HirAnalysisDb) -> Vec<Self::Diagnostic> {
        let mut out = self.ty_errors(db);
        out.extend(self.ty_wf_errors(db));
        out.extend(GenericParamOwner::TypeAlias(self).diags(db));
        out
    }
}

impl<'db> Trait<'db> {
    /// Diagnostics for associated type defaults (bounds satisfaction), in the trait's context.
    pub fn diags_assoc_defaults(self, db: &'db dyn HirAnalysisDb) -> Vec<TyDiagCollection<'db>> {
        let mut diags = Vec::new();
        let assumptions = constraints_for(db, self.into());
        for assoc in self.assoc_types(db) {
            let Some(default_ty) = assoc.default_ty(db) else {
                continue;
            };
            for trait_inst in assoc.bounds_on_subject(db, default_ty) {
                let canonical_inst = ty::canonical::Canonical::new(db, trait_inst);
                match ty::trait_resolution::is_goal_satisfiable(
                    db,
                    ty::trait_resolution::TraitSolveCx::new(db, self.scope())
                        .with_assumptions(assumptions),
                    canonical_inst,
                ) {
                    ty::trait_resolution::GoalSatisfiability::Satisfied(_) => {}
                    ty::trait_resolution::GoalSatisfiability::UnSat(_) => {
                        diags.push(
                            TraitConstraintDiag::TraitBoundNotSat {
                                span: self.span().into(),
                                primary_goal: trait_inst,
                                unsat_subgoal: None,
                            }
                            .into(),
                        );
                    }
                    _ => {}
                }
            }
        }
        diags
    }

    /// Diagnostics for generic parameter issues (duplicates, defined in parent).
    pub fn diags_generic_params(self, db: &'db dyn HirAnalysisDb) -> Vec<TyDiagCollection<'db>> {
        let owner = GenericParamOwner::Trait(self);
        let mut out: Vec<TyDiagCollection> = owner.diags_check_duplicate_names(db).collect();
        out.extend(owner.diags_params_defined_in_parent(db));
        out
    }

    /// Diagnostics for super-traits (semantic, kind-mismatch only).
    pub fn diags_super_traits(self, db: &'db dyn HirAnalysisDb) -> Vec<TyDiagCollection<'db>> {
        use ty::trait_resolution::{WellFormedness, check_trait_inst_wf};
        use ty::ty_def::TyFlags;

        let mut diags = Vec::new();
        for view in self.super_trait_refs(db) {
            if let Some((expected, actual)) = view.kind_mismatch_for_self(db) {
                diags.push(
                    TraitConstraintDiag::TraitArgKindMismatch {
                        span: view.span(),
                        expected,
                        actual,
                    }
                    .into(),
                );
            }

            // Additionally, ensure that the super-trait reference is well-formed
            if let Ok(inst) = view.trait_inst(db) {
                let flags = ty::visitor::collect_flags(db, inst);
                if !flags.contains(TyFlags::HAS_INVALID)
                    && flags.intersects(TyFlags::HAS_PARAM | TyFlags::HAS_VAR)
                {
                    continue;
                }
                match check_trait_inst_wf(
                    db,
                    ty::trait_resolution::TraitSolveCx::new(db, self.scope())
                        .with_assumptions(view.assumptions(db)),
                    inst,
                ) {
                    WellFormedness::WellFormed => {}
                    WellFormedness::IllFormed { goal, .. } => {
                        diags.push(
                            TraitConstraintDiag::TraitBoundNotSat {
                                span: view.span().into(),
                                primary_goal: goal,
                                unsat_subgoal: None,
                            }
                            .into(),
                        );
                    }
                }
            }
        }
        diags
    }
}

impl<'db> Impl<'db> {
    /// Impl-specific preconditions and implementor-type diagnostics.
    /// Generic parameter diagnostics are handled by `Diagnosable::diags`.
    pub fn diags_preconditions(self, db: &'db dyn HirAnalysisDb) -> Vec<TyDiagCollection<'db>> {
        use ty::diagnostics::ImplDiag;
        use ty::trait_resolution::WellFormedness;

        let mut out = self.ty_errors(db);

        let ty = self.ty(db);
        let ingot = self.top_mod(db).ingot(db);
        if !ty.is_inherent_impl_allowed(db, ingot) {
            let base = ty.base_ty(db);
            out.push(
                ImplDiag::InherentImplIsNotAllowed {
                    primary: self.span().target_ty().into(),
                    ty: base.pretty_print(db).to_string(),
                    is_nominal: !base.is_param(db),
                }
                .into(),
            );
            return out;
        }

        if let Some(diag) =
            ty::ty_error::emit_invalid_ty_error(db, ty, self.span().target_ty().into())
        {
            out.push(diag);
        }

        if ty.has_invalid(db) {
            return out;
        }

        match self.target_ty_wf(db) {
            WellFormedness::WellFormed => {}
            WellFormedness::IllFormed { goal, subgoal } => {
                out.push(
                    TraitConstraintDiag::TraitBoundNotSat {
                        span: self.span().target_ty().into(),
                        primary_goal: goal,
                        unsat_subgoal: subgoal,
                    }
                    .into(),
                );
            }
        }

        out
    }
}

impl<'db> ImplTrait<'db> {
    fn admission_summary(
        self,
        db: &'db dyn HirAnalysisDb,
    ) -> &'db ty::admission::AdmissionSummary<'db> {
        ty::trait_lower::admission_summary(db, self.top_mod(db).ingot(db))
    }

    fn header_diags(self, db: &'db dyn HirAnalysisDb) -> Vec<TyDiagCollection<'db>> {
        self.admission_summary(db)
            .header_issues
            .get(&self)
            .into_iter()
            .flat_map(|issues| issues.iter())
            .flat_map(|issue| issue.to_diags(db, self))
            .collect()
    }

    fn interface_issue_diags<F>(
        self,
        db: &'db dyn HirAnalysisDb,
        include: F,
    ) -> Vec<TyDiagCollection<'db>>
    where
        F: Fn(&ty::admission::ImplInterfaceIssue<'db>) -> bool,
    {
        self.admission_summary(db)
            .interface_issues
            .get(&self)
            .into_iter()
            .flat_map(|issues| issues.iter())
            .filter(|issue| include(issue))
            .flat_map(|issue| issue.to_diags())
            .collect()
    }

    pub(crate) fn interface_diags(self, db: &'db dyn HirAnalysisDb) -> Vec<TyDiagCollection<'db>> {
        self.interface_issue_diags(db, |_| true)
    }

    /// Lower the implementor view and report validity diagnostics (WF, conflicts, kind mismatch).
    /// Returns the implementor view if successful, or None if critical errors occurred.
    pub(crate) fn diags_implementor_validity(
        self,
        db: &'db dyn HirAnalysisDb,
    ) -> (
        Option<Binder<ImplementorId<'db>>>,
        Vec<TyDiagCollection<'db>>,
    ) {
        self.implementor_with_errors(db)
    }

    /// Diagnostics for missing associated types (required by the trait).
    pub fn diags_missing_assoc_types(
        self,
        db: &'db dyn HirAnalysisDb,
    ) -> Vec<TyDiagCollection<'db>> {
        self.interface_issue_diags(db, |issue| {
            matches!(
                issue,
                ty::admission::ImplInterfaceIssue::MissingAssocType { .. }
            )
        })
    }

    /// Diagnostics for missing associated consts (required by the trait).
    pub fn diags_missing_assoc_consts(
        self,
        db: &'db dyn HirAnalysisDb,
    ) -> Vec<TyDiagCollection<'db>> {
        self.interface_issue_diags(db, |issue| {
            matches!(
                issue,
                ty::admission::ImplInterfaceIssue::MissingAssocConst { .. }
            )
        })
    }

    /// Diagnostics for associated const values and validity.
    pub fn diags_assoc_consts(self, db: &'db dyn HirAnalysisDb) -> Vec<TyDiagCollection<'db>> {
        self.interface_issue_diags(db, |issue| {
            matches!(
                issue,
                ty::admission::ImplInterfaceIssue::ExtraAssocConst { .. }
                    | ty::admission::ImplInterfaceIssue::MissingAssocConstValue { .. }
                    | ty::admission::ImplInterfaceIssue::AssocConstInvalidDiag(..)
                    | ty::admission::ImplInterfaceIssue::AssocConstInvalid { .. }
            )
        })
    }

    /// Diagnostics for associated type bounds on implemented assoc types.
    pub fn diags_assoc_types_bounds(
        self,
        db: &'db dyn HirAnalysisDb,
    ) -> Vec<TyDiagCollection<'db>> {
        self.interface_issue_diags(db, |issue| {
            matches!(
                issue,
                ty::admission::ImplInterfaceIssue::AssocTypeBoundViolation { .. }
            )
        })
    }

    /// Diagnostics for implemented associated types' WF and invalid types.
    pub fn diags_assoc_types_wf(self, db: &'db dyn HirAnalysisDb) -> Vec<TyDiagCollection<'db>> {
        self.assoc_types(db)
            .flat_map(|view| view.diags(db))
            .collect()
    }
}

impl<'db> Diagnosable<'db> for ImplAssocTypeView<'db> {
    type Diagnostic = TyDiagCollection<'db>;

    fn diags(self, db: &'db dyn HirAnalysisDb) -> Vec<Self::Diagnostic> {
        self.ty_diags(db)
    }
}

impl<'db> Diagnosable<'db> for Struct<'db> {
    type Diagnostic = TyDiagCollection<'db>;

    fn diags(self, db: &'db dyn HirAnalysisDb) -> Vec<Self::Diagnostic> {
        let mut out = Vec::new();

        out.extend(check_duplicate_names(
            FieldParent::Struct(self).fields(db).map(|v| v.name(db)),
            |idxs| TyLowerDiag::DuplicateFieldName(FieldParent::Struct(self), idxs).into(),
        ));

        for v in FieldParent::Struct(self).fields(db) {
            out.extend(v.diags(db));
        }

        for pred in WhereClauseOwner::Struct(self).clause(db).predicates(db) {
            out.extend(pred.diags(db));
        }

        out.extend(GenericParamOwner::Struct(self).diags(db));
        out
    }
}

impl<'db> VariantView<'db> {
    /// Diagnostics for tuple-variant element types: star-kind and non-const checks.
    /// Returns an empty list if this is not a tuple variant.
    pub fn diags_tuple_elems_wf(self, db: &'db dyn HirAnalysisDb) -> Vec<TyDiagCollection<'db>> {
        use crate::hir_def::types::TypeKind as HirTyKind;
        use name_resolution::{PathRes, resolve_path};
        use ty::trait_resolution::{TraitSolveCx, WellFormedness, check_ty_wf};
        use ty::ty_lower::lower_hir_ty;

        let mut out = Vec::new();
        let VariantKind::Tuple(tuple_id) = self.kind(db) else {
            return out;
        };

        let enum_ = self.owner;
        let var = EnumVariant::new(enum_, self.idx);
        let scope = var.scope();
        let assumptions = constraints_for(db, enum_.into());

        for (elem_idx, p) in tuple_id.data(db).iter().enumerate() {
            let Some(hir_ty) = p.to_opt() else {
                continue;
            };

            let span = self.span().tuple_type().elem_ty(elem_idx);

            // For non-const subjects, surface name-resolution/path-domain errors first.
            let is_const_path = match hir_ty.data(db) {
                HirTyKind::Path(path) => {
                    if let Some(path) = path.to_opt() {
                        matches!(
                            resolve_path(db, path, scope, assumptions, true),
                            Ok(PathRes::Const(..))
                        )
                    } else {
                        false
                    }
                }
                _ => false,
            };

            if !is_const_path {
                let mut errs = ty::ty_error::collect_ty_lower_errors(
                    db,
                    scope,
                    hir_ty,
                    span.clone(),
                    assumptions,
                );
                if !errs.is_empty() {
                    out.append(&mut errs);
                    continue;
                }
            }

            let ty = lower_hir_ty(db, hir_ty, scope, assumptions);
            if ty.has_invalid(db) {
                continue;
            }
            if !ty.has_star_kind(db) {
                out.push(TyLowerDiag::ExpectedStarKind(span.clone().into()).into());
                continue;
            }
            if ty.is_const_ty(db) {
                out.push(
                    TyLowerDiag::NormalTypeExpected {
                        span: span.clone().into(),
                        given: ty,
                    }
                    .into(),
                );
                continue;
            }

            // Trait-bound well-formedness for element type.
            match check_ty_wf(
                db,
                TraitSolveCx::new(db, scope).with_assumptions(assumptions),
                ty,
            ) {
                WellFormedness::WellFormed => {}
                WellFormedness::IllFormed { goal, subgoal } => {
                    out.push(
                        TraitConstraintDiag::TraitBoundNotSat {
                            span: span.clone().into(),
                            primary_goal: goal,
                            unsat_subgoal: subgoal,
                        }
                        .into(),
                    );
                }
            }
        }

        out
    }
}

impl<'db> Diagnosable<'db> for FieldView<'db> {
    type Diagnostic = TyDiagCollection<'db>;

    fn diags(self, db: &'db dyn HirAnalysisDb) -> Vec<Self::Diagnostic> {
        self.ty_diags(db)
    }
}

impl<'db> Diagnosable<'db> for Enum<'db> {
    type Diagnostic = TyDiagCollection<'db>;

    fn diags(self, db: &'db dyn HirAnalysisDb) -> Vec<Self::Diagnostic> {
        let mut out = Vec::new();

        out.extend(check_duplicate_names(
            self.variants(db).map(|v| v.name(db)),
            |idxs| TyLowerDiag::DuplicateVariantName(self, idxs).into(),
        ));

        for v in self.variants(db) {
            if matches!(v.kind(db), VariantKind::Record(_)) {
                out.extend(check_duplicate_names(
                    v.fields(db).map(|f| f.name(db)),
                    |idxs| {
                        TyLowerDiag::DuplicateFieldName(
                            FieldParent::Variant(EnumVariant::new(self, v.idx)),
                            idxs,
                        )
                        .into()
                    },
                ));
                for f in v.fields(db) {
                    out.extend(f.diags(db));
                }
            } else if matches!(v.kind(db), VariantKind::Tuple(_)) {
                out.extend(v.diags_tuple_elems_wf(db));
            }
        }

        for pred in WhereClauseOwner::Enum(self).clause(db).predicates(db) {
            out.extend(pred.diags(db));
        }

        out.extend(GenericParamOwner::Enum(self).diags(db));
        out
    }
}

impl<'db> Diagnosable<'db> for Contract<'db> {
    type Diagnostic = TyDiagCollection<'db>;

    fn diags(self, db: &'db dyn HirAnalysisDb) -> Vec<Self::Diagnostic> {
        let mut out = Vec::new();
        out.extend(check_duplicate_names(
            FieldParent::Contract(self).fields(db).map(|v| v.name(db)),
            |idxs| TyLowerDiag::DuplicateFieldName(FieldParent::Contract(self), idxs).into(),
        ));
        for v in FieldParent::Contract(self).fields(db) {
            out.extend(v.diags(db));
        }
        out
    }
}

impl<'db> Diagnosable<'db> for AdtRef<'db> {
    type Diagnostic = TyDiagCollection<'db>;

    fn diags(self, db: &'db dyn HirAnalysisDb) -> Vec<Self::Diagnostic> {
        match self {
            AdtRef::Struct(s) => s.diags(db),
            AdtRef::Enum(e) => e.diags(db),
        }
    }
}

impl<'db> GenericParamOwner<'db> {
    pub fn diags_params_defined_in_parent(
        self,
        db: &'db dyn HirAnalysisDb,
    ) -> impl Iterator<Item = TyDiagCollection<'db>> + 'db {
        self.params(db).filter_map(|param| {
            param
                .diag_param_defined_in_parent(db)
                .map(TyDiagCollection::from)
        })
    }

    pub fn diags_check_duplicate_names(
        self,
        db: &'db dyn HirAnalysisDb,
    ) -> impl Iterator<Item = TyDiagCollection<'db>> + 'db {
        let params_iter = self.params(db).map(|v| v.name().to_opt());
        check_duplicate_names(params_iter, |idxs| {
            TyDiagCollection::from(TyLowerDiag::DuplicateGenericParamName(self, idxs))
        })
        .into_iter()
    }

    pub fn diags_non_trailing_defaults(
        self,
        db: &'db dyn HirAnalysisDb,
    ) -> Vec<TyDiagCollection<'db>> {
        let mut out = Vec::new();
        let mut default_idxs = Vec::new();
        for view in self.params(db) {
            let is_defaulted_type =
                matches!(view.param, GenericParam::Type(tp) if tp.default_ty.is_some());
            if is_defaulted_type {
                default_idxs.push(view.idx);
            } else if !default_idxs.is_empty() {
                for &idx in &default_idxs {
                    let span = self.param_view(db, idx).span();
                    out.push(TyLowerDiag::NonTrailingDefaultGenericParam(span).into());
                }
                break;
            }
        }
        out
    }

    pub fn diags_const_param_types(self, db: &'db dyn HirAnalysisDb) -> Vec<TyDiagCollection<'db>> {
        use ty::ty_def::{InvalidCause, TyData};

        let mut out = Vec::new();
        let param_set = ty::ty_lower::collect_generic_params(db, self);
        for view in self.params(db) {
            let GenericParam::Const(c) = view.param else {
                continue;
            };
            if c.ty.to_opt().is_none() {
                continue;
            }
            if let Some(ty) = param_set.param_by_original_idx(db, view.idx) {
                let cause_opt = match ty.data(db) {
                    TyData::Invalid(cause) => Some(cause.clone()),
                    TyData::ConstTy(ct) => match ct.ty(db).data(db) {
                        TyData::Invalid(cause) => Some(cause.clone()),
                        _ => None,
                    },
                    _ => None,
                };
                if let Some(cause) = cause_opt {
                    let span = view.span().into_const_param().ty();
                    match cause {
                        InvalidCause::InvalidConstParamTy => {
                            out.push(TyLowerDiag::InvalidConstParamTy(span.into()).into());
                        }
                        InvalidCause::RecursiveConstParamTy => {
                            out.push(TyLowerDiag::RecursiveConstParamTy(span.into()).into());
                        }
                        InvalidCause::ConstTyExpected { expected } => {
                            out.push(
                                TyLowerDiag::ConstTyExpected {
                                    span: span.into(),
                                    expected,
                                }
                                .into(),
                            );
                        }
                        InvalidCause::ConstTyMismatch { expected, given } => {
                            out.push(const_ty_mismatch_diag(span.into(), expected, given));
                        }
                        _ => {}
                    }
                }
            }
        }
        out
    }

    pub fn diags_default_forward_refs(
        self,
        db: &'db dyn HirAnalysisDb,
    ) -> Vec<TyDiagCollection<'db>> {
        use ty::{
            ty_def::{TyId, TyParam},
            ty_lower::lower_hir_ty,
            visitor::{TyVisitable, TyVisitor},
        };

        let mut out = Vec::new();
        let owner_item = ItemKind::from(self);
        let assumptions = constraints_for(db, owner_item);
        let scope = self.scope();

        for view in self.params(db) {
            let default_ty = match view.param {
                GenericParam::Type(tp) => tp.default_ty,
                GenericParam::Const(_) => None,
            };
            let Some(default_ty) = default_ty else {
                continue;
            };

            let lowered = lower_hir_ty(db, default_ty, scope, assumptions);

            struct Collector<'db> {
                db: &'db dyn HirAnalysisDb,
                scope: ScopeId<'db>,
                out: Vec<usize>,
            }
            impl<'db> TyVisitor<'db> for Collector<'db> {
                fn db(&self) -> &'db dyn HirAnalysisDb {
                    self.db
                }
                fn visit_param(&mut self, tp: &TyParam<'db>) {
                    if !tp.is_trait_self() && tp.owner == self.scope {
                        self.out.push(tp.original_idx(self.db));
                    }
                }
                fn visit_const_param(&mut self, tp: &TyParam<'db>, _ty: TyId<'db>) {
                    if tp.owner == self.scope {
                        self.out.push(tp.original_idx(self.db));
                    }
                }
            }

            let mut collector = Collector {
                db,
                scope,
                out: Vec::new(),
            };
            lowered.visit_with(&mut collector);

            for j in collector.out.into_iter().filter(|j| *j >= view.idx) {
                if let Some(name) = self.param_view(db, j).param.name().to_opt() {
                    let span = view.span();
                    out.push(TyLowerDiag::GenericDefaultForwardRef { span, name }.into());
                }
            }
        }

        out
    }

    pub fn diags_kind_bounds(self, db: &'db dyn HirAnalysisDb) -> Vec<TyDiagCollection<'db>> {
        let mut out = Vec::new();
        let param_set = ty::ty_lower::collect_generic_params(db, self);

        for view in self.params(db) {
            let GenericParam::Type(tp) = view.param else {
                continue;
            };
            let Some(ty) = param_set.param_by_original_idx(db, view.idx) else {
                continue;
            };
            let actual = ty.kind(db);

            for (i, bound) in tp.bounds.iter().enumerate() {
                if let TypeBound::Kind(Partial::Present(kb)) = bound {
                    let expected = lower_hir_kind_local(kb);
                    if !actual.does_match(&expected) {
                        let span = view.span().into_type_param().bounds().bound(i).kind_bound();
                        out.push(
                            TyLowerDiag::InconsistentKindBound {
                                span: span.into(),
                                ty,
                                bound: expected,
                            }
                            .into(),
                        );
                    }
                }
            }
        }

        out
    }

    pub fn diags_trait_bounds(self, db: &'db dyn HirAnalysisDb) -> Vec<TyDiagCollection<'db>> {
        use name_resolution::{ExpectedPathKind, diagnostics::PathResDiag};
        use ty::trait_lower::{self, TraitRefLowerError};
        use ty::trait_resolution::{WellFormedness, check_trait_inst_wf};

        let mut out = Vec::new();
        let param_set = ty::ty_lower::collect_generic_params(db, self);
        let scope = self.scope();
        let assumptions = constraints_for(db, self.into());

        for view in self.params(db) {
            let GenericParam::Type(tp) = view.param else {
                continue;
            };
            let Some(subject) = param_set.param_by_original_idx(db, view.idx) else {
                continue;
            };

            for (i, bound) in tp.bounds.iter().enumerate() {
                let TypeBound::Trait(tr) = bound else {
                    continue;
                };
                let span = view
                    .span()
                    .into_type_param()
                    .bounds()
                    .bound(i)
                    .trait_bound();
                match trait_lower::lower_trait_ref(db, subject, *tr, scope, assumptions, None) {
                    Ok(inst) => {
                        let expected = inst.def(db).self_param(db).kind(db);
                        if !expected.does_match(subject.kind(db)) {
                            out.push(
                                TraitConstraintDiag::TraitArgKindMismatch {
                                    span: span.clone(),
                                    expected: expected.clone(),
                                    actual: subject,
                                }
                                .into(),
                            );
                        }

                        if inst.self_ty(db).contains_assoc_ty_of_param(db) {
                            continue;
                        }

                        match check_trait_inst_wf(
                            db,
                            ty::trait_resolution::TraitSolveCx::new(db, scope)
                                .with_assumptions(assumptions),
                            inst,
                        ) {
                            WellFormedness::WellFormed => {}
                            WellFormedness::IllFormed { goal, .. } => out.push(
                                TraitConstraintDiag::TraitBoundNotSat {
                                    span: span.into(),
                                    primary_goal: goal,
                                    unsat_subgoal: None,
                                }
                                .into(),
                            ),
                        }
                    }
                    Err(TraitRefLowerError::PathResError(err)) => {
                        if let Some(path) = tr.path(db).to_opt()
                            && let Some(diag) =
                                err.into_diag(db, path, span.path(), ExpectedPathKind::Trait)
                        {
                            out.push(diag.into());
                        }
                    }
                    Err(TraitRefLowerError::InvalidDomain(res)) => {
                        if let Some(path) = tr.path(db).to_opt()
                            && let Some(ident) = path.ident(db).to_opt()
                        {
                            out.push(
                                PathResDiag::ExpectedTrait(
                                    span.path().into(),
                                    ident,
                                    res.kind_name(),
                                )
                                .into(),
                            );
                        }
                    }
                    Err(TraitRefLowerError::Ignored) => {}
                }
            }
        }

        out
    }
}

impl<'db> GenericParamView<'db> {
    pub fn diag_param_defined_in_parent(
        self,
        db: &'db dyn HirAnalysisDb,
    ) -> Option<TyLowerDiag<'db>> {
        use crate::analysis::name_resolution::{PathRes, resolve_path};
        use crate::analysis::ty::trait_resolution::PredicateListId;

        let name = self.param.name().to_opt()?;
        let parent_scope = self.owner.scope().parent_item(db)?.scope();
        let path = PathId::from_ident(db, name);
        let span = self.span();

        match resolve_path(
            db,
            path,
            parent_scope,
            PredicateListId::empty_list(db),
            false,
        ) {
            Ok(r @ PathRes::Ty(ty)) if ty.is_param(db) => {
                Some(TyLowerDiag::GenericParamAlreadyDefinedInParent {
                    span,
                    conflict_with: r.name_span(db).unwrap(),
                    name,
                })
            }
            _ => None,
        }
    }
}

impl<'db> Diagnosable<'db> for GenericParamOwner<'db> {
    type Diagnostic = TyDiagCollection<'db>;

    fn diags(self, db: &'db dyn HirAnalysisDb) -> Vec<Self::Diagnostic> {
        let mut out = Vec::new();
        out.extend(self.diags_check_duplicate_names(db));
        out.extend(self.diags_const_param_types(db));
        out.extend(self.diags_params_defined_in_parent(db));
        out.extend(self.diags_kind_bounds(db));
        out.extend(self.diags_trait_bounds(db));
        out.extend(self.diags_non_trailing_defaults(db));
        out.extend(self.diags_default_forward_refs(db));
        out
    }
}

impl<'db> Diagnosable<'db> for Func<'db> {
    type Diagnostic = TyDiagCollection<'db>;

    fn diags(self, db: &'db dyn HirAnalysisDb) -> Vec<Self::Diagnostic> {
        use ty::canonical::Canonical;
        use ty::method_table::probe_method;

        let mut out = Vec::new();
        out.extend(self.diags_const_fn(db));
        out.extend(self.diags_parameters(db));
        out.extend(self.diags_param_types(db));
        out.extend(self.diags_return(db));

        for pred in WhereClauseOwner::Func(self).clause(db).predicates(db) {
            out.extend(pred.diags(db));
        }

        // Method conflict check only for inherent impls
        if let Some(crate::hir_def::scope_graph::ScopeId::Item(ItemKind::Impl(impl_))) =
            self.scope().parent(db)
            && let Some(func_def) = self.as_callable(db)
        {
            let self_ty = impl_.ty(db);
            if !self_ty.has_invalid(db) {
                let ingot = self.top_mod(db).ingot(db);
                for &cand in probe_method(
                    db,
                    ingot,
                    Canonical::new(db, self_ty),
                    func_def.name(db).expect("impl methods have names"),
                ) {
                    if cand != func_def {
                        out.push(
                            ty::diagnostics::ImplDiag::ConflictMethodImpl {
                                primary: func_def,
                                conflict_with: cand,
                            }
                            .into(),
                        );
                        break;
                    }
                }
            }
        }

        out.extend(GenericParamOwner::Func(self).diags(db));
        out
    }
}

impl<'db> Diagnosable<'db> for Trait<'db> {
    type Diagnostic = TyDiagCollection<'db>;

    fn diags(self, db: &'db dyn HirAnalysisDb) -> Vec<Self::Diagnostic> {
        let mut out = Vec::new();
        out.extend(self.diags_assoc_defaults(db));
        out.extend(self.diags_super_traits(db));

        for pred in WhereClauseOwner::Trait(self).clause(db).predicates(db) {
            out.extend(pred.diags(db));
        }

        out.extend(GenericParamOwner::Trait(self).diags(db));
        out
    }
}

impl<'db> Diagnosable<'db> for Impl<'db> {
    type Diagnostic = TyDiagCollection<'db>;

    fn diags(self, db: &'db dyn HirAnalysisDb) -> Vec<Self::Diagnostic> {
        let mut out = self.diags_preconditions(db);
        out.extend(GenericParamOwner::Impl(self).diags(db));
        out
    }
}

impl<'db> Diagnosable<'db> for ImplTrait<'db> {
    type Diagnostic = TyDiagCollection<'db>;

    fn diags(self, db: &'db dyn HirAnalysisDb) -> Vec<Self::Diagnostic> {
        // Early path/domain/WF checks; bail out on errors to avoid noisy follow-ups
        let (implementor_opt, validity_diags) = self.diags_implementor_validity(db);
        let Some(_implementor) = implementor_opt else {
            return validity_diags;
        };

        let mut out = validity_diags;
        let header_diags = self.header_diags(db);
        if !header_diags.is_empty() {
            out.extend(header_diags);
            return out;
        }

        out.extend(self.interface_diags(db));
        out.extend(GenericParamOwner::ImplTrait(self).diags(db));
        out
    }
}
