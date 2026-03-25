use crate::analysis::{
    HirAnalysisDb,
    name_resolution::PathRes,
    ty::{
        binder::Binder,
        canonical::Canonical,
        const_expr::ConstExpr,
        const_ty::{ConstTyData, ConstTyId, const_body_simple_path, const_ty_from_trait_const},
        context::{AnalysisCx, ImplOverlay, LoweringMode, ProofCx},
        fold::{AssocTySubst, TyFoldable as _},
        normalize::normalize_ty_without_consts_with_solve_cx,
        trait_def::{
            ImplementorId, ImplementorOrigin, TraitInstId, impls_for_ty_with_constraints_in_cx,
        },
        trait_resolution::Selection,
        ty_def::{InvalidCause, TyData, TyFlags, TyId},
        ty_lower::contextual_path_resolution_in_cx,
        unify::UnificationTable,
        visitor::collect_flags,
    },
};
use crate::hir_def::{Body, IdentId};
use common::indexmap::IndexSet;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AssocConstBodyOrigin {
    ImplOverride,
    TraitDefault,
}

#[derive(Debug, Clone)]
pub struct SelectedAssocConstBody<'db> {
    pub selected_trait_inst: TraitInstId<'db>,
    pub implementor: ImplementorId<'db>,
    pub body: Body<'db>,
    pub impl_args: Vec<TyId<'db>>,
    pub origin: AssocConstBodyOrigin,
}

#[derive(Debug, Clone)]
pub struct AssocConstSelection<'db> {
    pub trait_inst: TraitInstId<'db>,
    pub name: IdentId<'db>,
    pub declared_ty: TyId<'db>,
    pub body: Option<SelectedAssocConstBody<'db>>,
}

#[derive(Debug, Clone)]
pub enum TraitConstUseResolution<'db> {
    Concrete(AssocConstSelection<'db>),
    Abstract {
        trait_inst: TraitInstId<'db>,
        name: IdentId<'db>,
        declared_ty: TyId<'db>,
    },
    MissingConcreteImpl {
        trait_inst: TraitInstId<'db>,
        name: IdentId<'db>,
        declared_ty: TyId<'db>,
    },
}

fn selected_assoc_const_body_analysis_cx<'db>(
    db: &'db dyn HirAnalysisDb,
    proof: ProofCx<'db>,
    body: &SelectedAssocConstBody<'db>,
) -> AnalysisCx<'db> {
    let trait_inst = body.selected_trait_inst;
    let implementor = body.implementor;
    let mode = match body.origin {
        AssocConstBodyOrigin::ImplOverride => LoweringMode::ImplTraitSignature {
            trait_inst,
            self_ty: trait_inst.self_ty(db),
            current_impl: Some(implementor),
        },
        AssocConstBodyOrigin::TraitDefault => LoweringMode::SelectedTraitBody {
            trait_inst,
            self_ty: trait_inst.self_ty(db),
            current_impl: Some(implementor),
        },
    };

    AnalysisCx::new(proof)
        .with_overlay(ImplOverlay::with_current_impl(implementor))
        .with_mode(mode)
}

pub(crate) fn normalize_ty_for_trait_inst<'db>(
    db: &'db dyn HirAnalysisDb,
    cx: &AnalysisCx<'db>,
    ty: TyId<'db>,
    trait_inst: TraitInstId<'db>,
) -> TyId<'db> {
    let mut substituter = AssocTySubst::new(trait_inst);
    let ty = ty.fold_with(db, &mut substituter);
    let scope = cx.proof.normalization_scope_for_trait_inst(db, trait_inst);
    let ty = normalize_ty_without_consts_with_solve_cx(
        db,
        ty,
        scope,
        cx.proof.assumptions(),
        Some(cx.proof.solve_cx()),
    );
    normalize_const_tys_for_trait_inst(db, cx, ty, trait_inst)
}

pub(crate) fn normalize_const_tys_for_trait_inst<'db>(
    db: &'db dyn HirAnalysisDb,
    cx: &AnalysisCx<'db>,
    ty: TyId<'db>,
    trait_inst: TraitInstId<'db>,
) -> TyId<'db> {
    struct ConstFolder<'db> {
        cx: AnalysisCx<'db>,
        trait_inst: TraitInstId<'db>,
        current_implementor: Option<ImplementorId<'db>>,
    }

    impl<'db> super::fold::TyFolder<'db> for ConstFolder<'db> {
        fn fold_ty(&mut self, db: &'db dyn HirAnalysisDb, ty: TyId<'db>) -> TyId<'db> {
            let TyData::ConstTy(const_ty) = ty.data(db) else {
                return ty.super_fold_with(db, self);
            };

            match const_ty.data(db) {
                ConstTyData::Abstract(expr, expected_ty) => {
                    let ConstExpr::TraitConst { inst, name } = expr.data(db) else {
                        return ty.super_fold_with(db, self);
                    };

                    let Some(const_ty) = const_ty_from_trait_const(
                        db,
                        self.cx.proof.solve_cx(),
                        *inst,
                        *name,
                        self.current_implementor,
                    ) else {
                        return ty.super_fold_with(db, self);
                    };

                    let evaluated = const_ty.evaluate_with_solve_cx(
                        db,
                        Some(*expected_ty),
                        self.cx.proof.solve_cx(),
                    );
                    if matches!(
                        evaluated.ty(db).invalid_cause(db),
                        Some(InvalidCause::ConstEvalUnsupported { .. })
                    ) {
                        return ty.super_fold_with(db, self);
                    }

                    TyId::new(db, TyData::ConstTy(evaluated))
                }
                ConstTyData::UnEvaluated {
                    body,
                    ty: expected_ty,
                    const_def,
                    generic_args,
                    trait_inst: body_trait_inst,
                    mode_kind,
                    current_implementor,
                    ..
                } => {
                    let const_ty =
                        if !matches!(mode_kind, super::const_ty::ConstBodyModeKind::Normal)
                            && (*body_trait_inst != Some(self.trait_inst)
                                || *current_implementor != self.current_implementor)
                        {
                            ConstTyId::new(
                                db,
                                ConstTyData::UnEvaluated {
                                    body: *body,
                                    ty: *expected_ty,
                                    const_def: *const_def,
                                    generic_args: generic_args.clone(),
                                    trait_inst: Some(self.trait_inst),
                                    mode_kind: *mode_kind,
                                    current_implementor: self.current_implementor,
                                },
                            )
                        } else {
                            *const_ty
                        };
                    let Some(path) = const_body_simple_path(db, *body) else {
                        return TyId::new(db, TyData::ConstTy(const_ty)).super_fold_with(db, self);
                    };

                    if let Some(PathRes::TraitConst(_, inst, name)) =
                        contextual_path_resolution_in_cx(db, body.scope(), path, true, &self.cx)
                        && let Some(repl) = const_ty_from_trait_const(
                            db,
                            self.cx.proof.solve_cx(),
                            inst,
                            name,
                            self.current_implementor,
                        )
                    {
                        return TyId::new(
                            db,
                            TyData::ConstTy(repl.evaluate_with_solve_cx(
                                db,
                                expected_ty.or(Some(repl.ty(db))),
                                self.cx.proof.solve_cx(),
                            )),
                        );
                    }

                    let Some(expected_ty) = *expected_ty else {
                        return TyId::new(db, TyData::ConstTy(const_ty)).super_fold_with(db, self);
                    };
                    TyId::new(
                        db,
                        TyData::ConstTy(const_ty.evaluate_with_solve_cx(
                            db,
                            Some(expected_ty),
                            self.cx.proof.solve_cx(),
                        )),
                    )
                }
                _ => ty.super_fold_with(db, self),
            }
        }
    }

    let mut folder = ConstFolder {
        cx: *cx,
        trait_inst,
        current_implementor: cx.overlay.current_impl().or(cx.mode.current_impl()),
    };
    ty.fold_with(db, &mut folder)
}

fn select_implementor_for_trait_inst<'db>(
    db: &'db dyn HirAnalysisDb,
    cx: &AnalysisCx<'db>,
    trait_inst: TraitInstId<'db>,
) -> Selection<ImplementorId<'db>> {
    let self_ty = Canonical::new(db, trait_inst.self_ty(db));
    let (primary, secondary) = cx.proof.search_ingots_for_trait_inst(db, trait_inst);
    let search_ingots = if primary.is_none() && secondary.is_none() {
        vec![None]
    } else {
        vec![primary, secondary]
    };
    let mut matches = IndexSet::default();

    for ingot in search_ingots {
        for implementor in
            impls_for_ty_with_constraints_in_cx(db, ingot, self_ty, cx.proof.solve_cx())
        {
            let mut table = UnificationTable::new(db);
            let implementor = table.instantiate_with_fresh_vars(implementor);
            if table.unify(implementor.trait_inst(db), trait_inst).is_err() {
                continue;
            }
            matches.insert(implementor.fold_with(db, &mut table));
        }
    }

    match matches.len() {
        0 => Selection::NotFound,
        1 => Selection::Unique(*matches.iter().next().unwrap()),
        _ => Selection::Ambiguous(matches.into_iter().collect()),
    }
}

pub(crate) fn normalize_trait_inst<'db>(
    db: &'db dyn HirAnalysisDb,
    cx: &AnalysisCx<'db>,
    trait_inst: TraitInstId<'db>,
) -> TraitInstId<'db> {
    let scope = cx.proof.normalization_scope_for_trait_inst(db, trait_inst);
    trait_inst.normalize_with_solve_cx(db, cx.proof.solve_cx(), scope, cx.proof.assumptions())
}

fn selected_assoc_const_body_for_implementor<'db>(
    db: &'db dyn HirAnalysisDb,
    implementor: ImplementorId<'db>,
    inst: TraitInstId<'db>,
    const_name: IdentId<'db>,
) -> Option<SelectedAssocConstBody<'db>> {
    let mut table = UnificationTable::new(db);
    let instantiated = table.instantiate_with_fresh_vars(Binder::bind(implementor));
    table.unify(instantiated.trait_inst(db), inst).ok()?;
    let selected_trait_inst = instantiated.trait_inst(db).fold_with(db, &mut table);
    let generic_args = instantiated
        .params(db)
        .iter()
        .map(|&ty| ty.fold_with(db, &mut table))
        .collect::<Vec<_>>();

    if let ImplementorOrigin::Hir(impl_trait) = implementor.origin(db)
        && let Some(assoc_const) = impl_trait.const_(db, const_name)
        && let Some(body) = assoc_const.value_body(db)
    {
        return Some(SelectedAssocConstBody {
            selected_trait_inst,
            implementor,
            body,
            impl_args: generic_args,
            origin: AssocConstBodyOrigin::ImplOverride,
        });
    }

    Some(SelectedAssocConstBody {
        selected_trait_inst,
        implementor,
        body: implementor
            .trait_inst(db)
            .def(db)
            .const_(db, const_name)?
            .default_body(db)?,
        impl_args: generic_args,
        origin: AssocConstBodyOrigin::TraitDefault,
    })
}

fn overlay_selected_assoc_const_body<'db>(
    db: &'db dyn HirAnalysisDb,
    overlay: ImplOverlay<'db>,
    inst: TraitInstId<'db>,
    const_name: IdentId<'db>,
) -> Option<SelectedAssocConstBody<'db>> {
    let current_impl = overlay.current_impl()?;
    (current_impl.trait_def(db) == inst.def(db))
        .then_some(())
        .and_then(|_| selected_assoc_const_body_for_implementor(db, current_impl, inst, const_name))
}

pub fn assoc_const_declared_ty<'db>(
    db: &'db dyn HirAnalysisDb,
    cx: &AnalysisCx<'db>,
    trait_inst: TraitInstId<'db>,
    const_name: IdentId<'db>,
) -> Option<TyId<'db>> {
    let trait_inst = normalize_trait_inst(db, cx, trait_inst);
    let binder = trait_inst
        .def(db)
        .const_(db, const_name)?
        .ty_binder_in_cx(db, cx)?;
    let declared_ty = binder.instantiate(db, trait_inst.args(db));
    Some(normalize_ty_for_trait_inst(db, cx, declared_ty, trait_inst))
}

pub fn resolve_assoc_const_selection<'db>(
    db: &'db dyn HirAnalysisDb,
    cx: &AnalysisCx<'db>,
    trait_inst: TraitInstId<'db>,
    name: IdentId<'db>,
) -> Option<AssocConstSelection<'db>> {
    let trait_inst = normalize_trait_inst(db, cx, trait_inst);

    let current_impl_overlay = cx
        .mode
        .current_impl()
        .map(ImplOverlay::with_current_impl)
        .unwrap_or_default();
    let body = overlay_selected_assoc_const_body(db, cx.overlay, trait_inst, name)
        .or_else(|| overlay_selected_assoc_const_body(db, current_impl_overlay, trait_inst, name))
        .or_else(|| {
            let selection = match cx.proof.select_impl(db, trait_inst) {
                Selection::NotFound => select_implementor_for_trait_inst(db, cx, trait_inst),
                other => other,
            };
            match selection {
                Selection::Unique(implementor) => {
                    selected_assoc_const_body_for_implementor(db, implementor, trait_inst, name)
                }
                Selection::Ambiguous(_) | Selection::NotFound => None,
            }
        });
    let trait_inst = body
        .as_ref()
        .map(|body| body.selected_trait_inst)
        .unwrap_or(trait_inst);
    let declared_ty = if let Some(body) = body.as_ref() {
        let cx = selected_assoc_const_body_analysis_cx(db, cx.proof, body);
        assoc_const_declared_ty(db, &cx, trait_inst, name)?
    } else {
        assoc_const_declared_ty(db, cx, trait_inst, name)?
    };

    Some(AssocConstSelection {
        trait_inst,
        name,
        declared_ty,
        body,
    })
}

pub(crate) fn analysis_cx_for_selected_assoc_const_body<'db>(
    db: &'db dyn HirAnalysisDb,
    proof: ProofCx<'db>,
    selection: &AssocConstSelection<'db>,
) -> Option<AnalysisCx<'db>> {
    let body = selection.body.as_ref()?;
    Some(selected_assoc_const_body_analysis_cx(db, proof, body))
}

pub fn resolve_trait_const_use<'db>(
    db: &'db dyn HirAnalysisDb,
    cx: &AnalysisCx<'db>,
    trait_inst: TraitInstId<'db>,
    name: IdentId<'db>,
) -> Option<TraitConstUseResolution<'db>> {
    let selection = resolve_assoc_const_selection(db, cx, trait_inst, name)?;
    if selection.body.is_some() {
        return Some(TraitConstUseResolution::Concrete(selection));
    }

    let flags = collect_flags(db, selection.trait_inst);
    let has_matching_assumption =
        cx.proof
            .assumptions()
            .list(db)
            .iter()
            .copied()
            .any(|assumption| {
                let mut table = UnificationTable::new(db);
                table.unify(assumption, selection.trait_inst).is_ok()
            });
    let can_stay_abstract = !flags.contains(TyFlags::HAS_INVALID)
        && (flags.intersects(TyFlags::HAS_PARAM | TyFlags::HAS_VAR)
            || selection.trait_inst.self_ty(db).is_trait_self(db)
            || has_matching_assumption);
    let resolution = if can_stay_abstract {
        TraitConstUseResolution::Abstract {
            trait_inst: selection.trait_inst,
            name: selection.name,
            declared_ty: selection.declared_ty,
        }
    } else {
        TraitConstUseResolution::MissingConcreteImpl {
            trait_inst: selection.trait_inst,
            name: selection.name,
            declared_ty: selection.declared_ty,
        }
    };
    Some(resolution)
}
