use crate::analysis::{
    HirAnalysisDb,
    name_resolution::PathRes,
    ty::{
        assoc_const::AssocConstUse,
        binder::Binder,
        canonical::{Canonical, Canonicalized},
        const_expr::ConstExpr,
        const_ty::{ConstTyData, const_body_simple_path, const_ty_from_trait_const},
        context::{AnalysisCx, ImplOverlay, LoweringMode, ProofCx},
        fold::{AssocTySubst, TyFoldable as _},
        normalize::normalize_ty_without_consts_with_solve_cx,
        trait_def::{
            ImplementorId, ImplementorOrigin, TraitInstId, impls_for_ty_with_constraints_in_cx,
        },
        trait_resolution::{Selection, normalize_trait_inst_preserving_validity_with_solve_cx},
        ty_def::{InvalidCause, TyData, TyId, inference_keys},
        ty_lower::contextual_path_resolution_in_cx,
        unify::UnificationTable,
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
pub(crate) struct SelectedAssocConstBody<'db> {
    pub(crate) selected_trait_inst: TraitInstId<'db>,
    pub(crate) implementor: ImplementorId<'db>,
    pub(crate) body: Body<'db>,
    pub(crate) impl_args: Vec<TyId<'db>>,
    pub(crate) origin: AssocConstBodyOrigin,
}

#[derive(Debug, Clone)]
pub(crate) struct AssocConstSelection<'db> {
    pub(crate) trait_inst: TraitInstId<'db>,
    pub(crate) declared_ty: TyId<'db>,
    pub(crate) body: Option<SelectedAssocConstBody<'db>>,
}

fn selected_assoc_const_body_analysis_cx<'db>(
    db: &'db dyn HirAnalysisDb,
    proof: ProofCx<'db>,
    body: &SelectedAssocConstBody<'db>,
) -> AnalysisCx<'db> {
    let trait_inst = body.selected_trait_inst;
    let implementor = body.implementor;
    let assumptions = proof
        .assumptions()
        .merge(db, implementor.constraints(db))
        .extend_all_bounds(db);
    let proof = proof.with_assumptions(assumptions);
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
    fn is_identity_trait_inst<'db>(db: &'db dyn HirAnalysisDb, inst: TraitInstId<'db>) -> bool {
        let trait_def = inst.def(db);
        inst.assoc_type_bindings(db).is_empty() && inst.args(db) == trait_def.params(db)
    }

    fn rebase_trait_const_inst<'db>(
        db: &'db dyn HirAnalysisDb,
        inst: TraitInstId<'db>,
        trait_inst: TraitInstId<'db>,
    ) -> TraitInstId<'db> {
        if inst.def(db) == trait_inst.def(db) && is_identity_trait_inst(db, inst) {
            trait_inst
        } else {
            inst
        }
    }

    struct ConstFolder<'db> {
        cx: AnalysisCx<'db>,
        trait_inst: TraitInstId<'db>,
    }

    impl<'db> super::fold::TyFolder<'db> for ConstFolder<'db> {
        fn fold_ty_app(
            &mut self,
            db: &'db dyn HirAnalysisDb,
            abs: TyId<'db>,
            arg: TyId<'db>,
        ) -> TyId<'db> {
            TyId::app_metadata_only(db, abs, arg)
        }

        fn fold_ty(&mut self, db: &'db dyn HirAnalysisDb, ty: TyId<'db>) -> TyId<'db> {
            let TyData::ConstTy(const_ty) = ty.data(db) else {
                return ty.super_fold_with(db, self);
            };

            match const_ty.data(db) {
                ConstTyData::Abstract(expr, expected_ty) => {
                    let ConstExpr::TraitConst(assoc) = expr.data(db) else {
                        return ty.super_fold_with(db, self);
                    };
                    let inst = rebase_trait_const_inst(db, assoc.inst(), self.trait_inst);

                    let Some(const_ty) =
                        const_ty_from_trait_const(db, self.cx.proof.solve_cx(), inst, assoc.name())
                    else {
                        if inst == assoc.inst() {
                            return ty.super_fold_with(db, self);
                        }
                        let assoc =
                            assoc.with_env(assoc.origin_scope(), self.cx.proof.assumptions());
                        let expr = super::const_expr::ConstExprId::new(
                            db,
                            ConstExpr::TraitConst(assoc.with_inst(inst)),
                        );
                        return TyId::new(
                            db,
                            TyData::ConstTy(super::const_ty::ConstTyId::new(
                                db,
                                ConstTyData::Abstract(expr, *expected_ty),
                            )),
                        );
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
                    ..
                } => {
                    let Some(path) = const_body_simple_path(db, *body) else {
                        return TyId::new(db, TyData::ConstTy(*const_ty)).super_fold_with(db, self);
                    };
                    if let Some(parent) = path.parent(db)
                        && parent.is_self_ty(db)
                        && let Some(name) = path.ident(db).to_opt()
                        && path.generic_args(db).is_empty(db)
                    {
                        if let Some(repl) = const_ty_from_trait_const(
                            db,
                            self.cx.proof.solve_cx(),
                            self.trait_inst,
                            name,
                        ) {
                            return TyId::new(
                                db,
                                TyData::ConstTy(repl.evaluate_with_solve_cx(
                                    db,
                                    expected_ty.or(Some(repl.ty(db))),
                                    self.cx.proof.solve_cx(),
                                )),
                            );
                        }
                        if let Some(expected_ty) = *expected_ty {
                            let assoc = AssocConstUse::new(
                                body.scope(),
                                self.cx.proof.assumptions(),
                                self.trait_inst,
                                name,
                            )
                            .with_analysis_cx(self.cx);
                            return TyId::new(
                                db,
                                TyData::ConstTy(super::const_ty::ConstTyId::new(
                                    db,
                                    ConstTyData::Abstract(
                                        super::const_expr::ConstExprId::new(
                                            db,
                                            ConstExpr::TraitConst(assoc),
                                        ),
                                        expected_ty,
                                    ),
                                )),
                            );
                        }
                    }

                    if let Some(PathRes::TraitConst(_, inst, name)) =
                        contextual_path_resolution_in_cx(db, body.scope(), path, true, &self.cx)
                            .map(|res| match res {
                                PathRes::TraitConst(recv_ty, inst, name) => PathRes::TraitConst(
                                    recv_ty,
                                    rebase_trait_const_inst(db, inst, self.trait_inst),
                                    name,
                                ),
                                other => other,
                            })
                        && let Some(repl) =
                            const_ty_from_trait_const(db, self.cx.proof.solve_cx(), inst, name)
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
                        return TyId::new(db, TyData::ConstTy(*const_ty)).super_fold_with(db, self);
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
    };
    ty.fold_with(db, &mut folder)
}

fn select_implementor_for_trait_inst<'db>(
    db: &'db dyn HirAnalysisDb,
    cx: &AnalysisCx<'db>,
    trait_inst: TraitInstId<'db>,
) -> Selection<ImplementorId<'db>> {
    let (primary, secondary) = cx.proof.search_ingots_for_trait_inst(db, trait_inst);
    let mut search_ingots = vec![Some(primary)];
    if let Some(secondary) = secondary {
        search_ingots.push(Some(secondary));
    }
    let canonical_target = Canonicalized::new(db, trait_inst);
    let canonical_self_ty = Canonical::new(db, canonical_target.value.value.self_ty(db));
    let mut matches = IndexSet::default();
    let mut table = UnificationTable::new(db);
    let target_inst = canonical_target.value.extract_identity(&mut table);
    let target_keys = inference_keys(db, &target_inst);

    for ingot in search_ingots {
        for implementor in
            impls_for_ty_with_constraints_in_cx(db, ingot, canonical_self_ty, cx.proof.solve_cx())
        {
            let snapshot = table.snapshot();
            let implementor: ImplementorId<'db> = table.instantiate_with_fresh_vars(implementor);
            if table
                .unify(implementor.trait_inst(db), target_inst)
                .is_err()
            {
                table.rollback_to(snapshot);
                continue;
            }
            let implementor = implementor.fold_with(db, &mut table);
            if !inference_keys(db, &implementor).is_subset(&target_keys) {
                table.rollback_to(snapshot);
                continue;
            }
            matches.insert(canonical_target.decanonicalize(db, implementor));
            table.rollback_to(snapshot);
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
    normalize_trait_inst_preserving_validity_with_solve_cx(db, trait_inst, cx.proof.solve_cx())
}

fn selected_assoc_const_body_for_implementor<'db>(
    db: &'db dyn HirAnalysisDb,
    implementor: ImplementorId<'db>,
    inst: TraitInstId<'db>,
    const_name: IdentId<'db>,
) -> Option<SelectedAssocConstBody<'db>> {
    if !inference_keys(db, &implementor).is_empty() || !inference_keys(db, &inst).is_empty() {
        return None;
    }

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

pub(crate) fn assoc_const_declared_ty<'db>(
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

pub(crate) fn resolve_assoc_const_selection<'db>(
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
