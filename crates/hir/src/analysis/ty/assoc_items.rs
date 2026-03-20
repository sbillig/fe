use crate::analysis::{
    HirAnalysisDb,
    ty::{
        binder::Binder,
        canonical::Canonical,
        const_expr::ConstExpr,
        const_ty::{ConstTyData, const_ty_from_trait_const},
        context::{AnalysisCx, ImplOverlay, LoweringMode, ProofCx},
        fold::{AssocTySubst, TyFoldable as _},
        normalize::normalize_ty_with_solve_cx,
        trait_def::{
            ImplementorId, ImplementorOrigin, TraitInstId, impls_for_ty_with_constraints_in_cx,
        },
        trait_resolution::Selection,
        ty_def::{InvalidCause, TyData, TyId},
        unify::UnificationTable,
    },
};
use crate::hir_def::{Body, Expr, IdentId, Partial, PathKind};
use common::indexmap::IndexSet;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AssocConstBodyOrigin {
    ImplOverride,
    TraitDefault,
}

#[derive(Debug, Clone)]
pub struct SelectedAssocConstBody<'db> {
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

fn selected_assoc_const_body_analysis_cx<'db>(
    db: &'db dyn HirAnalysisDb,
    proof: ProofCx<'db>,
    trait_inst: TraitInstId<'db>,
    body: &SelectedAssocConstBody<'db>,
) -> AnalysisCx<'db> {
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
    let ty = normalize_ty_with_solve_cx(
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
                    ..
                } => {
                    let Some(expected_ty) = *expected_ty else {
                        return ty.super_fold_with(db, self);
                    };
                    let expr = body.expr(db);
                    let Partial::Present(expr) = expr.data(db, *body) else {
                        return ty.super_fold_with(db, self);
                    };
                    let Expr::Path(path) = expr else {
                        return ty.super_fold_with(db, self);
                    };
                    let Some(path) = path.to_opt() else {
                        return ty.super_fold_with(db, self);
                    };

                    let mut const_ty = *const_ty;
                    if let Some(parent) = path.parent(db)
                        && parent.is_self_ty(db)
                        && let PathKind::Ident {
                            ident,
                            generic_args,
                        } = path.kind(db)
                        && generic_args.is_empty(db)
                        && let Some(name) = ident.to_opt()
                        && let Some(repl) = const_ty_from_trait_const(
                            db,
                            self.cx.proof.solve_cx(),
                            self.trait_inst,
                            name,
                            self.current_implementor,
                        )
                    {
                        const_ty = repl;
                    }

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
            implementor,
            body,
            impl_args: generic_args,
            origin: AssocConstBodyOrigin::ImplOverride,
        });
    }

    Some(SelectedAssocConstBody {
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
    let declared_ty = trait_inst
        .def(db)
        .const_(db, const_name)?
        .ty_binder(db)?
        .instantiate(db, trait_inst.args(db));
    Some(normalize_ty_for_trait_inst(db, cx, declared_ty, trait_inst))
}

pub fn resolve_assoc_const_selection<'db>(
    db: &'db dyn HirAnalysisDb,
    cx: &AnalysisCx<'db>,
    trait_inst: TraitInstId<'db>,
    name: IdentId<'db>,
) -> Option<AssocConstSelection<'db>> {
    let trait_inst = normalize_trait_inst(db, cx, trait_inst);

    let body = overlay_selected_assoc_const_body(db, cx.overlay, trait_inst, name).or_else(|| {
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
        .map(|body| body.implementor.trait_inst(db))
        .unwrap_or(trait_inst);
    let declared_ty = if let Some(body) = body.as_ref() {
        let cx = selected_assoc_const_body_analysis_cx(db, cx.proof, trait_inst, body);
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
    Some(selected_assoc_const_body_analysis_cx(
        db,
        proof,
        selection.trait_inst,
        body,
    ))
}

pub fn resolve_assoc_type_in_mode<'db>(
    db: &'db dyn HirAnalysisDb,
    cx: &AnalysisCx<'db>,
    receiver_ty: TyId<'db>,
    trait_inst: TraitInstId<'db>,
    name: IdentId<'db>,
) -> TyId<'db> {
    let trait_inst = match cx.mode.trait_inst() {
        Some(mode_trait_inst) if receiver_ty == cx.mode.self_ty().unwrap_or(receiver_ty) => {
            mode_trait_inst
        }
        _ => trait_inst,
    };

    trait_inst
        .assoc_ty(db, name)
        .map(|ty| normalize_ty_for_trait_inst(db, cx, ty, trait_inst))
        .unwrap_or_else(|| TyId::invalid(db, crate::analysis::ty::ty_def::InvalidCause::Other))
}
