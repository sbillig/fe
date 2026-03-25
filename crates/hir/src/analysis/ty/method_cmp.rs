use thin_vec::ThinVec;

use super::{
    assoc_items::normalize_const_tys_for_trait_inst,
    canonical::Canonical,
    context::AnalysisCx,
    diagnostics::{ImplDiag, TyDiagCollection},
    fold::TyFoldable,
    normalize::normalize_ty_without_consts_with_solve_cx,
    trait_def::TraitInstId,
    trait_resolution::{
        GoalSatisfiability, TraitSolveCx, constraint::collect_func_def_constraints,
        is_goal_satisfiable,
    },
    ty_def::{TyData, TyId},
};
use crate::analysis::HirAnalysisDb;
use crate::hir_def::{CallableDef, ItemKind, Trait};

struct MethodCmpSubst<'db> {
    trait_inst: TraitInstId<'db>,
}

impl<'db> MethodCmpSubst<'db> {
    fn new(trait_inst: TraitInstId<'db>) -> Self {
        Self { trait_inst }
    }
}

impl<'db> super::fold::TyFolder<'db> for MethodCmpSubst<'db> {
    fn fold_ty(&mut self, db: &'db dyn HirAnalysisDb, ty: TyId<'db>) -> TyId<'db> {
        match ty.data(db) {
            TyData::TyParam(param) => {
                if param.is_trait_self() {
                    let owner_trait = param.owner.resolve_to::<Trait>(db).or_else(|| {
                        match param.owner.parent_item(db)? {
                            ItemKind::Trait(trait_) => Some(trait_),
                            _ => None,
                        }
                    });
                    if owner_trait.is_some_and(|trait_def| trait_def == self.trait_inst.def(db)) {
                        let self_ty = self.trait_inst.self_ty(db);
                        if self_ty == ty { ty } else { self_ty }
                    } else {
                        ty.super_fold_with(db, self)
                    }
                } else {
                    ty.super_fold_with(db, self)
                }
            }
            TyData::AssocTy(assoc_ty) => {
                let folded_trait = assoc_ty.trait_.fold_with(db, self);
                let matches_current_trait = folded_trait.def(db) == self.trait_inst.def(db)
                    && folded_trait.self_ty(db) == self.trait_inst.self_ty(db)
                    && folded_trait.args(db).len() <= self.trait_inst.args(db).len()
                    && folded_trait
                        .args(db)
                        .iter()
                        .skip(1)
                        .zip(self.trait_inst.args(db).iter().skip(1))
                        .all(|(lhs, rhs)| lhs == rhs)
                    && folded_trait
                        .assoc_type_bindings(db)
                        .iter()
                        .all(|(name, ty)| {
                            self.trait_inst
                                .assoc_type_bindings(db)
                                .get(name)
                                .is_some_and(|bound| bound == ty)
                        });
                if matches_current_trait
                    && let Some(&bound_ty) =
                        self.trait_inst.assoc_type_bindings(db).get(&assoc_ty.name)
                {
                    bound_ty
                } else if folded_trait == assoc_ty.trait_ {
                    ty
                } else {
                    TyId::assoc_ty(db, folded_trait, assoc_ty.name)
                }
            }
            _ => ty.super_fold_with(db, self),
        }
    }
}

/// Compares the implementation method with the trait method to ensure they
/// match.
///
/// This function performs the following checks:
///
/// 1. Number of generic parameters.
/// 2. Kinds of generic parameters.
/// 3. Arity (number of arguments).
/// 4. Argument labels.
/// 5. Argument types and return type.
/// 6. Method constraints.
///
/// If any of these checks fail, the function will record the appropriate
/// diagnostics.
///
/// # Arguments
///
/// * `db` - Reference to the database implementing the `HirAnalysisDb` trait.
/// * `impl_m` - The implementation method to compare.
/// * `trait_m` - The trait method to compare against.
/// * `trait_inst` - The instance of the trait being checked.
/// * `implementor` - The implementor that contains associated type bindings.
/// * `sink` - A mutable reference to a vector where diagnostic messages will be
///   collected.
pub(super) fn compare_impl_method<'db>(
    db: &'db dyn HirAnalysisDb,
    impl_m: CallableDef<'db>,
    trait_m: CallableDef<'db>,
    trait_inst: TraitInstId<'db>,
    cx: &AnalysisCx<'db>,
    sink: &mut Vec<TyDiagCollection<'db>>,
) {
    if !compare_generic_param_num(db, impl_m, trait_m, sink) {
        return;
    }

    if !compare_generic_param_kind(db, impl_m, trait_m, sink) {
        return;
    }

    if !compare_arity(db, impl_m, trait_m, sink) {
        return;
    }

    // Compare the argument labels, argument types, and return type of the impl
    // method with the trait method.
    let mut err = !compare_arg_label(db, impl_m, trait_m, sink);

    let map_to_impl: Vec<_> = trait_inst
        .args(db)
        .iter()
        .chain(impl_m.explicit_params(db).iter())
        .copied()
        .collect();
    err |= !compare_ty(db, impl_m, trait_m, &map_to_impl, trait_inst, cx, sink);
    if err {
        return;
    }

    compare_constraints(db, impl_m, trait_m, &map_to_impl, cx.proof.solve_cx(), sink);
}

fn extend_solve_cx<'db>(
    db: &'db dyn HirAnalysisDb,
    solve_cx: TraitSolveCx<'db>,
    assumptions: super::trait_resolution::PredicateListId<'db>,
) -> TraitSolveCx<'db> {
    solve_cx.with_assumptions(
        solve_cx
            .assumptions()
            .merge(db, assumptions)
            .extend_all_bounds(db),
    )
}

fn normalize_method_cmp_ty<'db>(
    db: &'db dyn HirAnalysisDb,
    cx: &AnalysisCx<'db>,
    ty: TyId<'db>,
    trait_inst: TraitInstId<'db>,
) -> TyId<'db> {
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

/// Checks if the number of generic parameters of the implemented method is the
/// same as the number of generic parameters of the trait method.
/// Returns `false` if the comparison fails.
fn compare_generic_param_num<'db>(
    db: &'db dyn HirAnalysisDb,
    impl_m: CallableDef<'db>,
    trait_m: CallableDef<'db>,
    sink: &mut Vec<TyDiagCollection<'db>>,
) -> bool {
    let impl_params = impl_m.explicit_params(db);
    let trait_params = trait_m.explicit_params(db);

    if impl_params.len() == trait_params.len() {
        true
    } else {
        sink.push(ImplDiag::MethodTypeParamNumMismatch { trait_m, impl_m }.into());
        false
    }
}

/// Checks if the generic parameter kinds are the same.
/// Returns `false` if the comparison fails.
fn compare_generic_param_kind<'db>(
    db: &'db dyn HirAnalysisDb,
    impl_m: CallableDef<'db>,
    trait_m: CallableDef<'db>,
    sink: &mut Vec<TyDiagCollection<'db>>,
) -> bool {
    let mut err = false;
    for (idx, (&trait_m_param, &impl_m_param)) in trait_m
        .explicit_params(db)
        .iter()
        .zip(impl_m.explicit_params(db))
        .enumerate()
    {
        let trait_m_kind = trait_m_param.kind(db);
        let impl_m_kind = impl_m_param.kind(db);

        if !trait_m_kind.does_match(impl_m_kind) {
            sink.push(
                ImplDiag::MethodTypeParamKindMismatch {
                    trait_m,
                    impl_m,
                    param_idx: idx,
                }
                .into(),
            );
            err = true;
        }
    }

    !err
}

/// Checks if the arity of the implemented method is the same as the arity of
/// the trait method.
/// Returns `false` if the comparison fails.
fn compare_arity<'db>(
    db: &'db dyn HirAnalysisDb,
    impl_m: CallableDef<'db>,
    trait_m: CallableDef<'db>,
    sink: &mut Vec<TyDiagCollection<'db>>,
) -> bool {
    let impl_m_arity = impl_m.arg_tys(db).len();
    let trait_m_arity = trait_m.arg_tys(db).len();

    // Checks if the arity are the same.
    if impl_m_arity == trait_m_arity {
        true
    } else {
        sink.push(ImplDiag::MethodArgNumMismatch { impl_m, trait_m }.into());
        false
    }
}

/// Checks if the argument labels of the implemented method are the same as the
/// argument labels of the trait method.
/// Returns `false` if the comparison fails.
fn compare_arg_label<'db>(
    db: &'db dyn HirAnalysisDb,
    impl_m: CallableDef<'db>,
    trait_m: CallableDef<'db>,
    sink: &mut Vec<TyDiagCollection<'db>>,
) -> bool {
    let mut err = false;
    let len = impl_m.arg_tys(db).len().min(trait_m.arg_tys(db).len());
    for idx in 0..len {
        let Some(expected_label) = trait_m.param_label_or_name(db, idx) else {
            continue;
        };
        let Some(method_label) = impl_m.param_label_or_name(db, idx) else {
            continue;
        };
        if expected_label != method_label {
            sink.push(
                ImplDiag::MethodArgLabelMismatch {
                    trait_m,
                    impl_m,
                    param_idx: idx,
                }
                .into(),
            );
            err = true;
        }
    }

    !err
}

/// Checks if the argument types and return type of the implemented method are
/// the same as the argument types and return type of the trait method.
/// Returns `false` if the comparison fails.
fn compare_ty<'db>(
    db: &'db dyn HirAnalysisDb,
    impl_m: CallableDef<'db>,
    trait_m: CallableDef<'db>,
    map_to_impl: &[TyId<'db>],
    trait_inst: TraitInstId<'db>,
    cx: &AnalysisCx<'db>,
    sink: &mut Vec<TyDiagCollection<'db>>,
) -> bool {
    let mut err = false;
    let impl_m_arg_tys = impl_m.arg_tys(db);
    let trait_m_arg_tys = trait_m.arg_tys(db);

    let mut substituter = MethodCmpSubst::new(trait_inst);
    let assumptions = collect_func_def_constraints(db, impl_m, true).instantiate_identity();
    let solve_cx = extend_solve_cx(db, cx.proof.solve_cx(), assumptions);
    let cx = AnalysisCx::new(super::context::ProofCx::from_solve_cx(solve_cx))
        .with_overlay(cx.overlay)
        .with_mode(cx.mode);

    for (idx, (trait_m_ty, impl_m_ty)) in trait_m_arg_tys
        .iter()
        .zip(impl_m_arg_tys.iter())
        .enumerate()
    {
        // 1) Instantiate trait method's type params into the impl's generics
        let trait_m_ty = trait_m_ty.instantiate(db, map_to_impl);
        if trait_m_ty.has_invalid(db) {
            continue;
        }
        let impl_m_ty = impl_m_ty.instantiate_identity();

        // 2) Substitute associated types using the provided trait instance.
        let trait_m_ty_substituted = trait_m_ty.fold_with(db, &mut substituter);
        let impl_m_ty_substituted = impl_m_ty.fold_with(db, &mut substituter);
        let trait_m_ty_normalized =
            normalize_method_cmp_ty(db, &cx, trait_m_ty_substituted, trait_inst);
        let impl_m_ty_normalized =
            normalize_method_cmp_ty(db, &cx, impl_m_ty_substituted, trait_inst);
        if !impl_m_ty.has_invalid(db) && trait_m_ty_normalized == impl_m_ty_normalized {
            continue;
        }
        if trait_m_ty_normalized.contains_assoc_ty_of_param(db)
            || impl_m_ty_normalized.contains_assoc_ty_of_param(db)
        {
            if !impl_m_ty.has_invalid(db) && trait_m_ty_normalized != impl_m_ty_normalized {
                sink.push(
                    ImplDiag::MethodArgTyMismatch {
                        trait_m,
                        impl_m,
                        trait_m_ty: trait_m_ty_normalized,
                        impl_m_ty: impl_m_ty_normalized,
                        param_idx: idx,
                    }
                    .into(),
                );
                err = true;
            }
            continue;
        }
        if !impl_m_ty.has_invalid(db) && trait_m_ty_normalized != impl_m_ty_normalized {
            sink.push(
                ImplDiag::MethodArgTyMismatch {
                    trait_m,
                    impl_m,
                    trait_m_ty: trait_m_ty_normalized,
                    impl_m_ty: impl_m_ty_normalized,
                    param_idx: idx,
                }
                .into(),
            );
            err = true;
        }
    }

    let impl_m_ret_ty = impl_m.ret_ty(db).instantiate_identity();
    let trait_m_ret_ty = trait_m.ret_ty(db).instantiate(db, map_to_impl);

    let trait_m_ret_ty_substituted = trait_m_ret_ty.fold_with(db, &mut substituter);
    let impl_m_ret_ty_substituted = impl_m_ret_ty.fold_with(db, &mut substituter);
    let trait_m_ret_ty_normalized =
        normalize_method_cmp_ty(db, &cx, trait_m_ret_ty_substituted, trait_inst);
    let impl_m_ret_ty_normalized =
        normalize_method_cmp_ty(db, &cx, impl_m_ret_ty_substituted, trait_inst);
    if !impl_m_ret_ty.has_invalid(db)
        && !trait_m_ret_ty.has_invalid(db)
        && trait_m_ret_ty_normalized == impl_m_ret_ty_normalized
    {
        return !err;
    }
    if trait_m_ret_ty_normalized.contains_assoc_ty_of_param(db)
        || impl_m_ret_ty_normalized.contains_assoc_ty_of_param(db)
    {
        if !impl_m_ret_ty.has_invalid(db)
            && !trait_m_ret_ty.has_invalid(db)
            && trait_m_ret_ty_normalized != impl_m_ret_ty_normalized
        {
            sink.push(
                ImplDiag::MethodRetTyMismatch {
                    trait_m,
                    impl_m,
                    trait_ty: trait_m_ret_ty_normalized,
                    impl_ty: impl_m_ret_ty_normalized,
                }
                .into(),
            );

            err = true;
        }
        return !err;
    }
    if !impl_m_ret_ty.has_invalid(db)
        && !trait_m_ret_ty.has_invalid(db)
        && trait_m_ret_ty_normalized != impl_m_ret_ty_normalized
    {
        sink.push(
            ImplDiag::MethodRetTyMismatch {
                trait_m,
                impl_m,
                trait_ty: trait_m_ret_ty_normalized,
                impl_ty: impl_m_ret_ty_normalized,
            }
            .into(),
        );

        err = true;
    }

    !err
}
/// Checks if the method constraints are stricter than the trait constraints.
/// This check is performed by checking if the `impl_method` constraints are
/// satisfied under the assumptions that is obtained from the `expected_method`
/// constraints.
/// Returns `false` if the comparison fails.
fn compare_constraints<'db>(
    db: &'db dyn HirAnalysisDb,
    impl_m: CallableDef<'db>,
    trait_m: CallableDef<'db>,
    map_to_impl: &[TyId<'db>],
    solve_cx: TraitSolveCx<'db>,
    sink: &mut Vec<TyDiagCollection<'db>>,
) -> bool {
    let impl_m_constraints = collect_func_def_constraints(db, impl_m, false).instantiate_identity();
    let trait_m_constraints =
        collect_func_def_constraints(db, trait_m, false).instantiate(db, map_to_impl);
    let solve_cx = extend_solve_cx(db, solve_cx, trait_m_constraints);
    let mut unsatisfied_goals = ThinVec::new();
    for &goal in impl_m_constraints.list(db) {
        let canonical_goal = Canonical::new(db, goal);
        match is_goal_satisfiable(db, solve_cx, canonical_goal) {
            GoalSatisfiability::Satisfied(_)
            | GoalSatisfiability::ContainsInvalid
            | GoalSatisfiability::NeedsConfirmation(_) => {}
            GoalSatisfiability::UnSat(_) => {
                unsatisfied_goals.push(goal);
            }
        }
    }

    if unsatisfied_goals.is_empty() {
        true
    } else {
        sink.push(
            ImplDiag::MethodStricterBound {
                span: impl_m.name_span(),
                stricter_bounds: unsatisfied_goals,
            }
            .into(),
        );
        false
    }
}
