use num_bigint::{BigInt, BigUint, Sign};
use num_traits::{One, ToPrimitive, Zero};

use crate::core::hir_def::{
    BinOp, Body, Const, EnumVariant, Expr, Func, GenericArgListId, GenericParamOwner, IdentId,
    IntegerId, LitKind, Partial, PathId, Stmt, TypeAlias as HirTypeAlias, TypeId as HirTypeId,
};
use salsa::Update;

use super::const_expr::{ConstExpr, ConstExprId, pretty_print_un_op};
use super::{
    assoc_const::AssocConstUse,
    assoc_items::{analysis_cx_for_selected_assoc_const_body, resolve_assoc_const_selection},
    context::AnalysisCx,
    ctfe::{CtfeConfig, CtfeInterpreter, instantiate_typed_body},
    diagnostics::{BodyDiag, FuncBodyDiag},
    fold::{TyFoldable, TyFolder},
    trait_def::TraitInstId,
    trait_resolution::{
        TraitSolveCx, concretized_missing_trait_const_goal,
        constraint::{collect_constraints, collect_func_decl_constraints},
    },
    ty_check::{TypedBody, check_anon_const_body, check_anon_const_body_in_cx, check_const_body},
    ty_def::{InvalidCause, TyId, TyParam, TyVar},
    ty_lower::contextual_path_resolution_in_cx,
    unify::UnificationTable,
};
use crate::analysis::{
    HirAnalysisDb,
    name_resolution::{PathRes, resolve_path, resolve_path_in_cx},
    ty::trait_resolution::PredicateListId,
    ty::ty_def::{Kind, PrimTy, TyBase, TyData, TyVarSort},
};
use crate::hir_def::ItemKind;
use common::indexmap::IndexMap;
use rustc_hash::FxHashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Update)]
pub struct StoredAnalysisCx<'db> {
    analysis_cx: AnalysisCx<'db>,
}

impl<'db> StoredAnalysisCx<'db> {
    fn new(analysis_cx: AnalysisCx<'db>) -> Self {
        Self { analysis_cx }
    }

    pub(crate) fn get(self) -> AnalysisCx<'db> {
        self.analysis_cx
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Update)]
pub enum LayoutHoleArgSite<'db> {
    Path(PathId<'db>),
    GenericArgList(GenericArgListId<'db>),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Update)]
pub enum CallableInputLayoutHoleOrigin {
    Receiver,
    ValueParam(usize),
    Effect(usize),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HoleId<'db> {
    Structural(StructuralHoleId<'db>),
    Bound(BoundHoleId<'db>),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BoundHoleId<'db> {
    Opaque,
    CallableInput {
        func: Func<'db>,
        origin: CallableInputLayoutHoleOrigin,
        ordinal: usize,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Update)]
pub enum StructuralHoleOrigin<'db> {
    ExplicitWildcard {
        site: LayoutHoleArgSite<'db>,
        arg_idx: usize,
    },
    DefaultHoleParam {
        owner: GenericParamOwner<'db>,
        param_idx: usize,
    },
    EffectKeyExistential {
        path: PathId<'db>,
        arg_idx: usize,
        owner: GenericParamOwner<'db>,
        param_idx: usize,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Update)]
pub enum LocalFrameSite<'db> {
    HirType(HirTypeId<'db>),
    TypeComponent { ty: HirTypeId<'db>, slot: usize },
    RootPath(PathId<'db>),
    GenericArgList(GenericArgListId<'db>),
    AliasTemplate(HirTypeAlias<'db>),
}

#[salsa::interned]
#[derive(Debug)]
pub struct LocalFrameId<'db> {
    pub parent: Option<LocalFrameId<'db>>,
    pub site: LocalFrameSite<'db>,
}

impl<'db> LocalFrameId<'db> {
    pub(crate) fn root_hir_ty(db: &'db dyn HirAnalysisDb, hir_ty: HirTypeId<'db>) -> Self {
        Self::new(db, None, LocalFrameSite::HirType(hir_ty))
    }

    pub(crate) fn child_type_component(
        self,
        db: &'db dyn HirAnalysisDb,
        ty: HirTypeId<'db>,
        slot: usize,
    ) -> Self {
        Self::new(db, Some(self), LocalFrameSite::TypeComponent { ty, slot })
    }

    pub(crate) fn root_path(db: &'db dyn HirAnalysisDb, path: PathId<'db>) -> Self {
        Self::new(db, None, LocalFrameSite::RootPath(path))
    }

    pub(crate) fn root_generic_arg_list(
        db: &'db dyn HirAnalysisDb,
        args: GenericArgListId<'db>,
    ) -> Self {
        Self::new(db, None, LocalFrameSite::GenericArgList(args))
    }

    pub(crate) fn prepend_parent(self, db: &'db dyn HirAnalysisDb, parent: Self) -> Self {
        let rebased_parent = self
            .parent(db)
            .map(|current| current.prepend_parent(db, parent));
        Self::new(db, rebased_parent.or(Some(parent)), self.site(db))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Update)]
pub enum AppFrameSite<'db> {
    CallableInput {
        func: Func<'db>,
        origin: CallableInputLayoutHoleOrigin,
    },
    TypeComponent {
        ty: HirTypeId<'db>,
        slot: usize,
    },
    RootPath(PathId<'db>),
    GenericArgList(GenericArgListId<'db>),
}

#[salsa::interned]
#[derive(Debug)]
pub struct AppFrameId<'db> {
    pub parent: Option<AppFrameId<'db>>,
    pub site: AppFrameSite<'db>,
}

impl<'db> AppFrameId<'db> {
    pub(crate) fn root_callable_input(
        db: &'db dyn HirAnalysisDb,
        func: Func<'db>,
        origin: CallableInputLayoutHoleOrigin,
    ) -> Self {
        Self::new(db, None, AppFrameSite::CallableInput { func, origin })
    }

    pub(crate) fn child_type_component(
        self,
        db: &'db dyn HirAnalysisDb,
        ty: HirTypeId<'db>,
        slot: usize,
    ) -> Self {
        Self::new(db, Some(self), AppFrameSite::TypeComponent { ty, slot })
    }

    pub(crate) fn root_path(db: &'db dyn HirAnalysisDb, path: PathId<'db>) -> Self {
        Self::new(db, None, AppFrameSite::RootPath(path))
    }

    pub(crate) fn root_generic_arg_list(
        db: &'db dyn HirAnalysisDb,
        args: GenericArgListId<'db>,
    ) -> Self {
        Self::new(db, None, AppFrameSite::GenericArgList(args))
    }

    pub(crate) fn prepend_parent(self, db: &'db dyn HirAnalysisDb, parent: Self) -> Self {
        let rebased_parent = self
            .parent(db)
            .map(|current| current.prepend_parent(db, parent));
        Self::new(db, rebased_parent.or(Some(parent)), self.site(db))
    }
}

#[salsa::interned]
#[derive(Debug)]
pub struct StructuralHoleId<'db> {
    pub expected_ty: TyId<'db>,
    pub origin: StructuralHoleOrigin<'db>,
    pub local_frame: LocalFrameId<'db>,
    pub app_frame: Option<AppFrameId<'db>>,
}

impl<'db> StructuralHoleId<'db> {
    pub(crate) fn prepend_local_parent(
        self,
        db: &'db dyn HirAnalysisDb,
        parent: LocalFrameId<'db>,
    ) -> Self {
        Self::new(
            db,
            self.expected_ty(db),
            self.origin(db),
            self.local_frame(db).prepend_parent(db, parent),
            self.app_frame(db),
        )
    }

    pub(crate) fn rebase_app_under(
        self,
        db: &'db dyn HirAnalysisDb,
        parent: AppFrameId<'db>,
    ) -> Self {
        let app_frame = self
            .app_frame(db)
            .map(|frame| frame.prepend_parent(db, parent))
            .or(Some(parent));
        Self::new(
            db,
            self.expected_ty(db),
            self.origin(db),
            self.local_frame(db),
            app_frame,
        )
    }
}

impl<'db> HoleId<'db> {
    pub(crate) fn bound_callable(
        func: Func<'db>,
        origin: CallableInputLayoutHoleOrigin,
        ordinal: usize,
    ) -> Self {
        Self::Bound(BoundHoleId::CallableInput {
            func,
            origin,
            ordinal,
        })
    }

    pub(crate) fn bound_opaque() -> Self {
        Self::Bound(BoundHoleId::Opaque)
    }

    pub(crate) fn structural(
        db: &'db dyn HirAnalysisDb,
        expected_ty: TyId<'db>,
        origin: StructuralHoleOrigin<'db>,
        local_frame: LocalFrameId<'db>,
    ) -> Self {
        Self::Structural(StructuralHoleId::new(
            db,
            expected_ty,
            origin,
            local_frame,
            None,
        ))
    }

    pub(crate) fn structural_with_app(
        db: &'db dyn HirAnalysisDb,
        expected_ty: TyId<'db>,
        origin: StructuralHoleOrigin<'db>,
        local_frame: LocalFrameId<'db>,
        app_frame: Option<AppFrameId<'db>>,
    ) -> Self {
        Self::Structural(StructuralHoleId::new(
            db,
            expected_ty,
            origin,
            local_frame,
            app_frame,
        ))
    }
}

fn pretty_print_const_arg<'db>(db: &'db dyn HirAnalysisDb, ty: TyId<'db>) -> String {
    match ty.data(db) {
        TyData::ConstTy(const_ty) if matches!(const_ty.data(db), ConstTyData::TyParam(param, _) if param.is_normal()) =>
        {
            let ConstTyData::TyParam(param, _) = const_ty.data(db) else {
                unreachable!()
            };
            param.name.data(db).to_string()
        }
        _ => ty.pretty_print(db).to_string(),
    }
}

fn generic_const_param_display_map<'db>(
    db: &'db dyn HirAnalysisDb,
    body: Body<'db>,
    generic_args: &[TyId<'db>],
) -> FxHashMap<IdentId<'db>, String> {
    let Some(owner) = body
        .scope()
        .parent_item(db)
        .and_then(GenericParamOwner::from_item_opt)
    else {
        return FxHashMap::default();
    };

    super::ty_lower::collect_generic_params(db, owner)
        .params(db)
        .iter()
        .copied()
        .enumerate()
        .filter_map(|(idx, param_ty)| {
            let TyData::ConstTy(const_ty) = param_ty.data(db) else {
                return None;
            };
            let ConstTyData::TyParam(param, _) = const_ty.data(db) else {
                return None;
            };
            generic_args
                .get(idx)
                .map(|arg| (param.name, pretty_print_const_arg(db, *arg)))
        })
        .collect()
}

fn pretty_print_const_body_expr<'db>(
    db: &'db dyn HirAnalysisDb,
    body: Body<'db>,
    expr_id: crate::hir_def::ExprId,
    generic_param_display: &FxHashMap<IdentId<'db>, String>,
) -> Option<String> {
    ConstBodyExprPrinter {
        db,
        body,
        generic_param_display,
    }
    .pretty_print(expr_id)
}

struct ConstBodyExprPrinter<'a, 'db> {
    db: &'db dyn HirAnalysisDb,
    body: Body<'db>,
    generic_param_display: &'a FxHashMap<IdentId<'db>, String>,
}

impl<'a, 'db> ConstBodyExprPrinter<'a, 'db> {
    fn pretty_print(&self, expr_id: crate::hir_def::ExprId) -> Option<String> {
        let Partial::Present(expr) = expr_id.data(self.db, self.body) else {
            return None;
        };

        match expr {
            Expr::Lit(lit) => Some(lit.pretty_print(self.db)),
            Expr::Path(path) if path.is_present() => Some(self.pretty_print_path(path.unwrap())),
            Expr::Call(callee, args) => {
                let callee = self.pretty_print(*callee)?;
                let args = args
                    .iter()
                    .map(|arg| self.pretty_print(arg.expr))
                    .collect::<Option<Vec<_>>>()?;
                Some(format!("{callee}({})", args.join(", ")))
            }
            Expr::Bin(lhs, rhs, op) if !matches!(op, BinOp::Index) => {
                let lhs = self.pretty_print(*lhs)?;
                let rhs = self.pretty_print(*rhs)?;
                Some(format!("({lhs} {} {rhs})", op.pretty_print()))
            }
            Expr::Un(expr, op) => Some(pretty_print_un_op(*op, self.pretty_print(*expr)?)),
            Expr::Cast(expr, to) => Some(format!(
                "({} as {})",
                self.pretty_print(*expr)?,
                to.to_opt()?.pretty_print(self.db)
            )),
            Expr::Block(stmts) if stmts.len() == 1 => match stmts[0].data(self.db, self.body) {
                Partial::Present(Stmt::Expr(tail_expr)) => self.pretty_print(*tail_expr),
                Partial::Present(_) | Partial::Absent => None,
            },
            _ => None,
        }
    }

    fn pretty_print_path(&self, path: PathId<'db>) -> String {
        if path.parent(self.db).is_none()
            && let Some(ident) = path.as_ident(self.db)
            && let Some(replacement) = self.generic_param_display.get(&ident)
        {
            replacement.clone()
        } else {
            path.pretty_print(self.db)
        }
    }
}

pub(crate) fn const_body_simple_path<'db>(
    db: &'db dyn HirAnalysisDb,
    body: Body<'db>,
) -> Option<PathId<'db>> {
    fn expr_simple_path<'db>(
        db: &'db dyn HirAnalysisDb,
        body: Body<'db>,
        expr: &Expr<'db>,
    ) -> Option<PathId<'db>> {
        match expr {
            Expr::Path(path) => path.to_opt(),
            Expr::Block(stmts) => {
                let [stmt] = stmts.as_slice() else {
                    return None;
                };
                let Partial::Present(stmt) = stmt.data(db, body) else {
                    return None;
                };
                let Stmt::Expr(expr_id) = stmt else {
                    return None;
                };
                let Partial::Present(expr) = expr_id.data(db, body) else {
                    return None;
                };
                expr_simple_path(db, body, expr)
            }
            _ => None,
        }
    }

    let Partial::Present(expr) = body.expr(db).data(db, body) else {
        return None;
    };
    expr_simple_path(db, body, expr)
}

pub fn normalize_const_tys_for_comparison<'db>(
    db: &'db dyn HirAnalysisDb,
    ty: TyId<'db>,
) -> TyId<'db> {
    struct ComparisonConstFolder;

    impl<'db> TyFolder<'db> for ComparisonConstFolder {
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
            let ConstTyData::UnEvaluated {
                ty: Some(expected_ty),
                ..
            } = const_ty.data(db)
            else {
                return ty.super_fold_with(db, self);
            };

            let normalized = const_ty.evaluate(db, Some(*expected_ty));
            if normalized.ty(db).invalid_cause(db).is_none()
                && matches!(
                    normalized.data(db),
                    ConstTyData::Evaluated(..) | ConstTyData::Abstract(..)
                )
            {
                TyId::const_ty(db, normalized)
            } else {
                ty.super_fold_with(db, self)
            }
        }
    }

    ty.fold_with(db, &mut ComparisonConstFolder)
}

pub(crate) struct ValidatedUnEvaluatedConst<'db> {
    pub expected_ty: TyId<'db>,
    pub typed_body: TypedBody<'db>,
}

pub(crate) fn retype_hole_const_ty<'db>(
    db: &'db dyn HirAnalysisDb,
    const_ty: ConstTyId<'db>,
    expected_ty: TyId<'db>,
) -> Option<ConstTyId<'db>> {
    matches!(const_ty.data(db), ConstTyData::Hole(..)).then(|| const_ty.with_ty(db, expected_ty))
}

fn validate_unevaluated_const_ty_with_solve_cx<'db>(
    db: &'db dyn HirAnalysisDb,
    const_ty: ConstTyId<'db>,
    expected_ty: Option<TyId<'db>>,
    solve_cx: Option<TraitSolveCx<'db>>,
) -> Result<ValidatedUnEvaluatedConst<'db>, InvalidCause<'db>> {
    let ConstTyData::UnEvaluated {
        body,
        ty: const_ty_ty,
        const_def,
        generic_args,
        unevaluated_cx,
        ..
    } = const_ty.data(db)
    else {
        return Err(InvalidCause::Other);
    };

    let Some(expected_ty) = expected_ty.or(*const_ty_ty) else {
        return Err(InvalidCause::InvalidConstTyExpr { body: *body });
    };
    let check_ty = if generic_args.is_empty() {
        expected_ty
    } else {
        const_ty_ty.unwrap_or(expected_ty)
    };

    let (diags, typed_body) = match const_def {
        Some(const_def) => {
            let result = check_const_body(db, *const_def);
            (result.0.clone(), result.1.clone())
        }
        None => (*unevaluated_cx)
            .map(StoredAnalysisCx::get)
            .or_else(|| solve_cx.map(AnalysisCx::from_solve_cx))
            .map_or_else(
                || {
                    let result = check_anon_const_body(db, *body, check_ty);
                    (result.0.clone(), result.1.clone())
                },
                |cx| {
                    let result = check_anon_const_body_in_cx(db, *body, check_ty, cx);
                    (result.0.clone(), result.1.clone())
                },
            ),
    };

    if let Some((expected, given)) = diags.iter().find_map(|diag| match diag {
        FuncBodyDiag::Body(BodyDiag::TypeMismatch {
            expected, given, ..
        }) => Some((*expected, *given)),
        _ => None,
    }) {
        if matches!(body.scope().parent_item(db), Some(ItemKind::ImplTrait(_))) {
            return Err(InvalidCause::Other);
        }
        return Err(InvalidCause::ConstTyMismatch { expected, given });
    }

    if !diags.is_empty() {
        return Err(InvalidCause::Other);
    }

    check_const_ty(
        db,
        check_ty,
        Some(expected_ty),
        &mut UnificationTable::new(db),
    )?;
    Ok(ValidatedUnEvaluatedConst {
        expected_ty,
        typed_body,
    })
}

pub(crate) fn check_unevaluated_const_ty_without_eval<'db>(
    db: &'db dyn HirAnalysisDb,
    const_ty: ConstTyId<'db>,
    expected_ty: TyId<'db>,
) -> Result<ConstTyId<'db>, InvalidCause<'db>> {
    validate_unevaluated_const_ty_with_solve_cx(db, const_ty, Some(expected_ty), None)?;
    Ok(const_ty.with_ty(db, expected_ty))
}

#[salsa::interned]
#[derive(Debug)]
pub struct ConstTyId<'db> {
    #[return_ref]
    pub data: ConstTyData<'db>,
}

#[salsa::tracked(cycle_initial=evaluate_const_ty_cycle_initial, cycle_fn=evaluate_const_ty_cycle_recover)]
pub(crate) fn evaluate_const_ty<'db>(
    db: &'db dyn HirAnalysisDb,
    const_ty: ConstTyId<'db>,
    expected_ty: Option<TyId<'db>>,
) -> ConstTyId<'db> {
    evaluate_const_ty_with_solve_cx(db, const_ty, expected_ty, None)
}

fn evaluate_const_ty_with_solve_cx<'db>(
    db: &'db dyn HirAnalysisDb,
    const_ty: ConstTyId<'db>,
    expected_ty: Option<TyId<'db>>,
    solve_cx: Option<TraitSolveCx<'db>>,
) -> ConstTyId<'db> {
    if let Some(expected_ty) = expected_ty
        && let Some(retyped) = retype_hole_const_ty(db, const_ty, expected_ty)
    {
        return retyped;
    }
    if matches!(const_ty.data(db), ConstTyData::Hole(..)) {
        return const_ty;
    }

    if let ConstTyData::Abstract(expr, ty) = const_ty.data(db)
        && let ConstExpr::TraitConst(assoc) = expr.data(db)
        && let Some(resolved) = solve_cx.map_or_else(
            || const_ty_from_assoc_const_use(db, *assoc),
            |solve_cx| {
                assoc.analysis_cx(db, Some(solve_cx)).map_or_else(
                    || {
                        const_ty_from_trait_const(
                            db,
                            solve_cx.with_assumptions(assoc.assumptions()),
                            assoc.inst(),
                            assoc.name(),
                        )
                    },
                    |cx| const_ty_from_trait_const_in_cx(db, &cx, assoc.inst(), assoc.name()),
                )
            },
        )
    {
        let evaluated = solve_cx.map_or_else(
            || resolved.evaluate(db, expected_ty.or(Some(*ty))),
            |solve_cx| resolved.evaluate_with_solve_cx(db, expected_ty.or(Some(*ty)), solve_cx),
        );
        if matches!(
            evaluated.ty(db).invalid_cause(db),
            Some(InvalidCause::ConstEvalUnsupported { .. })
        ) && expected_ty.is_some()
        {
            return const_ty;
        }
        return evaluated;
    }

    let (body, const_ty_ty, generic_args, unevaluated_cx) = match const_ty.data(db) {
        ConstTyData::UnEvaluated {
            body,
            ty,
            generic_args,
            unevaluated_cx,
            ..
        } => (*body, *ty, generic_args.clone(), *unevaluated_cx),
        _ => {
            let const_ty_ty = const_ty.ty(db);
            return match check_const_ty(
                db,
                const_ty_ty,
                expected_ty,
                &mut UnificationTable::new(db),
            ) {
                Ok(_) => const_ty,
                Err(cause) => {
                    let ty = TyId::invalid(db, cause);
                    return const_ty.swap_ty(db, ty);
                }
            };
        }
    };
    let analysis_cx = unevaluated_cx.map(StoredAnalysisCx::get);

    let expected_ty = expected_ty.or(const_ty_ty);
    let check_ty = if generic_args.is_empty() {
        expected_ty
    } else {
        const_ty_ty.or(expected_ty)
    };

    let Partial::Present(expr) = body.expr(db).data(db, body) else {
        let data = ConstTyData::Evaluated(
            EvaluatedConstTy::Invalid,
            TyId::invalid(db, InvalidCause::ParseError),
        );
        return ConstTyId::new(db, data);
    };

    let expr = expr.clone();

    #[derive(Clone, Copy, Debug)]
    struct CheckedIntTy {
        bits: u16,
        signed: bool,
    }

    fn checked_int_ty_from_ty<'db>(
        db: &'db dyn HirAnalysisDb,
        expected: Option<TyId<'db>>,
    ) -> Option<CheckedIntTy> {
        let expected = expected?;
        let base_ty = expected.base_ty(db);
        let TyData::TyBase(TyBase::Prim(prim)) = base_ty.data(db) else {
            return None;
        };
        Some(match prim {
            // unsigned
            PrimTy::U8 => CheckedIntTy {
                bits: 8,
                signed: false,
            },
            PrimTy::U16 => CheckedIntTy {
                bits: 16,
                signed: false,
            },
            PrimTy::U32 => CheckedIntTy {
                bits: 32,
                signed: false,
            },
            PrimTy::U64 => CheckedIntTy {
                bits: 64,
                signed: false,
            },
            PrimTy::U128 => CheckedIntTy {
                bits: 128,
                signed: false,
            },
            PrimTy::U256 | PrimTy::Usize => CheckedIntTy {
                bits: 256,
                signed: false,
            },
            // signed
            PrimTy::I8 => CheckedIntTy {
                bits: 8,
                signed: true,
            },
            PrimTy::I16 => CheckedIntTy {
                bits: 16,
                signed: true,
            },
            PrimTy::I32 => CheckedIntTy {
                bits: 32,
                signed: true,
            },
            PrimTy::I64 => CheckedIntTy {
                bits: 64,
                signed: true,
            },
            PrimTy::I128 => CheckedIntTy {
                bits: 128,
                signed: true,
            },
            PrimTy::I256 | PrimTy::Isize => CheckedIntTy {
                bits: 256,
                signed: true,
            },
            _ => return None,
        })
    }

    fn u256_modulus() -> BigUint {
        BigUint::one() << 256usize
    }

    fn signed_bounds(ty: CheckedIntTy) -> (BigInt, BigInt) {
        debug_assert!(ty.signed);
        let half = BigInt::one() << ((ty.bits - 1) as usize);
        let min = -half.clone();
        let max = half - BigInt::one();
        (min, max)
    }

    fn unsigned_max(ty: CheckedIntTy) -> BigInt {
        debug_assert!(!ty.signed);
        (BigInt::one() << (ty.bits as usize)) - BigInt::one()
    }

    fn in_range(value: &BigInt, ty: CheckedIntTy) -> bool {
        if ty.signed {
            let (min, max) = signed_bounds(ty);
            value >= &min && value <= &max
        } else {
            value >= &BigInt::zero() && value <= &unsigned_max(ty)
        }
    }

    fn bigint_to_u256_word(value: &BigInt) -> Option<BigUint> {
        let modulus = u256_modulus();
        match value.sign() {
            Sign::Minus => {
                let abs = value.magnitude();
                if abs > &modulus {
                    return None;
                }
                if abs.is_zero() {
                    Some(BigUint::zero())
                } else {
                    Some(&modulus - abs)
                }
            }
            _ => value.to_biguint().and_then(|v| (v < modulus).then_some(v)),
        }
    }

    fn u256_word_to_bigint(word: &BigUint, ty: CheckedIntTy) -> BigInt {
        if !ty.signed {
            return BigInt::from(word.clone());
        }

        let bits = ty.bits as usize;
        let mask = if bits == 256 {
            (BigUint::one() << 256usize) - BigUint::one()
        } else {
            (BigUint::one() << (ty.bits as usize)) - BigUint::one()
        };
        let value_bits = word & mask;
        let sign_bit = BigUint::one() << ((ty.bits - 1) as usize);
        if (value_bits.clone() & sign_bit).is_zero() {
            BigInt::from(value_bits)
        } else {
            BigInt::from(value_bits) - (BigInt::one() << (ty.bits as usize))
        }
    }

    #[derive(Clone, Copy, Debug)]
    enum ConstIntError {
        Overflow,
        DivisionByZero,
        NegativeExponent,
        /// The expression is not a pure integer expression (e.g. contains
        /// function calls). Fall through to CTFE instead of reporting an error.
        NotIntExpr,
    }

    fn eval_int_expr<'db>(
        db: &'db dyn HirAnalysisDb,
        body: Body<'db>,
        expr: &Expr<'db>,
        expected: Option<CheckedIntTy>,
        generic_args: &[TyId<'db>],
        analysis_cx: Option<AnalysisCx<'db>>,
    ) -> Result<BigInt, ConstIntError> {
        match expr {
            Expr::Block(stmts) => {
                let [stmt] = stmts.as_slice() else {
                    return Err(ConstIntError::NotIntExpr);
                };
                let Partial::Present(stmt) = stmt.data(db, body) else {
                    return Err(ConstIntError::NotIntExpr);
                };
                let Stmt::Expr(expr_id) = stmt else {
                    return Err(ConstIntError::NotIntExpr);
                };
                let Partial::Present(inner) = expr_id.data(db, body) else {
                    return Err(ConstIntError::NotIntExpr);
                };
                eval_int_expr(db, body, inner, expected, generic_args, analysis_cx)
            }

            Expr::Lit(LitKind::Int(i)) => Ok(BigInt::from(i.data(db).clone())),

            Expr::Un(inner, op) => {
                let Partial::Present(inner) = inner.data(db, body) else {
                    return Err(ConstIntError::Overflow);
                };
                let value = eval_int_expr(db, body, inner, expected, generic_args, analysis_cx)?;
                match op {
                    crate::core::hir_def::expr::UnOp::Minus => {
                        let Some(expected) = expected else {
                            return Err(ConstIntError::Overflow);
                        };
                        let neg = -value;
                        if !in_range(&neg, expected) {
                            return Err(ConstIntError::Overflow);
                        }
                        Ok(neg)
                    }
                    crate::core::hir_def::expr::UnOp::Plus => Ok(value),
                    _ => Err(ConstIntError::NotIntExpr),
                }
            }

            Expr::Bin(lhs_id, rhs_id, op) => {
                let Partial::Present(lhs) = lhs_id.data(db, body) else {
                    return Err(ConstIntError::Overflow);
                };
                let Partial::Present(rhs) = rhs_id.data(db, body) else {
                    return Err(ConstIntError::Overflow);
                };
                let expected = expected.unwrap_or(CheckedIntTy {
                    bits: 256,
                    signed: false,
                });

                let lhs = eval_int_expr(db, body, lhs, Some(expected), generic_args, analysis_cx)?;
                let rhs = eval_int_expr(db, body, rhs, Some(expected), generic_args, analysis_cx)?;

                match op {
                    crate::core::hir_def::expr::BinOp::Arith(op) => match op {
                        crate::core::hir_def::expr::ArithBinOp::Add => {
                            let result = lhs + rhs;
                            if !in_range(&result, expected) {
                                Err(ConstIntError::Overflow)
                            } else {
                                Ok(result)
                            }
                        }
                        crate::core::hir_def::expr::ArithBinOp::Sub => {
                            let result = lhs - rhs;
                            if !in_range(&result, expected) {
                                Err(ConstIntError::Overflow)
                            } else {
                                Ok(result)
                            }
                        }
                        crate::core::hir_def::expr::ArithBinOp::Mul => {
                            let result = lhs * rhs;
                            if !in_range(&result, expected) {
                                Err(ConstIntError::Overflow)
                            } else {
                                Ok(result)
                            }
                        }
                        crate::core::hir_def::expr::ArithBinOp::Div => {
                            if rhs.is_zero() {
                                return Err(ConstIntError::DivisionByZero);
                            }
                            if expected.signed {
                                let (min, _) = signed_bounds(expected);
                                if lhs == min && rhs == -BigInt::one() {
                                    return Err(ConstIntError::Overflow);
                                }
                            }
                            let result = lhs / rhs;
                            if !in_range(&result, expected) {
                                Err(ConstIntError::Overflow)
                            } else {
                                Ok(result)
                            }
                        }
                        crate::core::hir_def::expr::ArithBinOp::Rem => {
                            if rhs.is_zero() {
                                return Err(ConstIntError::DivisionByZero);
                            }
                            let result = lhs % rhs;
                            if !in_range(&result, expected) {
                                Err(ConstIntError::Overflow)
                            } else {
                                Ok(result)
                            }
                        }
                        crate::core::hir_def::expr::ArithBinOp::Pow => {
                            if rhs.sign() == Sign::Minus {
                                return Err(ConstIntError::NegativeExponent);
                            }
                            let Some(exp) = rhs.to_biguint() else {
                                return Err(ConstIntError::NegativeExponent);
                            };
                            let mut acc = BigInt::one();
                            let mut base = lhs;
                            let mut exp = exp;
                            while !exp.is_zero() {
                                if (&exp & BigUint::one()) == BigUint::one() {
                                    acc *= base.clone();
                                    if !in_range(&acc, expected) {
                                        return Err(ConstIntError::Overflow);
                                    }
                                }
                                exp >>= 1usize;
                                if exp.is_zero() {
                                    break;
                                }
                                base = base.clone() * base;
                                if !in_range(&base, expected) {
                                    return Err(ConstIntError::Overflow);
                                }
                            }
                            Ok(acc)
                        }
                        _ => Err(ConstIntError::NotIntExpr),
                    },
                    _ => Err(ConstIntError::NotIntExpr),
                }
            }

            Expr::Path(path) => {
                if !generic_args.is_empty() {
                    return Err(ConstIntError::NotIntExpr);
                }
                let Some(path) = path.to_opt() else {
                    return Err(ConstIntError::NotIntExpr);
                };
                let assumptions = assumptions_for_body(db, body);
                let resolved = analysis_cx
                    .and_then(|cx| {
                        contextual_path_resolution_in_cx(db, body.scope(), path, true, &cx)
                    })
                    .map(Ok)
                    .or_else(|| {
                        analysis_cx.map(|cx| resolve_path_in_cx(db, path, body.scope(), true, &cx))
                    })
                    .unwrap_or_else(|| resolve_path(db, path, body.scope(), assumptions, true))
                    .map_err(|_| ConstIntError::NotIntExpr)?;

                let const_ty = match resolved {
                    PathRes::Const(const_def, declared_ty) => {
                        let body = const_def
                            .body(db)
                            .to_opt()
                            .ok_or(ConstIntError::NotIntExpr)?;
                        ConstTyId::from_body(db, body, Some(declared_ty), Some(const_def))
                    }
                    PathRes::TraitConst(_recv_ty, inst, name) => {
                        let solve_cx =
                            TraitSolveCx::new(db, body.scope()).with_assumptions(assumptions);
                        const_ty_from_trait_const(db, solve_cx, inst, name)
                            .ok_or(ConstIntError::NotIntExpr)?
                    }
                    _ => return Err(ConstIntError::NotIntExpr),
                };

                let evaluated = const_ty.evaluate(db, None);
                match evaluated.data(db) {
                    ConstTyData::Evaluated(EvaluatedConstTy::LitInt(i), _) => {
                        let word = i.data(db);
                        let expected_for_interpretation = expected.unwrap_or(CheckedIntTy {
                            bits: 256,
                            signed: false,
                        });
                        Ok(u256_word_to_bigint(word, expected_for_interpretation))
                    }
                    _ => Err(ConstIntError::NotIntExpr),
                }
            }

            _ => Err(ConstIntError::NotIntExpr),
        }
    }

    struct GenericArgSubst<'a, 'db> {
        db: &'db dyn HirAnalysisDb,
        generic_args: &'a [TyId<'db>],
    }

    impl<'a, 'db> TyFolder<'db> for GenericArgSubst<'a, 'db> {
        fn fold_ty(&mut self, db: &'db dyn HirAnalysisDb, ty: TyId<'db>) -> TyId<'db> {
            match ty.data(self.db) {
                TyData::TyParam(param) => self
                    .generic_args
                    .get(param.idx)
                    .copied()
                    .unwrap_or_else(|| ty.super_fold_with(db, self)),
                TyData::ConstTy(const_ty)
                    if matches!(const_ty.data(self.db), ConstTyData::TyParam(_, _)) =>
                {
                    let ConstTyData::TyParam(param, _) = const_ty.data(self.db) else {
                        unreachable!();
                    };
                    self.generic_args
                        .get(param.idx)
                        .copied()
                        .unwrap_or_else(|| ty.super_fold_with(db, self))
                }
                _ => ty.super_fold_with(db, self),
            }
        }
    }

    if let Expr::Path(path) = &expr {
        let subst_ty = |ty: TyId<'db>| {
            if generic_args.is_empty() {
                ty
            } else {
                let mut subst = GenericArgSubst {
                    db,
                    generic_args: &generic_args,
                };
                ty.fold_with(db, &mut subst)
            }
        };
        let subst_inst = |inst: TraitInstId<'db>| {
            if generic_args.is_empty() {
                inst
            } else {
                let mut subst = GenericArgSubst {
                    db,
                    generic_args: &generic_args,
                };
                inst.fold_with(db, &mut subst)
            }
        };

        let Some(path) = path.to_opt() else {
            return ConstTyId::new(
                db,
                ConstTyData::Evaluated(
                    EvaluatedConstTy::Invalid,
                    TyId::invalid(db, InvalidCause::ParseError),
                ),
            );
        };

        let assumptions =
            solve_cx.map_or_else(|| assumptions_for_body(db, body), TraitSolveCx::assumptions);
        let resolved_path = analysis_cx
            .and_then(|cx| contextual_path_resolution_in_cx(db, body.scope(), path, true, &cx))
            .map(Ok)
            .or_else(|| analysis_cx.map(|cx| resolve_path_in_cx(db, path, body.scope(), true, &cx)))
            .unwrap_or_else(|| resolve_path(db, path, body.scope(), assumptions, true));
        if let Ok(resolved_path) = resolved_path {
            match resolved_path {
                PathRes::Ty(ty) | PathRes::TyAlias(_, ty) => {
                    let ty = subst_ty(ty);
                    if let TyData::ConstTy(const_ty) = ty.data(db) {
                        return const_ty.evaluate(db, expected_ty);
                    }
                }
                PathRes::Const(const_def, ty) => {
                    if let Some(body) = const_def.body(db).to_opt() {
                        let const_ty = ConstTyId::from_body(db, body, Some(ty), Some(const_def));
                        let expected = expected_ty.or(Some(ty));
                        return const_ty.evaluate(db, expected);
                    }
                }
                PathRes::TraitConst(recv_ty, inst, name) => {
                    let recv_ty = subst_ty(recv_ty);
                    let inst = subst_inst(inst);
                    let mut args = inst.args(db).clone();
                    if let Some(self_arg) = args.first_mut() {
                        *self_arg = recv_ty;
                    }
                    let inst = TraitInstId::new(
                        db,
                        inst.def(db),
                        args,
                        inst.assoc_type_bindings(db).clone(),
                    );

                    let mk_abstract = |expected_ty: TyId<'db>| {
                        let expr = ConstExprId::new(
                            db,
                            ConstExpr::TraitConst(analysis_cx.map_or_else(
                                || AssocConstUse::new(body.scope(), assumptions, inst, name),
                                |cx| {
                                    AssocConstUse::new(body.scope(), assumptions, inst, name)
                                        .with_analysis_cx(cx)
                                },
                            )),
                        );
                        ConstTyId::new(db, ConstTyData::Abstract(expr, expected_ty))
                    };

                    let solve_cx = solve_cx.unwrap_or_else(|| {
                        TraitSolveCx::new(db, body.scope()).with_assumptions(assumptions)
                    });
                    if let Some(const_ty) = const_ty_from_trait_const(db, solve_cx, inst, name) {
                        let evaluated = const_ty.evaluate(db, expected_ty);
                        if matches!(
                            evaluated.ty(db).invalid_cause(db),
                            Some(InvalidCause::ConstEvalUnsupported { .. })
                        ) && let Some(expected_ty) = expected_ty
                        {
                            return mk_abstract(expected_ty);
                        }
                        return evaluated;
                    }

                    if let Some(inst) =
                        concretized_missing_trait_const_goal(db, solve_cx, inst, name)
                    {
                        return ConstTyId::invalid(
                            db,
                            InvalidCause::TraitConstNotImplemented { inst, name },
                        );
                    }

                    if let Some(expected_ty) = expected_ty {
                        return mk_abstract(expected_ty);
                    }
                }
                PathRes::EnumVariant(variant) if variant.ty.is_unit_variant_only_enum(db) => {
                    let evaluated = EvaluatedConstTy::EnumVariant(variant.variant);
                    let const_ty =
                        ConstTyId::new(db, ConstTyData::Evaluated(evaluated, variant.ty));
                    return const_ty.evaluate(db, expected_ty);
                }
                _ => {}
            }
        }

        // If the path failed to resolve but looks like a path to a value
        // (e.g., a trait associated const like `Type::CONST`), keep it
        // unevaluated and assume the expected type if available, avoiding
        // spurious diagnostics here. Downstream checks will validate usage.
        if path.parent(db).is_some() {
            return ConstTyId::from_body(db, body, expected_ty, None);
        }

        return ConstTyId::new(
            db,
            ConstTyData::Evaluated(
                EvaluatedConstTy::Invalid,
                TyId::invalid(db, InvalidCause::InvalidConstTyExpr { body }),
            ),
        );
    }

    // Try BigInt-based evaluation for integer arithmetic expressions (checked arithmetic).
    if matches!(
        expr,
        Expr::Block(..) | Expr::Un(..) | Expr::Bin(..) | Expr::Lit(LitKind::Int(..))
    ) {
        let expected_int_ty = expected_ty.and_then(|ty| checked_int_ty_from_ty(db, Some(ty)));
        match eval_int_expr(db, body, &expr, expected_int_ty, &generic_args, analysis_cx) {
            Ok(value) => {
                if let Some(word) = bigint_to_u256_word(&value) {
                    let mut table = UnificationTable::new(db);
                    let resolved = EvaluatedConstTy::LitInt(IntegerId::new(db, word));
                    let ty = table.new_var(TyVarSort::Integral, &Kind::Star);
                    let data = match check_const_ty(db, ty, expected_ty, &mut table) {
                        Ok(ty) => ConstTyData::Evaluated(resolved, ty),
                        Err(err) => ConstTyData::Evaluated(resolved, TyId::invalid(db, err)),
                    };
                    return ConstTyId::new(db, data);
                }
            }
            Err(ConstIntError::NotIntExpr) => {
                // Expression contains constructs we can't evaluate with BigInt
                // (e.g. function calls). Fall through to CTFE.
            }
            Err(_) => {
                // Keep BigInt evaluation as a fast path for pure integer arithmetic, but let CTFE
                // remain authoritative for actual failure shaping.
            }
        }
    }

    if check_ty.is_none() {
        return ConstTyId::invalid(db, InvalidCause::InvalidConstTyExpr { body });
    }
    let validated =
        match validate_unevaluated_const_ty_with_solve_cx(db, const_ty, expected_ty, solve_cx) {
            Ok(validated) => validated,
            Err(InvalidCause::InvalidConstTyExpr { body }) => {
                return ConstTyId::invalid(
                    db,
                    InvalidCause::ConstEvalUnsupported {
                        body,
                        expr: body.expr(db),
                    },
                );
            }
            Err(cause) => return ConstTyId::invalid(db, cause),
        };

    let mut interp = CtfeInterpreter::new(db, CtfeConfig::default());
    let typed_body = if generic_args.is_empty() {
        validated.typed_body
    } else {
        instantiate_typed_body(db, validated.typed_body.clone(), &generic_args)
    };
    let evaluated = interp
        .eval_expr_in_body(body, typed_body, generic_args, body.expr(db))
        .unwrap_or_else(|cause| ConstTyId::invalid(db, cause));

    let mut table = UnificationTable::new(db);
    match check_const_ty(
        db,
        evaluated.ty(db),
        Some(validated.expected_ty),
        &mut table,
    ) {
        Ok(ty) => evaluated.swap_ty(db, ty),
        Err(cause) => evaluated.swap_ty(db, TyId::invalid(db, cause)),
    }
}

pub(crate) fn assumptions_for_body<'db>(
    db: &'db dyn HirAnalysisDb,
    body: Body<'db>,
) -> PredicateListId<'db> {
    let containing_func = match body.scope().parent_item(db) {
        Some(ItemKind::Func(func)) => Some(func),
        Some(ItemKind::Body(parent)) => parent.containing_func(db),
        _ => None,
    };
    if let Some(func) = containing_func {
        let mut preds = collect_func_decl_constraints(db, func.into(), true).instantiate_identity();
        if let Some(ItemKind::Trait(trait_)) = func.scope().parent_item(db) {
            let self_pred =
                TraitInstId::new(db, trait_, trait_.params(db).to_vec(), IndexMap::new());
            let mut merged = preds.list(db).to_vec();
            merged.push(self_pred);
            preds = PredicateListId::new(db, merged);
        }
        return preds.extend_all_bounds(db);
    }

    let mut enclosing = body.scope();
    let mut parent_item = enclosing.parent_item(db);
    while let Some(ItemKind::Body(parent)) = parent_item {
        enclosing = parent.scope();
        parent_item = enclosing.parent_item(db);
    }

    match parent_item {
        Some(ItemKind::Trait(trait_)) => {
            let self_pred =
                TraitInstId::new(db, trait_, trait_.params(db).to_vec(), IndexMap::new());
            PredicateListId::new(db, vec![self_pred]).extend_all_bounds(db)
        }
        Some(ItemKind::ImplTrait(impl_trait)) => collect_constraints(db, impl_trait.into())
            .instantiate_identity()
            .extend_all_bounds(db),
        Some(ItemKind::Impl(impl_)) => collect_constraints(db, impl_.into())
            .instantiate_identity()
            .extend_all_bounds(db),
        _ => PredicateListId::empty_list(db),
    }
}

fn evaluate_const_ty_cycle_initial<'db>(
    _db: &'db dyn HirAnalysisDb,
    const_ty: ConstTyId<'db>,
    _expected_ty: Option<TyId<'db>>,
) -> ConstTyId<'db> {
    const_ty
}

fn evaluate_const_ty_cycle_recover<'db>(
    _db: &'db dyn HirAnalysisDb,
    _value: &ConstTyId<'db>,
    _count: u32,
    _const_ty: ConstTyId<'db>,
    _expected_ty: Option<TyId<'db>>,
) -> salsa::CycleRecoveryAction<ConstTyId<'db>> {
    salsa::CycleRecoveryAction::Iterate
}

pub fn const_ty_from_assoc_const_use<'db>(
    db: &'db dyn HirAnalysisDb,
    assoc: AssocConstUse<'db>,
) -> Option<ConstTyId<'db>> {
    assoc
        .analysis_cx(db, None)
        .map(|cx| const_ty_from_trait_const_in_cx(db, &cx, assoc.inst(), assoc.name()))
        .unwrap_or_else(|| {
            const_ty_from_trait_const(db, assoc.solve_cx(db), assoc.inst(), assoc.name())
        })
}

pub(super) fn const_ty_from_trait_const<'db>(
    db: &'db dyn HirAnalysisDb,
    solve_cx: TraitSolveCx<'db>,
    inst: TraitInstId<'db>,
    name: IdentId<'db>,
) -> Option<ConstTyId<'db>> {
    let cx = AnalysisCx::from_solve_cx(solve_cx);
    const_ty_from_trait_const_in_cx(db, &cx, inst, name)
}

fn const_ty_from_trait_const_in_cx<'db>(
    db: &'db dyn HirAnalysisDb,
    cx: &AnalysisCx<'db>,
    inst: TraitInstId<'db>,
    name: IdentId<'db>,
) -> Option<ConstTyId<'db>> {
    let selection = resolve_assoc_const_selection(db, cx, inst, name)?;
    let source = selection.body.as_ref()?;

    Some(ConstTyId::from_body_with_generic_args_and_cx(
        db,
        source.body,
        Some(selection.declared_ty),
        None,
        source.impl_args.clone(),
        analysis_cx_for_selected_assoc_const_body(db, cx.proof, &selection)
            .map(StoredAnalysisCx::new),
    ))
}

// FIXME: When we add type inference, we need to use the inference engine to
// check the type of the expression instead of this function.
pub(crate) fn check_const_ty<'db>(
    db: &'db dyn HirAnalysisDb,
    const_ty_ty: TyId<'db>,
    expected_ty: Option<TyId<'db>>,
    table: &mut UnificationTable<'db>,
) -> Result<TyId<'db>, InvalidCause<'db>> {
    fn string_capacity(db: &dyn HirAnalysisDb, ty: TyId<'_>) -> Option<usize> {
        let (base, args) = ty.decompose_ty_app(db);
        if !base.is_string(db) || args.len() != 1 {
            return None;
        }
        let TyData::ConstTy(const_ty) = args[0].data(db) else {
            return None;
        };
        let ConstTyData::Evaluated(EvaluatedConstTy::LitInt(value), _) = const_ty.data(db) else {
            return None;
        };
        value.data(db).to_usize()
    }

    let const_ty_ty = normalize_const_tys_for_comparison(db, const_ty_ty);
    if let Some(cause) = const_ty_ty.invalid_cause(db) {
        return Err(cause);
    }

    if const_ty_ty.has_invalid(db) {
        return Err(InvalidCause::Other);
    }

    let Some(expected_ty) = expected_ty.map(|ty| normalize_const_tys_for_comparison(db, ty)) else {
        return Ok(const_ty_ty);
    };

    if let (Some(expected_capacity), Some(given_capacity)) = (
        string_capacity(db, expected_ty),
        string_capacity(db, const_ty_ty),
    ) && given_capacity <= expected_capacity
    {
        return Ok(expected_ty);
    }

    if table.unify(expected_ty, const_ty_ty).is_ok() {
        Ok(expected_ty)
    } else {
        let invalid = InvalidCause::ConstTyMismatch {
            expected: expected_ty,
            given: const_ty_ty,
        };
        Err(invalid)
    }
}

impl<'db> ConstTyId<'db> {
    pub fn ty(self, db: &'db dyn HirAnalysisDb) -> TyId<'db> {
        match self.data(db) {
            ConstTyData::TyVar(_, ty) => *ty,
            ConstTyData::TyParam(_, ty) => *ty,
            ConstTyData::Hole(ty, _) => *ty,
            ConstTyData::Evaluated(_, ty) => *ty,
            ConstTyData::Abstract(_, ty) => *ty,
            ConstTyData::UnEvaluated { ty, .. } => {
                ty.unwrap_or_else(|| TyId::invalid(db, InvalidCause::Other))
            }
        }
    }

    pub(super) fn pretty_print(self, db: &dyn HirAnalysisDb) -> String {
        match &self.data(db) {
            ConstTyData::TyVar(var, _) => var.pretty_print(),
            ConstTyData::TyParam(param, ty) => {
                format!("const {}: {}", param.pretty_print(db), ty.pretty_print(db))
            }
            ConstTyData::Hole(..) => "_".to_string(),
            ConstTyData::Evaluated(resolved, _) => resolved.pretty_print(db),
            ConstTyData::Abstract(expr, _) => expr.pretty_print(db),
            ConstTyData::UnEvaluated {
                body,
                ty,
                const_def,
                generic_args,
                ..
            } => {
                if let Some(const_def) = const_def
                    && let Some(name) = const_def.name(db).to_opt()
                {
                    return format!("const {}", name.data(db));
                }

                let expr = body.expr(db);
                if let Some(rendered) = pretty_print_const_body_expr(
                    db,
                    *body,
                    expr,
                    &generic_const_param_display_map(db, *body, generic_args),
                ) {
                    return rendered;
                }

                let fallback = self.evaluate(db, *ty);
                if fallback != self {
                    return fallback.pretty_print(db);
                }

                "const value".into()
            }
        }
    }

    pub(super) fn evaluate(
        self,
        db: &'db dyn HirAnalysisDb,
        expected_ty: Option<TyId<'db>>,
    ) -> Self {
        evaluate_const_ty(db, self, expected_ty)
    }

    pub(crate) fn evaluate_with_solve_cx(
        self,
        db: &'db dyn HirAnalysisDb,
        expected_ty: Option<TyId<'db>>,
        solve_cx: TraitSolveCx<'db>,
    ) -> Self {
        evaluate_const_ty_with_solve_cx(db, self, expected_ty, Some(solve_cx))
    }

    pub(super) fn from_body(
        db: &'db dyn HirAnalysisDb,
        body: Body<'db>,
        ty: Option<TyId<'db>>,
        const_def: Option<Const<'db>>,
    ) -> Self {
        Self::from_body_with_generic_args_and_cx(db, body, ty, const_def, Vec::new(), None)
    }

    pub(super) fn from_body_with_generic_args_and_cx(
        db: &'db dyn HirAnalysisDb,
        body: Body<'db>,
        ty: Option<TyId<'db>>,
        const_def: Option<Const<'db>>,
        generic_args: Vec<TyId<'db>>,
        unevaluated_cx: Option<StoredAnalysisCx<'db>>,
    ) -> Self {
        Self::from_body_with_generic_args_and_preservation(
            db,
            body,
            ty,
            const_def,
            generic_args,
            false,
            unevaluated_cx,
        )
    }

    pub(super) fn from_body_with_generic_args_and_preservation(
        db: &'db dyn HirAnalysisDb,
        body: Body<'db>,
        ty: Option<TyId<'db>>,
        const_def: Option<Const<'db>>,
        generic_args: Vec<TyId<'db>>,
        preserve_unevaluated: bool,
        unevaluated_cx: Option<StoredAnalysisCx<'db>>,
    ) -> Self {
        let data = ConstTyData::UnEvaluated {
            body,
            ty,
            const_def,
            generic_args,
            preserve_unevaluated,
            unevaluated_cx,
        };
        Self::new(db, data)
    }

    pub fn from_opt_body(db: &'db dyn HirAnalysisDb, body: Partial<Body<'db>>) -> Self {
        match body {
            Partial::Present(body) => Self::from_body(db, body, None, None),
            Partial::Absent => Self::invalid(db, InvalidCause::ParseError),
        }
    }

    pub(crate) fn from_opt_body_in_cx(
        db: &'db dyn HirAnalysisDb,
        body: Partial<Body<'db>>,
        cx: &AnalysisCx<'db>,
    ) -> Self {
        match body {
            Partial::Present(body) => Self::from_body_with_generic_args_and_cx(
                db,
                body,
                None,
                None,
                Vec::new(),
                Some(StoredAnalysisCx::new(*cx)),
            ),
            Partial::Absent => Self::invalid(db, InvalidCause::ParseError),
        }
    }

    pub(super) fn from_opt_body_with_ty_and_generic_args(
        db: &'db dyn HirAnalysisDb,
        body: Partial<Body<'db>>,
        ty: Option<TyId<'db>>,
        generic_args: Vec<TyId<'db>>,
        preserve_unevaluated: bool,
    ) -> Self {
        match body {
            Partial::Present(body) => Self::from_body_with_generic_args_and_preservation(
                db,
                body,
                ty,
                None,
                generic_args,
                preserve_unevaluated,
                None,
            ),
            Partial::Absent => Self::invalid(db, InvalidCause::ParseError),
        }
    }

    pub(super) fn with_ty(self, db: &'db dyn HirAnalysisDb, ty: TyId<'db>) -> Self {
        self.swap_ty(db, ty)
    }

    pub(super) fn invalid(db: &'db dyn HirAnalysisDb, cause: InvalidCause<'db>) -> Self {
        let resolved = EvaluatedConstTy::Invalid;
        let ty = TyId::invalid(db, cause);
        let data = ConstTyData::Evaluated(resolved, ty);
        Self::new(db, data)
    }

    pub fn hole(db: &'db dyn HirAnalysisDb) -> Self {
        Self::hole_with_ty(db, TyId::invalid(db, InvalidCause::Other))
    }

    pub fn hole_with_ty(db: &'db dyn HirAnalysisDb, ty: TyId<'db>) -> Self {
        Self::hole_with_id(db, ty, HoleId::bound_opaque())
    }

    pub fn hole_with_id(db: &'db dyn HirAnalysisDb, ty: TyId<'db>, hole_id: HoleId<'db>) -> Self {
        Self::new(db, ConstTyData::Hole(ty, hole_id))
    }

    pub fn structural_hole(
        db: &'db dyn HirAnalysisDb,
        ty: TyId<'db>,
        origin: StructuralHoleOrigin<'db>,
        local_frame: LocalFrameId<'db>,
    ) -> Self {
        Self::hole_with_id(db, ty, HoleId::structural(db, ty, origin, local_frame))
    }

    pub fn structural_hole_with_app(
        db: &'db dyn HirAnalysisDb,
        ty: TyId<'db>,
        origin: StructuralHoleOrigin<'db>,
        local_frame: LocalFrameId<'db>,
        app_frame: Option<AppFrameId<'db>>,
    ) -> Self {
        Self::hole_with_id(
            db,
            ty,
            HoleId::structural_with_app(db, ty, origin, local_frame, app_frame),
        )
    }

    pub fn bound_callable_hole(
        db: &'db dyn HirAnalysisDb,
        ty: TyId<'db>,
        func: Func<'db>,
        origin: CallableInputLayoutHoleOrigin,
        ordinal: usize,
    ) -> Self {
        Self::hole_with_id(db, ty, HoleId::bound_callable(func, origin, ordinal))
    }

    fn swap_ty(self, db: &'db dyn HirAnalysisDb, ty: TyId<'db>) -> Self {
        let data = match self.data(db) {
            ConstTyData::TyVar(var, _) => ConstTyData::TyVar(var.clone(), ty),
            ConstTyData::TyParam(param, _) => ConstTyData::TyParam(param.clone(), ty),
            ConstTyData::Hole(_, hole_id) => ConstTyData::Hole(
                ty,
                match hole_id {
                    HoleId::Structural(hole_id) => HoleId::Structural(StructuralHoleId::new(
                        db,
                        ty,
                        hole_id.origin(db),
                        hole_id.local_frame(db),
                        hole_id.app_frame(db),
                    )),
                    HoleId::Bound(hole_id) => HoleId::Bound(*hole_id),
                },
            ),
            ConstTyData::Evaluated(evaluated, _) => ConstTyData::Evaluated(evaluated.clone(), ty),
            ConstTyData::Abstract(expr, _) => ConstTyData::Abstract(*expr, ty),
            ConstTyData::UnEvaluated {
                body,
                const_def,
                generic_args,
                preserve_unevaluated,
                unevaluated_cx,
                ..
            } => ConstTyData::UnEvaluated {
                body: *body,
                ty: Some(ty),
                const_def: *const_def,
                generic_args: generic_args.clone(),
                preserve_unevaluated: *preserve_unevaluated,
                unevaluated_cx: *unevaluated_cx,
            },
        };

        Self::new(db, data)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ConstTyData<'db> {
    TyVar(TyVar<'db>, TyId<'db>),
    TyParam(TyParam<'db>, TyId<'db>),
    Hole(TyId<'db>, HoleId<'db>),
    Evaluated(EvaluatedConstTy<'db>, TyId<'db>),
    Abstract(ConstExprId<'db>, TyId<'db>),
    UnEvaluated {
        body: Body<'db>,
        ty: Option<TyId<'db>>,
        const_def: Option<Const<'db>>,
        generic_args: Vec<TyId<'db>>,
        preserve_unevaluated: bool,
        unevaluated_cx: Option<StoredAnalysisCx<'db>>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum EvaluatedConstTy<'db> {
    LitInt(IntegerId<'db>),
    LitBool(bool),
    Unit,
    Tuple(Vec<TyId<'db>>),
    Array(Vec<TyId<'db>>),
    Bytes(Vec<u8>),
    Record(Vec<TyId<'db>>),
    EnumVariant(EnumVariant<'db>),
    Invalid,
}

impl EvaluatedConstTy<'_> {
    pub fn pretty_print(&self, db: &dyn HirAnalysisDb) -> String {
        match self {
            EvaluatedConstTy::LitInt(val) => {
                format!("{}", val.data(db))
            }
            EvaluatedConstTy::LitBool(val) => format!("{val}"),
            EvaluatedConstTy::Unit => "()".to_string(),
            EvaluatedConstTy::Tuple(elems) => {
                let elems = elems
                    .iter()
                    .map(|elem| elem.pretty_print(db).as_str())
                    .collect::<Vec<_>>()
                    .join(", ");
                format!("({elems})")
            }
            EvaluatedConstTy::Array(elems) => {
                let elems = elems
                    .iter()
                    .map(|elem| elem.pretty_print(db).as_str())
                    .collect::<Vec<_>>()
                    .join(", ");
                format!("[{elems}]")
            }
            EvaluatedConstTy::Bytes(bytes) => {
                let bytes = bytes
                    .iter()
                    .map(|b| b.to_string())
                    .collect::<Vec<_>>()
                    .join(", ");
                format!("[{bytes}]")
            }
            EvaluatedConstTy::Record(fields) => {
                let fields = fields
                    .iter()
                    .map(|field| field.pretty_print(db).as_str())
                    .collect::<Vec<_>>()
                    .join(", ");
                format!("{{{fields}}}")
            }
            EvaluatedConstTy::EnumVariant(variant) => {
                let enum_name = variant
                    .enum_
                    .name(db)
                    .to_opt()
                    .map(|n| n.data(db).to_string())
                    .unwrap_or_else(|| "<unknown>".to_string());
                let variant_name = variant.name(db).unwrap_or("<unknown>");
                format!("{enum_name}::{variant_name}")
            }
            EvaluatedConstTy::Invalid => "<invalid>".to_string(),
        }
    }
}
