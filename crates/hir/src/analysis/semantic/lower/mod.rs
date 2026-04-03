use crate::{
    analysis::{
        HirAnalysisDb,
        ty::{
            normalize::normalize_ty,
            ty_check::TypedBody,
            ty_def::{PrimTy, TyBase, TyData},
        },
    },
    hir_def::{
        Body, CallArg, Expr, ExprId,
        expr::{ArithBinOp, BinOp, UnOp},
    },
};

mod body;
mod effects;
mod pattern;
mod place;

pub use body::lower_to_smir;
pub use effects::{
    effect_param_site, owner_effect_bindings, resolved_provider_binding_for_owner_effect,
    same_owner_effect_binding,
};

pub(crate) fn expr_lowers_to_semantic_call<'db>(
    db: &'db dyn HirAnalysisDb,
    typed_body: &TypedBody<'db>,
    body: Body<'db>,
    expr_id: ExprId,
    expr: &Expr<'db>,
) -> bool {
    let Some(callable) = typed_body.callable_expr(expr_id) else {
        return false;
    };

    match expr {
        Expr::Call(_, args) => !is_code_region_intrinsic_call(db, typed_body, callable, args),
        Expr::MethodCall(..) => true,
        Expr::Un(inner, op) if !matches!(op, UnOp::Mut | UnOp::Ref) => {
            !supports_direct_operator_lowering(db, typed_body, body, *inner)
        }
        Expr::Bin(_, _, BinOp::Index) | Expr::Bin(_, _, BinOp::Arith(ArithBinOp::Range)) => false,
        Expr::Bin(lhs, _, _) | Expr::AugAssign(lhs, _, _) => {
            !supports_direct_operator_lowering(db, typed_body, body, *lhs)
        }
        _ => false,
    }
}

fn supports_direct_operator_lowering<'db>(
    db: &'db dyn HirAnalysisDb,
    typed_body: &TypedBody<'db>,
    body: Body<'db>,
    expr: ExprId,
) -> bool {
    let mut ty = normalize_ty(
        db,
        typed_body.expr_ty(db, expr),
        body.scope(),
        typed_body.assumptions(),
    );
    while let Some((_, inner)) = ty.as_capability(db) {
        ty = normalize_ty(db, inner, body.scope(), typed_body.assumptions());
    }

    match ty.base_ty(db).data(db) {
        TyData::TyBase(TyBase::Prim(prim)) => !matches!(
            prim,
            PrimTy::Array
                | PrimTy::Tuple(_)
                | PrimTy::Ptr
                | PrimTy::View
                | PrimTy::BorrowMut
                | PrimTy::BorrowRef
        ),
        TyData::TyBase(TyBase::Contract(_)) => true,
        _ => false,
    }
}

fn is_code_region_intrinsic_call<'db>(
    db: &'db dyn HirAnalysisDb,
    typed_body: &TypedBody<'db>,
    callable: &crate::analysis::ty::ty_check::Callable<'db>,
    args: &[CallArg<'db>],
) -> bool {
    let [arg] = args else {
        return false;
    };
    let crate::hir_def::CallableDef::Func(func) = callable.callable_def() else {
        return false;
    };
    let Some(name) = func.name(db).to_opt() else {
        return false;
    };
    matches!(
        name.data(db).as_str(),
        "code_region_offset" | "code_region_len"
    ) && typed_body.code_region_ref(arg.expr).is_some()
}
