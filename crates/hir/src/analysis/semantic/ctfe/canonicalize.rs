use crate::analysis::{
    HirAnalysisDb,
    semantic::{
        SConst, SExpr, SStmt, SemConstId, SemConstValue, SemanticBody, array_const, enum_const,
        instance::SemanticInstance, instantiate_with_generic_args, sem_const_from_ty, struct_const,
        tuple_const,
    },
    ty::{
        const_ty::ConstTyData,
        ty_def::{TyData, TyId},
    },
};

use super::eval_const_ref;

#[salsa::tracked]
pub fn canonicalize_semantic_consts<'db>(
    db: &'db dyn HirAnalysisDb,
    instance: SemanticInstance<'db>,
) -> SemanticBody<'db> {
    let mut body = instance.body(db).clone();
    for block in &mut body.blocks {
        for stmt in &mut block.stmts {
            canonicalize_stmt(db, instance, stmt);
        }
    }
    body
}

fn canonicalize_stmt<'db>(
    db: &'db dyn HirAnalysisDb,
    instance: SemanticInstance<'db>,
    stmt: &mut SStmt<'db>,
) {
    match stmt {
        SStmt::Assign { expr, .. } => canonicalize_expr(db, instance, expr),
        SStmt::Store { .. } => {}
    }
}

fn canonicalize_expr<'db>(
    db: &'db dyn HirAnalysisDb,
    instance: SemanticInstance<'db>,
    expr: &mut SExpr<'db>,
) {
    match expr {
        SExpr::Const(SConst::Ref(cref)) => {
            let value = eval_const_ref(db, *cref)
                .unwrap_or_else(|err| panic!("CTFE failed for {cref:?}: {err:?}"));
            *expr = SExpr::Const(SConst::Value(value));
        }
        SExpr::Const(SConst::Value(value)) => {
            *expr = SExpr::Const(SConst::Value(canonicalize_const_value(
                db, instance, *value,
            )));
        }
        SExpr::Use(_)
        | SExpr::Unary { .. }
        | SExpr::Binary { .. }
        | SExpr::Cast { .. }
        | SExpr::AggregateMake { .. }
        | SExpr::EnumMake { .. }
        | SExpr::Field { .. }
        | SExpr::Index { .. }
        | SExpr::Borrow { .. }
        | SExpr::GetEnumTag { .. }
        | SExpr::IsEnumVariant { .. }
        | SExpr::ExtractEnumField { .. }
        | SExpr::CodeRegionOffset { .. }
        | SExpr::CodeRegionLen { .. }
        | SExpr::Call { .. } => {}
    }
}

fn canonicalize_const_value<'db>(
    db: &'db dyn HirAnalysisDb,
    instance: SemanticInstance<'db>,
    value: SemConstId<'db>,
) -> SemConstId<'db> {
    match value.value(db) {
        SemConstValue::Unit | SemConstValue::Scalar { .. } => value,
        SemConstValue::TypeLevel { ty, const_ty } => {
            let instantiated = instantiate_with_generic_args(
                db,
                const_ty,
                instance.key(db).subst(db).generic_args(db),
            );
            let TyData::ConstTy(const_ty) = instantiated.data(db) else {
                return value;
            };
            let mut evaluated = const_ty.evaluate(db, Some(ty));
            if matches!(evaluated.data(db), ConstTyData::Abstract(..)) {
                let instantiated = instantiate_with_generic_args(
                    db,
                    TyId::const_ty(db, evaluated),
                    instance.key(db).subst(db).generic_args(db),
                );
                let TyData::ConstTy(instantiated) = instantiated.data(db) else {
                    unreachable!("instantiating a const ty must yield a const ty");
                };
                evaluated = instantiated.evaluate(db, Some(ty));
            }
            sem_const_from_ty(db, TyId::const_ty(db, evaluated)).unwrap_or(value)
        }
        SemConstValue::Tuple { ty, elems } => tuple_const(
            db,
            ty,
            elems
                .iter()
                .copied()
                .map(|elem| canonicalize_const_value(db, instance, elem))
                .collect::<Vec<_>>()
                .into_boxed_slice(),
        ),
        SemConstValue::Struct { ty, fields } => struct_const(
            db,
            ty,
            fields
                .iter()
                .copied()
                .map(|field| canonicalize_const_value(db, instance, field))
                .collect::<Vec<_>>()
                .into_boxed_slice(),
        ),
        SemConstValue::Array { ty, elems } => array_const(
            db,
            ty,
            elems
                .iter()
                .copied()
                .map(|elem| canonicalize_const_value(db, instance, elem))
                .collect::<Vec<_>>()
                .into_boxed_slice(),
        ),
        SemConstValue::Enum {
            ty,
            variant,
            fields,
        } => enum_const(
            db,
            ty,
            variant,
            fields
                .iter()
                .copied()
                .map(|field| canonicalize_const_value(db, instance, field))
                .collect::<Vec<_>>()
                .into_boxed_slice(),
        ),
    }
}
