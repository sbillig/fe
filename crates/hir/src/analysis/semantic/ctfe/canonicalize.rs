use std::collections::VecDeque;

use cranelift_entity::EntityRef;

use crate::analysis::{
    HirAnalysisDb,
    semantic::{
        SBlock, SBlockId, SConst, SExpr, SStmt, SStmtKind, STerminatorKind, SemConstId,
        SemConstValue, SemanticBody, array_const, enum_const, instance::SemanticInstance,
        instantiate_with_generic_args, reify_runtime_const_for_ty, sem_const_from_ty, struct_const,
        tuple_const,
    },
    ty::{
        const_ty::{ConstTyData, evaluate_type_level_int_const_expr},
        ty_def::{TyData, TyId},
    },
};

use super::{eval_const_ref, machine::try_eval_expr_to_const};

type LocalConstMap<'db> = Vec<Option<SemConstId<'db>>>;

#[salsa::tracked]
pub fn canonicalize_semantic_consts<'db>(
    db: &'db dyn HirAnalysisDb,
    instance: SemanticInstance<'db>,
) -> SemanticBody<'db> {
    let original = instance.body(db).clone();
    let mut body = original.clone();
    if body.blocks.is_empty() {
        return body;
    }

    let mut incoming = vec![None; body.blocks.len()];
    incoming[0] = Some(vec![None; body.locals.len()]);
    let mut pending = VecDeque::from([SBlockId::from_u32(0)]);

    while let Some(bb) = pending.pop_front() {
        let Some(mut locals) = incoming[bb.index()].clone() else {
            continue;
        };
        body.blocks[bb.index()] = canonicalize_block(
            db,
            instance,
            &original.blocks[bb.index()],
            &mut locals,
            &original,
        );
        for succ in block_successors(&original.blocks[bb.index()].terminator.kind) {
            if merge_local_consts(&mut incoming[succ.index()], &locals) {
                pending.push_back(succ);
            }
        }
    }

    let mut unknown_locals = vec![None; body.locals.len()];
    for (idx, state) in incoming.iter().enumerate() {
        if state.is_none() {
            body.blocks[idx] = canonicalize_block(
                db,
                instance,
                &original.blocks[idx],
                &mut unknown_locals,
                &original,
            );
            unknown_locals.fill(None);
        }
    }

    body
}

fn canonicalize_block<'db>(
    db: &'db dyn HirAnalysisDb,
    instance: SemanticInstance<'db>,
    block: &SBlock<'db>,
    locals: &mut LocalConstMap<'db>,
    body: &SemanticBody<'db>,
) -> SBlock<'db> {
    SBlock {
        stmts: block
            .stmts
            .iter()
            .map(|stmt| canonicalize_stmt(db, instance, stmt, locals, body))
            .collect(),
        terminator: block.terminator.clone(),
    }
}

fn canonicalize_stmt<'db>(
    db: &'db dyn HirAnalysisDb,
    instance: SemanticInstance<'db>,
    stmt: &SStmt<'db>,
    locals: &mut LocalConstMap<'db>,
    body: &SemanticBody<'db>,
) -> SStmt<'db> {
    let kind = match &stmt.kind {
        SStmtKind::Assign { dst, expr } => {
            let (expr, value) =
                canonicalize_expr(db, instance, expr, body.locals[dst.index()].ty, locals);
            locals[dst.index()] = value;
            SStmtKind::Assign { dst: *dst, expr }
        }
        SStmtKind::Store { dst, src } => {
            locals[dst.local.index()] = None;
            SStmtKind::Store {
                dst: dst.clone(),
                src: *src,
            }
        }
    };
    SStmt {
        origin: stmt.origin,
        kind,
    }
}

fn canonicalize_expr<'db>(
    db: &'db dyn HirAnalysisDb,
    instance: SemanticInstance<'db>,
    expr: &SExpr<'db>,
    result_ty: TyId<'db>,
    locals: &LocalConstMap<'db>,
) -> (SExpr<'db>, Option<SemConstId<'db>>) {
    if let SExpr::Const(SConst::Ref(cref)) = expr {
        let value = eval_const_ref(db, *cref)
            .unwrap_or_else(|err| panic!("CTFE failed for {cref:?}: {err:?}"));
        let value = canonicalize_const_value(db, instance, value);
        let runtime = reify_runtime_const_for_ty(db, instance, result_ty, value);
        let value = runtime.unwrap_or(value);
        return (SExpr::Const(SConst::Value(value)), runtime);
    }

    if let Some(value) = try_eval_expr_to_const(db, instance, result_ty, expr, locals, synthetic())
        && !matches!(value.value(db), SemConstValue::TypeLevel { .. })
    {
        let value = canonicalize_const_value(db, instance, value);
        if let Some(value) = reify_runtime_const_for_ty(db, instance, result_ty, value) {
            return (SExpr::Const(SConst::Value(value)), Some(value));
        }
    }

    match expr {
        SExpr::Const(SConst::Value(value)) => {
            let value = canonicalize_const_value(db, instance, *value);
            let runtime = reify_runtime_const_for_ty(db, instance, result_ty, value);
            let value = runtime.unwrap_or(value);
            (SExpr::Const(SConst::Value(value)), runtime)
        }
        _ => (expr.clone(), None),
    }
}

fn merge_local_consts<'db>(
    current: &mut Option<LocalConstMap<'db>>,
    incoming: &LocalConstMap<'db>,
) -> bool {
    match current {
        None => {
            *current = Some(incoming.clone());
            true
        }
        Some(current) => {
            let mut changed = false;
            for (slot, incoming) in current.iter_mut().zip(incoming.iter().copied()) {
                let merged = if *slot == incoming { incoming } else { None };
                if *slot != merged {
                    *slot = merged;
                    changed = true;
                }
            }
            changed
        }
    }
}

fn block_successors<'db>(term: &STerminatorKind<'db>) -> Vec<SBlockId> {
    match term {
        STerminatorKind::Goto(bb) => vec![*bb],
        STerminatorKind::Branch {
            then_bb, else_bb, ..
        } => vec![*then_bb, *else_bb],
        STerminatorKind::MatchEnum { cases, default, .. } => {
            let mut succs = cases.iter().map(|(_, bb)| *bb).collect::<Vec<_>>();
            if let Some(default) = default {
                succs.push(*default);
            }
            succs
        }
        STerminatorKind::Return(None | Some(_)) => Vec::new(),
    }
}

fn synthetic<'db>() -> crate::analysis::semantic::SemOrigin<'db> {
    crate::analysis::semantic::SemOrigin::Synthetic
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
            if let ConstTyData::Abstract(expr, expected_ty) = evaluated.data(db)
                && let Some(concrete) = evaluate_type_level_int_const_expr(db, *expr, *expected_ty)
            {
                evaluated = concrete;
            }
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
                if let ConstTyData::Abstract(expr, expected_ty) = evaluated.data(db)
                    && let Some(concrete) =
                        evaluate_type_level_int_const_expr(db, *expr, *expected_ty)
                {
                    evaluated = concrete;
                }
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
