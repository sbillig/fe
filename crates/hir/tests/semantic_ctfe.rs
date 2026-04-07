use std::collections::VecDeque;

use fe_hir::test_db::HirAnalysisTestDb;
use fe_hir::{
    analysis::{
        semantic::{
            SConst, SExpr, SStmtKind, SemConstId, SemConstValue, canonicalize_semantic_consts,
            get_or_build_semantic_instance, identity_semantic_instance_key,
            reify_runtime_const_for_ty,
        },
        ty::ty_check::BodyOwner,
    },
    hir_def::{ItemKind, Partial},
};

#[test]
fn canonicalize_folds_const_calls_into_nested_aggregate_consts() {
    let mut db = HirAnalysisTestDb::default();
    let file = db.new_stand_alone(
        "semantic_ctfe.fe".into(),
        r#"
const fn values() -> [u256; 4] {
    [1, 2, 3, 4]
}

fn wrap() {
    let args: ([u256; 4],) = (values(),)
}
"#,
    );
    let (top_mod, _) = db.top_mod(file);
    let func = top_mod
        .all_funcs(&db)
        .iter()
        .find(|func| matches!(func.name(&db), Partial::Present(name) if name.data(&db) == "wrap"))
        .expect("expected wrap function");

    let semantic = get_or_build_semantic_instance(
        &db,
        identity_semantic_instance_key(&db, BodyOwner::Func(*func)),
    );
    let body = canonicalize_semantic_consts(&db, semantic);

    let found = body
        .blocks
        .iter()
        .flat_map(|block| block.stmts.iter())
        .any(|stmt| {
            let SStmtKind::Assign {
                expr: SExpr::Const(SConst::Value(value)),
                ..
            } = &stmt.kind
            else {
                return false;
            };
            let SemConstValue::Tuple { elems, .. } = value.value(&db) else {
                return false;
            };
            let [array] = elems.as_ref() else {
                return false;
            };
            matches!(
                array.value(&db),
                SemConstValue::Array { elems, .. } if elems.len() == 4
            )
        });

    assert!(
        found,
        "expected canonicalization to fold const call into tuple(array) semantic const"
    );
}

fn owner_name(db: &HirAnalysisTestDb, owner: BodyOwner<'_>) -> String {
    match owner {
        BodyOwner::Func(func) => match func.name(db) {
            Partial::Present(name) => name.data(db).to_string(),
            Partial::Absent => "<fn>".to_string(),
        },
        BodyOwner::Const(const_) => match const_.name(db) {
            Partial::Present(name) => name.data(db).to_string(),
            Partial::Absent => "<const>".to_string(),
        },
        BodyOwner::AnonConstBody { .. } => "<anon const>".to_string(),
        BodyOwner::ContractInit { contract } => match contract.name(db) {
            Partial::Present(name) => format!("{}::__init__", name.data(db)),
            Partial::Absent => "<contract>::__init__".to_string(),
        },
        BodyOwner::ContractRecvArm {
            contract,
            recv_idx,
            arm_idx,
        } => match contract.name(db) {
            Partial::Present(name) => format!("{}::recv[{recv_idx}][{arm_idx}]", name.data(db)),
            Partial::Absent => format!("<contract>::recv[{recv_idx}][{arm_idx}]"),
        },
    }
}

#[test]
fn contract_init_fixed_array_arg_fixture_has_no_type_level_semantic_consts() {
    let mut db = HirAnalysisTestDb::default();
    let file = db.new_stand_alone(
        "semantic_ctfe.fe".into(),
        include_str!("../../fe/tests/fixtures/fe_test/contract_init_fixed_array_arg.fe"),
    );
    let (top_mod, _) = db.top_mod(file);
    let mut pending = VecDeque::new();

    for item in top_mod.all_items(&db) {
        match item {
            ItemKind::Func(func) => pending.push_back(get_or_build_semantic_instance(
                &db,
                identity_semantic_instance_key(&db, BodyOwner::Func(*func)),
            )),
            ItemKind::Contract(contract) => {
                pending.push_back(get_or_build_semantic_instance(
                    &db,
                    identity_semantic_instance_key(
                        &db,
                        BodyOwner::ContractInit {
                            contract: *contract,
                        },
                    ),
                ));
                for (recv_idx, recv) in contract.recvs(&db).data(&db).iter().enumerate() {
                    for arm_idx in 0..recv.arms.data(&db).len() {
                        pending.push_back(get_or_build_semantic_instance(
                            &db,
                            identity_semantic_instance_key(
                                &db,
                                BodyOwner::ContractRecvArm {
                                    contract: *contract,
                                    recv_idx: recv_idx as u32,
                                    arm_idx: arm_idx as u32,
                                },
                            ),
                        ));
                    }
                }
            }
            ItemKind::Const(_)
            | ItemKind::Mod(_)
            | ItemKind::Struct(_)
            | ItemKind::Enum(_)
            | ItemKind::Trait(_)
            | ItemKind::Impl(_)
            | ItemKind::ImplTrait(_)
            | ItemKind::TypeAlias(_)
            | ItemKind::Use(_)
            | ItemKind::TopMod(_)
            | ItemKind::Body(_) => {}
        }
    }

    let mut seen = rustc_hash::FxHashSet::default();
    let mut offenders = Vec::new();
    while let Some(instance) = pending.pop_front() {
        if !seen.insert(instance.key(&db)) {
            continue;
        }
        let body = canonicalize_semantic_consts(&db, instance);
        for stmt in body.blocks.iter().flat_map(|block| block.stmts.iter()) {
            if let SStmtKind::Assign {
                expr: SExpr::Const(SConst::Value(value)),
                ..
            } = &stmt.kind
                && matches!(value.value(&db), SemConstValue::TypeLevel { .. })
            {
                offenders.push(format!(
                    "{} args={:?} {:?} ty={} {:?}",
                    owner_name(&db, instance.key(&db).owner(&db)),
                    instance
                        .key(&db)
                        .subst(&db)
                        .generic_args(&db)
                        .iter()
                        .map(|ty| ty.pretty_print(&db))
                        .collect::<Vec<_>>(),
                    stmt.origin,
                    match value.value(&db) {
                        SemConstValue::TypeLevel { const_ty, .. } => {
                            const_ty.pretty_print(&db).to_string()
                        }
                        _ => "<non-typelevel>".to_string(),
                    },
                    value.value(&db)
                ));
            }
        }
        for callee in instance.callees(&db) {
            pending.push_back(get_or_build_semantic_instance(&db, callee.key));
        }
    }

    assert!(
        offenders.is_empty(),
        "unexpected type-level semantic consts:\n{}",
        offenders.join("\n")
    );
}

#[test]
fn canonicalize_does_not_fold_abstract_checked_runtime_calls() {
    let mut db = HirAnalysisTestDb::default();
    let file = db.new_stand_alone(
        "semantic_ctfe.fe".into(),
        r#"
fn test_add_overflow_u8() {
    let x: u8 = 255
    let y: u8 = x + 1
}
"#,
    );
    let (top_mod, _) = db.top_mod(file);
    let func = top_mod
        .all_funcs(&db)
        .iter()
        .find(|func| {
            matches!(func.name(&db), Partial::Present(name) if name.data(&db) == "test_add_overflow_u8")
        })
        .expect("expected test_add_overflow_u8 function");

    let semantic = get_or_build_semantic_instance(
        &db,
        identity_semantic_instance_key(&db, BodyOwner::Func(*func)),
    );
    let body = canonicalize_semantic_consts(&db, semantic);

    assert!(
        body.blocks
            .iter()
            .flat_map(|block| block.stmts.iter())
            .any(|stmt| {
                matches!(
                    &stmt.kind,
                    SStmtKind::Assign {
                        expr: SExpr::Call { .. } | SExpr::Binary { .. },
                        ..
                    }
                )
            }),
        "checked runtime arithmetic should remain executable after semantic const canonicalization:\n{body:#?}"
    );
}

fn sem_const_has_type_level_leaf(db: &HirAnalysisTestDb, value: SemConstId<'_>) -> bool {
    match value.value(db) {
        SemConstValue::Unit | SemConstValue::Scalar { .. } => false,
        SemConstValue::TypeLevel { .. } => true,
        SemConstValue::Tuple { elems, .. } | SemConstValue::Array { elems, .. } => elems
            .iter()
            .copied()
            .any(|elem| sem_const_has_type_level_leaf(db, elem)),
        SemConstValue::Struct { fields, .. } | SemConstValue::Enum { fields, .. } => fields
            .iter()
            .copied()
            .any(|field| sem_const_has_type_level_leaf(db, field)),
    }
}

fn sem_const_has_abstract_scalar_leaf(db: &HirAnalysisTestDb, value: SemConstId<'_>) -> bool {
    match value.value(db) {
        SemConstValue::Unit | SemConstValue::TypeLevel { .. } => false,
        SemConstValue::Scalar { ty, .. } => ty.pretty_print(db) == "{integer}",
        SemConstValue::Tuple { elems, .. } | SemConstValue::Array { elems, .. } => elems
            .iter()
            .copied()
            .any(|elem| sem_const_has_abstract_scalar_leaf(db, elem)),
        SemConstValue::Struct { fields, .. } | SemConstValue::Enum { fields, .. } => fields
            .iter()
            .copied()
            .any(|field| sem_const_has_abstract_scalar_leaf(db, field)),
    }
}

#[test]
fn runtime_const_reification_removes_nested_type_level_leaves() {
    let mut db = HirAnalysisTestDb::default();
    let file = db.new_stand_alone(
        "semantic_ctfe.fe".into(),
        r#"
fn first(xs: ref [u256; 2]) -> u256 {
    xs[0]
}

fn entry() -> u256 {
    first([10, 20])
}
"#,
    );
    let (top_mod, _) = db.top_mod(file);
    let func = top_mod
        .all_funcs(&db)
        .iter()
        .find(|func| matches!(func.name(&db), Partial::Present(name) if name.data(&db) == "entry"))
        .expect("expected entry function");

    let semantic = get_or_build_semantic_instance(
        &db,
        identity_semantic_instance_key(&db, BodyOwner::Func(*func)),
    );
    let body = canonicalize_semantic_consts(&db, semantic);
    let expected_ty = top_mod
        .all_funcs(&db)
        .iter()
        .find(|func| matches!(func.name(&db), Partial::Present(name) if name.data(&db) == "first"))
        .and_then(|func| func.arg_tys(&db).first().copied())
        .map(|ty| ty.instantiate_identity())
        .expect("expected first(xs: ref [u256; 2]) parameter type");
    let (_, args) = expected_ty.decompose_ty_app(&db);
    let array_expected_ty = args.first().copied().expect("ref pointee type");

    let array_const = body
        .blocks
        .iter()
        .flat_map(|block| block.stmts.iter())
        .find_map(|stmt| {
            let SStmtKind::Assign {
                expr: SExpr::Const(SConst::Value(value)),
                ..
            } = &stmt.kind
            else {
                return None;
            };
            matches!(value.value(&db), SemConstValue::Array { .. }).then_some(*value)
        })
        .expect("expected array constant in canonicalized body");

    let reified = reify_runtime_const_for_ty(&db, semantic, array_expected_ty, array_const)
        .expect("array constant should reify");
    assert!(
        !sem_const_has_type_level_leaf(&db, reified),
        "reified runtime const should not contain nested TypeLevel leaves: {:?}",
        reified.value(&db)
    );
    assert!(
        !sem_const_has_abstract_scalar_leaf(&db, reified),
        "reified runtime const should not contain nested abstract scalar leaves: {:?}",
        reified.value(&db)
    );
}
