use fe_hir::{
    analysis::{
        semantic::{
            CtfeError, GenericSubst, SemConstId, SemConstScalar, SemConstValue,
            SemanticInstanceKey, eval_body_owner_const, get_or_build_semantic_instance,
            identity_semantic_instance_key, instantiate_with_generic_args, reify_runtime_const,
            reify_runtime_const_for_ty, sem_const_ty,
        },
        ty::{
            const_ty::{ConstTyData, ConstTyId, EvaluatedConstTy},
            ty_check::BodyOwner,
            ty_def::{PrimTy, TyBase, TyData, TyId},
        },
    },
    hir_def::{Func, IntegerId, Partial, TopLevelMod},
    test_db::{HirAnalysisTestDb, format_diagnostics},
};
use num_bigint::BigUint;
use num_traits::ToPrimitive;

fn function<'db>(db: &'db HirAnalysisTestDb, module: TopLevelMod<'db>, name: &str) -> Func<'db> {
    module
        .all_funcs(db)
        .iter()
        .copied()
        .find(|func| matches!(func.name(db), Partial::Present(found) if found.data(db) == name))
        .unwrap_or_else(|| panic!("missing function {name}"))
}

fn scalar(db: &HirAnalysisTestDb, value: SemConstId<'_>) -> usize {
    let SemConstValue::Scalar {
        value: SemConstScalar::Int { value },
        ..
    } = value.value(db)
    else {
        panic!("expected integer, got {:?}", value.value(db));
    };
    value.to_usize().expect("small integer")
}

#[test]
fn generic_inherent_const_array_repeats_specialize() {
    let mut db = HirAnalysisTestDb::default();
    let file = db.new_stand_alone(
        "array_repeat.fe".into(),
        r#"
struct Marker<const N: usize> {}
impl<const N: usize> Marker<N> {
    const VALUES: [u8; N] = [7; N]
    const MATRIX: [[u8; N]; 2] = [[9; N]; 2]
    const FIRST: u8 = Self::VALUES[0]
}
const fn first() -> u8 { Marker<1>::FIRST }
const fn last() -> u8 { Marker<3>::VALUES[2] }
const fn nested() -> u8 { Marker<4>::MATRIX[1][3] }
const fn empty() -> [u8; 0] { Marker<0>::VALUES }
const fn empty_nested() -> [[u8; 0]; 2] { Marker<0>::MATRIX }
"#,
    );
    let (module, _) = db.top_mod(file);
    db.assert_no_diags(module);
    for (name, expected) in [("first", 7), ("last", 7), ("nested", 9)] {
        let value =
            eval_body_owner_const(&db, BodyOwner::Func(function(&db, module, name)), vec![])
                .unwrap();
        assert_eq!(scalar(&db, value), expected, "{name}");
    }
    let empty = eval_body_owner_const(&db, BodyOwner::Func(function(&db, module, "empty")), vec![])
        .unwrap();
    assert!(matches!(empty.value(&db), SemConstValue::Array { elems, .. } if elems.is_empty()));
    let nested = eval_body_owner_const(
        &db,
        BodyOwner::Func(function(&db, module, "empty_nested")),
        vec![],
    )
    .unwrap();
    let SemConstValue::Array { elems, .. } = nested.value(&db) else {
        panic!("expected array")
    };
    assert_eq!(elems.len(), 2);
    assert!(elems.iter().all(
        |elem| matches!(elem.value(&db), SemConstValue::Array { elems, .. } if elems.is_empty())
    ));
}

#[test]
fn generic_record_const_array_repeats_specialize() {
    let mut db = HirAnalysisTestDb::default();
    let file = db.new_stand_alone(
        "record_repeat.fe".into(),
        r#"
struct Marker<const N: usize> { values: [u8; N] }
impl<const N: usize> Marker<N> {
    const VALUE: Marker<N> = Marker<N> { values: [9; N] }
}
const fn value() -> u8 { Marker<3>::VALUE.values[2] }
"#,
    );
    let (module, _) = db.top_mod(file);
    db.assert_no_diags(module);
    let value = eval_body_owner_const(&db, BodyOwner::Func(function(&db, module, "value")), vec![])
        .unwrap();
    assert_eq!(scalar(&db, value), 9);
}

#[test]
fn symbolic_repeat_stores_defer_until_specialization() {
    let mut db = HirAnalysisTestDb::default();
    let file = db.new_stand_alone(
        "repeat_stores.fe".into(),
        r#"
struct Marker<const N: usize> { rows: [[u8; N]; 2] }
const fn changed<const N: usize>() -> [u8; N] {
    let mut values = [0; N]
    values[0] = 1
    values
}
const fn changed_record<const N: usize>() -> [[u8; N]; 2] {
    let mut value = Marker<N> { rows: [[7; N]; 2] }
    value.rows[0][1] = 2
    value.rows
}
const fn replaced<const N: usize>() -> [u8; N] {
    let mut values = [0; N]
    values = [3; N]
    values
}
impl<const N: usize> Marker<N> {
    const VALUES: [u8; N] = changed<N>()
    const ROWS: [[u8; N]; 2] = changed_record<N>()
    const REPLACED: [u8; N] = replaced<N>()
}
const fn first() -> u8 { Marker<1>::VALUES[0] }
const fn last() -> u8 { Marker<3>::VALUES[2] }
const fn nested() -> u8 { Marker<3>::ROWS[0][1] }
const fn sibling() -> u8 { Marker<3>::ROWS[1][1] }
const fn replacement() -> u8 { Marker<3>::REPLACED[2] }
const fn empty_replacement() -> [u8; 0] { Marker<0>::REPLACED }
"#,
    );
    let (module, _) = db.top_mod(file);
    db.assert_no_diags(module);
    for (name, expected) in [
        ("first", 1),
        ("last", 0),
        ("nested", 2),
        ("sibling", 7),
        ("replacement", 3),
    ] {
        let value =
            eval_body_owner_const(&db, BodyOwner::Func(function(&db, module, name)), vec![])
                .unwrap();
        assert_eq!(scalar(&db, value), expected, "{name}");
    }
    let empty = eval_body_owner_const(
        &db,
        BodyOwner::Func(function(&db, module, "empty_replacement")),
        vec![],
    )
    .unwrap();
    assert!(matches!(empty.value(&db), SemConstValue::Array { elems, .. } if elems.is_empty()));
}

#[test]
fn symbolic_repeat_stores_preserve_bounds_checks() {
    for (len, index) in [(0, 0), (1, 1)] {
        let mut db = HirAnalysisTestDb::default();
        let file = db.new_stand_alone(
            "repeat_store_bounds.fe".into(),
            &format!(
                r#"
const fn changed<const N: usize>() -> [u8; N] {{
    let mut values = [0; N]
    values[{index}] = 1
    values
}}
struct Marker<const N: usize> {{}}
impl<const N: usize> Marker<N> {{
    const VALUES: [u8; N] = changed<N>()
}}
const fn bad() -> [u8; {len}] {{ changed<{len}>() }}
const BAD: [u8; {len}] = Marker<{len}>::VALUES
"#
            ),
        );
        let (module, _) = db.top_mod(file);
        let diags = db.run_on_top_mod(module);
        let rendered = format_diagnostics(&db, &diags);
        assert!(!diags.is_empty(), "out-of-bounds store must be diagnosed");
        assert!(!rendered.contains("internal"), "{rendered}");
        let mut err =
            eval_body_owner_const(&db, BodyOwner::Func(function(&db, module, "bad")), vec![])
                .unwrap_err();
        while let CtfeError::CalleeError { source, .. } = err {
            err = *source;
        }
        assert!(matches!(err, CtfeError::OutOfBounds { .. }), "{err:?}");
    }
}

#[test]
fn symbolic_repeat_values_reify_after_element_and_length_substitution() {
    let mut db = HirAnalysisTestDb::default();
    let file = db.new_stand_alone(
        "repeat_reification.fe".into(),
        r#"
const fn repeat<const N: usize, const X: u8>() -> [[u8; N]; 2] { [[X; N]; 2] }
const fn project<const N: usize, const X: u8>() -> u8 { [X; N][1] }
const fn checked<const N: usize, const X: u8>() -> [[u8; N]; 2] { [[10 / X; N]; 2] }
const fn store<const N: usize>(_ value: u8) -> [[u8; N]; 2] {
    let mut rows = [[value; N]; 2]
    rows[0][1] = 10
    rows
}
const fn changed<const N: usize, const X: u8>() -> [[u8; N]; 2] { store<N>(X) }
const fn checked_changed<const N: usize, const X: u8>() -> [[u8; N]; 2] { store<N>(10 / X) }
struct Rows<const N: usize> { values: [[u8; N]; 2] }
const fn store_record<const N: usize>(_ value: u8) -> Rows<N> {
    let mut rows = Rows<N> { values: [[value; N]; 2] }
    rows.values[0][1] = 10
    rows
}
const fn record<const N: usize, const X: u8>() -> Rows<N> { store_record<N>(X) }
"#,
    );
    let (module, _) = db.top_mod(file);
    db.assert_no_diags(module);
    for name in [
        "repeat",
        "project",
        "checked",
        "changed",
        "checked_changed",
        "record",
    ] {
        let owner = BodyOwner::Func(function(&db, module, name));
        let value = eval_body_owner_const(&db, owner, vec![]).unwrap();
        let identity = identity_semantic_instance_key(&db, owner);
        assert!(
            reify_runtime_const(&db, get_or_build_semantic_instance(&db, identity), value)
                .is_none()
        );
        for (len, element) in [(0, 0), (0, 5), (1, 5), (2, 5), (4, 9)] {
            let args =
                [(PrimTy::Usize, len), (PrimTy::U8, element)].map(|(prim, value): (_, u32)| {
                    let ty = TyId::new(&db, TyData::TyBase(TyBase::Prim(prim)));
                    TyId::new(
                        &db,
                        TyData::ConstTy(ConstTyId::new(
                            &db,
                            ConstTyData::Evaluated(
                                EvaluatedConstTy::LitInt(IntegerId::new(&db, BigUint::from(value))),
                                ty,
                            ),
                        )),
                    )
                });
            let key = SemanticInstanceKey::new(
                &db,
                owner,
                GenericSubst::new(&db, args.to_vec()),
                identity.effect_providers(&db),
                identity.impl_env(&db),
            );
            let expected = instantiate_with_generic_args(&db, sem_const_ty(&db, value), &args);
            let instance = get_or_build_semantic_instance(&db, key);
            let reified = reify_runtime_const_for_ty(&db, instance, expected, value);
            assert_eq!(
                reify_runtime_const(&db, instance, value),
                reified,
                "{name} inferred result type"
            );
            if (matches!(name, "project" | "changed" | "checked_changed" | "record") && len < 2)
                || (matches!(name, "checked" | "checked_changed") && element == 0)
            {
                assert!(
                    reified.is_none(),
                    "{name} must retain its failure after substitution: len={len}, element={element}"
                );
                continue;
            }
            let reified = reified
                .unwrap_or_else(|| panic!("{name} failed to reify: len={len}, element={element}"));
            assert_eq!(sem_const_ty(&db, reified), expected, "{name} result type");
            assert!(
                !sem_const_ty(&db, reified).has_param(&db),
                "{name} result type"
            );
            let reified = if name == "record" {
                let SemConstValue::Struct { fields, .. } = reified.value(&db) else {
                    panic!("expected record")
                };
                fields[0]
            } else {
                reified
            };
            let expected = if matches!(name, "checked" | "checked_changed") {
                10 / element
            } else {
                element
            };
            if name == "project" {
                assert_eq!(scalar(&db, reified), element as usize);
            } else {
                let SemConstValue::Array { elems, .. } = reified.value(&db) else {
                    panic!("expected outer array")
                };
                assert_eq!(elems.len(), 2);
                assert!(!sem_const_ty(&db, reified).has_param(&db));
                for (row_index, row) in elems.iter().enumerate() {
                    assert!(!sem_const_ty(&db, *row).has_param(&db));
                    let SemConstValue::Array { elems, .. } = row.value(&db) else {
                        panic!("expected inner array")
                    };
                    assert_eq!(elems.len(), len as usize);
                    for (index, elem) in elems.iter().enumerate() {
                        let expected = if matches!(name, "changed" | "checked_changed" | "record")
                            && row_index == 0
                            && index == 1
                        {
                            10
                        } else {
                            expected as usize
                        };
                        assert_eq!(scalar(&db, *elem), expected);
                    }
                }
            }
        }
    }
}

#[test]
fn symbolic_repeat_element_access_preserves_bounds_checks() {
    let mut db = HirAnalysisTestDb::default();
    let file = db.new_stand_alone(
        "repeat_bounds.fe".into(),
        r#"
struct Marker<const N: usize> {}
impl<const N: usize> Marker<N> {
    const FIRST: u8 = [7; N][0]
}
const fn empty() -> u8 { Marker<0>::FIRST }
const BAD: u8 = Marker<0>::FIRST
"#,
    );
    let (module, _) = db.top_mod(file);
    let result =
        eval_body_owner_const(&db, BodyOwner::Func(function(&db, module, "empty")), vec![]);
    assert!(
        result.is_err(),
        "empty repeated array must not produce an element: {result:?}"
    );
    assert!(
        !db.run_on_top_mod(module).is_empty(),
        "empty array indexing must be diagnosed"
    );
}

#[test]
fn symbolic_repeat_still_evaluates_its_element() {
    for source in [
        "fn runtime() -> u8 { 7 }\nstruct Marker<const N: usize> {}\nimpl<const N: usize> Marker<N> { const VALUE: [u8; N] = [runtime(); N] }",
        "struct Marker<const N: usize> {}\nimpl<const N: usize> Marker<N> { const VALUE: [u8; N] = [1 / 0; N] }",
        "fn runtime() -> u8 { 7 }\nconst BAD: [u8; 0] = [runtime(); 0]",
    ] {
        let mut db = HirAnalysisTestDb::default();
        let file = db.new_stand_alone("invalid_repeat.fe".into(), source);
        let (module, _) = db.top_mod(file);
        let diags = db.run_on_top_mod(module);
        assert!(
            !diags.is_empty(),
            "invalid repeat element was accepted: {source}"
        );
        let rendered = format_diagnostics(&db, &diags);
        assert!(!rendered.contains("internal"), "{rendered}");
    }
}

#[test]
fn trait_associated_lengths_and_elements_stay_symbolic() {
    let mut db = HirAnalysisTestDb::default();
    let file = db.new_stand_alone(
        "trait_repeat.fe".into(),
        r#"
trait Shape {
    const LEN: usize = 1
    const VALUE: u8 = 2
}
struct Large {}
impl Shape for Large {
    const LEN: usize = 4
    const VALUE: u8 = 9
}
struct Buffer<T> {}
impl<T: Shape> Buffer<T> {
    const VALUES: [u8; T::LEN] = [T::VALUE; T::LEN]
}
const fn value() -> u8 { Buffer<Large>::VALUES[3] }
"#,
    );
    let (module, _) = db.top_mod(file);
    db.assert_no_diags(module);
    let value = eval_body_owner_const(&db, BodyOwner::Func(function(&db, module, "value")), vec![])
        .unwrap();
    assert_eq!(scalar(&db, value), 9);
}
