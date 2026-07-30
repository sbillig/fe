use std::collections::VecDeque;

use cranelift_entity::EntityRef;
use fe_hir::test_db::{HirAnalysisTestDb, format_diagnostics};
use fe_hir::{
    analysis::{
        semantic::{
            BorrowInput, BorrowResult, BorrowTransform, FieldIndex, LayoutBackingProjection,
            NBorrowRoot, NExpr, NLocalOrigin, NSPlace, NSPlaceRoot, NSStmtKind,
            NormalizedBindingLowering, ReadMode, SStmtKind, SemanticBorrowDiagKind,
            SemanticInstance, SemanticLocalKind, check_semantic_borrows, check_semantic_noesc,
            collect_semantic_borrow_diagnostic_vouchers, get_or_build_semantic_instance,
            identity_semantic_instance_key, normalize_semantic_body, semantic_borrow_summary,
        },
        ty::{
            ProviderAddressSpace,
            ty_check::{BodyOwner, LocalBinding},
            ty_def::{BorrowKind, TyData},
        },
    },
    hir_def::{ItemKind, Partial},
    projection::{IndexSource, Projection},
};

fn borrow_diags(src: &str) -> String {
    let mut db = HirAnalysisTestDb::default();
    let file = db.new_stand_alone("semantic_borrowck.fe".into(), src);
    let (top_mod, _) = db.top_mod(file);
    format_diagnostics(
        &db,
        &collect_semantic_borrow_diagnostic_vouchers(&db, top_mod),
    )
}

fn contract_init_instance<'db>(
    db: &'db HirAnalysisTestDb,
    top_mod: fe_hir::hir_def::TopLevelMod<'db>,
    contract_name: &str,
) -> SemanticInstance<'db> {
    top_mod
        .all_items(db)
        .iter()
        .find_map(|item| match item {
            ItemKind::Contract(contract)
                if contract
                    .name(db)
                    .to_opt()
                    .is_some_and(|name| name.data(db) == contract_name) =>
            {
                Some(get_or_build_semantic_instance(
                    db,
                    identity_semantic_instance_key(
                        db,
                        BodyOwner::ContractInit {
                            contract: *contract,
                        },
                    ),
                ))
            }
            _ => None,
        })
        .unwrap_or_else(|| panic!("missing contract init `{contract_name}`"))
}

fn mixed_returned_borrow_provenance_src() -> &'static str {
    r#"
struct Ledger {
    b: u256,
}

impl Ledger {
    fn pick_mixed(mut self, cond: bool, value: mut u256) -> mut u256 {
        if cond {
            value
        } else {
            mut self.b
        }
    }
}

fn add(by: u256) -> u256 uses (value: mut u256) {
    value += by
    value
}

pub contract Mixed {
    mut ledger: Ledger

    init() uses (mut ledger) {
        let mut local: u256 = 0
        let target = ledger.pick_mixed(cond: true, value: mut local)
        with (target) {
            add(by: 1)
        }
    }
}
"#
}

#[test]
fn closure_returned_capture_preserves_borrow_provenance() {
    let src = r#"
fn probe() -> u256 {
    let mut x = 0
    let borrowed = mut x
    let pass = |_ unit: own ()| -> mut u256 { borrowed }
    let returned = pass.call(())
    let other = mut x
    other = 1
    returned
}
"#;
    let diags = borrow_diags(src);
    assert!(diags.contains("borrow conflict in `fn probe`"), "{diags}");
    assert!(diags.contains("cannot mutably borrow"), "{diags}");
}

#[test]
fn closure_returned_capture_does_not_retain_unrelated_capture_loans() {
    let diags = borrow_diags(
        r#"
fn probe() {
    let mut x = 0
    let mut y = 0
    let borrowed_x = mut x
    let borrowed_y = mut y
    let pass = |_ unit: own ()| -> mut u256 {
        borrowed_y = 1
        borrowed_x
    }
    let returned = pass.call(())
    let other_y = mut y
    other_y = 2
    returned = 3
}
"#,
    );
    assert!(diags.is_empty(), "{diags}");
}

#[test]
fn repeated_closure_calls_cannot_duplicate_returned_mut_capture() {
    let diags = borrow_diags(
        r#"
fn probe() {
    let mut x = 0
    let borrowed = mut x
    let pass = |_ unit: own ()| -> mut u256 { borrowed }
    let first = pass.call(())
    let second = pass.call(())
    first = 1
    second = 2
}
"#,
    );
    assert!(diags.contains("borrow conflict in `fn probe`"), "{diags}");
}

#[test]
fn returned_ref_capture_remains_live_in_the_caller() {
    let diags = borrow_diags(
        r#"
fn probe() -> u256 {
    let mut x = 0
    let borrowed = ref x
    let pass = |_ unit: own ()| -> ref u256 { borrowed }
    let returned = pass.call(())
    let other = mut x
    other = 1
    returned
}
"#,
    );
    assert!(diags.contains("borrow conflict in `fn probe`"), "{diags}");
}

#[test]
fn repeated_closure_calls_may_duplicate_returned_ref_capture() {
    let diags = borrow_diags(
        r#"
fn probe() -> u256 {
    let x = 1
    let borrowed = ref x
    let pass = |_ unit: own ()| -> ref u256 { borrowed }
    let first = pass.call(())
    let second = pass.call(())
    first + second
}
"#,
    );
    assert!(diags.is_empty(), "{diags}");
}

#[test]
fn repeated_function_calls_cannot_duplicate_returned_mut_borrow() {
    let diags = borrow_diags(
        r#"
fn pass(_ value: mut u256) -> mut u256 {
    value
}

fn probe() {
    let mut x = 0
    let borrowed = mut x
    let first = pass(borrowed)
    let second = pass(borrowed)
    first = 1
    second = 2
}
"#,
    );
    assert!(diags.contains("borrow conflict in `fn probe`"), "{diags}");
}

#[test]
fn returned_mut_reborrow_releases_to_its_parent_after_last_use() {
    let diags = borrow_diags(
        r#"
fn pass(_ value: mut u256) -> mut u256 {
    value
}

fn probe() {
    let mut x = 0
    let borrowed = mut x
    let first = pass(borrowed)
    first = 1
    let second = pass(borrowed)
    second = 2
}
"#,
    );
    assert!(diags.is_empty(), "{diags}");
}

#[test]
fn opaque_trait_call_cannot_duplicate_returned_mut_borrow() {
    let diags = borrow_diags(
        r#"
trait ValueMut {
    fn value_mut(mut self) -> mut u256
}

fn probe<T: ValueMut>(mut value: T) {
    let first = value.value_mut()
    let second = value.value_mut()
    first = 1
    second = 2
}
"#,
    );
    assert!(diags.contains("borrow conflict in `fn probe`"), "{diags}");
}

#[test]
fn opaque_mut_borrow_result_does_not_alias_immutable_scalar_args() {
    let diags = borrow_diags(
        r#"
trait ValueMut {
    fn value_mut(mut self, flag: bool) -> mut u256
}

fn probe<T: ValueMut>(mut value: T, mut flag: bool) {
    let returned = value.value_mut(flag)
    let other = mut flag
    other = true
    returned = 1
}
"#,
    );
    assert!(diags.is_empty(), "{diags}");
}

#[test]
fn opaque_trait_call_tracks_borrows_nested_in_aggregate_results() {
    let diags = borrow_diags(
        r#"
struct Borrowed {
    value: mut u256,
}

trait BorrowValue {
    fn borrow_value(mut self) -> Borrowed
}

fn probe<T: BorrowValue>(mut value: T) {
    let first = value.borrow_value()
    let second = value.borrow_value()
    first.value = 1
    second.value = 2
}
"#,
    );
    assert!(diags.contains("borrow conflict in `fn probe`"), "{diags}");
}

#[test]
fn returned_aggregate_tracks_fresh_mut_borrow_from_view_param() {
    let diags = borrow_diags(
        r#"
struct Owner {
    value: u256,
}

struct Borrowed {
    value: mut u256,
}

fn borrow_value(mut owner: Owner) -> Borrowed {
    Borrowed { value: mut owner.value }
}

fn probe() {
    let mut owner = Owner { value: 0 }
    let borrowed = borrow_value(owner)
    let other = mut owner.value
    other = 1
    borrowed.value = 2
}
"#,
    );
    assert!(diags.contains("borrow conflict in `fn probe`"), "{diags}");
}

#[test]
fn fresh_borrow_of_plain_field_in_borrow_holding_aggregate_is_tracked() {
    let diags = borrow_diags(
        r#"
struct Mixed {
    owner: u256,
    held: mut u256,
}

fn borrow_owner(mut mixed: Mixed) -> mut u256 {
    mut mixed.owner
}

fn probe() {
    let mut held_owner = 0
    let mixed = Mixed { owner: 0, held: mut held_owner }
    let borrowed = borrow_owner(mixed)
    let other = mut mixed.owner
    other = 1
    borrowed = 2
}
"#,
    );
    assert!(diags.contains("borrow conflict in `fn probe`"), "{diags}");
}

#[test]
fn fresh_borrow_of_whole_borrow_holding_aggregate_tracks_its_storage() {
    let diags = borrow_diags(
        r#"
struct Mixed {
    owner: u256,
    held: mut u256,
}

fn borrow_mixed(mut mixed: Mixed) -> mut Mixed {
    mut mixed
}

fn probe() {
    let mut held_owner = 0
    let mixed = Mixed { owner: 0, held: mut held_owner }
    let borrowed = borrow_mixed(mixed)
    let other = mut mixed.owner
    other = 1
    borrowed.owner = 2
}
"#,
    );
    assert!(diags.contains("borrow conflict in `fn probe`"), "{diags}");
}

#[test]
fn reading_one_aggregate_borrow_field_does_not_retain_sibling_borrows() {
    let diags = borrow_diags(
        r#"
struct Pair {
    left: mut u256,
    right: mut u256,
}

fn probe() {
    let mut x = 0
    let mut y = 0
    let pair = Pair { left: mut x, right: mut y }
    let left = pair.left
    let other_y = mut y
    other_y = 1
    left = 2
}
"#,
    );
    assert!(diags.is_empty(), "{diags}");
}

#[test]
fn overwriting_aggregate_field_releases_its_old_held_borrows() {
    let diags = borrow_diags(
        r#"
struct Borrowed {
    value: mut u256,
}

struct Wrapper {
    borrowed: Borrowed,
}

fn probe() {
    let mut x = 0
    let mut y = 0
    let mut wrapper = Wrapper { borrowed: Borrowed { value: mut x } }
    wrapper.borrowed = Borrowed { value: mut y }
    let other_x = mut x
    other_x = 1
    wrapper.borrowed.value = 2
}
"#,
    );
    assert!(diags.is_empty(), "{diags}");
}

#[test]
fn writing_through_borrow_field_keeps_its_loan_active() {
    let diags = borrow_diags(
        r#"
struct Borrowed {
    value: mut u256,
}

fn probe() {
    let mut x = 0
    let borrowed = Borrowed { value: mut x }
    borrowed.value = 1
    let other = mut x
    other = 2
    borrowed.value = 3
}
"#,
    );
    assert!(diags.contains("borrow conflict in `fn probe`"), "{diags}");
}

#[test]
fn returned_enum_tracks_only_the_active_borrow_variant() {
    let diags = borrow_diags(
        r#"
enum Choice {
    Left(mut u256),
    Right(mut u256),
}

fn borrow_left(mut value: u256) -> Choice {
    Choice::Left(mut value)
}

fn probe() {
    let mut x = 0
    let choice = borrow_left(x)
    let other = mut x
    other = 1
    match choice {
        Choice::Left(value) => value = 2,
        Choice::Right(value) => value = 3,
    }
}
"#,
    );
    assert!(diags.contains("borrow conflict in `fn probe`"), "{diags}");
    assert!(!diags.contains("internal borrow checker error"), "{diags}");
}

#[test]
fn nested_inactive_enum_borrow_variant_does_not_require_a_source() {
    let diags = borrow_diags(
        r#"
enum Choice {
    Empty,
    Borrowed(mut u256),
}

struct Wrapper {
    choice: Choice,
}

fn empty() -> Wrapper {
    Wrapper { choice: Choice::Empty }
}

fn probe() {
    let wrapper = empty()
    match wrapper.choice {
        Choice::Empty => (),
        Choice::Borrowed(value) => value = 1,
    }
}
"#,
    );
    assert!(diags.is_empty(), "{diags}");
}

#[test]
fn recursive_aggregate_borrow_summary_converges() {
    let diags = borrow_diags(
        r#"
struct Owner {
    value: u256,
}

struct Borrowed {
    value: mut u256,
}

fn borrow_value(mut owner: Owner, recurse: bool) -> Borrowed {
    if recurse {
        borrow_value(owner, recurse: false)
    } else {
        Borrowed { value: mut owner.value }
    }
}

fn probe() {
    let mut owner = Owner { value: 0 }
    let borrowed = borrow_value(owner, recurse: true)
    let other = mut owner.value
    other = 1
    borrowed.value = 2
}
"#,
    );
    assert!(diags.contains("borrow conflict in `fn probe`"), "{diags}");
}

#[test]
fn mutually_recursive_aggregate_borrow_summaries_converge() {
    let diags = borrow_diags(
        r#"
struct Owner {
    value: u256,
}

struct Borrowed {
    value: mut u256,
}

fn borrow_left(mut owner: Owner, recurse: bool) -> Borrowed {
    if recurse {
        borrow_right(owner)
    } else {
        Borrowed { value: mut owner.value }
    }
}

fn borrow_right(mut owner: Owner) -> Borrowed {
    borrow_left(owner, recurse: false)
}

fn probe() {
    let mut owner = Owner { value: 0 }
    let borrowed = borrow_left(owner, recurse: true)
    let other = mut owner.value
    other = 1
    borrowed.value = 2
}
"#,
    );
    assert!(diags.contains("borrow conflict in `fn probe`"), "{diags}");
}

#[test]
fn reading_one_array_borrow_element_does_not_retain_siblings() {
    let diags = borrow_diags(
        r#"
fn probe() {
    let mut x = 0
    let mut y = 0
    let values = [mut x, mut y]
    let first = values[0]
    let other_y = mut y
    other_y = 1
    first = 2
}
"#,
    );
    assert!(diags.is_empty(), "{diags}");
}

#[test]
fn overwriting_constant_array_element_releases_only_that_elements_borrows() {
    let diags = borrow_diags(
        r#"
struct Borrowed {
    value: mut u256,
}

fn probe() {
    let mut x = 0
    let mut y = 0
    let mut z = 0
    let mut values = [
        Borrowed { value: mut x },
        Borrowed { value: mut y },
    ]
    values[0] = Borrowed { value: mut z }
    let other_x = mut x
    other_x = 1
    values[1].value = 2
}
"#,
    );
    assert!(diags.is_empty(), "{diags}");
}

#[test]
fn dynamic_array_element_overwrite_retains_possibly_untouched_borrows() {
    let diags = borrow_diags(
        r#"
struct Borrowed {
    value: mut u256,
}

fn probe(index: usize) {
    let mut x = 0
    let mut y = 0
    let mut z = 0
    let mut values = [
        Borrowed { value: mut x },
        Borrowed { value: mut y },
    ]
    values[index] = Borrowed { value: mut z }
    let other_x = mut x
    other_x = 1
    values[1].value = 2
}
"#,
    );
    assert!(diags.contains("borrow conflict in `fn probe`"), "{diags}");
}

#[test]
fn returned_dynamic_index_borrow_is_tracked_conservatively() {
    let diags = borrow_diags(
        r#"
fn get(mut values: [u256; 2], index: usize) -> mut u256 {
    mut values[index]
}

fn probe(index: usize) {
    let mut values = [0, 1]
    let returned = get(values, index)
    let other = mut values[0]
    other = 1
    returned = 2
}
"#,
    );
    assert!(diags.contains("borrow conflict in `fn probe`"), "{diags}");
    assert!(
        !diags.contains("return borrows with dynamic indices are not supported"),
        "{diags}"
    );
}

#[test]
fn returned_dynamic_index_borrow_releases_after_its_last_use() {
    let diags = borrow_diags(
        r#"
fn get(mut values: [u256; 2], index: usize) -> mut u256 {
    mut values[index]
}

fn probe(index: usize) {
    let mut values = [0, 1]
    let returned = get(values, index)
    returned = 1
    let other = mut values[0]
    other = 2
}
"#,
    );
    assert!(diags.is_empty(), "{diags}");
}

#[test]
fn returned_constant_index_borrow_does_not_alias_sibling_elements() {
    let diags = borrow_diags(
        r#"
fn first(mut values: [u256; 2]) -> mut u256 {
    mut values[0]
}

fn probe() {
    let mut values = [0, 1]
    let returned = first(values)
    let other = mut values[1]
    other = 1
    returned = 2
}
"#,
    );
    assert!(diags.is_empty(), "{diags}");
}

#[test]
fn returned_aggregate_preserves_each_borrow_fields_kind() {
    let diags = borrow_diags(
        r#"
struct Borrows {
    immutable: ref u256,
    mutable: mut u256,
}

fn borrow_both(left: u256, mut right: u256) -> Borrows {
    Borrows { immutable: ref left, mutable: mut right }
}

fn probe() {
    let left = 0
    let mut right = 0
    let borrowed = borrow_both(left, right)
    let second_ref = ref left
    borrowed.mutable = 1
    let sum = borrowed.immutable + second_ref
}
"#,
    );
    assert!(diags.is_empty(), "{diags}");
}

#[test]
fn nested_returned_closure_preserves_captured_borrow_provenance() {
    let diags = borrow_diags(
        r#"
fn probe() -> u256 {
    let mut x = 0
    let borrowed = mut x
    let outer = |_ unit: own ()| {
        |_ inner_unit: own ()| -> mut u256 { borrowed }
    }
    let inner = outer.call(())
    let returned = inner.call(())
    let other = mut x
    other = 1
    returned
}
"#,
    );
    assert!(diags.contains("borrow conflict in `fn probe`"), "{diags}");
}

#[test]
fn returned_nested_closure_does_not_retain_unrelated_outer_capture() {
    let diags = borrow_diags(
        r#"
fn probe() {
    let mut x = 0
    let mut y = 0
    let borrowed_x = mut x
    let borrowed_y = mut y
    let outer = |_ unit: own ()| {
        borrowed_y = 1
        let inner = |_ inner_unit: own ()| -> mut u256 { borrowed_x }
        inner
    }
    let inner = outer.call(())
    let other_y = mut y
    other_y = 2
    let returned = inner.call(())
    returned = 3
}
"#,
    );
    assert!(diags.is_empty(), "{diags}");
}

fn for_each_fixture_instance(
    src: &str,
    mut f: impl FnMut(&HirAnalysisTestDb, SemanticInstance<'_>),
) {
    let mut db = HirAnalysisTestDb::default();
    let file = db.new_stand_alone("semantic_borrowck.fe".into(), src);
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
            | ItemKind::StaticAssert(_)
            | ItemKind::Use(_)
            | ItemKind::TopMod(_)
            | ItemKind::Body(_) => {}
        }
    }

    let mut seen = rustc_hash::FxHashSet::default();
    while let Some(instance) = pending.pop_front() {
        if !seen.insert(instance.key(&db)) {
            continue;
        }
        f(&db, instance);
        for callee in instance.callees(&db) {
            pending.push_back(get_or_build_semantic_instance(&db, callee.key));
        }
    }
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
        BodyOwner::Closure { .. } => "<closure>".to_string(),
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

fn normalized_func_body<'db>(
    db: &'db HirAnalysisTestDb,
    top_mod: fe_hir::hir_def::TopLevelMod<'db>,
    func_name: &str,
) -> fe_hir::analysis::semantic::NormalizedSemanticBody<'db> {
    let instance = top_mod
        .all_items(db)
        .iter()
        .find_map(|item| match item {
            ItemKind::Func(func)
                if func
                    .name(db)
                    .to_opt()
                    .is_some_and(|name| name.data(db) == func_name) =>
            {
                Some(get_or_build_semantic_instance(
                    db,
                    identity_semantic_instance_key(db, BodyOwner::Func(*func)),
                ))
            }
            _ => None,
        })
        .unwrap_or_else(|| panic!("missing function `{func_name}`"));
    normalize_semantic_body(db, instance).expect("normalized body")
}

#[test]
fn self_referential_param_layout_backing_sources_use_the_param_root() {
    let mut db = HirAnalysisTestDb::default();
    let file = db.new_stand_alone(
        "semantic_borrowck.fe".into(),
        r#"
struct Pair {
    x: u256,
    y: u256,
}

fn rebuild(mut _ value: own Pair) -> Pair {
    let x = value.x
    value = Pair { x, y: value.y }
    value
}
"#,
    );
    let (top_mod, _) = db.top_mod(file);
    let normalized = normalized_func_body(&db, top_mod, "rebuild");
    let param = normalized
        .locals
        .iter()
        .find(|local| matches!(local.source, Some(LocalBinding::Param { idx: 0, .. })))
        .expect("missing value parameter");
    let root = param
        .lowering
        .root()
        .expect("mutable owned aggregate parameter must have a root");

    assert!(!param.layout_backing_sources().is_empty());
    assert!(
        param
            .layout_backing_sources()
            .iter()
            .all(|source| source.source.root.borrow_root() == Some(root))
    );
}

#[test]
fn branch_return_borrow_summary_flows_through_empty_entry_blocks() {
    let mut db = HirAnalysisTestDb::default();
    let file = db.new_stand_alone(
        "semantic_borrowck.fe".into(),
        r#"
struct Ledger {
    a: u256,
    b: u256,
    c: u256,
}

impl Ledger {
    fn pick(mut self, _ pick_c: bool) -> mut u256 {
        if pick_c {
            mut self.c
        } else {
            mut self.a
        }
    }
}
"#,
    );
    let (top_mod, _) = db.top_mod(file);
    let instance = top_mod
        .all_items(&db)
        .iter()
        .find_map(|item| match item {
            ItemKind::Func(func)
                if func
                    .name(&db)
                    .to_opt()
                    .is_some_and(|name| name.data(&db) == "pick") =>
            {
                Some(get_or_build_semantic_instance(
                    &db,
                    identity_semantic_instance_key(&db, BodyOwner::Func(*func)),
                ))
            }
            _ => None,
        })
        .expect("pick instance");
    let summary = semantic_borrow_summary(&db, instance)
        .expect("borrow summary")
        .expect("borrow-returning function should produce a summary");
    assert_eq!(summary.len(), 2, "unexpected summary: {summary:#?}");
    assert!(summary.iter().any(|transform| {
        transform.input
            == BorrowInput::Place {
                param: 0,
                projection: vec![LayoutBackingProjection::Field(FieldIndex(2))],
            }
    }));
    assert!(summary.iter().any(|transform| {
        transform.input
            == BorrowInput::Place {
                param: 0,
                projection: vec![LayoutBackingProjection::Field(FieldIndex(0))],
            }
    }));
    check_semantic_borrows(&db, instance).expect("borrowck should accept branch-returned borrow");
}

#[test]
fn forwarded_memory_borrow_param_keeps_incoming_loan_targets() {
    let mut db = HirAnalysisTestDb::default();
    let file = db.new_stand_alone(
        "semantic_borrowck.fe".into(),
        r#"
struct Holder {
    tag: u256,
}

impl Holder {
    fn forward(mut self, _ value: mut u256) -> mut u256 {
        value
    }
}
"#,
    );
    let (top_mod, _) = db.top_mod(file);
    let instance = top_mod
        .all_items(&db)
        .iter()
        .find_map(|item| match item {
            ItemKind::Func(func)
                if func
                    .name(&db)
                    .to_opt()
                    .is_some_and(|name| name.data(&db) == "forward") =>
            {
                Some(get_or_build_semantic_instance(
                    &db,
                    identity_semantic_instance_key(&db, BodyOwner::Func(*func)),
                ))
            }
            _ => None,
        })
        .expect("forward instance");
    let summary = semantic_borrow_summary(&db, instance)
        .expect("borrow summary")
        .expect("forward should produce a borrow summary");
    assert_eq!(
        summary,
        vec![BorrowTransform {
            result: BorrowResult {
                kind: BorrowKind::Mut,
                projection: Vec::new(),
            },
            input: BorrowInput::Place {
                param: 1,
                projection: Vec::new(),
            },
        }]
    );
    check_semantic_borrows(&db, instance).expect("borrowck should accept forwarded borrows");
}

#[test]
fn writing_through_borrow_field_preserves_returned_handle_source() {
    let mut db = HirAnalysisTestDb::default();
    let file = db.new_stand_alone(
        "semantic_borrowck.fe".into(),
        r#"
struct Borrowed {
    value: mut u256,
}

fn stash(mut borrowed: own Borrowed, value: mut u256) -> Borrowed {
    borrowed.value = value
    borrowed
}
"#,
    );
    let (top_mod, _) = db.top_mod(file);
    let instance = top_mod
        .all_items(&db)
        .iter()
        .find_map(|item| match item {
            ItemKind::Func(func)
                if func
                    .name(&db)
                    .to_opt()
                    .is_some_and(|name| name.data(&db) == "stash") =>
            {
                Some(get_or_build_semantic_instance(
                    &db,
                    identity_semantic_instance_key(&db, BodyOwner::Func(*func)),
                ))
            }
            _ => None,
        })
        .expect("stash instance");
    let summary = semantic_borrow_summary(&db, instance)
        .expect("borrow summary")
        .expect("stash should produce a borrow summary");
    assert_eq!(
        summary,
        vec![BorrowTransform {
            result: BorrowResult {
                kind: BorrowKind::Mut,
                projection: vec![LayoutBackingProjection::Field(FieldIndex(0))],
            },
            input: BorrowInput::Place {
                param: 0,
                projection: vec![LayoutBackingProjection::Field(FieldIndex(0))],
            },
        }]
    );
}

#[test]
fn contract_field_mut_borrow_matrix_fixture_borrowchecks() {
    for_each_fixture_instance(
        include_str!("../../fe/tests/fixtures/fe_test/contract_field_mut_borrow_matrix.fe"),
        |db, instance| {
            if let Err(diag) = check_semantic_borrows(db, instance) {
                panic!(
                    "borrowck failed for {} ({:?}): {diag:#?}",
                    owner_name(db, instance.key(db).owner(db)),
                    instance.key(db),
                );
            }
        },
    );
}

#[test]
fn returned_storage_borrow_effect_args_are_finalized_in_normalized_body() {
    let mut saw_storage_add_effect = false;
    for_each_fixture_instance(
        include_str!("../../fe/tests/fixtures/fe_test/contract_field_mut_borrow_matrix.fe"),
        |db, instance| {
            let normalized = normalize_semantic_body(db, instance).expect("normalized body");
            for stmt in normalized
                .blocks
                .iter()
                .flat_map(|block| block.stmts.iter())
            {
                let NSStmtKind::Assign {
                    expr:
                        NExpr::Call {
                            callee,
                            effect_args,
                            ..
                        },
                    ..
                } = &stmt.kind
                else {
                    continue;
                };
                let BodyOwner::Func(func) = callee.key.owner(db) else {
                    continue;
                };
                if func
                    .name(db)
                    .to_opt()
                    .is_some_and(|name| name.data(db) == "add")
                    && effect_args
                        .iter()
                        .any(|arg| arg.provider == Some(ProviderAddressSpace::Storage))
                {
                    saw_storage_add_effect = true;
                }
            }
        },
    );
    assert!(
        saw_storage_add_effect,
        "expected storage provider on normalized add effect arg"
    );
}

#[test]
fn mixed_returned_borrow_provenance_is_rejected_before_runtime_lowering() {
    let diags = borrow_diags(mixed_returned_borrow_provenance_src());

    assert!(
        diags.contains("provider provenance conflict in `fn Mixed::__init__`"),
        "{diags:?}"
    );
    assert!(
        diags.contains("effect argument may come from multiple address spaces"),
        "{diags:?}"
    );
}

#[test]
fn mixed_returned_borrow_provenance_poison_normalization() {
    let mut db = HirAnalysisTestDb::default();
    let file = db.new_stand_alone(
        "semantic_borrowck.fe".into(),
        mixed_returned_borrow_provenance_src(),
    );
    let (top_mod, _) = db.top_mod(file);
    let instance = contract_init_instance(&db, top_mod, "Mixed");

    let err = normalize_semantic_body(&db, instance)
        .expect_err("mixed provider provenance must poison normalization");
    assert_eq!(err.kind, SemanticBorrowDiagKind::ProviderProvenanceConflict);
    assert_eq!(
        err.primary.message,
        "effect argument may come from multiple address spaces: memory, storage"
    );
}

#[test]
fn mixed_returned_borrow_provenance_poison_noesc() {
    let mut db = HirAnalysisTestDb::default();
    let file = db.new_stand_alone(
        "semantic_borrowck.fe".into(),
        mixed_returned_borrow_provenance_src(),
    );
    let (top_mod, _) = db.top_mod(file);
    let instance = contract_init_instance(&db, top_mod, "Mixed");

    let err = check_semantic_noesc(&db, instance)
        .expect_err("mixed provider provenance must poison noesc");
    assert_eq!(
        err.message,
        "provider provenance conflict in `fn Mixed::__init__`"
    );
    assert_eq!(
        err.sub_diagnostics[0].message,
        "effect argument may come from multiple address spaces: memory, storage"
    );
}

#[test]
fn mixed_returned_borrow_provenance_collects_one_diagnostic() {
    let mut db = HirAnalysisTestDb::default();
    let file = db.new_stand_alone(
        "semantic_borrowck.fe".into(),
        mixed_returned_borrow_provenance_src(),
    );
    let (top_mod, _) = db.top_mod(file);
    let diags = collect_semantic_borrow_diagnostic_vouchers(&db, top_mod);
    assert_eq!(
        diags.len(),
        1,
        "unexpected diagnostics: {:#?}",
        borrow_diags(mixed_returned_borrow_provenance_src())
    );
    let rendered = format_diagnostics(&db, &diags);
    assert!(
        rendered.contains("provider provenance conflict in `fn Mixed::__init__`"),
        "{rendered:?}"
    );
}

#[test]
fn reports_mut_borrow_conflict() {
    let diags = borrow_diags(
        r#"
fn bad() {
    let mut x: u256 = 0
    let p: mut u256 = mut x
    let q: mut u256 = mut x
    q = 1
    p = 2
}
"#,
    );

    assert!(diags.contains("borrow conflict in `fn bad`"), "{diags:?}");
    assert!(
        diags.contains("cannot mutably borrow") || diags.contains("mutable borrow"),
        "{diags:?}",
    );
}

#[test]
fn writing_borrowed_place_while_ref_is_live_is_rejected() {
    let diags = borrow_diags(
        r#"
fn bad() -> u256 {
    let mut x = 0
    let borrowed = ref x
    x = 1
    borrowed
}
"#,
    );

    assert!(diags.contains("borrow conflict in `fn bad`"), "{diags:?}");
}

#[test]
fn reading_borrowed_place_while_mut_borrow_is_live_is_rejected() {
    let diags = borrow_diags(
        r#"
fn bad() -> u256 {
    let mut x = 0
    let borrowed = mut x
    let value = x
    borrowed = 1
    value
}
"#,
    );

    assert!(diags.contains("borrow conflict in `fn bad`"), "{diags:?}");
}

#[test]
fn access_to_disjoint_field_while_mut_borrow_is_live_is_allowed() {
    let diags = borrow_diags(
        r#"
struct Pair {
    left: u256,
    right: u256,
}

fn valid() -> u256 {
    let mut pair = Pair { left: 0, right: 1 }
    let borrowed = mut pair.left
    let value = pair.right
    borrowed = 2
    value
}
"#,
    );

    assert!(diags.is_empty(), "{diags:?}");
}

#[test]
fn reading_parent_while_field_mut_borrow_is_live_is_rejected() {
    let diags = borrow_diags(
        r#"
struct Pair {
    left: u256,
    right: u256,
}

fn consume(_ pair: Pair) -> u256 {
    pair.right
}

fn bad() -> u256 {
    let mut pair = Pair { left: 0, right: 1 }
    let borrowed = mut pair.left
    let value = consume(pair)
    borrowed = 2
    value
}
"#,
    );

    assert!(diags.contains("borrow conflict in `fn bad`"), "{diags:?}");
}

#[test]
fn reading_and_writing_through_active_mut_borrow_is_allowed() {
    let diags = borrow_diags(
        r#"
fn valid() -> u256 {
    let mut x = 0
    let borrowed = mut x
    borrowed = 1
    borrowed
}
"#,
    );

    assert!(diags.is_empty(), "{diags:?}");
}

#[test]
fn writing_borrowed_place_after_last_borrow_use_is_allowed() {
    let diags = borrow_diags(
        r#"
fn valid() -> u256 {
    let mut x = 0
    let borrowed = ref x
    let value = borrowed + 0
    x = 1
    value
}
"#,
    );

    assert!(diags.is_empty(), "{diags:?}");
}

#[test]
fn moving_borrowed_value_into_aggregate_is_rejected() {
    let diags = borrow_diags(
        r#"
struct Boxed {
    value: u256,
}

struct Holder {
    boxed: Boxed,
}

fn bad() -> u256 {
    let boxed = Boxed { value: 1 }
    let borrowed = ref boxed
    let holder = Holder { boxed }
    borrowed.value + holder.boxed.value
}
"#,
    );

    assert!(
        diags.contains("cannot move out of a value while it is borrowed"),
        "{diags:?}"
    );
}

#[test]
fn store_source_cannot_reuse_moved_value() {
    let diags = borrow_diags(
        r#"
struct Boxed {
    value: u256,
}

struct Holder {
    boxed: Boxed,
}

fn bad() -> u256 {
    let mut holder = Holder { boxed: Boxed { value: 0 } }
    let boxed = Boxed { value: 1 }
    let moved = boxed
    holder.boxed = boxed
    moved.value
}
"#,
    );

    assert!(
        diags.contains("cannot use a value after it was moved"),
        "{diags:?}"
    );
}

#[test]
fn aggregate_cannot_move_same_value_twice_in_one_expression() {
    let diags = borrow_diags(
        r#"
struct Boxed {
    value: u256,
}

struct Pair {
    left: Boxed,
    right: Boxed,
}

fn bad(_ boxed: own Boxed) -> Pair {
    Pair { left: boxed, right: boxed }
}
"#,
    );

    assert!(
        diags.contains("cannot use a value after it was moved"),
        "{diags:?}"
    );
}

#[test]
fn call_cannot_move_same_value_into_two_owned_params() {
    let diags = borrow_diags(
        r#"
struct Boxed {
    value: u256,
}

fn consume(_ left: own Boxed, _ right: own Boxed) {}

fn bad(_ boxed: own Boxed) {
    consume(boxed, boxed)
}
"#,
    );

    assert!(
        diags.contains("cannot use a value after it was moved"),
        "{diags:?}"
    );
}

#[test]
fn call_cannot_alias_one_mut_handle_across_two_params() {
    let diags = borrow_diags(
        r#"
fn write_both(_ left: mut u256, _ right: mut u256) {
    left = 1
    right = 2
}

fn bad() {
    let mut x = 0
    let borrowed = mut x
    write_both(borrowed, borrowed)
}
"#,
    );

    assert!(diags.contains("borrow conflict in `fn bad`"), "{diags:?}");
}

#[test]
fn call_may_alias_one_ref_handle_across_two_params() {
    let diags = borrow_diags(
        r#"
fn sum(_ left: ref u256, _ right: ref u256) -> u256 {
    left + right
}

fn valid() -> u256 {
    let x = 21
    let borrowed = ref x
    sum(borrowed, borrowed)
}
"#,
    );

    assert!(diags.is_empty(), "{diags:?}");
}

#[test]
fn call_may_mutably_access_disjoint_fields() {
    let diags = borrow_diags(
        r#"
struct Pair {
    left: u256,
    right: u256,
}

fn write_both(_ left: mut u256, _ right: mut u256) {
    left = 1
    right = 2
}

fn valid() {
    let mut pair = Pair { left: 0, right: 0 }
    let left = mut pair.left
    let right = mut pair.right
    write_both(left, right)
}
"#,
    );

    assert!(diags.is_empty(), "{diags:?}");
}

#[test]
fn call_cannot_mix_parent_mut_and_child_ref_access() {
    let diags = borrow_diags(
        r#"
fn read_and_write(_ writer: mut u256, reader: ref u256) {
    writer = reader
}

fn bad() {
    let mut value = 0
    let writer = mut value
    let reader = ref writer
    read_and_write(writer, reader)
}
"#,
    );

    assert!(diags.contains("borrow conflict in `fn bad`"), "{diags:?}");
}

#[test]
fn call_cannot_alias_mut_borrows_nested_in_aggregate_args() {
    let diags = borrow_diags(
        r#"
struct Borrowed {
    value: mut u256,
}

fn write_both(left: Borrowed, right: Borrowed) {
    left.value = 1
    right.value = 2
}

fn bad() {
    let mut value = 0
    let borrowed = mut value
    let left = Borrowed { value: borrowed }
    let right = Borrowed { value: borrowed }
    write_both(left, right)
}
"#,
    );

    assert!(
        diags.contains("call arguments require conflicting access to the same place"),
        "{diags:?}"
    );
}

#[test]
fn call_may_alias_ref_borrows_nested_in_aggregate_args() {
    let diags = borrow_diags(
        r#"
struct Borrowed {
    value: ref u256,
}

fn sum(left: Borrowed, right: Borrowed) -> u256 {
    left.value + right.value
}

fn valid() -> u256 {
    let value = 21
    let borrowed = ref value
    let left = Borrowed { value: borrowed }
    let right = Borrowed { value: borrowed }
    sum(left, right)
}
"#,
    );

    assert!(diags.is_empty(), "{diags:?}");
}

#[test]
fn call_cannot_alias_mut_borrows_within_one_aggregate_arg() {
    let diags = borrow_diags(
        r#"
struct BorrowedPair {
    left: mut u256,
    right: mut u256,
}

fn write_both(pair: BorrowedPair) {
    pair.left = 1
    pair.right = 2
}

fn bad() {
    let mut value = 0
    let borrowed = mut value
    let pair = BorrowedPair { left: borrowed, right: borrowed }
    write_both(pair)
}
"#,
    );

    assert!(
        diags.contains("call arguments require conflicting access to the same place"),
        "{diags:?}"
    );
}

#[test]
fn call_cannot_alias_regular_and_effect_mut_access() {
    let diags = borrow_diags(
        r#"
fn write_both(regular: mut u256) uses (effect: mut u256) {
    regular = 1
    effect = 2
}

fn bad() {
    let mut value = 0
    let borrowed = mut value
    with (borrowed) {
        write_both(borrowed)
    }
}
"#,
    );

    assert!(diags.contains("borrow conflict in `fn bad`"), "{diags:?}");
}

#[test]
fn call_may_copy_value_before_mut_effect_access() {
    let diags = borrow_diags(
        r#"
fn write(_ snapshot: u256) uses (target: mut u256) {
    target = snapshot
}

fn valid() {
    let mut value: u256 = 1
    with (value) {
        write(value)
    }
}
"#,
    );

    assert!(diags.is_empty(), "{diags:?}");
}

#[test]
fn mutable_receiver_reservation_allows_nested_shared_receiver_call() {
    let diags = borrow_diags(
        r#"
struct Cell {
    value: u256,
}

impl Cell {
    fn read(self) -> u256 {
        self.value
    }

    fn write(mut self, value: u256) {
        self.value = value
    }
}

fn valid() {
    let mut cell = Cell { value: 1 }
    cell.write(value: cell.read())
}
"#,
    );

    assert!(diags.is_empty(), "{diags:?}");
}

#[test]
fn mutable_receiver_activation_rejects_lingering_shared_borrow() {
    let diags = borrow_diags(
        r#"
struct Cell {
    value: u256,
}

impl Cell {
    fn read(self) -> u256 {
        self.value
    }

    fn write(mut self, value: u256) {
        self.value = value
    }
}

fn bad() -> u256 {
    let mut cell = Cell { value: 1 }
    let borrowed = ref cell.value
    cell.write(value: cell.read())
    borrowed
}
"#,
    );

    assert!(diags.contains("borrow conflict in `fn bad`"), "{diags:?}");
}

#[test]
fn mutable_receiver_reservation_allows_nested_mut_receiver_call() {
    let diags = borrow_diags(
        r#"
struct Cell {
    value: u256,
}

impl Cell {
    fn take(mut self) -> u256 {
        self.value += 1
        self.value
    }

    fn write(mut self, value: u256) {
        self.value = value
    }
}

fn bad() {
    let mut cell = Cell { value: 1 }
    cell.write(value: cell.take())
}
"#,
    );

    assert!(diags.is_empty(), "{diags:?}");
}

#[test]
fn mutable_receiver_activation_rejects_borrow_returned_by_nested_call() {
    let diags = borrow_diags(
        r#"
struct Cell {
    value: u256,
}

impl Cell {
    fn borrow(mut self) -> mut u256 {
        mut self.value
    }

    fn write(mut self, value: mut u256) {
        self.value = value
    }
}

fn bad() {
    let mut cell = Cell { value: 1 }
    cell.write(value: cell.borrow())
}
"#,
    );

    assert!(diags.contains("borrow conflict in `fn bad`"), "{diags:?}");
}

#[test]
fn call_cannot_alias_nested_mut_borrows_across_regular_and_effect_args() {
    let diags = borrow_diags(
        r#"
struct Borrowed {
    value: mut u256,
}

fn write_both(regular: Borrowed) uses (effect: Borrowed) {
    regular.value = 1
    effect.value = 2
}

fn bad() {
    let mut value = 0
    let borrowed = mut value
    let regular = Borrowed { value: borrowed }
    let effect = Borrowed { value: borrowed }
    with (effect) {
        write_both(regular)
    }
}
"#,
    );

    assert!(
        diags.contains("call arguments require conflicting access to the same place"),
        "{diags:?}"
    );
}

#[test]
fn call_may_alias_nested_ref_borrows_across_regular_and_effect_args() {
    let diags = borrow_diags(
        r#"
struct Borrowed {
    value: ref u256,
}

fn sum(regular: Borrowed) -> u256 uses (effect: Borrowed) {
    regular.value + effect.value
}

fn valid() -> u256 {
    let value = 21
    let borrowed = ref value
    let regular = Borrowed { value: borrowed }
    let effect = Borrowed { value: borrowed }
    with (effect) {
        sum(regular)
    }
}
"#,
    );

    assert!(diags.is_empty(), "{diags:?}");
}

#[test]
fn mutable_effect_call_cannot_write_borrowed_place() {
    let diags = borrow_diags(
        r#"
fn set() uses (value: mut u256) {
    value = 1
}

fn bad() -> u256 {
    let mut x = 0
    let borrowed = ref x
    with (x) {
        set()
    }
    borrowed
}
"#,
    );

    assert!(diags.contains("borrow conflict in `fn bad`"), "{diags:?}");
}

#[test]
fn mutable_effect_call_through_active_mut_borrow_is_allowed() {
    let diags = borrow_diags(
        r#"
fn set() uses (value: mut u256) {
    value = 1
}

fn valid() -> u256 {
    let mut x = 0
    let borrowed = mut x
    with (borrowed) {
        set()
    }
    borrowed
}
"#,
    );

    assert!(diags.is_empty(), "{diags:?}");
}

#[test]
fn immutable_effect_call_cannot_read_mutably_borrowed_place() {
    let diags = borrow_diags(
        r#"
fn get() -> u256 uses (value: u256) {
    value
}

fn bad() -> u256 {
    let mut x = 0
    let borrowed = mut x
    let value = with (x) {
        get()
    }
    borrowed = 1
    value
}
"#,
    );

    assert!(diags.contains("borrow conflict in `fn bad`"), "{diags:?}");
}

#[test]
fn mutable_enum_payload_reborrow_suspends_the_parent_loan() {
    let diags = borrow_diags(
        r#"
struct Item {
    value: u256,
}

impl Item {
    fn set(mut self, value: u256) {
        self.value = value
    }
}

enum Choice {
    Pair([Item; 2]),
    Triple([Item; 3]),
}

impl Choice {
    fn set(mut self, index: usize, value: u256) {
        match self {
            Choice::Pair(mut items) => items[index].set(value: value),
            Choice::Triple(mut items) => items[index].set(value: value),
        }
    }
}
"#,
    );

    assert!(diags.is_empty(), "{diags:?}");
}

#[test]
fn mutable_enum_payload_reborrow_still_rejects_independent_aliases() {
    let diags = borrow_diags(
        r#"
struct Item {
    value: u256,
}

enum Choice {
    Item(Item),
}

fn bad(mut choice: Choice) {
    match choice {
        Choice::Item(mut item) => {
            let first: mut u256 = mut item.value
            let second: mut u256 = mut item.value
            second = 1
            first = 2
        }
    }
}
"#,
    );

    assert!(diags.contains("borrow conflict in `fn bad`"), "{diags:?}");
    assert!(diags.contains("cannot mutably borrow"), "{diags:?}");
}

#[test]
fn destructured_tuple_param_field_projection_resolves_its_carrier_root() {
    let diags = borrow_diags(
        r#"
struct Byte {
    val: u8,
}

fn read(input: (Byte, u256)) -> u8 {
    let (byte, _) = input
    byte.val
}
"#,
    );

    assert!(diags.is_empty(), "{diags:?}");
}

#[test]
fn ordinary_effect_handle_fields_use_the_handle_backing_place() {
    let source = r#"
use core::{AddressSpace, EffectHandle}

struct TaggedPtr<T> {
    tag: u256,
    addr: u256,
}

impl<T> EffectHandle for TaggedPtr<T> {
    type Target = T
    const SPACE: AddressSpace = AddressSpace::Memory

    fn from_raw(_ raw: u256) -> Self {
        Self { tag: 1, addr: raw }
    }

    fn raw(self) -> u256 {
        self.addr
    }
}

fn identity(_ ptr: TaggedPtr<u256>) -> TaggedPtr<u256> {
    ptr
}

fn read_call_result() -> u256 {
    let ptr = identity(TaggedPtr { tag: 7, addr: 8 })
    ptr.tag
}

fn read_nested_array() -> u256 {
    let ptrs: [TaggedPtr<u256>; 2] = [
        TaggedPtr { tag: 7, addr: 8 },
        TaggedPtr { tag: 9, addr: 10 },
    ]
    ptrs[1].tag
}

fn mutate(mut _ ptr: own TaggedPtr<u256>) -> u256 {
    ptr.tag = 11
    ptr.tag
}
"#;
    let diags = borrow_diags(source);
    assert!(diags.is_empty(), "{diags}");

    let mut db = HirAnalysisTestDb::default();
    let file = db.new_stand_alone("semantic_borrowck.fe".into(), source);
    let (top_mod, _) = db.top_mod(file);
    let normalized = normalized_func_body(&db, top_mod, "read_call_result");
    let place = normalized
        .blocks
        .iter()
        .flat_map(|block| block.stmts.iter())
        .find_map(|stmt| match &stmt.kind {
            NSStmtKind::Assign {
                expr: NExpr::ReadPlace { place, .. },
                ..
            } if matches!(place.path.iter().next(), Some(Projection::Field(0))) => Some(place),
            _ => None,
        })
        .expect("tag field read");
    let root = place
        .root
        .borrow_root()
        .expect("ordinary handle backing root");
    assert!(
        matches!(normalized.root(root), Some(NBorrowRoot::LocalSlot { .. })),
        "ordinary handle field should read its ADT backing local: {place:#?}"
    );
}

#[test]
fn ordinary_effect_handle_field_borrows_still_conflict() {
    let diags = borrow_diags(
        r#"
use core::{AddressSpace, EffectHandle}

struct TaggedPtr<T> {
    tag: u256,
    addr: u256,
}

impl<T> EffectHandle for TaggedPtr<T> {
    type Target = T
    const SPACE: AddressSpace = AddressSpace::Memory

    fn from_raw(_ raw: u256) -> Self {
        Self { tag: 1, addr: raw }
    }

    fn raw(self) -> u256 {
        self.addr
    }
}

fn conflict(mut _ ptr: own TaggedPtr<u256>) {
    let first: mut u256 = mut ptr.tag
    let second: mut u256 = mut ptr.tag
    second = 1
    first = 2
}
"#,
    );
    assert!(
        diags.contains("borrow conflict in `fn conflict`"),
        "{diags}"
    );
}

#[test]
fn reports_noesc_storage_escape_through_whole_assignment() {
    let diags = borrow_diags(
        r#"
struct Esc {
    h: mut u256,
    tag: u256,
}

pub contract NoEscStore {
    mut slot: Esc

    init() uses (mut slot) {
        let mut x: u256 = 0
        let e: Esc = Esc { h: mut x, tag: 0 }
        slot = e
    }
}
"#,
    );

    assert!(
        diags.contains("noesc violation in `fn NoEscStore::__init__`"),
        "{diags:?}"
    );
    assert!(diags.contains("cannot store `Esc` in storage"), "{diags:?}");
}

#[test]
fn reports_noesc_storage_escape_through_field_assignment() {
    let diags = borrow_diags(
        r#"
struct Esc {
    h: mut u256,
    tag: u256,
}

struct Wrapper {
    e: Esc,
}

pub contract NoEscFieldStore {
    mut slot: Wrapper

    init() uses (mut slot) {
        let mut x: u256 = 0
        let e: Esc = Esc { h: mut x, tag: 0 }
        slot.e = e
    }
}
"#,
    );

    assert!(
        diags.contains("noesc violation in `fn NoEscFieldStore::__init__`"),
        "{diags:?}"
    );
    assert!(diags.contains("cannot store `Esc` in storage"), "{diags:?}");
}

#[test]
fn reports_noesc_storage_escape_through_inline_aggregate_store() {
    let diags = borrow_diags(
        r#"
struct Esc {
    h: mut u256,
    tag: u256,
}

pub contract NoEscInlineStore {
    mut slot: Esc

    init() uses (mut slot) {
        let mut x: u256 = 0
        slot = Esc { h: mut x, tag: 0 }
    }
}
"#,
    );

    assert!(
        diags.contains("noesc violation in `fn NoEscInlineStore::__init__`"),
        "{diags:?}"
    );
    assert!(diags.contains("cannot store `Esc` in storage"), "{diags:?}");
}

#[test]
fn reports_noesc_storage_escape_for_ref_handle_in_stored_aggregate() {
    let diags = borrow_diags(
        r#"
struct Esc {
    h: ref u256,
    tag: u256,
}

pub contract NoEscRefStore {
    mut slot: Esc

    init() uses (mut slot) {
        let x: u256 = 0
        let e: Esc = Esc { h: ref x, tag: 0 }
        slot = e
    }
}
"#,
    );

    assert!(
        diags.contains("noesc violation in `fn NoEscRefStore::__init__`"),
        "{diags:?}"
    );
    assert!(diags.contains("cannot store `Esc` in storage"), "{diags:?}");
}

#[test]
fn reports_storage_borrow_passed_as_regular_function_argument() {
    let diags = borrow_diags(
        r#"
fn bump(_ handle: mut u256) {
    handle += 1
}

pub contract NoEscCallArg {
    mut slot: u256

    init() uses (mut slot) {
        bump(mut slot)
    }
}
"#,
    );

    assert!(
        diags.contains("noesc violation in `fn NoEscCallArg::__init__`"),
        "{diags:?}"
    );
    assert!(
        diags.contains("cannot pass `mut u256` from storage as function argument"),
        "{diags:?}"
    );
}

#[test]
fn allows_memory_noesc_values_and_memory_borrow_call_args() {
    let diags = borrow_diags(
        r#"
struct Esc {
    h: mut u256,
    tag: u256,
}

fn bump(_ handle: mut u256) {
    handle += 1
}

fn ok() {
    let mut x: u256 = 0
    let e: Esc = Esc { h: mut x, tag: 0 }
    let mut y: u256 = 1
    let mut dst: Esc = Esc { h: mut y, tag: 1 }
    dst = e
    let mut z: u256 = 2
    bump(mut z)
}
"#,
    );

    assert!(!diags.contains("noesc violation"), "{diags:?}");
    assert!(
        !diags.contains("internal borrow checking error"),
        "{diags:?}"
    );
}

#[test]
fn generic_noesc_store_is_rejected_only_after_storage_specialization() {
    let mut db = HirAnalysisTestDb::default();
    let file = db.new_stand_alone(
        "semantic_borrowck.fe".into(),
        r#"
struct Box<T> {
    value: T,
}

fn store_generic<T>(value: own T) uses (slot: mut Box<T>) {
    slot = Box<T> { value }
}

pub contract GenericNoEsc {
    mut slot: Box<mut u256>

    init() uses (mut slot) {
        let mut x: u256 = 0
        store_generic<mut u256>(mut x)
    }
}
"#,
    );
    let (top_mod, _) = db.top_mod(file);
    let store_generic = top_mod
        .all_items(&db)
        .iter()
        .find_map(|item| match item {
            ItemKind::Func(func)
                if func
                    .name(&db)
                    .to_opt()
                    .is_some_and(|name| name.data(&db) == "store_generic") =>
            {
                Some(*func)
            }
            _ => None,
        })
        .expect("store_generic function");
    let identity = get_or_build_semantic_instance(
        &db,
        identity_semantic_instance_key(&db, BodyOwner::Func(store_generic)),
    );
    check_semantic_noesc(&db, identity).expect("generic identity noesc should be accepted");

    let init = top_mod
        .all_items(&db)
        .iter()
        .find_map(|item| match item {
            ItemKind::Contract(contract) => Some(get_or_build_semantic_instance(
                &db,
                identity_semantic_instance_key(
                    &db,
                    BodyOwner::ContractInit {
                        contract: *contract,
                    },
                ),
            )),
            _ => None,
        })
        .expect("contract init instance");
    check_semantic_borrows(&db, init)
        .expect("passing a local capability to generic noesc analysis should be borrow-safe");
    let specialized = init
        .callees(&db)
        .iter()
        .find_map(|callee| match callee.key.owner(&db) {
            BodyOwner::Func(func) if func == store_generic => {
                Some(get_or_build_semantic_instance(&db, callee.key))
            }
            _ => None,
        })
        .expect("specialized store_generic callee");
    let err = check_semantic_noesc(&db, specialized)
        .expect_err("specialized noesc store should be rejected");
    assert!(
        err.message
            .contains("noesc violation in `fn store_generic`"),
        "{err:#?}"
    );
    assert!(
        format!("{err:#?}").contains("cannot store `Box<mut u256>` in storage"),
        "{err:#?}"
    );
}

#[test]
fn rejects_return_borrow_to_local() {
    let diags = borrow_diags(
        r#"
struct Pair {
    a: u256,
    b: u256,
}

fn bad() -> mut u256 {
    let mut x = Pair { a: 0, b: 0 }
    mut x.a
}
"#,
    );

    assert!(
        diags.contains("invalid return borrow in `fn bad`"),
        "{diags:?}"
    );
    assert!(
        diags.contains("cannot return a borrow to local"),
        "{diags:?}"
    );
}

#[test]
fn rejects_return_borrow_derived_from_uses_effect_parameter() {
    let diags = borrow_diags(
        r#"
struct Store {
    value: u256,
}

fn bad() -> mut u256 uses (store: mut Store) {
    mut store.value
}
"#,
    );

    assert!(
        diags.contains("invalid return borrow in `fn bad`"),
        "{diags:?}"
    );
    assert!(
        diags.contains("cannot return a borrow derived from an effect parameter"),
        "{diags:?}"
    );
}

#[test]
fn array_index_reads_do_not_hit_internal_borrowck_error() {
    let diags = borrow_diags(
        r#"
pub fn cast_u8_usize_cmp(indices: [u8; 8], i: usize, j: usize) -> u8 {
    let path = indices[i]
    if j < path as usize {
        return 1
    }
    if j == path as usize {
        return 2
    }
    if j > path as usize {
        return 3
    }
    0
}
"#,
    );

    assert!(
        !diags.contains("internal borrow checking error"),
        "{diags:?}"
    );
}

#[test]
fn raw_mem_allocate_does_not_report_move_conflict() {
    let diags = borrow_diags(
        r#"
use std::evm::RawMem

fn allocate(bytes: u256) -> u256 uses (mem: mut RawMem) {
    let mut ptr = mem.mload(0x40)
    if ptr == 0 {
        ptr = 0x60
    }
    mem.mstore(0x40, ptr + bytes)
    ptr
}
"#,
    );

    assert!(!diags.contains("move conflict"), "{diags:?}");
    assert!(
        !diags.contains("internal borrow checking error"),
        "{diags:?}"
    );
}

#[test]
fn code_region_fixture_does_not_report_move_conflict() {
    let diags = borrow_diags(include_str!("../../codegen/tests/fixtures/code_region.fe"));
    assert!(!diags.contains("move conflict"), "{diags:?}");
    assert!(
        !diags.contains("internal borrow checking error"),
        "{diags:?}"
    );
}

#[test]
fn create_contract_fixture_does_not_report_top_level_semantic_borrow_errors() {
    let diags = borrow_diags(include_str!(
        "../../codegen/tests/fixtures/create_contract.fe"
    ));
    assert!(!diags.contains("borrow conflict"), "{diags:?}");
    assert!(!diags.contains("move conflict"), "{diags:?}");
    assert!(
        !diags.contains("internal borrow checking error"),
        "{diags:?}"
    );
}

#[test]
fn reports_move_conflict_for_reused_owned_binding() {
    let diags = borrow_diags(
        r#"
struct Inner {}

fn bad(x: own Inner) {
    let y = x
    let z = x
}
"#,
    );

    assert!(diags.contains("move conflict in `fn bad`"), "{diags:?}");
}

#[test]
fn reports_move_conflict_for_non_copy_projection_from_view_param() {
    let diags = borrow_diags(
        r#"
struct Wrapper {
    p: Pair,
}

struct Pair {
    x: u32,
    y: u32,
}

fn unwrap(w: Wrapper) -> Pair {
    let p = w.p
    p
}
"#,
    );

    assert!(diags.contains("move conflict in `fn unwrap`"), "{diags:?}");
    assert!(
        diags.contains("cannot move out of a view parameter"),
        "{diags:?}"
    );
}

#[test]
fn non_copy_projection_to_view_receiver_does_not_move_from_view_param() {
    let mut db = HirAnalysisTestDb::default();
    let file = db.new_stand_alone(
        "semantic_borrowck.fe".into(),
        r#"
struct Row {
    cells: [u256; 4],
}

impl Row {
    fn get_cell(self, col: usize) -> u256 {
        self.cells[col]
    }

    fn has_value(self, val: u256) -> bool {
        let mut c: usize = 0
        while c < 4 {
            if self.cells[c] == val {
                return true
            }
            c += 1
        }
        return false
    }
}

struct Board {
    rows: [Row; 4],
}

fn read_board(board: Board, row: usize, col: usize) -> u256 {
    board.rows[row].get_cell(col: col)
}

fn find_empty(board: Board, row: usize, col: usize) -> bool {
    if board.rows[row].has_value(val: 0) {
        return board.rows[row].get_cell(col: col) == 0
    }
    false
}
"#,
    );
    let (top_mod, _) = db.top_mod(file);
    let diags = format_diagnostics(
        &db,
        &collect_semantic_borrow_diagnostic_vouchers(&db, top_mod),
    );
    assert!(!diags.contains("move conflict"), "{diags:?}");
    assert!(
        !diags.contains("internal borrow checking error"),
        "{diags:?}"
    );

    let normalized = normalized_func_body(&db, top_mod, "read_board");
    let row_read_mode = normalized
        .blocks
        .iter()
        .flat_map(|block| block.stmts.iter())
        .find_map(|stmt| match &stmt.kind {
            NSStmtKind::Assign {
                dst,
                expr: NExpr::ReadPlace { mode, .. },
            } if normalized.locals[dst.index()].ty.pretty_print(&db) == "Row" => Some(mode),
            _ => None,
        })
        .expect("row projection read");
    assert_eq!(*row_read_mode, ReadMode::Read);
}

#[test]
fn closure_owned_param_projection_ignores_enclosing_view_param_mode() {
    let src = r#"
struct Row {
    cells: [u256; 4],
}

impl Row {
    fn get_cell(self, col: usize) -> u256 {
        self.cells[col]
    }
}

struct Board {
    rows: [Row; 4],
}

fn read_via_closure(_ marker: u256, _ decoy: Board, board: own Board) -> u256 {
    let read = |value: own Board| -> u256 { value.rows[0].get_cell(col: 0) }
    read.call(board)
}
"#;
    let mut found_closure = false;
    for_each_fixture_instance(src, |db, instance| {
        if !matches!(instance.key(db).owner(db), BodyOwner::Closure { .. }) {
            return;
        }
        found_closure = true;
        let normalized = normalize_semantic_body(db, instance).expect("normalized closure body");
        let (row_read_mode, logical_param_local) = normalized
            .blocks
            .iter()
            .flat_map(|block| block.stmts.iter())
            .find_map(|stmt| match &stmt.kind {
                NSStmtKind::Assign {
                    dst,
                    expr: NExpr::ReadPlace { mode, place },
                } if normalized.locals[dst.index()].ty.pretty_print(db) == "Row" => {
                    let NSPlaceRoot::Root(root) = place.root else {
                        return None;
                    };
                    let Some(NBorrowRoot::LocalSlot { local }) = normalized.root(root) else {
                        return None;
                    };
                    Some((mode, *local))
                }
                _ => None,
            })
            .expect("row projection read from the owned closure parameter");
        assert_eq!(*row_read_mode, ReadMode::Move);
        let snapshot_source = normalized.locals[logical_param_local.index()]
            .facts
            .snapshot_source_place
            .as_ref()
            .expect("logical closure parameter must retain its tuple-field source");
        let NSPlaceRoot::Root(root) = snapshot_source.root else {
            panic!("closure argument source must be a physical parameter root");
        };
        assert!(matches!(
            normalized.root(root),
            Some(NBorrowRoot::Param { param_idx: 1, .. })
        ));
        assert_eq!(
            snapshot_source.path.iter().cloned().collect::<Vec<_>>(),
            vec![Projection::Field(0)]
        );
    });
    assert!(found_closure);
}

#[test]
fn closure_logical_param_modes_govern_moves_from_tuple_pack_fields() {
    for (mode, src, message) in [
        (
            "view",
            r#"
struct Wrapper {
    value: Boxed,
}

struct Boxed {
    value: u256,
}

fn probe() {
    let take = |wrapped: Wrapper| -> Boxed { wrapped.value }
}
"#,
            "cannot move out of a view parameter",
        ),
        (
            "ref",
            r#"
struct Wrapper {
    value: Boxed,
}

struct Boxed {
    value: u256,
}

fn probe() {
    let take = |wrapped: ref Wrapper| -> Boxed { wrapped.value }
}
"#,
            "cannot move out through a borrow handle",
        ),
        (
            "mut",
            r#"
struct Wrapper {
    value: Boxed,
}

struct Boxed {
    value: u256,
}

fn probe() {
    let take = |wrapped: mut Wrapper| -> Boxed { wrapped.value }
}
"#,
            "cannot move out through a borrow handle",
        ),
    ] {
        let diags = borrow_diags(src);
        assert!(
            diags.contains("move conflict in `fn <closure>`") && diags.contains(message),
            "{mode} parameter diagnostics: {diags}"
        );
    }

    let own_diags = borrow_diags(
        r#"
struct Wrapper {
    value: Boxed,
}

struct Boxed {
    value: u256,
}

fn probe() -> Boxed {
    let take = |wrapped: own Wrapper| -> Boxed { wrapped.value }
    take.call_once(Wrapper { value: Boxed { value: 42 } })
}
"#,
    );
    assert!(own_diags.is_empty(), "{own_diags}");
}

#[test]
fn closure_view_param_call_reads_non_copy_argument() {
    let src = r#"
struct Boxed {
    value: u256,
}

fn read_twice(_ value: own Boxed) -> u256 {
    let read = |item: Boxed| -> u256 { item.value }
    let first = read.call(value)
    first + value.value
}
"#;
    let mut db = HirAnalysisTestDb::default();
    let file = db.new_stand_alone("closure_view_param_call.fe".into(), src);
    let (top_mod, _) = db.top_mod(file);
    db.assert_no_diags(top_mod);
    let diags = format_diagnostics(
        &db,
        &collect_semantic_borrow_diagnostic_vouchers(&db, top_mod),
    );
    assert!(diags.is_empty(), "{diags}");

    let normalized = normalized_func_body(&db, top_mod, "read_twice");
    let pack_field_mode = normalized
        .blocks
        .iter()
        .flat_map(|block| block.stmts.iter())
        .find_map(|stmt| match &stmt.kind {
            NSStmtKind::Assign {
                expr: NExpr::AggregateMake { ty, fields },
                ..
            } if ty.is_tuple(&db) => fields.first().map(|field| field.mode),
            _ => None,
        })
        .expect("closure call argument-pack field");
    let call_pack_mode = normalized
        .blocks
        .iter()
        .flat_map(|block| block.stmts.iter())
        .find_map(|stmt| match &stmt.kind {
            NSStmtKind::Assign {
                expr: NExpr::Call { callee, args, .. },
                ..
            } if matches!(callee.key.owner(&db), BodyOwner::Closure { .. }) => {
                args.get(1).map(|arg| arg.mode)
            }
            _ => None,
        })
        .expect("closure call argument pack");
    assert_eq!(pack_field_mode, ReadMode::Read);
    assert_eq!(call_pack_mode, ReadMode::Move);
}

#[test]
fn closure_multiple_mut_args_reject_aliasing_the_same_place() {
    let diags = borrow_diags(
        r#"
fn probe() {
    let mut value = 0
    let write = |left: mut u256, right: mut u256| {
        left = 1
        right = 2
    }
    write.call(mut value, mut value)
}
"#,
    );
    assert!(diags.contains("borrow conflict in `fn probe`"), "{diags}");
}

#[test]
fn closure_mut_and_ref_args_reject_aliasing_the_same_place() {
    let diags = borrow_diags(
        r#"
fn probe() -> u256 {
    let mut value = 0
    let write_and_read = |write: mut u256, read: ref u256| -> u256 {
        write = 1
        read
    }
    write_and_read.call(mut value, ref value)
}
"#,
    );
    assert!(diags.contains("borrow conflict in `fn probe`"), "{diags}");
}

#[test]
fn closure_multiple_ref_args_allow_aliasing_the_same_place() {
    let diags = borrow_diags(
        r#"
fn probe() -> u256 {
    let value = 21
    let add = |left: ref u256, right: ref u256| -> u256 { left + right }
    add.call(ref value, ref value)
}
"#,
    );
    assert!(diags.is_empty(), "{diags}");
}

#[test]
fn closure_multiple_inferred_view_args_allow_aliasing_noncopy_place() {
    let diags = borrow_diags(
        r#"
struct Boxed {
    value: u256,
}

fn probe() -> u256 {
    let value = Boxed { value: 21 }
    let add = |left, right| left.value + right.value
    add.call(value, value)
}
"#,
    );
    assert!(diags.is_empty(), "{diags}");
}

#[test]
fn closure_inferred_mut_and_view_args_reject_aliasing_the_same_place() {
    let diags = borrow_diags(
        r#"
fn probe() -> u256 {
    let mut value = 0
    let write_and_read = |write: mut, read| -> u256 {
        write = 1
        read
    }
    write_and_read.call(mut value, value)
}
"#,
    );
    assert!(diags.contains("borrow conflict in `fn probe`"), "{diags}");
}

#[test]
fn closure_inferred_owned_and_view_args_reject_aliasing_the_same_place() {
    let diags = borrow_diags(
        r#"
struct Boxed {
    value: u256,
}

fn probe() -> u256 {
    let value = Boxed { value: 42 }
    let consume = |owned: own, borrowed| -> u256 {
        owned.value + borrowed.value
    }
    consume.call_once(value, value)
}
"#,
    );
    assert!(
        diags.contains("move conflict in `fn probe`")
            || diags.contains("borrow conflict in `fn probe`"),
        "{diags}"
    );
}

#[test]
fn closure_owned_then_borrowed_arg_rejects_aliasing_the_same_place() {
    let diags = borrow_diags(
        r#"
struct Boxed {
    value: u256,
}

fn probe() -> u256 {
    let value = Boxed { value: 42 }
    let consume = |owned: own Boxed, borrowed: ref Boxed| -> u256 {
        owned.value + borrowed.value
    }
    consume.call_once(value, ref value)
}
"#,
    );
    assert!(
        diags.contains("move conflict in `fn probe`")
            || diags.contains("borrow conflict in `fn probe`"),
        "{diags}"
    );
}

#[test]
fn closure_borrowed_then_owned_arg_rejects_aliasing_the_same_place() {
    let diags = borrow_diags(
        r#"
struct Boxed {
    value: u256,
}

fn probe() -> u256 {
    let value = Boxed { value: 42 }
    let consume = |borrowed: ref Boxed, owned: own Boxed| -> u256 {
        borrowed.value + owned.value
    }
    consume.call_once(ref value, value)
}
"#,
    );
    assert!(
        diags.contains("move conflict in `fn probe`")
            || diags.contains("borrow conflict in `fn probe`"),
        "{diags}"
    );
}

#[test]
fn closure_multiple_mut_args_allow_distinct_places() {
    let diags = borrow_diags(
        r#"
fn probe() -> u256 {
    let mut left = 0
    let mut right = 0
    let write = |left: mut u256, right: mut u256| {
        left = 20
        right = 22
    }
    write.call(mut left, mut right)
    left + right
}
"#,
    );
    assert!(diags.is_empty(), "{diags}");
}

#[test]
fn closure_returned_first_mut_arg_retains_only_first_provenance() {
    let selected_diags = borrow_diags(
        r#"
fn probe() {
    let mut left = 0
    let mut right = 0
    let first = |left: mut u256, right: mut u256| -> mut u256 { left }
    let returned = first.call(mut left, mut right)
    let other = mut left
    other = 1
    returned = 2
}
"#,
    );
    assert!(
        selected_diags.contains("borrow conflict in `fn probe`"),
        "{selected_diags}"
    );

    let unrelated_diags = borrow_diags(
        r#"
fn probe() {
    let mut left = 0
    let mut right = 0
    let first = |left: mut u256, right: mut u256| -> mut u256 { left }
    let returned = first.call(mut left, mut right)
    let other = mut right
    other = 1
    returned = 2
}
"#,
    );
    assert!(unrelated_diags.is_empty(), "{unrelated_diags}");
}

#[test]
fn closure_returned_last_mut_arg_retains_only_last_provenance() {
    let selected_diags = borrow_diags(
        r#"
fn probe() {
    let mut left = 0
    let mut right = 0
    let last = |left: mut u256, right: mut u256| -> mut u256 { right }
    let returned = last.call(mut left, mut right)
    let other = mut right
    other = 1
    returned = 2
}
"#,
    );
    assert!(
        selected_diags.contains("borrow conflict in `fn probe`"),
        "{selected_diags}"
    );

    let unrelated_diags = borrow_diags(
        r#"
fn probe() {
    let mut left = 0
    let mut right = 0
    let last = |left: mut u256, right: mut u256| -> mut u256 { right }
    let returned = last.call(mut left, mut right)
    let other = mut left
    other = 1
    returned = 2
}
"#,
    );
    assert!(unrelated_diags.is_empty(), "{unrelated_diags}");
}

#[test]
fn inferred_closure_return_retains_selected_mut_arg_provenance() {
    let selected_diags = borrow_diags(
        r#"
fn probe() {
    let mut left = 0
    let mut right = 0
    let first = |left: mut, right: mut| left
    let returned = first.call(mut left, mut right)
    let other = mut left
    other = 1
    returned = 2
}
"#,
    );
    assert!(
        selected_diags.contains("borrow conflict in `fn probe`"),
        "{selected_diags}"
    );

    let unrelated_diags = borrow_diags(
        r#"
fn probe() {
    let mut left = 0
    let mut right = 0
    let first = |left: mut, right: mut| left
    let returned = first.call(mut left, mut right)
    let other = mut right
    other = 1
    returned = 2
}
"#,
    );
    assert!(unrelated_diags.is_empty(), "{unrelated_diags}");
}

#[test]
fn inferred_closure_return_retains_default_view_arg_provenance() {
    let diags = borrow_diags(
        r#"
struct Boxed {
    value: u256,
}

fn probe() -> u256 {
    let value = Boxed { value: 42 }
    let identity = |value| value
    let returned = identity.call(value)
    let moved = value
    returned.value + moved.value
}
"#,
    );
    assert!(diags.contains("borrow conflict in `fn probe`"), "{diags}");
}

#[test]
fn consuming_closure_may_return_borrow_from_owned_argument() {
    let diags = borrow_diags(
        r#"
struct Boxed {
    value: u256,
}

fn probe() -> u256 {
    let borrow = |value: own Boxed| -> ref u256 { ref value.value }
    let returned = borrow.call_once(Boxed { value: 42 })
    returned
}
"#,
    );
    assert!(diags.is_empty(), "{diags}");
}

#[test]
fn consuming_closure_may_return_aggregate_borrow_from_owned_argument() {
    let diags = borrow_diags(
        r#"
struct Boxed {
    value: u256,
}

struct Borrowed {
    value: ref u256,
}

fn probe() -> u256 {
    let borrow = |value: own Boxed| -> Borrowed {
        Borrowed { value: ref value.value }
    }
    let returned = borrow.call_once(Boxed { value: 42 })
    returned.value
}
"#,
    );
    assert!(diags.is_empty(), "{diags}");
}

#[test]
fn returned_borrow_does_not_restore_moved_closure_argument() {
    let diags = borrow_diags(
        r#"
struct Boxed {
    value: u256,
}

fn probe(_ value: own Boxed) -> u256 {
    let borrow = |argument: own Boxed| -> ref u256 { ref argument.value }
    let returned = borrow.call_once(value)
    value.value
    returned
}
"#,
    );
    assert!(
        diags.contains("cannot use a value after it was moved"),
        "{diags}"
    );
}

#[test]
fn layout_backing_does_not_let_borrows_of_fresh_fields_escape() {
    for (owner, src) in [
        (
            "function",
            r#"
struct Boxed {
    value: u256,
}

fn probe(_ input: own Boxed) -> ref u256 {
    let local = Boxed { value: input.value }
    ref local.value
}
"#,
        ),
        (
            "closure",
            r#"
struct Boxed {
    value: u256,
}

fn probe(_ input: own Boxed) -> ref u256 {
    let borrow_fresh = |value: own Boxed| -> ref u256 {
        let local = Boxed { value: value.value }
        ref local.value
    }
    borrow_fresh.call_once(input)
}
"#,
        ),
    ] {
        let diags = borrow_diags(src);
        assert!(
            diags.contains("cannot return a borrow to local `local`"),
            "{owner}: {diags}"
        );
    }

    let aggregate_shape_diags = borrow_diags(
        r#"
use core::option::Option::{self, Some}

struct Boxed {
    value: u256,
}

fn tuple_field(_ input: u256) -> ref u256 {
    let local = (input,)
    ref local.0
}

fn array_element(_ input: u256) -> ref u256 {
    let local = [input]
    ref local[0]
}

fn nested_field(_ input: own Boxed) -> ref u256 {
    let local = (Boxed { value: input.value },)
    ref local.0.value
}

fn enum_field(_ input: u256) -> ref u256 {
    let local = Some(input)
    match local {
        Some(value) => ref value
        Option::None => ref input
    }
}
"#,
    );
    assert_eq!(
        aggregate_shape_diags
            .matches("cannot return a borrow to local `local`")
            .count(),
        4,
        "{aggregate_shape_diags}"
    );

    let capability_field_diags = borrow_diags(
        r#"
struct Holder {
    value: ref u256,
}

fn probe(_ input: u256) -> ref u256 {
    let local = Holder { value: ref input }
    local.value
}
"#,
    );
    assert!(
        capability_field_diags.is_empty(),
        "{capability_field_diags}"
    );
}

#[test]
fn inferred_view_closure_payload_operations_do_not_return_local_borrows() {
    for (kind, src) in [
        (
            "contextual identity",
            r#"
use core::functional::Fn

fn apply<T, F: Fn<(view T,), T>>(_ func: F, _ value: own T) -> T {
    func.call(value)
}

fn probe() -> u256 {
    apply(|value| value, 42 as u256)
}
"#,
        ),
        (
            "arithmetic",
            r#"
fn probe() -> u256 {
    (|value| value + 1 as u256).call(41 as u256)
}
"#,
        ),
        (
            "field",
            r#"
struct Boxed {
    value: u256,
}

fn probe() -> u256 {
    let field = |value| value.value
    field.call(Boxed { value: 42 })
}
"#,
        ),
        (
            "cast",
            r#"
fn probe() -> u256 {
    let cast = |value| value as u256
    cast.call(42 as u8)
}
"#,
        ),
        (
            "unary",
            r#"
fn probe() -> bool {
    let invert = |value| !value
    invert.call(false)
}
"#,
        ),
        (
            "index",
            r#"
fn probe() -> u256 {
    let at = |values| values[0]
    at.call([42 as u256, 0])
}
"#,
        ),
        (
            "higher order",
            r#"
fn probe() -> u256 {
    let call_with = |function, value| function.call(value)
    call_with.call(|value| value + 1 as u256, 41 as u256)
}
"#,
        ),
        (
            "for loop",
            r#"
fn probe() -> u256 {
    let sum_values = |values| -> u256 {
        let mut total: u256 = 0
        for value in values {
            total += value
        }
        total
    }
    sum_values.call([20 as u256, 22 as u256])
}
"#,
        ),
        (
            "conditional",
            r#"
fn probe() -> u256 {
    let branch = |value| -> u256 { if true { value } else { 0 as u256 } }
    branch.call(42 as u256)
}
"#,
        ),
        (
            "match",
            r#"
fn probe() -> u256 {
    let branch = |value| -> u256 {
        match true {
            true => value
            false => 0 as u256
        }
    }
    branch.call(42 as u256)
}
"#,
        ),
        (
            "tuple destructure",
            r#"
fn probe() -> u256 {
    let unpack = |pair| {
        let (left, right) = pair
        left + right
    }
    unpack.call((20 as u256, 22 as u256))
}
"#,
        ),
        (
            "named view field",
            r#"
struct view {
    value: u256,
}

fn probe() -> u256 {
    let field = |value: view view| value.value
    field.call(view { value: 42 })
}
"#,
        ),
        (
            "nested capture",
            r#"
fn probe() -> u256 {
    let outer = |left| {
        let inner = |right| left + right
        inner.call(22 as u256)
    }
    outer.call(20 as u256)
}
"#,
        ),
    ] {
        let diags = borrow_diags(src);
        assert!(diags.is_empty(), "{kind}: {diags}");
    }
}

#[test]
fn inferred_copy_view_control_flow_join_materializes_values() {
    for src in [
        r#"
fn probe() {
    let branch = |value| if true { value } else { 0 as u256 }
    branch.call(42 as u256)
}
"#,
        r#"
fn probe() {
    let branch = |value| match true {
        true => value
        false => 0 as u256
    }
    branch.call(42 as u256)
}
"#,
    ] {
        let diags = borrow_diags(src);
        assert!(diags.is_empty(), "{diags}");
    }
}

#[test]
fn inferred_noncopy_view_control_flow_join_cannot_return_a_temporary_view() {
    for body in [
        "if true { value } else { Boxed { value: 0 } }",
        "match true { true => value, false => Boxed { value: 0 } }",
    ] {
        let diags = borrow_diags(&format!(
            r#"
struct Boxed {{
    value: u256,
}}

fn probe() {{
    let branch = |value| {body}
    branch.call(Boxed {{ value: 42 }})
}}
"#,
        ));
        assert!(
            diags.contains("invalid return borrow in `fn <closure>`"),
            "{body}: {diags}"
        );
    }
}

#[test]
fn returned_view_from_function_retains_argument_provenance() {
    let diags = borrow_diags(
        r#"
struct Boxed {
    value: u256,
}

fn identity(_ value: view Boxed) -> view Boxed {
    value
}

fn probe() -> u256 {
    let value = Boxed { value: 42 }
    let returned = identity(value)
    let moved = value
    returned.value + moved.value
}
"#,
    );
    assert!(diags.contains("borrow conflict in `fn probe`"), "{diags}");
}

#[test]
fn contextual_views_of_owned_rvalues_materialize_stable_backing_values() {
    let src = r#"
fn value() -> u256 {
    42
}

fn probe() -> u256 {
    let from_call: view u256 = value()
    let from_binary: view u256 = 20 + 22
    from_call + from_binary
}
"#;
    let mut db = HirAnalysisTestDb::default();
    let file = db.new_stand_alone("contextual_view_rvalue.fe".into(), src);
    let (top_mod, _) = db.top_mod(file);
    let diags = format_diagnostics(
        &db,
        &collect_semantic_borrow_diagnostic_vouchers(&db, top_mod),
    );
    assert!(diags.is_empty(), "{diags}");

    let normalized = normalized_func_body(&db, top_mod, "probe");
    let call_result = normalized
        .blocks
        .iter()
        .flat_map(|block| &block.stmts)
        .find_map(|stmt| match stmt.kind {
            NSStmtKind::Assign {
                dst,
                expr: NExpr::Call { .. },
            } => Some(dst),
            _ => None,
        })
        .expect("owned call result");
    assert_eq!(
        normalized.locals[call_result.index()].ty.pretty_print(&db),
        "u256"
    );
    assert!(
        normalized
            .blocks
            .iter()
            .flat_map(|block| &block.stmts)
            .any(|stmt| matches!(
                &stmt.kind,
                NSStmtKind::Assign {
                    expr: NExpr::ReadPlace {
                        place:
                            NSPlace {
                                root: NSPlaceRoot::Root(root),
                                ..
                            },
                        ..
                    },
                    ..
                } if matches!(
                    normalized.root(*root),
                    Some(NBorrowRoot::LocalSlot { local }) if *local == call_result
                )
            )),
        "the contextual view must read from the separately materialized owned result"
    );

    let invalid_return_diags = borrow_diags(
        r#"
fn value() -> u256 {
    42
}

fn bad() -> view u256 {
    value()
}
"#,
    );
    assert!(
        invalid_return_diags.contains("invalid return borrow in `fn bad`"),
        "{invalid_return_diags}"
    );
}

#[test]
fn implicit_view_aggregate_field_retains_argument_provenance() {
    let diags = borrow_diags(
        r#"
struct Boxed {
    value: u256,
}

struct Holder {
    value: view Boxed,
}

fn probe() -> u256 {
    let value = Boxed { value: 42 }
    let holder = Holder { value }
    let moved = value
    holder.value.value + moved.value
}
"#,
    );
    assert!(diags.contains("borrow conflict in `fn probe`"), "{diags}");
}

#[test]
fn implicit_view_tuple_array_and_enum_fields_retain_argument_provenance() {
    for (kind, src) in [
        (
            "tuple",
            r#"
struct Boxed {
    value: u256,
}

fn probe() -> u256 {
    let value = Boxed { value: 42 }
    let holder: (view Boxed,) = (value,)
    let moved = value
    holder.0.value + moved.value
}
"#,
        ),
        (
            "array repeat",
            r#"
struct Boxed {
    value: u256,
}

fn probe() -> u256 {
    let value = Boxed { value: 42 }
    let holder: [view Boxed; 2] = [value; 2]
    let moved = value
    holder[0].value + moved.value
}
"#,
        ),
        (
            "enum",
            r#"
struct Boxed {
    value: u256,
}

enum Holder {
    Value(view Boxed),
}

fn probe() -> u256 {
    let value = Boxed { value: 42 }
    let holder = Holder::Value(value)
    let moved = value
    match holder {
        Holder::Value(borrowed) => borrowed.value + moved.value,
    }
}
"#,
        ),
        (
            "record enum",
            r#"
struct Boxed {
    value: u256,
}

enum Holder {
    Value { value: view Boxed },
}

fn probe() -> u256 {
    let value = Boxed { value: 42 }
    let holder = Holder::Value { value }
    let moved = value
    match holder {
        Holder::Value { value: borrowed } => borrowed.value + moved.value,
    }
}
"#,
        ),
    ] {
        let diags = borrow_diags(src);
        assert!(
            diags.contains("borrow conflict in `fn probe`"),
            "{kind}: {diags}"
        );
    }
}

#[test]
fn returned_aggregate_with_view_field_retains_argument_provenance() {
    let diags = borrow_diags(
        r#"
struct Boxed {
    value: u256,
}

struct Holder {
    value: view Boxed,
}

fn wrap(_ value: view Boxed) -> Holder {
    Holder { value }
}

fn probe() -> u256 {
    let value = Boxed { value: 42 }
    let holder = wrap(value)
    let moved = value
    holder.value.value + moved.value
}
"#,
    );
    assert!(diags.contains("borrow conflict in `fn probe`"), "{diags}");
}

#[test]
fn user_defined_callable_mixed_capability_args_enforce_aliasing() {
    let distinct_diags = borrow_diags(
        r#"
use core::functional::FnOnce

struct AddInto {}

impl FnOnce<(mut u256, view u256), ()> for AddInto {
    fn call_once(self: own Self, mut _ args: own (mut u256, view u256)) {
        args.0 += args.1
    }
}

fn probe() {
    let mut left = 20
    let right = 22
    AddInto {}.call_once(mut left, right)
}
"#,
    );
    assert!(distinct_diags.is_empty(), "{distinct_diags}");

    let aliased_diags = borrow_diags(
        r#"
use core::functional::FnOnce

struct AddInto {}

impl FnOnce<(mut u256, view u256), ()> for AddInto {
    fn call_once(self: own Self, mut _ args: own (mut u256, view u256)) {
        args.0 += args.1
    }
}

fn probe() {
    let mut value = 21
    AddInto {}.call_once(mut value, value)
}
"#,
    );
    assert!(
        aliased_diags.contains("borrow conflict in `fn probe`"),
        "{aliased_diags}"
    );
}

#[test]
fn user_defined_fn_once_return_retains_tuple_field_provenance() {
    let selected_diags = borrow_diags(
        r#"
use core::functional::FnOnce

struct First {}

impl FnOnce<(mut u256, mut u256), mut u256> for First {
    fn call_once(self: own Self, _ args: own (mut u256, mut u256)) -> mut u256 {
        args.0
    }
}

fn probe() {
    let mut left = 0
    let mut right = 0
    let returned = First {}.call_once(mut left, mut right)
    let other = mut left
    other = 1
    returned = 2
}
"#,
    );
    assert!(
        selected_diags.contains("borrow conflict in `fn probe`"),
        "{selected_diags}"
    );

    let unrelated_diags = borrow_diags(
        r#"
use core::functional::FnOnce

struct First {}

impl FnOnce<(mut u256, mut u256), mut u256> for First {
    fn call_once(self: own Self, _ args: own (mut u256, mut u256)) -> mut u256 {
        args.0
    }
}

fn probe() {
    let mut left = 0
    let mut right = 0
    let returned = First {}.call_once(mut left, mut right)
    let other = mut right
    other = 1
    returned = 2
}
"#,
    );
    assert!(unrelated_diags.is_empty(), "{unrelated_diags}");
}

#[test]
fn closure_capture_and_mut_arg_reject_aliasing_the_same_place() {
    let diags = borrow_diags(
        r#"
fn probe() {
    let mut value = 0
    let captured = mut value
    let write = |arg: mut u256| {
        captured = 1
        arg = 2
    }
    write.call(mut value)
}
"#,
    );
    assert!(diags.contains("borrow conflict in `fn probe`"), "{diags}");
}

#[test]
fn closure_preexisting_mut_handles_preserve_argument_aliasing() {
    let distinct_diags = borrow_diags(
        r#"
fn probe() {
    let mut left = 0
    let mut right = 0
    let left_handle = mut left
    let right_handle = mut right
    let write = |left: mut u256, right: mut u256| {
        left = 1
        right = 2
    }
    write.call(left_handle, right_handle)
}
"#,
    );
    assert!(distinct_diags.is_empty(), "{distinct_diags}");

    let aliased_diags = borrow_diags(
        r#"
fn probe() {
    let mut value = 0
    let handle = mut value
    let write = |left: mut u256, right: mut u256| {
        left = 1
        right = 2
    }
    write.call(handle, handle)
}
"#,
    );
    assert!(
        aliased_diags.contains("borrow conflict in `fn probe`"),
        "{aliased_diags}"
    );
}

#[test]
fn closure_returned_borrow_nested_in_owned_arg_pack_is_tracked() {
    let diags = borrow_diags(
        r#"
struct Borrowed {
    value: mut u256,
}

fn probe() {
    let mut value = 0
    let unwrap = |borrowed: own Borrowed| -> mut u256 { borrowed.value }
    let returned = unwrap.call_once(Borrowed { value: mut value })
    let other = mut value
    other = 1
    returned = 2
}
"#,
    );
    assert!(diags.contains("borrow conflict in `fn probe`"), "{diags}");
}

#[test]
fn closure_control_flow_pattern_move_consumes_captured_value() {
    let diags = borrow_diags(
        r#"
use core::functional::FnOnce

struct Boxed {
    value: u256,
}

fn bad(_ boxed: own Boxed, _ choose_left: bool) -> u256 {
    let take = |_ unit: own ()| -> u256 {
        let selected = if choose_left { boxed } else { boxed }
        selected.value + boxed.value
    }
    take.call_once(())
}
"#,
    );

    assert!(
        diags.contains("move conflict in `fn <closure>`"),
        "{diags:?}",
    );
    assert!(
        diags.contains("cannot use a value after it was moved"),
        "{diags:?}",
    );
}

#[test]
fn closure_control_flow_moves_cover_divergent_and_fresh_predecessors() {
    for body in [
        "if choose_left { left } else { right }",
        "if choose_left { left } else { Boxed { value: 0 } }",
        "match choose_left { true => left, false => right }",
    ] {
        let diags = borrow_diags(&format!(
            r#"
use core::functional::FnOnce

struct Boxed {{
    value: u256,
}}

fn bad(_ left: own Boxed, _ right: own Boxed, _ choose_left: bool) -> u256 {{
    let take = |_ unit: own ()| -> u256 {{
        let selected = {body}
        selected.value + left.value + right.value
    }}
    take.call_once(())
}}
"#,
        ));

        assert!(
            diags.contains("move conflict in `fn <closure>`")
                && diags.contains("cannot use a value after it was moved"),
            "{body}: {diags:?}",
        );
    }
}

#[test]
fn closure_control_flow_copy_patterns_remain_non_consuming() {
    let src = r#"
use core::functional::Fn

struct Boxed {
    value: u256,
}

struct Mixed {
    copy: u256,
    owned: Boxed,
}

fn valid(_ mixed: own Mixed, _ choose_left: bool) -> u256 {
    let inspect = |_ unit: own ()| -> u256 {
        let Mixed { copy, owned: _ } =
            if choose_left { mixed } else { mixed }
        copy + mixed.copy
    }
    inspect.call(()) + inspect.call(())
}
"#;
    let diags = borrow_diags(src);

    assert!(diags.is_empty(), "{diags}");
}

#[test]
fn non_copy_field_projection_to_view_receiver_does_not_move_from_mut_receiver() {
    let diags = borrow_diags(
        r#"
struct LockStore {
    active: bool,
}

impl LockStore {
    fn is_active(self) -> bool {
        self.active
    }
}

struct RegistryStore {
    lock_store: LockStore,
}

impl RegistryStore {
    fn check(mut self) -> bool {
        self.lock_store.is_active()
    }
}
"#,
    );

    assert!(!diags.contains("move conflict"), "{diags:?}");
    assert!(
        !diags.contains("internal borrow checking error"),
        "{diags:?}"
    );
}

#[test]
fn nested_copy_projection_from_view_param_remains_allowed() {
    let diags = borrow_diags(
        r#"
struct Wrapper {
    p: Pair,
}

struct Pair {
    x: u32,
    y: u32,
}

fn read_x(w: Wrapper) -> u32 {
    w.p.x
}
"#,
    );

    assert!(!diags.contains("move conflict"), "{diags:?}");
    assert!(
        !diags.contains("internal borrow checking error"),
        "{diags:?}"
    );
}

#[test]
fn non_copy_projection_move_does_not_report_conflict() {
    let diags = borrow_diags(
        r#"
struct E {}
struct Inner {}
struct Container {
    value: Inner,
}

fn sink(_ value: own Inner, _ e: mut E) {}

impl Container {
    fn enc(own self, e: mut E) {
        sink(self.value, mut e)
    }
}
"#,
    );

    assert!(!diags.contains("move conflict"), "{diags:?}");
    assert!(
        !diags.contains("internal borrow checking error"),
        "{diags:?}"
    );
}

#[test]
fn generic_tuple_projection_move_does_not_report_conflict() {
    let diags = borrow_diags(
        r#"
struct E {}

fn sink<T>(_ value: own T, _ e: mut E) {}

trait Enc {
    fn enc(own self, e: mut E)
}

impl<T0> Enc for (T0,) {
    fn enc(own self, e: mut E) {
        sink<T0>(self.0, mut e)
    }
}
"#,
    );

    assert!(!diags.contains("move conflict"), "{diags:?}");
    assert!(
        !diags.contains("internal borrow checking error"),
        "{diags:?}"
    );
}

#[test]
fn enum_variant_test_does_not_consume_owned_value() {
    let diags = borrow_diags(
        r#"
fn decode(word: u256) -> u64 {
    if let Option::Some(value) = word.downcast() {
        return value
    }
    0
}
"#,
    );

    assert!(!diags.contains("move conflict"), "{diags:?}");
    assert!(
        !diags.contains("internal borrow checking error"),
        "{diags:?}"
    );
}

#[test]
fn nested_owned_enum_match_does_not_report_move_conflict() {
    let diags = borrow_diags(
        r#"
enum Inner {
    Unit,
    Value(u8),
}

enum Outer {
    First(Inner),
    Second(u8),
}

fn read(outer: own Outer) -> u8 {
    match outer {
        Outer::First(Inner::Unit) => 0
        Outer::First(Inner::Value(x)) => x
        Outer::Second(y) => y
    }
}
"#,
    );

    assert!(!diags.contains("move conflict"), "{diags:?}");
    assert!(
        !diags.contains("internal borrow checking error"),
        "{diags:?}"
    );
}

#[test]
fn multi_field_owned_enum_match_does_not_report_move_conflict() {
    let diags = borrow_diags(
        r#"
struct Boxed {}

enum Pair {
    Both(Boxed, Boxed),
}

fn take(_ value: own Boxed) {}

fn read(pair: own Pair) {
    match pair {
        Pair::Both(lhs, rhs) => {
            take(lhs)
            take(rhs)
        }
    }
}
"#,
    );

    assert!(!diags.contains("move conflict"), "{diags:?}");
    assert!(
        !diags.contains("internal borrow checking error"),
        "{diags:?}"
    );
}

#[test]
fn constant_folding_does_not_erase_non_copy_moves() {
    let diags = borrow_diags(
        r#"
use core::option::Option

struct Boxed {
    value: u256,
}

struct Holder {
    value: Boxed,
}

const fn consume(_ value: own Boxed) -> u256 {
    value.value
}

fn move_direct_twice() {
    let original = Boxed { value: 21 }
    let first = original
    let second = original
    assert(first.value + second.value == 42)
}

fn move_into_aggregate_twice() {
    let original = Boxed { value: 21 }
    let first = Holder { value: original }
    let second = Holder { value: original }
    assert(first.value.value + second.value.value == 42)
}

fn match_local_twice() {
    let wrapped: Option<Boxed> = Option::Some(Boxed { value: 21 })
    let first = match wrapped {
        Option::Some(value) => value,
        Option::None => Boxed { value: 0 },
    }
    let second = match wrapped {
        Option::Some(value) => value,
        Option::None => Boxed { value: 0 },
    }
    assert(first.value + second.value == 42)
}

fn if_let_local_twice() {
    let wrapped: Option<Boxed> = Option::Some(Boxed { value: 21 })
    if let Option::Some(value) = wrapped {
        assert(value.value == 21)
    }
    if let Option::Some(value) = wrapped {
        assert(value.value == 21)
    }
}

fn nested_match_local_twice() {
    let wrapped: Option<Holder> = Option::Some(Holder {
        value: Boxed { value: 21 },
    })
    let first = match wrapped {
        Option::Some(Holder { value }) => value,
        Option::None => Boxed { value: 0 },
    }
    let second = match wrapped {
        Option::Some(Holder { value }) => value,
        Option::None => Boxed { value: 0 },
    }
    assert(first.value + second.value == 42)
}

fn const_call_local_twice() {
    let original = Boxed { value: 21 }
    let first = consume(original)
    let second = consume(original)
    assert(first + second == 42)
}

fn while_let_reuses_local() {
    let wrapped: Option<Boxed> = Option::Some(Boxed { value: 21 })
    while let Option::Some(value) = wrapped {
        assert(value.value == 21)
    }
}

fn move_constant_while_borrowed() {
    let original = Boxed { value: 21 }
    let borrowed: ref Boxed = ref original
    let moved = original
    assert(borrowed.value + moved.value == 42)
}
"#,
    );

    for name in [
        "move_direct_twice",
        "move_into_aggregate_twice",
        "match_local_twice",
        "if_let_local_twice",
        "nested_match_local_twice",
        "const_call_local_twice",
        "while_let_reuses_local",
    ] {
        assert!(
            diags.contains(&format!("move conflict in `fn {name}`")),
            "missing move conflict for {name}: {diags:?}"
        );
    }
    assert!(
        diags.contains("borrow conflict in `fn move_constant_while_borrowed`")
            && diags.contains("cannot move out of a value while it is borrowed"),
        "{diags:?}"
    );
}

#[test]
fn constant_folded_single_moves_and_copy_reads_remain_valid() {
    let diags = borrow_diags(
        r#"
use core::option::Option

struct Boxed {
    value: u256,
}

fn move_direct_once() -> u256 {
    let original = Boxed { value: 21 }
    let moved = original
    moved.value
}

fn match_local_once() -> u256 {
    let wrapped: Option<Boxed> = Option::Some(Boxed { value: 21 })
    let moved = match wrapped {
        Option::Some(value) => value,
        Option::None => Boxed { value: 0 },
    }
    moved.value
}

fn inspect_tag_twice() -> u256 {
    let wrapped: Option<Boxed> = Option::Some(Boxed { value: 21 })
    let first = match wrapped {
        Option::Some(_) => 1,
        Option::None => 0,
    }
    let second = match wrapped {
        Option::Some(_) => 1,
        Option::None => 0,
    }
    first + second
}

fn copy_scrutinee_twice() -> u256 {
    let original: u256 = 21
    let first = match original {
        value => value,
    }
    let second = match original {
        value => value,
    }
    first + second
}

fn borrowed_match_twice() -> u256 {
    let wrapped: Option<Boxed> = Option::Some(Boxed { value: 21 })
    let borrowed: ref Option<Boxed> = ref wrapped
    let first = match borrowed {
        Option::Some(value) => value.value,
        Option::None => 0,
    }
    let second = match borrowed {
        Option::Some(value) => value.value,
        Option::None => 0,
    }
    first + second
}

fn reassign_after_match() -> u256 {
    let mut wrapped: Option<Boxed> = Option::Some(Boxed { value: 20 })
    let first = match wrapped {
        Option::Some(value) => value,
        Option::None => Boxed { value: 0 },
    }
    wrapped = Option::Some(Boxed { value: 22 })
    let second = match wrapped {
        Option::Some(value) => value,
        Option::None => Boxed { value: 0 },
    }
    first.value + second.value
}

fn borrow_ends_before_move() -> u256 {
    let original = Boxed { value: 21 }
    let borrowed: ref Boxed = ref original
    let value = borrowed.value
    let moved = original
    value + moved.value
}
"#,
    );

    assert!(!diags.contains("move conflict"), "{diags:?}");
    assert!(
        !diags.contains("internal borrow checking error"),
        "{diags:?}"
    );
}

#[test]
fn copy_pattern_bindings_do_not_move_non_copy_containers() {
    let diags = borrow_diags(
        r#"
struct Boxed {
    value: u256,
}

struct MixedStruct {
    copy: u256,
    owned: Boxed,
}

enum MixedEnum {
    Value(u256, Boxed),
}

fn consume(_ value: own Boxed) {}

fn match_enum_twice(mixed: own MixedEnum) -> u256 {
    let first = match mixed {
        MixedEnum::Value(value, _) => value,
    }
    let second = match mixed {
        MixedEnum::Value(value, _) => value,
    }
    first + second
}

fn if_let_enum_twice(mixed: own MixedEnum) -> u256 {
    let mut total: u256 = 0
    if let MixedEnum::Value(value, _) = mixed {
        total += value
    }
    if let MixedEnum::Value(value, _) = mixed {
        total += value
    }
    total
}

fn wildcarded_struct_field_remains_available(mixed: own MixedStruct) -> u256 {
    let MixedStruct { copy, owned: _ } = mixed
    copy + mixed.copy
}

fn copied_struct_field_remains_available_after_sibling_move(
    mixed: own MixedStruct,
) -> u256 {
    let MixedStruct { copy, owned } = mixed
    consume(owned)
    copy + mixed.copy
}

fn binding_free_projected_place_remains_available(mixed: own MixedStruct) -> u256 {
    let _ = mixed.owned
    let marker = match mixed.owned {
        _ => 1,
    }
    marker + mixed.owned.value
}

fn block_wrapped_noncopy_binding_moves_once(mixed: own MixedStruct) {
    let owned = { mixed.owned }
    consume(owned)
}
"#,
    );

    assert!(!diags.contains("move conflict"), "{diags:?}");
    assert!(
        !diags.contains("internal borrow checking error"),
        "{diags:?}"
    );
}

#[test]
fn effect_handle_field_deref_fixture_does_not_report_semantic_borrow_errors() {
    let diags = borrow_diags(include_str!(
        "../../codegen/tests/fixtures/effect_handle_field_deref.fe"
    ));
    assert!(!diags.contains("borrow conflict"), "{diags:?}");
    assert!(!diags.contains("move conflict"), "{diags:?}");
    assert!(
        !diags.contains("internal borrow checking error"),
        "{diags:?}"
    );
}

#[test]
fn root_object_direct_values_preserve_provider_roots_in_normalized_borrowck() {
    let mut db = HirAnalysisTestDb::default();
    let file = db.new_stand_alone(
        "semantic_borrowck.fe".into(),
        r#"
use std::evm::{Address, StorageMap}

struct TokenStore {
    balances: StorageMap<Address, u256>,
}

fn read_balance(addr: Address) -> u256 uses (store: TokenStore) {
    let balance = store.balances.get(key: addr)
    balance
}
"#,
    );
    let (top_mod, _) = db.top_mod(file);
    let instance = top_mod
        .all_items(&db)
        .iter()
        .find_map(|item| match item {
            ItemKind::Func(func)
                if func
                    .name(&db)
                    .to_opt()
                    .is_some_and(|name| name.data(&db) == "read_balance") =>
            {
                Some(get_or_build_semantic_instance(
                    &db,
                    identity_semantic_instance_key(&db, BodyOwner::Func(*func)),
                ))
            }
            _ => None,
        })
        .expect("read_balance instance");
    if let Err(diag) = check_semantic_borrows(&db, instance) {
        panic!("{diag:?}");
    }
    let normalized = normalize_semantic_body(&db, instance).expect("normalized body");
    let store_local = normalized
        .locals
        .iter()
        .enumerate()
        .find_map(|(idx, local)| match local.source {
            Some(fe_hir::analysis::ty::ty_check::LocalBinding::EffectParam { .. }) => {
                Some((idx, local))
            }
            _ => None,
        })
        .expect("store effect binding");
    let root = match &store_local.1.lowering {
        NormalizedBindingLowering::ValueLocal { place } => place
            .root
            .borrow_root()
            .expect("store binding should preserve a borrow root"),
        ref lowering => panic!("unexpected lowering for store binding: {lowering:?}"),
    };
    let Some(NBorrowRoot::Provider { value_ty, .. }) = normalized.root(root) else {
        panic!(
            "expected provider root for store binding, got {:?}",
            normalized.root(root)
        );
    };
    assert_eq!(*value_ty, store_local.1.layout_ty());
    assert_eq!(
        store_local.1.facts.interface,
        SemanticLocalKind::DirectValue
    );
    assert!(matches!(
        store_local.1.facts.origin,
        NLocalOrigin::RootProvider(_)
    ));
    assert!(store_local.1.snapshot_source_place().is_some());
    let field_local = normalized
        .locals
        .get(3)
        .expect("field projection temp should exist");
    let root = match &field_local.lowering {
        NormalizedBindingLowering::ValueLocal { place } => place
            .root
            .borrow_root()
            .expect("field projection should preserve a local root"),
        ref lowering => panic!("unexpected lowering for provider field temp: {lowering:?}"),
    };
    assert!(
        matches!(
            normalized.root(root),
            Some(NBorrowRoot::LocalSlot { local }) if *local == fe_hir::analysis::semantic::SLocalId::from_u32(3)
        ),
        "expected self-rooted local slot for provider field temp, got {:?}",
        normalized.root(root)
    );
    assert_eq!(field_local.facts.interface, SemanticLocalKind::DirectValue);
    assert!(matches!(field_local.facts.origin, NLocalOrigin::SelfRooted));
    let backing_place = field_local
        .backing_place()
        .expect("field projection temp should keep its own backing place");
    let backing_root = backing_place
        .root
        .borrow_root()
        .expect("field projection backing root");
    assert!(
        matches!(
            normalized.root(backing_root),
            Some(NBorrowRoot::LocalSlot { local }) if *local == fe_hir::analysis::semantic::SLocalId::from_u32(3)
        ),
        "expected self-rooted backing place for provider field temp, got {:?}",
        normalized.root(backing_root)
    );
    assert!(backing_place.path.is_empty());
    let snapshot_source = field_local
        .snapshot_source_place()
        .expect("field projection temp should preserve its source place");
    let snapshot_root = snapshot_source
        .root
        .borrow_root()
        .expect("field projection snapshot source root");
    let Some(NBorrowRoot::Provider { value_ty, .. }) = normalized.root(snapshot_root) else {
        panic!(
            "expected provider-root snapshot source for provider field temp, got {:?}",
            normalized.root(snapshot_root)
        );
    };
    assert_eq!(*value_ty, store_local.1.layout_ty());
    assert_eq!(
        snapshot_source.path.iter().next(),
        Some(&fe_hir::projection::Projection::Field(0))
    );
    assert!(!field_local.facts.root_demand.needs_runtime_root());
}

#[test]
fn ref_projection_preserves_place_borrow_lowering() {
    let mut db = HirAnalysisTestDb::default();
    let file = db.new_stand_alone(
        "semantic_borrowck.fe".into(),
        r#"
struct Pair {
    x: u256,
}

fn read(pair: Pair) -> u256 {
    let r: ref u256 = ref pair.x
    r
}
"#,
    );
    let (top_mod, _) = db.top_mod(file);
    let instance = top_mod
        .all_items(&db)
        .iter()
        .find_map(|item| match item {
            ItemKind::Func(func)
                if func
                    .name(&db)
                    .to_opt()
                    .is_some_and(|name| name.data(&db) == "read") =>
            {
                Some(get_or_build_semantic_instance(
                    &db,
                    identity_semantic_instance_key(&db, BodyOwner::Func(*func)),
                ))
            }
            _ => None,
        })
        .expect("read instance");
    let normalized = normalize_semantic_body(&db, instance).expect("normalized body");
    let borrow = normalized
        .blocks
        .iter()
        .flat_map(|block| block.stmts.iter())
        .find_map(|stmt| match &stmt.kind {
            NSStmtKind::Assign {
                expr:
                    NExpr::Borrow {
                        place,
                        kind: BorrowKind::Ref,
                        ..
                    },
                ..
            } => Some(place),
            _ => None,
        })
        .expect("borrow expression");
    assert!(
        matches!(borrow.root, NSPlaceRoot::CarrierDerefLocal(local) if normalized.local(local).is_some_and(|local| {
            matches!(
                local.source,
                Some(fe_hir::analysis::ty::ty_check::LocalBinding::Param { .. })
            )
        })),
        "expected carrier-rooted view param place for ref projection, got {:?}",
        borrow.root
    );
    assert_eq!(borrow.path.len(), 1);
    assert_eq!(
        borrow.path.iter().next(),
        Some(&fe_hir::projection::Projection::Field(0))
    );
}

#[test]
fn projected_direct_value_snapshots_keep_lineage_without_reviving_aliases() {
    let mut db = HirAnalysisTestDb::default();
    let file = db.new_stand_alone(
        "semantic_borrowck.fe".into(),
        r#"
struct Pair {
    x: u256,
}

struct Wrapper {
    pair: Pair,
}

fn read(wrapper: own Wrapper) -> u256 {
    let pair = wrapper.pair
    let copy = pair
    let r: ref Pair = ref copy
    r.x
}
"#,
    );
    let (top_mod, _) = db.top_mod(file);
    let instance = top_mod
        .all_items(&db)
        .iter()
        .find_map(|item| match item {
            ItemKind::Func(func)
                if func
                    .name(&db)
                    .to_opt()
                    .is_some_and(|name| name.data(&db) == "read") =>
            {
                Some(get_or_build_semantic_instance(
                    &db,
                    identity_semantic_instance_key(&db, BodyOwner::Func(*func)),
                ))
            }
            _ => None,
        })
        .expect("read instance");
    let normalized = normalize_semantic_body(&db, instance).expect("normalized body");
    let pair_ty = normalized
        .locals
        .iter()
        .find(|local| {
            matches!(
                local.source,
                Some(fe_hir::analysis::ty::ty_check::LocalBinding::Local { .. })
            ) && local.ty.is_struct(&db)
        })
        .map(|local| local.ty)
        .expect("pair locals should exist");
    let locals = normalized
        .locals
        .iter()
        .enumerate()
        .filter_map(|(idx, local)| match local.source {
            Some(fe_hir::analysis::ty::ty_check::LocalBinding::Local { .. })
                if local.ty == pair_ty =>
            {
                Some((
                    fe_hir::analysis::semantic::SLocalId::from_u32(idx as u32),
                    local,
                ))
            }
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(
        locals.len(),
        2,
        "expected pair/copy locals, got {locals:#?}"
    );
    let (pair_local_id, pair_local) = locals[0];
    let (copy_local_id, copy_local) = locals[1];

    for (local_id, local) in [(pair_local_id, pair_local), (copy_local_id, copy_local)] {
        assert_eq!(local.facts.interface, SemanticLocalKind::DirectValue);
        assert!(matches!(local.facts.origin, NLocalOrigin::SelfRooted));
        let backing_place = local
            .backing_place()
            .expect("projected snapshot should keep a backing place");
        let backing_root = backing_place
            .root
            .borrow_root()
            .expect("projected snapshot backing root");
        assert!(
            matches!(
                normalized.root(backing_root),
                Some(NBorrowRoot::LocalSlot { local: root_local }) if *root_local == local_id
            ),
            "expected self-rooted backing place for {local_id:?}, got {:?}",
            normalized.root(backing_root)
        );
        assert!(backing_place.path.is_empty());
    }

    let pair_snapshot = pair_local
        .snapshot_source_place()
        .expect("projected snapshot should preserve source lineage");
    let pair_snapshot_root = pair_snapshot
        .root
        .borrow_root()
        .expect("projected snapshot source root");
    assert!(
        matches!(
            normalized.root(pair_snapshot_root),
            Some(NBorrowRoot::Param { .. })
        ),
        "expected param-root snapshot lineage for projected local, got {:?}",
        normalized.root(pair_snapshot_root)
    );
    assert_eq!(
        pair_snapshot.path.iter().next(),
        Some(&fe_hir::projection::Projection::Field(0))
    );

    let copy_snapshot = copy_local
        .snapshot_source_place()
        .expect("forwarded snapshot should preserve source lineage");
    assert_eq!(copy_snapshot, pair_snapshot);

    let borrow = normalized
        .blocks
        .iter()
        .flat_map(|block| block.stmts.iter())
        .find_map(|stmt| match &stmt.kind {
            NSStmtKind::Assign {
                expr:
                    NExpr::Borrow {
                        place,
                        kind: BorrowKind::Ref,
                        ..
                    },
                ..
            } => Some(place),
            _ => None,
        })
        .expect("borrow expression");
    let borrow_root = borrow.root.borrow_root().expect("borrow root");
    assert!(
        matches!(
            normalized.root(borrow_root),
            Some(NBorrowRoot::LocalSlot { local }) if *local == copy_local_id
        ),
        "expected borrow of copied snapshot to use its own local root, got {:?}",
        normalized.root(borrow_root)
    );
    assert!(borrow.path.is_empty());
}

#[test]
fn nested_place_reads_normalize_as_one_composite_place() {
    let mut db = HirAnalysisTestDb::default();
    let file = db.new_stand_alone(
        "semantic_borrowck.fe".into(),
        r#"
struct Table {
    used: [u8; 4],
    keys: [u256; 4],
    values: [u256; 4],
}

impl Table {
    fn get_used(self, _ slot: usize) -> u8 {
        self.used[slot]
    }

    fn get_keys(self, _ slot: usize) -> u256 {
        self.keys[slot]
    }

    fn get_values(self, _ slot: usize) -> u256 {
        self.values[slot]
    }
}
"#,
    );
    let (top_mod, _) = db.top_mod(file);
    for (name, field, elem_ty) in [
        ("get_used", 0, "u8"),
        ("get_keys", 1, "u256"),
        ("get_values", 2, "u256"),
    ] {
        let normalized = normalized_func_body(&db, top_mod, name);
        let mut saw_nested_read = false;
        for stmt in normalized
            .blocks
            .iter()
            .flat_map(|block| block.stmts.iter())
        {
            let NSStmtKind::Assign {
                dst,
                expr: NExpr::ReadPlace { place, .. },
            } = &stmt.kind
            else {
                continue;
            };
            let local = &normalized.locals[dst.index()];
            if local.ty.pretty_print(&db) == elem_ty {
                assert!(
                    matches!(
                        place.path.iter().cloned().collect::<Vec<_>>().as_slice(),
                        [
                            Projection::Field(path_field),
                            Projection::Index(IndexSource::Dynamic(_))
                        ] if *path_field == field
                    ),
                    "unexpected nested place path in {name}: {:?}",
                    place.path
                );
                saw_nested_read = true;
            }
            assert!(
                !(local.ty.array_len(&db).is_some()
                    && place.path.iter().cloned().collect::<Vec<_>>()
                        == vec![Projection::Field(field)]),
                "unexpected intermediate whole-array read in {name}: {stmt:?}"
            );
        }
        assert!(
            saw_nested_read,
            "missing nested array element read in {name}"
        );
    }
}

#[test]
fn owned_aggregate_value_boundaries_project_from_the_owned_local() {
    let mut db = HirAnalysisTestDb::default();
    let file = db.new_stand_alone(
        "semantic_borrowck.fe".into(),
        r#"
struct Table {
    used: [u8; 4],
}

impl Table {
    fn get(own self, _ slot: usize) -> u8 {
        let used = self.used
        used[slot]
    }
}
"#,
    );
    let (top_mod, _) = db.top_mod(file);
    let normalized = normalized_func_body(&db, top_mod, "get");

    let (used_local_id, used_local) = normalized
        .locals
        .iter()
        .enumerate()
        .find_map(|(idx, local)| match local.source {
            Some(fe_hir::analysis::ty::ty_check::LocalBinding::Local { .. })
                if local.ty.array_len(&db).is_some() =>
            {
                Some((
                    fe_hir::analysis::semantic::SLocalId::from_u32(idx as u32),
                    local,
                ))
            }
            _ => None,
        })
        .expect("owned array local");

    assert_eq!(used_local.facts.interface, SemanticLocalKind::DirectValue);
    assert!(matches!(used_local.facts.origin, NLocalOrigin::SelfRooted));
    let backing_place = used_local
        .backing_place()
        .expect("owned aggregate local should keep backing storage");
    let backing_root = backing_place.root.borrow_root().expect("backing root");
    assert!(
        matches!(
            normalized.root(backing_root),
            Some(NBorrowRoot::LocalSlot { local }) if *local == used_local_id
        ),
        "expected owned aggregate backing root to be the local itself, got {:?}",
        normalized.root(backing_root)
    );
    assert!(backing_place.path.is_empty());
    let snapshot_source = used_local
        .snapshot_source_place()
        .expect("owned aggregate should preserve lineage");
    let snapshot_root = snapshot_source.root.borrow_root().expect("snapshot root");
    assert!(
        matches!(
            normalized.root(snapshot_root),
            Some(NBorrowRoot::Param { .. })
        ),
        "expected source lineage to point at the parameter root, got {:?}",
        normalized.root(snapshot_root)
    );
    assert_eq!(
        snapshot_source.path.iter().cloned().collect::<Vec<_>>(),
        vec![Projection::Field(0)]
    );

    let element_read = normalized
        .blocks
        .iter()
        .flat_map(|block| block.stmts.iter())
        .find_map(|stmt| match &stmt.kind {
            NSStmtKind::Assign {
                dst,
                expr: NExpr::ReadPlace { place, .. },
            } if normalized.locals[dst.index()].ty.pretty_print(&db) == "u8" => Some(place),
            _ => None,
        })
        .expect("element read");
    let read_root = element_read.root.borrow_root().expect("element read root");
    assert!(
        matches!(
            normalized.root(read_root),
            Some(NBorrowRoot::LocalSlot { local }) if *local == used_local_id
        ),
        "expected owned aggregate projection to read from the owned local, got {:?}",
        normalized.root(read_root)
    );
    assert!(
        matches!(
            element_read
                .path
                .iter()
                .cloned()
                .collect::<Vec<_>>()
                .as_slice(),
            [Projection::Index(IndexSource::Dynamic(_))]
        ),
        "unexpected owned-local projection path: {:?}",
        element_read.path
    );
}

#[test]
fn zero_sized_aggregate_fixture_instances_normalize_and_borrowcheck() {
    for_each_fixture_instance(
        include_str!("../../codegen/tests/fixtures/zero_sized_aggregates.fe"),
        |db, instance| {
            if let Err(err) = normalize_semantic_body(db, instance) {
                panic!(
                    "normalize failed for {} ({:?}): {err:?}",
                    owner_name(db, instance.key(db).owner(db)),
                    instance.key(db),
                );
            }
            if let Err(diag) = check_semantic_borrows(db, instance) {
                panic!(
                    "borrowck failed for {} ({:?}): {diag:#?}",
                    owner_name(db, instance.key(db).owner(db)),
                    instance.key(db),
                );
            }
        },
    );
}

#[test]
fn decompose_ty_app_handles_deep_ty_app_chains_iteratively() {
    let db = HirAnalysisTestDb::default();
    let arg = fe_hir::analysis::ty::ty_def::TyId::u256(&db);
    let mut ty = fe_hir::analysis::ty::ty_def::TyId::bool(&db);
    for _ in 0..10_000 {
        ty = fe_hir::analysis::ty::ty_def::TyId::new(&db, TyData::TyApp(ty, arg));
    }
    assert_eq!(
        ty.base_ty(&db),
        fe_hir::analysis::ty::ty_def::TyId::bool(&db)
    );
    assert_eq!(ty.generic_args(&db).len(), 10_000);
}

#[test]
fn erc20_has_role_self_ty_app_chain_is_acyclic() {
    let mut db = HirAnalysisTestDb::default();
    let file = db.new_stand_alone(
        "semantic_borrowck.fe".into(),
        include_str!("../../codegen/tests/fixtures/erc20.fe"),
    );
    let (top_mod, _) = db.top_mod(file);
    let has_role = top_mod
        .all_items(&db)
        .iter()
        .find_map(|item| match item {
            ItemKind::Func(func)
                if func
                    .name(&db)
                    .to_opt()
                    .is_some_and(|name| name.data(&db) == "has_role") =>
            {
                Some(func)
            }
            _ => None,
        })
        .expect("has_role fixture function");
    let instance = get_or_build_semantic_instance(
        &db,
        identity_semantic_instance_key(&db, BodyOwner::Func(*has_role)),
    );
    let ty = instance.body(&db).locals[0].ty;
    let mut seen = rustc_hash::FxHashSet::default();
    let mut cursor = ty;
    loop {
        assert!(seen.insert(cursor), "cyclic ty app chain at {:?}", cursor);
        match cursor.data(&db) {
            TyData::TyApp(lhs, _) => cursor = *lhs,
            _ => break,
        }
    }
}

#[test]
fn array_of_struct_place_lowers_with_resolved_index_then_field() {
    let mut db = HirAnalysisTestDb::default();
    let file = db.new_stand_alone(
        "semantic_borrowck.fe".into(),
        r#"
struct Subtree {
    left: u256,
    right: u256,
}

struct Tree {
    last_subtrees: [Subtree; 8],
}

fn write(mut tree: Tree, i: usize, h: u256) -> Tree {
    tree.last_subtrees[i].left = h
    tree
}
"#,
    );
    let (top_mod, _) = db.top_mod(file);
    let instance = top_mod
        .all_items(&db)
        .iter()
        .find_map(|item| match item {
            ItemKind::Func(func)
                if func
                    .name(&db)
                    .to_opt()
                    .is_some_and(|name| name.data(&db) == "write") =>
            {
                Some(get_or_build_semantic_instance(
                    &db,
                    identity_semantic_instance_key(&db, BodyOwner::Func(*func)),
                ))
            }
            _ => None,
        })
        .expect("write instance");
    let body = instance.body(&db);
    let dst = body
        .blocks
        .iter()
        .flat_map(|block| block.stmts.iter())
        .find_map(|stmt| match &stmt.kind {
            SStmtKind::Store { dst, .. } => Some(dst),
            SStmtKind::Assign { .. } => None,
        })
        .expect("store statement");

    assert_eq!(dst.path.len(), 3);
    let path = dst.path.iter().collect::<Vec<_>>();
    assert!(matches!(path[0], Projection::Field(0)));
    assert!(matches!(path[1], Projection::Index(_)));
    assert!(matches!(path[2], Projection::Field(0)));
}
