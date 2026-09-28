use fe_hir::{
    analysis::{
        semantic::{
            ImplEnv, SemanticInstanceKey, check_semantic_borrows, get_or_build_semantic_instance,
            identity_semantic_instance_key, semantic_borrow_summary,
        },
        ty::ty_check::BodyOwner,
    },
    hir_def::TopLevelMod,
    test_db::{HirAnalysisTestDb, find_func, format_diagnostics},
};

fn callee_key<'db>(
    db: &'db HirAnalysisTestDb,
    module: TopLevelMod<'db>,
    caller: &str,
    target: &str,
) -> SemanticInstanceKey<'db> {
    let key = identity_semantic_instance_key(db, BodyOwner::Func(find_func(db, module, caller)));
    get_or_build_semantic_instance(db, key)
        .call_sites(db)
        .iter()
        .flatten()
        .filter_map(|site| site.callee)
        .find(|callee| {
            matches!(callee.key.owner(db), BodyOwner::Func(func)
                if func.name(db).to_opt().is_some_and(|name| name.data(db) == target))
        })
        .expect("expected source call")
        .key
}

#[test]
fn inherent_calls_share_direct_keys_in_every_query_order() {
    let source = r#"
trait A {}
trait B {}
struct Machine { value: u256 }
fn free(_ value: u256) -> u256 { value }
impl Machine {
    fn leaf(_ value: u256) -> u256 { value }
    fn read(self) -> u256 { self.value }
    fn run(mut self, count: u256) {
        let mut i = 0
        while i < count {
            self.value += 1
            i += 1
        }
    }
}
fn left<T: A>(machine: mut Machine, value: u256) -> u256 {
    machine.run(count: value)
    free(value) + Machine::leaf(value) + machine.read()
}
fn right<U: B>(machine: mut Machine, value: u256) -> u256 {
    machine.run(count: value)
    free(value) + Machine::leaf(value) + machine.read()
}
"#;
    for first in ["direct", "left", "right"] {
        for summary_first in [false, true] {
            let mut db = HirAnalysisTestDb::default();
            let file = db.new_stand_alone("inherent_keys.fe".into(), source);
            let (module, _) = db.top_mod(file);
            let methods: Vec<_> = module
                .all_impls(&db)
                .iter()
                .flat_map(|implementation| implementation.funcs(&db))
                .collect();
            let initial = if first == "direct" {
                methods
                    .iter()
                    .map(|&method| identity_semantic_instance_key(&db, BodyOwner::Func(method)))
                    .collect()
            } else {
                ["leaf", "read", "run"]
                    .map(|target| callee_key(&db, module, first, target))
                    .to_vec()
            };
            for key in initial {
                let instance = get_or_build_semantic_instance(&db, key);
                if summary_first {
                    semantic_borrow_summary(&db, instance).unwrap();
                    check_semantic_borrows(&db, instance).unwrap();
                } else {
                    check_semantic_borrows(&db, instance).unwrap();
                    semantic_borrow_summary(&db, instance).unwrap();
                }
            }
            for target in ["free", "leaf", "read", "run"] {
                let left = callee_key(&db, module, "left", target);
                let right = callee_key(&db, module, "right", target);
                let direct = identity_semantic_instance_key(&db, left.owner(&db));
                assert_eq!(
                    left, direct,
                    "{first}, summary_first={summary_first}, {target}"
                );
                assert_eq!(
                    right, direct,
                    "{first}, summary_first={summary_first}, {target}"
                );
                assert_eq!(
                    get_or_build_semantic_instance(&db, left),
                    get_or_build_semantic_instance(&db, right)
                );
            }
            db.assert_no_diags(module);
        }
    }
}

#[test]
fn ground_inherent_calls_prove_parent_and_method_bounds_and_signatures() {
    let source = r#"
trait A {}
trait B {}
trait Output { type Item }
impl Output for bool { type Item = u256 }
trait Mark {}
impl Mark for u256 {}
struct Holder<T: Output> { value: T::Item }
impl<T: Output> Holder<T> {
    fn leaf<U: Mark>(_ value: own T::Item) -> T::Item { value }
}
fn left<T: A>(_ value: u256) -> u256 { Holder<bool>::leaf<u256>(value) }
fn right<U: B>(_ value: u256) -> u256 { Holder<bool>::leaf<u256>(value) }
"#;
    for first in ["left", "right"] {
        let mut db = HirAnalysisTestDb::default();
        let file = db.new_stand_alone("ground_inherent_keys.fe".into(), source);
        let (module, _) = db.top_mod(file);
        let key = callee_key(&db, module, first, "leaf");
        let expected = SemanticInstanceKey::new(
            &db,
            key.owner(&db),
            key.subst(&db),
            key.effect_providers(&db),
            ImplEnv::empty(&db, key.owner(&db).scope()),
        );
        assert_eq!(key, expected);
        assert_eq!(
            callee_key(&db, module, "left", "leaf"),
            callee_key(&db, module, "right", "leaf")
        );
        db.assert_no_diags(module);
    }
}

#[test]
fn symbolic_inherent_calls_keep_caller_evidence() {
    let source = r#"
trait Output { type Item }
struct Holder<T: Output> { value: T::Item }
impl<T: Output> Holder<T> {
    fn leaf(_ value: own T::Item) -> T::Item { value }
}
fn left<T: Output>(_ value: own T::Item) -> T::Item { Holder<T>::leaf(value) }
fn right<T: Output>(_ value: own T::Item) -> T::Item { Holder<T>::leaf(value) }
"#;
    for first in ["left", "right"] {
        let mut db = HirAnalysisTestDb::default();
        let file = db.new_stand_alone("symbolic_inherent_keys.fe".into(), source);
        let (module, _) = db.top_mod(file);
        for caller in [first, if first == "left" { "right" } else { "left" }] {
            let key = callee_key(&db, module, caller, "leaf");
            assert_eq!(
                key.impl_env(&db).normalization_scope(&db),
                find_func(&db, module, caller).scope()
            );
            assert!(!key.impl_env(&db).assumptions(&db).is_empty(&db));
        }
        assert_ne!(
            callee_key(&db, module, "left", "leaf"),
            callee_key(&db, module, "right", "leaf")
        );
        db.assert_no_diags(module);
    }
}

#[test]
fn inherent_effect_calls_keep_caller_context_and_provider_identity() {
    let source = r#"
struct Machine {}
impl Machine {
    fn leaf() -> u256 uses (value: mut u256) {
        value += 1
        value
    }
}
fn left() -> u256 {
    let mut value: u256 = 0
    with (value) { Machine::leaf() }
}
fn right() -> u256 {
    let mut value: u256 = 0
    with (value) { Machine::leaf() }
}
"#;
    for first in ["left", "right"] {
        let mut db = HirAnalysisTestDb::default();
        let file = db.new_stand_alone("effect_inherent_keys.fe".into(), source);
        let (module, _) = db.top_mod(file);
        for caller in [first, if first == "left" { "right" } else { "left" }] {
            let key = callee_key(&db, module, caller, "leaf");
            assert_eq!(
                key.impl_env(&db).normalization_scope(&db),
                find_func(&db, module, caller).scope()
            );
            assert!(!key.effect_providers(&db).providers(&db).is_empty());
        }
        assert_ne!(
            callee_key(&db, module, "left", "leaf"),
            callee_key(&db, module, "right", "leaf")
        );
        db.assert_no_diags(module);
    }
}

#[test]
fn inherent_calls_still_require_parent_and_method_bounds() {
    for (declaration, call) in [
        (
            "impl<T: Mark> Holder<T> { fn leaf() -> u256 { 0 } }",
            "Holder<bool>::leaf()",
        ),
        (
            "impl<T> Holder<T> { fn leaf<U: Mark>() -> u256 { 0 } }",
            "Holder<bool>::leaf<bool>()",
        ),
    ] {
        let source = format!(
            r#"
trait Mark {{}}
struct Holder<T> {{ value: T }}
{declaration}
fn caller() -> u256 {{ {call} }}
"#
        );
        let mut db = HirAnalysisTestDb::default();
        let file = db.new_stand_alone("missing_inherent_bounds.fe".into(), &source);
        let (module, _) = db.top_mod(file);
        let diagnostics = format_diagnostics(&db, &db.run_on_top_mod(module));
        assert!(
            diagnostics.contains("trait bound is not satisfied"),
            "{diagnostics}"
        );
        assert!(
            diagnostics.contains("`bool` doesn't implement `Mark`"),
            "{diagnostics}"
        );
    }
}

#[test]
fn inherent_calls_preserve_hidden_layout_slots() {
    let source = r#"
struct Slot<const ROOT: u256 = _> { value: u256 }
struct Machine {}
impl Machine {
    fn leaf(_ slot: Slot) -> u256 { slot.value }
}
fn left(_ slot: Slot<7>) -> u256 { Machine::leaf(slot) }
fn right(_ slot: Slot<7>) -> u256 { Machine::leaf(slot) }
"#;
    let mut db = HirAnalysisTestDb::default();
    let file = db.new_stand_alone("layout_inherent_keys.fe".into(), source);
    let (module, _) = db.top_mod(file);
    db.assert_no_diags(module);
    let left = callee_key(&db, module, "left", "leaf");
    let right = callee_key(&db, module, "right", "leaf");
    assert_eq!(left.subst(&db), right.subst(&db));
    assert_ne!(left, right);
    for (caller, key) in [("left", left), ("right", right)] {
        assert_eq!(
            key.impl_env(&db).normalization_scope(&db),
            find_func(&db, module, caller).scope()
        );
    }
}
