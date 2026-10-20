use fe_hir::analysis::semantic::{
    DropStep, drop_elaboration, get_or_build_semantic_instance, identity_semantic_instance_key,
};
use fe_hir::analysis::ty::ty_check::BodyOwner;
use fe_hir::test_db::HirAnalysisTestDb;

/// Each drop of function `name`, in statement order: its parts as paths
/// (`.n` a field, `:v.n` a variant's field, `[i]` an element, `*` the whole
/// target), each followed by `?` and its flags when only some paths reach
/// it initialized, or `-` for a drop with nothing to destroy.
fn drops(source: &str, name: &str) -> Vec<String> {
    let mut db = HirAnalysisTestDb::default();
    let file = db.new_stand_alone("drops.fe".into(), source);
    let (top_mod, _) = db.top_mod(file);
    db.assert_no_diags(top_mod);
    let func = top_mod
        .all_funcs(&db)
        .iter()
        .copied()
        .find(|func| {
            func.name(&db)
                .to_opt()
                .is_some_and(|ident| ident.data(&db) == name)
        })
        .unwrap_or_else(|| panic!("no function `{name}`"));
    let key = identity_semantic_instance_key(&db, BodyOwner::Func(func, None));
    let instance = get_or_build_semantic_instance(&db, key);
    let elaboration = drop_elaboration(&db, instance).expect("a checked body");
    elaboration
        .drops
        .iter()
        .map(|(_, parts)| {
            if parts.is_empty() {
                return "-".to_string();
            }
            parts
                .iter()
                .map(|part| {
                    let mut text: String = part
                        .path
                        .iter()
                        .map(|step| match step {
                            DropStep::Field(field) => format!(".{field}"),
                            DropStep::Variant { variant, field } => format!(":{variant}.{field}"),
                            DropStep::Index(index) => format!("[{index}]"),
                        })
                        .collect();
                    if text.is_empty() {
                        text.push('*');
                    }
                    if !part.flags.is_empty() {
                        text.push('?');
                        text.push_str(
                            &part
                                .flags
                                .iter()
                                .map(u32::to_string)
                                .collect::<Vec<_>>()
                                .join(","),
                        );
                    }
                    text
                })
                .collect::<Vec<_>>()
                .join(" ")
        })
        .collect()
}

const PRELUDE: &str = r#"
use core::ops::Drop
use core::Option

struct Buf { id: u64 }
impl Drop for Buf { fn drop(own self) {} }

struct Pair { a: Buf, b: Buf }

fn make(_ id: u64) -> Buf { Buf { id } }
fn consume(_ b: own Buf) -> u64 { b.id }
fn view(_ b: Buf) -> u64 { b.id }
"#;

fn check(body: &str, name: &str, expected: &[&str]) {
    assert_eq!(drops(&format!("{PRELUDE}{body}"), name), expected);
}

#[test]
fn bindings_drop_at_scope_end_unless_moved() {
    check(
        "fn f() -> u64 { let a = make(1)\n let b = make(2)\n consume(a) + view(b) }",
        "f",
        // The temporaries, the argument, `b`, and the moved `a`.
        &["-", "-", "-", "*", "-"],
    );
}

#[test]
fn a_conditional_move_is_flagged() {
    check(
        "fn f(_ c: bool) { let a = make(1)\n if c { consume(a) } }",
        "f",
        &["-", "-", "*?0"],
    );
}

#[test]
fn a_partial_move_drops_the_rest() {
    check(
        "fn f(_ p: own Pair) -> u64 { let a = p.a\n consume(a) }",
        "f",
        &["-", "-", "-", ".1"],
    );
}

#[test]
fn reassignment_drops_the_old_value() {
    check(
        "fn f() { var x = make(1)\n x = make(2)\n consume(x) }",
        "f",
        // The first value's temporary, `x` before the store, the second
        // value's temporary, the argument and `x` at the end.
        &["-", "*", "-", "-", "-"],
    );
}

#[test]
fn a_returned_binding_is_not_dropped() {
    check(
        "fn f(_ c: bool) -> Buf { let a = make(1)\n let b = make(2)\n if c { return a }\n b }",
        "f",
        // The temporaries, `b` where `a` returns, and `a` where `b` does.
        &["-", "-", "*", "*"],
    );
}

#[test]
fn a_matched_payload_moved_in_one_arm_is_flagged() {
    check(
        "fn f(_ o: own Option<Buf>) -> u64 { match o { Option::Some(b) => consume(b)\n Option::None => 0 } }",
        "f",
        &["-", "-", ":1.0?0"],
    );
}

#[test]
fn types_without_drop_glue_are_not_dropped() {
    check(
        "struct Plain { x: u64 }\nfn plain() -> Plain { Plain { x: 1 } }\nfn f() -> u64 { let p = plain()\n p.x }",
        "f",
        &[],
    );
}
