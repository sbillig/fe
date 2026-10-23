use common::InputDb;
use driver::DriverDataBase;
use fe_codegen::emit_module_sonatina_ir;
use mir::{RExpr, RStmt, build_runtime_package};
use url::Url;

const PROGRAM: &str = r#"
use core::ops::Drop
use core::Option

struct Token { counter: *u256 }

DROP

fn consume(_ token: own Token) {}

fn run(_ c: bool) -> u256 {
    let counter = core::ptr::alloc<u256>()
    let a = Token { counter }
    let b = Option::Some(Token { counter })
    if c {
        consume(a)
    }
    var d = Token { counter }
    d = Token { counter }
    unsafe { *counter }
}

msg Msg {
    #[selector = 0]
    Run { c: bool } -> u256,
}

contract C {
    recv Msg {
        Run { c } -> u256 {
            run(c)
        }
    }
}
"#;

fn evm_ir(drop: &str) -> String {
    let mut db = DriverDataBase::default();
    let url = Url::parse("file:///evm_drops.fe").unwrap();
    db.workspace()
        .touch(&mut db, url.clone(), Some(PROGRAM.replace("DROP", drop)));
    let file = db.workspace().get(&db, &url).unwrap();
    let top_mod = db.top_mod(file);
    emit_module_sonatina_ir(&db, top_mod).expect("Sonatina IR should emit")
}

#[test]
fn effect_free_drops_leave_evm_code_unchanged() {
    let none = evm_ir("");
    assert_eq!(evm_ir("impl Drop for Token { fn drop(own self) {} }"), none);
    // A call is an effect only if its callee has one.
    assert_eq!(
        evm_ir(
            "fn same(_ p: *u256) -> *u256 { p }\nimpl Drop for Token { fn drop(own self) { let _ = same(self.counter) } }"
        ),
        none
    );
    let counted =
        evm_ir("impl Drop for Token { fn drop(own self) { unsafe { *self.counter += 1 } } }");
    assert_ne!(counted, none);
    assert!(counted.contains("drop"), "{counted}");
}

#[test]
fn indexed_stores_check_bounds_before_destroying() {
    let mut db = DriverDataBase::default();
    let url = Url::parse("file:///evm_drops.fe").unwrap();
    db.workspace().touch(
        &mut db,
        url.clone(),
        Some(
            r#"
use core::ops::Drop

struct Token { counter: *u256 }

impl Drop for Token { fn drop(own self) { unsafe { *self.counter += 1 } } }

// Its only calls are drops.
fn store(_ i: usize, _ counter: *u256) -> u256 {
    var tokens = [Token { counter }, Token { counter }]
    tokens[i] = Token { counter }
    unsafe { *counter }
}

msg Msg {
    #[selector = 0]
    Store { i: usize } -> u256,
}

contract C {
    recv Msg {
        Store { i } -> u256 {
            store(i, core::ptr::alloc<u256>())
        }
    }
}
"#
            .to_string(),
        ),
    );
    let file = db.workspace().get(&db, &url).unwrap();
    let package = build_runtime_package(&db, db.top_mod(file)).expect("runtime package");
    let store = package
        .functions(&db)
        .iter()
        .find(|function| function.symbol(&db).contains("store"))
        .expect("runtime function `store`");
    let body = store.instance(&db).body(&db);
    // The drop of the old `tokens[i]` reads it, so `i` is checked first.
    let first = body
        .blocks
        .iter()
        .flat_map(|block| &block.stmts)
        .find(|stmt| {
            matches!(
                stmt,
                RStmt::AssertIndexInBounds { .. }
                    | RStmt::Assign {
                        expr: RExpr::Call { .. },
                        ..
                    }
            )
        });
    assert!(
        matches!(first, Some(RStmt::AssertIndexInBounds { .. })),
        "{body:#?}"
    );
}
