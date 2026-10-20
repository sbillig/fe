use common::InputDb;
use driver::DriverDataBase;
use fe_codegen::emit_module_sonatina_ir;
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
    let counted =
        evm_ir("impl Drop for Token { fn drop(own self) { unsafe { *self.counter += 1 } } }");
    assert_ne!(counted, none);
    assert!(counted.contains("drop"), "{counted}");
}
