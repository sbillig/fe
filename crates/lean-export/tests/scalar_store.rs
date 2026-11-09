use fe_lean_export::export_contract;
use hir::test_db::HirAnalysisTestDb;

#[test]
fn scalar_store_arms_export() {
    let mut db = HirAnalysisTestDb::default();
    let file = db.new_stand_alone(
        "scalar_store.fe".into(),
        r#"use std::abi::sol

msg FeScalarStoreMsg {
    #[selector = sol("get()")]
    Get -> u256,
    #[selector = sol("set(uint256)")]
    Set { value: u256 },
}

pub contract FeScalarStore {
    mut stored: u256

    init(initial: u256) uses (mut stored) {
        stored = initial
    }

    recv FeScalarStoreMsg {
        Get -> u256 uses (stored) {
            stored
        }

        Set { value } uses (mut stored) {
            stored = value
        }
    }
}"#,
    );
    let (top_mod, _) = db.top_mod(file);
    db.assert_no_diags(top_mod);
    let [contract] = top_mod.all_contracts(&db).as_slice() else {
        panic!("expected one contract");
    };
    let lean = export_contract(&db, *contract).expect("exported");
    assert_eq!(
        lean,
        "def init (v0 : UInt256) : FeM Unit := do
  let v1 ← pure v0
  sstore (UInt256.ofNat 0) v1
  let v2 ← pure ()
  return ()

def recv0_arm0 : FeM UInt256 := do
  let v0 ← sload (UInt256.ofNat 0)
  return v0

def recv0_arm1 (v0 : UInt256) : FeM Unit := do
  sstore (UInt256.ofNat 0) v0
  let v1 ← pure ()
  return ()

"
    );
}
