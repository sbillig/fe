mod load;

use fe_lean_export::{Library, export_contract, export_contract_with_entries};
use load::{FIXTURES, compare, with_fixture};

/// The last contract of each `tests/fixtures/<name>.fe`, after those it
/// deploys, or the one contract of the root module of the ingot
/// `tests/fixtures/<name>/`, exported with one library, `golden/lib.lean`.
/// Each golden file has the contract's own definitions, after the names they
/// call the library's by, for a library in a namespace `Lib`; `<name>.trust`
/// has what they take on trust, and `<name>.checks` the checks of its
/// constants, if it defines any.
#[test]
fn golden() {
    let mut library = Library::default();
    for &(name, entries) in FIXTURES {
        let export = if entries {
            export_contract_with_entries
        } else {
            export_contract
        };
        let own = with_fixture(name, |db, contract| export(db, contract, &mut library))
            .expect("exported");
        compare(
            &format!("{name}.lean"),
            &format!("{}{}", own.uses_command("Lib"), own.text),
        );
        compare(&format!("{name}.trust"), &own.trust_report());
        if !own.checks.is_empty() {
            compare(&format!("{name}.checks"), &own.checks);
        }
    }
    compare("lib.lean", library.text());
    compare("lib.checks", library.checks());
}
