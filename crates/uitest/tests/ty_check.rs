use common::InputDb;
use dir_test::{Fixture, dir_test};
use driver::DriverDataBase;
use test_utils::snap_test;

#[cfg(target_arch = "wasm32")]
use test_utils::url_utils::UrlExt;

use url::Url;

#[dir_test(
    dir: "$CARGO_MANIFEST_DIR/fixtures/ty_check",
    glob: "**/*.fe"
)]
fn run_ty_check(fixture: Fixture<&str>) {
    let mut db = DriverDataBase::default();
    let file = db.workspace().touch(
        &mut db,
        Url::from_file_path(fixture.path()).expect("path should be absolute"),
        Some(fixture.content().to_string()),
    );

    let top_mod = db.top_mod(file);

    let diags = db.run_on_top_mod(top_mod);
    let diags = diags.format_diags(&db);
    snap_test!(diags, fixture.path());
}

/// The error headlines of a program checked as a standalone file, in order.
#[cfg(not(target_arch = "wasm32"))]
fn error_headlines(path: &str, content: &str) -> Vec<String> {
    let mut db = DriverDataBase::default();
    let file = db.workspace().touch(
        &mut db,
        Url::from_file_path(path).expect("path should be absolute"),
        Some(content.to_string()),
    );
    let top_mod = db.top_mod(file);
    let diags = db.run_on_top_mod(top_mod);
    diags
        .format_diags(&db)
        .lines()
        .filter(|line| line.starts_with("error[") || line.starts_with("warning["))
        .map(str::to_string)
        .collect()
}

/// `fe fmt` keeps the meaning of the const `where` fixtures: formatting twice
/// gives the same text as formatting once, and the formatted program reports
/// the same errors. Fixtures that do not parse are left alone by `fe fmt`.
#[cfg(not(target_arch = "wasm32"))]
#[dir_test(
    dir: "$CARGO_MANIFEST_DIR/fixtures/ty_check/const_where",
    glob: "**/*.fe",
    postfix: "fmt"
)]
fn formatting_keeps_meaning(fixture: Fixture<&str>) {
    let config = fmt::Config::default();
    let content = fixture.content();
    let Ok(formatted) = fmt::format_str(content, &config) else {
        return;
    };
    let reformatted = fmt::format_str(&formatted, &config).expect("formatted source should parse");
    assert_eq!(formatted, reformatted, "formatting twice changed the text");
    assert_eq!(
        error_headlines(fixture.path(), content),
        error_headlines(fixture.path(), &formatted),
        "formatting changed the errors:\n{formatted}"
    );
}

#[cfg(target_family = "wasm")]
mod wasm {
    use super::*;
    use test_utils::url_utils::UrlExt;
    use url::Url;
    use wasm_bindgen_test::wasm_bindgen_test;

    #[dir_test(
        dir: "$CARGO_MANIFEST_DIR/fixtures/ty_check",
        glob: "*.fe",
        postfix: "wasm"
    )]
    #[dir_test_attr(
        #[wasm_bindgen_test]
    )]
    fn run_ty_check(fixture: Fixture<&str>) {
        let mut db = DriverDataBase::default();
        let file = db.workspace().touch(
            &mut db,
            <Url as UrlExt>::from_file_path_lossy(fixture.path()),
            Some(fixture.content().to_string()),
        );

        let top_mod = db.top_mod(file);
        db.run_on_top_mod(top_mod);
    }
}
