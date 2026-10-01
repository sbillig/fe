use std::path::Path;

use dir_test::{Fixture, dir_test};
use fe_hir::core::semantic::index::module_references;
use fe_hir::span::LazySpan;
use fe_hir::test_db::HirAnalysisTestDb;
use test_utils::snap_test;

/// Every reference to an item of the fixture that the module's reference
/// index records, in source order: the referring text at its span and the
/// path of what it refers to. Rename edits each of these spans, so a reference
/// recorded twice, or at the wrong span, shows here.
#[dir_test(
    dir: "$CARGO_MANIFEST_DIR/test_files/module_references",
    glob: "*.fe"
)]
fn module_references_snapshot(fixture: Fixture<&str>) {
    let mut db = HirAnalysisTestDb::default();
    let path = Path::new(fixture.path());
    let file_name = path.file_name().and_then(|file| file.to_str()).unwrap();
    let file = db.new_stand_alone(file_name.into(), fixture.content());
    let (top_mod, _) = db.top_mod(file);
    let text = fixture.content();
    let mut lines: Vec<(usize, String)> = module_references(&db, top_mod)
        .iter()
        .filter(|reference| reference.target.top_mod(&db) == top_mod)
        .map(|reference| {
            let target = reference
                .target
                .pretty_path(&db)
                .unwrap_or_else(|| reference.target.kind_name().to_string());
            match reference.span.resolve(&db) {
                Some(span) => {
                    let start = usize::from(span.range.start());
                    let end = usize::from(span.range.end());
                    let line = text[..start].matches('\n').count() + 1;
                    (
                        start,
                        format!("{line}: `{}` -> {target}", &text[start..end]),
                    )
                }
                None => (usize::MAX, format!("unresolved span -> {target}")),
            }
        })
        .collect();
    lines.sort();
    // A reference recorded more than once is one line with its count, since
    // rename would edit its span that many times.
    let mut output: Vec<(String, usize)> = Vec::new();
    for (_, line) in lines {
        match output.last_mut() {
            Some((last, count)) if *last == line => *count += 1,
            _ => output.push((line, 1)),
        }
    }
    let output = output
        .into_iter()
        .map(|(line, count)| match count {
            1 => line,
            count => format!("{line} ({count} times)"),
        })
        .collect::<Vec<_>>()
        .join("\n");
    snap_test!(output, fixture.path());
}
