use async_lsp::ResponseError;
use hir::{
    hir_def::{scope_graph::ScopeId, ItemKind, PathId, TopLevelMod},
    lower::map_file_to_mod,
    span::{DynLazySpan, LazySpan},
    visitor::{prelude::LazyPathSpan, Visitor, VisitorCtxt},
    SpannedHirDb,
};
use hir_analysis::name_resolution::{resolve_path, PathResErrorKind};

use crate::{
    backend::{db::LanguageServerDb, Backend},
    util::{to_lsp_location_from_scope, to_offset_from_position},
};
pub type Cursor = parser::TextSize;

#[derive(Default)]
struct PathSpanCollector<'db> {
    paths: Vec<(PathId<'db>, ScopeId<'db>, LazyPathSpan<'db>)>,
}

impl<'db, 'ast: 'db> Visitor<'ast> for PathSpanCollector<'db> {
    fn visit_path(&mut self, ctxt: &mut VisitorCtxt<'ast, LazyPathSpan<'ast>>, path: PathId<'db>) {
        let Some(span) = ctxt.span() else {
            return;
        };

        let scope = ctxt.scope();
        self.paths.push((path, scope, span));
    }
}

fn find_path_surrounding_cursor<'db>(
    db: &'db dyn LanguageServerDb,
    cursor: Cursor,
    full_paths: Vec<(PathId<'db>, ScopeId<'db>, LazyPathSpan<'db>)>,
) -> Option<(PathId<'db>, bool, ScopeId<'db>)> {
    let hir_db = db;
    for (path, scope, lazy_span) in full_paths {
        let span = lazy_span.resolve(db).unwrap();
        if span.range.contains(cursor) {
            for idx in 0..=path.segment_index(hir_db) {
                let seg_span = lazy_span.segment(idx).resolve(db).unwrap();
                if seg_span.range.contains(cursor) {
                    return Some((
                        path.segment(hir_db, idx).unwrap(),
                        idx != path.segment_index(hir_db),
                        scope,
                    ));
                }
            }
        }
    }
    None
}

pub fn find_enclosing_item<'db>(
    db: &'db dyn SpannedHirDb,
    top_mod: TopLevelMod<'db>,
    cursor: Cursor,
) -> Option<ItemKind<'db>> {
    let items = top_mod.scope_graph(db).items_dfs(db);

    let mut smallest_enclosing_item = None;
    let mut smallest_range_size = None;

    for item in items {
        let lazy_item_span = DynLazySpan::from(item.lazy_span());
        let item_span = lazy_item_span.resolve(db).unwrap();

        if item_span.range.contains(cursor) {
            let range_size = item_span.range.end() - item_span.range.start();
            if smallest_range_size.is_none() || range_size < smallest_range_size.unwrap() {
                smallest_enclosing_item = Some(item);
                smallest_range_size = Some(range_size);
            }
        }
    }

    smallest_enclosing_item
}

pub fn get_goto_target_scopes_for_cursor<'db>(
    db: &'db dyn LanguageServerDb,
    top_mod: TopLevelMod<'db>,
    cursor: Cursor,
) -> Option<Vec<ScopeId<'db>>> {
    let item: ItemKind = find_enclosing_item(db, top_mod, cursor)?;

    let mut visitor_ctxt = VisitorCtxt::with_item(db, item);
    let mut path_segment_collector = PathSpanCollector::default();
    path_segment_collector.visit_item(&mut visitor_ctxt, item);

    let (path, _is_intermediate, scope) =
        find_path_surrounding_cursor(db, cursor, path_segment_collector.paths)?;

    let resolved = resolve_path(db, path, scope, None, false);
    let scopes = match resolved {
        Ok(r) => r.as_scope(db).into_iter().collect::<Vec<_>>(),
        Err(err) => match err.kind {
            PathResErrorKind::NotFound(bucket) => {
                bucket.iter_ok().flat_map(|r| r.scope()).collect()
            }
            PathResErrorKind::Ambiguous(vec) => vec.into_iter().flat_map(|r| r.scope()).collect(),
            _ => vec![],
        },
    };

    Some(scopes)
}

use crate::backend::workspace::IngotFileContext;

pub async fn handle_goto_definition(
    backend: &mut Backend,
    params: async_lsp::lsp_types::GotoDefinitionParams,
) -> Result<Option<async_lsp::lsp_types::GotoDefinitionResponse>, ResponseError> {
    // Convert the position to an offset in the file
    let params = params.text_document_position_params;
    let file_text = std::fs::read_to_string(params.text_document.uri.path()).ok();
    let cursor: Cursor = to_offset_from_position(params.position, file_text.unwrap().as_str());

    // Get the module and the goto info
    let file_path = params.text_document.uri.path();
    let (ingot, file) = backend
        .workspace
        .get_input_for_file_path(file_path)
        .unwrap();
    let top_mod = map_file_to_mod(&backend.db, ingot, file);

    let scopes =
        get_goto_target_scopes_for_cursor(&backend.db, top_mod, cursor).unwrap_or_default();

    let locations = scopes
        .iter()
        .map(|scope| to_lsp_location_from_scope(&backend.db, ingot, *scope))
        .collect::<Vec<_>>();

    let result: Result<Option<async_lsp::lsp_types::GotoDefinitionResponse>, ()> =
        Ok(Some(async_lsp::lsp_types::GotoDefinitionResponse::Array(
            locations
                .into_iter()
                .filter_map(std::result::Result::ok)
                .collect(),
        )));
    let response = match result {
        Ok(response) => response,
        Err(e) => {
            eprintln!("Error handling goto definition: {:?}", e);
            None
        }
    };
    Ok(response)
}
// }
#[cfg(test)]
mod tests {
    use std::{collections::BTreeMap, path::Path};

    use common::input::IngotKind;
    use dir_test::{dir_test, Fixture};
    use salsa::Setter;
    use test_utils::snap_test;

    use super::*;
    use crate::backend::{
        db::LanguageServerDatabase,
        workspace::{IngotFileContext, Workspace},
    };

    // given a cursor position and a string, convert to cursor line and column
    fn line_col_from_cursor(cursor: Cursor, s: &str) -> (usize, usize) {
        let mut line = 0;
        let mut col = 0;
        for (i, c) in s.chars().enumerate() {
            if i == Into::<usize>::into(cursor) {
                return (line, col);
            }
            if c == '\n' {
                line += 1;
                col = 0;
            } else {
                col += 1;
            }
        }
        (line, col)
    }

    fn extract_multiple_cursor_positions_from_spans(
        db: &LanguageServerDatabase,
        top_mod: TopLevelMod,
    ) -> Vec<parser::TextSize> {
        let hir_db = db;
        let mut visitor_ctxt = VisitorCtxt::with_top_mod(hir_db, top_mod);
        let mut path_collector = PathSpanCollector::default();
        path_collector.visit_top_mod(&mut visitor_ctxt, top_mod);

        let mut cursors = Vec::new();
        for (path, _, lazy_span) in path_collector.paths {
            for idx in 0..=path.segment_index(hir_db) {
                let seg_span = lazy_span.segment(idx).resolve(db).unwrap();
                cursors.push(seg_span.range.start());
            }
        }

        cursors.sort();
        cursors.dedup();

        eprintln!("Found cursors: {:?}", cursors);
        cursors
    }

    fn make_goto_cursors_snapshot(
        db: &LanguageServerDatabase,
        fixture: &Fixture<&str>,
        top_mod: TopLevelMod,
    ) -> String {
        let cursors = extract_multiple_cursor_positions_from_spans(db, top_mod);
        let mut cursor_path_map: BTreeMap<Cursor, String> = BTreeMap::default();

        for cursor in &cursors {
            let scopes =
                get_goto_target_scopes_for_cursor(db, top_mod, *cursor).unwrap_or_default();

            if !scopes.is_empty() {
                cursor_path_map.insert(
                    *cursor,
                    scopes
                        .iter()
                        .flat_map(|x| x.pretty_path(db))
                        .collect::<Vec<_>>()
                        .join("\n"),
                );
            }
        }

        let cursor_lines = cursor_path_map
            .iter()
            .map(|(cursor, path)| {
                let (cursor_line, cursor_col) = line_col_from_cursor(*cursor, fixture.content());
                format!("cursor position ({cursor_line:?}, {cursor_col:?}), path: {path}")
            })
            .collect::<Vec<_>>();

        format!(
            "{}\n---\n{}",
            fixture
                .content()
                .lines()
                .enumerate()
                .map(|(i, line)| format!("{i:?}: {line}"))
                .collect::<Vec<_>>()
                .join("\n"),
            cursor_lines.join("\n")
        )
    }

    #[dir_test(
        dir: "$CARGO_MANIFEST_DIR/test_files/single_ingot",
        glob: "**/lib.fe",
    )]
    fn test_goto_multiple_files(fixture: Fixture<&str>) {
        let cargo_manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
        let ingot_base_dir = Path::new(&cargo_manifest_dir).join("test_files/single_ingot");

        let mut db = LanguageServerDatabase::default();
        let mut workspace = Workspace::default();

        let _ = workspace.set_workspace_root(&mut db, &ingot_base_dir);

        let fe_source_path = ingot_base_dir.join(fixture.path());
        let fe_source_path = fe_source_path.to_str().unwrap();
        let (ingot, file) = workspace
            .touch_input_for_file_path(&mut db, fixture.path())
            .unwrap();
        assert_eq!(ingot.kind(&db), IngotKind::Local);

        file.set_text(&mut db).to((*fixture.content()).to_string());

        // Introduce a new scope to limit the lifetime of `top_mod`
        {
            let (ingot, file) = workspace.get_input_for_file_path(fe_source_path).unwrap();
            let top_mod = map_file_to_mod(&db, ingot, file);

            let snapshot = make_goto_cursors_snapshot(&db, &fixture, top_mod);
            snap_test!(snapshot, fixture.path());
        }

        let ingot = workspace.touch_ingot_for_file_path(&mut db, fixture.path());
        assert_eq!(ingot.unwrap().kind(&db), IngotKind::Local);
    }

    #[dir_test(
        dir: "$CARGO_MANIFEST_DIR/test_files",
        glob: "goto*.fe"
    )]
    fn test_goto_cursor_target(fixture: Fixture<&str>) {
        let db = &mut LanguageServerDatabase::default();
        let workspace = &mut Workspace::default();
        let (ingot, file) = workspace
            .touch_input_for_file_path(db, fixture.path())
            .unwrap();
        file.set_text(db).to((*fixture.content()).to_string());
        let top_mod = map_file_to_mod(db, ingot, file);

        let snapshot = make_goto_cursors_snapshot(db, &fixture, top_mod);
        snap_test!(snapshot, fixture.path());
    }

    #[dir_test(
        dir: "$CARGO_MANIFEST_DIR/test_files",
        glob: "smallest_enclosing*.fe"
    )]
    fn test_find_path_surrounding_cursor(fixture: Fixture<&str>) {
        let db = &mut LanguageServerDatabase::default();
        let workspace = &mut Workspace::default();

        let (ingot, file) = workspace
            .touch_input_for_file_path(db, fixture.path())
            .unwrap();
        file.set_text(db).to((*fixture.content()).to_string());
        let top_mod = map_file_to_mod(db, ingot, file);

        let cursors = extract_multiple_cursor_positions_from_spans(db, top_mod);

        let mut cursor_paths: Vec<(Cursor, String)> = vec![];

        for cursor in &cursors {
            let mut visitor_ctxt = VisitorCtxt::with_top_mod(db, top_mod);
            let mut path_collector = PathSpanCollector::default();
            path_collector.visit_top_mod(&mut visitor_ctxt, top_mod);

            let full_paths = path_collector.paths;

            if let Some((path, _, scope)) = find_path_surrounding_cursor(db, *cursor, full_paths) {
                let resolved_enclosing_path = resolve_path(db, path, scope, None, false);

                let res = match resolved_enclosing_path {
                    Ok(res) => res.pretty_path(db).unwrap(),
                    Err(err) => match err.kind {
                        PathResErrorKind::Ambiguous(vec) => vec
                            .iter()
                            .map(|r| r.pretty_path(db).unwrap())
                            .collect::<Vec<_>>()
                            .join("\n"),
                        _ => "".into(),
                    },
                };
                cursor_paths.push((*cursor, res));
            }
        }

        let result = format!(
            "{}\n---\n{}",
            fixture.content(),
            cursor_paths
                .iter()
                .map(|(cursor, path)| { format!("cursor position: {cursor:?}, path: {path}") })
                .collect::<Vec<_>>()
                .join("\n")
        );
        snap_test!(result, fixture.path());
    }
}
