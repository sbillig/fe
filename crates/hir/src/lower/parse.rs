use common::InputFile;
use parser::GreenNode;
use salsa::Accumulator;

use crate::{hir_def::TopLevelMod, HirDb};

#[salsa::tracked]
pub fn parse_file_impl<'db>(db: &'db dyn HirDb, top_mod: TopLevelMod<'db>) -> GreenNode {
    let file = top_mod.file(db);
    let text = file.text(db);
    let (node, parse_errors) = parser::parse_source_file(text);

    for error in parse_errors {
        ParserError { file, error }.accumulate(db);
    }
    node
}

#[salsa::accumulator]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ParserError {
    pub file: InputFile,
    pub error: parser::ParseError,
}
