mod analyses;
mod canon;
mod check;
mod diagnostics;
mod facts;
mod ir;
mod normalize;
mod verify;

pub use check::{
    check_semantic_borrows, collect_semantic_borrow_diagnostics, semantic_borrow_summary,
};
pub use facts::*;
pub use ir::*;
pub use normalize::normalize_semantic_body;
pub use verify::verify_normalized_semantic_body;
