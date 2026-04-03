mod check;
mod ir;
mod normalize;

pub use check::{
    check_semantic_borrows, collect_semantic_borrow_diagnostics, semantic_borrow_summary,
    verify_normalized_semantic_body,
};
pub use ir::*;
pub use normalize::normalize_semantic_body;
