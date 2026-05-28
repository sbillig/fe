mod analyses;
mod callsite;
mod canon;
mod check;
mod diagnostics;
mod facts;
mod ir;
mod noesc;
mod normalize;
mod pointer;
mod verify;

pub(crate) use callsite::provisional_call_site_provider_refinements;
pub use check::{
    SemanticBorrowAnalysisPass, check_semantic_borrows,
    collect_semantic_borrow_diagnostic_vouchers, semantic_borrow_summary,
    semantic_pointer_provenance_summary,
};
pub use facts::*;
pub use ir::*;
pub use noesc::{check_semantic_noesc, check_semantic_noesc_voucher};
pub use normalize::normalize_semantic_body;
pub use verify::verify_normalized_semantic_body;
