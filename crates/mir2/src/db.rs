#[salsa::db]
pub trait MirDb: hir::analysis::diagnostics::SpannedHirAnalysisDb {}

#[salsa::db]
impl<T> MirDb for T where T: salsa::Database + hir::analysis::diagnostics::SpannedHirAnalysisDb {}
