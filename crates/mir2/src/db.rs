#[salsa::db]
pub trait MirDb: hir::analysis::HirAnalysisDb {}

#[salsa::db]
impl<T> MirDb for T where T: salsa::Database + hir::analysis::HirAnalysisDb {}
