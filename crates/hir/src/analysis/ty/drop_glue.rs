//! Which values the compiler destroys: a type's drop glue.

use super::{corelib::resolve_core_trait, trait_lower::lower_impl_trait};
use crate::{analysis::HirAnalysisDb, hir_def::Func};

/// Whether `func` is an implementation's `Drop::drop`.
pub fn implements_drop<'db>(db: &'db dyn HirAnalysisDb, func: Func<'db>) -> bool {
    func.containing_impl_trait(db).is_some_and(|impl_trait| {
        lower_impl_trait(db, impl_trait).is_some_and(|implementor| {
            Some(implementor.trait_def(db))
                == resolve_core_trait(db, func.scope(), &["ops", "Drop"])
        })
    })
}
