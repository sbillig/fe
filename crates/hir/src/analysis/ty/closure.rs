use crate::{
    analysis::{
        HirAnalysisDb,
        ty::{corelib::resolve_core_trait, ty_def::ClosureTy},
    },
    hir_def::{Trait, scope_graph::ScopeId},
};

use super::ty_def::ClosureCallMode;

/// A builtin callable trait implemented by a closure.
///
/// This is the single recognition point shared by trait solving and semantic
/// callee construction. A reusable closure implements both traits; a
/// consuming closure implements only `FnOnce`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClosureCallTrait {
    Fn,
    FnOnce,
}

impl ClosureCallTrait {
    const ALL: [Self; 2] = [Self::Fn, Self::FnOnce];

    pub const fn method_name(self) -> &'static str {
        match self {
            Self::Fn => "call",
            Self::FnOnce => "call_once",
        }
    }

    fn trait_name(self) -> &'static str {
        match self {
            Self::Fn => "Fn",
            Self::FnOnce => "FnOnce",
        }
    }

    fn is_implemented_by(self, db: &dyn HirAnalysisDb, closure: ClosureTy<'_>) -> bool {
        self == Self::FnOnce || closure.call_mode(db) == ClosureCallMode::Reusable
    }

    fn trait_def<'db>(self, db: &'db dyn HirAnalysisDb, scope: ScopeId<'db>) -> Option<Trait<'db>> {
        resolve_core_trait(db, scope, &["functional", self.trait_name()])
    }

    pub fn for_trait<'db>(
        db: &'db dyn HirAnalysisDb,
        scope: ScopeId<'db>,
        trait_: Trait<'db>,
    ) -> Option<Self> {
        Self::ALL
            .into_iter()
            .find(|call_trait| call_trait.trait_def(db, scope) == Some(trait_))
    }
}

pub fn closure_call_trait_for_method<'db>(
    db: &'db dyn HirAnalysisDb,
    scope: ScopeId<'db>,
    closure: ClosureTy<'db>,
    method_name: &str,
) -> Option<(ClosureCallTrait, Trait<'db>)> {
    ClosureCallTrait::ALL.into_iter().find_map(|call_trait| {
        (call_trait.method_name() == method_name && call_trait.is_implemented_by(db, closure))
            .then_some(call_trait)
            .and_then(|call_trait| {
                call_trait
                    .trait_def(db, scope)
                    .map(|trait_| (call_trait, trait_))
            })
    })
}

pub fn implemented_closure_call_trait<'db>(
    db: &'db dyn HirAnalysisDb,
    scope: ScopeId<'db>,
    closure: ClosureTy<'db>,
    trait_: Trait<'db>,
) -> Option<ClosureCallTrait> {
    ClosureCallTrait::for_trait(db, scope, trait_)
        .filter(|call_trait| call_trait.is_implemented_by(db, closure))
}
