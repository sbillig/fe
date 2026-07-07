//! Builtin `core::functional::Fn` support for closure types.
//!
//! Closures implement the core `Fn` trait without a HIR `impl` item. This
//! module is the single home for that rule: method selection, the trait
//! solver, and callee resolution all derive the closure/`Fn` relationship
//! from the helpers here.

use common::indexmap::IndexMap;

use super::{
    binder::Binder,
    corelib::resolve_core_trait,
    trait_def::{ImplementorId, TraitInstId},
    ty_def::{ClosureTy, TyId},
};
use crate::{
    analysis::HirAnalysisDb,
    hir_def::{Func, Trait, scope_graph::ScopeId},
};

/// The core `functional::Fn` trait, if resolvable from `scope`.
pub(crate) fn fn_trait<'db>(db: &'db dyn HirAnalysisDb, scope: ScopeId<'db>) -> Option<Trait<'db>> {
    resolve_core_trait(db, scope, &["functional", "Fn"])
}

/// The `Fn<closure, Arg, Ret>` trait instance realized by `closure`.
///
/// `None` when the closure's arity has no corresponding `Fn` trait (the core
/// trait is single-argument; other arities are rejected at the closure
/// definition site).
pub(crate) fn closure_fn_trait_inst<'db>(
    db: &'db dyn HirAnalysisDb,
    scope: ScopeId<'db>,
    closure: ClosureTy<'db>,
) -> Option<TraitInstId<'db>> {
    let [arg_ty] = closure.params(db).as_slice() else {
        return None;
    };
    let fn_trait = fn_trait(db, scope)?;
    Some(TraitInstId::new(
        db,
        fn_trait,
        vec![TyId::closure(db, closure), *arg_ty, closure.ret_ty(db)],
        IndexMap::new(),
    ))
}

/// A synthesized implementor for the builtin `Fn` impl of the closure at the
/// base of `goal`'s self type, for use as a trait-solving candidate.
///
/// Structural checks run before any core-trait path resolution so that
/// non-closure goals (the overwhelming majority) bail out cheaply.
pub(crate) fn builtin_fn_candidate_for_goal<'db>(
    db: &'db dyn HirAnalysisDb,
    scope: ScopeId<'db>,
    goal: TraitInstId<'db>,
) -> Option<Binder<ImplementorId<'db>>> {
    let self_ty = *goal.args(db).first()?;
    let closure = self_ty.as_closure(db)?;
    let inst = closure_fn_trait_inst(db, scope, closure)?;
    if goal.def(db) != inst.def(db) {
        return None;
    }
    Some(Binder::bind(ImplementorId::assumption(db, inst)))
}

/// The closure whose body is the callee when `func` is `Fn::call` invoked
/// through `inst` on a closure receiver.
pub(crate) fn callee_closure_for_fn_call<'db>(
    db: &'db dyn HirAnalysisDb,
    func: Func<'db>,
    inst: TraitInstId<'db>,
) -> Option<ClosureTy<'db>> {
    let closure = inst.self_ty(db).as_closure(db)?;
    if func
        .name(db)
        .to_opt()
        .is_none_or(|name| name.data(db) != "call")
    {
        return None;
    }
    if fn_trait(db, func.scope()) != Some(inst.def(db)) {
        return None;
    }
    Some(closure)
}
