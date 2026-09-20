//! Allocation lifetime events, independent of typed contents and ownership.
use std::collections::BTreeMap;

use super::{
    external::{ExternalOrigin, ExternalSource},
    guard::Guard,
    handle::OpaqueHandleRef,
    index::{BinderScope, IndexExpr, IndexSubst},
    region::RegionRoot,
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AllocationBirth<'db> {
    pub allocation: OpaqueHandleRef<'db>,
    pub guard: Guard<'db>,
}

impl<'db> AllocationBirth<'db> {
    /// A followed source still records the allocation of its container. This
    /// extracts that origin, never a claim that its loaded referent is fresh.
    pub fn from_source(source: &ExternalSource<'db>, guard: Guard<'db>) -> Option<Self> {
        let allocation = match &source.origin {
            ExternalOrigin::Allocation(allocation) => allocation.clone(),
            ExternalOrigin::Memory { base, .. } => {
                return Self::from_source(&base.source, guard);
            }
            _ => return None,
        };
        Some(Self { allocation, guard })
    }

    /// Select only proved members of this physical allocation family. Callee
    /// family binders are instantiated with the candidate's arguments; concrete
    /// arguments (including the caller generation) must all compare equal.
    /// Casts and offsets name the allocation's own bytes, but following a stored
    /// capability does not. Unknown aliases therefore cannot authorize a reset.
    pub fn selector(&self, root: &RegionRoot<'db>, scope: &BinderScope) -> Option<Guard<'db>> {
        let RegionRoot::External(source) = root else {
            return None;
        };
        let candidate = source.fresh_allocation()?;
        if candidate.occurrence != self.allocation.occurrence {
            return None;
        }
        assert_eq!(candidate.arguments.len(), self.allocation.arguments.len());
        let mut bindings = BTreeMap::new();
        for (&parameter, &argument) in self.allocation.arguments.iter().zip(&candidate.arguments) {
            if matches!(parameter, IndexExpr::Bound(_)) {
                bindings.entry(parameter).or_insert(argument);
            }
        }
        // Array/value traversal can contribute a range witness without making
        // it an allocation-family parameter (e.g. [one_pointer; N]). Project
        // such scalar witnesses instead of confusing them with object identity.
        let guard = self.guard.project_witnesses(|index| {
            matches!(index, IndexExpr::Bound(_)) && !bindings.contains_key(&index)
        });
        let used = guard.indices();
        for parameter in guard.scope().variables() {
            if !bindings.contains_key(&parameter) && used.contains(&parameter) {
                // A hidden selector cannot prove that this member was born.
                return None;
            }
            bindings.entry(parameter).or_insert(IndexExpr::Const(0));
        }
        let subst = IndexSubst::new(guard.scope(), scope, bindings)
            .expect("allocation family selector substitution");
        let mut guard = guard.substitute(&subst)?;
        for (&parameter, &argument) in self.allocation.arguments.iter().zip(&candidate.arguments) {
            guard = guard.with_equality(subst.apply(parameter), argument)?;
        }
        Some(guard)
    }
}
