//! Native validity obligations retained across symbolic input aliases.
use std::ops::BitOrAssign;

use super::solver::Borrowck;
use super::summary::CallInputs;
use crate::analysis::semantic::capability::{
    external::ExternalOrigin,
    guard::ValueOccurrence,
    index::BinderScope,
    loan::CapabilityRef,
    region::{RegionRoot, RegionSet},
    source::SourceExpr,
    state::{BorrowState, CapabilityValue},
};
use crate::analysis::semantic::{diagnostics::SemanticDiagnostic, normalized::NValueId};

impl<'db> Borrowck<'db> {
    pub fn call_native_validity(
        &mut self,
        state: &BorrowState<'db>,
        result: NValueId,
        inputs: CallInputs<'_, 'db>,
    ) -> Result<NativeValidity<'db>, SemanticDiagnostic<'db>> {
        let mut validity = NativeValidity::default();
        let Some(call) = self.calls.get(&result).cloned() else {
            return Ok(validity);
        };
        for clause in call.summary.native_requirements.clauses() {
            let Some(guard) = self.instantiate_guard(&clause.guard, result, inputs)? else {
                continue;
            };
            let source =
                SourceExpr::from_place(&clause.payload).expect("native validity summary source");
            let resolved =
                self.instantiate_source(state, &source, result, guard.scope(), inputs)?;
            validity |= resolved.invalidated;
            validity |= NativeValidity::from_region(&resolved.region.with_guard(&guard));
        }
        Ok(validity)
    }

    pub fn value_validity(
        &self,
        value: &CapabilityValue<'db>,
        occurrence: ValueOccurrence,
    ) -> NativeValidity<'db> {
        let mut validity = NativeValidity::default();
        for leaf in self.inventory.values.leaves(value, occurrence) {
            if let CapabilityRef::Invalidated { region, .. } = leaf.payload {
                validity |= NativeValidity::from_region(&region.with_guard(&leaf.guard));
            }
        }
        validity
    }
}

#[derive(Clone, Debug)]
pub(super) struct NativeValidity<'db> {
    pub invalid: bool,
    pub requirements: RegionSet<'db>,
}

impl Default for NativeValidity<'_> {
    fn default() -> Self {
        Self {
            invalid: false,
            requirements: RegionSet::empty(&BinderScope::default()),
        }
    }
}

impl<'db> NativeValidity<'db> {
    /// Requirements keep the region's family binders and close only its
    /// clause-local existential witnesses.
    pub fn from_region(region: &RegionSet<'db>) -> Self {
        let owner = region.scope().without_existentials();
        let mut result = Self {
            invalid: false,
            requirements: RegionSet::empty(&owner),
        };
        for clause in region.clauses() {
            let deferred = if let RegionRoot::External(source) = &clause.payload.root
                && let Some(clobber) = &source.clobber
            {
                let target = physical_base(&clobber.target);
                let written = physical_base(&clobber.written);
                let overlaps = target.source == written.source
                    && (target.path.as_slice().starts_with(written.path.as_slice())
                        || written.path.as_slice().starts_with(target.path.as_slice()));
                !overlaps && (target.source.param().is_some() || written.source.param().is_some())
            } else {
                false
            };
            result.requirements = result.requirements.union(
                &RegionSet::new(region.scope(), [clause.clone()]).close_existentials(&owner),
            );
            result.invalid |= !deferred;
        }
        result
    }
}

impl<'db> BitOrAssign for NativeValidity<'db> {
    fn bitor_assign(&mut self, other: Self) {
        self.invalid |= other.invalid;
        // An empty requirement set means the same thing in every scope.
        if self.requirements.is_empty() {
            self.requirements = other.requirements;
        } else if !other.requirements.is_empty() {
            self.requirements = self.requirements.union(&other.requirements);
        }
    }
}

fn physical_base<'a, 'db>(mut source: &'a SourceExpr<'db>) -> &'a SourceExpr<'db> {
    while source.path.is_empty()
        && source.views.iter().next().is_none()
        && source.source.dereferences().is_empty()
        && let ExternalOrigin::Memory {
            base,
            element: None,
            ..
        } = &source.source.origin
    {
        source = base;
    }
    source
}
