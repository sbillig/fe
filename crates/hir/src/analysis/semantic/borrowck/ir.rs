use crate::analysis::semantic::diagnostics::{BlockedSemanticBody, SemanticDiagnosticId};
use salsa::Update;
use std::collections::BTreeSet;

use crate::analysis::{
    semantic::{
        SemOrigin, SemanticInstance, SemanticInstanceKey,
        capability::{
            footprint::{AccessExtent, AccessFootprint},
            region::RegionSet,
            source::SourceExpr,
            value::ValueId,
        },
    },
    ty::{corelib::MemoryAccessKind, ty_def::TyId},
};

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct BorrowSummary<'db> {
    /// Whether any admitted path reaches a return.
    pub may_return: bool,
    pub result: ValueId<'db, SourceExpr<'db>>,
    pub mutable_inputs: Vec<InputPoststate<'db>>,
    /// Preconditions on the actual regions supplied by callers. They are
    /// independent of the callee's return value and mutable-input poststates.
    pub requirements: Vec<BoundaryRequirement<'db>>,
    /// Reads and writes through addresses, including effects of transitive calls.
    pub accesses: Vec<MemoryAccess<'db>>,
    pub availability: AvailabilitySummary<'db>,
    /// Conditional overwrites that must be disjoint before a native capability is used.
    pub native_requirements: RegionSet<'db>,
}

/// A normal-return ownership transformer, separate from unordered access history.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct AvailabilitySummary<'db> {
    pub incoming: Vec<AvailabilityRequirement<'db>>,
    /// Each clause is a guaranteed write, rather than one possible destination.
    pub reinitialized: RegionSet<'db>,
    pub unavailable: RegionSet<'db>,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct AvailabilityRequirement<'db> {
    /// Write requires an available parent; other accesses require the contents.
    pub kind: MemoryAccessKind,
    pub region: RegionSet<'db>,
    pub extent: AccessExtent<'db>,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MemoryAccess<'db> {
    pub kind: MemoryAccessKind,
    pub region: RegionSet<'db>,
    pub extent: AccessExtent<'db>,
    /// Explicit receiver authority for an otherwise unknown memory effect.
    pub authorizers: RegionSet<'db>,
}

impl<'db> MemoryAccess<'db> {
    pub fn footprint(&self) -> AccessFootprint<'_, 'db> {
        AccessFootprint {
            region: &self.region,
            extent: self.extent,
        }
    }
}

impl<'db> AvailabilityRequirement<'db> {
    pub fn footprint(&self) -> AccessFootprint<'_, 'db> {
        AccessFootprint {
            region: &self.region,
            extent: self.extent,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BoundaryRule<'db> {
    MemoryTransport(TyId<'db>),
    Writable,
    BorrowedStore(TyId<'db>),
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct BoundaryRequirement<'db> {
    pub rule: BoundaryRule<'db>,
    pub instance: SemanticInstance<'db>,
    pub origin: SemOrigin<'db>,
    pub region: RegionSet<'db>,
    /// A storage rule applies only if this supplied capability is populated.
    /// This retains empty-variant precision across forwarding calls.
    pub populated: Option<RegionSet<'db>>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct InputPoststate<'db> {
    pub destination: SourceExpr<'db>,
    pub value: ValueId<'db, SourceExpr<'db>>,
}

#[salsa::interned]
#[derive(Debug)]
pub struct BorrowSummaryId<'db> {
    #[return_ref]
    pub items: BorrowSummary<'db>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Update)]
pub enum SemanticBorrowSummaryResult<'db> {
    Ok(Option<BorrowSummaryId<'db>>),
    Pending {
        validation: PendingSemanticValidation<'db>,
        summary: Option<BorrowSummaryId<'db>>,
    },
    Blocked {
        body: BlockedSemanticBody<'db>,
        summary: Option<BorrowSummaryId<'db>>,
    },
    Err(SemanticDiagnosticId<'db>),
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Update)]
pub enum SemanticBorrowCheckResult<'db> {
    Ok,
    Pending(PendingSemanticValidation<'db>),
    Blocked(BlockedSemanticBody<'db>),
    Err(SemanticDiagnosticId<'db>),
}

/// Obligations that must be discharged by rebuilding the concrete semantic
/// instance. A conservative template summary is not a successful validation.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash, Update)]
pub struct PendingSemanticValidation<'db> {
    pub callees: BTreeSet<SemanticInstanceKey<'db>>,
}
