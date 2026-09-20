//! Resolve the shared operation contract before publishing analysis facts.
use super::validity::NativeValidity;
use crate::analysis::semantic::diagnostics::{SemanticDiagnostic, operand_origin};
use cranelift_entity::EntityRef;

use super::{
    availability::ResolvedAvailability,
    events::{CapabilityOccurrence, CapabilityTraversal},
    memory::ResolvedMemoryAccess,
    solver::Borrowck,
    summary::CallInputs,
};
use crate::analysis::{
    semantic::{
        SemOrigin,
        capability::{
            guard::ValueOccurrence,
            index::BinderScope,
            loan::{CapabilityRef, LoanRef},
            path::RegionPath,
            region::{OverlapResult, RegionRoot, RegionSet},
            state::BorrowState,
            value::Guarded,
        },
        normalized::{
            NEffectArgValue, NExpr, NPlaceBase, NStatementKind, ReadMode,
            access::{AccessTarget, OperationAccess},
        },
    },
    ty::{corelib::MemoryAccessKind, ty_def::BorrowKind},
};

#[derive(Clone)]
pub(super) struct ResolvedAccess<'db> {
    pub kind: MemoryAccessKind,
    pub conflict_kind: BorrowKind,
    pub region: RegionSet<'db>,
    pub authority: Vec<Guarded<'db, LoanRef<'db>>>,
    pub origin: SemOrigin<'db>,
    pub forbidden_move: bool,
    pub invalidated: NativeValidity<'db>,
}

#[derive(Clone, Default)]
pub(super) struct ResolvedOperation<'db> {
    pub accesses: Vec<ResolvedAccess<'db>>,
    pub calls: Vec<ResolvedMemoryAccess<'db>>,
    pub availability: Option<ResolvedAvailability<'db>>,
    pub native_validity: NativeValidity<'db>,
    pub active: Vec<CapabilityOccurrence<'db>>,
    pub arguments: Vec<(usize, CapabilityOccurrence<'db>)>,
}

pub(super) fn effect_occurrence(argument: &NEffectArgValue<'_>) -> ValueOccurrence {
    match argument {
        NEffectArgValue::Value(value) => ValueOccurrence::Value(value.value),
        NEffectArgValue::Place(place) => match place.base {
            NPlaceBase::Root(root) => ValueOccurrence::Root(root),
            NPlaceBase::CapabilityTarget { carrier } => ValueOccurrence::Value(carrier),
        },
    }
}

impl<'db> Borrowck<'db> {
    pub fn resolve_access(
        &self,
        state: &BorrowState<'db>,
        access: OperationAccess<'_, 'db>,
        origin: SemOrigin<'db>,
    ) -> ResolvedAccess<'db> {
        let (region, authority, origin, forbidden_move) = match access.target {
            AccessTarget::Place(place) => (
                self.resolve_region(state, place),
                self.authority(state, place),
                origin,
                access.kind == MemoryAccessKind::Move
                    && matches!(place.base, NPlaceBase::CapabilityTarget { carrier }
                        if self.body.values[carrier.index()].ty.as_ptr(self.db).is_none()),
            ),
            AccessTarget::Value { operand, path } => (
                RegionSet::singleton(
                    &BinderScope::default(),
                    RegionRoot::Value(operand.value),
                    path.map_or_else(RegionPath::default, |path| self.path(&path.0)),
                )
                .with_guard(state.guard()),
                Vec::new(),
                operand_origin(operand, origin),
                path.is_some()
                    && operand.mode == ReadMode::Move
                    && self.body.values[operand.value.index()]
                        .ty
                        .as_view(self.db)
                        .is_some(),
            ),
        };
        let invalidated = if let AccessTarget::Value { operand, .. } = access.target {
            self.inventory
                .values
                .leaves(
                    state.value(operand.value),
                    ValueOccurrence::Value(operand.value),
                )
                .iter()
                .filter(|leaf| {
                    matches!(leaf.payload, CapabilityRef::Invalidated { .. })
                        && !matches!(
                            region.overlap(
                                &RegionSet::singleton(
                                    leaf.guard.scope(),
                                    RegionRoot::Value(operand.value),
                                    RegionPath::new(leaf.path.as_slice())
                                )
                                .with_guard(&leaf.guard)
                                .close_existentials(region.scope())
                            ),
                            OverlapResult::Disjoint
                        )
                })
                .fold(NativeValidity::default(), |mut validity, leaf| {
                    if let CapabilityRef::Invalidated { region, .. } = &leaf.payload {
                        validity |= NativeValidity::from_region(&region.with_guard(&leaf.guard));
                    }
                    validity
                })
        } else {
            match access.target {
                AccessTarget::Place(place) => self.resolve_place(state, place).invalidated,
                AccessTarget::Value { .. } => unreachable!(),
            }
        };
        ResolvedAccess {
            kind: access.kind,
            conflict_kind: access.conflict_kind(),
            region,
            authority,
            origin,
            forbidden_move,
            invalidated,
        }
    }

    pub fn resolve_operations(&mut self) -> Result<(), SemanticDiagnostic<'db>> {
        for block in 0..self.body.blocks.len() {
            let statements = self.body.blocks[block].statements.clone();
            let mut operations = Vec::new();
            for (index, statement) in statements.iter().enumerate() {
                let Some(state) = self.before[block].get(index).cloned() else {
                    break;
                };
                let mut accesses = Vec::new();
                for access in statement.kind.accesses(self.db, &self.body) {
                    let mut resolved = self.resolve_access(&state, access, statement.origin);
                    if access.kind != MemoryAccessKind::Write
                        && let AccessTarget::Place(place) = access.target
                    {
                        let shape = self.shape(place.ty)?;
                        if shape.contains_capability(self.db) {
                            let contents = self.read_region(
                                &state,
                                &resolved.region,
                                shape,
                                ValueOccurrence::Summary,
                                statement.origin,
                            )?;
                            resolved.invalidated |=
                                self.value_validity(&contents, ValueOccurrence::Summary);
                        }
                    }
                    accesses.push(resolved);
                }
                if let NStatementKind::Store { value, .. } = statement.kind
                    && value.mode == ReadMode::Move
                {
                    let authority: Vec<_> = self
                        .capabilities(
                            &state,
                            state.value(value.value),
                            ValueOccurrence::Value(value.value),
                            statement.origin,
                            CapabilityTraversal::Held,
                        )?
                        .into_iter()
                        .filter_map(|capability| capability.authorizer())
                        .collect();
                    for access in &mut accesses {
                        if access.kind == MemoryAccessKind::Write {
                            access.authority.extend(authority.iter().cloned());
                        }
                    }
                }
                let (calls, availability, native_validity) = if let NStatementKind::Define {
                    result,
                    expr:
                        NExpr::Call {
                            args, effect_args, ..
                        },
                } = &statement.kind
                {
                    let inputs = CallInputs {
                        args,
                        effects: effect_args,
                        origin: statement.origin,
                    };
                    (
                        self.call_memory_accesses(&state, *result, inputs)?,
                        self.call_availability(&state, *result, inputs)?,
                        self.call_native_validity(&state, *result, inputs)?,
                    )
                } else {
                    (Vec::new(), None, NativeValidity::default())
                };
                operations.push(ResolvedOperation {
                    accesses,
                    calls,
                    availability,
                    native_validity,
                    ..Default::default()
                });
            }
            self.operations[block] = operations;
        }
        self.resolve_conflict_facts()
    }
}
