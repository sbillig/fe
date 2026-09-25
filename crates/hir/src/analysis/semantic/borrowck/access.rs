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
            birth::AllocationBirth,
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
            access::{AccessPhase, AccessTarget, OperationAccess},
        },
    },
    ty::{corelib::MemoryAccessKind, ty_def::BorrowKind},
};

#[derive(Clone)]
pub(super) struct ResolvedAccess<'db> {
    pub kind: MemoryAccessKind,
    pub phase: AccessPhase,
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
    pub births: Vec<AllocationBirth<'db>>,
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
                                self.db,
                                &RegionSet::singleton(
                                    leaf.guard.scope(),
                                    RegionRoot::Value(operand.value),
                                    RegionPath::new(leaf.path.as_slice())
                                )
                                .with_guard(&leaf.guard)
                                .quantify_into(self.db, region.scope())
                            ),
                            OverlapResult::Disjoint
                        )
                })
                .fold(NativeValidity::default(), |mut validity, leaf| {
                    if let CapabilityRef::Invalidated {
                        region: invalidated,
                        ..
                    } = &leaf.payload
                    {
                        validity |= NativeValidity::from_region(
                            &invalidated
                                .with_guard(&leaf.guard)
                                .quantify_into(self.db, region.scope()),
                        );
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
            phase: access.phase,
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
                let (calls, availability, native_validity, births) =
                    if let NStatementKind::Define {
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
                            self.call_births(*result, inputs)?,
                        )
                    } else if let NStatementKind::Define {
                        result,
                        expr: NExpr::Const(constant),
                    } = &statement.kind
                    {
                        (
                            Vec::new(),
                            None,
                            NativeValidity::default(),
                            self.literal_birth(*result, constant)
                                .map(|(_, birth)| birth)
                                .into_iter()
                                .collect(),
                        )
                    } else {
                        (Vec::new(), None, NativeValidity::default(), Vec::new())
                    };
                operations.push(ResolvedOperation {
                    accesses,
                    calls,
                    availability,
                    births,
                    native_validity,
                    ..Default::default()
                });
            }
            self.operations[block] = operations;
        }
        self.resolve_conflict_facts()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        analysis::{
            semantic::{
                capability::{
                    external::ExternalSource, guard::Guard, handle::AddressOccurrence,
                    index::IndexExpr,
                },
                get_or_build_semantic_instance, identity_semantic_instance_key,
                normalized::literal_allocation,
            },
            ty::ty_check::BodyOwner,
        },
        test_db::{HirAnalysisTestDb, find_func},
    };

    #[test]
    fn literal_birth_actions_match_the_explicit_constant_contract() {
        let mut db = HirAnalysisTestDb::default();
        let file = db.new_stand_alone(
            "literal_actions.fe".into(),
            r#"
fn inspect(_ count: u256) {
    let mut i: u256 = 0
    while i < count {
        let text: Text = "hello"
        let inline: String<5> = "hello"
        let number: u256 = 7
        i += 1
    }
}
"#,
        );
        let (module, _) = db.top_mod(file);
        db.assert_no_diags(module);
        let instance = get_or_build_semantic_instance(
            &db,
            identity_semantic_instance_key(&db, BodyOwner::Func(find_func(&db, module, "inspect"))),
        );
        let mut checker = Borrowck::new(&db, instance).unwrap();
        checker.solve().unwrap();
        assert!(
            checker.inventory.allocation_cells.is_empty(),
            "byte-only literal allocation has no capability contents"
        );
        let mut allocating = 0;
        let mut ordinary = 0;
        let mut inline_strings = 0;
        for (block_index, block) in checker.body.blocks.iter().enumerate() {
            for (index, statement) in block.statements.iter().enumerate() {
                let NStatementKind::Define {
                    result,
                    expr: NExpr::Const(constant),
                } = &statement.kind
                else {
                    continue;
                };
                let births = &checker.operations[block_index][index].births;
                if let Some((_, contract)) =
                    literal_allocation(&db, checker.body.values[result.index()].ty, constant)
                {
                    allocating += 1;
                    let [birth] = births.as_slice() else {
                        panic!("allocating constant must publish one birth: {births:?}")
                    };
                    assert_eq!(birth.allocation.contract, contract);
                    assert_eq!(
                        birth.allocation.occurrence,
                        AddressOccurrence::Value {
                            instance,
                            value: *result,
                            choice: 0
                        }
                    );
                    assert_eq!(
                        birth.allocation.arguments.as_ref(),
                        &[IndexExpr::Iteration(
                            checker
                                .inventory
                                .loops
                                .for_value(&checker.body, *result)
                                .unwrap()
                        )]
                    );
                    assert_eq!(birth.guard, Guard::always(&BinderScope::default()));
                    let state = checker.before[block_index].get(index + 1).unwrap();
                    let leaves = checker
                        .inventory
                        .values
                        .leaves(state.value(*result), ValueOccurrence::Value(*result));
                    assert!(
                        leaves.iter().any(|leaf| leaf
                            .payload
                            .region(&db, &checker.inventory.loans, leaf.guard.scope())
                            .clauses()
                            .iter()
                            .any(|clause| {
                                clause.payload.root
                                    == RegionRoot::External(ExternalSource::allocation(
                                        &db,
                                        birth.allocation.clone(),
                                    ))
                            })),
                        "resolved birth must match the source published by transfer"
                    );
                } else {
                    ordinary += 1;
                    if checker.body.values[result.index()].ty.is_string(&db) {
                        inline_strings += 1;
                    }
                    assert!(births.is_empty());
                }
            }
        }
        assert!(
            allocating > 0,
            "the recognized allocation contract is exercised"
        );
        assert!(
            inline_strings > 0,
            "inline string constants are checked separately"
        );
        assert!(
            ordinary >= 3,
            "scalar and inline string constants are covered"
        );
    }
}
