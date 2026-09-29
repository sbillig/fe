//! Canonical guards, including unions, over semantic indices and scoped enum choices.
//!
//! Index conditions use reduced bit decisions over Fe's 256-bit `usize`, so equality,
//! disequality, and bounds share one Boolean algebra. Enum decisions have index conditions
//! as leaves. Neither graph enumerates array elements or depends on construction order.
use rustc_hash::FxHashMap;
use std::{
    cell::RefCell,
    cmp::{Ordering, Reverse},
    collections::{BTreeMap, BTreeSet},
    hash::Hash,
    sync::Arc,
};

use super::{
    decision::{Decision, Variable},
    index::{BinderScope, IndexExpr, IndexSubst},
    path::{Projection, StructuralPath},
};
use crate::analysis::semantic::{
    VariantIndex,
    normalized::{NRootId, NValueId},
};

const INDEX_BITS: u16 = 256;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ValueOccurrence {
    Value(NValueId),
    Root(NRootId),
    Argument(u32),
    Summary,
    SummaryChoice(u32),
    CallChoice { result: NValueId, choice: u32 },
}

impl Ord for ValueOccurrence {
    fn cmp(&self, other: &Self) -> Ordering {
        // Keep callee-local observations next to the caller values at that call.
        // Separating all Value and CallChoice occurrences makes unions of related
        // conditions exponential, including after they become SummaryChoices.
        let key = |occurrence: &Self| match *occurrence {
            Self::Value(value) => (0, value.as_u32(), None),
            Self::CallChoice { result, choice } => (0, result.as_u32(), Some(choice)),
            Self::Root(root) => (1, root.as_u32(), None),
            Self::Argument(argument) => (2, argument, None),
            Self::Summary => (3, 0, None),
            Self::SummaryChoice(choice) => (4, choice, None),
        };
        key(self).cmp(&key(other))
    }
}

impl PartialOrd for ValueOccurrence {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ChoiceKey<'db> {
    occurrence: ValueOccurrence,
    path: StructuralPath<IndexExpr<'db>>,
}

impl<'db> ChoiceKey<'db> {
    pub fn new(occurrence: ValueOccurrence, path: StructuralPath<IndexExpr<'db>>) -> Self {
        Self { occurrence, path }
    }
    fn alias_condition(&self, other: &Self) -> Option<IndexCondition<'db>> {
        if self.occurrence != other.occurrence
            || self.path.as_slice().len() != other.path.as_slice().len()
        {
            return None;
        }
        let mut condition = IndexCondition::always();
        for (left, right) in self.path.as_slice().iter().zip(other.path.as_slice()) {
            match (left, right) {
                (Projection::Index(left), Projection::Index(right)) => {
                    condition = condition.and(&IndexCondition::equal(*left, *right));
                }
                (left, right) if left == right => {}
                _ => return None,
            }
        }
        (!condition.is_never()).then_some(condition)
    }

    fn map_indices(&self, map: impl FnMut(&IndexExpr<'db>) -> IndexExpr<'db>) -> Self {
        Self {
            occurrence: self.occurrence,
            path: self.path.map_indices(map),
        }
    }
}

/// A bit of the index in one slot of a condition's sorted index table.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct SlotBit {
    // Interleaving words avoids an exponential equality graph.
    bit: Reverse<u16>,
    slot: u16,
}

impl SlotBit {
    fn new(slot: usize, bit: u16) -> Self {
        Self {
            bit: Reverse(bit),
            slot: u16::try_from(slot).expect("index condition table fits a slot"),
        }
    }
}

fn constant_bit(value: usize, bit: u16) -> bool {
    value
        .checked_shr(u32::from(bit))
        .is_some_and(|value| value & 1 != 0)
}

type BitDecision = Decision<SlotBit, bool>;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum SlotTarget {
    Slot(u16),
    Const(usize),
}

/// Bit-decision operations name table slots rather than indices, so equal
/// operations over different indices share one result. Guard algebra repeats
/// the same few operations across fixpoint iterations and summaries.
#[derive(Clone, PartialEq, Eq, Hash)]
enum BitOperation {
    And(BitDecision, BitDecision),
    Or(BitDecision, BitDecision),
    Not(BitDecision),
    Restrict(BitDecision, BitDecision),
    Substitute(BitDecision, Box<[SlotTarget]>),
    Exists(BitDecision, Box<[u16]>),
}

thread_local! {
    static CONSTANT_BITS: [BitDecision; 2] = [Decision::leaf(false), Decision::leaf(true)];
    static BIT_OPERATIONS: RefCell<FxHashMap<BitOperation, Option<BitDecision>>> =
        RefCell::default();
}

impl BitOperation {
    const LIMIT: usize = 4096;

    fn run(self) -> Option<BitDecision> {
        if let Some(result) = BIT_OPERATIONS.with_borrow(|results| results.get(&self).cloned()) {
            return result;
        }
        let result = match &self {
            Self::And(lhs, rhs) => Some(lhs.apply(rhs, |left, right| *left && *right)),
            Self::Or(lhs, rhs) => Some(lhs.apply(rhs, |left, right| *left || *right)),
            Self::Not(decision) => Some(decision.map(|bit| Variable::Symbol(*bit), |value| !value)),
            Self::Restrict(decision, care) => {
                decision.restrict(care, &false, |value, care| care.then_some(*value))
            }
            Self::Substitute(decision, targets) => Some(decision.map(
                |bit| match targets[usize::from(bit.slot)] {
                    SlotTarget::Const(value) => Variable::Constant(constant_bit(value, bit.bit.0)),
                    SlotTarget::Slot(slot) => Variable::Symbol(SlotBit { slot, ..*bit }),
                },
                |value| *value,
            )),
            Self::Exists(decision, slots) => Some(decision.exists(
                |bit| slots.contains(&bit.slot),
                |left, right| *left || *right,
            )),
        };
        BIT_OPERATIONS.with_borrow_mut(|results| {
            if results.len() >= Self::LIMIT {
                results.clear();
            }
            results.insert(self, result.clone());
        });
        result
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct IndexCondition<'db> {
    // Sorted and distinct: exactly the indices the decision reads.
    indices: Arc<[IndexExpr<'db>]>,
    decision: BitDecision,
}

impl Ord for IndexCondition<'_> {
    // Order as the decision over the indices themselves.
    fn cmp(&self, other: &Self) -> Ordering {
        if self == other {
            return Ordering::Equal;
        }
        self.decision.cmp_by(&other.decision, |left, right| {
            left.bit.cmp(&right.bit).then_with(|| {
                self.indices[usize::from(left.slot)].cmp(&other.indices[usize::from(right.slot)])
            })
        })
    }
}

impl PartialOrd for IndexCondition<'_> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<'db> IndexCondition<'db> {
    fn constant(value: bool) -> Self {
        Self {
            indices: Arc::new([]),
            // Every guard leaf is one of these two, and a fresh leaf would be
            // another backing graph denoting the same constant.
            decision: CONSTANT_BITS.with(|bits| bits[usize::from(value)].clone()),
        }
    }
    fn always() -> Self {
        Self::constant(true)
    }
    fn never() -> Self {
        Self::constant(false)
    }
    fn is_always(&self) -> bool {
        self.decision.is_leaf(&true)
    }
    fn is_never(&self) -> bool {
        self.decision.is_leaf(&false)
    }

    /// Drop table slots the decision no longer reads.
    fn compact(indices: Arc<[IndexExpr<'db>]>, decision: BitDecision) -> Self {
        let mut used = vec![false; indices.len()];
        for bit in decision.variables() {
            used[usize::from(bit.slot)] = true;
        }
        if used.iter().all(|used| *used) {
            return Self { indices, decision };
        }
        let mut next = 0;
        // An unread slot's target is never consulted.
        let targets = used
            .iter()
            .map(|used| {
                if *used {
                    next += 1;
                    SlotTarget::Slot(next - 1)
                } else {
                    SlotTarget::Const(0)
                }
            })
            .collect();
        Self {
            indices: indices
                .iter()
                .zip(&used)
                .filter_map(|(index, used)| used.then_some(*index))
                .collect(),
            decision: BitOperation::Substitute(decision, targets).run().unwrap(),
        }
    }

    /// Express both decisions over one merged table.
    fn aligned(&self, other: &Self) -> (Arc<[IndexExpr<'db>]>, BitDecision, BitDecision) {
        if self.indices == other.indices {
            return (
                self.indices.clone(),
                self.decision.clone(),
                other.decision.clone(),
            );
        }
        let indices: Arc<[_]> = self
            .indices
            .iter()
            .chain(other.indices.iter())
            .copied()
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        let relabel = |condition: &Self| {
            if condition.indices.len() == indices.len() {
                return condition.decision.clone();
            }
            let targets = condition
                .indices
                .iter()
                .map(|index| {
                    let slot = indices.binary_search(index).unwrap();
                    SlotTarget::Slot(u16::try_from(slot).unwrap())
                })
                .collect();
            BitOperation::Substitute(condition.decision.clone(), targets)
                .run()
                .unwrap()
        };
        let (lhs, rhs) = (relabel(self), relabel(other));
        (indices, lhs, rhs)
    }

    fn equal(lhs: IndexExpr<'db>, rhs: IndexExpr<'db>) -> Self {
        let (lhs, rhs) = if lhs <= rhs { (lhs, rhs) } else { (rhs, lhs) };
        match (lhs, rhs) {
            (lhs, rhs) if lhs == rhs => Self::always(),
            (IndexExpr::Const(_), IndexExpr::Const(_)) => Self::never(),
            (IndexExpr::Const(value), index) => Self {
                indices: Arc::new([index]),
                decision: Decision::chain(
                    (0..INDEX_BITS).map(|bit| (SlotBit::new(0, bit), constant_bit(value, bit))),
                    true,
                    false,
                ),
            },
            (lhs, rhs) => Self {
                indices: Arc::new([lhs, rhs]),
                decision: Decision::equal_bits(
                    (0..INDEX_BITS).map(|bit| (SlotBit::new(0, bit), SlotBit::new(1, bit))),
                    true,
                    false,
                ),
            },
        }
    }

    fn bounded(index: IndexExpr<'db>, len: IndexExpr<'db>) -> Self {
        if index == len {
            return Self::never();
        }
        if let (IndexExpr::Const(value), IndexExpr::Const(len)) = (index, len) {
            return Self::constant(value < len);
        }
        if let IndexExpr::Const(len) = len {
            return Self::compact(
                Arc::new([index]),
                Decision::upper_bound_bits(
                    (0..INDEX_BITS).map(|bit| (SlotBit::new(0, bit), constant_bit(len, bit))),
                ),
            );
        }
        let indices: Vec<_> = [index, len]
            .into_iter()
            .filter(|index| !matches!(index, IndexExpr::Const(_)))
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        let decision = Decision::less_bits((0..INDEX_BITS).map(|bit| {
            let word_bit = |index| match index {
                IndexExpr::Const(value) => Variable::Constant(constant_bit(value, bit)),
                index => {
                    Variable::Symbol(SlotBit::new(indices.binary_search(&index).unwrap(), bit))
                }
            };
            (word_bit(index), word_bit(len))
        }));
        Self::compact(indices.into(), decision)
    }

    fn and(&self, other: &Self) -> Self {
        if self == other || other.is_always() {
            return self.clone();
        }
        if self.is_always() {
            return other.clone();
        }
        if self.is_never() || other.is_never() {
            return Self::never();
        }
        let (indices, lhs, rhs) = self.aligned(other);
        Self::compact(indices, BitOperation::And(lhs, rhs).run().unwrap())
    }

    fn or(&self, other: &Self) -> Self {
        if self == other || other.is_never() {
            return self.clone();
        }
        if self.is_never() {
            return other.clone();
        }
        if self.is_always() || other.is_always() {
            return Self::always();
        }
        let (indices, lhs, rhs) = self.aligned(other);
        Self::compact(indices, BitOperation::Or(lhs, rhs).run().unwrap())
    }

    fn not(&self) -> Self {
        Self {
            indices: self.indices.clone(),
            decision: BitOperation::Not(self.decision.clone()).run().unwrap(),
        }
    }
    fn implies(&self, other: &Self) -> bool {
        self.and(&other.not()).is_never()
    }
    fn substitute(&self, subst: &IndexSubst<'db>) -> Self {
        let targets: Vec<_> = self
            .indices
            .iter()
            .map(|index| subst.apply(*index))
            .collect();
        if targets.iter().eq(self.indices.iter()) {
            return self.clone();
        }
        let indices: Vec<_> = targets
            .iter()
            .filter(|index| !matches!(index, IndexExpr::Const(_)))
            .copied()
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        // An order-preserving rename leaves every slot in place.
        if indices.len() == targets.len() && targets.is_sorted() {
            return Self {
                indices: indices.into(),
                decision: self.decision.clone(),
            };
        }
        let targets = targets
            .into_iter()
            .map(|index| match index {
                IndexExpr::Const(value) => SlotTarget::Const(value),
                index => {
                    SlotTarget::Slot(u16::try_from(indices.binary_search(&index).unwrap()).unwrap())
                }
            })
            .collect();
        Self::compact(
            indices.into(),
            BitOperation::Substitute(self.decision.clone(), targets)
                .run()
                .unwrap(),
        )
    }
    fn indices(&self) -> impl Iterator<Item = IndexExpr<'db>> + '_ {
        self.indices.iter().copied()
    }

    /// Existentially quantify the selected indices.
    fn project(&self, mut hidden: impl FnMut(IndexExpr<'db>) -> bool) -> Self {
        let slots: Box<[u16]> = self
            .indices
            .iter()
            .enumerate()
            .filter(|(_, index)| hidden(**index))
            .map(|(slot, _)| u16::try_from(slot).unwrap())
            .collect();
        if slots.is_empty() {
            return self.clone();
        }
        Self::compact(
            self.indices.clone(),
            BitOperation::Exists(self.decision.clone(), slots)
                .run()
                .unwrap(),
        )
    }

    fn restrict(&self, care: &Self) -> Option<Self> {
        let (indices, decision, care) = self.aligned(care);
        BitOperation::Restrict(decision, care)
            .run()
            .map(|decision| Self::compact(indices, decision))
    }

    fn node_count(&self) -> usize {
        self.decision.node_count()
    }

    // Find a representative only when the complete condition proves equality. Partial
    // known bits and coincident numeric IDs never establish index correlation.
    fn representatives(
        &self,
        indices: &BTreeSet<IndexExpr<'db>>,
    ) -> BTreeMap<IndexExpr<'db>, IndexExpr<'db>> {
        let mut representatives = BTreeMap::new();
        let witness = self.decision.witness(|value| *value).unwrap_or_default();
        for index in indices {
            if matches!(index, IndexExpr::Const(_)) {
                continue;
            }
            let mut candidate = Some(0usize);
            for (bit, value) in &witness {
                if self.indices[usize::from(bit.slot)] == *index && *value {
                    candidate = candidate.and_then(|candidate| {
                        1usize
                            .checked_shl(u32::from(bit.bit.0))
                            .map(|bit| candidate | bit)
                    });
                }
            }
            if let Some(candidate) = candidate
                && self.implies(&Self::equal(*index, IndexExpr::Const(candidate)))
            {
                representatives.insert(*index, IndexExpr::Const(candidate));
                continue;
            }
            if let Some(previous) = indices
                .range(..*index)
                .find(|previous| self.implies(&Self::equal(**previous, *index)))
            {
                representatives.insert(
                    *index,
                    representatives.get(previous).copied().unwrap_or(*previous),
                );
            }
        }
        representatives
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct ChoiceBit<'db> {
    bit: Reverse<u16>,
    // One key is read by every bit node of its choice, and inlining it in each of
    // them dominated the size of a branch.
    choice: Arc<ChoiceKey<'db>>,
}

/// Rebuild keys once per distinct source key rather than once per bit node. The
/// source graph's bits already share one key, so its address identifies it.
fn renamed_choices<'db>(
    mut rename: impl FnMut(&ChoiceKey<'db>) -> ChoiceKey<'db>,
) -> impl FnMut(&ChoiceBit<'db>) -> Variable<ChoiceBit<'db>> {
    let mut renamed = FxHashMap::default();
    move |bit| {
        let choice = renamed
            .entry(Arc::as_ptr(&bit.choice) as usize)
            .or_insert_with(|| Arc::new(rename(&bit.choice)))
            .clone();
        Variable::Symbol(ChoiceBit {
            choice,
            bit: bit.bit,
        })
    }
}

impl Ord for ChoiceBit<'_> {
    fn cmp(&self, other: &Self) -> Ordering {
        // Interleaving unrelated enum words makes unions of tag tests exponential.
        // Keep each independent occurrence/structural slot together. Within one
        // slot, indexed selections may alias, so interleave their bits to keep
        // the exact tag-equality relations in Guard::canonical compact as well.
        self.choice
            .occurrence
            .cmp(&other.choice.occurrence)
            .then_with(|| {
                self.choice
                    .path
                    .as_slice()
                    .iter()
                    .map(|step| step.map_index(|_| ()))
                    .cmp(
                        other
                            .choice
                            .path
                            .as_slice()
                            .iter()
                            .map(|step| step.map_index(|_| ())),
                    )
            })
            .then_with(|| self.bit.cmp(&other.bit))
            .then_with(|| self.choice.path.cmp(&other.choice.path))
    }
}

impl PartialOrd for ChoiceBit<'_> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Leaves sit behind a handle: inlining an index condition in every node made a
/// branch pay for a payload only the leaves carry.
type Leaf<'db> = Arc<IndexCondition<'db>>;

/// A guard's decision over enum choices, with index conditions at its leaves.
///
/// The operations are named in terms of choice keys and leaves rather than the
/// graph underneath, because that is the whole vocabulary guards need and it is
/// what a slot-indexed table would have to intercept.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct Condition<'db>(Decision<ChoiceBit<'db>, Leaf<'db>>);

thread_local! {
    static CONSTANT_LEAVES: [Leaf<'static>; 2] =
        [Arc::new(IndexCondition::never()), Arc::new(IndexCondition::always())];
}

/// The two leaves every guard bottoms out in, shared rather than rebuilt.
fn constant_leaf<'db>(value: bool) -> Leaf<'db> {
    CONSTANT_LEAVES.with(|leaves| leaves[usize::from(value)].clone())
}

fn joined<'db>(
    join: impl Fn(&IndexCondition<'db>, &IndexCondition<'db>) -> IndexCondition<'db>,
) -> impl Fn(&Leaf<'db>, &Leaf<'db>) -> Leaf<'db> {
    move |left, right| Arc::new(join(left, right))
}

impl<'db> Condition<'db> {
    fn constant(value: bool) -> Self {
        Self(Decision::leaf(constant_leaf(value)))
    }

    /// One choice's tag bits, accepting exactly the valuation `value` describes.
    fn choice_bits(choice: &Arc<ChoiceKey<'db>>, bits: u16, value: impl Fn(u16) -> bool) -> Self {
        Self(Decision::chain(
            (0..bits).map(|bit| {
                (
                    ChoiceBit {
                        choice: choice.clone(),
                        bit: Reverse(bit),
                    },
                    value(bit),
                )
            }),
            constant_leaf(true),
            constant_leaf(false),
        ))
    }

    /// Two choices agree on every tag bit, or `mismatch` holds.
    fn tag_equality(
        left: &Arc<ChoiceKey<'db>>,
        right: &Arc<ChoiceKey<'db>>,
        mismatch: Leaf<'db>,
    ) -> Self {
        Self(Decision::equal_bits(
            (0..u16::BITS as u16).map(|bit| {
                (
                    ChoiceBit {
                        bit: Reverse(bit),
                        choice: left.clone(),
                    },
                    ChoiceBit {
                        bit: Reverse(bit),
                        choice: right.clone(),
                    },
                )
            }),
            constant_leaf(true),
            mismatch,
        ))
    }

    fn is_always(&self) -> bool {
        self.0.leaf_value().is_some_and(|leaf| leaf.is_always())
    }

    fn is_never(&self) -> bool {
        self.0.leaf_value().is_some_and(|leaf| leaf.is_never())
    }

    fn and(&self, other: &Self) -> Self {
        Self(self.0.apply(&other.0, joined(IndexCondition::and)))
    }

    fn or(&self, other: &Self) -> Self {
        Self(self.0.apply(&other.0, joined(IndexCondition::or)))
    }

    /// Everything this decision accepts that `other` rejects.
    fn without(&self, other: &Self) -> Self {
        Self(
            self.0
                .apply(&other.0, |left, right| Arc::new(left.and(&right.not()))),
        )
    }

    /// Rebuild the choice keys this decision reads.
    fn map_keys(&self, rename: impl FnMut(&ChoiceKey<'db>) -> ChoiceKey<'db>) -> Self {
        Self(self.0.map(renamed_choices(rename), Clone::clone))
    }

    fn map_leaves(&self, mut leaf: impl FnMut(&Leaf<'db>) -> Leaf<'db>) -> Self {
        Self(
            self.0
                .map(|bit| Variable::Symbol(bit.clone()), |old| leaf(old)),
        )
    }

    /// Rebuild keys and leaves together, as canonicalization does per alternative.
    fn map_keys_and_leaves(
        &self,
        rename: impl FnMut(&ChoiceKey<'db>) -> ChoiceKey<'db>,
        mut leaf: impl FnMut(&Leaf<'db>) -> Leaf<'db>,
    ) -> Self {
        Self(self.0.map(renamed_choices(rename), |old| leaf(old)))
    }

    /// Existentially quantify every key the predicate selects.
    fn forget_keys(&self, mut selected: impl FnMut(&ChoiceKey<'db>) -> bool) -> Self {
        Self(
            self.0
                .exists(|bit| selected(&bit.choice), joined(IndexCondition::or)),
        )
    }

    /// Complete this decision outside the valuations `care` admits.
    fn restricted(&self, care: &Self) -> Option<Self> {
        self.0
            .restrict(&care.0, &constant_leaf(false), |value, care| {
                value.restrict(care).map(Arc::new)
            })
            .map(Self)
    }

    fn keys(&self) -> impl Iterator<Item = &ChoiceKey<'db>> {
        self.0.variables().map(|bit| &*bit.choice)
    }

    /// The keys as shared handles, for rebuilding decisions over them.
    fn key_handles(&self) -> impl Iterator<Item = &Arc<ChoiceKey<'db>>> {
        self.0.variables().map(|bit| &bit.choice)
    }

    fn leaves(&self) -> impl Iterator<Item = &Leaf<'db>> {
        self.0.leaves()
    }

    fn node_count(&self) -> usize {
        self.0.node_count() + self.0.leaves().map(|leaf| leaf.node_count()).sum::<usize>()
    }

    fn is_sole_owner(&self) -> bool {
        self.0.is_sole_owner()
    }
}

/// Fixpoint iteration rebuilds values from the same guards, so their unions and
/// intersections repeat. Guards are immutable; a cached result is the result.
#[derive(Default)]
pub struct GuardCache<'db> {
    conjunctions: FxHashMap<(Guard<'db>, Guard<'db>), Option<Guard<'db>>>,
    disjunctions: FxHashMap<(Guard<'db>, Guard<'db>), Option<Guard<'db>>>,
    substitutions: FxHashMap<(Guard<'db>, IndexSubst<'db>), Option<Guard<'db>>>,
    canonical: CanonicalGuards<'db>,
    pending: usize,
}

impl<'db> GuardCache<'db> {
    /// Operations recorded between reclamation sweeps. A sweep costs one pass over
    /// the memo, so amortize it rather than rescanning once it is full.
    const SWEEP: usize = 4096;

    pub fn and(&mut self, lhs: &Guard<'db>, rhs: &Guard<'db>) -> Option<Guard<'db>> {
        let mut canonical = std::mem::take(&mut self.canonical);
        let result = Self::cached(
            &mut self.conjunctions,
            lhs,
            rhs,
            &mut canonical,
            &mut self.pending,
            Guard::and,
        );
        self.canonical = canonical;
        result
    }

    pub fn or(&mut self, lhs: &Guard<'db>, rhs: &Guard<'db>) -> Guard<'db> {
        let mut canonical = std::mem::take(&mut self.canonical);
        let result = Self::cached(
            &mut self.disjunctions,
            lhs,
            rhs,
            &mut canonical,
            &mut self.pending,
            |lhs, rhs| Some(lhs.or(rhs)),
        );
        self.canonical = canonical;
        result.expect("a union of satisfiable guards is satisfiable")
    }

    pub fn substitute(
        &mut self,
        guard: &Guard<'db>,
        subst: &IndexSubst<'db>,
    ) -> Option<Guard<'db>> {
        let mut canonical = std::mem::take(&mut self.canonical);
        let result = Self::cached(
            &mut self.substitutions,
            guard,
            subst,
            &mut canonical,
            &mut self.pending,
            Guard::substitute,
        );
        self.canonical = canonical;
        result
    }

    pub fn canonical(&mut self) -> &mut CanonicalGuards<'db> {
        &mut self.canonical
    }

    /// Recomputing an operation rebuilds a complete decision graph even when an equal
    /// one is already live, so hand back the shared representative rather than the
    /// fresh copy. Entries hold representatives, and a representative nothing else
    /// holds is dead storage, so sweep those instead of discarding the whole memo:
    /// clearing it would only make the same graphs be rebuilt again.
    fn cached<K: Clone + Eq + Hash>(
        results: &mut FxHashMap<(Guard<'db>, K), Option<Guard<'db>>>,
        lhs: &Guard<'db>,
        rhs: &K,
        canonical: &mut CanonicalGuards<'db>,
        pending: &mut usize,
        operation: impl FnOnce(&Guard<'db>, &K) -> Option<Guard<'db>>,
    ) -> Option<Guard<'db>> {
        let key = (lhs.clone(), rhs.clone());
        if let Some(result) = results.get(&key) {
            return result.clone();
        }
        let result = operation(lhs, rhs).map(|guard| canonical.share(guard));
        *pending += 1;
        if *pending >= Self::SWEEP {
            *pending = 0;
            // An entry nothing else holds is dead storage; an infeasible one owns
            // nothing and costs only its key.
            results.retain(|_, result| result.as_ref().is_some_and(Guard::is_sole_owner));
        }
        results.insert(key, result.clone());
        result
    }
}

/// Separate operations that complete an equal guard each allocate a complete
/// decision graph, and interned values, regions and summaries then retain every
/// copy. Equal guards are interchangeable, so keep one representative per
/// structure and let the other copies drop.
#[derive(Default)]
pub struct CanonicalGuards<'db> {
    /// Each entry keeps its own node count, so reclaiming never rewalks the graphs.
    guards: FxHashMap<Guard<'db>, usize>,
    nodes: usize,
    pending: usize,
}

impl<'db> CanonicalGuards<'db> {
    /// Inserts between reclamation sweeps. A sweep costs one pass over the table, so
    /// amortized it is constant per insert.
    const SWEEP: usize = 1024;

    pub fn share(&mut self, guard: Guard<'db>) -> Guard<'db> {
        if let Some((shared, _)) = self.guards.get_key_value(&guard) {
            return shared.clone();
        }
        // A representative nothing else holds is dead storage. Dropping those keeps
        // the table's own retention proportional to what the analysis still holds,
        // so sharing never has to be given up to bound memory.
        self.pending += 1;
        if self.pending >= Self::SWEEP {
            self.pending = 0;
            self.guards.retain(|guard, _| !guard.is_sole_owner());
            self.nodes = self.guards.values().sum();
        }
        let nodes = guard.node_count();
        self.nodes += nodes;
        self.guards.insert(guard.clone(), nodes);
        guard
    }

    pub fn retained_nodes(&self) -> usize {
        self.nodes
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Guard<'db> {
    scope: BinderScope,
    condition: Condition<'db>,
}

impl<'db> Guard<'db> {
    pub fn always(scope: &BinderScope) -> Self {
        Self {
            scope: scope.clone(),
            condition: Condition::constant(true),
        }
    }
    pub fn scope(&self) -> &BinderScope {
        &self.scope
    }

    pub fn and(&self, other: &Self) -> Option<Self> {
        assert_eq!(self.scope, other.scope, "guard scopes must match");
        if self == other || other.condition.is_always() {
            return Some(self.clone());
        }
        if self.condition.is_always() {
            return Some(other.clone());
        }
        Self::canonical(&self.scope, self.condition.and(&other.condition))
    }

    pub fn or(&self, other: &Self) -> Self {
        assert_eq!(self.scope, other.scope, "guard scopes must match");
        if self == other || self.condition.is_always() {
            return self.clone();
        }
        if other.condition.is_always() {
            return other.clone();
        }
        Self::canonical(&self.scope, self.condition.or(&other.condition))
            .expect("a union of satisfiable guards is satisfiable")
    }

    pub fn with_equality(&self, lhs: IndexExpr<'db>, rhs: IndexExpr<'db>) -> Option<Self> {
        self.scope.validate(lhs).expect("free equality binder");
        self.scope.validate(rhs).expect("free equality binder");
        self.with_index_condition(&IndexCondition::equal(lhs, rhs))
    }

    pub fn with_equalities(
        &self,
        pairs: impl IntoIterator<Item = (IndexExpr<'db>, IndexExpr<'db>)>,
    ) -> Option<Self> {
        pairs
            .into_iter()
            .try_fold(self.clone(), |guard, (lhs, rhs)| {
                guard.with_equality(lhs, rhs)
            })
    }

    pub fn with_disequality(&self, lhs: IndexExpr<'db>, rhs: IndexExpr<'db>) -> Option<Self> {
        self.scope.validate(lhs).expect("free disequality binder");
        self.scope.validate(rhs).expect("free disequality binder");
        self.with_index_condition(&IndexCondition::equal(lhs, rhs).not())
    }

    pub fn with_bound(
        &self,
        index: IndexExpr<'db>,
        len: impl Into<IndexExpr<'db>>,
    ) -> Option<Self> {
        self.scope.validate(index).expect("free bound binder");
        let len = len.into();
        self.scope.validate(len).expect("free bound length binder");
        self.with_index_condition(&IndexCondition::bounded(index, len))
    }

    pub fn with_variant(&self, choice: ChoiceKey<'db>, variant: VariantIndex) -> Option<Self> {
        self.with_choice_bits(choice, u16::BITS as u16, |bit| variant.0 & (1 << bit) != 0)
    }

    /// Restrict an actual boolean value occurrence to one of its two outcomes.
    pub fn with_boolean(&self, choice: ChoiceKey<'db>, value: bool) -> Option<Self> {
        self.with_choice_bits(choice, 1, |_| value)
    }

    fn with_choice_bits(
        &self,
        choice: ChoiceKey<'db>,
        bits: u16,
        value: impl Fn(u16) -> bool,
    ) -> Option<Self> {
        for index in choice.path.indices() {
            self.scope.validate(index).expect("free choice binder");
        }
        let selected = Condition::choice_bits(&Arc::new(choice), bits, value);
        Self::canonical(&self.scope, self.condition.and(&selected))
    }

    pub fn substitute(&self, subst: &IndexSubst<'db>) -> Option<Self> {
        assert_eq!(
            &self.scope,
            subst.source(),
            "substitution source scope must match"
        );
        // A substitution without entries only extends the scope; the decision
        // graph and its variables are unchanged.
        if subst.preserves_indices() {
            return Some(Self {
                scope: subst.destination().clone(),
                condition: self.condition.clone(),
            });
        }
        Self::canonical(
            subst.destination(),
            self.condition.map_keys_and_leaves(
                |choice| choice.map_indices(|index| subst.apply(*index)),
                |condition| Arc::new(condition.substitute(subst)),
            ),
        )
    }

    pub fn in_scope(&self, scope: &BinderScope) -> Self {
        if &self.scope == scope {
            return self.clone();
        }
        self.substitute(&IndexSubst::new(&self.scope, scope, []).expect("guard scope extension"))
            .expect("scope extension preserves satisfiability")
    }

    pub fn occurrences(&self) -> BTreeSet<ValueOccurrence> {
        self.condition
            .keys()
            .map(|choice| choice.occurrence)
            .collect()
    }

    pub fn map_occurrences(
        &self,
        mut map: impl FnMut(ValueOccurrence) -> ValueOccurrence,
    ) -> Option<Self> {
        let occurrences: BTreeSet<_> = self
            .condition
            .keys()
            .map(|choice| choice.occurrence)
            .collect();
        let mappings: BTreeMap<_, _> = occurrences
            .into_iter()
            .map(|occurrence| (occurrence, map(occurrence)))
            .collect();
        if mappings.iter().all(|(from, to)| from == to) {
            return Some(self.clone());
        }
        Self::canonical(
            &self.scope,
            self.condition.map_keys(|choice| {
                ChoiceKey::new(mappings[&choice.occurrence], choice.path.clone())
            }),
        )
    }

    /// A new execution of a loop may choose another enum alternative.
    pub fn forget_occurrences(&self, mut repeated: impl FnMut(ValueOccurrence) -> bool) -> Self {
        if !self.occurrences().into_iter().any(&mut repeated) {
            return self.clone();
        }
        Self::canonical(
            &self.scope,
            self.condition
                .forget_keys(|choice| repeated(choice.occurrence)),
        )
        .expect("existential quantification preserves feasibility")
    }

    /// Forget old scalar selectors without identifying them with a new execution.
    pub fn forget_indices(&self, mut repeated: impl FnMut(IndexExpr<'db>) -> bool) -> Self {
        let indices: BTreeSet<_> = self
            .indices()
            .into_iter()
            .filter(|index| repeated(*index))
            .collect();
        if indices.is_empty() {
            return self.clone();
        }
        let condition = self
            .condition
            .forget_keys(|choice| choice.path.indices().any(|index| indices.contains(&index)))
            .map_leaves(|condition| Arc::new(condition.project(|index| indices.contains(&index))));
        Self::canonical(&self.scope, condition)
            .expect("existential quantification preserves feasibility")
    }

    pub fn difference(&self, other: &Self) -> Option<Self> {
        assert_eq!(self.scope, other.scope, "guard scopes must match");
        Self::canonical(&self.scope, self.condition.without(&other.condition))
    }

    /// Eliminate clause-local witnesses that are observable only through scalar
    /// constraints. A witness indexing an enum choice remains observable: its
    /// quantification would require also quantifying that indexed choice.
    pub fn project_witnesses(&self, hidden: impl Fn(IndexExpr<'db>) -> bool) -> Self {
        let indexed: BTreeSet<_> = self
            .condition
            .keys()
            .flat_map(|choice| choice.path.indices())
            .collect();
        let projected: BTreeSet<_> = self
            .indices()
            .into_iter()
            .filter(|index| hidden(*index) && !indexed.contains(index))
            .collect();
        if projected.is_empty() {
            return self.clone();
        }
        Self::canonical(
            &self.scope,
            self.condition.map_leaves(|condition| {
                Arc::new(condition.project(|index| projected.contains(&index)))
            }),
        )
        .expect("existential projection preserves feasibility")
    }

    pub fn implies(&self, other: &Self) -> bool {
        assert_eq!(self.scope, other.scope, "guard scopes must match");
        self.condition.without(&other.condition).is_never()
    }

    pub fn proves_equal(&self, lhs: IndexExpr<'db>, rhs: IndexExpr<'db>) -> bool {
        self.scope.validate(lhs).expect("free equality binder");
        self.scope.validate(rhs).expect("free equality binder");
        let equality = IndexCondition::equal(lhs, rhs);
        self.condition
            .leaves()
            .all(|condition| condition.implies(&equality))
    }

    pub fn indices(&self) -> BTreeSet<IndexExpr<'db>> {
        let mut indices: BTreeSet<_> = self
            .condition
            .leaves()
            .flat_map(|leaf| leaf.indices())
            .collect();
        for choice in self.condition.keys() {
            indices.extend(choice.path.indices());
        }
        indices
    }

    /// No other guard holds this condition's backing graph.
    pub fn is_sole_owner(&self) -> bool {
        self.condition.is_sole_owner()
    }

    pub fn node_count(&self) -> usize {
        self.condition.node_count()
    }

    fn with_index_condition(&self, condition: &IndexCondition<'db>) -> Option<Self> {
        Self::canonical(
            &self.scope,
            self.condition
                .map_leaves(|old| Arc::new(old.and(condition))),
        )
    }

    // Indexed choice keys must also respect equalities proved by their index condition.
    // Identifying choice bits rebuilds the ordered graph and rejects conflicting variants;
    // no map collection is allowed to overwrite a contradictory requirement.
    fn canonical(scope: &BinderScope, condition: Condition<'db>) -> Option<Self> {
        if condition.is_never() {
            return None;
        }
        if condition.keys().all(|choice| {
            choice
                .path
                .indices()
                .all(|index| matches!(index, IndexExpr::Const(_)))
        }) {
            return Some(Self {
                scope: scope.clone(),
                condition,
            });
        }
        let choice_indices: BTreeSet<_> = condition
            .keys()
            .flat_map(|choice| choice.path.indices())
            .collect();
        let mut canonical = Condition::constant(false);
        // Partition by distinct terminal conditions, not graph paths: shared suffixes
        // can represent exponentially many paths through a compact decision graph.
        for indices in condition.leaves().filter(|indices| !indices.is_never()) {
            let terms = choice_indices
                .iter()
                .copied()
                .chain(indices.indices())
                .collect();
            let representatives = indices.representatives(&terms);
            let alternative = condition.map_keys_and_leaves(
                |choice| {
                    choice
                        .map_indices(|index| representatives.get(index).copied().unwrap_or(*index))
                },
                |leaf| {
                    if leaf == indices {
                        leaf.clone()
                    } else {
                        constant_leaf(false)
                    }
                },
            );
            canonical = canonical.or(&alternative);
        }
        // Choice occurrences at equal indices must have equal tags. Complete the
        // graph outside these feasible valuations so equality partitions can reunite
        // without retaining a spurious dependence on an extra indexed choice.
        let choices: BTreeSet<_> = canonical.key_handles().collect();
        let mut care = Condition::constant(true);
        for (position, left) in choices.iter().copied().enumerate() {
            for right in choices.iter().copied().skip(position + 1) {
                if let Some(alias) = left.alias_condition(right) {
                    care = care.and(&Condition::tag_equality(left, right, Arc::new(alias.not())));
                }
            }
        }
        let canonical = canonical.restricted(&care)?;
        (!canonical.is_never()).then(|| Self {
            scope: scope.clone(),
            condition: canonical,
        })
    }
}
