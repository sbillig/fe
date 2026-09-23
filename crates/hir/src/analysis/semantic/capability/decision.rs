//! Reduced ordered decision graphs with canonical, allocation-independent node numbering.
#[cfg(test)]
use std::cell::Cell;

use rustc_hash::FxHashMap;
use std::{
    collections::{BTreeMap, BTreeSet},
    hash::Hash,
    sync::Arc,
};

#[cfg(test)]
thread_local! {
    static INTERN_ATTEMPTS: Cell<usize> = const { Cell::new(0) };
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum Node<V, T> {
    Leaf(T),
    Branch {
        variable: V,
        low: usize,
        high: usize,
    },
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(super) struct Decision<V, T> {
    // Postorder, low edge first. The last node is the root; no unreachable nodes remain.
    nodes: Arc<[Node<V, T>]>,
}

#[derive(Clone, Debug)]
pub(super) enum Variable<V> {
    Constant(bool),
    Symbol(V),
}

struct Builder<V, T> {
    nodes: Vec<Node<V, T>>,
    interned: FxHashMap<Node<V, T>, usize>,
    selections: FxHashMap<(V, usize, usize), usize>,
}

impl<V: Clone + Ord + Hash, T: Clone + Eq + Hash> Builder<V, T> {
    fn new() -> Self {
        Self {
            nodes: Vec::new(),
            interned: FxHashMap::default(),
            selections: FxHashMap::default(),
        }
    }

    fn intern(&mut self, node: Node<V, T>) -> usize {
        #[cfg(test)]
        INTERN_ATTEMPTS.set(INTERN_ATTEMPTS.get() + 1);
        if let Some(id) = self.interned.get(&node) {
            return *id;
        }
        let id = self.nodes.len();
        self.nodes.push(node.clone());
        self.interned.insert(node, id);
        id
    }

    fn branch(&mut self, variable: V, low: usize, high: usize) -> usize {
        if low == high {
            low
        } else {
            self.intern(Node::Branch {
                variable,
                low,
                high,
            })
        }
    }

    fn variable(&self, node: usize) -> Option<&V> {
        match &self.nodes[node] {
            Node::Leaf(_) => None,
            Node::Branch { variable, .. } => Some(variable),
        }
    }

    fn cofactors(&self, node: usize, split: &V) -> (usize, usize) {
        match &self.nodes[node] {
            Node::Branch {
                variable,
                low,
                high,
            } if variable == split => (*low, *high),
            _ => (node, node),
        }
    }

    fn import(&mut self, decision: &Decision<V, T>) -> usize {
        let mut mapped = Vec::with_capacity(decision.nodes.len());
        for node in decision.nodes.iter() {
            let id = match node {
                Node::Leaf(value) => self.intern(Node::Leaf(value.clone())),
                Node::Branch {
                    variable,
                    low,
                    high,
                } => self.branch(variable.clone(), mapped[*low], mapped[*high]),
            };
            mapped.push(id);
        }
        mapped[decision.root()]
    }

    /// An `idempotent` join is also commutative, as in quantification, so
    /// equal operands and swapped pairs share work.
    fn apply(
        &mut self,
        lhs: usize,
        rhs: usize,
        join: &impl Fn(&T, &T) -> T,
        idempotent: bool,
        memo: &mut FxHashMap<(usize, usize), usize>,
    ) -> usize {
        if idempotent && lhs == rhs {
            return lhs;
        }
        let key = if idempotent {
            (lhs.min(rhs), lhs.max(rhs))
        } else {
            (lhs, rhs)
        };
        if let Some(result) = memo.get(&key) {
            return *result;
        }
        // Keep recursive frames small: only the split variable lives across calls.
        let result =
            if let (Node::Leaf(left), Node::Leaf(right)) = (&self.nodes[lhs], &self.nodes[rhs]) {
                let leaf = join(left, right);
                self.intern(Node::Leaf(leaf))
            } else {
                let variable = match (self.variable(lhs), self.variable(rhs)) {
                    (Some(left), Some(right)) => left.min(right),
                    (Some(variable), None) | (None, Some(variable)) => variable,
                    (None, None) => unreachable!("two leaves are joined directly"),
                }
                .clone();
                let (left_low, left_high) = self.cofactors(lhs, &variable);
                let (right_low, right_high) = self.cofactors(rhs, &variable);
                let low = self.apply(left_low, right_low, join, idempotent, memo);
                let high = self.apply(left_high, right_high, join, idempotent, memo);
                self.branch(variable, low, high)
            };
        memo.insert(key, result);
        result
    }

    // Substitution may reorder variables or identify two decisions. Rebuild by Shannon
    // expansion; directly relabeling an ordered graph would violate its invariant.
    fn select(&mut self, variable: V, low: usize, high: usize) -> usize {
        if low == high {
            return low;
        }
        let key = (variable.clone(), low, high);
        if let Some(result) = self.selections.get(&key) {
            return *result;
        }
        let first = self
            .variable(low)
            .into_iter()
            .chain(self.variable(high))
            .chain([&variable])
            .min()
            .unwrap()
            .clone();
        let (low_false, low_true) = self.cofactors(low, &first);
        let (high_false, high_true) = self.cofactors(high, &first);
        let (low, high) = if first == variable {
            (low_false, high_true)
        } else {
            (
                self.select(variable.clone(), low_false, high_false),
                self.select(variable, low_true, high_true),
            )
        };
        let result = self.branch(first, low, high);
        self.selections.insert(key, result);
        result
    }

    fn finish(self, root: usize) -> Decision<V, T> {
        let mut nodes = Vec::new();
        let mut numbering = FxHashMap::default();
        self.visit(root, &mut nodes, &mut numbering);
        Decision {
            nodes: nodes.into(),
        }
    }

    fn visit(
        &self,
        id: usize,
        nodes: &mut Vec<Node<V, T>>,
        numbering: &mut FxHashMap<usize, usize>,
    ) -> usize {
        if let Some(id) = numbering.get(&id) {
            return *id;
        }
        let node = match &self.nodes[id] {
            Node::Leaf(value) => Node::Leaf(value.clone()),
            Node::Branch {
                variable,
                low,
                high,
            } => Node::Branch {
                variable: variable.clone(),
                low: self.visit(*low, nodes, numbering),
                high: self.visit(*high, nodes, numbering),
            },
        };
        let canonical = nodes.len();
        nodes.push(node);
        numbering.insert(id, canonical);
        canonical
    }
}

impl<V: Clone + Ord + Hash, T: Clone + Eq + Hash> Decision<V, T> {
    /// Pairs arrive from the least significant bit to the most significant bit.
    /// The variable order must interleave both words by bit significance.
    pub(super) fn equal_bits(
        pairs: impl IntoIterator<Item = (V, V)>,
        accepted: T,
        rejected: T,
    ) -> Self {
        let mut builder = Builder::new();
        let reject = builder.intern(Node::Leaf(rejected));
        let mut tail = builder.intern(Node::Leaf(accepted));
        for (left, right) in pairs {
            let (left, right) = if left <= right {
                (left, right)
            } else {
                (right, left)
            };
            if left == right {
                continue;
            }
            let low = builder.branch(right.clone(), tail, reject);
            let high = builder.branch(right, reject, tail);
            tail = builder.branch(left, low, high);
        }
        builder.finish(tail)
    }

    /// Canonical completion outside a set of feasible valuations. An infeasible
    /// branch takes its sibling's value and therefore contributes no decision.
    pub(super) fn restrict(
        &self,
        care: &Self,
        empty: &T,
        leaf: impl Fn(&T, &T) -> Option<T>,
    ) -> Option<Self> {
        let mut builder = Builder::new();
        let mut context = Restriction {
            source: self,
            care,
            empty,
            leaf: &leaf,
            builder: &mut builder,
            memo: FxHashMap::default(),
        };
        let root = context.visit(self.root(), care.root())?;
        Some(builder.finish(root))
    }
    pub(super) fn leaf(value: T) -> Self {
        Self {
            nodes: vec![Node::Leaf(value)].into(),
        }
    }

    pub(super) fn chain(
        steps: impl IntoIterator<Item = (V, bool)>,
        accepted: T,
        rejected: T,
    ) -> Self {
        let mut builder = Builder::new();
        let mut tail = builder.intern(Node::Leaf(accepted));
        let reject = builder.intern(Node::Leaf(rejected));
        let mut steps: Vec<_> = steps.into_iter().collect();
        steps.sort_unstable_by(|lhs, rhs| rhs.0.cmp(&lhs.0));
        for (variable, expected) in steps {
            tail = if expected {
                builder.select(variable, reject, tail)
            } else {
                builder.select(variable, tail, reject)
            };
        }
        builder.finish(tail)
    }

    pub(super) fn is_leaf(&self, expected: &T) -> bool {
        matches!(&self.nodes[self.root()], Node::Leaf(actual) if actual == expected)
    }

    pub(super) fn leaves(&self) -> impl Iterator<Item = &T> {
        self.nodes.iter().filter_map(|node| match node {
            Node::Leaf(value) => Some(value),
            _ => None,
        })
    }

    /// Borrow decision variables without cloning or sorting their payloads.
    /// A variable may occur at more than one node; consumers needing unique
    /// variables collect only those keys they actually use.
    pub(super) fn variables(&self) -> impl Iterator<Item = &V> {
        self.nodes.iter().filter_map(|node| match node {
            Node::Branch { variable, .. } => Some(variable),
            _ => None,
        })
    }

    pub(super) fn map<W: Clone + Ord + Hash, U: Clone + Eq + Hash>(
        &self,
        mut variable: impl FnMut(&V) -> Variable<W>,
        mut leaf: impl FnMut(&T) -> U,
    ) -> Decision<W, U> {
        // An injective, order-preserving rename keeps every branch ordered.
        // Other substitutions can identify or reorder decisions and need select.
        let mapped_variables: BTreeMap<_, _> = self
            .variables()
            .map(|key| {
                let mapped = variable(key);
                (key, mapped)
            })
            .collect();
        let mut previous = None;
        let ordered = mapped_variables.values().all(|mapped| match mapped {
            Variable::Constant(_) => true,
            Variable::Symbol(key) => {
                let increasing = previous.is_none_or(|old| old < key);
                previous = Some(key);
                increasing
            }
        });
        let mut builder = Builder::new();
        let mut mapped = Vec::with_capacity(self.nodes.len());
        for node in self.nodes.iter() {
            let id = match node {
                Node::Leaf(value) => builder.intern(Node::Leaf(leaf(value))),
                Node::Branch {
                    variable: key,
                    low,
                    high,
                } => match mapped_variables[key].clone() {
                    Variable::Constant(value) => mapped[if value { *high } else { *low }],
                    Variable::Symbol(key) if ordered => {
                        builder.branch(key, mapped[*low], mapped[*high])
                    }
                    Variable::Symbol(key) => builder.select(key, mapped[*low], mapped[*high]),
                },
            };
            mapped.push(id);
        }
        builder.finish(mapped[self.root()])
    }

    /// Existentially quantify selected decisions using an associative,
    /// commutative, idempotent terminal join.
    pub(super) fn exists(
        &self,
        mut selected: impl FnMut(&V) -> bool,
        join: impl Fn(&T, &T) -> T,
    ) -> Self {
        // Ask about each distinct variable once, in decision order.
        let selected: BTreeSet<_> = self
            .variables()
            .collect::<BTreeSet<_>>()
            .into_iter()
            .filter(|variable| selected(variable))
            .collect();
        if selected.is_empty() {
            return self.clone();
        }
        let mut builder = Builder::new();
        let mut mapped = Vec::with_capacity(self.nodes.len());
        let mut memo = FxHashMap::default();
        for node in self.nodes.iter() {
            let result = match node {
                Node::Leaf(value) => builder.intern(Node::Leaf(value.clone())),
                Node::Branch {
                    variable,
                    low,
                    high,
                } if selected.contains(variable) => {
                    builder.apply(mapped[*low], mapped[*high], &join, true, &mut memo)
                }
                Node::Branch {
                    variable,
                    low,
                    high,
                } => builder.branch(variable.clone(), mapped[*low], mapped[*high]),
            };
            mapped.push(result);
        }
        builder.finish(mapped[self.root()])
    }

    pub(super) fn apply(&self, other: &Self, leaf: impl Fn(&T, &T) -> T) -> Self {
        let mut builder = Builder::new();
        let lhs = builder.import(self);
        let rhs = builder.import(other);
        let root = builder.apply(lhs, rhs, &leaf, false, &mut FxHashMap::default());
        builder.finish(root)
    }

    fn cofactors(&self, node: usize, split: &V) -> (usize, usize) {
        match &self.nodes[node] {
            Node::Branch {
                variable,
                low,
                high,
            } if variable == split => (*low, *high),
            _ => (node, node),
        }
    }

    fn root(&self) -> usize {
        self.nodes.len() - 1
    }

    pub(super) fn node_count(&self) -> usize {
        self.nodes.len()
    }

    pub(super) fn witness(&self, mut accepted: impl FnMut(&T) -> bool) -> Option<Vec<(V, bool)>> {
        let mut possible = Vec::with_capacity(self.nodes.len());
        for node in self.nodes.iter() {
            possible.push(match node {
                Node::Leaf(value) => accepted(value),
                Node::Branch { low, high, .. } => possible[*low] || possible[*high],
            });
        }
        if !possible[self.root()] {
            return None;
        }
        let mut path = Vec::new();
        let mut node = self.root();
        while let Node::Branch {
            variable,
            low,
            high,
        } = &self.nodes[node]
        {
            let value = !possible[*low];
            path.push((variable.clone(), value));
            node = if value { *high } else { *low };
        }
        Some(path)
    }
}

impl<V: Clone + Ord + Hash> Decision<V, bool> {
    pub(super) fn less_bits(bits: impl IntoIterator<Item = (Variable<V>, Variable<V>)>) -> Self {
        let mut builder = Builder::new();
        let reject = builder.intern(Node::Leaf(false));
        let accept = builder.intern(Node::Leaf(true));
        let mut tail = reject;
        for (left, right) in bits {
            let (low, high) = match right {
                Variable::Constant(false) => (tail, reject),
                Variable::Constant(true) => (accept, tail),
                Variable::Symbol(right) => (
                    builder.select(right.clone(), tail, accept),
                    builder.select(right, reject, tail),
                ),
            };
            tail = match left {
                Variable::Constant(false) => low,
                Variable::Constant(true) => high,
                Variable::Symbol(left) => builder.select(left, low, high),
            };
        }
        builder.finish(tail)
    }

    pub(super) fn upper_bound_bits(bits: impl IntoIterator<Item = (V, bool)>) -> Self {
        let mut builder = Builder::new();
        let reject = builder.intern(Node::Leaf(false));
        let accept = builder.intern(Node::Leaf(true));
        let mut tail = reject;
        for (variable, bound) in bits {
            tail = if bound {
                builder.branch(variable, accept, tail)
            } else {
                builder.branch(variable, tail, reject)
            };
        }
        builder.finish(tail)
    }
}

struct Restriction<'a, V, T, F> {
    source: &'a Decision<V, T>,
    care: &'a Decision<V, T>,
    empty: &'a T,
    leaf: &'a F,
    builder: &'a mut Builder<V, T>,
    memo: FxHashMap<(usize, usize), Option<usize>>,
}

impl<V: Clone + Ord + Hash, T: Clone + Eq + Hash, F: Fn(&T, &T) -> Option<T>>
    Restriction<'_, V, T, F>
{
    fn visit(&mut self, source: usize, care: usize) -> Option<usize> {
        if let Some(result) = self.memo.get(&(source, care)) {
            return *result;
        }
        if matches!(&self.care.nodes[care], Node::Leaf(value) if value == self.empty) {
            return None;
        }
        let result = match (&self.source.nodes[source], &self.care.nodes[care]) {
            (Node::Leaf(value), Node::Leaf(care)) => {
                (self.leaf)(value, care).map(|value| self.builder.intern(Node::Leaf(value)))
            }
            (value, care_value) => {
                let variable = match (value, care_value) {
                    (
                        Node::Branch { variable: left, .. },
                        Node::Branch {
                            variable: right, ..
                        },
                    ) => left.min(right),
                    (Node::Branch { variable, .. }, _) | (_, Node::Branch { variable, .. }) => {
                        variable
                    }
                    _ => unreachable!(),
                }
                .clone();
                let (source_low, source_high) = self.source.cofactors(source, &variable);
                let (care_low, care_high) = self.care.cofactors(care, &variable);
                match (
                    self.visit(source_low, care_low),
                    self.visit(source_high, care_high),
                ) {
                    (Some(low), Some(high)) => Some(self.builder.branch(variable, low, high)),
                    (left, right) => left.or(right),
                }
            }
        };
        self.memo.insert((source, care), result);
        result
    }
}

#[cfg(test)]
mod tests {
    use std::array;

    use super::*;

    #[test]
    fn ordered_renaming_preserves_correlated_boolean_decisions() {
        let choice = |variable| Decision::chain([(variable, true)], true, false);
        let source = choice(0).apply(
            &choice(1).apply(&choice(2), |left, right| *left || *right),
            |left, right| *left && *right,
        );
        let expected = choice(10).apply(
            &choice(11).apply(&choice(12), |left, right| *left || *right),
            |left, right| *left && *right,
        );
        assert_eq!(
            source.map(|key| Variable::Symbol(*key + 10), |value| *value),
            expected
        );
        assert_eq!(
            source.map(
                |key| {
                    if *key == 1 {
                        Variable::Constant(false)
                    } else {
                        Variable::Symbol(*key + 10)
                    }
                },
                |value| *value
            ),
            choice(10).apply(&choice(12), |left, right| *left && *right)
        );
    }

    #[test]
    fn existential_projection_preserves_correlated_alternatives() {
        let choice = |variable| Decision::chain([(variable, true)], true, false);
        let (x, y, z) = (choice(0), choice(1), choice(2));
        let not_x = x.map(
            |variable| super::Variable::Symbol(*variable),
            |value| !value,
        );
        let left = x.apply(&y, |left, right| *left && *right);
        let right = not_x.apply(&z, |left, right| *left && *right);
        let relation = left.apply(&right, |left, right| *left || *right);
        let union = y.apply(&z, |left, right| *left || *right);

        assert_eq!(
            relation.exists(|variable| *variable == 0, |left, right| *left || *right),
            union
        );
        assert_eq!(
            relation.exists(|variable| *variable != 1, |left, right| *left || *right),
            Decision::leaf(true)
        );
    }

    #[test]
    fn quantification_shares_work_across_selected_variables() {
        let decision = Decision::chain((0..256).map(|variable| (variable, true)), true, false);
        for select_all in [true, false] {
            let before = INTERN_ATTEMPTS.get();
            let quantified = decision.exists(
                |variable| select_all || variable % 2 == 0,
                |left, right| *left || *right,
            );
            let attempts = INTERN_ATTEMPTS.get() - before;
            let expected = Decision::chain(
                (0..256)
                    .filter(|variable| !select_all && variable % 2 != 0)
                    .map(|variable| (variable, true)),
                true,
                false,
            );
            assert_eq!(quantified, expected);
            assert!(
                attempts <= decision.node_count() * 8,
                "{attempts} interning attempts for {} source nodes",
                decision.node_count()
            );
        }
    }

    #[test]
    fn quantification_matches_exhaustive_terminal_unions() {
        // Every Boolean function of three variables, plus distinct singleton
        // sets at all eight terminals to exercise non-Boolean joins.
        let tables = (0u16..256)
            .map(|bits| array::from_fn::<_, 8, _>(|assignment| ((bits >> assignment) & 1) as u8))
            .chain([array::from_fn(|assignment| 1u8 << assignment)]);
        for table in tables {
            let mut builder = Builder::new();
            let mut level: Vec<_> = table
                .iter()
                .map(|value| builder.intern(Node::Leaf(*value)))
                .collect();
            for variable in (0u8..3).rev() {
                level = level
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|children| builder.branch(variable, children[0], children[1]))
                    .collect();
            }
            let decision = builder.finish(level[0]);
            for selected in 0u8..8 {
                let quantified = decision.exists(
                    |variable| selected & (1 << (2 - variable)) != 0,
                    |left, right| left | right,
                );
                for assignment in 0u8..8 {
                    let expected = table
                        .iter()
                        .enumerate()
                        .filter(|(other, _)| *other as u8 & !selected == assignment & !selected)
                        .fold(0, |union, (_, value)| union | value);
                    let actual = quantified.map(
                        |variable| {
                            Variable::<u8>::Constant(assignment & (1 << (2 - variable)) != 0)
                        },
                        Clone::clone,
                    );
                    assert!(
                        actual.is_leaf(&expected),
                        "table={table:?}, selected={selected}, assignment={assignment}"
                    );
                }
            }
        }
    }

    #[test]
    fn quantification_visits_shared_variables_once_in_order() {
        let first = Decision::chain([(0u8, true)], true, false);
        let second = Decision::chain([(1u8, true)], true, false);
        let parity = first.apply(&second, |left, right| left ^ right);
        assert_eq!(parity.variables().count(), 3);
        let mut observed = Vec::new();
        let quantified = parity.exists(
            |variable| {
                observed.push(*variable);
                *variable == 1
            },
            |left, right| *left || *right,
        );
        assert_eq!(observed, [0, 1]);
        assert!(quantified.is_leaf(&true));
    }
}
