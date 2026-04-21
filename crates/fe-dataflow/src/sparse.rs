use std::convert::Infallible;

use cranelift_entity::EntityRef;

use crate::queue::WorkQueue;

pub trait SparseAnalysis {
    type Node: EntityRef;
    type State;
    type Error;

    fn node_count(&self) -> usize;
    fn seed_nodes(&self) -> Vec<Self::Node>;
    fn step(&mut self, node: Self::Node, state: &mut Self::State) -> Result<bool, Self::Error>;
    fn dependents(&self, node: Self::Node, out: &mut Vec<Self::Node>);
}

pub fn try_solve_sparse<A: SparseAnalysis>(
    analysis: &mut A,
    state: &mut A::State,
) -> Result<(), A::Error> {
    let mut queue = WorkQueue::with_seed(analysis.node_count(), analysis.seed_nodes());
    let mut dependents = Vec::new();
    while let Some(node) = queue.pop() {
        if analysis.step(node, state)? {
            dependents.clear();
            analysis.dependents(node, &mut dependents);
            for dependent in dependents.drain(..) {
                queue.push(dependent);
            }
        }
    }
    Ok(())
}

pub fn solve_sparse<A: SparseAnalysis<Error = Infallible>>(analysis: &mut A, state: &mut A::State) {
    match try_solve_sparse(analysis, state) {
        Ok(()) => {}
        Err(err) => match err {},
    }
}

#[cfg(test)]
mod tests {
    use std::convert::Infallible;

    use cranelift_entity::{EntityRef, entity_impl};

    use super::{SparseAnalysis, solve_sparse, try_solve_sparse};

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    struct Node(u32);
    entity_impl!(Node);

    struct ChainAnalysis {
        runs: Vec<usize>,
    }

    impl SparseAnalysis for ChainAnalysis {
        type Node = Node;
        type State = Vec<bool>;
        type Error = Infallible;

        fn node_count(&self) -> usize {
            3
        }

        fn seed_nodes(&self) -> Vec<Self::Node> {
            vec![Node::new(0)]
        }

        fn step(&mut self, node: Self::Node, state: &mut Self::State) -> Result<bool, Self::Error> {
            self.runs[node.index()] += 1;
            let changed = !state[node.index()];
            state[node.index()] = true;
            Ok(changed)
        }

        fn dependents(&self, node: Self::Node, out: &mut Vec<Self::Node>) {
            if node.index() + 1 < self.node_count() {
                out.push(Node::new(node.index() + 1));
            }
        }
    }

    #[test]
    fn sparse_solver_propagates_to_dependents() {
        let mut analysis = ChainAnalysis { runs: vec![0; 3] };
        let mut state = vec![false; 3];

        solve_sparse(&mut analysis, &mut state);

        assert_eq!(state, vec![true, true, true]);
        assert_eq!(analysis.runs, vec![1, 1, 1]);
    }

    struct StableAnalysis {
        runs: Vec<usize>,
    }

    impl SparseAnalysis for StableAnalysis {
        type Node = Node;
        type State = ();
        type Error = Infallible;

        fn node_count(&self) -> usize {
            2
        }

        fn seed_nodes(&self) -> Vec<Self::Node> {
            vec![Node::new(0)]
        }

        fn step(&mut self, node: Self::Node, _: &mut Self::State) -> Result<bool, Self::Error> {
            self.runs[node.index()] += 1;
            Ok(false)
        }

        fn dependents(&self, _node: Self::Node, out: &mut Vec<Self::Node>) {
            out.push(Node::new(1));
        }
    }

    #[test]
    fn sparse_solver_does_not_reenqueue_when_unchanged() {
        let mut analysis = StableAnalysis { runs: vec![0; 2] };
        let mut state = ();

        solve_sparse(&mut analysis, &mut state);

        assert_eq!(analysis.runs, vec![1, 0]);
    }

    #[derive(Debug, PartialEq, Eq)]
    struct TestError;

    struct FallibleAnalysis;

    impl SparseAnalysis for FallibleAnalysis {
        type Node = Node;
        type State = ();
        type Error = TestError;

        fn node_count(&self) -> usize {
            1
        }

        fn seed_nodes(&self) -> Vec<Self::Node> {
            vec![Node::new(0)]
        }

        fn step(&mut self, _node: Self::Node, _: &mut Self::State) -> Result<bool, Self::Error> {
            Err(TestError)
        }

        fn dependents(&self, _node: Self::Node, _out: &mut Vec<Self::Node>) {}
    }

    #[test]
    fn sparse_solver_propagates_errors() {
        let mut analysis = FallibleAnalysis;
        let mut state = ();

        assert_eq!(try_solve_sparse(&mut analysis, &mut state), Err(TestError));
    }
}
