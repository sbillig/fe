use crate::{JoinSemiLattice, queue::WorkQueue};

pub trait ForwardCfgAnalysis {
    type State: Clone + JoinSemiLattice;

    fn block_count(&self) -> usize;
    fn bottom(&self) -> Self::State;
    fn initialize(&mut self, entry_states: &mut [Self::State]);
    fn transfer(&mut self, block: usize, in_state: &Self::State) -> Self::State;
    fn successors(&self, block: usize) -> &[usize];
}

pub trait BackwardCfgAnalysis {
    type State: Clone + JoinSemiLattice;

    fn block_count(&self) -> usize;
    fn bottom(&self) -> Self::State;
    fn initialize(&mut self, exit_states: &mut [Self::State]);
    fn transfer(&mut self, block: usize, out_state: &Self::State) -> Self::State;
    fn predecessors(&self, block: usize) -> &[usize];
}

pub fn solve_forward_cfg<A: ForwardCfgAnalysis>(analysis: &mut A) -> Vec<A::State> {
    let mut entry_states = vec![analysis.bottom(); analysis.block_count()];
    analysis.initialize(&mut entry_states);

    let mut queue = WorkQueue::with_all(entry_states.len());
    while let Some(block) = queue.pop() {
        let out_state = analysis.transfer(block, &entry_states[block]);
        for &succ in analysis.successors(block) {
            if entry_states[succ].join_into(&out_state) {
                queue.push(succ);
            }
        }
    }

    entry_states
}

pub fn solve_backward_cfg<A: BackwardCfgAnalysis>(analysis: &mut A) -> Vec<A::State> {
    let mut exit_states = vec![analysis.bottom(); analysis.block_count()];
    analysis.initialize(&mut exit_states);

    let mut queue = WorkQueue::with_all(exit_states.len());
    while let Some(block) = queue.pop() {
        let in_state = analysis.transfer(block, &exit_states[block]);
        for &pred in analysis.predecessors(block) {
            if exit_states[pred].join_into(&in_state) {
                queue.push(pred);
            }
        }
    }

    exit_states
}

#[cfg(test)]
mod tests {
    use super::{BackwardCfgAnalysis, ForwardCfgAnalysis, solve_backward_cfg, solve_forward_cfg};
    use crate::JoinSemiLattice;

    const SUCCESSORS: [&[usize]; 4] = [&[1, 2], &[3], &[3], &[]];
    const PREDECESSORS: [&[usize]; 4] = [&[], &[0], &[0], &[1, 2]];

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    struct Bits(u8);

    impl JoinSemiLattice for Bits {
        fn join_into(&mut self, other: &Self) -> bool {
            let joined = self.0 | other.0;
            let changed = joined != self.0;
            self.0 = joined;
            changed
        }
    }

    struct ForwardBitsAnalysis;

    impl ForwardCfgAnalysis for ForwardBitsAnalysis {
        type State = Bits;

        fn block_count(&self) -> usize {
            SUCCESSORS.len()
        }

        fn bottom(&self) -> Self::State {
            Bits(0)
        }

        fn initialize(&mut self, entry_states: &mut [Self::State]) {
            entry_states[0] = Bits(0b0001);
        }

        fn transfer(&mut self, block: usize, in_state: &Self::State) -> Self::State {
            Bits(in_state.0 | (1 << block))
        }

        fn successors(&self, block: usize) -> &[usize] {
            SUCCESSORS[block]
        }
    }

    struct BackwardBitsAnalysis;

    impl BackwardCfgAnalysis for BackwardBitsAnalysis {
        type State = Bits;

        fn block_count(&self) -> usize {
            PREDECESSORS.len()
        }

        fn bottom(&self) -> Self::State {
            Bits(0)
        }

        fn initialize(&mut self, exit_states: &mut [Self::State]) {
            exit_states[3] = Bits(0b1000);
        }

        fn transfer(&mut self, block: usize, out_state: &Self::State) -> Self::State {
            Bits(out_state.0 | (1 << block))
        }

        fn predecessors(&self, block: usize) -> &[usize] {
            PREDECESSORS[block]
        }
    }

    #[test]
    fn forward_cfg_solver_propagates_through_diamond() {
        let states = solve_forward_cfg(&mut ForwardBitsAnalysis);

        assert_eq!(
            states,
            vec![Bits(0b0001), Bits(0b0001), Bits(0b0001), Bits(0b0111)]
        );
    }

    #[test]
    fn backward_cfg_solver_propagates_through_diamond() {
        let states = solve_backward_cfg(&mut BackwardBitsAnalysis);

        assert_eq!(
            states,
            vec![Bits(0b1110), Bits(0b1000), Bits(0b1000), Bits(0b1000)]
        );
    }
}
