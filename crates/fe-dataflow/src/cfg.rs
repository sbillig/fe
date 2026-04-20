use std::convert::Infallible;

use crate::{JoinSemiLattice, queue::WorkQueue};

pub trait ForwardCfgAnalysis {
    type State: Clone + JoinSemiLattice;
    type Error;

    fn block_count(&self) -> usize;
    fn seed_blocks(&self) -> Vec<usize>;
    fn bottom(&self) -> Self::State;
    fn initialize(&mut self, _entry_states: &mut [Self::State]) -> Result<(), Self::Error> {
        Ok(())
    }

    fn transfer(
        &mut self,
        block: usize,
        in_state: &Self::State,
    ) -> Result<Self::State, Self::Error>;
    fn successors(&self, block: usize) -> &[usize];
}

pub trait BackwardCfgAnalysis {
    type State: Clone + JoinSemiLattice;

    fn block_count(&self) -> usize;
    fn seed_blocks(&self) -> Vec<usize>;
    fn bottom(&self) -> Self::State;
    fn initialize(&mut self, exit_states: &mut [Self::State]);
    fn transfer(&mut self, block: usize, out_state: &Self::State) -> Self::State;
    fn predecessors(&self, block: usize) -> &[usize];
}

pub fn solve_forward_cfg<A: ForwardCfgAnalysis<Error = Infallible>>(
    analysis: &mut A,
) -> Vec<A::State> {
    match try_solve_forward_cfg(analysis) {
        Ok(states) => states,
        Err(err) => match err {},
    }
}

pub fn solve_backward_cfg<A: BackwardCfgAnalysis>(analysis: &mut A) -> Vec<A::State> {
    let mut exit_states = vec![analysis.bottom(); analysis.block_count()];
    analysis.initialize(&mut exit_states);

    let seed_blocks = analysis.seed_blocks();
    let mut reached = vec![false; exit_states.len()];
    for &block in &seed_blocks {
        reached[block] = true;
    }
    let mut queue = WorkQueue::with_seed(exit_states.len(), seed_blocks);
    while let Some(block) = queue.pop() {
        let in_state = analysis.transfer(block, &exit_states[block]);
        for &pred in analysis.predecessors(block) {
            let changed = exit_states[pred].join_into(&in_state);
            let newly_reached = !reached[pred];
            reached[pred] = true;
            if newly_reached || changed {
                queue.push(pred);
            }
        }
    }

    exit_states
}

pub fn try_solve_forward_cfg<A: ForwardCfgAnalysis>(
    analysis: &mut A,
) -> Result<Vec<A::State>, A::Error> {
    let mut entry_states = vec![analysis.bottom(); analysis.block_count()];
    analysis.initialize(&mut entry_states)?;

    let seed_blocks = analysis.seed_blocks();
    let mut reached = vec![false; entry_states.len()];
    for &block in &seed_blocks {
        reached[block] = true;
    }
    let mut queue = WorkQueue::with_seed(entry_states.len(), seed_blocks);
    while let Some(block) = queue.pop() {
        let out_state = analysis.transfer(block, &entry_states[block])?;
        for &succ in analysis.successors(block) {
            let changed = entry_states[succ].join_into(&out_state);
            let newly_reached = !reached[succ];
            reached[succ] = true;
            if newly_reached || changed {
                queue.push(succ);
            }
        }
    }

    Ok(entry_states)
}

#[cfg(test)]
mod tests {
    use std::convert::Infallible;

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
        type Error = Infallible;

        fn block_count(&self) -> usize {
            SUCCESSORS.len()
        }

        fn seed_blocks(&self) -> Vec<usize> {
            vec![0]
        }

        fn bottom(&self) -> Self::State {
            Bits(0)
        }

        fn initialize(&mut self, entry_states: &mut [Self::State]) -> Result<(), Self::Error> {
            entry_states[0] = Bits(0b0001);
            Ok(())
        }

        fn transfer(
            &mut self,
            block: usize,
            in_state: &Self::State,
        ) -> Result<Self::State, Self::Error> {
            Ok(Bits(in_state.0 | (1 << block)))
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

        fn seed_blocks(&self) -> Vec<usize> {
            vec![3]
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

    struct ReachabilityAnalysis;

    impl ForwardCfgAnalysis for ReachabilityAnalysis {
        type State = Bits;
        type Error = Infallible;

        fn block_count(&self) -> usize {
            3
        }

        fn seed_blocks(&self) -> Vec<usize> {
            vec![0]
        }

        fn bottom(&self) -> Self::State {
            Bits(0)
        }

        fn initialize(&mut self, entry_states: &mut [Self::State]) -> Result<(), Self::Error> {
            entry_states[0] = Bits(1);
            Ok(())
        }

        fn transfer(
            &mut self,
            block: usize,
            in_state: &Self::State,
        ) -> Result<Self::State, Self::Error> {
            Ok(Bits(in_state.0 | (1 << block)))
        }

        fn successors(&self, block: usize) -> &[usize] {
            match block {
                0 => &[1],
                1 | 2 => &[],
                _ => unreachable!(),
            }
        }
    }

    #[test]
    fn forward_cfg_solver_does_not_process_unreachable_blocks() {
        let states = solve_forward_cfg(&mut ReachabilityAnalysis);

        assert_eq!(states, vec![Bits(1), Bits(1), Bits(0)]);
    }

    struct ReachBottomThenGenerateAnalysis;

    impl ForwardCfgAnalysis for ReachBottomThenGenerateAnalysis {
        type State = Bits;
        type Error = Infallible;

        fn block_count(&self) -> usize {
            3
        }

        fn seed_blocks(&self) -> Vec<usize> {
            vec![0]
        }

        fn bottom(&self) -> Self::State {
            Bits(0)
        }

        fn transfer(
            &mut self,
            block: usize,
            in_state: &Self::State,
        ) -> Result<Self::State, Self::Error> {
            Ok(match block {
                0 => *in_state,
                1 => Bits(0b10),
                2 => Bits(in_state.0 | 0b100),
                _ => unreachable!(),
            })
        }

        fn successors(&self, block: usize) -> &[usize] {
            match block {
                0 => &[1],
                1 => &[2],
                2 => &[],
                _ => unreachable!(),
            }
        }
    }

    #[test]
    fn forward_cfg_solver_processes_newly_reached_bottom_state_blocks() {
        let states = solve_forward_cfg(&mut ReachBottomThenGenerateAnalysis);

        assert_eq!(states, vec![Bits(0), Bits(0), Bits(0b10)]);
    }
}
