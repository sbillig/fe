pub trait JoinSemiLattice {
    fn join_into(&mut self, other: &Self) -> bool;
}
