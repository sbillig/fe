use std::collections::VecDeque;

use cranelift_entity::{EntityRef, SecondaryMap};

pub(crate) struct WorkQueue<N: EntityRef> {
    queued: SecondaryMap<N, bool>,
    pending: VecDeque<N>,
}

impl<N: EntityRef> WorkQueue<N> {
    pub(crate) fn with_seed(count: usize, seed: impl IntoIterator<Item = N>) -> Self {
        let mut queued = SecondaryMap::new();
        queued.resize(count);
        let mut queue = Self {
            queued,
            pending: VecDeque::new(),
        };
        for node in seed {
            queue.push(node);
        }
        queue
    }

    pub(crate) fn push(&mut self, node: N) {
        if !self.queued[node] {
            self.queued[node] = true;
            self.pending.push_back(node);
        }
    }

    pub(crate) fn pop(&mut self) -> Option<N> {
        let node = self.pending.pop_front()?;
        self.queued[node] = false;
        Some(node)
    }
}

#[cfg(test)]
mod tests {
    use cranelift_entity::{EntityRef, entity_impl};

    use super::WorkQueue;

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    struct Node(u32);
    entity_impl!(Node);

    #[test]
    fn deduplicates_nodes() {
        let mut queue = WorkQueue::with_seed(3, [Node::new(0), Node::new(1), Node::new(2)]);

        assert_eq!(queue.pop(), Some(Node::new(0)));
        queue.push(Node::new(1));
        queue.push(Node::new(1));
        queue.push(Node::new(1));

        assert_eq!(queue.pop(), Some(Node::new(1)));
        assert_eq!(queue.pop(), Some(Node::new(2)));
        assert_eq!(queue.pop(), None);
    }
}
