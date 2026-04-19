use std::collections::VecDeque;

pub(crate) struct WorkQueue {
    queued: Vec<bool>,
    pending: VecDeque<usize>,
}

impl WorkQueue {
    pub(crate) fn with_seed(count: usize, seed: impl IntoIterator<Item = usize>) -> Self {
        let mut queue = Self {
            queued: vec![false; count],
            pending: VecDeque::new(),
        };
        for node in seed {
            queue.push(node);
        }
        queue
    }

    pub(crate) fn push(&mut self, node: usize) {
        if !self.queued[node] {
            self.queued[node] = true;
            self.pending.push_back(node);
        }
    }

    pub(crate) fn pop(&mut self) -> Option<usize> {
        let node = self.pending.pop_front()?;
        self.queued[node] = false;
        Some(node)
    }
}

#[cfg(test)]
mod tests {
    use super::WorkQueue;

    #[test]
    fn deduplicates_nodes() {
        let mut queue = WorkQueue::with_seed(3, [0, 1, 2]);

        assert_eq!(queue.pop(), Some(0));
        queue.push(1);
        queue.push(1);
        queue.push(1);

        assert_eq!(queue.pop(), Some(1));
        assert_eq!(queue.pop(), Some(2));
        assert_eq!(queue.pop(), None);
    }
}
