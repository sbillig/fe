use std::collections::VecDeque;

pub(crate) struct WorkQueue {
    queued: Vec<bool>,
    pending: VecDeque<usize>,
}

impl WorkQueue {
    pub(crate) fn with_all(count: usize) -> Self {
        let mut pending = VecDeque::with_capacity(count);
        pending.extend(0..count);

        Self {
            queued: vec![true; count],
            pending,
        }
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
        let mut queue = WorkQueue::with_all(3);

        assert_eq!(queue.pop(), Some(0));
        queue.push(1);
        queue.push(1);
        queue.push(1);

        assert_eq!(queue.pop(), Some(1));
        assert_eq!(queue.pop(), Some(2));
        assert_eq!(queue.pop(), None);
    }
}
