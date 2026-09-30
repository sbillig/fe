//! Optional accounting for retained graphs, construction churn, and solver progress.
use rustc_hash::FxHashMap;
use std::{cell::RefCell, time::Instant};

type Fingerprint = (u64, usize, usize, &'static str);

struct Scope {
    label: String,
    started: Instant,
    created: usize,
    sweep: usize,
    block: usize,
    statement: usize,
    phase: &'static str,
}

struct Profile {
    started: Instant,
    reported: Instant,
    live: [usize; 2],
    created: [usize; 2],
    completed: usize,
    large: FxHashMap<Fingerprint, usize>,
    scopes: Vec<Scope>,
    operation: &'static str,
    operation_counts: [usize; 4],
    input_nodes: [usize; 4],
}

impl Default for Profile {
    fn default() -> Self {
        Self {
            started: Instant::now(),
            reported: Instant::now(),
            live: [0; 2],
            created: [0; 2],
            completed: 0,
            large: FxHashMap::default(),
            scopes: Vec::new(),
            operation: "none",
            operation_counts: [0; 4],
            input_nodes: [0; 4],
        }
    }
}

thread_local! {
    static PROFILE: RefCell<Profile> = RefCell::default();
}

impl Profile {
    fn report(&mut self, event: &str) {
        let duplicate_bytes: usize = self
            .large
            .iter()
            .map(|((_, nodes, size, _), copies)| nodes * size * (copies - 1))
            .sum();
        let unique_bytes: usize = self
            .large
            .keys()
            .map(|(_, nodes, size, _)| nodes * size)
            .sum();
        let largest = self
            .large
            .keys()
            .map(|(_, nodes, _, _)| *nodes)
            .max()
            .unwrap_or(0);
        let scope = self.scopes.last();
        eprintln!(
            "GRAPH_PROFILE event={event} seconds={:.3} scope={:?} phase={:?} sweep={} block={} statement={} operation={} choice_live_bytes={} bit_live_bytes={} choice_created_bytes={} bit_created_bytes={} completed={} large_unique={} largest_nodes={largest} duplicate_bytes={duplicate_bytes} unique_bytes={unique_bytes} operation_counts={:?} input_nodes={:?}",
            self.started.elapsed().as_secs_f64(),
            scope.map(|s| &s.label),
            scope.map(|s| s.phase),
            scope.map_or(0, |s| s.sweep),
            scope.map_or(0, |s| s.block),
            scope.map_or(0, |s| s.statement),
            self.operation,
            self.live[1],
            self.live[0],
            self.created[1],
            self.created[0],
            self.completed,
            self.large.len(),
            self.operation_counts,
            self.input_nodes,
        );
        self.reported = Instant::now();
    }
}

pub(crate) fn created(key: Fingerprint) {
    PROFILE.with_borrow_mut(|p| {
        let (_, nodes, size, name) = key;
        let category = usize::from(name.contains("SlotChoice"));
        p.live[category] += nodes * size;
        p.created[category] += nodes * size;
        p.completed += 1;
        if nodes >= 1024 {
            *p.large.entry(key).or_default() += 1;
        }
        if p.completed.is_multiple_of(1024) && p.reported.elapsed().as_secs_f64() >= 1.0 {
            p.report("tick");
        }
    });
}

pub(crate) fn dropped(key: Fingerprint) {
    let _ = PROFILE.try_with(|p| {
        let mut p = p.borrow_mut();
        let (_, nodes, size, name) = key;
        p.live[usize::from(name.contains("SlotChoice"))] -= nodes * size;
        if nodes >= 1024 {
            let count = p.large.get_mut(&key).expect("counted graph");
            *count -= 1;
            if *count == 0 {
                p.large.remove(&key);
            }
        }
    });
}

pub(crate) struct ProfileScope;

impl ProfileScope {
    pub(crate) fn new(label: String) -> Self {
        PROFILE.with_borrow_mut(|p| {
            p.scopes.push(Scope {
                label,
                started: Instant::now(),
                created: p.created.iter().sum(),
                sweep: 0,
                block: 0,
                statement: 0,
                phase: "prepare",
            });
        });
        Self
    }

    pub(crate) fn point(&self, phase: &'static str, block: usize, statement: usize) {
        PROFILE.with_borrow_mut(|p| {
            let scope = p.scopes.last_mut().expect("active scope");
            scope.phase = phase;
            scope.block = block;
            scope.statement = statement;
        });
    }

    pub(crate) fn sweep(&self, loans: usize, generation: usize) {
        PROFILE.with_borrow_mut(|p| {
            let scope = p.scopes.last_mut().expect("active scope");
            scope.sweep += 1;
            scope.phase = "sweep";
            if scope.started.elapsed().as_secs_f64() >= 0.5 {
                eprintln!(
                    "SOLVER_SWEEP scope={:?} sweep={} loans={loans} generation={generation}",
                    scope.label, scope.sweep
                );
                p.report("sweep");
            }
        });
    }
}

impl Drop for ProfileScope {
    fn drop(&mut self) {
        let _ = PROFILE.try_with(|p| {
            let mut p = p.borrow_mut();
            let scope = p.scopes.last().expect("active scope");
            if scope.started.elapsed().as_secs_f64() >= 0.1
                || p.created.iter().sum::<usize>() - scope.created >= 32 * 1024 * 1024
            {
                eprintln!(
                    "SOLVER_SCOPE seconds={:.3} scope={:?}",
                    scope.started.elapsed().as_secs_f64(),
                    scope.label
                );
                p.report("scope_exit");
            }
            p.scopes.pop();
        });
    }
}

pub(crate) struct Operation(&'static str);

impl Operation {
    pub(crate) fn new(kind: usize, nodes: usize) -> Self {
        PROFILE.with_borrow_mut(|p| {
            p.operation_counts[kind] += 1;
            p.input_nodes[kind] += nodes;
            Self(std::mem::replace(
                &mut p.operation,
                ["map", "apply", "exists", "restrict"][kind],
            ))
        })
    }
}

impl Drop for Operation {
    fn drop(&mut self) {
        let _ = PROFILE.try_with(|p| p.borrow_mut().operation = self.0);
    }
}
