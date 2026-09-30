//! Optional accounting for retained graphs, construction churn, and solver progress.
//! Enable output with `FE_BORROWCK_PROFILE=1` in a `borrowck-profile` build.
use super::{
    decision::{GraphFingerprint, LiveGraphs, live_graphs},
    guard,
};
use std::{
    cell::RefCell,
    sync::{Arc, Mutex, OnceLock},
    time::Instant,
};

pub(super) fn enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("FE_BORROWCK_PROFILE").is_some())
}

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
    live: Arc<Mutex<LiveGraphs>>,
    created: [usize; 2],
    completed: usize,
    peak_graph_and_scratch_capacity: usize,
    next_overlap_bytes: usize,
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
            live: live_graphs(),
            created: [0; 2],
            completed: 0,
            peak_graph_and_scratch_capacity: 0,
            next_overlap_bytes: if std::env::var_os("FE_BORROWCK_SUBGRAPHS").is_some() {
                128 * 1024 * 1024
            } else {
                usize::MAX
            },
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
        let live = self.live.lock().expect("graph accounting");
        let large = live
            .copies
            .iter()
            .filter(|((_, nodes, _, _), _)| *nodes >= 1024);
        let duplicate_bytes: usize = large
            .clone()
            .map(|((_, nodes, size, _), copies)| nodes * size * (copies - 1))
            .sum();
        let unique_bytes: usize = large
            .clone()
            .map(|((_, nodes, size, _), _)| nodes * size)
            .sum();
        let largest = large
            .clone()
            .map(|((_, nodes, _, _), _)| *nodes)
            .max()
            .unwrap_or(0);
        let scope = self.scopes.last();
        eprintln!(
            "GRAPH_PROFILE event={event} seconds={:.3} scope={:?} phase={:?} sweep={} block={} statement={} operation={} choice_live_bytes={} bit_live_bytes={} choice_created_bytes={} bit_created_bytes={} completed={} large_unique={} largest_nodes={largest} duplicate_bytes={duplicate_bytes} unique_bytes={unique_bytes} operation_counts={:?} input_nodes={:?} peak_graph_and_scratch_capacity={}",
            self.started.elapsed().as_secs_f64(),
            scope.map(|s| &s.label),
            scope.map(|s| s.phase),
            scope.map_or(0, |s| s.sweep),
            scope.map_or(0, |s| s.block),
            scope.map_or(0, |s| s.statement),
            self.operation,
            live.bytes[1],
            live.bytes[0],
            self.created[1],
            self.created[0],
            self.completed,
            large.count(),
            self.operation_counts,
            self.input_nodes,
            self.peak_graph_and_scratch_capacity,
        );
        self.reported = Instant::now();
    }
}

pub(crate) fn scratch(bytes: usize) {
    if !enabled() {
        return;
    }
    PROFILE.with_borrow_mut(|p| {
        p.peak_graph_and_scratch_capacity = p.peak_graph_and_scratch_capacity.max(
            p.live
                .lock()
                .expect("graph accounting")
                .bytes
                .iter()
                .sum::<usize>()
                + bytes,
        );
    });
}

pub(crate) fn created(key: GraphFingerprint) {
    if !enabled() {
        return;
    }
    PROFILE.with_borrow_mut(|p| {
        let (_, nodes, size, name) = key;
        let category = usize::from(name.contains("SlotChoice"));
        p.created[category] += nodes * size;
        p.completed += 1;
        if p.completed.is_multiple_of(1024) && p.reported.elapsed().as_secs_f64() >= 1.0 {
            p.report("tick");
        }
    });
}

pub(crate) struct ProfileScope;

impl ProfileScope {
    pub(crate) fn new(label: String) -> Self {
        if !enabled() {
            return Self;
        }
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
        if !enabled() {
            return;
        }
        PROFILE.with_borrow_mut(|p| {
            let scope = p.scopes.last_mut().expect("active scope");
            scope.phase = phase;
            scope.block = block;
            scope.statement = statement;
        });
    }

    pub(crate) fn sweep(&self, loans: usize, generation: usize) {
        if !enabled() {
            return;
        }
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
        if !enabled() {
            return;
        }
        let overlap = PROFILE
            .try_with(|p| {
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
                let overlap = p
                    .live
                    .lock()
                    .expect("graph accounting")
                    .bytes
                    .iter()
                    .sum::<usize>()
                    >= p.next_overlap_bytes;
                if overlap {
                    p.report("subgraph_snapshot");
                    p.next_overlap_bytes = p.next_overlap_bytes.saturating_mul(2);
                }
                p.scopes.pop();
                overlap
            })
            .unwrap_or(false);
        if overlap {
            guard::profile_subgraphs();
        }
    }
}

pub(crate) struct Operation(&'static str);

impl Operation {
    pub(crate) fn new(kind: usize, nodes: usize) -> Self {
        if !enabled() {
            return Self("none");
        }
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
        if !enabled() {
            return;
        }
        let _ = PROFILE.try_with(|p| p.borrow_mut().operation = self.0);
    }
}
