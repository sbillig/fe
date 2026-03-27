use hir::analysis::semantic::SemanticCalleeRef;

use crate::{
    db::MirDb,
    instance::RuntimeInstance,
    runtime::{RExpr, RStmt, RuntimeCallEdge},
};

pub fn collect_runtime_calls<'db>(
    db: &'db dyn MirDb,
    instance: RuntimeInstance<'db>,
) -> Vec<RuntimeCallEdge<'db>> {
    let body = instance.body(db);
    let mut calls = Vec::new();
    for block in &body.blocks {
        for stmt in &block.stmts {
            let RStmt::Assign {
                expr: RExpr::Call { callee, args },
                ..
            } = stmt
            else {
                continue;
            };
            let runtime_arg_classes = args
                .iter()
                .map(|arg| {
                    body.value_class(*arg)
                        .cloned()
                        .expect("call arguments should not be erased")
                })
                .collect();
            calls.push(RuntimeCallEdge {
                semantic_callee: SemanticCalleeRef {
                    key: callee.key(db).semantic(db).key(db),
                },
                runtime_arg_classes,
            });
        }
    }
    calls
}
