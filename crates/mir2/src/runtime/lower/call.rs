use crate::{
    db::MirDb,
    instance::RuntimeInstance,
    runtime::{RExpr, RStmt, RTerminator, RuntimeCallEdge},
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
            let _ = args;
            calls.push(RuntimeCallEdge { callee: *callee });
        }
        if let RTerminator::TerminalCall { callee, args } = &block.terminator {
            let _ = args;
            calls.push(RuntimeCallEdge { callee: *callee });
        }
    }
    calls
}
