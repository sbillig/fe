use hir::analysis::semantic::{SemanticInstance, get_or_build_semantic_instance};

use crate::{
    db::MirDb,
    runtime::{
        RuntimeBody, RuntimeCallEdge, RuntimeClass,
        lower::{collect_runtime_calls as collect_runtime_calls_lowered, lower_to_rmir},
    },
    verify::verify_runtime_body,
};

#[salsa::interned]
#[derive(Debug)]
pub struct RuntimeInstanceKey<'db> {
    pub semantic: SemanticInstance<'db>,
    #[return_ref]
    pub params: Vec<RuntimeClass<'db>>,
}

#[salsa::tracked]
#[derive(Debug)]
pub struct RuntimeInstance<'db> {
    pub key: RuntimeInstanceKey<'db>,
}

#[salsa::tracked]
impl<'db> RuntimeInstance<'db> {
    #[salsa::tracked]
    pub fn body(self, db: &'db dyn MirDb) -> RuntimeBody<'db> {
        lower_runtime_body(db, self)
    }

    #[salsa::tracked(return_ref)]
    pub fn calls(self, db: &'db dyn MirDb) -> Vec<RuntimeCallEdge<'db>> {
        collect_runtime_calls(db, self)
    }
}

pub fn get_or_build_runtime_instance<'db>(
    db: &'db dyn MirDb,
    key: RuntimeInstanceKey<'db>,
) -> RuntimeInstance<'db> {
    let instance = RuntimeInstance::new(db, key);
    for call in instance.calls(db) {
        let callee_key = RuntimeInstanceKey::new(
            db,
            get_or_build_semantic_instance(db, call.semantic_callee.key),
            call.runtime_arg_classes.clone(),
        );
        get_or_build_runtime_instance(db, callee_key);
    }
    instance
}

fn lower_runtime_body<'db>(db: &'db dyn MirDb, instance: RuntimeInstance<'db>) -> RuntimeBody<'db> {
    let body = lower_to_rmir(db, instance);
    verify_runtime_body(db, &db, &body).expect("invalid runtime MIR");
    body
}

fn collect_runtime_calls<'db>(
    db: &'db dyn MirDb,
    instance: RuntimeInstance<'db>,
) -> Vec<RuntimeCallEdge<'db>> {
    collect_runtime_calls_lowered(db, instance)
}
