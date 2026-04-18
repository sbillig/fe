use hir::analysis::semantic::{SemanticInstance, check_semantic_borrows};
use salsa::Update;

use crate::{
    db::MirDb,
    runtime::{
        LoweredRuntimeBody, RuntimeBody, RuntimeCallEdge, RuntimeClass, RuntimeSignature,
        RuntimeSyntheticSpec,
        lower::{
            body::lower_to_rmir,
            call::{
                collect_referenced_code_regions, collect_referenced_const_regions,
                collect_runtime_calls as collect_runtime_calls_lowered,
            },
            classify::runtime_signature_for_key,
        },
        synthetic::{lower_synthetic_runtime_body, runtime_synthetic_signature},
    },
};

#[salsa::interned]
#[derive(Debug)]
pub struct RuntimeSyntheticInstance<'db> {
    pub spec: RuntimeSyntheticSpec<'db>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Update)]
pub enum RuntimeInstanceSource<'db> {
    Semantic(SemanticInstance<'db>),
    Synthetic(RuntimeSyntheticInstance<'db>),
}

#[salsa::interned]
#[derive(Debug)]
pub struct RuntimeInstanceKey<'db> {
    pub source: RuntimeInstanceSource<'db>,
    #[return_ref]
    pub params: Vec<RuntimeClass<'db>>,
}

impl<'db> RuntimeInstanceKey<'db> {
    pub fn semantic(self, db: &'db dyn MirDb) -> Option<SemanticInstance<'db>> {
        match self.source(db) {
            RuntimeInstanceSource::Semantic(semantic) => Some(semantic),
            RuntimeInstanceSource::Synthetic(_) => None,
        }
    }
}

#[salsa::tracked]
#[derive(Debug)]
pub struct RuntimeInstance<'db> {
    pub key: RuntimeInstanceKey<'db>,
}

#[salsa::tracked]
impl<'db> RuntimeInstance<'db> {
    #[salsa::tracked]
    pub fn signature(self, db: &'db dyn MirDb) -> RuntimeSignature<'db> {
        match self.key(db).source(db) {
            RuntimeInstanceSource::Semantic(semantic) => {
                runtime_signature_for_key(db, semantic, self.key(db).params(db))
            }
            RuntimeInstanceSource::Synthetic(synthetic) => {
                runtime_synthetic_signature(synthetic.spec(db).clone())
            }
        }
    }

    #[salsa::tracked]
    pub fn body(self, db: &'db dyn MirDb) -> RuntimeBody<'db> {
        lower_runtime_body(db, self).body(db)
    }

    #[salsa::tracked(return_ref)]
    pub fn calls(self, db: &'db dyn MirDb) -> Vec<RuntimeCallEdge<'db>> {
        lower_runtime_body(db, self).direct_callees(db)
    }

    #[salsa::tracked(return_ref)]
    pub fn referenced_const_regions(
        self,
        db: &'db dyn MirDb,
    ) -> Vec<crate::runtime::ConstRegionId<'db>> {
        lower_runtime_body(db, self).referenced_const_regions(db)
    }

    #[salsa::tracked(return_ref)]
    pub fn referenced_code_regions(
        self,
        db: &'db dyn MirDb,
    ) -> Vec<crate::runtime::RuntimeCodeRegion<'db>> {
        lower_runtime_body(db, self).referenced_code_regions(db)
    }
}

#[salsa::tracked]
pub fn get_or_build_runtime_instance<'db>(
    db: &'db dyn MirDb,
    key: RuntimeInstanceKey<'db>,
) -> RuntimeInstance<'db> {
    RuntimeInstance::new(db, key)
}

#[salsa::tracked]
fn lower_runtime_body<'db>(
    db: &'db dyn MirDb,
    instance: RuntimeInstance<'db>,
) -> LoweredRuntimeBody<'db> {
    let body = match instance.key(db).source(db) {
        RuntimeInstanceSource::Semantic(semantic) => {
            if let Err(diag) = check_semantic_borrows(db, semantic) {
                panic!(
                    "semantic borrow checking failed for {:?}: {}",
                    semantic.key(db),
                    diag.message,
                );
            }
            lower_to_rmir(db, instance)
        }
        RuntimeInstanceSource::Synthetic(synthetic) => {
            lower_synthetic_runtime_body(db, instance, synthetic.spec(db).clone())
        }
    };
    let direct_callees = collect_runtime_calls_lowered(&body);
    let referenced_const_regions = collect_referenced_const_regions(&body);
    let referenced_code_regions = collect_referenced_code_regions(&body);
    LoweredRuntimeBody::new(
        db,
        body,
        direct_callees,
        referenced_const_regions,
        referenced_code_regions,
    )
}

pub(crate) fn runtime_instance_lowered_body<'db>(
    db: &'db dyn MirDb,
    instance: RuntimeInstance<'db>,
) -> LoweredRuntimeBody<'db> {
    lower_runtime_body(db, instance)
}
