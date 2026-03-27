use rustc_hash::FxHashSet;

use crate::{
    analysis::{
        HirAnalysisDb,
        semantic::{SemanticBody, SemanticCalleeRef, lower::lower_to_smir, verify_semantic_body},
        ty::ty_check::{BodyOwner, TypedBody},
    },
    hir_def::{Expr, Partial},
};

use super::{
    GenericSubst, ImplEnv, instantiate_typed_body, semantic_callee_key, typed_body_template,
};

#[salsa::interned]
#[derive(Debug)]
pub struct SemanticInstanceKey<'db> {
    pub owner: BodyOwner<'db>,
    pub subst: GenericSubst<'db>,
    pub impl_env: ImplEnv<'db>,
}

impl<'db> SemanticInstanceKey<'db> {
    pub fn instantiate_typed_body(self, db: &'db dyn HirAnalysisDb) -> TypedBody<'db> {
        instantiate_typed_body(db, typed_body_template(db, self.owner(db)), self.subst(db))
    }
}

#[salsa::tracked]
#[derive(Debug)]
pub struct SemanticInstance<'db> {
    pub key: SemanticInstanceKey<'db>,
}

#[salsa::tracked]
impl<'db> SemanticInstance<'db> {
    #[salsa::tracked]
    pub fn body(self, db: &'db dyn HirAnalysisDb) -> SemanticBody<'db> {
        lower_semantic_body(db, self)
    }

    #[salsa::tracked(return_ref)]
    pub fn callees(self, db: &'db dyn HirAnalysisDb) -> Vec<SemanticCalleeRef<'db>> {
        collect_semantic_callees(db, self)
    }
}

pub fn get_or_build_semantic_instance<'db>(
    db: &'db dyn HirAnalysisDb,
    key: SemanticInstanceKey<'db>,
) -> SemanticInstance<'db> {
    let instance = SemanticInstance::new(db, key);
    for callee in instance.callees(db) {
        get_or_build_semantic_instance(db, callee.key);
    }
    instance
}

fn lower_semantic_body<'db>(
    db: &'db dyn HirAnalysisDb,
    instance: SemanticInstance<'db>,
) -> SemanticBody<'db> {
    let key = instance.key(db);
    let typed_body = key.instantiate_typed_body(db);
    let body = lower_to_smir(db, instance, key.owner(db), typed_body);
    verify_semantic_body(&body).expect("invalid semantic MIR");
    body
}

fn collect_semantic_callees<'db>(
    db: &'db dyn HirAnalysisDb,
    instance: SemanticInstance<'db>,
) -> Vec<SemanticCalleeRef<'db>> {
    let key = instance.key(db);
    let typed_body = key.instantiate_typed_body(db);
    let Some(body) = typed_body.body() else {
        return Vec::new();
    };

    let mut seen = FxHashSet::default();
    let mut callees = Vec::new();
    for (expr_id, expr) in body.exprs(db).iter() {
        let Partial::Present(expr) = expr else {
            continue;
        };
        if !matches!(expr, Expr::Call(..) | Expr::MethodCall(..)) {
            continue;
        }

        let Some(callable) = typed_body.callable_expr(expr_id) else {
            continue;
        };
        let Some(callee_key) = semantic_callee_key(db, key, &typed_body, callable) else {
            continue;
        };

        if seen.insert(callee_key) {
            callees.push(SemanticCalleeRef { key: callee_key });
        }
    }

    callees
}
