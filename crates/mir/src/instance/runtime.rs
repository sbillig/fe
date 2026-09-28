use hir::analysis::{
    semantic::{SemanticInstance, check_semantic_borrows, check_semantic_boundaries},
    ty::{corelib::resolve_lib_type_path, ty_check::BodyOwner, ty_def::TyId},
};
use hir::hir_def::{HirIngot, scope_graph::ScopeId};
use salsa::Update;

use crate::{
    db::MirDb,
    runtime::{
        LowerError, LoweredRuntimeBody, RuntimeBody, RuntimeCallEdge, RuntimeClass,
        RuntimeExitBehavior, RuntimeInterfaceSignature, RuntimeSyntheticSpec,
        lower::{
            abi::runtime_declaration_abi_plan,
            body::lower_to_rmir,
            call::{
                collect_referenced_code_regions, collect_referenced_const_regions,
                collect_runtime_calls as collect_runtime_calls_lowered,
            },
            returns::runtime_exit_behavior,
        },
        synthetic::lower_synthetic_runtime_body,
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

/// Runtime ABI selected at the package root and inherited by callees.
/// This is deliberately independent of semantic normalization/evidence scope.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Update)]
pub struct RuntimeAbi<'db> {
    pub sol: Option<TyId<'db>>,
}

impl<'db> RuntimeAbi<'db> {
    pub fn for_scope(db: &'db dyn MirDb, scope: ScopeId<'db>) -> Self {
        Self {
            sol: resolve_lib_type_path(db, scope.ingot(db).root_mod(db).scope(), "std::abi::Sol"),
        }
    }

    pub fn library_scope(self, db: &'db dyn MirDb) -> Option<ScopeId<'db>> {
        Some(self.sol?.ingot(db)?.root_mod(db).scope())
    }
}

#[salsa::interned]
#[derive(Debug)]
pub struct RuntimeInstanceKey<'db> {
    pub source: RuntimeInstanceSource<'db>,
    pub abi: RuntimeAbi<'db>,
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
    pub fn interface_signature(self, db: &'db dyn MirDb) -> RuntimeInterfaceSignature<'db> {
        runtime_interface_signature_for_key(db, self.key(db))
    }

    #[salsa::tracked]
    pub fn exit_behavior(self, db: &'db dyn MirDb) -> RuntimeExitBehavior {
        runtime_exit_behavior(db, self.key(db))
    }

    #[salsa::tracked]
    pub fn body(self, db: &'db dyn MirDb) -> RuntimeBody<'db> {
        expect_lowered_runtime_body(db, self).body(db)
    }

    #[salsa::tracked(return_ref)]
    pub fn calls(self, db: &'db dyn MirDb) -> Vec<RuntimeCallEdge<'db>> {
        if self.is_external_declaration(db) {
            return Vec::new();
        }
        expect_lowered_runtime_body(db, self).direct_callees(db)
    }

    #[salsa::tracked(return_ref)]
    pub fn referenced_const_regions(
        self,
        db: &'db dyn MirDb,
    ) -> Vec<crate::runtime::ConstRegionId<'db>> {
        if self.is_external_declaration(db) {
            return Vec::new();
        }
        expect_lowered_runtime_body(db, self).referenced_const_regions(db)
    }

    #[salsa::tracked(return_ref)]
    pub fn referenced_code_regions(
        self,
        db: &'db dyn MirDb,
    ) -> Vec<crate::runtime::RuntimeCodeRegion<'db>> {
        if self.is_external_declaration(db) {
            return Vec::new();
        }
        expect_lowered_runtime_body(db, self).referenced_code_regions(db)
    }
}

impl<'db> RuntimeInstance<'db> {
    pub fn is_external_declaration(self, db: &'db dyn MirDb) -> bool {
        self.key(db).semantic(db).is_some_and(|semantic| {
            matches!(semantic.key(db).owner(db), BodyOwner::Func(func) if func.is_extern(db))
        })
    }
}

pub(crate) fn runtime_interface_signature_for_key<'db>(
    db: &'db dyn MirDb,
    key: RuntimeInstanceKey<'db>,
) -> RuntimeInterfaceSignature<'db> {
    runtime_declaration_abi_plan(db, key).signature()
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
) -> Result<LoweredRuntimeBody<'db>, LowerError> {
    let body = match instance.key(db).source(db) {
        RuntimeInstanceSource::Semantic(semantic) => {
            if let Err(diag) = check_semantic_borrows(db, semantic) {
                return Err(LowerError::Unsupported(format!(
                    "semantic borrow checking failed for {:?}: {}",
                    semantic.key(db),
                    diag
                )));
            }
            if let Err(diag) = check_semantic_boundaries(db, semantic) {
                return Err(LowerError::Unsupported(format!(
                    "semantic boundary checking failed for {:?}: {}",
                    semantic.key(db),
                    diag
                )));
            }
            lower_to_rmir(db, instance)?
        }
        RuntimeInstanceSource::Synthetic(synthetic) => {
            lower_synthetic_runtime_body(db, instance, synthetic.spec(db).clone())?
        }
    };
    // Anchor lowering to the canonical class discipline: in debug builds,
    // verify each body as it is produced so a divergence is attributed to
    // the instance being lowered instead of surfacing later at package
    // assembly. Release builds rely on the unconditional package-level
    // verification.
    #[cfg(debug_assertions)]
    if let Err(failure) = crate::verify::verify_runtime_body_detailed(db, &db, &body) {
        panic!(
            "lowering produced an invalid runtime body for {:?}:\n{}",
            instance.key(db).source(db),
            crate::runtime::format_runtime_verify_failure(db, &body, &failure),
        );
    }
    let direct_callees = collect_runtime_calls_lowered(&body);
    let referenced_const_regions = collect_referenced_const_regions(&body);
    let referenced_code_regions = collect_referenced_code_regions(&body);
    Ok(LoweredRuntimeBody::new(
        db,
        body,
        direct_callees,
        referenced_const_regions,
        referenced_code_regions,
    ))
}

pub(crate) fn runtime_instance_lowered_body<'db>(
    db: &'db dyn MirDb,
    instance: RuntimeInstance<'db>,
) -> Result<LoweredRuntimeBody<'db>, LowerError> {
    lower_runtime_body(db, instance)
}

fn expect_lowered_runtime_body<'db>(
    db: &'db dyn MirDb,
    instance: RuntimeInstance<'db>,
) -> LoweredRuntimeBody<'db> {
    lower_runtime_body(db, instance).unwrap_or_else(|err| {
        panic!(
            "runtime lowering failed for {:?}: {err}",
            instance.key(db).source(db)
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::{
        RTerminator,
        package::{
            root_runtime_instance, runtime_instance_for_semantic, runtime_instance_stable_key,
        },
    };
    use common::{InputDb, stdlib::HasBuiltinStd};
    use driver::DriverDataBase;
    use hir::analysis::{
        semantic::{
            ImplEnv, SemanticInstanceKey, get_or_build_semantic_instance,
            identity_semantic_instance_key, semantic_borrow_summary,
        },
        ty::ty_check::BodyOwner,
    };
    use hir::{analysis::ty::corelib::resolve_lib_func_path, hir_def::TopLevelMod};
    use url::Url;

    #[test]
    fn runtime_lowering_cannot_discharge_pending_semantic_validation() {
        for source in [
            "extern { fn opaque() }\nfn entry() { opaque() }",
            "extern { fn opaque() -> ! }\nfn entry() -> u256 { opaque() }",
            "trait Operation { fn apply() }\nfn entry<T: Operation>() { T::apply() }",
        ] {
            let mut db = DriverDataBase::default();
            let file = db.workspace().touch(
                &mut db,
                Url::parse("file:///pending_runtime_validation.fe").unwrap(),
                Some(source.into()),
            );
            let module = db.top_mod(file);
            let func = module
                .all_funcs(&db)
                .iter()
                .copied()
                .find(|func| {
                    func.name(&db)
                        .to_opt()
                        .is_some_and(|name| name.data(&db) == "entry")
                })
                .unwrap();
            let semantic = get_or_build_semantic_instance(
                &db,
                identity_semantic_instance_key(&db, BodyOwner::Func(func)),
            );
            let key = RuntimeInstanceKey::new(
                &db,
                RuntimeInstanceSource::Semantic(semantic),
                RuntimeAbi::for_scope(&db, func.scope()),
                Vec::new(),
            );
            let instance = get_or_build_runtime_instance(&db, key);
            let Err(LowerError::Unsupported(message)) = lower_runtime_body(&db, instance) else {
                panic!("pending validation reached runtime lowering");
            };
            assert!(
                message.contains("requires concrete implementations"),
                "{message}"
            );
        }
    }

    fn unwrap_runtime<'db>(
        db: &'db DriverDataBase,
        module: TopLevelMod<'db>,
    ) -> RuntimeInstance<'db> {
        let func = module
            .all_funcs(db)
            .iter()
            .copied()
            .find(|func| {
                func.name(db)
                    .to_opt()
                    .is_some_and(|name| name.data(db) == "entry")
            })
            .expect("entry function");
        let caller = get_or_build_semantic_instance(
            db,
            identity_semantic_instance_key(db, BodyOwner::Func(func)),
        );
        let source_key = caller
            .call_sites(db)
            .iter()
            .flatten()
            .filter_map(|site| site.callee)
            .find(|callee| {
                matches!(callee.key.owner(db), BodyOwner::Func(func)
                if func.name(db).to_opt().is_some_and(|name| name.data(db) == "unwrap"))
            })
            .expect("unwrap call")
            .key;
        // Exercise the canonical core-owned key even without the inherent-key repair.
        let key = SemanticInstanceKey::new(
            db,
            source_key.owner(db),
            source_key.subst(db),
            source_key.effect_providers(db),
            ImplEnv::empty(db, source_key.owner(db).scope()),
        );
        let semantic = get_or_build_semantic_instance(db, key);
        runtime_instance_for_semantic(db, semantic, RuntimeAbi::for_scope(db, module.scope()))
    }

    #[test]
    fn canonical_unwrap_preserves_payload_behavior_in_either_query_order() {
        for (declaration, error_ty, helper) in [
            ("", "u256", Some("revert")),
            (
                "#[error]\npub struct Payload { pub code: u256 }",
                "Payload",
                Some("revert_error"),
            ),
            ("enum PlainError { Oops }", "PlainError", None),
        ] {
            for summary_first in [false, true] {
                let mut db = DriverDataBase::default();
                let source = format!(
                    "{declaration}\nfn entry(_ value: own Result<{error_ty}, u256>) -> u256 {{ value.unwrap() }}"
                );
                let file = db.workspace().touch(
                    &mut db,
                    Url::parse("file:///panic_payload.fe").unwrap(),
                    Some(source),
                );
                let module = db.top_mod(file);
                let diagnostics = db.run_on_top_mod(module);
                assert!(diagnostics.is_empty(), "{}", diagnostics.format_diags(&db));
                let instance = unwrap_runtime(&db, module);
                let semantic = instance.key(&db).semantic(&db).unwrap();
                if summary_first {
                    semantic_borrow_summary(&db, semantic).unwrap();
                    instance.interface_signature(&db);
                }
                let no_abi = runtime_instance_for_semantic(&db, semantic, RuntimeAbi { sol: None });
                if summary_first {
                    assert!(
                        lower_runtime_body(&db, no_abi)
                            .unwrap_err()
                            .to_string()
                            .contains("missing std::abi::Sol")
                    );
                }
                let lowered = lower_runtime_body(&db, instance);
                if let Some(helper) = helper {
                    let lowered = lowered.unwrap();
                    let body = lowered.body(&db);
                    let helper_ingot = resolve_lib_type_path(&db, module.scope(), "std::abi::Sol")
                        .unwrap()
                        .ingot(&db)
                        .unwrap();
                    assert!(body.blocks.iter().any(|block| {
                        let RTerminator::TerminalCall { callee, .. } = block.terminator else {
                            return false;
                        };
                        let BodyOwner::Func(func) =
                            callee.key(&db).semantic(&db).unwrap().key(&db).owner(&db)
                        else {
                            return false;
                        };
                        func.name(&db)
                            .to_opt()
                            .is_some_and(|name| name.data(&db) == helper)
                            && func.scope().ingot(&db) == helper_ingot
                            && callee.key(&db).abi(&db) == instance.key(&db).abi(&db)
                    }));
                } else {
                    let message = lowered.unwrap_err().to_string();
                    assert!(
                        message.contains("`PlainError` to implement `Encode<Sol>` and `AbiSize`"),
                        "{message}"
                    );
                }
                if !summary_first {
                    assert!(
                        lower_runtime_body(&db, no_abi)
                            .unwrap_err()
                            .to_string()
                            .contains("missing std::abi::Sol")
                    );
                    instance.interface_signature(&db);
                    semantic_borrow_summary(&db, semantic).unwrap();
                }
            }
        }
    }

    #[test]
    fn shared_semantic_unwrap_keeps_selected_runtime_abis_distinct() {
        for reverse in [false, true] {
            let mut db = DriverDataBase::default();
            let std_files: Vec<_> = db
                .builtin_std()
                .files(&db)
                .iter()
                .map(|(url, file)| {
                    (
                        url.path().trim_start_matches('/').to_owned(),
                        file.text(&db).to_owned(),
                    )
                })
                .collect();
            let mut entries = Vec::new();
            for library in ["first", "second"] {
                let base = Url::parse(&format!("file:///{library}/")).unwrap();
                for (path, source) in &std_files {
                    db.workspace()
                        .touch(&mut db, base.join(path).unwrap(), Some(source.clone()));
                }
                // Rooting in two std ingots models two packages selecting different std libraries.
                let file = db.workspace().touch(
                    &mut db,
                    base.join("src/runtime_entry.fe").unwrap(),
                    Some(
                        "pub fn entry(_ value: own Result<u256, u256>) -> u256 { value.unwrap() }"
                            .into(),
                    ),
                );
                entries.push(file);
            }
            if reverse {
                entries.reverse();
            }
            let instances: Vec<_> = entries
                .into_iter()
                .map(|file| {
                    let module = db.top_mod(file);
                    let func = module.all_funcs(&db).iter().copied().find(|func|
                        func.name(&db).to_opt().is_some_and(|name| name.data(&db) == "entry")).unwrap();
                    let semantic = get_or_build_semantic_instance(&db, identity_semantic_instance_key(&db, BodyOwner::Func(func)));
                    let root = root_runtime_instance(&db, semantic);
                    let callee = root.calls(&db).iter().find(|edge| matches!(edge.callee.key(&db).semantic(&db).unwrap().key(&db).owner(&db),
                        BodyOwner::Func(func) if func.name(&db).to_opt().is_some_and(|name| name.data(&db) == "unwrap"))).unwrap().callee;
                    assert_eq!(callee.key(&db).abi(&db), root.key(&db).abi(&db));
                    callee
                })
                .collect();
            let [left, right] = instances.as_slice() else {
                unreachable!()
            };
            assert_eq!(left.key(&db).semantic(&db), right.key(&db).semantic(&db));
            assert_ne!(left.key(&db).abi(&db), right.key(&db).abi(&db));
            assert_ne!(left, right);
            assert_ne!(
                runtime_instance_stable_key(&db, *left),
                runtime_instance_stable_key(&db, *right)
            );
            for instance in instances {
                let body = lower_runtime_body(&db, instance).unwrap().body(&db);
                let library = instance.key(&db).abi(&db).sol.unwrap().ingot(&db).unwrap();
                let expected =
                    resolve_lib_func_path(&db, library.root_mod(&db).scope(), "std::evm::revert")
                        .unwrap();
                assert!(body.blocks.iter().any(|block| matches!(block.terminator,
                    RTerminator::TerminalCall { callee, .. }
                    if callee.key(&db).semantic(&db).unwrap().key(&db).owner(&db) == BodyOwner::Func(expected)
                        && callee.key(&db).abi(&db) == instance.key(&db).abi(&db))));
            }
        }
    }
}
