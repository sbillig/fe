use hir::semantic::{ContractFieldLayoutInfo, ProviderSource, RecvArmAbiInfo, RecvArmView};
use hir::{
    analysis::{
        semantic::{
            GenericSubst, ImplEnv, ManualContractSection, RootSemanticInstanceError,
            SemanticInstance, SemanticInstanceKey, ValueProvenance, get_or_build_semantic_instance,
            owner_effect_bindings, resolved_provider_binding_for_instance_effect,
            root_semantic_instance_key, semantic_binding_lowering, semantic_binding_ty,
            semantic_instance_assumptions,
        },
        ty::{
            corelib::{resolve_core_trait, resolve_lib_type_path},
            trait_def::{TraitInstId, resolve_trait_method_instance},
            trait_resolution::TraitSolveCx,
            ty_check::BodyOwner,
            ty_def::TyId,
        },
    },
    hir_def::{Contract, Func, IdentId, InlineHint, ItemKind, ManualContractRootAttr, TopLevelMod},
};
use rustc_hash::{FxHashMap, FxHashSet};

use crate::{
    db::MirDb,
    instance::{
        RuntimeInstance, RuntimeInstanceKey, RuntimeInstanceSource, RuntimeSyntheticInstance,
        get_or_build_runtime_instance,
    },
    runtime::code_region::{code_region_symbol, runtime_code_region_for_manual_root},
    runtime::lower::class::{
        owner_effect_binding_boundary, provider_class_for_target_in_env,
        runtime_class_for_direct_value_provider_in_env,
        runtime_class_for_effect_binding_provider_in_env, runtime_param_class,
        runtime_visible_binding_class, top_level_class_for_ty_in_env,
    },
    runtime::lower::layout::RuntimeTypeEnv,
    runtime::{
        AddressSpaceKind, ConstRegionId, ContractEffectArgPlan, ContractFieldBinding,
        ContractInitAbiPlan, ContractRecvAbiPlan, DispatchArm, DispatchDefault, InitArgsPlan,
        RefKind, ResolvedCodeRegion, RuntimeCodeRegion, RuntimeCodeRegionKey, RuntimeFunction,
        RuntimeFunctionOwner, RuntimeInlineHint, RuntimeInputPlan, RuntimeLinkage, RuntimeObject,
        RuntimePackage, RuntimePackagePlan, RuntimeReturnPlan, RuntimeSection, RuntimeSectionName,
        RuntimeSectionRef, RuntimeSyntheticSpec,
    },
    verify::verify_runtime_package,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LowerError {
    Unsupported(String),
}

impl std::fmt::Display for LowerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LowerError::Unsupported(message) => write!(f, "{message}"),
        }
    }
}

impl std::error::Error for LowerError {}

#[derive(Clone, Copy)]
struct ManualContractRoot<'db> {
    func: Func<'db>,
    instance: RuntimeInstance<'db>,
    contract_name: &'db str,
    section: ManualContractSection,
}

type ManualContractObjectSpec<'db> = (
    String,
    Vec<(RuntimeSectionName, RuntimeInstance<'db>)>,
    Vec<RuntimeInstance<'db>>,
);

pub fn build_runtime_package<'db>(
    db: &'db dyn MirDb,
    top_mod: TopLevelMod<'db>,
) -> Result<RuntimePackage<'db>, LowerError> {
    if !top_mod.all_contracts(db).is_empty()
        || !discover_manual_contract_roots(db, top_mod)?.is_empty()
    {
        return build_contract_package(db, top_mod);
    }

    let funcs = top_mod
        .all_funcs(db)
        .iter()
        .copied()
        .filter(|func| func.top_mod(db) == top_mod)
        .filter(|func| !func.is_extern(db) && !is_test_func(db, *func))
        .collect::<Vec<_>>();
    let mut funcs = funcs;
    funcs.sort_by_key(|func| {
        func.name(db)
            .to_opt()
            .map(|name| name.data(db).to_string())
            .unwrap_or_default()
    });
    let mut entry_funcs = Vec::new();
    for func in funcs.iter().copied() {
        if runtime_root_candidate(db, func)? {
            entry_funcs.push(func);
        }
    }
    if entry_funcs.is_empty() {
        return Ok(RuntimePackage::new(
            db,
            top_mod,
            Vec::new(),
            RuntimePackagePlan::new(db, Vec::new(), Vec::new(), Vec::new(), Vec::new(), None),
        ));
    }

    let mut roots = Vec::new();
    for func in entry_funcs.iter().copied() {
        roots.push((
            func,
            runtime_instance_for_semantic(
                db,
                semantic_instance_for_root_owner(db, BodyOwner::Func(func))?,
            ),
        ));
    }
    let entry = roots
        .iter()
        .find(|(func, _)| {
            func.name(db)
                .to_opt()
                .is_some_and(|name| name.data(db) == "main")
        })
        .or_else(|| roots.first())
        .map(|(_, instance)| *instance)
        .expect("entry root candidates should include the chosen entry function");
    let root = synthetic_instance(
        db,
        RuntimeSyntheticSpec::MainRoot { callee: entry },
        Vec::new(),
    );
    let mut package_roots = roots
        .into_iter()
        .map(|(_, instance)| instance)
        .collect::<Vec<_>>();
    package_roots.push(root);
    let package = build_non_contract_package(
        db,
        top_mod,
        package_roots,
        vec![(sanitize_object_name("main"), RuntimeSectionName::Main, root)],
        Some("main"),
    )?;
    verify_runtime_package(db, package)
        .map_err(|err| LowerError::Unsupported(format!("invalid runtime package: {err:?}")))?;
    Ok(package)
}

pub fn build_test_runtime_package<'db>(
    db: &'db dyn MirDb,
    top_mod: TopLevelMod<'db>,
    filter: Option<&str>,
) -> Result<RuntimePackage<'db>, LowerError> {
    if !top_mod.all_contracts(db).is_empty()
        || !discover_manual_contract_roots(db, top_mod)?.is_empty()
    {
        return build_contract_test_package(db, top_mod, filter);
    }

    let mut roots = Vec::new();
    let mut objects = Vec::new();
    for &func in top_mod.all_funcs(db) {
        if func.top_mod(db) != top_mod {
            continue;
        }
        let Some(attrs) = ItemKind::from(func).attrs(db) else {
            continue;
        };
        if attrs.get_attr(db, "test").is_none() {
            continue;
        }

        let name = func
            .name(db)
            .to_opt()
            .map(|name| name.data(db).to_string())
            .unwrap_or_else(|| "<anonymous>".to_string());
        if let Some(filter) = filter
            && !name.contains(filter)
        {
            continue;
        }

        let semantic = semantic_instance_for_root_owner(db, BodyOwner::Func(func))?;
        let runtime_root = runtime_instance_for_semantic(db, semantic);
        let root = synthetic_instance(
            db,
            RuntimeSyntheticSpec::TestRoot {
                name: name.clone(),
                callee: runtime_root,
            },
            Vec::new(),
        );
        roots.push(root);
        objects.push((
            sanitize_object_name(&name),
            RuntimeSectionName::Test(name),
            root,
        ));
    }

    let package = build_non_contract_package(db, top_mod, roots, objects, None)?;
    verify_runtime_package(db, package)
        .map_err(|err| LowerError::Unsupported(format!("invalid runtime package: {err:?}")))?;
    Ok(package)
}

fn build_contract_package<'db>(
    db: &'db dyn MirDb,
    top_mod: TopLevelMod<'db>,
) -> Result<RuntimePackage<'db>, LowerError> {
    let mut roots = Vec::new();
    let mut objects = Vec::new();
    let contracts = top_mod.all_contracts(db);
    for &contract in contracts {
        let (name, sections, section_roots) = contract_object_spec(db, contract)?;
        roots.extend(section_roots);
        objects.push((name, sections));
    }
    for (name, sections, section_roots) in manual_contract_objects(db, top_mod)? {
        roots.extend(section_roots);
        objects.push((name, sections));
    }

    let primary = (objects.len() == 1).then(|| objects[0].0.clone());
    let package = build_sectioned_package(db, top_mod, roots, objects, primary.as_deref())?;
    verify_runtime_package(db, package)
        .map_err(|err| LowerError::Unsupported(format!("invalid runtime package: {err:?}")))?;
    Ok(package)
}

fn build_contract_test_package<'db>(
    db: &'db dyn MirDb,
    top_mod: TopLevelMod<'db>,
    filter: Option<&str>,
) -> Result<RuntimePackage<'db>, LowerError> {
    let mut roots = Vec::new();
    let mut objects = Vec::new();
    let contracts = top_mod.all_contracts(db);
    for &contract in contracts {
        let (name, sections, section_roots) = contract_object_spec(db, contract)?;
        roots.extend(section_roots);
        objects.push((name, sections));
    }
    for (name, sections, section_roots) in manual_contract_objects(db, top_mod)? {
        roots.extend(section_roots);
        objects.push((name, sections));
    }

    for &func in top_mod.all_funcs(db) {
        if func.top_mod(db) != top_mod {
            continue;
        }
        let Some(attrs) = ItemKind::from(func).attrs(db) else {
            continue;
        };
        if attrs.get_attr(db, "test").is_none() {
            continue;
        }
        let name = func
            .name(db)
            .to_opt()
            .map(|name| name.data(db).to_string())
            .unwrap_or_else(|| "<anonymous>".to_string());
        if let Some(filter) = filter
            && !name.contains(filter)
        {
            continue;
        }
        let semantic = semantic_instance_for_root_owner(db, BodyOwner::Func(func))?;
        let runtime_root = runtime_instance_for_semantic(db, semantic);
        let root = synthetic_instance(
            db,
            RuntimeSyntheticSpec::TestRoot {
                name: name.clone(),
                callee: runtime_root,
            },
            Vec::new(),
        );
        roots.push(root);
        objects.push((
            sanitize_object_name(&name),
            vec![(RuntimeSectionName::Test(name), root)],
        ));
    }

    let primary = (objects.len() == 1).then(|| objects[0].0.clone());
    let package = build_sectioned_package(db, top_mod, roots, objects, primary.as_deref())?;
    verify_runtime_package(db, package)
        .map_err(|err| LowerError::Unsupported(format!("invalid runtime package: {err:?}")))?;
    Ok(package)
}

fn manual_contract_objects<'db>(
    db: &'db dyn MirDb,
    top_mod: TopLevelMod<'db>,
) -> Result<Vec<ManualContractObjectSpec<'db>>, LowerError> {
    let roots = discover_manual_contract_roots(db, top_mod)?;
    let mut by_contract =
        FxHashMap::<String, (Option<RuntimeInstance<'db>>, Option<RuntimeInstance<'db>>)>::default(
        );
    for root in roots {
        let entry = by_contract
            .entry(root.contract_name.to_string())
            .or_insert((None, None));
        match root.section {
            ManualContractSection::Init => {
                if entry.0.replace(root.instance).is_some() {
                    return Err(LowerError::Unsupported(format!(
                        "duplicate #[contract_init({})] root in package",
                        root.contract_name
                    )));
                }
            }
            ManualContractSection::Runtime => {
                if entry.1.replace(root.instance).is_some() {
                    return Err(LowerError::Unsupported(format!(
                        "duplicate #[contract_runtime({})] root in package",
                        root.contract_name
                    )));
                }
            }
        }
    }

    let high_level_names = top_mod
        .all_contracts(db)
        .iter()
        .filter_map(|contract| {
            contract
                .name(db)
                .to_opt()
                .map(|name| name.data(db).to_string())
        })
        .collect::<FxHashSet<_>>();
    for contract_name in by_contract.keys() {
        if high_level_names.contains(contract_name) {
            return Err(LowerError::Unsupported(format!(
                "manual contract roots for `{contract_name}` conflict with a high-level contract of the same name"
            )));
        }
    }

    let mut objects = by_contract
        .into_iter()
        .map(|(contract_name, (init, runtime))| {
            let mut sections = Vec::new();
            let mut roots = Vec::new();
            if let Some(init) = init {
                sections.push((RuntimeSectionName::Init, init));
                roots.push(init);
            }
            if let Some(runtime) = runtime {
                sections.push((RuntimeSectionName::Runtime, runtime));
                roots.push(runtime);
            }
            (sanitize_object_name(&contract_name), sections, roots)
        })
        .collect::<Vec<_>>();
    objects.sort_by(|lhs, rhs| lhs.0.cmp(&rhs.0));
    Ok(objects)
}

fn discover_manual_contract_roots<'db>(
    db: &'db dyn MirDb,
    top_mod: TopLevelMod<'db>,
) -> Result<Vec<ManualContractRoot<'db>>, LowerError> {
    let mut roots = Vec::new();
    for &func in top_mod.all_funcs(db) {
        if func.top_mod(db) != top_mod {
            continue;
        }
        let Some(attr) = func.manual_contract_root_attr(db) else {
            continue;
        };
        let (contract_name, section) = match attr {
            ManualContractRootAttr::Init { contract_name } => {
                (contract_name.data(db), ManualContractSection::Init)
            }
            ManualContractRootAttr::Runtime { contract_name } => {
                (contract_name.data(db), ManualContractSection::Runtime)
            }
            ManualContractRootAttr::Error(err) => {
                return Err(LowerError::Unsupported(format!(
                    "invalid manual contract root attr on `{}`: {err:?}",
                    func.name(db)
                        .to_opt()
                        .map(|name| name.data(db).to_string())
                        .unwrap_or_else(|| "<anonymous>".to_string())
                )));
            }
        };
        if !func.arg_tys(db).is_empty() || func.return_ty(db) != TyId::unit(db) {
            return Err(LowerError::Unsupported(format!(
                "manual contract root `{}` must be monomorphic, unit-returning, and take no ordinary value params",
                func.name(db)
                    .to_opt()
                    .map(|name| name.data(db).to_string())
                    .unwrap_or_else(|| "<anonymous>".to_string())
            )));
        }
        roots.push(ManualContractRoot {
            func,
            instance: runtime_instance_for_semantic(
                db,
                semantic_instance_for_root_owner(db, BodyOwner::Func(func))?,
            ),
            contract_name,
            section,
        });
    }
    roots.sort_by_key(|root| {
        (
            root.contract_name.to_string(),
            matches!(root.section, ManualContractSection::Runtime),
            root.func
                .name(db)
                .to_opt()
                .map(|name| name.data(db).to_string()),
        )
    });
    Ok(roots)
}

fn contract_runtime_root<'db>(
    db: &'db dyn MirDb,
    contract: Contract<'db>,
) -> Result<RuntimeInstance<'db>, LowerError> {
    let abi_ty = sol_abi_ty(db, contract.scope())?;
    let mut dispatch = Vec::new();
    let mut default = DispatchDefault::RevertEmpty;
    for arm in contract.recv_views(db).flat_map(|recv| recv.arms(db)) {
        let (abi_info, wrapper) = contract_recv_wrapper(db, arm, abi_ty)?;
        if abi_info.is_fallback {
            if matches!(default, DispatchDefault::Call { .. }) {
                return Err(LowerError::Unsupported(format!(
                    "contract `{}` has multiple fallback recv arms",
                    contract_name(db, contract)
                )));
            }
            default = DispatchDefault::Call { wrapper };
            continue;
        }

        let selector = abi_info.selector_value.ok_or_else(|| {
            LowerError::Unsupported(format!(
                "recv arm in `{}` is missing a resolved selector",
                contract_name(db, contract)
            ))
        })?;
        dispatch.push(DispatchArm { selector, wrapper });
    }
    dispatch.sort_by_key(|arm| arm.selector);
    Ok(synthetic_instance(
        db,
        RuntimeSyntheticSpec::ContractRuntimeRoot {
            contract,
            dispatch: dispatch.into_boxed_slice(),
            default,
        },
        Vec::new(),
    ))
}

fn contract_init_root<'db>(
    db: &'db dyn MirDb,
    contract: Contract<'db>,
) -> Result<RuntimeInstance<'db>, LowerError> {
    let init_abi = contract_init_abi(db, contract)?;
    Ok(synthetic_instance(
        db,
        RuntimeSyntheticSpec::ContractInitRoot {
            contract,
            init_abi,
            runtime_region: RuntimeCodeRegion::new(
                db,
                RuntimeCodeRegionKey::ContractRuntime { contract },
            ),
        },
        Vec::new(),
    ))
}

fn contract_init_abi<'db>(
    db: &'db dyn MirDb,
    contract: Contract<'db>,
) -> Result<Option<RuntimeInstance<'db>>, LowerError> {
    let Some(init) = contract.init(db) else {
        return Ok(None);
    };

    let user_init = Some(runtime_instance_for_semantic(
        db,
        semantic_instance_for_root_owner(db, BodyOwner::ContractInit { contract })?,
    ));
    let owner_effect_args = contract_effect_arg_plans_for_site(
        db,
        contract,
        semantic_instance_for_root_owner(db, BodyOwner::ContractInit { contract })?,
    )?;
    let init_args = if contract.init_args_ty(db) == TyId::unit(db) {
        InitArgsPlan::None
    } else {
        InitArgsPlan::DecodeInitTail {
            tuple_ty: contract.init_args_ty(db),
            decode_fn: resolve_decode_instance(db, contract.scope(), contract.init_args_ty(db))?,
        }
    };
    Ok(Some(synthetic_instance(
        db,
        RuntimeSyntheticSpec::ContractInitAbi {
            plan: ContractInitAbiPlan {
                contract,
                payable: init.is_payable(db),
                user_init,
                owner_effect_args: owner_effect_args.into_boxed_slice(),
                init_args,
            },
        },
        Vec::new(),
    )))
}

fn contract_recv_wrapper<'db>(
    db: &'db dyn MirDb,
    arm: RecvArmView<'db>,
    abi_ty: TyId<'db>,
) -> Result<(RecvArmAbiInfo<'db>, RuntimeInstance<'db>), LowerError> {
    let contract = arm.contract(db);
    let abi_info = arm.abi_info(db, abi_ty);
    let recv = arm.recv(db);
    let user_recv = runtime_instance_for_semantic(
        db,
        semantic_instance_for_root_owner(
            db,
            BodyOwner::ContractRecvArm {
                contract,
                recv_idx: recv.recv_idx(db),
                arm_idx: arm.arm_idx(db),
            },
        )?,
    );
    let owner_effect_args = contract_effect_arg_plans_for_site(
        db,
        contract,
        semantic_instance_for_root_owner(
            db,
            BodyOwner::ContractRecvArm {
                contract,
                recv_idx: recv.recv_idx(db),
                arm_idx: arm.arm_idx(db),
            },
        )?,
    )?;
    let input = if abi_info.args_ty == TyId::unit(db) {
        RuntimeInputPlan::None
    } else {
        RuntimeInputPlan::DecodeCalldataPayload {
            msg_ty: abi_info.args_ty,
            decode_fn: resolve_decode_instance(db, contract.scope(), abi_info.args_ty)?,
        }
    };
    let ret = if let Some(ret_ty) = abi_info.ret_ty {
        RuntimeReturnPlan::Value {
            ty: ret_ty,
            encode_fn: resolve_encode_instance(db, contract.scope(), ret_ty)?,
        }
    } else {
        RuntimeReturnPlan::Unit
    };
    let wrapper = synthetic_instance(
        db,
        RuntimeSyntheticSpec::ContractRecvAbi {
            plan: ContractRecvAbiPlan {
                contract,
                selector: abi_info.selector_value,
                payable: match arm.arm(db) {
                    Some(recv_arm) => recv_arm.is_payable(db),
                    None => false,
                },
                user_recv,
                owner_effect_args: owner_effect_args.into_boxed_slice(),
                input,
                ret,
            },
        },
        Vec::new(),
    );
    Ok((abi_info, wrapper))
}

fn build_non_contract_package<'db>(
    db: &'db dyn MirDb,
    top_mod: TopLevelMod<'db>,
    roots: Vec<RuntimeInstance<'db>>,
    object_specs: Vec<(String, RuntimeSectionName, RuntimeInstance<'db>)>,
    primary_object_name: Option<&str>,
) -> Result<RuntimePackage<'db>, LowerError> {
    build_sectioned_package(
        db,
        top_mod,
        roots,
        object_specs
            .into_iter()
            .map(|(name, section, entry)| (name, vec![(section, entry)]))
            .collect(),
        primary_object_name,
    )
}

fn build_sectioned_package<'db>(
    db: &'db dyn MirDb,
    top_mod: TopLevelMod<'db>,
    mut roots: Vec<RuntimeInstance<'db>>,
    mut object_specs: Vec<(String, Vec<(RuntimeSectionName, RuntimeInstance<'db>)>)>,
    primary_object_name: Option<&str>,
) -> Result<RuntimePackage<'db>, LowerError> {
    let root_object_names = object_specs
        .iter()
        .map(|(name, _)| name.clone())
        .collect::<FxHashSet<_>>();
    let mut code_region_roots = Vec::new();
    let mut seen_region_roots = FxHashSet::default();
    let mut materialized_contracts = materialized_contracts_for_roots(db, &roots);

    loop {
        let functions = collect_runtime_functions(db, &roots);
        let region_specs = collect_function_root_regions(db, &functions);
        let mut changed = false;
        for region in region_specs {
            if !seen_region_roots.insert(region) {
                continue;
            }
            let RuntimeCodeRegionKey::FunctionRoot { symbol, callee } = region.key(db).clone()
            else {
                continue;
            };
            let root = synthetic_instance(
                db,
                RuntimeSyntheticSpec::CodeRegionRoot { symbol, callee },
                Vec::new(),
            );
            code_region_roots.push((region, root));
            roots.push(root);
            changed = true;
        }
        for contract in collect_referenced_contract_regions(db, &functions) {
            if !materialized_contracts.insert(contract) {
                continue;
            }
            let (name, sections, section_roots) = contract_object_spec(db, contract)?;
            object_specs.push((name, sections));
            roots.extend(section_roots);
            changed = true;
        }
        if !changed {
            break;
        }
    }

    let functions = collect_runtime_functions(db, &roots);
    let functions_by_instance = functions
        .iter()
        .map(|function| (function.instance(db), *function))
        .collect::<FxHashMap<_, _>>();
    let const_regions = collect_const_regions(db, &functions);

    let mut objects = object_specs
        .into_iter()
        .map(|(name, sections)| {
            make_runtime_object(
                db,
                name,
                sections
                    .into_iter()
                    .map(|(section_name, entry_instance)| {
                        let entry = *functions_by_instance
                            .get(&entry_instance)
                            .expect("section entry should be declared as a runtime function");
                        let reachable = collect_reachable_from_entry(db, entry_instance);
                        RuntimeSection {
                            name: section_name,
                            entry,
                            embeds: Vec::new(),
                            const_regions: collect_const_regions_for_reachable(
                                db,
                                &functions_by_instance,
                                &reachable,
                            ),
                        }
                    })
                    .collect(),
            )
        })
        .collect::<Vec<_>>();

    if !code_region_roots.is_empty() {
        let code_regions_object =
            build_code_regions_object(db, &functions_by_instance, &code_region_roots);
        objects.push(code_regions_object);
    }

    let code_regions =
        resolve_code_regions(db, &objects, &functions_by_instance, &code_region_roots);
    let code_region_map = code_regions
        .iter()
        .map(|region| (region.region(db), *region))
        .collect::<FxHashMap<_, _>>();
    objects = objects
        .into_iter()
        .map(|object| rewrite_object_embeds(db, object, &functions_by_instance, &code_region_map))
        .collect();
    objects = remap_object_section_refs(db, &objects);
    let code_regions = remap_resolved_code_regions(db, &objects, code_regions);

    let root_objects: Vec<_> = objects
        .iter()
        .filter(|object| root_object_names.contains(&object.name(db)))
        .copied()
        .collect();
    let primary_object = primary_object_name.and_then(|primary| {
        objects
            .iter()
            .find(|object| object.name(db) == primary)
            .copied()
    });

    Ok(RuntimePackage::new(
        db,
        top_mod,
        functions,
        RuntimePackagePlan::new(
            db,
            objects,
            const_regions,
            code_regions,
            root_objects,
            primary_object,
        ),
    ))
}

fn build_code_regions_object<'db>(
    db: &'db dyn MirDb,
    functions_by_instance: &FxHashMap<RuntimeInstance<'db>, RuntimeFunction<'db>>,
    roots: &[(RuntimeCodeRegion<'db>, RuntimeInstance<'db>)],
) -> RuntimeObject<'db> {
    let sections = roots
        .iter()
        .map(|(region, instance)| {
            let RuntimeCodeRegionKey::FunctionRoot { symbol, .. } = region.key(db).clone() else {
                unreachable!();
            };
            let entry = *functions_by_instance
                .get(instance)
                .expect("code-region root should be declared as a runtime function");
            let reachable = collect_reachable_from_entry(db, *instance);
            RuntimeSection {
                name: RuntimeSectionName::CodeRegion(symbol),
                entry,
                embeds: Vec::new(),
                const_regions: collect_const_regions_for_reachable(
                    db,
                    functions_by_instance,
                    &reachable,
                ),
            }
        })
        .collect();
    make_runtime_object(db, "CodeRegions".to_string(), sections)
}

fn collect_referenced_contract_regions<'db>(
    db: &'db dyn MirDb,
    functions: &[RuntimeFunction<'db>],
) -> Vec<Contract<'db>> {
    let mut seen = FxHashSet::default();
    let mut contracts = Vec::new();
    for function in functions {
        for region in function
            .instance(db)
            .referenced_code_regions(db)
            .iter()
            .copied()
        {
            let contract = match region.key(db) {
                RuntimeCodeRegionKey::ContractInit { contract }
                | RuntimeCodeRegionKey::ContractRuntime { contract } => contract,
                RuntimeCodeRegionKey::ManualContractRoot { .. }
                | RuntimeCodeRegionKey::FunctionRoot { .. } => continue,
            };
            if seen.insert(contract) {
                contracts.push(contract);
            }
        }
    }
    contracts.sort_by_key(|contract| contract_name(db, *contract));
    contracts
}

fn rewrite_object_embeds<'db>(
    db: &'db dyn MirDb,
    object: RuntimeObject<'db>,
    functions_by_instance: &FxHashMap<RuntimeInstance<'db>, RuntimeFunction<'db>>,
    code_region_map: &FxHashMap<RuntimeCodeRegion<'db>, ResolvedCodeRegion<'db>>,
) -> RuntimeObject<'db> {
    let section_refs = code_region_map
        .iter()
        .map(|(region, resolved)| (*region, resolved.source(db).clone()))
        .collect::<FxHashMap<_, _>>();
    let sections = object
        .sections(db)
        .iter()
        .cloned()
        .map(|mut section| {
            let reachable = collect_reachable_from_entry(db, section.entry.instance(db));
            section.embeds = collect_region_embeds(
                db,
                &reachable,
                functions_by_instance,
                &section_refs,
                RuntimeSectionRef::Local {
                    object,
                    section: section.name.clone(),
                },
            );
            section
        })
        .collect();
    make_runtime_object(db, object.name(db).clone(), sections)
}

fn remap_resolved_code_regions<'db>(
    db: &'db dyn MirDb,
    objects: &[RuntimeObject<'db>],
    code_regions: Vec<ResolvedCodeRegion<'db>>,
) -> Vec<ResolvedCodeRegion<'db>> {
    code_regions
        .into_iter()
        .map(|region| {
            make_resolved_code_region(
                db,
                region.region(db),
                region.symbol(db).clone(),
                remap_section_ref(db, objects, region.source(db).clone()),
                region.root(db),
            )
        })
        .collect()
}

fn remap_object_section_refs<'db>(
    db: &'db dyn MirDb,
    objects: &[RuntimeObject<'db>],
) -> Vec<RuntimeObject<'db>> {
    objects
        .iter()
        .map(|object| {
            let sections = object
                .sections(db)
                .iter()
                .cloned()
                .map(|mut section| {
                    section.embeds = section
                        .embeds
                        .into_iter()
                        .map(|embed| crate::runtime::RuntimeEmbed {
                            source: remap_section_ref(db, objects, embed.source),
                            as_symbol: embed.as_symbol,
                        })
                        .collect();
                    section
                })
                .collect();
            make_runtime_object(db, object.name(db).clone(), sections)
        })
        .collect()
}

fn remap_section_ref<'db>(
    db: &'db dyn MirDb,
    objects: &[RuntimeObject<'db>],
    section_ref: RuntimeSectionRef<'db>,
) -> RuntimeSectionRef<'db> {
    let (old_object, section, is_local) = match section_ref {
        RuntimeSectionRef::Local { object, section } => (object, section, true),
        RuntimeSectionRef::External { object, section } => (object, section, false),
    };
    let object = objects
        .iter()
        .find(|candidate| candidate.name(db) == old_object.name(db))
        .copied()
        .unwrap_or_else(|| {
            panic!(
                "missing rewritten runtime object `{}` while remapping section ref",
                old_object.name(db)
            )
        });
    if is_local {
        RuntimeSectionRef::Local { object, section }
    } else {
        RuntimeSectionRef::External { object, section }
    }
}

fn resolve_code_regions<'db>(
    db: &'db dyn MirDb,
    objects: &[RuntimeObject<'db>],
    functions_by_instance: &FxHashMap<RuntimeInstance<'db>, RuntimeFunction<'db>>,
    function_roots: &[(RuntimeCodeRegion<'db>, RuntimeInstance<'db>)],
) -> Vec<ResolvedCodeRegion<'db>> {
    let mut resolved = Vec::new();
    for object in objects {
        for section in object.sections(db) {
            let Some((region, symbol)) = resolved_section_region(db, &section) else {
                continue;
            };
            resolved.push(make_resolved_code_region(
                db,
                region,
                symbol,
                RuntimeSectionRef::Local {
                    object: *object,
                    section: section.name.clone(),
                },
                section.entry,
            ));
        }
    }

    if let Some(code_regions_object) = objects
        .iter()
        .find(|object| object.name(db) == "CodeRegions")
    {
        for (region, root_instance) in function_roots {
            let RuntimeCodeRegionKey::FunctionRoot { symbol, .. } = region.key(db).clone() else {
                continue;
            };
            let Some(_section) = code_regions_object
                .sections(db)
                .iter()
                .find(|section| section.name == RuntimeSectionName::CodeRegion(symbol.clone()))
            else {
                continue;
            };
            resolved.push(make_resolved_code_region(
                db,
                *region,
                symbol.clone(),
                RuntimeSectionRef::Local {
                    object: *code_regions_object,
                    section: RuntimeSectionName::CodeRegion(symbol),
                },
                *functions_by_instance
                    .get(root_instance)
                    .expect("code-region root should be declared as a runtime function"),
            ));
        }
    }

    resolved.sort_by_key(|region| region.symbol(db).clone());
    resolved
}

fn resolved_section_region<'db>(
    db: &'db dyn MirDb,
    section: &RuntimeSection<'db>,
) -> Option<(RuntimeCodeRegion<'db>, String)> {
    match section.entry.owner(db) {
        RuntimeFunctionOwner::Synthetic(
            RuntimeSyntheticSpec::ContractInitRoot { contract, .. }
            | RuntimeSyntheticSpec::ContractRuntimeRoot { contract, .. },
        ) => match section.name {
            RuntimeSectionName::Init => Some((
                RuntimeCodeRegion::new(db, RuntimeCodeRegionKey::ContractInit { contract }),
                format!("{}_init", contract_name(db, contract)),
            )),
            RuntimeSectionName::Runtime => Some((
                RuntimeCodeRegion::new(db, RuntimeCodeRegionKey::ContractRuntime { contract }),
                format!("{}_runtime", contract_name(db, contract)),
            )),
            RuntimeSectionName::Main
            | RuntimeSectionName::Test(_)
            | RuntimeSectionName::CodeRegion(_) => None,
        },
        RuntimeFunctionOwner::Semantic(semantic) => {
            let BodyOwner::Func(func) = semantic.key(db).owner(db) else {
                return None;
            };
            let region = runtime_code_region_for_manual_root(db, func)?;
            Some((region, code_region_symbol(db, region)))
        }
        RuntimeFunctionOwner::Synthetic(
            RuntimeSyntheticSpec::MainRoot { .. }
            | RuntimeSyntheticSpec::TestRoot { .. }
            | RuntimeSyntheticSpec::ContractInitAbi { .. }
            | RuntimeSyntheticSpec::ContractRecvAbi { .. }
            | RuntimeSyntheticSpec::CodeRegionRoot { .. },
        ) => None,
    }
}

fn collect_region_embeds<'db>(
    db: &'db dyn MirDb,
    reachable: &FxHashSet<RuntimeInstance<'db>>,
    functions_by_instance: &FxHashMap<RuntimeInstance<'db>, RuntimeFunction<'db>>,
    section_refs: &FxHashMap<RuntimeCodeRegion<'db>, RuntimeSectionRef<'db>>,
    current_section: RuntimeSectionRef<'db>,
) -> Vec<crate::runtime::RuntimeEmbed<'db>> {
    let current_object = match &current_section {
        RuntimeSectionRef::Local { object, .. } | RuntimeSectionRef::External { object, .. } => {
            *object
        }
    };
    let mut seen = FxHashSet::default();
    let mut embeds = Vec::new();
    for instance in reachable {
        let Some(function) = functions_by_instance.get(instance) else {
            continue;
        };
        for region in function
            .instance(db)
            .referenced_code_regions(db)
            .iter()
            .copied()
        {
            let Some(source) = section_refs.get(&region) else {
                continue;
            };
            if *source == current_section || !seen.insert(region) {
                continue;
            }
            let source = match source {
                RuntimeSectionRef::Local { object, section }
                | RuntimeSectionRef::External { object, section }
                    if *object == current_object =>
                {
                    RuntimeSectionRef::Local {
                        object: *object,
                        section: section.clone(),
                    }
                }
                RuntimeSectionRef::Local { object, section }
                | RuntimeSectionRef::External { object, section } => RuntimeSectionRef::External {
                    object: *object,
                    section: section.clone(),
                },
            };
            embeds.push(crate::runtime::RuntimeEmbed {
                source,
                as_symbol: code_region_symbol(db, region),
            });
        }
    }
    embeds.sort_by(|lhs, rhs| lhs.as_symbol.cmp(&rhs.as_symbol));
    embeds
}

fn collect_reachable_from_entry<'db>(
    db: &'db dyn MirDb,
    entry: RuntimeInstance<'db>,
) -> FxHashSet<RuntimeInstance<'db>> {
    let mut seen = FxHashSet::default();
    let mut stack = vec![entry];
    while let Some(instance) = stack.pop() {
        if !seen.insert(instance) {
            continue;
        }
        for call in instance.calls(db) {
            stack.push(call.callee);
        }
    }
    seen
}

fn collect_const_regions_for_reachable<'db>(
    db: &'db dyn MirDb,
    functions_by_instance: &FxHashMap<RuntimeInstance<'db>, RuntimeFunction<'db>>,
    reachable: &FxHashSet<RuntimeInstance<'db>>,
) -> Vec<ConstRegionId<'db>> {
    let mut seen = FxHashSet::default();
    let mut regions = Vec::new();
    for instance in reachable {
        let Some(function) = functions_by_instance.get(instance) else {
            continue;
        };
        for region in function.referenced_const_regions(db) {
            if seen.insert(region) {
                regions.push(region);
            }
        }
    }
    regions
}

fn collect_function_root_regions<'db>(
    db: &'db dyn MirDb,
    functions: &[RuntimeFunction<'db>],
) -> Vec<RuntimeCodeRegion<'db>> {
    let mut seen = FxHashSet::default();
    let mut regions = Vec::new();
    for function in functions {
        for region in function
            .instance(db)
            .referenced_code_regions(db)
            .iter()
            .copied()
        {
            if seen.insert(region) {
                regions.push(region);
            }
        }
    }
    regions
}

fn materialized_contracts_for_roots<'db>(
    db: &'db dyn MirDb,
    roots: &[RuntimeInstance<'db>],
) -> FxHashSet<Contract<'db>> {
    roots
        .iter()
        .filter_map(|root| match root.key(db).source(db) {
            RuntimeInstanceSource::Synthetic(synthetic) => match synthetic.spec(db) {
                RuntimeSyntheticSpec::ContractInitRoot { contract, .. }
                | RuntimeSyntheticSpec::ContractRuntimeRoot { contract, .. } => Some(contract),
                RuntimeSyntheticSpec::MainRoot { .. }
                | RuntimeSyntheticSpec::TestRoot { .. }
                | RuntimeSyntheticSpec::ContractInitAbi { .. }
                | RuntimeSyntheticSpec::ContractRecvAbi { .. }
                | RuntimeSyntheticSpec::CodeRegionRoot { .. } => None,
            },
            RuntimeInstanceSource::Semantic(_) => None,
        })
        .collect()
}

fn semantic_instance_for_root_owner<'db>(
    db: &'db dyn MirDb,
    owner: BodyOwner<'db>,
) -> Result<SemanticInstance<'db>, LowerError> {
    let key = root_semantic_instance_key(db, owner).map_err(|err| match err {
        RootSemanticInstanceError::UnsupportedGenericParam {
            owner,
            owner_scope,
            offending_ty,
            param_idx,
        } => LowerError::Unsupported(format!(
            "root semantic instance for {owner:?} has unsupported generic param {param_idx} in {owner_scope:?}: {}",
            offending_ty.pretty_print(db),
        )),
        RootSemanticInstanceError::MissingRootProvider { owner } => LowerError::Unsupported(
            format!("root semantic instance for {owner:?} is missing a root provider binding"),
        ),
        RootSemanticInstanceError::UnclosedEffectEnv(err) => LowerError::Unsupported(format!(
            "root semantic instance for {:?} is not closed under synthesized root substitution: owner_scope={:?} param_idx={} args_len={} offending_ty={}",
            err.owner,
            err.owner_scope,
            err.param_idx,
            err.args_len,
            err.offending_ty.pretty_print(db),
        )),
    })?;
    Ok(get_or_build_semantic_instance(db, key))
}

fn contract_object_spec<'db>(
    db: &'db dyn MirDb,
    contract: Contract<'db>,
) -> Result<ManualContractObjectSpec<'db>, LowerError> {
    let runtime_root = contract_runtime_root(db, contract)?;
    let init_root = contract_init_root(db, contract)?;
    Ok((
        sanitize_object_name(&contract_name(db, contract)),
        vec![
            (RuntimeSectionName::Init, init_root),
            (RuntimeSectionName::Runtime, runtime_root),
        ],
        vec![init_root, runtime_root],
    ))
}

fn is_test_func<'db>(db: &'db dyn MirDb, func: Func<'db>) -> bool {
    ItemKind::from(func)
        .attrs(db)
        .is_some_and(|attrs| attrs.get_attr(db, "test").is_some())
}

fn runtime_root_candidate<'db>(db: &'db dyn MirDb, func: Func<'db>) -> Result<bool, LowerError> {
    if func.is_associated_func(db) || func.params(db).next().is_some() {
        return Ok(false);
    }
    let semantic = match semantic_instance_for_root_owner(db, BodyOwner::Func(func)) {
        Ok(semantic) => semantic,
        Err(LowerError::Unsupported(_)) => return Ok(false),
    };
    Ok(owner_effect_bindings(db, BodyOwner::Func(func))
        .into_iter()
        .all(|binding| owner_effect_binding_class(db, semantic, binding).is_some()))
}

pub(crate) fn runtime_instance_for_semantic<'db>(
    db: &'db dyn MirDb,
    semantic: SemanticInstance<'db>,
) -> RuntimeInstance<'db> {
    let typed_body = semantic.key(db).instantiate_typed_body(db);
    let owner = semantic.key(db).owner(db);
    if let BodyOwner::Func(func) = owner
        && func.body(db).is_none()
    {
        panic!(
            "bodyless semantic function leaked into runtime instance construction: func={func:?} key={:?}",
            semantic.key(db)
        );
    }
    let env = RuntimeTypeEnv::new(
        Some(owner.scope()),
        semantic_instance_assumptions(db, semantic),
    );
    let mut params = Vec::new();
    let mut idx = 0;
    while let Some(binding) = typed_body.param_binding(idx) {
        if let Some(class) = runtime_visible_binding_class(db, semantic, binding)
            .map(|class| runtime_param_class(db, &typed_body, binding, class))
        {
            params.push(class);
        }
        idx += 1;
    }
    if let BodyOwner::ContractRecvArm {
        contract,
        recv_idx,
        arm_idx,
    } = owner
    {
        let recv = hir::semantic::RecvView::new(db, contract, recv_idx);
        let arm = RecvArmView::new(db, recv, arm_idx);
        for arg_binding in arm.arg_bindings(db) {
            let Some(binding) = typed_body.pat_binding(arg_binding.pat) else {
                continue;
            };
            let ty = semantic_binding_ty(db, semantic, binding);
            if let Some(class) =
                top_level_class_for_ty_in_env(db, env, ty, AddressSpaceKind::Memory)
            {
                params.push(class);
            }
        }
    }
    for binding in owner_effect_bindings(db, owner) {
        if let Some(class) = owner_effect_binding_class(db, semantic, binding) {
            params.push(class);
        }
    }
    let key = RuntimeInstanceKey::new(db, RuntimeInstanceSource::Semantic(semantic), params);
    get_or_build_runtime_instance(db, key)
}

fn owner_effect_binding_class<'db>(
    db: &'db dyn MirDb,
    semantic: SemanticInstance<'db>,
    binding: hir::analysis::ty::ty_check::LocalBinding<'db>,
) -> Option<crate::runtime::RuntimeClass<'db>> {
    let owner = semantic.key(db).owner(db);
    let env = RuntimeTypeEnv::new(
        Some(owner.scope()),
        semantic_instance_assumptions(db, semantic),
    );
    let binding_ty = semantic_binding_ty(db, semantic, binding);
    match semantic_binding_lowering(db, semantic, binding) {
        hir::analysis::semantic::SemanticBindingLowering::Erased => None,
        hir::analysis::semantic::SemanticBindingLowering::DirectValue {
            provenance: ValueProvenance::RootProvider(provider),
        } => runtime_class_for_direct_value_provider_in_env(db, env, &provider),
        hir::analysis::semantic::SemanticBindingLowering::DirectValue { .. }
        | hir::analysis::semantic::SemanticBindingLowering::DirectCarrier { .. } => {
            top_level_class_for_ty_in_env(db, env, binding_ty, AddressSpaceKind::Memory)
        }
        hir::analysis::semantic::SemanticBindingLowering::PlaceCarrier { value_ty } => {
            top_level_class_for_ty_in_env(db, env, value_ty, AddressSpaceKind::Memory)
        }
        hir::analysis::semantic::SemanticBindingLowering::PlaceBoundValue {
            provenance: hir::analysis::semantic::PlaceProvenance::RootProvider(provider),
            ..
        } => runtime_class_for_effect_binding_provider_in_env(db, env, &provider),
        hir::analysis::semantic::SemanticBindingLowering::PlaceBoundValue {
            provenance: hir::analysis::semantic::PlaceProvenance::Derived { .. },
            ..
        } => None,
    }
}

fn synthetic_instance<'db>(
    db: &'db dyn MirDb,
    spec: RuntimeSyntheticSpec<'db>,
    params: Vec<crate::runtime::RuntimeClass<'db>>,
) -> RuntimeInstance<'db> {
    let synthetic = RuntimeSyntheticInstance::new(db, spec);
    let key = RuntimeInstanceKey::new(db, RuntimeInstanceSource::Synthetic(synthetic), params);
    get_or_build_runtime_instance(db, key)
}

fn contract_effect_arg_plans_for_site<'db>(
    db: &'db dyn MirDb,
    contract: Contract<'db>,
    semantic: SemanticInstance<'db>,
) -> Result<Vec<ContractEffectArgPlan<'db>>, LowerError> {
    let field_layout = contract.field_layout(db);
    let field_by_index = field_layout
        .values()
        .cloned()
        .map(|field| (field.index, field))
        .collect::<FxHashMap<_, _>>();
    owner_effect_bindings(db, semantic.key(db).owner(db))
        .into_iter()
        .filter_map(|binding| {
            let provider = resolved_provider_binding_for_instance_effect(db, semantic, binding)?;
            Some((|| {
                Ok(match provider.source.clone() {
                    ProviderSource::ContractField { field_idx, .. } => {
                        let field = field_by_index.get(&field_idx).ok_or_else(|| {
                            LowerError::Unsupported(format!(
                                "missing contract field layout for field {field_idx} in `{}`",
                                contract_name(db, contract)
                            ))
                        })?;
                        ContractEffectArgPlan::ContractField(contract_field_binding(
                            db, contract, field,
                        )?)
                    }
                    ProviderSource::UsesParam { .. } | ProviderSource::RootProvider { .. } => {
                        let Some(boundary) = owner_effect_binding_boundary(db, semantic, binding)
                        else {
                            return Err(LowerError::Unsupported(format!(
                                "missing runtime boundary for owner effect binding {binding:?} in `{}`",
                                contract_name(db, contract)
                            )));
                        };
                        ContractEffectArgPlan::Placeholder {
                            declared_ty: semantic_binding_ty(db, semantic, binding),
                            boundary,
                        }
                    }
                })
            })())
        })
        .collect()
}

pub(crate) fn contract_field_binding<'db>(
    db: &'db dyn MirDb,
    contract: Contract<'db>,
    field: &ContractFieldLayoutInfo<'db>,
) -> Result<ContractFieldBinding<'db>, LowerError> {
    let space = address_space_from_ty(db, contract.scope(), field.address_space)?;
    let class = provider_class_for_target_in_env(
        db,
        RuntimeTypeEnv::new(
            Some(contract.scope()),
            hir::analysis::ty::trait_resolution::PredicateListId::empty_list(db),
        ),
        Some(field.target_ty),
        space,
    );
    let kind = match &class {
        crate::runtime::RuntimeClass::Ref { kind, .. } => kind.clone(),
        crate::runtime::RuntimeClass::RawAddr { .. } => RefKind::Provider {
            provider_ty: TyId::borrow_ref_of(db, field.target_ty),
            space,
        },
        crate::runtime::RuntimeClass::Scalar(_)
        | crate::runtime::RuntimeClass::AggregateValue { .. } => {
            return Err(LowerError::Unsupported(format!(
                "contract field `{}` in `{}` does not lower to a provider-style runtime class",
                field.name.data(db),
                contract_name(db, contract)
            )));
        }
    };
    Ok(ContractFieldBinding {
        slot: field.slot_offset as u128,
        declared_ty: field.declared_ty,
        class,
        kind,
    })
}

fn resolve_decode_instance<'db>(
    db: &'db dyn MirDb,
    scope: hir::hir_def::scope_graph::ScopeId<'db>,
    ty: TyId<'db>,
) -> Result<RuntimeInstance<'db>, LowerError> {
    let abi_ty = sol_abi_ty(db, scope)?;
    let decoder_ty = sol_decoder_ty(db, scope)?;
    let decode_trait = resolve_core_trait(db, scope, &["abi", "Decode"])
        .ok_or_else(|| LowerError::Unsupported("missing required core::abi::Decode".to_string()))?;
    let inst = TraitInstId::new_simple(db, decode_trait, vec![ty, abi_ty]);
    resolve_trait_runtime_instance(db, scope, inst, "decode", vec![decoder_ty])
}

fn resolve_encode_instance<'db>(
    db: &'db dyn MirDb,
    scope: hir::hir_def::scope_graph::ScopeId<'db>,
    ty: TyId<'db>,
) -> Result<RuntimeInstance<'db>, LowerError> {
    let abi_ty = sol_abi_ty(db, scope)?;
    let encoder_ty = sol_encoder_ty(db, scope)?;
    let encode_trait = resolve_core_trait(db, scope, &["abi", "Encode"])
        .ok_or_else(|| LowerError::Unsupported("missing required core::abi::Encode".to_string()))?;
    let inst = TraitInstId::new_simple(db, encode_trait, vec![ty, abi_ty]);
    resolve_trait_runtime_instance(db, scope, inst, "encode", vec![encoder_ty])
}

fn resolve_trait_runtime_instance<'db>(
    db: &'db dyn MirDb,
    scope: hir::hir_def::scope_graph::ScopeId<'db>,
    inst: TraitInstId<'db>,
    method: &str,
    extra_generic_args: Vec<TyId<'db>>,
) -> Result<RuntimeInstance<'db>, LowerError> {
    let assumptions = hir::analysis::ty::trait_resolution::PredicateListId::empty_list(db);
    let method = IdentId::new(db, method.to_string());
    let (func, mut impl_args) = resolve_trait_method_instance(
        db,
        TraitSolveCx::new(db, scope).with_assumptions(assumptions),
        inst,
        method,
    )
    .ok_or_else(|| {
        LowerError::Unsupported(format!(
            "failed to resolve trait method `{}` for runtime package planning",
            method.data(db)
        ))
    })?;
    impl_args.extend(extra_generic_args);
    let key = SemanticInstanceKey::new(
        db,
        BodyOwner::Func(func),
        GenericSubst::new(db, impl_args),
        hir::analysis::semantic::EffectProviderSubst::empty(db),
        ImplEnv::new(db, scope, assumptions, vec![inst]),
    );
    Ok(runtime_instance_for_semantic(
        db,
        get_or_build_semantic_instance(db, key),
    ))
}

fn address_space_from_ty<'db>(
    db: &'db dyn MirDb,
    scope: hir::hir_def::scope_graph::ScopeId<'db>,
    ty: TyId<'db>,
) -> Result<AddressSpaceKind, LowerError> {
    let memory = resolve_lib_type_path(db, scope, "core::effect_ref::Memory")
        .ok_or_else(|| LowerError::Unsupported("missing Memory address space type".to_string()))?;
    let storage = resolve_lib_type_path(db, scope, "core::effect_ref::Storage")
        .ok_or_else(|| LowerError::Unsupported("missing Storage address space type".to_string()))?;
    let transient = resolve_lib_type_path(db, scope, "core::effect_ref::TransientStorage")
        .ok_or_else(|| {
            LowerError::Unsupported("missing TransientStorage address space type".to_string())
        })?;
    let calldata =
        resolve_lib_type_path(db, scope, "core::effect_ref::Calldata").ok_or_else(|| {
            LowerError::Unsupported("missing Calldata address space type".to_string())
        })?;
    if ty == memory {
        Ok(AddressSpaceKind::Memory)
    } else if ty == storage {
        Ok(AddressSpaceKind::Storage)
    } else if ty == transient {
        Ok(AddressSpaceKind::Transient)
    } else if ty == calldata {
        Ok(AddressSpaceKind::Calldata)
    } else {
        Err(LowerError::Unsupported(format!(
            "unsupported contract field address space `{:?}`",
            ty
        )))
    }
}

fn sol_abi_ty<'db>(
    db: &'db dyn MirDb,
    scope: hir::hir_def::scope_graph::ScopeId<'db>,
) -> Result<TyId<'db>, LowerError> {
    resolve_lib_type_path(db, scope, "std::abi::Sol")
        .ok_or_else(|| LowerError::Unsupported("missing std::abi::Sol".to_string()))
}

fn memory_bytes_ty<'db>(
    db: &'db dyn MirDb,
    scope: hir::hir_def::scope_graph::ScopeId<'db>,
) -> Result<TyId<'db>, LowerError> {
    resolve_lib_type_path(db, scope, "std::evm::memory_input::MemoryBytes").ok_or_else(|| {
        LowerError::Unsupported("missing std::evm::memory_input::MemoryBytes".to_string())
    })
}

fn sol_decoder_ty<'db>(
    db: &'db dyn MirDb,
    scope: hir::hir_def::scope_graph::ScopeId<'db>,
) -> Result<TyId<'db>, LowerError> {
    let ctor = resolve_lib_type_path(db, scope, "std::abi::sol::SolDecoder")
        .ok_or_else(|| LowerError::Unsupported("missing std::abi::sol::SolDecoder".to_string()))?;
    Ok(TyId::app(db, ctor, memory_bytes_ty(db, scope)?))
}

fn sol_encoder_ty<'db>(
    db: &'db dyn MirDb,
    scope: hir::hir_def::scope_graph::ScopeId<'db>,
) -> Result<TyId<'db>, LowerError> {
    resolve_lib_type_path(db, scope, "std::abi::sol::SolEncoder")
        .ok_or_else(|| LowerError::Unsupported("missing std::abi::sol::SolEncoder".to_string()))
}

fn make_runtime_function<'db>(
    db: &'db dyn MirDb,
    instance: RuntimeInstance<'db>,
    symbol: String,
    linkage: RuntimeLinkage,
    inline_hint: RuntimeInlineHint,
    owner: RuntimeFunctionOwner<'db>,
    referenced_const_regions: Vec<ConstRegionId<'db>>,
) -> RuntimeFunction<'db> {
    RuntimeFunction::new(
        db,
        instance,
        symbol,
        linkage,
        inline_hint,
        owner,
        referenced_const_regions,
    )
}

fn make_runtime_object<'db>(
    db: &'db dyn MirDb,
    name: String,
    sections: Vec<RuntimeSection<'db>>,
) -> RuntimeObject<'db> {
    RuntimeObject::new(db, name, sections)
}

fn make_resolved_code_region<'db>(
    db: &'db dyn MirDb,
    region: RuntimeCodeRegion<'db>,
    symbol: String,
    source: RuntimeSectionRef<'db>,
    root: RuntimeFunction<'db>,
) -> ResolvedCodeRegion<'db> {
    ResolvedCodeRegion::new(db, region, symbol, source, root)
}

fn collect_runtime_functions<'db>(
    db: &'db dyn MirDb,
    roots: &[RuntimeInstance<'db>],
) -> Vec<RuntimeFunction<'db>> {
    let mut seen = FxHashSet::default();
    let mut stack = roots.to_vec();
    let mut instances = Vec::new();

    while let Some(instance) = stack.pop() {
        if !seen.insert(instance) {
            continue;
        }
        for call in instance.calls(db) {
            stack.push(call.callee);
        }
        instances.push(instance);
    }

    instances.sort_by_key(|instance| runtime_instance_sort_key(db, *instance));
    let duplicate_counts = instances
        .iter()
        .fold(FxHashMap::default(), |mut counts, instance| {
            *counts
                .entry(runtime_instance_symbol_base(db, *instance))
                .or_insert(0usize) += 1;
            counts
        });
    let mut emitted_counts = FxHashMap::<String, usize>::default();
    let mut functions = instances
        .into_iter()
        .map(|instance| {
            let base = runtime_instance_symbol_base(db, instance);
            let symbol = match duplicate_counts.get(&base).copied().unwrap_or_default() {
                0 | 1 => base,
                _ => {
                    let ordinal = emitted_counts.entry(base.clone()).or_insert(0);
                    let symbol = format!("{base}_{ordinal}");
                    *ordinal += 1;
                    symbol
                }
            };
            runtime_function_for_instance(db, instance, symbol)
        })
        .collect::<Vec<_>>();
    functions.sort_by_key(|function| function.symbol(db));
    functions
}

fn runtime_function_for_instance<'db>(
    db: &'db dyn MirDb,
    instance: RuntimeInstance<'db>,
    symbol: String,
) -> RuntimeFunction<'db> {
    match instance.key(db).source(db) {
        RuntimeInstanceSource::Semantic(semantic) => make_runtime_function(
            db,
            instance,
            symbol,
            RuntimeLinkage::Private,
            inline_hint_for_semantic(db, semantic),
            RuntimeFunctionOwner::Semantic(semantic),
            instance.referenced_const_regions(db).to_vec(),
        ),
        RuntimeInstanceSource::Synthetic(synthetic) => {
            let spec = synthetic.spec(db).clone();
            make_runtime_function(
                db,
                instance,
                symbol,
                RuntimeLinkage::Private,
                RuntimeInlineHint::Auto,
                RuntimeFunctionOwner::Synthetic(spec),
                instance.referenced_const_regions(db).to_vec(),
            )
        }
    }
}

fn runtime_instance_sort_key<'db>(db: &'db dyn MirDb, instance: RuntimeInstance<'db>) -> String {
    match instance.key(db).source(db) {
        RuntimeInstanceSource::Semantic(_) => {
            format!("semantic:{}", runtime_instance_symbol_base(db, instance))
        }
        RuntimeInstanceSource::Synthetic(synthetic) => match synthetic.spec(db).clone() {
            RuntimeSyntheticSpec::MainRoot { .. } => "__synthetic:main_root".to_string(),
            RuntimeSyntheticSpec::TestRoot { name, .. } => {
                format!("__synthetic:test_root:{name}")
            }
            RuntimeSyntheticSpec::ContractInitAbi { plan } => format!(
                "__synthetic:contract_init_abi:{}",
                contract_name(db, plan.contract)
            ),
            RuntimeSyntheticSpec::ContractRecvAbi { plan } => format!(
                "__synthetic:contract_recv_abi:{}:{}",
                contract_name(db, plan.contract),
                plan.selector
                    .map_or_else(|| "fallback".to_string(), |selector| selector.to_string())
            ),
            RuntimeSyntheticSpec::ContractInitRoot { contract, .. } => format!(
                "__synthetic:contract_init_root:{}",
                contract_name(db, contract)
            ),
            RuntimeSyntheticSpec::ContractRuntimeRoot { contract, .. } => format!(
                "__synthetic:contract_runtime_root:{}",
                contract_name(db, contract)
            ),
            RuntimeSyntheticSpec::CodeRegionRoot { symbol, .. } => {
                format!("__synthetic:code_region_root:{symbol}")
            }
        },
    }
}

fn runtime_instance_symbol_base<'db>(db: &'db dyn MirDb, instance: RuntimeInstance<'db>) -> String {
    match instance.key(db).source(db) {
        RuntimeInstanceSource::Semantic(semantic) => {
            symbol_base_for_semantic_instance(db, semantic)
        }
        RuntimeInstanceSource::Synthetic(synthetic) => {
            symbol_base_for_runtime_instance(db, &synthetic.spec(db))
        }
    }
}

fn symbol_base_for_semantic_instance<'db>(
    db: &'db dyn MirDb,
    semantic: SemanticInstance<'db>,
) -> String {
    let owner = semantic.key(db).owner(db);
    match owner {
        BodyOwner::Func(func) => func
            .name(db)
            .to_opt()
            .map(|name| name.data(db).to_string())
            .unwrap_or_else(|| "__anon".to_string()),
        BodyOwner::ContractInit { contract } => format!(
            "__{}_init",
            contract
                .name(db)
                .to_opt()
                .map(|name| name.data(db).to_string())
                .unwrap_or_else(|| "contract".to_string())
        ),
        BodyOwner::ContractRecvArm {
            contract,
            recv_idx,
            arm_idx,
        } => format!(
            "__{}_recv_{}_{}",
            contract
                .name(db)
                .to_opt()
                .map(|name| name.data(db).to_string())
                .unwrap_or_else(|| "contract".to_string()),
            recv_idx,
            arm_idx
        ),
        BodyOwner::Const(_) | BodyOwner::AnonConstBody { .. } => "__const".to_string(),
    }
}

fn symbol_base_for_runtime_instance<'db>(
    db: &'db dyn MirDb,
    spec: &RuntimeSyntheticSpec<'db>,
) -> String {
    match spec {
        RuntimeSyntheticSpec::MainRoot { .. } => "__fe_main_root".to_string(),
        RuntimeSyntheticSpec::TestRoot { name, .. } => {
            format!("__fe_test_root_{}", sanitize_symbol(name))
        }
        RuntimeSyntheticSpec::ContractInitAbi { plan } => {
            format!(
                "__fe_contract_init_abi_{}",
                contract_name(db, plan.contract)
            )
        }
        RuntimeSyntheticSpec::ContractRecvAbi { plan } => format!(
            "__fe_contract_recv_abi_{}_{}",
            contract_name(db, plan.contract),
            plan.selector
                .map_or_else(|| "fallback".to_string(), |selector| selector.to_string()),
        ),
        RuntimeSyntheticSpec::ContractInitRoot { contract, .. } => {
            format!("__fe_contract_init_root_{}", contract_name(db, *contract))
        }
        RuntimeSyntheticSpec::ContractRuntimeRoot { contract, .. } => {
            format!(
                "__fe_contract_runtime_root_{}",
                contract_name(db, *contract)
            )
        }
        RuntimeSyntheticSpec::CodeRegionRoot { symbol, .. } => {
            format!("__fe_code_region_root_{}", sanitize_symbol(symbol))
        }
    }
}

fn inline_hint_for_semantic<'db>(
    db: &'db dyn MirDb,
    semantic: SemanticInstance<'db>,
) -> RuntimeInlineHint {
    match semantic.key(db).owner(db) {
        BodyOwner::Func(func) => match func.inline_hint(db) {
            Some(InlineHint::Hint) => RuntimeInlineHint::Hint,
            Some(InlineHint::Always) => RuntimeInlineHint::Always,
            Some(InlineHint::Never) => RuntimeInlineHint::Never,
            None => RuntimeInlineHint::Auto,
        },
        BodyOwner::Const(_)
        | BodyOwner::AnonConstBody { .. }
        | BodyOwner::ContractInit { .. }
        | BodyOwner::ContractRecvArm { .. } => RuntimeInlineHint::Auto,
    }
}

fn collect_const_regions<'db>(
    db: &'db dyn MirDb,
    functions: &[RuntimeFunction<'db>],
) -> Vec<ConstRegionId<'db>> {
    let mut seen = FxHashSet::default();
    let mut regions = Vec::new();
    for function in functions {
        for region in function.referenced_const_regions(db) {
            if seen.insert(region) {
                regions.push(region);
            }
        }
    }
    regions
}

fn contract_name<'db>(db: &'db dyn MirDb, contract: hir::hir_def::Contract<'db>) -> String {
    contract
        .name(db)
        .to_opt()
        .map(|name| sanitize_symbol(name.data(db)))
        .unwrap_or_else(|| "contract".to_string())
}

fn sanitize_symbol(value: &str) -> String {
    value
        .chars()
        .map(|ch| if ch.is_ascii_alphanumeric() { ch } else { '_' })
        .collect()
}

fn sanitize_object_name(value: &str) -> String {
    let sanitized = sanitize_symbol(value);
    if sanitized.is_empty() {
        "object".to_string()
    } else {
        sanitized
    }
}
