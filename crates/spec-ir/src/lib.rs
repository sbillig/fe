//! Serializes Fe's normalized semantic IR for fe-spec's Lean interpreter
//! ([`export_ir`]), with what the serializer and the Lean exporter
//! (`fe-lean-export`) both need of a contract: its entry code's callees and
//! the names of its arms and instances.

mod ir;
mod key;

use common::{
    diagnostics::Severity,
    ingot::{Ingot, IngotKind},
};
use hir::{
    analysis::{
        HirAnalysisDb,
        analysis_pass::AnalysisPassManager,
        diagnostics::SpannedHirAnalysisDb,
        initialize_analysis_pass,
        semantic::{SemanticAccessAnalysisPass, SemanticInstanceKey, generated_callee_key},
        ty::{
            binder::Binder,
            corelib::{resolve_core_trait, resolve_lib_func_path, resolve_lib_type_path},
            registered_root_providers,
            trait_def::TraitInstId,
            trait_resolution::PredicateListId,
            ty_check::{BodyOwner, EffectParamSite, LocalBinding},
            ty_def::TyId,
            ty_lower::{CompleteSubst, ParamSchemaId},
        },
    },
    hir_def::{
        Body, Contract, GenericParamOwner, HirIngot, IdentId, ItemKind, PatId, scope_graph::ScopeId,
    },
    semantic::{RecvArmAbiInfo, RecvArmView},
};

pub use ir::{IR_VERSION, SpecIr, export_ir, ir_text, merge_library};
pub use key::{ingot_name, ingot_namespace};

/// Why a contract is not exported: a construct the exporter has no model
/// for, or the errors the compiler reports in its ingot.
#[derive(Debug, PartialEq, Eq)]
pub struct Unsupported(pub String);

/// What the definition of `key` defines, as Fe names it: the function
/// or constant it is an instance of, a method or an associated constant
/// by the type its impl is for at the instance's types, and the
/// instance's other types.
pub fn describe<'db>(db: &'db dyn HirAnalysisDb, key: SemanticInstanceKey<'db>) -> String {
    let owner = key.owner(db);
    let args = key.subst(db).generic_args(db);
    // The item the instance's types are for, its scope and its name.
    let (generic, scope, name) = match owner {
        BodyOwner::Func(func, _) => {
            let name = func.name(db).to_opt().map(|name| name.data(db).as_str());
            (Some(GenericParamOwner::Func(func)), func.scope(), name)
        }
        BodyOwner::AnonConstBody { body, .. } => {
            let generic = match body.scope().parent_item(db) {
                Some(ItemKind::Impl(item)) => Some(GenericParamOwner::Impl(item)),
                Some(ItemKind::ImplTrait(item)) => Some(GenericParamOwner::ImplTrait(item)),
                _ => None,
            };
            (generic, body.scope(), assoc_const_name(db, body))
        }
        BodyOwner::Closure { .. } => {
            return format!("A closure in `{}`", fe_path(db, owner.scope()));
        }
        owner => return format!("`{}`", const_path(db, owner)),
    };
    let types: Vec<_> = args
        .iter()
        .map(|arg| arg.pretty_print(db).as_str())
        .collect();
    let Some(name) = name else {
        let block = format!("A `const` block in `{}`", fe_path(db, scope));
        return match types.as_slice() {
            [] => block,
            types => format!("{block}, at `<{}>`", types.join(", ")),
        };
    };
    // The type an impl is for, at the instance's types, and the number
    // of those the impl takes.
    let instantiated = |ty: TyId<'db>, parent: GenericParamOwner<'db>| {
        let owner = generic.unwrap_or(parent);
        let subst = CompleteSubst::for_owner(db, owner, args.to_vec()).ok();
        let ty = subst
            .and_then(|subst| Binder::bind(owner, ty).instantiate_subst(db, &subst).ok())
            .unwrap_or(ty);
        let taken = ParamSchemaId::full(db, parent).keys(db).len();
        (ty.pretty_print(db).to_string(), taken)
    };
    let (path, taken) = match scope.parent_item(db) {
        Some(ItemKind::Impl(item)) => {
            let (ty, taken) = instantiated(item.ty(db), GenericParamOwner::Impl(item));
            (format!("{ty}::{name}"), taken)
        }
        Some(ItemKind::ImplTrait(item)) => {
            let (ty, taken) = instantiated(item.ty(db), GenericParamOwner::ImplTrait(item));
            let trait_name = item.trait_def(db).and_then(|def| def.name(db).to_opt());
            let trait_name = trait_name.map_or("?", |name| name.data(db));
            (format!("<{ty} as {trait_name}>::{name}"), taken)
        }
        _ => (fe_path(db, scope), 0),
    };
    let own = match types.get(taken..).unwrap_or_default() {
        [] => String::new(),
        own => format!("<{}>", own.join(", ")),
    };
    match owner {
        BodyOwner::Func(_, Some(mode)) => {
            let mode = format!("{mode:?}").to_lowercase();
            format!("`{path}{own}`, its `{mode}` body")
        }
        _ => format!("`{path}{own}`"),
    }
}

/// What the compiler's entry code calls around one recv arm: the root
/// provider the arm runs under, the arm's ABI information, the instance of
/// `core::contracts::decode_runtime_args` that decodes its arguments, if it
/// takes any, and the instance of `ContractHost::return_value` that returns
/// its result, if it has one.
pub struct ArmCallees<'db> {
    pub host: TyId<'db>,
    pub info: RecvArmAbiInfo<'db>,
    pub decode: Option<SemanticInstanceKey<'db>>,
    pub return_value: Option<SemanticInstanceKey<'db>>,
}

/// The callees of the entry code of recv arm `arm` of `contract`.
pub fn arm_callees<'db>(
    db: &'db dyn HirAnalysisDb,
    contract: Contract<'db>,
    arm: RecvArmView<'db>,
) -> Result<ArmCallees<'db>, Unsupported> {
    let scope = contract.scope();
    let missing = |what: &str| Unsupported(format!("runtime: missing `{what}`"));
    let generated = |error| Unsupported(format!("runtime: generated call: {error:?}"));
    let abi = resolve_lib_type_path(db, scope, "std::abi::Sol")
        .ok_or_else(|| missing("std::abi::Sol"))?;
    let site = EffectParamSite::ContractRecvArm {
        contract,
        recv_idx: arm.recv(db).recv_idx(db),
        arm_idx: arm.arm_idx(db),
    };
    let host = registered_root_providers(db, site)
        .first()
        .ok_or_else(|| missing("root provider"))?
        .provider_ty;
    let info = arm.abi_info(db, abi);
    let decode = if info.args_ty != TyId::unit(db) {
        let decode = resolve_lib_func_path(db, scope, "core::contracts::decode_runtime_args")
            .ok_or_else(|| missing("core::contracts::decode_runtime_args"))?;
        let key = generated_callee_key(
            db,
            scope,
            PredicateListId::empty_list(db),
            decode,
            None,
            &[host, abi, info.args_ty],
        )
        .map_err(generated)?;
        Some(key)
    } else {
        None
    };
    let return_value = match info.ret_ty {
        Some(ret_ty) => {
            let host_trait = resolve_core_trait(db, scope, &["contracts", "ContractHost"])
                .ok_or_else(|| missing("core::contracts::ContractHost"))?;
            let inst = TraitInstId::new_simple(db, host_trait, vec![host]);
            let method = IdentId::new(db, "return_value".to_string());
            let func = host_trait
                .method_defs(db)
                .get(&method)
                .copied()
                .ok_or_else(|| missing("ContractHost::return_value"))?;
            let key = generated_callee_key(
                db,
                scope,
                PredicateListId::empty_list(db),
                func,
                Some(inst),
                &[abi, ret_ty],
            )
            .map_err(generated)?;
            Some(key)
        }
        None => None,
    };
    Ok(ArmCallees {
        host,
        info,
        decode,
        return_value,
    })
}

/// What the compiler's deployment code calls to decode the constructor
/// arguments: `Abi::decoder_new` and `Decode::decode_payload` at the tuple of
/// `init`'s parameter types, which decodes from a `core::ptr::MemSpan`.
pub struct InitCallees<'db> {
    pub decoder_new: SemanticInstanceKey<'db>,
    pub decode_payload: SemanticInstanceKey<'db>,
    pub span_ty: TyId<'db>,
}

/// The callees of the deployment code of `contract`, if `init` takes
/// arguments.
pub fn init_callees<'db>(
    db: &'db dyn HirAnalysisDb,
    contract: Contract<'db>,
) -> Result<Option<InitCallees<'db>>, Unsupported> {
    let scope = contract.scope();
    let missing = |what: &str| Unsupported(format!("init_entry: missing `{what}`"));
    let generated = |error| Unsupported(format!("init_entry: generated call: {error:?}"));
    let args_ty = contract.init_args_ty(db);
    if args_ty == TyId::unit(db) {
        return Ok(None);
    }
    let lib_ty = |path: &str| resolve_lib_type_path(db, scope, path).ok_or_else(|| missing(path));
    let abi = lib_ty("std::abi::Sol")?;
    let decoder_ty = lib_ty("std::abi::sol::SolDecoder")?;
    let span_ty = lib_ty("core::ptr::MemSpan")?;
    let method = |segments: &[&str], inst_args: Vec<TyId<'db>>, method: &str, own: &[TyId<'db>]| {
        let def =
            resolve_core_trait(db, scope, segments).ok_or_else(|| missing(&segments.join("::")))?;
        let func = def
            .method_defs(db)
            .get(&IdentId::new(db, method.to_string()))
            .copied()
            .ok_or_else(|| missing(method))?;
        let inst = TraitInstId::new_simple(db, def, inst_args);
        generated_callee_key(
            db,
            scope,
            PredicateListId::empty_list(db),
            func,
            Some(inst),
            own,
        )
        .map_err(generated)
    };
    Ok(Some(InitCallees {
        decoder_new: method(&["abi", "Abi"], vec![abi], "decoder_new", &[])?,
        decode_payload: method(
            &["abi", "Decode"],
            vec![args_ty, abi],
            "decode_payload",
            &[decoder_ty, span_ty],
        )?,
        span_ty,
    }))
}

/// The constant whose value `owner` is, as Fe names it: its path, or its
/// impl's type and its name.
pub fn const_path(db: &dyn HirAnalysisDb, owner: BodyOwner<'_>) -> String {
    match owner {
        BodyOwner::AnonConstBody { body, .. } => {
            let ty = match body.scope().parent_item(db) {
                Some(ItemKind::Impl(item)) => item.ty(db).pretty_print(db).as_str(),
                Some(ItemKind::ImplTrait(item)) => item.ty(db).pretty_print(db).as_str(),
                _ => "",
            };
            format!("{ty}::{}", assoc_const_name(db, body).unwrap_or("?"))
        }
        owner => fe_path(db, owner.scope()),
    }
}

/// The path of `scope` as Fe code names it: from `core` or `std` for theirs,
/// whose root module the compiler calls `lib`.
pub fn fe_path(db: &dyn HirAnalysisDb, scope: ScopeId<'_>) -> String {
    let path = scope.pretty_path(db).unwrap_or_default();
    let root = match scope.ingot(db).kind(db) {
        IngotKind::Core => "core",
        IngotKind::Std => "std",
        _ => return path,
    };
    let (_, rest) = path.split_once("::").unwrap_or_default();
    format!("{root}::{rest}")
}

/// The name of the associated constant whose value is `body`, an anonymous
/// body of its impl.
pub fn assoc_const_name<'db>(db: &'db dyn HirAnalysisDb, body: Body<'db>) -> Option<&'db str> {
    let consts = match body.scope().parent_item(db) {
        Some(ItemKind::Impl(item)) => item.hir_consts(db),
        Some(ItemKind::ImplTrait(item)) => item.hir_consts(db),
        _ => &[],
    };
    let constant = consts
        .iter()
        .find(|constant| constant.value.to_opt() == Some(body))?;
    Some(constant.name.to_opt()?.data(db))
}

/// Each recv arm of `contract` with its Lean name: `recv_` and the name of
/// the msg variant it handles, with the msg's name before the variant's if
/// another arm handles a variant of the same name, and `recv_fallback` for
/// the arm that takes any other call. An arm whose name or entry's name is
/// still another arm's, or that handles neither, is named by its place,
/// `recv<n>_arm<m>`.
pub fn arm_names<'db>(
    db: &'db dyn HirAnalysisDb,
    contract: Contract<'db>,
) -> Vec<(RecvArmView<'db>, String)> {
    let arms: Vec<_> = contract
        .recv_views(db)
        .flat_map(|recv| recv.arms(db))
        .collect();
    let variants: Vec<_> = arms
        .iter()
        .map(|arm| {
            let variant = arm.resolution(db).variant?.variant_struct;
            let msg = match variant.scope().parent_item(db) {
                Some(ItemKind::Mod(module)) => module.name(db).to_opt(),
                _ => None,
            };
            Some((msg, variant.name(db).to_opt()?))
        })
        .collect();
    let short: Vec<_> = arms
        .iter()
        .zip(&variants)
        .map(|(arm, variant)| match variant {
            Some((_, variant)) => Some(format!("recv_{}", variant.data(db))),
            None => arm.is_fallback(db).then(|| "recv_fallback".to_string()),
        })
        .collect();
    let chosen: Vec<_> = variants
        .iter()
        .zip(&short)
        .map(|(variant, name)| {
            if short.iter().filter(|other| *other == name).count() == 1 {
                return name.clone();
            }
            let (msg, variant) = (*variant)?;
            Some(format!("recv_{}_{}", msg?.data(db), variant.data(db)))
        })
        .collect();
    arms.iter()
        .zip(&chosen)
        .map(|(arm, name)| {
            let clashes = |name: &str| {
                chosen.iter().flatten().any(|other| {
                    *other == format!("{name}_entry") || name == format!("{other}_entry")
                }) || chosen
                    .iter()
                    .flatten()
                    .filter(|other| *other == name)
                    .count()
                    > 1
            };
            let name = match name {
                Some(name) if !clashes(name) => name.clone(),
                _ => format!("recv{}_arm{}", arm.recv(db).recv_idx(db), arm.arm_idx(db)),
            };
            (*arm, name)
        })
        .collect()
}

/// The errors the compiler reports in `ingot`, as `fe check` finds them: the
/// analysis of its modules, and the check of their accesses if that finds
/// none.
pub fn ingot_errors<'db>(db: &'db dyn SpannedHirAnalysisDb, ingot: Ingot<'db>) -> Vec<String> {
    let tree = ingot.module_tree(db);
    let errors = |mut passes: AnalysisPassManager| -> Vec<String> {
        let diagnostics = passes.run_on_module_tree(db, tree);
        let diagnostics = diagnostics
            .iter()
            .map(|diagnostic| diagnostic.to_complete(db));
        diagnostics
            .filter(|diagnostic| diagnostic.severity == Severity::Error)
            .map(|diagnostic| diagnostic.message)
            .collect()
    };
    let analysis = errors(initialize_analysis_pass());
    if !analysis.is_empty() {
        return analysis;
    }
    let mut access = AnalysisPassManager::new();
    access.add_module_pass("SemanticAccess", Box::new(SemanticAccessAnalysisPass));
    errors(access)
}

/// `contract` as its documentation names it: "contract `<name>`".
pub fn contract_title(db: &dyn HirAnalysisDb, contract: Contract<'_>) -> String {
    let name = contract.name(db).to_opt();
    format!("contract `{}`", name.map_or("", |name| name.data(db)))
}

/// The provider of the effect `binding` binds, if it is an effect's: an
/// effect a body takes by value, as an input after its parameters.
pub fn effect_input(binding: Option<LocalBinding<'_>>) -> Option<u32> {
    match binding? {
        LocalBinding::EffectParam { provider_idx, .. } => Some(provider_idx),
        _ => None,
    }
}

/// The place of `binding` among the patterns `pats`, if it is one of them.
pub fn arg_position(binding: Option<LocalBinding<'_>>, pats: &[PatId]) -> Option<usize> {
    let Some(LocalBinding::Local { pat, .. }) = binding else {
        return None;
    };
    pats.iter().position(|bound| *bound == pat)
}
