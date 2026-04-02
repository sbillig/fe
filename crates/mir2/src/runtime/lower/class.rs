use cranelift_entity::EntityRef;
use hir::analysis::{
    semantic::{
        Mutability, SConst, SEffectArg, SEffectArgValue, SExpr, SLocal, SLocalId, SStmt,
        STerminator, SemanticBody, SemanticInstance, ctfe::canonicalize_semantic_consts,
        owner_effect_bindings, sem_const_ty,
    },
    ty::{
        ProviderAddressSpace, ProviderKind,
        corelib::resolve_lib_func_path,
        provider_semantics,
        trait_resolution::PredicateListId,
        ty_check::{BodyOwner, EffectPassMode, LocalBinding, ParamSite},
        ty_def::{MAX_INLINE_STRING_BYTES, PrimTy, TyBase, TyData, TyId},
    },
};
use hir::semantic::ProviderBinding;

use crate::{
    db::MirDb,
    instance::{RuntimeInstanceKey, RuntimeInstanceSource},
    runtime::{
        AddressSpaceKind, HandleKind, HandleView, RuntimeCarrier, RuntimeClass, RuntimeCodeRegion,
        RuntimeCodeRegionKey, RuntimeParam, RuntimeSignature, ScalarClass, ScalarRepr, ScalarRole,
    },
};

use super::{
    consts::const_scalar_from_value,
    layout::{
        is_zero_sized_in_context, layout_for_ty, layout_for_ty_in_context, runtime_repr_ty,
        runtime_repr_ty_in_context,
    },
    place::{address_space_from_provider, effect_arg_address_space},
};

pub fn runtime_signature_for_key<'db>(
    db: &'db dyn MirDb,
    semantic: SemanticInstance<'db>,
    params: &[RuntimeClass<'db>],
) -> RuntimeSignature<'db> {
    let key = RuntimeInstanceKey::new(
        db,
        RuntimeInstanceSource::Semantic(semantic),
        params.to_vec(),
    );
    RuntimeSignature {
        params: params
            .iter()
            .zip(runtime_param_locals(db, semantic, params))
            .map(|(class, local)| RuntimeParam {
                local: crate::runtime::RLocalId::from_u32(local.index() as u32),
                class: class.clone(),
            })
            .collect(),
        ret: runtime_return_class_for_key(db, key),
    }
}

#[salsa::tracked(
    cycle_fn=runtime_return_class_cycle_recover,
    cycle_initial=runtime_return_class_cycle_initial
)]
pub fn runtime_return_class_for_key<'db>(
    db: &'db dyn MirDb,
    key: RuntimeInstanceKey<'db>,
) -> Option<RuntimeClass<'db>> {
    let semantic = key
        .semantic(db)
        .expect("return-class inference only applies to semantic runtime instances");
    let typed_body = semantic.key(db).instantiate_typed_body(db);
    let semantic_body = canonicalize_semantic_consts(db, semantic);
    let carriers = infer_local_carriers(
        db,
        &semantic_body,
        key.params(db),
        &runtime_param_locals(db, semantic, key.params(db)),
        typed_body.body().map(|body| body.scope()),
        typed_body.assumptions(),
    );
    let mut returned = semantic_body
        .blocks
        .iter()
        .filter_map(|block| match &block.terminator {
            STerminator::Return(Some(value)) => match carriers.get(value.index())? {
                RuntimeCarrier::Erased => None,
                RuntimeCarrier::Value(class) => Some(class.clone()),
            },
            STerminator::Goto(_)
            | STerminator::Branch { .. }
            | STerminator::MatchEnum { .. }
            | STerminator::Return(None) => None,
        })
        .collect::<Vec<_>>();
    let Some(first) = returned.pop() else {
        return default_return_class(db, &typed_body);
    };
    if returned.iter().all(|class| class == &first) {
        Some(first)
    } else {
        default_return_class(db, &typed_body)
    }
}

pub(super) fn infer_local_carriers<'db>(
    db: &'db dyn MirDb,
    body: &SemanticBody<'db>,
    params: &[RuntimeClass<'db>],
    param_locals: &[SLocalId],
    scope: Option<hir::hir_def::scope_graph::ScopeId<'db>>,
    assumptions: PredicateListId<'db>,
) -> Vec<RuntimeCarrier<'db>> {
    let mut carriers = vec![RuntimeCarrier::Erased; body.locals.len()];
    for (class, local) in params.iter().zip(param_locals.iter().copied()) {
        carriers[local.index()] = RuntimeCarrier::Value(class.clone());
    }

    loop {
        let mut changed = false;
        for block in &body.blocks {
            for stmt in &block.stmts {
                let SStmt::Assign { dst, expr } = stmt else {
                    continue;
                };
                if matches!(carriers[dst.index()], RuntimeCarrier::Value(_)) {
                    continue;
                }
                let desired = match &body.locals[dst.index()] {
                    SLocal {
                        ty,
                        mutability: Mutability::Mutable,
                        source: Some(_),
                    } if matches!(
                        top_level_class_for_ty_in_context(
                            db,
                            *ty,
                            AddressSpaceKind::Memory,
                            scope,
                            assumptions,
                        ),
                        Some(RuntimeClass::AggregateValue { .. })
                    ) =>
                    {
                        let ty = runtime_repr_ty_in_context(db, *ty, scope, assumptions);
                        RuntimeCarrier::Value(RuntimeClass::Handle {
                            layout: layout_for_ty(db, ty),
                            kind: HandleKind::ObjectValue,
                            view: HandleView::Whole,
                        })
                    }
                    local => match expr_direct_class(db, body, expr, local.ty, &carriers) {
                        Some(class) => RuntimeCarrier::Value(class),
                        None => continue,
                    },
                };
                carriers[dst.index()] = desired;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }

    carriers
}

pub(crate) fn runtime_param_locals<'db>(
    db: &'db dyn MirDb,
    semantic: SemanticInstance<'db>,
    params: &[RuntimeClass<'db>],
) -> Vec<SLocalId> {
    let typed_body = semantic.key(db).instantiate_typed_body(db);
    let semantic_body = semantic.body(db);
    let owner = semantic.key(db).owner(db);
    let scope = Some(owner.scope());
    let assumptions = typed_body.assumptions();
    let mut direct_bindings = Vec::new();
    let mut idx = 0;
    while let Some(binding) = typed_body.param_binding(idx) {
        let ty = typed_body.binding_ty(db, binding);
        if top_level_class_for_ty_in_context(db, ty, AddressSpaceKind::Memory, scope, assumptions)
            .is_some()
            || runtime_abstract_param_ty(db, ty, scope, assumptions)
        {
            direct_bindings.push(binding);
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
        let arm = hir::semantic::RecvArmView::new(db, recv, arm_idx);
        for arg_binding in arm.arg_bindings(db) {
            let Some(binding) = typed_body.pat_binding(arg_binding.pat) else {
                continue;
            };
            let ty = typed_body.binding_ty(db, binding);
            if top_level_class_for_ty_in_context(
                db,
                ty,
                AddressSpaceKind::Memory,
                scope,
                assumptions,
            )
            .is_some()
                || runtime_abstract_param_ty(db, ty, scope, assumptions)
            {
                direct_bindings.push(binding);
            }
        }
    }
    let mut owner_slots = Vec::new();
    let mut known_owner_count = 0usize;
    for binding in owner_effect_bindings(db, owner) {
        if !semantic_body.locals.iter().any(|local| {
            local
                .source
                .is_some_and(|source| same_owner_effect_binding(source, binding))
        }) {
            continue;
        }
        let known = owner_effect_binding_expected_class(db, semantic, binding).is_some();
        known_owner_count += usize::from(known);
        owner_slots.push((binding, known));
    }
    let extra_unknown_owners = params
        .len()
        .saturating_sub(direct_bindings.len() + known_owner_count);
    let owner_taken = known_owner_count + extra_unknown_owners;
    let direct_taken = params.len().saturating_sub(owner_taken);
    let direct_start = direct_bindings.len().saturating_sub(direct_taken);
    let mut bindings = direct_bindings
        .into_iter()
        .skip(direct_start)
        .collect::<Vec<_>>();
    let mut remaining_unknown = extra_unknown_owners;
    for (binding, known) in owner_slots {
        if known {
            bindings.push(binding);
        } else if remaining_unknown > 0 {
            bindings.push(binding);
            remaining_unknown -= 1;
        }
    }
    if bindings.len() != params.len() {
        panic!(
            "failed to map runtime params to semantic locals for {:?}: mapped {} of {}; params={params:?}",
            semantic.key(db),
            bindings.len(),
            params.len(),
        );
    }
    bindings
        .into_iter()
        .map(|binding| {
            semantic_body
                .locals
                .iter()
                .position(|local| {
                    local.source.is_some_and(|source| {
                        source == binding || same_owner_effect_binding(source, binding)
                    })
                })
                .map(|idx| SLocalId::from_u32(idx as u32))
                .unwrap_or_else(|| {
                    panic!("missing semantic local for runtime-visible binding {binding:?}")
                })
        })
        .collect()
}

pub(crate) fn resolved_provider_binding_for_owner_effect<'db>(
    db: &'db dyn MirDb,
    owner: BodyOwner<'db>,
    binding_idx: usize,
) -> Option<ProviderBinding<'db>> {
    let site = match owner {
        BodyOwner::Func(func) => hir::analysis::ty::ty_check::EffectParamSite::Func(func),
        BodyOwner::ContractInit { contract } => {
            hir::analysis::ty::ty_check::EffectParamSite::ContractInit { contract }
        }
        BodyOwner::ContractRecvArm {
            contract,
            recv_idx,
            arm_idx,
        } => hir::analysis::ty::ty_check::EffectParamSite::ContractRecvArm {
            contract,
            recv_idx,
            arm_idx,
        },
        BodyOwner::Const(_) | BodyOwner::AnonConstBody { .. } => return None,
    };
    let view = hir::semantic::EffectEnvView::new(site);
    let provider_idx = view
        .resolutions(db)
        .into_iter()
        .find(|resolution| resolution.requirement_idx as usize == binding_idx)?
        .provider_idx;
    view.providers(db)
        .into_iter()
        .find(|provider| provider.provider_idx == provider_idx)
}

pub(crate) fn runtime_class_for_provider_binding<'db>(
    db: &'db dyn MirDb,
    provider: &ProviderBinding<'db>,
    scope: Option<hir::hir_def::scope_graph::ScopeId<'db>>,
    assumptions: PredicateListId<'db>,
) -> Option<RuntimeClass<'db>> {
    match provider.semantics.kind {
        ProviderKind::RootObject => top_level_class_for_ty_in_context(
            db,
            provider.provider_ty,
            AddressSpaceKind::Memory,
            scope,
            assumptions,
        ),
        ProviderKind::Handle | ProviderKind::RawAddress => {
            Some(provider_class_for_target_in_context(
                db,
                provider.semantics.target_ty,
                provider_address_space_to_runtime(provider.semantics.address_space?),
                scope,
                assumptions,
            ))
        }
    }
}

fn owner_effect_binding_expected_class<'db>(
    db: &'db dyn MirDb,
    semantic: SemanticInstance<'db>,
    binding: LocalBinding<'db>,
) -> Option<RuntimeClass<'db>> {
    let owner = semantic.key(db).owner(db);
    let assumptions = semantic.key(db).instantiate_typed_body(db).assumptions();
    let binding_idx = match binding {
        LocalBinding::EffectParam { idx, .. }
        | LocalBinding::Param {
            site: ParamSite::EffectField(_),
            idx,
            ..
        } => idx,
        LocalBinding::Local { .. } | LocalBinding::Param { .. } => return None,
    };
    let provider = resolved_provider_binding_for_owner_effect(db, owner, binding_idx)
        .unwrap_or_else(|| {
            panic!("missing provider binding metadata for {owner:?} binding {binding:?}")
        });
    runtime_class_for_provider_binding(db, &provider, Some(owner.scope()), assumptions)
}

fn expr_direct_class<'db>(
    db: &'db dyn MirDb,
    body: &SemanticBody<'db>,
    expr: &SExpr<'db>,
    result_ty: TyId<'db>,
    carriers: &[RuntimeCarrier<'db>],
) -> Option<RuntimeClass<'db>> {
    let owner = body.owner.key(db).owner(db);
    let typed_body = body.owner.key(db).instantiate_typed_body(db);
    let scope = Some(owner.scope());
    let assumptions = typed_body.assumptions();
    Some(match expr {
        SExpr::Use(value) => {
            if let Some(class) = effect_binding_value_class(db, body, *value, carriers) {
                return Some(class);
            }
            match carriers.get(value.index())? {
                RuntimeCarrier::Erased => return None,
                RuntimeCarrier::Value(class) => class.clone(),
            }
        }
        SExpr::Const(const_) => match const_ {
            SConst::Value(value) => {
                let ty = sem_const_ty(db, *value);
                if ty == TyId::unit(db) {
                    return None;
                }
                if const_scalar_from_value(db, *value).is_some() {
                    scalar_class_for_ty(db, ty).map(RuntimeClass::Scalar)?
                } else {
                    RuntimeClass::Handle {
                        layout: layout_for_ty(db, ty),
                        kind: HandleKind::ConstValue,
                        view: HandleView::Whole,
                    }
                }
            }
            SConst::Ref(cref) => {
                panic!("unresolved const ref reached runtime class inference: {cref:?}")
            }
        },
        SExpr::Unary { .. }
        | SExpr::Binary { .. }
        | SExpr::Cast { .. }
        | SExpr::CodeRegionOffset { .. }
        | SExpr::CodeRegionLen { .. }
        | SExpr::GetEnumTag { .. } => RuntimeClass::Scalar(scalar_class_for_ty(db, result_ty)?),
        SExpr::AggregateMake { ty, .. } => top_level_class_for_ty_in_context(
            db,
            *ty,
            AddressSpaceKind::Memory,
            scope,
            assumptions,
        )?,
        SExpr::EnumMake { enum_ty, .. } => RuntimeClass::AggregateValue {
            layout: layout_for_ty_in_context(db, *enum_ty, scope, assumptions),
        },
        SExpr::Field { .. } | SExpr::Index { .. } | SExpr::ExtractEnumField { .. } => {
            top_level_class_for_ty_in_context(
                db,
                result_ty,
                AddressSpaceKind::Memory,
                scope,
                assumptions,
            )?
        }
        SExpr::Borrow { provider, .. } => provider_class_for_target_in_context(
            db,
            Some(
                result_ty
                    .as_borrow(db)
                    .map_or(result_ty, |(_, inner)| inner),
            ),
            provider.map_or(AddressSpaceKind::Memory, address_space_from_provider),
            scope,
            assumptions,
        ),
        SExpr::IsEnumVariant { .. } => RuntimeClass::Scalar(ScalarClass {
            repr: ScalarRepr::Bool,
            role: ScalarRole::Plain,
        }),
        SExpr::Call {
            callee,
            args,
            effect_args,
        } => {
            let semantic = SemanticInstance::new(db, callee.key);
            if let Some(class) = extern_builtin_return_class(db, semantic, result_ty) {
                return class;
            }
            let typed_body = semantic.key(db).instantiate_typed_body(db);
            let scope = typed_body.body().map(|body| body.scope());
            let assumptions = typed_body.assumptions();
            let mut param_classes = Vec::new();
            for (idx, arg) in args.iter().enumerate() {
                let desired = typed_body.param_binding(idx).and_then(|binding| {
                    top_level_class_for_ty_in_context(
                        db,
                        typed_body.binding_ty(db, binding),
                        AddressSpaceKind::Memory,
                        scope,
                        assumptions,
                    )
                    .map(|class| runtime_param_class(db, &typed_body, binding, class))
                });
                let actual = runtime_visible_arg_class(body, *arg, carriers);
                if let (Some(actual), Some(desired)) = (actual, desired) {
                    param_classes.push(preserve_provider_space(&actual, &desired));
                }
            }
            for arg in effect_args {
                if let Some(class) = effect_arg_class(db, arg, carriers) {
                    param_classes.push(class);
                }
            }
            for binding in owner_effect_bindings(db, semantic.key(db).owner(db)) {
                if let Some(class) = owner_effect_arg_class(body, carriers, binding) {
                    param_classes.push(class);
                }
            }
            runtime_return_class_for_key(
                db,
                RuntimeInstanceKey::new(
                    db,
                    RuntimeInstanceSource::Semantic(semantic),
                    param_classes,
                ),
            )?
        }
    })
}

fn runtime_visible_arg_class<'db>(
    body: &SemanticBody<'db>,
    local: SLocalId,
    carriers: &[RuntimeCarrier<'db>],
) -> Option<RuntimeClass<'db>> {
    let source = runtime_visible_arg_source(body, local);
    match carriers.get(source.index())? {
        RuntimeCarrier::Erased => None,
        RuntimeCarrier::Value(class) => Some(class.clone()),
    }
}

fn runtime_visible_arg_source<'db>(body: &SemanticBody<'db>, mut local: SLocalId) -> SLocalId {
    while let Some(source) = use_alias_source(body, local) {
        local = source;
    }
    local
}

fn use_alias_source<'db>(body: &SemanticBody<'db>, local: SLocalId) -> Option<SLocalId> {
    body.blocks.iter().find_map(|block| {
        block.stmts.iter().find_map(|stmt| match stmt {
            SStmt::Assign {
                dst,
                expr: SExpr::Use(src),
            } if *dst == local => Some(*src),
            SStmt::Assign { .. } | SStmt::Store { .. } => None,
        })
    })
}

fn preserve_provider_space<'db>(
    actual: &RuntimeClass<'db>,
    desired: &RuntimeClass<'db>,
) -> RuntimeClass<'db> {
    match (actual, desired) {
        (
            RuntimeClass::Handle {
                layout: actual_layout,
                kind:
                    HandleKind::Provider {
                        provider_ty: actual_provider_ty,
                        space: actual_space,
                    },
                view: HandleView::Whole,
            },
            RuntimeClass::Handle {
                layout: desired_layout,
                kind:
                    HandleKind::Provider {
                        provider_ty: desired_provider_ty,
                        ..
                    },
                view: HandleView::Whole,
            },
        ) if actual_layout == desired_layout && actual_provider_ty == desired_provider_ty => {
            RuntimeClass::Handle {
                layout: *actual_layout,
                kind: HandleKind::Provider {
                    provider_ty: *actual_provider_ty,
                    space: *actual_space,
                },
                view: HandleView::Whole,
            }
        }
        (
            RuntimeClass::RawAddr {
                space: actual_space,
                target: actual_target,
            },
            RuntimeClass::RawAddr {
                target: desired_target,
                ..
            },
        ) if actual_target == desired_target => RuntimeClass::RawAddr {
            space: *actual_space,
            target: *actual_target,
        },
        _ => desired.clone(),
    }
}

pub(crate) fn effect_binding_value_class<'db>(
    db: &'db dyn MirDb,
    body: &SemanticBody<'db>,
    local: SLocalId,
    carriers: &[RuntimeCarrier<'db>],
) -> Option<RuntimeClass<'db>> {
    let local_data = body.local(local)?;
    let binding = local_data.source?;
    if !is_effect_binding(binding) {
        return None;
    }
    let RuntimeCarrier::Value(carrier_class) = carriers.get(local.index())? else {
        return None;
    };
    let typed_body = body.owner.key(db).instantiate_typed_body(db);
    let top_level = top_level_class_for_ty_in_context(
        db,
        local_data.ty,
        AddressSpaceKind::Memory,
        body.owner.key(db).owner(db).scope().into(),
        typed_body.assumptions(),
    )?;
    if &top_level == carrier_class {
        return None;
    }
    Some(stored_class_for_ty_in_context(
        db,
        local_data.ty,
        body.owner.key(db).owner(db).scope().into(),
        typed_body.assumptions(),
    ))
}

pub(crate) fn is_effect_binding<'db>(binding: LocalBinding<'db>) -> bool {
    matches!(
        binding,
        LocalBinding::EffectParam { .. }
            | LocalBinding::Param {
                site: ParamSite::EffectField(_),
                ..
            }
    )
}

pub(crate) fn same_owner_effect_binding<'db>(
    lhs: LocalBinding<'db>,
    rhs: LocalBinding<'db>,
) -> bool {
    match (lhs, rhs) {
        (
            LocalBinding::EffectParam {
                idx: lhs_idx,
                key_path: lhs_key,
                ..
            },
            LocalBinding::EffectParam {
                idx: rhs_idx,
                key_path: rhs_key,
                ..
            },
        ) => lhs_idx == rhs_idx && lhs_key == rhs_key,
        (
            LocalBinding::Param {
                site: ParamSite::EffectField(_),
                idx: lhs_idx,
                ty: lhs_ty,
                ..
            },
            LocalBinding::Param {
                site: ParamSite::EffectField(_),
                idx: rhs_idx,
                ty: rhs_ty,
                ..
            },
        ) => lhs_idx == rhs_idx && lhs_ty == rhs_ty,
        _ => false,
    }
}

fn owner_effect_arg_class<'db>(
    body: &SemanticBody<'db>,
    carriers: &[RuntimeCarrier<'db>],
    binding: LocalBinding<'db>,
) -> Option<RuntimeClass<'db>> {
    let idx = body.locals.iter().position(|local| {
        local
            .source
            .is_some_and(|source| same_owner_effect_binding(source, binding))
    })?;
    match carriers.get(idx)? {
        RuntimeCarrier::Erased => None,
        RuntimeCarrier::Value(class) => Some(class.clone()),
    }
}

fn effect_arg_class<'db>(
    db: &'db dyn MirDb,
    arg: &SEffectArg<'db>,
    carriers: &[RuntimeCarrier<'db>],
) -> Option<RuntimeClass<'db>> {
    match arg.pass_mode {
        EffectPassMode::ByValue | EffectPassMode::Unknown => match arg.arg {
            SEffectArgValue::Place(_) => Some(provider_class_for_target_in_context(
                db,
                arg.target_ty,
                effect_arg_address_space(arg),
                None,
                PredicateListId::empty_list(db),
            )),
            SEffectArgValue::Value(value) => match carriers.get(value.index())? {
                RuntimeCarrier::Value(class) => Some(class.clone()),
                RuntimeCarrier::Erased if arg.provider.is_none() && arg.target_ty.is_none() => None,
                RuntimeCarrier::Erased => Some(provider_class_for_target_in_context(
                    db,
                    arg.target_ty,
                    effect_arg_address_space(arg),
                    None,
                    PredicateListId::empty_list(db),
                )),
            },
        },
        EffectPassMode::ByPlace | EffectPassMode::ByTempPlace => {
            Some(provider_class_for_target_in_context(
                db,
                arg.target_ty,
                effect_arg_address_space(arg),
                None,
                PredicateListId::empty_list(db),
            ))
        }
    }
}

pub(crate) enum ContractMetadataBuiltin<'db> {
    InitCodeOffset(RuntimeCodeRegion<'db>),
    InitCodeLen(RuntimeCodeRegion<'db>),
}

pub(crate) fn contract_metadata_builtin<'db>(
    db: &'db dyn MirDb,
    semantic: SemanticInstance<'db>,
) -> Option<ContractMetadataBuiltin<'db>> {
    let BodyOwner::Func(func) = semantic.key(db).owner(db) else {
        return None;
    };
    let name = func.name(db).to_opt()?.data(db);
    let trait_ = func.containing_trait(db)?;
    if trait_.name(db).to_opt()?.data(db) != "Contract" {
        return None;
    }
    let contract = semantic
        .key(db)
        .subst(db)
        .generic_args(db)
        .iter()
        .find_map(|ty| ty.as_contract(db))?;
    let region = RuntimeCodeRegion::new(db, RuntimeCodeRegionKey::ContractInit { contract });
    match name.as_str() {
        "init_code_offset" => Some(ContractMetadataBuiltin::InitCodeOffset(region)),
        "init_code_len" => Some(ContractMetadataBuiltin::InitCodeLen(region)),
        _ => None,
    }
}

fn extern_builtin_return_class<'db>(
    db: &'db dyn MirDb,
    semantic: SemanticInstance<'db>,
    result_ty: TyId<'db>,
) -> Option<Option<RuntimeClass<'db>>> {
    if contract_metadata_builtin(db, semantic).is_some() {
        return Some(top_level_class_for_ty(
            db,
            result_ty,
            AddressSpaceKind::Memory,
        ));
    }
    let hir::analysis::ty::ty_check::BodyOwner::Func(func) = semantic.key(db).owner(db) else {
        return None;
    };
    if func.body(db).is_none()
        && func
            .name(db)
            .to_opt()
            .is_some_and(|name| is_runtime_intrinsic_name(name.data(db).as_str()))
    {
        return Some(top_level_class_for_ty(
            db,
            result_ty,
            AddressSpaceKind::Memory,
        ));
    }
    let matches = |path: &str| resolve_lib_func_path(db, func.scope(), path) == Some(func);
    if matches("std::evm::mem::alloc") {
        return Some(top_level_class_for_ty(
            db,
            result_ty,
            AddressSpaceKind::Memory,
        ));
    }
    if matches("core::intrinsic::__keccak256") {
        return Some(top_level_class_for_ty(
            db,
            result_ty,
            AddressSpaceKind::Memory,
        ));
    }

    let known = [
        "std::evm::ops::mload",
        "std::evm::ops::mstore",
        "std::evm::ops::mstore8",
        "std::evm::ops::msize",
        "std::evm::ops::sload",
        "std::evm::ops::sstore",
        "std::evm::ops::calldataload",
        "std::evm::ops::calldatacopy",
        "std::evm::ops::calldatasize",
        "std::evm::ops::returndatacopy",
        "std::evm::ops::returndatasize",
        "std::evm::ops::codecopy",
        "std::evm::ops::codesize",
        "std::evm::ops::keccak256",
        "std::evm::ops::addmod",
        "std::evm::ops::mulmod",
        "std::evm::ops::address",
        "std::evm::ops::caller",
        "std::evm::ops::callvalue",
        "std::evm::ops::origin",
        "std::evm::ops::gasprice",
        "std::evm::ops::coinbase",
        "std::evm::ops::timestamp",
        "std::evm::ops::number",
        "std::evm::ops::prevrandao",
        "std::evm::ops::gaslimit",
        "std::evm::ops::chainid",
        "std::evm::ops::basefee",
        "std::evm::ops::selfbalance",
        "std::evm::ops::blockhash",
        "std::evm::ops::gas",
        "std::evm::ops::call",
        "std::evm::ops::staticcall",
        "std::evm::ops::delegatecall",
        "std::evm::ops::create",
        "std::evm::ops::create2",
        "std::evm::ops::log0",
        "std::evm::ops::log1",
        "std::evm::ops::log2",
        "std::evm::ops::log3",
        "std::evm::ops::log4",
        "std::evm::ops::revert",
        "std::evm::ops::return_data",
        "std::evm::ops::selfdestruct",
        "std::evm::ops::stop",
        "core::panic",
        "core::panic_with_value",
        "core::todo",
    ]
    .iter()
    .any(|path| matches(path));
    known.then(|| top_level_class_for_ty(db, result_ty, AddressSpaceKind::Memory))
}

fn is_runtime_intrinsic_name(name: &str) -> bool {
    if matches!(
        name,
        "alloc" | "__bitcast" | "__saturating_add" | "__saturating_sub" | "__saturating_mul"
    ) {
        return true;
    }
    intrinsic_numeric_name_parts(name).is_some()
}

fn intrinsic_numeric_name_parts(name: &str) -> Option<(&str, &str)> {
    let op = name.strip_prefix("__")?;
    [
        "_u8", "_u16", "_u32", "_u64", "_u128", "_u256", "_usize", "_i8", "_i16", "_i32", "_i64",
        "_i128", "_i256", "_isize", "_bool",
    ]
    .iter()
    .find_map(|suffix| op.strip_suffix(suffix).map(|prefix| (prefix, *suffix)))
}

fn runtime_return_class_cycle_initial<'db>(
    db: &'db dyn MirDb,
    key: RuntimeInstanceKey<'db>,
) -> Option<RuntimeClass<'db>> {
    let typed_body = key
        .semantic(db)
        .expect("cycle handling only applies to semantic runtime instances")
        .key(db)
        .instantiate_typed_body(db);
    default_return_class(db, &typed_body)
}

fn runtime_return_class_cycle_recover<'db>(
    _db: &'db dyn MirDb,
    _value: &Option<RuntimeClass<'db>>,
    _count: u32,
    _key: RuntimeInstanceKey<'db>,
) -> salsa::CycleRecoveryAction<Option<RuntimeClass<'db>>> {
    salsa::CycleRecoveryAction::Iterate
}

pub(crate) fn runtime_param_class<'db>(
    db: &'db dyn MirDb,
    typed_body: &hir::analysis::ty::ty_check::TypedBody<'db>,
    binding: hir::analysis::ty::ty_check::LocalBinding<'db>,
    actual: RuntimeClass<'db>,
) -> RuntimeClass<'db> {
    let ty = runtime_repr_ty_in_context(
        db,
        typed_body.binding_ty(db, binding),
        typed_body.body().map(|body| body.scope()),
        typed_body.assumptions(),
    );
    if runtime_abstract_param_ty(
        db,
        typed_body.binding_ty(db, binding),
        typed_body.body().map(|body| body.scope()),
        typed_body.assumptions(),
    ) || matches!(
        ty.base_ty(db).data(db),
        TyData::TyParam(param) if param.is_effect() || param.is_effect_provider()
    ) {
        return actual;
    }
    if binding.is_mut() && ty.as_enum(db).is_some() {
        return RuntimeClass::Handle {
            layout: layout_for_ty_in_context(
                db,
                ty,
                typed_body.body().map(|body| body.scope()),
                typed_body.assumptions(),
            ),
            kind: HandleKind::ObjectValue,
            view: HandleView::Whole,
        };
    }
    actual
}

pub(crate) fn semantic_return_ty<'db>(
    db: &'db dyn MirDb,
    semantic: SemanticInstance<'db>,
) -> TyId<'db> {
    semantic.key(db).instantiate_typed_body(db).result_ty()
}

fn default_return_class<'db>(
    db: &'db dyn MirDb,
    typed_body: &hir::analysis::ty::ty_check::TypedBody<'db>,
) -> Option<RuntimeClass<'db>> {
    let default_space = typed_body
        .return_borrow_provider()
        .map_or(AddressSpaceKind::Memory, address_space_from_provider);
    let body = typed_body.body()?;
    top_level_class_for_ty_in_context(
        db,
        typed_body.expr_ty(db, body.expr(db)),
        default_space,
        Some(body.scope()),
        typed_body.assumptions(),
    )
}

pub(crate) fn top_level_class_for_ty<'db>(
    db: &'db dyn MirDb,
    ty: TyId<'db>,
    default_space: AddressSpaceKind,
) -> Option<RuntimeClass<'db>> {
    top_level_class_for_ty_in_context(
        db,
        ty,
        default_space,
        ty.as_scope(db),
        PredicateListId::empty_list(db),
    )
}

pub(crate) fn top_level_class_for_ty_in_context<'db>(
    db: &'db dyn MirDb,
    ty: TyId<'db>,
    default_space: AddressSpaceKind,
    scope: Option<hir::hir_def::scope_graph::ScopeId<'db>>,
    assumptions: PredicateListId<'db>,
) -> Option<RuntimeClass<'db>> {
    let ty = runtime_repr_ty_in_context(db, ty, scope, assumptions);
    if ty == TyId::unit(db) || is_zero_sized_in_context(db, ty, scope, assumptions) {
        return None;
    }
    if let Some(scalar) = scalar_class_for_ty(db, ty) {
        return Some(RuntimeClass::Scalar(scalar));
    }
    if let Some((_, inner)) = ty.as_borrow(db) {
        if is_zero_sized_in_context(db, inner, scope, assumptions) {
            return None;
        }
        return Some(provider_class_for_target_in_context(
            db,
            Some(inner),
            default_space,
            scope,
            assumptions,
        ));
    }
    if let Some((_, inner)) = ty.as_capability(db) {
        if is_zero_sized_in_context(db, inner, scope, assumptions) {
            return None;
        }
        return Some(provider_class_for_target_in_context(
            db,
            Some(inner),
            default_space,
            scope,
            assumptions,
        ));
    }
    if let Some(class) = effect_handle_class_for_ty(db, ty, scope, assumptions) {
        return Some(class);
    }
    if ty.as_enum(db).is_some() {
        return Some(RuntimeClass::AggregateValue {
            layout: layout_for_ty_in_context(db, ty, scope, assumptions),
        });
    }
    if ty.is_struct(db) || ty.is_array(db) || ty.is_tuple(db) {
        return Some(RuntimeClass::AggregateValue {
            layout: layout_for_ty_in_context(db, ty, scope, assumptions),
        });
    }
    None
}

pub(crate) fn stored_class_for_ty_in_context<'db>(
    db: &'db dyn MirDb,
    ty: TyId<'db>,
    scope: Option<hir::hir_def::scope_graph::ScopeId<'db>>,
    assumptions: PredicateListId<'db>,
) -> RuntimeClass<'db> {
    let ty = runtime_repr_ty_in_context(db, ty, scope, assumptions);
    if let Some(scalar) = scalar_class_for_ty(db, ty) {
        return RuntimeClass::Scalar(scalar);
    }
    if let Some((_, inner)) = ty.as_capability(db) {
        return provider_class_for_target_in_context(
            db,
            Some(inner),
            AddressSpaceKind::Memory,
            scope,
            assumptions,
        );
    }
    if let Some(class) = effect_handle_class_for_ty(db, ty, scope, assumptions) {
        return class;
    }
    RuntimeClass::AggregateValue {
        layout: layout_for_ty_in_context(db, ty, scope, assumptions),
    }
}

pub(crate) fn provider_class_for_target<'db>(
    db: &'db dyn MirDb,
    target_ty: Option<TyId<'db>>,
    space: AddressSpaceKind,
) -> RuntimeClass<'db> {
    provider_class_for_target_in_context(
        db,
        target_ty,
        space,
        target_ty.and_then(|ty| ty.as_scope(db)),
        PredicateListId::empty_list(db),
    )
}

pub(crate) fn provider_class_for_target_in_context<'db>(
    db: &'db dyn MirDb,
    target_ty: Option<TyId<'db>>,
    space: AddressSpaceKind,
    scope: Option<hir::hir_def::scope_graph::ScopeId<'db>>,
    assumptions: PredicateListId<'db>,
) -> RuntimeClass<'db> {
    match target_ty.map(|ty| runtime_repr_ty_in_context(db, ty, scope, assumptions)) {
        Some(target_ty)
            if target_ty.is_struct(db)
                || target_ty.is_array(db)
                || target_ty.is_tuple(db)
                || target_ty.as_enum(db).is_some() =>
        {
            RuntimeClass::Handle {
                layout: layout_for_ty_in_context(db, target_ty, scope, assumptions),
                kind: HandleKind::Provider {
                    provider_ty: TyId::borrow_ref_of(db, target_ty),
                    space,
                },
                view: HandleView::Whole,
            }
        }
        Some(target_ty) if scalar_class_for_ty(db, target_ty).is_some() => RuntimeClass::RawAddr {
            space,
            target: None,
        },
        Some(target_ty) => RuntimeClass::RawAddr {
            space,
            target: layout_for_ty(db, target_ty).into(),
        },
        None => RuntimeClass::RawAddr {
            space,
            target: None,
        },
    }
}

pub(crate) fn scalar_class_for_ty<'db>(
    db: &'db dyn MirDb,
    ty: TyId<'db>,
) -> Option<ScalarClass<'db>> {
    let ty = runtime_repr_ty(db, ty);
    let repr = match ty.base_ty(db).data(db) {
        TyData::TyBase(TyBase::Prim(prim)) => match prim {
            PrimTy::Bool => ScalarRepr::Bool,
            PrimTy::U8 => ScalarRepr::Int {
                bits: 8,
                signed: false,
            },
            PrimTy::U16 => ScalarRepr::Int {
                bits: 16,
                signed: false,
            },
            PrimTy::U32 => ScalarRepr::Int {
                bits: 32,
                signed: false,
            },
            PrimTy::U64 => ScalarRepr::Int {
                bits: 64,
                signed: false,
            },
            PrimTy::U128 => ScalarRepr::Int {
                bits: 128,
                signed: false,
            },
            PrimTy::U256 | PrimTy::Usize => ScalarRepr::Int {
                bits: 256,
                signed: false,
            },
            PrimTy::I8 => ScalarRepr::Int {
                bits: 8,
                signed: true,
            },
            PrimTy::I16 => ScalarRepr::Int {
                bits: 16,
                signed: true,
            },
            PrimTy::I32 => ScalarRepr::Int {
                bits: 32,
                signed: true,
            },
            PrimTy::I64 => ScalarRepr::Int {
                bits: 64,
                signed: true,
            },
            PrimTy::I128 => ScalarRepr::Int {
                bits: 128,
                signed: true,
            },
            PrimTy::I256 | PrimTy::Isize => ScalarRepr::Int {
                bits: 256,
                signed: true,
            },
            PrimTy::String => ScalarRepr::FixedBytes {
                len: MAX_INLINE_STRING_BYTES as u16,
            },
            PrimTy::Array
            | PrimTy::Tuple(_)
            | PrimTy::Ptr
            | PrimTy::View
            | PrimTy::BorrowMut
            | PrimTy::BorrowRef => return None,
        },
        TyData::TyBase(TyBase::Contract(_)) => ScalarRepr::Address { bits: 256 },
        _ => return None,
    };

    Some(ScalarClass {
        repr,
        role: ScalarRole::Plain,
    })
}

fn effect_handle_class_for_ty<'db>(
    db: &'db dyn MirDb,
    ty: TyId<'db>,
    scope: Option<hir::hir_def::scope_graph::ScopeId<'db>>,
    assumptions: PredicateListId<'db>,
) -> Option<RuntimeClass<'db>> {
    let scope = scope.or_else(|| ty.as_scope(db))?;
    let semantics = provider_semantics(db, scope, assumptions, ty);
    if matches!(semantics.kind, ProviderKind::RootObject) {
        return None;
    }
    if let Some(target_ty) = semantics.target_ty {
        if is_zero_sized_in_context(db, target_ty, Some(scope), assumptions) {
            return None;
        }
        return Some(provider_class_for_target_in_context(
            db,
            Some(target_ty),
            provider_address_space_to_runtime(semantics.address_space?),
            Some(scope),
            assumptions,
        ));
    }
    None
}

fn runtime_abstract_param_ty<'db>(
    db: &'db dyn MirDb,
    ty: TyId<'db>,
    scope: Option<hir::hir_def::scope_graph::ScopeId<'db>>,
    assumptions: PredicateListId<'db>,
) -> bool {
    let ty = runtime_repr_ty_in_context(db, ty, scope, assumptions);
    ty.has_param(db) || ty.contains_assoc_ty_of_param(db)
}

fn provider_address_space_to_runtime(space: ProviderAddressSpace) -> AddressSpaceKind {
    match space {
        ProviderAddressSpace::Memory => AddressSpaceKind::Memory,
        ProviderAddressSpace::Storage => AddressSpaceKind::Storage,
        ProviderAddressSpace::Transient => AddressSpaceKind::Transient,
        ProviderAddressSpace::Calldata => AddressSpaceKind::Calldata,
    }
}
