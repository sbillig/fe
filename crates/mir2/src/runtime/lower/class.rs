use cranelift_entity::EntityRef;
use hir::analysis::{
    semantic::{
        Mutability, SConst, SEffectArg, SEffectArgValue, SExpr, SLocal, SStmt, STerminator,
        SemanticBody, SemanticInstance, ctfe::canonicalize_semantic_consts, sem_const_ty,
    },
    ty::{
        ty_check::EffectPassMode,
        ty_def::{MAX_INLINE_STRING_BYTES, PrimTy, TyBase, TyData, TyId},
    },
};

use crate::{
    db::MirDb,
    instance::RuntimeInstanceKey,
    runtime::{
        AddressSpaceKind, HandleKind, HandleView, RuntimeCarrier, RuntimeClass, RuntimeParam,
        RuntimeSignature, ScalarClass, ScalarRepr, ScalarRole,
    },
};

use super::{
    consts::const_scalar_from_value,
    layout::layout_for_ty,
    place::{address_space_from_provider, effect_arg_address_space},
};

pub fn runtime_signature_for_key<'db>(
    db: &'db dyn MirDb,
    semantic: SemanticInstance<'db>,
    params: &[RuntimeClass<'db>],
) -> RuntimeSignature<'db> {
    let key = RuntimeInstanceKey::new(db, semantic, params.to_vec());
    RuntimeSignature {
        params: params
            .iter()
            .enumerate()
            .map(|(idx, class)| RuntimeParam {
                local: crate::runtime::RLocalId::from_u32(idx as u32),
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
    let semantic = key.semantic(db);
    let typed_body = semantic.key(db).instantiate_typed_body(db);
    let semantic_body = canonicalize_semantic_consts(db, semantic);
    let carriers = infer_local_carriers(db, &semantic_body, key.params(db));
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
) -> Vec<RuntimeCarrier<'db>> {
    let mut carriers = vec![RuntimeCarrier::Erased; body.locals.len()];
    for (idx, class) in params.iter().enumerate().take(body.locals.len()) {
        carriers[idx] = RuntimeCarrier::Value(class.clone());
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
                    } if ty.is_struct(db) || ty.is_array(db) || ty.as_enum(db).is_some() => {
                        RuntimeCarrier::Value(RuntimeClass::Handle {
                            layout: layout_for_ty(db, *ty),
                            kind: HandleKind::ObjectValue,
                            view: HandleView::Whole,
                        })
                    }
                    local => match expr_direct_class(db, expr, local.ty, &carriers) {
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

fn expr_direct_class<'db>(
    db: &'db dyn MirDb,
    expr: &SExpr<'db>,
    result_ty: TyId<'db>,
    carriers: &[RuntimeCarrier<'db>],
) -> Option<RuntimeClass<'db>> {
    Some(match expr {
        SExpr::Use(value) => match carriers.get(value.index())? {
            RuntimeCarrier::Erased => return None,
            RuntimeCarrier::Value(class) => class.clone(),
        },
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
        | SExpr::GetEnumTag { .. } => RuntimeClass::Scalar(scalar_class_for_ty(db, result_ty)?),
        SExpr::AggregateMake { ty, .. } => RuntimeClass::Handle {
            layout: layout_for_ty(db, *ty),
            kind: HandleKind::ObjectValue,
            view: HandleView::Whole,
        },
        SExpr::EnumMake { enum_ty, .. } => RuntimeClass::AggregateValue {
            layout: layout_for_ty(db, *enum_ty),
        },
        SExpr::Field { .. } | SExpr::Index { .. } | SExpr::ExtractEnumField { .. } => {
            top_level_class_for_ty(db, result_ty, AddressSpaceKind::Memory)?
        }
        SExpr::Borrow { provider, .. } => provider_class_for_target(
            db,
            Some(
                result_ty
                    .as_borrow(db)
                    .map_or(result_ty, |(_, inner)| inner),
            ),
            provider.map_or(AddressSpaceKind::Memory, address_space_from_provider),
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
            let mut param_classes = args
                .iter()
                .filter_map(|arg| match carriers.get(arg.index())? {
                    RuntimeCarrier::Erased => None,
                    RuntimeCarrier::Value(class) => Some(class.clone()),
                })
                .collect::<Vec<_>>();
            for arg in effect_args {
                param_classes.push(effect_arg_class(db, arg, carriers)?);
            }
            let semantic = SemanticInstance::new(db, callee.key);
            runtime_return_class_for_key(db, RuntimeInstanceKey::new(db, semantic, param_classes))?
        }
    })
}

fn effect_arg_class<'db>(
    db: &'db dyn MirDb,
    arg: &SEffectArg<'db>,
    carriers: &[RuntimeCarrier<'db>],
) -> Option<RuntimeClass<'db>> {
    match arg.pass_mode {
        EffectPassMode::ByValue | EffectPassMode::Unknown => match arg.arg {
            SEffectArgValue::Place(_) => Some(provider_class_for_target(
                db,
                arg.target_ty,
                effect_arg_address_space(arg),
            )),
            SEffectArgValue::Value(value) => match carriers.get(value.index())? {
                RuntimeCarrier::Erased => None,
                RuntimeCarrier::Value(class) => Some(class.clone()),
            },
        },
        EffectPassMode::ByPlace | EffectPassMode::ByTempPlace => Some(provider_class_for_target(
            db,
            arg.target_ty,
            effect_arg_address_space(arg),
        )),
    }
}

fn runtime_return_class_cycle_initial<'db>(
    db: &'db dyn MirDb,
    key: RuntimeInstanceKey<'db>,
) -> Option<RuntimeClass<'db>> {
    let typed_body = key.semantic(db).key(db).instantiate_typed_body(db);
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

pub(super) fn runtime_param_class<'db>(
    db: &'db dyn MirDb,
    typed_body: &hir::analysis::ty::ty_check::TypedBody<'db>,
    binding: hir::analysis::ty::ty_check::LocalBinding<'db>,
    actual: RuntimeClass<'db>,
) -> RuntimeClass<'db> {
    let ty = typed_body.binding_ty(db, binding);
    if binding.is_mut() && ty.as_enum(db).is_some() {
        return RuntimeClass::Handle {
            layout: layout_for_ty(db, ty),
            kind: HandleKind::ObjectValue,
            view: HandleView::Whole,
        };
    }
    actual
}

pub(super) fn semantic_return_ty<'db>(
    db: &'db dyn MirDb,
    semantic: SemanticInstance<'db>,
) -> TyId<'db> {
    let typed_body = semantic.key(db).instantiate_typed_body(db);
    typed_body
        .body()
        .map(|body| typed_body.expr_ty(db, body.expr(db)))
        .unwrap_or_else(|| TyId::unit(db))
}

fn default_return_class<'db>(
    db: &'db dyn MirDb,
    typed_body: &hir::analysis::ty::ty_check::TypedBody<'db>,
) -> Option<RuntimeClass<'db>> {
    let default_space = typed_body
        .return_borrow_provider()
        .map_or(AddressSpaceKind::Memory, address_space_from_provider);
    let body = typed_body.body()?;
    top_level_class_for_ty(db, typed_body.expr_ty(db, body.expr(db)), default_space)
}

pub(super) fn top_level_class_for_ty<'db>(
    db: &'db dyn MirDb,
    ty: TyId<'db>,
    default_space: AddressSpaceKind,
) -> Option<RuntimeClass<'db>> {
    if ty == TyId::unit(db) || ty.is_zero_sized(db) {
        return None;
    }
    if let Some(scalar) = scalar_class_for_ty(db, ty) {
        return Some(RuntimeClass::Scalar(scalar));
    }
    if let Some((_, inner)) = ty.as_borrow(db).or_else(|| {
        ty.as_view(db)
            .map(|inner| (hir::analysis::ty::ty_def::BorrowKind::Ref, inner))
    }) {
        return Some(provider_class_for_target(db, Some(inner), default_space));
    }
    if let Some((_, inner)) = ty.as_capability(db) {
        return Some(provider_class_for_target(db, Some(inner), default_space));
    }
    if ty.as_enum(db).is_some() {
        return Some(RuntimeClass::AggregateValue {
            layout: layout_for_ty(db, ty),
        });
    }
    if ty.is_struct(db) || ty.is_array(db) || ty.is_tuple(db) {
        return Some(RuntimeClass::Handle {
            layout: layout_for_ty(db, ty),
            kind: HandleKind::ObjectValue,
            view: HandleView::Whole,
        });
    }
    None
}

pub(super) fn stored_class_for_ty<'db>(db: &'db dyn MirDb, ty: TyId<'db>) -> RuntimeClass<'db> {
    if let Some(scalar) = scalar_class_for_ty(db, ty) {
        return RuntimeClass::Scalar(scalar);
    }
    if let Some((_, inner)) = ty.as_capability(db) {
        return provider_class_for_target(db, Some(inner), AddressSpaceKind::Memory);
    }
    RuntimeClass::AggregateValue {
        layout: layout_for_ty(db, ty),
    }
}

pub(super) fn provider_class_for_target<'db>(
    db: &'db dyn MirDb,
    target_ty: Option<TyId<'db>>,
    space: AddressSpaceKind,
) -> RuntimeClass<'db> {
    match target_ty {
        Some(target_ty)
            if target_ty.is_struct(db)
                || target_ty.is_array(db)
                || target_ty.is_tuple(db)
                || target_ty.as_enum(db).is_some() =>
        {
            RuntimeClass::Handle {
                layout: layout_for_ty(db, target_ty),
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

pub(super) fn scalar_class_for_ty<'db>(
    db: &'db dyn MirDb,
    ty: TyId<'db>,
) -> Option<ScalarClass<'db>> {
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
