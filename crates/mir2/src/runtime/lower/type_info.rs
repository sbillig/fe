use common::indexmap::IndexSet;
use hir::analysis::ty::{
    ProviderAddressSpace, ProviderKind,
    normalize::normalize_ty,
    provider::provider_semantics,
    trait_resolution::PredicateListId,
    ty_def::{BorrowKind, MAX_INLINE_STRING_BYTES, PrimTy, TyBase, TyData, TyId},
};
use salsa::Update;

use crate::{
    db::MirDb,
    runtime::{
        AddressSpaceKind, BorrowAccess, BorrowTransportSet, RefKind, RefView, RuntimeBoundarySpec,
        RuntimeClass, ScalarClass, ScalarRepr, ScalarRole,
    },
};

use super::layout::layout_for_ty_in_context;

#[derive(Clone, Copy)]
pub(crate) struct RuntimeTypeEnv<'db> {
    pub(crate) scope: Option<hir::hir_def::scope_graph::ScopeId<'db>>,
    pub(crate) assumptions: PredicateListId<'db>,
}

impl<'db> RuntimeTypeEnv<'db> {
    pub(crate) fn new(
        scope: Option<hir::hir_def::scope_graph::ScopeId<'db>>,
        assumptions: PredicateListId<'db>,
    ) -> Self {
        Self { scope, assumptions }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Update)]
struct RuntimeEffectHandleInfo<'db> {
    target_ty: TyId<'db>,
    space: AddressSpaceKind,
}

pub(crate) fn runtime_repr_ty_in_context<'db>(
    db: &'db dyn MirDb,
    ty: TyId<'db>,
    scope: Option<hir::hir_def::scope_graph::ScopeId<'db>>,
    assumptions: PredicateListId<'db>,
) -> TyId<'db> {
    let mut ty = scope.map_or(ty, |scope| normalize_ty(db, ty, scope, assumptions));
    while let Some(inner) = ty.as_view(db) {
        ty = scope.map_or(inner, |scope| normalize_ty(db, inner, scope, assumptions));
    }
    ty
}

pub(crate) fn is_zero_sized_in_context<'db>(
    db: &'db dyn MirDb,
    ty: TyId<'db>,
    scope: Option<hir::hir_def::scope_graph::ScopeId<'db>>,
    assumptions: PredicateListId<'db>,
) -> bool {
    runtime_zero_sized_ty(db, ty, scope, assumptions)
}

#[salsa::tracked(
    cycle_fn=runtime_zero_sized_ty_cycle_recover,
    cycle_initial=runtime_zero_sized_ty_cycle_initial
)]
fn runtime_zero_sized_ty<'db>(
    db: &'db dyn MirDb,
    ty: TyId<'db>,
    scope: Option<hir::hir_def::scope_graph::ScopeId<'db>>,
    assumptions: PredicateListId<'db>,
) -> bool {
    let repr_ty = runtime_repr_ty_in_context(db, ty, scope, assumptions);
    if repr_ty != ty {
        return runtime_zero_sized_ty(db, repr_ty, scope, assumptions);
    }
    if repr_ty.is_never(db)
        || matches!(
            repr_ty.base_ty(db).data(db),
            TyData::TyBase(hir::analysis::ty::ty_def::TyBase::Func(_))
        )
    {
        return true;
    }
    if repr_ty.is_array(db) {
        let (_, args) = repr_ty.decompose_ty_app(db);
        return repr_ty.array_len(db).is_some_and(|len| {
            len == 0
                || args
                    .first()
                    .copied()
                    .is_some_and(|elem| runtime_zero_sized_ty(db, elem, scope, assumptions))
        });
    }
    if repr_ty.is_tuple(db) || repr_ty.is_struct(db) {
        return repr_ty
            .field_types(db)
            .into_iter()
            .all(|field| runtime_zero_sized_ty(db, field, scope, assumptions));
    }
    false
}

pub(crate) fn top_level_class_for_ty_in_env<'db>(
    db: &'db dyn MirDb,
    env: RuntimeTypeEnv<'db>,
    ty: TyId<'db>,
    default_space: AddressSpaceKind,
) -> Option<RuntimeClass<'db>> {
    top_level_class_for_ty_in_context(db, ty, default_space, env.scope, env.assumptions)
}

pub(crate) fn boundary_spec_for_ty_in_env<'db>(
    db: &'db dyn MirDb,
    env: RuntimeTypeEnv<'db>,
    ty: TyId<'db>,
    default_space: AddressSpaceKind,
) -> Option<RuntimeBoundarySpec<'db>> {
    boundary_spec_for_ty_in_context(db, ty, default_space, env.scope, env.assumptions)
}

pub(crate) fn provider_class_for_target_in_env<'db>(
    db: &'db dyn MirDb,
    env: RuntimeTypeEnv<'db>,
    target_ty: Option<TyId<'db>>,
    space: AddressSpaceKind,
) -> RuntimeClass<'db> {
    provider_class_for_target_in_context(db, target_ty, space, env.scope, env.assumptions)
}

pub(crate) fn scalar_class_for_ty_in_env<'db>(
    db: &'db dyn MirDb,
    env: RuntimeTypeEnv<'db>,
    ty: TyId<'db>,
) -> Option<ScalarClass<'db>> {
    scalar_class_for_ty_in_context(db, ty, env.scope, env.assumptions)
}

pub(crate) fn boundary_spec_for_ty_in_context<'db>(
    db: &'db dyn MirDb,
    ty: TyId<'db>,
    default_space: AddressSpaceKind,
    scope: Option<hir::hir_def::scope_graph::ScopeId<'db>>,
    assumptions: PredicateListId<'db>,
) -> Option<RuntimeBoundarySpec<'db>> {
    runtime_boundary_spec(db, ty, default_space, scope, assumptions)
}

pub(crate) fn default_borrow_transport_set(
    access: BorrowAccess,
    default_space: AddressSpaceKind,
) -> BorrowTransportSet {
    let mut provider_spaces = IndexSet::new();
    provider_spaces.insert(default_space);
    provider_spaces.insert(AddressSpaceKind::Memory);
    provider_spaces.insert(AddressSpaceKind::Storage);
    provider_spaces.insert(AddressSpaceKind::Transient);
    if matches!(access, BorrowAccess::ReadOnly) {
        provider_spaces.insert(AddressSpaceKind::Calldata);
    }
    BorrowTransportSet {
        allow_object: true,
        allow_const: matches!(access, BorrowAccess::ReadOnly),
        provider_spaces: provider_spaces.into_iter().collect(),
        allow_raw_addr: true,
    }
}

pub(crate) fn aggregate_transport_depends_on_runtime_source<'db>(
    db: &'db dyn MirDb,
    ty: TyId<'db>,
    scope: Option<hir::hir_def::scope_graph::ScopeId<'db>>,
    assumptions: PredicateListId<'db>,
) -> bool {
    runtime_transport_sensitive_aggregate(db, ty, scope, assumptions)
}

pub(crate) fn boundary_source_uses_transport_sensitive_aggregate<'db>(
    db: &'db dyn MirDb,
    ty: TyId<'db>,
    scope: Option<hir::hir_def::scope_graph::ScopeId<'db>>,
    assumptions: PredicateListId<'db>,
) -> bool {
    runtime_boundary_source_uses_transport_sensitive_aggregate(db, ty, scope, assumptions)
}

pub(crate) fn top_level_class_for_ty_in_context<'db>(
    db: &'db dyn MirDb,
    ty: TyId<'db>,
    default_space: AddressSpaceKind,
    scope: Option<hir::hir_def::scope_graph::ScopeId<'db>>,
    assumptions: PredicateListId<'db>,
) -> Option<RuntimeClass<'db>> {
    runtime_top_level_class(db, ty, default_space, scope, assumptions)
}

pub(crate) fn stored_class_for_ty_in_context<'db>(
    db: &'db dyn MirDb,
    ty: TyId<'db>,
    scope: Option<hir::hir_def::scope_graph::ScopeId<'db>>,
    assumptions: PredicateListId<'db>,
) -> RuntimeClass<'db> {
    let ty = runtime_repr_ty_in_context(db, ty, scope, assumptions);
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
    if let Some(scalar) = scalar_class_for_ty_in_context(db, ty, scope, assumptions) {
        return RuntimeClass::Scalar(scalar);
    }
    RuntimeClass::AggregateValue {
        layout: layout_for_ty_in_context(db, ty, scope, assumptions),
    }
}

pub(crate) fn object_ref_class_for_target_in_context<'db>(
    db: &'db dyn MirDb,
    target_ty: TyId<'db>,
    scope: Option<hir::hir_def::scope_graph::ScopeId<'db>>,
    assumptions: PredicateListId<'db>,
) -> RuntimeClass<'db> {
    let target_ty = runtime_repr_ty_in_context(db, target_ty, scope, assumptions);
    RuntimeClass::Ref {
        pointee: Box::new(stored_class_for_ty_in_context(
            db,
            target_ty,
            scope,
            assumptions,
        )),
        kind: RefKind::Object,
        view: RefView::Whole,
    }
}

pub(crate) fn provider_class_for_target_in_context<'db>(
    db: &'db dyn MirDb,
    target_ty: Option<TyId<'db>>,
    space: AddressSpaceKind,
    scope: Option<hir::hir_def::scope_graph::ScopeId<'db>>,
    assumptions: PredicateListId<'db>,
) -> RuntimeClass<'db> {
    match target_ty.map(|ty| runtime_repr_ty_in_context(db, ty, scope, assumptions)) {
        Some(target_ty) => RuntimeClass::Ref {
            pointee: Box::new(stored_class_for_ty_in_context(
                db,
                target_ty,
                scope,
                assumptions,
            )),
            kind: RefKind::Provider {
                provider_ty: TyId::borrow_ref_of(db, target_ty),
                space,
            },
            view: RefView::Whole,
        },
        None => RuntimeClass::RawAddr {
            space,
            target: None,
        },
    }
}

pub(crate) fn scalar_class_for_ty_in_context<'db>(
    db: &'db dyn MirDb,
    ty: TyId<'db>,
    scope: Option<hir::hir_def::scope_graph::ScopeId<'db>>,
    assumptions: PredicateListId<'db>,
) -> Option<ScalarClass<'db>> {
    let ty = runtime_repr_ty_in_context(db, ty, scope, assumptions);
    scalar_class_from_repr_ty(db, ty)
}

pub(crate) fn provider_address_space_to_runtime(space: ProviderAddressSpace) -> AddressSpaceKind {
    match space {
        ProviderAddressSpace::Memory => AddressSpaceKind::Memory,
        ProviderAddressSpace::Storage => AddressSpaceKind::Storage,
        ProviderAddressSpace::Transient => AddressSpaceKind::Transient,
        ProviderAddressSpace::Calldata => AddressSpaceKind::Calldata,
    }
}

fn scalar_class_from_repr_ty<'db>(db: &'db dyn MirDb, ty: TyId<'db>) -> Option<ScalarClass<'db>> {
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
    let ty = runtime_repr_ty_in_context(db, ty, scope, assumptions);
    let scope = scope.or_else(|| ty.as_scope(db))?;
    let info = runtime_effect_handle_info(db, ty, Some(scope), assumptions)?;
    Some(provider_class_for_target_in_context(
        db,
        Some(info.target_ty),
        info.space,
        Some(scope),
        assumptions,
    ))
}

#[salsa::tracked]
fn runtime_boundary_spec<'db>(
    db: &'db dyn MirDb,
    ty: TyId<'db>,
    default_space: AddressSpaceKind,
    scope: Option<hir::hir_def::scope_graph::ScopeId<'db>>,
    assumptions: PredicateListId<'db>,
) -> Option<RuntimeBoundarySpec<'db>> {
    let repr_ty = runtime_repr_ty_in_context(db, ty, scope, assumptions);
    if repr_ty != ty {
        return runtime_boundary_spec(db, repr_ty, default_space, scope, assumptions);
    }
    if repr_ty == TyId::unit(db) || is_zero_sized_in_context(db, repr_ty, scope, assumptions) {
        return None;
    }
    if let Some((kind, inner)) = repr_ty.as_borrow(db) {
        if is_zero_sized_in_context(db, inner, scope, assumptions) {
            return None;
        }
        let access = match kind {
            BorrowKind::Ref => BorrowAccess::ReadOnly,
            BorrowKind::Mut => BorrowAccess::ReadWrite,
        };
        return Some(RuntimeBoundarySpec::BorrowLike {
            pointee: stored_class_for_ty_in_context(db, inner, scope, assumptions),
            access,
            allow: default_borrow_transport_set(access, default_space),
        });
    }
    runtime_top_level_class(db, repr_ty, default_space, scope, assumptions)
        .map(RuntimeBoundarySpec::Exact)
}

#[salsa::tracked]
fn runtime_top_level_class<'db>(
    db: &'db dyn MirDb,
    ty: TyId<'db>,
    default_space: AddressSpaceKind,
    scope: Option<hir::hir_def::scope_graph::ScopeId<'db>>,
    assumptions: PredicateListId<'db>,
) -> Option<RuntimeClass<'db>> {
    let repr_ty = runtime_repr_ty_in_context(db, ty, scope, assumptions);
    if repr_ty != ty {
        return runtime_top_level_class(db, repr_ty, default_space, scope, assumptions);
    }
    if repr_ty == TyId::unit(db) || is_zero_sized_in_context(db, repr_ty, scope, assumptions) {
        return None;
    }
    if let Some((_, inner)) = repr_ty.as_borrow(db) {
        if is_zero_sized_in_context(db, inner, scope, assumptions) {
            return None;
        }
        return Some(object_ref_class_for_target_in_context(
            db,
            inner,
            scope,
            assumptions,
        ));
    }
    if let Some((_, inner)) = repr_ty.as_capability(db) {
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
    if let Some(class) = effect_handle_class_for_ty(db, repr_ty, scope, assumptions) {
        return Some(class);
    }
    if let Some(scalar) = scalar_class_from_repr_ty(db, repr_ty) {
        return Some(RuntimeClass::Scalar(scalar));
    }
    if repr_ty.as_enum(db).is_some()
        || repr_ty.is_struct(db)
        || repr_ty.is_array(db)
        || repr_ty.is_tuple(db)
    {
        return Some(RuntimeClass::AggregateValue {
            layout: layout_for_ty_in_context(db, repr_ty, scope, assumptions),
        });
    }
    None
}

#[salsa::tracked]
fn runtime_effect_handle_info<'db>(
    db: &'db dyn MirDb,
    ty: TyId<'db>,
    scope: Option<hir::hir_def::scope_graph::ScopeId<'db>>,
    assumptions: PredicateListId<'db>,
) -> Option<RuntimeEffectHandleInfo<'db>> {
    let repr_ty = runtime_repr_ty_in_context(db, ty, scope, assumptions);
    if repr_ty != ty {
        return runtime_effect_handle_info(db, repr_ty, scope, assumptions);
    }
    let scope = scope.or_else(|| repr_ty.as_scope(db))?;
    let semantics = provider_semantics(db, scope, assumptions, repr_ty);
    if matches!(semantics.kind, ProviderKind::RootObject) {
        return None;
    }
    let target_ty = semantics.target_ty?;
    if is_zero_sized_in_context(db, target_ty, Some(scope), assumptions) {
        return None;
    }
    Some(RuntimeEffectHandleInfo {
        target_ty,
        space: provider_address_space_to_runtime(semantics.address_space?),
    })
}

#[salsa::tracked]
fn runtime_boundary_source_uses_transport_sensitive_aggregate<'db>(
    db: &'db dyn MirDb,
    ty: TyId<'db>,
    scope: Option<hir::hir_def::scope_graph::ScopeId<'db>>,
    assumptions: PredicateListId<'db>,
) -> bool {
    if let Some((_, inner)) = ty.as_borrow(db) {
        return runtime_transport_sensitive_aggregate(db, inner, scope, assumptions);
    }
    let repr_ty = runtime_repr_ty_in_context(db, ty, scope, assumptions);
    if let Some((_, inner)) = repr_ty.as_borrow(db) {
        return runtime_transport_sensitive_aggregate(db, inner, scope, assumptions);
    }
    runtime_transport_sensitive_aggregate(db, repr_ty, scope, assumptions)
}

#[salsa::tracked(
    cycle_fn=runtime_transport_sensitive_aggregate_cycle_recover,
    cycle_initial=runtime_transport_sensitive_aggregate_cycle_initial
)]
fn runtime_transport_sensitive_aggregate<'db>(
    db: &'db dyn MirDb,
    ty: TyId<'db>,
    scope: Option<hir::hir_def::scope_graph::ScopeId<'db>>,
    assumptions: PredicateListId<'db>,
) -> bool {
    let repr_ty = runtime_repr_ty_in_context(db, ty, scope, assumptions);
    if repr_ty != ty {
        return runtime_transport_sensitive_aggregate(db, repr_ty, scope, assumptions);
    }
    if repr_ty.as_borrow(db).is_some() {
        return true;
    }
    if repr_ty.as_capability(db).is_some()
        || runtime_effect_handle_info(db, repr_ty, scope, assumptions).is_some()
        || scalar_class_from_repr_ty(db, repr_ty).is_some()
    {
        return false;
    }
    if repr_ty.is_array(db) {
        let (_, args) = repr_ty.decompose_ty_app(db);
        return args.first().copied().is_some_and(|elem| {
            runtime_transport_sensitive_aggregate(db, elem, scope, assumptions)
        });
    }
    if repr_ty.is_tuple(db) || repr_ty.is_struct(db) {
        return repr_ty
            .field_types(db)
            .into_iter()
            .any(|field| runtime_transport_sensitive_aggregate(db, field, scope, assumptions));
    }
    if let Some(enum_) = repr_ty.as_enum(db) {
        let adt = enum_.as_adt(db);
        let args = repr_ty.generic_args(db);
        return adt
            .fields(db)
            .iter()
            .enumerate()
            .any(|(variant_idx, variant)| {
                (0..variant.num_types()).any(|field_idx| {
                    runtime_transport_sensitive_aggregate(
                        db,
                        adt.fields(db)[variant_idx]
                            .ty(db, field_idx)
                            .instantiate(db, args),
                        scope,
                        assumptions,
                    )
                })
            });
    }
    false
}

fn runtime_zero_sized_ty_cycle_initial<'db>(
    _db: &'db dyn MirDb,
    _ty: TyId<'db>,
    _scope: Option<hir::hir_def::scope_graph::ScopeId<'db>>,
    _assumptions: PredicateListId<'db>,
) -> bool {
    false
}

fn runtime_zero_sized_ty_cycle_recover<'db>(
    _db: &'db dyn MirDb,
    _value: &bool,
    _count: u32,
    _ty: TyId<'db>,
    _scope: Option<hir::hir_def::scope_graph::ScopeId<'db>>,
    _assumptions: PredicateListId<'db>,
) -> salsa::CycleRecoveryAction<bool> {
    salsa::CycleRecoveryAction::Iterate
}

fn runtime_transport_sensitive_aggregate_cycle_initial<'db>(
    _db: &'db dyn MirDb,
    _ty: TyId<'db>,
    _scope: Option<hir::hir_def::scope_graph::ScopeId<'db>>,
    _assumptions: PredicateListId<'db>,
) -> bool {
    false
}

fn runtime_transport_sensitive_aggregate_cycle_recover<'db>(
    _db: &'db dyn MirDb,
    _value: &bool,
    _count: u32,
    _ty: TyId<'db>,
    _scope: Option<hir::hir_def::scope_graph::ScopeId<'db>>,
    _assumptions: PredicateListId<'db>,
) -> salsa::CycleRecoveryAction<bool> {
    salsa::CycleRecoveryAction::Iterate
}
