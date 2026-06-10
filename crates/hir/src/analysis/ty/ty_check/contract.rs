//! Contract-specific type checking functions.
//!
//! This module contains functions for checking contract init bodies,
//! recv blocks, and recv arm bodies.
use std::hash::Hash;

use num_bigint::BigUint;
use rustc_hash::{FxHashMap, FxHashSet};

use super::{
    EffectArg, EffectParamSite, LocalBinding, ParamSite, ResolvedEffectArg, TypedBody,
    check_func_body, owner::BodyOwner,
};

use num_traits::ToPrimitive;

use crate::{
    analysis::{
        HirAnalysisDb,
        name_resolution::{ExpectedPathKind, PathRes, diagnostics::PathResDiag, resolve_path},
        place::{Place, PlaceBase},
        semantic::{SemConstScalar, SemConstValue, eval_body_owner_const},
        ty::{
            adt_def::AdtRef,
            canonical::Canonical,
            corelib::resolve_core_trait,
            diagnostics::{BodyDiag, FuncBodyDiag, TraitConstraintDiag, TyDiagCollection},
            provider::{ProviderAddressSpace, address_space_from_ty},
            trait_def::TraitInstId,
            trait_def::impls_for_ty,
            trait_resolution::{
                GoalSatisfiability, PredicateListId, TraitSolveCx, is_goal_satisfiable,
            },
            ty_check::check_body,
            ty_def::{PrimTy, TyBase, TyData, TyId},
        },
    },
    hir_def::{
        ArithBinOp, BinOp, Body, CallableDef, CompBinOp, Cond, Contract, Expr, ExprId, FieldParent,
        Func, IdentId, ItemKind, LitKind, LogicalBinOp, Mod, Partial, PatId, PathId, Stmt, Struct,
        scope_graph::ScopeId,
    },
    semantic::{EffectEnvView, FieldView, ProviderSource},
    span::{DynLazySpan, path::LazyPathSpan},
};
use common::{indexmap::IndexMap, ingot::IngotKind};

#[allow(clippy::enum_variant_names)]
pub enum VariantResError<'db> {
    /// Path doesn't resolve at all.
    NotFound,
    /// Path resolves to a type that doesn't implement MsgVariant.
    NotMsgVariant(TyId<'db>),
    /// Path resolves to a type that implements MsgVariant but is not a variant
    /// of the specified msg module.
    NotVariantOfMsg(TyId<'db>),
}

/// Returns true if a struct implements the core MsgVariant trait.
fn implements_msg_variant<'db>(db: &'db dyn HirAnalysisDb, struct_: Struct<'db>) -> bool {
    let Some(msg_variant_trait) =
        resolve_core_trait(db, struct_.scope(), &["message", "MsgVariant"])
    else {
        return false;
    };

    let adt_def = AdtRef::from(struct_).as_adt(db);
    let ty = TyId::adt(db, adt_def);
    let canonical_ty = Canonical::new(db, ty);
    let ingot = struct_.top_mod(db).ingot(db);

    impls_for_ty(db, ingot, canonical_ty)
        .iter()
        .any(|impl_| impl_.skip_binder().trait_def(db).eq(&msg_variant_trait))
}

fn resolve_sol_abi_ty<'db>(
    db: &'db dyn HirAnalysisDb,
    scope: ScopeId<'db>,
    assumptions: PredicateListId<'db>,
) -> Option<TyId<'db>> {
    let ingot = scope.ingot(db);
    let std_root = if ingot.kind(db) == IngotKind::Std {
        IdentId::make_ingot(db)
    } else {
        IdentId::new(db, "std".to_string())
    };

    let sol_path = PathId::from_ident(db, std_root)
        .push_ident(db, IdentId::new(db, "abi".to_string()))
        .push_ident(db, IdentId::new(db, "Sol".to_string()));

    match resolve_path(db, sol_path, scope, assumptions, false).ok()? {
        PathRes::Ty(ty) | PathRes::TyAlias(_, ty) => Some(ty),
        _ => None,
    }
}

#[allow(clippy::too_many_arguments)]
fn check_ty_decodable<'db>(
    db: &'db dyn HirAnalysisDb,
    solve_cx: TraitSolveCx<'db>,
    decode_trait: crate::hir_def::Trait<'db>,
    sol_ty: TyId<'db>,
    ty: TyId<'db>,
    span: DynLazySpan<'db>,
    diags: &mut Vec<FuncBodyDiag<'db>>,
) {
    if ty.has_invalid(db) {
        return;
    }

    if ty.is_tuple(db) {
        for elem in ty.field_types(db) {
            check_ty_decodable(
                db,
                solve_cx,
                decode_trait,
                sol_ty,
                elem,
                span.clone(),
                diags,
            );
        }
        return;
    }

    if ty.has_var(db) {
        return;
    }

    let inst = TraitInstId::new(db, decode_trait, vec![ty, sol_ty], IndexMap::new());
    if let GoalSatisfiability::UnSat(_) = is_goal_satisfiable(db, solve_cx, inst) {
        diags.push(
            TyDiagCollection::from(TraitConstraintDiag::TraitBoundNotSat {
                span,
                primary_goal: inst,
                unsat_subgoal: None,
                required_by: None,
            })
            .into(),
        );
    }
}

fn check_recv_variant_param_types_decodable<'db>(
    db: &'db dyn HirAnalysisDb,
    contract: Contract<'db>,
    variant: ResolvedRecvVariant<'db>,
    span: DynLazySpan<'db>,
    assumptions: PredicateListId<'db>,
    diags: &mut Vec<FuncBodyDiag<'db>>,
) {
    let Some(sol_ty) = resolve_sol_abi_ty(db, contract.scope(), assumptions) else {
        return;
    };
    let Some(decode_trait) = resolve_core_trait(db, contract.scope(), &["abi", "Decode"]) else {
        return;
    };
    let solve_cx = TraitSolveCx::new(db, contract.scope()).with_assumptions(assumptions);

    for field_ty in variant.ty.field_types(db) {
        check_ty_decodable(
            db,
            solve_cx,
            decode_trait,
            sol_ty,
            field_ty,
            span.clone(),
            diags,
        );
    }
}

/// Returns all variant structs in a msg module (structs that implement MsgVariant).
fn msg_variants<'db>(
    db: &'db dyn HirAnalysisDb,
    msg_mod: Mod<'db>,
) -> impl Iterator<Item = Struct<'db>> + 'db {
    msg_mod
        .children_non_nested(db)
        .filter_map(|item| match item {
            ItemKind::Struct(s) => Some(s),
            _ => None,
        })
        .filter(move |s| implements_msg_variant(db, *s))
}

/// Resolved msg variant in a recv arm.
#[derive(Debug, Clone, Copy)]
pub struct ResolvedRecvVariant<'db> {
    pub variant_struct: Struct<'db>,
    pub ty: TyId<'db>,
}

/// Resolves a variant path within a msg module.
pub fn resolve_variant_in_msg<'db>(
    db: &'db dyn HirAnalysisDb,
    msg_mod: Mod<'db>,
    variant_path: PathId<'db>,
    assumptions: PredicateListId<'db>,
) -> Result<ResolvedRecvVariant<'db>, VariantResError<'db>> {
    let Ok(PathRes::Ty(ty)) = resolve_path(db, variant_path, msg_mod.scope(), assumptions, false)
    else {
        return Err(VariantResError::NotFound);
    };

    if let Some(adt_def) = ty.adt_def(db)
        && let AdtRef::Struct(struct_) = adt_def.adt_ref(db)
        && implements_msg_variant(db, struct_)
    {
        if let Some(parent) = struct_.scope().parent(db)
            && parent == ScopeId::Item(ItemKind::Mod(msg_mod))
        {
            return Ok(ResolvedRecvVariant {
                variant_struct: struct_,
                ty,
            });
        }
        return Err(VariantResError::NotVariantOfMsg(ty));
    }
    // Resolved to a type but it doesn't implement MsgVariant
    Err(VariantResError::NotMsgVariant(ty))
}

/// Resolves a variant path in a bare recv block (no msg module specified).
/// Paths are resolved from the contract's scope.
pub fn resolve_variant_bare<'db>(
    db: &'db dyn HirAnalysisDb,
    contract: Contract<'db>,
    variant_path: PathId<'db>,
    assumptions: PredicateListId<'db>,
) -> Result<ResolvedRecvVariant<'db>, VariantResError<'db>> {
    match resolve_path(db, variant_path, contract.scope(), assumptions, false) {
        Ok(PathRes::Ty(ty)) => {
            if let Some(adt_def) = ty.adt_def(db)
                && let AdtRef::Struct(s) = adt_def.adt_ref(db)
                && implements_msg_variant(db, s)
            {
                return Ok(ResolvedRecvVariant {
                    variant_struct: s,
                    ty,
                });
            }
            // Resolved to a type but it doesn't implement MsgVariant
            Err(VariantResError::NotMsgVariant(ty))
        }
        _ => Err(VariantResError::NotFound),
    }
}

#[salsa::tracked(return_ref)]
pub fn check_contract_recv_block<'db>(
    db: &'db dyn HirAnalysisDb,
    contract: Contract<'db>,
    recv_idx: u32,
) -> Vec<FuncBodyDiag<'db>> {
    let mut diags = Vec::new();

    let Some(recv) = contract.recvs(db).data(db).get(recv_idx as usize) else {
        return diags;
    };

    let recv_span = contract.span().recv(recv_idx as usize);
    let path_span = recv_span.clone().path();

    // Check if this is a named recv block (recv MsgType { ... }) or bare (recv { ... })
    if let Some(msg_mod) = resolve_recv_msg_mod(
        db,
        contract,
        recv.msg_path,
        path_span.clone(),
        &mut diags,
        true,
    ) {
        // Named recv block - validate against the specific msg module
        check_named_recv_block(db, contract, recv_idx, msg_mod, &mut diags);
    } else if recv.msg_path.is_none() {
        // Bare recv block - no msg module specified
        check_bare_recv_block(db, contract, recv_idx, &mut diags);
    }
    // If msg_path was Some but didn't resolve, diagnostics were already emitted

    diags
}

/// Check a named recv block (recv MsgType { ... }).
/// All variants must be children of the specified msg module.
fn check_named_recv_block<'db>(
    db: &'db dyn HirAnalysisDb,
    contract: Contract<'db>,
    recv_idx: u32,
    msg_mod: Mod<'db>,
    diags: &mut Vec<FuncBodyDiag<'db>>,
) {
    let recv = &contract.recvs(db).data(db)[recv_idx as usize];
    let recv_span = contract.span().recv(recv_idx as usize);
    let assumptions = PredicateListId::empty_list(db);

    // Use TyId for duplicate detection to correctly handle generic types
    let mut seen = FxHashMap::<TyId<'db>, DynLazySpan<'db>>::default();
    // Use Struct for exhaustiveness checking (tracks which base structs are covered)
    let mut covered = FxHashSet::<Struct<'db>>::default();
    let mut checked_decode = FxHashSet::<Struct<'db>>::default();

    // Get msg name for diagnostics
    let Some(msg_name) = msg_mod.name(db).to_opt() else {
        return;
    };

    for (arm_idx, arm) in recv.arms.data(db).iter().enumerate() {
        let arm_span = recv_span.clone().arms().arm(arm_idx);
        let pat_span: DynLazySpan<'db> = arm_span.clone().pat().into();

        if arm.is_fallback(db) {
            diags.push(
                BodyDiag::RecvFallbackOnlyInBareBlock {
                    primary: pat_span.clone(),
                }
                .into(),
            );
            if arm.ret_ty.is_some() {
                diags.push(
                    BodyDiag::RecvFallbackReturnTypeNotAllowed {
                        primary: arm_span.ret_ty().into(),
                    }
                    .into(),
                );
            }
            continue;
        }

        let Some(path) = arm.variant_path(db) else {
            continue;
        };

        match resolve_variant_in_msg(db, msg_mod, path, assumptions) {
            Ok(resolved) => {
                let Some(ident) = resolved.variant_struct.name(db).to_opt() else {
                    continue;
                };

                if let Some(first_span) = seen.get(&resolved.ty) {
                    diags.push(
                        BodyDiag::RecvArmDuplicateVariant {
                            primary: pat_span.clone(),
                            first_use: first_span.clone(),
                            variant: ident,
                        }
                        .into(),
                    );
                } else {
                    seen.insert(resolved.ty, pat_span.clone());
                }

                covered.insert(resolved.variant_struct);
                if checked_decode.insert(resolved.variant_struct) {
                    check_recv_variant_param_types_decodable(
                        db,
                        contract,
                        resolved,
                        pat_span.clone(),
                        assumptions,
                        diags,
                    );
                }
            }
            Err(VariantResError::NotVariantOfMsg(ty)) => {
                // Type implements MsgVariant but is not a child of this msg module
                diags.push(
                    BodyDiag::RecvArmNotVariantOfMsg {
                        primary: pat_span,
                        variant_ty: ty,
                        msg_name,
                    }
                    .into(),
                );
            }
            Err(VariantResError::NotMsgVariant(ty)) => {
                // Type doesn't implement MsgVariant
                diags.push(
                    BodyDiag::RecvArmNotMsgVariantTrait {
                        primary: pat_span,
                        given_ty: ty,
                    }
                    .into(),
                );
            }
            Err(VariantResError::NotFound) => {
                // Path doesn't resolve at all - use the generic error
                diags.push(
                    BodyDiag::RecvArmNotMsgVariant {
                        primary: pat_span,
                        msg_name,
                    }
                    .into(),
                );
            }
        }
    }

    // Check for missing variants (exhaustiveness)
    let missing: Vec<_> = msg_variants(db, msg_mod)
        .filter_map(|variant| {
            if !covered.contains(&variant) {
                variant.name(db).to_opt()
            } else {
                None
            }
        })
        .collect();

    if !missing.is_empty() {
        diags.push(
            BodyDiag::RecvMissingMsgVariants {
                primary: recv_span.clone().path().into(),
                variants: missing,
            }
            .into(),
        );
    }
}

/// Check a bare recv block (recv { ... }).
/// Variants can be any type that implements MsgVariant.
fn check_bare_recv_block<'db>(
    db: &'db dyn HirAnalysisDb,
    contract: Contract<'db>,
    recv_idx: u32,
    diags: &mut Vec<FuncBodyDiag<'db>>,
) {
    let recv = &contract.recvs(db).data(db)[recv_idx as usize];
    let recv_span = contract.span().recv(recv_idx as usize);
    let assumptions = PredicateListId::empty_list(db);

    // Use TyId as key to correctly handle generic types like GenericMsg<u8> vs GenericMsg<u16>
    let mut seen = FxHashMap::<TyId<'db>, DynLazySpan<'db>>::default();
    let mut checked_decode = FxHashSet::<Struct<'db>>::default();

    for (arm_idx, arm) in recv.arms.data(db).iter().enumerate() {
        let arm_span = recv_span.clone().arms().arm(arm_idx);
        let pat_span: DynLazySpan<'db> = arm_span.clone().pat().into();

        if arm.is_fallback(db) {
            if arm.ret_ty.is_some() {
                diags.push(
                    BodyDiag::RecvFallbackReturnTypeNotAllowed {
                        primary: arm_span.ret_ty().into(),
                    }
                    .into(),
                );
            }
            continue;
        }

        let Some(path) = arm.variant_path(db) else {
            continue;
        };

        match resolve_variant_bare(db, contract, path, assumptions) {
            Ok(resolved) => {
                let Some(ident) = resolved.variant_struct.name(db).to_opt() else {
                    continue;
                };

                if let Some(first_span) = seen.get(&resolved.ty) {
                    diags.push(
                        BodyDiag::RecvArmDuplicateVariant {
                            primary: pat_span.clone(),
                            first_use: first_span.clone(),
                            variant: ident,
                        }
                        .into(),
                    );
                } else {
                    seen.insert(resolved.ty, pat_span.clone());
                }

                if checked_decode.insert(resolved.variant_struct) {
                    check_recv_variant_param_types_decodable(
                        db,
                        contract,
                        resolved,
                        pat_span.clone(),
                        assumptions,
                        diags,
                    );
                }
            }
            Err(VariantResError::NotMsgVariant(ty)) => {
                // Type doesn't implement MsgVariant
                diags.push(
                    BodyDiag::RecvArmNotMsgVariantTrait {
                        primary: pat_span,
                        given_ty: ty,
                    }
                    .into(),
                );
            }
            Err(VariantResError::NotVariantOfMsg(_)) => {
                // This shouldn't happen in bare recv blocks
                unreachable!("NotVariantOfMsg should not occur in bare recv blocks");
            }
            Err(VariantResError::NotFound) => {
                // Path doesn't resolve - this will be caught by name resolution
                // We don't emit a recv-specific error here
            }
        }
    }

    // No exhaustiveness check for bare recv blocks
}

#[salsa::tracked(return_ref)]
pub fn check_contract_recv_blocks<'db>(
    db: &'db dyn HirAnalysisDb,
    contract: Contract<'db>,
) -> Vec<FuncBodyDiag<'db>> {
    let mut diags = Vec::new();
    let mut seen_msg_blocks = FxHashMap::<Mod<'db>, (DynLazySpan<'db>, IdentId<'db>)>::default();
    let mut seen_fallback: Option<DynLazySpan<'db>> = None;

    // Track selectors across ALL recv blocks: selector -> (span, variant_name, struct)
    // We store the struct to correctly identify duplicates - comparing by name alone fails
    // when different msg blocks have variants with the same name but different selectors.
    let mut seen_selectors =
        FxHashMap::<u32, (DynLazySpan<'db>, IdentId<'db>, Struct<'db>)>::default();

    // Track handler types across ALL recv blocks for duplicate detection.
    // We use TyId to handle type aliases correctly.
    let mut seen_handlers = FxHashMap::<TyId<'db>, DynLazySpan<'db>>::default();

    let assumptions = PredicateListId::empty_list(db);

    for (idx, recv) in contract.recvs(db).data(db).iter().enumerate() {
        let recv_span = contract.span().recv(idx);
        let path_span = recv_span.clone().path();

        for (arm_idx, arm) in recv.arms.data(db).iter().enumerate() {
            if !arm.is_fallback(db) {
                continue;
            }

            let pat_span: DynLazySpan<'db> = recv_span.clone().arms().arm(arm_idx).pat().into();
            if let Some(first_use) = &seen_fallback {
                diags.push(
                    BodyDiag::RecvDuplicateFallback {
                        primary: pat_span,
                        first_use: first_use.clone(),
                    }
                    .into(),
                );
            } else {
                seen_fallback = Some(pat_span);
            }
        }

        // Check if this is a named recv block
        if let Some(msg_mod) = resolve_recv_msg_mod(
            db,
            contract,
            recv.msg_path,
            path_span.clone(),
            &mut diags,
            false,
        ) {
            let Some(msg_name) = msg_mod.name(db).to_opt() else {
                continue;
            };

            let path_span: DynLazySpan<'db> = path_span.into();
            let is_duplicate_msg_block = seen_msg_blocks.contains_key(&msg_mod);
            if is_duplicate_msg_block {
                if let Some((first_span, first_name)) = seen_msg_blocks.get(&msg_mod) {
                    diags.push(
                        BodyDiag::RecvDuplicateMsgBlock {
                            primary: path_span.clone(),
                            first_use: first_span.clone(),
                            msg_name: *first_name,
                        }
                        .into(),
                    );
                }
                // Skip handler/selector conflict checks for duplicate msg blocks
                continue;
            } else {
                seen_msg_blocks.insert(msg_mod, (path_span.clone(), msg_name));
            }

            // Check for selector and handler conflicts across all msg variants in this recv block
            for variant in msg_variants(db, msg_mod) {
                let Some(variant_name) = variant.name(db).to_opt() else {
                    continue;
                };

                let variant_span: DynLazySpan<'db> = variant.span().name().into();

                // Check selector conflicts
                let variant_ty = TyId::adt(db, AdtRef::from(variant).as_adt(db));
                if let Some(selector) =
                    eval_msg_variant_selector(db, variant_ty, variant.scope(), &mut diags)
                {
                    check_selector_conflict(
                        selector,
                        variant,
                        variant_name,
                        variant_span.clone(),
                        &mut seen_selectors,
                        &mut diags,
                    );
                }

                // Check handler type conflicts
                let adt_def = AdtRef::from(variant).as_adt(db);
                let ty = TyId::adt(db, adt_def);
                check_handler_conflict(ty, variant_span, &mut seen_handlers, &mut diags);
            }
        } else if recv.msg_path.is_none() {
            // Bare recv block - check each arm individually
            for (arm_idx, arm) in recv.arms.data(db).iter().enumerate() {
                let pat_span: DynLazySpan<'db> = recv_span.clone().arms().arm(arm_idx).pat().into();

                if arm.is_fallback(db) {
                    continue;
                }

                let Some(path) = arm.variant_path(db) else {
                    continue;
                };

                if let Ok(resolved) = resolve_variant_bare(db, contract, path, assumptions) {
                    let Some(variant_name) = resolved.variant_struct.name(db).to_opt() else {
                        continue;
                    };

                    // Check selector conflicts
                    if let Some(selector) =
                        eval_msg_variant_selector(db, resolved.ty, contract.scope(), &mut diags)
                    {
                        check_selector_conflict(
                            selector,
                            resolved.variant_struct,
                            variant_name,
                            pat_span.clone(),
                            &mut seen_selectors,
                            &mut diags,
                        );
                    }

                    // Check handler type conflicts
                    check_handler_conflict(resolved.ty, pat_span, &mut seen_handlers, &mut diags);
                }
            }
        }
    }

    diags
}

/// Check for selector conflicts and emit diagnostics if found.
fn check_selector_conflict<'db>(
    selector: u32,
    variant: Struct<'db>,
    variant_name: IdentId<'db>,
    variant_span: DynLazySpan<'db>,
    seen_selectors: &mut FxHashMap<u32, (DynLazySpan<'db>, IdentId<'db>, Struct<'db>)>,
    diags: &mut Vec<FuncBodyDiag<'db>>,
) {
    if let Some((first_span, first_variant, first_struct)) = seen_selectors.get(&selector) {
        // Don't report if it's the same variant (duplicate msg block already reported)
        if *first_struct != variant {
            diags.push(
                BodyDiag::RecvDuplicateSelector {
                    primary: variant_span,
                    first_use: first_span.clone(),
                    selector,
                    first_variant: *first_variant,
                    second_variant: variant_name,
                }
                .into(),
            );
        }
    } else {
        seen_selectors.insert(selector, (variant_span, variant_name, variant));
    }
}

/// Check for handler type conflicts and emit diagnostics if found.
fn check_handler_conflict<'db>(
    ty: TyId<'db>,
    variant_span: DynLazySpan<'db>,
    seen_handlers: &mut FxHashMap<TyId<'db>, DynLazySpan<'db>>,
    diags: &mut Vec<FuncBodyDiag<'db>>,
) {
    if let Some(first_span) = seen_handlers.get(&ty) {
        diags.push(
            BodyDiag::RecvDuplicateHandler {
                primary: variant_span,
                first_use: first_span.clone(),
                handler_ty: ty,
            }
            .into(),
        );
    } else {
        seen_handlers.insert(ty, variant_span);
    }
}

/// Evaluates a msg variant's `SELECTOR` associated const via CTFE.
pub(crate) fn eval_msg_variant_selector<'db>(
    db: &'db dyn HirAnalysisDb,
    variant_ty: TyId<'db>,
    scope: ScopeId<'db>,
    diags: &mut Vec<FuncBodyDiag<'db>>,
) -> Option<u32> {
    let msg_variant_trait = resolve_core_trait(db, scope, &["message", "MsgVariant"])?;

    let canonical_ty = Canonical::new(db, variant_ty);
    let scope_ingot = scope.ingot(db);
    let search_ingots = [
        Some(scope_ingot),
        variant_ty.ingot(db).filter(|&ingot| ingot != scope_ingot),
    ];
    let implementor = search_ingots.into_iter().flatten().find_map(|ingot| {
        impls_for_ty(db, ingot, canonical_ty)
            .iter()
            .find(|impl_| impl_.skip_binder().trait_def(db) == msg_variant_trait)
            .copied()
    })?;
    let impl_ = implementor.skip_binder();

    let selector_name = IdentId::new(db, "SELECTOR".to_string());
    let selector_const = impl_
        .hir_impl_trait(db)
        .hir_consts(db)
        .iter()
        .find(|c| c.name.to_opt() == Some(selector_name))?;

    let body = selector_const.value.to_opt()?;
    if matches!(
        body.expr(db).data(db, body),
        crate::hir_def::Partial::Absent
    ) {
        return None;
    }

    let expected_ty = TyId::new(db, TyData::TyBase(TyBase::Prim(PrimTy::U32)));
    let result = super::check_anon_const_body(db, body, expected_ty);
    diags.extend(result.0.clone());
    if !result.0.is_empty() {
        return None;
    }

    match eval_body_owner_const(
        db,
        BodyOwner::AnonConstBody {
            body,
            expected: expected_ty,
        },
        Vec::new(),
    ) {
        Ok(value) => match value.value(db) {
            SemConstValue::Scalar {
                value: SemConstScalar::Int { value },
                ..
            } => value.to_u32(),
            _ => {
                diags.push(BodyDiag::ConstValueMustBeKnown(body.span().into()).into());
                None
            }
        },
        Err(_) => {
            diags.push(BodyDiag::ConstValueMustBeKnown(body.span().into()).into());
            None
        }
    }
}

#[salsa::tracked(return_ref)]
pub fn check_contract_recv_arm_body<'db>(
    db: &'db dyn HirAnalysisDb,
    contract: Contract<'db>,
    recv_idx: u32,
    arm_idx: u32,
) -> (Vec<FuncBodyDiag<'db>>, TypedBody<'db>) {
    check_body(
        db,
        BodyOwner::ContractRecvArm {
            contract,
            recv_idx,
            arm_idx,
        },
    )
}

#[derive(Debug, Clone)]
struct AssignmentState<K> {
    assigned: FxHashSet<K>,
    values: FxHashMap<ValueFactBinding, ValueFact>,
}

impl<K> Default for AssignmentState<K> {
    fn default() -> Self {
        Self {
            assigned: FxHashSet::default(),
            values: FxHashMap::default(),
        }
    }
}

impl<K: Copy + Eq + Hash> AssignmentState<K> {
    fn intersection(lhs: &Self, rhs: &Self) -> Self {
        Self {
            assigned: lhs.assigned.intersection(&rhs.assigned).copied().collect(),
            values: lhs
                .values
                .iter()
                .filter(|(binding, value)| rhs.values.get(binding) == Some(*value))
                .map(|(binding, value)| (*binding, value.clone()))
                .collect(),
        }
    }

    fn without_values(mut self) -> Self {
        self.values.clear();
        self
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum ValueFactBinding {
    Local(PatId),
    Param(usize),
    Effect(usize),
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum ValueFact {
    Bool(bool),
    Int(BigUint),
}

#[derive(Debug, Clone)]
struct AssignmentFlow<K> {
    normal: Option<AssignmentState<K>>,
    returns: Vec<AssignmentState<K>>,
    breaks: Vec<AssignmentState<K>>,
    continues: Vec<AssignmentState<K>>,
}

impl<K> AssignmentFlow<K> {
    fn normal(state: AssignmentState<K>) -> Self {
        Self {
            normal: Some(state),
            returns: Vec::new(),
            breaks: Vec::new(),
            continues: Vec::new(),
        }
    }

    fn divergent() -> Self {
        Self {
            normal: None,
            returns: Vec::new(),
            breaks: Vec::new(),
            continues: Vec::new(),
        }
    }

    fn with_break(mut self, state: AssignmentState<K>) -> Self {
        self.breaks.push(state);
        self
    }

    fn with_continue(mut self, state: AssignmentState<K>) -> Self {
        self.continues.push(state);
        self
    }
}

fn merge_normal_states<K: Copy + Eq + Hash>(
    states: impl IntoIterator<Item = Option<AssignmentState<K>>>,
) -> Option<AssignmentState<K>> {
    states
        .into_iter()
        .flatten()
        .reduce(|lhs, rhs| AssignmentState::intersection(&lhs, &rhs))
}

trait AssignmentDomain<'db> {
    type Item: Copy + Eq + Hash;

    fn direct_assignment(
        &mut self,
        db: &'db dyn HirAnalysisDb,
        body: Body<'db>,
        typed_body: &TypedBody<'db>,
        lhs: ExprId,
    ) -> Option<Self::Item>;

    fn call_assignments(
        &mut self,
        db: &'db dyn HirAnalysisDb,
        body: Body<'db>,
        typed_body: &TypedBody<'db>,
        call_expr: ExprId,
        expr_data: &Expr<'db>,
    ) -> Vec<Self::Item>;
}

struct DefiniteAssignmentAnalyzer<'a, 'db, D>
where
    D: AssignmentDomain<'db>,
{
    db: &'db dyn HirAnalysisDb,
    body: Body<'db>,
    typed_body: &'a TypedBody<'db>,
    domain: D,
}

impl<'a, 'db, D> DefiniteAssignmentAnalyzer<'a, 'db, D>
where
    D: AssignmentDomain<'db>,
{
    fn new(
        db: &'db dyn HirAnalysisDb,
        body: Body<'db>,
        typed_body: &'a TypedBody<'db>,
        domain: D,
    ) -> Self {
        Self {
            db,
            body,
            typed_body,
            domain,
        }
    }

    fn successful_exit_states(&mut self) -> Vec<AssignmentState<D::Item>> {
        let flow = self.analyze_expr(self.body.expr(self.db), AssignmentState::default());
        let mut exits = flow.returns;
        if let Some(normal) = flow.normal {
            exits.push(normal);
        }
        exits
    }

    fn successful_assignments(&mut self) -> FxHashSet<D::Item> {
        let mut exits = self.successful_exit_states().into_iter();
        let Some(first) = exits.next() else {
            return FxHashSet::default();
        };
        exits
            .fold(first, |lhs, rhs| AssignmentState::intersection(&lhs, &rhs))
            .assigned
    }

    fn analyze_stmt(
        &mut self,
        stmt: Stmt<'db>,
        state: AssignmentState<D::Item>,
    ) -> AssignmentFlow<D::Item> {
        match stmt {
            Stmt::Let(pat, _, Some(init)) => {
                let mut flow = self.analyze_expr(init, state);
                if let Some(normal) = &mut flow.normal {
                    self.apply_let_value_fact(pat, init, normal);
                }
                flow
            }
            Stmt::Let(_, _, None) => AssignmentFlow::normal(state),
            Stmt::Expr(expr) => self.analyze_expr(expr, state),
            Stmt::Return(expr) => {
                let flow = if let Some(expr) = expr {
                    self.analyze_expr(expr, state)
                } else {
                    AssignmentFlow::normal(state)
                };
                let mut returns = flow.returns;
                returns.extend(flow.normal);
                AssignmentFlow {
                    normal: None,
                    returns,
                    breaks: flow.breaks,
                    continues: flow.continues,
                }
            }
            Stmt::While(cond, body) => {
                let initial_truth = self.cond_value(cond, &state);
                let cond_flow = self.analyze_cond(cond, state);
                let mut returns = cond_flow.returns;
                let mut breaks = cond_flow.breaks;
                let mut continues = cond_flow.continues;
                let mut normal_exits = Vec::new();
                if initial_truth != Some(true) {
                    normal_exits.push(cond_flow.normal.clone());
                }
                if initial_truth != Some(false)
                    && let Some(cond_state) = cond_flow.normal.clone()
                {
                    let body_flow = self.analyze_expr(body, cond_state);
                    returns.extend(body_flow.returns);
                    normal_exits.extend(body_flow.breaks.into_iter().map(Some));

                    let mut backedge_states = body_flow.normal.into_iter().collect::<Vec<_>>();
                    backedge_states.extend(body_flow.continues);
                    for backedge_state in backedge_states {
                        let backedge_truth = self.cond_value(cond, &backedge_state);
                        let backedge_cond_flow = self.analyze_cond(cond, backedge_state);
                        returns.extend(backedge_cond_flow.returns);
                        breaks.extend(backedge_cond_flow.breaks);
                        continues.extend(backedge_cond_flow.continues);
                        if backedge_truth != Some(true) {
                            normal_exits.push(backedge_cond_flow.normal);
                        }
                    }
                }
                AssignmentFlow {
                    normal: merge_normal_states(normal_exits).map(AssignmentState::without_values),
                    returns,
                    breaks,
                    continues,
                }
            }
            Stmt::For(_, iter, body, _) => {
                let iter_flow = self.analyze_expr(iter, state);
                let mut returns = iter_flow.returns;
                let breaks = iter_flow.breaks;
                let continues = iter_flow.continues;
                let mut normal_exits = vec![iter_flow.normal.clone()];
                if let Some(iter_state) = iter_flow.normal.clone() {
                    let body_flow = self.analyze_expr(body, iter_state);
                    returns.extend(body_flow.returns);
                    normal_exits.extend(body_flow.breaks.into_iter().map(Some));
                    // Body continuations target this loop and are consumed here.
                }
                AssignmentFlow {
                    normal: merge_normal_states(normal_exits).map(AssignmentState::without_values),
                    returns,
                    breaks,
                    continues,
                }
            }
            Stmt::Break => AssignmentFlow::divergent().with_break(state),
            Stmt::Continue => AssignmentFlow::divergent().with_continue(state),
        }
    }

    fn analyze_cond(
        &mut self,
        cond: crate::hir_def::CondId,
        state: AssignmentState<D::Item>,
    ) -> AssignmentFlow<D::Item> {
        let Partial::Present(cond) = cond.data(self.db, self.body) else {
            return AssignmentFlow::normal(state);
        };
        match cond {
            Cond::Expr(expr) | Cond::Let(_, expr) => self.analyze_expr(*expr, state),
            Cond::Bin(lhs, rhs, _) => {
                let lhs_flow = self.analyze_cond(*lhs, state);
                let mut returns = lhs_flow.returns;
                let mut breaks = lhs_flow.breaks;
                let mut continues = lhs_flow.continues;
                let normal = lhs_flow.normal.and_then(|lhs_state| {
                    let rhs_flow = self.analyze_cond(*rhs, lhs_state.clone());
                    returns.extend(rhs_flow.returns);
                    breaks.extend(rhs_flow.breaks);
                    continues.extend(rhs_flow.continues);
                    // RHS of `&&`/`||` is short-circuited, so only writes that happen with and
                    // without RHS evaluation are definite after the condition.
                    merge_normal_states([Some(lhs_state), rhs_flow.normal])
                });
                AssignmentFlow {
                    normal,
                    returns,
                    breaks,
                    continues,
                }
            }
        }
    }

    fn analyze_expr(
        &mut self,
        expr: ExprId,
        state: AssignmentState<D::Item>,
    ) -> AssignmentFlow<D::Item> {
        let Partial::Present(expr_data) = expr.data(self.db, self.body) else {
            return AssignmentFlow::normal(state);
        };

        let flow = match expr_data {
            Expr::Lit(_) | Expr::Path(_) => AssignmentFlow::normal(state),
            Expr::Block(stmts) => self.analyze_block(stmts, state),
            Expr::Tuple(elems) | Expr::Array(elems) => self.analyze_exprs(elems, state),
            Expr::ArrayRep(elem, _) | Expr::Un(elem, _) | Expr::Cast(elem, _) => {
                self.analyze_expr(*elem, state)
            }
            Expr::Bin(lhs, rhs, _) => self.analyze_expr_pair(*lhs, *rhs, state),
            Expr::Call(callee, args) => {
                let exprs = std::iter::once(*callee)
                    .chain(args.iter().map(|arg| arg.expr))
                    .collect::<Vec<_>>();
                let mut flow = self.analyze_exprs(&exprs, state);
                self.clear_value_facts(&mut flow);
                self.apply_call_assignments(expr, expr_data, &mut flow);
                flow
            }
            Expr::Assert(args) => {
                let exprs = args.iter().map(|arg| arg.expr).collect::<Vec<_>>();
                self.analyze_exprs(&exprs, state)
            }
            Expr::MethodCall(receiver, _, _, args) => {
                let exprs = std::iter::once(*receiver)
                    .chain(args.iter().map(|arg| arg.expr))
                    .collect::<Vec<_>>();
                let mut flow = self.analyze_exprs(&exprs, state);
                self.clear_value_facts(&mut flow);
                self.apply_call_assignments(expr, expr_data, &mut flow);
                flow
            }
            Expr::RecordInit(_, fields) => {
                let exprs = fields.iter().map(|field| field.expr).collect::<Vec<_>>();
                self.analyze_exprs(&exprs, state)
            }
            Expr::Field(base, _) => self.analyze_expr(*base, state),
            Expr::If(cond, then_expr, else_expr) => {
                let truth = self.cond_value(*cond, &state);
                let cond_flow = self.analyze_cond(*cond, state);
                let mut returns = cond_flow.returns;
                let mut breaks = cond_flow.breaks;
                let mut continues = cond_flow.continues;
                let normal = cond_flow.normal.and_then(|cond_state| match truth {
                    Some(true) => {
                        let then_flow = self.analyze_expr(*then_expr, cond_state);
                        returns.extend(then_flow.returns);
                        breaks.extend(then_flow.breaks);
                        continues.extend(then_flow.continues);
                        then_flow.normal
                    }
                    Some(false) => {
                        let else_flow = if let Some(else_expr) = else_expr {
                            self.analyze_expr(*else_expr, cond_state)
                        } else {
                            AssignmentFlow::normal(cond_state)
                        };
                        returns.extend(else_flow.returns);
                        breaks.extend(else_flow.breaks);
                        continues.extend(else_flow.continues);
                        else_flow.normal
                    }
                    None => {
                        let then_flow = self.analyze_expr(*then_expr, cond_state.clone());
                        returns.extend(then_flow.returns);
                        breaks.extend(then_flow.breaks);
                        continues.extend(then_flow.continues);
                        let else_flow = if let Some(else_expr) = else_expr {
                            self.analyze_expr(*else_expr, cond_state)
                        } else {
                            AssignmentFlow::normal(cond_state)
                        };
                        returns.extend(else_flow.returns);
                        breaks.extend(else_flow.breaks);
                        continues.extend(else_flow.continues);
                        merge_normal_states([then_flow.normal, else_flow.normal])
                    }
                });
                AssignmentFlow {
                    normal,
                    returns,
                    breaks,
                    continues,
                }
            }
            Expr::Match(scrutinee, arms) => {
                let scrutinee_flow = self.analyze_expr(*scrutinee, state);
                let mut returns = scrutinee_flow.returns;
                let mut breaks = scrutinee_flow.breaks;
                let mut continues = scrutinee_flow.continues;
                let normal = scrutinee_flow.normal.and_then(|scrutinee_state| {
                    let Partial::Present(arms) = arms else {
                        return Some(scrutinee_state);
                    };
                    let mut normals = Vec::new();
                    for arm in arms {
                        let arm_flow = self.analyze_expr(arm.body, scrutinee_state.clone());
                        returns.extend(arm_flow.returns);
                        breaks.extend(arm_flow.breaks);
                        continues.extend(arm_flow.continues);
                        normals.push(arm_flow.normal);
                    }
                    merge_normal_states(normals)
                });
                AssignmentFlow {
                    normal,
                    returns,
                    breaks,
                    continues,
                }
            }
            Expr::Assign(lhs, rhs) => {
                let mut flow = self.analyze_expr(*rhs, state);
                let value_fact = flow
                    .normal
                    .as_ref()
                    .and_then(|normal| self.assignment_value_fact(*lhs, *rhs, normal));
                let direct_assignment =
                    self.domain
                        .direct_assignment(self.db, self.body, self.typed_body, *lhs);
                if let Some(normal) = &mut flow.normal {
                    if let Some(item) = direct_assignment {
                        normal.assigned.insert(item);
                    }
                    if let Some((binding, value)) = value_fact {
                        update_value_fact(normal, binding, value);
                    }
                }
                flow
            }
            Expr::AugAssign(lhs, rhs, op) => self.analyze_aug_assign(*lhs, *rhs, *op, state),
            Expr::With(bindings, body) => {
                let exprs = bindings
                    .iter()
                    .map(|binding| binding.value)
                    .chain(std::iter::once(*body))
                    .collect::<Vec<_>>();
                self.analyze_exprs(&exprs, state)
            }
        };

        if self.typed_body.expr_ty(self.db, expr).is_never(self.db) {
            AssignmentFlow {
                normal: None,
                returns: flow.returns,
                breaks: flow.breaks,
                continues: flow.continues,
            }
        } else {
            flow
        }
    }

    fn analyze_block(
        &mut self,
        stmts: &[crate::hir_def::StmtId],
        state: AssignmentState<D::Item>,
    ) -> AssignmentFlow<D::Item> {
        let mut current = Some(state);
        let mut returns = Vec::new();
        let mut breaks = Vec::new();
        let mut continues = Vec::new();
        for stmt in stmts {
            let Some(state) = current.take() else {
                break;
            };
            let Partial::Present(stmt_data) = stmt.data(self.db, self.body) else {
                current = Some(state);
                continue;
            };
            let flow = self.analyze_stmt(stmt_data.clone(), state);
            returns.extend(flow.returns);
            breaks.extend(flow.breaks);
            continues.extend(flow.continues);
            current = flow.normal;
        }
        AssignmentFlow {
            normal: current,
            returns,
            breaks,
            continues,
        }
    }

    fn analyze_expr_pair(
        &mut self,
        lhs: ExprId,
        rhs: ExprId,
        state: AssignmentState<D::Item>,
    ) -> AssignmentFlow<D::Item> {
        let lhs_flow = self.analyze_expr(lhs, state);
        let mut returns = lhs_flow.returns;
        let mut breaks = lhs_flow.breaks;
        let mut continues = lhs_flow.continues;
        let normal = lhs_flow.normal.and_then(|lhs_state| {
            let rhs_flow = self.analyze_expr(rhs, lhs_state);
            returns.extend(rhs_flow.returns);
            breaks.extend(rhs_flow.breaks);
            continues.extend(rhs_flow.continues);
            rhs_flow.normal
        });
        AssignmentFlow {
            normal,
            returns,
            breaks,
            continues,
        }
    }

    fn analyze_exprs(
        &mut self,
        exprs: &[ExprId],
        state: AssignmentState<D::Item>,
    ) -> AssignmentFlow<D::Item> {
        let mut current = Some(state);
        let mut returns = Vec::new();
        let mut breaks = Vec::new();
        let mut continues = Vec::new();
        for expr in exprs {
            let Some(state) = current.take() else {
                break;
            };
            let flow = self.analyze_expr(*expr, state);
            returns.extend(flow.returns);
            breaks.extend(flow.breaks);
            continues.extend(flow.continues);
            current = flow.normal;
        }
        AssignmentFlow {
            normal: current,
            returns,
            breaks,
            continues,
        }
    }

    fn analyze_aug_assign(
        &mut self,
        lhs: ExprId,
        rhs: ExprId,
        op: ArithBinOp,
        state: AssignmentState<D::Item>,
    ) -> AssignmentFlow<D::Item> {
        let lhs_flow = self.analyze_expr(lhs, state);
        let mut returns = lhs_flow.returns;
        let mut breaks = lhs_flow.breaks;
        let mut continues = lhs_flow.continues;
        let normal = lhs_flow.normal.and_then(|lhs_state| {
            let lhs_value = self.expr_value(lhs, &lhs_state);
            let rhs_flow = self.analyze_expr(rhs, lhs_state);
            returns.extend(rhs_flow.returns);
            breaks.extend(rhs_flow.breaks);
            continues.extend(rhs_flow.continues);

            let value_fact = rhs_flow.normal.as_ref().and_then(|normal| {
                let binding = self.value_fact_binding_for_place(lhs)?;
                let rhs_value = self.expr_value(rhs, normal);
                let value = match (lhs_value, rhs_value) {
                    (Some(ValueFact::Int(lhs)), Some(ValueFact::Int(rhs))) => {
                        eval_arith_value(op, &lhs, &rhs).map(ValueFact::Int)
                    }
                    _ => None,
                };
                Some((binding, value))
            });

            let mut normal = rhs_flow.normal;
            if let Some(normal) = &mut normal
                && let Some((binding, value)) = value_fact
            {
                update_value_fact(normal, binding, value);
            }
            normal
        });
        AssignmentFlow {
            normal,
            returns,
            breaks,
            continues,
        }
    }

    fn apply_let_value_fact(&self, pat: PatId, init: ExprId, state: &mut AssignmentState<D::Item>) {
        let Some(binding) = self.typed_body.pat_binding(pat).map(value_fact_binding) else {
            return;
        };
        let value = self.expr_value(init, state);
        update_value_fact(state, binding, value);
    }

    fn assignment_value_fact(
        &self,
        lhs: ExprId,
        rhs: ExprId,
        state: &AssignmentState<D::Item>,
    ) -> Option<(ValueFactBinding, Option<ValueFact>)> {
        let binding = self.value_fact_binding_for_place(lhs)?;
        Some((binding, self.expr_value(rhs, state)))
    }

    fn clear_value_facts(&self, flow: &mut AssignmentFlow<D::Item>) {
        if let Some(normal) = &mut flow.normal {
            normal.values.clear();
        }
    }

    fn value_fact_binding_for_place(&self, expr: ExprId) -> Option<ValueFactBinding> {
        self.typed_body
            .expr_place(expr)
            .and_then(root_binding_from_place)
            .map(value_fact_binding)
    }

    fn value_fact_binding_for_expr(&self, expr: ExprId) -> Option<ValueFactBinding> {
        self.typed_body.expr_binding(expr).map(value_fact_binding)
    }

    fn cond_value(
        &self,
        cond: crate::hir_def::CondId,
        state: &AssignmentState<D::Item>,
    ) -> Option<bool> {
        let Partial::Present(cond) = cond.data(self.db, self.body) else {
            return None;
        };
        match cond {
            Cond::Expr(expr) => match self.expr_value(*expr, state) {
                Some(ValueFact::Bool(value)) => Some(value),
                _ => None,
            },
            Cond::Let(pat, _) => self.pattern_is_irrefutable(*pat).then_some(true),
            Cond::Bin(lhs, rhs, LogicalBinOp::And) => {
                match (self.cond_value(*lhs, state), self.cond_value(*rhs, state)) {
                    (Some(false), _) | (_, Some(false)) => Some(false),
                    (Some(true), Some(true)) => Some(true),
                    _ => None,
                }
            }
            Cond::Bin(lhs, rhs, LogicalBinOp::Or) => {
                match (self.cond_value(*lhs, state), self.cond_value(*rhs, state)) {
                    (Some(true), _) | (_, Some(true)) => Some(true),
                    (Some(false), Some(false)) => Some(false),
                    _ => None,
                }
            }
        }
    }

    fn expr_value(&self, expr: ExprId, state: &AssignmentState<D::Item>) -> Option<ValueFact> {
        let Partial::Present(expr_data) = expr.data(self.db, self.body) else {
            return None;
        };
        match expr_data {
            Expr::Lit(LitKind::Bool(value)) => Some(ValueFact::Bool(*value)),
            Expr::Lit(LitKind::Int(value)) => Some(ValueFact::Int(value.data(self.db).clone())),
            Expr::Path(_) => self
                .value_fact_binding_for_expr(expr)
                .and_then(|binding| state.values.get(&binding).cloned()),
            Expr::Cast(inner, _) => self.expr_value(*inner, state),
            Expr::Bin(lhs, rhs, BinOp::Arith(op)) => {
                match (self.expr_value(*lhs, state), self.expr_value(*rhs, state)) {
                    (Some(ValueFact::Int(lhs)), Some(ValueFact::Int(rhs))) => {
                        eval_arith_value(*op, &lhs, &rhs).map(ValueFact::Int)
                    }
                    _ => None,
                }
            }
            Expr::Bin(lhs, rhs, BinOp::Comp(op)) => compare_values(
                *op,
                &self.expr_value(*lhs, state)?,
                &self.expr_value(*rhs, state)?,
            )
            .map(ValueFact::Bool),
            Expr::Bin(lhs, rhs, BinOp::Logical(LogicalBinOp::And)) => {
                match (self.expr_value(*lhs, state), self.expr_value(*rhs, state)) {
                    (Some(ValueFact::Bool(false)), _) | (_, Some(ValueFact::Bool(false))) => {
                        Some(ValueFact::Bool(false))
                    }
                    (Some(ValueFact::Bool(true)), Some(ValueFact::Bool(true))) => {
                        Some(ValueFact::Bool(true))
                    }
                    _ => None,
                }
            }
            Expr::Bin(lhs, rhs, BinOp::Logical(LogicalBinOp::Or)) => {
                match (self.expr_value(*lhs, state), self.expr_value(*rhs, state)) {
                    (Some(ValueFact::Bool(true)), _) | (_, Some(ValueFact::Bool(true))) => {
                        Some(ValueFact::Bool(true))
                    }
                    (Some(ValueFact::Bool(false)), Some(ValueFact::Bool(false))) => {
                        Some(ValueFact::Bool(false))
                    }
                    _ => None,
                }
            }
            _ => None,
        }
    }

    fn pattern_is_irrefutable(&self, pat: PatId) -> bool {
        self.typed_body.pattern_root(pat).is_none_or(|root| {
            self.typed_body
                .pattern_store()
                .is_irrefutable(self.db, root)
        })
    }

    fn apply_call_assignments(
        &mut self,
        call_expr: ExprId,
        expr_data: &Expr<'db>,
        flow: &mut AssignmentFlow<D::Item>,
    ) {
        let Some(normal) = &mut flow.normal else {
            return;
        };
        for item in
            self.domain
                .call_assignments(self.db, self.body, self.typed_body, call_expr, expr_data)
        {
            normal.assigned.insert(item);
        }
    }
}

fn value_fact_binding(binding: LocalBinding<'_>) -> ValueFactBinding {
    match binding {
        LocalBinding::Local { pat, .. } => ValueFactBinding::Local(pat),
        LocalBinding::Param { idx, .. } => ValueFactBinding::Param(idx),
        LocalBinding::EffectParam { idx, .. } => ValueFactBinding::Effect(idx),
    }
}

fn update_value_fact<K>(
    state: &mut AssignmentState<K>,
    binding: ValueFactBinding,
    value: Option<ValueFact>,
) {
    if let Some(value) = value {
        state.values.insert(binding, value);
    } else {
        state.values.remove(&binding);
    }
}

fn eval_arith_value(op: ArithBinOp, lhs: &BigUint, rhs: &BigUint) -> Option<BigUint> {
    match op {
        ArithBinOp::Add => Some(lhs + rhs),
        ArithBinOp::Sub => (lhs >= rhs).then(|| lhs - rhs),
        ArithBinOp::Mul => Some(lhs * rhs),
        _ => None,
    }
}

fn compare_values(op: CompBinOp, lhs: &ValueFact, rhs: &ValueFact) -> Option<bool> {
    match (lhs, rhs) {
        (ValueFact::Bool(lhs), ValueFact::Bool(rhs)) => match op {
            CompBinOp::Eq => Some(lhs == rhs),
            CompBinOp::NotEq => Some(lhs != rhs),
            _ => None,
        },
        (ValueFact::Int(lhs), ValueFact::Int(rhs)) => match op {
            CompBinOp::Eq => Some(lhs == rhs),
            CompBinOp::NotEq => Some(lhs != rhs),
            CompBinOp::Lt => Some(lhs < rhs),
            CompBinOp::LtEq => Some(lhs <= rhs),
            CompBinOp::Gt => Some(lhs > rhs),
            CompBinOp::GtEq => Some(lhs >= rhs),
        },
        _ => None,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum FuncWriteRoot {
    Param(usize),
    Effect(usize),
}

#[derive(Default)]
struct FuncWriteSummaryCx<'db> {
    cache: FxHashMap<Func<'db>, FxHashSet<FuncWriteRoot>>,
    in_progress: FxHashSet<Func<'db>>,
}

impl<'db> FuncWriteSummaryCx<'db> {
    fn summarize_func(
        &mut self,
        db: &'db dyn HirAnalysisDb,
        func: Func<'db>,
    ) -> FxHashSet<FuncWriteRoot> {
        if let Some(cached) = self.cache.get(&func) {
            return cached.clone();
        }

        if !self.in_progress.insert(func) {
            return FxHashSet::default();
        }

        let (_, typed_body) = check_func_body(db, func);
        let roots = if let Some(body) = typed_body.body() {
            DefiniteAssignmentAnalyzer::new(
                db,
                body,
                typed_body,
                FuncWriteDomain {
                    func,
                    summaries: self,
                },
            )
            .successful_assignments()
        } else {
            FxHashSet::default()
        };

        self.in_progress.remove(&func);
        self.cache.insert(func, roots.clone());
        roots
    }
}

struct FuncWriteDomain<'a, 'db> {
    func: Func<'db>,
    summaries: &'a mut FuncWriteSummaryCx<'db>,
}

impl<'db> AssignmentDomain<'db> for FuncWriteDomain<'_, 'db> {
    type Item = FuncWriteRoot;

    fn direct_assignment(
        &mut self,
        _db: &'db dyn HirAnalysisDb,
        _body: Body<'db>,
        typed_body: &TypedBody<'db>,
        lhs: ExprId,
    ) -> Option<Self::Item> {
        let binding = typed_body
            .expr_place(lhs)
            .and_then(root_binding_from_place)?;
        func_write_root_for_binding(self.func, binding)
    }

    fn call_assignments(
        &mut self,
        db: &'db dyn HirAnalysisDb,
        _body: Body<'db>,
        typed_body: &TypedBody<'db>,
        call_expr: ExprId,
        expr_data: &Expr<'db>,
    ) -> Vec<Self::Item> {
        call_assigned_root_bindings(db, typed_body, self.summaries, call_expr, expr_data)
            .into_iter()
            .filter_map(|binding| func_write_root_for_binding(self.func, binding))
            .collect()
    }
}

struct InitFieldDomain<'a, 'db> {
    contract: Contract<'db>,
    required: &'a FxHashSet<u32>,
    summaries: &'a mut FuncWriteSummaryCx<'db>,
}

impl<'db> AssignmentDomain<'db> for InitFieldDomain<'_, 'db> {
    type Item = u32;

    fn direct_assignment(
        &mut self,
        db: &'db dyn HirAnalysisDb,
        _body: Body<'db>,
        typed_body: &TypedBody<'db>,
        lhs: ExprId,
    ) -> Option<Self::Item> {
        let binding = typed_body
            .expr_place(lhs)
            .and_then(root_binding_from_place)?;
        contract_field_for_binding(db, self.contract, self.required, binding)
    }

    fn call_assignments(
        &mut self,
        db: &'db dyn HirAnalysisDb,
        _body: Body<'db>,
        typed_body: &TypedBody<'db>,
        call_expr: ExprId,
        expr_data: &Expr<'db>,
    ) -> Vec<Self::Item> {
        call_assigned_root_bindings(db, typed_body, self.summaries, call_expr, expr_data)
            .into_iter()
            .filter_map(|binding| {
                contract_field_for_binding(db, self.contract, self.required, binding)
            })
            .collect()
    }
}

fn root_binding_from_place<'db>(place: &Place<'db>) -> Option<LocalBinding<'db>> {
    if !place.projections.is_empty() {
        return None;
    }
    let PlaceBase::Binding(binding) = place.base;
    Some(binding)
}

fn func_write_root_for_binding<'db>(
    func: Func<'db>,
    binding: LocalBinding<'db>,
) -> Option<FuncWriteRoot> {
    match binding {
        LocalBinding::Param {
            site: ParamSite::Func(binding_func),
            idx,
            ..
        } if binding_func == func => Some(FuncWriteRoot::Param(idx)),
        LocalBinding::EffectParam {
            site: EffectParamSite::Func(binding_func),
            idx,
            ..
        } if binding_func == func => Some(FuncWriteRoot::Effect(idx)),
        _ => None,
    }
}

fn contract_field_for_binding<'db>(
    db: &'db dyn HirAnalysisDb,
    contract: Contract<'db>,
    required: &FxHashSet<u32>,
    binding: LocalBinding<'db>,
) -> Option<u32> {
    let LocalBinding::EffectParam { site, idx, .. } = binding else {
        return None;
    };
    if site != (EffectParamSite::ContractInit { contract }) {
        return None;
    }
    let provider = EffectEnvView::new(site)
        .resolved_binding(db, idx)
        .map(|binding| binding.provider)?;
    let ProviderSource::ContractField {
        contract: provider_contract,
        field_idx,
    } = provider.source
    else {
        return None;
    };
    (provider_contract == contract && required.contains(&field_idx)).then_some(field_idx)
}

fn call_assigned_root_bindings<'db>(
    db: &'db dyn HirAnalysisDb,
    typed_body: &TypedBody<'db>,
    summaries: &mut FuncWriteSummaryCx<'db>,
    call_expr: ExprId,
    expr_data: &Expr<'db>,
) -> Vec<LocalBinding<'db>> {
    let Some(callable) = typed_body.callable_expr(call_expr) else {
        return Vec::new();
    };
    let CallableDef::Func(func) = callable.callable_def() else {
        return Vec::new();
    };

    summaries
        .summarize_func(db, func)
        .into_iter()
        .filter_map(|root| caller_binding_for_callee_root(typed_body, call_expr, expr_data, root))
        .collect()
}

fn caller_binding_for_callee_root<'db>(
    typed_body: &TypedBody<'db>,
    call_expr: ExprId,
    expr_data: &Expr<'db>,
    root: FuncWriteRoot,
) -> Option<LocalBinding<'db>> {
    match root {
        FuncWriteRoot::Param(param_idx) => call_param_expr(expr_data, param_idx)
            .and_then(|expr| typed_body.expr_place(expr))
            .and_then(root_binding_from_place),
        FuncWriteRoot::Effect(effect_idx) => typed_body
            .call_effect_args(call_expr)?
            .iter()
            .find(|arg| arg.binding_idx as usize == effect_idx)
            .and_then(|arg| effect_arg_root_binding(typed_body, arg)),
    }
}

fn call_param_expr<'db>(expr_data: &Expr<'db>, param_idx: usize) -> Option<ExprId> {
    match expr_data {
        Expr::Call(_, args) => args.get(param_idx).map(|arg| arg.expr),
        Expr::MethodCall(receiver, _, _, args) => {
            if param_idx == 0 {
                Some(*receiver)
            } else {
                args.get(param_idx - 1).map(|arg| arg.expr)
            }
        }
        _ => None,
    }
}

fn effect_arg_root_binding<'db>(
    typed_body: &TypedBody<'db>,
    arg: &ResolvedEffectArg<'db>,
) -> Option<LocalBinding<'db>> {
    match &arg.arg {
        EffectArg::Place(place) => root_binding_from_place(place),
        EffectArg::Value(expr) => typed_body
            .expr_place(*expr)
            .and_then(root_binding_from_place),
        EffectArg::Binding(binding) => Some(*binding),
        EffectArg::Unknown => None,
    }
}

#[salsa::tracked(return_ref)]
pub fn check_contract_immutable_fields_initialized<'db>(
    db: &'db dyn HirAnalysisDb,
    contract: Contract<'db>,
) -> Vec<FuncBodyDiag<'db>> {
    let valid_fields = contract
        .field_layout(db)
        .values()
        .filter(|field| field.slot_count != 0)
        .filter(|field| {
            !field.declared_ty.has_invalid(db)
                && !field.target_ty.has_invalid(db)
                && !field.address_space.has_invalid(db)
        })
        .filter(|field| {
            FieldView {
                parent: FieldParent::Contract(contract),
                idx: field.index as usize,
            }
            .ty_diags(db)
            .is_empty()
        })
        .cloned()
        .collect::<Vec<_>>();
    let mut diags = valid_fields
        .iter()
        .filter(|field| {
            address_space_from_ty(db, contract.scope(), field.address_space)
                == Some(ProviderAddressSpace::Memory)
        })
        .map(|field| {
            BodyDiag::UnsupportedMemoryContractField {
                primary: FieldView {
                    parent: FieldParent::Contract(contract),
                    idx: field.index as usize,
                }
                .ty_span(),
                field: field.name,
            }
            .into()
        })
        .collect::<Vec<_>>();
    let required_fields = valid_fields
        .into_iter()
        .filter(|field| {
            address_space_from_ty(db, contract.scope(), field.address_space)
                == Some(ProviderAddressSpace::Code)
        })
        .collect::<Vec<_>>();
    if required_fields.is_empty() {
        return diags;
    }

    let required = required_fields
        .iter()
        .map(|field| field.index)
        .collect::<FxHashSet<_>>();
    let init = contract.init(db);
    let missing = if init.is_some() {
        let (_, typed_body) = check_contract_init_body(db, contract);
        if let Some(body) = typed_body.body() {
            let mut summaries = FuncWriteSummaryCx::default();
            let exits = DefiniteAssignmentAnalyzer::new(
                db,
                body,
                typed_body,
                InitFieldDomain {
                    contract,
                    required: &required,
                    summaries: &mut summaries,
                },
            )
            .successful_exit_states();
            if exits.is_empty() {
                FxHashSet::default()
            } else {
                required
                    .iter()
                    .copied()
                    .filter(|field| !exits.iter().all(|state| state.assigned.contains(field)))
                    .collect()
            }
        } else {
            required.clone()
        }
    } else {
        required
    };

    let init_span = init.map(|_| contract.span().init_block().body().into());
    diags.extend(required_fields.into_iter().filter_map(|field| {
        missing.contains(&field.index).then(|| {
            BodyDiag::ImmutableContractFieldNotInitialized {
                primary: FieldParent::Contract(contract).field_name_span(field.index as usize),
                field: field.name,
                init: init_span.clone(),
            }
            .into()
        })
    }));
    diags
}

#[salsa::tracked(return_ref)]
pub fn check_contract_init_body<'db>(
    db: &'db dyn HirAnalysisDb,
    contract: Contract<'db>,
) -> (Vec<FuncBodyDiag<'db>>, TypedBody<'db>) {
    check_body(db, BodyOwner::ContractInit { contract })
}

pub(super) fn resolve_recv_msg_mod<'db>(
    db: &'db dyn HirAnalysisDb,
    contract: Contract<'db>,
    msg_path: Option<PathId<'db>>,
    span: LazyPathSpan<'db>,
    diags: &mut Vec<FuncBodyDiag<'db>>,
    emit_diag: bool,
) -> Option<Mod<'db>> {
    let msg_path = msg_path?;
    let assumptions = PredicateListId::empty_list(db);

    match resolve_path(db, msg_path, contract.scope(), assumptions, false) {
        Ok(PathRes::Ty(ty) | PathRes::TyAlias(_, ty)) => {
            if emit_diag {
                diags.push(
                    BodyDiag::RecvExpectedMsgType {
                        primary: span.clone().into(),
                        given: ty,
                    }
                    .into(),
                );
            }
            None
        }
        Ok(PathRes::Mod(scope)) => {
            // Accept any module as a recv root (both msg-desugared and manually defined)
            if let ScopeId::Item(ItemKind::Mod(mod_)) = scope {
                return Some(mod_);
            }
            unreachable!();
        }
        Ok(other) => {
            let ident = msg_path.ident(db).to_opt()?;
            if emit_diag {
                diags.push(PathResDiag::ExpectedType(span.into(), ident, other.kind_name()).into());
            }
            None
        }
        Err(err) => {
            if emit_diag
                && let Some(diag) = err.into_diag(db, msg_path, span, ExpectedPathKind::Type)
            {
                diags.push(diag.into());
            }
            None
        }
    }
}

/// Gets the Return type from a type's MsgVariant trait implementation.
/// Specifically looks for the MsgVariant trait and returns `None` if no impl is found.
pub(super) fn get_msg_variant_return_type<'db>(
    db: &'db dyn HirAnalysisDb,
    variant_ty: TyId<'db>,
    scope: ScopeId<'db>,
) -> Option<TyId<'db>> {
    let msg_variant_trait = resolve_core_trait(db, scope, &["message", "MsgVariant"])?;

    let canonical_ty = Canonical::new(db, variant_ty);
    let scope_ingot = scope.ingot(db);
    let search_ingots = [
        Some(scope_ingot),
        variant_ty.ingot(db).filter(|&ingot| ingot != scope_ingot),
    ];

    // Find the MsgVariant impl specifically, probing both:
    // - the call-site ingot (for local traits implemented for external types), and
    // - the receiver type's ingot (for external traits implemented in the type's ingot).
    let msg_variant_impl = search_ingots.into_iter().flatten().find_map(|ingot| {
        impls_for_ty(db, ingot, canonical_ty)
            .iter()
            .find(|impl_| impl_.skip_binder().trait_def(db).eq(&msg_variant_trait))
            .copied()
    })?;

    // Get the Return associated type from the impl
    let return_name = IdentId::new(db, "Return".to_string());
    msg_variant_impl.skip_binder().assoc_ty(db, return_name)
}
