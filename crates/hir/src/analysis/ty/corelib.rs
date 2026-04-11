use common::ingot::IngotKind;

use crate::{
    analysis::{
        HirAnalysisDb,
        name_resolution::{NameDomain, PathRes, resolve_ident_to_bucket, resolve_path},
        ty::{
            trait_resolution::PredicateListId,
            ty_def::{TyBase, TyData, TyId},
        },
    },
    hir_def::{CallableDef, Func, IdentId, PathId, Trait, scope_graph::ScopeId},
};

/// Resolve a trait in the core library by an explicit trait path, excluding the "core" root segment.
pub fn resolve_core_trait<'db>(
    db: &'db dyn HirAnalysisDb,
    scope: ScopeId<'db>,
    segments: &[&str],
) -> Option<Trait<'db>> {
    let (module_segments, [trait_name]) = segments.split_last_chunk::<1>()?;
    let mut module_path = lib_root_path(db, scope, "core");

    for segment in module_segments {
        module_path = module_path.push_str(db, segment);
    }

    let assumptions = PredicateListId::empty_list(db);
    let Ok(PathRes::Mod(module_scope)) = resolve_path(db, module_path, scope, assumptions, false)
    else {
        return None;
    };

    let trait_name = IdentId::new(db, trait_name.to_string());
    let bucket = resolve_ident_to_bucket(db, PathId::from_ident(db, trait_name), module_scope);
    bucket.pick(NameDomain::TYPE).as_ref().ok()?.trait_()
}

#[salsa::interned]
#[derive(Debug)]
pub struct LibPath<'db> {
    #[return_ref]
    pub string: String,
}

/// Resolve a type by a fully-qualified `core::...` or `std::...` path string.
///
/// This is a cached wrapper around `resolve_path` intended for backend consumers (e.g. MIR)
/// that need stable access to a small set of core/std helper types.
pub fn resolve_lib_type_path<'db>(
    db: &'db dyn HirAnalysisDb,
    scope: ScopeId<'db>,
    path: &str,
) -> Option<TyId<'db>> {
    let path_id = LibPath::new(db, path.to_string());
    resolve_lib_path(db, scope, path_id)
}

/// Resolve a function by a fully-qualified `core::...` or `std::...` path string.
///
/// Returns the `Func` HIR item for the resolved function.
pub fn resolve_lib_func_path<'db>(
    db: &'db dyn HirAnalysisDb,
    scope: ScopeId<'db>,
    path: &str,
) -> Option<crate::hir_def::Func<'db>> {
    let path_id = LibPath::new(db, path.to_string());
    resolve_lib_func(db, scope, path_id)
}

/// Returns `true` if `func` is the library function at the fully-qualified
/// `core::...` or `std::...` path.
///
/// This resolves from the owning ingot root instead of an arbitrary caller or
/// nested module scope, so backend consumers can classify already-resolved
/// library callees without reintroducing lookup drift.
pub fn lib_func_matches<'db>(db: &'db dyn HirAnalysisDb, func: Func<'db>, path: &str) -> bool {
    let Some((root, target_suffix)) = path.split_once("::") else {
        return false;
    };
    let expected_kind = match root {
        "core" => IngotKind::Core,
        "std" => IngotKind::Std,
        _ => return false,
    };
    if func.top_mod(db).ingot(db).kind(db) != expected_kind {
        return false;
    }
    let Some(actual_path) = func.scope().pretty_path(db) else {
        return false;
    };
    let Some((_, actual_suffix)) = actual_path.split_once("::") else {
        return false;
    };
    actual_suffix == target_suffix
}

pub struct CoreRangeTypes<'db> {
    pub range: TyId<'db>,
    pub known: TyId<'db>,
    pub unknown: TyId<'db>,
}

pub fn resolve_core_range_types<'db>(
    db: &'db dyn HirAnalysisDb,
    scope: ScopeId<'db>,
) -> Option<CoreRangeTypes<'db>> {
    let range = resolve_lib_type_path(db, scope, "core::range::Range")?;
    let known = resolve_lib_type_path(db, scope, "core::range::Known")?;
    let unknown = resolve_lib_type_path(db, scope, "core::range::Unknown")?;
    Some(CoreRangeTypes {
        range,
        known,
        unknown,
    })
}

#[salsa::tracked]
fn resolve_lib_path<'db>(
    db: &'db dyn HirAnalysisDb,
    scope: ScopeId<'db>,
    path: LibPath<'db>,
) -> Option<TyId<'db>> {
    let mut segments = path.string(db).split("::");
    let mut path = lib_root_path(db, scope, segments.next()?);

    for segment in segments {
        path = path.push_str(db, segment);
    }

    let assumptions = PredicateListId::empty_list(db);
    match resolve_path(db, path, scope, assumptions, true).ok()? {
        PathRes::Ty(ty) | PathRes::TyAlias(_, ty) => Some(ty),
        _ => None,
    }
}

#[salsa::tracked]
fn resolve_lib_func<'db>(
    db: &'db dyn HirAnalysisDb,
    scope: ScopeId<'db>,
    path: LibPath<'db>,
) -> Option<Func<'db>> {
    let mut segments = path.string(db).split("::");
    let mut path = lib_root_path(db, scope, segments.next()?);

    for segment in segments {
        path = path.push_str(db, segment);
    }

    let assumptions = PredicateListId::empty_list(db);
    match resolve_path(db, path, scope, assumptions, true).ok()? {
        PathRes::Func(ty) => {
            let TyData::TyBase(TyBase::Func(CallableDef::Func(func))) = ty.data(db) else {
                return None;
            };
            Some(*func)
        }
        _ => None,
    }
}

pub(crate) fn lib_root_path<'db>(
    db: &'db dyn HirAnalysisDb,
    scope: ScopeId<'db>,
    root: &str,
) -> PathId<'db> {
    let ingot_kind = scope.top_mod(db).ingot(db).kind(db);
    if (ingot_kind == IngotKind::Core && root == "core")
        || (ingot_kind == IngotKind::Std && root == "std")
    {
        PathId::from_ident(db, IdentId::make_ingot(db))
    } else {
        PathId::from_str(db, root)
    }
}
