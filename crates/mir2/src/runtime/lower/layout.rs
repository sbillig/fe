use hir::analysis::ty::{
    const_ty::{ConstTyData, EvaluatedConstTy},
    ty_def::{TyData, TyId},
};
use num_traits::ToPrimitive;

use crate::{
    db::MirDb,
    runtime::{ArrayLayout, EnumLayoutKey, EnumVariantLayout, LayoutId, LayoutKey, StructLayout},
};

use super::class::stored_class_for_ty;

pub(super) fn layout_for_ty<'db>(db: &'db dyn MirDb, ty: TyId<'db>) -> LayoutId<'db> {
    if ty.as_enum(db).is_some() {
        return LayoutId::new(db, LayoutKey::Enum(enum_layout_key(db, ty)));
    }
    if ty.is_array(db) {
        let (_, args) = ty.decompose_ty_app(db);
        let elem = args.first().copied().expect("array element type");
        return LayoutId::new(
            db,
            LayoutKey::Array(ArrayLayout {
                source_ty: ty,
                elem: stored_class_for_ty(db, elem),
                len: array_len(db, ty).expect("array length"),
            }),
        );
    }
    LayoutId::new(
        db,
        LayoutKey::Struct(StructLayout {
            source_ty: ty,
            fields: ty
                .field_types(db)
                .into_iter()
                .map(|field| stored_class_for_ty(db, field))
                .collect(),
        }),
    )
}

fn enum_layout_key<'db>(db: &'db dyn MirDb, ty: TyId<'db>) -> EnumLayoutKey<'db> {
    let enum_ = ty.as_enum(db).expect("enum layout requested for non-enum");
    let args = ty.generic_args(db);
    EnumLayoutKey {
        source_ty: ty,
        variants: enum_
            .variants(db)
            .map(|variant| EnumVariantLayout {
                name: variant
                    .name(db)
                    .map(|name| name.data(db).to_string())
                    .unwrap_or_else(|| format!("variant_{}", variant.idx)),
                fields: variant
                    .field_tys(db)
                    .into_iter()
                    .map(|field| stored_class_for_ty(db, field.instantiate(db, args)))
                    .collect(),
            })
            .collect(),
    }
}

fn array_len<'db>(db: &'db dyn MirDb, ty: TyId<'db>) -> Option<u64> {
    let (_, args) = ty.decompose_ty_app(db);
    let len_ty = *args.get(1)?;
    let TyData::ConstTy(const_ty) = len_ty.data(db) else {
        return None;
    };
    match const_ty.data(db) {
        ConstTyData::Evaluated(EvaluatedConstTy::LitInt(int_id), _) => int_id.data(db).to_u64(),
        _ => None,
    }
}
