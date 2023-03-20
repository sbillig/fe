use fe_parser2::ast::{self, prelude::*};

use crate::{
    hir_def::{Body, GenericArgListId, MaybeInvalid, PathId, TraitRef, TypeId, TypeKind},
    span::FileId,
    HirDb,
};

impl TypeId {
    pub(crate) fn from_ast(db: &dyn HirDb, fid: FileId, ast: ast::Type) -> Self {
        let kind = match ast.kind() {
            ast::TypeKind::Ptr(ty) => {
                let inner = Self::maybe_from_ast(db, fid, ty.inner());
                TypeKind::Ptr(inner)
            }

            ast::TypeKind::Path(ty) => {
                let path = PathId::maybe_from_ast(db, ty.path()).into();
                let generic_args = GenericArgListId::from_ast_opt(db, fid, ty.generic_args());
                TypeKind::Path(path, generic_args.into())
            }

            ast::TypeKind::SelfType(_) => TypeKind::SelfType,

            ast::TypeKind::Tuple(ty) => {
                let mut elem_tys = Vec::new();
                for elem in ty {
                    elem_tys.push(Some(TypeId::from_ast(db, fid, elem)).into());
                }
                TypeKind::Tuple(elem_tys)
            }

            ast::TypeKind::Array(ty) => {
                let elem_ty = Self::maybe_from_ast(db, fid, ty.elem_ty());
                let body = ty
                    .len()
                    .map(|ast| Body::from_ast(db, fid, None, ast))
                    .into();
                TypeKind::Array(elem_ty, body)
            }
        };

        TypeId::new(db, kind)
    }

    pub(crate) fn maybe_from_ast(
        db: &dyn HirDb,
        fid: FileId,
        ast: Option<ast::Type>,
    ) -> MaybeInvalid<Self> {
        ast.map(|ast| Self::from_ast(db, fid, ast)).into()
    }
}

impl TraitRef {
    pub(crate) fn from_ast(db: &dyn HirDb, fid: FileId, ast: ast::PathType) -> Self {
        let path = PathId::maybe_from_ast(db, ast.path()).into();
        let generic_args = GenericArgListId::from_ast_opt(db, fid, ast.generic_args());
        Self { path, generic_args }
    }

    pub(crate) fn maybe_from_ast(
        db: &dyn HirDb,
        fid: FileId,
        ast: Option<ast::PathType>,
    ) -> MaybeInvalid<Self> {
        ast.map(|ast| Self::from_ast(db, fid, ast)).into()
    }
}
