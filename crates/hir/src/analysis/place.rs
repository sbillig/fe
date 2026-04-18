use crate::{
    analysis::{
        HirAnalysisDb,
        ty::{
            ty_check::{LocalBinding, RecordLike, TypedBody},
            ty_def::TyId,
        },
    },
    hir_def::{BinOp, Body, Expr, ExprId, FieldIndex, Partial, UnOp},
};
use num_traits::ToPrimitive;
use salsa::Update;

/// A "place" is an assignable location (an lvalue): a base binding plus zero or
/// more resolved projections (field/index).
///
/// Places are used to model effect arguments as implicit references and to
/// select the correct load/store operations based on address space.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Update)]
pub struct Place<'db> {
    pub base: PlaceBase<'db>,
    pub projections: Vec<PlaceProjection<'db>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Update)]
pub enum PlaceBase<'db> {
    Binding(LocalBinding<'db>),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Update)]
pub enum PlaceProjection<'db> {
    Field {
        index: u16,
        result_ty: TyId<'db>,
    },
    Index {
        index_expr: ExprId,
        result_ty: TyId<'db>,
    },
}

impl<'db> PlaceProjection<'db> {
    pub fn result_ty(self) -> TyId<'db> {
        match self {
            Self::Field { result_ty, .. } | Self::Index { result_ty, .. } => result_ty,
        }
    }
}

impl<'db> Place<'db> {
    pub fn new(base: PlaceBase<'db>) -> Self {
        Self {
            base,
            projections: Vec::new(),
        }
    }

    pub fn push_projection(&mut self, proj: PlaceProjection<'db>) {
        self.projections.push(proj);
    }

    pub fn is_definitely_place_expr(typed_body: &TypedBody<'db>, expr: ExprId) -> bool {
        typed_body.expr_place(expr).is_some()
    }

    pub fn from_expr(typed_body: &TypedBody<'db>, expr: ExprId) -> Option<Self> {
        typed_body.expr_place(expr).cloned()
    }

    pub fn from_expr_in_body<F, G>(
        db: &'db dyn HirAnalysisDb,
        body: Body<'db>,
        expr: ExprId,
        mut expr_binding: F,
        mut expr_ty: G,
    ) -> Option<Self>
    where
        F: FnMut(ExprId) -> Option<LocalBinding<'db>>,
        G: FnMut(ExprId) -> TyId<'db>,
    {
        Self::from_expr_in_body_with(db, body, expr, &mut expr_binding, &mut expr_ty)
    }

    fn from_expr_in_body_with(
        db: &'db dyn HirAnalysisDb,
        body: Body<'db>,
        expr: ExprId,
        expr_binding: &mut dyn FnMut(ExprId) -> Option<LocalBinding<'db>>,
        expr_ty: &mut dyn FnMut(ExprId) -> TyId<'db>,
    ) -> Option<Self> {
        let Partial::Present(expr_data) = expr.data(db, body) else {
            return None;
        };

        match expr_data {
            Expr::Path(..) => {
                let binding = expr_binding(expr)?;
                Some(Place::new(PlaceBase::Binding(binding)))
            }
            Expr::Un(base, UnOp::Mut | UnOp::Ref) => {
                Place::from_expr_in_body_with(db, body, *base, expr_binding, expr_ty)
            }
            Expr::Field(base, field) => {
                let field = field.to_opt()?;
                let mut place =
                    Place::from_expr_in_body_with(db, body, *base, expr_binding, expr_ty)?;
                let index = resolve_place_field_index(db, expr_ty(*base), field)?;
                place.push_projection(PlaceProjection::Field {
                    index,
                    result_ty: expr_ty(expr),
                });
                Some(place)
            }
            Expr::Bin(base, index, op) if *op == BinOp::Index => {
                let mut place =
                    Place::from_expr_in_body_with(db, body, *base, expr_binding, expr_ty)?;
                if !projectable_place_ty(db, expr_ty(*base)).is_array(db) {
                    return None;
                }
                place.push_projection(PlaceProjection::Index {
                    index_expr: *index,
                    result_ty: expr_ty(expr),
                });
                Some(place)
            }
            _ => None,
        }
    }
}

pub fn projectable_place_ty<'db>(db: &'db dyn HirAnalysisDb, mut ty: TyId<'db>) -> TyId<'db> {
    while let Some((_, inner)) = ty.as_capability(db) {
        ty = inner;
    }
    ty
}

pub fn resolve_place_field_index<'db>(
    db: &'db dyn HirAnalysisDb,
    base_ty: TyId<'db>,
    field: FieldIndex<'db>,
) -> Option<u16> {
    let base_ty = projectable_place_ty(db, base_ty);
    let idx = match field {
        FieldIndex::Index(index) => index.data(db).to_usize()?,
        FieldIndex::Ident(ident) => RecordLike::Type(base_ty).record_field_idx(db, ident)?,
    };
    u16::try_from(idx).ok()
}
