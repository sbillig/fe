use crate::{
    analysis::{
        place::{Place, PlaceBase, PlaceProjection, projectable_place_ty},
        semantic::{FieldIndex, SPlace, SPlaceElem},
        ty::ty_def::TyId,
    },
    hir_def::ExprId,
};

use super::body::SmirLowerCtxt;

impl<'db> SmirLowerCtxt<'db> {
    pub(super) fn projectable_place_ty(&self, ty: TyId<'db>) -> TyId<'db> {
        projectable_place_ty(self.db, ty)
    }

    pub(super) fn lower_place(&mut self, expr: ExprId) -> SPlace {
        let place = self
            .typed_body
            .expr_place(expr)
            .unwrap_or_else(|| panic!("expected place expression: {expr:?}"));
        self.lower_place_data(place)
    }

    pub(super) fn lower_place_data(&mut self, place: &Place<'db>) -> SPlace {
        let PlaceBase::Binding(binding) = place.base;
        let local = *self
            .binding_locals
            .get(&binding)
            .expect("binding local should be allocated");
        let mut path = Vec::with_capacity(place.projections.len());

        for projection in &place.projections {
            match *projection {
                PlaceProjection::Field { index, .. } => {
                    path.push(SPlaceElem::Field(FieldIndex(index)));
                }
                PlaceProjection::Index { index_expr, .. } => {
                    path.push(SPlaceElem::Index(self.lower_expr(index_expr)));
                }
            }
        }

        SPlace {
            local,
            path: path.into_boxed_slice(),
        }
    }
}
