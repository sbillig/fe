use crate::{
    analysis::{
        place::{Place, PlaceBase, PlaceProjection, projectable_place_ty},
        semantic::{FieldIndex, SPlace},
        ty::ty_def::TyId,
    },
    hir_def::ExprId,
};

use super::body::SmirLowerCtxt;

impl<'db> SmirLowerCtxt<'db> {
    pub(super) fn projectable_place_ty(&self, ty: TyId<'db>) -> TyId<'db> {
        projectable_place_ty(self.db, ty)
    }

    pub(super) fn lower_place(&mut self, expr: ExprId) -> SPlace<'db> {
        let place = self
            .typed_body
            .expr_place(expr)
            .unwrap_or_else(|| panic!("expected place expression: {expr:?}"));
        self.lower_place_data(place)
    }

    pub(super) fn lower_place_data(&mut self, source_place: &Place<'db>) -> SPlace<'db> {
        let PlaceBase::Binding(binding) = source_place.base;
        let local = *self
            .binding_locals
            .get(&binding)
            .expect("binding local should be allocated");
        let mut place = SPlace::new(local);

        for projection in &source_place.projections {
            match *projection {
                PlaceProjection::Field { index, .. } => {
                    place.push_field(FieldIndex(index));
                }
                PlaceProjection::Index { index_expr, .. } => {
                    let index = self.lower_expr(index_expr);
                    place.push_dynamic_index(index);
                }
            }
        }

        place
    }
}
