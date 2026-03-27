use num_traits::ToPrimitive;

use crate::{
    analysis::{
        place::{Place, PlaceBase, PlaceProjection},
        semantic::{FieldIndex, SPlace, SPlaceElem},
        ty::{
            ty_check::RecordLike,
            ty_def::{InvalidCause, TyId},
        },
    },
    hir_def::{ExprId, FieldIndex as HirFieldIndex},
};

use super::body::SmirLowerCtxt;

impl<'db> SmirLowerCtxt<'db> {
    pub(super) fn lower_place(&mut self, expr: ExprId) -> SPlace {
        let place = self
            .typed_body
            .expr_place(self.db, expr)
            .unwrap_or_else(|| panic!("expected place expression: {expr:?}"));
        self.lower_place_data(place)
    }

    pub(super) fn lower_place_data(&mut self, place: Place<'db>) -> SPlace {
        let PlaceBase::Binding(binding) = place.base;
        let mut current_ty = self.typed_body.binding_ty(self.db, binding);
        let local = *self
            .binding_locals
            .get(&binding)
            .expect("binding local should be allocated");
        let mut path = Vec::with_capacity(place.projections.len());

        for projection in place.projections {
            match projection {
                PlaceProjection::Field(field) => {
                    let lowered = self.lower_field_index(current_ty, Some(field));
                    path.push(SPlaceElem::Field(lowered));
                    current_ty = current_ty.field_types(self.db)[lowered.0 as usize];
                }
                PlaceProjection::Index { index_expr } => {
                    path.push(SPlaceElem::Index(self.lower_expr(index_expr)));
                    current_ty = current_ty
                        .field_types(self.db)
                        .into_iter()
                        .next()
                        .unwrap_or_else(|| TyId::invalid(self.db, InvalidCause::Other));
                }
            }
        }

        SPlace {
            local,
            path: path.into_boxed_slice(),
        }
    }

    pub(super) fn lower_field_index(
        &self,
        base_ty: TyId<'db>,
        field: Option<HirFieldIndex<'db>>,
    ) -> FieldIndex {
        let idx = match field {
            Some(HirFieldIndex::Index(index)) => index
                .data(self.db)
                .to_usize()
                .expect("tuple field index should fit usize"),
            Some(HirFieldIndex::Ident(ident)) => RecordLike::Type(base_ty)
                .record_field_idx(self.db, ident)
                .expect("record field should resolve"),
            None => panic!("missing field index"),
        };
        FieldIndex(idx as u16)
    }
}
