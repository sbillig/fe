use cranelift_entity::EntityRef;

use crate::{
    analysis::{
        place::{Place, PlaceBase, PlaceProjection, projectable_place_ty},
        semantic::{FieldIndex, SExpr, SOperand, SPlace, SemOrigin},
        ty::ty_def::TyId,
    },
    hir_def::{ExprId, Partial},
};

use super::body::SmirLowerCtxt;

impl<'a, 'db> SmirLowerCtxt<'a, 'db> {
    pub(super) fn projectable_place_ty(&self, ty: TyId<'db>) -> TyId<'db> {
        projectable_place_ty(self.db, ty)
    }

    pub(super) fn lower_place(&mut self, expr: ExprId) -> SPlace<'db> {
        if let Partial::Present(crate::hir_def::Expr::Un(inner, crate::hir_def::UnOp::Deref)) =
            expr.data(self.db, self.body)
        {
            let inner_ty = self.expr_ty(*inner);
            if let Some((_, ptr_ty)) = inner_ty.as_capability(self.db)
                && ptr_ty.as_ptr(self.db).is_some()
                && let Some(place) = self.typed_body.expr_place(*inner)
            {
                let place = self.lower_place_data(place);
                let ptr = self.emit_expr_with_origin(
                    SemOrigin::Expr(*inner),
                    ptr_ty,
                    SExpr::ReadPlace { place },
                );
                return SPlace::deref(ptr);
            }
            let ptr = self.lower_expr(*inner);
            return SPlace::deref(ptr);
        }
        let place = self
            .typed_body
            .expr_place(expr)
            .unwrap_or_else(|| panic!("expected place expression: {expr:?}"));
        self.lower_place_data(place)
    }

    pub(super) fn lower_place_data(&mut self, source_place: &Place<'db>) -> SPlace<'db> {
        self.lower_place_source(source_place, false)
    }

    pub(super) fn capture_place(&mut self, source_place: &Place<'db>) -> SPlace<'db> {
        self.lower_place_source(source_place, true)
    }

    fn lower_place_source(&mut self, source_place: &Place<'db>, capture: bool) -> SPlace<'db> {
        let PlaceBase::Binding(binding) = source_place.base;
        let local = *self
            .binding_locals
            .get(&binding)
            .expect("binding local should be allocated");
        let mut place = SPlace::new(local);
        let mut ty = self.locals[local.index()].ty;

        for projection in &source_place.projections {
            match *projection {
                PlaceProjection::Deref { .. } => {
                    if capture {
                        let ptr_ty = self.projectable_place_ty(ty);
                        let ptr = self.emit_expr(ptr_ty, SExpr::ReadPlace { place });
                        place = SPlace::deref(ptr);
                    } else {
                        place.push_deref();
                    }
                }
                PlaceProjection::Field { index, .. } => {
                    place.push_field(FieldIndex(index));
                }
                PlaceProjection::Index { index_expr, .. } => {
                    let value = self.lower_expr(index_expr);
                    let index = if capture {
                        self.emit_expr_with_origin(
                            SemOrigin::Expr(index_expr),
                            self.expr_ty(index_expr),
                            SExpr::UseValue(SOperand::expr(value, index_expr)),
                        )
                    } else {
                        value
                    };
                    place.push_dynamic_index(index);
                }
            }
            ty = projection.result_ty();
        }

        place
    }
}
