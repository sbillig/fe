use num_bigint::BigInt;

use crate::{
    analysis::{
        semantic::{
            FieldIndex, SBlockId, SConst, SExpr, SStmt, STerminator, SValueId, VariantIndex,
            bool_const, bytes_const, int_const,
        },
        ty::{
            pattern_ir::{ConstructorKind, ValidatedPatId, ValidatedPatKind},
            ty_def::TyId,
        },
    },
    hir_def::{
        LitKind, PatId,
        expr::{BinOp, CompBinOp},
    },
};

use super::body::SmirLowerCtxt;

pub(super) enum ArmVariants {
    Variants(Vec<VariantIndex>),
    Default,
}

impl<'db> SmirLowerCtxt<'db> {
    pub(super) fn pattern_is_irrefutable(&self, pat: PatId) -> bool {
        self.typed_body.pattern_root(pat).is_none_or(|root| {
            self.typed_body
                .pattern_store()
                .is_irrefutable(self.db, root)
        })
    }

    pub(super) fn pattern_is_enum_dispatchable(&self, pat: PatId) -> bool {
        let Some(root) = self.typed_body.pattern_root(pat) else {
            return true;
        };
        self.is_enum_dispatchable_root(root)
    }

    pub(super) fn bind_pattern(&mut self, pat: PatId, value: SValueId) {
        if let Some(root) = self.typed_body.pattern_root(pat) {
            self.bind_validated_pattern(root, value);
        }
    }

    pub(super) fn lower_pattern_branch(
        &mut self,
        pat: PatId,
        value: SValueId,
        then_bb: SBlockId,
        else_bb: SBlockId,
    ) {
        let Some(root) = self.typed_body.pattern_root(pat) else {
            self.set_terminator(self.current, STerminator::Goto(then_bb));
            return;
        };
        self.lower_validated_pattern_branch(root, value, then_bb, else_bb);
    }

    fn bind_validated_pattern(&mut self, pat: ValidatedPatId, value: SValueId) {
        let node = self.typed_body.pattern_store().node(pat).clone();
        match node.kind {
            ValidatedPatKind::Wildcard { binding } => {
                if let Some(binding) = binding
                    && let Some(local_binding) =
                        self.typed_body.pat_binding(binding.representative_pat)
                {
                    let dst = self.alloc_binding_local(local_binding);
                    self.push_stmt(SStmt::Assign {
                        dst,
                        expr: SExpr::Use(value),
                    });
                }
            }
            ValidatedPatKind::Constructor { ctor, fields } => match ctor {
                ConstructorKind::Variant(variant, _) => {
                    for (idx, field_pat) in fields.into_iter().enumerate() {
                        let field = self.emit_expr(
                            self.typed_body.pattern_store().node(field_pat).ty,
                            SExpr::ExtractEnumField {
                                value,
                                variant: VariantIndex(variant.idx),
                                field: FieldIndex(idx as u16),
                            },
                        );
                        self.bind_validated_pattern(field_pat, field);
                    }
                }
                ConstructorKind::Type(_) => {
                    for (idx, field_pat) in fields.into_iter().enumerate() {
                        let field = self.emit_expr(
                            self.typed_body.pattern_store().node(field_pat).ty,
                            SExpr::Field {
                                base: value,
                                field: FieldIndex(idx as u16),
                            },
                        );
                        self.bind_validated_pattern(field_pat, field);
                    }
                }
                ConstructorKind::Literal(..) => {}
            },
            ValidatedPatKind::Or(pats) => {
                if let Some(first) = pats.first().copied() {
                    self.bind_validated_pattern(first, value);
                }
            }
        }
    }

    pub(super) fn arm_variants(&self, pat: PatId) -> ArmVariants {
        let Some(root) = self.typed_body.pattern_root(pat) else {
            return ArmVariants::Default;
        };
        self.arm_variants_from_root(root)
    }

    fn arm_variants_from_root(&self, pat: ValidatedPatId) -> ArmVariants {
        match &self.typed_body.pattern_store().node(pat).kind {
            ValidatedPatKind::Wildcard { .. } => ArmVariants::Default,
            ValidatedPatKind::Constructor {
                ctor: ConstructorKind::Variant(variant, _),
                ..
            } => ArmVariants::Variants(vec![VariantIndex(variant.idx)]),
            ValidatedPatKind::Constructor {
                ctor: ConstructorKind::Type(_) | ConstructorKind::Literal(..),
                ..
            } => ArmVariants::Default,
            ValidatedPatKind::Or(pats) => {
                let mut variants = Vec::new();
                for pat in pats {
                    match self.arm_variants_from_root(*pat) {
                        ArmVariants::Variants(mut pat_variants) => {
                            variants.append(&mut pat_variants)
                        }
                        ArmVariants::Default => return ArmVariants::Default,
                    }
                }
                ArmVariants::Variants(variants)
            }
        }
    }

    fn is_enum_dispatchable_root(&self, pat: ValidatedPatId) -> bool {
        match &self.typed_body.pattern_store().node(pat).kind {
            ValidatedPatKind::Wildcard { .. } => true,
            ValidatedPatKind::Constructor {
                ctor: ConstructorKind::Variant(..),
                ..
            } => true,
            ValidatedPatKind::Constructor {
                ctor: ConstructorKind::Type(_) | ConstructorKind::Literal(..),
                ..
            } => false,
            ValidatedPatKind::Or(pats) => {
                pats.iter().all(|pat| self.is_enum_dispatchable_root(*pat))
            }
        }
    }

    fn lower_validated_pattern_branch(
        &mut self,
        pat: ValidatedPatId,
        value: SValueId,
        then_bb: SBlockId,
        else_bb: SBlockId,
    ) {
        let node = self.typed_body.pattern_store().node(pat).clone();
        match node.kind {
            ValidatedPatKind::Wildcard { .. } => {
                self.set_terminator(self.current, STerminator::Goto(then_bb));
            }
            ValidatedPatKind::Constructor { ctor, fields } => match ctor {
                ConstructorKind::Literal(lit, ty) => {
                    let rhs = self.literal_pattern_value(ty, lit);
                    let cond = self.emit_expr(
                        TyId::bool(self.db),
                        SExpr::Binary {
                            op: BinOp::Comp(CompBinOp::Eq),
                            lhs: value,
                            rhs,
                        },
                    );
                    self.set_terminator(
                        self.current,
                        STerminator::Branch {
                            cond,
                            then_bb,
                            else_bb,
                        },
                    );
                }
                ConstructorKind::Variant(variant, _) => {
                    let success_bb = if fields.is_empty() {
                        then_bb
                    } else {
                        self.new_block()
                    };
                    let cond = self.emit_expr(
                        TyId::bool(self.db),
                        SExpr::IsEnumVariant {
                            value,
                            variant: VariantIndex(variant.idx),
                        },
                    );
                    self.set_terminator(
                        self.current,
                        STerminator::Branch {
                            cond,
                            then_bb: success_bb,
                            else_bb,
                        },
                    );
                    if !fields.is_empty() {
                        self.switch_to(success_bb);
                        self.lower_variant_field_branches(
                            value,
                            VariantIndex(variant.idx),
                            &fields,
                            then_bb,
                            else_bb,
                        );
                    }
                }
                ConstructorKind::Type(_) => {
                    self.lower_field_branches(value, &fields, then_bb, else_bb);
                }
            },
            ValidatedPatKind::Or(pats) => {
                let Some((last, rest)) = pats.split_last() else {
                    self.set_terminator(self.current, STerminator::Goto(else_bb));
                    return;
                };
                for pat in rest {
                    let next_else = self.new_block();
                    self.lower_validated_pattern_branch(*pat, value, then_bb, next_else);
                    self.switch_to(next_else);
                }
                self.lower_validated_pattern_branch(*last, value, then_bb, else_bb);
            }
        }
    }

    fn lower_field_branches(
        &mut self,
        value: SValueId,
        fields: &[ValidatedPatId],
        then_bb: SBlockId,
        else_bb: SBlockId,
    ) {
        if fields.is_empty() {
            self.set_terminator(self.current, STerminator::Goto(then_bb));
            return;
        }

        for (idx, field_pat) in fields.iter().copied().enumerate() {
            let field_value = self.emit_expr(
                self.typed_body.pattern_store().node(field_pat).ty,
                SExpr::Field {
                    base: value,
                    field: FieldIndex(idx as u16),
                },
            );
            let next_then = if idx + 1 == fields.len() {
                then_bb
            } else {
                self.new_block()
            };
            self.lower_validated_pattern_branch(field_pat, field_value, next_then, else_bb);
            if idx + 1 != fields.len() {
                self.switch_to(next_then);
            }
        }
    }

    fn lower_variant_field_branches(
        &mut self,
        value: SValueId,
        variant: VariantIndex,
        fields: &[ValidatedPatId],
        then_bb: SBlockId,
        else_bb: SBlockId,
    ) {
        if fields.is_empty() {
            self.set_terminator(self.current, STerminator::Goto(then_bb));
            return;
        }

        for (idx, field_pat) in fields.iter().copied().enumerate() {
            let field_value = self.emit_expr(
                self.typed_body.pattern_store().node(field_pat).ty,
                SExpr::ExtractEnumField {
                    value,
                    variant,
                    field: FieldIndex(idx as u16),
                },
            );
            let next_then = if idx + 1 == fields.len() {
                then_bb
            } else {
                self.new_block()
            };
            self.lower_validated_pattern_branch(field_pat, field_value, next_then, else_bb);
            if idx + 1 != fields.len() {
                self.switch_to(next_then);
            }
        }
    }

    fn literal_pattern_value(&mut self, ty: TyId<'db>, lit: LitKind<'db>) -> SValueId {
        let value = match lit {
            LitKind::Int(int_id) => {
                int_const(self.db, ty, BigInt::from(int_id.data(self.db).clone()))
            }
            LitKind::String(string_id) => {
                bytes_const(self.db, ty, string_id.data(self.db).as_bytes().to_vec())
            }
            LitKind::Bool(value) => bool_const(self.db, value),
        };
        self.emit_expr(ty, SExpr::Const(SConst::Value(value)))
    }
}
