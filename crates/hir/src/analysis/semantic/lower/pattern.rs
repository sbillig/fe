use cranelift_entity::EntityRef;
use num_bigint::BigInt;

use crate::{
    analysis::{
        HirAnalysisDb,
        semantic::{
            FieldIndex, SBlockId, SConst, SExpr, SStmtKind, STerminatorKind, SValueId,
            VariantIndex, bool_const, bytes_const, int_const,
        },
        ty::{
            decision_tree::{
                Case, DecisionTree, LeafNode, Projection, ProjectionPath, SwitchNode,
                build_decision_tree,
            },
            pattern_analysis::PatternMatrix,
            pattern_ir::{ConstructorKind, ValidatedPatId, ValidatedPatKind},
            ty_def::{PrimTy, TyBase, TyData, TyId, instantiate_adt_field_ty},
        },
    },
    hir_def::{
        LitKind, MatchArm, PatId,
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
            self.set_synthetic_terminator(self.current, STerminatorKind::Goto(then_bb));
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
                    self.push_synthetic_stmt(SStmtKind::Assign {
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

    pub(super) fn pattern_enum_ty(&self, pat: PatId) -> Option<TyId<'db>> {
        let root = self.typed_body.pattern_root(pat)?;
        self.pattern_enum_ty_from_root(root)
    }

    pub(super) fn lower_match_expr_with_decision_tree(
        &mut self,
        value: SValueId,
        result: crate::analysis::semantic::SLocalId,
        join_bb: SBlockId,
        arms: &[MatchArm],
    ) -> SValueId {
        let roots = arms
            .iter()
            .map(|arm| self.typed_body.pattern_root(arm.pat))
            .collect::<Option<Vec<_>>>()
            .unwrap_or_else(|| panic!("decision-tree match lowering requires validated patterns"));
        let tree = build_decision_tree(
            self.db,
            &PatternMatrix::from_roots(self.typed_body.pattern_store(), &roots),
        );
        if !self.lower_decision_tree(&tree, value, result, join_bb, arms) {
            self.set_synthetic_terminator(join_bb, STerminatorKind::Goto(join_bb));
        }
        self.switch_to(join_bb);
        result
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

    fn pattern_enum_ty_from_root(&self, pat: ValidatedPatId) -> Option<TyId<'db>> {
        match &self.typed_body.pattern_store().node(pat).kind {
            ValidatedPatKind::Wildcard { .. } => None,
            ValidatedPatKind::Constructor {
                ctor: ConstructorKind::Variant(_, enum_ty),
                ..
            } => Some(*enum_ty),
            ValidatedPatKind::Constructor {
                ctor: ConstructorKind::Type(_) | ConstructorKind::Literal(..),
                ..
            } => None,
            ValidatedPatKind::Or(pats) => pats
                .iter()
                .find_map(|pat| self.pattern_enum_ty_from_root(*pat)),
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
                self.set_synthetic_terminator(self.current, STerminatorKind::Goto(then_bb));
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
                    self.set_synthetic_terminator(
                        self.current,
                        STerminatorKind::Branch {
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
                    self.set_synthetic_terminator(
                        self.current,
                        STerminatorKind::Branch {
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
                    self.set_synthetic_terminator(self.current, STerminatorKind::Goto(else_bb));
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

    fn lower_decision_tree(
        &mut self,
        tree: &DecisionTree<'db>,
        root_value: SValueId,
        result: crate::analysis::semantic::SLocalId,
        join_bb: SBlockId,
        arms: &[MatchArm],
    ) -> bool {
        match tree {
            DecisionTree::Leaf(leaf) => {
                self.bind_decision_tree_leaf(leaf, root_value);
                let arm = &arms[leaf.arm_index];
                let arm_value = self.lower_expr(arm.body);
                if self.is_terminated(self.current) {
                    false
                } else {
                    self.push_synthetic_stmt(SStmtKind::Assign {
                        dst: result,
                        expr: SExpr::Use(arm_value),
                    });
                    self.set_synthetic_terminator(self.current, STerminatorKind::Goto(join_bb));
                    true
                }
            }
            DecisionTree::Switch(switch) => {
                self.lower_decision_tree_switch(switch, root_value, result, join_bb, arms)
            }
        }
    }

    fn lower_decision_tree_switch(
        &mut self,
        switch: &SwitchNode<'db>,
        root_value: SValueId,
        result: crate::analysis::semantic::SLocalId,
        join_bb: SBlockId,
        arms: &[MatchArm],
    ) -> bool {
        let occurrence = self.project_decision_tree_path(root_value, &switch.occurrence);
        let case_blocks = switch
            .arms
            .iter()
            .map(|(case, tree)| (case, tree, self.new_block()))
            .collect::<Vec<_>>();
        let mut dispatch_bb = self.current;
        for (idx, (case, _, case_bb)) in case_blocks.iter().enumerate() {
            if idx > 0 {
                self.switch_to(dispatch_bb);
            }
            let is_last = idx + 1 == case_blocks.len();
            match case {
                Case::Default | Case::Constructor(ConstructorKind::Type(_)) if is_last => {
                    self.set_synthetic_terminator(self.current, STerminatorKind::Goto(*case_bb));
                    break;
                }
                Case::Default | Case::Constructor(ConstructorKind::Type(_)) => {
                    self.set_synthetic_terminator(self.current, STerminatorKind::Goto(*case_bb));
                    break;
                }
                Case::Constructor(ctor) if is_last => {
                    let _ = ctor;
                    self.set_synthetic_terminator(self.current, STerminatorKind::Goto(*case_bb));
                    break;
                }
                Case::Constructor(ctor) => {
                    let test = match ctor {
                        ConstructorKind::Literal(lit, ty) => {
                            let rhs = self.literal_pattern_value(*ty, *lit);
                            SExpr::Binary {
                                op: BinOp::Comp(CompBinOp::Eq),
                                lhs: occurrence,
                                rhs,
                            }
                        }
                        ConstructorKind::Variant(variant, _) => SExpr::IsEnumVariant {
                            value: occurrence,
                            variant: VariantIndex(variant.idx),
                        },
                        ConstructorKind::Type(_) => unreachable!(),
                    };
                    let next_bb = self.new_block();
                    let cond = self.emit_expr(TyId::bool(self.db), test);
                    self.set_synthetic_terminator(
                        self.current,
                        STerminatorKind::Branch {
                            cond,
                            then_bb: *case_bb,
                            else_bb: next_bb,
                        },
                    );
                    dispatch_bb = next_bb;
                }
            }
        }

        let mut join_reachable = false;
        for (_, subtree, block) in case_blocks {
            self.switch_to(block);
            join_reachable |= self.lower_decision_tree(subtree, root_value, result, join_bb, arms);
        }
        join_reachable
    }

    fn bind_decision_tree_leaf(&mut self, leaf: &LeafNode<'db>, root_value: SValueId) {
        for (binding_ref, path) in &leaf.bindings {
            if let Some(binding) = self.typed_body.pat_binding(binding_ref.representative_pat) {
                let dst = self.alloc_binding_local(binding);
                let src = self.project_decision_tree_path(root_value, path);
                self.push_synthetic_stmt(SStmtKind::Assign {
                    dst,
                    expr: SExpr::Use(src),
                });
            }
        }
    }

    fn project_decision_tree_path(
        &mut self,
        root_value: SValueId,
        path: &ProjectionPath<'db>,
    ) -> SValueId {
        let mut value = root_value;
        for projection in path.iter() {
            value = self.project_decision_tree_value(value, projection);
        }
        value
    }

    fn project_decision_tree_value(
        &mut self,
        base: SValueId,
        projection: &Projection<'db>,
    ) -> SValueId {
        let base_ty = self.projectable_place_ty(self.locals[base.index()].ty);
        match projection {
            Projection::Field(field) => {
                let field_ty = base_ty
                    .field_types(self.db)
                    .get(*field)
                    .copied()
                    .unwrap_or_else(|| {
                        TyId::invalid(self.db, crate::analysis::ty::ty_def::InvalidCause::Other)
                    });
                self.emit_expr(
                    field_ty,
                    SExpr::Field {
                        base,
                        field: FieldIndex(*field as u16),
                    },
                )
            }
            Projection::VariantField {
                variant,
                enum_ty,
                field_idx,
            } => {
                let field_ty = instantiate_adt_field_ty(
                    self.db,
                    enum_ty
                        .adt_def(self.db)
                        .expect("enum variant projection should have an ADT definition"),
                    variant.idx as usize,
                    *field_idx,
                    enum_ty.generic_args(self.db),
                );
                self.emit_expr(
                    field_ty,
                    SExpr::ExtractEnumField {
                        value: base,
                        variant: VariantIndex(variant.idx),
                        field: FieldIndex(*field_idx as u16),
                    },
                )
            }
            Projection::Discriminant => self.emit_expr(
                enum_tag_ty(self.db, base_ty),
                SExpr::GetEnumTag { value: base },
            ),
            Projection::Deref => {
                panic!("decision-tree lowering does not support deref projections yet")
            }
            Projection::Index(_) => {
                panic!("decision-tree lowering does not support index projections yet")
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
            self.set_synthetic_terminator(self.current, STerminatorKind::Goto(then_bb));
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
            self.set_synthetic_terminator(self.current, STerminatorKind::Goto(then_bb));
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

fn enum_tag_ty<'db>(db: &'db dyn HirAnalysisDb, enum_ty: TyId<'db>) -> TyId<'db> {
    let variant_count = enum_ty
        .as_enum(db)
        .map(|enum_| enum_.len_variants(db))
        .unwrap_or(0);
    if variant_count <= u8::MAX as usize + 1 {
        TyId::new(db, TyData::TyBase(TyBase::Prim(PrimTy::U8)))
    } else if variant_count <= u16::MAX as usize + 1 {
        TyId::new(db, TyData::TyBase(TyBase::Prim(PrimTy::U16)))
    } else if variant_count <= u32::MAX as usize + 1 {
        TyId::new(db, TyData::TyBase(TyBase::Prim(PrimTy::U32)))
    } else if variant_count <= u64::MAX as usize {
        TyId::new(db, TyData::TyBase(TyBase::Prim(PrimTy::U64)))
    } else {
        TyId::u256(db)
    }
}
