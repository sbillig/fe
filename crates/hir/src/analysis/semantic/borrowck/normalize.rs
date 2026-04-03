use cranelift_entity::EntityRef;

use crate::{
    analysis::{
        HirAnalysisDb,
        semantic::{
            SExpr, SLocalId, SPlace, SPlaceElem, SStmtKind, STerminatorKind, SemanticBody,
            SemanticInstance, SemanticLocalRole, ValueProvenance,
            ctfe::canonicalize_semantic_consts,
        },
        ty::{ty_check::LocalBinding, ty_def::TyId, ty_is_copy},
    },
    projection::{IndexSource, Projection, ProjectionPath},
};

use super::ir::{
    NBorrowRoot, NBorrowRootId, NEffectArg, NEffectArgValue, NExpr, NOperand, NSBlock, NSLocal,
    NSPlace, NSPlaceRoot, NSStmt, NSStmtKind, NSTerminator, NSTerminatorKind,
    NormalizedBindingLowering, NormalizedSemanticBody, ReadMode, SemanticNormalizeError,
    empty_normalized_body,
};

pub fn normalize_semantic_body<'db>(
    db: &'db dyn HirAnalysisDb,
    instance: SemanticInstance<'db>,
) -> Result<NormalizedSemanticBody<'db>, SemanticNormalizeError<'db>> {
    let raw = canonicalize_semantic_consts(db, instance);
    let typed_body = instance.key(db).instantiate_typed_body(db);
    let cx = NormalizeCtxt::new(db, instance, raw, typed_body.assumptions());
    cx.normalize()
}

struct NormalizeCtxt<'db> {
    db: &'db dyn HirAnalysisDb,
    instance: SemanticInstance<'db>,
    raw: SemanticBody<'db>,
    assumptions: crate::analysis::ty::trait_resolution::PredicateListId<'db>,
    locals: Vec<NSLocal<'db>>,
    borrow_roots: Vec<NBorrowRoot<'db>>,
}

impl<'db> NormalizeCtxt<'db> {
    fn new(
        db: &'db dyn HirAnalysisDb,
        instance: SemanticInstance<'db>,
        raw: SemanticBody<'db>,
        assumptions: crate::analysis::ty::trait_resolution::PredicateListId<'db>,
    ) -> Self {
        let local_capacity = raw.locals.len();
        Self {
            db,
            instance,
            raw,
            assumptions,
            locals: Vec::with_capacity(local_capacity),
            borrow_roots: Vec::new(),
        }
    }

    fn normalize(mut self) -> Result<NormalizedSemanticBody<'db>, SemanticNormalizeError<'db>> {
        self.normalize_locals();
        if self.raw.blocks.is_empty() {
            return Ok(empty_normalized_body(
                &self.raw,
                self.locals,
                self.borrow_roots,
            ));
        }

        let mut blocks = Vec::with_capacity(self.raw.blocks.len());
        for block in &self.raw.blocks {
            let stmts = block
                .stmts
                .iter()
                .map(|stmt| {
                    Ok(NSStmt {
                        origin: stmt.origin,
                        kind: self.normalize_stmt(stmt.origin, &stmt.kind)?,
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
            let terminator = NSTerminator {
                origin: block.terminator.origin,
                kind: self.normalize_terminator(block.terminator.origin, &block.terminator.kind),
            };
            blocks.push(NSBlock { stmts, terminator });
        }

        Ok(NormalizedSemanticBody {
            owner: self.instance,
            template_owner: self.raw.template_owner,
            locals: self.locals,
            blocks,
            borrow_roots: self.borrow_roots,
        })
    }

    fn normalize_locals(&mut self) {
        let raw_locals = self.raw.locals.clone();
        for (idx, local) in raw_locals.iter().enumerate() {
            let local_id = SLocalId::from_u32(idx as u32);
            let lowering = self.normalize_local_lowering(local_id, local);
            self.locals.push(NSLocal {
                ty: local.ty,
                mutability: local.mutability,
                source: local.source,
                lowering,
            });
        }
    }

    fn normalize_local_lowering(
        &mut self,
        local: SLocalId,
        raw_local: &crate::analysis::semantic::SLocal<'db>,
    ) -> NormalizedBindingLowering<'db> {
        match raw_local.role.clone() {
            SemanticLocalRole::Erased => NormalizedBindingLowering::Erased,
            SemanticLocalRole::DirectValue { provenance } => {
                let root = match provenance {
                    ValueProvenance::Ordinary => self.push_local_root(local, raw_local.source),
                    ValueProvenance::RootProvider(binding) => self.push_provider_root(binding),
                };
                NormalizedBindingLowering::ValueLocal { root }
            }
            SemanticLocalRole::PlaceBoundValue { provider, value_ty } => {
                let root = self.push_provider_root(provider);
                NormalizedBindingLowering::PlaceBoundValue { root, value_ty }
            }
            SemanticLocalRole::PlaceCarrier { value_ty }
            | SemanticLocalRole::DirectCarrier {
                provider: None,
                target_ty: value_ty,
            } => NormalizedBindingLowering::CarrierLocal {
                root: Some(self.push_local_root(local, raw_local.source)),
                provider: None,
                target_ty: value_ty,
            },
            SemanticLocalRole::DirectCarrier {
                provider,
                target_ty,
            } => NormalizedBindingLowering::CarrierLocal {
                root: Some(self.push_local_root(local, raw_local.source)),
                provider,
                target_ty,
            },
        }
    }

    fn push_local_root(
        &mut self,
        local: SLocalId,
        source: Option<LocalBinding<'db>>,
    ) -> NBorrowRootId {
        let root = NBorrowRootId::from_u32(self.borrow_roots.len() as u32);
        let param_idx = source.and_then(|binding| match binding {
            LocalBinding::Param { idx, .. } => Some(idx as u32),
            _ => None,
        });
        self.borrow_roots.push(if let Some(param_idx) = param_idx {
            NBorrowRoot::Param { local, param_idx }
        } else {
            NBorrowRoot::LocalSlot { local }
        });
        root
    }

    fn push_provider_root(
        &mut self,
        binding: crate::semantic::ProviderBinding<'db>,
    ) -> NBorrowRootId {
        let root = NBorrowRootId::from_u32(self.borrow_roots.len() as u32);
        self.borrow_roots.push(NBorrowRoot::Provider { binding });
        root
    }

    fn normalize_stmt(
        &self,
        origin: crate::analysis::semantic::SemOrigin<'db>,
        stmt: &SStmtKind<'db>,
    ) -> Result<NSStmtKind<'db>, SemanticNormalizeError<'db>> {
        match stmt {
            SStmtKind::Assign { dst, expr } => Ok(NSStmtKind::Assign {
                dst: *dst,
                expr: self.normalize_expr(origin, *dst, expr)?,
            }),
            SStmtKind::Store { dst, src } => Ok(NSStmtKind::Store {
                dst: self.normalize_place(dst, origin)?,
                src: *src,
            }),
        }
    }

    fn normalize_terminator(
        &self,
        origin: crate::analysis::semantic::SemOrigin<'db>,
        term: &STerminatorKind<'db>,
    ) -> NSTerminatorKind<'db> {
        match term {
            STerminatorKind::Goto(bb) => NSTerminatorKind::Goto(*bb),
            STerminatorKind::Branch {
                cond,
                then_bb,
                else_bb,
            } => NSTerminatorKind::Branch {
                cond: self.normalize_operand(*cond, origin),
                then_bb: *then_bb,
                else_bb: *else_bb,
            },
            STerminatorKind::MatchEnum {
                value,
                enum_ty,
                cases,
                default,
            } => NSTerminatorKind::MatchEnum {
                value: self.normalize_operand(*value, origin),
                enum_ty: *enum_ty,
                cases: cases.clone(),
                default: *default,
            },
            STerminatorKind::Return(value) => {
                NSTerminatorKind::Return(value.map(|value| self.normalize_operand(value, origin)))
            }
        }
    }

    fn normalize_expr(
        &self,
        origin: crate::analysis::semantic::SemOrigin<'db>,
        dst: SLocalId,
        expr: &SExpr<'db>,
    ) -> Result<NExpr<'db>, SemanticNormalizeError<'db>> {
        let dst_ty = self.locals[dst.index()].ty;
        Ok(match expr {
            SExpr::Use(value) => self
                .normalize_direct_read(origin, *value, dst_ty)?
                .unwrap_or(NExpr::Use(self.normalize_operand(*value, origin))),
            SExpr::Const(const_) => NExpr::Const(const_.clone()),
            SExpr::Unary { op, value } => NExpr::Unary {
                op: *op,
                value: self.normalize_operand(*value, origin),
            },
            SExpr::Binary { op, lhs, rhs } => NExpr::Binary {
                op: *op,
                lhs: self.normalize_operand(*lhs, origin),
                rhs: self.normalize_operand(*rhs, origin),
            },
            SExpr::Cast { value, to } => NExpr::Cast {
                value: self.normalize_operand(*value, origin),
                to: *to,
            },
            SExpr::AggregateMake { ty, fields } => NExpr::AggregateMake {
                ty: *ty,
                fields: fields
                    .iter()
                    .map(|field| self.normalize_operand(*field, origin))
                    .collect::<Vec<_>>()
                    .into_boxed_slice(),
            },
            SExpr::EnumMake {
                enum_ty,
                variant,
                fields,
            } => NExpr::EnumMake {
                enum_ty: *enum_ty,
                variant: *variant,
                fields: fields
                    .iter()
                    .map(|field| self.normalize_operand(*field, origin))
                    .collect::<Vec<_>>()
                    .into_boxed_slice(),
            },
            SExpr::Field { base, field } => {
                let place =
                    self.project_local_place(*base, Projection::Field(field.0 as usize), origin)?;
                NExpr::ReadPlace {
                    mode: self.read_mode_for_place(origin, dst_ty, &place),
                    place,
                }
            }
            SExpr::Index { base, index } => {
                let place = self.project_local_place(
                    *base,
                    Projection::Index(IndexSource::Dynamic(*index)),
                    origin,
                )?;
                NExpr::ReadPlace {
                    mode: self.read_mode_for_place(origin, dst_ty, &place),
                    place,
                }
            }
            SExpr::Borrow {
                place,
                kind,
                provider,
            } => NExpr::Borrow {
                place: self.normalize_place(place, origin)?,
                kind: *kind,
                provider: *provider,
            },
            SExpr::GetEnumTag { value } => NExpr::GetEnumTag {
                value: self.normalize_operand(*value, origin),
            },
            SExpr::IsEnumVariant { value, variant } => NExpr::IsEnumVariant {
                value: self.normalize_operand(*value, origin),
                variant: *variant,
            },
            SExpr::ExtractEnumField {
                value,
                variant,
                field,
            } => NExpr::ExtractEnumField {
                value: self.normalize_operand(*value, origin),
                variant: *variant,
                field: *field,
            },
            SExpr::CodeRegionOffset { region } => NExpr::CodeRegionOffset {
                region: region.clone(),
            },
            SExpr::CodeRegionLen { region } => NExpr::CodeRegionLen {
                region: region.clone(),
            },
            SExpr::Call {
                callee,
                args,
                effect_args,
            } => NExpr::Call {
                callee: *callee,
                args: args
                    .iter()
                    .enumerate()
                    .map(|(idx, arg)| self.normalize_call_arg(*callee, idx, *arg, origin))
                    .collect::<Vec<_>>()
                    .into_boxed_slice(),
                effect_args: effect_args
                    .iter()
                    .map(|arg| self.normalize_effect_arg(arg, origin))
                    .collect::<Result<Vec<_>, _>>()?
                    .into_boxed_slice(),
            },
        })
    }

    fn normalize_effect_arg(
        &self,
        arg: &crate::analysis::semantic::SEffectArg<'db>,
        origin: crate::analysis::semantic::SemOrigin<'db>,
    ) -> Result<NEffectArg<'db>, SemanticNormalizeError<'db>> {
        Ok(NEffectArg {
            arg: match &arg.arg {
                crate::analysis::semantic::SEffectArgValue::Place(place) => {
                    NEffectArgValue::Place(self.normalize_place(place, origin)?)
                }
                crate::analysis::semantic::SEffectArgValue::Value(value) => {
                    NEffectArgValue::Value(self.normalize_operand(*value, origin))
                }
            },
            pass_mode: arg.pass_mode,
            target_ty: arg.target_ty,
            provider: arg.provider,
        })
    }

    fn normalize_direct_read(
        &self,
        origin: crate::analysis::semantic::SemOrigin<'db>,
        local: SLocalId,
        ty: TyId<'db>,
    ) -> Result<Option<NExpr<'db>>, SemanticNormalizeError<'db>> {
        let Some(crate::analysis::semantic::SemOrigin::Expr(_)) = Some(origin) else {
            return Ok(None);
        };
        let Some(place) = self.local_read_place(local, false, origin)? else {
            return Ok(None);
        };
        let mode = self.read_mode_for_place(origin, ty, &place);
        Ok(Some(NExpr::ReadPlace { place, mode }))
    }

    fn normalize_operand(
        &self,
        local: SLocalId,
        origin: crate::analysis::semantic::SemOrigin<'db>,
    ) -> NOperand {
        NOperand {
            local,
            mode: self.read_mode_for_local(local, origin, self.locals[local.index()].ty),
        }
    }

    fn normalize_call_arg(
        &self,
        callee: crate::analysis::semantic::SemanticCalleeRef<'db>,
        idx: usize,
        local: SLocalId,
        origin: crate::analysis::semantic::SemOrigin<'db>,
    ) -> NOperand {
        let mode = match callee.key.owner(self.db) {
            crate::analysis::ty::ty_check::BodyOwner::Func(func) => func
                .params(self.db)
                .nth(idx)
                .map(|param| param.mode(self.db))
                .filter(|mode| *mode == crate::hir_def::FuncParamMode::View)
                .map(|_| ReadMode::Copy)
                .unwrap_or_else(|| {
                    self.read_mode_for_local(local, origin, self.locals[local.index()].ty)
                }),
            _ => self.read_mode_for_local(local, origin, self.locals[local.index()].ty),
        };
        NOperand { local, mode }
    }

    fn project_local_place(
        &self,
        local: SLocalId,
        projection: Projection<TyId<'db>, crate::analysis::semantic::VariantIndex, SLocalId>,
        origin: crate::analysis::semantic::SemOrigin<'db>,
    ) -> Result<NSPlace<'db>, SemanticNormalizeError<'db>> {
        let mut place = self
            .local_read_place(local, true, origin)?
            .ok_or(SemanticNormalizeError::MissingBorrowRoot { local })?;
        place.path.push(projection);
        Ok(place)
    }

    fn normalize_place(
        &self,
        place: &SPlace,
        origin: crate::analysis::semantic::SemOrigin<'db>,
    ) -> Result<NSPlace<'db>, SemanticNormalizeError<'db>> {
        let mut lowered = self
            .local_read_place(place.local, true, origin)?
            .ok_or(SemanticNormalizeError::MissingBorrowRoot { local: place.local })?;
        for elem in place.path.iter() {
            lowered.path.push(match elem {
                SPlaceElem::Field(field) => Projection::Field(field.0 as usize),
                SPlaceElem::Index(index) => Projection::Index(IndexSource::Dynamic(*index)),
            });
        }
        Ok(lowered)
    }

    fn local_read_place(
        &self,
        local: SLocalId,
        allow_carrier: bool,
        origin: crate::analysis::semantic::SemOrigin<'db>,
    ) -> Result<Option<NSPlace<'db>>, SemanticNormalizeError<'db>> {
        let Some(local_data) = self.locals.get(local.index()) else {
            return Ok(None);
        };
        Ok(match &local_data.lowering {
            NormalizedBindingLowering::Erased => None,
            NormalizedBindingLowering::ValueLocal { root }
            | NormalizedBindingLowering::PlaceBoundValue { root, .. } => Some(NSPlace {
                root: NSPlaceRoot::Root(*root),
                path: ProjectionPath::default(),
            }),
            NormalizedBindingLowering::CarrierLocal { .. } if !allow_carrier => None,
            NormalizedBindingLowering::CarrierLocal { .. } => {
                if local_data.ty.as_capability(self.db).is_none() {
                    return Err(SemanticNormalizeError::IllegalCarrierPlace { local, origin });
                }
                Some(NSPlace {
                    root: NSPlaceRoot::CarrierDerefLocal(local),
                    path: ProjectionPath::default(),
                })
            }
        })
    }

    fn read_mode(
        &self,
        origin: crate::analysis::semantic::SemOrigin<'db>,
        ty: TyId<'db>,
    ) -> ReadMode {
        match origin {
            crate::analysis::semantic::SemOrigin::Expr(expr)
                if self
                    .instance
                    .key(self.db)
                    .instantiate_typed_body(self.db)
                    .is_implicit_move(expr)
                    || !ty_is_copy(
                        self.db,
                        self.raw.template_owner.scope(),
                        ty,
                        self.assumptions,
                    ) =>
            {
                ReadMode::Move
            }
            _ => ReadMode::Copy,
        }
    }

    fn read_mode_for_local(
        &self,
        local: SLocalId,
        origin: crate::analysis::semantic::SemOrigin<'db>,
        ty: TyId<'db>,
    ) -> ReadMode {
        match self.locals.get(local.index()).map(|local| &local.lowering) {
            Some(NormalizedBindingLowering::ValueLocal { root })
            | Some(NormalizedBindingLowering::PlaceBoundValue { root, .. }) => {
                self.read_mode_for_root(origin, ty, *root)
            }
            Some(NormalizedBindingLowering::CarrierLocal { .. }) => ReadMode::Copy,
            Some(NormalizedBindingLowering::Erased) | None => self.read_mode(origin, ty),
        }
    }

    fn read_mode_for_place(
        &self,
        origin: crate::analysis::semantic::SemOrigin<'db>,
        ty: TyId<'db>,
        place: &NSPlace<'db>,
    ) -> ReadMode {
        match place.root {
            NSPlaceRoot::CarrierDerefLocal(_) => ReadMode::Copy,
            NSPlaceRoot::Root(root) => self.read_mode_for_root(origin, ty, root),
        }
    }

    fn read_mode_for_root(
        &self,
        origin: crate::analysis::semantic::SemOrigin<'db>,
        ty: TyId<'db>,
        root: NBorrowRootId,
    ) -> ReadMode {
        match self.borrow_roots.get(root.index()) {
            Some(NBorrowRoot::Provider { .. }) => ReadMode::Copy,
            Some(NBorrowRoot::Param { .. }) | Some(NBorrowRoot::LocalSlot { .. }) | None => {
                self.read_mode(origin, ty)
            }
        }
    }
}
