use cranelift_entity::EntityRef;

use crate::{
    analysis::{
        HirAnalysisDb,
        semantic::{
            PlaceProvenance, SExpr, SLocalId, SPlace, SPlaceElem, SStmtKind, STerminatorKind,
            SemanticBody, SemanticInstance, SemanticLocalRole, SemanticProjection, ValueProvenance,
            ctfe::canonicalize_semantic_consts, semantic_instance_assumptions,
        },
        ty::{ty_check::LocalBinding, ty_def::TyId, ty_is_copy},
    },
    hir_def::{Expr, ExprId, Partial},
    projection::{IndexSource, Projection, ProjectionPath},
};

use super::ir::{
    NBorrowRoot, NBorrowRootId, NEffectArg, NEffectArgValue, NExpr, NLocalFacts, NLocalInterface,
    NLocalOrigin, NLocalRootDemand, NOperand, NSBlock, NSLocal, NSPlace, NSPlaceRoot, NSStmt,
    NSStmtKind, NSTerminator, NSTerminatorKind, NormalizedBindingLowering, NormalizedSemanticBody,
    NormalizedSemanticBodyId, ReadMode, SemanticNormalizeError, SemanticNormalizeErrorId,
    SemanticNormalizeResult, empty_normalized_body, local_has_runtime_move_semantics,
};

pub fn normalize_semantic_body<'db>(
    db: &'db dyn HirAnalysisDb,
    instance: SemanticInstance<'db>,
) -> Result<NormalizedSemanticBody<'db>, SemanticNormalizeError<'db>> {
    match normalized_semantic_body_query(db, instance) {
        SemanticNormalizeResult::Ok(body) => Ok(body.body(db).clone()),
        SemanticNormalizeResult::Err(err) => Err(err.err(db).clone()),
    }
}

#[salsa::tracked]
fn normalized_semantic_body_query<'db>(
    db: &'db dyn HirAnalysisDb,
    instance: SemanticInstance<'db>,
) -> SemanticNormalizeResult<'db> {
    let raw = canonicalize_semantic_consts(db, instance);
    let cx = NormalizeCtxt::new(
        db,
        instance,
        raw,
        semantic_instance_assumptions(db, instance),
    );
    match cx.normalize() {
        Ok(body) => SemanticNormalizeResult::Ok(NormalizedSemanticBodyId::new(db, body)),
        Err(err) => SemanticNormalizeResult::Err(SemanticNormalizeErrorId::new(db, err)),
    }
}

struct NormalizeCtxt<'db> {
    db: &'db dyn HirAnalysisDb,
    instance: SemanticInstance<'db>,
    raw: SemanticBody<'db>,
    assumptions: crate::analysis::ty::trait_resolution::PredicateListId<'db>,
    locals: Vec<Option<NSLocal<'db>>>,
    local_state: Vec<LocalNormState>,
    borrow_roots: Vec<NBorrowRoot<'db>>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LocalNormState {
    Unseen,
    Visiting,
    Done,
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
            locals: vec![None; local_capacity],
            local_state: vec![LocalNormState::Unseen; local_capacity],
            borrow_roots: Vec::new(),
        }
    }

    fn normalize(mut self) -> Result<NormalizedSemanticBody<'db>, SemanticNormalizeError<'db>> {
        self.normalize_locals()?;
        if self.raw.blocks.is_empty() {
            return Ok(empty_normalized_body(
                &self.raw,
                self.locals
                    .into_iter()
                    .map(|local| local.expect("all locals normalized"))
                    .collect(),
                self.borrow_roots,
            ));
        }

        let mut blocks = Vec::with_capacity(self.raw.blocks.len());
        let raw_blocks = self.raw.blocks.clone();
        for block in &raw_blocks {
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
        self.populate_root_demand(&blocks);

        Ok(NormalizedSemanticBody {
            owner: self.instance,
            template_owner: self.raw.template_owner,
            locals: self
                .locals
                .into_iter()
                .map(|local| local.expect("all locals normalized"))
                .collect(),
            blocks,
            borrow_roots: self.borrow_roots,
        })
    }

    fn normalize_locals(&mut self) -> Result<(), SemanticNormalizeError<'db>> {
        let raw_locals = self.raw.locals.clone();
        for (idx, local) in raw_locals.iter().enumerate() {
            let local_id = SLocalId::from_u32(idx as u32);
            self.ensure_local_normalized(local_id, local)?;
        }
        Ok(())
    }

    fn ensure_local_normalized(
        &mut self,
        local: SLocalId,
        raw_local: &crate::analysis::semantic::SLocal<'db>,
    ) -> Result<(), SemanticNormalizeError<'db>> {
        match self.local_state[local.index()] {
            LocalNormState::Done => return Ok(()),
            LocalNormState::Visiting => {
                return Err(SemanticNormalizeError::LocalProvenanceCycle {
                    owner: self.instance,
                    local,
                });
            }
            LocalNormState::Unseen => {}
        }

        self.local_state[local.index()] = LocalNormState::Visiting;
        let lowering = self.normalize_local_lowering(local, raw_local)?;
        let facts = self.normalize_local_facts(local, raw_local, &lowering)?;
        self.locals[local.index()] = Some(NSLocal {
            ty: raw_local.ty,
            mutability: raw_local.mutability,
            source: raw_local.source,
            lowering,
            facts,
        });
        self.local_state[local.index()] = LocalNormState::Done;
        Ok(())
    }

    fn normalize_local_lowering(
        &mut self,
        local: SLocalId,
        raw_local: &crate::analysis::semantic::SLocal<'db>,
    ) -> Result<NormalizedBindingLowering<'db>, SemanticNormalizeError<'db>> {
        match raw_local.role.clone() {
            SemanticLocalRole::Erased => Ok(NormalizedBindingLowering::Erased),
            SemanticLocalRole::DirectValue { provenance } => {
                let place = self.normalize_value_provenance(local, raw_local.source, provenance)?;
                Ok(NormalizedBindingLowering::ValueLocal { place })
            }
            SemanticLocalRole::PlaceBoundValue {
                provenance,
                value_ty,
            } => {
                let place = self.normalize_place_provenance(local, provenance)?;
                Ok(NormalizedBindingLowering::PlaceBoundValue { place, value_ty })
            }
            SemanticLocalRole::PlaceCarrier { value_ty }
            | SemanticLocalRole::DirectCarrier {
                provider: None,
                target_ty: value_ty,
            } => Ok(NormalizedBindingLowering::CarrierLocal {
                root: Some(self.push_local_root(local, raw_local.source)),
                provider: None,
                target_ty: value_ty,
            }),
            SemanticLocalRole::DirectCarrier {
                provider,
                target_ty,
            } => Ok(NormalizedBindingLowering::CarrierLocal {
                root: Some(self.push_local_root(local, raw_local.source)),
                provider,
                target_ty,
            }),
        }
    }

    fn normalize_local_facts(
        &mut self,
        local: SLocalId,
        raw_local: &crate::analysis::semantic::SLocal<'db>,
        _: &NormalizedBindingLowering<'db>,
    ) -> Result<NLocalFacts<'db>, SemanticNormalizeError<'db>> {
        let (interface, origin) = match &raw_local.role {
            SemanticLocalRole::Erased => (NLocalInterface::Erased, NLocalOrigin::SelfRooted),
            SemanticLocalRole::DirectValue { provenance } => (
                NLocalInterface::DirectValue,
                match provenance {
                    ValueProvenance::Ordinary => NLocalOrigin::SelfRooted,
                    ValueProvenance::RootProvider(provider) => {
                        NLocalOrigin::RootProvider(provider.clone())
                    }
                },
            ),
            SemanticLocalRole::PlaceBoundValue { provenance, .. } => (
                NLocalInterface::PlaceBoundValue,
                match provenance {
                    PlaceProvenance::RootProvider(provider) => {
                        NLocalOrigin::RootProvider(provider.clone())
                    }
                    PlaceProvenance::Derived { .. } => NLocalOrigin::AliasedPlace,
                },
            ),
            SemanticLocalRole::PlaceCarrier { .. } => {
                (NLocalInterface::PlaceCarrier, NLocalOrigin::SelfRooted)
            }
            SemanticLocalRole::DirectCarrier { provider, .. } => (
                NLocalInterface::DirectCarrier,
                provider
                    .clone()
                    .map_or(NLocalOrigin::SelfRooted, NLocalOrigin::RootProvider),
            ),
        };
        let snapshot_source_place = raw_local
            .snapshot_source
            .clone()
            .map(|snapshot_source| self.normalize_snapshot_source(local, snapshot_source))
            .transpose()?;
        let mut root_demand = NLocalRootDemand::default();
        if matches!(
            interface,
            NLocalInterface::PlaceBoundValue | NLocalInterface::PlaceCarrier
        ) {
            root_demand.always_rooted = true;
        }
        Ok(NLocalFacts {
            interface,
            origin,
            snapshot_source_place,
            root_demand,
        })
    }

    fn populate_root_demand(&mut self, blocks: &[NSBlock<'db>]) {
        let mut root_demand = self
            .locals
            .iter()
            .map(|local| {
                local
                    .as_ref()
                    .map_or(NLocalRootDemand::default(), |local| local.facts.root_demand)
            })
            .collect::<Vec<_>>();

        for block in blocks {
            for stmt in &block.stmts {
                match &stmt.kind {
                    NSStmtKind::Assign { expr, .. } => {
                        self.mark_expr_root_demand(expr, &mut root_demand);
                    }
                    NSStmtKind::Store { dst, .. } => {
                        self.mark_place_root_demand(dst, &mut root_demand, |demand| {
                            demand.written_by_place = true;
                        });
                    }
                }
            }
        }

        for (idx, local) in self.locals.iter().enumerate() {
            let Some(local) = local.as_ref() else {
                continue;
            };
            if let NormalizedBindingLowering::ValueLocal { place } = &local.lowering {
                let local_id = SLocalId::from_u32(idx as u32);
                if !self.is_self_rooted_value_place(local_id, place) {
                    self.mark_place_root_demand(place, &mut root_demand, |demand| {
                        demand.nonself_backing_place = true;
                    });
                }
            }
            if let Some(place) = local.snapshot_source_place() {
                self.mark_place_root_demand(place, &mut root_demand, |demand| {
                    demand.nonself_backing_place = true;
                });
            }
        }

        for (idx, demand) in root_demand.into_iter().enumerate() {
            if let Some(local) = self.locals[idx].as_mut() {
                local.facts.root_demand = demand;
            }
        }
    }

    fn mark_expr_root_demand(&self, expr: &NExpr<'db>, root_demand: &mut [NLocalRootDemand]) {
        match expr {
            NExpr::Use(_)
            | NExpr::Const(_)
            | NExpr::Unary { .. }
            | NExpr::Binary { .. }
            | NExpr::Cast { .. }
            | NExpr::AggregateMake { .. }
            | NExpr::EnumMake { .. }
            | NExpr::GetEnumTag { .. }
            | NExpr::IsEnumVariant { .. }
            | NExpr::ExtractEnumField { .. }
            | NExpr::CodeRegionRef { .. }
            | NExpr::CodeRegionOffset { .. }
            | NExpr::CodeRegionLen { .. } => {}
            NExpr::ReadPlace { place, .. } => {
                self.mark_place_root_demand(place, root_demand, |demand| {
                    demand.read_by_place = true;
                });
            }
            NExpr::Borrow { place, .. } => {
                self.mark_place_root_demand(place, root_demand, |demand| {
                    demand.borrowed_or_addr_taken = true;
                });
            }
            NExpr::Call { effect_args, .. } => {
                for arg in effect_args {
                    if let NEffectArgValue::Place(place) = &arg.arg {
                        self.mark_place_root_demand(place, root_demand, |demand| {
                            demand.passed_by_place = true;
                        });
                    }
                }
            }
        }
    }

    fn mark_place_root_demand(
        &self,
        place: &NSPlace<'db>,
        root_demand: &mut [NLocalRootDemand],
        mut mark: impl FnMut(&mut NLocalRootDemand),
    ) {
        let local = match place.root {
            NSPlaceRoot::CarrierDerefLocal(local) => Some(local),
            NSPlaceRoot::Root(root) => match self.borrow_roots.get(root.index()) {
                Some(NBorrowRoot::Param { local, .. }) | Some(NBorrowRoot::LocalSlot { local }) => {
                    Some(*local)
                }
                Some(NBorrowRoot::Provider { .. }) | None => None,
            },
        };
        if let Some(local) = local
            && let Some(demand) = root_demand.get_mut(local.index())
        {
            mark(demand);
        }
    }

    fn is_self_rooted_value_place(&self, local: SLocalId, place: &NSPlace<'db>) -> bool {
        if !place.path.is_empty() {
            return false;
        }
        match place.root {
            NSPlaceRoot::CarrierDerefLocal(root_local) => root_local == local,
            NSPlaceRoot::Root(root) => matches!(
                self.borrow_roots.get(root.index()),
                Some(NBorrowRoot::Param { local: root_local, .. }
                    | NBorrowRoot::LocalSlot { local: root_local }) if *root_local == local
            ),
        }
    }

    fn propagated_place(&mut self, local: SLocalId) -> Option<NSPlace<'db>> {
        let local_data = self.locals.get(local.index())?.as_ref()?;
        match &local_data.lowering {
            NormalizedBindingLowering::ValueLocal { place }
            | NormalizedBindingLowering::PlaceBoundValue { place, .. } => Some(place.clone()),
            NormalizedBindingLowering::CarrierLocal { provider, .. } => {
                let provider = provider.clone();
                let is_capability = local_data.ty.as_capability(self.db).is_some();
                provider
                    .map(|provider| self.provider_root_place(provider))
                    .or_else(|| {
                        is_capability.then_some(NSPlace {
                            root: NSPlaceRoot::CarrierDerefLocal(local),
                            path: ProjectionPath::default(),
                        })
                    })
            }
            NormalizedBindingLowering::Erased => None,
        }
    }

    fn normalize_value_provenance(
        &mut self,
        local: SLocalId,
        source: Option<LocalBinding<'db>>,
        provenance: ValueProvenance<'db>,
    ) -> Result<NSPlace<'db>, SemanticNormalizeError<'db>> {
        match provenance {
            ValueProvenance::Ordinary => Ok(self.local_root_place(local, source)),
            ValueProvenance::RootProvider(binding) => Ok(self.provider_root_place(binding)),
        }
    }

    fn normalize_place_provenance(
        &mut self,
        local: SLocalId,
        provenance: PlaceProvenance<'db>,
    ) -> Result<NSPlace<'db>, SemanticNormalizeError<'db>> {
        match provenance {
            PlaceProvenance::RootProvider(binding) => Ok(self.provider_root_place(binding)),
            PlaceProvenance::Derived { base, path } => {
                self.normalize_derived_place(local, base, &path)
            }
        }
    }

    fn normalize_snapshot_source(
        &mut self,
        local: SLocalId,
        snapshot_source: PlaceProvenance<'db>,
    ) -> Result<NSPlace<'db>, SemanticNormalizeError<'db>> {
        match snapshot_source {
            PlaceProvenance::RootProvider(binding) => Ok(self.provider_root_place(binding)),
            PlaceProvenance::Derived { base, path } => {
                self.normalize_snapshot_derived_place(local, base, &path)
            }
        }
    }

    fn push_semantic_projection_path(place: &mut NSPlace<'db>, path: &[SemanticProjection<'db>]) {
        for projection in path {
            place
                .path
                .push(Self::normalize_semantic_projection(projection));
        }
    }

    fn normalize_semantic_projection(
        projection: &SemanticProjection<'db>,
    ) -> Projection<TyId<'db>, crate::analysis::semantic::VariantIndex, SLocalId> {
        match projection {
            SemanticProjection::Field(field_idx) => Projection::Field(*field_idx),
            SemanticProjection::VariantField {
                variant,
                enum_ty,
                field_idx,
            } => Projection::VariantField {
                variant: *variant,
                enum_ty: *enum_ty,
                field_idx: *field_idx,
            },
            SemanticProjection::Index(index) => Projection::Index(IndexSource::Dynamic(*index)),
        }
    }

    fn normalize_snapshot_derived_place(
        &mut self,
        local: SLocalId,
        base: SLocalId,
        path: &[SemanticProjection<'db>],
    ) -> Result<NSPlace<'db>, SemanticNormalizeError<'db>> {
        let raw_base = self.raw.locals[base.index()].clone();
        self.ensure_local_normalized(base, &raw_base)?;
        let mut place = self.snapshot_source_base_place(local, base)?;
        Self::push_semantic_projection_path(&mut place, path);
        Ok(place)
    }

    fn snapshot_source_base_place(
        &mut self,
        local: SLocalId,
        base: SLocalId,
    ) -> Result<NSPlace<'db>, SemanticNormalizeError<'db>> {
        self.locals
            .get(base.index())
            .and_then(|local| local.as_ref())
            .and_then(|local| local.snapshot_source_place().cloned())
            .or_else(|| self.propagated_place(base))
            .ok_or(SemanticNormalizeError::NonPlaceDerivedValue {
                owner: self.instance,
                local,
                base,
            })
    }

    fn normalize_derived_place(
        &mut self,
        local: SLocalId,
        base: SLocalId,
        path: &[SemanticProjection<'db>],
    ) -> Result<NSPlace<'db>, SemanticNormalizeError<'db>> {
        let raw_base = self.raw.locals[base.index()].clone();
        self.ensure_local_normalized(base, &raw_base)?;
        let mut place =
            self.propagated_place(base)
                .ok_or(SemanticNormalizeError::NonPlaceDerivedValue {
                    owner: self.instance,
                    local,
                    base,
                })?;
        Self::push_semantic_projection_path(&mut place, path);
        Ok(place)
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

    fn local_root_place(
        &mut self,
        local: SLocalId,
        source: Option<LocalBinding<'db>>,
    ) -> NSPlace<'db> {
        NSPlace {
            root: NSPlaceRoot::Root(self.push_local_root(local, source)),
            path: ProjectionPath::default(),
        }
    }

    fn push_provider_root(
        &mut self,
        binding: crate::semantic::ProviderBinding<'db>,
    ) -> NBorrowRootId {
        let root = NBorrowRootId::from_u32(self.borrow_roots.len() as u32);
        self.borrow_roots.push(NBorrowRoot::Provider { binding });
        root
    }

    fn provider_root_place(
        &mut self,
        binding: crate::semantic::ProviderBinding<'db>,
    ) -> NSPlace<'db> {
        NSPlace {
            root: NSPlaceRoot::Root(self.push_provider_root(binding)),
            path: ProjectionPath::default(),
        }
    }

    fn normalize_stmt(
        &mut self,
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
        &mut self,
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
        &mut self,
        origin: crate::analysis::semantic::SemOrigin<'db>,
        dst: SLocalId,
        expr: &SExpr<'db>,
    ) -> Result<NExpr<'db>, SemanticNormalizeError<'db>> {
        let dst_ty = self.locals[dst.index()]
            .as_ref()
            .expect("all locals normalized before block lowering")
            .ty;
        Ok(match expr {
            SExpr::Forward(value) => NExpr::Use(self.normalize_operand(*value, origin)),
            SExpr::UseValue(value) => self
                .normalize_direct_read(origin, *value, dst_ty)?
                .unwrap_or(NExpr::Use(self.normalize_operand(*value, origin))),
            SExpr::CodeRegionRef { region } => NExpr::CodeRegionRef {
                region: region.clone(),
            },
            SExpr::Const(const_) => NExpr::Const(const_.clone()),
            SExpr::Unary { op, value } => NExpr::Unary {
                op: *op,
                value: self.normalize_operand_at(*value, origin, 0),
            },
            SExpr::Binary { op, lhs, rhs } => NExpr::Binary {
                op: *op,
                lhs: self.normalize_operand_at(*lhs, origin, 0),
                rhs: self.normalize_operand_at(*rhs, origin, 1),
            },
            SExpr::Cast { value, to } => NExpr::Cast {
                value: self.normalize_operand_at(*value, origin, 0),
                to: *to,
            },
            SExpr::AggregateMake { ty, fields } => NExpr::AggregateMake {
                ty: *ty,
                fields: fields
                    .iter()
                    .enumerate()
                    .map(|(idx, field)| self.normalize_operand_at(*field, origin, idx))
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
                    .enumerate()
                    .map(|(idx, field)| self.normalize_operand_at(*field, origin, idx))
                    .collect::<Vec<_>>()
                    .into_boxed_slice(),
            },
            SExpr::ReadPlace { place } => {
                let place = self.normalize_place(place, origin)?;
                NExpr::ReadPlace {
                    mode: self.read_mode_for_place(origin, dst_ty, &place),
                    place,
                }
            }
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
                value: self.normalize_copy_operand(*value, origin),
            },
            SExpr::IsEnumVariant { value, variant } => NExpr::IsEnumVariant {
                value: self.normalize_copy_operand(*value, origin),
                variant: *variant,
            },
            SExpr::ExtractEnumField {
                value,
                variant,
                field,
            } => NExpr::ExtractEnumField {
                value: self.normalize_operand_at(*value, origin, 0),
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
                    .map(|(idx, arg)| self.normalize_call_arg_at(*callee, idx, *arg, origin))
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
        &mut self,
        arg: &crate::analysis::semantic::SEffectArg<'db>,
        origin: crate::analysis::semantic::SemOrigin<'db>,
    ) -> Result<NEffectArg<'db>, SemanticNormalizeError<'db>> {
        Ok(NEffectArg {
            binding_idx: arg.binding_idx,
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
        &mut self,
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
        let ty = self.locals[local.index()]
            .as_ref()
            .expect("all locals normalized before operand lowering")
            .ty;
        NOperand {
            local,
            origin: Self::origin_expr(origin),
            mode: self.read_mode_for_operand(local, origin, ty),
        }
    }

    fn normalize_operand_at(
        &self,
        local: SLocalId,
        origin: crate::analysis::semantic::SemOrigin<'db>,
        idx: usize,
    ) -> NOperand {
        self.normalize_operand(local, self.operand_origin(origin, idx).unwrap_or(origin))
    }

    fn normalize_copy_operand(
        &self,
        local: SLocalId,
        origin: crate::analysis::semantic::SemOrigin<'db>,
    ) -> NOperand {
        NOperand {
            local,
            origin: Self::origin_expr(origin),
            mode: ReadMode::Copy,
        }
    }

    fn normalize_call_arg(
        &self,
        callee: crate::analysis::semantic::SemanticCalleeRef<'db>,
        idx: usize,
        local: SLocalId,
        origin: crate::analysis::semantic::SemOrigin<'db>,
    ) -> NOperand {
        let ty = self.locals[local.index()]
            .as_ref()
            .expect("all locals normalized before call arg lowering")
            .ty;
        let mode = match callee.key.owner(self.db) {
            crate::analysis::ty::ty_check::BodyOwner::Func(func) => func
                .params(self.db)
                .nth(idx)
                .map(|param| param.mode(self.db))
                .filter(|mode| *mode == crate::hir_def::FuncParamMode::View)
                .map(|_| ReadMode::Copy)
                .unwrap_or_else(|| self.read_mode_for_operand(local, origin, ty)),
            _ => self.read_mode_for_operand(local, origin, ty),
        };
        NOperand {
            local,
            origin: Self::origin_expr(origin),
            mode,
        }
    }

    fn normalize_call_arg_at(
        &self,
        callee: crate::analysis::semantic::SemanticCalleeRef<'db>,
        idx: usize,
        local: SLocalId,
        origin: crate::analysis::semantic::SemOrigin<'db>,
    ) -> NOperand {
        self.normalize_call_arg(
            callee,
            idx,
            local,
            self.operand_origin(origin, idx).unwrap_or(origin),
        )
    }

    fn origin_expr(origin: crate::analysis::semantic::SemOrigin<'db>) -> Option<ExprId> {
        match origin {
            crate::analysis::semantic::SemOrigin::Expr(expr) => Some(expr),
            crate::analysis::semantic::SemOrigin::Stmt(_)
            | crate::analysis::semantic::SemOrigin::Body(_)
            | crate::analysis::semantic::SemOrigin::Synthetic => None,
        }
    }

    fn operand_origin(
        &self,
        origin: crate::analysis::semantic::SemOrigin<'db>,
        idx: usize,
    ) -> Option<crate::analysis::semantic::SemOrigin<'db>> {
        self.operand_expr(origin, idx)
            .map(crate::analysis::semantic::SemOrigin::Expr)
    }

    fn operand_expr(
        &self,
        origin: crate::analysis::semantic::SemOrigin<'db>,
        idx: usize,
    ) -> Option<ExprId> {
        let crate::analysis::semantic::SemOrigin::Expr(expr) = origin else {
            return None;
        };
        let body = self
            .instance
            .key(self.db)
            .instantiate_typed_body(self.db)
            .body()?;
        let Partial::Present(expr_data) = expr.data(self.db, body) else {
            return None;
        };
        match expr_data {
            Expr::Un(inner, _) | Expr::Cast(inner, _) | Expr::Field(inner, _) => {
                (idx == 0).then_some(*inner)
            }
            Expr::Bin(lhs, rhs, _) | Expr::Assign(lhs, rhs) | Expr::AugAssign(lhs, rhs, _) => {
                [*lhs, *rhs].get(idx).copied()
            }
            Expr::Tuple(items) | Expr::Array(items) => items.get(idx).copied(),
            Expr::ArrayRep(item, _) => Some(*item),
            Expr::Call(_, args) => args.get(idx).map(|arg| arg.expr),
            Expr::MethodCall(receiver, _, _, args) => {
                if idx == 0 {
                    Some(*receiver)
                } else {
                    args.get(idx - 1).map(|arg| arg.expr)
                }
            }
            Expr::Lit(_)
            | Expr::Path(_)
            | Expr::RecordInit(_, _)
            | Expr::Block(_)
            | Expr::If(_, _, _)
            | Expr::Match(_, _)
            | Expr::With(_, _) => None,
        }
    }

    fn project_local_place(
        &mut self,
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

    fn normalize_place_elem(
        elem: &SPlaceElem,
    ) -> Projection<TyId<'db>, crate::analysis::semantic::VariantIndex, SLocalId> {
        match elem {
            SPlaceElem::Field(field) => Projection::Field(field.0 as usize),
            SPlaceElem::Index(index) => Projection::Index(IndexSource::Dynamic(*index)),
        }
    }

    fn push_place_path(place: &mut NSPlace<'db>, path: &[SPlaceElem]) {
        for elem in path {
            place.path.push(Self::normalize_place_elem(elem));
        }
    }

    fn normalize_place(
        &mut self,
        place: &SPlace,
        origin: crate::analysis::semantic::SemOrigin<'db>,
    ) -> Result<NSPlace<'db>, SemanticNormalizeError<'db>> {
        let mut lowered = self
            .local_read_place(place.local, true, origin)?
            .ok_or(SemanticNormalizeError::MissingBorrowRoot { local: place.local })?;
        Self::push_place_path(&mut lowered, &place.path);
        Ok(lowered)
    }

    fn local_read_place(
        &mut self,
        local: SLocalId,
        allow_carrier: bool,
        origin: crate::analysis::semantic::SemOrigin<'db>,
    ) -> Result<Option<NSPlace<'db>>, SemanticNormalizeError<'db>> {
        let Some(local_data) = self
            .locals
            .get(local.index())
            .and_then(|local| local.as_ref())
        else {
            return Ok(None);
        };
        Ok(match &local_data.lowering {
            NormalizedBindingLowering::Erased => None,
            NormalizedBindingLowering::ValueLocal { place }
            | NormalizedBindingLowering::PlaceBoundValue { place, .. } => Some(place.clone()),
            NormalizedBindingLowering::CarrierLocal { .. } if !allow_carrier => None,
            NormalizedBindingLowering::CarrierLocal { provider, .. } => {
                let provider = provider.clone();
                let is_capability = local_data.ty.as_capability(self.db).is_some();
                if provider.is_none() && !is_capability {
                    return Err(SemanticNormalizeError::IllegalCarrierPlace { local, origin });
                }
                provider
                    .map(|provider| self.provider_root_place(provider))
                    .or_else(|| {
                        Some(NSPlace {
                            root: NSPlaceRoot::CarrierDerefLocal(local),
                            path: ProjectionPath::default(),
                        })
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

    fn read_mode_for_operand(
        &self,
        local: SLocalId,
        origin: crate::analysis::semantic::SemOrigin<'db>,
        ty: TyId<'db>,
    ) -> ReadMode {
        let Some(local) = self
            .locals
            .get(local.index())
            .and_then(|local| local.as_ref())
        else {
            return self.read_mode(origin, ty);
        };
        if !local_has_runtime_move_semantics(self.db, local, &self.borrow_roots) {
            return ReadMode::Copy;
        }
        if !ty_is_copy(
            self.db,
            self.raw.template_owner.scope(),
            ty,
            self.assumptions,
        ) {
            return ReadMode::Move;
        }
        match origin {
            crate::analysis::semantic::SemOrigin::Expr(expr)
                if self
                    .instance
                    .key(self.db)
                    .instantiate_typed_body(self.db)
                    .is_implicit_move(expr) =>
            {
                ReadMode::Move
            }
            _ => ReadMode::Copy,
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
            Some(NBorrowRoot::Param { param_idx, .. })
                if self
                    .instance
                    .key(self.db)
                    .instantiate_typed_body(self.db)
                    .param_binding(*param_idx as usize)
                    .is_some_and(|binding| {
                        matches!(
                            binding,
                            LocalBinding::Param {
                                mode: crate::hir_def::FuncParamMode::View,
                                ..
                            }
                        )
                    }) =>
            {
                match origin {
                    crate::analysis::semantic::SemOrigin::Expr(expr)
                        if self
                            .instance
                            .key(self.db)
                            .instantiate_typed_body(self.db)
                            .is_implicit_move(expr) =>
                    {
                        ReadMode::Move
                    }
                    _ => ReadMode::Copy,
                }
            }
            Some(NBorrowRoot::Param { .. }) | Some(NBorrowRoot::LocalSlot { .. }) | None => {
                self.read_mode(origin, ty)
            }
        }
    }
}
