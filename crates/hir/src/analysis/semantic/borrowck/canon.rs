use cranelift_entity::SecondaryMap;
use dataflow::JoinSemiLattice;
use rustc_hash::{FxHashMap, FxHashSet};
use smallvec::SmallVec;

use crate::{
    analysis::{
        HirAnalysisDb,
        semantic::{
            BorrowActivation, FieldIndex, LayoutBackingProjection, SBlockId, SLocalId, SemOrigin,
            SemanticInstance,
        },
        ty::{
            provider::{ProviderAddressSpace, ProviderKind},
            ty_check::LocalBinding,
            ty_contains_borrow,
            ty_def::{BorrowKind, TyId},
            ty_is_noesc,
        },
    },
    projection::{Aliasing, IndexSource, Projection},
};

use super::{
    diagnostics::normalized_body_internal_diag,
    ir::{
        BorrowResult, NBorrowRoot, NBorrowRootId, NExpr, NSPlace, NSPlaceRoot, NSProjectionPath,
        NSStmt, NSStmtKind, NValueOwnershipSource, NormalizedBindingLowering,
        NormalizedSemanticBody, SemanticBorrowDiagnostic, layout_path_for_semantic_projection,
        resolved_layout_backing_places, semantic_projection_for_layout_path,
        semantic_projection_ty,
    },
};

pub(super) fn address_space_for_borrow_root<'db>(
    db: &'db dyn HirAnalysisDb,
    instance: SemanticInstance<'db>,
    body: &NormalizedSemanticBody<'db>,
    root: &BorrowRoot<'db>,
    origin: SemOrigin<'db>,
) -> Result<ProviderAddressSpace, SemanticBorrowDiagnostic<'db>> {
    match root {
        BorrowRoot::Param(_) | BorrowRoot::Local(_) => Ok(ProviderAddressSpace::Memory),
        BorrowRoot::Provider(binding) => match binding.semantics.address_space {
            Some(space) => Ok(space),
            None if matches!(binding.semantics.kind, ProviderKind::RootObject) => {
                Ok(ProviderAddressSpace::Memory)
            }
            None => Err(normalized_body_internal_diag(
                db,
                instance,
                body,
                origin,
                format!(
                    "provider `{}` has no address space",
                    binding.provider_ty.pretty_print(db)
                ),
            )),
        },
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) struct LoanId(pub(super) u32);

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(super) enum BorrowRoot<'db> {
    Param(u32),
    Local(SLocalId),
    Provider(crate::semantic::ProviderBinding<'db>),
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(super) struct CanonPlace<'db> {
    pub(super) root: BorrowRoot<'db>,
    pub(super) proj: NSProjectionPath<'db>,
}

#[derive(Clone, Debug)]
pub(super) struct Loan<'db> {
    pub(super) kind: BorrowKind,
    pub(super) activation: BorrowActivation,
    pub(super) targets: FxHashSet<CanonPlace<'db>>,
    pub(super) parents: FxHashSet<LoanId>,
    pub(super) origin: SemOrigin<'db>,
}

#[derive(Clone, Debug)]
pub(super) struct MoveSite<'db> {
    pub(super) origin: SemOrigin<'db>,
    pub(super) note: String,
}

pub(super) type MovedPlaces<'db> = FxHashMap<CanonPlace<'db>, MoveSite<'db>>;
pub(super) type BlockAdjacency = SmallVec<SBlockId, 2>;
pub(super) type CfgAdjacency = SecondaryMap<SBlockId, BlockAdjacency>;

pub(super) type HeldLoanPath = Vec<LayoutBackingProjection>;
pub(super) type HeldLoans = FxHashMap<HeldLoanPath, FxHashSet<LoanId>>;

fn layout_projection_matches(lhs: LayoutBackingProjection, rhs: LayoutBackingProjection) -> bool {
    lhs == rhs
        || matches!(
            (lhs, rhs),
            (
                LayoutBackingProjection::Index(None),
                LayoutBackingProjection::Index(_)
            ) | (
                LayoutBackingProjection::Index(_),
                LayoutBackingProjection::Index(None)
            )
        )
}

fn layout_path_is_prefix(
    prefix: &[LayoutBackingProjection],
    path: &[LayoutBackingProjection],
) -> bool {
    prefix.len() <= path.len()
        && prefix
            .iter()
            .copied()
            .zip(path.iter().copied())
            .all(|(lhs, rhs)| layout_projection_matches(lhs, rhs))
}

fn prefixed_held_loans(held: HeldLoans, prefix: &[LayoutBackingProjection]) -> HeldLoans {
    held.into_iter()
        .map(|(path, loans)| {
            let mut prefixed = prefix.to_vec();
            prefixed.extend(path);
            (prefixed, loans)
        })
        .collect()
}

fn merge_held_loans(into: &mut HeldLoans, from: HeldLoans) {
    for (path, loans) in from {
        into.entry(path).or_default().extend(loans);
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(super) struct State {
    pub(super) local_loans: FxHashMap<SLocalId, HeldLoans>,
}

impl State {
    pub(super) fn loans_in(&self, local: SLocalId) -> FxHashSet<LoanId> {
        self.local_loans
            .get(&local)
            .into_iter()
            .flat_map(|held| held.values())
            .flatten()
            .copied()
            .collect()
    }

    pub(super) fn held_loans_in(&self, local: SLocalId) -> HeldLoans {
        self.local_loans.get(&local).cloned().unwrap_or_default()
    }

    pub(super) fn assign_loans(&mut self, local: SLocalId, loans: FxHashSet<LoanId>) {
        let held = if loans.is_empty() {
            FxHashMap::default()
        } else {
            FxHashMap::from_iter([(HeldLoanPath::new(), loans)])
        };
        self.assign_held_loans(local, held);
    }

    pub(super) fn assign_held_loans(&mut self, local: SLocalId, held: HeldLoans) {
        if held.is_empty() {
            self.local_loans.remove(&local);
        } else {
            self.local_loans.insert(local, held);
        }
    }

    pub(super) fn projected_held_loans(
        &self,
        local: SLocalId,
        projection: &[LayoutBackingProjection],
    ) -> HeldLoans {
        let mut out = HeldLoans::default();
        for (path, loans) in self.local_loans.get(&local).into_iter().flatten() {
            let projected = if layout_path_is_prefix(projection, path) {
                path[projection.len()..].to_vec()
            } else if layout_path_is_prefix(path, projection) {
                HeldLoanPath::new()
            } else {
                continue;
            };
            out.entry(projected).or_default().extend(loans);
        }
        out
    }

    pub(super) fn replace_held_loans(
        &mut self,
        local: SLocalId,
        projection: &[LayoutBackingProjection],
        replacement: HeldLoans,
    ) {
        let mut held = self.held_loans_in(local);
        if !projection.contains(&LayoutBackingProjection::Index(None)) {
            held.retain(|path, _| {
                !layout_path_is_prefix(projection, path)
                    || projection.iter().copied().zip(path.iter().copied()).any(
                        |(written, held)| {
                            matches!(
                                (written, held),
                                (
                                    LayoutBackingProjection::Index(Some(_)),
                                    LayoutBackingProjection::Index(None)
                                )
                            )
                        },
                    )
            });
        }
        merge_held_loans(&mut held, prefixed_held_loans(replacement, projection));
        self.assign_held_loans(local, held);
    }
}

impl JoinSemiLattice for State {
    fn join_into(&mut self, other: &Self) -> bool {
        let mut changed = false;
        for (local, held) in &other.local_loans {
            let entry = self.local_loans.entry(*local).or_default();
            for (path, loans) in held {
                let entry = entry.entry(path.clone()).or_default();
                let before = entry.len();
                entry.extend(loans.iter().copied());
                changed |= before != entry.len();
            }
        }
        changed
    }
}

pub(super) struct BorrowCanonCx<'a, 'db> {
    db: &'db dyn HirAnalysisDb,
    instance: SemanticInstance<'db>,
    body: &'a NormalizedSemanticBody<'db>,
    loans: &'a [Loan<'db>],
    loan_for_local: &'a FxHashMap<SLocalId, LoanId>,
    constant_indices: &'a SecondaryMap<SLocalId, Option<usize>>,
}

impl<'a, 'db> BorrowCanonCx<'a, 'db> {
    pub(super) fn new(
        db: &'db dyn HirAnalysisDb,
        instance: SemanticInstance<'db>,
        body: &'a NormalizedSemanticBody<'db>,
        loans: &'a [Loan<'db>],
        loan_for_local: &'a FxHashMap<SLocalId, LoanId>,
        constant_indices: &'a SecondaryMap<SLocalId, Option<usize>>,
    ) -> Self {
        Self {
            db,
            instance,
            body,
            loans,
            loan_for_local,
            constant_indices,
        }
    }

    fn materialize_constant_indices(&self, path: &NSProjectionPath<'db>) -> NSProjectionPath<'db> {
        let mut out = NSProjectionPath::new();
        for projection in path.iter() {
            out.push(match projection {
                Projection::Index(IndexSource::Dynamic(index)) => self.constant_indices[*index]
                    .map_or_else(
                        || projection.clone(),
                        |index| Projection::Index(IndexSource::Constant(index)),
                    ),
                projection => projection.clone(),
            });
        }
        out
    }

    fn layout_path(&self, path: &NSProjectionPath<'db>) -> Option<Vec<LayoutBackingProjection>> {
        layout_path_for_semantic_projection(&self.materialize_constant_indices(path))
    }

    pub(super) fn apply_stmt_state_with_call_loans(
        &self,
        state: &mut State,
        stmt: &NSStmt<'db>,
        call_result_loans: Option<&[(BorrowResult, LoanId)]>,
    ) {
        match &stmt.kind {
            NSStmtKind::Assign { dst, expr } => {
                let held = match expr {
                    NExpr::Use(src) => {
                        let own = self.own_held_loan_for_local(*dst);
                        if own.is_empty() {
                            self.propagated_held_loans(*dst, state.held_loans_in(src.local))
                        } else {
                            own
                        }
                    }
                    NExpr::Borrow { .. } => self.own_held_loan_for_local(*dst),
                    NExpr::Call { args, .. } => {
                        let own = self.own_held_loan_for_local(*dst);
                        if !own.is_empty() {
                            own
                        } else if let Some(call_result_loans) = call_result_loans {
                            let mut held = HeldLoans::default();
                            for (result, loan) in call_result_loans {
                                held.entry(result.projection.clone())
                                    .or_default()
                                    .insert(*loan);
                            }
                            held
                        } else {
                            let mut held = HeldLoans::default();
                            for arg in args {
                                merge_held_loans(&mut held, state.held_loans_in(arg.local));
                            }
                            self.propagated_held_loans(*dst, held)
                        }
                    }
                    NExpr::AggregateMake { ty, fields } => {
                        let mut held = HeldLoans::default();
                        for (idx, field) in fields.iter().enumerate() {
                            let projection = if ty.is_array(self.db) {
                                LayoutBackingProjection::Index(Some(idx))
                            } else {
                                let Ok(idx) = u16::try_from(idx) else {
                                    continue;
                                };
                                LayoutBackingProjection::Field(FieldIndex(idx))
                            };
                            merge_held_loans(
                                &mut held,
                                prefixed_held_loans(
                                    state.held_loans_in(field.local),
                                    &[projection],
                                ),
                            );
                        }
                        held
                    }
                    NExpr::EnumMake {
                        variant, fields, ..
                    } => {
                        let mut held = HeldLoans::default();
                        for (idx, field) in fields.iter().enumerate() {
                            let Ok(field_idx) = u16::try_from(idx) else {
                                continue;
                            };
                            merge_held_loans(
                                &mut held,
                                prefixed_held_loans(
                                    state.held_loans_in(field.local),
                                    &[LayoutBackingProjection::VariantField {
                                        variant: *variant,
                                        field: FieldIndex(field_idx),
                                    }],
                                ),
                            );
                        }
                        held
                    }
                    NExpr::ArrayRepeat { value, .. } => prefixed_held_loans(
                        state.held_loans_in(value.local),
                        &[LayoutBackingProjection::Index(None)],
                    ),
                    NExpr::ExtractEnumField {
                        value,
                        variant,
                        field,
                    } => {
                        let held = state.projected_held_loans(
                            value.local,
                            &[LayoutBackingProjection::VariantField {
                                variant: *variant,
                                field: *field,
                            }],
                        );
                        self.propagated_held_loans(*dst, held)
                    }
                    NExpr::ReadPlace { place, .. } => {
                        let own = self.own_held_loan_for_local(*dst);
                        if own.is_empty() {
                            self.place_base_local(place)
                                .map(|base| {
                                    let projection =
                                        self.layout_path(&place.path).unwrap_or_default();
                                    self.propagated_held_loans(
                                        *dst,
                                        state.projected_held_loans(base, &projection),
                                    )
                                })
                                .unwrap_or_default()
                        } else {
                            own
                        }
                    }
                    _ => HeldLoans::default(),
                };
                state.assign_held_loans(*dst, held);
            }
            NSStmtKind::Store { dst, src } => {
                if let NSPlaceRoot::Root(root) = dst.root
                    && let Some(base) = self.root_base_local(root)
                {
                    let path = self.materialize_constant_indices(&dst.path);
                    if self
                        .body
                        .place_root_ty(&dst.root)
                        .and_then(|ty| semantic_projection_ty(self.db, ty, &path))
                        .is_some_and(|(_, traverses_capability)| traverses_capability)
                    {
                        return;
                    }
                    let projection = layout_path_for_semantic_projection(&path).unwrap_or_default();
                    state.replace_held_loans(base, &projection, state.held_loans_in(src.local));
                }
            }
        }
    }

    fn propagated_held_loans(&self, dst: SLocalId, held: HeldLoans) -> HeldLoans {
        if held.is_empty() {
            return held;
        }
        let Some(dst_local) = self.body.local(dst) else {
            return HeldLoans::default();
        };
        if dst_local.ty.as_capability(self.db).is_some() || ty_is_noesc(self.db, dst_local.ty) {
            held
        } else {
            HeldLoans::default()
        }
    }

    fn own_held_loan_for_local(&self, local: SLocalId) -> HeldLoans {
        self.loan_for_local
            .get(&local)
            .copied()
            .map(|loan| FxHashMap::from_iter([(HeldLoanPath::new(), FxHashSet::from_iter([loan]))]))
            .unwrap_or_default()
    }

    fn root_base_local(&self, root: NBorrowRootId) -> Option<SLocalId> {
        match self.body.root(root)? {
            NBorrowRoot::Param { local, .. } | NBorrowRoot::LocalSlot { local } => Some(*local),
            NBorrowRoot::Provider { .. } => None,
        }
    }

    fn place_base_local(&self, place: &NSPlace<'db>) -> Option<SLocalId> {
        match place.root {
            NSPlaceRoot::CarrierDerefLocal(local) => Some(local),
            NSPlaceRoot::Root(root) => self.root_base_local(root),
        }
    }

    pub(super) fn canonicalize_value_base(
        &self,
        state: &State,
        local: SLocalId,
    ) -> FxHashSet<CanonPlace<'db>> {
        if self
            .body
            .local(local)
            .is_some_and(|local| local.ty.as_borrow(self.db).is_some())
        {
            return self.borrow_local_targets(state, local);
        }

        let Some(local_data) = self.body.local(local) else {
            return FxHashSet::default();
        };
        if let Some(place) = local_data.lowering.place() {
            return self.canonicalize_place_targets(state, place);
        }
        let root = match &local_data.lowering {
            NormalizedBindingLowering::CarrierLocal { root, provider, .. } => provider
                .clone()
                .map(BorrowRoot::Provider)
                .or_else(|| root.and_then(|root| self.root_to_borrow_root(root))),
            NormalizedBindingLowering::Erased => None,
            NormalizedBindingLowering::ValueLocal { .. }
            | NormalizedBindingLowering::PlaceBoundValue { .. } => unreachable!(),
        };
        root.into_iter()
            .map(|root| CanonPlace {
                root,
                proj: NSProjectionPath::default(),
            })
            .collect()
    }

    pub(super) fn canonicalize_value_projection(
        &self,
        state: &State,
        local: SLocalId,
        projection: &NSProjectionPath<'db>,
    ) -> FxHashSet<CanonPlace<'db>> {
        let Some(local_data) = self.body.local(local) else {
            return FxHashSet::default();
        };
        let projection = self.materialize_constant_indices(projection);
        let traverses_capability = semantic_projection_ty(self.db, local_data.ty, &projection)
            .is_none_or(|(_, traverses_capability)| traverses_capability);
        if local_data.ty.as_borrow(self.db).is_none() && traverses_capability {
            let held = layout_path_for_semantic_projection(&projection).map_or_else(
                || state.loans_in(local),
                |projection| {
                    state
                        .projected_held_loans(local, &projection)
                        .into_values()
                        .flatten()
                        .collect()
                },
            );
            let targets = held
                .into_iter()
                .flat_map(|loan| self.loans[loan.0 as usize].targets.iter().cloned())
                .collect::<FxHashSet<_>>();
            if !targets.is_empty() {
                return targets;
            }
        }
        if local_data.ty.as_borrow(self.db).is_none() {
            let resolved =
                resolved_layout_backing_places(local_data.layout_backing_sources(), &projection);
            if !resolved.is_empty() {
                return resolved
                    .iter()
                    .flat_map(|place| self.canonicalize_place_targets(state, place))
                    .collect();
            }
        }
        if local_data.ty.as_borrow(self.db).is_none()
            && ty_contains_borrow(self.db, local_data.ty)
            && !matches!(
                local_data.source,
                Some(LocalBinding::Param { .. } | LocalBinding::EffectParam { .. })
            )
            && traverses_capability
        {
            return FxHashSet::default();
        }

        self.canonicalize_value_base(state, local)
            .into_iter()
            .map(|base| CanonPlace {
                root: base.root,
                proj: base.proj.concat(&projection),
            })
            .collect()
    }

    pub(super) fn canonicalize_value_layout_projection(
        &self,
        state: &State,
        local: SLocalId,
        projection: &[LayoutBackingProjection],
    ) -> FxHashSet<CanonPlace<'db>> {
        let Some(local_ty) = self.body.local(local).map(|local| local.ty) else {
            return FxHashSet::default();
        };
        semantic_projection_for_layout_path(self.db, local_ty, projection).map_or_else(
            || self.borrow_local_targets(state, local),
            |projection| self.canonicalize_value_projection(state, local, &projection),
        )
    }

    pub(super) fn canonicalize_place_layout_projection(
        &self,
        state: &State,
        place: &NSPlace<'db>,
        target_ty: TyId<'db>,
        projection: &[LayoutBackingProjection],
    ) -> FxHashSet<CanonPlace<'db>> {
        if let Some(suffix) = semantic_projection_for_layout_path(self.db, target_ty, projection) {
            if let Some(local) = self.place_base_local(place) {
                let path = self
                    .materialize_constant_indices(&place.path)
                    .concat(&suffix);
                return self.canonicalize_value_projection(state, local, &path);
            }
            let mut projected = place.clone();
            projected.path = projected.path.concat(&suffix);
            return self.canonicalize_place_targets(state, &projected);
        }

        if let Some(local) = self.place_base_local(place) {
            let mut path = self
                .layout_path(&self.materialize_constant_indices(&place.path))
                .unwrap_or_default();
            path.extend_from_slice(projection);
            let targets = state
                .projected_held_loans(local, &path)
                .into_values()
                .flatten()
                .flat_map(|loan| self.loans[loan.0 as usize].targets.iter().cloned())
                .collect::<FxHashSet<_>>();
            if !targets.is_empty() {
                return targets;
            }
            if self.body.local(local).is_some_and(|local| {
                ty_contains_borrow(self.db, local.ty)
                    && !matches!(
                        local.source,
                        Some(LocalBinding::Param { .. } | LocalBinding::EffectParam { .. })
                    )
            }) {
                return FxHashSet::default();
            }
        }

        self.canonicalize_place_targets(state, place)
    }

    pub(super) fn borrow_local_targets(
        &self,
        state: &State,
        local: SLocalId,
    ) -> FxHashSet<CanonPlace<'db>> {
        let Some(local_data) = self.body.local(local) else {
            return FxHashSet::default();
        };
        let mut out = FxHashSet::default();
        for loan in state.loans_in(local) {
            out.extend(self.loans[loan.0 as usize].targets.iter().cloned());
        }
        if !out.is_empty() {
            return out;
        }

        if local_data.ty.as_borrow(self.db).is_none() && ty_contains_borrow(self.db, local_data.ty)
        {
            let targets = local_data
                .layout_backing_sources()
                .iter()
                .flat_map(|source| self.canonicalize_place_targets(state, &source.source))
                .collect::<FxHashSet<_>>();
            if !targets.is_empty() {
                return targets;
            }
        }

        if let Some(place) = local_data.lowering.place() {
            return self.canonicalize_place_targets(state, place);
        }
        match &local_data.lowering {
            NormalizedBindingLowering::CarrierLocal { root, provider, .. } => provider
                .clone()
                .map(BorrowRoot::Provider)
                .or_else(|| root.and_then(|root| self.root_to_borrow_root(root)))
                .into_iter()
                .map(|root| CanonPlace {
                    root,
                    proj: NSProjectionPath::default(),
                })
                .collect(),
            NormalizedBindingLowering::Erased => FxHashSet::default(),
            NormalizedBindingLowering::ValueLocal { .. }
            | NormalizedBindingLowering::PlaceBoundValue { .. } => FxHashSet::default(),
        }
    }

    pub(super) fn all_value_targets(
        &self,
        state: &State,
        local: SLocalId,
    ) -> FxHashSet<CanonPlace<'db>> {
        let mut out =
            self.canonicalize_value_projection(state, local, &NSProjectionPath::default());
        out.extend(self.borrow_local_targets(state, local));
        out
    }

    pub(super) fn canonicalize_place(
        &self,
        state: &State,
        place: &NSPlace<'db>,
        origin: SemOrigin<'db>,
    ) -> Result<FxHashSet<CanonPlace<'db>>, SemanticBorrowDiagnostic<'db>> {
        let out = self.canonicalize_place_targets(state, place);
        if out.is_empty() {
            return Err(self.internal_diag(
                origin,
                "cannot canonicalize carrier-rooted place".to_string(),
            ));
        }
        Ok(out)
    }

    fn canonicalize_place_targets(
        &self,
        state: &State,
        place: &NSPlace<'db>,
    ) -> FxHashSet<CanonPlace<'db>> {
        let path = self.materialize_constant_indices(&place.path);
        match place.root {
            NSPlaceRoot::Root(root) => {
                let borrow_root = self
                    .root_to_borrow_root(root)
                    .expect("normalized borrow root");
                if let Some(local) = self.root_base_local(root)
                    && let Some(local_data) = self.body.local(local)
                    && !local_data.ownership_sources().is_empty()
                {
                    let mut out = FxHashSet::default();
                    for source in local_data.ownership_sources() {
                        match source {
                            NValueOwnershipSource::Local => {
                                out.insert(CanonPlace {
                                    root: borrow_root.clone(),
                                    proj: path.clone(),
                                });
                            }
                            NValueOwnershipSource::Place(source) => {
                                for mut target in self.canonicalize_place_targets(state, source) {
                                    target.proj = target.proj.concat(&path);
                                    out.insert(target);
                                }
                            }
                        }
                    }
                    return out;
                }
                FxHashSet::from_iter([CanonPlace {
                    root: borrow_root,
                    proj: path,
                }])
            }
            NSPlaceRoot::CarrierDerefLocal(local) => {
                let suffix = path;
                let mut out = FxHashSet::default();
                let mut resolved = false;
                for loan in state.loans_in(local) {
                    resolved = true;
                    for target in &self.loans[loan.0 as usize].targets {
                        out.insert(CanonPlace {
                            root: target.root.clone(),
                            proj: target.proj.concat(&suffix),
                        });
                    }
                }
                if !resolved
                    && let Some(NormalizedBindingLowering::CarrierLocal { root, provider, .. }) =
                        self.body.local(local).map(|local| &local.lowering)
                {
                    if let Some(provider) = provider {
                        out.insert(CanonPlace {
                            root: BorrowRoot::Provider(provider.clone()),
                            proj: suffix.clone(),
                        });
                    } else if let Some(root) = root.and_then(|root| self.root_to_borrow_root(root))
                    {
                        out.insert(CanonPlace { root, proj: suffix });
                    }
                }
                out
            }
        }
    }

    pub(super) fn root_to_borrow_root(&self, root: NBorrowRootId) -> Option<BorrowRoot<'db>> {
        match self.body.root(root)? {
            NBorrowRoot::Param { param_idx, .. } => Some(BorrowRoot::Param(*param_idx)),
            NBorrowRoot::LocalSlot { local } => Some(BorrowRoot::Local(*local)),
            NBorrowRoot::Provider { binding, .. } => Some(BorrowRoot::Provider(binding.clone())),
        }
    }

    pub(super) fn mut_loans_for_place(
        &self,
        state: &State,
        place: &NSPlace<'db>,
    ) -> FxHashSet<LoanId> {
        self.loans_for_place(state, place)
            .into_iter()
            .filter(|loan| self.loans[loan.0 as usize].kind == BorrowKind::Mut)
            .collect()
    }

    pub(super) fn mut_loans_for_place_targets(
        &self,
        state: &State,
        place: &NSPlace<'db>,
        targets: &FxHashSet<CanonPlace<'db>>,
    ) -> FxHashSet<LoanId> {
        self.loans_for_place_targets(state, place, targets)
            .into_iter()
            .filter(|loan| self.loans[loan.0 as usize].kind == BorrowKind::Mut)
            .collect()
    }

    pub(super) fn loans_for_place_targets(
        &self,
        state: &State,
        place: &NSPlace<'db>,
        targets: &FxHashSet<CanonPlace<'db>>,
    ) -> FxHashSet<LoanId> {
        self.place_base_local(place).map_or_else(
            || self.loans_for_place(state, place),
            |local| self.loans_for_value_targets(state, local, targets),
        )
    }

    pub(super) fn loans_for_place(&self, state: &State, place: &NSPlace<'db>) -> FxHashSet<LoanId> {
        match place.root {
            NSPlaceRoot::CarrierDerefLocal(local) => state.loans_in(local),
            NSPlaceRoot::Root(_) => FxHashSet::default(),
        }
    }

    pub(super) fn mut_loans_for_value(&self, state: &State, local: SLocalId) -> FxHashSet<LoanId> {
        state
            .loans_in(local)
            .into_iter()
            .filter(|loan| self.loans[loan.0 as usize].kind == BorrowKind::Mut)
            .collect()
    }

    pub(super) fn mut_loans_for_value_targets(
        &self,
        state: &State,
        local: SLocalId,
        targets: &FxHashSet<CanonPlace<'db>>,
    ) -> FxHashSet<LoanId> {
        self.loans_for_value_targets(state, local, targets)
            .into_iter()
            .filter(|loan| self.loans[loan.0 as usize].kind == BorrowKind::Mut)
            .collect()
    }

    pub(super) fn loans_for_value_targets(
        &self,
        state: &State,
        local: SLocalId,
        targets: &FxHashSet<CanonPlace<'db>>,
    ) -> FxHashSet<LoanId> {
        state
            .loans_in(local)
            .into_iter()
            .filter(|loan| {
                let loan = &self.loans[loan.0 as usize];
                place_set_overlaps(&loan.targets, targets)
            })
            .collect()
    }

    fn internal_diag(
        &self,
        origin: SemOrigin<'db>,
        message: String,
    ) -> SemanticBorrowDiagnostic<'db> {
        normalized_body_internal_diag(self.db, self.instance, self.body, origin, message)
    }
}

pub(super) fn place_set_overlaps<'db>(
    lhs: &FxHashSet<CanonPlace<'db>>,
    rhs: &FxHashSet<CanonPlace<'db>>,
) -> bool {
    lhs.iter()
        .any(|lhs| rhs.iter().any(|rhs| places_overlap(lhs, rhs)))
}

pub(super) fn places_overlap<'db>(lhs: &CanonPlace<'db>, rhs: &CanonPlace<'db>) -> bool {
    lhs.root == rhs.root && !matches!(lhs.proj.may_alias(&rhs.proj), Aliasing::No)
}
