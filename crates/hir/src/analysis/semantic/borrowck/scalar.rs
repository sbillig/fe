//! Trusted scalar predicates share the borrow checker's scoped guard algebra.
use cranelift_entity::EntityRef;
use rustc_hash::FxHashSet;

use crate::{
    analysis::{
        HirAnalysisDb,
        semantic::{
            SemOrigin,
            capability::{
                external::ExternalOrigin,
                guard::Guard,
                index::{BinderScope, IndexExpr},
                region::{RegionRoot, RegionSet},
                state::BorrowState,
            },
            definite_assignment::literal_bool_cond,
            diagnostics::{SemanticDiagnostic, normalized_body_internal_diag},
            normalized::{
                NDataPath, NDataProjection, NExpr, NIndex, NPlace, NPlaceBase, NRootKind,
                NStatement, NStatementKind, NTerminatorKind, NValueDefinition, NValueId,
                NormalizedBody, copied_scalar_ty,
            },
        },
        ty::{
            corelib::{PrimitiveWrapperCallKind, core_primitive_wrapper_call_kind},
            ty_check::BodyOwner,
            ty_def::{TyBase, TyData, TyId, prim_int_bits},
        },
    },
    hir_def::{BinOp, CompBinOp, LogicalBinOp, UnOp},
};

use super::{loop_certificate::frontier_candidates, solver::Borrowck};

/// Nested boolean operations followed when deriving a branch condition.
pub(super) const CONDITION_BUDGET: u8 = 16;

/// Scalar facts are generated on demand: for index selectors, loop frontiers,
/// representable integer returns, and parameters an assertion constrains on
/// every normal return.
#[derive(Default)]
pub(super) struct ScalarDemand<'db> {
    /// Values whose trusted comparisons contribute guard facts.
    pub values: FxHashSet<NValueId>,
    pub indices: FxHashSet<IndexExpr<'db>>,
    /// Selector indices may also carry unsigned bounds.
    pub selectors: FxHashSet<IndexExpr<'db>>,
    /// Block parameters that keep their incoming equalities.
    pub phis: FxHashSet<NValueId>,
    /// Exact scalar cells whose store versions are tracked.
    pub cells: FxHashSet<NPlaceBase>,
    /// Reader-loop conditions whose bounds apply once a fill is certified.
    pub bounded_readers: FxHashSet<NValueId>,
    readers: Vec<(NValueId, Vec<NValueId>)>,
}

pub(super) fn integer_model(db: &dyn HirAnalysisDb, ty: TyId<'_>) -> Option<(usize, bool)> {
    let TyData::TyBase(TyBase::Prim(primitive)) = copied_scalar_ty(db, ty).data(db) else {
        return None;
    };
    Some((prim_int_bits(*primitive)?, primitive.is_signed_int()))
}

pub(super) fn lossless_integer_cast(
    db: &dyn HirAnalysisDb,
    source: TyId<'_>,
    target: TyId<'_>,
) -> bool {
    integer_model(db, source)
        .zip(integer_model(db, target))
        .is_some_and(
            |((source_bits, source_signed), (target_bits, target_signed))| {
                source_signed == target_signed
                    && source_bits <= target_bits
                    && (!source_signed || source_bits == target_bits)
            },
        )
}

impl<'db> Borrowck<'db> {
    pub(super) fn prepare_scalar_demand(&mut self) -> Result<(), SemanticDiagnostic<'db>> {
        let (mut selectors, cells, mut values) = scalar_seeds(self.db, &self.body);
        values.retain(|value| self.supported_scalar_return(*value));
        if self.inventory.loops.has_cycle()
            && self.body.blocks.iter().flat_map(|block| &block.statements).any(|statement| {
                self.stores_capability(statement)
                    || matches!(&statement.kind, NStatementKind::Define { result, expr: NExpr::Call { .. } }
                        if self.inventory.shapes[result.index()].contains_capability(self.db))
            })
        {
            self.frontiers = frontier_candidates(self.db, &self.body)
                .map_err(|error| {
                    normalized_body_internal_diag(
                        self.db,
                        self.instance,
                        &self.body,
                        SemOrigin::Body(self.body.template_owner),
                        format!("invalid normalized loop control flow: {error:?}"),
                    )
                })?
                .into_iter()
                .flatten()
                .collect();
        }
        let mut readers = Vec::new();
        for candidate in &self.frontiers {
            let mut frontier: Vec<_> = candidate
                .loop_region
                .blocks
                .iter()
                .flat_map(|block| &self.body.blocks[block.index()].statements)
                .filter_map(|statement| match &statement.kind {
                    NStatementKind::Define {
                        result,
                        expr: NExpr::Load { place, .. },
                    } if place.base == candidate.frontier_root && place.path.is_empty() => {
                        Some(*result)
                    }
                    _ => None,
                })
                .collect();
            let indices: FxHashSet<_> = frontier.iter().map(|value| self.index(*value)).collect();
            let statements = &self.body.blocks[candidate.body.index()].statements;
            let reads = statements.iter().any(|statement| {
                matches!(&statement.kind,
                    NStatementKind::Define { result, expr: NExpr::Call { args, .. } }
                        if self.inventory.shapes[result.index()].contains_capability(self.db)
                            && args.iter().any(|argument| indices.contains(&self.index(argument.value))))
            });
            frontier.extend([candidate.header_value, candidate.bound]);
            if statements
                .iter()
                .any(|statement| self.stores_capability(statement))
            {
                selectors.extend(frontier);
            } else if reads {
                readers.push((candidate.condition, frontier));
            }
        }
        self.scalar.readers = readers;
        self.add_scalar_demand(selectors, values, cells);
        Ok(())
    }

    /// A reader loop uses its unsigned bound only once a fill is certified.
    pub(super) fn enable_bounded_readers(&mut self) {
        let mut selectors = FxHashSet::default();
        for (condition, frontier) in std::mem::take(&mut self.scalar.readers) {
            self.scalar.bounded_readers.insert(condition);
            selectors.extend(frontier);
        }
        if !selectors.is_empty() {
            self.add_scalar_demand(selectors, [], FxHashSet::default());
        }
    }

    /// Track scalar facts for new selectors and values, and everything they load.
    fn add_scalar_demand(
        &mut self,
        mut selectors: FxHashSet<NValueId>,
        values: impl IntoIterator<Item = NValueId>,
        cells: FxHashSet<NPlaceBase>,
    ) {
        let mut selector_cells = FxHashSet::default();
        close_scalar_values(self.db, &self.body, &mut selectors, &mut selector_cells);
        let selector_indices: Vec<_> = selectors.iter().map(|value| self.index(*value)).collect();
        self.scalar.selectors.extend(selector_indices);
        self.scalar
            .values
            .extend(selectors.into_iter().chain(values));
        self.scalar
            .cells
            .extend(cells.into_iter().chain(selector_cells));
        close_scalar_values(
            self.db,
            &self.body,
            &mut self.scalar.values,
            &mut self.scalar.cells,
        );
        self.scalar.indices = self
            .scalar
            .values
            .iter()
            .map(|value| self.index(*value))
            .collect();
        self.scalar.phis = self
            .scalar
            .values
            .iter()
            .copied()
            .filter(|value| self.compact_scalar_phi(*value))
            .collect();
    }

    pub(super) fn stores_capability(&self, statement: &NStatement<'db>) -> bool {
        matches!(&statement.kind,
            NStatementKind::Store { destination, value }
                if matches!(destination.base, NPlaceBase::CapabilityTarget { .. })
                    && self.inventory.shapes[value.value.index()].contains_capability(self.db))
    }

    fn compact_scalar_phi(&self, value: NValueId) -> bool {
        let NValueDefinition::BlockParam { block, index } =
            self.body.values[value.index()].definition
        else {
            return false;
        };
        let incoming: Vec<_> = self
            .body
            .blocks
            .iter()
            .flat_map(|predecessor| predecessor.terminator.kind.successors())
            .filter(|successor| successor.block == block)
            .filter_map(|successor| successor.args.get(index as usize))
            .map(|argument| self.index(argument.value))
            .collect();
        let Some(first) = incoming.first() else {
            return false;
        };
        incoming
            .iter()
            .all(|index| matches!(index, IndexExpr::Const(_)))
            || incoming.iter().all(|index| index == first)
    }

    fn supported_scalar_return(&self, mut value: NValueId) -> bool {
        loop {
            match self.body.defining_expr(value) {
                Some((_, NExpr::Forward { src })) => value = src.value,
                Some((_, NExpr::ScalarCast { value: src, to }))
                    if self.lossless_scalar_cast(src.value, *to) =>
                {
                    value = src.value;
                }
                _ => {
                    return self.compact_scalar_phi(value)
                        || !matches!(
                            self.body.values[value.index()].definition,
                            NValueDefinition::BlockParam { .. }
                        );
                }
            }
        }
    }

    pub(super) fn scalar_cell(
        &self,
        place: &NPlace<'db>,
        region: &RegionSet<'db>,
        state: &BorrowState<'db>,
    ) -> Option<RegionRoot<'db>> {
        if !place.path.is_empty()
            || !self.scalar.cells.contains(&place.base)
            || integer_model(self.db, place.ty).is_none()
        {
            return None;
        }
        let [clause] = region.clauses() else {
            return None;
        };
        if !clause.payload.path.is_empty()
            || clause.payload.views.iter().next().is_some()
            || clause.guard.scope() != state.guard().scope()
            || !state.guard().implies(&clause.guard)
        {
            return None;
        }
        match &clause.payload.root {
            RegionRoot::Root { root, .. }
                if matches!(
                    self.body.roots[root.index()].kind,
                    NRootKind::LocalSlot { .. } | NRootKind::ParamPlace { .. }
                ) =>
            {
                Some(clause.payload.root.clone())
            }
            RegionRoot::External(source)
                if matches!(&source.origin, ExternalOrigin::Input(input) if !input.is_reachable() && input.dereferences().is_empty())
                    && !source.uncertain()
                    && source.dereferences().is_empty() =>
            {
                Some(clause.payload.root.clone())
            }
            _ => None,
        }
    }

    pub(super) fn lossless_scalar_cast(&self, source: NValueId, target: TyId<'db>) -> bool {
        lossless_integer_cast(self.db, self.body.values[source.index()].ty, target)
    }

    /// The guard under which `value` equals `expected`, following trusted
    /// boolean operations within a budget of nested conditions.
    pub(super) fn condition_guard(
        &self,
        value: NValueId,
        expected: bool,
        include_bounds: bool,
        budget: u8,
    ) -> Option<Guard<'db>> {
        let always = Guard::always(&BinderScope::default());
        if let Some(actual) = literal_bool_cond(self.db, &self.body, value) {
            return (actual == expected).then_some(always);
        }
        let selected = always.with_boolean(self.boolean_choice(value), expected)?;
        if budget == 0 {
            return Some(selected);
        }
        let Some((_, expr)) = self.body.defining_expr(value) else {
            return Some(selected);
        };
        let relation = match expr {
            NExpr::Forward { src } => {
                self.condition_guard(src.value, expected, include_bounds, budget - 1)
            }
            NExpr::Unary {
                op: UnOp::Not,
                value,
            } => self.condition_guard(value.value, !expected, include_bounds, budget - 1),
            NExpr::Binary {
                op: BinOp::Logical(operator),
                lhs,
                rhs,
            } => self.logical_guard(
                *operator,
                lhs.value,
                rhs.value,
                expected,
                include_bounds,
                budget - 1,
            ),
            NExpr::Binary {
                op: BinOp::Comp(operator),
                lhs,
                rhs,
            } => self.comparison_guard(*operator, lhs.value, rhs.value, expected, include_bounds),
            NExpr::Call { callee, args, .. } => {
                let BodyOwner::Func(function) = callee.key.owner(self.db) else {
                    return Some(selected);
                };
                match (
                    core_primitive_wrapper_call_kind(
                        self.db,
                        function,
                        self.body.values[value.index()].ty,
                    ),
                    args.as_ref(),
                ) {
                    (Some(PrimitiveWrapperCallKind::Unary(UnOp::Not)), [operand]) => {
                        self.condition_guard(operand.value, !expected, include_bounds, budget - 1)
                    }
                    (Some(PrimitiveWrapperCallKind::Binary(BinOp::Comp(operator))), [lhs, rhs]) => {
                        self.comparison_guard(
                            operator,
                            lhs.value,
                            rhs.value,
                            expected,
                            include_bounds,
                        )
                    }
                    _ => Some(always),
                }
            }
            _ => Some(always),
        }?;
        selected.and(&relation)
    }

    fn logical_guard(
        &self,
        operator: LogicalBinOp,
        lhs: NValueId,
        rhs: NValueId,
        expected: bool,
        include_bounds: bool,
        budget: u8,
    ) -> Option<Guard<'db>> {
        let both = matches!(operator, LogicalBinOp::And) == expected;
        let left = self.condition_guard(lhs, expected, include_bounds, budget);
        let right = self.condition_guard(rhs, expected, include_bounds, budget);
        if both {
            left?.and(&right?)
        } else {
            match (left, right) {
                (Some(left), Some(right)) => Some(left.or(&right)),
                (left, right) => left.or(right),
            }
        }
    }

    fn comparison_guard(
        &self,
        operator: CompBinOp,
        lhs: NValueId,
        rhs: NValueId,
        expected: bool,
        include_bounds: bool,
    ) -> Option<Guard<'db>> {
        let always = Guard::always(&BinderScope::default());
        let lhs_index = self.index(lhs);
        let rhs_index = self.index(rhs);
        if !self.scalar.indices.contains(&lhs_index) && !self.scalar.indices.contains(&rhs_index) {
            return Some(always);
        }
        let lhs_ty = copied_scalar_ty(self.db, self.body.values[lhs.index()].ty);
        let rhs_ty = copied_scalar_ty(self.db, self.body.values[rhs.index()].ty);
        let Some((_, signed)) = integer_model(self.db, lhs_ty) else {
            return Some(always);
        };
        if lhs_ty != rhs_ty
            || ((!include_bounds
                || signed
                || (!self.scalar.selectors.contains(&lhs_index)
                    && !self.scalar.selectors.contains(&rhs_index)))
                && !matches!(operator, CompBinOp::Eq | CompBinOp::NotEq))
        {
            return Some(always);
        }
        let (condition, positive) = match operator {
            CompBinOp::Eq => (always.with_equality(lhs_index, rhs_index), true),
            CompBinOp::NotEq => (always.with_equality(lhs_index, rhs_index), false),
            CompBinOp::Lt => (always.with_bound(lhs_index, rhs_index), true),
            CompBinOp::LtEq => (always.with_bound(rhs_index, lhs_index), false),
            CompBinOp::Gt => (always.with_bound(rhs_index, lhs_index), true),
            CompBinOp::GtEq => (always.with_bound(lhs_index, rhs_index), false),
        };
        if expected == positive {
            condition
        } else {
            condition.map_or(Some(always.clone()), |condition| {
                always.difference(&condition)
            })
        }
    }
}

fn scalar_seeds<'db>(
    db: &dyn HirAnalysisDb,
    body: &NormalizedBody<'db>,
) -> (FxHashSet<NValueId>, FxHashSet<NPlaceBase>, Vec<NValueId>) {
    let mut selectors = FxHashSet::default();
    let mut cells = FxHashSet::default();
    let mut values = Vec::new();
    for block in &body.blocks {
        for statement in &block.statements {
            let mut add_indices = |path: &NDataPath| {
                selectors.extend(path.iter().filter_map(|projection| match projection {
                    NDataProjection::Index(NIndex::Value(value)) => Some(*value),
                    _ => None,
                }));
            };
            match &statement.kind {
                NStatementKind::Define { expr, .. } => {
                    expr.for_each_place_operand(|place| add_indices(&place.path));
                    if let NExpr::ProjectValue { path, .. } = expr {
                        add_indices(&path.0);
                    }
                }
                NStatementKind::Store { destination, value } => {
                    add_indices(&destination.path);
                    if destination.path.is_empty()
                        && destination.ty.is_integral(db)
                        && matches!(destination.base, NPlaceBase::CapabilityTarget { carrier }
                            if matches!(body.values[carrier.index()].definition, NValueDefinition::EntryParam { .. }))
                        && matches!(body.defining_expr(value.value), Some((_, NExpr::Const(_))))
                    {
                        cells.insert(destination.base);
                    }
                }
            }
        }
        if let NTerminatorKind::Return(Some(value)) = block.terminator.kind
            && body.values[value.value.index()].ty.is_integral(db)
        {
            values.push(value.value);
        }
    }
    // A branch around a failed assertion constrains every normal return, which
    // a summary exports over the integer parameters it compares.
    if body.blocks.iter().any(|block| {
        block.terminator.kind.successors().iter().any(|successor| {
            matches!(
                body.blocks[successor.block.index()].terminator.kind,
                NTerminatorKind::Assert { .. }
            )
        })
    }) {
        values.extend(
            body.values
                .iter()
                .enumerate()
                .filter(|(_, value)| {
                    matches!(value.definition, NValueDefinition::EntryParam { .. })
                        && value.ty.is_integral(db)
                })
                .map(|(index, _)| NValueId::new(index)),
        );
    }
    (selectors, cells, values)
}

fn close_scalar_values(
    db: &dyn HirAnalysisDb,
    body: &NormalizedBody<'_>,
    values: &mut FxHashSet<NValueId>,
    cells: &mut FxHashSet<NPlaceBase>,
) {
    loop {
        let previous = (values.len(), cells.len());
        for value in values.iter().copied().collect::<Vec<_>>() {
            match body.defining_expr(value) {
                Some((_, NExpr::Load { place, .. })) if place.path.is_empty() => {
                    cells.insert(place.base);
                }
                Some((_, NExpr::Forward { src })) => {
                    values.insert(src.value);
                }
                Some((_, NExpr::ScalarCast { value: src, to }))
                    if lossless_integer_cast(db, body.values[src.value.index()].ty, *to) =>
                {
                    values.insert(src.value);
                }
                _ => {}
            }
            if let NValueDefinition::BlockParam { block, index } =
                body.values[value.index()].definition
            {
                values.extend(
                    body.blocks
                        .iter()
                        .flat_map(|predecessor| predecessor.terminator.kind.successors())
                        .filter(|successor| successor.block == block)
                        .filter_map(|successor| successor.args.get(index as usize))
                        .map(|argument| argument.value),
                );
            }
        }
        for statement in body.blocks.iter().flat_map(|block| &block.statements) {
            if let NStatementKind::Define {
                result,
                expr: NExpr::Load { place, .. },
            } = &statement.kind
                && place.path.is_empty()
                && cells.contains(&place.base)
            {
                values.insert(*result);
            }
        }
        if previous == (values.len(), cells.len()) {
            break;
        }
    }
}
