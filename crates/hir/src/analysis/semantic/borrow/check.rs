//! Liveness and the checks of a solved analysis.
use std::collections::BTreeSet;

use cranelift_entity::EntityRef;

use super::{
    analysis::{Analysis, MoveKey, State, Token, TokenId, TokenKind, TokenSet, union_into},
    place::{AbsPlace, Base, Path, paths_overlap},
    requires_memory_receiver,
};
use crate::{
    analysis::{
        semantic::{
            SemOrigin, SemanticInstanceKey,
            diagnostics::{
                SemanticDiagnostic, SemanticDiagnosticKind, SemanticDiagnosticSpan, operand_origin,
            },
            get_or_build_semantic_instance,
            normalized::{
                NBlockId, NEffectArg, NEffectArgValue, NExpr, NOperand, NPlace, NPlaceBase,
                NRootKind, NStatement, NStatementKind, NStructuralPath, NTerminatorKind, NValueId,
                ReadMode,
                access::{AccessTarget, OperationAccess, path_indices},
            },
        },
        ty::{
            corelib::{
                MemoryAccessKind, effect_param_starts_external_calls, external_call_state_access,
            },
            provider::{ProviderAddressSpace, ProviderKind},
            ty_check::BodyOwner,
            ty_def::{BorrowKind, CapabilityKind, TyId},
        },
    },
    hir_def::Func,
};

type Diag<'db> = SemanticDiagnostic<'db>;
type Moved = BTreeSet<(MoveKey, Path)>;

/// The more precise of two origins for a diagnostic: an expression, then a
/// statement, then the body.
fn specific_origin<'db>(first: SemOrigin<'db>, second: SemOrigin<'db>) -> SemOrigin<'db> {
    match (first, second) {
        (SemOrigin::Expr(_), _)
        | (SemOrigin::Stmt(_), SemOrigin::Body(_) | SemOrigin::Synthetic) => first,
        _ => second,
    }
}

/// Values and roots that may be read later.
#[derive(Clone, Default, PartialEq, Eq)]
struct Live {
    values: Vec<bool>,
    roots: Vec<bool>,
}

impl Live {
    fn new(values: usize, roots: usize) -> Self {
        Self {
            values: vec![false; values],
            roots: vec![false; roots],
        }
    }

    fn join(&mut self, other: &Self) -> bool {
        let mut changed = false;
        for (target, source) in self
            .values
            .iter_mut()
            .chain(self.roots.iter_mut())
            .zip(other.values.iter().chain(&other.roots))
        {
            changed |= *source && !*target;
            *target |= *source;
        }
        changed
    }
}

impl<'db> Analysis<'_, 'db> {
    /// Check the solved body; the first violation in block order is reported.
    pub fn check(&mut self) -> Result<(), Diag<'db>> {
        self.check_pointee_types()?;
        let live_out = self.live_out();
        for block in self.reverse_postorder() {
            let Some(mut state) = self.block_entry(block) else {
                continue;
            };
            let live_after = self.live_after_statements(block, &live_out[block.index()]);
            for (index, statement) in self.statements(block) {
                let before = state.clone();
                let mut changed = false;
                self.transfer(block, index, statement, &mut state, &mut changed);
                let live = self.live_loans(&state, &live_after[index]);
                let point = Point {
                    before: &before,
                    after: &state,
                    live: &live,
                    own: self.statement_tokens.get(&(block, index)).copied(),
                };
                self.check_statement(statement, &point)?;
            }
            if !self.diverges(block) {
                self.check_terminator(block, &state)?;
            }
        }
        Ok(())
    }

    fn check_pointee_types(&self) -> Result<(), Diag<'db>> {
        let values = self
            .body
            .values
            .iter()
            .map(|value| (value.ty, value.origin));
        let roots = self.body.roots.iter().map(|root| (root.ty, root.origin));
        for (ty, origin) in values.chain(roots) {
            if self.carried(ty).pointer_to_borrow {
                return Err(self.diag(
                    SemanticDiagnosticKind::InvalidConcreteType,
                    origin,
                    format!(
                        "raw pointers cannot point to values that hold borrows: `{}`",
                        ty.pretty_print(self.db)
                    ),
                ));
            }
        }
        Ok(())
    }

    pub(super) fn diag(
        &self,
        kind: SemanticDiagnosticKind,
        origin: SemOrigin<'db>,
        message: String,
    ) -> Diag<'db> {
        SemanticDiagnostic::new(self.instance, kind, message, self.span(origin))
    }

    fn span(&self, origin: SemOrigin<'db>) -> SemanticDiagnosticSpan<'db> {
        SemanticDiagnosticSpan::OriginWithTemplateFallback {
            owner: self.instance.key(self.db).owner(self.db),
            template_owner: self.body.template_owner,
            origin,
        }
    }

    // --- liveness ---------------------------------------------------------

    fn statement_uses(&self, statement: &NStatement<'db>, live: &mut Live) {
        let use_place = |place: &NPlace<'db>, live: &mut Live| {
            for value in self.body.place_values(place) {
                live.values[value.index()] = true;
            }
            for region in self.resolve(place).regions {
                if let Base::Root(root) = region.base {
                    live.roots[root.index()] = true;
                }
            }
        };
        match &statement.kind {
            NStatementKind::Store { destination, value } => {
                if let NPlaceBase::Root(root) = destination.base
                    && destination.path.is_empty()
                {
                    live.roots[root.index()] = false;
                }
                for value in self.body.place_values(destination) {
                    live.values[value.index()] = true;
                }
                live.values[value.value.index()] = true;
            }
            NStatementKind::Define { result, expr } => {
                live.values[result.index()] = false;
                expr.for_each_value_operand(|operand| live.values[operand.value.index()] = true);
                if let NExpr::ProjectValue { path, .. } = expr {
                    for value in path_indices(&path.0) {
                        live.values[value.index()] = true;
                    }
                }
                expr.for_each_place_operand(|place| use_place(place, live));
            }
        }
    }

    fn terminator_uses(&self, block: NBlockId, live: &mut Live) {
        let kind = &self.body.blocks[block.index()].terminator.kind;
        match kind {
            NTerminatorKind::Branch { cond: value, .. }
            | NTerminatorKind::MatchEnum { value, .. }
            | NTerminatorKind::Return(Some(value)) => live.values[value.value.index()] = true,
            NTerminatorKind::Goto(_)
            | NTerminatorKind::Assert { .. }
            | NTerminatorKind::Return(None) => {}
        }
        for successor in kind.successors() {
            for arg in &successor.args {
                live.values[arg.value.index()] = true;
            }
        }
    }

    fn live_in(&self, block: NBlockId, out: &Live) -> Live {
        let mut live = out.clone();
        if !self.diverges(block) {
            self.terminator_uses(block, &mut live);
        }
        let statements: Vec<_> = self.statements(block).collect();
        for (_, statement) in statements.into_iter().rev() {
            self.statement_uses(statement, &mut live);
        }
        for param in &self.body.blocks[block.index()].params {
            live.values[param.index()] = false;
        }
        live
    }

    fn live_out(&self) -> Vec<Live> {
        let empty = Live::new(self.body.values.len(), self.body.roots.len());
        let mut out = vec![empty; self.body.blocks.len()];
        let order = self.reverse_postorder();
        loop {
            let mut changed = false;
            for &block in order.iter().rev() {
                let mut joined = out[block.index()].clone();
                for successor in self.successors(block) {
                    joined.join(&self.live_in(successor, &out[successor.index()]));
                }
                if joined != out[block.index()] {
                    out[block.index()] = joined;
                    changed = true;
                }
            }
            if !changed {
                return out;
            }
        }
    }

    /// Liveness after each statement of `block`.
    fn live_after_statements(&self, block: NBlockId, out: &Live) -> Vec<Live> {
        let mut live = out.clone();
        if !self.diverges(block) {
            self.terminator_uses(block, &mut live);
        }
        let statements: Vec<_> = self.statements(block).collect();
        let mut after = vec![Live::default(); statements.len()];
        for (index, statement) in statements.into_iter().rev() {
            after[index] = live.clone();
            self.statement_uses(statement, &mut live);
        }
        after
    }

    /// Loans held by live values and live roots, and by the contents of
    /// storage those reach.
    fn live_loans(&self, state: &State, live: &Live) -> TokenSet {
        let mut held = TokenSet::new();
        for (value, _) in live.values.iter().enumerate().filter(|(_, live)| **live) {
            union_into(&mut held, &self.values[value]);
        }
        for (root, _) in live.roots.iter().enumerate().filter(|(_, live)| **live) {
            union_into(&mut held, &self.contents(state, root));
        }
        self.loans(state, &held)
    }

    /// The loans among `tokens` and everything they reach.
    fn loans(&self, state: &State, tokens: &[TokenId]) -> TokenSet {
        self.reachable(state, tokens)
            .into_iter()
            .filter(|token| self.tokens[*token as usize].is_loan())
            .collect()
    }
}

/// What the checks of one statement see: the states before and after it, the
/// loans live after it, and the loan it creates.
struct Point<'a> {
    before: &'a State,
    after: &'a State,
    live: &'a TokenSet,
    own: Option<TokenId>,
}

/// An access checked against the live loans it does not hold.
struct Access<'a, 'db> {
    regions: &'a [AbsPlace],
    kind: BorrowKind,
    authority: &'a TokenSet,
    verb: Verb,
    origin: SemOrigin<'db>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Verb {
    Read,
    Borrow,
    MutBorrow,
    Write,
    Move,
    Access,
}

impl Verb {
    fn describe(self) -> &'static str {
        match self {
            Self::Read => "read",
            Self::Borrow => "immutably borrow",
            Self::MutBorrow => "mutably borrow",
            Self::Write => "write to",
            Self::Move => "move out of",
            Self::Access => "access",
        }
    }
}

impl<'db> Analysis<'_, 'db> {
    fn check_statement(
        &mut self,
        statement: &NStatement<'db>,
        point: &Point<'_>,
    ) -> Result<(), Diag<'db>> {
        let verb = match &statement.kind {
            NStatementKind::Store { .. } => Verb::Write,
            NStatementKind::Define { expr, .. } => match expr {
                NExpr::Borrow {
                    kind: BorrowKind::Mut,
                    ..
                } => Verb::MutBorrow,
                NExpr::Borrow { .. } | NExpr::MakeView { .. } => Verb::Borrow,
                NExpr::Load {
                    mode: ReadMode::Move,
                    ..
                } => Verb::Move,
                NExpr::Load { .. } => Verb::Read,
                _ => Verb::Access,
            },
        };
        // Operands are checked in order, so one operation cannot consume the
        // same value twice.
        let mut moved = point.before.moved.clone();
        for access in statement.kind.accesses(self.db, self.body) {
            match access.target {
                AccessTarget::Value { operand, path } => {
                    let consume = access.kind == MemoryAccessKind::Move;
                    self.check_value_use(operand, path, &mut moved, consume, statement.origin)?;
                }
                AccessTarget::Place(place) => {
                    let origin = specific_origin(place.origin, statement.origin);
                    self.check_place_access(place, &access, verb, point, &mut moved, origin)?;
                }
            }
        }
        match &statement.kind {
            NStatementKind::Store { destination, value } => {
                self.check_store(destination, *value, statement.origin)
            }
            NStatementKind::Define {
                expr:
                    NExpr::Borrow {
                        place,
                        kind: BorrowKind::Mut,
                        ..
                    },
                ..
            } => self.check_writable(&self.resolve(place).regions, place.origin),
            NStatementKind::Define {
                expr:
                    NExpr::Call {
                        callee,
                        args,
                        effect_args,
                        ..
                    },
                ..
            } => self.check_call(statement, callee.key, args, effect_args, point),
            NStatementKind::Define { .. } => Ok(()),
        }
    }

    /// Reject a use of a possibly moved value, then record its move.
    fn check_value_use(
        &self,
        operand: NOperand,
        path: Option<&NStructuralPath>,
        moved: &mut Moved,
        consume: bool,
        origin: SemOrigin<'db>,
    ) -> Result<(), Diag<'db>> {
        let path = path.map(|path| self.path(&path.0)).unwrap_or_default();
        let key = MoveKey::Value(operand.value);
        if let Some(found) = moved
            .iter()
            .find(|(moved, moved_path)| *moved == key && paths_overlap(moved_path, &path))
        {
            return Err(self.moved_diag(
                "cannot use a value after it was moved",
                operand_origin(operand, origin),
                found,
            ));
        }
        if consume
            && self.body.values[operand.value.index()]
                .ty
                .as_capability(self.db)
                .is_none()
        {
            moved.insert((key, path));
        }
        Ok(())
    }

    fn moved_diag(
        &self,
        message: &str,
        origin: SemOrigin<'db>,
        moved: &(MoveKey, Path),
    ) -> Diag<'db> {
        let mut diag = self.diag(SemanticDiagnosticKind::MoveConflict, origin, message.into());
        if let Some(moved_at) = self.moved_at.get(moved) {
            diag.push_secondary("value is moved here".into(), self.span(*moved_at));
        }
        diag
    }

    fn check_place_access(
        &self,
        place: &NPlace<'db>,
        access: &OperationAccess<'_, 'db>,
        verb: Verb,
        point: &Point<'_>,
        moved: &mut Moved,
        origin: SemOrigin<'db>,
    ) -> Result<(), Diag<'db>> {
        let resolved = self.resolve(place);
        if let NPlaceBase::Root(root) = place.base
            && self.is_local_root(root.index())
        {
            let key = MoveKey::Root(root);
            let path = self.path(&place.path);
            let found = moved.iter().find(|(moved, moved_path)| {
                *moved == key
                    && if access.kind == MemoryAccessKind::Write {
                        moved_path.len() < path.len() && path.starts_with(moved_path)
                    } else {
                        paths_overlap(moved_path, &path)
                    }
            });
            if let Some(found) = found {
                let message = if access.kind == MemoryAccessKind::Write {
                    "cannot assign to part of a moved value"
                } else if self.moved_at.contains_key(found) {
                    "cannot use a value after it was moved"
                } else {
                    "cannot use a value before it is initialized"
                };
                return Err(self.moved_diag(message, origin, found));
            }
            if access.kind == MemoryAccessKind::Move {
                moved.insert((key, path));
            }
        }
        if access.kind == MemoryAccessKind::Move
            && matches!(place.base, NPlaceBase::CapabilityTarget { .. })
            && resolved
                .regions
                .iter()
                .any(|region| region.base != Base::Raw)
        {
            return Err(self.diag(
                SemanticDiagnosticKind::MoveConflict,
                origin,
                "cannot move out of a view parameter or through a borrow handle".into(),
            ));
        }
        if place.ty.is_zero_sized(self.db) {
            return Ok(());
        }
        self.check_conflicts(
            point,
            &Access {
                regions: &resolved.regions,
                kind: access.conflict_kind(),
                authority: &resolved.authority,
                verb,
                origin,
            },
        )
    }

    /// Reject an access that overlaps a live loan it does not hold.
    fn check_conflicts(
        &self,
        point: &Point<'_>,
        access: &Access<'_, 'db>,
    ) -> Result<(), Diag<'db>> {
        for &loan in point.live {
            if Some(loan) == point.own || access.authority.binary_search(&loan).is_ok() {
                continue;
            }
            let data = &self.tokens[loan as usize];
            let Some(loan_kind) = self.loan_kind(point.after, loan) else {
                continue;
            };
            if data.zero_sized
                || data.view && access.verb == Verb::Write
                || access.kind == BorrowKind::Ref && loan_kind == BorrowKind::Ref
            {
                continue;
            }
            let overlaps = |token: &Token<'db>| {
                access
                    .regions
                    .iter()
                    .any(|region| token.regions.iter().any(|loaned| loaned.overlaps(region)))
            };
            if !overlaps(data) {
                continue;
            }
            // Point at the borrow of the same kind the conflicting loan was
            // derived from, as a reborrow or a call result borrows its sources.
            let mut seen = vec![loan];
            let mut origin = data;
            while let Some(&parent) = origin.parents.iter().find(|&&parent| {
                !seen.contains(&parent)
                    && access.authority.binary_search(&parent).is_err()
                    && self.loan_kind(point.after, parent) == Some(loan_kind)
                    && overlaps(&self.tokens[parent as usize])
            }) {
                seen.push(parent);
                origin = &self.tokens[parent as usize];
            }
            let held = match loan_kind {
                BorrowKind::Mut => "a mutable",
                BorrowKind::Ref => "an immutable",
            };
            let mut diag = self.diag(
                SemanticDiagnosticKind::BorrowConflict,
                access.origin,
                format!(
                    "cannot {} this place while {held} borrow is active",
                    access.verb.describe()
                ),
            );
            diag.push_secondary("borrow created here".into(), self.span(origin.origin));
            return Err(diag);
        }
        Ok(())
    }

    /// The conflict kind of a live loan: a reserved receiver is shared until
    /// its call runs.
    fn loan_kind(&self, state: &State, loan: TokenId) -> Option<BorrowKind> {
        match self.tokens[loan as usize].kind {
            TokenKind::Loan(kind) => Some(kind),
            TokenKind::Reserved(_) if state.reserved.contains(&loan) => Some(BorrowKind::Ref),
            TokenKind::Reserved(_) => Some(BorrowKind::Mut),
            TokenKind::Input { .. } | TokenKind::Handle => None,
        }
    }

    fn is_local_root(&self, root: usize) -> bool {
        !matches!(self.body.roots[root].kind, NRootKind::Provider { .. })
    }

    pub(super) fn space(&self, base: Base) -> Option<ProviderAddressSpace> {
        match base {
            Base::Root(root) => Some(self.body.roots[root.index()].address_space),
            Base::Provider(provider) => {
                let semantics = &self.providers[provider as usize].semantics;
                semantics.address_space.or_else(|| {
                    matches!(semantics.kind, ProviderKind::RootObject)
                        .then_some(ProviderAddressSpace::Memory)
                })
            }
            Base::Param(param) => self.param_spaces.get(&param).copied(),
            Base::Raw => Some(ProviderAddressSpace::Memory),
        }
    }

    fn check_writable(
        &self,
        regions: &[AbsPlace],
        origin: SemOrigin<'db>,
    ) -> Result<(), Diag<'db>> {
        for region in regions {
            if let Some(space @ (ProviderAddressSpace::Calldata | ProviderAddressSpace::Code)) =
                self.space(region.base)
            {
                return Err(self.diag(
                    SemanticDiagnosticKind::StorageViolation,
                    origin,
                    format!("cannot write to {}", space.pretty()),
                ));
            }
        }
        Ok(())
    }

    /// A body loan of storage owned by this body, if any token holds one.
    fn local_loan(&self, tokens: &[TokenId]) -> Option<usize> {
        tokens.iter().find_map(|token| {
            let data = &self.tokens[*token as usize];
            if !data.is_loan() {
                return None;
            }
            data.regions.iter().find_map(|region| match region.base {
                Base::Root(root) if self.is_local_root(root.index()) => Some(root.index()),
                _ => None,
            })
        })
    }

    fn root_name(&self, root: usize) -> Option<String> {
        let body = self.body.template_owner.body(self.db)?;
        let binding = match self.body.roots[root].kind {
            NRootKind::LocalSlot { binding } => binding?,
            NRootKind::ParamPlace { param } => self
                .instance
                .key(self.db)
                .typed_body(self.db)
                .param_binding(param as usize)?,
            NRootKind::Provider { .. }
            | NRootKind::CapabilityRepresentation { .. }
            | NRootKind::Temporary { .. } => return None,
        };
        Some(binding.pretty_name_in_body(self.db, body))
    }

    fn local_storage(&self, root: usize) -> String {
        self.root_name(root)
            .map_or_else(|| "local storage".into(), |name| format!("local `{name}`"))
    }

    /// A call may leave the capabilities it was given in a `mut` referent:
    /// storage cannot hold borrows, and caller-visible places cannot hold
    /// borrows of this body's locals.
    fn check_flow(
        &self,
        destinations: &[AbsPlace],
        ty: TyId<'db>,
        tokens: &[TokenId],
        origin: SemOrigin<'db>,
    ) -> Result<(), Diag<'db>> {
        if tokens
            .iter()
            .any(|token| self.tokens[*token as usize].kind != TokenKind::Handle)
        {
            for region in destinations {
                if let Some(
                    space @ (ProviderAddressSpace::Storage | ProviderAddressSpace::Transient),
                ) = self.space(region.base)
                {
                    return Err(self.diag(
                        SemanticDiagnosticKind::NoEscViolation,
                        origin,
                        format!(
                            "cannot store `{}` in {}",
                            ty.pretty_print(self.db),
                            space.pretty()
                        ),
                    ));
                }
            }
        }
        self.check_retained(destinations, tokens, origin)
    }

    fn check_retained(
        &self,
        destinations: &[AbsPlace],
        tokens: &[TokenId],
        origin: SemOrigin<'db>,
    ) -> Result<(), Diag<'db>> {
        if destinations
            .iter()
            .any(|region| matches!(region.base, Base::Param(_) | Base::Provider(_)))
            && let Some(root) = self.local_loan(tokens)
        {
            return Err(self.diag(
                SemanticDiagnosticKind::InvalidReturnBorrow,
                origin,
                format!(
                    "cannot leave a borrow of {} in caller-accessible storage",
                    self.local_storage(root)
                ),
            ));
        }
        Ok(())
    }

    fn check_store(
        &self,
        destination: &NPlace<'db>,
        value: NOperand,
        origin: SemOrigin<'db>,
    ) -> Result<(), Diag<'db>> {
        let regions = self.resolve(destination).regions;
        self.check_writable(&regions, destination.origin)?;
        let ty = self.body.values[value.value.index()].ty;
        if self.carried(ty).borrowing() {
            for region in &regions {
                if let Some(
                    space @ (ProviderAddressSpace::Storage | ProviderAddressSpace::Transient),
                ) = self.space(region.base)
                {
                    return Err(self.diag(
                        SemanticDiagnosticKind::NoEscViolation,
                        operand_origin(value, origin),
                        format!(
                            "cannot store `{}` in {}",
                            ty.pretty_print(self.db),
                            space.pretty()
                        ),
                    ));
                }
            }
        }
        self.check_retained(&regions, &self.values[value.value.index()], origin)
    }
}

/// One capability a call receives, for the pairwise argument check.
struct CallInput<'db> {
    /// The argument or effect it comes from.
    arg: usize,
    regions: Vec<AbsPlace>,
    mutable: bool,
    origin: SemOrigin<'db>,
}

impl<'db> Analysis<'_, 'db> {
    fn token_regions(
        &self,
        tokens: &[TokenId],
        filter: impl Fn(&Token<'db>) -> bool,
    ) -> Vec<AbsPlace> {
        let mut regions: Vec<AbsPlace> = tokens
            .iter()
            .map(|token| &self.tokens[*token as usize])
            .filter(|data| data.kind != TokenKind::Handle && !data.zero_sized && filter(data))
            .flat_map(|data| data.regions.iter().cloned())
            .collect();
        regions.sort();
        regions.dedup();
        regions
    }

    /// Whether an argument hands its referents to the callee mutably.
    fn arg_is_mutable(&self, value: NValueId) -> bool {
        match self.body.values[value.index()].ty.as_capability(self.db) {
            Some((CapabilityKind::Mut, _)) => true,
            Some((CapabilityKind::Ref | CapabilityKind::View, _)) => false,
            None => self.values[value.index()]
                .iter()
                .any(|token| self.tokens[*token as usize].is_mutable()),
        }
    }

    fn check_call(
        &mut self,
        statement: &NStatement<'db>,
        callee: SemanticInstanceKey<'db>,
        args: &[NOperand],
        effect_args: &[NEffectArg<'db>],
        point: &Point<'_>,
    ) -> Result<(), Diag<'db>> {
        let origin = statement.origin;
        // Each argument reaches its own borrows and the borrows stored in the
        // storage they refer to. Handles confer no exclusivity.
        let is_borrow = |data: &Token<'db>| data.kind != TokenKind::Handle;
        let mut inputs = Vec::new();
        for (position, arg) in args.iter().enumerate() {
            let ty = self.body.values[arg.value.index()].ty;
            if !self.carried(ty).any()
                || ty
                    .as_capability(self.db)
                    .is_some_and(|(_, target)| target.is_zero_sized(self.db))
            {
                continue;
            }
            let direct = &self.values[arg.value.index()];
            let mutable = self.arg_is_mutable(arg.value);
            let arg_origin = operand_origin(*arg, origin);
            inputs.push(CallInput {
                arg: position,
                regions: self.token_regions(direct, is_borrow),
                mutable,
                origin: arg_origin,
            });
            for contained in self.reachable(point.before, direct) {
                let data = &self.tokens[contained as usize];
                if is_borrow(data) && direct.binary_search(&contained).is_err() {
                    inputs.push(CallInput {
                        arg: position,
                        regions: self.token_regions(&[contained], |_| true),
                        mutable: mutable && data.is_mutable(),
                        origin: arg_origin,
                    });
                }
            }
        }
        for (position, effect) in effect_args.iter().enumerate() {
            let regions = match &effect.arg {
                NEffectArgValue::Place(place) if place.ty.is_zero_sized(self.db) => continue,
                NEffectArgValue::Place(place) => self.resolve(place).regions,
                NEffectArgValue::Value(value) => {
                    self.token_regions(&self.values[value.value.index()], |_| true)
                }
            };
            inputs.push(CallInput {
                arg: args.len() + position,
                regions,
                mutable: effect.required_mut,
                origin,
            });
        }
        for (position, input) in inputs.iter().enumerate() {
            for other in &inputs[position + 1..] {
                if input.arg != other.arg
                    && (input.mutable || other.mutable)
                    && input
                        .regions
                        .iter()
                        .any(|region| other.regions.iter().any(|place| place.overlaps(region)))
                {
                    let mut diag = self.diag(
                        SemanticDiagnosticKind::BorrowConflict,
                        other.origin,
                        "cannot pass overlapping places to one call when one of them is mutable"
                            .into(),
                    );
                    diag.push_secondary("overlapping argument".into(), self.span(input.origin));
                    return Err(diag);
                }
            }
        }

        for arg in args {
            let tokens = self.values[arg.value.index()].clone();
            if tokens.is_empty() {
                continue;
            }
            self.check_conflicts(
                point,
                &Access {
                    regions: &self.token_regions(&tokens, |_| true),
                    kind: if self.arg_is_mutable(arg.value) {
                        BorrowKind::Mut
                    } else {
                        BorrowKind::Ref
                    },
                    authority: &self.ancestors(&tokens),
                    verb: Verb::Access,
                    origin: operand_origin(*arg, origin),
                },
            )?;
        }

        let callee_func = match callee.owner(self.db) {
            BodyOwner::Func(func) => Some(func),
            _ => None,
        };
        let receiver = callee_func.is_some_and(|func| func.receiver_ty(self.db).is_some());
        let mut flow = TokenSet::new();
        for arg in args {
            union_into(&mut flow, &self.values[arg.value.index()]);
        }
        for effect in effect_args {
            if let NEffectArgValue::Value(value) = &effect.arg {
                union_into(&mut flow, &self.values[value.value.index()]);
            }
        }
        for (position, arg) in args.iter().enumerate() {
            let ty = self.body.values[arg.value.index()].ty;
            let held = self.reachable(point.before, &self.values[arg.value.index()]);
            let mutable_regions = self.token_regions(&held, |data| data.is_mutable());
            let arg_origin = operand_origin(*arg, origin);
            if position == 0 && receiver {
                self.check_writable(&mutable_regions, arg_origin)?;
                if mutable_regions
                    .iter()
                    .any(|region| self.space(region.base) != Some(ProviderAddressSpace::Memory))
                    && requires_memory_receiver(
                        self.db,
                        get_or_build_semantic_instance(self.db, callee),
                    )
                {
                    self.require_memory(&mutable_regions, ty, arg_origin)?;
                }
            } else {
                self.require_memory(&mutable_regions, ty, arg_origin)?;
            }
            for target in self.flow_targets(point.before, arg.value) {
                let regions = self.token_regions(&[target], |_| true);
                self.check_flow(&regions, ty, &self.incoming(&flow, target), arg_origin)?;
            }
        }
        for effect in effect_args.iter().filter(|effect| effect.required_mut) {
            let place = match &effect.arg {
                NEffectArgValue::Place(place) => place,
                NEffectArgValue::Value(value) => {
                    // A handle supplied as a mutable effect must name writable state.
                    if let Some(
                        space @ (ProviderAddressSpace::Calldata | ProviderAddressSpace::Code),
                    ) = self.handle_space(self.body.values[value.value.index()].ty)
                    {
                        return Err(self.diag(
                            SemanticDiagnosticKind::StorageViolation,
                            operand_origin(*value, origin),
                            format!("cannot write to {}", space.pretty()),
                        ));
                    }
                    continue;
                }
            };
            let regions = self.resolve(place).regions;
            self.check_writable(&regions, place.origin)?;
            if let Some(ty) = effect.provider_target_ty
                && self.carried(ty).any()
            {
                self.check_flow(&regions, ty, &flow, origin)?;
            }
        }
        if let Some(func) = callee_func {
            // Borrows the call receives stay live while it runs.
            let mut live = self.loans(point.before, &flow);
            union_into(&mut live, point.live);
            self.check_state_effects(func, effect_args, point.after, &live, origin)?;
        }
        Ok(())
    }

    /// A mutable borrow passed as an ordinary argument must refer to memory.
    /// A borrow derived from this method's own receiver has an unknown space;
    /// `receiver_reaches_memory_argument` defers that obligation to callers.
    fn require_memory(
        &self,
        regions: &[AbsPlace],
        ty: TyId<'db>,
        origin: SemOrigin<'db>,
    ) -> Result<(), Diag<'db>> {
        for region in regions {
            if let Some(space) = self.space(region.base)
                && space != ProviderAddressSpace::Memory
            {
                return Err(self.diag(
                    SemanticDiagnosticKind::TransportViolation,
                    origin,
                    format!(
                        "cannot pass `{}` from {} as function argument",
                        ty.pretty_print(self.db),
                        space.pretty()
                    ),
                ));
            }
        }
        Ok(())
    }

    pub(super) fn receiver_reaches_memory_argument(&self) -> bool {
        if !self.has_receiver {
            return false;
        }
        let order = self.reverse_postorder();
        order.iter().any(|block| {
            self.statements(*block).any(|(_, statement)| {
                let NStatementKind::Define {
                    expr: NExpr::Call { callee, args, .. },
                    ..
                } = &statement.kind
                else {
                    return false;
                };
                let receiver = matches!(callee.key.owner(self.db),
                        BodyOwner::Func(func) if func.receiver_ty(self.db).is_some());
                args.iter().enumerate().any(|(position, arg)| {
                    let from_self = self
                        .token_regions(&self.values[arg.value.index()], |data| data.is_mutable())
                        .iter()
                        .any(|region| region.base == Base::Param(0));
                    from_self
                        && (position != 0
                            || !receiver
                            || requires_memory_receiver(
                                self.db,
                                get_or_build_semantic_instance(self.db, callee.key),
                            ))
                })
            })
        })
    }

    /// External execution can reenter the contract and access any persistent
    /// or transient slot, so no state borrow may stay live across it. A call
    /// starts one when it is a trusted call intrinsic, a method of a reentrant
    /// std capability, or receives such a capability as a mutable effect.
    fn check_state_effects(
        &self,
        func: Func<'db>,
        effect_args: &[NEffectArg<'db>],
        state: &State,
        live: &TokenSet,
        origin: SemOrigin<'db>,
    ) -> Result<(), Diag<'db>> {
        let Some(kind) = external_call_state_access(self.db, func).or_else(|| {
            effect_args
                .iter()
                .any(|effect| {
                    effect_param_starts_external_calls(self.db, func, effect.binding_idx as usize)
                })
                .then_some(MemoryAccessKind::Write)
        }) else {
            return Ok(());
        };
        let accesses = [
            ProviderAddressSpace::Storage,
            ProviderAddressSpace::Transient,
        ]
        .map(|space| (space, kind));
        for (space, kind) in accesses {
            let kind = kind.borrow_kind();
            for &loan in live {
                let data = &self.tokens[loan as usize];
                let Some(loan_kind) = self.loan_kind(state, loan) else {
                    continue;
                };
                if !data.zero_sized
                    && (kind == BorrowKind::Mut || loan_kind == BorrowKind::Mut)
                    && data
                        .regions
                        .iter()
                        .any(|region| self.space(region.base) == Some(space))
                {
                    let mut diag = self.diag(
                        SemanticDiagnosticKind::BorrowConflict,
                        origin,
                        format!(
                            "this call may {} {} while a borrow of it is active",
                            if kind == BorrowKind::Mut {
                                "write"
                            } else {
                                "read"
                            },
                            space.pretty()
                        ),
                    );
                    diag.push_secondary("borrow created here".into(), self.span(data.origin));
                    return Err(diag);
                }
            }
        }
        Ok(())
    }

    fn check_terminator(&self, block: NBlockId, state: &State) -> Result<(), Diag<'db>> {
        let terminator = &self.body.blocks[block.index()].terminator;
        let mut moved = state.moved.clone();
        if let Some(access) = terminator.kind.access(self.db, self.body)
            && let AccessTarget::Value { operand, path } = access.target
        {
            let consume = access.kind == MemoryAccessKind::Move;
            self.check_value_use(operand, path, &mut moved, consume, terminator.origin)?;
        }
        for successor in terminator.kind.successors() {
            let mut edge = moved.clone();
            for arg in &successor.args {
                let consume = arg.mode == ReadMode::Move;
                self.check_value_use(*arg, None, &mut edge, consume, terminator.origin)?;
            }
        }
        if let NTerminatorKind::Return(Some(value)) = &terminator.kind {
            self.check_return(*value, operand_origin(*value, terminator.origin))?;
        }
        Ok(())
    }

    fn check_return(&self, value: NOperand, origin: SemOrigin<'db>) -> Result<(), Diag<'db>> {
        let ty = self.body.values[value.value.index()].ty;
        if !self.carried(ty).borrowing() {
            return Ok(());
        }
        let direct = ty.as_capability(self.db).is_some();
        let reject = |message: String| {
            Err(self.diag(SemanticDiagnosticKind::InvalidReturnBorrow, origin, message))
        };
        for &token in &self.values[value.value.index()] {
            let data = &self.tokens[token as usize];
            match data.kind {
                TokenKind::Handle => continue,
                TokenKind::Input { effect: true, .. } => {
                    return reject(
                        "cannot return a borrow derived from an effect parameter".into(),
                    );
                }
                TokenKind::Input { .. } | TokenKind::Loan(_) | TokenKind::Reserved(_) => {}
            }
            for region in &data.regions {
                match region.base {
                    // A view borrows memory, which outlives the local owner
                    // it was derived from once the body returns.
                    Base::Root(root) if self.is_local_root(root.index()) && data.view => {}
                    Base::Root(root) if self.is_local_root(root.index()) => {
                        let local = self.local_storage(root.index());
                        return reject(if direct {
                            format!("cannot return a borrow to {local}")
                        } else {
                            format!("cannot return a value that holds a borrow of {local}")
                        });
                    }
                    Base::Root(_) | Base::Provider(_) => {
                        return reject(
                            "cannot return a borrow derived from an effect parameter".into(),
                        );
                    }
                    Base::Param(param) if param >= self.param_count => {
                        return reject(
                            "cannot return a borrow derived from an effect parameter".into(),
                        );
                    }
                    Base::Param(param) if param != 0 && self.self_carries => {
                        return reject(
                            "a method can only return borrows derived from `self`".into(),
                        );
                    }
                    Base::Param(_) | Base::Raw => {}
                }
            }
        }
        Ok(())
    }
}
