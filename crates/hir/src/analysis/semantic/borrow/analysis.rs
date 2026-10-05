//! Borrow analysis of one normalized body.
//!
//! Every value and root carries a set of tokens: loans created in this body,
//! caller-supplied inputs, and provider handles. A place reached through a
//! capability carrier resolves to the regions of the carrier's tokens, so
//! copies of one carrier share its loans. Calls are checked against the
//! callee's signature only; raw memory is unchecked.
use std::collections::BTreeSet;

use cranelift_entity::EntityRef;
use rustc_hash::FxHashMap;
use smallvec::SmallVec;

use super::{
    carried::{Carried, carried_capabilities},
    place::{AbsPlace, Base, Path, Step, path_of},
};
use crate::{
    analysis::{
        HirAnalysisDb,
        semantic::{
            BorrowActivation, CallSiteId, SemOrigin, SemanticInstance, SemanticInstanceKey,
            borrow::control::semantic_may_return,
            capability::semantics::{CapabilityClass, capability_semantics},
            definite_assignment::literal_index,
            diagnostics::operand_origin,
            get_or_build_semantic_instance,
            normalized::{
                HandleOrigin, NBlockId, NDataPath, NEffectArgValue, NExpr, NOperand, NPlace,
                NPlaceBase, NRootId, NRootKind, NStatement, NStatementKind, NValueDefinition,
                NValueId, NormalizedBody, ReadMode, access::AccessTarget,
            },
        },
        ty::{
            corelib::MemoryAccessKind,
            provider::{ProviderAddressSpace, provider_semantics},
            trait_resolution::PredicateListId,
            ty_check::BodyOwner,
            ty_def::{BorrowKind, CapabilityKind, TyId},
        },
    },
    hir_def::scope_graph::ScopeId,
    semantic::ProviderBinding,
};

pub(super) type TokenId = u32;
pub(super) type TokenSet = SmallVec<TokenId, 4>;

pub(super) fn union_into(target: &mut TokenSet, source: &[TokenId]) -> bool {
    let mut changed = false;
    for &token in source {
        if let Err(index) = target.binary_search(&token) {
            target.insert(index, token);
            changed = true;
        }
    }
    changed
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum TokenKind {
    Loan(BorrowKind),
    /// A two-phase receiver borrow: shared until the call it reserves.
    Reserved(CallSiteId),
    /// Caller-supplied capabilities of an entry value or effect provider.
    Input {
        mutable: bool,
        effect: bool,
    },
    /// A provider handle: names a region but confers no exclusivity.
    Handle,
}

#[derive(Clone, Debug)]
pub(super) struct Token<'db> {
    pub kind: TokenKind,
    pub regions: BTreeSet<AbsPlace>,
    /// Tokens of the carrier a reborrow went through. Accessing through a
    /// reborrow is authorized by its ancestors as well.
    pub parents: TokenSet,
    pub origin: SemOrigin<'db>,
    /// A loan of a zero-sized place protects no data and never conflicts.
    pub zero_sized: bool,
}

impl Token<'_> {
    pub fn is_loan(&self) -> bool {
        matches!(self.kind, TokenKind::Loan(_) | TokenKind::Reserved(_))
    }

    pub fn is_mutable(&self) -> bool {
        matches!(
            self.kind,
            TokenKind::Loan(BorrowKind::Mut)
                | TokenKind::Reserved(_)
                | TokenKind::Input { mutable: true, .. }
        )
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(super) enum MoveKey {
    Value(NValueId),
    Root(u32),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct State {
    /// Tokens held by each root's contents.
    pub contents: Vec<TokenSet>,
    /// Possibly moved (or uninitialized) values and root paths.
    pub moved: BTreeSet<(MoveKey, Path)>,
    /// Two-phase borrows whose call has not run yet.
    pub reserved: BTreeSet<TokenId>,
}

impl State {
    fn join(&mut self, other: &Self) -> bool {
        let mut changed = false;
        for (target, source) in self.contents.iter_mut().zip(&other.contents) {
            changed |= union_into(target, source);
        }
        for moved in &other.moved {
            changed |= self.moved.insert(moved.clone());
        }
        for token in &other.reserved {
            changed |= self.reserved.insert(*token);
        }
        changed
    }
}

/// Where an access goes and which tokens authorize it.
pub(super) struct Resolved {
    pub regions: Vec<AbsPlace>,
    pub authority: TokenSet,
}

pub(super) struct Analysis<'a, 'db> {
    pub db: &'db dyn HirAnalysisDb,
    pub instance: SemanticInstance<'db>,
    pub body: &'a NormalizedBody<'db>,
    scope: ScopeId<'db>,
    assumptions: PredicateListId<'db>,
    pub param_count: u32,
    /// Whether entry value 0 is a capability-carrying `self` receiver.
    pub self_carries: bool,
    pub has_receiver: bool,
    /// The address space of each caller-supplied entry value's referents, when
    /// its type fixes it.
    pub param_spaces: FxHashMap<u32, ProviderAddressSpace>,
    pub providers: Vec<ProviderBinding<'db>>,
    root_provider: Vec<Option<u32>>,
    pub tokens: Vec<Token<'db>>,
    pub values: Vec<TokenSet>,
    input_tokens: FxHashMap<Base, TokenId>,
    pub statement_tokens: FxHashMap<(NBlockId, usize), TokenId>,
    /// Handles standing for effect places passed to a call.
    effect_tokens: FxHashMap<(NBlockId, usize, usize), TokenId>,
    pub entry: Vec<Option<State>>,
    pub moved_at: FxHashMap<(MoveKey, Path), SemOrigin<'db>>,
    /// The statement at which each block diverges into a call that never returns.
    divergence: Vec<Option<usize>>,
}

impl<'a, 'db> Analysis<'a, 'db> {
    /// `divergence` cuts each block at a call that never returns. Provisional
    /// analyses run while callee instances are still being built, so they
    /// keep every path.
    pub fn new(
        db: &'db dyn HirAnalysisDb,
        instance: SemanticInstance<'db>,
        body: &'a NormalizedBody<'db>,
        divergence: bool,
    ) -> Self {
        let owner = instance.key(db).owner(db);
        let typed_body = instance.key(db).typed_body(db);
        let param_count = (0..)
            .take_while(|index| typed_body.param_binding(*index).is_some())
            .count() as u32;
        let has_receiver = matches!(owner, BodyOwner::Func(func) if func.receiver_ty(db).is_some());
        let mut analysis = Self {
            db,
            instance,
            body,
            scope: instance.key(db).impl_env(db).normalization_scope(db),
            assumptions: instance.assumptions(db),
            param_count,
            self_carries: false,
            has_receiver,
            param_spaces: FxHashMap::default(),
            providers: Vec::new(),
            root_provider: vec![None; body.roots.len()],
            tokens: Vec::new(),
            values: vec![TokenSet::new(); body.values.len()],
            input_tokens: FxHashMap::default(),
            statement_tokens: FxHashMap::default(),
            effect_tokens: FxHashMap::default(),
            entry: vec![None; body.blocks.len()],
            moved_at: FxHashMap::default(),
            divergence: vec![None; body.blocks.len()],
        };
        if divergence {
            for (block, data) in body.blocks.iter().enumerate() {
                analysis.divergence[block] =
                    data.statements
                        .iter()
                        .position(|statement| match &statement.kind {
                            NStatementKind::Define {
                                expr: NExpr::Call { callee, .. },
                                ..
                            } => !semantic_may_return(
                                db,
                                get_or_build_semantic_instance(db, callee.key),
                            ),
                            _ => false,
                        });
            }
        }
        for (index, root) in body.roots.iter().enumerate() {
            if let NRootKind::Provider { binding } = &root.kind {
                analysis.root_provider[index] = Some(analysis.provider_index(binding));
            }
        }
        analysis.seed_entry();
        analysis
    }

    /// The address space a provider handle type declares.
    pub fn handle_space(&self, ty: TyId<'db>) -> Option<ProviderAddressSpace> {
        matches!(
            capability_semantics(self.db, self.scope, self.assumptions, ty),
            Ok(Some(semantics)) if semantics.class == CapabilityClass::Handle
        )
        .then(|| provider_semantics(self.db, self.scope, self.assumptions, ty).address_space)
        .flatten()
    }

    pub fn carried(&self, ty: TyId<'db>) -> Carried {
        carried_capabilities(self.db, self.scope, self.assumptions, ty)
    }

    /// Providers are identified by their source: two bindings of one contract
    /// field or effect parameter name the same storage.
    fn provider_index(&mut self, binding: &ProviderBinding<'db>) -> u32 {
        if let Some(index) = self
            .providers
            .iter()
            .position(|known| known.source == binding.source)
        {
            return index as u32;
        }
        self.providers.push(binding.clone());
        (self.providers.len() - 1) as u32
    }

    fn new_token(&mut self, kind: TokenKind, origin: SemOrigin<'db>) -> TokenId {
        self.tokens.push(Token {
            kind,
            regions: BTreeSet::new(),
            parents: TokenSet::new(),
            origin,
            zero_sized: false,
        });
        (self.tokens.len() - 1) as TokenId
    }

    /// The token for everything a caller supplied through `base`.
    fn input_token(&mut self, base: Base, mutable: bool, effect: bool) -> TokenId {
        if let Some(token) = self.input_tokens.get(&base) {
            return *token;
        }
        let token = self.new_token(
            TokenKind::Input { mutable, effect },
            SemOrigin::Body(self.body.template_owner),
        );
        self.tokens[token as usize]
            .regions
            .insert(AbsPlace::new(base));
        self.input_tokens.insert(base, token);
        token
    }

    fn root_base(&self, root: usize) -> Base {
        self.root_provider[root].map_or(Base::Root(NRootId::new(root)), Base::Provider)
    }

    fn seed_entry(&mut self) {
        let body = self.body;
        for (index, value) in body.values.iter().enumerate() {
            let NValueDefinition::EntryParam { param } = value.definition else {
                continue;
            };
            let carried = self.carried(value.ty);
            if !carried.any() {
                continue;
            }
            let effect = param >= self.param_count;
            // Ordinary `mut` arguments refer to memory; handles declare their space.
            let receiver = param == 0 && self.has_receiver;
            let space = match capability_semantics(self.db, self.scope, self.assumptions, value.ty)
            {
                Ok(Some(semantics))
                    if semantics.class == CapabilityClass::Borrow(BorrowKind::Mut) && !receiver =>
                {
                    Some(ProviderAddressSpace::Memory)
                }
                _ => self.handle_space(value.ty),
            };
            if let Some(space) = space {
                self.param_spaces.insert(param, space);
            }
            let token = self.input_token(Base::Param(param), carried.mut_borrows, effect);
            self.values[index] = TokenSet::from_iter([token]);
            if param == 0 && self.has_receiver {
                self.self_carries = true;
            }
        }
        let mut state = State {
            contents: vec![TokenSet::new(); body.roots.len()],
            moved: BTreeSet::new(),
            reserved: BTreeSet::new(),
        };
        // Entry bindings, such as recv arm fields, initialize their slots. The
        // provisional body has the same entry locals and does not depend on
        // the call-site refinements this analysis may be computing.
        let raw = self.instance.provisional_body(self.db);
        let entry_bindings: Vec<_> = raw
            .entry_locals
            .iter()
            .filter_map(|local| raw.locals[local.index()].source)
            .collect();
        for (index, root) in body.roots.iter().enumerate() {
            match &root.kind {
                NRootKind::LocalSlot { binding } => {
                    if !binding.is_some_and(|binding| entry_bindings.contains(&binding)) {
                        state
                            .moved
                            .insert((MoveKey::Root(index as u32), Path::new()));
                    }
                }
                NRootKind::ParamPlace { param } => {
                    let carried = self.carried(root.ty);
                    if carried.any() {
                        let token =
                            self.input_token(Base::Param(*param), carried.mut_borrows, false);
                        state.contents[index] = TokenSet::from_iter([token]);
                    }
                    if *param == 0 && self.has_receiver && carried.any() {
                        self.self_carries = true;
                    }
                }
                NRootKind::Provider { .. } => {
                    if self.carried(root.ty).any() {
                        let base = self.root_base(index);
                        let token = self.input_token(base, true, true);
                        state.contents[index] = TokenSet::from_iter([token]);
                    }
                }
                NRootKind::CapabilityRepresentation { .. } | NRootKind::Temporary { .. } => {}
            }
        }
        self.entry[body.entry.index()] = Some(state);
    }

    /// Tokens held by a root's contents, including the value an implicitly
    /// initialized root was created from.
    pub fn contents(&self, state: &State, root: usize) -> TokenSet {
        let mut tokens = state.contents[root].clone();
        if let NRootKind::CapabilityRepresentation { carrier: value }
        | NRootKind::Temporary { value } = self.body.roots[root].kind
        {
            union_into(&mut tokens, &self.values[value.index()]);
        }
        tokens
    }

    /// The steps of a normalized path, resolving literal index values.
    pub fn path(&self, path: &NDataPath) -> Path {
        path_of(path, |value| literal_index(self.db, self.body, value))
    }

    pub fn ancestors(&self, tokens: &[TokenId]) -> TokenSet {
        let mut closure = tokens.iter().copied().collect::<TokenSet>();
        let mut index = 0;
        while index < closure.len() {
            let parents = self.tokens[closure[index] as usize].parents.clone();
            union_into(&mut closure, &parents);
            index += 1;
        }
        closure
    }

    pub fn resolve(&self, place: &NPlace<'db>) -> Resolved {
        let path = self.path(&place.path);
        match place.base {
            NPlaceBase::Root(root) => Resolved {
                regions: vec![AbsPlace {
                    base: self.root_base(root.index()),
                    path,
                }],
                authority: TokenSet::new(),
            },
            NPlaceBase::CapabilityTarget { carrier } => {
                let tokens = &self.values[carrier.index()];
                let mut regions: Vec<AbsPlace> = tokens
                    .iter()
                    .flat_map(|token| &self.tokens[*token as usize].regions)
                    .map(|region| region.extended(&path))
                    .collect();
                if self.body.values[carrier.index()]
                    .ty
                    .as_ptr(self.db)
                    .is_some()
                    || regions.is_empty()
                {
                    regions = vec![AbsPlace {
                        base: Base::Raw,
                        path,
                    }];
                }
                regions.sort();
                regions.dedup();
                Resolved {
                    regions,
                    authority: self.ancestors(tokens),
                }
            }
        }
    }

    /// Tokens a load from `regions` yields.
    fn load_tokens(&mut self, state: &State, regions: &[AbsPlace]) -> TokenSet {
        let mut tokens = TokenSet::new();
        for region in regions {
            match region.base {
                Base::Root(root) => union_into(&mut tokens, &self.contents(state, root.index())),
                Base::Provider(_) | Base::Param(_) => {
                    let effect =
                        !matches!(region.base, Base::Param(param) if param < self.param_count);
                    let token = self.input_token(region.base, true, effect);
                    union_into(&mut tokens, &[token])
                }
                Base::Raw => false,
            };
        }
        tokens
    }

    fn set_value(&mut self, value: NValueId, tokens: &[TokenId], changed: &mut bool) {
        *changed |= union_into(&mut self.values[value.index()], tokens);
    }

    fn operand_tokens(&self, operands: impl IntoIterator<Item = NOperand>) -> TokenSet {
        let mut tokens = TokenSet::new();
        for operand in operands {
            union_into(&mut tokens, &self.values[operand.value.index()]);
        }
        tokens
    }

    /// Run the forward fixed point over root contents, moves and value tokens.
    pub fn solve(&mut self) {
        let order = self.reverse_postorder();
        loop {
            let mut changed = false;
            for &block in &order {
                let Some(mut state) = self.block_entry(block) else {
                    continue;
                };
                for (index, statement) in self.statements(block) {
                    self.transfer(block, index, statement, &mut state, &mut changed);
                }
                if self.divergence[block.index()].is_none() {
                    self.transfer_terminator(block, &mut state, &mut changed);
                }
            }
            if !changed {
                break;
            }
        }
    }

    /// The state on entry to `block`, after its parameters are (re)defined.
    pub fn block_entry(&self, block: NBlockId) -> Option<State> {
        let mut state = self.entry[block.index()].clone()?;
        for param in &self.body.blocks[block.index()].params {
            state
                .moved
                .retain(|(key, _)| *key != MoveKey::Value(*param));
        }
        Some(state)
    }

    /// The statements of `block` that can execute.
    pub fn statements(
        &self,
        block: NBlockId,
    ) -> impl Iterator<Item = (usize, &'a NStatement<'db>)> + use<'a, 'db> {
        let statements = &self.body.blocks[block.index()].statements;
        let end = self.divergence[block.index()].map_or(statements.len(), |end| end + 1);
        statements[..end].iter().enumerate()
    }

    /// The successors of `block`, unless it diverges.
    pub fn successors(&self, block: NBlockId) -> Vec<NBlockId> {
        if self.divergence[block.index()].is_some() {
            return Vec::new();
        }
        self.body.blocks[block.index()]
            .terminator
            .kind
            .successors()
            .into_iter()
            .map(|successor| successor.block)
            .collect()
    }

    pub fn diverges(&self, block: NBlockId) -> bool {
        self.divergence[block.index()].is_some()
    }

    pub fn reverse_postorder(&self) -> Vec<NBlockId> {
        let mut visited = vec![false; self.body.blocks.len()];
        let mut postorder = Vec::new();
        let mut stack = vec![(self.body.entry, 0)];
        visited[self.body.entry.index()] = true;
        while let Some((block, next)) = stack.last_mut() {
            let successors = self.successors(*block);
            if let Some(&successor) = successors.get(*next) {
                *next += 1;
                if !visited[successor.index()] {
                    visited[successor.index()] = true;
                    stack.push((successor, 0));
                }
            } else {
                postorder.push(*block);
                stack.pop();
            }
        }
        postorder.reverse();
        postorder
    }

    fn transfer_terminator(&mut self, block: NBlockId, state: &mut State, changed: &mut bool) {
        let terminator = &self.body.blocks[block.index()].terminator;
        if let Some(access) = terminator.kind.access(self.db, self.body) {
            self.transfer_value_access(access.target, access.kind, terminator.origin, state);
        }
        for successor in terminator.kind.successors() {
            let target = &self.body.blocks[successor.block.index()];
            for (param, arg) in target.params.iter().zip(&successor.args) {
                let tokens = self.values[arg.value.index()].clone();
                self.set_value(*param, &tokens, changed);
            }
            let mut successor_state = state.clone();
            for arg in &successor.args {
                self.transfer_value_access(
                    AccessTarget::Value {
                        operand: *arg,
                        path: None,
                    },
                    if arg.mode == ReadMode::Move {
                        MemoryAccessKind::Move
                    } else {
                        MemoryAccessKind::Read
                    },
                    terminator.origin,
                    &mut successor_state,
                );
            }
            match &mut self.entry[successor.block.index()] {
                Some(existing) => *changed |= existing.join(&successor_state),
                slot @ None => {
                    *slot = Some(successor_state);
                    *changed = true;
                }
            }
        }
    }

    fn transfer_value_access(
        &mut self,
        target: AccessTarget<'_, 'db>,
        kind: MemoryAccessKind,
        origin: SemOrigin<'db>,
        state: &mut State,
    ) {
        if kind != MemoryAccessKind::Move {
            return;
        }
        let AccessTarget::Value { operand, path } = target else {
            return;
        };
        if self.body.values[operand.value.index()]
            .ty
            .as_capability(self.db)
            .is_some()
        {
            return;
        }
        let path = path.map(|path| self.path(&path.0)).unwrap_or_default();
        let key = (MoveKey::Value(operand.value), path);
        self.moved_at
            .entry(key.clone())
            .or_insert(operand_origin(operand, origin));
        state.moved.insert(key);
    }

    pub fn transfer(
        &mut self,
        block: NBlockId,
        index: usize,
        statement: &NStatement<'db>,
        state: &mut State,
        changed: &mut bool,
    ) {
        let db = self.db;
        let body = self.body;
        for access in statement.kind.accesses(db, body) {
            match access.target {
                AccessTarget::Value { .. } => {
                    self.transfer_value_access(access.target, access.kind, statement.origin, state)
                }
                AccessTarget::Place(place) => {
                    if access.kind == MemoryAccessKind::Move
                        && let NPlaceBase::Root(root) = place.base
                        && self.root_provider[root.index()].is_none()
                    {
                        let key = (MoveKey::Root(root.index() as u32), self.path(&place.path));
                        self.moved_at.entry(key.clone()).or_insert(place.origin);
                        state.moved.insert(key);
                    }
                }
            }
        }
        match &statement.kind {
            NStatementKind::Store { destination, value } => {
                let tokens = self.values[value.value.index()].clone();
                self.store(destination, &tokens, state);
            }
            NStatementKind::Define { result, expr } => {
                state
                    .moved
                    .retain(|(key, _)| *key != MoveKey::Value(*result));
                let tokens = self.define(block, index, statement, *result, expr, state, changed);
                if self.carried(body.values[result.index()].ty).any() {
                    self.set_value(*result, &tokens, changed);
                }
            }
        }
    }

    fn store(&mut self, destination: &NPlace<'db>, tokens: &[TokenId], state: &mut State) {
        let resolved = self.resolve(destination);
        let whole_root = match destination.base {
            NPlaceBase::Root(root)
                if destination.path.is_empty() && self.root_provider[root.index()].is_none() =>
            {
                Some(root.index())
            }
            _ => None,
        };
        if let Some(root) = whole_root {
            state.contents[root] = tokens.iter().copied().collect::<TokenSet>();
        } else {
            for region in &resolved.regions {
                if let Base::Root(root) = region.base {
                    union_into(&mut state.contents[root.index()], tokens);
                }
            }
        }
        let path = self.path(&destination.path);
        if let NPlaceBase::Root(root) = destination.base
            && !path.contains(&Step::Index(None))
        {
            let key = MoveKey::Root(root.index() as u32);
            state
                .moved
                .retain(|(moved, moved_path)| *moved != key || !moved_path.starts_with(&path));
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn define(
        &mut self,
        block: NBlockId,
        index: usize,
        statement: &NStatement<'db>,
        result: NValueId,
        expr: &NExpr<'db>,
        state: &mut State,
        changed: &mut bool,
    ) -> TokenSet {
        match expr {
            NExpr::Forward { src }
            | NExpr::ProjectValue { value: src, .. }
            | NExpr::StructuralRepack { value: src, .. }
            | NExpr::ArrayRepeat { value: src, .. } => self.values[src.value.index()].clone(),
            NExpr::AggregateMake { fields, .. } | NExpr::EnumMake { fields, .. } => {
                self.operand_tokens(fields.iter().copied())
            }
            NExpr::Load { place, .. } => {
                if !self.carried(self.body.values[result.index()].ty).any() {
                    return TokenSet::new();
                }
                let resolved = self.resolve(place);
                self.load_tokens(state, &resolved.regions)
            }
            NExpr::Borrow {
                place,
                kind,
                activation,
                ..
            } => {
                let kind = match (kind, activation) {
                    (BorrowKind::Mut, BorrowActivation::AtCall { call_site, .. }) => {
                        TokenKind::Reserved(*call_site)
                    }
                    (kind, _) => TokenKind::Loan(*kind),
                };
                let tokens = self.loan(block, index, kind, place, statement.origin, changed);
                if matches!(kind, TokenKind::Reserved(_)) {
                    state.reserved.extend(tokens.iter().copied());
                }
                tokens
            }
            NExpr::MakeView { place, .. } => self.loan(
                block,
                index,
                TokenKind::Loan(BorrowKind::Ref),
                place,
                statement.origin,
                changed,
            ),
            NExpr::MakeHandle { fields, origin, .. } => {
                let base = match origin {
                    HandleOrigin::Provider(binding) => Base::Provider(self.provider_index(binding)),
                    HandleOrigin::Opaque(_) => Base::Raw,
                };
                let token = *self
                    .statement_tokens
                    .entry((block, index))
                    .or_insert_with(|| {
                        self.tokens.push(Token {
                            kind: TokenKind::Handle,
                            regions: BTreeSet::from([AbsPlace::new(base)]),
                            parents: TokenSet::new(),
                            origin: statement.origin,
                            zero_sized: false,
                        });
                        (self.tokens.len() - 1) as TokenId
                    });
                let mut tokens = self.operand_tokens(fields.iter().copied());
                union_into(&mut tokens, &[token]);
                tokens
            }
            NExpr::Call {
                call_site,
                callee,
                args,
                effect_args,
            } => {
                state.reserved.retain(|token| {
                    !matches!(self.tokens[*token as usize].kind, TokenKind::Reserved(site) if site == *call_site)
                });
                let mut flow = TokenSet::new();
                for arg in args.iter() {
                    union_into(&mut flow, &self.values[arg.value.index()]);
                }
                let mut handles = TokenSet::new();
                for (position, effect) in effect_args.iter().enumerate() {
                    match &effect.arg {
                        NEffectArgValue::Value(value) => {
                            union_into(&mut flow, &self.values[value.value.index()]);
                        }
                        NEffectArgValue::Place(place) => {
                            let regions = self.resolve(place).regions;
                            let token = *self
                                .effect_tokens
                                .entry((block, index, position))
                                .or_insert_with(|| {
                                    self.tokens.push(Token {
                                        kind: TokenKind::Handle,
                                        regions: BTreeSet::new(),
                                        parents: TokenSet::new(),
                                        origin: statement.origin,
                                        zero_sized: false,
                                    });
                                    (self.tokens.len() - 1) as TokenId
                                });
                            for region in regions {
                                *changed |= self.tokens[token as usize].regions.insert(region);
                            }
                            union_into(&mut handles, &[token]);
                        }
                    }
                }
                // Each `mut` referent that can hold capabilities may receive
                // every other capability the call was given.
                for arg in args.iter() {
                    if let Some((kind, target)) = self.body.values[arg.value.index()]
                        .ty
                        .as_capability(self.db)
                        && kind == CapabilityKind::Mut
                        && self.carried(target).any()
                    {
                        let own = &self.values[arg.value.index()];
                        let incoming: TokenSet = flow
                            .iter()
                            .copied()
                            .filter(|token| own.binary_search(token).is_err())
                            .collect();
                        let regions = own
                            .iter()
                            .flat_map(|token| self.tokens[*token as usize].regions.clone())
                            .collect::<Vec<_>>();
                        for region in regions {
                            if let Base::Root(root) = region.base {
                                union_into(&mut state.contents[root.index()], &incoming);
                            }
                        }
                    }
                }
                for effect in effect_args.iter() {
                    if effect.required_mut
                        && effect
                            .provider_target_ty
                            .is_some_and(|ty| self.carried(ty).any())
                        && let NEffectArgValue::Place(place) = &effect.arg
                    {
                        for region in self.resolve(place).regions {
                            if let Base::Root(root) = region.base {
                                union_into(&mut state.contents[root.index()], &flow);
                            }
                        }
                    }
                }
                let mut tokens = if self.callee_has_carrying_self(callee.key, args) {
                    self.values[args[0].value.index()].clone()
                } else {
                    flow.clone()
                };
                union_into(&mut tokens, &handles);
                let flow_handles: TokenSet = flow
                    .iter()
                    .copied()
                    .filter(|token| self.tokens[*token as usize].kind == TokenKind::Handle)
                    .collect();
                union_into(&mut tokens, &flow_handles);
                tokens
            }
            NExpr::CodeRegionRef { .. }
            | NExpr::Const(_)
            | NExpr::Unary { .. }
            | NExpr::Binary { .. }
            | NExpr::PointerCast { .. }
            | NExpr::ScalarCast { .. }
            | NExpr::GetEnumTag { .. }
            | NExpr::IsEnumVariant { .. }
            | NExpr::CodeRegionOffset { .. }
            | NExpr::CodeRegionLen { .. } => TokenSet::new(),
        }
    }

    pub fn callee_has_carrying_self(
        &self,
        callee: SemanticInstanceKey<'db>,
        args: &[NOperand],
    ) -> bool {
        let BodyOwner::Func(func) = callee.owner(self.db) else {
            return false;
        };
        func.receiver_ty(self.db).is_some()
            && args
                .first()
                .is_some_and(|arg| self.carried(self.body.values[arg.value.index()].ty).any())
    }

    fn loan(
        &mut self,
        block: NBlockId,
        index: usize,
        kind: TokenKind,
        place: &NPlace<'db>,
        origin: SemOrigin<'db>,
        changed: &mut bool,
    ) -> TokenSet {
        let token = *self
            .statement_tokens
            .entry((block, index))
            .or_insert_with(|| {
                self.tokens.push(Token {
                    kind,
                    regions: BTreeSet::new(),
                    parents: TokenSet::new(),
                    origin,
                    zero_sized: place.ty.is_zero_sized(self.db),
                });
                (self.tokens.len() - 1) as TokenId
            });
        let resolved = self.resolve(place);
        let token_data = &mut self.tokens[token as usize];
        for region in resolved.regions {
            *changed |= token_data.regions.insert(region);
        }
        if let NPlaceBase::CapabilityTarget { carrier } = place.base {
            let parents = self.values[carrier.index()].clone();
            *changed |= union_into(&mut self.tokens[token as usize].parents, &parents);
        }
        TokenSet::from_iter([token])
    }
}
