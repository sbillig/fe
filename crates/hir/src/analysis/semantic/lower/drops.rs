//! Lexical destruction: each scope destroys the owners it holds where control
//! leaves it.
//!
//! A binding belongs to the block or arm that binds it, and a temporary to
//! the scope it is allocated in: the statement, the block tail, a condition
//! or an arm. A scope ends where its code falls through, or where `break`,
//! `continue`, `return`, `?` or a failed condition leaves it; the value a
//! scope ends with passes to the enclosing scope. A `Drop` is placed for
//! every owner at each such exit, and drop elaboration decides what of the
//! owner is still initialized there.
use cranelift_entity::EntityRef;

use super::body::SmirLowerCtxt;
use crate::analysis::{
    semantic::{
        SBlockId, SExpr, SLocalId, SPlace, SStmtKind, STerminatorKind, SemanticLocalRole,
        ValueProvenance,
    },
    ty::{drop_glue::ty_has_drop_glue, ty_check::LocalBinding, ty_def::TyId, ty_is_copy},
};
use crate::projection::Projection;

pub(super) struct DropScope {
    id: u32,
    /// Whether bindings bound in the scope are its own, as a block's or an
    /// arm's are, or belong to an enclosing one, as a statement's do.
    bindings: bool,
    owners: Vec<SLocalId>,
}

impl<'db> SmirLowerCtxt<'_, 'db> {
    pub(super) fn push_drop_scope(&mut self, bindings: bool) {
        self.drop_scopes.push(DropScope {
            id: self.next_drop_scope,
            bindings,
            owners: Vec::new(),
        });
        self.next_drop_scope += 1;
    }

    /// Ends the innermost scope; a path falling through it destroys its
    /// owners. `value`, what the scope evaluates to, passes to the enclosing
    /// scope instead.
    pub(super) fn pop_drop_scope(&mut self, value: Option<SLocalId>) {
        let mut owners = self.take_drop_scope();
        if let Some(value) = value
            && let Some(position) = owners.iter().position(|owner| *owner == value)
        {
            owners.remove(position);
            self.hold_in_innermost(value);
        }
        if !self.is_terminated(self.current) {
            self.destroy(&owners);
        }
    }

    /// Ends the innermost scope without destroying its owners, which the
    /// caller hands on.
    pub(super) fn take_drop_scope(&mut self) -> Vec<SLocalId> {
        self.drop_scopes
            .pop()
            .expect("lowering ends only the scopes it begins")
            .owners
    }

    /// Destroys the owners of the scopes from `depth` in, for a path leaving
    /// them; `value` leaves with the path.
    pub(super) fn destroy_scopes_from(&mut self, depth: usize, value: Option<SLocalId>) {
        let owners: Vec<_> = self.drop_scopes[depth..]
            .iter()
            .flat_map(|scope| scope.owners.iter().copied())
            .filter(|owner| Some(*owner) != value)
            .collect();
        self.destroy(&owners);
    }

    /// The owners each scope holds so far.
    pub(super) fn held_counts(&self) -> Vec<usize> {
        self.drop_scopes
            .iter()
            .map(|scope| scope.owners.len())
            .collect()
    }

    /// Lowers `f` without the owners the scopes came to hold since `counts`:
    /// it lowers a path they are not defined on, as a `let`'s `else` lies
    /// outside the scope of the pattern's bindings.
    pub(super) fn without_held_since(&mut self, counts: &[usize], f: impl FnOnce(&mut Self)) {
        let held: Vec<_> = self
            .drop_scopes
            .iter_mut()
            .zip(counts)
            .map(|(scope, &count)| scope.owners.split_off(count))
            .collect();
        f(self);
        for (scope, owners) in self.drop_scopes.iter_mut().zip(held) {
            scope.owners.extend(owners);
        }
    }

    /// A block a failed condition takes on its way to `else_bb`, destroying
    /// `owners` and what the scope at `held` holds so far: the bindings and
    /// scrutinees of the conditions before it.
    pub(super) fn condition_exit(
        &mut self,
        else_bb: SBlockId,
        held: Option<usize>,
        owners: &[SLocalId],
    ) -> SBlockId {
        let owners: Vec<_> = held
            .into_iter()
            .flat_map(|depth| self.drop_scopes[depth].owners.iter())
            .chain(owners)
            .copied()
            .collect();
        if owners.is_empty() {
            return else_bb;
        }
        let current = self.current;
        let exit = self.new_block();
        self.switch_to(exit);
        self.destroy(&owners);
        self.set_synthetic_terminator(exit, STerminatorKind::Goto(else_bb));
        self.switch_to(current);
        exit
    }

    /// A write of a value of type `ty` destroys what its place held. A raw
    /// pointer's pointee is written as it is: raw memory is outside the
    /// analysis.
    pub(super) fn drop_before_write(&mut self, place: &SPlace<'db>, ty: TyId<'db>) {
        if self.needs_destroying(ty)
            && !place
                .path
                .iter()
                .any(|projection| matches!(projection, Projection::Deref))
        {
            self.push_synthetic_stmt(SStmtKind::Drop {
                place: place.clone(),
            });
        }
    }

    fn destroy(&mut self, owners: &[SLocalId]) {
        for &owner in owners.iter().rev() {
            self.push_synthetic_stmt(SStmtKind::Drop {
                place: SPlace::new(owner),
            });
        }
    }

    /// The scope a temporary is allocated in, which holds it.
    pub(super) fn allocation_scope(&self) -> Option<u32> {
        self.drop_scopes.last().map(|scope| scope.id)
    }

    /// Holds what an assignment defines, unless it reads a place in place
    /// or projects a value no scope owns, such as an inspected one.
    pub(super) fn hold_assigned(&mut self, kind: &SStmtKind<'db>) {
        if let SStmtKind::Assign { dst, expr } = kind
            && !matches!(expr, SExpr::InspectPlace { .. } | SExpr::Borrow { .. })
            && !matches!(
                expr,
                SExpr::Field { base, .. }
                    | SExpr::Index { base, .. }
                    | SExpr::ExtractEnumField { value: base, .. }
                    if !self.is_held(base.value)
            )
        {
            let depth = match self.locals[dst.index()].source {
                Some(LocalBinding::Local { .. }) => {
                    self.drop_scopes.iter().rposition(|scope| scope.bindings)
                }
                _ => self.temp_scopes[dst.index()]
                    .and_then(|id| self.drop_scopes.iter().position(|scope| scope.id == id)),
            };
            match depth {
                Some(depth) => self.hold_in(*dst, depth),
                None => self.hold_in_innermost(*dst),
            }
        }
    }

    /// Holds the owners of an entry: its `own` parameters and the fields a
    /// receive arm binds.
    pub(super) fn hold_entry(&mut self) {
        for local in self.entry_locals.clone() {
            if matches!(
                self.locals[local.index()].source,
                Some(
                    LocalBinding::Local { .. }
                        | LocalBinding::Param {
                            mode: crate::hir_def::params::FuncParamMode::Own,
                            ..
                        }
                )
            ) {
                self.hold_in(local, 0);
            }
        }
    }

    /// Whether a value of `ty` may own something to destroy. A `Copy` value
    /// owns nothing, whatever its type parameters.
    pub(super) fn needs_destroying(&self, ty: TyId<'db>) -> bool {
        ty_has_drop_glue(self.db, ty)
            && !ty_is_copy(self.db, self.body.scope(), ty, self.assumptions)
    }

    fn is_held(&self, local: SLocalId) -> bool {
        self.drop_scopes
            .iter()
            .any(|scope| scope.owners.contains(&local))
    }

    pub(super) fn hold_in_innermost(&mut self, local: SLocalId) {
        if let Some(depth) = self.drop_scopes.len().checked_sub(1) {
            self.hold_in(local, depth);
        }
    }

    /// Makes the scope at `depth` the owner of `local`'s value, if the value
    /// may need destroying and no scope holds it yet.
    pub(super) fn hold_in(&mut self, local: SLocalId, depth: usize) {
        let data = &self.locals[local.index()];
        if matches!(
            data.role,
            SemanticLocalRole::DirectValue {
                provenance: ValueProvenance::Ordinary
            }
        ) && self.needs_destroying(data.ty)
            && !self.is_held(local)
        {
            self.drop_scopes[depth].owners.push(local);
        }
    }
}
