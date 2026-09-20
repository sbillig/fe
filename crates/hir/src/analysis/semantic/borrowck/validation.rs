//! Track validation dependencies without assigning a harmless effect to opaque calls.
use crate::analysis::{
    HirAnalysisDb,
    semantic::{
        SemOrigin, SemanticInstance,
        diagnostics::{SemanticDiagnostic, SemanticDiagnosticKind, SemanticDiagnosticSpan},
        normalized::{NExpr, NStatementKind},
    },
};
use cranelift_entity::EntityRef;

use super::{ir::PendingSemanticValidation, solver::Borrowck};

pub(super) fn can_specialize(db: &dyn HirAnalysisDb, instance: SemanticInstance<'_>) -> bool {
    instance
        .key(db)
        .subst(db)
        .generic_args(db)
        .iter()
        .any(|ty| ty.has_param(db) || ty.has_var(db) || ty.contains_assoc_ty_of_param(db))
}

impl<'db> PendingSemanticValidation<'db> {
    pub(super) fn diagnostic(
        &self,
        db: &'db dyn HirAnalysisDb,
        instance: SemanticInstance<'db>,
    ) -> SemanticDiagnostic<'db> {
        let owner = instance.key(db).owner(db);
        let mut diagnostic = SemanticDiagnostic::new(
            instance,
            SemanticDiagnosticKind::UnresolvedCall,
            "executable calls require concrete implementations or verified effect contracts".into(),
            SemanticDiagnosticSpan::Origin {
                owner,
                origin: SemOrigin::Body(owner),
            },
        );
        for callee in &self.callees {
            let owner = callee.owner(db);
            diagnostic.push_secondary(
                "validation depends on this opaque callee".into(),
                SemanticDiagnosticSpan::Origin {
                    owner,
                    origin: SemOrigin::Body(owner),
                },
            );
        }
        diagnostic
    }
}

impl Borrowck<'_> {
    /// An unbounded opaque effect makes subsequent memory facts conditional.
    /// Keep checks before that effect, including the call's operand checks.
    pub(super) fn prepare_validation_dependencies(&mut self) {
        self.validation_dependencies = self
            .body
            .blocks
            .iter()
            .map(|block| vec![false; block.statements.len() + 1])
            .collect();
        if !can_specialize(self.db, self.instance) || self.pending.callees.is_empty() {
            return;
        }
        let mut entries = vec![false; self.body.blocks.len()];
        loop {
            let mut changed = false;
            for (index, block) in self.body.blocks.iter().enumerate() {
                let mut pending = entries[index];
                for (statement_index, statement) in block.statements.iter().enumerate() {
                    self.validation_dependencies[index][statement_index] = pending;
                    pending |= matches!(statement.kind,
                        NStatementKind::Define { result, expr: NExpr::Call { .. } }
                            if self.calls.get(&result).is_some_and(|call| call.pending));
                }
                self.validation_dependencies[index][block.statements.len()] = pending;
                for successor in block.terminator.kind.successors() {
                    if pending && !entries[successor.block.index()] {
                        entries[successor.block.index()] = true;
                        changed = true;
                    }
                }
            }
            if !changed {
                break;
            }
        }
    }

    pub(super) fn call_validation_pending(&self, block: usize, index: usize) -> bool {
        can_specialize(self.db, self.instance)
            && matches!(self.body.blocks[block].statements[index].kind,
                NStatementKind::Define { result, expr: NExpr::Call { .. } }
                    if self.calls.get(&result).is_some_and(|call| call.pending))
    }
}
