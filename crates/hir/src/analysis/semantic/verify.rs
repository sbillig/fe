use rustc_hash::FxHashSet;

use crate::analysis::semantic::{
    SBlockId, SLocalId, SStmt, STerminator, SemanticBody, VariantIndex,
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SemanticVerifyError {
    MissingSemanticBlock(SBlockId),
    MissingSemanticLocal(SLocalId),
    DuplicateSemanticVariantCase(VariantIndex),
}

pub fn verify_semantic_body<'db>(body: &SemanticBody<'db>) -> Result<(), SemanticVerifyError> {
    for block in &body.blocks {
        for stmt in &block.stmts {
            match stmt {
                SStmt::Assign { dst, .. } => {
                    if body.local(*dst).is_none() {
                        return Err(SemanticVerifyError::MissingSemanticLocal(*dst));
                    }
                }
                SStmt::Store { dst, src } => {
                    if body.local(dst.local).is_none() {
                        return Err(SemanticVerifyError::MissingSemanticLocal(dst.local));
                    }
                    if body.local(*src).is_none() {
                        return Err(SemanticVerifyError::MissingSemanticLocal(*src));
                    }
                }
            }
        }

        if let STerminator::MatchEnum { cases, .. } = &block.terminator {
            let mut seen = FxHashSet::default();
            for (variant, target) in cases.iter().copied() {
                if !seen.insert(variant) {
                    return Err(SemanticVerifyError::DuplicateSemanticVariantCase(variant));
                }
                if body.block(target).is_none() {
                    return Err(SemanticVerifyError::MissingSemanticBlock(target));
                }
            }
        }
    }

    Ok(())
}
