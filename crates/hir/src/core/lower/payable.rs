use parser::ast;

use super::{
    FileLowerCtxt,
    attr::{
        AttrForm, AttrRule, AttrTarget, validate_attr_rules,
        validate_unknown_attrs_in_restricted_context,
    },
};
use crate::hir_def::AttrListId;

/// Lower contract-entry attributes after validating the restricted attribute set.
pub(super) fn lower_contract_entry_attrs_opt<'db>(
    ctxt: &mut FileLowerCtxt<'db>,
    attrs: Option<ast::AttrList>,
    entry_kind: &'static str,
) -> AttrListId<'db> {
    let target = AttrTarget::new(entry_kind, None);
    validate_unknown_attrs_in_restricted_context(
        ctxt,
        attrs.clone(),
        target.clone(),
        &["payable"],
        "`#[payable]`",
    );
    validate_attr_rules(
        ctxt,
        attrs.clone(),
        target,
        &[AttrRule::supported(
            "payable",
            AttrForm::Bare,
            "`#[payable]`",
        )],
    );
    AttrListId::lower_ast_opt(ctxt, attrs)
}
