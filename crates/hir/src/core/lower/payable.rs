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
    allow_manual_decode: bool,
) -> AttrListId<'db> {
    let target = AttrTarget::new(entry_kind, None);
    let (allowed, expected): (&[&str], &str) = if allow_manual_decode {
        (
            &["payable", "decode"],
            "`#[payable]` or `#[decode(manual)]`",
        )
    } else {
        (&["payable"], "`#[payable]`")
    };
    validate_unknown_attrs_in_restricted_context(
        ctxt,
        attrs.clone(),
        target.clone(),
        allowed,
        expected,
    );
    validate_attr_rules(
        ctxt,
        attrs.clone(),
        target,
        &[
            AttrRule::supported("payable", AttrForm::Bare, "`#[payable]`"),
            if allow_manual_decode {
                AttrRule::supported(
                    "decode",
                    AttrForm::SingleArg {
                        allow_bare: false,
                        allowed_args: &["manual"],
                    },
                    "`#[decode(manual)]`",
                )
            } else {
                AttrRule::unsupported("decode", "recv arms")
            },
        ],
    );
    AttrListId::lower_ast_opt(ctxt, attrs)
}
