use parser::ast::{self, prelude::AstNode as _};
use salsa::Accumulator as _;

use super::FileLowerCtxt;
use crate::hir_def::{AttrListId, attr::Attr};

/// Payable-related errors accumulated during lowering / validation.
#[salsa::accumulator]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PayableError {
    pub kind: PayableErrorKind,
    pub file: common::file::File,
    pub primary_range: parser::TextRange,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum PayableErrorKind {
    /// `#[payable]` placed on an item that doesn't support it.
    PayableAttrOnUnsupportedItem { item_kind: &'static str },
    /// `#[payable]` placed on a msg variant instead of the corresponding recv arm.
    PayableAttrOnMsgVariant,
    /// `#[payable]` used with arguments or value, e.g. `#[payable(foo)]` or `#[payable = 1]`.
    InvalidPayableAttrForm,
    /// Unknown attribute on an `init` block or `recv` arm (only `#[payable]` is allowed).
    UnknownAttrOnContractEntry {
        attr_name: String,
        entry_kind: &'static str,
    },
}

/// Report an error if `#[payable]` appears on an item that doesn't support it
/// (anything other than a recv arm or init block).
pub(super) fn report_payable_attr_on_unsupported_item<'db>(
    ctxt: &mut FileLowerCtxt<'db>,
    attrs: Option<ast::AttrList>,
    item_kind: &'static str,
) {
    let Some(attrs) = attrs else { return };
    let db = ctxt.db();
    let file = ctxt.top_mod().file(db);

    for attr in attrs {
        let ast::AttrKind::Normal(normal) = attr.kind() else {
            continue;
        };
        let is_payable = normal.path().is_some_and(|p| p.text() == "payable");
        if !is_payable {
            continue;
        }

        PayableError {
            kind: PayableErrorKind::PayableAttrOnUnsupportedItem { item_kind },
            file,
            primary_range: attr.syntax().text_range(),
        }
        .accumulate(db);
    }
}

/// Report a targeted error if `#[payable]` appears on a msg variant.
pub(super) fn report_payable_attr_on_msg_variant<'db>(
    ctxt: &mut FileLowerCtxt<'db>,
    attrs: Option<ast::AttrList>,
) {
    let Some(attrs) = attrs else { return };
    let db = ctxt.db();
    let file = ctxt.top_mod().file(db);

    for attr in attrs {
        let ast::AttrKind::Normal(normal) = attr.kind() else {
            continue;
        };
        let is_payable = normal.path().is_some_and(|p| p.text() == "payable");
        if !is_payable {
            continue;
        }

        PayableError {
            kind: PayableErrorKind::PayableAttrOnMsgVariant,
            file,
            primary_range: attr.syntax().text_range(),
        }
        .accumulate(db);
    }
}

/// Report an error for any non-`#[payable]` normal attribute on an `init` block
/// or `recv` arm.  Only `#[payable]` (and doc comments) are valid here.
pub(super) fn report_unknown_attrs_on_contract_entry<'db>(
    ctxt: &mut FileLowerCtxt<'db>,
    attrs: Option<ast::AttrList>,
    entry_kind: &'static str,
) {
    let Some(attrs) = attrs else { return };
    let db = ctxt.db();
    let file = ctxt.top_mod().file(db);

    for attr in attrs {
        let ast::AttrKind::Normal(normal) = attr.kind() else {
            // Doc comments are always allowed.
            continue;
        };
        let attr_name = normal
            .path()
            .map(|p| p.text().to_string())
            .unwrap_or_default();
        if attr_name == "payable" {
            continue;
        }

        PayableError {
            kind: PayableErrorKind::UnknownAttrOnContractEntry {
                attr_name,
                entry_kind,
            },
            file,
            primary_range: attr.syntax().text_range(),
        }
        .accumulate(db);
    }
}

/// Validate that a `#[payable]` attribute on an init block or recv arm has the
/// correct form (no arguments, no value).
pub(super) fn validate_payable_attr_form<'db>(
    ctxt: &mut FileLowerCtxt<'db>,
    attrs: Option<ast::AttrList>,
) {
    let Some(attrs) = attrs else { return };
    let db = ctxt.db();
    let file = ctxt.top_mod().file(db);

    for attr in attrs {
        let ast::AttrKind::Normal(normal) = attr.kind() else {
            continue;
        };
        let is_payable = normal.path().is_some_and(|p| p.text() == "payable");
        if !is_payable {
            continue;
        }

        // #[payable] must have no arguments and no value
        let has_args = normal.args().is_some();
        let has_value = normal.value().is_some();
        if has_args || has_value {
            PayableError {
                kind: PayableErrorKind::InvalidPayableAttrForm,
                file,
                primary_range: attr.syntax().text_range(),
            }
            .accumulate(db);
        }
    }
}

/// Lower contract-entry attributes while dropping malformed `#[payable]` forms
/// so they do not affect downstream payability semantics.
pub(super) fn lower_contract_entry_attrs_opt<'db>(
    ctxt: &mut FileLowerCtxt<'db>,
    attrs: Option<ast::AttrList>,
) -> AttrListId<'db> {
    let Some(attrs) = attrs else {
        return AttrListId::new(ctxt.db(), vec![]);
    };

    let attrs: Vec<_> = attrs
        .into_iter()
        .filter(|attr| !is_malformed_payable_attr(attr))
        .map(|attr| Attr::lower_ast(ctxt, attr))
        .collect();
    AttrListId::new(ctxt.db(), attrs)
}

fn is_malformed_payable_attr(attr: &ast::Attr) -> bool {
    let ast::AttrKind::Normal(normal) = attr.kind() else {
        return false;
    };

    normal.path().is_some_and(|p| p.text() == "payable")
        && (normal.args().is_some() || normal.value().is_some())
}
