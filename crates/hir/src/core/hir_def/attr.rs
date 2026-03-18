use crate::HirDb;

use super::{IdentId, Partial, PathId, StringId};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ArithmeticMode {
    Checked,
    Unchecked,
}

impl ArithmeticMode {
    pub fn parse(name: &str) -> Option<Self> {
        match name {
            "checked" => Some(Self::Checked),
            "unchecked" => Some(Self::Unchecked),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, salsa::Update)]
pub enum InlineHint {
    Hint,
    Always,
    Never,
}

impl InlineHint {
    pub fn pretty_print(self) -> &'static str {
        match self {
            Self::Hint => "#[inline]",
            Self::Always => "#[inline(always)]",
            Self::Never => "#[inline(never)]",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, salsa::Update)]
pub enum InlineAttrErrorKind {
    Duplicate,
    InvalidForm,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, salsa::Update)]
pub enum LoopUnrollAttrErrorKind {
    Duplicate,
    InvalidForm,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, salsa::Update)]
pub enum InlineAttr {
    Hint(InlineHint),
    Error(InlineAttrErrorKind),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, salsa::Update)]
pub enum LoopUnrollAttr {
    Hint(bool),
    Error(LoopUnrollAttrErrorKind),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct InlineAttrSpec {
    pub has_value: bool,
    pub args: Vec<InlineAttrArgSpec>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct InlineAttrArgSpec {
    pub key: Option<String>,
    pub has_value: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct InlineAttrParseError {
    pub kind: InlineAttrErrorKind,
    pub attr_index: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct LoopUnrollAttrSpec {
    pub enabled: bool,
    pub has_value: bool,
    pub has_args: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct LoopUnrollAttrParseError {
    pub kind: LoopUnrollAttrErrorKind,
    pub attr_index: usize,
}

pub(crate) fn parse_inline_attr_specs(
    attrs: impl IntoIterator<Item = InlineAttrSpec>,
) -> Result<Option<InlineHint>, InlineAttrParseError> {
    let mut inline_hint = None;

    for (attr_index, attr) in attrs.into_iter().enumerate() {
        let parsed_hint = parse_inline_attr_spec(&attr)
            .map_err(|kind| InlineAttrParseError { kind, attr_index })?;

        if inline_hint.is_some() {
            return Err(InlineAttrParseError {
                kind: InlineAttrErrorKind::Duplicate,
                attr_index,
            });
        }

        inline_hint = Some(parsed_hint);
    }

    Ok(inline_hint)
}

pub(crate) fn parse_loop_unroll_attr_specs(
    attrs: impl IntoIterator<Item = LoopUnrollAttrSpec>,
) -> Result<Option<bool>, LoopUnrollAttrParseError> {
    let mut unroll_hint = None;

    for (attr_index, attr) in attrs.into_iter().enumerate() {
        if attr.has_value || attr.has_args {
            return Err(LoopUnrollAttrParseError {
                kind: LoopUnrollAttrErrorKind::InvalidForm,
                attr_index,
            });
        }

        if unroll_hint.is_some() {
            return Err(LoopUnrollAttrParseError {
                kind: LoopUnrollAttrErrorKind::Duplicate,
                attr_index,
            });
        }

        unroll_hint = Some(attr.enabled);
    }

    Ok(unroll_hint)
}

fn parse_inline_attr_spec(attr: &InlineAttrSpec) -> Result<InlineHint, InlineAttrErrorKind> {
    if attr.has_value {
        return Err(InlineAttrErrorKind::InvalidForm);
    }
    if attr.args.is_empty() {
        return Ok(InlineHint::Hint);
    }
    if attr.args.len() != 1 {
        return Err(InlineAttrErrorKind::InvalidForm);
    }

    let arg = &attr.args[0];
    if arg.has_value {
        return Err(InlineAttrErrorKind::InvalidForm);
    }

    match arg.key.as_deref() {
        Some("always") => Ok(InlineHint::Always),
        Some("never") => Ok(InlineHint::Never),
        Some(_) | None => Err(InlineAttrErrorKind::InvalidForm),
    }
}

#[salsa::interned]
#[derive(Debug)]
pub struct AttrListId<'db> {
    #[return_ref]
    pub data: Vec<Attr<'db>>,
}

impl<'db> AttrListId<'db> {
    /// Returns true if this attribute list contains an attribute with the given name.
    ///
    /// Only checks simple identifier attributes (e.g., `#[msg]`), not path attributes.
    pub fn has_attr(self, db: &'db dyn HirDb, name: &str) -> bool {
        self.data(db).iter().any(|attr| {
            if let Attr::Normal(normal_attr) = attr
                && let Some(path) = normal_attr.path.to_opt()
                && let Some(ident) = path.as_ident(db)
            {
                ident.data(db) == name
            } else {
                false
            }
        })
    }

    /// Returns true if this attribute list contains a marker attribute with the given name.
    ///
    /// Marker attributes have no arguments in lowered HIR, for example `#[payable]`.
    pub fn has_marker_attr(self, db: &'db dyn HirDb, name: &str) -> bool {
        self.data(db).iter().any(|attr| {
            if let Attr::Normal(normal_attr) = attr
                && normal_attr.args.is_empty()
                && let Some(path) = normal_attr.path.to_opt()
                && let Some(ident) = path.as_ident(db)
            {
                ident.data(db) == name
            } else {
                false
            }
        })
    }

    /// Returns the attribute with the given name, if present.
    pub fn get_attr(self, db: &'db dyn HirDb, name: &str) -> Option<&'db NormalAttr<'db>> {
        self.data(db).iter().find_map(|attr| {
            if let Attr::Normal(normal_attr) = attr
                && let Some(path) = normal_attr.path.to_opt()
                && let Some(ident) = path.as_ident(db)
                && ident.data(db) == name
            {
                Some(normal_attr)
            } else {
                None
            }
        })
    }

    pub fn arithmetic_mode(self, db: &'db dyn HirDb) -> Option<ArithmeticMode> {
        self.data(db)
            .iter()
            .filter_map(|attr| {
                let Attr::Normal(normal_attr) = attr else {
                    return None;
                };
                let path = normal_attr.path.to_opt()?;
                let ident = path.as_ident(db)?;
                if ident.data(db) != "arithmetic" {
                    return None;
                }
                normal_attr.arithmetic_mode_arg(db)
            })
            .last()
    }
    pub fn inline_attr(self, db: &'db dyn HirDb) -> Option<InlineAttr> {
        match parse_inline_attr_specs(self.data(db).iter().filter_map(|attr| {
            let Attr::Normal(normal_attr) = attr else {
                return None;
            };
            if normal_attr
                .path
                .to_opt()
                .and_then(|path| path.as_ident(db))
                .is_none_or(|ident| ident.data(db) != "inline")
            {
                return None;
            }

            Some(normal_attr.inline_attr_spec(db))
        })) {
            Ok(Some(hint)) => Some(InlineAttr::Hint(hint)),
            Ok(None) => None,
            Err(err) => Some(InlineAttr::Error(err.kind)),
        }
    }

    pub(crate) fn parse_loop_unroll_attr(
        self,
        db: &'db dyn HirDb,
    ) -> Result<Option<bool>, LoopUnrollAttrParseError> {
        parse_loop_unroll_attr_specs(self.data(db).iter().filter_map(|attr| {
            let Attr::Normal(normal_attr) = attr else {
                return None;
            };
            normal_attr.loop_unroll_attr_spec(db)
        }))
    }

    pub fn loop_unroll_attr(self, db: &'db dyn HirDb) -> Option<LoopUnrollAttr> {
        match self.parse_loop_unroll_attr(db) {
            Ok(Some(hint)) => Some(LoopUnrollAttr::Hint(hint)),
            Ok(None) => None,
            Err(err) => Some(LoopUnrollAttr::Error(err.kind)),
        }
    }
}

impl<'db> NormalAttr<'db> {
    /// Returns true if this attribute has an argument with the given key (no value).
    ///
    /// For example, `#[test(should_revert)]` has the argument `should_revert`.
    pub fn has_arg(&self, db: &'db dyn HirDb, key: &str) -> bool {
        self.args.iter().any(|arg| {
            !arg.has_value
                && arg
                    .key
                    .to_opt()
                    .and_then(|p| p.as_ident(db))
                    .is_some_and(|ident| ident.data(db) == key)
        })
    }

    pub fn arithmetic_mode_arg(&self, db: &'db dyn HirDb) -> Option<ArithmeticMode> {
        let [arg] = self.args.as_slice() else {
            return None;
        };
        if arg.has_value {
            return None;
        }
        let mode = arg
            .key
            .to_opt()
            .and_then(|path| path.as_ident(db))
            .map(|ident| ident.data(db).as_str())?;
        ArithmeticMode::parse(mode)
    }

    pub(crate) fn inline_attr_spec(&self, db: &'db dyn HirDb) -> InlineAttrSpec {
        InlineAttrSpec {
            has_value: self.has_value,
            args: self
                .args
                .iter()
                .map(|arg| InlineAttrArgSpec {
                    key: arg.key_str(db).map(str::to_owned),
                    has_value: arg.has_value,
                })
                .collect(),
        }
    }

    pub(crate) fn loop_unroll_attr_spec(&self, db: &'db dyn HirDb) -> Option<LoopUnrollAttrSpec> {
        let enabled = match self
            .path
            .to_opt()
            .and_then(|path| path.as_ident(db))
            .map(|ident| ident.data(db).as_str())
        {
            Some("unroll") => true,
            Some("no_unroll") => false,
            Some(_) | None => return None,
        };

        Some(LoopUnrollAttrSpec {
            enabled,
            has_value: self.has_value,
            has_args: !self.args.is_empty(),
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, derive_more::From)]
pub enum Attr<'db> {
    Normal(NormalAttr<'db>),
    DocComment(DocCommentAttr<'db>),
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct NormalAttr<'db> {
    pub path: Partial<PathId<'db>>,
    /// The value after `=` in `#[attr = value]`.
    pub value: Option<AttrArgValue<'db>>,
    /// True when the source contained `= ...`, even if the typed value was not lowerable.
    pub has_value: bool,
    pub args: Vec<AttrArg<'db>>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct DocCommentAttr<'db> {
    /// This is the text of the doc comment, excluding the `///` prefix.
    pub text: StringId<'db>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct AttrArg<'db> {
    pub key: Partial<PathId<'db>>,
    /// The value after `=` in `#[attr(key = value)]`.
    pub value: Option<AttrArgValue<'db>>,
    /// True when the source contained `= ...`, even if the typed value was not lowerable.
    pub has_value: bool,
}

impl<'db> AttrArg<'db> {
    pub fn key_str(&self, db: &'db dyn HirDb) -> Option<&str> {
        self.key
            .to_opt()
            .and_then(|p| p.as_ident(db))
            .map(|i| i.data(db).as_str())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum AttrArgValue<'db> {
    Ident(IdentId<'db>),
    Lit(super::LitKind<'db>),
}
