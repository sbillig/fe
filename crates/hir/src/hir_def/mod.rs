// TODO: Remove this when https://github.com/salsa-rs/salsa/pull/513 is fixed.
#![allow(clippy::unused_unit)]
pub mod attr;
pub mod body;
pub mod expr;
pub mod ident;
pub mod item;
pub mod params;
pub mod pat;
pub mod path;
pub mod prim_ty;
pub mod scope_graph;
pub mod stmt;
pub mod types;
pub mod use_tree;

mod scope_graph_viz;

pub(crate) mod module_tree;

pub use attr::*;
pub use body::*;
use common::{input::IngotKind, InputIngot};
pub use expr::*;
pub use ident::*;
pub use item::*;
pub use module_tree::*;
use num_bigint::BigUint;
pub use params::*;
pub use pat::*;
pub use path::*;
use salsa::update::Update;
pub use stmt::*;
pub use types::*;
pub use use_tree::*;

use crate::{external_ingots_impl, HirDb};

#[salsa::tracked]
pub struct IngotId<'db> {
    inner: InputIngot,
}
impl<'db> IngotId<'db> {
    pub fn module_tree(self, db: &dyn HirDb) -> &ModuleTree {
        module_tree_impl(db, self.inner(db))
    }

    pub fn all_modules(self, db: &'db dyn HirDb) -> &[TopLevelMod] {
        all_top_modules_in_ingot(db, self)
    }

    pub fn root_mod(self, db: &dyn HirDb) -> TopLevelMod {
        self.module_tree(db).root_data().top_mod
    }

    pub fn external_ingots(self, db: &dyn HirDb) -> &[(IdentId, IngotId)] {
        external_ingots_impl(db, self.inner(db))
    }

    pub fn kind(self, db: &dyn HirDb) -> IngotKind {
        self.inner(db).kind(db.as_input_db())
    }

    pub fn all_enums(self, db: &'db dyn HirDb) -> &[Enum] {
        all_enums_in_ingot(db, self)
    }

    pub fn all_impl_traits(self, db: &'db dyn HirDb) -> &[ImplTrait] {
        all_impl_traits_in_ingot(db, self)
    }

    pub fn all_impls(self, db: &'db dyn HirDb) -> &Vec<Impl> {
        all_impls_in_ingot(db, self)
    }

    pub fn is_std(self, db: &'db dyn HirDb) -> bool {
        matches!(self.kind(db), IngotKind::Std)
    }
}

#[salsa::interned]
pub struct IntegerId<'db> {
    #[return_ref]
    pub data: BigUint,
}

impl<'db> IntegerId<'db> {
    pub fn from_usize(db: &'db dyn HirDb, value: usize) -> Self {
        let data = BigUint::from(value);
        Self::new(db, data)
    }
}

#[salsa::interned]
pub struct StringId<'db> {
    /// The text of the string literal, without the quotes.
    #[return_ref]
    pub data: String,
}

impl<'db> StringId<'db> {
    pub fn from_str(db: &'db dyn HirDb, value: &str) -> Self {
        let data = value.to_string();
        Self::new(db, data)
    }

    pub fn len_bytes(&self, db: &dyn HirDb) -> usize {
        self.data(db).as_bytes().len()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, derive_more::From, salsa::Update)]
pub enum LitKind<'db> {
    Int(IntegerId<'db>),
    String(StringId<'db>),
    Bool(bool),
}

/// `Partial<T>` is a type that explicitly indicates the possibility that an HIR
/// node cannot be generated due to syntax errors in the source file.
///
/// If a node is `Partial::Absent`, it means that the corresponding AST either
/// does not exist or is erroneous. When a `Partial::Absent` is generated, the
/// relevant error is always generated by the parser, so in Analysis phases, it
/// can often be ignored.
///
/// This type is clearly distinguished from `Option<T>`. The
/// `Option<T>` type is used to hold syntactically valid optional nodes, while
/// `Partial<T>` means that a syntactically required element may be missing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Partial<T> {
    Present(T),
    Absent,
}
unsafe impl<T> Update for Partial<T>
where
    T: Update,
{
    unsafe fn maybe_update(old_ptr: *mut Self, new_val: Self) -> bool {
        use Partial::*;

        let old_val = unsafe { &mut *old_ptr };
        match (old_val, new_val) {
            (Present(old), Present(new)) => T::maybe_update(old, new),
            (Absent, Absent) => false,
            (old_value, new_value) => {
                *old_value = new_value;
                true
            }
        }
    }
}

impl<T> Partial<T> {
    pub fn unwrap(&self) -> &T {
        match self {
            Self::Present(value) => value,
            Self::Absent => panic!("unwrap called on absent value"),
        }
    }

    pub fn to_opt(self) -> Option<T> {
        match self {
            Self::Present(value) => Some(value),
            Self::Absent => None,
        }
    }

    pub fn is_present(&self) -> bool {
        matches!(self, Self::Present(_))
    }
}

impl<T> Default for Partial<T> {
    fn default() -> Self {
        Self::Absent
    }
}

impl<T> From<Option<T>> for Partial<T> {
    fn from(value: Option<T>) -> Self {
        if let Some(value) = value {
            Self::Present(value)
        } else {
            Self::Absent
        }
    }
}

impl<T> From<Partial<T>> for Option<T> {
    fn from(value: Partial<T>) -> Option<T> {
        value.to_opt()
    }
}
