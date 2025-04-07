use super::{Body, IdentId, Partial, PathId};
use crate::{hir_def::TypeId, HirDb};

#[salsa::interned]
#[derive(Debug)]
pub struct GenericArgListId<'db> {
    #[return_ref]
    pub data: Vec<GenericArg<'db>>,
    pub is_given: bool,
}

impl<'db> GenericArgListId<'db> {
    pub fn none(db: &'db dyn HirDb) -> Self {
        Self::new(db, vec![], false)
    }

    pub fn len(self, db: &dyn HirDb) -> usize {
        self.data(db).len()
    }

    pub fn is_empty(self, db: &dyn HirDb) -> bool {
        self.data(db).is_empty()
    }
}

#[salsa::interned]
#[derive(Debug)]
pub struct GenericParamListId<'db> {
    #[return_ref]
    pub data: Vec<GenericParam<'db>>,
}

impl GenericParamListId<'_> {
    pub fn len(&self, db: &dyn HirDb) -> usize {
        self.data(db).len()
    }
}

#[salsa::interned]
#[derive(Debug)]
pub struct FuncParamListId<'db> {
    #[return_ref]
    pub data: Vec<FuncParam<'db>>,
}

#[salsa::interned]
#[derive(Debug)]
pub struct WhereClauseId<'db> {
    #[return_ref]
    pub data: Vec<WherePredicate<'db>>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, derive_more::From)]
pub enum GenericParam<'db> {
    Type(TypeGenericParam<'db>),
    Const(ConstGenericParam<'db>),
}

impl<'db> GenericParam<'db> {
    pub fn name(&self) -> Partial<IdentId<'db>> {
        match self {
            Self::Type(ty) => ty.name,
            Self::Const(c) => c.name,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TypeGenericParam<'db> {
    pub name: Partial<IdentId<'db>>,
    pub bounds: Vec<TypeBound<'db>>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ConstGenericParam<'db> {
    pub name: Partial<IdentId<'db>>,
    pub ty: Partial<TypeId<'db>>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, derive_more::From)]
pub enum GenericArg<'db> {
    Type(TypeGenericArg<'db>),
    Const(ConstGenericArg<'db>),
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TypeGenericArg<'db> {
    pub ty: Partial<TypeId<'db>>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ConstGenericArg<'db> {
    pub body: Partial<Body<'db>>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FuncParam<'db> {
    pub is_mut: bool,
    pub label: Option<FuncParamName<'db>>,
    pub name: Partial<FuncParamName<'db>>,
    pub ty: Partial<TypeId<'db>>,

    /// `true` if this parameter is `self` and the type is not specified.
    /// `ty` should have `Self` type without any type arguments.
    pub self_ty_fallback: bool,
}

impl<'db> FuncParam<'db> {
    pub fn label_eagerly(&self) -> Option<IdentId<'db>> {
        match self.label {
            Some(FuncParamName::Ident(ident)) => return Some(ident),
            Some(FuncParamName::Underscore) => return None,
            _ => {}
        }

        if let FuncParamName::Ident(ident) = self.name.to_opt()? {
            Some(ident)
        } else {
            None
        }
    }

    pub fn name(&self) -> Option<IdentId<'db>> {
        match self.name.to_opt()? {
            FuncParamName::Ident(name) => Some(name),
            _ => None,
        }
    }

    pub fn is_self_param(&self, db: &dyn HirDb) -> bool {
        self.name.to_opt().is_some_and(|name| name.is_self(db))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct WherePredicate<'db> {
    pub ty: Partial<TypeId<'db>>,
    pub bounds: Vec<TypeBound<'db>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FuncParamName<'db> {
    Ident(IdentId<'db>),
    Underscore,
}

impl<'db> FuncParamName<'db> {
    pub fn ident(&self) -> Option<IdentId<'db>> {
        match self {
            FuncParamName::Ident(name) => Some(*name),
            _ => None,
        }
    }

    pub fn is_self(&self, db: &dyn HirDb) -> bool {
        self.ident().is_some_and(|id| id.is_self(db))
    }

    pub fn pretty_print(&self, db: &dyn HirDb) -> String {
        match self {
            FuncParamName::Ident(name) => name.data(db).to_string(),
            FuncParamName::Underscore => "_".to_string(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, salsa::Update)]
pub enum TypeBound<'db> {
    Trait(TraitRefId<'db>),
    Kind(Partial<KindBound>),
}

#[salsa::interned]
#[derive(Debug)]
pub struct TraitRefId<'db> {
    /// The path to the trait.
    pub path: Partial<PathId<'db>>,
}

impl<'db> TraitRefId<'db> {
    /// Returns the generic arg list of the last segment of the trait ref path
    pub fn generic_args(self, db: &'db dyn HirDb) -> Option<GenericArgListId<'db>> {
        self.path(db).to_opt().map(|path| path.generic_args(db))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum KindBound {
    /// `*`
    Mono,
    /// `* -> *`
    Abs(Partial<Box<KindBound>>, Partial<Box<KindBound>>),
}
