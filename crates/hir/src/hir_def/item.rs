// This is necessary because `salsa::tracked` structs generates a
// constructor
// that may take many arguments depending on the number of fields in the struct.
#![allow(clippy::too_many_arguments)]

use common::InputFile;
use parser::ast;

use super::{
    scope_graph::{ScopeGraph, ScopeId},
    AttrListId, Body, FuncParamListId, FuncParamName, GenericParamListId, IdentId, IngotId,
    Partial, TupleTypeId, TypeId, UseAlias, WhereClauseId,
};
use crate::{
    hir_def::TraitRefId,
    lower,
    span::{
        item::{
            LazyConstSpan, LazyContractSpan, LazyEnumSpan, LazyFuncSpan, LazyImplSpan,
            LazyImplTraitSpan, LazyItemSpan, LazyModSpan, LazyStructSpan, LazyTopModSpan,
            LazyTraitSpan, LazyTypeAliasSpan, LazyUseSpan,
        },
        params::{LazyGenericParamListSpan, LazyWhereClauseSpan},
        DynLazySpan, HirOrigin,
    },
    HirDb,
};

#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    Hash,
    PartialOrd,
    Ord,
    derive_more::From,
    derive_more::TryInto,
    salsa::Update,
)]
pub enum ItemKind<'db> {
    TopMod(TopLevelMod<'db>),
    Mod(Mod<'db>),
    Func(Func<'db>),
    Struct(Struct<'db>),
    Contract(Contract<'db>),
    Enum(Enum<'db>),
    TypeAlias(TypeAlias<'db>),
    Impl(Impl<'db>),
    Trait(Trait<'db>),
    ImplTrait(ImplTrait<'db>),
    Const(Const<'db>),
    Use(Use<'db>),
    /// Body is not an `Item`, but this makes it easier for analyzers to handle
    /// it.
    Body(Body<'db>),
}

impl<'db> ItemKind<'db> {
    pub fn lazy_span(self) -> LazyItemSpan<'db> {
        LazyItemSpan::new(self)
    }

    pub fn scope(self) -> ScopeId<'db> {
        ScopeId::from_item(self)
    }

    pub fn name(self, db: &'db dyn HirDb) -> Option<IdentId<'db>> {
        use ItemKind::*;
        match self {
            TopMod(top_mod) => Some(top_mod.name(db)),
            Mod(mod_) => mod_.name(db).to_opt(),
            Func(func_) => func_.name(db).to_opt(),
            Struct(struct_) => struct_.name(db).to_opt(),
            Contract(contract_) => contract_.name(db).to_opt(),
            Enum(enum_) => enum_.name(db).to_opt(),
            TypeAlias(alias) => alias.name(db).to_opt(),
            Trait(trait_) => trait_.name(db).to_opt(),
            Const(const_) => const_.name(db).to_opt(),
            Use(_) | Body(_) | Impl(_) | ImplTrait(_) => None,
        }
    }

    /// Returns attributes being applied to this item.
    pub fn attrs(self, db: &'db dyn HirDb) -> Option<AttrListId<'db>> {
        match self {
            Self::Mod(mod_) => mod_.attributes(db),
            Self::Func(func) => func.attributes(db),
            Self::Struct(struct_) => struct_.attributes(db),
            Self::Contract(contract) => contract.attributes(db),
            Self::Enum(enum_) => enum_.attributes(db),
            Self::TypeAlias(alias) => alias.attributes(db),
            Self::Impl(impl_) => impl_.attributes(db),
            Self::Trait(trait_) => trait_.attributes(db),
            Self::ImplTrait(impl_trait) => impl_trait.attributes(db),
            Self::Const(const_) => const_.attributes(db),
            _ => return None,
        }
        .into()
    }

    pub fn kind_name(self) -> &'static str {
        use ItemKind::*;
        match self {
            TopMod(_) => "mod",
            Mod(_) => "mod",
            Func(_) => "fn",
            Struct(_) => "struct",
            Contract(_) => "contract",
            Enum(_) => "enum",
            TypeAlias(_) => "type",
            Trait(_) => "trait",
            Impl(_) => "impl",
            ImplTrait(_) => "impl trait",
            Const(_) => "const",
            Use(_) => "use",
            Body(_) => "body",
        }
    }

    pub fn name_span(self) -> Option<DynLazySpan<'db>> {
        use ItemKind::*;
        match self {
            Mod(mod_) => Some(mod_.lazy_span().name().into()),
            Func(func_) => Some(func_.lazy_span().name().into()),
            Struct(struct_) => Some(struct_.lazy_span().name().into()),
            Contract(contract_) => Some(contract_.lazy_span().name().into()),
            Enum(enum_) => Some(enum_.lazy_span().name().into()),
            TypeAlias(alias) => Some(alias.lazy_span().alias().into()),
            Trait(trait_) => Some(trait_.lazy_span().name().into()),
            Const(const_) => Some(const_.lazy_span().name().into()),
            TopMod(_) | Use(_) | Body(_) | Impl(_) | ImplTrait(_) => None,
        }
    }

    pub fn vis(self, db: &dyn HirDb) -> Visibility {
        use ItemKind::*;
        match self {
            TopMod(top_mod) => top_mod.vis(db),
            Mod(mod_) => mod_.vis(db),
            Func(func) => func.vis(db),
            Struct(struct_) => struct_.vis(db),
            Contract(contract) => contract.vis(db),
            Enum(enum_) => enum_.vis(db),
            TypeAlias(type_) => type_.vis(db),
            Trait(trait_) => trait_.vis(db),
            Const(const_) => const_.vis(db),
            Use(use_) => use_.vis(db),
            Impl(_) | ImplTrait(_) | Body(_) => Visibility::Private,
        }
    }

    pub fn ingot(self, db: &'db dyn HirDb) -> IngotId<'db> {
        let top_mod = self.top_mod(db);
        top_mod.ingot(db)
    }

    pub fn top_mod(self, db: &'db dyn HirDb) -> TopLevelMod<'db> {
        match self {
            ItemKind::TopMod(top_mod) => top_mod,
            ItemKind::Mod(mod_) => mod_.top_mod(db),
            ItemKind::Func(func) => func.top_mod(db),
            ItemKind::Struct(struct_) => struct_.top_mod(db),
            ItemKind::Contract(contract) => contract.top_mod(db),
            ItemKind::Enum(enum_) => enum_.top_mod(db),
            ItemKind::TypeAlias(type_) => type_.top_mod(db),
            ItemKind::Trait(trait_) => trait_.top_mod(db),
            ItemKind::Impl(impl_) => impl_.top_mod(db),
            ItemKind::ImplTrait(impl_trait) => impl_trait.top_mod(db),
            ItemKind::Const(const_) => const_.top_mod(db),
            ItemKind::Use(use_) => use_.top_mod(db),
            ItemKind::Body(body) => body.top_mod(db),
        }
    }

    pub fn is_type(self) -> bool {
        matches!(
            self,
            Self::Struct(_) | Self::Enum(_) | Self::Contract(_) | Self::TypeAlias(_)
        )
    }

    pub fn is_trait(self) -> bool {
        matches!(self, Self::Trait(_))
    }
}

impl<'db> From<GenericParamOwner<'db>> for ItemKind<'db> {
    fn from(owner: GenericParamOwner<'db>) -> Self {
        match owner {
            GenericParamOwner::Func(func) => ItemKind::Func(func),
            GenericParamOwner::Struct(struct_) => ItemKind::Struct(struct_),
            GenericParamOwner::Enum(enum_) => ItemKind::Enum(enum_),
            GenericParamOwner::TypeAlias(type_alias) => ItemKind::TypeAlias(type_alias),
            GenericParamOwner::Impl(impl_) => ItemKind::Impl(impl_),
            GenericParamOwner::Trait(trait_) => ItemKind::Trait(trait_),
            GenericParamOwner::ImplTrait(impl_trait) => ItemKind::ImplTrait(impl_trait),
        }
    }
}

impl<'db> From<WhereClauseOwner<'db>> for ItemKind<'db> {
    fn from(owner: WhereClauseOwner<'db>) -> Self {
        match owner {
            WhereClauseOwner::Func(func) => ItemKind::Func(func),
            WhereClauseOwner::Struct(struct_) => ItemKind::Struct(struct_),
            WhereClauseOwner::Enum(enum_) => ItemKind::Enum(enum_),
            WhereClauseOwner::Impl(impl_) => ItemKind::Impl(impl_),
            WhereClauseOwner::Trait(trait_) => ItemKind::Trait(trait_),
            WhereClauseOwner::ImplTrait(impl_trait) => ItemKind::ImplTrait(impl_trait),
        }
    }
}

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, derive_more::From, salsa::Supertype,
)]
pub enum GenericParamOwner<'db> {
    Func(Func<'db>),
    Struct(Struct<'db>),
    Enum(Enum<'db>),
    TypeAlias(TypeAlias<'db>),
    Impl(Impl<'db>),
    Trait(Trait<'db>),
    ImplTrait(ImplTrait<'db>),
}

impl<'db> GenericParamOwner<'db> {
    pub fn top_mod(self, db: &'db dyn HirDb) -> TopLevelMod<'db> {
        ItemKind::from(self).top_mod(db)
    }

    pub fn params(self, db: &'db dyn HirDb) -> GenericParamListId<'db> {
        match self {
            GenericParamOwner::Func(func) => func.generic_params(db),
            GenericParamOwner::Struct(struct_) => struct_.generic_params(db),
            GenericParamOwner::Enum(enum_) => enum_.generic_params(db),
            GenericParamOwner::TypeAlias(type_alias) => type_alias.generic_params(db),
            GenericParamOwner::Impl(impl_) => impl_.generic_params(db),
            GenericParamOwner::Trait(trait_) => trait_.generic_params(db),
            GenericParamOwner::ImplTrait(impl_trait) => impl_trait.generic_params(db),
        }
    }

    pub fn params_span(self) -> LazyGenericParamListSpan<'db> {
        match self {
            GenericParamOwner::Func(func) => func.lazy_span().generic_params_moved(),
            GenericParamOwner::Struct(struct_) => struct_.lazy_span().generic_params_moved(),
            GenericParamOwner::Enum(enum_) => enum_.lazy_span().generic_params_moved(),
            GenericParamOwner::TypeAlias(type_alias) => {
                type_alias.lazy_span().generic_params_moved()
            }
            GenericParamOwner::Impl(impl_) => impl_.lazy_span().generic_params_moved(),
            GenericParamOwner::Trait(trait_) => trait_.lazy_span().generic_params_moved(),
            GenericParamOwner::ImplTrait(impl_trait) => {
                impl_trait.lazy_span().generic_params_moved()
            }
        }
    }

    pub fn scope(self) -> ScopeId<'db> {
        ItemKind::from(self).scope()
    }

    pub fn from_item_opt(item: ItemKind<'db>) -> Option<Self> {
        match item {
            ItemKind::Func(func) => Some(GenericParamOwner::Func(func)),
            ItemKind::Struct(struct_) => Some(GenericParamOwner::Struct(struct_)),
            ItemKind::Enum(enum_) => Some(GenericParamOwner::Enum(enum_)),
            ItemKind::TypeAlias(type_alias) => Some(GenericParamOwner::TypeAlias(type_alias)),
            ItemKind::Impl(impl_) => Some(GenericParamOwner::Impl(impl_)),
            ItemKind::Trait(trait_) => Some(GenericParamOwner::Trait(trait_)),
            ItemKind::ImplTrait(impl_trait) => Some(GenericParamOwner::ImplTrait(impl_trait)),
            _ => None,
        }
    }

    pub fn parent(self, db: &'db dyn HirDb) -> Option<Self> {
        let ScopeId::Item(item) = self.scope().parent(db)? else {
            return None;
        };

        match item {
            ItemKind::Func(func) => Some(GenericParamOwner::Func(func)),
            ItemKind::Struct(struct_) => Some(GenericParamOwner::Struct(struct_)),
            ItemKind::Enum(enum_) => Some(GenericParamOwner::Enum(enum_)),
            ItemKind::TypeAlias(type_alias) => Some(GenericParamOwner::TypeAlias(type_alias)),
            ItemKind::Impl(impl_) => Some(GenericParamOwner::Impl(impl_)),
            ItemKind::Trait(trait_) => Some(GenericParamOwner::Trait(trait_)),
            ItemKind::ImplTrait(impl_trait) => Some(GenericParamOwner::ImplTrait(impl_trait)),
            _ => None,
        }
    }

    pub fn where_clause_owner(self) -> Option<WhereClauseOwner<'db>> {
        let item = ItemKind::from(self);
        WhereClauseOwner::from_item_opt(item)
    }

    pub fn where_clause(self, db: &'db dyn HirDb) -> Option<WhereClauseId<'db>> {
        self.where_clause_owner()
            .map(|owner| owner.where_clause(db))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, derive_more::From)]
pub enum WhereClauseOwner<'db> {
    Func(Func<'db>),
    Struct(Struct<'db>),
    Enum(Enum<'db>),
    Impl(Impl<'db>),
    Trait(Trait<'db>),
    ImplTrait(ImplTrait<'db>),
}

impl<'db> WhereClauseOwner<'db> {
    pub fn top_mod(self, db: &'db dyn HirDb) -> TopLevelMod<'db> {
        ItemKind::from(self).top_mod(db)
    }

    pub fn where_clause(self, db: &'db dyn HirDb) -> WhereClauseId<'db> {
        match self {
            Self::Func(func) => func.where_clause(db),
            Self::Struct(struct_) => struct_.where_clause(db),
            Self::Enum(enum_) => enum_.where_clause(db),
            Self::Impl(impl_) => impl_.where_clause(db),
            Self::Trait(trait_) => trait_.where_clause(db),
            Self::ImplTrait(impl_trait) => impl_trait.where_clause(db),
        }
    }

    pub fn where_clause_span(self) -> LazyWhereClauseSpan<'db> {
        match self {
            Self::Func(func) => func.lazy_span().where_clause_moved(),
            Self::Struct(struct_) => struct_.lazy_span().where_clause_moved(),
            Self::Enum(enum_) => enum_.lazy_span().where_clause_moved(),
            Self::Impl(impl_) => impl_.lazy_span().where_clause_moved(),
            Self::Trait(trait_) => trait_.lazy_span().where_clause_moved(),
            Self::ImplTrait(impl_trait) => impl_trait.lazy_span().where_clause_moved(),
        }
    }

    pub fn scope(self) -> ScopeId<'db> {
        ItemKind::from(self).scope()
    }

    pub fn from_item_opt(item: ItemKind<'db>) -> Option<Self> {
        match item {
            ItemKind::Func(func) => Some(Self::Func(func)),
            ItemKind::Struct(struct_) => Some(Self::Struct(struct_)),
            ItemKind::Enum(enum_) => Some(Self::Enum(enum_)),
            ItemKind::Impl(impl_) => Some(Self::Impl(impl_)),
            ItemKind::Trait(trait_) => Some(Self::Trait(trait_)),
            ItemKind::ImplTrait(impl_trait) => Some(Self::ImplTrait(impl_trait)),
            _ => None,
        }
    }
}

#[salsa::tracked]
#[derive(Debug)]
pub struct TopLevelMod<'db> {
    // No #[id] here, because `TopLevelMod` is always unique to a `InputFile` that is an argument
    // of `module_scope_graph`.
    pub name: IdentId<'db>,

    pub ingot: IngotId<'db>,
    pub(crate) file: InputFile,
}

#[salsa::tracked]
impl<'db> TopLevelMod<'db> {
    pub fn lazy_span(self) -> LazyTopModSpan<'db> {
        LazyTopModSpan::new(self)
    }

    pub fn scope(self) -> ScopeId<'db> {
        ScopeId::from_item(self.into())
    }

    pub fn scope_graph(self, db: &'db dyn HirDb) -> &'db ScopeGraph<'db> {
        lower::scope_graph_impl(db, self)
    }

    /// Returns the child top level modules of `self`.
    pub fn child_top_mods(
        self,
        db: &'db dyn HirDb,
    ) -> impl Iterator<Item = TopLevelMod<'db>> + 'db {
        let module_tree = self.ingot(db).module_tree(db);
        module_tree.children(self)
    }

    /// Returns the top level children of this module.
    /// If you need all the children, use
    /// [`children_nested`](Self::children_nested) instead.
    pub fn children_non_nested(
        self,
        db: &'db dyn HirDb,
    ) -> impl Iterator<Item = ItemKind<'db>> + 'db {
        let s_graph = self.scope_graph(db);
        let scope = ScopeId::from_item(self.into());
        s_graph.child_items(scope)
    }

    /// Returns all the children of this module, including nested items.
    pub fn children_nested(self, db: &'db dyn HirDb) -> impl Iterator<Item = ItemKind<'db>> + 'db {
        let s_graph = self.scope_graph(db);
        s_graph.items_dfs(db)
    }

    pub fn parent(self, db: &'db dyn HirDb) -> Option<TopLevelMod<'db>> {
        let module_tree = self.ingot(db).module_tree(db);
        module_tree.parent(self)
    }

    pub fn vis(self, _db: &dyn HirDb) -> Visibility {
        // We don't have a way to specify visibility of a top level module.
        // Please change here if we introduce it.
        Visibility::Public
    }

    /// Returns all items in the top level module including ones in nested
    /// modules.
    #[salsa::tracked(return_ref)]
    pub fn all_items(self, db: &'db dyn HirDb) -> Vec<ItemKind<'db>> {
        self.children_nested(db).collect()
    }

    /// Returns all structs in the top level module including ones in nested
    /// modules.
    #[salsa::tracked(return_ref)]
    pub fn all_structs(self, db: &'db dyn HirDb) -> Vec<Struct<'db>> {
        self.all_items(db)
            .iter()
            .filter_map(|item| match item {
                ItemKind::Struct(struct_) => Some(*struct_),
                _ => None,
            })
            .collect()
    }

    /// Returns all enums in the top level module including ones in nested
    /// modules.
    #[salsa::tracked(return_ref)]
    pub fn all_enums(self, db: &'db dyn HirDb) -> Vec<Enum<'db>> {
        self.all_items(db)
            .iter()
            .filter_map(|item| match item {
                ItemKind::Enum(enum_) => Some(*enum_),
                _ => None,
            })
            .collect()
    }

    /// Returns all contracts in the top level module including ones in nested
    /// modules.
    #[salsa::tracked(return_ref)]
    pub fn all_contracts(self, db: &'db dyn HirDb) -> Vec<Contract<'db>> {
        self.all_items(db)
            .iter()
            .filter_map(|item| match item {
                ItemKind::Contract(contract) => Some(*contract),
                _ => None,
            })
            .collect()
    }

    /// Returns all type aliases in the top level module including ones in
    /// nested modules.
    #[salsa::tracked(return_ref)]
    pub fn all_type_aliases(self, db: &'db dyn HirDb) -> Vec<TypeAlias<'db>> {
        self.all_items(db)
            .iter()
            .filter_map(|item| match item {
                ItemKind::TypeAlias(alias) => Some(*alias),
                _ => None,
            })
            .collect()
    }

    /// Returns all traits in the top level module including ones in nested
    /// modules.
    #[salsa::tracked(return_ref)]
    pub fn all_traits(self, db: &'db dyn HirDb) -> Vec<Trait<'db>> {
        self.all_items(db)
            .iter()
            .filter_map(|item| match item {
                ItemKind::Trait(trait_) => Some(*trait_),
                _ => None,
            })
            .collect()
    }

    #[salsa::tracked(return_ref)]
    pub fn all_funcs(self, db: &'db dyn HirDb) -> Vec<Func<'db>> {
        self.all_items(db)
            .iter()
            .filter_map(|item| match item {
                ItemKind::Func(func_) => Some(*func_),
                _ => None,
            })
            .collect()
    }

    /// Returns all traits in the top level module including ones in nested
    /// modules.
    #[salsa::tracked(return_ref)]
    pub fn all_impl_traits(self, db: &'db dyn HirDb) -> Vec<ImplTrait<'db>> {
        self.all_items(db)
            .iter()
            .filter_map(|item| match item {
                ItemKind::ImplTrait(impl_trait) => Some(*impl_trait),
                _ => None,
            })
            .collect()
    }

    /// Returns all impls in the top level module including ones in nested
    /// modules.
    #[salsa::tracked(return_ref)]
    pub fn all_impls(self, db: &'db dyn HirDb) -> Vec<Impl<'db>> {
        self.all_items(db)
            .iter()
            .filter_map(|item| match item {
                ItemKind::Impl(impl_) => Some(*impl_),
                _ => None,
            })
            .collect()
    }
}

#[salsa::tracked]
#[derive(Debug)]
pub struct Mod<'db> {
    #[id]
    id: TrackedItemId<'db>,

    pub name: Partial<IdentId<'db>>,
    pub attributes: AttrListId<'db>,
    pub vis: Visibility,

    pub top_mod: TopLevelMod<'db>,

    #[return_ref]
    pub(crate) origin: HirOrigin<ast::Mod>,
}
impl<'db> Mod<'db> {
    pub fn lazy_span(self) -> LazyModSpan<'db> {
        LazyModSpan::new(self)
    }

    pub fn scope(self) -> ScopeId<'db> {
        ScopeId::from_item(self.into())
    }

    pub fn children_non_nested(
        self,
        db: &'db dyn HirDb,
    ) -> impl Iterator<Item = ItemKind<'db>> + 'db {
        let s_graph = self.top_mod(db).scope_graph(db);
        let scope = ScopeId::from_item(self.into());
        s_graph.child_items(scope)
    }
}

#[salsa::tracked]
#[derive(Debug)]
pub struct Func<'db> {
    #[id]
    id: TrackedItemId<'db>,

    pub name: Partial<IdentId<'db>>,
    pub attributes: AttrListId<'db>,
    pub generic_params: GenericParamListId<'db>,
    pub where_clause: WhereClauseId<'db>,
    pub params: Partial<FuncParamListId<'db>>,
    pub ret_ty: Option<TypeId<'db>>,
    pub modifier: ItemModifier,
    pub body: Option<Body<'db>>,
    pub is_extern: bool,
    pub top_mod: TopLevelMod<'db>,

    #[return_ref]
    pub(crate) origin: HirOrigin<ast::Func>,
}
impl<'db> Func<'db> {
    pub fn lazy_span(self) -> LazyFuncSpan<'db> {
        LazyFuncSpan::new(self)
    }

    pub fn scope(self) -> ScopeId<'db> {
        ScopeId::from_item(self.into())
    }

    pub fn vis(self, db: &dyn HirDb) -> Visibility {
        self.modifier(db).to_visibility()
    }

    pub fn is_method(self, db: &dyn HirDb) -> bool {
        let Some(params) = self.params(db).to_opt() else {
            return false;
        };

        let Some(first_param) = params.data(db).first() else {
            return false;
        };

        first_param.is_self_param(db)
    }

    /// Returns `true` if the function is method or associated functions.
    pub fn is_associated_func(self, db: &dyn HirDb) -> bool {
        let item = match self.scope().parent(db) {
            Some(ScopeId::Item(item)) => item,
            _ => return false,
        };

        matches!(
            item,
            ItemKind::Trait(_) | ItemKind::Impl(_) | ItemKind::ImplTrait(_)
        )
    }

    pub fn param_label(self, db: &'db dyn HirDb, idx: usize) -> Option<IdentId<'db>> {
        self.params(db).to_opt()?.data(db).get(idx)?.label_eagerly()
    }

    pub fn param_label_or_name(self, db: &'db dyn HirDb, idx: usize) -> Option<FuncParamName<'db>> {
        let param = self.params(db).to_opt()?.data(db).get(idx)?;
        param.label.or(param.name.to_opt())
    }
}

#[salsa::tracked]
#[derive(Debug)]
pub struct Struct<'db> {
    #[id]
    id: TrackedItemId<'db>,

    pub name: Partial<IdentId<'db>>,
    pub attributes: AttrListId<'db>,
    pub vis: Visibility,
    pub generic_params: GenericParamListId<'db>,
    pub where_clause: WhereClauseId<'db>,
    pub fields: FieldDefListId<'db>,
    pub top_mod: TopLevelMod<'db>,

    #[return_ref]
    pub(crate) origin: HirOrigin<ast::Struct>,
}
impl<'db> Struct<'db> {
    pub fn lazy_span(self) -> LazyStructSpan<'db> {
        LazyStructSpan::new(self)
    }

    pub fn scope(self) -> ScopeId<'db> {
        ScopeId::from_item(self.into())
    }

    /// Returns the human readable string of the expected struct initializer.
    /// ## Example
    /// When `S` is a struct defined as below:
    /// ```fe
    /// struct S {
    ///    x: u64,
    ///    y: i32,
    /// }
    /// ```
    /// Then this method returns ` { x, y }`.
    pub fn format_initializer_args(self, db: &dyn HirDb) -> String {
        self.fields(db).format_initializer_args(db)
    }
}

#[salsa::tracked]
#[derive(Debug)]
pub struct Contract<'db> {
    #[id]
    id: TrackedItemId<'db>,

    pub name: Partial<IdentId<'db>>,
    pub attributes: AttrListId<'db>,
    pub vis: Visibility,
    pub fields: FieldDefListId<'db>,
    pub top_mod: TopLevelMod<'db>,

    #[return_ref]
    pub(crate) origin: HirOrigin<ast::Contract>,
}
impl<'db> Contract<'db> {
    pub fn lazy_span(self) -> LazyContractSpan<'db> {
        LazyContractSpan::new(self)
    }

    pub fn scope(self) -> ScopeId<'db> {
        ScopeId::from_item(self.into())
    }
}

#[salsa::tracked]
#[derive(Debug)]
pub struct Enum<'db> {
    #[id]
    id: TrackedItemId<'db>,

    pub name: Partial<IdentId<'db>>,
    pub attributes: AttrListId<'db>,
    pub vis: Visibility,
    pub generic_params: GenericParamListId<'db>,
    pub where_clause: WhereClauseId<'db>,
    pub variants: VariantDefListId<'db>,
    pub top_mod: TopLevelMod<'db>,

    #[return_ref]
    pub(crate) origin: HirOrigin<ast::Enum>,
}
impl<'db> Enum<'db> {
    pub fn lazy_span(self) -> LazyEnumSpan<'db> {
        LazyEnumSpan::new(self)
    }

    pub fn scope(self) -> ScopeId<'db> {
        ScopeId::from_item(self.into())
    }
}

#[salsa::tracked]
#[derive(Debug)]
pub struct TypeAlias<'db> {
    #[id]
    id: TrackedItemId<'db>,

    pub name: Partial<IdentId<'db>>,
    pub attributes: AttrListId<'db>,
    pub vis: Visibility,
    pub generic_params: GenericParamListId<'db>,
    pub ty: Partial<TypeId<'db>>,
    pub top_mod: TopLevelMod<'db>,

    #[return_ref]
    pub(crate) origin: HirOrigin<ast::TypeAlias>,
}
impl<'db> TypeAlias<'db> {
    pub fn lazy_span(self) -> LazyTypeAliasSpan<'db> {
        LazyTypeAliasSpan::new(self)
    }

    pub fn scope(self) -> ScopeId<'db> {
        ScopeId::from_item(self.into())
    }
}

#[salsa::tracked]
#[derive(Debug)]
pub struct Impl<'db> {
    #[id]
    id: TrackedItemId<'db>,

    pub ty: super::Partial<TypeId<'db>>,
    pub attributes: AttrListId<'db>,
    pub generic_params: GenericParamListId<'db>,
    pub where_clause: WhereClauseId<'db>,
    pub top_mod: TopLevelMod<'db>,

    #[return_ref]
    pub(crate) origin: HirOrigin<ast::Impl>,
}
impl<'db> Impl<'db> {
    pub fn lazy_span(self) -> LazyImplSpan<'db> {
        LazyImplSpan::new(self)
    }

    pub fn children_non_nested(
        self,
        db: &'db dyn HirDb,
    ) -> impl Iterator<Item = ItemKind<'db>> + 'db {
        let s_graph = self.top_mod(db).scope_graph(db);
        let scope = ScopeId::from_item(self.into());
        s_graph.child_items(scope)
    }

    pub fn funcs(self, db: &'db dyn HirDb) -> impl Iterator<Item = Func<'db>> + 'db {
        let s_graph = self.top_mod(db).scope_graph(db);
        let scope = ScopeId::from_item(self.into());
        s_graph.child_items(scope).filter_map(|item| match item {
            ItemKind::Func(func) => Some(func),
            _ => None,
        })
    }

    pub fn scope(self) -> ScopeId<'db> {
        ScopeId::from_item(self.into())
    }
}

#[salsa::tracked]
#[derive(Debug)]
pub struct Trait<'db> {
    #[id]
    id: TrackedItemId<'db>,

    pub name: Partial<IdentId<'db>>,

    pub attributes: AttrListId<'db>,
    pub vis: Visibility,
    pub generic_params: GenericParamListId<'db>,
    #[return_ref]
    pub super_traits: Vec<TraitRefId<'db>>,
    pub where_clause: WhereClauseId<'db>,
    pub top_mod: TopLevelMod<'db>,

    #[return_ref]
    pub(crate) origin: HirOrigin<ast::Trait>,
}
impl<'db> Trait<'db> {
    pub fn lazy_span(self) -> LazyTraitSpan<'db> {
        LazyTraitSpan::new(self)
    }

    pub fn children_non_nested(
        self,
        db: &'db dyn HirDb,
    ) -> impl Iterator<Item = ItemKind<'db>> + 'db {
        let s_graph = self.top_mod(db).scope_graph(db);
        let scope = ScopeId::from_item(self.into());
        s_graph.child_items(scope)
    }

    pub fn scope(self) -> ScopeId<'db> {
        ScopeId::from_item(self.into())
    }

    pub fn methods(self, db: &'db dyn HirDb) -> impl Iterator<Item = Func<'db>> + 'db {
        let s_graph = self.top_mod(db).scope_graph(db);
        let scope = ScopeId::from_item(self.into());
        s_graph.child_items(scope).filter_map(|item| match item {
            ItemKind::Func(func) => Some(func),
            _ => None,
        })
    }
}

#[salsa::tracked]
#[derive(Debug)]
pub struct ImplTrait<'db> {
    #[id]
    id: TrackedItemId<'db>,

    pub trait_ref: Partial<TraitRefId<'db>>,
    pub ty: Partial<TypeId<'db>>,
    pub attributes: AttrListId<'db>,
    pub generic_params: GenericParamListId<'db>,
    pub where_clause: WhereClauseId<'db>,
    pub top_mod: TopLevelMod<'db>,

    #[return_ref]
    pub(crate) origin: HirOrigin<ast::ImplTrait>,
}
impl<'db> ImplTrait<'db> {
    pub fn lazy_span(self) -> LazyImplTraitSpan<'db> {
        LazyImplTraitSpan::new(self)
    }

    pub fn children_non_nested(
        self,
        db: &'db dyn HirDb,
    ) -> impl Iterator<Item = ItemKind<'db>> + 'db {
        let s_graph = self.top_mod(db).scope_graph(db);
        let scope = ScopeId::from_item(self.into());
        s_graph.child_items(scope)
    }

    pub fn scope(self) -> ScopeId<'db> {
        ScopeId::from_item(self.into())
    }

    pub fn methods(self, db: &'db dyn HirDb) -> impl Iterator<Item = Func<'db>> + 'db {
        self.children_non_nested(db).filter_map(|item| match item {
            ItemKind::Func(func) => Some(func),
            _ => None,
        })
    }
}

#[salsa::tracked]
#[derive(Debug)]
pub struct Const<'db> {
    #[id]
    id: TrackedItemId<'db>,

    pub name: Partial<IdentId<'db>>,
    pub attributes: AttrListId<'db>,
    pub ty: Partial<TypeId<'db>>,
    pub body: Partial<Body<'db>>,
    pub vis: Visibility,
    pub top_mod: TopLevelMod<'db>,

    #[return_ref]
    pub(crate) origin: HirOrigin<ast::Const>,
}
impl<'db> Const<'db> {
    pub fn lazy_span(self) -> LazyConstSpan<'db> {
        LazyConstSpan::new(self)
    }

    pub fn scope(self) -> ScopeId<'db> {
        ScopeId::from_item(self.into())
    }
}

#[salsa::tracked]
#[derive(Debug)]
pub struct Use<'db> {
    #[id]
    id: TrackedItemId<'db>,

    pub path: Partial<super::UsePathId<'db>>,
    pub alias: Option<Partial<UseAlias<'db>>>,
    pub vis: Visibility,
    pub top_mod: TopLevelMod<'db>,

    #[return_ref]
    pub(crate) origin: HirOrigin<ast::Use>,
}
impl<'db> Use<'db> {
    pub fn lazy_span(self) -> LazyUseSpan<'db> {
        LazyUseSpan::new(self)
    }

    pub fn scope(self) -> ScopeId<'db> {
        ScopeId::from_item(self.into())
    }

    /// Returns imported name if it is present and not a glob.
    pub fn imported_name(&self, db: &'db dyn HirDb) -> Option<IdentId<'db>> {
        if let Some(alias) = self.alias(db) {
            return match alias {
                Partial::Present(UseAlias::Ident(name)) => Some(name),
                _ => None,
            };
        }

        self.path(db).to_opt()?.last_ident(db)
    }

    /// Returns the span of imported name span if the use is not a glob.
    /// The returned span is
    /// 1. If the use has an alias, the span of the alias.
    /// 2. If the use has no alias, the span of the last segment of the path.
    pub fn imported_name_span(&self, db: &'db dyn HirDb) -> Option<DynLazySpan<'db>> {
        if self.is_glob(db) {
            return None;
        }

        if self.alias(db).is_some() {
            Some(self.lazy_span().alias().into())
        } else {
            let segment_len = self.path(db).to_opt()?.segment_len(db);
            Some(self.lazy_span().path().segment(segment_len - 1).into())
        }
    }

    pub fn glob_span(&self, db: &dyn HirDb) -> Option<DynLazySpan<'db>> {
        if !self.is_glob(db) {
            return None;
        }

        let segment_len = self.path(db).to_opt()?.segment_len(db);
        Some(self.lazy_span().path().segment(segment_len - 1).into())
    }

    pub fn is_glob(&self, db: &dyn HirDb) -> bool {
        self.path(db).to_opt().is_some_and(|path| path.is_glob(db))
    }

    pub fn is_unnamed(&self, db: &dyn HirDb) -> bool {
        if let Some(alias) = self.alias(db) {
            !matches!(alias, Partial::Present(UseAlias::Ident(_)))
        } else {
            false
        }
    }

    pub(crate) fn pretty_path(&self, db: &dyn HirDb) -> String {
        self.path(db)
            .to_opt()
            .map_or_else(|| "{invalid}".to_string(), |path| path.pretty_path(db))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ItemModifier {
    Pub,
    Unsafe,
    PubAndUnsafe,
    None,
}

impl ItemModifier {
    pub fn to_visibility(self) -> Visibility {
        match self {
            ItemModifier::Pub | ItemModifier::PubAndUnsafe => Visibility::Public,
            ItemModifier::Unsafe | ItemModifier::None => Visibility::Private,
        }
    }
}

#[salsa::interned]
#[derive(Debug)]
pub struct FieldDefListId<'db> {
    #[return_ref]
    pub data: Vec<FieldDef<'db>>,
}

impl<'db> FieldDefListId<'db> {
    pub fn get_field(self, db: &'db dyn HirDb, name: IdentId<'db>) -> Option<&'db FieldDef<'db>> {
        self.data(db)
            .iter()
            .find(|field| field.name.to_opt() == Some(name))
    }

    pub fn field_idx(self, db: &dyn HirDb, name: IdentId<'db>) -> Option<usize> {
        self.data(db)
            .iter()
            .position(|field| field.name.to_opt() == Some(name))
    }

    fn format_initializer_args(self, db: &dyn HirDb) -> String {
        let args = self
            .data(db)
            .iter()
            .map(|field| {
                field
                    .name
                    .to_opt()
                    .map_or_else(|| "_".to_string(), |name| name.data(db).to_string())
            })
            .collect::<Vec<_>>()
            .join(", ");

        format!(" {{ {} }}", args)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FieldDef<'db> {
    pub attributes: AttrListId<'db>,
    pub name: Partial<IdentId<'db>>,
    pub ty: Partial<TypeId<'db>>,
    pub vis: Visibility,
}

#[salsa::interned]
#[derive(Debug)]
pub struct VariantDefListId<'db> {
    #[return_ref]
    pub data: Vec<VariantDef<'db>>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct VariantDef<'db> {
    pub attributes: AttrListId<'db>,
    pub name: Partial<IdentId<'db>>,
    pub kind: VariantKind<'db>,
}

impl VariantDef<'_> {
    /// Returns the human readable string of the expected variant initializer.
    /// ## Example
    /// When enum `E` is an variant defined as below:
    /// ```fe
    /// enum E {
    ///     V(u64, i32),
    ///     S { x: u64, y: i32 },
    /// }
    /// ```
    ///
    /// Then the method returns `(_, _)` for the first variant and ` { x, y }`
    /// for the second variant.
    pub fn format_initializer_args(&self, db: &dyn HirDb) -> String {
        match self.kind {
            VariantKind::Unit => "".to_string(),
            VariantKind::Tuple(tup) => {
                let args = (0..tup.len(db)).map(|_| "_").collect::<Vec<_>>().join(", ");
                format!("({})", args)
            }

            VariantKind::Record(fields) => fields.format_initializer_args(db),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VariantKind<'db> {
    Unit,
    Tuple(TupleTypeId<'db>),
    Record(FieldDefListId<'db>),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Visibility {
    Public,
    Private,
}

impl Visibility {
    pub fn is_pub(self) -> bool {
        self == Self::Public
    }
}

#[salsa::interned]
#[derive(Debug)]
pub struct TrackedItemId<'db> {
    variant: TrackedItemVariant<'db>,
}

impl<'db> TrackedItemId<'db> {
    pub(crate) fn join(self, db: &'db dyn HirDb, variant: TrackedItemVariant<'db>) -> Self {
        let old = self.variant(db);
        let joined = old.join(variant);
        Self::new(db, joined)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum TrackedItemVariant<'db> {
    TopLevelMod(IdentId<'db>),
    Mod(Partial<IdentId<'db>>),
    Func(Partial<IdentId<'db>>),
    Struct(Partial<IdentId<'db>>),
    Contract(Partial<IdentId<'db>>),
    Enum(Partial<IdentId<'db>>),
    TypeAlias(Partial<IdentId<'db>>),
    Impl(Partial<TypeId<'db>>),
    Trait(Partial<IdentId<'db>>),
    ImplTrait(Partial<TraitRefId<'db>>, Partial<TypeId<'db>>),
    Const(Partial<IdentId<'db>>),
    Use(Partial<super::UsePathId<'db>>),
    FuncBody,
    NamelessBody,
    Joined(Box<Self>, Box<Self>),
}
impl TrackedItemVariant<'_> {
    pub(crate) fn join(self, rhs: Self) -> Self {
        Self::Joined(self.into(), rhs.into())
    }
}
