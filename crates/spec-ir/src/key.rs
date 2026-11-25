//! The keys the serialized IR names its types and functions by: readable
//! renderings of what the compiler identifies a type or an instance by,
//! independent of the order a serialization reaches them in.
//!
//! An item is named by its path from its ingot (`core`, `std`, or the
//! ingot's name). A type is rendered as Fe writes it, with every item by its
//! path and every capability (`view`, `ref`, `mut`) shown, and a type argument
//! the compiler has not evaluated marked as such. An instance is the function
//! or constant it is an instance of, by the type its impl is for and the trait
//! it implements at the instance's types, with its own type arguments, its
//! receiver if it is `mut self`, the body mode of a `mut?` function, the providers its
//! effects are bound to, and the spaces of the places its parameters name.
//! The environment an instance was resolved in (`ImplEnv`) and the proofs
//! its caller supplied are not part of it: a closed instance means the same
//! wherever it is resolved, and the serializer checks that each instance
//! under one key serializes the same.

use common::ingot::{Ingot, IngotKind};
use hir::{
    analysis::{
        HirAnalysisDb,
        semantic::SemanticInstanceKey,
        ty::{
            AddressSpace, RootProviderScope,
            binder::Binder,
            const_ty::ConstTyData,
            ty_check::{BodyOwner, EffectParamSite},
            ty_def::{BorrowKind, PrimTy, TyBase, TyData, TyId},
            ty_lower::{CompleteSubst, ParamSchemaId},
        },
    },
    hir_def::{CallableDef, GenericParamOwner, ItemKind, params::FuncParamMode},
    semantic::{ProviderBinding, ProviderSource},
};
use mir::runtime::stable_key::module_path_components_for_scope;

use crate::{assoc_const_name, fe_path};

/// Renders keys for a serialization of a contract of `ingot`, and records
/// whether a key names an item of `ingot`: such an entry is the contract's,
/// any other the library's.
pub(crate) struct Keys<'db> {
    db: &'db dyn HirAnalysisDb,
    ingot: Ingot<'db>,
    local: bool,
}

impl<'db> Keys<'db> {
    pub(crate) fn new(db: &'db dyn HirAnalysisDb, ingot: Ingot<'db>) -> Self {
        Self {
            db,
            ingot,
            local: false,
        }
    }

    /// The key of `ty`, and whether it names an item of the contract's ingot.
    pub(crate) fn ty_key(&mut self, ty: TyId<'db>) -> (String, bool) {
        self.local = false;
        let key = self.ty(ty);
        (key, self.local)
    }

    /// The key of the instance `key`, and whether it names an item of the
    /// contract's ingot.
    pub(crate) fn func_key(&mut self, key: SemanticInstanceKey<'db>) -> (String, bool) {
        self.local = false;
        let rendered = self.instance(key);
        (rendered, self.local)
    }

    /// The path of `item` from its ingot.
    fn item(&mut self, item: ItemKind<'db>) -> String {
        let db = self.db;
        let top_mod = item.top_mod(db);
        let ingot = top_mod.ingot(db);
        self.local |= ingot == self.ingot;
        let root = match ingot.kind(db) {
            IngotKind::Core => "core".to_string(),
            IngotKind::Std => "std".to_string(),
            _ => ingot
                .config(db)
                .clone()
                .and_then(|config| config.metadata.name)
                .map_or_else(
                    || top_mod.name(db).data(db).to_string(),
                    |name| name.to_string(),
                ),
        };
        let modules = module_path_components_for_scope(db, item.scope());
        let name = item.name(db).map_or_else(
            || item.kind_name().replace(' ', "_"),
            |name| name.data(db).to_string(),
        );
        let mut path = vec![root];
        path.extend(modules.into_iter().skip(1));
        path.push(name);
        path.join("::")
    }

    fn ty(&mut self, ty: TyId<'db>) -> String {
        let db = self.db;
        match ty.data(db) {
            TyData::TyApp(..) => {
                let (base, args) = ty.decompose_ty_app(db);
                let args: Vec<_> = args.iter().map(|arg| self.ty(*arg)).collect();
                match base.data(db) {
                    TyData::TyBase(TyBase::Prim(PrimTy::Tuple(1))) => format!("({},)", args[0]),
                    TyData::TyBase(TyBase::Prim(PrimTy::Tuple(_))) => {
                        format!("({})", args.join(", "))
                    }
                    TyData::TyBase(TyBase::Prim(PrimTy::Array)) => match args.as_slice() {
                        [elem, len] => format!("[{elem}; {len}]"),
                        _ => format!("[{}]", args.join(", ")),
                    },
                    TyData::TyBase(TyBase::Prim(
                        prim @ (PrimTy::View | PrimTy::BorrowRef | PrimTy::BorrowMut),
                    )) => {
                        let kind = match prim {
                            PrimTy::View => "view",
                            PrimTy::BorrowRef => "ref",
                            _ => "mut",
                        };
                        format!("{kind} {}", args.join(", "))
                    }
                    TyData::TyBase(TyBase::Prim(PrimTy::Ptr(AddressSpace::Memory))) => {
                        format!("*{}", args.join(", "))
                    }
                    _ => format!("{}<{}>", self.ty(base), args.join(", ")),
                }
            }
            TyData::TyBase(base) => match base {
                TyBase::Prim(_) => ty.pretty_print(db).to_string(),
                TyBase::Shape(_) => ty.pretty_print(db).to_string(),
                TyBase::Adt(adt) => self.item(adt.adt_ref(db).as_item()),
                TyBase::Contract(contract) => self.item((*contract).into()),
                TyBase::Func(callable) => match callable {
                    CallableDef::Func(func) => format!("fn {}", self.item((*func).into())),
                    CallableDef::VariantCtor(variant) => format!(
                        "fn {}::{}",
                        self.item(variant.enum_.into()),
                        variant.name(db).unwrap_or("variant")
                    ),
                    CallableDef::StructCtor(item) => format!("fn {}", self.item((*item).into())),
                },
                TyBase::Closure(closure) => {
                    let def = closure.def(db);
                    let owner = match BodyOwner::closure_parent(db, def) {
                        Some(owner) => self.owner_path(owner),
                        None => fe_path(db, def.body.scope()),
                    };
                    let args = self.args(closure.parent_args(db));
                    format!("{{closure {} in {owner}}}{args}", def.expr.as_u32())
                }
            },
            // A value as Fe writes it; a constant the compiler has not evaluated to one, which
            // can be a type of its own beside the value's, as what it is.
            TyData::ConstTy(constant) => {
                let printed = constant.pretty_print_concrete(db);
                let kind = match constant.data(db) {
                    ConstTyData::Value(_) => return printed.to_string(),
                    ConstTyData::TyVar(..) => "var",
                    ConstTyData::TyParam(..) => "param",
                    ConstTyData::Description(_) => "description",
                    ConstTyData::Invalid(_) => "invalid",
                    ConstTyData::Abstract(..) => "abstract",
                    ConstTyData::Computation { .. } => "computation",
                    ConstTyData::UnEvaluated { .. } => "unevaluated",
                };
                format!("{{{kind} {printed}}}")
            }
            TyData::Never => "!".to_string(),
            TyData::TyParam(_)
            | TyData::AssocTy(_)
            | TyData::QualifiedTy(_)
            | TyData::TyVar(_)
            | TyData::Invalid(_) => format!("{{{}}}", ty.pretty_print(db)),
        }
    }

    /// `<args>`, or nothing if there are none.
    fn args(&mut self, args: &[TyId<'db>]) -> String {
        if args.is_empty() {
            return String::new();
        }
        let args: Vec<_> = args.iter().map(|arg| self.ty(*arg)).collect();
        format!("<{}>", args.join(", "))
    }

    /// The body `owner` names, without the types of an instance of it.
    fn owner_path(&mut self, owner: BodyOwner<'db>) -> String {
        match owner {
            BodyOwner::Func(func, mode) => {
                format!("{}{}", self.item(func.into()), mode_suffix(mode))
            }
            BodyOwner::Const(constant) => self.item(constant.into()),
            BodyOwner::ContractInit { contract } => format!("{}::init", self.item(contract.into())),
            BodyOwner::ContractRecvArm {
                contract,
                recv_idx,
                arm_idx,
            } => format!(
                "{}::recv{recv_idx}::arm{arm_idx}",
                self.item(contract.into())
            ),
            BodyOwner::AnonConstBody { body, .. } => {
                format!("const block in {}", fe_path(self.db, body.scope()))
            }
            BodyOwner::Closure { ty, receiver, .. } => {
                self.closure_body(TyId::closure(self.db, ty), receiver)
            }
        }
    }

    fn closure_body(&mut self, closure: TyId<'db>, receiver: FuncParamMode) -> String {
        let receiver = match receiver {
            FuncParamMode::Mut => "#mut",
            FuncParamMode::View | FuncParamMode::Own => "",
        };
        format!("{}::call{receiver}", self.ty(closure))
    }

    fn instance(&mut self, key: SemanticInstanceKey<'db>) -> String {
        let db = self.db;
        let owner = key.owner(db);
        let args = key.subst(db).generic_args(db);
        // The item the instance's types are for, its scope and its name.
        let (generic, scope, name, mode) = match owner {
            BodyOwner::Func(func, mode) => {
                let name = func.name(db).to_opt().map(|name| name.data(db).to_string());
                (
                    Some(GenericParamOwner::Func(func)),
                    func.scope(),
                    name,
                    mode,
                )
            }
            BodyOwner::AnonConstBody { body, expected } => {
                let parent = body.scope().parent_item(db);
                let generic = match parent {
                    Some(ItemKind::Impl(item)) => Some(GenericParamOwner::Impl(item)),
                    Some(ItemKind::ImplTrait(item)) => Some(GenericParamOwner::ImplTrait(item)),
                    _ => None,
                };
                let name = match parent {
                    Some(ItemKind::Trait(item)) => item
                        .assoc_consts(db)
                        .find(|constant| constant.default_body(db) == Some(body))
                        .and_then(|constant| constant.name(db))
                        .map(|name| name.data(db).to_string()),
                    _ => assoc_const_name(db, body).map(str::to_string),
                };
                match name {
                    Some(name) => (generic, body.scope(), Some(name), None),
                    None => {
                        let expected = self.ty(expected);
                        let args = self.args(args);
                        let path = fe_path(db, body.scope());
                        return format!("const block in {path}{args}: {expected}")
                            + &self.specialization(key);
                    }
                }
            }
            BodyOwner::Closure { ty, receiver, .. } => {
                return self.closure_body(TyId::closure(db, ty), receiver)
                    + &self.specialization(key);
            }
            owner => return self.owner_path(owner) + &self.args(args) + &self.specialization(key),
        };
        let name = name.unwrap_or_else(|| "?".to_string());
        // The type an impl is for and the trait it implements, at the
        // instance's types, and the number of those the impl takes.
        let mut instantiate = |ty: TyId<'db>, parent: GenericParamOwner<'db>| {
            let owner = generic.unwrap_or(parent);
            let subst = CompleteSubst::for_owner(db, owner, args.to_vec()).ok();
            let ty = subst
                .and_then(|subst| Binder::bind(owner, ty).instantiate_subst(db, &subst).ok())
                .unwrap_or(ty);
            self.ty(ty)
        };
        let (path, taken) = match scope.parent_item(db) {
            Some(ItemKind::Impl(item)) => {
                let parent = GenericParamOwner::Impl(item);
                let ty = instantiate(item.ty(db), parent);
                let taken = ParamSchemaId::full(db, parent).keys(db).len();
                (format!("{ty}::{name}"), taken)
            }
            Some(ItemKind::ImplTrait(item)) => {
                let parent = GenericParamOwner::ImplTrait(item);
                let ty = instantiate(item.ty(db), parent);
                let trait_args: Vec<_> = item
                    .trait_inst(db)
                    .map(|inst| {
                        inst.trait_ref(db)
                            .args(db)
                            .iter()
                            .skip(1)
                            .copied()
                            .collect()
                    })
                    .unwrap_or_default();
                let trait_args: Vec<_> = trait_args
                    .into_iter()
                    .map(|arg| instantiate(arg, parent))
                    .collect();
                let trait_path = match item.trait_def(db) {
                    Some(def) => self.item(def.into()),
                    None => "?".to_string(),
                };
                let trait_args = match trait_args.as_slice() {
                    [] => String::new(),
                    args => format!("<{}>", args.join(", ")),
                };
                let taken = ParamSchemaId::full(db, parent).keys(db).len();
                (format!("<{ty} as {trait_path}{trait_args}>::{name}"), taken)
            }
            Some(ItemKind::Trait(item)) => (format!("{}::{name}", self.item(item.into())), 0),
            _ => match owner {
                BodyOwner::Func(func, _) => (self.item(func.into()), 0),
                _ => (format!("{}::{name}", fe_path(db, scope)), 0),
            },
        };
        let own = self.args(args.get(taken..).unwrap_or_default());
        // An impl can implement a `mut?` method by one function per receiver mode, both of its
        // name: the one that takes `mut self` is told apart by its receiver.
        let receiver = match owner {
            BodyOwner::Func(func, None)
                if func.is_method(db)
                    && func
                        .params(db)
                        .next()
                        .is_some_and(|param| param.mode_at(db, None) == FuncParamMode::Mut) =>
            {
                "(mut self)"
            }
            _ => "",
        };
        format!(
            "{path}{own}{receiver}{}{}",
            mode_suffix(mode),
            self.specialization(key)
        )
    }

    /// What `key` is specialized on beyond its types: the providers its
    /// effects are bound to, and the spaces of the places its parameters
    /// name, if any.
    fn specialization(&mut self, key: SemanticInstanceKey<'db>) -> String {
        let db = self.db;
        let effects = key.effect_providers(db);
        let mut out = String::new();
        if !effects.providers(db).is_empty() {
            let providers: Vec<_> = effects
                .providers(db)
                .iter()
                .map(|provider| self.provider(provider))
                .collect();
            out.push_str(&format!(" uses ({})", providers.join(", ")));
        }
        if !effects.param_spaces(db).is_empty() {
            let spaces: Vec<_> = effects
                .param_spaces(db)
                .iter()
                .map(|(param, space)| format!("{param}: {space:?}"))
                .collect();
            out.push_str(&format!(" at ({})", spaces.join(", ")));
        }
        out
    }

    fn provider(&mut self, binding: &ProviderBinding<'db>) -> String {
        let source = match &binding.source {
            ProviderSource::UsesParam {
                site,
                requirement_idx,
            } => format!("{} uses {requirement_idx}", self.site(*site)),
            ProviderSource::ContractField { field } => {
                format!("{}.field{}", self.item(field.contract.into()), field.index)
            }
            ProviderSource::RootProvider {
                scope,
                registration,
            } => {
                let scope = match scope {
                    RootProviderScope::Func(func) => self.item((*func).into()),
                    RootProviderScope::Contract(contract) => self.item((*contract).into()),
                };
                format!(
                    "root {} of {scope} {:?}: {}",
                    registration.idx,
                    registration.site_kind,
                    self.ty(registration.provider_ty)
                )
            }
        };
        let semantics = &binding.semantics;
        let target = semantics
            .target_ty
            .map(|ty| format!(" -> {}", self.ty(ty)))
            .unwrap_or_default();
        format!(
            "{}{}: {} = {source} as {:?} {:?}{target} {}",
            binding.provider_idx,
            if binding.is_mut { " mut" } else { "" },
            self.ty(binding.provider_ty),
            semantics.kind,
            semantics.address_space,
            self.ty(semantics.provider_ty),
        )
    }

    fn site(&mut self, site: EffectParamSite<'db>) -> String {
        match site {
            EffectParamSite::Func(func, mode) => {
                format!("{}{}", self.item(func.into()), mode_suffix(mode))
            }
            EffectParamSite::Contract(contract) => self.item(contract.into()),
            EffectParamSite::ContractInit { contract } => {
                format!("{}::init", self.item(contract.into()))
            }
            EffectParamSite::ContractRecvArm {
                contract,
                recv_idx,
                arm_idx,
            } => format!(
                "{}::recv{recv_idx}::arm{arm_idx}",
                self.item(contract.into())
            ),
            EffectParamSite::Closure(def) => {
                let owner = match BodyOwner::closure_parent(self.db, def) {
                    Some(owner) => self.owner_path(owner),
                    None => fe_path(self.db, def.body.scope()),
                };
                format!("{{closure {} in {owner}}}", def.expr.as_u32())
            }
        }
    }
}

/// A `mut?` function's bodies, one per mode, are distinct.
fn mode_suffix(mode: Option<BorrowKind>) -> &'static str {
    match mode {
        None => "",
        Some(BorrowKind::Ref) => "#ref",
        Some(BorrowKind::Mut) => "#mut",
    }
}
