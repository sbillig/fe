//! The keys the serialized IR names its types and functions by: readable
//! renderings of what the compiler identifies a type or an instance by,
//! independent of the order a serialization reaches them in.
//!
//! An item is named by its path from its ingot (`core`, `std`, or the
//! ingot's name, [`ingot_name`]; two ingots of one name in one serialization
//! are refused). A type is rendered as Fe writes it, with every item by its
//! path and every capability (`view`, `ref`, `mut`) shown, and a type argument
//! the compiler has not evaluated marked as such. An instance is the function
//! or constant it is an instance of, by the type its impl is for and the trait
//! it implements at the instance's types, with its own type arguments, its
//! receiver unless it is `self`, the body mode of a `mut?` function, the
//! providers its effects are bound to, and the spaces of the places its
//! parameters name.
//!
//! An entry is the contract's when anything its key names is defined in the
//! contract's ingot: an item, the scope of a method or a body, or a constant.
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
    hir_def::{
        CallableDef, GenericParamOwner, ItemKind, TopLevelMod, params::FuncParamMode,
        scope_graph::ScopeId,
    },
    semantic::{ProviderBinding, ProviderSource},
};
use mir::runtime::stable_key::module_path_components_for_scope;
use rustc_hash::FxHashMap;

use crate::{Unsupported, assoc_const_name};

/// Renders keys for a serialization of a contract of `ingot`, and records
/// whether a key names an item of `ingot`: such an entry is the contract's,
/// any other the library's.
pub(crate) struct Keys<'db> {
    db: &'db dyn HirAnalysisDb,
    ingot: Ingot<'db>,
    local: bool,
    /// The ingot each name names so far.
    ingots: FxHashMap<String, Ingot<'db>>,
    /// Why the key being rendered is refused.
    refused: Option<String>,
}

impl<'db> Keys<'db> {
    pub(crate) fn new(db: &'db dyn HirAnalysisDb, ingot: Ingot<'db>) -> Self {
        Self {
            db,
            ingot,
            local: false,
            ingots: FxHashMap::default(),
            refused: None,
        }
    }

    /// The key of `ty`, and whether it names something defined in the contract's ingot.
    pub(crate) fn ty_key(&mut self, ty: TyId<'db>) -> Result<(String, bool), Unsupported> {
        self.local = false;
        let key = self.ty(ty);
        self.finish(key)
    }

    /// The key of the instance `key`, and whether it names something defined in the
    /// contract's ingot.
    pub(crate) fn func_key(
        &mut self,
        key: SemanticInstanceKey<'db>,
    ) -> Result<(String, bool), Unsupported> {
        self.local = false;
        let rendered = self.instance(key);
        self.finish(rendered)
    }

    fn finish(&mut self, key: String) -> Result<(String, bool), Unsupported> {
        match self.refused.take() {
            Some(reason) => Err(Unsupported(format!("no key for `{key}`: {reason}"))),
            None => Ok((key, self.local)),
        }
    }

    /// The name of `ingot` in keys, recording that `ingot` has it: two ingots under one name
    /// would give their items one key.
    fn ingot(&mut self, ingot: Ingot<'db>, top_mod: TopLevelMod<'db>) -> String {
        let db = self.db;
        self.local |= ingot == self.ingot;
        let name = ingot_name(db, ingot, top_mod);
        let named = *self.ingots.entry(name.clone()).or_insert(ingot);
        if named != ingot {
            self.refused = Some(format!(
                "two ingots are named `{name}`: {} and {}",
                named.base(db),
                ingot.base(db)
            ));
        }
        name
    }

    /// The path of `scope`, a body or an item, from its ingot.
    fn scope(&mut self, scope: ScopeId<'db>) -> String {
        let db = self.db;
        let top_mod = scope.top_mod(db);
        let root = self.ingot(top_mod.ingot(db), top_mod);
        let path = scope.pretty_path(db).unwrap_or_default();
        match path.split_once("::") {
            Some((_, rest)) => format!("{root}::{rest}"),
            None => root,
        }
    }

    /// The path of `item` from its ingot.
    fn item(&mut self, item: ItemKind<'db>) -> String {
        let db = self.db;
        let top_mod = item.top_mod(db);
        let root = self.ingot(top_mod.ingot(db), top_mod);
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
                        None => self.scope(def.body.scope()),
                    };
                    let args = self.args(closure.parent_args(db));
                    format!("{{closure {} in {owner}}}{args}", def.expr.as_u32())
                }
            },
            // A value as Fe writes it; a constant the compiler has not evaluated to one, which
            // can be a type of its own beside the value's, as what it is and where the body it
            // comes from is. What the constant is of, and where it is defined, decide whose it is.
            // A type a closed instance names holds no other constant: no variable or parameter
            // (the instance is closed), no invalid constant (the program has no errors), no
            // abstract expression (a closed one is evaluated), and no description, which the
            // serializer refuses as a constant value too; one is refused by what it is.
            TyData::ConstTy(constant) => {
                let printed = constant.pretty_print_concrete(db);
                self.ty(constant.ty(db));
                let (kind, scope) = match constant.data(db) {
                    ConstTyData::Value(_) => return printed.to_string(),
                    ConstTyData::Computation { source, .. } => {
                        ("computation", Some(source.owner(db).scope()))
                    }
                    ConstTyData::UnEvaluated { body, .. } => ("unevaluated", Some(body.scope())),
                    ConstTyData::TyVar(..) => ("variable", None),
                    ConstTyData::TyParam(..) => ("parameter", None),
                    ConstTyData::Description(_) => ("description", None),
                    ConstTyData::Invalid(_) => ("invalid constant", None),
                    ConstTyData::Abstract(..) => ("abstract expression", None),
                };
                match scope {
                    Some(scope) => format!("{{{kind} {printed} in {}}}", self.scope(scope)),
                    None => {
                        self.refused = Some(format!("a constant `{printed}` that is a {kind}"));
                        format!("{{{kind} {printed}}}")
                    }
                }
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
                format!("const block in {}", self.scope(body.scope()))
            }
            BodyOwner::Closure { ty, receiver, .. } => {
                self.closure_body(TyId::closure(self.db, ty), receiver)
            }
        }
    }

    fn closure_body(&mut self, closure: TyId<'db>, receiver: FuncParamMode) -> String {
        format!("{}::call{}", self.ty(closure), receiver_suffix(receiver))
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
                    // A body reaches a constant body through a constant reference, which names
                    // a `const` item or an associated constant (`resolve_semantic_const_ref`),
                    // so a body that is neither has no name to key it by.
                    None => {
                        let path = self.scope(body.scope());
                        let expected = self.ty(expected);
                        self.refused = Some(format!(
                            "a constant body of type `{expected}` in `{path}` that is no named \
                             constant"
                        ));
                        return format!("const block in {path}");
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
        // A method defined in the contract's ingot is the contract's, whatever it is for.
        self.local |= scope.ingot(db) == self.ingot;
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
                _ => (format!("{}::{name}", self.scope(scope)), 0),
            },
        };
        let own = self.args(args.get(taken..).unwrap_or_default());
        // An impl can implement a `mut?` method by one function per mode, both of its name:
        // the `mut` body is the method at `mut`.
        let mode = match owner {
            BodyOwner::Func(func, None) if func.is_mut_body(db) => Some(BorrowKind::Mut),
            _ => mode,
        };
        format!(
            "{path}{own}{}{}",
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
                    None => self.scope(def.body.scope()),
                };
                format!("{{closure {} in {owner}}}", def.expr.as_u32())
            }
        }
    }
}

/// The name of `ingot`, whose module `top_mod` is, in keys: `core`, `std`, a standalone file's
/// module name, or the ingot's declared name (else the last segment of its base URL), as
/// `--emit metadata` and the spec IR's provenance name it.
pub fn ingot_name(db: &dyn HirAnalysisDb, ingot: Ingot<'_>, top_mod: TopLevelMod<'_>) -> String {
    match ingot.kind(db) {
        IngotKind::Core => "core".to_string(),
        IngotKind::Std => "std".to_string(),
        IngotKind::StandAlone => top_mod.name(db).data(db).to_string(),
        _ => ingot_namespace(db, ingot),
    }
}

/// An ingot's declared name, falling back to the last segment of its base URL.
pub fn ingot_namespace(db: &dyn HirAnalysisDb, ingot: Ingot<'_>) -> String {
    if let Some(name) = ingot
        .config(db)
        .as_ref()
        .and_then(|config| config.metadata.name.as_ref())
    {
        return name.to_string();
    }
    ingot
        .base(db)
        .path()
        .trim_end_matches('/')
        .rsplit('/')
        .next()
        .filter(|segment| !segment.is_empty())
        .unwrap_or("dep")
        .to_string()
}

/// A closure body's receiver, unless it is `self`.
fn receiver_suffix(receiver: FuncParamMode) -> &'static str {
    match receiver {
        FuncParamMode::View => "",
        FuncParamMode::Own => "#own",
        FuncParamMode::Mut => "#mut",
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
