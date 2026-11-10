//! Exports a contract as Lean definitions in the `Fe.EVM` model monad: the
//! checked normalized body of each entry, of each function and constant it
//! reaches, and the storage layout those bodies name.
//!
//! A body the exporter has no model for is reported by the construct it
//! stopped at, never skipped.

use std::fmt::Write;

use common::{compilation::Target as CompilationTarget, ingot::IngotKind};
use cranelift_entity::EntityRef;
use hir::{
    analysis::{
        HirAnalysisDb,
        semantic::{
            BinaryOp, CarrierBase, FailReason, NBlockId, NDataPath, NDataProjection, NEffectArg,
            NEffectArgValue, NExpr, NOperand, NPlace, NPlaceBase, NRootId, NRootKind, NStatement,
            NStatementKind, NSuccessor, NTerminatorKind, NValueDefinition, NValueId,
            NormalizedBody, Overflow, SConst, SemConstScalar, SemConstValue, SemanticInstance,
            SemanticInstanceKey, UnaryOp, drop_elaboration, generated_callee_key,
            identity_semantic_instance_key, normalize_semantic_body,
        },
        ty::{
            AddressSpace,
            adt_def::AdtRef,
            corelib::{resolve_core_trait, resolve_lib_func_path, resolve_lib_type_path},
            registered_root_providers,
            trait_def::TraitInstId,
            trait_resolution::PredicateListId,
            ty_check::{BodyOwner, EffectParamSite, LocalBinding},
            ty_def::{CapabilityKind, TyId, prim_int_bits},
            ty_is_snapshot,
        },
    },
    hir_def::{Contract, FieldParent, IdentId, ItemKind, PatId, expr::CompBinOp},
    semantic::{ProviderSource, storage_field_placements},
};
use rustc_hash::{FxHashMap, FxHashSet};

/// A construct the exporter has no model for.
#[derive(Debug, PartialEq, Eq)]
pub struct Unsupported(pub String);

/// How a body receives what an access parameter reaches.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Carrier {
    /// As a `Place`, which the body reads and writes.
    Place,
    /// As the value there. The body returns the value a `mut` access leaves.
    Value,
    /// Not at all: the target's root object holds no data.
    Erased,
}

/// The value an access or place of a local reaches: a field path below a
/// Lean variable.
struct ValueTarget<'db> {
    var: String,
    ty: TyId<'db>,
    path: Vec<usize>,
}

/// What a place is in the model.
enum Target<'db> {
    Place(String),
    Value(ValueTarget<'db>),
    /// The word in memory at an address.
    Memory(String),
}

/// The Lean definitions of `contract`: its storage layout, the functions
/// and constants its entries reach, then its `init` and `recv` arms.
pub fn export_contract<'db>(
    db: &'db dyn HirAnalysisDb,
    contract: Contract<'db>,
) -> Result<String, Unsupported> {
    export(db, contract, false)
}

/// As [`export_contract`], followed by the runtime entry: one definition per
/// arm that decodes its arguments, runs it and returns its result, and
/// `runtime`, which dispatches on the selector. They call std's decoder and
/// encoder as exported.
pub fn export_contract_with_runtime<'db>(
    db: &'db dyn HirAnalysisDb,
    contract: Contract<'db>,
) -> Result<String, Unsupported> {
    export(db, contract, true)
}

fn export<'db>(
    db: &'db dyn HirAnalysisDb,
    contract: Contract<'db>,
    runtime: bool,
) -> Result<String, Unsupported> {
    let mut entries = vec![("init".to_string(), BodyOwner::ContractInit { contract })];
    for (recv_idx, recv) in contract.recv_views(db).enumerate() {
        entries.extend(recv.arms(db).enumerate().map(|(arm_idx, _)| {
            (
                format!("recv{recv_idx}_arm{arm_idx}"),
                BodyOwner::ContractRecvArm {
                    contract,
                    recv_idx: recv_idx as u32,
                    arm_idx: arm_idx as u32,
                },
            )
        }));
    }
    let mut module = Module {
        db,
        names: FxHashMap::default(),
        taken: entries
            .iter()
            .flat_map(|(name, _)| [name.clone(), format!("{name}_entry")])
            .chain(["runtime".to_string()])
            .collect(),
        nevers: FxHashSet::default(),
        open: Vec::new(),
        layouts: Vec::new(),
        bodies: FxHashMap::default(),
        defs: String::new(),
    };
    for (name, owner) in entries {
        let (body, _) = module.body(&name, identity_semantic_instance_key(db, owner), &[])?;
        writeln!(module.defs, "def {name}{body}").unwrap();
    }
    if runtime {
        module.runtime(contract)?;
    }
    let layouts: String = module
        .layouts
        .iter()
        .map(|(_, def)| format!("{def}\n\n"))
        .collect();
    Ok(layouts + &module.defs)
}

/// The definitions exported so far.
struct Module<'db> {
    db: &'db dyn HirAnalysisDb,
    /// The Lean name of each exported function or constant instance, for
    /// each way it receives its access parameters.
    names: FxHashMap<(SemanticInstanceKey<'db>, Vec<Option<Carrier>>), String>,
    taken: FxHashSet<String>,
    /// The exported functions that never return.
    nevers: FxHashSet<String>,
    /// The instances whose export is under way.
    open: Vec<SemanticInstanceKey<'db>>,
    /// The layout definitions the bodies name, by Lean name.
    layouts: Vec<(String, String)>,
    /// The name of each exported function or constant, by its base name and
    /// the text of its definition after the name.
    bodies: FxHashMap<(String, String), String>,
    defs: String,
}

impl<'db> Module<'db> {
    /// The definition of `key`'s body after its name, which `name` is in
    /// reports, and whether the body never returns. The definitions it uses
    /// are exported first.
    fn body(
        &mut self,
        name: &str,
        key: SemanticInstanceKey<'db>,
        carriers: &[Option<Carrier>],
    ) -> Result<(String, bool), Unsupported> {
        let normalized = normalize_semantic_body(self.db, SemanticInstance::new(self.db, key))
            .map_err(|_| Unsupported(format!("{name}: body does not normalize")))?;
        BodyExporter {
            module: self,
            body: &normalized.body,
            carriers,
            aggregates: FxHashMap::default(),
            slots: Vec::new(),
            outs: Vec::new(),
            result_ty: String::new(),
            joins: FxHashMap::default(),
            joined: FxHashSet::default(),
        }
        .export()
        .map_err(|error| Unsupported(format!("{name}: {}", error.0)))
    }

    /// The patterns that bind the arguments of recv arm `owner`, in the order
    /// the arm takes them; none for any other body.
    fn arg_pats(&self, owner: BodyOwner<'db>) -> Vec<PatId> {
        let BodyOwner::ContractRecvArm {
            contract,
            recv_idx,
            arm_idx,
        } = owner
        else {
            return Vec::new();
        };
        let arm = contract
            .recv(self.db, recv_idx)
            .and_then(|recv| recv.arm(self.db, arm_idx));
        arm.map_or(Vec::new(), |arm| {
            let bindings = arm.arg_bindings(self.db);
            bindings.iter().map(|binding| binding.pat).collect()
        })
    }

    /// Appends the runtime entry: what the compiler synthesizes around the
    /// arms. An arm's entry rejects call value unless the arm is payable,
    /// decodes the message with `core::contracts::decode_runtime_args`, runs
    /// the arm on the fields it binds, and returns through the host's
    /// `return_value`, or with no data. `runtime` reverts on calldata shorter
    /// than a selector or with an unknown one, unless there is a fallback.
    fn runtime(&mut self, contract: Contract<'db>) -> Result<(), Unsupported> {
        let db = self.db;
        let scope = contract.scope();
        let missing = |what: &str| Unsupported(format!("runtime: missing `{what}`"));
        let abi = resolve_lib_type_path(db, scope, "std::abi::Sol")
            .ok_or_else(|| missing("std::abi::Sol"))?;
        let decode = resolve_lib_func_path(db, scope, "core::contracts::decode_runtime_args")
            .ok_or_else(|| missing("core::contracts::decode_runtime_args"))?;
        let host_trait = resolve_core_trait(db, scope, &["contracts", "ContractHost"])
            .ok_or_else(|| missing("core::contracts::ContractHost"))?;
        let generated = |error| Unsupported(format!("runtime: generated call: {error:?}"));
        let mut cases = Vec::new();
        let mut default = "revertEmpty".to_string();
        for (recv_idx, recv) in contract.recv_views(db).enumerate() {
            for (arm_idx, arm) in recv.arms(db).enumerate() {
                let name = format!("recv{recv_idx}_arm{arm_idx}");
                let owner = BodyOwner::ContractRecvArm {
                    contract,
                    recv_idx: recv_idx as u32,
                    arm_idx: arm_idx as u32,
                };
                let site = EffectParamSite::ContractRecvArm {
                    contract,
                    recv_idx: recv_idx as u32,
                    arm_idx: arm_idx as u32,
                };
                let host = registered_root_providers(db, site)
                    .first()
                    .ok_or_else(|| missing("root provider"))?
                    .provider_ty;
                let info = arm.abi_info(db, abi);
                let mut lines = String::new();
                if !arm.arm(db).is_some_and(|arm| arm.is_payable(db)) {
                    lines.push_str("  nonPayable\n");
                }
                // The arguments the arm's body takes, by their place among
                // the arm's bindings.
                let instance = SemanticInstance::new(db, identity_semantic_instance_key(db, owner));
                let normalized = normalize_semantic_body(db, instance)
                    .map_err(|_| Unsupported(format!("{name}: body does not normalize")))?;
                let pats = self.arg_pats(owner);
                let values = normalized.body.values.iter().filter_map(|value| {
                    matches!(value.definition, NValueDefinition::EntryParam { .. })
                        .then(|| arg_position(value.source, &pats))
                });
                let slots = normalized
                    .body
                    .roots
                    .iter()
                    .filter_map(|root| match root.kind {
                        NRootKind::LocalSlot { binding } => arg_position(binding, &pats).map(Some),
                        _ => None,
                    });
                let mut positions = values
                    .chain(slots)
                    .collect::<Option<Vec<_>>>()
                    .ok_or_else(|| Unsupported(format!("{name}: argument bound by no field")))?;
                positions.sort_unstable();
                let mut args = String::new();
                if info.args_ty != TyId::unit(db) {
                    let key = generated_callee_key(
                        db,
                        scope,
                        PredicateListId::empty_list(db),
                        decode,
                        None,
                        &[host, abi, info.args_ty],
                    )
                    .map_err(generated)?;
                    let decode = self.instance(key, vec![Some(Carrier::Erased)])?;
                    writeln!(lines, "  let message ← {decode}").unwrap();
                    let fields = self.fields(info.args_ty)?;
                    let bindings = arm.arg_bindings(db);
                    for position in positions {
                        let (steps, _) = &fields[bindings[position].tuple_index as usize];
                        write!(args, " message.{}", steps.join(".")).unwrap();
                    }
                }
                match info.ret_ty {
                    None => {
                        writeln!(lines, "  let _ ← {name}{args}").unwrap();
                        lines.push_str("  returnData (UInt256.ofNat 0) (UInt256.ofNat 0)\n");
                    }
                    Some(ret_ty) => {
                        let inst = TraitInstId::new_simple(db, host_trait, vec![host]);
                        let method = IdentId::new(db, "return_value".to_string());
                        let func = host_trait
                            .method_defs(db)
                            .get(&method)
                            .copied()
                            .ok_or_else(|| missing("ContractHost::return_value"))?;
                        let key = generated_callee_key(
                            db,
                            scope,
                            PredicateListId::empty_list(db),
                            func,
                            Some(inst),
                            &[abi, ret_ty],
                        )
                        .map_err(generated)?;
                        let return_value = self.instance(key, vec![Some(Carrier::Erased), None])?;
                        writeln!(lines, "  let result ← {name}{args}").unwrap();
                        writeln!(lines, "  {return_value} result").unwrap();
                    }
                }
                writeln!(self.defs, "def {name}_entry : FeM Empty := do\n{lines}").unwrap();
                match info.selector_value {
                    _ if info.is_fallback => default = format!("{name}_entry"),
                    Some(selector) => cases.push((selector, name)),
                    None => return Err(Unsupported(format!("{name}: no selector"))),
                }
            }
        }
        cases.sort_unstable();
        let mut dispatch = String::new();
        for (selector, name) in cases {
            write!(
                dispatch,
                "if selector = UInt256.ofNat {selector:#010x} then\n      {name}_entry\n    else "
            )
            .unwrap();
        }
        let last = if dispatch.is_empty() {
            default.clone()
        } else {
            format!("\n      {default}")
        };
        writeln!(
            self.defs,
            "def runtime : FeM Empty := do\n  let size ← calldatasize\n  \
             if size.toNat < 4 then\n    {default}\n  else\n    \
             let selector ← calldataSelector\n    {dispatch}{last}"
        )
        .unwrap();
        Ok(())
    }

    /// The Lean name of the function or constant instance `key` that
    /// receives its access parameters as `carriers`, exported on first use.
    fn instance(
        &mut self,
        key: SemanticInstanceKey<'db>,
        carriers: Vec<Option<Carrier>>,
    ) -> Result<String, Unsupported> {
        if let Some(name) = self.names.get(&(key, carriers.clone())) {
            return Ok(name.clone());
        }
        let db = self.db;
        let base = match key.owner(db) {
            BodyOwner::Func(func, mode) => {
                let name = func.name(db).to_opt().map(|name| name.data(db).as_str());
                let name = name.unwrap_or("fn");
                if func.is_extern(db) {
                    return Err(Unsupported(format!("extern function `{name}`")));
                }
                match mode {
                    Some(mode) => format!("{name}_{mode:?}").to_lowercase(),
                    None => name.to_string(),
                }
            }
            BodyOwner::Const(constant) => constant
                .name(db)
                .to_opt()
                .map_or("const".to_string(), |name| name.data(db).to_string()),
            // An associated constant's value is an anonymous body of its impl.
            BodyOwner::AnonConstBody { body, .. } => {
                let consts = match body.scope().parent_item(db) {
                    Some(ItemKind::Impl(item)) => item.hir_consts(db),
                    Some(ItemKind::ImplTrait(item)) => item.hir_consts(db),
                    _ => &[],
                };
                consts
                    .iter()
                    .find(|constant| constant.value.to_opt() == Some(body))
                    .and_then(|constant| constant.name.to_opt())
                    .map_or("const".to_string(), |name| name.data(db).to_string())
            }
            owner => return Err(Unsupported(format!("call to {owner:?}"))),
        };
        if self.open.contains(&key) {
            return Err(Unsupported(format!("recursion through `{base}`")));
        }
        let args = key.subst(db).generic_args(db).iter();
        let base = args.fold(base, |name, arg| {
            format!("{name}_{}", lean_ident(arg.pretty_print(db)))
        });
        self.open.push(key);
        let (body, never) = self.body(&base, key, &carriers)?;
        self.open.pop();
        // Instances with one definition share it.
        let name = match self.bodies.get(&(base.clone(), body.clone())) {
            Some(name) => name.clone(),
            None => {
                let name = (0..)
                    .map(|n| match n {
                        0 => base.clone(),
                        n => format!("{base}_{n}"),
                    })
                    .find(|name| !self.taken.contains(name))
                    .unwrap();
                self.taken.insert(name.clone());
                let name = lean_name(&name);
                writeln!(self.defs, "def {name}{body}").unwrap();
                self.bodies.insert((base, body), name.clone());
                name
            }
        };
        if never {
            self.nevers.insert(name.clone());
        }
        self.names.insert((key, carriers), name.clone());
        Ok(name)
    }

    /// The Lean name of `definition`, a layout definition named `name`.
    fn layout(&mut self, name: String, definition: impl FnOnce(&str) -> String) -> String {
        if !self.layouts.iter().any(|(known, _)| *known == name) {
            let definition = definition(&name);
            self.layouts.push((name.clone(), definition));
        }
        name
    }

    /// The Lean projections and the type of each field of struct or tuple
    /// `ty`. A tuple is a right-nested product, and a tuple of one field is
    /// that field.
    fn fields(&self, ty: TyId<'db>) -> Result<Vec<(Vec<String>, TyId<'db>)>, Unsupported> {
        let db = self.db;
        let tys = ty.field_types(db);
        // An address is its word, so its one field is the address itself.
        if ty.pretty_print(db) == "Address" {
            return Ok(tys.into_iter().map(|ty| (Vec::new(), ty)).collect());
        }
        if ty.is_tuple(db) {
            let last = tys.len().saturating_sub(1);
            let steps = |index: usize| {
                let mut steps = vec!["snd".to_string(); index];
                if index < last {
                    steps.push("fst".to_string());
                }
                steps
            };
            return Ok(tys
                .into_iter()
                .enumerate()
                .map(|(i, ty)| (steps(i), ty))
                .collect());
        }
        let Some(AdtRef::Struct(item)) = ty.adt_def(db).map(|adt| adt.adt_ref(db)) else {
            return Err(Unsupported(format!("type {}", ty.pretty_print(db))));
        };
        let names = FieldParent::Struct(item).fields(db).map(|field| {
            let name = field.name(db).map(|name| name.data(db).as_str());
            vec![lean_name(name.unwrap_or("field"))]
        });
        Ok(names.zip(tys).collect())
    }

    /// The width of `ty`, and whether it is signed, if it is an integer type.
    fn int_bits(&self, ty: TyId<'db>) -> Option<(usize, bool)> {
        let prim = ty.as_prim(self.db)?;
        let target = self.db.compilation_settings().target(self.db);
        Some((prim_int_bits(prim, target)?, prim.is_signed_int()))
    }

    /// The width of `ty` if it is an unsigned integer type.
    fn uint_bits(&self, ty: TyId<'db>) -> Option<usize> {
        self.int_bits(ty)
            .and_then(|(bits, signed)| (!signed).then_some(bits))
    }

    /// The Lean term for `value` of integer type `from` cast to `to`: the
    /// word, cut to the width of a narrower unsigned type.
    fn cast(&self, value: &str, from: TyId<'db>, to: TyId<'db>) -> Result<String, Unsupported> {
        if from.is_bool(self.db) && self.uint_bits(to).is_some() {
            return Ok(format!("Bool.toUInt256 {value}"));
        }
        match (self.int_bits(from), self.int_bits(to)) {
            // A word reinterpreted at its own width.
            (Some((256, _)), Some((256, _))) => Ok(value.to_string()),
            (Some((from, false)), Some((to, false))) if from <= to => Ok(value.to_string()),
            (Some((_, false)), Some((to, false))) => Ok(format!("truncate {to} {value}")),
            _ => Err(Unsupported(format!(
                "cast from {} to {}",
                from.pretty_print(self.db),
                to.pretty_print(self.db)
            ))),
        }
    }

    /// The Lean type of a value of `ty`: a word, a product for a tuple, or
    /// the structure a struct is defined as.
    fn value_ty(&mut self, ty: TyId<'db>) -> Result<String, Unsupported> {
        // An integer of any width is a word: below its bound if unsigned,
        // sign-extended if signed. A pointer into memory is its address.
        let memory_ptr = matches!(ty.as_ptr(self.db), Some((_, AddressSpace::Memory)));
        if self.int_bits(ty).is_some() || memory_ptr {
            return Ok("UInt256".to_string());
        }
        if ty.is_never(self.db) {
            return Ok("Empty".to_string());
        }
        let pretty = ty.pretty_print(self.db);
        match pretty.as_str() {
            // An address is its word.
            "Address" => return Ok("UInt256".to_string()),
            "bool" => return Ok("Bool".to_string()),
            "()" => return Ok("Unit".to_string()),
            _ => {}
        }
        let fields = self.fields(ty)?;
        if ty.is_tuple(self.db) {
            let tys: Result<Vec<_>, _> = fields.iter().map(|(_, ty)| self.value_ty(*ty)).collect();
            return Ok(match tys?.as_slice() {
                [ty] => ty.clone(),
                tys => format!("({})", tys.join(" × ")),
            });
        }
        let mut lines = String::new();
        for (name, field) in fields {
            writeln!(lines, "  {} : {}", name[0], self.value_ty(field)?).unwrap();
        }
        Ok(self.layout(lean_ident(pretty), |name| {
            format!("structure {name} where\n{lines}deriving Inhabited")
        }))
    }

    /// The layout of struct or tuple `ty`: the slot of each field below a
    /// place of `ty` in storage.
    fn struct_layout(&mut self, ty: TyId<'db>) -> Result<String, Unsupported> {
        let pretty = ty.pretty_print(self.db);
        let placements = storage_field_placements(self.db, ty)
            .ok_or_else(|| Unsupported(format!("field of {pretty}")))?;
        if placements
            .iter()
            .any(|placement| placement.partial_lane().is_some())
        {
            return Err(Unsupported(format!("packed fields of {pretty}")));
        }
        let slots: Vec<_> = placements
            .iter()
            .map(|placement| placement.slot.to_string())
            .collect();
        Ok(
            self.layout(format!("Layout.{}", lean_ident(pretty)), |name| {
                format!(
                    "def {name} : Layout :=\n  {{ storage := [{}] }}",
                    slots.join(", ")
                )
            }),
        )
    }
}

struct BodyExporter<'a, 'db> {
    module: &'a mut Module<'db>,
    body: &'a NormalizedBody<'db>,
    /// How the body receives each access parameter, by parameter.
    carriers: &'a [Option<Carrier>],
    /// The type and fields of each aggregate the body builds, printed where
    /// it is used.
    aggregates: FxHashMap<NValueId, (TyId<'db>, Vec<NValueId>)>,
    /// The slots of locals and of parameters, each a mutable Lean variable.
    slots: Vec<NRootId>,
    /// The `mut` access parameters received as values, which the body
    /// returns after its result.
    outs: Vec<NValueId>,
    /// The Lean type of the value the body returns.
    result_ty: String,
    /// The blocks printed as local functions, under the block that
    /// immediately dominates each, in the order they are defined.
    joins: FxHashMap<NBlockId, Vec<NBlockId>>,
    joined: FxHashSet<NBlockId>,
}

impl<'db> BodyExporter<'_, 'db> {
    /// The body's definition after its name (parameters, type and value),
    /// and whether the body never returns.
    fn export(mut self) -> Result<(String, bool), Unsupported> {
        let db = self.module.db;
        // A parameter is an entry value, or a slot if the body needs its
        // address: the slot starts out holding the argument. A recv arm's
        // are in the order of the fields its pattern binds.
        let owner = self.body.owner.key(db).owner(db);
        let arm = matches!(owner, BodyOwner::ContractRecvArm { .. });
        let pats = self.module.arg_pats(owner);
        let mut params = Vec::new();
        for (index, value) in self.body.values.iter().enumerate() {
            let NValueDefinition::EntryParam { param } = value.definition else {
                continue;
            };
            let param = if arm {
                arg_position(value.source, &pats)
                    .ok_or_else(|| Unsupported("argument bound by no field".to_string()))?
                    as u32
            } else {
                param
            };
            let carrier = self.carriers.get(param as usize).copied().flatten();
            let ty = match (value.ty.as_capability(db), carrier) {
                (Some(_), Some(Carrier::Erased)) => continue,
                (Some((kind, inner)), Some(Carrier::Value)) => {
                    if kind == CapabilityKind::Mut {
                        self.outs.push(NValueId::new(index));
                    }
                    self.module.value_ty(inner)?
                }
                _ => self.lean_ty(value.ty)?,
            };
            params.push((param, format!(" (v{index} : {ty})")));
        }
        let mut state = String::new();
        for (index, root) in self.body.roots.iter().enumerate() {
            let ty = match root.kind {
                NRootKind::LocalSlot { .. } | NRootKind::ParamPlace { .. } => {
                    self.module.value_ty(root.ty)?
                }
                _ => continue,
            };
            self.slots.push(NRootId::new(index));
            let param = match root.kind {
                NRootKind::ParamPlace { param } => Some(param),
                NRootKind::LocalSlot { binding } => {
                    arg_position(binding, &pats).map(|position| position as u32)
                }
                _ => None,
            };
            if let Some(param) = param {
                params.push((param, format!(" (r{index} : {ty})")));
                writeln!(state, "  let mut r{index} := r{index}").unwrap();
            } else {
                writeln!(state, "  let mut r{index} : {ty} := default").unwrap();
            }
        }
        params.sort_by_key(|(param, _)| *param);
        let params: String = params.into_iter().map(|(_, text)| text).collect();
        for out in &self.outs {
            writeln!(state, "  let mut {0} := {0}", value_name(*out)).unwrap();
        }
        self.result_ty = self.returned_ty()?;
        let never = self.result_ty == "Empty";
        for out in self.outs.clone() {
            let (_, inner) = self.ty(out).as_capability(db).unwrap();
            let ty = self.module.value_ty(inner)?;
            self.result_ty = format!("{} × {ty}", self.result_ty);
        }
        if !self.outs.is_empty() {
            self.result_ty = format!("({})", self.result_ty);
        }
        self.find_joins()?;
        let mut out = format!("{params} : FeM {} := do\n{state}", self.result_ty);
        self.block(self.body.entry, 1, &mut out)?;
        Ok((out, never))
    }

    /// The mutable variables of the body: its local slots, then the values
    /// it returns for its `mut` access parameters.
    fn state(&self) -> Vec<String> {
        let slots = self.slots.iter().map(|root| format!("r{}", root.index()));
        slots
            .chain(self.outs.iter().copied().map(value_name))
            .collect()
    }

    /// The type every `Return` of the body gives.
    fn returned_ty(&mut self) -> Result<String, Unsupported> {
        let returned = self
            .body
            .blocks
            .iter()
            .find_map(|block| match block.terminator.kind {
                NTerminatorKind::Return(operand) => Some(operand),
                _ => None,
            });
        // A projection gives what it yields: the value its access reaches.
        let yielded = self
            .body
            .blocks
            .iter()
            .find_map(|block| match &block.terminator.kind {
                NTerminatorKind::Yield { value, .. } => Some(self.ty(value.value)),
                _ => None,
            });
        if let Some(yielded) = yielded {
            let inner = yielded
                .as_capability(self.module.db)
                .map_or(yielded, |(_, ty)| ty);
            return self.module.value_ty(inner);
        }
        match returned {
            Some(Some(operand)) => self.lean_ty(self.ty(operand.value)),
            Some(None) => Ok("Unit".to_string()),
            // A body that never returns gives no value.
            None => Ok("Empty".to_string()),
        }
    }

    /// Finds the join blocks: those more than one edge enters, and those
    /// that take parameters. Each is printed once, as a local function of
    /// the block that immediately dominates it. An edge to an earlier block
    /// is a loop.
    fn find_joins(&mut self) -> Result<(), Unsupported> {
        let order = self.body.reverse_postorder();
        let position: FxHashMap<_, _> = order.iter().enumerate().map(|(i, b)| (*b, i)).collect();
        let mut preds: FxHashMap<NBlockId, Vec<NBlockId>> = FxHashMap::default();
        for &block in &order {
            for successor in self.body.blocks[block.index()].terminator.kind.successors() {
                if position[&successor.block] <= position[&block] {
                    return Err(Unsupported("loop".to_string()));
                }
                preds.entry(successor.block).or_default().push(block);
            }
        }
        let mut idom: FxHashMap<NBlockId, NBlockId> = FxHashMap::default();
        for &block in &order[1..] {
            let dominator = preds[&block]
                .iter()
                .copied()
                .reduce(|mut a, mut b| {
                    while a != b {
                        if position[&a] > position[&b] {
                            a = idom[&a];
                        } else {
                            b = idom[&b];
                        }
                    }
                    a
                })
                .unwrap();
            idom.insert(block, dominator);
            if preds[&block].len() > 1 || !self.body.blocks[block.index()].params.is_empty() {
                self.joined.insert(block);
                // A later join can be the target of an earlier one, so it is
                // defined first.
                self.joins.entry(dominator).or_default().insert(0, block);
            }
        }
        Ok(())
    }

    /// Prints `block`: its statements, the joins it dominates, and where it
    /// goes, with each block only it enters inline.
    fn block(
        &mut self,
        block: NBlockId,
        depth: usize,
        out: &mut String,
    ) -> Result<(), Unsupported> {
        let data = &self.body.blocks[block.index()];
        let pad = "  ".repeat(depth);
        for statement in &data.statements {
            for line in self
                .statement(statement)?
                .iter()
                .flat_map(|text| text.lines())
            {
                writeln!(out, "{pad}{line}").unwrap();
            }
        }
        // A join takes the body's mutable variables before its parameters,
        // since a local function cannot assign its definer's.
        let state = self.state();
        for join in self.joins.get(&block).cloned().unwrap_or_default() {
            let params = &self.body.blocks[join.index()].params;
            let mut ty = String::new();
            for root in self.slots.clone() {
                let slot = self.body.roots[root.index()].ty;
                write!(ty, "{} → ", self.module.value_ty(slot)?).unwrap();
            }
            for out in self.outs.clone() {
                let (_, inner) = self.ty(out).as_capability(self.module.db).unwrap();
                write!(ty, "{} → ", self.module.value_ty(inner)?).unwrap();
            }
            for param in params {
                write!(ty, "{} → ", self.lean_ty(self.ty(*param))?).unwrap();
            }
            let names: Vec<_> = state
                .iter()
                .cloned()
                .chain(params.iter().copied().map(value_name))
                .collect();
            let binder = if names.is_empty() {
                String::new()
            } else {
                format!("fun {} => ", names.join(" "))
            };
            writeln!(
                out,
                "{pad}let b{} : {ty}FeM {} := {binder}do",
                join.index(),
                self.result_ty
            )
            .unwrap();
            for var in &state {
                writeln!(out, "{pad}  let mut {var} := {var}").unwrap();
            }
            self.block(join, depth + 1, out)?;
        }
        match &data.terminator.kind {
            NTerminatorKind::Return(returned) => {
                let result = match returned {
                    Some(returned) => self.operand(returned)?,
                    None => "()".to_string(),
                };
                let outs: String = self
                    .outs
                    .iter()
                    .map(|out| format!(", {}", value_name(*out)))
                    .collect();
                if outs.is_empty() {
                    writeln!(out, "{pad}return {result}").unwrap();
                } else {
                    writeln!(out, "{pad}return ({result}{outs})").unwrap();
                }
            }
            NTerminatorKind::Goto(target) => self.jump(target, depth, out)?,
            NTerminatorKind::Branch {
                cond,
                then_target,
                else_target,
            } => {
                writeln!(out, "{pad}if {} then", value_name(cond.value)).unwrap();
                self.jump(then_target, depth + 1, out)?;
                writeln!(out, "{pad}else").unwrap();
                self.jump(else_target, depth + 1, out)?;
            }
            NTerminatorKind::Fail(FailReason::Assert {
                message: Some(message),
            }) => writeln!(
                out,
                "{pad}M.revert (assertData {:?})",
                message.data(self.module.db)
            )
            .unwrap(),
            NTerminatorKind::Fail(FailReason::Assert { message: None }) => {
                writeln!(out, "{pad}M.panic 0x01").unwrap()
            }
            // A projection that grants read access to a value and then only
            // ends its own sessions returns the value.
            NTerminatorKind::Yield { value, resume } => {
                let db = self.module.db;
                let ty = self.ty(value.value);
                let read_only = !matches!(ty.as_capability(db), Some((CapabilityKind::Mut, _)));
                if !read_only || !self.outs.is_empty() || !self.slide_is_empty(resume.block) {
                    return Err(Unsupported(
                        "projection that grants `mut` access or has a slide".to_string(),
                    ));
                }
                let granted = if self.is_access(ty) {
                    let target = self.carrier_value(value.value)?;
                    self.read(&target)?
                } else {
                    self.operand(value)?
                };
                writeln!(out, "{pad}return {granted}").unwrap();
            }
            // Control stops at a call that never returns, whose result has
            // no values.
            NTerminatorKind::Unreachable => {
                let never =
                    data.statements
                        .iter()
                        .rev()
                        .find_map(|statement| match &statement.kind {
                            NStatementKind::Define { result, .. }
                                if self.ty(*result).is_never(self.module.db) =>
                            {
                                Some(*result)
                            }
                            _ => None,
                        });
                let never = never
                    .ok_or_else(|| Unsupported("unreachable code after no call".to_string()))?;
                writeln!(out, "{pad}nomatch {}", value_name(never)).unwrap();
            }
            other => return Err(Unsupported(format!("terminator {other:?}"))),
        }
        Ok(())
    }

    /// Whether the slide from `block` does nothing: it ends sessions,
    /// destroys nothing and returns.
    fn slide_is_empty(&self, block: NBlockId) -> bool {
        let data = &self.body.blocks[block.index()];
        let drops = drop_elaboration(self.module.db, self.body.owner);
        let inert = data
            .statements
            .iter()
            .all(|statement| match statement.kind {
                NStatementKind::End { .. } => true,
                NStatementKind::Drop { .. } => {
                    drops.is_some_and(|drops| drops.parts(statement.id).is_empty())
                }
                _ => false,
            });
        inert
            && match &data.terminator.kind {
                NTerminatorKind::Return(None) => true,
                NTerminatorKind::Goto(next) => {
                    next.args.is_empty() && self.slide_is_empty(next.block)
                }
                _ => false,
            }
    }

    /// Prints the step to `target`: a call of its local function if it is a
    /// join, and the block itself otherwise.
    fn jump(
        &mut self,
        target: &NSuccessor,
        depth: usize,
        out: &mut String,
    ) -> Result<(), Unsupported> {
        if !self.joined.contains(&target.block) {
            return self.block(target.block, depth, out);
        }
        let mut args: String = self.state().iter().map(|var| format!(" {var}")).collect();
        for arg in &target.args {
            write!(args, " {}", self.operand(arg)?).unwrap();
        }
        writeln!(out, "{}b{}{args}", "  ".repeat(depth), target.block.index()).unwrap();
        Ok(())
    }

    /// The lines of `statement`, if it prints any.
    fn statement(&mut self, statement: &NStatement<'db>) -> Result<Option<String>, Unsupported> {
        Ok(match &statement.kind {
            NStatementKind::Define { result, expr } => {
                let name = value_name(*result);
                match expr {
                    // A view or borrow of the `Evm` root carries no data.
                    NExpr::MakeView { place, .. } | NExpr::Borrow { place, .. }
                        if self.is_root_object(place) =>
                    {
                        None
                    }
                    // An access to a local is read and written through the
                    // local's variable, where it is used; one to the root
                    // object reaches no data.
                    NExpr::Forward { .. } | NExpr::MakeView { .. } | NExpr::Borrow { .. }
                        if self.is_access(self.ty(*result))
                            && self.carrier(*result)? != Carrier::Place =>
                    {
                        None
                    }
                    NExpr::Forward { src } => Some(format!("let {name} := {}", self.operand(src)?)),
                    NExpr::Const(SConst::Ref(constant)) => Some(format!(
                        "let {name} ← {}",
                        self.module
                            .instance(constant.instance(self.module.db), Vec::new())?
                    )),
                    NExpr::Const(constant) => {
                        Some(format!("let {name} := {}", self.lean_const(constant)?))
                    }
                    NExpr::Load { place, .. } => Some(match self.target(place)? {
                        Target::Place(lean) => {
                            self.word(place)?;
                            format!("let {name} ← load {lean}")
                        }
                        Target::Value(target) => format!("let {name} := {}", self.read(&target)?),
                        Target::Memory(ptr) => format!("let {name} ← mload {ptr}"),
                    }),
                    NExpr::PointerCast { value, .. } => {
                        Some(format!("let {name} := {}", self.operand(value)?))
                    }
                    NExpr::ProjectValue { value, path } => {
                        let target = ValueTarget {
                            var: self.operand(value)?,
                            ty: self.ty(value.value),
                            path: field_path(&path.0)?,
                        };
                        Some(format!("let {name} := {}", self.read(&target)?))
                    }
                    NExpr::Binary { op, lhs, rhs } => {
                        let ty = self.ty(lhs.value);
                        let int = self.module.int_bits(ty).ok_or_else(|| {
                            let ty = ty.pretty_print(self.module.db);
                            Unsupported(format!("operator {op:?} on {ty}"))
                        })?;
                        let (lhs, rhs) = (self.operand(lhs)?, self.operand(rhs)?);
                        let rhs = binary(*op, int, lhs, rhs).map_err(|error| {
                            let ty = ty.pretty_print(self.module.db);
                            Unsupported(format!("{} on {ty}", error.0))
                        })?;
                        Some(format!("let {name} {rhs}"))
                    }
                    NExpr::Unary { op, value } => {
                        let ty = self.ty(value.value);
                        let value = self.operand(value)?;
                        Some(match (op, self.module.uint_bits(ty)) {
                            (UnaryOp::BitNot, Some(256)) => format!("let {name} := ~~~{value}"),
                            (UnaryOp::Not, None) if ty.is_bool(self.module.db) => {
                                format!("let {name} := !{value}")
                            }
                            _ => return Err(Unsupported(format!("operator {op:?}"))),
                        })
                    }
                    NExpr::ScalarCast { value, to } => {
                        let from = self.ty(value.value);
                        let value = self.operand(value)?;
                        Some(format!(
                            "let {name} := {}",
                            self.module.cast(&value, from, *to)?
                        ))
                    }
                    NExpr::MakeView { place, .. } | NExpr::Borrow { place, .. } => {
                        Some(
                            match (self.target(place)?, self.is_access(self.ty(*result))) {
                                (Target::Place(lean), true) => format!("let {name} := {lean}"),
                                // A view of a snapshot type is the value now there.
                                (Target::Value(target), false) => {
                                    format!("let {name} := {}", self.read(&target)?)
                                }
                                _ => {
                                    return Err(Unsupported(
                                        "view of a one-word place".to_string(),
                                    ));
                                }
                            },
                        )
                    }
                    NExpr::AggregateMake { ty, fields } => {
                        let fields = fields.iter().map(|field| field.value).collect();
                        self.aggregates.insert(*result, (*ty, fields));
                        None
                    }
                    NExpr::Call {
                        callee,
                        args,
                        effect_args,
                        ..
                    } => {
                        // A callee's instance names the provider behind each of
                        // its effects, so a provider passed whole needs no
                        // argument.
                        let whole_provider = |arg: &NEffectArg<'db>| {
                            matches!(&arg.arg, NEffectArgValue::Place(place)
                            if place.path.is_empty()
                                && matches!(place.base, NPlaceBase::Root(root)
                                    if matches!(
                                        self.body.roots[root.index()].kind,
                                        NRootKind::Provider { .. }
                                    )))
                        };
                        if !effect_args.iter().all(whole_provider) {
                            return Err(Unsupported(
                                "effect argument below a provider".to_string(),
                            ));
                        }
                        let call = self.call(*result, callee.key, args)?;
                        Some(self.after_never(*result, call)?)
                    }
                    other => return Err(Unsupported(format!("expression {other:?}"))),
                }
            }
            NStatementKind::Store { destination, value } => {
                let value = self.operand(value)?;
                Some(match self.target(destination)? {
                    Target::Place(lean) => {
                        self.word(destination)?;
                        format!("store {lean} {value}")
                    }
                    Target::Value(target) => self.write(&target, &value)?,
                    Target::Memory(ptr) => format!("mstore {ptr} {value}"),
                })
            }
            NStatementKind::End { .. } => None,
            // A drop that destroys nothing: its target holds no value with
            // a destructor here.
            NStatementKind::Drop { .. }
                if drop_elaboration(self.module.db, self.body.owner)
                    .is_some_and(|drops| drops.parts(statement.id).is_empty()) =>
            {
                None
            }
            other => return Err(Unsupported(format!("statement {other:?}"))),
        })
    }

    /// A call: to a std function the prelude models, by name, or to an
    /// exported function. An access to a local passes the value there, and
    /// a `mut` one stores the value the callee returns for it.
    fn call(
        &mut self,
        result: NValueId,
        callee: SemanticInstanceKey<'db>,
        args: &[NOperand],
    ) -> Result<String, Unsupported> {
        let db = self.module.db;
        let name = &value_name(result);
        let BodyOwner::Func(func, _) = callee.owner(db) else {
            return Err(Unsupported("call to a non-function".to_string()));
        };
        let in_std = matches!(
            func.scope().ingot(db).kind(db),
            IngotKind::Std | IngotKind::Core
        );
        let callee_name = func.name(db).to_opt().map(|name| name.data(db).as_str());
        let evm = db.compilation_settings().target(db) == CompilationTarget::Evm;
        match (in_std, callee_name, args) {
            (true, Some("caller"), [_evm]) => return Ok(format!("let {name} ← caller")),
            (true, Some("zero"), []) => return Ok(format!("let {name} := UInt256.ofNat 0")),
            (true, Some("emit"), [_evm, event]) => {
                let (ty, fields) = self
                    .aggregates
                    .get(&event.value)
                    .ok_or_else(|| Unsupported("emit of a value not built in place".to_string()))?;
                let fields = fields
                    .iter()
                    .map(|field| format!(" {}", value_name(*field)));
                return Ok(format!(
                    "let {name} ← emit{}{}",
                    ty.pretty_print(db),
                    fields.collect::<String>()
                ));
            }
            // The externs the prelude defines.
            (true, Some("calldataload"), [offset]) if func.is_extern(db) => {
                return Ok(format!(
                    "let {name} ← calldataload {}",
                    self.operand(offset)?
                ));
            }
            (true, Some("calldatasize"), []) if func.is_extern(db) => {
                return Ok(format!("let {name} ← calldatasize"));
            }
            (true, Some("revert_empty"), []) if func.is_extern(db) => {
                return Ok(format!("let {name} ← revertEmpty\nnever"));
            }
            (true, Some("alloc_raw"), [size]) if func.is_extern(db) => {
                return Ok(format!("let {name} ← alloc {}", self.operand(size)?));
            }
            (true, Some("dealloc_raw"), [_]) if func.is_extern(db) && evm => {
                return Ok(format!("let {name} := ()"));
            }
            (true, Some("offset_bytes"), [ptr, offset]) if func.is_extern(db) => {
                let (ptr, offset) = (self.operand(ptr)?, self.operand(offset)?);
                return Ok(format!("let {name} := {ptr} + {offset}"));
            }
            (true, Some("zero_mem"), [ptr, len]) if func.is_extern(db) => {
                let (ptr, len) = (self.operand(ptr)?, self.operand(len)?);
                return Ok(format!("let {name} ← zeroMem {ptr} {len}"));
            }
            (true, Some("revert"), [ptr, len]) if func.is_extern(db) => {
                let (ptr, len) = (self.operand(ptr)?, self.operand(len)?);
                return Ok(format!("let {name} ← revertData {ptr} {len}\nnever"));
            }
            (true, Some("return_data"), [ptr, len]) if func.is_extern(db) => {
                let (ptr, len) = (self.operand(ptr)?, self.operand(len)?);
                return Ok(format!("let {name} ← returnData {ptr} {len}\nnever"));
            }
            (true, Some("panic_code"), [code]) if func.is_extern(db) => {
                return Ok(format!(
                    "let {name} ← panicCode {}\nnever",
                    self.operand(code)?
                ));
            }
            // EVM memory is big-endian, and `usize` is a word there.
            (true, Some("to_be" | "from_be"), [value]) if func.is_extern(db) && evm => {
                return Ok(format!("let {name} := {}", self.operand(value)?));
            }
            (true, Some("__bitnot_usize"), [value]) if func.is_extern(db) && evm => {
                return Ok(format!("let {name} := ~~~{}", self.operand(value)?));
            }
            (true, Some("__bitcast"), [value]) if func.is_extern(db) => {
                let from = self.ty(value.value);
                let value = self.operand(value)?;
                let cast = self.module.cast(&value, from, self.ty(result))?;
                return Ok(format!("let {name} := {cast}"));
            }
            _ => {}
        }
        let mut carriers = Vec::new();
        let mut lean_args = String::new();
        let mut outs = Vec::new();
        for arg in args {
            let ty = self.ty(arg.value);
            let carrier = if self.is_access(ty) {
                Some(self.carrier(arg.value)?)
            } else {
                None
            };
            carriers.push(carrier);
            if carrier == Some(Carrier::Erased) {
                continue;
            }
            if carrier != Some(Carrier::Value) {
                write!(lean_args, " {}", self.operand(arg)?).unwrap();
                continue;
            }
            let target = self.carrier_value(arg.value)?;
            write!(lean_args, " {}", self.read(&target)?).unwrap();
            if matches!(ty.as_capability(db), Some((CapabilityKind::Mut, _))) {
                outs.push(target);
            }
        }
        let callee = self.module.instance(callee, carriers)?;
        if self.module.nevers.contains(&callee) {
            return Ok(format!("let {name} ← {callee}{lean_args}\nnever"));
        }
        if outs.is_empty() {
            return Ok(format!("let {name} ← {callee}{lean_args}"));
        }
        let returned: String = (0..outs.len()).map(|n| format!(", {name}_{n}")).collect();
        let mut lines = format!("let ({name}{returned}) ← {callee}{lean_args}");
        for (n, out) in outs.iter().enumerate() {
            write!(lines, "\n{}", self.write(out, &format!("{name}_{n}"))?).unwrap();
        }
        Ok(lines)
    }

    /// The lines of a call, which end in a `never` line if the callee never
    /// returns. The result of such a call has no values, yet the body may
    /// give it the type of the expression it stands in; the name is then
    /// rebound at that type.
    fn after_never(&mut self, result: NValueId, call: String) -> Result<String, Unsupported> {
        let Some(call) = call.strip_suffix("\nnever") else {
            return Ok(call);
        };
        let ty = self.ty(result);
        if ty.is_never(self.module.db) {
            return Ok(call.to_string());
        }
        let name = value_name(result);
        Ok(format!(
            "{call}\nlet {name} : {} := nomatch {name}",
            self.lean_ty(ty)?
        ))
    }

    /// `operand` as a Lean term: its variable, or the structure an aggregate
    /// builds.
    fn operand(&mut self, operand: &NOperand) -> Result<String, Unsupported> {
        let Some((ty, fields)) = self.aggregates.get(&operand.value).cloned() else {
            return Ok(value_name(operand.value));
        };
        let fields: Vec<_> = fields.iter().copied().map(value_name).collect();
        // A type that is its one field, a tuple of one or an address.
        if let [(steps, _)] = self.module.fields(ty)?.as_slice()
            && steps.is_empty()
        {
            return Ok(fields[0].clone());
        }
        if ty.is_tuple(self.module.db) {
            return Ok(format!("({})", fields.join(", ")));
        }
        let fields: String = fields.iter().map(|field| format!(" {field}")).collect();
        Ok(format!("({}.mk{fields})", self.module.value_ty(ty)?))
    }

    /// Whether a value of `ty` is an access to a place. A view of a snapshot
    /// type is the value.
    fn is_access(&self, ty: TyId<'db>) -> bool {
        let db = self.module.db;
        let owner = self.body.owner;
        let scope = owner.key(db).owner(db).scope();
        match ty.as_capability(db) {
            Some((CapabilityKind::View, inner)) => {
                !ty_is_snapshot(db, scope, inner, owner.assumptions(db))
            }
            Some(_) => true,
            None => false,
        }
    }

    /// How the model holds the access `value`: as a `Place` if it reaches
    /// contract state, and by the value there if it reaches a local.
    fn carrier(&self, value: NValueId) -> Result<Carrier, Unsupported> {
        let target = self
            .body
            .carrier_target(self.module.db, value)
            .ok_or_else(|| Unsupported("access without one target".to_string()))?;
        match target.base {
            CarrierBase::Root(root) => match &self.body.roots[root.index()].kind {
                NRootKind::Provider { binding }
                    if matches!(binding.source, ProviderSource::RootProvider { .. }) =>
                {
                    Ok(Carrier::Erased)
                }
                NRootKind::Provider { .. } => Ok(Carrier::Place),
                NRootKind::LocalSlot { .. } | NRootKind::ParamPlace { .. } => Ok(Carrier::Value),
                kind => Err(Unsupported(format!("access to root {kind:?}"))),
            },
            CarrierBase::Param { param } => self
                .carriers
                .get(param as usize)
                .copied()
                .flatten()
                .ok_or_else(|| Unsupported("access parameter of an entry".to_string())),
            // A read-only grant is the value the projection returned.
            CarrierBase::Grant { session, .. }
                if !matches!(
                    self.ty(session).as_capability(self.module.db),
                    Some((CapabilityKind::Mut, _))
                ) =>
            {
                Ok(Carrier::Value)
            }
            base => Err(Unsupported(format!("access to {base:?}"))),
        }
    }

    /// The local value the access `value` reaches.
    fn carrier_value(&self, value: NValueId) -> Result<ValueTarget<'db>, Unsupported> {
        let db = self.module.db;
        let target = self
            .body
            .carrier_target(db, value)
            .ok_or_else(|| Unsupported("access without one target".to_string()))?;
        let (var, ty) = match target.base {
            CarrierBase::Root(root) => (
                format!("r{}", root.index()),
                self.body.roots[root.index()].ty,
            ),
            CarrierBase::Param { param } => {
                let (index, value) = self
                    .body
                    .values
                    .iter()
                    .enumerate()
                    .find(|(_, value)| value.definition == NValueDefinition::EntryParam { param })
                    .ok_or_else(|| Unsupported("access to an unknown parameter".to_string()))?;
                let (_, inner) = value.ty.as_capability(db).unwrap();
                (format!("v{index}"), inner)
            }
            CarrierBase::Grant { session, component } => {
                let (_, inner) = self.ty(session).as_capability(db).unwrap();
                let mut path = field_path(&component)?;
                path.extend(field_path(&target.path)?);
                return Ok(ValueTarget {
                    var: value_name(session),
                    ty: inner,
                    path,
                });
            }
            base => return Err(Unsupported(format!("access to {base:?}"))),
        };
        Ok(ValueTarget {
            var,
            ty,
            path: field_path(&target.path)?,
        })
    }

    /// What `place` is in the model: a `Place` in contract state, or a
    /// value below a local's variable.
    fn target(&mut self, place: &NPlace<'db>) -> Result<Target<'db>, Unsupported> {
        let local = match place.base {
            NPlaceBase::Root(root) => {
                let data = &self.body.roots[root.index()];
                matches!(
                    data.kind,
                    NRootKind::LocalSlot { .. } | NRootKind::ParamPlace { .. }
                )
                .then(|| ValueTarget {
                    var: format!("r{}", root.index()),
                    ty: data.ty,
                    path: Vec::new(),
                })
            }
            // A view of a snapshot type is the value itself.
            NPlaceBase::CapabilityTarget { carrier } if !self.is_access(self.ty(carrier)) => {
                let (_, ty) = self.ty(carrier).as_capability(self.module.db).unwrap();
                Some(ValueTarget {
                    var: value_name(carrier),
                    ty,
                    path: Vec::new(),
                })
            }
            NPlaceBase::CapabilityTarget { carrier }
                if self.carrier(carrier)? == Carrier::Value =>
            {
                Some(self.carrier_value(carrier)?)
            }
            _ => None,
        };
        if let NPlaceBase::Deref { pointer } = place.base {
            let word = self
                .module
                .int_bits(place.ty)
                .is_some_and(|(bits, _)| bits == 256);
            let memory = matches!(
                self.ty(pointer).as_ptr(self.module.db),
                Some((_, AddressSpace::Memory))
            );
            if !word || !memory || !place.path.is_empty() {
                return Err(Unsupported(format!(
                    "place of {} through {}",
                    place.ty.pretty_print(self.module.db),
                    self.ty(pointer).pretty_print(self.module.db)
                )));
            }
            return Ok(Target::Memory(value_name(pointer)));
        }
        let Some(mut local) = local else {
            return Ok(Target::Place(self.place(place)?));
        };
        local.path.extend(field_path(&place.path)?);
        Ok(Target::Value(local))
    }

    /// The Lean term for the value at `target`.
    fn read(&self, target: &ValueTarget<'db>) -> Result<String, Unsupported> {
        let mut lean = target.var.clone();
        let mut ty = target.ty;
        for &index in &target.path {
            let (steps, field) = self.module.fields(ty)?.swap_remove(index);
            for step in steps {
                write!(lean, ".{step}").unwrap();
            }
            ty = field;
        }
        Ok(lean)
    }

    /// The assignment that leaves `value` at `target`.
    fn write(&self, target: &ValueTarget<'db>, value: &str) -> Result<String, Unsupported> {
        let mut prefix = target.var.clone();
        let mut ty = target.ty;
        let mut open = String::new();
        let mut depth = 0;
        for &index in &target.path {
            let (steps, field) = self.module.fields(ty)?.swap_remove(index);
            for step in steps {
                write!(open, "{{ {prefix} with {step} := ").unwrap();
                write!(prefix, ".{step}").unwrap();
                depth += 1;
            }
            ty = field;
        }
        Ok(format!(
            "{} := {open}{value}{}",
            target.var,
            " }".repeat(depth)
        ))
    }

    fn ty(&self, value: NValueId) -> TyId<'db> {
        self.body.values[value.index()].ty
    }

    fn is_root_object(&self, place: &NPlace<'db>) -> bool {
        let NPlaceBase::Root(root) = place.base else {
            return false;
        };
        matches!(
            &self.body.roots[root.index()].kind,
            NRootKind::Provider { binding }
                if matches!(binding.source, ProviderSource::RootProvider { .. })
        )
    }

    /// The Lean type of a value of `ty`: a `Place` for an access, and the
    /// type of the value otherwise.
    fn lean_ty(&mut self, ty: TyId<'db>) -> Result<String, Unsupported> {
        match (self.is_access(ty), ty.as_capability(self.module.db)) {
            (true, _) => Ok("Place".to_string()),
            (false, Some((_, inner))) => self.module.value_ty(inner),
            (false, None) => self.module.value_ty(ty),
        }
    }

    fn lean_const(&self, constant: &SConst<'db>) -> Result<String, Unsupported> {
        let SConst::Value(value) = constant else {
            return Err(Unsupported(format!("constant {constant:?}")));
        };
        match value.value().value(self.module.db) {
            SemConstValue::Unit => Ok("()".to_string()),
            SemConstValue::Scalar {
                value: SemConstScalar::Int { value },
                ..
            } => Ok(format!("(UInt256.ofNat {value})")),
            SemConstValue::Scalar {
                value: SemConstScalar::Bool(value),
                ..
            } => Ok(value.to_string()),
            other => Err(Unsupported(format!("constant {other:?}"))),
        }
    }

    /// Checks that `place` holds one word, as `load` and `store` take.
    fn word(&mut self, place: &NPlace<'db>) -> Result<(), Unsupported> {
        match self.lean_ty(place.ty)?.as_str() {
            "UInt256" => Ok(()),
            _ => Err(Unsupported("place that is not one word".to_string())),
        }
    }

    /// The Lean place `place` names: its base, then a layout step for each
    /// struct field and an entry step for each `StorageMap` key.
    fn place(&mut self, place: &NPlace<'db>) -> Result<String, Unsupported> {
        let db = self.module.db;
        let mut lean = match place.base {
            NPlaceBase::Root(root) => self.root(root)?,
            NPlaceBase::CapabilityTarget { carrier } => value_name(carrier),
            base => return Err(Unsupported(format!("place base {base:?}"))),
        };
        for (len, projection) in place.path.iter().enumerate() {
            let ty = self
                .body
                .place_prefix_ty(db, place, len)
                .ok_or_else(|| Unsupported("untyped place".to_string()))?;
            lean = match projection {
                NDataProjection::Field(index) => {
                    format!(
                        "({lean}.field {} {})",
                        self.module.struct_layout(ty)?,
                        index.0
                    )
                }
                NDataProjection::Entry(key) => {
                    let map = ty.adt_def(db).is_some_and(|adt| {
                        adt.adt_ref(db)
                            .name(db)
                            .is_some_and(|name| name.data(db) == "StorageMap")
                    }) && ty
                        .ingot(db)
                        .is_some_and(|ingot| ingot.kind(db) == IngotKind::Std);
                    if !map || self.lean_ty(self.ty(*key))? != "UInt256" {
                        return Err(Unsupported(format!("entry of {}", ty.pretty_print(db))));
                    }
                    format!("({lean}.entry {})", value_name(*key))
                }
                other => return Err(Unsupported(format!("place step {other:?}"))),
            };
        }
        Ok(lean)
    }

    /// The place of the contract field `root` provides.
    fn root(&mut self, root: NRootId) -> Result<String, Unsupported> {
        let db = self.module.db;
        let root = &self.body.roots[root.index()];
        let NRootKind::Provider { binding } = &root.kind else {
            return Err(Unsupported(format!("root {:?}", root.kind)));
        };
        let ProviderSource::ContractField { field } = binding.source else {
            return Err(Unsupported(
                "provider that is not a contract field".to_string(),
            ));
        };
        let layout = field.contract.storage_layout(db);
        let layout = layout
            .values()
            .find(|layout| layout.field == field)
            .ok_or_else(|| Unsupported("contract field without a layout".to_string()))?;
        let name = layout.name.data(db);
        if layout.address_space != AddressSpace::Storage
            || layout.lane.is_some_and(|lane| lane.partial)
        {
            return Err(Unsupported(format!("layout of contract field `{name}`")));
        }
        let slot = layout.slot_offset;
        Ok(self.module.layout(format!("Field.{name}"), |name| {
            format!("def {name} : Place :=\n  .storage (UInt256.ofNat {slot})")
        }))
    }
}

/// `text` with each run of characters a Lean identifier cannot hold as `_`.
fn lean_ident(text: impl AsRef<str>) -> String {
    text.as_ref()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|part| !part.is_empty())
        .collect::<Vec<_>>()
        .join("_")
}

fn value_name(value: NValueId) -> String {
    format!("v{}", value.index())
}

/// The place of `binding` among the patterns `pats`, if it is one of them.
fn arg_position(binding: Option<LocalBinding<'_>>, pats: &[PatId]) -> Option<usize> {
    let Some(LocalBinding::Local { pat, .. }) = binding else {
        return None;
    };
    pats.iter().position(|bound| *bound == pat)
}

/// The struct fields `path` steps through.
fn field_path(path: &NDataPath) -> Result<Vec<usize>, Unsupported> {
    path.iter()
        .map(|projection| match projection {
            NDataProjection::Field(index) => Ok(usize::from(index.0)),
            other => Err(Unsupported(format!("step {other:?} in a local"))),
        })
        .collect()
}

/// `name` as a Lean identifier, quoted if Lean reserves it.
fn lean_name(name: &str) -> String {
    const RESERVED: &[&str] = &[
        "at", "by", "do", "else", "end", "from", "fun", "have", "if", "in", "let", "match", "open",
        "show", "then", "this", "where", "with",
    ];
    if RESERVED.contains(&name) {
        format!("«{name}»")
    } else {
        name.to_string()
    }
}

/// The right-hand side of `let v` for a built-in operation on integers of
/// `bits` bits, signed or not, each a word. An operation whose result
/// depends on the width is printed only for a full word.
fn binary(
    op: BinaryOp,
    (bits, signed): (usize, bool),
    lhs: String,
    rhs: String,
) -> Result<String, Unsupported> {
    let word = bits == 256 && !signed;
    Ok(match op {
        BinaryOp::Add(Overflow::Checked) if word => format!("← checkedAdd {lhs} {rhs}"),
        BinaryOp::Add(Overflow::Wrapping) if word => format!(":= {lhs} + {rhs}"),
        BinaryOp::Sub(Overflow::Checked) if word => format!("← checkedSub {lhs} {rhs}"),
        BinaryOp::Sub(Overflow::Wrapping) if word => format!(":= {lhs} - {rhs}"),
        BinaryOp::Mul(Overflow::Checked) if word => format!("← checkedMul {lhs} {rhs}"),
        BinaryOp::Mul(Overflow::Wrapping) if word => format!(":= {lhs} * {rhs}"),
        BinaryOp::Shl if word => format!(":= {lhs} <<< {rhs}"),
        // Unsigned division, remainder and right shift stay below the
        // operand's bound.
        BinaryOp::Div(_) if !signed => format!("← checkedDiv {lhs} {rhs}"),
        BinaryOp::Rem if !signed => format!("← checkedRem {lhs} {rhs}"),
        BinaryOp::Shr if !signed => format!(":= {lhs} >>> {rhs}"),
        BinaryOp::BitAnd => format!(":= {lhs} &&& {rhs}"),
        BinaryOp::BitOr => format!(":= {lhs} ||| {rhs}"),
        BinaryOp::BitXor => format!(":= {lhs} ^^^ {rhs}"),
        BinaryOp::Comp(CompBinOp::Eq) => format!(":= decide ({lhs} = {rhs})"),
        BinaryOp::Comp(CompBinOp::NotEq) => format!(":= decide ({lhs} ≠ {rhs})"),
        BinaryOp::Comp(CompBinOp::GtEq) if !signed => {
            format!(":= decide ({rhs}.toNat ≤ {lhs}.toNat)")
        }
        BinaryOp::Comp(CompBinOp::LtEq) if !signed => {
            format!(":= decide ({lhs}.toNat ≤ {rhs}.toNat)")
        }
        BinaryOp::Comp(CompBinOp::Lt) if !signed => {
            format!(":= decide ({lhs}.toNat < {rhs}.toNat)")
        }
        BinaryOp::Comp(CompBinOp::Gt) if !signed => {
            format!(":= decide ({rhs}.toNat < {lhs}.toNat)")
        }
        // Sign-extended words compare as the EVM's signed comparisons do.
        BinaryOp::Comp(CompBinOp::Lt) => format!(":= UInt256.sltBool {lhs} {rhs}"),
        BinaryOp::Comp(CompBinOp::Gt) => format!(":= UInt256.sgtBool {lhs} {rhs}"),
        BinaryOp::Comp(CompBinOp::GtEq) => format!(":= !UInt256.sltBool {lhs} {rhs}"),
        BinaryOp::Comp(CompBinOp::LtEq) => format!(":= !UInt256.sgtBool {lhs} {rhs}"),
        other => return Err(Unsupported(format!("operator {other:?}"))),
    })
}
