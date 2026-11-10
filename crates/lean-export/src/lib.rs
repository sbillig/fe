//! Exports a contract as Lean definitions in the `Fe.EVM` model monad: the
//! checked normalized body of each entry, of each function and constant it
//! reaches, and the storage layout those bodies name.
//!
//! A body the exporter has no model for is reported by the construct it
//! stopped at, never skipped.

use std::fmt::Write;

use common::{
    compilation::Target as CompilationTarget, ingot::IngotKind, layout::StorageFieldPlacement,
};
use cranelift_entity::EntityRef;
use hir::{
    analysis::{
        HirAnalysisDb,
        semantic::{
            BinaryOp, CarrierBase, DropStep, EvalOutcome, FailReason, NBlockId, NDataPath,
            NDataProjection, NDropTarget, NEffectArg, NEffectArgValue, NExpr, NIndex, NOperand,
            NPlace, NPlaceBase, NRootId, NRootKind, NStatement, NStatementKind, NSuccessor,
            NTerminatorKind, NValueDefinition, NValueId, NormalizedBody, Overflow, SConst,
            SemConstId, SemConstScalar, SemConstValue, SemanticInstance, SemanticInstanceKey,
            UnaryOp, core_method_callee_key, drop_elaboration, eval_const_ref,
            generated_callee_key, identity_semantic_instance_key, normalize_semantic_body,
            projection_component_spaces, runtime_size_bytes,
        },
        ty::{
            AddressSpace,
            adt_def::{AdtField, AdtRef, instantiate_adt_field_shape},
            corelib::{
                NumericExternIntrinsic, RuntimeBuiltinFuncKind, SaturatingArithmetic,
                contract_metadata_kind, core_numeric_const_intrinsic, lib_func_matches,
                resolve_core_trait, resolve_lib_func_path, resolve_lib_type_path,
                runtime_builtin_func_kind,
            },
            drop_glue::{adt_drop_impl, implements_drop, ty_has_drop_glue},
            normalize::normalize_ty,
            registered_root_providers,
            trait_def::TraitInstId,
            trait_resolution::PredicateListId,
            ty_check::{BodyOwner, EffectParamSite, LocalBinding, ReturnProjectionStep},
            ty_def::{BorrowKind, CapabilityKind, TyId, prim_int_bits},
            ty_is_snapshot,
        },
    },
    hir_def::{
        Contract, EnumVariant, FieldParent, IdentId, ItemKind, PatId,
        expr::{ArithBinOp, BinOp, CompBinOp},
    },
    semantic::{ProviderSource, storage_field_placements, storage_lane_bytes, storage_slot_span},
};
use num_bigint::BigUint;
use num_traits::Signed;
use rustc_hash::{FxHashMap, FxHashSet};

/// The names exported code calls the prelude and the entries by, which no
/// exported function takes.
const PRELUDE: &[&str] = &[
    "alloc",
    "assertData",
    "balance",
    "basefee",
    "blobbasefee",
    "blobhash",
    "blockNumber",
    "blockhash",
    "calldataSelector",
    "calldatacopy",
    "calldataload",
    "calldatasize",
    "caller",
    "callvalue",
    "chainid",
    "checkIndex",
    "checkedAdd",
    "checkedDiv",
    "checkedMul",
    "checkedRem",
    "checkedSub",
    "codecopy",
    "codesize",
    "coinbase",
    "create2",
    "decide",
    "default",
    "delegatecall",
    "extcodecopy",
    "extcodehash",
    "extcodesize",
    "gas",
    "gaslimit",
    "gasprice",
    "init_entry",
    "keccak256",
    "load",
    "logData",
    "mcopy",
    "mload",
    "mstore",
    "mstore8",
    "mstoreBytes",
    "nonPayable",
    "origin",
    "packed",
    "panicCode",
    "prevrandao",
    "pure",
    "readPaddedWord",
    "returnData",
    "returndatacopy",
    "returndatasize",
    "revertData",
    "revertEmpty",
    "runtime",
    "selfAddress",
    "selfbalance",
    "sload",
    "sstore",
    "staticcall",
    "stopCall",
    "store",
    "timestamp",
    "tload",
    "trap",
    "truncate",
    "tstore",
    "zeroMem",
];

/// The types exported code names from Lean and the prelude, which no exported
/// type takes.
const TYPES: &[&str] = &[
    "Bool", "Empty", "FeM", "IntTy", "Lane", "Layout", "Nat", "Place", "Space", "UInt256", "Unit",
    "Vector",
];

/// The longest run of statements a definition starts with before it is cut
/// into pieces, and the statements of each piece. Lean compiles a `do` block
/// in time that grows with the fourth power of its binds, and elaborates it
/// at a depth that grows with them.
const LONGEST_RUN: usize = 40;
const PIECE: usize = 24;

/// What stands for a definition's own name in its text until it has one.
const SELF: char = '\u{1}';

/// What separates a definition's text from each piece cut from it.
const CUT: char = '\u{2}';

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

/// A step from a value to a part of it.
#[derive(Clone, Copy)]
enum Step {
    /// A field of a struct or tuple.
    Field(usize),
    /// A field of a variant, of an enum value that is that variant.
    Variant { variant: usize, field: usize },
    /// An element of an array.
    Index(NIndex),
}

/// The value an access or place of a local reaches: a path below a Lean
/// variable.
#[derive(Clone)]
struct ValueTarget<'db> {
    var: String,
    ty: TyId<'db>,
    path: Vec<Step>,
}

/// The bytes of a slot that hold a packed value, counted from its low end.
#[derive(Clone, Copy)]
struct Lane {
    off: u32,
    width: u32,
    /// Whether other values lie in the slot.
    shared: bool,
    /// Whether the lane is read and written apart from the rest of its slot,
    /// so that no reference can point at it.
    partial: bool,
    /// For an entry a collection packs into a lane, the body's record of
    /// the lane's bit offset and of the collection's codec, which reads and
    /// writes it; the byte offset and width are then unused.
    codec: Option<usize>,
}

impl Lane {
    fn lean(self) -> String {
        format!("⟨{}, {}, {}⟩", self.off, self.width, self.shared)
    }
}

/// An effect of a function that its callers pass an argument for.
#[derive(Clone, Copy)]
struct Effect<'db> {
    provider: u32,
    ty: TyId<'db>,
    /// Whether it lies in contract state unless a caller says otherwise.
    state: bool,
    mutable: bool,
}

/// How a body receives what its access parameters and its effects reach: by
/// parameter, and by provider. One function is exported once for each way.
#[derive(Clone, Default, PartialEq, Eq, Hash)]
struct Carriers {
    params: Vec<Option<Carrier>>,
    effects: Vec<(u32, Carrier)>,
}

impl From<Vec<Option<Carrier>>> for Carriers {
    fn from(params: Vec<Option<Carrier>>) -> Self {
        Self {
            params,
            effects: Vec::new(),
        }
    }
}

/// Where a call leaves the value its callee returns for a `mut` argument it
/// took by value.
#[derive(Clone)]
enum Out<'db> {
    /// Below a local's variable.
    Value(ValueTarget<'db>),
    /// In a packed lane of contract state, which has no address to pass.
    Lane(TyId<'db>, String, Lane),
}

/// What a place is in the model.
enum Target<'db> {
    /// A place in contract state, and the lane of its slot an access to it
    /// by this name reads and writes, if it is one.
    Place(String, Option<Lane>),
    Value(ValueTarget<'db>),
    /// A value in memory at an address.
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

/// As [`export_contract`], followed by the contract's entries. The runtime
/// entry is one definition per arm that decodes its arguments, runs it and
/// returns its result, and `runtime`, which dispatches on the selector. The
/// deployment entry, `init_entry`, decodes the constructor arguments and runs
/// `init`. They call std's decoder and encoder as exported.
pub fn export_contract_with_entries<'db>(
    db: &'db dyn HirAnalysisDb,
    contract: Contract<'db>,
) -> Result<String, Unsupported> {
    export(db, contract, true)
}

fn export<'db>(
    db: &'db dyn HirAnalysisDb,
    contract: Contract<'db>,
    with_entries: bool,
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
        root: SemanticInstance::new(
            db,
            identity_semantic_instance_key(db, BodyOwner::ContractInit { contract }),
        ),
        names: FxHashMap::default(),
        taken: entries
            .iter()
            .flat_map(|(name, _)| [name.clone(), format!("{name}_entry")])
            .chain(PRELUDE.iter().map(|name| name.to_string()))
            .collect(),
        nevers: FxHashSet::default(),
        lenses: FxHashSet::default(),
        lens: false,
        open: Vec::new(),
        layouts: Vec::new(),
        bodies: FxHashMap::default(),
        defs: String::new(),
    };
    for (name, owner) in entries {
        let key = identity_semantic_instance_key(db, owner);
        let (body, _) = module.body(&name, key, &Carriers::default())?;
        module.define(&name, &body);
    }
    if with_entries {
        module.runtime(contract)?;
        module.init_entry(contract)?;
    }
    let layouts: String = module
        .layouts
        .iter()
        .map(|(_, def)| format!("{def}\n\n"))
        .collect();
    Ok(layouts + &module.defs)
}

/// The definitions exported so far.
#[derive(Clone)]
struct Module<'db> {
    db: &'db dyn HirAnalysisDb,
    /// The contract's `init`, which types are normalized in.
    root: SemanticInstance<'db>,
    /// The Lean name of each exported function or constant instance, for
    /// each way it receives its access parameters.
    names: FxHashMap<(SemanticInstanceKey<'db>, Carriers), String>,
    taken: FxHashSet<String>,
    /// The exported functions that never return.
    nevers: FxHashSet<String>,
    /// The exported projections that return the value they grant with their
    /// slide: a function from the value the caller leaves there to the
    /// values the projection's `mut` parameters end with.
    lenses: FxHashSet<String>,
    /// Whether the body exported last is such a projection.
    lens: bool,
    /// The instances whose export is under way.
    open: Vec<(SemanticInstanceKey<'db>, Carriers)>,
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
        carriers: &Carriers,
    ) -> Result<(String, bool), Unsupported> {
        let normalized = normalize_semantic_body(self.db, SemanticInstance::new(self.db, key))
            .map_err(|_| Unsupported(format!("{name}: body does not normalize")))?;
        BodyExporter {
            module: self,
            body: &normalized.body,
            carriers,
            slots: Vec::new(),
            outs: Vec::new(),
            effects: Vec::new(),
            result_ty: String::new(),
            binders: Vec::new(),
            joins: FxHashMap::default(),
            joined: FxHashSet::default(),
            loops: FxHashSet::default(),
            helds: Vec::new(),
            slide: FxHashSet::default(),
            slide_ty: None,
            sessions: Vec::new(),
            slides: FxHashMap::default(),
            never: None,
            yielding: false,
            codecs: Vec::new(),
            located: Vec::new(),
            entries: 0,
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
                let values = normalized
                    .body
                    .values
                    .iter()
                    .filter(|value| matches!(value.definition, NValueDefinition::EntryParam { .. }))
                    .map(|value| arg_position(value.source, &pats));
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
                    let decode = self.instance(key, vec![Some(Carrier::Erased)].into())?;
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
                        let carriers = vec![Some(Carrier::Erased), None];
                        let return_value = self.instance(key, carriers.into())?;
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
        let tests: Vec<_> = cases
            .iter()
            .map(|(selector, name)| {
                format!(
                    "if selector = UInt256.ofNat {selector:#010x} then\n      {name}_entry\n    else"
                )
            })
            .collect();
        let dispatch = tests.join(" ");
        let last = if tests.is_empty() {
            default.clone()
        } else {
            format!("\n      {default}")
        };
        writeln!(
            self.defs,
            "def runtime : FeM Empty := do\n  let size ← calldatasize\n  \
             if size.toNat < 4 then\n    {default}\n  else\n    \
             let selector ← calldataSelector\n    {dispatch}{last}\n"
        )
        .unwrap();
        Ok(())
    }

    /// Appends the deployment entry: what the compiler synthesizes around
    /// `init`. It rejects call value unless `init` is payable, copies the
    /// constructor arguments, the bytes after the creation code, into fresh
    /// memory, decodes them with `Decode::decode_payload` at the tuple of
    /// `init`'s parameter types, and runs `init`. Returning the runtime code
    /// is left to the caller.
    fn init_entry(&mut self, contract: Contract<'db>) -> Result<(), Unsupported> {
        let db = self.db;
        let scope = contract.scope();
        let missing = |what: &str| Unsupported(format!("init_entry: missing `{what}`"));
        let generated = |error| Unsupported(format!("init_entry: generated call: {error:?}"));
        if contract.code_address_space_slot_count(db) != 0 {
            return Err(Unsupported("init_entry: immutable fields".to_string()));
        }
        let mut lines = String::new();
        if !contract.init(db).is_some_and(|init| init.is_payable(db)) {
            lines.push_str("  nonPayable\n");
        }
        let args_ty = contract.init_args_ty(db);
        let mut args = String::new();
        if args_ty != TyId::unit(db) {
            let lib_ty =
                |path: &str| resolve_lib_type_path(db, scope, path).ok_or_else(|| missing(path));
            let abi = lib_ty("std::abi::Sol")?;
            let decoder_ty = lib_ty("std::abi::sol::SolDecoder")?;
            let span_ty = lib_ty("core::ptr::MemSpan")?;
            let method =
                |segments: &[&str], inst_args: Vec<TyId<'db>>, method: &str, own: &[TyId<'db>]| {
                    let def = resolve_core_trait(db, scope, segments)
                        .ok_or_else(|| missing(&segments.join("::")))?;
                    let func = def
                        .method_defs(db)
                        .get(&IdentId::new(db, method.to_string()))
                        .copied()
                        .ok_or_else(|| missing(method))?;
                    let inst = TraitInstId::new_simple(db, def, inst_args);
                    generated_callee_key(
                        db,
                        scope,
                        PredicateListId::empty_list(db),
                        func,
                        Some(inst),
                        own,
                    )
                    .map_err(generated)
                };
            let decoder_new = method(&["abi", "Abi"], vec![abi], "decoder_new", &[])?;
            let decode = method(
                &["abi", "Decode"],
                vec![args_ty, abi],
                "decode_payload",
                &[decoder_ty, span_ty],
            )?;
            let decoder_new = self.instance(decoder_new, Carriers::default())?;
            // The input is a view: the value itself if a span is a snapshot.
            let input = (!ty_is_snapshot(db, scope, span_ty, PredicateListId::empty_list(db)))
                .then_some(Carrier::Value);
            let decode = self.instance(decode, vec![Some(Carrier::Value), input].into())?;
            let span = self.value_ty(span_ty)?;
            lines.push_str("  let size := UInt256.ofNat args.size\n");
            lines.push_str("  let tail ← alloc size\n  mstoreBytes tail args\n");
            writeln!(lines, "  let decoder ← {decoder_new}").unwrap();
            writeln!(
                lines,
                "  let (decoded, _) ← {decode} decoder ({span}.mk tail size)"
            )
            .unwrap();
            // `init` takes the fields its body has parameters for.
            let owner = BodyOwner::ContractInit { contract };
            let instance = SemanticInstance::new(db, identity_semantic_instance_key(db, owner));
            let normalized = normalize_semantic_body(db, instance)
                .map_err(|_| Unsupported("init: body does not normalize".to_string()))?;
            let values = normalized
                .body
                .values
                .iter()
                .filter_map(|value| match value.definition {
                    NValueDefinition::EntryParam { param } => Some(param),
                    _ => None,
                });
            let slots = normalized
                .body
                .roots
                .iter()
                .filter_map(|root| match root.kind {
                    NRootKind::ParamPlace { param } => Some(param),
                    _ => None,
                });
            let mut params: Vec<_> = values.chain(slots).collect();
            params.sort_unstable();
            let fields = self.fields(args_ty)?;
            for param in params {
                let (steps, _) = &fields[param as usize];
                let steps: String = steps.iter().map(|step| format!(".{step}")).collect();
                write!(args, " decoded{steps}").unwrap();
            }
        }
        writeln!(
            self.defs,
            "def init_entry (args : ByteArray) : FeM Unit := do\n{lines}  init{args}\n"
        )
        .unwrap();
        Ok(())
    }

    /// The Lean name of the function or constant instance `key` that
    /// receives its access parameters as `carriers`, exported on first use.
    fn instance(
        &mut self,
        key: SemanticInstanceKey<'db>,
        carriers: Carriers,
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
                // The compiler implements these for each contract from where
                // its init code lies in the artifact.
                if contract_metadata_kind(db, func.trait_method_def(db).unwrap_or(func)).is_some() {
                    return Err(Unsupported(format!("`{name}` of a contract's init code")));
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
        let args = key.subst(db).generic_args(db).iter();
        let base = args.fold(base, |name, arg| {
            format!("{name}_{}", lean_ident(arg.pretty_print(db)))
        });
        // A function that calls itself is named before its body is done.
        let instance = (key, carriers.clone());
        if self.open.last() == Some(&instance) {
            let name = self.fresh(&base);
            self.names.insert(instance, name.clone());
            return Ok(name);
        }
        if self.open.iter().any(|(open, _)| *open == key) {
            return Err(Unsupported(format!("recursion through `{base}`")));
        }
        self.open.push(instance.clone());
        let (body, never) = self.body(&base, key, &carriers)?;
        let lens = std::mem::take(&mut self.lens);
        self.open.pop();
        let name = match (
            self.names.get(&instance),
            self.bodies.get(&(base.clone(), body.clone())),
        ) {
            // The body called itself by this name: it is the least fixed
            // point of its definition, which diverges where the calls do
            // not end.
            (Some(_), _) if body.contains(CUT) => {
                return Err(Unsupported(format!("recursion through long `{base}`")));
            }
            (Some(name), _) => {
                let name = name.clone();
                self.define(&name, &format!("{body}partial_fixpoint\n"));
                name
            }
            // Instances with one definition share it.
            (None, Some(name)) => name.clone(),
            (None, None) => {
                let name = self.fresh(&base);
                self.define(&name, &body);
                self.bodies.insert((base, body), name.clone());
                name
            }
        };
        if never {
            self.nevers.insert(name.clone());
        }
        if lens {
            self.lenses.insert(name.clone());
        }
        self.names.insert((key, carriers), name.clone());
        Ok(name)
    }

    /// A Lean name no definition has yet: `base`, or `base` with a number.
    fn fresh(&mut self, base: &str) -> String {
        let name = (0..)
            .map(|n| match n {
                0 => base.to_string(),
                n => format!("{base}_{n}"),
            })
            .find(|name| !self.taken.contains(name))
            .unwrap();
        self.taken.insert(name.clone());
        lean_name(&name)
    }

    /// The effects of `key` its callers pass an argument for: those its
    /// instance does not name a contract field or the root object for. Each
    /// by provider, with its type, whether it lies in contract state, where
    /// the argument is its place, and otherwise whether it is `mut`, where
    /// the argument is its value and the callee returns the value it leaves.
    fn effect_params(&self, key: SemanticInstanceKey<'db>) -> Vec<Effect<'db>> {
        let Ok(normalized) = normalize_semantic_body(self.db, SemanticInstance::new(self.db, key))
        else {
            return Vec::new();
        };
        let mut effects: Vec<_> = normalized
            .body
            .roots
            .iter()
            .filter_map(|root| match &root.kind {
                NRootKind::Provider { binding }
                    if matches!(binding.source, ProviderSource::UsesParam { .. }) =>
                {
                    Some(Effect {
                        provider: binding.provider_idx,
                        ty: root.ty,
                        state: matches!(
                            root.address_space,
                            AddressSpace::Storage | AddressSpace::Transient
                        ),
                        mutable: binding.is_mut,
                    })
                }
                _ => None,
            })
            .collect();
        effects.sort_unstable_by_key(|effect| effect.provider);
        effects
    }

    /// Appends the definition of `name` with the text `body` gives after the
    /// name: the pieces cut from it, last first, each named after it, then
    /// the definition.
    fn define(&mut self, name: &str, body: &str) {
        let mut texts = body.split(CUT);
        let main = texts.next().unwrap();
        let piece = name.trim_matches(['«', '»']);
        for text in texts.rev() {
            writeln!(self.defs, "{}", text.replace(SELF, piece)).unwrap();
        }
        writeln!(self.defs, "def {name}{}", main.replace(SELF, piece)).unwrap();
    }

    /// The Lean name of `definition`, a type or layout definition named
    /// `name`, or `name` with a number if a different definition has it.
    fn layout(&mut self, name: String, definition: impl Fn(&str) -> String) -> String {
        for n in 0.. {
            let name = match n {
                0 => name.clone(),
                n => format!("{name}_{n}"),
            };
            let definition = definition(&name);
            match self.layouts.iter().find(|(known, _)| *known == name) {
                Some((_, known)) if *known == definition => return name,
                Some(_) => {}
                // A type the exported text names from Lean or the prelude
                // keeps its name.
                None if TYPES.contains(&name.as_str()) => {}
                None => {
                    self.layouts.push((name.clone(), definition));
                    return name;
                }
            }
        }
        unreachable!()
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

    /// The variants of enum `ty`: the Lean constructor and the field types
    /// of each.
    fn variants(&self, ty: TyId<'db>) -> Option<Vec<(String, Vec<TyId<'db>>)>> {
        let db = self.db;
        let adt = ty.adt_def(db)?;
        let AdtRef::Enum(item) = adt.adt_ref(db) else {
            return None;
        };
        let args = ty.generic_args(db);
        let variant = |(variant, fields): (usize, &AdtField<'db>)| {
            let name = EnumVariant::new(item, variant).name(db);
            let field = |field| {
                let ty = instantiate_adt_field_shape(db, adt, variant, field, args);
                normalize_ty(db, ty, adt.scope(db), PredicateListId::empty_list(db))
            };
            (
                lean_name(name.unwrap_or("variant")),
                (0..fields.num_types()).map(field).collect(),
            )
        };
        Some(adt.fields(db).iter().enumerate().map(variant).collect())
    }

    /// The Lean projections from a value of `ty` to the part `step` reaches,
    /// and the part's type.
    fn part(&self, ty: TyId<'db>, step: Step) -> Result<(Vec<String>, TyId<'db>), Unsupported> {
        match step {
            Step::Field(index) => Ok(self.fields(ty)?.swap_remove(index)),
            Step::Variant { variant, field } => {
                let (name, mut fields) = self
                    .variants(ty)
                    .ok_or_else(|| Unsupported(format!("type {}", ty.pretty_print(self.db))))?
                    .swap_remove(variant);
                Ok((vec![format!("{name}_{field}")], fields.swap_remove(field)))
            }
            // The one name of an element is its index.
            Step::Index(index) => {
                let (_, args) = ty.decompose_ty_app(self.db);
                let elem = args
                    .first()
                    .filter(|_| ty.is_array(self.db))
                    .ok_or_else(|| Unsupported(format!("index of {}", ty.pretty_print(self.db))))?;
                let index = match index {
                    NIndex::Const(index) => index.to_string(),
                    NIndex::Value(value) => format!("{}.toNat", value_name(value)),
                };
                Ok((vec![index], *elem))
            }
        }
    }

    /// The Lean term for a value of struct, tuple or array type `ty` with
    /// the parts `parts`. A type that is its one field, a tuple of one or an
    /// address, is that field.
    fn aggregate(&mut self, ty: TyId<'db>, parts: &[String]) -> Result<String, Unsupported> {
        if ty.is_array(self.db) {
            return Ok(format!("#v[{}]", parts.join(", ")));
        }
        if let [(steps, _)] = self.fields(ty)?.as_slice()
            && steps.is_empty()
        {
            return Ok(parts[0].clone());
        }
        if ty.is_tuple(self.db) {
            return Ok(format!("({})", parts.join(", ")));
        }
        let parts: String = parts.iter().map(|part| format!(" {part}")).collect();
        Ok(format!("({}.mk{parts})", self.value_ty(ty)?))
    }

    /// The Lean term for the constant `value`.
    fn const_value(&mut self, value: &SemConstValue<'db>) -> Result<String, Unsupported> {
        let db = self.db;
        let mut parts = |parts: &[_]| -> Result<Vec<String>, Unsupported> {
            let part = |part: &SemConstId<'db>| self.const_value(part.value(db));
            parts.iter().map(part).collect()
        };
        match value {
            SemConstValue::Unit => Ok("()".to_string()),
            SemConstValue::Scalar {
                value: SemConstScalar::Int { value },
                ..
            } if !value.is_negative() => Ok(format!("(UInt256.ofNat {value})")),
            // A negative integer's canonical word is the same at every width.
            SemConstValue::Scalar {
                value: SemConstScalar::Int { value },
                ..
            } => Ok(format!("(IntTy.wrap (.i 256) ({value}))")),
            SemConstValue::Scalar {
                value: SemConstScalar::Bool(value),
                ..
            } => Ok(value.to_string()),
            SemConstValue::Scalar {
                ty,
                value: SemConstScalar::Bytes(bytes),
            } if ty.is_string(db) && bytes.len() <= 32 => {
                let word = bytes
                    .iter()
                    .fold(BigUint::ZERO, |word, byte| word * 256u32 + *byte);
                Ok(format!("(UInt256.ofNat {word})"))
            }
            SemConstValue::Tuple { ty, elems: fields }
            | SemConstValue::Struct { ty, fields }
            | SemConstValue::Array { ty, elems: fields } => {
                let parts = parts(fields)?;
                self.aggregate(*ty, &parts)
            }
            SemConstValue::Enum {
                ty,
                variant,
                fields,
            } => {
                let parts: String = parts(fields)?
                    .iter()
                    .map(|part| format!(" {part}"))
                    .collect();
                let name = self.value_ty(*ty)?;
                let variants = self.variants(*ty).unwrap_or_default();
                let (variant, _) = &variants[usize::from(variant.0)];
                Ok(format!("({name}.{variant}{parts})"))
            }
            other => Err(Unsupported(format!("constant {other:?}"))),
        }
    }

    /// The width of `ty`, and whether it is signed, if it is an integer type.
    fn int_bits(&self, ty: TyId<'db>) -> Option<(usize, bool)> {
        // A fixed string is a word: its bytes as a big-endian number.
        if ty.is_string(self.db) {
            return Some((256, false));
        }
        let prim = ty.as_prim(self.db)?;
        let target = self.db.compilation_settings().target(self.db);
        Some((prim_int_bits(prim, target)?, prim.is_signed_int()))
    }

    /// The Lean term for `value` of type `from` cast to integer type `to`.
    /// A canonical word is already that of a type that has its value; a
    /// narrower unsigned type cuts it, and any other cast wraps.
    fn cast(&self, value: &str, from: TyId<'db>, to: TyId<'db>) -> Result<String, Unsupported> {
        // A view of a scalar is the scalar.
        let from = from.as_capability(self.db).map_or(from, |(_, inner)| inner);
        if from.is_bool(self.db) && self.int_bits(to).is_some() {
            return Ok(format!("Bool.toUInt256 {value}"));
        }
        match (self.int_bits(from), self.int_bits(to)) {
            // A word reinterpreted at its own width.
            (Some((256, _)), Some((256, _))) => Ok(value.to_string()),
            (Some((from, false)), Some((to, _))) if from < to => Ok(value.to_string()),
            (Some((from, signed)), Some((to, to_signed))) if from <= to && signed == to_signed => {
                Ok(value.to_string())
            }
            (Some((_, false)), Some((to, false))) => Ok(format!("truncate {to} {value}")),
            (Some(from), Some(to)) => Ok(format!(
                "IntTy.cast {} {} {value}",
                int_ty(from),
                int_ty(to)
            )),
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
        // sign-extended if signed. A pointer is its address in memory, or
        // its slot number in contract state.
        if self.int_bits(ty).is_some() || ty.as_ptr(self.db).is_some() {
            return Ok("UInt256".to_string());
        }
        if ty.is_never(self.db) {
            return Ok("Empty".to_string());
        }
        // An access that is part of a value is the place it reaches in
        // contract state; a view of a snapshot type is the value.
        if let Some((kind, inner)) = ty.as_capability(self.db) {
            let scope = self.root.key(self.db).owner(self.db).scope();
            let snapshot = kind == CapabilityKind::View
                && ty_is_snapshot(self.db, scope, inner, self.root.assumptions(self.db))
                && self.root.pinned_space(self.db, inner).is_none();
            return if snapshot {
                self.value_ty(inner)
            } else {
                Ok("Place".to_string())
            };
        }
        let pretty = ty.pretty_print(self.db);
        match pretty.as_str() {
            // An address is its word.
            "Address" => return Ok("UInt256".to_string()),
            "bool" => return Ok("Bool".to_string()),
            "()" => return Ok("Unit".to_string()),
            _ => {}
        }
        if let Some(variants) = self.variants(ty) {
            return self.enum_ty(lean_ident(pretty), &variants);
        }
        if let Some(len) = ty.array_len(self.db) {
            let (_, args) = ty.decompose_ty_app(self.db);
            return Ok(format!("(Vector {} {len})", self.value_ty(args[0])?));
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

    /// The Lean type of an enum named `name`: an inductive type, with its
    /// tag, and for each field of each variant the projection that reads it
    /// and the function that replaces it. Both leave a value of another
    /// variant alone.
    fn enum_ty(
        &mut self,
        name: String,
        variants: &[(String, Vec<TyId<'db>>)],
    ) -> Result<String, Unsupported> {
        let mut ctors = String::new();
        let mut tags = String::new();
        let mut parts = String::new();
        let other = if variants.len() > 1 {
            ("  | _ => default\n", "  | x => x\n")
        } else {
            ("", "")
        };
        for (tag, (variant, fields)) in variants.iter().enumerate() {
            let tys: Result<Vec<_>, _> = fields.iter().map(|ty| self.value_ty(*ty)).collect();
            let tys = tys?;
            let binders: String = tys
                .iter()
                .enumerate()
                .map(|(field, ty)| format!(" (f{field} : {ty})"))
                .collect();
            writeln!(ctors, "  | {variant}{binders}").unwrap();
            let rest = if fields.is_empty() { "" } else { " .." };
            writeln!(tags, "  | .{variant}{rest} => UInt256.ofNat {tag}").unwrap();
            let bound: Vec<_> = (0..tys.len()).map(|field| format!("f{field}")).collect();
            for (field, ty) in tys.iter().enumerate() {
                let pattern = format!(".{variant} {}", bound.join(" "));
                let mut replaced = bound.clone();
                replaced[field] = "v".to_string();
                write!(
                    parts,
                    "\ndef NAME.{variant}_{field} : NAME → {ty}\n  | {pattern} => f{field}\n{}\n\
                     def NAME.set_{variant}_{field} (x : NAME) (v : {ty}) : NAME :=\n  match x with\n  \
                     | {pattern} => .{variant} {}\n{}",
                    other.0,
                    replaced.join(" "),
                    other.1,
                )
                .unwrap();
            }
        }
        Ok(self.layout(name, |name| {
            format!(
                "inductive {name} where\n{ctors}deriving Inhabited\n\n\
                 def {name}.tag : {name} → UInt256\n{tags}{}",
                parts.replace("NAME", name).trim_end()
            )
        }))
    }

    /// The slots a value of `ty` takes in contract state.
    fn span(&self, ty: TyId<'db>) -> Result<usize, Unsupported> {
        storage_slot_span(self.db, ty)
            .map_err(|_| Unsupported(format!("layout of {}", ty.pretty_print(self.db))))
    }

    /// The bytes a value of `ty` packs into and its one field, if `ty` is a
    /// newtype of a packable type.
    fn newtype(&self, ty: TyId<'db>) -> Option<(u32, TyId<'db>)> {
        let bytes = storage_lane_bytes(self.db, ty)?;
        let [field] = ty.field_types(self.db)[..] else {
            return None;
        };
        Some((bytes, field))
    }

    /// The bits of the scalar a packable type holds.
    fn scalar_bits(&self, ty: TyId<'db>) -> Option<usize> {
        match (self.int_bits(ty), self.newtype(ty)) {
            (Some((bits, _)), _) => Some(bits),
            (None, Some((_, field))) => self.scalar_bits(field),
            (None, None) => ty.is_bool(self.db).then_some(8),
        }
    }

    /// The lane a field of type `ty` placed by `placement` lies in. A lane
    /// narrower than its scalar is partial too: a write checks the value and
    /// a read masks it.
    fn lane(&self, ty: TyId<'db>, placement: StorageFieldPlacement) -> Option<Lane> {
        let lane = placement.lane?;
        let narrow = self
            .scalar_bits(ty)
            .is_some_and(|bits| bits > lane.bit_width() as usize);
        Some(Lane {
            off: lane.byte_offset,
            width: lane.byte_width,
            shared: placement.shared,
            partial: placement.partial_lane().is_some() || narrow,
            codec: None,
        })
    }

    /// `place` re-rooted in transient storage if values of `ty` are pinned
    /// there. A type pinned to storage lies in storage already.
    fn pinned(&self, ty: TyId<'db>, place: String) -> String {
        match self.root.pinned_space(self.db, ty) {
            Some(AddressSpace::Transient) => format!("({place}.pin .transient)"),
            _ => place,
        }
    }

    /// The Lean term for the part of `value`, of struct or tuple type `ty`,
    /// that `steps` reach.
    fn projected(value: &str, steps: &[String]) -> String {
        steps
            .iter()
            .fold(value.to_string(), |value, step| format!("{value}.{step}"))
    }

    /// The Lean computation that reads a value of `ty` at `place`: on the
    /// lane path if it lies in `lane`, and otherwise from whole slots.
    fn load(
        &mut self,
        ty: TyId<'db>,
        place: &str,
        lane: Option<Lane>,
    ) -> Result<String, Unsupported> {
        let db = self.db;
        if let Some(int) = self.int_bits(ty) {
            return Ok(match (lane, int) {
                (Some(lane), _) => format!("loadIntAt {} {place} {}", int_ty(int), lane.lean()),
                (None, (256, _)) => format!("load {place}"),
                (None, _) => format!("loadInt {} {place}", int_ty(int)),
            });
        }
        if ty.is_bool(db) {
            return Ok(match lane {
                Some(lane) => format!("loadBoolAt {place} {}", lane.lean()),
                None => format!("loadBool {place}"),
            });
        }
        // A pointer is its address or slot number.
        if ty.as_ptr(db).is_some() {
            return Ok(format!("load {place}"));
        }
        // A newtype lies in its field's lane: its own bytes if it has its
        // slot to itself.
        if let Some((width, field)) = self.newtype(ty) {
            let lane = lane.unwrap_or(Lane {
                off: 0,
                width,
                shared: false,
                partial: false,
                codec: None,
            });
            let inner = self.load(field, place, Some(lane))?;
            return Ok(match self.fields(ty)?.as_slice() {
                [(steps, _)] if steps.is_empty() => inner,
                _ => format!("{}.mk <$> {inner}", self.value_ty(ty)?),
            });
        }
        if let Some(len) = ty.array_len(db) {
            let (_, args) = ty.decompose_ty_app(db);
            let stride = self.span(args[0])?;
            let elem = self.pinned(args[0], "q".to_string());
            let elem = self.load(args[0], &elem, None)?;
            return Ok(format!(
                "loadArray {stride} (fun q => {elem}) {place} {len}"
            ));
        }
        let name = self.value_ty(ty)?;
        let mut body = String::new();
        if let Some(variants) = self.variants(ty) {
            writeln!(
                body,
                "  match (← loadTag {} {} p) with",
                tag_bits(variants.len()),
                variants.len()
            )
            .unwrap();
            for (tag, (variant, fields)) in variants.iter().enumerate() {
                // `loadTag` gives a variant's index, so the last one is every
                // other number.
                let pattern = if tag + 1 == variants.len() {
                    "_".to_string()
                } else {
                    tag.to_string()
                };
                writeln!(body, "  | {pattern} => do").unwrap();
                let mut slot = 1;
                for (index, field) in fields.iter().enumerate() {
                    let at = self.pinned(*field, format!("(p.offset {slot})"));
                    writeln!(body, "    let f{index} ← {}", self.load(*field, &at, None)?).unwrap();
                    slot += self.span(*field)?;
                }
                let parts: String = (0..fields.len())
                    .map(|index| format!(" f{index}"))
                    .collect();
                writeln!(body, "    return .{variant}{parts}").unwrap();
            }
        } else {
            let placements = storage_field_placements(db, ty)
                .ok_or_else(|| Unsupported(format!("layout of {}", ty.pretty_print(db))))?;
            let fields = self.fields(ty)?;
            for (index, ((_, field), placement)) in fields.iter().zip(&placements).enumerate() {
                let at = self.pinned(*field, format!("(p.offset {})", placement.slot));
                let lane = self.lane(*field, *placement);
                writeln!(body, "  let f{index} ← {}", self.load(*field, &at, lane)?).unwrap();
            }
            let parts: Vec<_> = (0..fields.len()).map(|index| format!("f{index}")).collect();
            writeln!(body, "  return {}", self.aggregate(ty, &parts)?).unwrap();
        }
        let loader = self.layout(
            format!("load_{}", lean_ident(ty.pretty_print(db))),
            |loader| {
                format!(
                    "def {loader} (p : Place) : FeM {name} := do\n{}",
                    body.trim_end()
                )
            },
        );
        Ok(format!("{loader} {place}"))
    }

    /// The bytes a value of `ty` takes in memory.
    fn size(&self, ty: TyId<'db>) -> Result<u64, Unsupported> {
        runtime_size_bytes(self.db, ty)
            .ok()
            .flatten()
            .ok_or_else(|| Unsupported(format!("size of {}", ty.pretty_print(self.db))))
    }

    /// The Lean computation that reads a value of `ty` in memory at
    /// `address`: a scalar's bytes, or an aggregate's parts one after
    /// another.
    fn mload(&mut self, ty: TyId<'db>, address: &str) -> Result<String, Unsupported> {
        let db = self.db;
        match self.int_bits(ty) {
            Some((256, _)) => return Ok(format!("mload {address}")),
            Some(int) => return Ok(format!("mloadInt {} {address}", int_ty(int))),
            None => {}
        }
        if ty.is_bool(db) {
            return Ok(format!("mloadBool {address}"));
        }
        if ty.as_ptr(db).is_some() {
            return Ok(format!("mload {address}"));
        }
        if let Some(len) = ty.array_len(db) {
            let (_, args) = ty.decompose_ty_app(db);
            let elem = self.mload(args[0], "q")?;
            let stride = self.size(args[0])?;
            return Ok(format!(
                "mloadArray {stride} (fun q => {elem}) {address} {len}"
            ));
        }
        if self.variants(ty).is_some() {
            return Err(Unsupported(format!("{} in memory", ty.pretty_print(db))));
        }
        let name = self.value_ty(ty)?;
        let fields = self.fields(ty)?;
        let mut body = String::new();
        let mut offset = 0;
        for (index, (_, field)) in fields.iter().enumerate() {
            let at = format!("(p + UInt256.ofNat {offset})");
            writeln!(body, "  let f{index} ← {}", self.mload(*field, &at)?).unwrap();
            offset += self.size(*field)?;
        }
        let parts: Vec<_> = (0..fields.len()).map(|index| format!("f{index}")).collect();
        writeln!(body, "  return {}", self.aggregate(ty, &parts)?).unwrap();
        let loader = self.layout(
            format!("mload_{}", lean_ident(ty.pretty_print(db))),
            |loader| {
                format!(
                    "def {loader} (p : UInt256) : FeM {name} := do\n{}",
                    body.trim_end()
                )
            },
        );
        Ok(format!("{loader} {address}"))
    }

    /// The Lean computation that writes `value` of `ty` in memory at
    /// `address`.
    fn mstore(&mut self, ty: TyId<'db>, address: &str, value: &str) -> Result<String, Unsupported> {
        let db = self.db;
        match self.int_bits(ty) {
            Some((256, _)) => return Ok(format!("mstore {address} {value}")),
            Some(int) => return Ok(format!("mstoreInt {} {address} {value}", int_ty(int))),
            None => {}
        }
        if ty.is_bool(db) {
            return Ok(format!("mstoreBool {address} {value}"));
        }
        if ty.as_ptr(db).is_some() {
            return Ok(format!("mstore {address} {value}"));
        }
        if ty.array_len(db).is_some() {
            let (_, args) = ty.decompose_ty_app(db);
            let elem = self.mstore(args[0], "q", "x")?;
            let stride = self.size(args[0])?;
            return Ok(format!(
                "mstoreArray {stride} (fun q x => {elem}) {address} {value}"
            ));
        }
        if self.variants(ty).is_some() {
            return Err(Unsupported(format!("{} in memory", ty.pretty_print(db))));
        }
        let name = self.value_ty(ty)?;
        let mut body = String::new();
        let mut offset = 0;
        for (steps, field) in self.fields(ty)? {
            let at = format!("(p + UInt256.ofNat {offset})");
            let part = Self::projected("v", &steps);
            if self.size(field)? > 0 {
                writeln!(body, "  {}", self.mstore(field, &at, &part)?).unwrap();
            }
            offset += self.size(field)?;
        }
        if body.is_empty() {
            body.push_str("  return ()\n");
        }
        let storer = self.layout(
            format!("mstore_{}", lean_ident(ty.pretty_print(db))),
            |storer| {
                format!(
                    "def {storer} (p : UInt256) (v : {name}) : FeM Unit := do\n{}",
                    body.trim_end()
                )
            },
        );
        Ok(format!("{storer} {address} {value}"))
    }

    /// The integer type and the word of the scalar `value` of packable type
    /// `ty` holds.
    fn scalar(&self, ty: TyId<'db>, value: String) -> Result<(String, String), Unsupported> {
        match (self.int_bits(ty), self.newtype(ty)) {
            (Some(int), _) => Ok((int_ty(int), value)),
            (None, Some((_, field))) => {
                let (steps, _) = self.fields(ty)?.swap_remove(0);
                self.scalar(field, Self::projected(&value, &steps))
            }
            (None, None) if ty.is_bool(self.db) => {
                Ok(("(.u 8)".to_string(), format!("(Bool.toUInt256 {value})")))
            }
            (None, None) => Err(Unsupported(format!("lane of {}", ty.pretty_print(self.db)))),
        }
    }

    /// The Lean computation that writes `value` of `ty` at `place`: on the
    /// lane path if it lies in `lane`, and otherwise over whole slots.
    fn store(
        &mut self,
        ty: TyId<'db>,
        place: &str,
        lane: Option<Lane>,
        value: &str,
    ) -> Result<String, Unsupported> {
        let db = self.db;
        if let Some(int) = self.int_bits(ty) {
            return Ok(match (lane, int) {
                (Some(lane), _) => {
                    format!("storeIntAt {} {place} {} {value}", int_ty(int), lane.lean())
                }
                (None, (256, _)) => format!("store {place} {value}"),
                (None, _) => format!("storeInt {} {place} {value}", int_ty(int)),
            });
        }
        if ty.is_bool(db) {
            return Ok(match lane {
                Some(lane) => format!("storeBoolAt {place} {} {value}", lane.lean()),
                None => format!("storeBool {place} {value}"),
            });
        }
        if ty.as_ptr(db).is_some() {
            return Ok(format!("store {place} {value}"));
        }
        if let Some((width, field)) = self.newtype(ty) {
            let lane = lane.unwrap_or(Lane {
                off: 0,
                width,
                shared: false,
                partial: false,
                codec: None,
            });
            let (steps, _) = self.fields(ty)?.swap_remove(0);
            return self.store(field, place, Some(lane), &Self::projected(value, &steps));
        }
        if ty.array_len(db).is_some() {
            let (_, args) = ty.decompose_ty_app(db);
            let stride = self.span(args[0])?;
            let elem = self.pinned(args[0], "q".to_string());
            let elem = self.store(args[0], &elem, None, "x")?;
            return Ok(format!(
                "storeArray {stride} (fun q x => {elem}) {place} {value}"
            ));
        }
        let name = self.value_ty(ty)?;
        let mut body = String::new();
        if let Some(variants) = self.variants(ty) {
            body.push_str("  match v with\n");
            for (tag, (variant, fields)) in variants.iter().enumerate() {
                let parts: String = (0..fields.len())
                    .map(|index| format!(" f{index}"))
                    .collect();
                writeln!(body, "  | .{variant}{parts} => do").unwrap();
                // The tag has the enum's first slot, and each field of the
                // variant whole slots after it. No other slot is written.
                writeln!(body, "    store p (UInt256.ofNat {tag})").unwrap();
                let mut slot = 1;
                for (index, field) in fields.iter().enumerate() {
                    let at = self.pinned(*field, format!("(p.offset {slot})"));
                    writeln!(
                        body,
                        "    {}",
                        self.store(*field, &at, None, &format!("f{index}"))?
                    )
                    .unwrap();
                    slot += self.span(*field)?;
                }
            }
        } else {
            let placements = storage_field_placements(db, ty)
                .ok_or_else(|| Unsupported(format!("layout of {}", ty.pretty_print(db))))?;
            // The fields packed into one slot are stored as one word, which
            // replaces the slot.
            let mut pending: Vec<(u32, String)> = Vec::new();
            let mut pending_slot = 0;
            let flush = |pending: &mut Vec<(u32, String)>, slot: u64, body: &mut String| {
                if pending.is_empty() {
                    return;
                }
                let lanes: Vec<_> = pending
                    .drain(..)
                    .map(|(off, bits)| format!("({off}, {bits})"))
                    .collect();
                writeln!(
                    body,
                    "  store (p.offset {slot}) (packed [{}])",
                    lanes.join(", ")
                )
                .unwrap();
            };
            for (index, ((steps, field), placement)) in
                self.fields(ty)?.iter().zip(&placements).enumerate()
            {
                let part = Self::projected("v", steps);
                let Some(lane) = self.lane(*field, *placement) else {
                    flush(&mut pending, pending_slot, &mut body);
                    if self.span(*field)? > 0 {
                        let at = self.pinned(*field, format!("(p.offset {})", placement.slot));
                        writeln!(body, "  {}", self.store(*field, &at, None, &part)?).unwrap();
                    }
                    continue;
                };
                if placement.slot != pending_slot {
                    flush(&mut pending, pending_slot, &mut body);
                    pending_slot = placement.slot;
                }
                let (int, word) = self.scalar(*field, part)?;
                writeln!(
                    body,
                    "  let b{index} ← Lane.pack {} {int} {word}",
                    lane.lean()
                )
                .unwrap();
                pending.push((lane.off, format!("b{index}")));
            }
            flush(&mut pending, pending_slot, &mut body);
            if body.is_empty() {
                body.push_str("  return ()\n");
            }
        }
        let storer = self.layout(
            format!("store_{}", lean_ident(ty.pretty_print(db))),
            |storer| {
                format!(
                    "def {storer} (p : Place) (v : {name}) : FeM Unit := do\n{}",
                    body.trim_end()
                )
            },
        );
        Ok(format!("{storer} {place} {value}"))
    }

    /// The layout of struct or tuple `ty`: the slot of each field below a
    /// place of `ty` in storage.
    fn struct_layout(&mut self, ty: TyId<'db>) -> Result<String, Unsupported> {
        let pretty = ty.pretty_print(self.db);
        let placements = storage_field_placements(self.db, ty)
            .ok_or_else(|| Unsupported(format!("field of {pretty}")))?;
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
    carriers: &'a Carriers,
    /// The slots of locals and of parameters, each a mutable Lean variable.
    slots: Vec<NRootId>,
    /// The `mut` access parameters received as values, which the body
    /// returns after its result.
    outs: Vec<NValueId>,
    /// The effects received as values, each a mutable variable `eN`. The
    /// body returns those of the `mut` ones after `outs`.
    effects: Vec<Effect<'db>>,
    /// The Lean type of the value the body returns.
    result_ty: String,
    /// The name and type of each parameter of the definition.
    binders: Vec<(String, String)>,
    /// The blocks printed as local functions, under the block that
    /// immediately dominates each, in the order they are defined.
    joins: FxHashMap<NBlockId, Vec<NBlockId>>,
    joined: FxHashSet<NBlockId>,
    /// The joins an edge from a later block enters: the heads of loops,
    /// printed as recursive local functions.
    loops: FxHashSet<NBlockId>,
    /// The borrows of packed lanes, with the place each borrows and whether
    /// it is `mut`. A lane has no address, so the borrow holds its value in a
    /// mutable variable, which a `mut` borrow writes back where it ends.
    helds: Vec<(NValueId, NPlace<'db>, bool)>,
    /// The blocks of a projection's slides: those a yield resumes into.
    slide: FxHashSet<NBlockId>,
    /// The Lean type of what a slide returns, if the body is a projection
    /// that returns its slide.
    slide_ty: Option<String>,
    /// The calls of projections that grant `mut` access to a local: the
    /// session is a mutable variable holding the granted value.
    sessions: Vec<NValueId>,
    /// Where the slide of each open session of a projection that returns its
    /// slide leaves the values it returns.
    slides: FxHashMap<NValueId, Vec<Out<'db>>>,
    /// The result of the call printed last in the block being printed that
    /// never returns.
    never: Option<String>,
    /// Whether the block being printed ends in a yield, which a call that
    /// never returns before it replaces.
    yielding: bool,
    /// The lanes of entries a collection packs: the Lean term for each lane's
    /// bit offset, the collection codec's `get` and `set`, the entry's type,
    /// and the steps from the entry to the part of it the place names.
    codecs: Vec<(String, String, String, TyId<'db>, Vec<Step>)>,
    /// The lines that locate the entries the statement being printed names,
    /// which go before it.
    located: Vec<String>,
    /// The number of entries located so far, each bound to its own name.
    entries: usize,
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
            // An entry's access parameters are views of decoded arguments,
            // held by value.
            let entry = matches!(
                owner,
                BodyOwner::ContractInit { .. } | BodyOwner::ContractRecvArm { .. }
            );
            let carrier = match self.carriers.params.get(param as usize).copied().flatten() {
                None if entry && self.is_access(value.ty) => Some(Carrier::Value),
                carrier => carrier,
            };
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
            params.push((param, format!("v{index}"), ty));
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
                params.push((param, format!("r{index}"), ty.clone()));
                writeln!(state, "  let mut r{index} := r{index}").unwrap();
            } else {
                writeln!(state, "  let mut r{index} : {ty} := default").unwrap();
            }
        }
        params.sort_by_key(|(param, ..)| *param);
        self.binders = params.into_iter().map(|(_, name, ty)| (name, ty)).collect();
        for mut effect in self.module.effect_params(self.body.owner.key(db)) {
            let name = format!("e{}", effect.provider);
            let given = self
                .carriers
                .effects
                .iter()
                .find(|(provider, _)| *provider == effect.provider);
            effect.state = given.map_or(effect.state, |(_, carrier)| *carrier == Carrier::Place);
            if effect.state {
                self.binders.push((name, "Place".to_string()));
            } else {
                writeln!(state, "  let mut {name} := {name}").unwrap();
                self.binders.push((name, self.module.value_ty(effect.ty)?));
                self.effects.push(effect);
            }
        }
        let params = binders(&self.binders);
        for out in &self.outs {
            writeln!(state, "  let mut {0} := {0}", value_name(*out)).unwrap();
        }
        // A call of a projection that grants `mut` access to a local holds
        // the granted value in a mutable variable.
        for statement in self.body.blocks.iter().flat_map(|block| &block.statements) {
            if let NStatementKind::Define {
                result,
                expr: NExpr::Call { callee, .. },
            } = &statement.kind
                && SemanticInstance::new(db, callee.key).is_projection(db)
                && let Some((CapabilityKind::Mut, inner)) = self.ty(*result).as_capability(db)
                && !self.granted_in_state(*result, &NDataPath::empty())
            {
                let ty = self.module.value_ty(inner)?;
                writeln!(state, "  let mut {} : {ty} := default", value_name(*result)).unwrap();
                self.sessions.push(*result);
            }
        }
        for statement in self.body.blocks.iter().flat_map(|block| &block.statements) {
            let NStatementKind::Define { result, expr } = &statement.kind else {
                continue;
            };
            let (place, mutable) = match expr {
                NExpr::Borrow { place, kind, .. } => (place, *kind == BorrowKind::Mut),
                NExpr::MakeView { place, .. } => (place, false),
                _ => continue,
            };
            if !self.is_access(self.ty(*result))
                || self.is_root_object(place)
                || !self.in_state(place)?
            {
                continue;
            }
            // Only the lane matters here; the entries on the way are located
            // where the borrow is printed.
            let entries = self.entries;
            let (_, lane) = self.place(place)?;
            self.located.clear();
            self.entries = entries;
            if !lane.is_some_and(|lane| lane.partial) {
                continue;
            }
            if self.held(*result).is_none() {
                let ty = self.module.value_ty(place.ty)?;
                writeln!(state, "  let mut h{} : {ty} := default", result.index()).unwrap();
            }
            self.helds.push((*result, place.clone(), mutable));
        }
        self.result_ty = self.returned_ty()?;
        // A function that returns a value and whose body returns none has
        // no body of its own: the compiler supplies its result.
        if let BodyOwner::Func(func, _) = owner
            && self.result_ty == "Unit"
            && func.return_ty(db) != TyId::unit(db)
        {
            return Err(Unsupported("body the compiler supplies".to_string()));
        }
        let never = self.result_ty == "Empty";
        let mut returned = Vec::new();
        for out in self.outs.clone() {
            let (_, inner) = self.ty(out).as_capability(db).unwrap();
            returned.push(self.module.value_ty(inner)?);
        }
        for effect in self.effects.clone() {
            if effect.mutable {
                returned.push(self.module.value_ty(effect.ty)?);
            }
        }
        // A projection returns what it grants. If the caller's writes must
        // come back, or its slide does anything, it returns the slide too: a
        // function from the value the caller leaves to the values the
        // projection's `mut` parameters end with.
        let yields: Vec<_> = self
            .body
            .blocks
            .iter()
            .filter_map(|block| match &block.terminator.kind {
                NTerminatorKind::Yield { value, resume } => Some((value.value, resume.block)),
                _ => None,
            })
            .collect();
        let mut direct = returned.is_empty();
        for (value, resume) in &yields {
            let ty = self.ty(*value);
            let place = self.is_access(ty) && self.carrier(*value)? == Carrier::Place;
            let read_only = !matches!(ty.as_capability(db), Some((CapabilityKind::Mut, _)));
            direct &= (read_only || place) && self.slide_is_empty(*resume);
        }
        if !yields.is_empty() && !direct {
            let mut pending: Vec<_> = yields.iter().map(|(_, resume)| *resume).collect();
            while let Some(block) = pending.pop() {
                if self.slide.insert(block) {
                    let successors = self.body.blocks[block.index()].terminator.kind.successors();
                    pending.extend(successors.iter().map(|successor| successor.block));
                }
            }
            let outs = match returned.as_slice() {
                [] => "Unit".to_string(),
                [ty] => ty.clone(),
                tys => format!("({})", tys.join(" × ")),
            };
            self.result_ty = format!("({0} × ({0} → FeM {outs}))", self.result_ty);
            self.slide_ty = Some(outs);
        } else if !returned.is_empty() {
            self.result_ty = format!("({} × {})", self.result_ty, returned.join(" × "));
        }
        self.find_joins();
        let mut out = format!("{params} : FeM {} := do\n{state}", self.result_ty);
        self.block(self.body.entry, 1, &mut out)?;
        self.module.lens = self.slide_ty.is_some();
        Ok((out, never))
    }

    /// The mutable variables of the body: its local slots, then the values
    /// it returns for its `mut` access parameters.
    fn state(&self) -> Vec<String> {
        let slots = self.slots.iter().map(|root| format!("r{}", root.index()));
        let helds = self.helds.iter().filter_map(|(held, ..)| {
            let (var, _) = self.held(*held)?;
            (var == *held).then(|| format!("h{}", held.index()))
        });
        slots
            .chain(self.outs.iter().copied().map(value_name))
            .chain(
                self.effects
                    .iter()
                    .map(|effect| format!("e{}", effect.provider)),
            )
            .chain(self.sessions.iter().copied().map(value_name))
            .chain(helds)
            .collect()
    }

    /// The variables the body returns after its result: the values its `mut`
    /// access parameters and `mut` effects held by value leave.
    fn returned(&self) -> Vec<String> {
        let effects = self.effects.iter().filter(|effect| effect.mutable);
        self.outs
            .iter()
            .copied()
            .map(value_name)
            .chain(effects.map(|effect| format!("e{}", effect.provider)))
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
        // A projection gives what it yields: the place its access reaches
        // in contract state, or the value it reaches in a local.
        let yielded = self
            .body
            .blocks
            .iter()
            .find_map(|block| match &block.terminator.kind {
                NTerminatorKind::Yield { value, .. } => Some(value.value),
                _ => None,
            });
        if let Some(yielded) = yielded {
            let ty = self.ty(yielded);
            if self.is_access(ty) && self.carrier(yielded)? == Carrier::Place {
                return Ok("Place".to_string());
            }
            let inner = ty.as_capability(self.module.db).map_or(ty, |(_, ty)| ty);
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
    /// the block that immediately dominates it. An edge from a later block
    /// closes a loop: its target is the loop's head, a recursive join, and
    /// the edge is no part of what dominates a block.
    fn find_joins(&mut self) {
        let order = self.body.reverse_postorder();
        let position: FxHashMap<_, _> = order.iter().enumerate().map(|(i, b)| (*b, i)).collect();
        let mut preds: FxHashMap<NBlockId, Vec<NBlockId>> = FxHashMap::default();
        for &block in &order {
            for successor in self.body.blocks[block.index()].terminator.kind.successors() {
                if position[&successor.block] <= position[&block] {
                    self.loops.insert(successor.block);
                    self.joined.insert(successor.block);
                } else {
                    preds.entry(successor.block).or_default().push(block);
                }
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
            if preds[&block].len() > 1
                || !self.body.blocks[block.index()].params.is_empty()
                || self.loops.contains(&block)
            {
                self.joined.insert(block);
                // A later join can be the target of an earlier one, so it is
                // defined first.
                self.joins.entry(dominator).or_default().insert(0, block);
            }
        }
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
        // The text of each statement that prints any, and the value it
        // defines.
        let mut printed = Vec::new();
        self.never = None;
        self.yielding = matches!(data.terminator.kind, NTerminatorKind::Yield { .. });
        for statement in &data.statements {
            let text = self.statement(statement)?;
            let located = std::mem::take(&mut self.located);
            let lines = located.iter().chain(&text).flat_map(|text| text.lines());
            let text: String = lines.map(|line| format!("{pad}{line}\n")).collect();
            let defined = match statement.kind {
                NStatementKind::Define { result, .. } => Some(result),
                _ => None,
            };
            if !text.is_empty() {
                printed.push((text, defined));
            }
        }
        // Nothing is granted after a call that never returns: the yield that
        // follows one names what the path that returns grants elsewhere.
        if let Some(never) = self.never.take()
            && std::mem::take(&mut self.yielding)
        {
            printed.iter().for_each(|(text, _)| out.push_str(text));
            writeln!(out, "{pad}nomatch {never}").unwrap();
            return Ok(());
        }
        if block != self.body.entry || printed.len() <= LONGEST_RUN {
            printed.iter().for_each(|(text, _)| out.push_str(text));
            return self.tail(block, depth, out);
        }
        // A long definition is cut into pieces. Each ends by calling the
        // next with the variables still in use, and the last has the rest of
        // the body.
        let mut pieces: Vec<String> = printed
            .chunks(PIECE)
            .map(|piece| piece.iter().map(|(text, _)| text.as_str()).collect())
            .collect();
        self.tail(block, depth, pieces.last_mut().unwrap())?;
        let state = self.state();
        let state_tys = self.state_tys()?;
        let mut defined = Vec::new();
        for index in 1..pieces.len() {
            defined.extend(
                printed[(index - 1) * PIECE..index * PIECE]
                    .iter()
                    .filter_map(|(_, value)| *value),
            );
            let rest = pieces[index..].concat();
            let mut live: Vec<_> = self
                .binders
                .iter()
                .filter(|(name, _)| !state.contains(name))
                .cloned()
                .collect();
            for value in &defined {
                live.push((value_name(*value), self.lean_ty(self.ty(*value))?));
            }
            live.retain(|(name, _)| mentions(&rest, name));
            live.extend(state.iter().cloned().zip(state_tys.iter().cloned()));
            let args: String = live.iter().map(|(name, _)| format!(" {name}")).collect();
            writeln!(pieces[index - 1], "{pad}{SELF}_k{index}{args}").unwrap();
            let again: String = state
                .iter()
                .map(|var| format!("  let mut {var} := {var}\n"))
                .collect();
            pieces[index] = format!(
                "def {SELF}_k{index}{} : FeM {} := do\n{again}{}",
                binders(&live),
                self.result_ty,
                pieces[index]
            );
        }
        out.push_str(&pieces.join(&CUT.to_string()));
        Ok(())
    }

    /// The Lean types of the body's mutable variables, in the order of
    /// `state`.
    fn state_tys(&mut self) -> Result<Vec<String>, Unsupported> {
        let mut tys = Vec::new();
        for root in self.slots.clone() {
            let slot = self.body.roots[root.index()].ty;
            tys.push(self.module.value_ty(slot)?);
        }
        for out in self.outs.clone() {
            let (_, inner) = self.ty(out).as_capability(self.module.db).unwrap();
            tys.push(self.module.value_ty(inner)?);
        }
        for effect in self.effects.clone() {
            tys.push(self.module.value_ty(effect.ty)?);
        }
        for session in self.sessions.clone() {
            let (_, inner) = self.ty(session).as_capability(self.module.db).unwrap();
            tys.push(self.module.value_ty(inner)?);
        }
        for (held, place, _) in self.helds.clone() {
            if self.held(held).is_some_and(|(var, _)| var == held) {
                tys.push(self.module.value_ty(place.ty)?);
            }
        }
        Ok(tys)
    }

    /// Prints what follows the statements of `block`: the joins it
    /// dominates, and where it goes, with each block only it enters inline.
    fn tail(&mut self, block: NBlockId, depth: usize, out: &mut String) -> Result<(), Unsupported> {
        let data = &self.body.blocks[block.index()];
        let pad = "  ".repeat(depth);
        // A join takes the body's mutable variables before its parameters,
        // since a local function cannot assign its definer's.
        let state = self.state();
        for join in self.joins.get(&block).cloned().unwrap_or_default() {
            let params = &self.body.blocks[join.index()].params;
            let mut tys = self.state_tys()?;
            for param in params {
                tys.push(self.lean_ty(self.ty(*param))?);
            }
            let names = state
                .iter()
                .cloned()
                .chain(params.iter().copied().map(value_name));
            let index = join.index();
            let result = match &self.slide_ty {
                Some(outs) if self.slide.contains(&join) => outs.clone(),
                _ => self.result_ty.clone(),
            };
            // The head of a loop calls itself: the least fixed point, which
            // diverges where the loop does.
            if self.loops.contains(&join) {
                let binders: String = names
                    .zip(&tys)
                    .map(|(name, ty)| format!(" ({name} : {ty})"))
                    .collect();
                writeln!(out, "{pad}let rec b{index}{binders} : FeM {result} := do").unwrap();
            } else {
                let ty: String = tys.iter().map(|ty| format!("{ty} → ")).collect();
                let names: Vec<_> = names.collect();
                let binder = if names.is_empty() {
                    String::new()
                } else {
                    format!("fun {} => ", names.join(" "))
                };
                writeln!(out, "{pad}let b{index} : {ty}FeM {result} := {binder}do").unwrap();
            }
            for var in &state {
                writeln!(out, "{pad}  let mut {var} := {var}").unwrap();
            }
            self.block(join, depth + 1, out)?;
            if self.loops.contains(&join) {
                writeln!(out, "{pad}partial_fixpoint").unwrap();
            }
        }
        match &data.terminator.kind {
            // A slide returns the values the projection's `mut` parameters
            // end with.
            NTerminatorKind::Return(_) if self.slide.contains(&block) => {
                let outs = match self.returned().as_slice() {
                    [] => "()".to_string(),
                    [out] => out.clone(),
                    outs => format!("({})", outs.join(", ")),
                };
                writeln!(out, "{pad}return {outs}").unwrap();
            }
            NTerminatorKind::Return(returned) => {
                let result = match returned {
                    Some(returned) => self.operand(returned)?,
                    None => "()".to_string(),
                };
                let outs: String = self
                    .returned()
                    .iter()
                    .map(|out| format!(", {out}"))
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
            NTerminatorKind::MatchEnum {
                value,
                enum_ty,
                cases,
                default,
            } => {
                let variants = self.module.variants(*enum_ty);
                let variants =
                    variants.ok_or_else(|| Unsupported("match on no enum".to_string()))?;
                // Lean cannot show a recursive function monotone through a
                // match on a `let`-bound value, so a body with a loop binds
                // the value first.
                let (load, value) = self.enum_value(value)?;
                if let Some(load) = load {
                    writeln!(out, "{pad}{load}").unwrap();
                }
                if self.loops.is_empty() {
                    writeln!(out, "{pad}match {value} with").unwrap();
                } else {
                    writeln!(out, "{pad}match (← pure {value}) with").unwrap();
                }
                for (variant, target) in cases {
                    let (variant, fields) = &variants[usize::from(variant.0)];
                    let rest = if fields.is_empty() { "" } else { " .." };
                    writeln!(out, "{pad}| .{variant}{rest} =>").unwrap();
                    self.jump(target, depth + 1, out)?;
                }
                if let Some(default) = default
                    && cases.len() < variants.len()
                {
                    writeln!(out, "{pad}| _ =>").unwrap();
                    self.jump(default, depth + 1, out)?;
                }
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
                let place = self.is_access(ty) && self.carrier(value.value)? == Carrier::Place;
                let target = (self.is_access(ty) && !place)
                    .then(|| self.carrier_value(value.value))
                    .transpose()?;
                let granted = match &target {
                    Some(target) => self.read(target)?,
                    None => self.operand(value)?,
                };
                let Some(outs) = self.slide_ty.clone() else {
                    writeln!(out, "{pad}return {granted}").unwrap();
                    return Ok(());
                };
                // The slide takes the value the caller leaves: a write
                // through a `mut` grant of a local comes back with it.
                let granted_ty = self.returned_ty()?;
                writeln!(
                    out,
                    "{pad}let slide : {granted_ty} → FeM {outs} := fun g => do"
                )
                .unwrap();
                for var in self.state() {
                    writeln!(out, "{pad}  let mut {var} := {var}").unwrap();
                }
                if let Some(target) = &target
                    && matches!(ty.as_capability(db), Some((CapabilityKind::Mut, _)))
                {
                    writeln!(out, "{pad}  {}", self.write(target, "g")?).unwrap();
                }
                self.jump(resume, depth + 1, out)?;
                writeln!(out, "{pad}return ({granted}, slide)").unwrap();
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
                    // A borrow of a packed lane holds the lane's value.
                    NExpr::MakeView { place, .. } | NExpr::Borrow { place, .. }
                        if self.helds.iter().any(|(held, ..)| held == result) =>
                    {
                        let (var, _) = self.held(*result).unwrap();
                        let (lean, lane) = self.place(place)?;
                        let load = self.load(place.ty, &lean, lane)?;
                        Some(format!("h{} := (← {load})", var.index()))
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
                    NExpr::Const(SConst::Ref(constant)) => {
                        let db = self.module.db;
                        let saved = self.module.clone();
                        Some(
                            match self
                                .module
                                .instance(constant.instance(db), Carriers::default())
                            {
                                Ok(constant) => format!("let {name} ← {constant}"),
                                // A constant whose definition has no model is
                                // the value the compiler computes for it.
                                Err(unsupported) => {
                                    *self.module = saved;
                                    let EvalOutcome::Ready(value) = eval_const_ref(db, *constant)
                                    else {
                                        return Err(unsupported);
                                    };
                                    let value = self.module.const_value(value.value(db))?;
                                    format!(
                                        "let {name} := {value} -- a constant, as the compiler evaluates it"
                                    )
                                }
                            },
                        )
                    }
                    NExpr::Const(SConst::Value(value)) => {
                        let value = value.value().value(self.module.db);
                        Some(format!("let {name} := {}", self.module.const_value(value)?))
                    }
                    NExpr::Load { place, .. } => Some(match self.target(place)? {
                        Target::Place(lean, lane) => {
                            format!("let {name} ← {}", self.load(place.ty, &lean, lane)?)
                        }
                        Target::Value(target) => format!("let {name} := {}", self.read(&target)?),
                        Target::Memory(address) => {
                            format!("let {name} ← {}", self.module.mload(place.ty, &address)?)
                        }
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
                    NExpr::Binary { op, lhs, rhs }
                        if self.ty(lhs.value).is_bool(self.module.db) =>
                    {
                        let (lhs, rhs) = (self.operand(lhs)?, self.operand(rhs)?);
                        Some(match op {
                            BinaryOp::Comp(CompBinOp::Eq) => {
                                format!("let {name} := {lhs} == {rhs}")
                            }
                            BinaryOp::Comp(CompBinOp::NotEq) | BinaryOp::BitXor => {
                                format!("let {name} := {lhs} != {rhs}")
                            }
                            BinaryOp::BitAnd => format!("let {name} := {lhs} && {rhs}"),
                            BinaryOp::BitOr => format!("let {name} := {lhs} || {rhs}"),
                            _ => return Err(Unsupported(format!("operator {op:?} on bool"))),
                        })
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
                        let int = self.module.int_bits(self.ty(value.value));
                        let value = self.operand(value)?;
                        Some(format!("let {name} {}", unary(*op, int, &value)?))
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
                                (Target::Place(lean, _), true) => format!("let {name} := {lean}"),
                                // A view of a snapshot type is the value now there.
                                (Target::Place(lean, lane), false) => {
                                    let load = self.load(place.ty, &lean, lane)?;
                                    format!("let {name} ← {load}")
                                }
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
                        self.parts_in_state(fields)?;
                        let fields: Vec<_> =
                            fields.iter().map(|field| value_name(field.value)).collect();
                        let value = self.module.aggregate(*ty, &fields)?;
                        Some(format!("let {name} := {value}"))
                    }
                    NExpr::ArrayRepeat { ty, value } => {
                        let len = ty
                            .array_len(self.module.db)
                            .ok_or_else(|| Unsupported("array of unknown length".to_string()))?;
                        let value = self.operand(value)?;
                        Some(format!("let {name} := Vector.replicate {len} {value}"))
                    }
                    NExpr::EnumMake {
                        enum_ty,
                        variant,
                        fields,
                    } => {
                        self.parts_in_state(fields)?;
                        let ty = self.module.value_ty(*enum_ty)?;
                        let variants = self.module.variants(*enum_ty).unwrap_or_default();
                        let (variant, _) = &variants[usize::from(variant.0)];
                        let fields: String = fields
                            .iter()
                            .map(|field| format!(" {}", value_name(field.value)))
                            .collect();
                        Some(format!("let {name} := {ty}.{variant}{fields}"))
                    }
                    NExpr::GetEnumTag { value } => {
                        let (load, value) = self.enum_value(value)?;
                        let load = load.map_or(String::new(), |load| format!("{load}\n"));
                        Some(format!("{load}let {name} := {value}.tag"))
                    }
                    NExpr::IsEnumVariant { value, variant } => {
                        let ty = self.ty(value.value);
                        let ty = ty.as_capability(self.module.db).map_or(ty, |(_, ty)| ty);
                        let variants = self.module.variants(ty);
                        let variants = variants
                            .ok_or_else(|| Unsupported("variant of no enum".to_string()))?;
                        let (variant, fields) = &variants[usize::from(variant.0)];
                        let rest = if fields.is_empty() { "" } else { " .." };
                        let (load, value) = self.enum_value(value)?;
                        let load = load.map_or(String::new(), |load| format!("{load}\n"));
                        Some(if variants.len() == 1 {
                            format!("{load}let {name} := true")
                        } else {
                            format!(
                                "{load}let {name} := match {value} with | .{variant}{rest} => true | _ => false"
                            )
                        })
                    }
                    NExpr::Call {
                        callee,
                        args,
                        effect_args,
                        ..
                    } => {
                        let call = self.call(*result, callee.key, args, effect_args)?;
                        Some(self.after_never(*result, call)?)
                    }
                    other => return Err(Unsupported(format!("expression {other:?}"))),
                }
            }
            NStatementKind::Store { destination, value } => {
                let value = self.operand(value)?;
                Some(match self.target(destination)? {
                    Target::Place(lean, lane) => self.store(destination.ty, &lean, lane, &value)?,
                    Target::Value(target) => self.write(&target, &value)?,
                    Target::Memory(address) => {
                        self.module.mstore(destination.ty, &address, &value)?
                    }
                })
            }
            // The end of a session runs the projection's slide on the value
            // the session leaves.
            NStatementKind::End { access } if self.slides.contains_key(access) => {
                let outs = self.slides[access].clone();
                let name = value_name(*access);
                let names: Vec<_> = (0..outs.len()).map(|n| format!("{name}_{n}")).collect();
                let binder = match names.as_slice() {
                    [] => "_".to_string(),
                    [one] => one.clone(),
                    names => format!("({})", names.join(", ")),
                };
                let mut lines = format!("let {binder} ← {name}_k {name}");
                for (out, value) in outs.iter().zip(&names) {
                    let line = match out {
                        Out::Value(target) => self.write(target, value)?,
                        Out::Lane(ty, place, lane) => self.store(*ty, place, Some(*lane), value)?,
                    };
                    write!(lines, "\n{line}").unwrap();
                }
                Some(lines)
            }
            // A `mut` borrow of a lane writes the value it held back.
            NStatementKind::End { access } => {
                let held = self
                    .helds
                    .iter()
                    .find(|(held, _, mutable)| held == access && *mutable)
                    .cloned();
                match held {
                    Some((held, place, _)) => {
                        let (var, _) = self.held(held).unwrap();
                        let (lean, lane) = self.place(&place)?;
                        let value = format!("h{}", var.index());
                        Some(self.store(place.ty, &lean, lane, &value)?)
                    }
                    None => None,
                }
            }
            // An index known when compiling was checked then.
            NStatementKind::CheckIndex {
                index: NIndex::Const(_),
                ..
            } => None,
            NStatementKind::CheckIndex {
                index: NIndex::Value(index),
                array,
            } => {
                let len = array
                    .array_len(self.module.db)
                    .ok_or_else(|| Unsupported("array of unknown length".to_string()))?;
                Some(format!("checkIndex {} {len}", value_name(*index)))
            }
            NStatementKind::Drop { target } => {
                let db = self.module.db;
                let parts = drop_elaboration(db, self.body.owner)
                    .ok_or_else(|| Unsupported("drop in an unchecked body".to_string()))?
                    .parts(statement.id);
                // The target as a value: a local's, or the value itself.
                let whole = match target {
                    NDropTarget::Value(value) => ValueTarget {
                        var: value_name(*value),
                        ty: self.ty(*value),
                        path: Vec::new(),
                    },
                    NDropTarget::Place(place) => match self.target(place)? {
                        Target::Value(target) => target,
                        _ => return Err(Unsupported("drop of contract state".to_string())),
                    },
                };
                // Inside `Drop::drop`, destroying `self` whole destroys its
                // fields: the destructor is already running.
                let own_self = match target {
                    NDropTarget::Value(value) => matches!(
                        self.body.values[value.index()].definition,
                        NValueDefinition::EntryParam { param: 0 }
                    ),
                    NDropTarget::Place(place) => {
                        place.path.is_empty()
                            && matches!(place.base, NPlaceBase::Root(root)
                            if matches!(
                                self.body.roots[root.index()].kind,
                                NRootKind::ParamPlace { param: 0 }
                            ))
                    }
                };
                let in_drop = matches!(self.body.owner.key(db).owner(db),
                    BodyOwner::Func(func, _) if implements_drop(db, func));
                let mut lines = Vec::new();
                for part in parts {
                    let mut target = ValueTarget {
                        var: whole.var.clone(),
                        ty: whole.ty,
                        path: whole.path.clone(),
                    };
                    for step in &part.path {
                        let DropStep::Field(field) = step else {
                            return Err(Unsupported(format!("drop of {step:?}")));
                        };
                        target.path.push(Step::Field(usize::from(*field)));
                    }
                    if !part.flags.is_empty() {
                        return Err(Unsupported("drop under a flag".to_string()));
                    }
                    let mut ty = target.ty;
                    for &step in &target.path {
                        ty = self.module.part(ty, step)?.1;
                    }
                    let value = self.read(&target)?;
                    let field_wise = own_self && in_drop && part.path.is_empty();
                    self.destroy(&value, ty, field_wise, &mut lines)?;
                }
                (!lines.is_empty()).then(|| lines.join("\n"))
            }
        })
    }

    /// Appends the lines that destroy `value` of type `ty`: a call of its
    /// `Drop::drop` if its type implements `Drop` and the value is not
    /// destroyed `field_wise`, and otherwise the destruction of each field
    /// that needs it.
    fn destroy(
        &mut self,
        value: &str,
        ty: TyId<'db>,
        field_wise: bool,
        lines: &mut Vec<String>,
    ) -> Result<(), Unsupported> {
        let db = self.module.db;
        if !ty_has_drop_glue(db, ty) {
            return Ok(());
        }
        let adt = ty.adt_def(db).map(|adt| adt.adt_ref(db));
        if !field_wise && adt.is_some_and(|adt| adt_drop_impl(db, adt).is_some()) {
            let scope = self.body.owner.key(db).owner(db).scope();
            let key = core_method_callee_key(
                db,
                scope,
                PredicateListId::empty_list(db),
                &["ops", "Drop"],
                "drop",
                vec![ty],
            )
            .map_err(|error| Unsupported(format!("drop: generated call: {error:?}")))?;
            let drop = self.module.instance(key, vec![None].into())?;
            lines.push(format!("let _ ← {drop} {value}"));
            return Ok(());
        }
        if !matches!(adt, Some(AdtRef::Struct(_))) && !ty.is_tuple(db) {
            let ty = ty.pretty_print(db);
            return Err(Unsupported(format!("drop of {ty}")));
        }
        for (steps, field) in self.module.fields(ty)? {
            let steps: String = steps.iter().map(|step| format!(".{step}")).collect();
            self.destroy(&format!("{value}{steps}"), field, false, lines)?;
        }
        Ok(())
    }

    /// A call: to a std function the prelude models, by name, or to an
    /// exported function. An access to a local passes the value there, and
    /// a `mut` one stores the value the callee returns for it.
    fn call(
        &mut self,
        result: NValueId,
        callee: SemanticInstanceKey<'db>,
        args: &[NOperand],
        effect_args: &[NEffectArg<'db>],
    ) -> Result<String, Unsupported> {
        let db = self.module.db;
        let name = &value_name(result);
        let BodyOwner::Func(func, _) = callee.owner(db) else {
            return Err(Unsupported("call to a non-function".to_string()));
        };
        if let Some(kind) = runtime_builtin_func_kind(db, func) {
            return self.builtin(kind, result, callee, args);
        }
        if let Some(kind) = core_numeric_const_intrinsic(db, func) {
            return self.numeric(kind, result, args);
        }
        if lib_func_matches(db, func, "core::num::__bitcast")
            && let [value] = args
        {
            let from = self.ty(value.value);
            let value = self.operand(value)?;
            let cast = self.module.cast(&value, from, self.ty(result))?;
            return Ok(format!("let {name} := {cast}"));
        }
        let mut carriers = Vec::new();
        let mut effects = Vec::new();
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
                outs.push(Out::Value(target));
            }
        }
        // A callee's instance names the contract field behind an effect.
        // Any other effect is an argument: its place in contract state, or
        // the value of the local it is, which a `mut` effect leaves changed.
        for effect in self.module.effect_params(callee) {
            let arg = effect_args
                .iter()
                .find(|arg| arg.binding_idx == effect.provider)
                .ok_or_else(|| Unsupported("effect without an argument".to_string()))?;
            let target = match &arg.arg {
                NEffectArgValue::Place(place) => self.target(place)?,
                NEffectArgValue::Value(operand) if self.is_access(self.ty(operand.value)) => {
                    match self.carrier(operand.value)? {
                        Carrier::Value => Target::Value(self.carrier_value(operand.value)?),
                        _ => Target::Place(value_name(operand.value), None),
                    }
                }
                NEffectArgValue::Value(operand) => {
                    if effect.mutable {
                        return Err(Unsupported(
                            "`mut` effect argument that is a value".to_string(),
                        ));
                    }
                    write!(lean_args, " {}", self.operand(operand)?).unwrap();
                    effects.push((effect.provider, Carrier::Value));
                    continue;
                }
            };
            // The callee takes the effect as the caller has it: a place in
            // contract state, or a value.
            match target {
                Target::Place(lean, lane) if !lane.is_some_and(|lane| lane.partial) => {
                    write!(lean_args, " {lean}").unwrap();
                    effects.push((effect.provider, Carrier::Place));
                }
                Target::Value(target) => {
                    write!(lean_args, " {}", self.read(&target)?).unwrap();
                    effects.push((effect.provider, Carrier::Value));
                    if effect.mutable {
                        outs.push(Out::Value(target));
                    }
                }
                // A packed lane is passed as the value it holds, and a `mut`
                // effect writes back the value it leaves.
                Target::Place(lean, Some(lane)) => {
                    let NEffectArgValue::Place(place) = &arg.arg else {
                        return Err(Unsupported("lane behind an access".to_string()));
                    };
                    let held = format!("{name}_e{}", effect.provider);
                    let load = self.load(place.ty, &lean, Some(lane))?;
                    self.located.push(format!("let {held} ← {load}"));
                    write!(lean_args, " {held}").unwrap();
                    effects.push((effect.provider, Carrier::Value));
                    if effect.mutable {
                        outs.push(Out::Lane(place.ty, lean, lane));
                    }
                }
                _ => return Err(Unsupported("effect argument in memory".to_string())),
            }
        }
        let carriers = Carriers {
            params: carriers,
            effects,
        };
        let callee = self.module.instance(callee, carriers)?;
        if self.module.nevers.contains(&callee) {
            return Ok(format!("let {name} ← {callee}{lean_args}\nnever"));
        }
        // A projection that returns its slide: the slide runs where the
        // session ends, and the values it returns are written back there.
        if self.module.lenses.contains(&callee) {
            self.slides.insert(result, outs);
            return Ok(if self.sessions.contains(&result) {
                format!("let ({name}_g, {name}_k) ← {callee}{lean_args}\n{name} := {name}_g")
            } else {
                format!("let ({name}, {name}_k) ← {callee}{lean_args}")
            });
        }
        if outs.is_empty() {
            return Ok(format!("let {name} ← {callee}{lean_args}"));
        }
        let returned: String = (0..outs.len()).map(|n| format!(", {name}_{n}")).collect();
        let mut lines = format!("let ({name}{returned}) ← {callee}{lean_args}");
        for (n, out) in outs.iter().enumerate() {
            let value = format!("{name}_{n}");
            let line = match out {
                Out::Value(target) => self.write(target, &value)?,
                Out::Lane(ty, place, lane) => self.store(*ty, place, Some(*lane), &value)?,
            };
            write!(lines, "\n{line}").unwrap();
        }
        Ok(lines)
    }

    /// A call of the builtin `kind`: the prelude's definition of it. One that
    /// never returns ends in a `never` line.
    fn builtin(
        &mut self,
        kind: RuntimeBuiltinFuncKind,
        result: NValueId,
        callee: SemanticInstanceKey<'db>,
        args: &[NOperand],
    ) -> Result<String, Unsupported> {
        use RuntimeBuiltinFuncKind as Kind;
        let db = self.module.db;
        let name = value_name(result);
        let evm = db.compilation_settings().target(db) == CompilationTarget::Evm;
        let operands: Result<Vec<_>, _> = args.iter().map(|arg| self.operand(arg)).collect();
        let operands = operands?;
        let applied: String = operands.iter().map(|arg| format!(" {arg}")).collect();
        // The type an integer builtin works at.
        let int = args
            .first()
            .and_then(|arg| self.module.int_bits(self.ty(arg.value)))
            .map(int_ty);
        let action = match kind {
            Kind::Malloc | Kind::TryMalloc if evm => "alloc",
            Kind::Mload => "mload",
            Kind::Mstore => "mstore",
            Kind::Mstore8 => "mstore8",
            Kind::Mcopy => "mcopy",
            Kind::ZeroMem => "zeroMem",
            Kind::ReadPaddedWord => "readPaddedWord",
            Kind::Sload | Kind::SloadHashed => "sload",
            Kind::Sstore | Kind::SstoreHashed => "sstore",
            Kind::CallDataLoad => "calldataload",
            Kind::CallDataCopy => "calldatacopy",
            Kind::CallDataSize => "calldatasize",
            Kind::CodeCopy => "codecopy",
            Kind::CodeSize => "codesize",
            Kind::ExtCodeSize => "extcodesize",
            Kind::Keccak256 => "keccak256",
            // External calls, and the gas a model does not have: what the
            // prelude's feed supplies.
            Kind::Gas => "gas",
            Kind::Call => "Fe.EVM.call",
            Kind::StaticCall => "staticcall",
            Kind::StaticCallPrecompile => "staticcallPrecompile",
            Kind::DelegateCall => "delegatecall",
            Kind::Create => "Fe.EVM.create",
            Kind::Create2 => "create2",
            Kind::ReturnDataSize => "returndatasize",
            Kind::ReturnDataCopy => "returndatacopy",
            Kind::ExtCodeHash => "extcodehash",
            Kind::ExtCodeCopy => "extcodecopy",
            Kind::Address => "selfAddress",
            Kind::Caller => "caller",
            Kind::CallValue => "callvalue",
            Kind::Origin => "origin",
            Kind::GasPrice => "gasprice",
            Kind::CoinBase => "coinbase",
            Kind::Balance => "balance",
            Kind::Timestamp => "timestamp",
            Kind::Number => "blockNumber",
            Kind::PrevRandao => "prevrandao",
            Kind::GasLimit => "gaslimit",
            Kind::ChainId => "chainid",
            Kind::BaseFee => "basefee",
            Kind::SelfBalance => "selfbalance",
            Kind::BlockHash => "blockhash",
            Kind::BlobHash => "blobhash",
            Kind::BlobBaseFee => "blobbasefee",
            Kind::Revert => "revertData",
            Kind::RevertEmpty => "revertEmpty",
            Kind::ReturnData => "returnData",
            Kind::Stop => "stopCall",
            Kind::Panic | Kind::Todo => "trap",
            Kind::PanicCode => "panicCode",
            _ => "",
        };
        let never = matches!(
            kind,
            Kind::Revert
                | Kind::RevertEmpty
                | Kind::ReturnData
                | Kind::Stop
                | Kind::Panic
                | Kind::Todo
                | Kind::PanicCode
        );
        if !action.is_empty() {
            let never = if never { "\nnever" } else { "" };
            return Ok(format!("let {name} ← {action}{applied}{never}"));
        }
        // `clear` zeroes a place: its slots in contract state, or the value a
        // local or a held lane has.
        if kind == Kind::Clear
            && let [place] = args
            && let Some((_, inner)) = self.ty(place.value).as_capability(db)
        {
            return Ok(match self.carrier(place.value)? {
                Carrier::Place => format!(
                    "let {name} ← clearSlots {} {}",
                    value_name(place.value),
                    self.module.span(inner)?
                ),
                Carrier::Value => {
                    let target = self.carrier_value(place.value)?;
                    format!("{}\nlet {name} := ()", self.write(&target, "default")?)
                }
                Carrier::Erased => format!("let {name} := ()"),
            });
        }
        let value = match (kind, operands.as_slice(), int) {
            (
                Kind::Log0 | Kind::Log1 | Kind::Log2 | Kind::Log3 | Kind::Log4,
                [ptr, len, topics @ ..],
                _,
            ) => {
                return Ok(format!(
                    "let {name} ← logData {ptr} {len} #[{}]",
                    topics.join(", ")
                ));
            }
            (Kind::PtrOffsetBytes, [ptr, offset], _) => format!("{ptr} + {offset}"),
            (Kind::PtrEq, [lhs, rhs], _) => format!("decide ({lhs} = {rhs})"),
            // EVM memory is never reclaimed, and a forgotten value is not
            // destroyed.
            (Kind::Free, [_], _) if evm => "()".to_string(),
            (Kind::Forget, [_], _) => "()".to_string(),
            // EVM memory is big-endian, so an integer is its own big-endian
            // representation there.
            (Kind::ToBigEndian | Kind::FromBigEndian | Kind::BlackBox, [value], _) if evm => {
                value.clone()
            }
            (
                Kind::ToLittleEndian | Kind::FromLittleEndian | Kind::SwapBytes,
                [value],
                Some(int),
            ) if evm => format!("IntTy.swapBytes {int} {value}"),
            (Kind::LeadingZeros, [value], Some(int)) => {
                format!("IntTy.leadingZeros {int} {value}")
            }
            (Kind::TrailingZeros, [value], Some(int)) => {
                format!("IntTy.trailingZeros {int} {value}")
            }
            (Kind::AddMod, [a, b, modulus], _) => format!("UInt256.addMod {a} {b} {modulus}"),
            (Kind::MulMod, [a, b, modulus], _) => format!("UInt256.mulMod {a} {b} {modulus}"),
            (Kind::Byte, [pos, value], _) => format!("UInt256.byteAt {pos} {value}"),
            (Kind::SignExtend, [byte, value], _) => format!("UInt256.signextend {byte} {value}"),
            (Kind::IntrinsicKeccakWords, [words], _) => format!("keccakWords {words}"),
            (Kind::NeedsDrop, [], _) => {
                let ty = callee.subst(db).generic_args(db)[0];
                ty_has_drop_glue(db, ty).to_string()
            }
            _ => return Err(Unsupported(format!("builtin {kind:?}"))),
        };
        Ok(format!("let {name} := {value}"))
    }

    /// A call of a numeric intrinsic: the operation it names, at the type of
    /// its operands.
    fn numeric(
        &mut self,
        kind: NumericExternIntrinsic,
        result: NValueId,
        args: &[NOperand],
    ) -> Result<String, Unsupported> {
        use NumericExternIntrinsic as Kind;
        let db = self.module.db;
        let name = value_name(result);
        let ty = args.first().map(|arg| self.ty(arg.value));
        let int = ty.and_then(|ty| self.module.int_bits(ty));
        let operands: Result<Vec<_>, _> = args.iter().map(|arg| self.operand(arg)).collect();
        let unsupported = || {
            Unsupported(format!(
                "numeric intrinsic on {:?}",
                ty.map(|ty| ty.pretty_print(db))
            ))
        };
        let binary_op = |op, overflow| BinaryOp::new(BinOp::Arith(op), overflow);
        let rhs = match (kind, operands?.as_slice(), int) {
            (Kind::CheckedBinary(op), [lhs, rhs], Some(int)) => {
                let op = binary_op(op, Overflow::Checked).ok_or_else(unsupported)?;
                binary(op, int, lhs.clone(), rhs.clone())?
            }
            (Kind::WrappingBinary(op), [lhs, rhs], Some(int)) => {
                let op = binary_op(op, Overflow::Wrapping).ok_or_else(unsupported)?;
                binary(op, int, lhs.clone(), rhs.clone())?
            }
            (Kind::Comparison(op), [lhs, rhs], Some(int)) => {
                binary(BinaryOp::Comp(op), int, lhs.clone(), rhs.clone())?
            }
            (Kind::Comparison(CompBinOp::Eq), [lhs, rhs], None) => {
                format!(":= decide ({lhs} = {rhs})")
            }
            (Kind::Comparison(CompBinOp::NotEq), [lhs, rhs], None) => {
                format!(":= decide ({lhs} ≠ {rhs})")
            }
            (Kind::BoolBinary(ArithBinOp::BitAnd), [lhs, rhs], None) => {
                format!(":= {lhs} && {rhs}")
            }
            (Kind::BoolBinary(ArithBinOp::BitOr), [lhs, rhs], None) => {
                format!(":= {lhs} || {rhs}")
            }
            (Kind::BoolBinary(ArithBinOp::BitXor), [lhs, rhs], None) => {
                format!(":= {lhs} != {rhs}")
            }
            (Kind::BoolNot, [value], None) => unary(UnaryOp::Not, None, value)?,
            (Kind::BitNot, [value], Some(_)) => unary(UnaryOp::BitNot, int, value)?,
            (Kind::CheckedNeg, [value], Some(_)) => {
                unary(UnaryOp::Neg(Overflow::Checked), int, value)?
            }
            (Kind::SaturatingBinary(op), [lhs, rhs], Some(int)) => {
                let op = match op {
                    SaturatingArithmetic::Add => "satAdd",
                    SaturatingArithmetic::Sub => "satSub",
                    SaturatingArithmetic::Mul => "satMul",
                };
                format!(":= IntTy.{op} {} {lhs} {rhs}", int_ty(int))
            }
            _ => return Err(unsupported()),
        };
        Ok(format!("let {name} {rhs}"))
    }

    /// The lines of a call, which end in a `never` line if the callee never
    /// returns. The result of such a call has no values, yet the body may
    /// give it the type of the expression it stands in; the name is then
    /// rebound at that type.
    fn after_never(&mut self, result: NValueId, call: String) -> Result<String, Unsupported> {
        let Some(call) = call.strip_suffix("\nnever") else {
            return Ok(call);
        };
        self.never = Some(value_name(result));
        let ty = self.ty(result);
        if ty.is_never(self.module.db) || self.yielding {
            return Ok(call.to_string());
        }
        let name = value_name(result);
        Ok(format!(
            "{call}\nlet {name} : {} := nomatch {name}",
            self.lean_ty(ty)?
        ))
    }

    /// The enum value `operand` holds or, if it is an access, reaches, and
    /// the line that reads it from contract state first, if it lies there.
    fn enum_value(&mut self, operand: &NOperand) -> Result<(Option<String>, String), Unsupported> {
        let ty = self.ty(operand.value);
        if !self.is_access(ty) {
            return Ok((None, self.operand(operand)?));
        }
        match self.carrier(operand.value)? {
            Carrier::Value => Ok((None, self.read(&self.carrier_value(operand.value)?)?)),
            Carrier::Place => {
                let (_, inner) = ty.as_capability(self.module.db).unwrap();
                let name = format!("{}_e", value_name(operand.value));
                let load = self.load(inner, &value_name(operand.value), None)?;
                Ok((Some(format!("let {name} ← {load}")), name))
            }
            Carrier::Erased => Err(Unsupported("enum in the root object".to_string())),
        }
    }

    /// `operand` as a Lean term: its variable.
    fn operand(&self, operand: &NOperand) -> Result<String, Unsupported> {
        Ok(value_name(operand.value))
    }

    /// Whether a value of `ty` is an access to a place. A view of a snapshot
    /// type is the value.
    fn is_access(&self, ty: TyId<'db>) -> bool {
        let db = self.module.db;
        let owner = self.body.owner;
        let scope = owner.key(db).owner(db).scope();
        match ty.as_capability(db) {
            // A view of a type pinned to contract state is the place its
            // value lies at, though the type may be a snapshot.
            Some((CapabilityKind::View, inner)) => {
                !ty_is_snapshot(db, scope, inner, owner.assumptions(db))
                    || owner.pinned_space(db, inner).is_some()
            }
            Some(_) => true,
            None => false,
        }
    }

    /// How the model holds the access `value`: as a `Place` if it reaches
    /// contract state, and by the value there if it reaches a local.
    fn carrier(&self, value: NValueId) -> Result<Carrier, Unsupported> {
        if self.held(value).is_some() {
            return Ok(Carrier::Value);
        }
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
                _ if self.root_var(root).is_some() => Ok(Carrier::Value),
                NRootKind::Provider { .. } => Ok(Carrier::Place),
                kind => Err(Unsupported(format!("access to root {kind:?}"))),
            },
            CarrierBase::Param { param } => {
                let owner = self.body.owner.key(self.module.db).owner(self.module.db);
                let entry = matches!(
                    owner,
                    BodyOwner::ContractInit { .. } | BodyOwner::ContractRecvArm { .. }
                );
                let carrier = self.carriers.params.get(param as usize).copied().flatten();
                carrier
                    .or(entry.then_some(Carrier::Value))
                    .ok_or_else(|| Unsupported("access parameter without a carrier".to_string()))
            }
            // A grant into contract state is the place the projection
            // returned.
            CarrierBase::Grant { session, component }
                if self.granted_in_state(session, &component) =>
            {
                Ok(Carrier::Place)
            }
            // A `mut` grant of a local is the value the session holds.
            CarrierBase::Grant { session, component }
                if component.is_empty() && self.sessions.contains(&session) =>
            {
                Ok(Carrier::Value)
            }
            // A read-only grant is the value the projection returned.
            CarrierBase::Grant { session, .. }
                if !matches!(
                    self.ty(session).as_capability(self.module.db),
                    Some((CapabilityKind::Mut, _))
                ) =>
            {
                Ok(Carrier::Value)
            }
            // An access through a pointer into contract state reaches the
            // place at that slot.
            CarrierBase::Raw { pointer }
                if matches!(
                    self.ty(pointer).as_ptr(self.module.db),
                    Some((_, AddressSpace::Storage | AddressSpace::Transient))
                ) =>
            {
                Ok(Carrier::Place)
            }
            base => Err(Unsupported(format!("access to {base:?}"))),
        }
    }

    /// Checks that each access among `parts`, the parts of a value being
    /// built, reaches contract state: there it is a place, which a value can
    /// hold.
    fn parts_in_state(&self, parts: &[NOperand]) -> Result<(), Unsupported> {
        for part in parts {
            if self.is_access(self.ty(part.value)) && self.carrier(part.value)? != Carrier::Place {
                return Err(Unsupported(
                    "value that holds an access to a local".to_string(),
                ));
            }
        }
        Ok(())
    }

    /// Whether the access at `component` of what the projection call
    /// `session` grants reaches contract state.
    fn granted_in_state(&self, session: NValueId, component: &NDataPath) -> bool {
        let db = self.module.db;
        let NValueDefinition::Statement { block, statement } =
            self.body.values[session.index()].definition
        else {
            return false;
        };
        let statement = &self.body.blocks[block.index()].statements[statement as usize];
        let NStatementKind::Define {
            expr: NExpr::Call { callee, .. },
            ..
        } = &statement.kind
        else {
            return false;
        };
        let step = |step: &NDataProjection| match *step {
            NDataProjection::Field(field) => Some(ReturnProjectionStep::Field(field.0)),
            NDataProjection::VariantField { variant, field } => {
                Some(ReturnProjectionStep::VariantField {
                    variant: variant.0,
                    field: field.0,
                })
            }
            _ => None,
        };
        let component: Option<Vec<_>> = component.iter().map(step).collect();
        projection_component_spaces(db, SemanticInstance::new(db, callee.key))
            .iter()
            .any(|(steps, space)| {
                Some(steps) == component.as_ref()
                    && matches!(space, Some(AddressSpace::Storage | AddressSpace::Transient))
            })
    }

    /// The first borrow of the packed lane the access `value` reaches into,
    /// whose variable holds the lane's value, and the path below the lane.
    fn held(&self, value: NValueId) -> Option<(NValueId, NDataPath)> {
        if self.helds.is_empty() {
            return None;
        }
        let db = self.module.db;
        let target = self.body.carrier_target(db, value)?;
        self.helds.iter().find_map(|(held, ..)| {
            let lane = self.body.carrier_target(db, *held)?;
            (lane.base == target.base && lane.path.is_prefix_of(&target.path)).then(|| {
                let below = target.path.as_slice()[lane.path.len()..].to_vec();
                (*held, NDataPath::new(below))
            })
        })
    }

    /// The local value the access `value` reaches.
    fn carrier_value(&self, value: NValueId) -> Result<ValueTarget<'db>, Unsupported> {
        let db = self.module.db;
        if let Some((held, path)) = self.held(value) {
            let place = self.helds.iter().find(|(borrow, ..)| *borrow == held);
            return Ok(ValueTarget {
                var: format!("h{}", held.index()),
                ty: place.unwrap().1.ty,
                path: field_path(&path)?,
            });
        }
        let target = self
            .body
            .carrier_target(db, value)
            .ok_or_else(|| Unsupported("access without one target".to_string()))?;
        let several = || Unsupported("access below a value of several accesses".to_string());
        let (var, ty) = match target.base {
            CarrierBase::Root(root) => (
                self.root_var(root)
                    .ok_or_else(|| Unsupported("access to a root with no value".to_string()))?,
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
                let (_, inner) = value.ty.as_capability(db).ok_or_else(several)?;
                (format!("v{index}"), inner)
            }
            CarrierBase::Grant { session, component } => {
                let (_, inner) = self.ty(session).as_capability(db).ok_or_else(several)?;
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

    /// The mutable variable that holds the value of `root`: a local's slot,
    /// a parameter's, or an effect received by value.
    fn root_var(&self, root: NRootId) -> Option<String> {
        match &self.body.roots[root.index()].kind {
            NRootKind::LocalSlot { .. } | NRootKind::ParamPlace { .. } => {
                Some(format!("r{}", root.index()))
            }
            // A temporary is the value it was made for.
            NRootKind::Temporary { value } => Some(value_name(*value)),
            NRootKind::Provider { binding } => self
                .effects
                .iter()
                .any(|effect| {
                    effect.provider == binding.provider_idx
                        && matches!(binding.source, ProviderSource::UsesParam { .. })
                })
                .then(|| format!("e{}", binding.provider_idx)),
            _ => None,
        }
    }

    /// The value of a local that `place` is below, if it is.
    fn local(&self, place: &NPlace<'db>) -> Result<Option<ValueTarget<'db>>, Unsupported> {
        Ok(match place.base {
            NPlaceBase::Root(root) => self.root_var(root).map(|var| ValueTarget {
                var,
                ty: self.body.roots[root.index()].ty,
                path: Vec::new(),
            }),
            // A value of several accesses holds places.
            NPlaceBase::CapabilityTarget { carrier }
                if self.ty(carrier).as_capability(self.module.db).is_none() =>
            {
                None
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
        })
    }

    /// Whether `place` lies in contract state.
    fn in_state(&self, place: &NPlace<'db>) -> Result<bool, Unsupported> {
        let memory = matches!(place.base, NPlaceBase::Deref { pointer }
            if matches!(self.ty(pointer).as_ptr(self.module.db), Some((_, AddressSpace::Memory))));
        Ok(!memory && self.local(place)?.is_none())
    }

    /// What `place` is in the model: a `Place` in contract state, a value
    /// below a local's variable, or a word in memory.
    fn target(&mut self, place: &NPlace<'db>) -> Result<Target<'db>, Unsupported> {
        let db = self.module.db;
        if let NPlaceBase::Deref { pointer } = place.base
            && let Some((_, AddressSpace::Memory)) = self.ty(pointer).as_ptr(db)
        {
            // The parts of a value in memory follow one another, so a part
            // lies the sizes of those before it after the value's address.
            let untyped = || Unsupported("untyped place".to_string());
            let mut address = value_name(pointer);
            let mut offset = 0;
            for (len, projection) in place.path.iter().enumerate() {
                let ty = self
                    .body
                    .place_prefix_ty(db, place, len)
                    .ok_or_else(untyped)?;
                match projection {
                    NDataProjection::Field(index) => {
                        for field in &ty.field_types(db)[..usize::from(index.0)] {
                            offset += self.module.size(*field)?;
                        }
                    }
                    NDataProjection::Index(index) => {
                        let part = self
                            .body
                            .place_prefix_ty(db, place, len + 1)
                            .ok_or_else(untyped)?;
                        let stride = self.module.size(part)?;
                        match index {
                            NIndex::Const(index) => offset += *index as u64 * stride,
                            NIndex::Value(index) => {
                                address = format!(
                                    "({address} + {} * UInt256.ofNat {stride})",
                                    value_name(*index)
                                );
                            }
                        }
                    }
                    other => return Err(Unsupported(format!("step {other:?} in memory"))),
                }
            }
            if offset > 0 {
                address = format!("({address} + UInt256.ofNat {offset})");
            }
            return Ok(Target::Memory(address));
        }
        let Some(mut local) = self.local(place)? else {
            let (lean, lane) = self.place(place)?;
            return Ok(Target::Place(lean, lane));
        };
        local.path.extend(field_path(&place.path)?);
        Ok(Target::Value(local))
    }

    /// The Lean term for the value at `target`.
    fn read(&self, target: &ValueTarget<'db>) -> Result<String, Unsupported> {
        let mut lean = target.var.clone();
        let mut ty = target.ty;
        for &step in &target.path {
            let (names, part) = self.module.part(ty, step)?;
            for name in names {
                match step {
                    Step::Index(_) => write!(lean, "[{name}]!").unwrap(),
                    _ => write!(lean, ".{name}").unwrap(),
                }
            }
            ty = part;
        }
        Ok(lean)
    }

    /// The assignment that leaves `value` at `target`.
    fn write(&self, target: &ValueTarget<'db>, value: &str) -> Result<String, Unsupported> {
        let mut prefix = target.var.clone();
        let mut ty = target.ty;
        let mut open = String::new();
        let mut close = String::new();
        for &step in &target.path {
            let (steps, part) = self.module.part(ty, step)?;
            for name in steps {
                match step {
                    Step::Field(_) => {
                        write!(open, "{{ {prefix} with {name} := ").unwrap();
                        close.insert_str(0, " }");
                    }
                    Step::Variant { .. } => {
                        write!(open, "({prefix}.set_{name} ").unwrap();
                        close.insert(0, ')');
                    }
                    Step::Index(_) => {
                        write!(open, "({prefix}.set! {name} ").unwrap();
                        close.insert(0, ')');
                    }
                }
                match step {
                    Step::Index(_) => write!(prefix, "[{name}]!").unwrap(),
                    _ => write!(prefix, ".{name}").unwrap(),
                }
            }
            ty = part;
        }
        Ok(format!("{} := {open}{value}{close}", target.var))
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

    /// The Lean place `place` names, and the lane of its slot that an access
    /// by this name reads and writes, if it ends at a packed field. Its base,
    /// then a step for each field, variant field, element and `StorageMap`
    /// key; a part pinned to transient storage lies there at the slot number
    /// it has.
    fn place(&mut self, place: &NPlace<'db>) -> Result<(String, Option<Lane>), Unsupported> {
        let db = self.module.db;
        let untyped = || Unsupported("untyped place".to_string());
        let base_ty = self
            .body
            .place_prefix_ty(db, place, 0)
            .ok_or_else(untyped)?;
        // A value of several accesses holds a place for each: the steps to
        // one of them select it, and the rest go on below that place.
        let mut selected = 0;
        let mut base_ty = base_ty;
        let mut selecting = None;
        if let NPlaceBase::CapabilityTarget { carrier } = place.base
            && self.ty(carrier).as_capability(db).is_none()
        {
            let mut ty = self.ty(carrier);
            let mut lean = value_name(carrier);
            while ty.as_capability(db).is_none() {
                let step = place.path.as_slice().get(selected).ok_or_else(untyped)?;
                let step = field_path(&NDataPath::new(vec![*step]))?[0];
                let (names, part) = self.module.part(ty, step)?;
                lean = names
                    .iter()
                    .fold(lean, |lean, name| format!("{lean}.{name}"));
                ty = part;
                selected += 1;
            }
            base_ty = ty.as_capability(db).map_or(ty, |(_, inner)| inner);
            selecting = Some(lean);
        }
        let (base, mut lane) = match place.base {
            NPlaceBase::Root(root) => self.root(root)?,
            NPlaceBase::CapabilityTarget { carrier } => {
                (selecting.unwrap_or_else(|| value_name(carrier)), None)
            }
            // A pointer into contract state is a slot number.
            NPlaceBase::Deref { pointer } => {
                let space = match self.ty(pointer).as_ptr(db) {
                    Some((_, AddressSpace::Storage)) => "storage",
                    Some((_, AddressSpace::Transient)) => "transient",
                    _ => return Err(Unsupported(format!("place base {:?}", place.base))),
                };
                (format!("(Place.mk .{space} {})", value_name(pointer)), None)
            }
        };
        let mut lean = self.module.pinned(base_ty, base);
        for (len, projection) in place.path.iter().enumerate().skip(selected) {
            let ty = self
                .body
                .place_prefix_ty(db, place, len)
                .ok_or_else(untyped)?;
            let part = self
                .body
                .place_prefix_ty(db, place, len + 1)
                .ok_or_else(untyped)?;
            // A step below an entry packed into a lane names a part of the
            // entry's value.
            if let Some(codec) = lane.and_then(|lane| lane.codec) {
                let step = field_path(&NDataPath::new(vec![*projection]))?;
                self.codecs[codec].4.extend(step);
                continue;
            }
            (lean, lane) = match projection {
                // The field of a newtype lies where the newtype does.
                NDataProjection::Field(_) if let Some((width, _)) = self.module.newtype(ty) => {
                    let own = Lane {
                        off: 0,
                        width,
                        shared: false,
                        partial: false,
                        codec: None,
                    };
                    (lean, Some(lane.unwrap_or(own)))
                }
                NDataProjection::Field(index) => {
                    let placements = storage_field_placements(db, ty).unwrap_or_default();
                    let lane = placements
                        .get(usize::from(index.0))
                        .and_then(|placement| self.module.lane(part, *placement));
                    let layout = self.module.struct_layout(ty)?;
                    (format!("({lean}.field {layout} {})", index.0), lane)
                }
                // The tag has the enum's first slot, and each field of a
                // variant whole slots after it.
                NDataProjection::VariantField { variant, field } => {
                    let variants = self.module.variants(ty).ok_or_else(untyped)?;
                    let (_, fields) = &variants[usize::from(variant.0)];
                    let mut slot = 1;
                    for field in &fields[..usize::from(field.0)] {
                        slot += self.module.span(*field)?;
                    }
                    (format!("({lean}.offset {slot})"), None)
                }
                NDataProjection::Index(index) => {
                    let index = match index {
                        NIndex::Const(index) => format!("(UInt256.ofNat {index})"),
                        NIndex::Value(value) => value_name(*value),
                    };
                    (
                        format!("({lean}.index {} {index})", self.module.span(part)?),
                        None,
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
                    if map && self.lean_ty(self.ty(*key))? == "UInt256" {
                        (format!("({lean}.entry {})", value_name(*key)), None)
                    } else {
                        self.entry(ty, &lean, *key)?
                    }
                }
            };
            lean = self.module.pinned(part, lean);
        }
        Ok((lean, lane))
    }

    /// The computation that reads a value of `ty` at `place` in contract
    /// state. An entry a collection packs into a lane is read by the
    /// collection's codec from the word there.
    fn load(
        &mut self,
        ty: TyId<'db>,
        place: &str,
        lane: Option<Lane>,
    ) -> Result<String, Unsupported> {
        match lane.and_then(|lane| lane.codec) {
            Some(codec) => {
                let (offset, get, _, entry, path) = self.codecs[codec].clone();
                let part = self.read(&ValueTarget {
                    var: "entry".to_string(),
                    ty: entry,
                    path,
                })?;
                Ok(format!(
                    "(do let entry ← {get} (← load {place}) {offset}; pure {part})"
                ))
            }
            None => self.module.load(ty, place, lane),
        }
    }

    /// The computation that writes `value` of `ty` at `place` in contract
    /// state. The codec of a packed entry puts the value into the word
    /// there, which is then stored.
    fn store(
        &mut self,
        ty: TyId<'db>,
        place: &str,
        lane: Option<Lane>,
        value: &str,
    ) -> Result<String, Unsupported> {
        match lane.and_then(|lane| lane.codec) {
            // A part of a packed entry is written into the entry as read.
            Some(codec) => {
                let (offset, get, set, entry, path) = self.codecs[codec].clone();
                let word = format!("(← load {place})");
                if path.is_empty() {
                    return Ok(format!(
                        "(do store {place} (← {set} {word} {offset} {value}))"
                    ));
                }
                let target = ValueTarget {
                    var: "entry".to_string(),
                    ty: entry,
                    path,
                };
                Ok(format!(
                    "(do let mut entry ← {get} {word} {offset}; {}; \
                     store {place} (← {set} {word} {offset} entry))",
                    self.write(&target, value)?
                ))
            }
            None => self.module.store(ty, place, lane, value),
        }
    }

    /// The place of entry `key` of the collection of type `ty` at
    /// `collection`: where the collection's `StateIndex::locate` puts it, in
    /// the state space the collection is pinned to. The compiler calls
    /// `locate` with the collection, the collection's own slot and the key.
    fn entry(
        &mut self,
        ty: TyId<'db>,
        collection: &str,
        key: NValueId,
    ) -> Result<(String, Option<Lane>), Unsupported> {
        let db = self.module.db;
        let unsupported = |what: &str| Unsupported(format!("{what} of {}", ty.pretty_print(db)));
        let entry = self
            .body
            .owner
            .entry_locate(db, ty)
            .map_err(|_| unsupported("entry"))?;
        let space = match entry.space {
            Some(AddressSpace::Storage) => "storage",
            Some(AddressSpace::Transient) => "transient",
            _ => return Err(unsupported("entry space")),
        };
        if self.is_access(self.ty(key)) {
            return Err(unsupported("entry key by access"));
        }
        // `locate` takes the collection as a place, and a key it views by
        // value.
        let callee = SemanticInstance::new(db, entry.callee);
        let normalized = normalize_semantic_body(db, callee).map_err(|_| unsupported("entry"))?;
        let scope = entry.callee.owner(db).scope();
        let mut carriers = vec![None; 3];
        for value in &normalized.body.values {
            if let NValueDefinition::EntryParam { param } = value.definition
                && let Some((kind, inner)) = value.ty.as_capability(db)
                && (kind != CapabilityKind::View
                    || !ty_is_snapshot(db, scope, inner, callee.assumptions(db)))
                && let Some(carrier) = carriers.get_mut(param as usize)
            {
                *carrier = Some(if param == 0 {
                    Carrier::Place
                } else {
                    Carrier::Value
                });
            }
        }
        let locate = self.module.instance(entry.callee, carriers.into())?;
        let location = format!("l{}", self.entries);
        self.entries += 1;
        self.located.push(format!(
            "let {location} ← {locate} {collection} {collection}.slot {}",
            value_name(key)
        ));
        let place = format!("(Place.mk .{space} {location}.slot)");
        let Some(lanes) = entry.lanes else {
            return Ok((place, None));
        };
        // The collection packs its entries: this one is the bits its codec
        // reads at the location's offset in the word there.
        let scope = self.body.owner.key(db).owner(db).scope();
        let mut codec = |method| {
            let key = core_method_callee_key(
                db,
                scope,
                self.body.owner.assumptions(db),
                &["ops", "LaneCodec"],
                method,
                vec![lanes.codec, entry.entry_ty],
            )
            .map_err(|_| unsupported("entry lane"))?;
            self.module.instance(key, Carriers::default())
        };
        let (get, set) = (codec("get")?, codec("set")?);
        self.codecs.push((
            format!("{location}.offset"),
            get,
            set,
            entry.entry_ty,
            Vec::new(),
        ));
        let lane = Lane {
            off: 0,
            width: 0,
            shared: true,
            partial: true,
            codec: Some(self.codecs.len() - 1),
        };
        Ok((place, Some(lane)))
    }

    /// The place of the contract field `root` provides, and its lane if it
    /// packs into its slot with other fields.
    fn root(&mut self, root: NRootId) -> Result<(String, Option<Lane>), Unsupported> {
        let db = self.module.db;
        let root = &self.body.roots[root.index()];
        let NRootKind::Provider { binding } = &root.kind else {
            return Err(Unsupported(format!("root {:?}", root.kind)));
        };
        let field = match binding.source {
            ProviderSource::ContractField { field } => field,
            // A place the caller passes.
            ProviderSource::UsesParam { .. } => {
                return Ok((format!("e{}", binding.provider_idx), None));
            }
            ProviderSource::RootProvider { .. } => {
                return Err(Unsupported("place in the root object".to_string()));
            }
        };
        let layout = field.contract.storage_layout(db);
        let layout = layout
            .values()
            .find(|layout| layout.field == field)
            .ok_or_else(|| Unsupported("contract field without a layout".to_string()))?;
        let name = layout.name.data(db);
        if layout.address_space != AddressSpace::Storage {
            return Err(Unsupported(format!("layout of contract field `{name}`")));
        }
        let lane = layout.lane.filter(|lane| lane.partial).map(|lane| Lane {
            off: u32::from(lane.byte_offset),
            width: u32::from(lane.byte_width),
            shared: lane.shared,
            partial: true,
            codec: None,
        });
        let slot = layout.slot_offset;
        let place = self.module.layout(format!("Field.{name}"), |name| {
            format!("def {name} : Place :=\n  .storage (UInt256.ofNat {slot})")
        });
        Ok((place, lane))
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

/// The steps of `path`, each to a field of a struct, tuple or variant, or to
/// an element of an array.
fn field_path(path: &NDataPath) -> Result<Vec<Step>, Unsupported> {
    path.iter()
        .map(|projection| match *projection {
            NDataProjection::Field(index) => Ok(Step::Field(usize::from(index.0))),
            NDataProjection::VariantField { variant, field } => Ok(Step::Variant {
                variant: usize::from(variant.0),
                field: usize::from(field.0),
            }),
            NDataProjection::Index(index) => Ok(Step::Index(index)),
            other => Err(Unsupported(format!("step {other:?} in a local"))),
        })
        .collect()
}

/// `name` as a Lean identifier, quoted if Lean reserves it.
fn lean_name(name: &str) -> String {
    const RESERVED: &[&str] = &[
        "abbrev",
        "at",
        "axiom",
        "break",
        "by",
        "calc",
        "catch",
        "class",
        "continue",
        "def",
        "deriving",
        "do",
        "else",
        "end",
        "example",
        "exists",
        "export",
        "extends",
        "finally",
        "for",
        "forall",
        "from",
        "fun",
        "have",
        "if",
        "import",
        "in",
        "inductive",
        "instance",
        "let",
        "local",
        "macro",
        "match",
        "mut",
        "namespace",
        "nomatch",
        "notation",
        "opaque",
        "open",
        "partial",
        "private",
        "protected",
        "return",
        "section",
        "show",
        "structure",
        "suffices",
        "syntax",
        "then",
        "theorem",
        "this",
        "to",
        "try",
        "universe",
        "unless",
        "unsafe",
        "using",
        "variable",
        "where",
        "with",
    ];
    if RESERVED.contains(&name) {
        format!("«{name}»")
    } else {
        name.to_string()
    }
}

/// The parameters `binders` of a definition, as Lean prints them.
fn binders(binders: &[(String, String)]) -> String {
    binders
        .iter()
        .map(|(name, ty)| format!(" ({name} : {ty})"))
        .collect()
}

/// Whether `text` names the variable `name`.
fn mentions(text: &str, name: &str) -> bool {
    let part = |c: char| c.is_alphanumeric() || c == '_';
    text.match_indices(name).any(|(at, _)| {
        !text[..at].chars().next_back().is_some_and(part)
            && !text[at + name.len()..].chars().next().is_some_and(part)
    })
}

/// The width of the tag of an enum of `variants` variants.
fn tag_bits(variants: usize) -> usize {
    match variants {
        0..=0x100 => 8,
        0x101..=0x1_0000 => 16,
        _ => 32,
    }
}

/// The Lean term for the integer type of `bits` bits, signed or not.
fn int_ty((bits, signed): (usize, bool)) -> String {
    format!("(.{} {bits})", if signed { 'i' } else { 'u' })
}

/// The right-hand side of `let v` for the unary operation `op` on `value`,
/// an integer of the width and signedness `int`, or a `bool`.
fn unary(op: UnaryOp, int: Option<(usize, bool)>, value: &str) -> Result<String, Unsupported> {
    Ok(match (op, int) {
        (UnaryOp::Not, None) => format!(":= !{value}"),
        (UnaryOp::BitNot, Some((256, false))) => format!(":= ~~~{value}"),
        (UnaryOp::BitNot, Some(int)) => format!(":= IntTy.not {} {value}", int_ty(int)),
        (UnaryOp::Neg(Overflow::Checked), Some(int)) => {
            format!("← IntTy.neg {} {value}", int_ty(int))
        }
        (UnaryOp::Neg(Overflow::Wrapping), Some(int)) => {
            format!(":= IntTy.negW {} {value}", int_ty(int))
        }
        _ => return Err(Unsupported(format!("operator {op:?}"))),
    })
}

/// The right-hand side of `let v` for a built-in operation on integers of
/// `bits` bits, signed or not, each a canonical word. `u256` has its own
/// definitions in the prelude; an operation whose result depends on the width
/// or the sign is otherwise the one of the prelude's `IntTy`.
fn binary(
    op: BinaryOp,
    (bits, signed): (usize, bool),
    lhs: String,
    rhs: String,
) -> Result<String, Unsupported> {
    let word = bits == 256 && !signed;
    let ty = int_ty((bits, signed));
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
        BinaryOp::Add(Overflow::Checked) => format!("← IntTy.add {ty} {lhs} {rhs}"),
        BinaryOp::Add(Overflow::Wrapping) => format!(":= IntTy.addW {ty} {lhs} {rhs}"),
        BinaryOp::Sub(Overflow::Checked) => format!("← IntTy.sub {ty} {lhs} {rhs}"),
        BinaryOp::Sub(Overflow::Wrapping) => format!(":= IntTy.subW {ty} {lhs} {rhs}"),
        BinaryOp::Mul(Overflow::Checked) => format!("← IntTy.mul {ty} {lhs} {rhs}"),
        BinaryOp::Mul(Overflow::Wrapping) => format!(":= IntTy.mulW {ty} {lhs} {rhs}"),
        BinaryOp::Div(Overflow::Checked) => format!("← IntTy.div {ty} {lhs} {rhs}"),
        BinaryOp::Div(Overflow::Wrapping) => format!("← IntTy.divW {ty} {lhs} {rhs}"),
        BinaryOp::Rem => format!("← IntTy.rem {ty} {lhs} {rhs}"),
        BinaryOp::Pow(Overflow::Checked) => format!("← IntTy.pow {ty} {lhs} {rhs}"),
        BinaryOp::Pow(Overflow::Wrapping) => format!(":= IntTy.powW {ty} {lhs} {rhs}"),
        BinaryOp::Shl => format!(":= IntTy.shl {ty} {lhs} {rhs}"),
        BinaryOp::Shr => format!(":= IntTy.shr {ty} {lhs} {rhs}"),
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
    })
}
