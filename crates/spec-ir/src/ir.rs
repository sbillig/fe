//! Serializes a contract's normalized semantic IR as data, for the Lean
//! interpreter of fe-spec: every normalized body the contract's entries
//! reach, the callees without a body by their stable names, the types those
//! bodies name with the layout facts the compiler computes for them, the
//! contract's fields, and the plan of the code the compiler synthesizes
//! around `init` and the recv arms.
//!
//! The output is structural: each body is `ir.rs`'s `NormalizedBody` with
//! its compiler-interned objects resolved into tables. What a construct does
//! is the interpreter's to say. A construct this serializer has no encoding
//! for is reported by name, never skipped.
//!
//! Every type and function is a table entry under its key (`key.rs`), and
//! is referred to by it. The entries whose keys name no item of the
//! contract's ingot form the library, which contracts share; the others are
//! the contract's own.

use std::collections::{BTreeMap, VecDeque};

use common::layout::StorageFieldPlacement;
use cranelift_entity::EntityRef;
use hir::{
    analysis::{
        HirAnalysisDb,
        semantic::{
            BinaryOp, CallSiteId, DropFlagPoint, DropStep, EvalOutcome, FailReason, NBlock,
            NDataPath, NDataProjection, NDropTarget, NEffectArg, NExpr, NIndex, NOperand, NPlace,
            NPlaceBase, NRoot, NRootKind, NStatement, NStatementKind, NSuccessor, NTerminatorKind,
            NValueDefinition, NormalizedBody, Overflow, ReadMode, SConst, SemConstId,
            SemConstScalar, SemConstValue, SemanticCodeRegionRef, SemanticCodeRegionTarget,
            SemanticInstance, SemanticInstanceKey, StructuralRepack, UnaryOp, ViewAccess,
            core_method_callee_key, drop_elaboration, eval_const_ref, fixed_string_capacity_bytes,
            identity_semantic_instance_key, normalize_semantic_body, owner_effect_bindings,
            reify_runtime_const_for_ty, resolved_provider_binding_for_instance_effect,
            runtime_size_bytes, semantic_may_return,
        },
        ty::{
            AddressSpace, ProviderKind,
            adt_def::AdtRef,
            corelib::{
                ContractMetadataKind, NumericExternIntrinsic, RuntimeBuiltinFuncKind,
                SaturatingArithmetic, core_numeric_const_intrinsic, lib_func_matches,
                runtime_builtin_func_kind,
            },
            drop_glue::{adt_drop_impl, implements_drop, ty_has_drop_glue},
            ty_check::{BodyOwner, LocalBinding},
            ty_def::{BorrowKind, CapabilityKind, PrimTy, TyBase, TyData, TyId, prim_int_bits},
        },
    },
    hir_def::{
        Contract, FieldParent,
        expr::{ArithBinOp, CompBinOp},
    },
    semantic::{
        ContractFieldId, ProviderBinding, ProviderSource, storage_field_lanes,
        storage_field_placements, storage_lane_bytes, storage_slot_span,
    },
};
use mir::{
    MirDb, RuntimeMemoryLayout, RuntimeMemoryOffsets, panic_payload_revert_key, runtime_needs_drop,
    stored_class_in_instance,
};
use rustc_hash::FxHashMap;
use serde_json::{Map, Value, json};

use crate::{
    Unsupported, arg_position, arm_callees, arm_names, contract_title, describe, effect_input,
    ingot_errors, init_callees, key::Keys,
};

/// The version of the format [`export_ir`] writes. A change to the format
/// changes it.
pub const IR_VERSION: u32 = 2;

/// A serialization of a contract: the entries of the library its code
/// reaches, and the contract's own entries with its plan.
#[derive(Debug, Clone, PartialEq)]
pub struct SpecIr {
    /// The types and functions whose keys name no item of the contract's
    /// ingot, sorted by key: `{version, target, types, funcs}`.
    pub library: Value,
    /// The other types and functions, sorted by key, and the contract plan:
    /// `{version, target, types, funcs, contract}`.
    pub contract: Value,
}

/// The normalized IR of `contract` and of everything its entries reach, as
/// JSON (`docs/design.md` of fe-spec has the schema). Every type and
/// function is an entry named by its key ([`crate::key`]), and every
/// reference to one is its key, `{"$ty": key}` or `{"$fn": key}`. A
/// contract of an ingot the compiler reports errors in is not serialized.
pub fn export_ir<'db>(db: &'db dyn MirDb, contract: Contract<'db>) -> Result<SpecIr, Unsupported> {
    let mir: &'db dyn MirDb = db;
    let errors = ingot_errors(db, contract.top_mod(db).ingot(db));
    if !errors.is_empty() {
        return Err(Unsupported(format!(
            "the program has errors: {}",
            errors.join("; ")
        )));
    }
    let db: &'db dyn HirAnalysisDb = db;
    let ingot = contract.top_mod(db).ingot(db);
    let root = SemanticInstance::new(
        db,
        identity_semantic_instance_key(db, BodyOwner::ContractInit { contract }),
    );
    let mut ir = IrExport {
        db,
        mir,
        contract,
        root,
        scope: root,
        keys: Keys::new(db, ingot),
        types: Table::default(),
        type_ids: FxHashMap::default(),
        funcs: Table::default(),
        func_ids: FxHashMap::default(),
        pending: VecDeque::new(),
    };
    let entry = ir.entry_plan()?;
    while let Some(instance) = ir.pending.pop_front() {
        let function = ir.function(instance)?;
        let key = ir.func_ids[&instance].clone();
        ir.funcs.insert(key, function)?;
    }
    let target = db.compilation_settings().target(db);
    let target = json!({
        "name": format!("{target:?}").to_lowercase(),
        "usizeBits": prim_int_bits(PrimTy::Usize, target),
    });
    let (library_types, types) = ir.types.split();
    let (library_funcs, funcs) = ir.funcs.split();
    let library = json!({
        "version": IR_VERSION,
        "target": target,
        "types": library_types,
        "funcs": library_funcs,
    });
    // The library is closed: none of its entries names one of the contract's.
    let mut refs = Vec::new();
    references(&library, &mut refs);
    let own = |table: &[Value], key: &str| table.iter().any(|entry| entry["key"] == key);
    if let Some((kind, key)) = refs
        .into_iter()
        .find(|(kind, key)| own(if *kind == "$ty" { &types } else { &funcs }, key))
    {
        return Err(Unsupported(format!(
            "the library refers to the contract's {kind} `{key}`"
        )));
    }
    Ok(SpecIr {
        library,
        contract: json!({
            "version": IR_VERSION,
            "target": target,
            "types": types,
            "funcs": funcs,
            "contract": entry,
        }),
    })
}

/// Each reference `{"$ty": key}` or `{"$fn": key}` in `value`, as the tag and the key.
fn references<'a>(value: &'a Value, refs: &mut Vec<(&'a str, &'a str)>) {
    match value {
        Value::Object(fields) => {
            if fields.len() == 1
                && let Some((tag, Value::String(key))) = fields.iter().next()
                && matches!(tag.as_str(), "$ty" | "$fn")
            {
                refs.push((tag, key));
                return;
            }
            fields.values().for_each(|value| references(value, refs));
        }
        Value::Array(values) => values.iter().for_each(|value| references(value, refs)),
        _ => {}
    }
}

/// Merges the library `other` into the library `into`, both [`SpecIr::library`]
/// outputs, possibly with more fields (such as a `provenance`) than
/// [`export_ir`] writes: their other fields must be equal, and an entry
/// both have must be the same in both. The result is sorted by key, so it
/// does not depend on the order libraries are merged in.
pub fn merge_library(into: &mut Value, other: &Value) -> Result<(), String> {
    let (Value::Object(into), Value::Object(other)) = (into, other) else {
        return Err("a library is a JSON object".into());
    };
    let fields = |object: &Map<String, Value>| {
        object
            .iter()
            .filter(|(field, _)| !matches!(field.as_str(), "types" | "funcs"))
            .map(|(field, value)| (field.clone(), value.clone()))
            .collect::<Map<_, _>>()
    };
    if fields(into) != fields(other) {
        return Err("the libraries differ in their version, target or provenance".into());
    }
    for table in ["types", "funcs"] {
        let mut entries = BTreeMap::new();
        for library in [&*into, other] {
            let Some(Value::Array(table_entries)) = library.get(table) else {
                return Err(format!("a library without a `{table}` table"));
            };
            for entry in table_entries {
                let key = entry["key"]
                    .as_str()
                    .ok_or_else(|| format!("an entry of `{table}` without a key"))?;
                if let Some(existing) = entries.insert(key.to_string(), entry.clone())
                    && existing != *entry
                {
                    return Err(format!("the libraries differ in `{key}`"));
                }
            }
        }
        into.insert(table.into(), Value::Array(entries.into_values().collect()));
    }
    Ok(())
}

/// A table of entries by key, each with whether it is the contract's own.
#[derive(Default)]
struct Table(BTreeMap<String, (bool, Value)>);

impl Table {
    /// Adds the entry `value` of `key`, which must be the same as the one the
    /// table has if it has one, and be the contract's own or not alike:
    /// instances and types under one key are one.
    fn insert(&mut self, (key, own): (String, bool), mut value: Value) -> Result<(), Unsupported> {
        value["key"] = json!(key);
        match self.0.get(&key) {
            Some((_, existing)) if *existing != value => Err(Unsupported(format!(
                "two entries under the key `{key}` differ:\n{existing}\n{value}"
            ))),
            Some((existing, _)) if *existing != own => Err(Unsupported(format!(
                "two entries under the key `{key}` are one the contract's, one the library's"
            ))),
            Some(_) => Ok(()),
            None => {
                self.0.insert(key, (own, value));
                Ok(())
            }
        }
    }

    /// The library's entries and the others, each sorted by key.
    fn split(self) -> (Vec<Value>, Vec<Value>) {
        let (own, library): (Vec<_>, Vec<_>) = self.0.into_values().partition(|(own, _)| *own);
        let values =
            |entries: Vec<(bool, Value)>| entries.into_iter().map(|(_, value)| value).collect();
        (values(library), values(own))
    }
}

/// `ir`, an [`export_ir`] output, as text: each top-level field on a line
/// of its own, and each entry of the type and function tables too, so that a
/// change to one body is a change to one line.
pub fn ir_text(ir: &Value) -> String {
    let Value::Object(fields) = ir else {
        return format!("{ir}\n");
    };
    let lines: Vec<_> = fields
        .iter()
        .map(|(key, value)| match value {
            Value::Array(entries) if matches!(key.as_str(), "types" | "funcs") => {
                let entries: Vec<_> = entries.iter().map(Value::to_string).collect();
                format!(
                    "{}: [\n{}\n]",
                    Value::from(key.as_str()),
                    entries.join(",\n")
                )
            }
            value => format!("{}: {value}", Value::from(key.as_str())),
        })
        .collect();
    format!("{{\n{}\n}}\n", lines.join(",\n"))
}

struct IrExport<'db> {
    db: &'db dyn HirAnalysisDb,
    /// The database, for what runtime lowering decides.
    mir: &'db dyn MirDb,
    contract: Contract<'db>,
    /// The contract's `init`, which the entry plan's types are normalized in.
    root: SemanticInstance<'db>,
    /// The instance whose types are being resolved: a type's entry (its fields, pinned space,
    /// drop impl and layouts) is resolved in the scope and assumptions of the instance that
    /// first names it.
    scope: SemanticInstance<'db>,
    keys: Keys<'db>,
    types: Table,
    /// The key of each type named so far, and whether it is the contract's own.
    type_ids: FxHashMap<TyId<'db>, (String, bool)>,
    funcs: Table,
    /// The key of each instance named so far, and whether it is the contract's own.
    func_ids: FxHashMap<SemanticInstanceKey<'db>, (String, bool)>,
    /// The instances named but not yet serialized.
    pending: VecDeque<SemanticInstanceKey<'db>>,
}

impl<'db> IrExport<'db> {
    /// The contract's fields and the plan of its entries.
    fn entry_plan(&mut self) -> Result<Value, Unsupported> {
        let db = self.db;
        let contract = self.contract;
        let layout = contract.storage_layout(db);
        let mut fields = Vec::new();
        for field in layout.values() {
            fields.push(json!({
                "name": field.name.data(db),
                "ty": self.ty(field.declared)?,
                "space": space(field.address_space),
                "mutable": field.is_mut,
                "slot": field.slot_offset.to_string(),
                "slotCount": field.slot_count,
                "lane": field.lane.map(|lane| json!({
                    "offset": lane.byte_offset,
                    "width": lane.byte_width,
                    "shared": lane.shared,
                    "partial": lane.partial,
                })),
            }));
        }
        let init_key = identity_semantic_instance_key(db, BodyOwner::ContractInit { contract });
        let init_decode = match init_callees(db, contract)? {
            Some(callees) => Some(json!({
                "decoderNew": self.func(callees.decoder_new),
                "decodePayload": self.func(callees.decode_payload),
                "spanTy": self.ty(callees.span_ty)?,
            })),
            None => None,
        };
        let init_args = contract.init_args_ty(db);
        // `init` takes the field of the decoded tuple each of its input bindings binds: a
        // parameter's own (`visible_init_arg_fields` in MIR).
        let init_fields = init_key
            .typed_body(db)
            .input_bindings()
            .iter()
            .map(|binding| match binding {
                LocalBinding::Param { idx, .. } => Ok(*idx),
                binding => Err(Unsupported(format!("init input binding {binding:?}"))),
            })
            .collect::<Result<Vec<_>, _>>()?;
        let init = json!({
            "func": self.func(init_key),
            "payable": contract.init(db).is_some_and(|init| init.is_payable(db)),
            "argsTy": self.ty(init_args)?,
            "argFields": init_fields,
            "decode": init_decode,
            "effects": self.entry_effects(init_key)?,
        });
        let mut arms = Vec::new();
        let mut root_provider = None;
        for (arm, name) in arm_names(db, contract) {
            let owner = BodyOwner::ContractRecvArm {
                contract,
                recv_idx: arm.recv(db).recv_idx(db),
                arm_idx: arm.arm_idx(db),
            };
            let callees = arm_callees(db, contract, arm)?;
            let host = self.ty(callees.host)?;
            if root_provider.is_some_and(|provider| provider != host) {
                return Err(Unsupported(
                    "recv arms under different root providers".into(),
                ));
            }
            root_provider = Some(host);
            // The field of the decoded message each of the arm body's
            // arguments takes, by the place of the binding among the arm's.
            let fields: Vec<_> = arm
                .arg_bindings(db)
                .iter()
                .map(|binding| binding.tuple_index)
                .collect();
            let info = &callees.info;
            arms.push(json!({
                "name": name,
                "func": self.func(identity_semantic_instance_key(db, owner)),
                "payable": arm.arm(db).is_some_and(|arm| arm.is_payable(db)),
                "fallback": info.is_fallback,
                "selector": info.selector_value,
                "signature": info.selector_signature,
                "argsTy": self.ty(info.args_ty)?,
                "argFields": fields,
                "decode": callees.decode.map(|key| self.func(key)),
                "retTy": info.ret_ty.map(|ty| self.ty(ty)).transpose()?,
                "returnValue": callees.return_value.map(|key| self.func(key)),
                "effects": self.entry_effects(identity_semantic_instance_key(db, owner))?,
            }));
        }
        Ok(json!({
            "name": contract.name(db).to_opt().map(|name| name.data(db).to_string()),
            "fields": fields,
            "codeSlotCount": contract.code_address_space_slot_count(db),
            "init": init,
            "runtime": {
                "rootProvider": root_provider,
                "arms": arms,
            },
        }))
    }

    /// The effects the entry code passes the body of `key`, an arm or `init`: each effect of
    /// the body by provider, with where its provider comes from (a contract field, or the root
    /// object), as MIR's `entry_semantic_args_plan` resolves them.
    fn entry_effects(&mut self, key: SemanticInstanceKey<'db>) -> Result<Vec<Value>, Unsupported> {
        let db = self.db;
        let instance = SemanticInstance::new(db, key);
        let mut effects = Vec::new();
        for binding in owner_effect_bindings(db, key.owner(db)) {
            let Some(provider) =
                resolved_provider_binding_for_instance_effect(db, instance, binding)
            else {
                continue;
            };
            let source = match provider.source {
                ProviderSource::ContractField { field } => json!({ "contractField": {
                    "field": self.field_index(field)?,
                }}),
                ProviderSource::RootProvider { .. } => json!("rootProvider"),
                ProviderSource::UsesParam { .. } => {
                    return Err(Unsupported("an entry effect from a uses parameter".into()));
                }
            };
            effects.push(json!({
                "provider": provider.provider_idx,
                "ty": self.ty(provider.provider_ty)?,
                "source": source,
            }));
        }
        Ok(effects)
    }

    /// The place of `field` among the contract's fields.
    fn field_index(&self, field: ContractFieldId<'db>) -> Result<usize, Unsupported> {
        if field.contract != self.contract {
            return Err(Unsupported("field of another contract".into()));
        }
        field
            .contract
            .storage_layout(self.db)
            .values()
            .position(|layout| layout.field == field)
            .ok_or_else(|| Unsupported("contract field without a layout".into()))
    }

    /// A reference to instance `key`, serialized once the instances named
    /// before it are.
    fn func(&mut self, key: SemanticInstanceKey<'db>) -> Value {
        if let Some((name, _)) = self.func_ids.get(&key) {
            return json!({ "$fn": name });
        }
        let id = self.keys.func_key(key);
        let reference = json!({ "$fn": id.0 });
        self.func_ids.insert(key, id);
        self.pending.push_back(key);
        reference
    }

    /// The entry of `key` in the function table: a body, or a callee
    /// without one by what the compiler knows it as.
    fn function(&mut self, key: SemanticInstanceKey<'db>) -> Result<Value, Unsupported> {
        let instance = SemanticInstance::new(self.db, key);
        let outer = std::mem::replace(&mut self.scope, instance);
        let function = self.function_in_scope(key, instance);
        self.scope = outer;
        function
    }

    fn function_in_scope(
        &mut self,
        key: SemanticInstanceKey<'db>,
        instance: SemanticInstance<'db>,
    ) -> Result<Value, Unsupported> {
        let db = self.db;
        let owner = key.owner(db);
        let name = match owner {
            BodyOwner::ContractInit { contract } => {
                format!("`init` of {}", contract_title(db, contract))
            }
            BodyOwner::ContractRecvArm {
                contract,
                recv_idx,
                arm_idx,
            } => format!(
                "recv arm {arm_idx} of recv block {recv_idx} of {}",
                contract_title(db, contract)
            ),
            _ => describe(db, key),
        };
        if let BodyOwner::Func(func, _) = owner {
            let intrinsic = if let Some(kind) = runtime_builtin_func_kind(db, func) {
                Some(json!({ "builtin": { "name": format!("{kind:?}") } }))
            } else if let Some(kind) = core_numeric_const_intrinsic(db, func) {
                Some(json!({ "numeric": { "op": numeric(kind) } }))
            } else if lib_func_matches(db, func, "core::num::__bitcast") {
                Some(json!("bitcast"))
            } else if let Some((kind, contract)) = instance.contract_metadata(db) {
                let kind = match kind {
                    ContractMetadataKind::InitCodeOffset => "initCodeOffset",
                    ContractMetadataKind::InitCodeLen => "initCodeLen",
                };
                let contract = contract
                    .name(db)
                    .to_opt()
                    .map(|name| name.data(db).to_string());
                Some(json!({ "contractMetadata": { "kind": kind, "contract": contract } }))
            } else if func.is_extern(db) {
                return Err(Unsupported(format!("extern function {name}")));
            } else {
                None
            };
            if let Some(intrinsic) = intrinsic {
                let typed = key.typed_body(db);
                let params = typed
                    .input_bindings()
                    .iter()
                    .map(|binding| self.ty(instance.normalized_binding_ty(db, *binding)))
                    .collect::<Result<Vec<_>, _>>()?;
                let ty_args = key
                    .subst(db)
                    .generic_args(db)
                    .iter()
                    .map(|ty| self.ty(instance.normalized_ty(db, *ty)))
                    .collect::<Result<Vec<_>, _>>()?;
                return Ok(json!({
                    "name": name,
                    "mayReturn": semantic_may_return(db, instance),
                    "kind": { "intrinsic": {
                        "kind": intrinsic,
                        "params": params,
                        "result": self.ty(instance.normalized_result_ty(db))?,
                        "tyArgs": ty_args,
                    }},
                }));
            }
        }
        let owner_kind = match owner {
            BodyOwner::Func(..) => "func",
            BodyOwner::Closure { .. } => "closure",
            BodyOwner::Const(_) | BodyOwner::AnonConstBody { .. } => "const",
            BodyOwner::ContractInit { .. } => "init",
            BodyOwner::ContractRecvArm { .. } => "recv",
        };
        let normalized = normalize_semantic_body(db, instance)
            .map_err(|_| Unsupported(format!("{name}: body does not normalize")))?;
        let mut body = BodyWriter {
            ir: self,
            body: &normalized.body,
        }
        .write()
        .map_err(|error| Unsupported(format!("{name}: {}", error.0)))?;
        let requirements: Vec<_> = instance
            .call_effect_requirements(db)
            .iter()
            .map(|requirement| json!({ "binding": requirement.binding_idx, "mutable": requirement.is_mut }))
            .collect();
        body["requirements"] = json!(requirements);
        Ok(json!({
            "name": name,
            "mayReturn": semantic_may_return(db, instance),
            "kind": { "body": {
                "owner": owner_kind,
                "projection": instance.is_projection(db),
                "dropImpl": matches!(owner, BodyOwner::Func(func, _) if implements_drop(db, func)),
                "result": self.ty(instance.normalized_result_ty(db))?,
                "body": body,
            }},
        }))
    }

    /// A reference to `ty`, with its entry in the type table.
    fn ty(&mut self, ty: TyId<'db>) -> Result<Value, Unsupported> {
        if let Some((name, _)) = self.type_ids.get(&ty) {
            return Ok(json!({ "$ty": name }));
        }
        let id = self.keys.ty_key(ty);
        let reference = json!({ "$ty": id.0 });
        self.type_ids.insert(ty, id.clone());
        let entry = self.type_entry(ty)?;
        self.types.insert(id, entry)?;
        Ok(reference)
    }

    /// What the compiler knows of `ty`: its shape, and its layouts in memory
    /// and in contract state.
    fn type_entry(&mut self, ty: TyId<'db>) -> Result<Value, Unsupported> {
        let db = self.db;
        let root = self.scope;
        let pretty = ty.pretty_print(db).to_string();
        let target = db.compilation_settings().target(db);
        let kind = if ty.is_never(db) {
            json!("never")
        } else if let Some((kind, inner)) = ty.as_capability(db) {
            let kind = match kind {
                CapabilityKind::Mut => "mut",
                CapabilityKind::Ref => "ref",
                CapabilityKind::View => "view",
            };
            json!({ "capability": { "kind": kind, "target": self.ty(inner)? } })
        } else if let Some((pointee, address_space)) = ty.as_ptr(db) {
            json!({ "ptr": { "pointee": self.ty(pointee)?, "space": space(address_space) } })
        } else if ty.is_bool(db) {
            json!("bool")
        } else if let Some(prim) = ty.as_prim(db)
            && let Some(bits) = prim_int_bits(prim, target)
        {
            json!({ "int": { "bits": bits, "signed": prim.is_signed_int() } })
        } else if ty.is_string(db) {
            // The capacity, where the compiler has evaluated it.
            json!({ "string": { "capacity": fixed_string_capacity_bytes(db, ty) } })
        } else if ty.is_array(db) {
            let (_, args) = ty.decompose_ty_app(db);
            let elem = args
                .first()
                .ok_or_else(|| Unsupported(format!("type {pretty}")))?;
            let len = ty
                .array_len(db)
                .ok_or_else(|| Unsupported(format!("array of unknown length {pretty}")))?;
            json!({ "array": { "elem": self.ty(root.normalized_ty(db, *elem))?, "len": len } })
        } else if ty.is_tuple(db) {
            let fields = self.tys(root.normalized_field_types(db, ty))?;
            json!({ "tuple": { "fields": fields } })
        } else if ty.as_closure(db).is_some() {
            let fields = self.tys(root.normalized_field_types(db, ty))?;
            json!({ "closure": { "captures": fields } })
        } else if let Some(adt) = ty.adt_def(db) {
            match adt.adt_ref(db) {
                AdtRef::Struct(item) => {
                    let names: Vec<_> = FieldParent::Struct(item)
                        .fields(db)
                        .map(|field| field.name(db).map(|name| name.data(db).to_string()))
                        .collect();
                    let tys = self.tys(root.normalized_field_types(db, ty))?;
                    let fields: Vec<_> = names
                        .into_iter()
                        .zip(tys)
                        .map(|(name, ty)| json!({ "name": name, "ty": ty }))
                        .collect();
                    json!({ "struct": { "fields": fields } })
                }
                AdtRef::Enum(item) => {
                    let mut variants = Vec::new();
                    for index in 0..adt.fields(db).len() {
                        let name = hir::hir_def::EnumVariant::new(item, index)
                            .name(db)
                            .map(str::to_string);
                        let fields = root.normalized_enum_variant_field_tys(
                            db,
                            ty,
                            hir::analysis::semantic::VariantIndex(index as u16),
                        );
                        variants.push(json!({ "name": name, "fields": self.tys(fields)? }));
                    }
                    json!({ "enum": { "variants": variants } })
                }
            }
        } else if matches!(
            ty.base_ty(db).data(db),
            TyData::TyBase(TyBase::Func(_) | TyBase::Contract(_))
        ) {
            json!("item")
        } else if ty.is_const_ty(db) {
            // A type argument that is a value, such as an array's length.
            json!("constArg")
        } else {
            return Err(Unsupported(format!("type {pretty}")));
        };
        Ok(json!({
            "name": pretty,
            "kind": kind,
            "memory": self.memory_layout(ty)?,
            "storage": self.storage_layout(ty),
            "pinned": root.pinned_space(db, ty).map(space),
            "dynString": ty.is_core_dyn_string(db),
            "dropGlue": ty_has_drop_glue(db, ty),
            "needsDrop": runtime_needs_drop(self.mir, ty),
            "dropImpl": self.drop_impl(ty)?,
        }))
    }

    /// The `Drop::drop` instance that destroys a value of `ty`, if its type
    /// implements `Drop`.
    fn drop_impl(&mut self, ty: TyId<'db>) -> Result<Option<Value>, Unsupported> {
        let db = self.db;
        let Some(adt) = ty.adt_def(db) else {
            return Ok(None);
        };
        if adt_drop_impl(db, adt.adt_ref(db)).is_none() {
            return Ok(None);
        }
        let scope = self.scope.key(db).owner(db).scope();
        let key = core_method_callee_key(
            db,
            scope,
            self.scope.assumptions(db),
            &["ops", "Drop"],
            "drop",
            vec![ty],
        )
        .map_err(|error| Unsupported(format!("drop: generated call: {error:?}")))?;
        Ok(Some(self.func(key)))
    }

    fn tys(&mut self, tys: &[TyId<'db>]) -> Result<Vec<Value>, Unsupported> {
        tys.iter().map(|ty| self.ty(*ty)).collect()
    }

    /// Where a value of `ty` lies in memory, as `RuntimeMemoryLayout` places it: its size, and
    /// the offsets of a product's fields or of an enum's tag, payload and variants' fields.
    fn memory_layout(&mut self, ty: TyId<'db>) -> Result<Value, Unsupported> {
        if runtime_size_bytes(self.db, ty).ok().flatten().is_none() {
            return Ok(Value::Null);
        }
        let class = stored_class_in_instance(self.mir, self.scope, ty);
        let Ok(facts) = RuntimeMemoryLayout::raw(self.mir).memory_facts(&class) else {
            return Ok(Value::Null);
        };
        let offsets = match facts.offsets {
            RuntimeMemoryOffsets::Product(fields) => json!({ "product": { "fields": fields } }),
            RuntimeMemoryOffsets::Enum {
                tag_bytes,
                payload,
                variants,
            } => {
                json!({ "enum": { "tagBytes": tag_bytes, "payload": payload, "variants": variants } })
            }
            RuntimeMemoryOffsets::None => Value::Null,
        };
        Ok(json!({ "size": facts.size, "offsets": offsets }))
    }

    /// Where a value of `ty` lies in storage or transient storage: the slots
    /// it takes, where each field of a struct or tuple lies below its first
    /// slot, the bytes it packs into as a field, and which of its fields
    /// are lanes apart from the rest of their slots.
    fn storage_layout(&self, ty: TyId<'db>) -> Value {
        let db = self.db;
        let Ok(slots) = storage_slot_span(db, ty) else {
            return Value::Null;
        };
        let placement = |placement: &StorageFieldPlacement| {
            json!({
                "slot": placement.slot,
                "lane": placement.lane.map(|lane| json!({
                    "offset": lane.byte_offset,
                    "width": lane.byte_width,
                })),
                "shared": placement.shared,
                "narrowed": placement.narrowed,
                "partial": placement.partial_lane().is_some(),
            })
        };
        json!({
            "slots": slots,
            "fields": storage_field_placements(db, ty)
                .map(|placements| placements.iter().map(placement).collect::<Vec<_>>()),
            "laneBytes": storage_lane_bytes(db, ty),
            "fieldLanes": storage_field_lanes(db, ty),
        })
    }
}

/// The serialization of one normalized body.
struct BodyWriter<'a, 'db> {
    ir: &'a mut IrExport<'db>,
    body: &'a NormalizedBody<'db>,
}

impl<'db> BodyWriter<'_, 'db> {
    /// A reference to `ty`, normalized in the body's instance.
    fn ty(&mut self, ty: TyId<'db>) -> Result<Value, Unsupported> {
        let ty = self.body.owner.normalized_ty(self.ir.db, ty);
        self.ir.ty(ty)
    }

    fn write(mut self) -> Result<Value, Unsupported> {
        let db = self.ir.db;
        let body = self.body;
        self.check_places()?;
        let mut values = Vec::new();
        for value in &body.values {
            let definition = match value.definition {
                NValueDefinition::EntryParam { param } => {
                    json!({ "entryParam": { "param": param } })
                }
                NValueDefinition::BlockParam { block, index } => {
                    json!({ "blockParam": { "block": block.index(), "index": index } })
                }
                NValueDefinition::Statement { block, statement } => {
                    json!({ "statement": { "block": block.index(), "statement": statement } })
                }
            };
            values.push(json!({
                "ty": self.ty(value.ty)?,
                "mutable": value.mutability == hir::analysis::semantic::Mutability::Mutable,
                "definition": definition,
                "name": self.binding_name(value.source),
            }));
        }
        let mut roots = Vec::new();
        for root in &body.roots {
            roots.push(self.root(root)?);
        }
        let mut blocks = Vec::new();
        for block in &body.blocks {
            blocks.push(self.block(block)?);
        }
        let drops = drop_elaboration(db, body.owner).map(|drops| {
            let point = |point: DropFlagPoint| match point {
                DropFlagPoint::After(statement) => {
                    json!({ "after": { "statement": statement.index() } })
                }
                DropFlagPoint::BlockStart(block) => {
                    json!({ "blockStart": { "block": block.index() } })
                }
                DropFlagPoint::BeforeTerminator(block) => {
                    json!({ "beforeTerminator": { "block": block.index() } })
                }
            };
            json!({
                "drops": drops.drops.iter().map(|(statement, parts)| json!({
                    "statement": statement.index(),
                    "parts": parts.iter().map(|part| json!({
                        "path": part.path.iter().map(|step| match *step {
                            DropStep::Field(field) => json!({ "field": { "field": field } }),
                            DropStep::Variant { variant, field } => {
                                json!({ "variant": { "variant": variant, "field": field } })
                            }
                            DropStep::Index(index) => json!({ "index": { "index": index } }),
                        }).collect::<Vec<_>>(),
                        "flags": part.flags,
                    })).collect::<Vec<_>>(),
                })).collect::<Vec<_>>(),
                "flags": drops.flags,
                "updates": drops.flag_updates.iter().map(|update| json!({
                    "at": point(update.at),
                    "flag": update.flag,
                    "initialized": update.initialized,
                })).collect::<Vec<_>>(),
            })
        });
        let entries = self.entries()?;
        Ok(json!({
            "paramCount": self.param_count(),
            "inputs": self.inputs()?,
            "values": values,
            "roots": roots,
            "blocks": blocks,
            "entry": body.entry.index(),
            "unsafeSplits": body.unsafe_splits.iter()
                .map(|split| split.iter().map(|value| value.index()).collect::<Vec<_>>())
                .collect::<Vec<_>>(),
            "drops": drops,
            "entries": entries,
        }))
    }

    /// The arguments a caller passes the body: a recv arm's, one for each of the arm's
    /// bindings; any other body's, one for each of its input bindings.
    fn param_count(&self) -> usize {
        let db = self.ir.db;
        let key = self.body.owner.key(db);
        match key.owner(db) {
            BodyOwner::ContractRecvArm {
                contract,
                recv_idx,
                arm_idx,
            } => contract
                .recv(db, recv_idx)
                .and_then(|recv| recv.arm(db, arm_idx))
                .map_or(0, |arm| arm.arg_bindings(db).len()),
            _ => key.typed_body(db).input_bindings().len(),
        }
    }

    /// The name the source gives a binding, for diagnostics.
    fn binding_name(&self, binding: Option<LocalBinding<'db>>) -> Option<String> {
        let db = self.ir.db;
        let hir = self.body.owner.key(db).owner(db).body(db)?;
        Some(binding?.name(db, hir)?.data(db).to_string())
    }

    /// How the body receives its inputs: each argument a caller passes, by
    /// its place among the arguments, and each effect it takes by value, by
    /// provider. Each is an entry value or a root the body starts with
    /// holding it. A recv arm's arguments are in the order of the arm's
    /// bindings.
    fn inputs(&self) -> Result<Vec<Value>, Unsupported> {
        let db = self.ir.db;
        let body = self.body;
        let owner = body.owner.key(db).owner(db);
        let pats: Vec<_> = match owner {
            BodyOwner::ContractRecvArm {
                contract,
                recv_idx,
                arm_idx,
            } => contract
                .recv(db, recv_idx)
                .and_then(|recv| recv.arm(db, arm_idx))
                .map_or(Vec::new(), |arm| {
                    arm.arg_bindings(db)
                        .iter()
                        .map(|binding| binding.pat)
                        .collect()
                }),
            _ => Vec::new(),
        };
        let arm = matches!(owner, BodyOwner::ContractRecvArm { .. });
        let instance = body.owner;
        let bindings = instance.key(db).typed_body(db).input_bindings();
        // Where an argument lies, and whether a projection holds a copy of
        // it for its session.
        let param_input = |param: u32| {
            json!({ "param": {
                "index": param,
                "space": space(instance.param_space(db, param)),
                "sessionCopy": bindings
                    .get(param as usize)
                    .is_some_and(|binding| instance.binding_is_session_copy(db, *binding)),
            }})
        };
        let mut inputs = Vec::new();
        for (index, value) in body.values.iter().enumerate() {
            let NValueDefinition::EntryParam { param } = value.definition else {
                continue;
            };
            let input = match effect_input(value.source) {
                Some(provider) => json!({ "effect": { "provider": provider } }),
                None if arm => param_input(
                    arg_position(value.source, &pats)
                        .ok_or_else(|| Unsupported("argument bound by no field".into()))?
                        as u32,
                ),
                None => param_input(param),
            };
            inputs.push(json!({ "input": input, "target": { "value": { "value": index } } }));
        }
        for (index, root) in body.roots.iter().enumerate() {
            let input = match root.kind {
                NRootKind::ParamPlace { param } => param_input(param),
                NRootKind::LocalSlot { binding } => match effect_input(binding) {
                    Some(provider) => json!({ "effect": { "provider": provider } }),
                    None => match arg_position(binding, &pats) {
                        Some(position) if arm => param_input(position as u32),
                        _ => continue,
                    },
                },
                _ => continue,
            };
            inputs.push(json!({ "input": input, "target": { "root": { "root": index } } }));
        }
        // Each argument is taken once.
        let mut params: Vec<_> = inputs
            .iter()
            .filter_map(|input| input["input"]["param"]["index"].as_u64())
            .collect();
        params.sort_unstable();
        if params.windows(2).any(|pair| pair[0] == pair[1]) {
            return Err(Unsupported(format!("inputs {params:?}")));
        }
        Ok(inputs)
    }

    fn root(&mut self, root: &NRoot<'db>) -> Result<Value, Unsupported> {
        let kind = match &root.kind {
            NRootKind::LocalSlot { binding } => {
                json!({ "localSlot": { "name": self.binding_name(*binding) } })
            }
            NRootKind::ParamPlace { param } => json!({ "paramPlace": { "param": param } }),
            NRootKind::Provider { binding } => {
                json!({ "provider": { "binding": self.provider(binding)? } })
            }
            NRootKind::CapabilityRepresentation { carrier } => {
                json!({ "capabilityRepresentation": { "carrier": carrier.index() } })
            }
            NRootKind::Temporary { value } => json!({ "temporary": { "value": value.index() } }),
        };
        Ok(json!({
            "kind": kind,
            "ty": self.ty(root.ty)?,
            "space": space(root.address_space),
            "mutable": root.mutability == hir::analysis::semantic::Mutability::Mutable,
        }))
    }

    fn provider(&mut self, binding: &ProviderBinding<'db>) -> Result<Value, Unsupported> {
        let source = match &binding.source {
            ProviderSource::UsesParam {
                requirement_idx, ..
            } => json!({ "usesParam": { "requirement": requirement_idx } }),
            ProviderSource::ContractField { field } => {
                json!({ "contractField": { "field": self.ir.field_index(*field)? } })
            }
            ProviderSource::RootProvider { .. } => json!("rootProvider"),
        };
        let semantics = &binding.semantics;
        Ok(json!({
            "provider": binding.provider_idx,
            "ty": self.ty(binding.provider_ty)?,
            "mutable": binding.is_mut,
            "source": source,
            "kind": match semantics.kind {
                ProviderKind::RootObject => "rootObject",
                ProviderKind::Place => "place",
            },
            "space": space(semantics.address_space),
            "target": semantics.target_ty.map(|ty| self.ty(ty)).transpose()?,
        }))
    }

    fn block(&mut self, block: &NBlock<'db>) -> Result<Value, Unsupported> {
        let mut statements = Vec::new();
        for statement in &block.statements {
            statements.push(self.statement(statement)?);
        }
        Ok(json!({
            "params": block.params.iter().map(|value| value.index()).collect::<Vec<_>>(),
            "statements": statements,
            "terminator": self.terminator(&block.terminator.kind)?,
        }))
    }

    fn statement(&mut self, statement: &NStatement<'db>) -> Result<Value, Unsupported> {
        let kind = match &statement.kind {
            NStatementKind::Define { result, expr } => {
                let ty = self.body.values[result.index()].ty;
                json!({ "define": { "result": result.index(), "expr": self.expr(expr, ty)? } })
            }
            NStatementKind::Store { destination, value } => json!({ "store": {
                "destination": self.place(destination)?,
                "value": operand(value),
            }}),
            NStatementKind::End { access } => json!({ "end": { "access": access.index() } }),
            NStatementKind::CheckIndex { index, array } => json!({ "checkIndex": {
                "index": nindex(index),
                "array": self.ty(*array)?,
            }}),
            NStatementKind::Drop { target } => {
                let target = match target {
                    NDropTarget::Value(value) => json!({ "value": { "value": value.index() } }),
                    NDropTarget::Place(place) => {
                        json!({ "place": { "place": self.place(place)? } })
                    }
                };
                json!({ "drop": { "target": target } })
            }
        };
        Ok(json!({ "id": statement.id.index(), "kind": kind }))
    }

    /// `expr`, which defines a value of type `ty`.
    fn expr(&mut self, expr: &NExpr<'db>, ty: TyId<'db>) -> Result<Value, Unsupported> {
        let db = self.ir.db;
        Ok(match expr {
            NExpr::Forward { src } => json!({ "forward": { "src": operand(src) } }),
            NExpr::ProjectValue { value, path } => json!({ "projectValue": {
                "value": operand(value),
                "path": path_json(&path.0),
            }}),
            NExpr::Load { place, mode } => json!({ "load": {
                "place": self.place(place)?,
                "mode": read_mode(*mode),
            }}),
            NExpr::Borrow {
                place,
                kind,
                provider,
            } => json!({ "borrow": {
                "place": self.place(place)?,
                "kind": match kind { BorrowKind::Ref => "ref", BorrowKind::Mut => "mut" },
                "provider": provider.map(space),
            }}),
            NExpr::MakeView {
                place,
                access: ViewAccess::Read,
            } => json!({ "makeView": { "place": self.place(place)?, "access": "read" } }),
            NExpr::StructuralRepack { value, mapping } => json!({ "structuralRepack": {
                "value": operand(value),
                "mapping": repack(mapping),
            }}),
            NExpr::CodeRegionRef { region } => {
                json!({ "codeRegionRef": { "region": code_region(db, region) } })
            }
            NExpr::Const(constant) => {
                json!({ "const": { "constant": self.constant(constant, ty)? } })
            }
            NExpr::Unary { op, value } => json!({ "unary": {
                "op": unary_op(*op),
                "value": operand(value),
            }}),
            NExpr::Binary { op, lhs, rhs } => json!({ "binary": {
                "op": binary_op(*op),
                "lhs": operand(lhs),
                "rhs": operand(rhs),
            }}),
            NExpr::PointerCast { value, to } => json!({ "pointerCast": {
                "value": operand(value),
                "to": self.ty(*to)?,
            }}),
            NExpr::ScalarCast { value, to } => json!({ "scalarCast": {
                "value": operand(value),
                "to": self.ty(*to)?,
            }}),
            NExpr::ArrayRepeat { ty, value } => json!({ "arrayRepeat": {
                "ty": self.ty(*ty)?,
                "value": operand(value),
            }}),
            NExpr::AggregateMake { ty, fields } => json!({ "aggregateMake": {
                "ty": self.ty(*ty)?,
                "fields": fields.iter().map(operand).collect::<Vec<_>>(),
            }}),
            NExpr::EnumMake {
                enum_ty,
                variant,
                fields,
            } => json!({ "enumMake": {
                "enumTy": self.ty(*enum_ty)?,
                "variant": variant.0,
                "fields": fields.iter().map(operand).collect::<Vec<_>>(),
            }}),
            NExpr::GetEnumTag { value } => json!({ "getEnumTag": { "value": operand(value) } }),
            NExpr::IsEnumVariant { value, variant } => json!({ "isEnumVariant": {
                "value": operand(value),
                "variant": variant.0,
            }}),
            NExpr::CodeRegionOffset { target } => {
                json!({ "codeRegionOffset": { "target": code_region_target(db, target)? } })
            }
            NExpr::CodeRegionLen { target } => {
                json!({ "codeRegionLen": { "target": code_region_target(db, target)? } })
            }
            NExpr::Call {
                call_site,
                callee,
                args,
                effect_args,
            } => {
                let mut effects = Vec::new();
                for arg in effect_args {
                    effects.push(self.effect_arg(arg)?);
                }
                json!({ "call": {
                    "callSite": match call_site {
                        CallSiteId::Expr(expr) => format!("expr{}", expr.index()),
                        CallSiteId::ForLoop(stmt, step) => format!("for{}:{step:?}", stmt.index()),
                    },
                    "callee": self.ir.func(self.callee_key(callee.key, args)),
                    "args": args.iter().map(operand).collect::<Vec<_>>(),
                    "effectArgs": effects,
                }})
            }
        })
    }

    /// The function a call of `callee` runs: on the EVM, `panic_with_value::<T>` reverts
    /// through the function MIR's `lower_panic_with_value` calls in its place, which the
    /// calling body's scope resolves.
    fn callee_key(
        &self,
        callee: SemanticInstanceKey<'db>,
        args: &[NOperand],
    ) -> SemanticInstanceKey<'db> {
        let db = self.ir.db;
        match (callee.owner(db), args) {
            (BodyOwner::Func(func, _), [value])
                if runtime_builtin_func_kind(db, func)
                    == Some(RuntimeBuiltinFuncKind::PanicWithValue) =>
            {
                let value_ty = self.body.values[value.value.index()].ty;
                panic_payload_revert_key(self.ir.mir, self.body.owner.key(db), value_ty)
            }
            _ => callee,
        }
    }

    fn effect_arg(&mut self, arg: &NEffectArg<'db>) -> Result<Value, Unsupported> {
        Ok(json!({
            "bindingIdx": arg.binding_idx,
            "place": self.place(&arg.place)?,
            "requiredMut": arg.required_mut,
            "providerTargetTy": arg.provider_target_ty.map(|ty| self.ty(ty)).transpose()?,
            "provider": arg.provider.map(space),
        }))
    }

    /// A constant that defines a value of type `ty`, each value reified for `ty` as runtime
    /// lowering reifies it (`reify_runtime_const_for_ty`).
    fn constant(&mut self, constant: &SConst<'db>, ty: TyId<'db>) -> Result<Value, Unsupported> {
        let db = self.ir.db;
        let owner = self.body.owner;
        let reified = |value: SemConstId<'db>| {
            reify_runtime_const_for_ty(db, owner, ty, value)
                .ok_or_else(|| Unsupported(format!("constant of type `{}`", ty.pretty_print(db))))
        };
        Ok(match constant {
            SConst::Value(value) => {
                json!({ "value": { "value": self.const_value(reified(value.value())?)? } })
            }
            SConst::Description(value) => {
                json!({ "description": { "value": self.const_value(reified(*value)?)? } })
            }
            SConst::Evidence(value) => {
                json!({ "evidence": { "value": self.const_value(reified(*value)?)? } })
            }
            SConst::Invalid(_) => json!("invalid"),
            SConst::Ref(reference) => {
                let value = match eval_const_ref(db, *reference) {
                    EvalOutcome::Ready(value) => Some(self.const_value(reified(value)?)?),
                    _ => None,
                };
                json!({ "ref": {
                    "func": self.ir.func(reference.instance(db)),
                    "ty": self.ty(reference.ty(db))?,
                    "value": value,
                }})
            }
        })
    }

    fn const_value(&mut self, value: SemConstId<'db>) -> Result<Value, Unsupported> {
        let db = self.ir.db;
        let parts = |ir: &mut Self, parts: &[SemConstId<'db>]| -> Result<Vec<Value>, Unsupported> {
            parts.iter().map(|part| ir.const_value(*part)).collect()
        };
        Ok(match value.value(db) {
            SemConstValue::Unit => json!("unit"),
            SemConstValue::Scalar { ty, value } => {
                let ty = self.ty(*ty)?;
                match value {
                    SemConstScalar::Bool(value) => json!({ "bool": { "ty": ty, "value": value } }),
                    SemConstScalar::Int { value } => {
                        json!({ "int": { "ty": ty, "value": value.to_string() } })
                    }
                    SemConstScalar::Bytes(bytes) => {
                        let hex: String = bytes.iter().map(|byte| format!("{byte:02x}")).collect();
                        json!({ "bytes": { "ty": ty, "value": hex } })
                    }
                }
            }
            SemConstValue::Description(term) => {
                return Err(Unsupported(format!(
                    "constant description of type `{}`",
                    term.ty(db).pretty_print(db)
                )));
            }
            SemConstValue::Tuple { ty, elems } => json!({ "tuple": {
                "ty": self.ty(*ty)?,
                "elems": parts(self, elems)?,
            }}),
            SemConstValue::Struct { ty, fields } => json!({ "struct": {
                "ty": self.ty(*ty)?,
                "fields": parts(self, fields)?,
            }}),
            SemConstValue::Array { ty, elems } => json!({ "array": {
                "ty": self.ty(*ty)?,
                "elems": parts(self, elems)?,
            }}),
            SemConstValue::Enum {
                ty,
                variant,
                fields,
            } => json!({ "enum": {
                "ty": self.ty(*ty)?,
                "variant": variant.0,
                "fields": parts(self, fields)?,
            }}),
        })
    }

    fn place(&mut self, place: &NPlace<'db>) -> Result<Value, Unsupported> {
        let base = match place.base {
            NPlaceBase::Root(root) => json!({ "root": { "root": root.index() } }),
            NPlaceBase::CapabilityTarget { carrier } => {
                json!({ "capabilityTarget": { "carrier": carrier.index() } })
            }
            NPlaceBase::Deref { pointer } => json!({ "deref": { "pointer": pointer.index() } }),
        };
        Ok(json!({
            "base": base,
            "path": path_json(&place.path),
            "ty": self.ty(place.ty)?,
        }))
    }

    fn terminator(&mut self, terminator: &NTerminatorKind<'db>) -> Result<Value, Unsupported> {
        let db = self.ir.db;
        Ok(match terminator {
            NTerminatorKind::Goto(target) => json!({ "goto": { "target": successor(target) } }),
            NTerminatorKind::Branch {
                cond,
                then_target,
                else_target,
            } => json!({ "branch": {
                "cond": operand(cond),
                "thenTarget": successor(then_target),
                "elseTarget": successor(else_target),
            }}),
            NTerminatorKind::MatchEnum {
                value,
                enum_ty,
                cases,
                default,
            } => json!({ "matchEnum": {
                "value": operand(value),
                "enumTy": self.ty(*enum_ty)?,
                "cases": cases.iter()
                    .map(|(variant, target)| json!({ "variant": variant.0, "target": successor(target) }))
                    .collect::<Vec<_>>(),
                "default": default.as_ref().map(successor),
            }}),
            NTerminatorKind::Fail(FailReason::Assert { message }) => json!({ "fail": { "reason": {
                "assert": { "message": message.map(|message| message.data(db).to_string()) },
            }}}),
            NTerminatorKind::Unreachable => json!("unreachable"),
            NTerminatorKind::Return(value) => {
                json!({ "return": { "value": value.as_ref().map(operand) } })
            }
            NTerminatorKind::Yield { value, resume } => json!({ "yield": {
                "value": operand(value),
                "resume": successor(resume),
            }}),
        })
    }

    /// How the body reaches an entry of each collection a place of it takes
    /// an entry of: the collection's `StateIndex::locate`, the state space
    /// the entries lie in, and the collection's lane codec if it packs them.
    fn entries(&mut self) -> Result<Vec<Value>, Unsupported> {
        let db = self.ir.db;
        let body = self.body;
        let mut collections = Vec::new();
        let mut places = Vec::new();
        for statement in body.blocks.iter().flat_map(|block| &block.statements) {
            match &statement.kind {
                NStatementKind::Define { expr, .. } => {
                    expr.for_each_place_operand(|place| places.push(place.clone()));
                }
                NStatementKind::Store { destination, .. } => places.push(destination.clone()),
                NStatementKind::Drop {
                    target: NDropTarget::Place(place),
                } => places.push(place.clone()),
                _ => {}
            }
        }
        for place in &places {
            for (len, projection) in place.path.iter().enumerate() {
                if !matches!(projection, NDataProjection::Entry(_)) {
                    continue;
                }
                let ty = body
                    .place_prefix_ty(db, place, len)
                    .ok_or_else(|| Unsupported("untyped place".into()))?;
                if !collections.contains(&ty) {
                    collections.push(ty);
                }
            }
        }
        let mut entries = Vec::new();
        for ty in collections {
            let pretty = ty.pretty_print(db);
            let entry = body
                .owner
                .entry_locate(db, ty)
                .map_err(|_| Unsupported(format!("entry of {pretty}")))?;
            let codec = match entry.lanes {
                Some(lanes) => {
                    let scope = body.owner.key(db).owner(db).scope();
                    let mut method = |name| {
                        core_method_callee_key(
                            db,
                            scope,
                            body.owner.assumptions(db),
                            &["ops", "LaneCodec"],
                            name,
                            vec![lanes.codec, entry.entry_ty],
                        )
                        .map(|key| self.ir.func(key))
                        .map_err(|_| Unsupported(format!("entry lane of {pretty}")))
                    };
                    Some(json!({ "get": method("get")?, "set": method("set")? }))
                }
                None => None,
            };
            entries.push(json!({
                "collection": self.ty(ty)?,
                "locate": self.ir.func(entry.callee),
                "keyTy": self.ty(entry.key_ty)?,
                "entryTy": self.ty(entry.entry_ty)?,
                "space": entry.space.map(space),
                "codec": codec,
            }));
        }
        Ok(entries)
    }

    /// Checks that the type table types every place and value path of the
    /// body as the compiler does, step by step, so that the interpreter can
    /// type a part of a place from the table.
    fn check_places(&mut self) -> Result<(), Unsupported> {
        let db = self.ir.db;
        let body = self.body;
        let mismatch =
            |what: &str| Unsupported(format!("the type table types {what} unlike the compiler"));
        let check =
            |base: TyId<'db>, path: &NDataPath, prefix: &dyn Fn(usize) -> Option<TyId<'db>>| {
                let mut ty = base;
                for (len, projection) in path.iter().enumerate() {
                    if prefix(len) != Some(ty) {
                        return Err(mismatch("a path"));
                    }
                    ty = self
                        .project(ty, projection)
                        .ok_or_else(|| mismatch("a step"))?;
                }
                (prefix(path.len()) == Some(ty))
                    .then_some(())
                    .ok_or_else(|| mismatch("a path's end"))
            };
        for statement in body.blocks.iter().flat_map(|block| &block.statements) {
            let mut places = Vec::new();
            match &statement.kind {
                NStatementKind::Define { expr, .. } => {
                    expr.for_each_place_operand(|place| places.push(place.clone()));
                    if let NExpr::ProjectValue { value, path } = expr {
                        let base = body.values[value.value.index()].ty;
                        check(base, &path.0, &|len| {
                            body.value_prefix_ty(db, value.value, &path.0, len)
                        })?;
                    }
                }
                NStatementKind::Store { destination, .. } => places.push(destination.clone()),
                NStatementKind::Drop {
                    target: NDropTarget::Place(place),
                } => places.push(place.clone()),
                _ => {}
            }
            for place in &places {
                let base = body
                    .place_prefix_ty(db, place, 0)
                    .ok_or_else(|| mismatch("a place's base"))?;
                check(base, &place.path, &|len| {
                    body.place_prefix_ty(db, place, len)
                })?;
                if body.place_prefix_ty(db, place, place.path.len()) != Some(place.ty) {
                    return Err(mismatch("a place"));
                }
            }
        }
        Ok(())
    }
}

impl<'db> BodyWriter<'_, 'db> {
    /// The type of the part `projection` reaches of a value of `ty`, as the
    /// type table has it.
    fn project(&self, ty: TyId<'db>, projection: &NDataProjection) -> Option<TyId<'db>> {
        let db = self.ir.db;
        let root = self.ir.root;
        match *projection {
            NDataProjection::Field(field) => root
                .normalized_field_types(db, ty)
                .get(usize::from(field.0))
                .copied(),
            NDataProjection::VariantField { variant, field } => root
                .normalized_enum_variant_field_tys(db, ty, variant)
                .get(usize::from(field.0))
                .copied(),
            NDataProjection::Index(_) => {
                let (_, args) = ty.decompose_ty_app(db);
                ty.is_array(db).then(|| root.normalized_ty(db, args[0]))
            }
            NDataProjection::Entry(_) => self
                .body
                .owner
                .state_index_tys(db, ty)
                .map(|(_, entry)| entry),
        }
    }
}

fn space(space: AddressSpace) -> &'static str {
    match space {
        AddressSpace::Memory => "memory",
        AddressSpace::Storage => "storage",
        AddressSpace::Transient => "transient",
        AddressSpace::Calldata => "calldata",
        AddressSpace::Code => "code",
    }
}

fn operand(operand: &NOperand) -> Value {
    json!({ "value": operand.value.index(), "mode": read_mode(operand.mode) })
}

fn read_mode(mode: ReadMode) -> &'static str {
    match mode {
        ReadMode::Copy => "copy",
        ReadMode::Read => "read",
        ReadMode::Move => "move",
    }
}

fn successor(successor: &NSuccessor) -> Value {
    json!({
        "block": successor.block.index(),
        "args": successor.args.iter().map(operand).collect::<Vec<_>>(),
    })
}

fn nindex(index: &NIndex) -> Value {
    match index {
        NIndex::Const(index) => json!({ "const": { "index": index } }),
        NIndex::Value(value) => json!({ "value": { "value": value.index() } }),
    }
}

fn path_json(path: &NDataPath) -> Vec<Value> {
    path.iter()
        .map(|projection| match projection {
            NDataProjection::Field(field) => json!({ "field": { "index": field.0 } }),
            NDataProjection::VariantField { variant, field } => {
                json!({ "variantField": { "variant": variant.0, "field": field.0 } })
            }
            NDataProjection::Index(index) => json!({ "index": { "index": nindex(index) } }),
            NDataProjection::Entry(key) => json!({ "entry": { "key": key.index() } }),
        })
        .collect()
}

fn repack(mapping: &StructuralRepack) -> Vec<Value> {
    mapping
        .fields
        .iter()
        .map(|(target, source)| json!({ "target": path_json(target), "source": path_json(source) }))
        .collect()
}

fn code_region(db: &dyn HirAnalysisDb, region: &SemanticCodeRegionRef<'_>) -> Value {
    match region {
        SemanticCodeRegionRef::ManualContractRoot { func } => json!({ "manualContractRoot": {
            "func": func.name(db).to_opt().map(|name| name.data(db).to_string()),
        }}),
    }
}

fn code_region_target(
    db: &dyn HirAnalysisDb,
    target: &SemanticCodeRegionTarget<'_>,
) -> Result<Value, Unsupported> {
    match target {
        SemanticCodeRegionTarget::Resolved(region) => {
            Ok(json!({ "resolved": { "region": code_region(db, region) } }))
        }
        SemanticCodeRegionTarget::Deferred { .. } => {
            Err(Unsupported("deferred code region".into()))
        }
    }
}

fn overflow(overflow: Overflow) -> &'static str {
    match overflow {
        Overflow::Checked => "checked",
        Overflow::Wrapping => "wrapping",
    }
}

fn unary_op(op: UnaryOp) -> Value {
    match op {
        UnaryOp::Neg(mode) => json!({ "neg": { "overflow": overflow(mode) } }),
        UnaryOp::Not => json!("not"),
        UnaryOp::BitNot => json!("bitNot"),
    }
}

fn comp(op: CompBinOp) -> &'static str {
    match op {
        CompBinOp::Eq => "eq",
        CompBinOp::NotEq => "notEq",
        CompBinOp::Lt => "lt",
        CompBinOp::LtEq => "ltEq",
        CompBinOp::Gt => "gt",
        CompBinOp::GtEq => "gtEq",
    }
}

fn binary_op(op: BinaryOp) -> Value {
    match op {
        BinaryOp::Add(mode) => json!({ "add": { "overflow": overflow(mode) } }),
        BinaryOp::Sub(mode) => json!({ "sub": { "overflow": overflow(mode) } }),
        BinaryOp::Mul(mode) => json!({ "mul": { "overflow": overflow(mode) } }),
        BinaryOp::Div(mode) => json!({ "div": { "overflow": overflow(mode) } }),
        BinaryOp::Rem => json!("rem"),
        BinaryOp::Pow(mode) => json!({ "pow": { "overflow": overflow(mode) } }),
        BinaryOp::Shl => json!("shl"),
        BinaryOp::Shr => json!("shr"),
        BinaryOp::BitAnd => json!("bitAnd"),
        BinaryOp::BitOr => json!("bitOr"),
        BinaryOp::BitXor => json!("bitXor"),
        BinaryOp::Comp(op) => json!({ "comp": { "op": comp(op) } }),
    }
}

/// A numeric intrinsic as the operation it performs on its operands.
fn numeric(kind: NumericExternIntrinsic) -> Value {
    let arith = |op: ArithBinOp, mode: Overflow| match BinaryOp::new(
        hir::hir_def::expr::BinOp::Arith(op),
        mode,
    ) {
        Some(op) => json!({ "binary": { "op": binary_op(op) } }),
        None => json!({ "unknown": { "op": format!("{op:?}") } }),
    };
    match kind {
        NumericExternIntrinsic::CheckedBinary(op) => arith(op, Overflow::Checked),
        NumericExternIntrinsic::WrappingBinary(op) => arith(op, Overflow::Wrapping),
        NumericExternIntrinsic::BoolBinary(op) => arith(op, Overflow::Wrapping),
        NumericExternIntrinsic::Comparison(op) => {
            json!({ "binary": { "op": { "comp": { "op": comp(op) } } } })
        }
        NumericExternIntrinsic::SaturatingBinary(op) => json!({ "saturating": { "op": match op {
            SaturatingArithmetic::Add => "add",
            SaturatingArithmetic::Sub => "sub",
            SaturatingArithmetic::Mul => "mul",
        }}}),
        NumericExternIntrinsic::CheckedNeg => {
            json!({ "unary": { "op": unary_op(UnaryOp::Neg(Overflow::Checked)) } })
        }
        NumericExternIntrinsic::BitNot => json!({ "unary": { "op": unary_op(UnaryOp::BitNot) } }),
        NumericExternIntrinsic::BoolNot => json!({ "unary": { "op": unary_op(UnaryOp::Not) } }),
    }
}
