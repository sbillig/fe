//! Exports a contract as Lean definitions in the `Fe.EVM` model monad: the
//! checked normalized body of each entry, of each function and constant it
//! reaches, and the storage layout those bodies name.
//!
//! A body the exporter has no model for is reported by the construct it
//! stopped at, never skipped.

use std::fmt::Write;

use common::ingot::IngotKind;
use cranelift_entity::EntityRef;
use hir::{
    analysis::{
        HirAnalysisDb,
        semantic::{
            BinaryOp, FailReason, NBlockId, NDataProjection, NEffectArg, NEffectArgValue, NExpr,
            NOperand, NPlace, NPlaceBase, NRootId, NRootKind, NStatement, NStatementKind,
            NTerminatorKind, NValueDefinition, NValueId, NormalizedBody, Overflow, SConst,
            SemConstScalar, SemConstValue, SemanticInstance, SemanticInstanceKey,
            identity_semantic_instance_key, normalize_semantic_body,
        },
        ty::{
            AddressSpace,
            ty_check::BodyOwner,
            ty_def::{CapabilityKind, TyId},
            ty_is_snapshot,
        },
    },
    hir_def::{Contract, ItemKind, expr::CompBinOp},
    semantic::{ProviderSource, storage_field_placements},
};
use rustc_hash::{FxHashMap, FxHashSet};

/// A construct the exporter has no model for.
#[derive(Debug, PartialEq, Eq)]
pub struct Unsupported(pub String);

/// The Lean definitions of `contract`: its storage layout, the functions
/// and constants its entries reach, then its `init` and `recv` arms.
pub fn export_contract<'db>(
    db: &'db dyn HirAnalysisDb,
    contract: Contract<'db>,
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
        taken: entries.iter().map(|(name, _)| name.clone()).collect(),
        open: Vec::new(),
        layouts: Vec::new(),
        bodies: FxHashMap::default(),
        defs: String::new(),
    };
    for (name, owner) in entries {
        let body = module.body(&name, identity_semantic_instance_key(db, owner))?;
        writeln!(module.defs, "def {name}{body}").unwrap();
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
    /// The Lean name of each exported function or constant instance.
    names: FxHashMap<SemanticInstanceKey<'db>, String>,
    taken: FxHashSet<String>,
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
    /// reports. The definitions it uses are exported first.
    fn body(&mut self, name: &str, key: SemanticInstanceKey<'db>) -> Result<String, Unsupported> {
        let normalized = normalize_semantic_body(self.db, SemanticInstance::new(self.db, key))
            .map_err(|_| Unsupported(format!("{name}: body does not normalize")))?;
        BodyExporter {
            module: self,
            body: &normalized.body,
            aggregates: FxHashMap::default(),
        }
        .export()
        .map_err(|error| Unsupported(format!("{name}: {}", error.0)))
    }

    /// The Lean name of the function or constant instance `key`, exported
    /// on first use.
    fn instance(&mut self, key: SemanticInstanceKey<'db>) -> Result<String, Unsupported> {
        if let Some(name) = self.names.get(&key) {
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
        let body = self.body(&base, key)?;
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
                writeln!(self.defs, "def {name}{body}").unwrap();
                self.taken.insert(name.clone());
                self.bodies.insert((base, body), name.clone());
                name
            }
        };
        self.names.insert(key, name.clone());
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
    /// The type name and fields of each aggregate a body builds, which only
    /// a std call consumes.
    aggregates: FxHashMap<NValueId, (String, Vec<NValueId>)>,
}

impl<'db> BodyExporter<'_, 'db> {
    /// The body's definition after its name: parameters, type and value.
    fn export(mut self) -> Result<String, Unsupported> {
        let mut params = String::new();
        for (index, value) in self.body.values.iter().enumerate() {
            if matches!(value.definition, NValueDefinition::EntryParam { .. }) {
                write!(params, " (v{index} : {})", self.lean_ty(value.ty)?).unwrap();
            }
        }
        let result_ty = self.result_ty()?;
        let mut out = format!("{params} : FeM {result_ty} := do\n");
        self.block(self.body.entry, 1, &mut vec![], &mut out)?;
        Ok(out)
    }

    /// The type every `Return` of the body gives.
    fn result_ty(&self) -> Result<String, Unsupported> {
        let returned = self
            .body
            .blocks
            .iter()
            .find_map(|block| match block.terminator.kind {
                NTerminatorKind::Return(operand) => Some(operand),
                _ => None,
            });
        match returned {
            Some(Some(operand)) => self.lean_ty(self.ty(operand.value)),
            Some(None) | None => Ok("Unit".to_string()),
        }
    }

    /// Prints `block` and, inline, the blocks it leads to. A block reached
    /// again on its own path is a loop.
    fn block(
        &mut self,
        block: NBlockId,
        depth: usize,
        path: &mut Vec<NBlockId>,
        out: &mut String,
    ) -> Result<(), Unsupported> {
        if path.contains(&block) {
            return Err(Unsupported("loop".to_string()));
        }
        let data = &self.body.blocks[block.index()];
        if !data.params.is_empty() {
            return Err(Unsupported("block parameters".to_string()));
        }
        let pad = "  ".repeat(depth);
        for statement in &data.statements {
            if let Some(line) = self.statement(statement)? {
                writeln!(out, "{pad}{line}").unwrap();
            }
        }
        path.push(block);
        match &data.terminator.kind {
            NTerminatorKind::Return(None) => writeln!(out, "{pad}return ()").unwrap(),
            NTerminatorKind::Return(Some(operand)) => {
                writeln!(out, "{pad}return {}", value_name(operand.value)).unwrap()
            }
            NTerminatorKind::Goto(target) => self.block(target.block, depth, path, out)?,
            NTerminatorKind::Branch {
                cond,
                then_target,
                else_target,
            } => {
                writeln!(out, "{pad}if {} then", value_name(cond.value)).unwrap();
                self.block(then_target.block, depth + 1, path, out)?;
                writeln!(out, "{pad}else").unwrap();
                self.block(else_target.block, depth + 1, path, out)?;
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
            other => return Err(Unsupported(format!("terminator {other:?}"))),
        }
        path.pop();
        Ok(())
    }

    fn statement(&mut self, statement: &NStatement<'db>) -> Result<Option<String>, Unsupported> {
        Ok(match &statement.kind {
            NStatementKind::Define { result, expr } => {
                let name = value_name(*result);
                match expr {
                    NExpr::Forward { src } => Some(format!("let {name} := {}", operand(src))),
                    NExpr::Const(SConst::Ref(constant)) => Some(format!(
                        "let {name} ← {}",
                        self.module.instance(constant.instance(self.module.db))?
                    )),
                    NExpr::Const(constant) => {
                        Some(format!("let {name} := {}", self.lean_const(constant)?))
                    }
                    NExpr::Load { place, .. } => {
                        Some(format!("let {name} ← load {}", self.word_place(place)?))
                    }
                    NExpr::Binary { op, lhs, rhs } => {
                        if self.ty(lhs.value).pretty_print(self.module.db) != "u256" {
                            return Err(Unsupported(format!("operator {op:?} off `u256`")));
                        }
                        Some(format!(
                            "let {name} {}",
                            binary(*op, operand(lhs), operand(rhs))?
                        ))
                    }
                    // A view or borrow of the `Evm` root carries no data.
                    NExpr::MakeView { place, .. } | NExpr::Borrow { place, .. }
                        if self.is_root_object(place) =>
                    {
                        None
                    }
                    NExpr::MakeView { place, .. } | NExpr::Borrow { place, .. } => {
                        if self.lean_ty(self.ty(*result))? != "Place" {
                            return Err(Unsupported("view of a one-word place".to_string()));
                        }
                        Some(format!("let {name} := {}", self.place(place)?))
                    }
                    NExpr::AggregateMake { ty, fields } => {
                        let fields = fields.iter().map(|field| field.value).collect();
                        let ty = ty.pretty_print(self.module.db).to_string();
                        self.aggregates.insert(*result, (ty, fields));
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
                        Some(self.call(&name, callee.key, args)?)
                    }
                    other => return Err(Unsupported(format!("expression {other:?}"))),
                }
            }
            NStatementKind::Store { destination, value } => Some(format!(
                "store {} {}",
                self.word_place(destination)?,
                operand(value)
            )),
            NStatementKind::End { .. } => None,
            other => return Err(Unsupported(format!("statement {other:?}"))),
        })
    }

    /// A call: to a std function the prelude models, by name, or to an
    /// exported function.
    fn call(
        &mut self,
        name: &str,
        callee: SemanticInstanceKey<'db>,
        args: &[NOperand],
    ) -> Result<String, Unsupported> {
        let db = self.module.db;
        let BodyOwner::Func(func, _) = callee.owner(db) else {
            return Err(Unsupported("call to a non-function".to_string()));
        };
        let in_std = matches!(
            func.scope().ingot(db).kind(db),
            IngotKind::Std | IngotKind::Core
        );
        let callee_name = func.name(db).to_opt().map(|name| name.data(db).as_str());
        match (in_std, callee_name, args) {
            (true, Some("caller"), [_evm]) => Ok(format!("let {name} ← caller")),
            (true, Some("zero"), []) => Ok(format!("let {name} := UInt256.ofNat 0")),
            (true, Some("emit"), [_evm, event]) => {
                let (ty, fields) = self
                    .aggregates
                    .get(&event.value)
                    .ok_or_else(|| Unsupported("emit of a value not built in place".to_string()))?;
                let fields = fields
                    .iter()
                    .map(|field| format!(" {}", value_name(*field)));
                Ok(format!(
                    "let {name} ← emit{ty}{}",
                    fields.collect::<String>()
                ))
            }
            _ => {
                let callee = self.module.instance(callee)?;
                let args: String = args
                    .iter()
                    .map(|arg| format!(" {}", operand(arg)))
                    .collect();
                Ok(format!("let {name} ← {callee}{args}"))
            }
        }
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

    /// The Lean type of a value of `ty`: a word, or the place an access
    /// reaches. A view of a snapshot type carries the value.
    fn lean_ty(&self, ty: TyId<'db>) -> Result<String, Unsupported> {
        let db = self.module.db;
        let word = |ty: TyId<'db>| match ty.pretty_print(db).as_str() {
            // An address is its word.
            "u256" | "Address" => Ok("UInt256".to_string()),
            "bool" => Ok("Bool".to_string()),
            "()" => Ok("Unit".to_string()),
            other => Err(Unsupported(format!("type {other}"))),
        };
        let owner = self.body.owner;
        let scope = owner.key(db).owner(db).scope();
        match ty.as_capability(db) {
            Some((CapabilityKind::View, inner))
                if ty_is_snapshot(db, scope, inner, owner.assumptions(db)) =>
            {
                word(inner)
            }
            Some(_) => Ok("Place".to_string()),
            None => word(ty),
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

    /// `place`, which holds one word.
    fn word_place(&mut self, place: &NPlace<'db>) -> Result<String, Unsupported> {
        if self.lean_ty(place.ty)? != "UInt256" {
            return Err(Unsupported("place that is not one word".to_string()));
        }
        self.place(place)
    }

    /// The Lean place `place` names: its base, then a layout step for each
    /// struct field and an entry step for each `StorageMap` key.
    fn place(&mut self, place: &NPlace<'db>) -> Result<String, Unsupported> {
        let db = self.module.db;
        let mut lean = match place.base {
            NPlaceBase::Root(root) => self.root(root)?,
            NPlaceBase::CapabilityTarget { carrier }
                if self.lean_ty(self.ty(carrier))? == "Place" =>
            {
                value_name(carrier)
            }
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

fn operand(operand: &NOperand) -> String {
    value_name(operand.value)
}

/// The right-hand side of `let v` for a built-in operation on `u256`.
fn binary(op: BinaryOp, lhs: String, rhs: String) -> Result<String, Unsupported> {
    Ok(match op {
        BinaryOp::Add(Overflow::Checked) => format!("← checkedAdd {lhs} {rhs}"),
        BinaryOp::Add(Overflow::Wrapping) => format!(":= {lhs} + {rhs}"),
        BinaryOp::Sub(Overflow::Checked) => format!("← checkedSub {lhs} {rhs}"),
        BinaryOp::Sub(Overflow::Wrapping) => format!(":= {lhs} - {rhs}"),
        BinaryOp::Mul(Overflow::Checked) => format!("← checkedMul {lhs} {rhs}"),
        BinaryOp::Mul(Overflow::Wrapping) => format!(":= {lhs} * {rhs}"),
        // Unsigned division cannot overflow.
        BinaryOp::Div(_) => format!("← checkedDiv {lhs} {rhs}"),
        BinaryOp::Rem => format!("← checkedRem {lhs} {rhs}"),
        BinaryOp::Shl => format!(":= {lhs} <<< {rhs}"),
        BinaryOp::Shr => format!(":= {lhs} >>> {rhs}"),
        BinaryOp::BitAnd => format!(":= {lhs} &&& {rhs}"),
        BinaryOp::BitOr => format!(":= {lhs} ||| {rhs}"),
        BinaryOp::BitXor => format!(":= {lhs} ^^^ {rhs}"),
        BinaryOp::Comp(CompBinOp::Eq) => format!(":= decide ({lhs} = {rhs})"),
        BinaryOp::Comp(CompBinOp::NotEq) => format!(":= decide ({lhs} ≠ {rhs})"),
        BinaryOp::Comp(CompBinOp::GtEq) => format!(":= decide ({rhs}.toNat ≤ {lhs}.toNat)"),
        BinaryOp::Comp(CompBinOp::LtEq) => format!(":= decide ({lhs}.toNat ≤ {rhs}.toNat)"),
        BinaryOp::Comp(CompBinOp::Lt) => format!(":= decide ({lhs}.toNat < {rhs}.toNat)"),
        BinaryOp::Comp(CompBinOp::Gt) => format!(":= decide ({rhs}.toNat < {lhs}.toNat)"),
        BinaryOp::Pow(_) => return Err(Unsupported("operator `**`".to_string())),
    })
}
