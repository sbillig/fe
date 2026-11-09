//! Exports the checked normalized body of each contract entry as a Lean
//! definition in the `Fe.EVM` model monad.
//!
//! A body the exporter has no model for is reported by the construct it
//! stopped at, never skipped.

use std::fmt::Write;

use cranelift_entity::EntityRef;
use hir::{
    analysis::{
        HirAnalysisDb,
        semantic::{
            BinaryOp, FailReason, NBlockId, NDataProjection, NExpr, NOperand, NPlace, NPlaceBase,
            NRootKind, NStatement, NStatementKind, NTerminatorKind, NValueDefinition, NValueId,
            NormalizedBody, Overflow, SConst, SemConstScalar, SemConstValue, SemanticInstance,
            identity_semantic_instance_key, normalize_semantic_body,
        },
        ty::{ty_check::BodyOwner, ty_def::TyId},
    },
    hir_def::{Contract, expr::CompBinOp},
    semantic::ProviderSource,
};
use rustc_hash::FxHashMap;

/// A construct the exporter has no model for.
#[derive(Debug, PartialEq, Eq)]
pub struct Unsupported(pub String);

/// The Lean definitions of `contract`'s `init` and `recv` arms.
pub fn export_contract<'db>(
    db: &'db dyn HirAnalysisDb,
    contract: Contract<'db>,
) -> Result<String, Unsupported> {
    let mut owners = vec![("init".to_string(), BodyOwner::ContractInit { contract })];
    for (recv_idx, recv) in contract.recv_views(db).enumerate() {
        owners.extend(recv.arms(db).enumerate().map(|(arm_idx, _)| {
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
    let mut out = String::new();
    for (name, owner) in owners {
        let instance = SemanticInstance::new(db, identity_semantic_instance_key(db, owner));
        let normalized = normalize_semantic_body(db, instance)
            .map_err(|_| Unsupported(format!("{name}: body does not normalize")))?;
        let exporter = BodyExporter {
            db,
            body: &normalized.body,
            aggregates: FxHashMap::default(),
        };
        out.push_str(
            &exporter
                .export(&name)
                .map_err(|e| Unsupported(format!("{name}: {}", e.0)))?,
        );
        out.push('\n');
    }
    Ok(out)
}

struct BodyExporter<'a, 'db> {
    db: &'db dyn HirAnalysisDb,
    body: &'a NormalizedBody<'db>,
    /// The type name and fields of each aggregate a body builds, which only
    /// a std call consumes.
    aggregates: FxHashMap<NValueId, (String, Vec<NValueId>)>,
}

impl<'db> BodyExporter<'_, 'db> {
    fn export(mut self, name: &str) -> Result<String, Unsupported> {
        let mut params = String::new();
        for (index, value) in self.body.values.iter().enumerate() {
            if matches!(value.definition, NValueDefinition::EntryParam { .. }) {
                write!(params, " (v{index} : {})", self.lean_ty(value.ty)?).unwrap();
            }
        }
        let result_ty = self.result_ty()?;
        let mut out = format!("def {name}{params} : FeM {result_ty} := do\n");
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
            Some(Some(operand)) => self.lean_ty(self.body.values[operand.value.index()].ty),
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
                message.data(self.db)
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
                    NExpr::Const(constant) => {
                        Some(format!("let {name} := {}", self.lean_const(constant)?))
                    }
                    NExpr::Load { place, .. } => {
                        Some(format!("let {name} ← sload {}", self.slot(place)?))
                    }
                    NExpr::Binary { op, lhs, rhs } => Some(format!(
                        "let {name} {}",
                        binary(*op, operand(lhs), operand(rhs))?
                    )),
                    // A view or borrow of the `Evm` root carries no data.
                    NExpr::MakeView { place, .. } | NExpr::Borrow { place, .. }
                        if self.is_root_object(place) =>
                    {
                        None
                    }
                    NExpr::AggregateMake { ty, fields } => {
                        let fields = fields.iter().map(|field| field.value).collect();
                        self.aggregates
                            .insert(*result, (ty.pretty_print(self.db).to_string(), fields));
                        None
                    }
                    NExpr::Call { callee, args, .. } => {
                        Some(self.call(&name, callee.key.owner(self.db), args)?)
                    }
                    other => return Err(Unsupported(format!("expression {other:?}"))),
                }
            }
            NStatementKind::Store { destination, value } => Some(format!(
                "sstore {} {}",
                self.slot(destination)?,
                operand(value)
            )),
            NStatementKind::End { .. } => None,
            other => return Err(Unsupported(format!("statement {other:?}"))),
        })
    }

    /// A call to a std function the prelude models. The table is by name:
    /// the functions are methods of `Evm`'s capability impls.
    fn call(
        &self,
        name: &str,
        owner: BodyOwner<'db>,
        args: &[NOperand],
    ) -> Result<String, Unsupported> {
        let BodyOwner::Func(func, _) = owner else {
            return Err(Unsupported("call to a non-function".to_string()));
        };
        let callee = func
            .name(self.db)
            .to_opt()
            .map(|ident| ident.data(self.db).to_string())
            .unwrap_or_default();
        match (callee.as_str(), args) {
            ("caller", [_evm]) => Ok(format!("let {name} ← caller")),
            ("zero", []) => Ok(format!("let {name} := UInt256.ofNat 0")),
            ("emit", [_evm, event]) => {
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
            _ => Err(Unsupported(format!("call to `{callee}`"))),
        }
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

    fn lean_ty(&self, ty: TyId<'db>) -> Result<String, Unsupported> {
        match ty.pretty_print(self.db).as_str() {
            // An address is its word.
            "u256" | "Address" => Ok("UInt256".to_string()),
            "bool" => Ok("Bool".to_string()),
            "()" => Ok("Unit".to_string()),
            other => Err(Unsupported(format!("type {other}"))),
        }
    }

    fn lean_const(&self, constant: &SConst<'db>) -> Result<String, Unsupported> {
        let SConst::Value(value) = constant else {
            return Err(Unsupported(format!("constant {constant:?}")));
        };
        match value.value().value(self.db) {
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

    /// The storage slot of a one-word place below a contract field: the
    /// field's index, plus the index of each struct field on the path, then
    /// `mapSlot` for each map entry. Counting one slot per field holds only
    /// while every field before the place takes one slot.
    fn slot(&self, place: &NPlace<'db>) -> Result<String, Unsupported> {
        let NPlaceBase::Root(root) = place.base else {
            return Err(Unsupported(format!("place base {:?}", place.base)));
        };
        let root = &self.body.roots[root.index()];
        let NRootKind::Provider { binding } = &root.kind else {
            return Err(Unsupported(format!("root {:?}", root.kind)));
        };
        let ProviderSource::ContractField { field } = binding.source else {
            return Err(Unsupported(
                "provider that is not a contract field".to_string(),
            ));
        };
        if self.lean_ty(place.ty)? != "UInt256" {
            return Err(Unsupported("place that is not one word".to_string()));
        }
        let mut number = field.index as usize;
        let mut slot = None;
        for projection in place.path.iter() {
            match (projection, &slot) {
                (NDataProjection::Field(index), None) => number += usize::from(index.0),
                (NDataProjection::Entry(key), _) => {
                    let root = slot.unwrap_or_else(|| format!("(UInt256.ofNat {number})"));
                    slot = Some(format!("(mapSlot {} {root})", value_name(*key)));
                }
                other => return Err(Unsupported(format!("place step {:?}", other.0))),
            }
        }
        Ok(slot.unwrap_or_else(|| format!("(UInt256.ofNat {number})")))
    }
}

fn value_name(value: NValueId) -> String {
    format!("v{}", value.index())
}

fn operand(operand: &NOperand) -> String {
    value_name(operand.value)
}

/// The right-hand side of `let v` for a built-in operation on words.
fn binary(op: BinaryOp, lhs: String, rhs: String) -> Result<String, Unsupported> {
    Ok(match op {
        BinaryOp::Add(Overflow::Checked) => format!("← checkedAdd {lhs} {rhs}"),
        BinaryOp::Sub(Overflow::Checked) => format!("← checkedSub {lhs} {rhs}"),
        BinaryOp::Comp(CompBinOp::GtEq) => format!(":= decide ({rhs}.toNat ≤ {lhs}.toNat)"),
        BinaryOp::Comp(CompBinOp::LtEq) => format!(":= decide ({lhs}.toNat ≤ {rhs}.toNat)"),
        BinaryOp::Comp(CompBinOp::Lt) => format!(":= decide ({lhs}.toNat < {rhs}.toNat)"),
        BinaryOp::Comp(CompBinOp::Gt) => format!(":= decide ({rhs}.toNat < {lhs}.toNat)"),
        other => return Err(Unsupported(format!("operator {other:?}"))),
    })
}
