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
            NExpr, NPlace, NPlaceBase, NRootKind, NStatementKind, NTerminatorKind,
            NValueDefinition, NValueId, NormalizedBody, SConst, SemConstScalar, SemConstValue,
            SemanticInstance, identity_semantic_instance_key, normalize_semantic_body,
        },
        ty::{ty_check::BodyOwner, ty_def::TyId},
    },
    hir_def::Contract,
    semantic::ProviderSource,
};

/// A construct the exporter has no model for.
#[derive(Debug, PartialEq, Eq)]
pub struct Unsupported(pub String);

/// The Lean definitions of `contract`'s `init` and `recv` arms.
pub fn export_contract<'db>(
    db: &'db dyn HirAnalysisDb,
    contract: Contract<'db>,
) -> Result<String, Unsupported> {
    let mut out = String::new();
    let mut owners = vec![("init".to_string(), BodyOwner::ContractInit { contract })];
    for (recv_idx, recv) in contract.recv_views(db).enumerate() {
        for (arm_idx, _) in recv.arms(db).enumerate() {
            owners.push((
                format!("recv{recv_idx}_arm{arm_idx}"),
                BodyOwner::ContractRecvArm {
                    contract,
                    recv_idx: recv_idx as u32,
                    arm_idx: arm_idx as u32,
                },
            ));
        }
    }
    for (name, owner) in owners {
        let instance = SemanticInstance::new(db, identity_semantic_instance_key(db, owner));
        let normalized = normalize_semantic_body(db, instance)
            .map_err(|_| Unsupported(format!("{name}: body does not normalize")))?;
        out.push_str(&export_body(db, &name, &normalized.body)?);
        out.push('\n');
    }
    Ok(out)
}

fn export_body<'db>(
    db: &'db dyn HirAnalysisDb,
    name: &str,
    body: &NormalizedBody<'db>,
) -> Result<String, Unsupported> {
    let [block] = body.blocks.as_slice() else {
        return Err(Unsupported(format!("{name}: control flow")));
    };
    let mut params = String::new();
    for (index, value) in body.values.iter().enumerate() {
        if matches!(value.definition, NValueDefinition::EntryParam { .. }) {
            write!(params, " (v{index} : {})", lean_ty(db, value.ty)?).unwrap();
        }
    }
    let result = match &block.terminator.kind {
        NTerminatorKind::Return(None) => None,
        NTerminatorKind::Return(Some(operand)) => Some(operand.value),
        other => return Err(Unsupported(format!("{name}: terminator {other:?}"))),
    };
    let result_ty = match result {
        Some(value) => lean_ty(db, body.values[value.index()].ty)?,
        None => "Unit".to_string(),
    };
    let mut out = format!("def {name}{params} : FeM {result_ty} := do\n");
    for statement in &block.statements {
        match &statement.kind {
            NStatementKind::Define { result, expr } => {
                let rhs = match expr {
                    NExpr::Forward { src } => format!("pure {}", value_name(src.value)),
                    NExpr::Const(constant) => format!("pure {}", lean_const(db, constant)?),
                    NExpr::Load { place, .. } => format!("sload {}", slot(db, body, place)?),
                    other => return Err(Unsupported(format!("{name}: expression {other:?}"))),
                };
                writeln!(out, "  let {} ← {rhs}", value_name(*result)).unwrap();
            }
            NStatementKind::Store { destination, value } => {
                let slot = slot(db, body, destination)?;
                writeln!(out, "  sstore {slot} {}", value_name(value.value)).unwrap();
            }
            other => return Err(Unsupported(format!("{name}: statement {other:?}"))),
        }
    }
    match result {
        Some(value) => writeln!(out, "  return {}", value_name(value)).unwrap(),
        None => writeln!(out, "  return ()").unwrap(),
    }
    Ok(out)
}

fn value_name(value: NValueId) -> String {
    format!("v{}", value.index())
}

fn lean_ty<'db>(db: &'db dyn HirAnalysisDb, ty: TyId<'db>) -> Result<String, Unsupported> {
    match ty.pretty_print(db).as_str() {
        "u256" => Ok("UInt256".to_string()),
        "()" => Ok("Unit".to_string()),
        other => Err(Unsupported(format!("type {other}"))),
    }
}

fn lean_const<'db>(
    db: &'db dyn HirAnalysisDb,
    constant: &SConst<'db>,
) -> Result<String, Unsupported> {
    let SConst::Value(value) = constant else {
        return Err(Unsupported(format!("constant {constant:?}")));
    };
    match value.value().value(db) {
        SemConstValue::Unit => Ok("()".to_string()),
        SemConstValue::Scalar {
            value: SemConstScalar::Int { value },
            ..
        } => Ok(format!("(UInt256.ofNat {value})")),
        other => Err(Unsupported(format!("constant {other:?}"))),
    }
}

/// The storage slot of a place that is a whole contract field holding one
/// word. The slot is the field's index, which holds only while every field
/// before it takes one slot.
fn slot<'db>(
    db: &'db dyn HirAnalysisDb,
    body: &NormalizedBody<'db>,
    place: &NPlace<'db>,
) -> Result<String, Unsupported> {
    let NPlaceBase::Root(root) = place.base else {
        return Err(Unsupported(format!("place base {:?}", place.base)));
    };
    let root = &body.roots[root.index()];
    let NRootKind::Provider { binding } = &root.kind else {
        return Err(Unsupported(format!("root {:?}", root.kind)));
    };
    let ProviderSource::ContractField { field } = binding.source else {
        return Err(Unsupported(
            "provider that is not a contract field".to_string(),
        ));
    };
    if !place.path.is_empty() || lean_ty(db, place.ty)? != "UInt256" {
        return Err(Unsupported(
            "place that is not a one-word contract field".to_string(),
        ));
    }
    Ok(format!("(UInt256.ofNat {})", field.index))
}
