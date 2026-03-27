use hir::analysis::HirAnalysisDb;
use hir::analysis::ty::pattern_ir::ConstructorKind;
use hir::analysis::ty::ty_check::{ReturnProvenance, check_func_body};
use hir::analysis::ty::ty_def::{PrimTy, TyBase, TyData, TyId};
use hir::hir_def::{CallableDef, EnumVariant};
use hir::projection::{IndexSource, Projection};
use rustc_hash::{FxHashMap, FxHashSet};

use crate::CoreLib;
use crate::ir::{
    AddressSpaceKind, CallOrigin, CallTargetRef, LocalData, LocalId, LocalPlaceRootLayout, MirBody,
    MirFunction, MirInst, MirModule, MirProjectionPath, Place, PointerInfo, RuntimeAbi,
    RuntimeShape, Rvalue, SourceInfoId, TerminatingCall, Terminator, ValueData, ValueId,
    ValueOrigin, ValueRepr,
};
use crate::layout;

mod lower_capability_to_repr;

pub(crate) use lower_capability_to_repr::{
    lower_capability_to_repr, prepare_body_for_evm_yul_codegen,
};

struct StabilizeCtx<'db, 'a, 'b> {
    db: &'db dyn HirAnalysisDb,
    values: &'a [ValueData<'db>],
    value_use_counts: &'a [usize],
    bound_in_block: &'b mut FxHashSet<ValueId>,
    rewritten: &'b mut Vec<MirInst<'db>>,
}

impl<'db> StabilizeCtx<'db, '_, '_> {
    fn stabilize_terminator(&mut self, term: &Terminator<'db>) {
        match term {
            Terminator::Return {
                value: Some(value), ..
            } => self.stabilize_value(*value, true, false),
            Terminator::TerminatingCall { call, .. } => match call {
                TerminatingCall::Call(call) => {
                    for arg in call.args.iter().chain(call.effect_args.iter()) {
                        self.stabilize_value(*arg, true, false);
                    }
                }
                TerminatingCall::Intrinsic { args, .. } => {
                    for arg in args {
                        self.stabilize_value(*arg, true, false);
                    }
                }
            },
            Terminator::Branch { cond, .. } => self.stabilize_value(*cond, true, false),
            Terminator::Switch { discr, .. } => self.stabilize_value(*discr, true, false),
            Terminator::Return { value: None, .. }
            | Terminator::Goto { .. }
            | Terminator::Unreachable { .. } => {}
        }
    }

    fn stabilize_path(&mut self, path: &crate::ir::MirProjectionPath<'db>) {
        for proj in path.iter() {
            if let Projection::Index(IndexSource::Dynamic(value)) = proj {
                self.stabilize_value(*value, true, false);
            }
        }
    }

    fn stabilize_value(&mut self, value: ValueId, bind_root: bool, force_root_bind: bool) {
        let mut visiting: FxHashSet<ValueId> = FxHashSet::default();
        self.stabilize_value_inner(value, bind_root, force_root_bind, &mut visiting);
    }

    fn stabilize_value_inner(
        &mut self,
        value: ValueId,
        bind_root: bool,
        force_root_bind: bool,
        visiting: &mut FxHashSet<ValueId>,
    ) {
        if !visiting.insert(value) {
            return;
        }

        let origin = &self.values[value.index()].origin;
        for dep in value_deps_in_eval_order(origin) {
            self.stabilize_value_inner(dep, true, false, visiting);
        }

        if bind_root
            && value_should_bind(
                self.db,
                value,
                &self.values[value.index()],
                origin,
                self.value_use_counts,
                force_root_bind,
            )
            && self.bound_in_block.insert(value)
        {
            self.rewritten.push(MirInst::BindValue {
                source: SourceInfoId::SYNTHETIC,
                value,
            });
        }
    }
}

pub(crate) fn insert_temp_binds<'db>(db: &'db dyn HirAnalysisDb, body: &mut MirBody<'db>) {
    let value_use_counts = compute_value_use_counts(body);
    let (blocks, values) = (&mut body.blocks, &body.values);
    for block in blocks {
        let mut bound_in_block: FxHashSet<ValueId> = FxHashSet::default();
        let mut rewritten: Vec<MirInst<'db>> = Vec::with_capacity(block.insts.len());
        {
            let mut ctx = StabilizeCtx {
                db,
                values,
                value_use_counts: &value_use_counts,
                bound_in_block: &mut bound_in_block,
                rewritten: &mut rewritten,
            };

            for inst in std::mem::take(&mut block.insts) {
                match inst {
                    MirInst::BindValue { value, .. } => {
                        ctx.stabilize_value(value, true, true);
                    }
                    MirInst::Assign {
                        source,
                        dest,
                        rvalue,
                    } => {
                        match &rvalue {
                            Rvalue::ZeroInit
                            | Rvalue::Alloc { .. }
                            | Rvalue::ConstAggregate { .. } => {}
                            Rvalue::Value(value) => {
                                ctx.stabilize_value(*value, dest.is_some(), false);
                            }
                            Rvalue::Call(call) => {
                                for arg in call.args.iter().chain(call.effect_args.iter()) {
                                    ctx.stabilize_value(*arg, true, false);
                                }
                            }
                            Rvalue::Intrinsic { args, .. } => {
                                for arg in args {
                                    ctx.stabilize_value(*arg, true, false);
                                }
                            }
                            Rvalue::Load { place } => {
                                ctx.stabilize_value(place.base, true, false);
                                ctx.stabilize_path(&place.projection);
                            }
                        }
                        ctx.rewritten.push(MirInst::Assign {
                            source,
                            dest,
                            rvalue,
                        });
                    }
                    MirInst::Store {
                        source,
                        place,
                        value,
                    } => {
                        ctx.stabilize_value(place.base, true, false);
                        ctx.stabilize_path(&place.projection);
                        ctx.stabilize_value(value, true, false);
                        ctx.rewritten.push(MirInst::Store {
                            source,
                            place,
                            value,
                        });
                    }
                    MirInst::InitAggregate {
                        source,
                        place,
                        inits,
                    } => {
                        ctx.stabilize_value(place.base, true, false);
                        ctx.stabilize_path(&place.projection);
                        for (path, value) in &inits {
                            ctx.stabilize_path(path);
                            ctx.stabilize_value(*value, true, false);
                        }
                        ctx.rewritten.push(MirInst::InitAggregate {
                            source,
                            place,
                            inits,
                        });
                    }
                    MirInst::SetDiscriminant {
                        source,
                        place,
                        variant,
                    } => {
                        ctx.stabilize_value(place.base, true, false);
                        ctx.stabilize_path(&place.projection);
                        ctx.rewritten.push(MirInst::SetDiscriminant {
                            source,
                            place,
                            variant,
                        });
                    }
                }
            }

            ctx.stabilize_terminator(&block.terminator);
        }
        block.insts = rewritten;
    }
}

pub fn prepare_module_for_evm_yul_codegen<'db>(
    db: &'db dyn HirAnalysisDb,
    module: &mut MirModule<'db>,
) {
    for func in &mut module.functions {
        let core = function_core_lib(db, func);
        prepare_body_for_evm_yul_codegen(db, &core, &mut func.body);
        canonicalize_transparent_newtypes(db, &mut func.body);
        insert_temp_binds(db, &mut func.body);
        canonicalize_zero_sized(db, &mut func.body);
    }
    normalize_runtime_shapes(db, module);
}

pub(crate) fn compute_live_values<'db>(body: &MirBody<'db>) -> Vec<bool> {
    fn mark_value_live<'db>(
        values: &[ValueData<'db>],
        live: &mut [bool],
        visiting: &mut FxHashSet<ValueId>,
        value: ValueId,
    ) {
        if !visiting.insert(value) {
            return;
        }
        let Some(slot) = live.get_mut(value.index()) else {
            return;
        };
        if *slot {
            return;
        }
        *slot = true;
        let Some(origin) = values.get(value.index()).map(|value| &value.origin) else {
            return;
        };
        for dep in value_deps_in_eval_order(origin) {
            mark_value_live(values, live, visiting, dep);
        }
        visiting.remove(&value);
    }

    let mut live = vec![false; body.values.len()];
    let values = &body.values;

    let mut mark_root = |value: ValueId| {
        let mut visiting = FxHashSet::default();
        mark_value_live(values, &mut live, &mut visiting, value);
    };

    for block in &body.blocks {
        for inst in &block.insts {
            match inst {
                MirInst::BindValue { value, .. } => mark_root(*value),
                MirInst::Assign { rvalue, .. } => match rvalue {
                    Rvalue::ZeroInit | Rvalue::Alloc { .. } | Rvalue::ConstAggregate { .. } => {}
                    Rvalue::Value(value) => mark_root(*value),
                    Rvalue::Call(call) => {
                        for arg in call.args.iter().chain(call.effect_args.iter()) {
                            mark_root(*arg);
                        }
                    }
                    Rvalue::Intrinsic { args, .. } => {
                        for arg in args {
                            mark_root(*arg);
                        }
                    }
                    Rvalue::Load { place } => {
                        mark_root(place.base);
                        bump_place_path(&mut mark_root, &place.projection);
                    }
                },
                MirInst::Store { place, value, .. } => {
                    mark_root(place.base);
                    bump_place_path(&mut mark_root, &place.projection);
                    mark_root(*value);
                }
                MirInst::InitAggregate { place, inits, .. } => {
                    mark_root(place.base);
                    bump_place_path(&mut mark_root, &place.projection);
                    for (path, value) in inits {
                        bump_place_path(&mut mark_root, path);
                        mark_root(*value);
                    }
                }
                MirInst::SetDiscriminant { place, .. } => {
                    mark_root(place.base);
                    bump_place_path(&mut mark_root, &place.projection);
                }
            }
        }

        match &block.terminator {
            Terminator::Return {
                value: Some(value), ..
            } => mark_root(*value),
            Terminator::TerminatingCall { call, .. } => match call {
                TerminatingCall::Call(call) => {
                    for arg in call.args.iter().chain(call.effect_args.iter()) {
                        mark_root(*arg);
                    }
                }
                TerminatingCall::Intrinsic { args, .. } => {
                    for arg in args {
                        mark_root(*arg);
                    }
                }
            },
            Terminator::Branch { cond, .. } => mark_root(*cond),
            Terminator::Switch { discr, .. } => mark_root(*discr),
            Terminator::Return { value: None, .. }
            | Terminator::Goto { .. }
            | Terminator::Unreachable { .. } => {}
        }
    }

    live
}

fn has_runtime_param_representation(db: &dyn HirAnalysisDb, ty: TyId<'_>) -> bool {
    !layout::is_zero_sized_ty(db, ty)
        && !ty
            .as_capability(db)
            .is_some_and(|(_, inner)| layout::is_zero_sized_ty(db, inner))
}

fn is_contract_entry_function(func: &MirFunction<'_>) -> bool {
    use crate::ir::{ContractFunctionKind, MirFunctionOrigin, SyntheticId};

    func.contract_function.as_ref().is_some_and(|cf| {
        matches!(
            cf.kind,
            ContractFunctionKind::Init | ContractFunctionKind::Runtime
        )
    }) || matches!(
        func.origin,
        MirFunctionOrigin::Synthetic(
            SyntheticId::ContractInitEntrypoint(_) | SyntheticId::ContractRuntimeEntrypoint(_)
        )
    )
}

fn add_place_runtime_uses<'db>(
    db: &'db dyn HirAnalysisDb,
    body: &MirBody<'db>,
    place: &Place<'db>,
    live_locals: &mut FxHashSet<LocalId>,
    seen_values: &mut FxHashSet<ValueId>,
) {
    add_value_runtime_uses(db, body, place.base, live_locals, seen_values);
    for proj in place.projection.iter() {
        if let Projection::Index(IndexSource::Dynamic(value)) = proj {
            add_value_runtime_uses(db, body, *value, live_locals, seen_values);
        }
    }
}

fn add_value_runtime_uses<'db>(
    db: &'db dyn HirAnalysisDb,
    body: &MirBody<'db>,
    value: ValueId,
    live_locals: &mut FxHashSet<LocalId>,
    seen_values: &mut FxHashSet<ValueId>,
) {
    if !seen_values.insert(value) {
        return;
    }
    let Some(value_data) = body.values.get(value.index()) else {
        return;
    };
    if layout::is_zero_sized_ty(db, value_data.ty) {
        return;
    }

    match &value_data.origin {
        ValueOrigin::Local(local) | ValueOrigin::PlaceRoot(local) => {
            live_locals.insert(*local);
        }
        ValueOrigin::Unary { inner, .. } => {
            add_value_runtime_uses(db, body, *inner, live_locals, seen_values);
        }
        ValueOrigin::Binary { lhs, rhs, .. } => {
            add_value_runtime_uses(db, body, *lhs, live_locals, seen_values);
            add_value_runtime_uses(db, body, *rhs, live_locals, seen_values);
        }
        ValueOrigin::FieldPtr(field_ptr) => {
            add_value_runtime_uses(db, body, field_ptr.base, live_locals, seen_values);
        }
        ValueOrigin::PlaceRef(place) | ValueOrigin::MoveOut { place } => {
            add_place_runtime_uses(db, body, place, live_locals, seen_values);
        }
        ValueOrigin::TransparentCast { value } => {
            add_value_runtime_uses(db, body, *value, live_locals, seen_values);
        }
        ValueOrigin::Expr(..)
        | ValueOrigin::ControlFlowResult { .. }
        | ValueOrigin::Unit
        | ValueOrigin::ConstRegion(_)
        | ValueOrigin::Synthetic(..)
        | ValueOrigin::CodeRegionRef(..) => {}
    }
}

fn add_runtime_call_arg_uses<'db>(
    db: &'db dyn HirAnalysisDb,
    body: &MirBody<'db>,
    call: &CallOrigin<'db>,
    runtime_abis: &[RuntimeAbi<'db>],
    func_indices: &FxHashMap<String, usize>,
    live_locals: &mut FxHashSet<LocalId>,
    seen_values: &mut FxHashSet<ValueId>,
) {
    let Some(callee_name) = call.resolved_name.as_deref() else {
        for &arg in call.args.iter().chain(call.effect_args.iter()) {
            add_value_runtime_uses(db, body, arg, live_locals, seen_values);
        }
        return;
    };
    let Some(&func_idx) = func_indices.get(callee_name) else {
        for &arg in call.args.iter().chain(call.effect_args.iter()) {
            add_value_runtime_uses(db, body, arg, live_locals, seen_values);
        }
        return;
    };
    let Some(runtime_abi) = runtime_abis.get(func_idx) else {
        return;
    };

    for (idx, &arg) in call.args.iter().enumerate() {
        if runtime_abi.value_param_visible(idx) {
            add_value_runtime_uses(db, body, arg, live_locals, seen_values);
        }
    }
    for (idx, &arg) in call.effect_args.iter().enumerate() {
        if runtime_abi.effect_param_visible(idx) {
            add_value_runtime_uses(db, body, arg, live_locals, seen_values);
        }
    }
}

fn runtime_successors(term: &Terminator<'_>) -> Vec<crate::ir::BasicBlockId> {
    match term {
        Terminator::Goto { target, .. } => vec![*target],
        Terminator::Branch {
            then_bb, else_bb, ..
        } => vec![*then_bb, *else_bb],
        Terminator::Switch {
            targets, default, ..
        } => {
            let mut out = Vec::with_capacity(targets.len() + 1);
            out.extend(targets.iter().map(|target| target.block));
            out.push(*default);
            out
        }
        Terminator::Return { .. }
        | Terminator::TerminatingCall { .. }
        | Terminator::Unreachable { .. } => Vec::new(),
    }
}

fn transfer_runtime_terminator<'db>(
    db: &'db dyn HirAnalysisDb,
    body: &MirBody<'db>,
    term: &Terminator<'db>,
    runtime_abis: &[RuntimeAbi<'db>],
    func_indices: &FxHashMap<String, usize>,
    live: &mut FxHashSet<LocalId>,
) {
    let mut seen_values = FxHashSet::default();
    match term {
        Terminator::Return {
            value: Some(value), ..
        } => add_value_runtime_uses(db, body, *value, live, &mut seen_values),
        Terminator::TerminatingCall { call, .. } => match call {
            TerminatingCall::Call(call) => add_runtime_call_arg_uses(
                db,
                body,
                call,
                runtime_abis,
                func_indices,
                live,
                &mut seen_values,
            ),
            TerminatingCall::Intrinsic { op, args, .. } => {
                if !op.returns_value() || matches!(op, crate::ir::IntrinsicOp::Alloc) {
                    for &arg in args {
                        add_value_runtime_uses(db, body, arg, live, &mut seen_values);
                    }
                }
            }
        },
        Terminator::Branch { cond, .. } => {
            add_value_runtime_uses(db, body, *cond, live, &mut seen_values);
        }
        Terminator::Switch { discr, .. } => {
            add_value_runtime_uses(db, body, *discr, live, &mut seen_values);
        }
        Terminator::Return { value: None, .. }
        | Terminator::Goto { .. }
        | Terminator::Unreachable { .. } => {}
    }
}

fn transfer_runtime_inst<'db>(
    db: &'db dyn HirAnalysisDb,
    body: &MirBody<'db>,
    inst: &MirInst<'db>,
    runtime_abis: &[RuntimeAbi<'db>],
    func_indices: &FxHashMap<String, usize>,
    live: &mut FxHashSet<LocalId>,
) {
    let mut seen_values = FxHashSet::default();
    match inst {
        MirInst::Assign { dest, rvalue, .. } => {
            let dest_live = dest.is_some_and(|local| live.remove(&local));
            match rvalue {
                Rvalue::ZeroInit | Rvalue::Alloc { .. } | Rvalue::ConstAggregate { .. } => {}
                Rvalue::Value(value) => {
                    if dest.is_none() || dest_live {
                        add_value_runtime_uses(db, body, *value, live, &mut seen_values);
                    }
                }
                Rvalue::Load { place } => {
                    if dest.is_none() || dest_live {
                        add_place_runtime_uses(db, body, place, live, &mut seen_values);
                    }
                }
                Rvalue::Call(call) => add_runtime_call_arg_uses(
                    db,
                    body,
                    call,
                    runtime_abis,
                    func_indices,
                    live,
                    &mut seen_values,
                ),
                Rvalue::Intrinsic { op, args } => {
                    if dest.is_none()
                        || dest_live
                        || !op.returns_value()
                        || matches!(op, crate::ir::IntrinsicOp::Alloc)
                    {
                        for &arg in args {
                            add_value_runtime_uses(db, body, arg, live, &mut seen_values);
                        }
                    }
                }
            }
        }
        MirInst::Store { place, value, .. } => {
            add_place_runtime_uses(db, body, place, live, &mut seen_values);
            add_value_runtime_uses(db, body, *value, live, &mut seen_values);
        }
        MirInst::InitAggregate { place, inits, .. } => {
            add_place_runtime_uses(db, body, place, live, &mut seen_values);
            for (path, value) in inits {
                for proj in path.iter() {
                    if let Projection::Index(IndexSource::Dynamic(index)) = proj {
                        add_value_runtime_uses(db, body, *index, live, &mut seen_values);
                    }
                }
                add_value_runtime_uses(db, body, *value, live, &mut seen_values);
            }
        }
        MirInst::SetDiscriminant { place, .. } => {
            add_place_runtime_uses(db, body, place, live, &mut seen_values);
        }
        MirInst::BindValue { .. } => {}
    }
}

fn compute_entry_runtime_live_locals<'db>(
    db: &'db dyn HirAnalysisDb,
    func: &MirFunction<'db>,
    runtime_abis: &[RuntimeAbi<'db>],
    func_indices: &FxHashMap<String, usize>,
) -> FxHashSet<LocalId> {
    let block_count = func.body.blocks.len();
    let mut live_in = vec![FxHashSet::default(); block_count];
    let mut changed = true;
    while changed {
        changed = false;
        for block_idx in (0..block_count).rev() {
            let block = &func.body.blocks[block_idx];
            let mut live = FxHashSet::default();
            for succ in runtime_successors(&block.terminator) {
                live.extend(live_in[succ.index()].iter().copied());
            }
            transfer_runtime_terminator(
                db,
                &func.body,
                &block.terminator,
                runtime_abis,
                func_indices,
                &mut live,
            );
            for inst in block.insts.iter().rev() {
                transfer_runtime_inst(db, &func.body, inst, runtime_abis, func_indices, &mut live);
            }
            if live != live_in[block_idx] {
                live_in[block_idx] = live;
                changed = true;
            }
        }
    }
    live_in
        .get(func.body.entry.index())
        .cloned()
        .unwrap_or_default()
}

fn compute_runtime_abi<'db>(
    db: &'db dyn HirAnalysisDb,
    func: &MirFunction<'db>,
    runtime_abis: &[RuntimeAbi<'db>],
    func_indices: &FxHashMap<String, usize>,
) -> RuntimeAbi<'db> {
    if is_contract_entry_function(func) {
        return RuntimeAbi {
            value_params: vec![false; func.body.param_locals.len()],
            effect_params: vec![false; func.body.effect_param_locals.len()],
            effect_param_provider_tys: func.runtime_abi.effect_param_provider_tys.clone(),
        };
    }

    let live_at_entry = compute_entry_runtime_live_locals(db, func, runtime_abis, func_indices);
    let value_params = func
        .body
        .param_locals
        .iter()
        .copied()
        .map(|local| {
            live_at_entry.contains(&local)
                && has_runtime_param_representation(db, func.body.local(local).ty)
        })
        .collect();
    let effect_params = func
        .body
        .effect_param_locals
        .iter()
        .copied()
        .enumerate()
        .map(|(idx, local)| {
            live_at_entry.contains(&local)
                && func
                    .runtime_abi
                    .effect_param_provider_tys
                    .get(idx)
                    .copied()
                    .flatten()
                    .map(|ty| has_runtime_param_representation(db, ty))
                    .unwrap_or_else(|| {
                        has_runtime_param_representation(db, func.body.local(local).ty)
                    })
        })
        .collect();

    RuntimeAbi {
        value_params,
        effect_params,
        effect_param_provider_tys: func.runtime_abi.effect_param_provider_tys.clone(),
    }
}

fn rewrite_runtime_call_args<'db>(
    call: &mut CallOrigin<'db>,
    runtime_abis: &[RuntimeAbi<'db>],
    func_indices: &FxHashMap<String, usize>,
) {
    let Some(callee_name) = call.resolved_name.as_deref() else {
        return;
    };
    let Some(&func_idx) = func_indices.get(callee_name) else {
        return;
    };
    let Some(runtime_abi) = runtime_abis.get(func_idx) else {
        return;
    };

    call.args = call
        .args
        .iter()
        .copied()
        .enumerate()
        .filter_map(|(idx, arg)| runtime_abi.value_param_visible(idx).then_some(arg))
        .collect();
    call.effect_args = call
        .effect_args
        .iter()
        .copied()
        .enumerate()
        .filter_map(|(idx, arg)| runtime_abi.effect_param_visible(idx).then_some(arg))
        .collect();
}

fn path_has_deref<'db>(path: &MirProjectionPath<'db>) -> bool {
    path.iter().any(|proj| matches!(proj, Projection::Deref))
}

fn path_has_index<'db>(path: &MirProjectionPath<'db>) -> bool {
    path.iter().any(|proj| matches!(proj, Projection::Index(_)))
}

/// Returns `true` when any projection in the full resolution chain of `place`
/// (both the base-value prefix obtained via `resolve_local_projection_root` and
/// the explicit `place.projection`) contains an `Index` step.  Evaluating such
/// a place has a side effect in codegen (an OOB bounds-check revert), so
/// instructions that touch it must not be eliminated as dead code.
fn place_has_indexed_projection<'db>(body: &MirBody<'db>, place: &Place<'db>) -> bool {
    if path_has_index(&place.projection) {
        return true;
    }
    value_has_indexed_origin(&body.values, place.base)
}

/// Walks the value origin chain and returns `true` if any `PlaceRef` or
/// `MoveOut` along the way contains an `Index` projection.  This catches
/// by-ref element accesses that are wrapped in `Rvalue::Value(value_id)`
/// rather than `Rvalue::Load { place }`.
fn value_has_indexed_origin<'db>(values: &[ValueData<'db>], value: ValueId) -> bool {
    use crate::ir::ValueOrigin;
    let mut current = value;
    loop {
        let Some(data) = values.get(current.index()) else {
            return false;
        };
        match &data.origin {
            ValueOrigin::TransparentCast { value } => current = *value,
            ValueOrigin::PlaceRef(place) | ValueOrigin::MoveOut { place } => {
                if path_has_index(&place.projection) {
                    return true;
                }
                current = place.base;
            }
            _ => return false,
        }
    }
}

fn removable_memory_root_local_for_place<'db>(
    body: &MirBody<'db>,
    place: &Place<'db>,
) -> Option<LocalId> {
    let (local, prefix) = crate::ir::resolve_local_projection_root(&body.values, place.base)?;
    if path_has_deref(&prefix) || path_has_deref(&place.projection) {
        return None;
    }
    // Stores/loads through indexed projections must not be eliminated: the
    // codegen emits a bounds check (revert on OOB) as a side effect of
    // evaluating the index.  Removing the instruction would silently drop
    // that check.
    if path_has_index(&prefix) || path_has_index(&place.projection) {
        return None;
    }
    (crate::ir::try_place_address_space_in(&body.values, &body.locals, place)
        == Some(AddressSpaceKind::Memory))
    .then_some(local)
}

fn removable_assign_rvalue<'db>(body: &MirBody<'db>, rvalue: &Rvalue<'db>) -> bool {
    match rvalue {
        Rvalue::ZeroInit | Rvalue::Alloc { .. } | Rvalue::ConstAggregate { .. } => true,
        // A plain value is removable unless it originates from an indexed
        // place (PlaceRef/MoveOut with an Index projection).  By-ref element
        // accesses lower to PlaceRef values, and evaluating the place in
        // codegen triggers a bounds check that must not be dropped.
        Rvalue::Value(value) => !value_has_indexed_origin(&body.values, *value),
        // Loads through indexed projections are not removable: the codegen
        // emits a bounds check (revert on OOB) as a side effect of evaluating
        // the index.  Removing the load would silently drop that check.
        // We check both `place.projection` and the base-value prefix to
        // cover nested accesses like `arr[i].field`.
        Rvalue::Load { place } => !place_has_indexed_projection(body, place),
        Rvalue::Call(_) | Rvalue::Intrinsic { .. } => false,
    }
}

fn removable_bind_value<'db>(body: &MirBody<'db>, value: ValueId) -> bool {
    // Binding an indexed place is not side-effect-free: lowering the value
    // evaluates the place and emits the OOB bounds check in codegen.
    !value_has_indexed_origin(&body.values, value)
}

fn mark_value_runtime_live<'db>(
    body: &MirBody<'db>,
    live_values: &mut [bool],
    live_locals: &mut FxHashSet<LocalId>,
    value: ValueId,
) -> bool {
    let Some(slot) = live_values.get_mut(value.index()) else {
        return false;
    };
    if *slot {
        return false;
    }
    *slot = true;

    let Some(data) = body.values.get(value.index()) else {
        return true;
    };
    let mut changed = true;
    match &data.origin {
        ValueOrigin::Unary { inner, .. } => {
            changed |= mark_value_runtime_live(body, live_values, live_locals, *inner);
        }
        ValueOrigin::Binary { lhs, rhs, .. } => {
            changed |= mark_value_runtime_live(body, live_values, live_locals, *lhs);
            changed |= mark_value_runtime_live(body, live_values, live_locals, *rhs);
        }
        ValueOrigin::Local(local) | ValueOrigin::PlaceRoot(local) => {
            changed |= live_locals.insert(*local);
        }
        ValueOrigin::FieldPtr(field_ptr) => {
            changed |= mark_value_runtime_live(body, live_values, live_locals, field_ptr.base);
        }
        ValueOrigin::PlaceRef(place) | ValueOrigin::MoveOut { place } => {
            changed |= mark_value_runtime_live(body, live_values, live_locals, place.base);
            for proj in place.projection.iter() {
                if let Projection::Index(IndexSource::Dynamic(value)) = proj {
                    changed |= mark_value_runtime_live(body, live_values, live_locals, *value);
                }
            }
        }
        ValueOrigin::TransparentCast { value } => {
            changed |= mark_value_runtime_live(body, live_values, live_locals, *value);
        }
        ValueOrigin::Expr(..)
        | ValueOrigin::ControlFlowResult { .. }
        | ValueOrigin::Unit
        | ValueOrigin::ConstRegion(_)
        | ValueOrigin::Synthetic(..)
        | ValueOrigin::CodeRegionRef(..) => {}
    }
    changed
}

fn mark_place_runtime_live<'db>(
    body: &MirBody<'db>,
    live_values: &mut [bool],
    live_locals: &mut FxHashSet<LocalId>,
    place: &Place<'db>,
) -> bool {
    let mut changed = mark_value_runtime_live(body, live_values, live_locals, place.base);
    for proj in place.projection.iter() {
        if let Projection::Index(IndexSource::Dynamic(value)) = proj {
            changed |= mark_value_runtime_live(body, live_values, live_locals, *value);
        }
    }
    changed
}

fn instruction_is_runtime_root<'db>(
    body: &MirBody<'db>,
    inst: &MirInst<'db>,
    live_values: &[bool],
    live_locals: &FxHashSet<LocalId>,
) -> bool {
    match inst {
        MirInst::Assign { dest, rvalue, .. } => match dest {
            Some(local) if removable_assign_rvalue(body, rvalue) => live_locals.contains(local),
            None if removable_assign_rvalue(body, rvalue) => false,
            _ => true,
        },
        MirInst::Store { place, .. }
        | MirInst::InitAggregate { place, .. }
        | MirInst::SetDiscriminant { place, .. } => {
            removable_memory_root_local_for_place(body, place)
                .is_none_or(|local| live_locals.contains(&local))
        }
        MirInst::BindValue { value, .. } => {
            !removable_bind_value(body, *value)
                || live_values.get(value.index()).copied().unwrap_or(false)
        }
    }
}

fn mark_runtime_inst_live_operands<'db>(
    body: &MirBody<'db>,
    inst: &MirInst<'db>,
    live_values: &mut [bool],
    live_locals: &mut FxHashSet<LocalId>,
) -> bool {
    let mut changed = false;
    match inst {
        MirInst::Assign { rvalue, .. } => match rvalue {
            Rvalue::ZeroInit | Rvalue::Alloc { .. } | Rvalue::ConstAggregate { .. } => {}
            Rvalue::Value(value) => {
                changed |= mark_value_runtime_live(body, live_values, live_locals, *value);
            }
            Rvalue::Call(call) => {
                for arg in call.args.iter().chain(call.effect_args.iter()) {
                    changed |= mark_value_runtime_live(body, live_values, live_locals, *arg);
                }
            }
            Rvalue::Intrinsic { args, .. } => {
                for arg in args {
                    changed |= mark_value_runtime_live(body, live_values, live_locals, *arg);
                }
            }
            Rvalue::Load { place } => {
                changed |= mark_place_runtime_live(body, live_values, live_locals, place);
            }
        },
        MirInst::Store { place, value, .. } => {
            changed |= mark_place_runtime_live(body, live_values, live_locals, place);
            changed |= mark_value_runtime_live(body, live_values, live_locals, *value);
        }
        MirInst::InitAggregate { place, inits, .. } => {
            changed |= mark_place_runtime_live(body, live_values, live_locals, place);
            for (path, value) in inits {
                for proj in path.iter() {
                    if let Projection::Index(IndexSource::Dynamic(index)) = proj {
                        changed |= mark_value_runtime_live(body, live_values, live_locals, *index);
                    }
                }
                changed |= mark_value_runtime_live(body, live_values, live_locals, *value);
            }
        }
        MirInst::SetDiscriminant { place, .. } => {
            changed |= mark_place_runtime_live(body, live_values, live_locals, place);
        }
        MirInst::BindValue { value, .. } => {
            changed |= mark_value_runtime_live(body, live_values, live_locals, *value);
        }
    }
    changed
}

fn mark_runtime_terminator_live_operands<'db>(
    body: &MirBody<'db>,
    terminator: &Terminator<'db>,
    live_values: &mut [bool],
    live_locals: &mut FxHashSet<LocalId>,
) -> bool {
    let mut changed = false;
    match terminator {
        Terminator::Return {
            value: Some(value), ..
        } => {
            changed |= mark_value_runtime_live(body, live_values, live_locals, *value);
        }
        Terminator::TerminatingCall { call, .. } => match call {
            TerminatingCall::Call(call) => {
                for arg in call.args.iter().chain(call.effect_args.iter()) {
                    changed |= mark_value_runtime_live(body, live_values, live_locals, *arg);
                }
            }
            TerminatingCall::Intrinsic { args, .. } => {
                for arg in args {
                    changed |= mark_value_runtime_live(body, live_values, live_locals, *arg);
                }
            }
        },
        Terminator::Branch { cond, .. } => {
            changed |= mark_value_runtime_live(body, live_values, live_locals, *cond);
        }
        Terminator::Switch { discr, .. } => {
            changed |= mark_value_runtime_live(body, live_values, live_locals, *discr);
        }
        Terminator::Return { value: None, .. }
        | Terminator::Goto { .. }
        | Terminator::Unreachable { .. } => {}
    }
    changed
}

pub(crate) fn eliminate_dead_erased_arg_materializations<'db>(
    _db: &'db dyn HirAnalysisDb,
    module: &mut MirModule<'db>,
) {
    for func in &mut module.functions {
        let body = &func.body;
        let mut live_values = vec![false; body.values.len()];
        let mut live_locals = FxHashSet::default();

        loop {
            let mut changed = false;
            for block in &body.blocks {
                changed |= mark_runtime_terminator_live_operands(
                    body,
                    &block.terminator,
                    &mut live_values,
                    &mut live_locals,
                );
                for inst in &block.insts {
                    if instruction_is_runtime_root(body, inst, &live_values, &live_locals) {
                        changed |= mark_runtime_inst_live_operands(
                            body,
                            inst,
                            &mut live_values,
                            &mut live_locals,
                        );
                    }
                }
            }
            if !changed {
                break;
            }
        }

        let body_ref = &func.body;
        let keep_insts: Vec<Vec<bool>> = body_ref
            .blocks
            .iter()
            .map(|block| {
                block
                    .insts
                    .iter()
                    .map(|inst| {
                        instruction_is_runtime_root(body_ref, inst, &live_values, &live_locals)
                    })
                    .collect()
            })
            .collect();

        for (block, keep) in func.body.blocks.iter_mut().zip(keep_insts) {
            block.insts = std::mem::take(&mut block.insts)
                .into_iter()
                .zip(keep)
                .filter_map(|(inst, keep)| keep.then_some(inst))
                .collect();
        }
    }
}

pub(crate) fn normalize_runtime_abi<'db>(db: &'db dyn HirAnalysisDb, module: &mut MirModule<'db>) {
    let func_indices: FxHashMap<_, _> = module
        .functions
        .iter()
        .enumerate()
        .filter(|(_, func)| !func.symbol_name.is_empty())
        .map(|(idx, func)| (func.symbol_name.clone(), idx))
        .collect();

    loop {
        let current_abis: Vec<_> = module
            .functions
            .iter()
            .map(|func| func.runtime_abi.clone())
            .collect();
        let mut changed = false;
        for func in &mut module.functions {
            let next = compute_runtime_abi(db, func, &current_abis, &func_indices);
            if func.runtime_abi.value_params != next.value_params
                || func.runtime_abi.effect_params != next.effect_params
            {
                func.runtime_abi.value_params = next.value_params;
                func.runtime_abi.effect_params = next.effect_params;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }

    let final_abis: Vec<_> = module
        .functions
        .iter()
        .map(|func| func.runtime_abi.clone())
        .collect();
    for func in &mut module.functions {
        for block in &mut func.body.blocks {
            for inst in &mut block.insts {
                match inst {
                    MirInst::Assign {
                        rvalue: Rvalue::Call(call),
                        ..
                    } => rewrite_runtime_call_args(call, &final_abis, &func_indices),
                    MirInst::Assign { .. }
                    | MirInst::Store { .. }
                    | MirInst::InitAggregate { .. }
                    | MirInst::SetDiscriminant { .. }
                    | MirInst::BindValue { .. } => {}
                }
            }
            if let Terminator::TerminatingCall {
                call: TerminatingCall::Call(call),
                ..
            } = &mut block.terminator
            {
                rewrite_runtime_call_args(call, &final_abis, &func_indices);
            }
        }
    }
}

fn function_core_lib<'db>(db: &'db dyn HirAnalysisDb, func: &MirFunction<'db>) -> CoreLib<'db> {
    let scope = match func.origin {
        crate::ir::MirFunctionOrigin::Hir(hir_func) => hir_func.scope(),
        crate::ir::MirFunctionOrigin::Synthetic(synth) => synth.contract().scope(),
    };
    CoreLib::new(db, scope)
}

fn runtime_return_shape_seed_for_func<'db>(
    db: &'db dyn HirAnalysisDb,
    func: &MirFunction<'db>,
    core: &CoreLib<'db>,
) -> RuntimeShape<'db> {
    if let crate::ir::MirFunctionOrigin::Hir(hir_func) = func.origin
        && let (diags, typed_body) = check_func_body(db, hir_func)
        && diags.is_empty()
        && matches!(
            typed_body.return_provenance(db),
            ReturnProvenance::ForwardedParams(ref indices) if !indices.is_empty()
        )
    {
        return RuntimeShape::Unresolved;
    }

    crate::repr::runtime_return_shape_seed_for_ty(db, core, func.ret_ty)
}

fn merge_runtime_shapes<'db>(
    existing: RuntimeShape<'db>,
    next: RuntimeShape<'db>,
) -> Option<RuntimeShape<'db>> {
    match (existing, next) {
        (RuntimeShape::Unresolved, shape) | (shape, RuntimeShape::Unresolved) => Some(shape),
        (RuntimeShape::Erased, shape) | (shape, RuntimeShape::Erased) => Some(shape),
        (RuntimeShape::Word(lhs), RuntimeShape::Word(rhs)) if lhs == rhs => {
            Some(RuntimeShape::Word(lhs))
        }
        (RuntimeShape::EnumTag { enum_ty: lhs }, RuntimeShape::EnumTag { enum_ty: rhs })
            if lhs == rhs =>
        {
            Some(RuntimeShape::EnumTag { enum_ty: lhs })
        }
        (
            RuntimeShape::Word(crate::ir::RuntimeWordKind::I256),
            RuntimeShape::EnumTag { enum_ty },
        )
        | (
            RuntimeShape::EnumTag { enum_ty },
            RuntimeShape::Word(crate::ir::RuntimeWordKind::I256),
        ) => Some(RuntimeShape::EnumTag { enum_ty }),
        (
            RuntimeShape::ObjectRef {
                target_ty: lhs_target,
            },
            RuntimeShape::ObjectRef {
                target_ty: rhs_target,
            },
        ) if lhs_target == rhs_target => Some(RuntimeShape::ObjectRef {
            target_ty: lhs_target,
        }),
        (
            RuntimeShape::ConstRef {
                target_ty: lhs_target,
            },
            RuntimeShape::ConstRef {
                target_ty: rhs_target,
            },
        ) if lhs_target == rhs_target => Some(RuntimeShape::ConstRef {
            target_ty: lhs_target,
        }),
        (
            RuntimeShape::MemoryPtr {
                target_ty: lhs_target,
            },
            RuntimeShape::MemoryPtr {
                target_ty: rhs_target,
            },
        ) => Some(RuntimeShape::MemoryPtr {
            target_ty: lhs_target.or(rhs_target),
        }),
        (RuntimeShape::AddressWord(lhs), RuntimeShape::AddressWord(rhs))
            if lhs.address_space == rhs.address_space =>
        {
            Some(RuntimeShape::AddressWord(crate::ir::PointerInfo {
                address_space: lhs.address_space,
                target_ty: lhs.target_ty.or(rhs.target_ty),
            }))
        }
        (
            RuntimeShape::Word(crate::ir::RuntimeWordKind::I256),
            RuntimeShape::MemoryPtr { target_ty },
        )
        | (
            RuntimeShape::MemoryPtr { target_ty },
            RuntimeShape::Word(crate::ir::RuntimeWordKind::I256),
        ) => Some(RuntimeShape::MemoryPtr { target_ty }),
        (RuntimeShape::Word(crate::ir::RuntimeWordKind::I256), RuntimeShape::AddressWord(info))
        | (RuntimeShape::AddressWord(info), RuntimeShape::Word(crate::ir::RuntimeWordKind::I256)) => {
            Some(RuntimeShape::AddressWord(info))
        }
        (RuntimeShape::MemoryPtr { target_ty }, RuntimeShape::AddressWord(info))
        | (RuntimeShape::AddressWord(info), RuntimeShape::MemoryPtr { target_ty })
            if info.address_space != AddressSpaceKind::Memory
                && match (target_ty, info.target_ty) {
                    (Some(memory_target_ty), Some(address_target_ty)) => {
                        memory_target_ty == address_target_ty
                    }
                    _ => true,
                } =>
        {
            Some(RuntimeShape::AddressWord(crate::ir::PointerInfo {
                address_space: info.address_space,
                target_ty: info.target_ty.or(target_ty),
            }))
        }
        _ => None,
    }
}

fn merge_runtime_shapes_in_context<'db>(
    db: &'db dyn HirAnalysisDb,
    core: &CoreLib<'db>,
    existing: RuntimeShape<'db>,
    next: RuntimeShape<'db>,
) -> Option<RuntimeShape<'db>> {
    merge_runtime_shapes(existing, next).or_else(|| match (existing, next) {
        (
            RuntimeShape::ConstRef { target_ty },
            RuntimeShape::ObjectRef {
                target_ty: object_target_ty,
            },
        )
        | (
            RuntimeShape::ObjectRef {
                target_ty: object_target_ty,
            },
            RuntimeShape::ConstRef { target_ty },
        ) => (crate::repr::object_layout_ty(db, core, target_ty) == object_target_ty).then_some(
            RuntimeShape::ObjectRef {
                target_ty: object_target_ty,
            },
        ),
        (
            RuntimeShape::ObjectRef {
                target_ty: lhs_target,
            },
            RuntimeShape::ObjectRef {
                target_ty: rhs_target,
            },
        ) if crate::repr::runtime_ty_matches(db, lhs_target, rhs_target) => {
            Some(RuntimeShape::ObjectRef {
                target_ty: lhs_target,
            })
        }
        (
            RuntimeShape::MemoryPtr { target_ty },
            RuntimeShape::ObjectRef {
                target_ty: object_target_ty,
            },
        )
        | (
            RuntimeShape::ObjectRef {
                target_ty: object_target_ty,
            },
            RuntimeShape::MemoryPtr { target_ty },
        ) => target_ty
            .filter(|target_ty| {
                crate::repr::runtime_ty_matches(
                    db,
                    crate::repr::object_layout_ty(db, core, *target_ty),
                    object_target_ty,
                )
            })
            .map(|target_ty| RuntimeShape::MemoryPtr {
                target_ty: Some(target_ty),
            }),
        (
            RuntimeShape::MemoryPtr {
                target_ty: Some(memory_target_ty),
            },
            RuntimeShape::ConstRef {
                target_ty: const_target_ty,
            },
        )
        | (
            RuntimeShape::ConstRef {
                target_ty: const_target_ty,
            },
            RuntimeShape::MemoryPtr {
                target_ty: Some(memory_target_ty),
            },
        ) => match crate::repr::runtime_shape_for_ty(
            db,
            core,
            memory_target_ty,
            AddressSpaceKind::Code,
        ) {
            RuntimeShape::ConstRef {
                target_ty: memory_const_target_ty,
            } if crate::repr::runtime_ty_matches(db, memory_const_target_ty, const_target_ty) => {
                Some(RuntimeShape::ConstRef {
                    target_ty: memory_const_target_ty,
                })
            }
            _ => None,
        },
        (
            RuntimeShape::MemoryPtr {
                target_ty: Some(memory_target_ty),
            },
            RuntimeShape::AddressWord(info),
        )
        | (
            RuntimeShape::AddressWord(info),
            RuntimeShape::MemoryPtr {
                target_ty: Some(memory_target_ty),
            },
        ) if info.address_space != AddressSpaceKind::Memory => {
            let expected_info = crate::repr::runtime_pointer_info_for_ty(
                db,
                core,
                memory_target_ty,
                info.address_space,
            );
            let expected_target_ty = expected_info.and_then(|info| info.target_ty);
            let targets_match = match (expected_target_ty, info.target_ty) {
                (Some(expected_target_ty), Some(actual_target_ty)) => {
                    crate::repr::runtime_ty_matches(db, expected_target_ty, actual_target_ty)
                }
                (None, None) => true,
                _ => false,
            };
            targets_match.then_some(RuntimeShape::AddressWord(crate::ir::PointerInfo {
                address_space: info.address_space,
                target_ty: expected_target_ty.or(info.target_ty),
            }))
        }
        _ => None,
    })
}

type PointerLeafInfoSet<'db> = Vec<(MirProjectionPath<'db>, PointerInfo<'db>)>;

#[derive(Clone)]
struct ReturnRuntimeLayoutState<'db> {
    shape: RuntimeShape<'db>,
    pointer_leaf_infos: PointerLeafInfoSet<'db>,
}

impl<'db> ReturnRuntimeLayoutState<'db> {
    fn from_function(func: &MirFunction<'db>) -> Self {
        Self {
            shape: func.runtime_return_shape,
            pointer_leaf_infos: func.runtime_return_pointer_leaf_infos.clone(),
        }
    }

    fn apply(self, func: &mut MirFunction<'db>) -> bool {
        let mut changed = false;
        if func.runtime_return_shape != self.shape {
            func.runtime_return_shape = self.shape;
            changed = true;
        }
        if func.runtime_return_pointer_leaf_infos != self.pointer_leaf_infos {
            func.runtime_return_pointer_leaf_infos = self.pointer_leaf_infos;
            changed = true;
        }
        changed
    }
}

#[derive(Clone)]
struct LocalRuntimeLayoutState<'db> {
    shapes: Vec<RuntimeShape<'db>>,
    pointer_leaf_infos: Vec<PointerLeafInfoSet<'db>>,
}

impl<'db> LocalRuntimeLayoutState<'db> {
    fn from_function(func: &MirFunction<'db>) -> Self {
        Self {
            shapes: func
                .body
                .locals
                .iter()
                .map(|local| local.runtime_shape)
                .collect(),
            pointer_leaf_infos: func
                .body
                .locals
                .iter()
                .map(|local| local.pointer_leaf_infos.clone())
                .collect(),
        }
    }

    fn shape(&self, local: LocalId) -> RuntimeShape<'db> {
        self.shapes[local.index()]
    }

    fn set_shape(&mut self, local: LocalId, shape: RuntimeShape<'db>) {
        self.shapes[local.index()] = shape;
    }

    fn pointer_leaf_infos(&self, local: LocalId) -> &PointerLeafInfoSet<'db> {
        &self.pointer_leaf_infos[local.index()]
    }

    fn set_pointer_leaf_infos(&mut self, local: LocalId, infos: PointerLeafInfoSet<'db>) {
        self.pointer_leaf_infos[local.index()] = infos;
    }

    fn apply(self, func: &mut MirFunction<'db>) -> bool {
        let mut changed = false;
        for (local_idx, next_shape) in self.shapes.into_iter().enumerate() {
            if func.body.locals[local_idx].runtime_shape != next_shape {
                func.body.locals[local_idx].runtime_shape = next_shape;
                changed = true;
            }
        }
        for (local_idx, next_infos) in self.pointer_leaf_infos.into_iter().enumerate() {
            if func.body.locals[local_idx].pointer_leaf_infos != next_infos {
                func.body.locals[local_idx].pointer_leaf_infos = next_infos;
                changed = true;
            }
        }
        changed
    }
}

fn sync_local_snapshot<'db>(
    db: &'db dyn HirAnalysisDb,
    core: &CoreLib<'db>,
    current_locals: &mut [LocalData<'db>],
    local_layouts: &LocalRuntimeLayoutState<'db>,
    local: LocalId,
) {
    current_locals[local.index()].runtime_shape = local_layouts.shape(local);
    current_locals[local.index()].pointer_leaf_infos =
        local_layouts.pointer_leaf_infos(local).to_vec();
    sync_local_runtime_root_metadata(db, core, &mut current_locals[local.index()]);
}

fn locals_snapshot_from_layouts<'db>(
    db: &'db dyn HirAnalysisDb,
    core: &CoreLib<'db>,
    func: &MirFunction<'db>,
    local_layouts: &LocalRuntimeLayoutState<'db>,
) -> Vec<LocalData<'db>> {
    let mut locals = func.body.locals.clone();
    for local_idx in 0..locals.len() {
        sync_local_snapshot(
            db,
            core,
            &mut locals,
            local_layouts,
            LocalId(local_idx as u32),
        );
    }
    locals
}

fn merge_pointer_leaf_infos<'db>(
    db: &'db dyn HirAnalysisDb,
    existing: &PointerLeafInfoSet<'db>,
    next: &PointerLeafInfoSet<'db>,
) -> Option<PointerLeafInfoSet<'db>> {
    crate::capability_space::normalize_pointer_leaf_info_entries_in_context(
        db,
        existing.iter().cloned().chain(next.iter().cloned()),
    )
    .ok()
}

fn replace_pointer_leaf_infos<'db>(
    db: &'db dyn HirAnalysisDb,
    existing: &PointerLeafInfoSet<'db>,
    next: &PointerLeafInfoSet<'db>,
) -> Option<PointerLeafInfoSet<'db>> {
    let next_paths: FxHashSet<_> = next.iter().map(|(path, _)| path.clone()).collect();
    merge_pointer_leaf_infos(
        db,
        &existing
            .iter()
            .filter(|(path, _)| !next_paths.contains(path))
            .cloned()
            .collect(),
        next,
    )
}

fn retag_pointer_leaf_infos_for_local<'db>(
    db: &'db dyn HirAnalysisDb,
    core: &CoreLib<'db>,
    local: &LocalData<'db>,
    infos: &PointerLeafInfoSet<'db>,
) -> PointerLeafInfoSet<'db> {
    if !matches!(local.place_root_layout, LocalPlaceRootLayout::Direct) {
        return infos.to_vec();
    }
    infos
        .iter()
        .filter_map(|(path, info)| {
            if !path.is_empty() {
                return Some((path.clone(), *info));
            }
            crate::capability_space::pointer_leaf_infos_for_ty_with_default(
                db,
                core,
                local.ty,
                info.address_space,
            )
            .iter()
            .find_map(|(typed_path, typed_info)| (typed_path == path).then_some(*typed_info))
            .or(Some(*info))
            .map(|typed_info| (path.clone(), typed_info))
        })
        .collect()
}

fn direct_local_root_pointer_shape<'db>(
    db: &'db dyn HirAnalysisDb,
    core: &CoreLib<'db>,
    local: &LocalData<'db>,
    infos: &PointerLeafInfoSet<'db>,
) -> Option<RuntimeShape<'db>> {
    if !matches!(local.place_root_layout, LocalPlaceRootLayout::Direct) {
        return None;
    }

    let root_info = infos
        .iter()
        .find_map(|(path, info)| path.is_empty().then_some(*info))?;
    if !local.runtime_shape.is_unresolved() {
        match local.runtime_shape {
            RuntimeShape::ObjectRef { target_ty }
                if root_info.address_space == AddressSpaceKind::Memory
                    && root_info.target_ty.is_none_or(|root_target_ty| {
                        crate::repr::runtime_ty_matches(
                            db,
                            crate::repr::object_layout_ty(db, core, root_target_ty),
                            target_ty,
                        )
                    }) =>
            {
                return Some(RuntimeShape::ObjectRef { target_ty });
            }
            RuntimeShape::ObjectRef { .. } => {}
            RuntimeShape::ConstRef { target_ty }
                if root_info.address_space == AddressSpaceKind::Code
                    && root_info.target_ty.is_none_or(|root_target_ty| {
                        crate::repr::runtime_ty_matches(db, root_target_ty, target_ty)
                    }) =>
            {
                return Some(RuntimeShape::ConstRef { target_ty });
            }
            RuntimeShape::ConstRef { .. } => {}
            RuntimeShape::MemoryPtr { .. } | RuntimeShape::AddressWord(_) => {
                return Some(crate::repr::runtime_shape_for_pointer_info(root_info));
            }
            RuntimeShape::Unresolved
            | RuntimeShape::Erased
            | RuntimeShape::Word(_)
            | RuntimeShape::EnumTag { .. } => {}
        }
    }

    if root_info.address_space == AddressSpaceKind::Memory {
        return Some(crate::repr::runtime_shape_for_pointer_info(root_info));
    }

    if root_info.address_space == AddressSpaceKind::Code
        && let RuntimeShape::ConstRef { target_ty } =
            crate::repr::runtime_shape_for_ty(db, core, local.ty, AddressSpaceKind::Code)
        && root_info.target_ty.is_none_or(|root_target_ty| {
            crate::repr::runtime_ty_matches(db, root_target_ty, target_ty)
        })
    {
        return Some(RuntimeShape::ConstRef { target_ty });
    }

    Some(crate::repr::runtime_shape_for_pointer_info(root_info))
}

fn normalize_local_layout_pointer_infos<'db>(
    db: &'db dyn HirAnalysisDb,
    core: &CoreLib<'db>,
    func: &MirFunction<'db>,
    local_layouts: &mut LocalRuntimeLayoutState<'db>,
    local: LocalId,
) {
    let normalized = pointer_leaf_infos_with_runtime_root(
        db,
        core,
        func.body.local(local).ty,
        local_layouts.pointer_leaf_infos(local),
        local_layouts.shape(local),
    );
    if local_layouts.pointer_leaf_infos(local).as_slice() != normalized.as_slice() {
        local_layouts.set_pointer_leaf_infos(local, normalized);
    }
}

fn clamp_plain_word_local_layouts<'db>(
    db: &'db dyn HirAnalysisDb,
    core: &CoreLib<'db>,
    func: &MirFunction<'db>,
    local_layouts: &mut LocalRuntimeLayoutState<'db>,
) {
    for local_idx in 0..func.body.locals.len() {
        let local = LocalId(local_idx as u32);
        let clamped = normalize_plain_word_local_runtime_shape(
            db,
            core,
            func.body.local(local),
            local_layouts.shape(local),
        );
        if clamped != local_layouts.shape(local) {
            local_layouts.set_shape(local, clamped);
        }
        normalize_local_layout_pointer_infos(db, core, func, local_layouts, local);
    }
}

fn place_store_updates_local_pointer_leaf_infos<'db>(
    db: &'db dyn HirAnalysisDb,
    func: &MirFunction<'db>,
    local_layouts: &mut LocalRuntimeLayoutState<'db>,
    place: &Place<'db>,
    value_pointer_infos: &PointerLeafInfoSet<'db>,
) {
    let Some((local, base_projection)) =
        crate::ir::resolve_local_projection_root(&func.body.values, place.base)
    else {
        return;
    };
    let full_projection = base_projection.concat(&place.projection);
    if full_projection
        .iter()
        .any(|projection| matches!(projection, Projection::Deref))
    {
        return;
    }

    let preserved_infos = prune_overlapping_non_root_pointer_leaf_infos(
        local_layouts.pointer_leaf_infos(local),
        &full_projection,
    );
    let mut updated_infos = Vec::with_capacity(value_pointer_infos.len());
    for (path, info) in value_pointer_infos {
        updated_infos.push((full_projection.concat(path), *info));
    }
    if full_projection.is_empty()
        && !matches!(
            func.body.local(local).address_space,
            AddressSpaceKind::Memory
        )
    {
        updated_infos.retain(|(path, _)| !path.is_empty());
    }
    let merged = merge_pointer_leaf_infos(db, &preserved_infos, &updated_infos).unwrap_or_else(|| {
        panic!(
            "incompatible pointer leaf infos for local v{} `{}` in `{}` after store to {:?}: {:?} vs {:?}",
            local.index(),
            func.body.local(local).name,
            func.symbol_name,
            place,
            preserved_infos,
            updated_infos,
        )
    });
    local_layouts.set_pointer_leaf_infos(local, merged);
}

fn prune_overlapping_non_root_pointer_leaf_infos<'db>(
    infos: &PointerLeafInfoSet<'db>,
    projection: &MirProjectionPath<'db>,
) -> PointerLeafInfoSet<'db> {
    infos
        .iter()
        .filter(|(path, _)| {
            path.is_empty() || !(path.is_prefix_of(projection) || projection.is_prefix_of(path))
        })
        .cloned()
        .collect()
}

fn invalidate_mutating_call_arg_local_layout<'db>(
    db: &'db dyn HirAnalysisDb,
    core: &CoreLib<'db>,
    func: &MirFunction<'db>,
    local_layouts: &mut LocalRuntimeLayoutState<'db>,
    current_locals: &mut [LocalData<'db>],
    arg: ValueId,
) {
    let Some((local, projection)) =
        crate::ir::resolve_local_projection_root(&func.body.values, arg)
    else {
        return;
    };
    if projection
        .iter()
        .any(|projection| matches!(projection, Projection::Deref))
    {
        return;
    }

    let preserved_infos = prune_overlapping_non_root_pointer_leaf_infos(
        local_layouts.pointer_leaf_infos(local),
        &projection,
    );
    let (shape, normalized_infos) = normalize_mutable_local_runtime_shape(
        db,
        core,
        func.body.local(local),
        &preserved_infos,
        local_layouts.shape(local),
    );
    local_layouts.set_shape(local, shape);
    local_layouts.set_pointer_leaf_infos(local, normalized_infos);
    sync_local_snapshot(db, core, current_locals, local_layouts, local);
}

#[derive(Clone, Copy)]
struct AssignedRvalueResolver<'a, 'db> {
    db: &'db dyn HirAnalysisDb,
    core: &'a CoreLib<'db>,
    func: &'a MirFunction<'db>,
    locals: &'a [LocalData<'db>],
    return_layouts: &'a [ReturnRuntimeLayoutState<'db>],
    func_indices: &'a FxHashMap<String, usize>,
}

impl<'a, 'db> AssignedRvalueResolver<'a, 'db> {
    fn forwarded_const_call_result_shape(
        &self,
        dest: LocalId,
        call: &CallOrigin<'db>,
    ) -> Option<RuntimeShape<'db>> {
        let CallTargetRef::Hir(hir_target) = call.target.as_ref()? else {
            return None;
        };
        let CallableDef::Func(hir_func) = hir_target.callable_def else {
            return None;
        };
        let (diags, typed_body) = check_func_body(self.db, hir_func);
        if !diags.is_empty() {
            return None;
        }
        let ReturnProvenance::ForwardedParams(indices) = typed_body.return_provenance(self.db)
        else {
            return None;
        };
        if indices.iter().copied().any(|idx| {
            hir_func.params(self.db).nth(idx).is_some_and(|param| {
                param.is_mut(self.db)
                    || crate::repr::ty_has_mut_capability(self.db, param.ty(self.db))
            })
        }) {
            return None;
        }

        let mut arg_target_ty = None;
        for idx in indices {
            let arg = *call.args.get(idx)?;
            let RuntimeShape::ConstRef { target_ty } = crate::repr::runtime_shape_for_value(
                self.db,
                self.core,
                &self.func.body.values,
                self.locals,
                arg,
            )?
            else {
                return None;
            };
            if let Some(existing) = arg_target_ty {
                if !crate::repr::runtime_ty_matches(self.db, existing, target_ty) {
                    return None;
                }
            } else {
                arg_target_ty = Some(target_ty);
            }
        }
        let arg_target_ty = arg_target_ty?;

        match crate::repr::runtime_shape_for_ty(
            self.db,
            self.core,
            self.func.body.local(dest).ty,
            AddressSpaceKind::Code,
        ) {
            RuntimeShape::ConstRef { target_ty }
                if crate::repr::runtime_ty_matches(self.db, target_ty, arg_target_ty) =>
            {
                Some(RuntimeShape::ConstRef { target_ty })
            }
            _ => Some(RuntimeShape::ConstRef {
                target_ty: arg_target_ty,
            }),
        }
    }

    fn runtime_shape(&self, dest: LocalId, rvalue: &Rvalue<'db>) -> Option<RuntimeShape<'db>> {
        match rvalue {
            Rvalue::Value(value) => crate::repr::runtime_shape_for_value(
                self.db,
                self.core,
                &self.func.body.values,
                self.locals,
                *value,
            ),
            Rvalue::Call(call) => self
                .forwarded_const_call_result_shape(dest, call)
                .or_else(|| {
                    call.resolved_name.as_deref().and_then(|name| {
                        self.func_indices
                            .get(name)
                            .copied()
                            .map(|callee_idx| self.return_layouts[callee_idx].shape)
                    })
                })
                .or_else(|| {
                    self.locals
                        .get(dest.index())
                        .map(|local| local.runtime_shape)
                }),
            Rvalue::Alloc {
                address_space: AddressSpaceKind::Memory,
            } => self.locals.get(dest.index()).map(|local| {
                crate::repr::place_root_runtime_shape_for_local(local).unwrap_or_else(|| {
                    crate::repr::runtime_shape_for_local(self.db, self.core, local)
                })
            }),
            Rvalue::ZeroInit
            | Rvalue::Intrinsic { .. }
            | Rvalue::Alloc { .. }
            | Rvalue::ConstAggregate { .. } => self
                .locals
                .get(dest.index())
                .map(|local| local.runtime_shape),
            Rvalue::Load { place } => crate::repr::runtime_shape_for_loaded_place(
                self.db,
                self.core,
                &self.func.body.values,
                self.locals,
                place,
            )
            .or_else(|| {
                self.locals
                    .get(dest.index())
                    .map(|local| local.runtime_shape)
            }),
        }
    }

    fn pointer_leaf_infos(&self, dest: LocalId, rvalue: &Rvalue<'db>) -> PointerLeafInfoSet<'db> {
        match rvalue {
            Rvalue::Value(value) => crate::repr::pointer_leaf_infos_for_value(
                self.db,
                self.core,
                &self.func.body.values,
                self.locals,
                *value,
            ),
            Rvalue::Load { place } => crate::repr::pointer_leaf_infos_for_place(
                self.db,
                self.core,
                &self.func.body.values,
                self.locals,
                place,
                self.func.body.local(dest).ty,
            ),
            Rvalue::Call(call) => {
                let inherited_infos = call
                    .resolved_name
                    .as_deref()
                    .and_then(|name| self.func_indices.get(name).copied())
                    .map(|callee_idx| self.return_layouts[callee_idx].pointer_leaf_infos.clone())
                    .unwrap_or_default();
                let forwarded_infos = crate::lower::call_return_pointer_leaf_infos(
                    self.db,
                    self.core,
                    &self.func.body.values,
                    self.locals,
                    call,
                    self.func.body.local(dest).ty,
                );
                if forwarded_infos.is_empty() {
                    inherited_infos
                } else {
                    replace_pointer_leaf_infos(self.db, &inherited_infos, &forwarded_infos)
                        .unwrap_or_else(|| {
                            panic!(
                                "incompatible call result pointer leaf infos for local v{} `{}` in `{}` from rvalue {:?}: {:?} vs {:?}",
                                dest.index(),
                                self.func.body.local(dest).name,
                                self.func.symbol_name,
                                rvalue,
                                inherited_infos,
                                forwarded_infos,
                            )
                        })
                }
            }
            Rvalue::Intrinsic { .. }
            | Rvalue::Alloc { .. }
            | Rvalue::ZeroInit
            | Rvalue::ConstAggregate { .. } => Vec::new(),
        }
    }
}

fn refine_call_param_local_shapes<'db>(
    db: &'db dyn HirAnalysisDb,
    module: &MirModule<'db>,
    caller_idx: usize,
    call: &CallOrigin<'db>,
    next_local_layouts: &mut [LocalRuntimeLayoutState<'db>],
    func_indices: &FxHashMap<String, usize>,
) {
    let Some(callee_name) = call.resolved_name.as_deref() else {
        return;
    };
    let Some(&callee_idx) = func_indices.get(callee_name) else {
        return;
    };
    let caller = &module.functions[caller_idx];
    let caller_core = function_core_lib(db, caller);
    let callee = &module.functions[callee_idx];
    let core = function_core_lib(db, callee);
    let caller_locals =
        locals_snapshot_from_layouts(db, &caller_core, caller, &next_local_layouts[caller_idx]);

    let runtime_value_params = callee.runtime_param_locals();
    assert_eq!(
        runtime_value_params.len(),
        call.args.len(),
        "runtime value arg shape mismatch for `{callee_name}`: params={}, args={}",
        runtime_value_params.len(),
        call.args.len(),
    );
    for (local, arg) in runtime_value_params.into_iter().zip(&call.args) {
        let arg_shape = crate::repr::runtime_shape_for_value(
            db,
            &caller_core,
            &caller.body.values,
            &caller_locals,
            *arg,
        )
        .unwrap_or(RuntimeShape::Unresolved);
        let local_shape = next_local_layouts[callee_idx].shape(local);
        let Some(merged) = specialize_call_param_runtime_shape(
            db,
            &core,
            callee.body.local(local),
            local_shape,
            arg_shape,
        ) else {
            continue;
        };
        let (merged, normalized_infos) = normalize_mutable_local_runtime_shape(
            db,
            &core,
            callee.body.local(local),
            next_local_layouts[callee_idx].pointer_leaf_infos(local),
            merged,
        );
        let merged =
            normalize_plain_word_local_runtime_shape(db, &core, callee.body.local(local), merged);
        next_local_layouts[callee_idx].set_shape(local, merged);
        next_local_layouts[callee_idx].set_pointer_leaf_infos(local, normalized_infos);
        let arg_pointer_infos = crate::repr::pointer_leaf_infos_for_value(
            db,
            &caller_core,
            &caller.body.values,
            &caller_locals,
            *arg,
        );
        if !arg_pointer_infos.is_empty() {
            next_local_layouts[callee_idx].set_pointer_leaf_infos(
                local,
                replace_pointer_leaf_infos(
                    db,
                    next_local_layouts[callee_idx].pointer_leaf_infos(local),
                    &arg_pointer_infos,
                )
                .unwrap_or_else(|| {
                    panic!(
                        "incompatible pointer leaf infos for call param local v{} `{}` in `{}` from caller `{}` arg v{}: {:?} vs {:?}",
                        local.index(),
                        callee.body.local(local).name,
                        callee.symbol_name,
                        caller.symbol_name,
                        arg.index(),
                        next_local_layouts[callee_idx].pointer_leaf_infos(local),
                        arg_pointer_infos,
                    )
                }),
            );
        }
    }

    let runtime_effect_params = callee.runtime_effect_param_locals();
    assert_eq!(
        runtime_effect_params.len(),
        call.effect_args.len(),
        "runtime effect arg shape mismatch for `{callee_name}`: params={}, args={}",
        runtime_effect_params.len(),
        call.effect_args.len(),
    );
    for (local, arg) in runtime_effect_params.into_iter().zip(&call.effect_args) {
        let arg_pointer_infos = retag_pointer_leaf_infos_for_local(
            db,
            &core,
            callee.body.local(local),
            &crate::repr::pointer_leaf_infos_for_value(
                db,
                &caller_core,
                &caller.body.values,
                &caller_locals,
                *arg,
            ),
        );
        let arg_shape = direct_local_root_pointer_shape(
            db,
            &core,
            callee.body.local(local),
            &arg_pointer_infos,
        )
        .or_else(|| {
            crate::repr::runtime_shape_for_value(
                db,
                &caller_core,
                &caller.body.values,
                &caller_locals,
                *arg,
            )
        })
        .unwrap_or(RuntimeShape::Unresolved);
        let local_shape = next_local_layouts[callee_idx].shape(local);
        let Some(merged) = specialize_call_param_runtime_shape(
            db,
            &core,
            callee.body.local(local),
            local_shape,
            arg_shape,
        ) else {
            continue;
        };
        let (merged, normalized_infos) = normalize_mutable_local_runtime_shape(
            db,
            &core,
            callee.body.local(local),
            next_local_layouts[callee_idx].pointer_leaf_infos(local),
            merged,
        );
        let merged =
            normalize_plain_word_local_runtime_shape(db, &core, callee.body.local(local), merged);
        next_local_layouts[callee_idx].set_shape(local, merged);
        next_local_layouts[callee_idx].set_pointer_leaf_infos(local, normalized_infos);
        if !arg_pointer_infos.is_empty() {
            next_local_layouts[callee_idx].set_pointer_leaf_infos(
                local,
                replace_pointer_leaf_infos(
                    db,
                    next_local_layouts[callee_idx].pointer_leaf_infos(local),
                    &arg_pointer_infos,
                )
                .unwrap_or_else(|| {
                    panic!(
                        "incompatible pointer leaf infos for effect param local v{} `{}` in `{}` from caller `{}` arg v{}: {:?} vs {:?}",
                        local.index(),
                        callee.body.local(local).name,
                        callee.symbol_name,
                        caller.symbol_name,
                        arg.index(),
                        next_local_layouts[callee_idx].pointer_leaf_infos(local),
                        arg_pointer_infos,
                    )
                }),
            );
        }
    }
}

fn specialize_call_param_runtime_shape<'db>(
    db: &'db dyn HirAnalysisDb,
    core: &CoreLib<'db>,
    local: &LocalData<'db>,
    local_shape: RuntimeShape<'db>,
    arg_shape: RuntimeShape<'db>,
) -> Option<RuntimeShape<'db>> {
    let local_ty = local.ty;
    let local_memory_shape =
        crate::repr::runtime_shape_for_ty(db, core, local_ty, AddressSpaceKind::Memory);
    let local_code_shape =
        crate::repr::runtime_shape_for_ty(db, core, local_ty, AddressSpaceKind::Code);
    let local_const_target_ty = match local_code_shape {
        RuntimeShape::ConstRef { target_ty } => Some(target_ty),
        _ => None,
    };
    let local_shape = match (local_shape, local_memory_shape, local_const_target_ty) {
        (
            RuntimeShape::ObjectRef {
                target_ty: shape_target_ty,
            },
            RuntimeShape::MemoryPtr {
                target_ty: memory_target_ty,
            },
            Some(const_target_ty),
        ) if memory_target_ty.is_some()
            && crate::repr::runtime_ty_matches(db, const_target_ty, shape_target_ty) =>
        {
            RuntimeShape::MemoryPtr {
                target_ty: memory_target_ty,
            }
        }
        (local_shape, _, _) => local_shape,
    };

    if let RuntimeShape::ObjectRef { target_ty } = arg_shape
        && matches!(local.address_space, AddressSpaceKind::Memory)
        && crate::repr::effect_provider_space_for_ty(db, core, local_ty).is_none()
        && !matches!(local_shape, RuntimeShape::AddressWord(info) if info.address_space != AddressSpaceKind::Memory)
    {
        return Some(RuntimeShape::ObjectRef { target_ty });
    }

    if let RuntimeShape::MemoryPtr {
        target_ty: arg_target_ty,
    } = arg_shape
        && matches!(local.address_space, AddressSpaceKind::Memory)
        && !matches!(local_shape, RuntimeShape::Erased)
    {
        let target_ty =
            crate::repr::runtime_pointer_info_for_ty(db, core, local_ty, AddressSpaceKind::Memory)
                .and_then(|info| info.target_ty)
                .or(arg_target_ty);
        if arg_target_ty.is_none_or(|arg_target_ty| {
            target_ty.is_none_or(|target_ty| {
                crate::repr::runtime_ty_matches(db, target_ty, arg_target_ty)
            })
        }) {
            return Some(RuntimeShape::MemoryPtr { target_ty });
        }
    }

    if let RuntimeShape::ConstRef {
        target_ty: arg_target_ty,
    } = arg_shape
        && let Some(local_target_ty) = local_const_target_ty
        && crate::repr::runtime_ty_matches(db, local_target_ty, arg_target_ty)
    {
        return Some(RuntimeShape::ConstRef {
            target_ty: local_target_ty,
        });
    }

    if let RuntimeShape::ConstRef {
        target_ty: arg_target_ty,
    } = arg_shape
        && (crate::repr::runtime_pointer_info_for_ty(db, core, local_ty, AddressSpaceKind::Code)
            .and_then(|info| info.target_ty)
            .is_some_and(|local_target_ty| {
                crate::repr::runtime_ty_matches(db, local_target_ty, arg_target_ty)
            })
            || crate::repr::runtime_ty_matches(
                db,
                crate::repr::object_layout_ty(db, core, local_ty),
                arg_target_ty,
            ))
    {
        return Some(RuntimeShape::ConstRef {
            target_ty: arg_target_ty,
        });
    }

    if let RuntimeShape::ObjectRef {
        target_ty: arg_target_ty,
    } = arg_shape
        && let RuntimeShape::ObjectRef {
            target_ty: local_target_ty,
        } = local_memory_shape
        && crate::repr::runtime_ty_matches(db, local_target_ty, arg_target_ty)
    {
        return merge_runtime_shapes_in_context(
            db,
            core,
            local_shape,
            RuntimeShape::ObjectRef {
                target_ty: local_target_ty,
            },
        );
    }

    if let RuntimeShape::ObjectRef {
        target_ty: arg_target_ty,
    } = arg_shape
        && let RuntimeShape::MemoryPtr { target_ty } = local_memory_shape
        && let Some(local_target_ty) = local_const_target_ty
        && target_ty.is_some()
        && crate::repr::runtime_ty_matches(db, local_target_ty, arg_target_ty)
    {
        return merge_runtime_shapes_in_context(
            db,
            core,
            local_shape,
            RuntimeShape::MemoryPtr { target_ty },
        );
    }

    if let RuntimeShape::MemoryPtr {
        target_ty: arg_target_ty,
    } = arg_shape
        && !matches!(local_shape, RuntimeShape::MemoryPtr { .. })
        && let RuntimeShape::ObjectRef {
            target_ty: local_target_ty,
        } = local_memory_shape
        && arg_target_ty.is_none_or(|arg_target_ty| {
            crate::repr::runtime_ty_matches(
                db,
                local_target_ty,
                crate::repr::object_layout_ty(db, core, arg_target_ty),
            )
        })
    {
        return merge_runtime_shapes_in_context(
            db,
            core,
            local_shape,
            RuntimeShape::ObjectRef {
                target_ty: local_target_ty,
            },
        );
    }

    if let RuntimeShape::MemoryPtr {
        target_ty: arg_target_ty,
    } = arg_shape
        && let RuntimeShape::MemoryPtr { target_ty } = local_memory_shape
        && let Some(local_target_ty) = local_const_target_ty
        && arg_target_ty.is_none_or(|arg_target_ty| {
            crate::repr::runtime_ty_matches(db, local_target_ty, arg_target_ty)
        })
    {
        return merge_runtime_shapes_in_context(
            db,
            core,
            local_shape,
            RuntimeShape::MemoryPtr { target_ty },
        );
    }

    merge_runtime_shapes_in_context(db, core, local_shape, arg_shape)
}

fn assert_normalized_runtime_shapes<'db>(db: &'db dyn HirAnalysisDb, func: &MirFunction<'db>) {
    let live_values = compute_live_values(&func.body);

    for (idx, local) in func.body.locals.iter().enumerate() {
        if local.runtime_shape.is_unresolved() {
            panic!(
                "unresolved local runtime shape after MIR normalization in `{}` for local v{idx} `{}`: ty={}, address_space={:?}, pointer_leaf_infos={:?}",
                func.symbol_name,
                local.name,
                local.ty.pretty_print(db),
                local.address_space,
                local.pointer_leaf_infos,
            );
        }
    }

    for (idx, value) in func.body.values.iter().enumerate() {
        if !live_values.get(idx).copied().unwrap_or(false) {
            continue;
        }
        if value.runtime_shape.is_unresolved() {
            panic!(
                "unresolved value runtime shape after MIR normalization in `{}` for v{idx}: origin={:?}, ty={}, repr={:?}, pointer_info={:?}",
                func.symbol_name,
                value.origin,
                value.ty.pretty_print(db),
                value.repr,
                value.pointer_info,
            );
        }
        let Some((local, expected_root_shape)) = (match value.origin {
            ValueOrigin::Local(local) => func
                .body
                .locals
                .get(local.index())
                .map(|local_data| (local, local_data.runtime_shape)),
            ValueOrigin::PlaceRoot(local) => {
                func.body.locals.get(local.index()).map(|local_data| {
                    (
                        local,
                        crate::repr::place_root_runtime_shape_for_local(local_data)
                            .unwrap_or(local_data.runtime_shape),
                    )
                })
            }
            _ => None,
        }) else {
            continue;
        };
        if expected_root_shape != value.runtime_shape {
            panic!(
                "mismatched root local runtime shapes after MIR normalization in `{}` for v{idx}: local v{} `{}` has {:?}, value has {:?}, origin={:?}, ty={}, repr={:?}",
                func.symbol_name,
                local.index(),
                func.body.local(local).name,
                expected_root_shape,
                value.runtime_shape,
                value.origin,
                value.ty.pretty_print(db),
                value.repr,
            );
        }
    }
}

fn seed_function_local_runtime_shapes<'db>(
    db: &'db dyn HirAnalysisDb,
    func: &mut MirFunction<'db>,
) -> Vec<Option<RuntimeShape<'db>>> {
    let core = function_core_lib(db, func);
    let live_values = compute_live_values(&func.body);
    let spill_locals: FxHashSet<_> = func.body.spill_slots.values().copied().collect();
    let mut local_fallbacks = vec![None; func.body.locals.len()];

    for (idx, local) in func.body.locals.iter_mut().enumerate() {
        local.runtime_shape = crate::repr::runtime_shape_for_local(db, &core, local);
        if spill_locals.contains(&LocalId(idx as u32))
            && let Some(place_root_shape) = crate::repr::place_root_runtime_shape_for_local(local)
        {
            local.runtime_shape = place_root_shape;
        }
    }

    for (local_id, local_fallback) in local_fallbacks.iter_mut().enumerate() {
        let Some(fallback) = crate::repr::deferred_const_ref_runtime_shape_fallback(
            db,
            &core,
            &func.body.locals[local_id],
        ) else {
            continue;
        };
        *local_fallback = Some(fallback);
        func.body.locals[local_id].runtime_shape = RuntimeShape::Unresolved;
    }

    for (idx, local_id) in func.body.param_locals.iter().copied().enumerate() {
        if !func.runtime_abi.value_param_visible(idx)
            && !live_values.iter().enumerate().any(|(value_idx, live)| {
                *live
                    && matches!(
                        func.body.values[value_idx].origin,
                        ValueOrigin::Local(local) | ValueOrigin::PlaceRoot(local) if local == local_id
                    )
            })
        {
            func.body.locals[local_id.index()].runtime_shape = RuntimeShape::Erased;
        }
    }

    for (idx, local_id) in func.body.effect_param_locals.iter().copied().enumerate() {
        if !func.runtime_abi.effect_param_visible(idx)
            && !live_values.iter().enumerate().any(|(value_idx, live)| {
                *live
                    && matches!(
                        func.body.values[value_idx].origin,
                        ValueOrigin::Local(local) | ValueOrigin::PlaceRoot(local) if local == local_id
                    )
            })
        {
            func.body.locals[local_id.index()].runtime_shape = RuntimeShape::Erased;
        }
    }

    local_fallbacks
}

fn deferred_runtime_return_shape_fallback<'db>(
    db: &'db dyn HirAnalysisDb,
    core: &CoreLib<'db>,
    ret_ty: TyId<'db>,
) -> Option<RuntimeShape<'db>> {
    let concrete_shape = crate::repr::runtime_return_shape_seed_for_ty(db, core, ret_ty);
    (matches!(concrete_shape, RuntimeShape::ObjectRef { .. })
        && crate::repr::supports_const_ref_runtime_ty(db, core, ret_ty))
    .then_some(concrete_shape)
}

fn materialize_local_runtime_layout<'db>(
    db: &'db dyn HirAnalysisDb,
    core: &CoreLib<'db>,
    local: &mut LocalData<'db>,
) {
    local.const_backing = crate::ir::LocalConstBacking::Runtime;
    if local.address_space != AddressSpaceKind::Memory {
        crate::repr::set_declared_local_address_space(db, core, local, AddressSpaceKind::Memory);
    }
    local.pointer_leaf_infos = crate::capability_space::pointer_leaf_infos_for_ty_with_default(
        db,
        core,
        local.ty,
        AddressSpaceKind::Memory,
    );
    local.runtime_shape = crate::repr::runtime_shape_for_local(db, core, local);
}

fn set_local_root_pointer_info<'db>(
    local: &mut LocalData<'db>,
    root_info: Option<crate::ir::PointerInfo<'db>>,
) {
    local
        .pointer_leaf_infos
        .retain(|(path, _)| !path.is_empty());
    if let Some(root_info) = root_info {
        local
            .pointer_leaf_infos
            .insert(0, (crate::MirProjectionPath::new(), root_info));
    }
}

fn pointer_leaf_infos_without_root<'db>(
    infos: &PointerLeafInfoSet<'db>,
) -> PointerLeafInfoSet<'db> {
    infos
        .iter()
        .filter(|(path, _)| !path.is_empty())
        .cloned()
        .collect()
}

fn sync_local_runtime_root_metadata<'db>(
    db: &'db dyn HirAnalysisDb,
    core: &CoreLib<'db>,
    local: &mut LocalData<'db>,
) {
    match local.runtime_shape {
        RuntimeShape::ConstRef { target_ty } => {
            local.const_backing = crate::ir::LocalConstBacking::Const;
            if local.address_space != AddressSpaceKind::Code {
                crate::repr::set_declared_local_address_space(
                    db,
                    core,
                    local,
                    AddressSpaceKind::Code,
                );
            }
            local.pointer_leaf_infos = pointer_leaf_infos_with_runtime_root(
                db,
                core,
                local.ty,
                &local.pointer_leaf_infos,
                local.runtime_shape,
            );
            set_local_root_pointer_info(
                local,
                Some(crate::ir::PointerInfo {
                    address_space: AddressSpaceKind::Code,
                    target_ty: Some(target_ty),
                }),
            );
        }
        RuntimeShape::ObjectRef { target_ty } => {
            local.const_backing = crate::ir::LocalConstBacking::Runtime;
            if local.address_space != AddressSpaceKind::Memory {
                crate::repr::set_declared_local_address_space(
                    db,
                    core,
                    local,
                    AddressSpaceKind::Memory,
                );
            }
            if !local.place_root_layout.is_object_root() {
                local.place_root_layout =
                    crate::repr::object_ref_place_root_layout_for_ty(db, core, local.ty, target_ty);
            }
            local.pointer_leaf_infos = pointer_leaf_infos_with_runtime_root(
                db,
                core,
                local.ty,
                &local.pointer_leaf_infos,
                local.runtime_shape,
            );
            set_local_root_pointer_info(
                local,
                Some(crate::ir::PointerInfo {
                    address_space: AddressSpaceKind::Memory,
                    target_ty: Some(target_ty),
                }),
            );
        }
        RuntimeShape::MemoryPtr { target_ty } => {
            local.const_backing = crate::ir::LocalConstBacking::Runtime;
            if local.address_space != AddressSpaceKind::Memory {
                crate::repr::set_declared_local_address_space(
                    db,
                    core,
                    local,
                    AddressSpaceKind::Memory,
                );
            }
            if local.place_root_layout.is_object_root() {
                local.place_root_layout = LocalPlaceRootLayout::MemorySlot;
            }
            local.pointer_leaf_infos = pointer_leaf_infos_with_runtime_root(
                db,
                core,
                local.ty,
                &local.pointer_leaf_infos,
                local.runtime_shape,
            );
            set_local_root_pointer_info(
                local,
                Some(crate::ir::PointerInfo {
                    address_space: AddressSpaceKind::Memory,
                    target_ty,
                }),
            );
        }
        RuntimeShape::AddressWord(info) => {
            local.const_backing = crate::ir::LocalConstBacking::Runtime;
            if local.address_space != info.address_space {
                crate::repr::set_declared_local_address_space(db, core, local, info.address_space);
            }
            local.pointer_leaf_infos = pointer_leaf_infos_with_runtime_root(
                db,
                core,
                local.ty,
                &local.pointer_leaf_infos,
                local.runtime_shape,
            );
            set_local_root_pointer_info(local, Some(info));
        }
        RuntimeShape::Word(_) | RuntimeShape::EnumTag { .. } | RuntimeShape::Erased => {
            set_local_root_pointer_info(local, None);
        }
        RuntimeShape::Unresolved => {}
    }
}

fn normalize_mutable_local_runtime_shape<'db>(
    db: &'db dyn HirAnalysisDb,
    core: &CoreLib<'db>,
    local: &LocalData<'db>,
    current_pointer_leaf_infos: &PointerLeafInfoSet<'db>,
    proposed_shape: RuntimeShape<'db>,
) -> (RuntimeShape<'db>, PointerLeafInfoSet<'db>) {
    if !crate::repr::local_is_semantically_mutable(db, local)
        || !matches!(proposed_shape, RuntimeShape::ConstRef { .. })
    {
        return (proposed_shape, current_pointer_leaf_infos.to_vec());
    }

    let mut materialized = local.clone();
    materialized.pointer_leaf_infos = current_pointer_leaf_infos.to_vec();
    materialized.runtime_shape = proposed_shape;
    materialize_local_runtime_layout(db, core, &mut materialized);
    (materialized.runtime_shape, materialized.pointer_leaf_infos)
}

fn normalize_plain_word_local_runtime_shape<'db>(
    db: &'db dyn HirAnalysisDb,
    core: &CoreLib<'db>,
    local: &LocalData<'db>,
    proposed_shape: RuntimeShape<'db>,
) -> RuntimeShape<'db> {
    crate::repr::normalize_plain_word_runtime_shape_for_ty(db, core, local.ty, proposed_shape)
}

fn pointer_leaf_infos_with_runtime_root<'db>(
    db: &'db dyn HirAnalysisDb,
    core: &CoreLib<'db>,
    owner_ty: TyId<'db>,
    infos: &PointerLeafInfoSet<'db>,
    runtime_shape: RuntimeShape<'db>,
) -> PointerLeafInfoSet<'db> {
    let (root_info, rebase_code_leaves_to_memory) = match runtime_shape {
        RuntimeShape::ConstRef { target_ty } => (
            Some(PointerInfo {
                address_space: AddressSpaceKind::Code,
                target_ty: Some(target_ty),
            }),
            false,
        ),
        RuntimeShape::ObjectRef { target_ty } => (
            Some(PointerInfo {
                address_space: AddressSpaceKind::Memory,
                target_ty: Some(target_ty),
            }),
            true,
        ),
        RuntimeShape::MemoryPtr { target_ty } => (
            Some(PointerInfo {
                address_space: AddressSpaceKind::Memory,
                target_ty,
            }),
            true,
        ),
        RuntimeShape::AddressWord(info) => (Some(info), false),
        RuntimeShape::Word(_) | RuntimeShape::EnumTag { .. } | RuntimeShape::Erased => {
            return infos
                .iter()
                .filter(|(path, _)| !path.is_empty())
                .cloned()
                .collect();
        }
        RuntimeShape::Unresolved => return infos.to_vec(),
    };

    let code_default_infos = crate::capability_space::pointer_leaf_infos_for_ty_with_default(
        db,
        core,
        owner_ty,
        AddressSpaceKind::Code,
    );

    let mut normalized_infos = infos
        .iter()
        .filter(|(path, _)| !path.is_empty())
        .map(|(path, info)| {
            let keeps_non_memory_leaf = code_default_infos
                .iter()
                .find_map(|(default_path, default_info)| {
                    (default_path == path).then_some(default_info.address_space)
                })
                .is_some_and(|space| space != AddressSpaceKind::Memory);
            let info = if rebase_code_leaves_to_memory
                && matches!(info.address_space, AddressSpaceKind::Code)
                && !keeps_non_memory_leaf
            {
                PointerInfo {
                    address_space: AddressSpaceKind::Memory,
                    target_ty: info.target_ty,
                }
            } else {
                *info
            };
            (path.clone(), info)
        })
        .collect::<Vec<_>>();
    if let Some(root_info) = root_info {
        normalized_infos.insert(0, (crate::MirProjectionPath::new(), root_info));
    }
    crate::repr::normalize_pointer_leaf_infos(normalized_infos)
}

struct RuntimeLayoutFixpoint<'db, 'm> {
    db: &'db dyn HirAnalysisDb,
    module: &'m mut MirModule<'db>,
    func_indices: FxHashMap<String, usize>,
    return_layout_seeds: Vec<ReturnRuntimeLayoutState<'db>>,
    deferred_return_shape_fallbacks: Vec<Option<RuntimeShape<'db>>>,
    deferred_local_fallbacks: Vec<Vec<Option<RuntimeShape<'db>>>>,
}

impl<'db, 'm> RuntimeLayoutFixpoint<'db, 'm> {
    fn new(db: &'db dyn HirAnalysisDb, module: &'m mut MirModule<'db>) -> Self {
        let func_indices = module
            .functions
            .iter()
            .enumerate()
            .map(|(idx, func)| (func.symbol_name.clone(), idx))
            .collect();
        let mut deferred_return_shape_fallbacks = Vec::with_capacity(module.functions.len());
        let return_layout_seeds = module
            .functions
            .iter()
            .map(|func| {
                let core = function_core_lib(db, func);
                let fallback = deferred_runtime_return_shape_fallback(db, &core, func.ret_ty);
                let shape_seed = runtime_return_shape_seed_for_func(db, func, &core);
                deferred_return_shape_fallbacks.push(fallback);
                ReturnRuntimeLayoutState {
                    shape: if fallback.is_some() && !matches!(shape_seed, RuntimeShape::Unresolved)
                    {
                        RuntimeShape::Unresolved
                    } else {
                        shape_seed
                    },
                    pointer_leaf_infos:
                        crate::capability_space::pointer_leaf_infos_for_ty_with_default(
                            db,
                            &core,
                            func.ret_ty,
                            AddressSpaceKind::Memory,
                        ),
                }
            })
            .collect();

        Self {
            db,
            module,
            func_indices,
            return_layout_seeds,
            deferred_return_shape_fallbacks,
            deferred_local_fallbacks: Vec::new(),
        }
    }

    fn run(&mut self) {
        self.deferred_local_fallbacks.clear();
        for func in &mut self.module.functions {
            self.deferred_local_fallbacks
                .push(seed_function_local_runtime_shapes(self.db, func));
        }

        while self.iterate() {}

        for func in &mut self.module.functions {
            let core = function_core_lib(self.db, func);
            for local in &mut func.body.locals {
                if crate::repr::local_is_semantically_mutable(self.db, local)
                    && matches!(local.runtime_shape, RuntimeShape::ConstRef { .. })
                {
                    materialize_local_runtime_layout(self.db, &core, local);
                    continue;
                }
                sync_local_runtime_root_metadata(self.db, &core, local);
            }
        }

        self.refresh_value_pointer_infos();
        self.recompute_value_runtime_shapes();

        for func in &self.module.functions {
            assert_normalized_runtime_shapes(self.db, func);
        }
    }

    fn iterate(&mut self) -> bool {
        self.refresh_value_pointer_infos();
        self.recompute_value_runtime_shapes();
        let return_layouts = self.current_return_layouts();
        let next_local_layouts = self.recompute_local_layouts(&return_layouts);

        let mut changed = self.apply_local_layouts(next_local_layouts)
            | self.apply_deferred_const_ref_fallbacks();

        if changed {
            self.refresh_value_pointer_infos();
            self.recompute_value_runtime_shapes();
        }

        changed |= self.apply_return_layouts(self.recompute_return_layouts());
        changed
    }

    fn current_return_layouts(&self) -> Vec<ReturnRuntimeLayoutState<'db>> {
        self.module
            .functions
            .iter()
            .enumerate()
            .map(|(func_idx, func)| {
                let fallback = self.deferred_return_shape_fallbacks[func_idx];
                if fallback.is_some_and(|fallback| func.runtime_return_shape == fallback) {
                    ReturnRuntimeLayoutState {
                        shape: RuntimeShape::Unresolved,
                        pointer_leaf_infos: pointer_leaf_infos_without_root(
                            &func.runtime_return_pointer_leaf_infos,
                        ),
                    }
                } else {
                    ReturnRuntimeLayoutState::from_function(func)
                }
            })
            .collect()
    }

    fn recompute_value_runtime_shapes(&mut self) {
        for func_idx in 0..self.module.functions.len() {
            let core = function_core_lib(self.db, &self.module.functions[func_idx]);
            let value_shapes: Vec<_> = {
                let func = &self.module.functions[func_idx];
                (0..func.body.values.len())
                    .map(|idx| {
                        let shape = crate::repr::inferred_runtime_shape_for_value(
                            self.db,
                            &core,
                            &func.body.values,
                            &func.body.locals,
                            ValueId(idx as u32),
                        )
                        .unwrap_or(RuntimeShape::Unresolved);
                        crate::repr::normalize_plain_word_runtime_shape_for_ty(
                            self.db,
                            &core,
                            func.body.values[idx].ty,
                            shape,
                        )
                    })
                    .collect()
            };

            for (idx, shape) in value_shapes.into_iter().enumerate() {
                self.module.functions[func_idx].body.values[idx].runtime_shape = shape;
            }
        }
    }

    fn refresh_value_pointer_infos(&mut self) {
        for func_idx in 0..self.module.functions.len() {
            let core = function_core_lib(self.db, &self.module.functions[func_idx]);
            let pointer_infos: Vec<_> = {
                let func = &self.module.functions[func_idx];
                (0..func.body.values.len())
                    .map(|idx| {
                        crate::repr::infer_value_pointer_info(
                            self.db,
                            &core,
                            &func.body.values,
                            &func.body.locals,
                            ValueId(idx as u32),
                        )
                    })
                    .collect()
            };

            for (idx, pointer_info) in pointer_infos.into_iter().enumerate() {
                self.module.functions[func_idx].body.values[idx].pointer_info = pointer_info;
            }
        }
    }

    fn recompute_return_layouts(&self) -> Vec<ReturnRuntimeLayoutState<'db>> {
        (0..self.module.functions.len())
            .map(|func_idx| {
                let func = &self.module.functions[func_idx];
                let core = function_core_lib(self.db, func);
                let mut return_layout = self.return_layout_seeds[func_idx].clone();

                for block in &func.body.blocks {
                    let Terminator::Return {
                        value: Some(value), ..
                    } = &block.terminator
                    else {
                        continue;
                    };
                    let Some(returned) = func
                        .body
                        .values
                        .get(value.index())
                        .map(|value| value.runtime_shape)
                    else {
                        continue;
                    };
                    return_layout.shape = merge_runtime_shapes_in_context(
                        self.db,
                        &core,
                        return_layout.shape,
                        returned,
                    )
                    .unwrap_or_else(|| {
                        let returned_value = func
                            .body
                            .values
                            .get(value.index())
                            .expect("checked above");
                        let returned_local = match returned_value.origin {
                            ValueOrigin::Local(local) => func.body.locals.get(local.index()),
                            _ => None,
                        };
                        panic!(
                            "incompatible runtime return shapes in `{}`: {:?} vs {:?}; return_value=v{} origin={:?} ty={} repr={:?} pointer_info={:?} returned_local={:?}",
                            func.symbol_name,
                            return_layout.shape,
                            returned,
                            value.index(),
                            returned_value.origin,
                            returned_value.ty.pretty_print(self.db),
                            returned_value.repr,
                            returned_value.pointer_info,
                            returned_local,
                        )
                    });
                    let returned_pointer_infos = crate::repr::pointer_leaf_infos_for_value(
                        self.db,
                        &core,
                        &func.body.values,
                        &func.body.locals,
                        *value,
                    );
                    return_layout.pointer_leaf_infos = replace_pointer_leaf_infos(
                        self.db,
                        &return_layout.pointer_leaf_infos,
                        &returned_pointer_infos,
                    )
                    .unwrap_or_else(|| {
                        panic!(
                            "incompatible runtime return pointer leaf infos in `{}`: {:?} vs {:?}",
                            func.symbol_name,
                            return_layout.pointer_leaf_infos,
                            returned_pointer_infos
                        )
                    });
                }

                if return_layout.shape.is_unresolved()
                    && let Some(fallback) = self.deferred_return_shape_fallbacks[func_idx]
                {
                    return_layout.shape = fallback;
                }

                if return_layout.shape.is_unresolved() {
                    panic!(
                        "failed to resolve runtime return shape in `{}` for `{}`",
                        func.symbol_name,
                        func.ret_ty.pretty_print(self.db)
                    );
                }

                return_layout
            })
            .collect()
    }

    fn recompute_local_layouts(
        &self,
        return_layouts: &[ReturnRuntimeLayoutState<'db>],
    ) -> Vec<LocalRuntimeLayoutState<'db>> {
        let mut next_local_layouts = self
            .module
            .functions
            .iter()
            .map(LocalRuntimeLayoutState::from_function)
            .collect::<Vec<_>>();

        self.reset_deferred_local_fallbacks(&mut next_local_layouts);
        for (func_idx, local_layouts) in next_local_layouts.iter_mut().enumerate() {
            self.propagate_local_layouts_from_body(func_idx, local_layouts, return_layouts);
        }
        self.propagate_callsite_param_runtime_shapes(&mut next_local_layouts);
        self.restore_unresolved_deferred_local_fallbacks(&mut next_local_layouts);
        for (func_idx, local_layouts) in next_local_layouts.iter_mut().enumerate() {
            let func = &self.module.functions[func_idx];
            let core = function_core_lib(self.db, func);
            clamp_plain_word_local_layouts(self.db, &core, func, local_layouts);
        }
        next_local_layouts
    }

    fn reset_deferred_local_fallbacks(
        &self,
        next_local_layouts: &mut [LocalRuntimeLayoutState<'db>],
    ) {
        for (func_idx, local_layouts) in next_local_layouts
            .iter_mut()
            .enumerate()
            .take(self.deferred_local_fallbacks.len())
        {
            let local_fallbacks = &self.deferred_local_fallbacks[func_idx];
            for (local_idx, fallback) in local_fallbacks.iter().copied().enumerate() {
                let Some(fallback) = fallback else {
                    continue;
                };
                let local = LocalId(local_idx as u32);
                if local_layouts.shape(local) != fallback {
                    continue;
                }
                local_layouts.set_shape(local, RuntimeShape::Unresolved);
                local_layouts.set_pointer_leaf_infos(
                    local,
                    pointer_leaf_infos_without_root(local_layouts.pointer_leaf_infos(local)),
                );
            }
        }
    }

    fn restore_unresolved_deferred_local_fallbacks(
        &self,
        next_local_layouts: &mut [LocalRuntimeLayoutState<'db>],
    ) {
        for (func_idx, local_layouts) in next_local_layouts
            .iter_mut()
            .enumerate()
            .take(self.deferred_local_fallbacks.len())
        {
            let local_fallbacks = &self.deferred_local_fallbacks[func_idx];
            for (local_idx, fallback) in local_fallbacks.iter().copied().enumerate() {
                let Some(fallback) = fallback else {
                    continue;
                };
                let local = LocalId(local_idx as u32);
                if !local_layouts.shape(local).is_unresolved() {
                    continue;
                }
                local_layouts.set_shape(local, fallback);
                local_layouts.set_pointer_leaf_infos(
                    local,
                    pointer_leaf_infos_with_runtime_root(
                        self.db,
                        &function_core_lib(self.db, &self.module.functions[func_idx]),
                        self.module.functions[func_idx].body.local(local).ty,
                        local_layouts.pointer_leaf_infos(local),
                        fallback,
                    ),
                );
            }
        }
    }

    fn propagate_local_layouts_from_body(
        &self,
        func_idx: usize,
        local_layouts: &mut LocalRuntimeLayoutState<'db>,
        return_layouts: &[ReturnRuntimeLayoutState<'db>],
    ) {
        let func = &self.module.functions[func_idx];
        let core = function_core_lib(self.db, func);
        let mut current_locals = func.body.locals.clone();

        for block in &func.body.blocks {
            for inst in &block.insts {
                match inst {
                    MirInst::Assign {
                        dest: Some(dest),
                        rvalue,
                        ..
                    } => {
                        let Some((shape, raw_assigned_pointer_infos)) = ({
                            let resolver = AssignedRvalueResolver {
                                db: self.db,
                                core: &core,
                                func,
                                locals: &current_locals,
                                return_layouts,
                                func_indices: &self.func_indices,
                            };
                            resolver
                                .runtime_shape(*dest, rvalue)
                                .map(|shape| (shape, resolver.pointer_leaf_infos(*dest, rvalue)))
                        }) else {
                            continue;
                        };
                        let assigned_pointer_infos = retag_pointer_leaf_infos_for_local(
                            self.db,
                            &core,
                            func.body.local(*dest),
                            &raw_assigned_pointer_infos,
                        );
                        let shape = if matches!(
                            shape,
                            RuntimeShape::ConstRef { .. } | RuntimeShape::ObjectRef { .. }
                        ) {
                            shape
                        } else {
                            direct_local_root_pointer_shape(
                                self.db,
                                &core,
                                func.body.local(*dest),
                                &assigned_pointer_infos,
                            )
                            .unwrap_or(shape)
                        };
                        if matches!(
                            func.body.local(*dest).address_space,
                            AddressSpaceKind::Storage
                                | AddressSpaceKind::TransientStorage
                                | AddressSpaceKind::Calldata
                        ) {
                            continue;
                        }
                        let Some(merged) = specialize_call_param_runtime_shape(
                            self.db,
                            &core,
                            func.body.local(*dest),
                            local_layouts.shape(*dest),
                            shape,
                        ) else {
                            continue;
                        };
                        let (merged, normalized_infos) = normalize_mutable_local_runtime_shape(
                            self.db,
                            &core,
                            func.body.local(*dest),
                            local_layouts.pointer_leaf_infos(*dest),
                            merged,
                        );
                        let merged = normalize_plain_word_local_runtime_shape(
                            self.db,
                            &core,
                            func.body.local(*dest),
                            merged,
                        );
                        local_layouts.set_shape(*dest, merged);
                        local_layouts.set_pointer_leaf_infos(*dest, normalized_infos);
                        normalize_local_layout_pointer_infos(
                            self.db,
                            &core,
                            func,
                            local_layouts,
                            *dest,
                        );
                        sync_local_snapshot(
                            self.db,
                            &core,
                            &mut current_locals,
                            local_layouts,
                            *dest,
                        );

                        if !assigned_pointer_infos.is_empty() {
                            local_layouts.set_pointer_leaf_infos(
                                *dest,
                                replace_pointer_leaf_infos(
                                    self.db,
                                    local_layouts.pointer_leaf_infos(*dest),
                                    &assigned_pointer_infos,
                                )
                                .unwrap_or_else(|| {
                                    panic!(
                                        "incompatible pointer leaf infos for local v{} `{}` in `{}` from rvalue {:?}: {:?} vs {:?}",
                                        dest.index(),
                                        func.body.local(*dest).name,
                                        func.symbol_name,
                                        rvalue,
                                        local_layouts.pointer_leaf_infos(*dest),
                                        assigned_pointer_infos
                                    )
                                }),
                            );
                            normalize_local_layout_pointer_infos(
                                self.db,
                                &core,
                                func,
                                local_layouts,
                                *dest,
                            );
                            sync_local_snapshot(
                                self.db,
                                &core,
                                &mut current_locals,
                                local_layouts,
                                *dest,
                            );
                        }

                        if let Rvalue::Value(value) = rvalue
                            && let Some((source_local, projection)) =
                                crate::ir::resolve_local_projection_root(&func.body.values, *value)
                            && projection.is_empty()
                            && matches!(
                                func.body.local(source_local).address_space,
                                AddressSpaceKind::Memory
                            )
                        {
                            // Keep direct memory-local aliases in sync when precise
                            // pointer facts are first discovered on the assignee.
                            // Without this backfill, later place-sensitive queries can
                            // depend on which alias happened to pick up leaf metadata
                            // first during fixpoint propagation.
                            local_layouts.set_pointer_leaf_infos(
                                source_local,
                                merge_pointer_leaf_infos(
                                    self.db,
                                    local_layouts.pointer_leaf_infos(source_local),
                                    local_layouts.pointer_leaf_infos(*dest),
                                )
                                .unwrap_or_else(|| {
                                    panic!(
                                        "incompatible pointer leaf infos for source local v{} `{}` in `{}` after assignment into v{} `{}`: {:?} vs {:?}",
                                        source_local.index(),
                                        func.body.local(source_local).name,
                                        func.symbol_name,
                                        dest.index(),
                                        func.body.local(*dest).name,
                                        local_layouts.pointer_leaf_infos(source_local),
                                        local_layouts.pointer_leaf_infos(*dest),
                                    )
                                }),
                            );
                            normalize_local_layout_pointer_infos(
                                self.db,
                                &core,
                                func,
                                local_layouts,
                                source_local,
                            );
                            sync_local_snapshot(
                                self.db,
                                &core,
                                &mut current_locals,
                                local_layouts,
                                source_local,
                            );
                        }
                    }
                    MirInst::Assign {
                        rvalue: Rvalue::Call(call),
                        ..
                    } => {
                        let mut invalidated = false;
                        if let Some(CallTargetRef::Hir(target)) = call.target.as_ref()
                            && let CallableDef::Func(hir_func) = target.callable_def
                        {
                            for (idx, param) in hir_func.params(self.db).enumerate() {
                                if !param.is_mut(self.db)
                                    && !crate::repr::ty_has_mut_capability(
                                        self.db,
                                        param.ty(self.db),
                                    )
                                {
                                    continue;
                                }
                                let Some(arg) = call.args.get(idx).copied() else {
                                    break;
                                };
                                invalidate_mutating_call_arg_local_layout(
                                    self.db,
                                    &core,
                                    func,
                                    local_layouts,
                                    &mut current_locals,
                                    arg,
                                );
                                invalidated = true;
                            }
                        }
                        if !invalidated
                            && let Some(callee_name) = call.resolved_name.as_deref()
                            && let Some(&callee_idx) = self.func_indices.get(callee_name)
                        {
                            let callee = &self.module.functions[callee_idx];
                            for (param_local, arg) in
                                callee.runtime_param_locals().into_iter().zip(&call.args)
                            {
                                if !crate::repr::local_is_semantically_mutable(
                                    self.db,
                                    callee.body.local(param_local),
                                ) {
                                    continue;
                                }
                                invalidate_mutating_call_arg_local_layout(
                                    self.db,
                                    &core,
                                    func,
                                    local_layouts,
                                    &mut current_locals,
                                    *arg,
                                );
                            }
                        }
                    }
                    MirInst::Store { place, value, .. } => {
                        let value_pointer_infos = crate::repr::pointer_leaf_infos_for_value(
                            self.db,
                            &core,
                            &func.body.values,
                            &current_locals,
                            *value,
                        );
                        place_store_updates_local_pointer_leaf_infos(
                            self.db,
                            func,
                            local_layouts,
                            place,
                            &value_pointer_infos,
                        );
                        if let Some((local, _)) =
                            crate::ir::resolve_local_projection_root(&func.body.values, place.base)
                        {
                            normalize_local_layout_pointer_infos(
                                self.db,
                                &core,
                                func,
                                local_layouts,
                                local,
                            );
                            sync_local_snapshot(
                                self.db,
                                &core,
                                &mut current_locals,
                                local_layouts,
                                local,
                            );
                        }
                    }
                    MirInst::InitAggregate { place, inits, .. } => {
                        for (path, value) in inits {
                            let value_pointer_infos = crate::repr::pointer_leaf_infos_for_value(
                                self.db,
                                &core,
                                &func.body.values,
                                &current_locals,
                                *value,
                            );
                            place_store_updates_local_pointer_leaf_infos(
                                self.db,
                                func,
                                local_layouts,
                                &Place::new(place.base, place.projection.concat(path)),
                                &value_pointer_infos,
                            );
                            if let Some((local, _)) = crate::ir::resolve_local_projection_root(
                                &func.body.values,
                                place.base,
                            ) {
                                normalize_local_layout_pointer_infos(
                                    self.db,
                                    &core,
                                    func,
                                    local_layouts,
                                    local,
                                );
                                sync_local_snapshot(
                                    self.db,
                                    &core,
                                    &mut current_locals,
                                    local_layouts,
                                    local,
                                );
                            }
                        }
                    }
                    MirInst::Assign { dest: None, .. }
                    | MirInst::SetDiscriminant { .. }
                    | MirInst::BindValue { .. } => {}
                }
            }
        }
    }

    fn propagate_callsite_param_runtime_shapes(
        &self,
        next_local_layouts: &mut [LocalRuntimeLayoutState<'db>],
    ) {
        for caller_idx in 0..self.module.functions.len() {
            let func = &self.module.functions[caller_idx];
            for block in &func.body.blocks {
                for inst in &block.insts {
                    let MirInst::Assign {
                        rvalue: Rvalue::Call(call),
                        ..
                    } = inst
                    else {
                        continue;
                    };
                    refine_call_param_local_shapes(
                        self.db,
                        self.module,
                        caller_idx,
                        call,
                        next_local_layouts,
                        &self.func_indices,
                    );
                }
                if let Terminator::TerminatingCall {
                    call: TerminatingCall::Call(call),
                    ..
                } = &block.terminator
                {
                    refine_call_param_local_shapes(
                        self.db,
                        self.module,
                        caller_idx,
                        call,
                        next_local_layouts,
                        &self.func_indices,
                    );
                }
            }
        }
    }

    fn apply_return_layouts(
        &mut self,
        next_return_layouts: Vec<ReturnRuntimeLayoutState<'db>>,
    ) -> bool {
        let mut changed = false;
        for (func_idx, next_layout) in next_return_layouts.into_iter().enumerate() {
            changed |= next_layout.apply(&mut self.module.functions[func_idx]);
        }
        changed
    }

    fn apply_local_layouts(
        &mut self,
        next_local_layouts: Vec<LocalRuntimeLayoutState<'db>>,
    ) -> bool {
        let mut changed = false;
        for (func_idx, next_layouts) in next_local_layouts.into_iter().enumerate() {
            changed |= next_layouts.apply(&mut self.module.functions[func_idx]);
        }
        changed
    }

    fn apply_deferred_const_ref_fallbacks(&mut self) -> bool {
        let mut changed = false;

        for (func_idx, local_fallbacks) in self.deferred_local_fallbacks.iter().enumerate() {
            for (local_idx, fallback) in local_fallbacks.iter().enumerate() {
                let Some(fallback) = *fallback else {
                    continue;
                };
                let local = &mut self.module.functions[func_idx].body.locals[local_idx];
                if !local.runtime_shape.is_unresolved() {
                    continue;
                }
                local.runtime_shape = fallback;
                changed = true;
            }
        }

        for (func_idx, fallback) in self.deferred_return_shape_fallbacks.iter().enumerate() {
            let Some(fallback) = *fallback else {
                continue;
            };
            let runtime_return_shape = &mut self.module.functions[func_idx].runtime_return_shape;
            if !runtime_return_shape.is_unresolved() {
                continue;
            }
            *runtime_return_shape = fallback;
            changed = true;
        }

        changed
    }
}

pub(crate) fn normalize_runtime_shapes<'db>(
    db: &'db dyn HirAnalysisDb,
    module: &mut MirModule<'db>,
) {
    RuntimeLayoutFixpoint::new(db, module).run();
}

/// Canonicalize transparent-newtype operations in MIR.
///
/// This pass enforces a single representation strategy for transparent single-field wrappers
/// (single-field structs and single-element tuples):
/// - Collapses chains of `ValueOrigin::TransparentCast` so downstream passes don't need to chase
///   multiple hops.
/// - Rewrites `Place` projection paths to peel `.0` field projections over transparent newtypes
///   by inserting type-only `TransparentCast`s on the base address value and removing the no-op
///   field projection.
///
/// This is intended as a post-lowering cleanup that reduces scattered newtype handling in later
/// passes and in codegen.
pub(crate) fn canonicalize_transparent_newtypes<'db>(
    db: &'db dyn HirAnalysisDb,
    body: &mut MirBody<'db>,
) {
    fn alloc_value<'db>(values: &mut Vec<ValueData<'db>>, data: ValueData<'db>) -> ValueId {
        let id = ValueId(values.len() as u32);
        values.push(data);
        id
    }

    fn flatten_transparent_cast_chains<'db>(values: &mut [ValueData<'db>]) {
        for idx in 0..values.len() {
            let ValueOrigin::TransparentCast { value: mut inner } = values[idx].origin else {
                continue;
            };
            let original_inner = inner;
            // Bound the walk defensively (cycles should be impossible).
            for _ in 0..values.len() {
                match values.get(inner.index()).map(|v| &v.origin) {
                    Some(ValueOrigin::TransparentCast { value }) => inner = *value,
                    _ => break,
                }
            }
            values[idx].origin = ValueOrigin::TransparentCast { value: inner };
            if inner != original_inner {
                // Transparent-cast flattening can invalidate any cached pointee metadata inherited
                // from the outer cast chain. Let later queries re-derive it from the new inner
                // value instead of preserving stale address-space information.
                values[idx].pointer_info = None;
            }
        }
    }

    fn apply_projection_to_ty<'db>(
        db: &'db dyn HirAnalysisDb,
        ty: TyId<'db>,
        proj: &Projection<TyId<'db>, EnumVariant<'db>, ValueId>,
    ) -> Option<TyId<'db>> {
        match proj {
            Projection::Field(field_idx) => ty.field_types(db).get(*field_idx).copied(),
            Projection::VariantField {
                variant,
                enum_ty,
                field_idx,
            } => {
                let ctor = ConstructorKind::Variant(*variant, *enum_ty);
                ctor.field_types(db).get(*field_idx).copied()
            }
            Projection::Discriminant => {
                Some(TyId::new(db, TyData::TyBase(TyBase::Prim(PrimTy::U256))))
            }
            Projection::Index(_idx) => {
                let (base, args) = ty.decompose_ty_app(db);
                (base.is_array(db) && !args.is_empty()).then(|| args[0])
            }
            Projection::Deref => None,
        }
    }

    fn canonicalize_place<'db>(
        db: &'db dyn HirAnalysisDb,
        values: &mut Vec<ValueData<'db>>,
        locals: &[LocalData<'db>],
        place: Place<'db>,
    ) -> Place<'db> {
        // Only attempt to canonicalize places rooted at pointer-like values. If the base is a
        // pure word with no address space, treating it as an address is a bug; this pass avoids
        // "fixing" such cases into memory loads/stores.
        if values
            .get(place.base.index())
            .is_none_or(|v| v.repr.address_space().is_none())
        {
            return place;
        }

        let mut base = place.base;
        let mut current_ty = values[base.index()].ty;
        let mut path = MirProjectionPath::new();

        for proj in place.projection.iter() {
            // Peel transparent-newtype field 0 projections by retyping the base address.
            if let Some(inner_ty) =
                crate::repr::transparent_field0_projection_step_ty(db, current_ty, proj)
            {
                let base_at_point = if path.is_empty() {
                    base
                } else {
                    let addr_space = crate::ir::try_value_address_space_in(values, locals, base)
                        .expect("pointer-like canonicalize base must carry an address space");
                    let prefix_place = Place::new(base, path);
                    alloc_value(
                        values,
                        ValueData {
                            ty: current_ty,
                            origin: ValueOrigin::PlaceRef(prefix_place),
                            source: SourceInfoId::SYNTHETIC,
                            repr: ValueRepr::Ref(addr_space),
                            pointer_info: Some(crate::ir::PointerInfo {
                                address_space: addr_space,
                                target_ty: Some(current_ty),
                            }),
                            runtime_shape: crate::ir::RuntimeShape::Unresolved,
                        },
                    )
                };

                let repr = values[base_at_point.index()].repr;
                base = alloc_value(
                    values,
                    ValueData {
                        ty: inner_ty,
                        origin: ValueOrigin::TransparentCast {
                            value: base_at_point,
                        },
                        source: SourceInfoId::SYNTHETIC,
                        repr,
                        pointer_info: None,
                        runtime_shape: crate::ir::RuntimeShape::Unresolved,
                    },
                );
                current_ty = inner_ty;
                path = MirProjectionPath::new();
                continue;
            }

            path.push(proj.clone());
            if let Some(next) = apply_projection_to_ty(db, current_ty, proj) {
                current_ty = next;
            }
        }

        Place::new(base, path)
    }

    flatten_transparent_cast_chains(&mut body.values);

    let (values, blocks, locals) = (&mut body.values, &mut body.blocks, &body.locals);

    let initial_values_len = values.len();
    for idx in 0..initial_values_len {
        let (place, is_move_out) = match &values[idx].origin {
            ValueOrigin::PlaceRef(place) => (Some(place.clone()), false),
            ValueOrigin::MoveOut { place } => (Some(place.clone()), true),
            _ => (None, false),
        };
        if let Some(place) = place {
            let updated = canonicalize_place(db, values, locals, place);
            values[idx].origin = if is_move_out {
                ValueOrigin::MoveOut { place: updated }
            } else {
                ValueOrigin::PlaceRef(updated)
            };
        }
    }

    for block in blocks {
        for inst in &mut block.insts {
            match inst {
                MirInst::Assign { rvalue, .. } => {
                    if let Rvalue::Load { place } = rvalue {
                        *place = canonicalize_place(db, values, locals, place.clone());
                    }
                }
                MirInst::Store { place, .. } => {
                    *place = canonicalize_place(db, values, locals, place.clone());
                }
                MirInst::InitAggregate { place, .. } => {
                    *place = canonicalize_place(db, values, locals, place.clone());
                }
                MirInst::SetDiscriminant { place, .. } => {
                    *place = canonicalize_place(db, values, locals, place.clone());
                }
                MirInst::BindValue { .. } => {}
            }
        }
    }

    flatten_transparent_cast_chains(values);
}

/// Canonicalize zero-sized types (ZSTs) in MIR.
///
/// After this pass:
/// - ZST-returning ops do not produce runtime values; their values are rewritten to `Unit`.
/// - `Eval`/`EvalValue` instructions for ZST values are removed (their effects
///   are preserved by explicit effectful MIR instructions (`Call`/`Intrinsic`/`Load`/`Alloc`).
/// - `Store`/`InitAggregate` instructions of ZST values are removed, but any
///   dynamic index computations and RHS evaluations are preserved via inserted
///   `EvalValue`s to maintain evaluation order.
/// - `Return(Some(v))` where `v` is ZST is rewritten to `Return(None)`.
pub(crate) fn canonicalize_zero_sized<'db>(db: &'db dyn HirAnalysisDb, body: &mut MirBody<'db>) {
    fn is_zst(db: &dyn HirAnalysisDb, ty: hir::analysis::ty::ty_def::TyId<'_>) -> bool {
        layout::is_zero_sized_ty(db, ty)
    }

    fn mark_unit<'db>(values: &mut [ValueData<'db>], value: ValueId) {
        let data = &mut values[value.index()];
        data.origin = ValueOrigin::Unit;
        data.repr = ValueRepr::Word;
    }

    fn push_eval_value<'db>(
        db: &'db dyn HirAnalysisDb,
        values: &mut [ValueData<'db>],
        out: &mut Vec<MirInst<'db>>,
        value: ValueId,
    ) {
        let value_ty = values[value.index()].ty;
        if !is_zst(db, value_ty) {
            out.push(MirInst::Assign {
                source: SourceInfoId::SYNTHETIC,
                dest: None,
                rvalue: Rvalue::Value(value),
            });
            return;
        }

        // ZST: the value has no runtime representation.
        mark_unit(values, value);
    }

    fn push_place_eval<'db>(
        db: &'db dyn HirAnalysisDb,
        values: &mut [ValueData<'db>],
        out: &mut Vec<MirInst<'db>>,
        place: &crate::ir::Place<'db>,
    ) {
        push_eval_value(db, values, out, place.base);
        for proj in place.projection.iter() {
            if let Projection::Index(IndexSource::Dynamic(value)) = proj {
                push_eval_value(db, values, out, *value);
            }
        }
    }

    fn push_path_eval<'db>(
        db: &'db dyn HirAnalysisDb,
        values: &mut [ValueData<'db>],
        out: &mut Vec<MirInst<'db>>,
        path: &crate::ir::MirProjectionPath<'db>,
    ) {
        for proj in path.iter() {
            if let Projection::Index(IndexSource::Dynamic(value)) = proj {
                push_eval_value(db, values, out, *value);
            }
        }
    }

    let zst_locals: Vec<bool> = body
        .locals
        .iter()
        .map(|local| is_zst(db, local.ty))
        .collect();
    let (blocks, values) = (&mut body.blocks, &mut body.values);
    for block in blocks {
        let mut rewritten: Vec<MirInst<'db>> = Vec::with_capacity(block.insts.len());
        for inst in std::mem::take(&mut block.insts) {
            match inst {
                MirInst::Assign {
                    source,
                    dest,
                    rvalue,
                } => match dest {
                    Some(dest) if zst_locals.get(dest.index()).copied().unwrap_or(false) => {
                        // Dest is ZST: keep side effects, drop runtime write.
                        match rvalue {
                            Rvalue::Call(call) => rewritten.push(MirInst::Assign {
                                source,
                                dest: None,
                                rvalue: Rvalue::Call(call),
                            }),
                            Rvalue::Intrinsic { op, args } => rewritten.push(MirInst::Assign {
                                source,
                                dest: None,
                                rvalue: Rvalue::Intrinsic { op, args },
                            }),
                            Rvalue::Value(value) => {
                                // Pure value, no runtime representation needed.
                                if is_zst(db, values[value.index()].ty) {
                                    mark_unit(values, value);
                                }
                            }
                            Rvalue::Load { place } => {
                                // Even though the loaded value is ZST (so the write can be
                                // dropped), we must still evaluate the load's place to preserve
                                // any side effects in the base/index expressions.
                                push_place_eval(db, values, &mut rewritten, &place);
                            }
                            Rvalue::Alloc { .. }
                            | Rvalue::ZeroInit
                            | Rvalue::ConstAggregate { .. } => {}
                        }
                    }
                    _ => {
                        // Dest is non-ZST (or none): canonicalize ZST-valued evals.
                        if dest.is_none()
                            && let Rvalue::Value(value) = &rvalue
                        {
                            let value_ty = values[value.index()].ty;
                            if is_zst(db, value_ty) {
                                mark_unit(values, *value);
                                continue;
                            }
                        }
                        rewritten.push(MirInst::Assign {
                            source,
                            dest,
                            rvalue,
                        });
                    }
                },
                MirInst::BindValue { source, value } => {
                    let value_ty = values[value.index()].ty;
                    if is_zst(db, value_ty) {
                        push_eval_value(db, values, &mut rewritten, value);
                    } else {
                        rewritten.push(MirInst::BindValue { source, value });
                    }
                }
                MirInst::Store {
                    source,
                    place,
                    value,
                } => {
                    let value_ty = values[value.index()].ty;
                    if is_zst(db, value_ty) {
                        push_place_eval(db, values, &mut rewritten, &place);
                        push_eval_value(db, values, &mut rewritten, value);
                    } else {
                        rewritten.push(MirInst::Store {
                            source,
                            place,
                            value,
                        });
                    }
                }
                MirInst::InitAggregate {
                    source,
                    place,
                    inits,
                } => {
                    let base_ty = values[place.base.index()].ty;
                    if is_zst(db, base_ty) {
                        push_place_eval(db, values, &mut rewritten, &place);
                        for (path, value) in &inits {
                            push_path_eval(db, values, &mut rewritten, path);
                            push_eval_value(db, values, &mut rewritten, *value);
                        }
                    } else {
                        let mut kept: Vec<(crate::ir::MirProjectionPath<'db>, ValueId)> =
                            Vec::with_capacity(inits.len());
                        let mut removed_any = false;
                        for (path, value) in inits {
                            let value_ty = values[value.index()].ty;
                            if is_zst(db, value_ty) {
                                if !removed_any {
                                    push_place_eval(db, values, &mut rewritten, &place);
                                    removed_any = true;
                                }
                                push_path_eval(db, values, &mut rewritten, &path);
                                push_eval_value(db, values, &mut rewritten, value);
                            } else {
                                kept.push((path, value));
                            }
                        }
                        if kept.is_empty() {
                            if !removed_any {
                                // No inits to keep, but still preserve evaluation of the base.
                                push_place_eval(db, values, &mut rewritten, &place);
                            }
                        } else {
                            rewritten.push(MirInst::InitAggregate {
                                source,
                                place,
                                inits: kept,
                            });
                        }
                    }
                }
                other => rewritten.push(other),
            }
        }

        if let Terminator::Return {
            source,
            value: Some(value),
        } = &mut block.terminator
        {
            let ty = values[value.index()].ty;
            if is_zst(db, ty) {
                // Ensure any side effects are emitted before the return.
                push_eval_value(db, values, &mut rewritten, *value);
                block.terminator = Terminator::Return {
                    source: *source,
                    value: None,
                };
            }
        }

        block.insts = rewritten;
    }

    // Ensure no runtime value references a zero-sized local.
    //
    // The instruction forms that "define" locals (e.g. `Alloc`, `Load`, `Call` dests) can be
    // removed for ZSTs above. Any MIR `ValueId` that still points at such locals must be
    // canonicalized to `Unit` so codegen never needs to bind/resolve a runtime representation.
    for value in values.iter_mut() {
        if let ValueOrigin::Local(local) | ValueOrigin::PlaceRoot(local) = &value.origin
            && zst_locals.get(local.index()).copied().unwrap_or(false)
        {
            value.origin = ValueOrigin::Unit;
            value.repr = ValueRepr::Word;
        }
    }

    let mut unit_values: Vec<bool> = values
        .iter()
        .map(|value| matches!(value.origin, ValueOrigin::Unit))
        .collect();
    let mut changed = true;
    while changed {
        changed = false;
        for (idx, value) in values.iter_mut().enumerate() {
            let collapses_to_unit = match &value.origin {
                ValueOrigin::TransparentCast { value } => {
                    unit_values.get(value.index()).copied().unwrap_or(false)
                }
                ValueOrigin::PlaceRef(place) | ValueOrigin::MoveOut { place } => unit_values
                    .get(place.base.index())
                    .copied()
                    .unwrap_or(false),
                _ => false,
            };
            if collapses_to_unit && !unit_values[idx] {
                value.origin = ValueOrigin::Unit;
                value.repr = ValueRepr::Word;
                unit_values[idx] = true;
                changed = true;
            }
        }
    }
}

fn value_should_bind(
    db: &dyn HirAnalysisDb,
    value_id: ValueId,
    value: &ValueData<'_>,
    origin: &ValueOrigin<'_>,
    value_use_counts: &[usize],
    force_root_bind: bool,
) -> bool {
    if force_root_bind {
        return true;
    }
    if layout::is_zero_sized_ty(db, value.ty) {
        return false;
    }
    value_use_counts.get(value_id.index()).copied().unwrap_or(0) > 1
        && !matches!(
            origin,
            ValueOrigin::Unit
                | ValueOrigin::Synthetic(..)
                | ValueOrigin::Local(..)
                | ValueOrigin::PlaceRoot(..)
                | ValueOrigin::CodeRegionRef(..)
        )
}

fn value_deps_in_eval_order(origin: &ValueOrigin<'_>) -> Vec<ValueId> {
    match origin {
        ValueOrigin::Unary { inner, .. } => vec![*inner],
        ValueOrigin::Binary { lhs, rhs, .. } => vec![*lhs, *rhs],
        ValueOrigin::FieldPtr(field_ptr) => vec![field_ptr.base],
        ValueOrigin::PlaceRef(place) | ValueOrigin::MoveOut { place } => {
            let mut deps = vec![place.base];
            for proj in place.projection.iter() {
                if let Projection::Index(IndexSource::Dynamic(value)) = proj {
                    deps.push(*value);
                }
            }
            deps
        }
        ValueOrigin::TransparentCast { value } => vec![*value],
        ValueOrigin::ConstRegion(_) => vec![],
        ValueOrigin::Expr(..)
        | ValueOrigin::ControlFlowResult { .. }
        | ValueOrigin::Unit
        | ValueOrigin::Synthetic(..)
        | ValueOrigin::Local(..)
        | ValueOrigin::PlaceRoot(..)
        | ValueOrigin::CodeRegionRef(..) => Vec::new(),
    }
}

fn compute_value_use_counts<'db>(body: &MirBody<'db>) -> Vec<usize> {
    let mut counts = vec![0usize; body.values.len()];

    let mut bump = |value: ValueId| {
        if let Some(slot) = counts.get_mut(value.index()) {
            *slot += 1;
        }
    };

    for value in &body.values {
        for dep in value_deps_in_eval_order(&value.origin) {
            bump(dep);
        }
    }

    for block in &body.blocks {
        for inst in &block.insts {
            match inst {
                MirInst::BindValue { value, .. } => bump(*value),
                MirInst::Assign { rvalue, .. } => match rvalue {
                    Rvalue::Value(value) => bump(*value),
                    Rvalue::Call(call) => {
                        for arg in call.args.iter().chain(call.effect_args.iter()) {
                            bump(*arg);
                        }
                    }
                    Rvalue::Intrinsic { args, .. } => {
                        for arg in args {
                            bump(*arg);
                        }
                    }
                    Rvalue::Load { place } => {
                        bump(place.base);
                        bump_place_path(&mut bump, &place.projection);
                    }
                    Rvalue::Alloc { .. } | Rvalue::ZeroInit | Rvalue::ConstAggregate { .. } => {}
                },
                MirInst::Store { place, value, .. } => {
                    bump(place.base);
                    bump_place_path(&mut bump, &place.projection);
                    bump(*value);
                }
                MirInst::InitAggregate { place, inits, .. } => {
                    bump(place.base);
                    bump_place_path(&mut bump, &place.projection);
                    for (path, value) in inits {
                        bump_place_path(&mut bump, path);
                        bump(*value);
                    }
                }
                MirInst::SetDiscriminant { place, .. } => {
                    bump(place.base);
                    bump_place_path(&mut bump, &place.projection);
                }
            }
        }

        match &block.terminator {
            Terminator::Return {
                value: Some(value), ..
            } => bump(*value),
            Terminator::TerminatingCall { call, .. } => match call {
                TerminatingCall::Call(call) => {
                    for arg in call.args.iter().chain(call.effect_args.iter()) {
                        bump(*arg);
                    }
                }
                TerminatingCall::Intrinsic { args, .. } => {
                    for arg in args {
                        bump(*arg);
                    }
                }
            },
            Terminator::Branch { cond, .. } => bump(*cond),
            Terminator::Switch { discr, .. } => bump(*discr),
            Terminator::Return { value: None, .. }
            | Terminator::Goto { .. }
            | Terminator::Unreachable { .. } => {}
        }
    }

    counts
}

fn bump_place_path<'db>(bump: &mut impl FnMut(ValueId), path: &crate::ir::MirProjectionPath<'db>) {
    for proj in path.iter() {
        if let Projection::Index(IndexSource::Dynamic(value)) = proj {
            bump(*value);
        }
    }
}

#[cfg(test)]
mod tests {
    use common::InputDb;
    use driver::DriverDataBase;
    use hir::analysis::ty::ty_check::check_func_body;
    use hir::analysis::ty::ty_def::TyId;
    use hir::projection::Projection;
    use url::Url;

    use crate::{
        CoreLib, MirInst,
        ir::{
            AddressSpaceKind, BasicBlock, IntrinsicOp, LocalData, LocalId, LocalPlaceRootLayout,
            MirBody, MirProjectionPath, ObjectRootSource, Place, PointerInfo, RuntimeShape,
            RuntimeWordKind, Rvalue, SourceInfoId, Terminator, ValueData, ValueId, ValueOrigin,
            ValueRepr,
        },
        lower::{lower_function, lower_module},
    };

    fn lower_inline_module<'db>(
        db: &'db mut DriverDataBase,
        path: &str,
        src: &str,
    ) -> crate::MirModule<'db> {
        let url = Url::parse(path).expect("test url should be valid");
        let file = db.workspace().touch(db, url, Some(src.to_owned()));
        let top_mod = db.top_mod(file);
        lower_module(db, top_mod).expect("module should lower")
    }

    #[test]
    fn transparent_newtype_projection_keeps_non_memory_space() {
        let mut db = DriverDataBase::default();
        let url =
            Url::parse("file:///transparent_newtype_projection_keeps_non_memory_space.fe").unwrap();
        let src = r#"
struct Wrap {
    inner: u256,
}

pub fn transparent_newtype_projection_keeps_non_memory_space(w: mut Wrap) {}
"#;
        let file = db.workspace().touch(&mut db, url, Some(src.to_string()));
        let top_mod = db.top_mod(file);
        let module = lower_module(&db, top_mod).expect("module should lower");
        let wrap_fn = module
            .functions
            .iter()
            .find(|func| {
                func.symbol_name == "transparent_newtype_projection_keeps_non_memory_space"
            })
            .expect("function should exist");
        let wrap_ty = wrap_fn.body.local(wrap_fn.body.param_locals[0]).ty;
        let (_, wrap_inner_ty) = wrap_ty
            .as_capability(&db)
            .expect("parameter should be a capability type");
        let inner_ty = wrap_inner_ty.field_types(&db)[0];

        let mut body = MirBody::new();
        body.blocks.push(BasicBlock {
            insts: Vec::new(),
            terminator: Terminator::Return {
                source: SourceInfoId::SYNTHETIC,
                value: None,
            },
        });

        let wrap_local = body.alloc_local(LocalData {
            name: "w".to_string(),
            ty: wrap_inner_ty,
            is_mut: true,
            source: SourceInfoId::SYNTHETIC,
            address_space: AddressSpaceKind::Storage,
            pointer_leaf_infos: Vec::new(),
            place_root_layout: crate::ir::LocalPlaceRootLayout::Direct,
            const_backing: crate::ir::LocalConstBacking::Unknown,
            runtime_shape: crate::ir::RuntimeShape::Unresolved,
        });
        let base = body.alloc_value(ValueData {
            ty: wrap_inner_ty,
            origin: ValueOrigin::Local(wrap_local),
            source: SourceInfoId::SYNTHETIC,
            repr: ValueRepr::Ref(AddressSpaceKind::Storage),
            pointer_info: None,
            runtime_shape: crate::ir::RuntimeShape::Unresolved,
        });
        let value = body.alloc_value(ValueData {
            ty: inner_ty,
            origin: ValueOrigin::Synthetic(crate::ir::SyntheticValue::Int(1u8.into())),
            source: SourceInfoId::SYNTHETIC,
            repr: ValueRepr::Word,
            pointer_info: None,
            runtime_shape: crate::ir::RuntimeShape::Unresolved,
        });
        body.blocks[0].insts.push(crate::MirInst::Store {
            source: SourceInfoId::SYNTHETIC,
            place: Place::new(
                base,
                MirProjectionPath::from_projection(hir::projection::Projection::Field(0)),
            ),
            value,
        });

        super::canonicalize_transparent_newtypes(&db, &mut body);

        let crate::MirInst::Store { place, .. } = &body.blocks[0].insts[0] else {
            panic!("expected rewritten store");
        };
        assert!(
            place.projection.is_empty(),
            "transparent field projection should be peeled"
        );
        assert_eq!(
            body.value(place.base).repr.address_space(),
            Some(AddressSpaceKind::Storage),
            "peeled base must preserve non-memory address space",
        );
    }

    #[test]
    fn canonicalize_transparent_newtypes_clears_stale_pointer_info_on_flattened_casts() {
        let db = DriverDataBase::default();
        let mut body = MirBody::new();
        body.blocks.push(BasicBlock {
            insts: Vec::new(),
            terminator: Terminator::Return {
                source: SourceInfoId::SYNTHETIC,
                value: None,
            },
        });

        let local = body.alloc_local(LocalData {
            name: "mem".to_string(),
            ty: TyId::u256(&db),
            is_mut: false,
            source: SourceInfoId::SYNTHETIC,
            address_space: AddressSpaceKind::Memory,
            pointer_leaf_infos: Vec::new(),
            place_root_layout: crate::ir::LocalPlaceRootLayout::Direct,
            const_backing: crate::ir::LocalConstBacking::Unknown,
            runtime_shape: RuntimeShape::Unresolved,
        });
        let root = body.alloc_value(ValueData {
            ty: TyId::u256(&db),
            origin: ValueOrigin::Local(local),
            source: SourceInfoId::SYNTHETIC,
            repr: ValueRepr::Ref(AddressSpaceKind::Memory),
            pointer_info: None,
            runtime_shape: RuntimeShape::Unresolved,
        });
        let inner = body.alloc_value(ValueData {
            ty: TyId::u256(&db),
            origin: ValueOrigin::TransparentCast { value: root },
            source: SourceInfoId::SYNTHETIC,
            repr: ValueRepr::Ref(AddressSpaceKind::Memory),
            pointer_info: Some(PointerInfo {
                address_space: AddressSpaceKind::Storage,
                target_ty: Some(TyId::u256(&db)),
            }),
            runtime_shape: RuntimeShape::Unresolved,
        });
        let outer = body.alloc_value(ValueData {
            ty: TyId::u256(&db),
            origin: ValueOrigin::TransparentCast { value: inner },
            source: SourceInfoId::SYNTHETIC,
            repr: ValueRepr::Ref(AddressSpaceKind::Memory),
            pointer_info: Some(PointerInfo {
                address_space: AddressSpaceKind::Storage,
                target_ty: Some(TyId::u256(&db)),
            }),
            runtime_shape: RuntimeShape::Unresolved,
        });

        super::canonicalize_transparent_newtypes(&db, &mut body);

        let ValueOrigin::TransparentCast { value } = body.value(outer).origin else {
            panic!("outer cast should remain a transparent cast");
        };
        assert_eq!(value, root, "cast chain should collapse to the root value");
        assert_eq!(
            crate::ir::try_value_address_space_in(&body.values, &body.locals, outer),
            Some(AddressSpaceKind::Memory),
            "flattened transparent casts must not keep stale non-memory pointer metadata",
        );
    }

    #[test]
    fn canonicalize_zero_sized_rewrites_place_values_over_unit_bases() {
        let db = DriverDataBase::default();
        let unit_ty = TyId::unit(&db);
        let mut body = MirBody::new();
        body.blocks.push(BasicBlock {
            insts: Vec::new(),
            terminator: Terminator::Return {
                source: SourceInfoId::SYNTHETIC,
                value: None,
            },
        });

        let local = body.alloc_local(LocalData {
            name: "zst".to_string(),
            ty: unit_ty,
            is_mut: false,
            source: SourceInfoId::SYNTHETIC,
            address_space: AddressSpaceKind::Memory,
            pointer_leaf_infos: Vec::new(),
            place_root_layout: crate::ir::LocalPlaceRootLayout::Direct,
            const_backing: crate::ir::LocalConstBacking::Unknown,
            runtime_shape: crate::ir::RuntimeShape::Unresolved,
        });
        let root = body.alloc_value(ValueData {
            ty: unit_ty,
            origin: ValueOrigin::Local(local),
            source: SourceInfoId::SYNTHETIC,
            repr: ValueRepr::Word,
            pointer_info: None,
            runtime_shape: crate::ir::RuntimeShape::Unresolved,
        });
        let place = Place::new(root, MirProjectionPath::new());
        let place_ref = body.alloc_value(ValueData {
            ty: unit_ty,
            origin: ValueOrigin::PlaceRef(place.clone()),
            source: SourceInfoId::SYNTHETIC,
            repr: ValueRepr::Word,
            pointer_info: None,
            runtime_shape: crate::ir::RuntimeShape::Unresolved,
        });
        let move_out = body.alloc_value(ValueData {
            ty: unit_ty,
            origin: ValueOrigin::MoveOut { place },
            source: SourceInfoId::SYNTHETIC,
            repr: ValueRepr::Word,
            pointer_info: None,
            runtime_shape: crate::ir::RuntimeShape::Unresolved,
        });

        super::canonicalize_zero_sized(&db, &mut body);

        assert!(matches!(body.value(root).origin, ValueOrigin::Unit));
        assert!(matches!(body.value(place_ref).origin, ValueOrigin::Unit));
        assert!(matches!(body.value(move_out).origin, ValueOrigin::Unit));
    }

    #[test]
    fn runtime_abi_erases_compile_time_only_range_helper_params() {
        let mut db = DriverDataBase::default();
        let module = lower_inline_module(
            &mut db,
            "file:///runtime_abi_erases_compile_time_only_range_helper_params.fe",
            include_str!("../../codegen/tests/fixtures/range_bounds.fe"),
        );

        let helper = module
            .functions
            .iter()
            .find(|func| {
                func.symbol_name.contains("range_known_const")
                    && func.symbol_name.contains("len__0_4")
            })
            .expect("expected generated range helper");
        assert_eq!(
            helper.runtime_param_count(),
            0,
            "compile-time-only range helpers must not keep runtime params",
        );
        assert_eq!(
            helper.runtime_effect_param_count(),
            0,
            "compile-time-only range helpers must not keep runtime effect params",
        );
    }

    #[test]
    fn runtime_abi_rewrites_calls_to_erased_helpers() {
        let mut db = DriverDataBase::default();
        let module = lower_inline_module(
            &mut db,
            "file:///runtime_abi_rewrites_calls_to_erased_helpers.fe",
            include_str!("../../codegen/tests/fixtures/range_bounds.fe"),
        );

        let caller = module
            .functions
            .iter()
            .find(|func| func.symbol_name == "sum_const")
            .expect("expected caller function");

        let call = caller
            .body
            .blocks
            .iter()
            .flat_map(|block| block.insts.iter())
            .find_map(|inst| match inst {
                crate::MirInst::Assign {
                    rvalue: Rvalue::Call(call),
                    ..
                } if call.resolved_name.as_deref().is_some_and(|name| {
                    name.contains("range_known_const") && name.contains("len__0_4")
                }) =>
                {
                    Some(call)
                }
                _ => None,
            })
            .expect("expected call to generated range helper");
        assert!(
            call.args.is_empty() && call.effect_args.is_empty(),
            "calls to erased helpers must be rewritten to the runtime ABI",
        );
    }

    #[test]
    fn runtime_abi_cleanup_removes_dead_zero_arg_create2_encoder_materialization() {
        let mut db = DriverDataBase::default();
        let module = lower_inline_module(
            &mut db,
            "file:///runtime_abi_cleanup_removes_dead_zero_arg_create2_encoder_materialization.fe",
            r#"
pub contract Counter {
    init() {}
}

#[test]
fn runtime_abi_cleanup_removes_dead_zero_arg_create2_encoder_materialization() uses (evm: mut Evm) {
    let addr = evm.create2<Counter>(value: 0, args: (), salt: 0)
    assert(addr.inner != 0)
}
"#,
        );

        let helper_body = module
            .functions
            .iter()
            .find(|func| {
                func.symbol_name.contains("create2_stor_arg0_root_stor")
                    && func.symbol_name.contains("Counter")
            })
            .expect("expected create2 helper")
            .body
            .clone();
        drop(module);

        let has_zero_arg_fast_path = helper_body.blocks.iter().any(|block| {
            let has_create2_raw_call = block.insts.iter().any(|inst| match inst {
                MirInst::Assign {
                    rvalue: Rvalue::Call(call),
                    ..
                } => call
                    .resolved_name
                    .as_deref()
                    .is_some_and(|name| name.contains("create2_raw")),
                MirInst::Store { .. }
                | MirInst::InitAggregate { .. }
                | MirInst::SetDiscriminant { .. }
                | MirInst::BindValue { .. }
                | MirInst::Assign { .. } => false,
            });
            let has_encoder_materialization = block.insts.iter().any(|inst| match inst {
                MirInst::Assign {
                    dest: Some(_),
                    rvalue: Rvalue::Alloc { .. },
                    ..
                } => true,
                MirInst::InitAggregate { .. } => true,
                MirInst::Store { .. }
                | MirInst::SetDiscriminant { .. }
                | MirInst::BindValue { .. }
                | MirInst::Assign { .. } => false,
            });

            has_create2_raw_call && !has_encoder_materialization
        });

        assert!(
            has_zero_arg_fast_path,
            "zero-arg create2 helpers should provide a direct create2_raw fast path without encoder materialization",
        );
    }

    #[test]
    fn runtime_abi_erases_contract_entry_params() {
        let mut db = DriverDataBase::default();
        let module = lower_inline_module(
            &mut db,
            "file:///runtime_abi_erases_contract_entry_params.fe",
            r#"
msg Msg {
    #[selector = 0x01]
    Ping -> u256,
}

contract C {
    recv Msg {
        Ping -> u256 {
            return 1
        }
    }
}
"#,
        );

        let entry = module
            .functions
            .iter()
            .find(|func| func.contract_function.is_some())
            .expect("expected contract entrypoint");
        assert_eq!(entry.runtime_param_count(), 0);
        assert_eq!(entry.runtime_effect_param_count(), 0);
    }

    #[test]
    fn contract_init_entry_keeps_storage_field_effect_calls_storage_specialized() {
        let mut db = DriverDataBase::default();
        let module = lower_inline_module(
            &mut db,
            "file:///contract_init_entry_keeps_storage_field_effect_calls_storage_specialized.fe",
            include_str!("../../codegen/tests/fixtures/high_level_contract.fe"),
        );

        let init_entry = module
            .functions
            .iter()
            .find(|func| func.symbol_name == "__EchoContract_init")
            .expect("expected contract init entrypoint");
        let init_call = init_entry
            .body
            .blocks
            .iter()
            .flat_map(|block| block.insts.iter())
            .find_map(|inst| match inst {
                MirInst::Assign {
                    rvalue: Rvalue::Call(call),
                    ..
                } if call
                    .resolved_name
                    .as_deref()
                    .is_some_and(|name| name.starts_with("__EchoContract_init_contract")) =>
                {
                    Some(call)
                }
                _ => None,
            })
            .expect("expected init handler call");
        assert_eq!(
            init_call.resolved_name.as_deref(),
            Some("__EchoContract_init_contract"),
            "storage-backed contract field effects must keep the storage-specialized init handler",
        );
    }

    #[test]
    fn contract_init_handler_keeps_storage_field_effect_places_storage_backed() {
        let mut db = DriverDataBase::default();
        let module = lower_inline_module(
            &mut db,
            "file:///by_ref_trait_provider_storage_bug.fe",
            include_str!("../../codegen/tests/fixtures/by_ref_trait_provider_storage_bug.fe"),
        );

        let init_handler = module
            .functions
            .iter()
            .find(|func| func.symbol_name == "__ByRefTraitProviderStorageBug_init_contract")
            .expect("expected contract init handler");
        assert!(
            init_handler
                .runtime_abi
                .effect_param_provider_tys
                .first()
                .is_some_and(Option::is_some),
            "synthetic init handlers should preserve effect provider metadata for field effects",
        );
        let effect_local = init_handler.body.effect_param_locals[0];
        assert_eq!(
            init_handler.body.local(effect_local).address_space,
            AddressSpaceKind::Storage,
            "field-backed init effect locals must stay storage-backed",
        );
        let store = init_handler
            .body
            .blocks
            .iter()
            .flat_map(|block| block.insts.iter())
            .find_map(|inst| match inst {
                MirInst::Store { place, .. } => Some(place),
                _ => None,
            })
            .expect("expected init handler store");
        assert!(
            init_handler
                .body
                .local(effect_local)
                .pointer_leaf_infos
                .iter()
                .any(|(path, info)| {
                    path.is_empty() && info.address_space == AddressSpaceKind::Storage
                }),
            "field-backed init effect locals must carry a storage root pointer leaf; local infos={:?}, store base origin={:?}, store base repr={:?}, place space={:?}",
            init_handler.body.local(effect_local).pointer_leaf_infos,
            init_handler.body.value(store.base).origin,
            init_handler.body.value(store.base).repr,
            crate::ir::try_place_address_space_in(
                &init_handler.body.values,
                &init_handler.body.locals,
                store,
            ),
        );
        assert_eq!(
            init_handler.body.place_address_space(store),
            AddressSpaceKind::Storage,
            "init handler stores through field-backed effect locals must target storage",
        );
    }

    #[test]
    fn runtime_shapes_preserve_pointer_effect_params() {
        let mut db = DriverDataBase::default();
        let module = lower_inline_module(
            &mut db,
            "file:///runtime_shapes_preserve_pointer_effect_params.fe",
            include_str!("../../codegen/tests/fixtures/pointer_field_aggregate.fe"),
        );

        assert!(
            module.functions.iter().any(|func| {
                func.symbol_name.contains("bump")
                    && func
                        .body
                        .param_locals
                        .iter()
                        .chain(func.body.effect_param_locals.iter())
                        .copied()
                        .any(|local| {
                            matches!(
                                func.body.local(local).runtime_shape,
                                RuntimeShape::MemoryPtr { target_ty: Some(_) }
                            )
                        })
            }),
            "expected at least one memory-specialized bump helper with a typed memory-pointer runtime param",
        );
    }

    #[test]
    fn free_type_key_effect_params_stay_memory_backed() {
        let mut db = DriverDataBase::default();
        let module = lower_inline_module(
            &mut db,
            "file:///mutable_array_args_and_effects.fe",
            include_str!("../../fe/tests/fixtures/fe_test/mutable_array_args_and_effects.fe"),
        );

        for func_name in [
            "write_effect__MemPtr__u256__4__",
            "effect_sum__MemPtr__u256__4__",
            "test_mut_array_effect_param__MemPtr__u256__4__",
        ] {
            let func = module
                .functions
                .iter()
                .find(|func| func.symbol_name.contains(func_name))
                .unwrap_or_else(|| {
                    panic!("expected memory-specialized function matching `{func_name}`")
                });
            let effect_local = func.body.local(func.body.effect_param_locals[0]);
            assert_eq!(
                effect_local.address_space,
                AddressSpaceKind::Memory,
                "free-function type-key effect params must stay memory-backed in `{}`",
                func.symbol_name,
            );
            assert!(
                matches!(
                    effect_local.runtime_shape,
                    RuntimeShape::MemoryPtr { target_ty: Some(_) } | RuntimeShape::ObjectRef { .. }
                ),
                "free-function type-key effect params must not default to storage in `{}`; got {:?}",
                func.symbol_name,
                effect_local.runtime_shape,
            );
            assert!(
                effect_local.pointer_leaf_infos.iter().any(|(path, info)| {
                    path.is_empty() && info.address_space == AddressSpaceKind::Memory
                }),
                "free-function type-key effect params must carry a memory root pointer leaf in `{}`; got {:?}",
                func.symbol_name,
                effect_local.pointer_leaf_infos,
            );
        }
    }

    #[test]
    fn seq_helpers_from_memory_effect_array_keep_memory_ptr_receivers() {
        let mut db = DriverDataBase::default();
        let module = lower_inline_module(
            &mut db,
            "file:///mutable_array_args_and_effects.fe",
            include_str!("../../fe/tests/fixtures/fe_test/mutable_array_args_and_effects.fe"),
        );

        let seq_funcs = module
            .functions
            .iter()
            .filter(|func| func.symbol_name.contains("get") || func.symbol_name.contains("len"))
            .collect::<Vec<_>>();

        assert!(
            !seq_funcs.is_empty(),
            "expected monomorphized `len`/`get` helpers in fixture"
        );

        assert!(
            seq_funcs.iter().all(|func| {
                matches!(
                    func.body.local(func.body.param_locals[0]).runtime_shape,
                    RuntimeShape::MemoryPtr { target_ty: Some(_) } | RuntimeShape::Erased
                )
            }),
            "expected memory-pointer-specialized `len`/`get` helpers, got: {:?}",
            seq_funcs
                .iter()
                .map(|func| (
                    func.symbol_name.as_str(),
                    func.body.local(func.body.param_locals[0]).runtime_shape,
                    func.body.local(func.body.param_locals[0]).place_root_layout,
                    func.body
                        .local(func.body.param_locals[0])
                        .pointer_leaf_infos
                        .clone(),
                ))
                .collect::<Vec<_>>(),
        );
        assert!(
            seq_funcs.iter().all(|func| {
                func.body.values.iter().all(|value| {
                    let root_param = func.body.param_locals[0];
                    match value.origin {
                        ValueOrigin::Local(local) | ValueOrigin::PlaceRoot(local)
                            if local == root_param =>
                        {
                            matches!(
                                value.runtime_shape,
                                RuntimeShape::MemoryPtr { target_ty: Some(_) }
                            )
                        }
                        _ => true,
                    }
                })
            }),
            "expected root receiver values in memory-specialized `len`/`get` helpers to stay MemoryPtr, got: {:?}",
            seq_funcs
                .iter()
                .map(|func| (
                    func.symbol_name.as_str(),
                    func.body
                        .values
                        .iter()
                        .filter_map(|value| match value.origin {
                            ValueOrigin::Local(local) | ValueOrigin::PlaceRoot(local)
                                if local == func.body.param_locals[0] =>
                            {
                                Some((value.origin.clone(), value.runtime_shape))
                            }
                            _ => None,
                        })
                        .collect::<Vec<_>>(),
                ))
                .collect::<Vec<_>>(),
        );
    }

    #[test]
    fn memory_effect_array_place_refs_do_not_repromote_to_object_refs() {
        let mut db = DriverDataBase::default();
        let module = lower_inline_module(
            &mut db,
            "file:///mutable_array_args_and_effects.fe",
            include_str!("../../fe/tests/fixtures/fe_test/mutable_array_args_and_effects.fe"),
        );

        let get_helper = module
            .functions
            .iter()
            .find(|func| {
                func.symbol_name.contains("seq") && func.symbol_name.contains("get_arg0_root_mem")
            })
            .expect("expected memory-specialized seq::get helper");
        let param_local = get_helper.body.param_locals[0];

        let mismatches = get_helper
            .body
            .values
            .iter()
            .filter_map(|value| match &value.origin {
                ValueOrigin::PlaceRef(place) | ValueOrigin::MoveOut { place }
                    if matches!(
                        get_helper.body.value(place.base).origin,
                        ValueOrigin::Local(local) | ValueOrigin::PlaceRoot(local) if local == param_local
                    ) =>
                {
                    Some((value.origin.clone(), value.runtime_shape, value.pointer_info))
                }
                _ => None,
            })
            .filter(|(_, shape, _)| {
                !matches!(shape, RuntimeShape::MemoryPtr { target_ty: Some(_) })
            })
            .collect::<Vec<_>>();

        assert!(
            mismatches.is_empty(),
            "memory-specialized seq::get helper place refs should stay memory-backed, got {mismatches:?}",
        );

        let root_related_values = get_helper
            .body
            .values
            .iter()
            .filter_map(|value| match &value.origin {
                ValueOrigin::Local(local) | ValueOrigin::PlaceRoot(local) if *local == param_local => {
                    Some((value.origin.clone(), value.repr, value.runtime_shape, value.pointer_info))
                }
                ValueOrigin::PlaceRef(place) | ValueOrigin::MoveOut { place }
                    if matches!(
                        get_helper.body.value(place.base).origin,
                        ValueOrigin::Local(local) | ValueOrigin::PlaceRoot(local) if local == param_local
                    ) =>
                {
                    Some((value.origin.clone(), value.repr, value.runtime_shape, value.pointer_info))
                }
                ValueOrigin::TransparentCast { value: inner }
                    if matches!(
                        get_helper.body.value(*inner).origin,
                        ValueOrigin::Local(local) | ValueOrigin::PlaceRoot(local) if local == param_local
                    ) =>
                {
                    Some((value.origin.clone(), value.repr, value.runtime_shape, value.pointer_info))
                }
                _ => None,
            })
            .collect::<Vec<_>>();
        assert!(
            root_related_values.iter().all(|(_, _, shape, _)| {
                matches!(shape, RuntimeShape::MemoryPtr { target_ty: Some(_) })
            }),
            "all root-related seq::get helper values should stay memory-backed, got {root_related_values:?}",
        );
    }

    #[test]
    fn mut_array_forwarded_return_keeps_memory_ptr_runtime_shape() {
        let mut db = DriverDataBase::default();
        let module = lower_inline_module(
            &mut db,
            "file:///mutable_array_args_and_effects.fe",
            include_str!("../../fe/tests/fixtures/fe_test/mutable_array_args_and_effects.fe"),
        );

        let rewrite = module
            .functions
            .iter()
            .find(|func| func.symbol_name == "rewrite")
            .expect("expected `rewrite` function in fixture");

        assert!(
            matches!(
                rewrite
                    .body
                    .local(rewrite.body.param_locals[0])
                    .runtime_shape,
                RuntimeShape::MemoryPtr { target_ty: Some(_) }
            ),
            "expected forwarded mut-array param to stay memory-backed, got {:?}",
            rewrite
                .body
                .local(rewrite.body.param_locals[0])
                .runtime_shape,
        );
        assert!(
            matches!(
                rewrite.runtime_return_shape,
                RuntimeShape::MemoryPtr { target_ty: Some(_) }
            ),
            "expected forwarded mut-array return to stay memory-backed, got {:?}",
            rewrite.runtime_return_shape,
        );

        let specialized_rewrite = module
            .functions
            .iter()
            .find(|func| func.symbol_name == "rewrite_arg0_root_mem")
            .expect("expected memory-specialized rewrite helper");
        let specialized_param = specialized_rewrite
            .body
            .local(specialized_rewrite.body.param_locals[0]);
        assert!(
            matches!(
                specialized_param.place_root_layout,
                LocalPlaceRootLayout::MemorySlot
            ),
            "expected memory-specialized rewrite param to use a memory slot root, got layout={:?}, address_space={:?}, runtime_shape={:?}, pointer_leaf_infos={:?}",
            specialized_param.place_root_layout,
            specialized_param.address_space,
            specialized_param.runtime_shape,
            specialized_param.pointer_leaf_infos,
        );
        assert!(
            matches!(
                specialized_param.runtime_shape,
                RuntimeShape::MemoryPtr { target_ty: Some(_) }
            ),
            "expected memory-specialized rewrite param to stay memory-backed, got {:?}",
            specialized_param.runtime_shape,
        );
    }

    #[test]
    fn mutating_array_call_results_do_not_stay_const_backed() {
        let mut db = DriverDataBase::default();
        let module = lower_inline_module(
            &mut db,
            "file:///mutable_array_args_and_effects.fe",
            include_str!("../../fe/tests/fixtures/fe_test/mutable_array_args_and_effects.fe"),
        );

        let test_fn = module
            .functions
            .iter()
            .find(|func| func.symbol_name == "test_mut_array_fn_argument")
            .expect("expected mut-array argument test helper");
        let rewrite_call = test_fn
            .body
            .blocks
            .iter()
            .flat_map(|block| block.insts.iter())
            .find_map(|inst| match inst {
                MirInst::Assign {
                    rvalue: Rvalue::Call(call),
                    ..
                } => Some(call),
                _ => None,
            })
            .expect("expected rewrite call in mut-array argument test");
        let updated = test_fn
            .body
            .locals
            .iter()
            .find(|local| local.name == "updated")
            .expect("expected updated local");

        assert!(
            !rewrite_call
                .resolved_name
                .as_deref()
                .is_some_and(|name| name.contains("root_code")),
            "mutating helper calls must not specialize through code-backed return forwarding, got {:?}",
            rewrite_call.resolved_name,
        );
        assert!(
            !matches!(updated.runtime_shape, RuntimeShape::ConstRef { .. }),
            "mutating helper results must not stay const-backed, got {:?}",
            updated.runtime_shape,
        );
        assert!(
            matches!(
                updated.runtime_shape,
                RuntimeShape::MemoryPtr { target_ty: Some(_) }
            ),
            "mutating helper results should preserve memory-backed carriers, got runtime_shape={:?}, address_space={:?}, place_root_layout={:?}, pointer_leaf_infos={:?}, const_backing={:?}",
            updated.runtime_shape,
            updated.address_space,
            updated.place_root_layout,
            updated.pointer_leaf_infos,
            updated.const_backing,
        );
        assert!(
            !matches!(updated.const_backing, crate::ir::LocalConstBacking::Const),
            "mutating helper results must not keep const backing, got {:?}",
            updated.const_backing,
        );
    }

    #[test]
    fn provider_backed_scalar_handle_locals_stay_memory_ptrs() {
        let mut db = DriverDataBase::default();
        let module = lower_inline_module(
            &mut db,
            "file:///borrow_storage_field_handle.fe",
            include_str!("../../codegen/tests/fixtures/borrow_storage_field_handle.fe"),
        );

        let func = module
            .functions
            .iter()
            .find(|func| func.symbol_name.contains("borrow_storage_field_handle"))
            .expect("expected borrow_storage_field_handle helper");
        let p_local = func
            .body
            .locals
            .iter()
            .find(|local| local.name == "p")
            .expect("expected borrowed field-handle local `p`");

        assert!(
            matches!(
                p_local.runtime_shape,
                RuntimeShape::MemoryPtr { target_ty: Some(_) }
            ),
            "provider-backed scalar handle local should stay memory-pointer-backed, got {:?}",
            p_local.runtime_shape,
        );
        assert!(
            matches!(
                p_local.pointer_leaf_infos.as_slice(),
                [(
                    path,
                    PointerInfo {
                        address_space: AddressSpaceKind::Memory,
                        target_ty: Some(_),
                    },
                )] if path.is_empty()
            ),
            "provider-backed scalar handle local should keep a rooted memory pointer leaf, got {:?}",
            p_local.pointer_leaf_infos,
        );
    }

    #[test]
    fn nested_const_struct_field_reads_lower_without_projection_index_panics() {
        let mut db = DriverDataBase::default();
        let module = lower_inline_module(
            &mut db,
            "file:///quinary_nested_field_read.fe",
            include_str!("../../fe/tests/fixtures/fe_test/quinary_nested_field_read.fe"),
        );

        let test_fn = module
            .functions
            .iter()
            .find(|func| func.symbol_name == "test_read_nested_field_directly")
            .expect("expected quinary nested-field read test helper");
        let load_place = test_fn
            .body
            .blocks
            .iter()
            .flat_map(|block| block.insts.iter())
            .find_map(|inst| match inst {
                MirInst::Assign {
                    rvalue: Rvalue::Load { place },
                    ..
                } => Some(place),
                _ => None,
            })
            .expect("expected scalar load from nested const struct field");

        assert_eq!(
            test_fn.body.place_address_space(load_place),
            AddressSpaceKind::Code,
            "nested reads from immutable const-backed struct fields should stay code-backed",
        );
    }

    #[test]
    fn contract_field_type_key_effect_params_stay_storage_backed() {
        let mut db = DriverDataBase::default();
        let module = lower_inline_module(
            &mut db,
            "file:///contract_field_array_effect.fe",
            r#"
msg Msg {
    #[selector = 1]
    Sum -> u256,
}

pub contract C {
    mut words: [u256; 4],

    init() uses (mut words) {
        words[0] = 1
        words[1] = 2
        words[2] = 3
        words[3] = 4
    }

    recv Msg {
        Sum -> u256 uses (words) {
            words[0] + words[1] + words[2] + words[3]
        }
    }
}
"#,
        );
        let func = module
            .functions
            .iter()
            .find(|func| func.symbol_name.starts_with("__C_recv_"))
            .unwrap_or_else(|| {
                panic!(
                    "expected contract handler, got {:?}",
                    module
                        .functions
                        .iter()
                        .map(|func| func.symbol_name.as_str())
                        .collect::<Vec<_>>()
                )
            });
        let effect_local = func.body.local(func.body.effect_param_locals[0]);
        assert_eq!(
            effect_local.address_space,
            AddressSpaceKind::Storage,
            "contract-field type-key effects must stay storage-backed in `{}`",
            func.symbol_name,
        );
        assert!(
            effect_local.pointer_leaf_infos.iter().any(|(path, info)| {
                path.is_empty() && info.address_space == AddressSpaceKind::Storage
            }),
            "contract-field type-key effects must carry a storage root pointer leaf in `{}`; got {:?}",
            func.symbol_name,
            effect_local.pointer_leaf_infos,
        );
    }

    #[test]
    fn runtime_shapes_preserve_pointer_return_shapes() {
        let mut db = DriverDataBase::default();
        let module = lower_inline_module(
            &mut db,
            "file:///runtime_shapes_preserve_pointer_return_shapes.fe",
            include_str!("../../codegen/tests/fixtures/effect_handle_field_deref.fe"),
        );

        let extract_ptr = module
            .functions
            .iter()
            .find(|func| func.symbol_name == "extract_ptr")
            .expect("expected extract_ptr helper");
        let target_ty = match extract_ptr.runtime_return_shape {
            RuntimeShape::AddressWord(PointerInfo {
                address_space: AddressSpaceKind::Storage,
                target_ty,
            }) => target_ty,
            other => panic!("unexpected runtime return shape: {other:?}"),
        };
        assert!(
            target_ty.is_some(),
            "pointer return shape should retain pointee target metadata",
        );
    }

    #[test]
    fn runtime_shapes_resolve_storage_capability_return_shapes() {
        let mut db = DriverDataBase::default();
        let module = lower_inline_module(
            &mut db,
            "file:///runtime_shapes_resolve_storage_capability_return_shapes.fe",
            include_str!("../../fe/tests/fixtures/fe_test/mut_self_storage_receiver_regression.fe"),
        );

        let value_mut = module
            .functions
            .iter()
            .find(|func| func.symbol_name.contains("value_mut_stor_arg0_root_stor"))
            .expect("expected storage-specialized value_mut helper");
        let target_ty = match value_mut.runtime_return_shape {
            RuntimeShape::AddressWord(PointerInfo {
                address_space: AddressSpaceKind::Storage,
                target_ty,
            }) => target_ty,
            other => panic!("unexpected runtime return shape: {other:?}"),
        };
        assert!(
            target_ty.is_some(),
            "storage-backed capability returns must resolve from real return values and retain pointee metadata",
        );
    }

    #[test]
    fn runtime_shapes_recompute_storage_backed_wrapper_locals() {
        let mut db = DriverDataBase::default();
        let module = lower_inline_module(
            &mut db,
            "file:///runtime_shapes_recompute_storage_backed_wrapper_locals.fe",
            include_str!("../../codegen/tests/fixtures/effect_handle_field_deref.fe"),
        );

        let recv = module
            .functions
            .iter()
            .find(|func| func.symbol_name == "__EffectHandleFieldDeref_recv_0_2")
            .expect("expected BumpMover receiver helper");
        let mover_local = recv
            .body
            .locals
            .iter()
            .find(|local| local.name == "mover")
            .expect("expected `mover` wrapper local");

        let target_ty = match mover_local.runtime_shape {
            RuntimeShape::AddressWord(PointerInfo {
                address_space: AddressSpaceKind::Storage,
                target_ty,
            }) => target_ty,
            other => panic!("unexpected wrapper local runtime shape: {other:?}"),
        };
        assert!(
            target_ty.is_some(),
            "storage-backed transparent wrapper locals must normalize to a storage address word with pointee metadata",
        );
    }

    #[test]
    fn runtime_shapes_preserve_alloc_backed_aug_assign_temporaries() {
        let mut db = DriverDataBase::default();
        let module = lower_inline_module(
            &mut db,
            "file:///runtime_shapes_preserve_alloc_backed_aug_assign_temporaries.fe",
            include_str!("../../fe/tests/fixtures/fe_test/aug_assign_traits.fe"),
        );

        let func = module
            .functions
            .iter()
            .find(|func| func.symbol_name == "add_assign_without_add_trait")
            .expect("expected add_assign_without_add_trait helper");
        let temp_local = func
            .body
            .locals
            .iter()
            .find(|local| local.name.starts_with("tmp_aug_assign_self"))
            .expect("expected aug-assign temp place local");

        let target_ty = match temp_local.runtime_shape {
            RuntimeShape::ObjectRef { target_ty } => Some(target_ty),
            other => panic!("unexpected aug-assign temp runtime shape: {other:?}"),
        };
        assert!(
            target_ty.is_some(),
            "alloc-backed aug-assign temp locals must normalize as typed object refs",
        );
        let LocalPlaceRootLayout::ObjectRootValue { target_ty, source } =
            temp_local.place_root_layout
        else {
            panic!(
                "alloc-backed temporaries should carry pass-through object-root provenance, got {:?}",
                temp_local.place_root_layout,
            );
        };
        assert_eq!(source, ObjectRootSource::AllocatedMemory);
        assert_eq!(
            temp_local.runtime_shape,
            RuntimeShape::ObjectRef { target_ty },
            "alloc-backed temporaries should carry object-root provenance consistent with their object runtime shape",
        );
    }

    #[test]
    fn runtime_shapes_refine_object_backed_scalar_handles_to_object_refs() {
        let mut db = DriverDataBase::default();
        let module = lower_inline_module(
            &mut db,
            "file:///runtime_shapes_refine_object_backed_scalar_handles_to_object_refs.fe",
            include_str!("../../fe/tests/fixtures/fe_test/borrow_handle_forwarding.fe"),
        );

        let read_balance = module
            .functions
            .iter()
            .find(|func| func.symbol_name == "read_balance")
            .expect("expected read_balance helper");
        let test_func = module
            .functions
            .iter()
            .find(|func| func.symbol_name == "test_borrow_handle_forwarding")
            .expect("expected borrow_handle_forwarding test");
        let balance_ref = test_func
            .body
            .locals
            .iter()
            .find(|local| local.name == "b_ref")
            .expect("expected b_ref local");
        let balance_ref_idx = test_func
            .body
            .locals
            .iter()
            .position(|local| local.name == "b_ref")
            .expect("expected b_ref local index");
        assert!(
            matches!(balance_ref.runtime_shape, RuntimeShape::ObjectRef { .. }),
            "b_ref local v{balance_ref_idx} should normalize to an object-backed scalar handle, got {:?}",
            balance_ref.runtime_shape,
        );
        let balance_ref_source = test_func
            .body
            .blocks
            .iter()
            .flat_map(|block| block.insts.iter())
            .find_map(|inst| match inst {
                MirInst::Assign {
                    dest,
                    rvalue: Rvalue::Value(value),
                    ..
                } if *dest == Some(LocalId(balance_ref_idx as u32)) => Some(*value),
                _ => None,
            })
            .expect("expected b_ref assignment source");
        let balance_ref_source_data = test_func.body.value(balance_ref_source);
        assert!(
            matches!(balance_ref_source_data.origin, ValueOrigin::PlaceRef(_)),
            "b_ref should come from an explicit borrow, got {:?}",
            balance_ref_source_data.origin,
        );
        assert!(
            matches!(
                balance_ref_source_data.runtime_shape,
                RuntimeShape::ObjectRef { .. }
            ),
            "explicit scalar borrows from object-backed aggregates should normalize to object refs, got {:?}",
            balance_ref_source_data.runtime_shape,
        );
        let (read_balance_name, call_arg) = test_func
            .body
            .blocks
            .iter()
            .flat_map(|block| block.insts.iter())
            .find_map(|inst| match inst {
                MirInst::Assign {
                    rvalue: Rvalue::Call(call),
                    ..
                } if call
                    .resolved_name
                    .as_deref()
                    .is_some_and(|name| name.contains("read_balance")) =>
                {
                    call.args.first().copied().zip(call.resolved_name.clone())
                }
                _ => None,
            })
            .map(|(arg, name)| (name, arg))
            .expect("expected read_balance call arg");
        let call_arg_origin = &test_func.body.value(call_arg).origin;
        if let ValueOrigin::Local(local) = call_arg_origin {
            assert_eq!(
                local.index(),
                balance_ref_idx,
                "read_balance should receive the `b_ref` local directly",
            );
        }
        let call_arg_shape = test_func.body.value(call_arg).runtime_shape;
        assert!(
            matches!(call_arg_shape, RuntimeShape::ObjectRef { .. }),
            "read_balance call arg should stay object-backed, got {:?} from {:?}",
            call_arg_shape,
            call_arg_origin,
        );

        let read_balance = module
            .functions
            .iter()
            .find(|func| func.symbol_name == read_balance_name)
            .unwrap_or(read_balance);
        let read_balance_param = read_balance.body.local(read_balance.body.param_locals[0]);
        assert!(
            matches!(
                read_balance_param.runtime_shape,
                RuntimeShape::ObjectRef { .. }
            ),
            "object-backed scalar refs should stay object-backed in helpers, got {:?}",
            read_balance_param.runtime_shape,
        );

        let bump_nonce_name = test_func
            .body
            .blocks
            .iter()
            .flat_map(|block| block.insts.iter())
            .find_map(|inst| match inst {
                MirInst::Assign {
                    rvalue: Rvalue::Call(call),
                    ..
                } if call
                    .resolved_name
                    .as_deref()
                    .is_some_and(|name| name.contains("bump_nonce")) =>
                {
                    call.resolved_name.clone()
                }
                _ => None,
            })
            .expect("expected bump_nonce call");
        let bump_nonce = module
            .functions
            .iter()
            .find(|func| func.symbol_name == bump_nonce_name)
            .expect("expected bump_nonce helper");
        let bump_nonce_param = bump_nonce.body.local(bump_nonce.body.param_locals[0]);
        assert!(
            matches!(
                bump_nonce_param.runtime_shape,
                RuntimeShape::ObjectRef { .. }
            ),
            "object-backed scalar mut handles should stay object-backed in helpers, got {:?}",
            bump_nonce_param.runtime_shape,
        );
    }

    #[test]
    fn runtime_shapes_keep_readonly_array_params_const_backed_across_private_calls() {
        let mut db = DriverDataBase::default();
        let module = lower_inline_module(
            &mut db,
            "file:///runtime_shapes_keep_readonly_array_params_const_backed_across_private_calls.fe",
            r#"
fn head(values: [u256; 4], i: usize) -> u256 {
    return values[i]
}

fn read(i: usize) -> u256 {
    let x: [u256; 4] = [1, 2, 3, 4]
    return head(x, i)
}
"#,
        );

        let head = module
            .functions
            .iter()
            .find(|func| func.symbol_name == "head")
            .expect("expected head helper");
        let values_param = head.body.local(head.body.param_locals[0]);
        assert!(
            matches!(values_param.runtime_shape, RuntimeShape::ConstRef { .. }),
            "readonly aggregate helper params should stay const-backed across private calls, got {:?}",
            values_param.runtime_shape,
        );
        assert!(
            matches!(
                values_param.const_backing,
                crate::ir::LocalConstBacking::Const
            ),
            "readonly aggregate helper params should keep const backing, got {:?}",
            values_param.const_backing,
        );
        assert_eq!(
            values_param.address_space,
            AddressSpaceKind::Code,
            "readonly aggregate helper params should stay in code space",
        );
    }

    #[test]
    fn runtime_shapes_keep_readonly_array_returns_const_backed_across_private_calls() {
        let mut db = DriverDataBase::default();
        let module = lower_inline_module(
            &mut db,
            "file:///runtime_shapes_keep_readonly_array_returns_const_backed_across_private_calls.fe",
            r#"
fn id(values: [u256; 4]) -> [u256; 4] {
    return values
}

fn read() -> u256 {
    let x: [u256; 4] = [1, 2, 3, 4]
    let y: [u256; 4] = id(x)
    let z: [u256; 4] = y
    return z[0]
}
"#,
        );

        let id = module
            .functions
            .iter()
            .find(|func| func.symbol_name == "id")
            .expect("expected id helper");
        assert!(
            matches!(id.runtime_return_shape, RuntimeShape::ConstRef { .. }),
            "readonly aggregate helper returns should stay const-backed across private calls, got {:?}",
            id.runtime_return_shape,
        );

        let read = module
            .functions
            .iter()
            .find(|func| func.symbol_name == "read")
            .expect("expected read helper");
        let y_local = read
            .body
            .locals
            .iter()
            .find(|local| local.name == "y")
            .expect("expected y local");
        assert!(
            matches!(y_local.runtime_shape, RuntimeShape::ConstRef { .. }),
            "readonly aggregate locals receiving readonly call results should stay const-backed, got {:?}",
            y_local.runtime_shape,
        );
        assert!(
            matches!(y_local.const_backing, crate::ir::LocalConstBacking::Const),
            "readonly aggregate locals receiving readonly call results should keep const backing, got {:?}",
            y_local.const_backing,
        );
        assert_eq!(
            y_local.address_space,
            AddressSpaceKind::Code,
            "readonly aggregate locals receiving readonly call results should stay in code space",
        );

        let z_local = read
            .body
            .locals
            .iter()
            .find(|local| local.name == "z")
            .expect("expected z local");
        assert!(
            matches!(z_local.runtime_shape, RuntimeShape::ConstRef { .. }),
            "readonly aggregate copy locals should stay const-backed, got {:?}",
            z_local.runtime_shape,
        );
        assert!(
            matches!(z_local.const_backing, crate::ir::LocalConstBacking::Const),
            "readonly aggregate copy locals should inherit const backing, got {:?}",
            z_local.const_backing,
        );
        assert_eq!(
            z_local.address_space,
            AddressSpaceKind::Code,
            "readonly aggregate copy locals should stay in code space",
        );
    }

    #[test]
    fn runtime_shapes_materialize_mutable_const_array_locals() {
        let mut db = DriverDataBase::default();
        let module = lower_inline_module(
            &mut db,
            "file:///runtime_shapes_materialize_mutable_const_array_locals.fe",
            r#"
fn read() -> u256 {
    let mut x: [u256; 4] = [1, 2, 3, 4]
    x[1] = 9
    return x[1]
}
"#,
        );

        let read = module
            .functions
            .iter()
            .find(|func| func.symbol_name == "read")
            .expect("expected read helper");
        let x_local = read
            .body
            .locals
            .iter()
            .find(|local| local.name == "x")
            .expect("expected x local");

        assert!(x_local.is_mut, "expected mutable local");
        assert_eq!(
            x_local.address_space,
            AddressSpaceKind::Memory,
            "mutable aggregate locals must materialize into memory-backed runtime carriers",
        );
        assert!(
            matches!(x_local.runtime_shape, RuntimeShape::ObjectRef { .. }),
            "mutable aggregate locals must not remain const-backed, got {:?}",
            x_local.runtime_shape,
        );
        assert!(
            matches!(x_local.const_backing, crate::ir::LocalConstBacking::Runtime),
            "mutable aggregate locals must clear const backing after materialization, got {:?}",
            x_local.const_backing,
        );
    }

    #[test]
    fn runtime_shapes_materialize_mutable_const_struct_wrapper_locals() {
        let mut db = DriverDataBase::default();
        let module = lower_inline_module(
            &mut db,
            "file:///runtime_shapes_materialize_mutable_const_struct_wrapper_locals.fe",
            r#"
const ROW5: [u256; 5] = [0, 0, 0, 0, 0]
const GRID5: [[u256; 5]; 3] = [ROW5, ROW5, ROW5]

pub struct Data {
    pub values: [[u256; 5]; 3],
}

fn read() -> u256 {
    let mut data = Data { values: GRID5 }
    data.values[1][2] = 7
    data.values[1][2]
}
"#,
        );

        let read = module
            .functions
            .iter()
            .find(|func| func.symbol_name == "read")
            .expect("expected read helper");
        let data_local = read
            .body
            .locals
            .iter()
            .find(|local| local.name == "data")
            .expect("expected data local");

        assert!(data_local.is_mut, "expected mutable local");
        assert_eq!(
            data_local.address_space,
            AddressSpaceKind::Memory,
            "mutable transparent struct-wrapper locals must materialize into memory-backed runtime carriers; shape={:?} const_backing={:?} place_root_layout={:?}",
            data_local.runtime_shape,
            data_local.const_backing,
            data_local.place_root_layout,
        );
        assert!(
            matches!(data_local.runtime_shape, RuntimeShape::ObjectRef { .. }),
            "mutable transparent struct-wrapper locals must not remain const-backed, got {:?}",
            data_local.runtime_shape,
        );
        assert!(
            matches!(
                data_local.const_backing,
                crate::ir::LocalConstBacking::Runtime
            ),
            "mutable transparent struct-wrapper locals must clear const backing after materialization, got {:?}",
            data_local.const_backing,
        );
    }

    #[test]
    fn nested_mutation_stores_do_not_keep_code_backed_row_metadata() {
        let mut db = DriverDataBase::default();
        let module = lower_inline_module(
            &mut db,
            "file:///nested_mutation_stores_do_not_keep_code_backed_row_metadata.fe",
            r#"
const ROW5: [u256; 5] = [0, 0, 0, 0, 0]
const GRID5: [[u256; 5]; 3] = [ROW5, ROW5, ROW5]

pub struct Data {
    pub values: [[u256; 5]; 3],
}

#[test]
fn test_mutate_field_then_pass_nested_field() {
    let mut data = Data { values: GRID5 }
    data.values[1][2] = 7
    assert(data.values[1][2] == 7)
}
"#,
        );

        let test_fn = module
            .functions
            .iter()
            .find(|func| func.symbol_name == "test_mutate_field_then_pass_nested_field")
            .expect("expected nested-field mutation test helper");
        let store = test_fn
            .body
            .blocks
            .iter()
            .flat_map(|block| block.insts.iter())
            .find_map(|inst| match inst {
                MirInst::Store { place, .. } => Some(place),
                _ => None,
            })
            .expect("expected nested-field store");

        assert_eq!(
            test_fn.body.place_address_space(store),
            AddressSpaceKind::Memory,
            "nested scalar stores into runtime-backed transparent wrappers must not retain stale code-backed row metadata; store={store:?}, locals={:?}",
            test_fn.body.locals,
        );
    }

    #[test]
    fn nested_mutation_call_params_do_not_specialize_back_to_code() {
        let mut db = DriverDataBase::default();
        let module = lower_inline_module(
            &mut db,
            "file:///nested_mutation_call_params_do_not_specialize_back_to_code.fe",
            r#"
const ROW5: [u256; 5] = [0, 0, 0, 0, 0]
const GRID5: [[u256; 5]; 3] = [ROW5, ROW5, ROW5]

pub struct Data {
    pub values: [[u256; 5]; 3],
}

impl Data {
    pub fn empty() -> Self {
        Data { values: GRID5 }
    }

    pub fn set(mut self, row: usize, col: usize, value: u256) {
        self.values[row][col] = value
    }
}

fn first(values: [[u256; 5]; 3]) -> u256 {
    values[1][2]
}

#[test]
fn test_mutate_then_pass_nested_field() {
    let mut data = Data::empty()
    data.set(1, 2, 7)
    assert(first(data.values) == 7)
}
"#,
        );

        let caller = module
            .functions
            .iter()
            .find(|func| func.symbol_name == "test_mutate_then_pass_nested_field")
            .expect("expected mutation caller")
            .clone();
        let caller_body = caller.body.clone();
        let (call_arg, callee_name) = caller_body
            .blocks
            .iter()
            .flat_map(|block| block.insts.iter())
            .find_map(|inst| match inst {
                MirInst::Assign {
                    rvalue: Rvalue::Call(call),
                    ..
                } if call
                    .resolved_name
                    .as_deref()
                    .is_some_and(|name| name.starts_with("first")) =>
                {
                    call.args.first().copied().zip(call.resolved_name.clone())
                }
                _ => None,
            })
            .expect("expected first call arg");
        let call_arg_shape = caller_body.value(call_arg).runtime_shape;
        assert!(
            matches!(
                caller_body.value(call_arg).origin,
                ValueOrigin::TransparentCast { .. }
            ),
            "mutated nested-array helper calls should lower transparent wrapper field access through a TransparentCast, got {:?}",
            caller_body.value(call_arg).origin,
        );
        assert!(
            matches!(call_arg_shape, RuntimeShape::ObjectRef { .. }),
            "mutated nested-array helper calls must pass runtime-backed array objects, got {:?}; origin={:?}; ptr={:?}",
            call_arg_shape,
            caller_body.value(call_arg).origin,
            caller_body.value_pointer_info(call_arg),
        );
        let first_body = module
            .functions
            .iter()
            .find(|func| func.symbol_name == callee_name)
            .expect("expected specialized first helper")
            .body
            .clone();
        let param_base = first_body
            .values
            .iter()
            .enumerate()
            .find_map(|(idx, value)| match value.origin {
                ValueOrigin::Local(local) | ValueOrigin::PlaceRoot(local)
                    if local == first_body.param_locals[0] =>
                {
                    Some(ValueId(idx as u32))
                }
                _ => None,
            })
            .expect("expected value rooted at first's param");
        let row_path = MirProjectionPath::from_projection(Projection::Index(
            hir::projection::IndexSource::Constant(1),
        ));
        let elem_path = row_path.concat(&MirProjectionPath::from_projection(Projection::Index(
            hir::projection::IndexSource::Constant(2),
        )));

        assert_eq!(
            first_body.place_address_space(&Place::new(param_base, row_path.clone())),
            AddressSpaceKind::Memory,
            "mutated nested-array rows passed through helpers must stay memory-backed; param={:?}",
            first_body.local(first_body.param_locals[0]),
        );
        assert_eq!(
            first_body.place_address_space(&Place::new(param_base, elem_path)),
            AddressSpaceKind::Memory,
            "mutated nested-array element loads in helpers must not resolve through code; param={:?}",
            first_body.local(first_body.param_locals[0]),
        );
    }

    #[test]
    fn direct_nested_mutation_call_params_do_not_specialize_back_to_code() {
        let mut db = DriverDataBase::default();
        let module = lower_inline_module(
            &mut db,
            "file:///direct_nested_mutation_call_params_do_not_specialize_back_to_code.fe",
            r#"
const ROW5: [u256; 5] = [0, 0, 0, 0, 0]
const GRID5: [[u256; 5]; 3] = [ROW5, ROW5, ROW5]

pub struct Data {
    pub values: [[u256; 5]; 3],
}

fn first(values: [[u256; 5]; 3]) -> u256 {
    values[1][2]
}

#[test]
fn test_mutate_field_then_pass_nested_field() {
    let mut data = Data { values: GRID5 }
    data.values[1][2] = 7
    assert(first(data.values) == 7)
}
"#,
        );

        let caller = module
            .functions
            .iter()
            .find(|func| func.symbol_name == "test_mutate_field_then_pass_nested_field")
            .expect("expected direct mutation caller")
            .clone();
        let caller_body = caller.body.clone();
        let (call_arg, callee_name) = caller_body
            .blocks
            .iter()
            .flat_map(|block| block.insts.iter())
            .find_map(|inst| match inst {
                MirInst::Assign {
                    rvalue: Rvalue::Call(call),
                    ..
                } if call
                    .resolved_name
                    .as_deref()
                    .is_some_and(|name| name.starts_with("first")) =>
                {
                    call.args.first().copied().zip(call.resolved_name.clone())
                }
                _ => None,
            })
            .expect("expected first call arg");
        let call_arg_shape = caller_body.value(call_arg).runtime_shape;
        assert!(
            matches!(
                caller_body.value(call_arg).origin,
                ValueOrigin::TransparentCast { .. }
            ),
            "direct nested-array helper calls should lower transparent wrapper field access through a TransparentCast, got {:?}",
            caller_body.value(call_arg).origin,
        );
        assert!(
            matches!(call_arg_shape, RuntimeShape::ObjectRef { .. }),
            "direct nested-array helper calls must pass runtime-backed array objects, got {:?}; origin={:?}; ptr={:?}",
            call_arg_shape,
            caller_body.value(call_arg).origin,
            caller_body.value_pointer_info(call_arg),
        );
        let ValueOrigin::TransparentCast { value: inner } = caller_body.value(call_arg).origin
        else {
            unreachable!("checked above");
        };
        assert!(
            matches!(
                caller_body.value(inner).runtime_shape,
                RuntimeShape::ObjectRef { .. }
            ),
            "direct nested-array helper inner value must also stay object-backed, got {:?}; origin={:?}; ptr={:?}",
            caller_body.value(inner).runtime_shape,
            caller_body.value(inner).origin,
            caller_body.value_pointer_info(inner),
        );

        let first_body = module
            .functions
            .iter()
            .find(|func| func.symbol_name == callee_name)
            .expect("expected specialized first helper")
            .body
            .clone();
        let param_base = first_body
            .values
            .iter()
            .enumerate()
            .find_map(|(idx, value)| match value.origin {
                ValueOrigin::Local(local) | ValueOrigin::PlaceRoot(local)
                    if local == first_body.param_locals[0] =>
                {
                    Some(ValueId(idx as u32))
                }
                _ => None,
            })
            .expect("expected value rooted at first's param");
        let row_path = MirProjectionPath::from_projection(Projection::Index(
            hir::projection::IndexSource::Constant(1),
        ));
        let elem_path = row_path.concat(&MirProjectionPath::from_projection(Projection::Index(
            hir::projection::IndexSource::Constant(2),
        )));

        assert_eq!(
            first_body.place_address_space(&Place::new(param_base, row_path.clone())),
            AddressSpaceKind::Memory,
            "direct nested-array rows passed through helpers must stay memory-backed; param={:?}",
            first_body.local(first_body.param_locals[0]),
        );
        assert_eq!(
            first_body.place_address_space(&Place::new(param_base, elem_path)),
            AddressSpaceKind::Memory,
            "direct nested-array element loads in helpers must not resolve through code; param={:?}",
            first_body.local(first_body.param_locals[0]),
        );
    }

    #[test]
    fn runtime_shapes_choose_object_refs_for_mixed_readonly_and_runtime_callers() {
        let mut db = DriverDataBase::default();
        let module = lower_inline_module(
            &mut db,
            "file:///runtime_shapes_choose_object_refs_for_mixed_readonly_and_runtime_callers.fe",
            r#"
fn helper(values: [u256; 4]) -> [u256; 4] {
    let mut out: [u256; 4] = [0, 0, 0, 0]
    for i in 0..4 {
        out[i] = values[i]
    }
    return out
}

fn readonly_caller() -> u256 {
    let values: [u256; 4] = [1, 2, 3, 4]
    let out: [u256; 4] = helper(values)
    return out[0]
}

fn runtime_caller() -> u256 {
    let mut values: [u256; 4] = [1, 2, 3, 4]
    values[0] = 9
    let out: [u256; 4] = helper(values)
    return out[0]
}
"#,
        );

        let helper = module
            .functions
            .iter()
            .find(|func| func.symbol_name == "helper")
            .expect("expected helper function");
        let values_param = helper.body.local(helper.body.param_locals[0]);
        assert!(
            matches!(values_param.runtime_shape, RuntimeShape::ObjectRef { .. }),
            "helpers with mixed readonly/runtime aggregate callers must converge to object refs, got {:?}",
            values_param.runtime_shape,
        );
        assert_eq!(
            values_param.address_space,
            AddressSpaceKind::Memory,
            "mixed readonly/runtime aggregate helper params must not stay code-backed",
        );
        let param_base = helper
            .body
            .values
            .iter()
            .enumerate()
            .find_map(|(idx, value)| match value.origin {
                ValueOrigin::Local(local) | ValueOrigin::PlaceRoot(local)
                    if local == helper.body.param_locals[0] =>
                {
                    Some(ValueId(idx as u32))
                }
                _ => None,
            })
            .expect("expected value rooted at the mixed-shape param");
        let param_loads = [Place::new(
            param_base,
            MirProjectionPath::from_projection(Projection::Index(
                hir::projection::IndexSource::Constant(0),
            )),
        )];
        assert!(
            param_loads
                .iter()
                .all(|place| helper.body.place_address_space(place) == AddressSpaceKind::Memory),
            "mixed readonly/runtime aggregate helper param loads must resolve through memory, got {:?}",
            param_loads
                .iter()
                .map(|place| helper.body.place_address_space(place))
                .collect::<Vec<_>>(),
        );
    }

    #[test]
    fn yul_prep_keeps_msg_nested_array_encode_helpers_memory_backed() {
        let mut db = DriverDataBase::default();
        let url = Url::parse("file:///msg_nested_array_arg.fe").expect("valid test url");
        let file = db.workspace().touch(
            &mut db,
            url,
            Some(
                include_str!("../../fe/tests/fixtures/fe_test/msg_nested_array_arg.fe").to_owned(),
            ),
        );
        let top_mod = db.top_mod(file);
        let mut module = lower_module(&db, top_mod).expect("module should lower");
        super::prepare_module_for_evm_yul_codegen(&db, &mut module);
        let verify_helper = module
            .functions
            .iter()
            .find(|func| {
                func.symbol_name.contains("verify_") && func.symbol_name.contains("encode_to_ptr")
            })
            .expect("expected Verify encode_to_ptr helper");
        let verify_core = super::function_core_lib(&db, verify_helper);
        let verify_param = verify_helper.body.local(verify_helper.body.param_locals[0]);
        assert!(
            !matches!(verify_param.runtime_shape, RuntimeShape::ConstRef { .. }),
            "Verify encode helper param should already be runtime-backed, got space {:?} const {:?} shape {:?}",
            verify_param.address_space,
            verify_param.const_backing,
            verify_param.runtime_shape,
        );
        let verify_caller = module
            .functions
            .iter()
            .find(|func| func.symbol_name.contains("encode_calldata__Verify"))
            .expect("expected encode_calldata__Verify helper");
        let verify_caller_core = super::function_core_lib(&db, verify_caller);
        let verify_arg = verify_caller
            .body
            .blocks
            .iter()
            .flat_map(|block| block.insts.iter())
            .find_map(|inst| match inst {
                MirInst::Assign {
                    rvalue: Rvalue::Call(call),
                    ..
                } if call.resolved_name.as_deref() == Some(verify_helper.symbol_name.as_str()) => {
                    call.args.first().copied()
                }
                _ => None,
            })
            .expect("expected caller arg for Verify encode helper");
        let verify_arg_shape = crate::repr::runtime_shape_for_value(
            &db,
            &verify_caller_core,
            &verify_caller.body.values,
            &verify_caller.body.locals,
            verify_arg,
        )
        .expect("Verify caller arg shape should resolve");
        assert!(
            !matches!(verify_arg_shape, RuntimeShape::ConstRef { .. }),
            "caller arg for Verify encode helper should already be runtime-backed, got {:?}",
            verify_arg_shape,
        );

        for expected_ty in ["[u256; 2]", "[u256; 38]"] {
            let helper = module
                .functions
                .iter()
                .find(|func| {
                    func.symbol_name.contains("encode_to_ptr")
                        && func.body.param_locals.first().is_some_and(|local| {
                            func.body.local(*local).ty.pretty_print(&db) == expected_ty
                        })
                })
                .unwrap_or_else(|| panic!("expected encode_to_ptr helper for {expected_ty}"));

            if expected_ty == "[u256; 2]" {
                let arg = verify_helper
                    .body
                    .blocks
                    .iter()
                    .flat_map(|block| block.insts.iter())
                    .find_map(|inst| match inst {
                        MirInst::Assign {
                            rvalue: Rvalue::Call(call),
                            ..
                        } if call.resolved_name.as_deref() == Some(helper.symbol_name.as_str()) => {
                            call.args.first().copied()
                        }
                        _ => None,
                    })
                    .expect("expected caller arg for nested [u256; 2] encode helper");
                let arg_shape = crate::repr::runtime_shape_for_value(
                    &db,
                    &verify_core,
                    &verify_helper.body.values,
                    &verify_helper.body.locals,
                    arg,
                )
                .expect("caller arg shape should resolve");
                let arg_data = verify_helper.body.value(arg);
                let arg_debug = match &arg_data.origin {
                    ValueOrigin::PlaceRef(place) | ValueOrigin::MoveOut { place } => {
                        let base = verify_helper.body.value(place.base);
                        format!(
                            "origin={:?} place={:?} base_origin={:?} base_shape={:?} base_ptr={:?}",
                            arg_data.origin,
                            place,
                            base.origin,
                            crate::repr::runtime_shape_for_value(
                                &db,
                                &verify_core,
                                &verify_helper.body.values,
                                &verify_helper.body.locals,
                                place.base,
                            ),
                            base.pointer_info,
                        )
                    }
                    _ => format!(
                        "origin={:?} repr={:?} ptr={:?}",
                        arg_data.origin, arg_data.repr, arg_data.pointer_info
                    ),
                };
                assert!(
                    !matches!(arg_shape, RuntimeShape::ConstRef { .. }),
                    "caller arg for `{}` should already be runtime-backed, got {:?}; {}; local_pointer_leaf_infos={:?}",
                    helper.symbol_name,
                    arg_shape,
                    arg_debug,
                    verify_helper
                        .body
                        .local(verify_helper.body.param_locals[0])
                        .pointer_leaf_infos,
                );
            }

            let param_local = helper.body.param_locals[0];
            let param_data = helper.body.local(param_local);
            assert_eq!(
                param_data.address_space,
                AddressSpaceKind::Memory,
                "Yul-prepared helper `{}` param `{expected_ty}` should stay memory-backed, got space {:?} const {:?} shape {:?}",
                helper.symbol_name,
                param_data.address_space,
                param_data.const_backing,
                param_data.runtime_shape,
            );
            assert!(
                !param_data.const_backing.is_const(),
                "Yul-prepared helper `{}` param `{expected_ty}` should not stay const-backed: space {:?} const {:?} shape {:?}",
                helper.symbol_name,
                param_data.address_space,
                param_data.const_backing,
                param_data.runtime_shape,
            );

            let param_base = helper
                .body
                .values
                .iter()
                .enumerate()
                .find_map(|(idx, value)| match value.origin {
                    ValueOrigin::Local(local) | ValueOrigin::PlaceRoot(local)
                        if local == param_local =>
                    {
                        Some(ValueId(idx as u32))
                    }
                    _ => None,
                })
                .expect("expected value rooted at the helper param");
            let indexed_place = Place::new(
                param_base,
                MirProjectionPath::from_projection(Projection::Index(
                    hir::projection::IndexSource::Constant(0),
                )),
            );
            assert_eq!(
                helper.body.place_address_space(&indexed_place),
                AddressSpaceKind::Memory,
                "Yul-prepared helper `{}` indexed param loads should stay in memory",
                helper.symbol_name,
            );
        }
    }

    #[test]
    fn runtime_shapes_preserve_nested_pointer_leaf_spaces_across_forwarded_returns() {
        let mut db = DriverDataBase::default();
        let module = lower_inline_module(
            &mut db,
            "file:///runtime_shapes_preserve_nested_pointer_leaf_spaces_across_forwarded_returns.fe",
            include_str!("../../fe/tests/fixtures/fe_test/view_param_local_ref_take_reverse.fe"),
        );

        let sum_first4 = module
            .functions
            .iter()
            .find(|func| func.symbol_name == "sum_first4")
            .expect("expected sum_first4 helper");
        let arr_param = sum_first4.body.local(sum_first4.body.param_locals[0]);
        let take_local = sum_first4
            .body
            .blocks
            .iter()
            .flat_map(|block| block.insts.iter())
            .find_map(|inst| match inst {
                MirInst::Assign {
                    dest: Some(local),
                    rvalue: Rvalue::Call(call),
                    ..
                } if call
                    .resolved_name
                    .as_deref()
                    .is_some_and(|name| name.contains("take_u256___u256__8")) =>
                {
                    Some((*local, *call.args.get(1).expect("expected take seq arg")))
                }
                _ => None,
            })
            .expect("expected take_u256 call result local");

        assert!(
            sum_first4
                .body
                .value_pointer_info(take_local.1)
                .is_some_and(|info| info.address_space == AddressSpaceKind::Code),
            "forwarded aggregate returns should preserve code-backed call args, got sum_first4 param space {:?} const {:?} shape {:?} infos {:?}, take arg origin {:?} base {:?} repr {:?} ptr {:?}",
            arr_param.address_space,
            arr_param.const_backing,
            arr_param.runtime_shape,
            arr_param.pointer_leaf_infos,
            sum_first4.body.value(take_local.1).origin,
            match &sum_first4.body.value(take_local.1).origin {
                ValueOrigin::PlaceRef(place) | ValueOrigin::MoveOut { place } => {
                    Some(sum_first4.body.value(place.base).origin.clone())
                }
                _ => None,
            },
            sum_first4.body.value(take_local.1).repr,
            sum_first4.body.value_pointer_info(take_local.1),
        );
    }

    #[test]
    fn code_backed_take_wrapper_results_preserve_nested_base_leaf_metadata() {
        let mut db = DriverDataBase::default();
        let module = lower_inline_module(
            &mut db,
            "file:///code_backed_take_wrapper_results_preserve_nested_base_leaf_metadata.fe",
            include_str!("../../fe/tests/fixtures/fe_test/view_param_local_ref_take_reverse.fe"),
        );

        let sum_first4 = module
            .functions
            .iter()
            .find(|func| func.symbol_name == "sum_first4_arg0_root_code")
            .expect("expected code-specialized sum_first4 helper");
        let head_local = sum_first4
            .body
            .blocks
            .iter()
            .flat_map(|block| block.insts.iter())
            .find_map(|inst| match inst {
                MirInst::Assign {
                    dest: Some(local),
                    rvalue: Rvalue::Call(call),
                    ..
                } if call
                    .resolved_name
                    .as_deref()
                    .is_some_and(|name| name.contains("take_u256_arg1_root_code___u256__8")) =>
                {
                    Some(*local)
                }
                _ => None,
            })
            .expect("expected take_u256 call result local");

        assert_eq!(
            sum_first4
                .body
                .local(head_local)
                .pointer_leaf_infos
                .iter()
                .find_map(|(path, info)| {
                    (path == &MirProjectionPath::from_projection(Projection::Field(1)))
                        .then_some(info.address_space)
                }),
            Some(AddressSpaceKind::Code),
            "code-backed take wrapper results should preserve the base field leaf metadata; head_local={:?}",
            sum_first4.body.local(head_local),
        );
    }

    #[test]
    fn runtime_shapes_keep_pointer_like_wrapper_returns_const_backed() {
        let mut db = DriverDataBase::default();
        let module = lower_inline_module(
            &mut db,
            "file:///runtime_shapes_keep_pointer_like_wrapper_returns_const_backed.fe",
            include_str!("../../fe/tests/fixtures/fe_test/view_param_local_ref_take_reverse.fe"),
        );

        let reverse_u256 = module
            .functions
            .iter()
            .find(|func| func.symbol_name.contains("reverse_u256___u256__8"))
            .expect("expected reverse_u256 helper");
        let _reverse_generic = module
            .functions
            .iter()
            .find(|func| func.symbol_name.starts_with("reverse__u256__u256__8"))
            .expect("expected reverse generic helper");
        let sum_last4 = module
            .functions
            .iter()
            .find(|func| func.symbol_name == "sum_last4")
            .expect("expected sum_last4 helper");
        assert!(
            matches!(
                reverse_u256.runtime_return_shape,
                RuntimeShape::ConstRef { .. }
            ),
            "pointer-like wrapper helper returns should stay const-backed, got {:?}",
            reverse_u256.runtime_return_shape,
        );
        let seq_param = reverse_u256.body.local(reverse_u256.body.param_locals[0]);
        assert!(
            matches!(seq_param.runtime_shape, RuntimeShape::ConstRef { .. }),
            "pointer-like wrapper helper params should stay const-backed, got {:?}",
            seq_param.runtime_shape,
        );
        assert!(
            matches!(seq_param.const_backing, crate::ir::LocalConstBacking::Const),
            "pointer-like wrapper helper params should keep const backing, got {:?}",
            seq_param.const_backing,
        );
        assert_eq!(
            seq_param.address_space,
            AddressSpaceKind::Code,
            "pointer-like wrapper helper params should stay in code space",
        );

        let rev_local = sum_last4
            .body
            .locals
            .iter()
            .find(|local| local.name == "rev")
            .expect("expected rev local");
        assert!(
            matches!(rev_local.runtime_shape, RuntimeShape::ConstRef { .. }),
            "pointer-like wrapper locals receiving readonly call results should stay const-backed, got {:?}",
            rev_local.runtime_shape,
        );
        assert!(
            matches!(rev_local.const_backing, crate::ir::LocalConstBacking::Const),
            "pointer-like wrapper locals receiving readonly call results should keep const backing, got {:?}",
            rev_local.const_backing,
        );
        assert_eq!(
            rev_local.address_space,
            AddressSpaceKind::Code,
            "pointer-like wrapper locals receiving readonly call results should stay in code space",
        );

        let take_reverse = module
            .functions
            .iter()
            .find(|func| {
                func.symbol_name
                    .contains("take_u256__Reverse_u256___u256__8__")
            })
            .expect("expected take_u256<Reverse> helper");
        let seq_param = take_reverse.body.local(take_reverse.body.param_locals[1]);
        assert!(
            matches!(seq_param.runtime_shape, RuntimeShape::ConstRef { .. }),
            "pointer-like wrapper ref params should specialize to const-backed runtime shape, got {:?}",
            seq_param.runtime_shape,
        );
        assert!(
            matches!(seq_param.const_backing, crate::ir::LocalConstBacking::Const),
            "pointer-like wrapper ref params should keep const backing, got {:?}",
            seq_param.const_backing,
        );
        assert_eq!(
            seq_param.address_space,
            AddressSpaceKind::Code,
            "pointer-like wrapper ref params should specialize to code space",
        );

        let tail_local = sum_last4
            .body
            .locals
            .iter()
            .find(|local| local.name == "tail")
            .expect("expected tail local");
        assert!(
            matches!(tail_local.runtime_shape, RuntimeShape::ObjectRef { .. }),
            "readonly wrapper locals returned through runtime-backed helpers should stay object-backed until projected loads, got {:?}",
            tail_local.runtime_shape,
        );
        assert_eq!(
            tail_local
                .pointer_leaf_infos
                .iter()
                .find_map(|(path, info)| {
                    (path == &MirProjectionPath::from_projection(Projection::Field(1)))
                        .then_some(info.address_space)
                }),
            Some(AddressSpaceKind::Code),
            "readonly wrapper locals should preserve code-space leaf metadata for forwarded ref fields",
        );

        for func in module.functions.iter().filter(|func| {
            func.symbol_name.contains("take_i__t__")
                && (func.symbol_name.contains("len__u256_Reverse")
                    || func.symbol_name.contains("get__u256_Reverse"))
        }) {
            for call in func.body.blocks.iter().flat_map(|block| {
                block.insts.iter().filter_map(|inst| match inst {
                    MirInst::Assign {
                        rvalue: Rvalue::Call(call),
                        ..
                    } if call
                        .resolved_name
                        .as_deref()
                        .is_some_and(|name| name.contains("reverse_i__t__")) =>
                    {
                        Some(call)
                    }
                    _ => None,
                })
            }) {
                let Some(&receiver) = call.args.first() else {
                    continue;
                };
                assert!(
                    matches!(
                        func.body.value(receiver).runtime_shape,
                        RuntimeShape::ConstRef { .. }
                    ),
                    "take<Reverse> should call Reverse helpers with a const-backed receiver, got {:?} origin {:?} ptr {:?} inner_origin {:?} inner_shape {:?} inner_ptr {:?} inner_local {:?} in `{}`",
                    func.body.value(receiver).runtime_shape,
                    func.body.value(receiver).origin,
                    func.body.value_pointer_info(receiver),
                    match func.body.value(receiver).origin {
                        ValueOrigin::TransparentCast { value } =>
                            Some(func.body.value(value).origin.clone()),
                        _ => None,
                    },
                    match func.body.value(receiver).origin {
                        ValueOrigin::TransparentCast { value } =>
                            Some(func.body.value(value).runtime_shape),
                        _ => None,
                    },
                    match func.body.value(receiver).origin {
                        ValueOrigin::TransparentCast { value } =>
                            func.body.value_pointer_info(value),
                        _ => None,
                    },
                    match func.body.value(receiver).origin {
                        ValueOrigin::TransparentCast { value } =>
                            match func.body.value(value).origin {
                                ValueOrigin::Local(local) => Some(func.body.local(local)),
                                _ => None,
                            },
                        _ => None,
                    },
                    func.symbol_name,
                );
                assert_eq!(
                    func.body
                        .value_pointer_info(receiver)
                        .map(|info| info.address_space),
                    Some(AddressSpaceKind::Code),
                    "take<Reverse> should preserve code-space pointer info on Reverse helper receivers in `{}`",
                    func.symbol_name,
                );
            }
        }

        let reverse_get = module
            .functions
            .iter()
            .find(|func| {
                func.symbol_name.contains("reverse_i__t__")
                    && func.symbol_name.contains("get__u256__u256__8__")
            })
            .expect("expected Reverse::get helper");
        let self_param = reverse_get.body.local(reverse_get.body.param_locals[0]);
        assert!(
            matches!(self_param.runtime_shape, RuntimeShape::ConstRef { .. }),
            "pointer-like wrapper method receivers should specialize to const-backed runtime shape, got {:?}",
            self_param.runtime_shape,
        );
        assert!(
            matches!(
                self_param.const_backing,
                crate::ir::LocalConstBacking::Const
            ),
            "pointer-like wrapper method receivers should keep const backing, got {:?}",
            self_param.const_backing,
        );
        assert_eq!(
            self_param.address_space,
            AddressSpaceKind::Code,
            "pointer-like wrapper method receivers should specialize to code space",
        );

        let take_get = module
            .functions
            .iter()
            .find(|func| {
                func.symbol_name.contains("take_i__t__")
                    && func.symbol_name.contains("get__u256_Reverse")
            })
            .expect("expected Take<Reverse>::get helper");
        let take_get_self = take_get.body.local(take_get.body.param_locals[0]);
        assert!(
            matches!(take_get_self.runtime_shape, RuntimeShape::ObjectRef { .. }),
            "wrapper helpers should keep runtime-backed outer receivers when only nested ref fields are readonly, got {:?}",
            take_get_self.runtime_shape,
        );
        assert_eq!(
            take_get_self
                .pointer_leaf_infos
                .iter()
                .find_map(|(path, info)| {
                    (path == &MirProjectionPath::from_projection(Projection::Field(1)))
                        .then_some(info.address_space)
                }),
            Some(AddressSpaceKind::Code),
            "wrapper helper receivers should preserve code-space leaf metadata for readonly ref fields",
        );
    }

    #[test]
    fn discriminant_loads_from_object_backed_enums_keep_enum_tag_runtime_shape() {
        let mut db = DriverDataBase::default();
        let module = lower_inline_module(
            &mut db,
            "file:///discriminant_loads_from_object_backed_enums_keep_enum_tag_runtime_shape.fe",
            include_str!("../../fe/tests/fixtures/fe_test/option_mut_scalar_match_regression.fe"),
        );

        let recv = module
            .functions
            .iter()
            .find(|func| func.symbol_name == "__C_recv_0_0")
            .expect("expected __C_recv_0_0");
        let (dest_local, discr_place) = recv
            .body
            .blocks
            .iter()
            .flat_map(|block| block.insts.iter())
            .find_map(|inst| match inst {
                MirInst::Assign {
                    dest: Some(local),
                    rvalue: Rvalue::Load { place },
                    ..
                } if matches!(
                    place.projection.iter().last(),
                    Some(Projection::Discriminant)
                ) =>
                {
                    Some((*local, place.clone()))
                }
                _ => None,
            })
            .expect("expected discriminant load");
        let locals = recv.body.locals.clone();
        assert!(
            matches!(
                locals[dest_local.index()].runtime_shape,
                RuntimeShape::EnumTag { .. }
            ),
            "discriminant load destinations must keep enum-tag runtime shape, got {:?}",
            locals[dest_local.index()].runtime_shape,
        );
        assert!(
            matches!(
                discr_place.projection.iter().last(),
                Some(Projection::Discriminant)
            ),
            "expected discriminant load place, got {:?}",
            discr_place,
        );
    }

    #[test]
    fn runtime_shapes_keep_receiver_field_returns_storage_backed_with_same_typed_args() {
        let mut db = DriverDataBase::default();
        let module = lower_inline_module(
            &mut db,
            "file:///runtime_shapes_keep_receiver_field_returns_storage_backed_with_same_typed_args.fe",
            include_str!(
                "../../fe/tests/fixtures/fe_test/receiver_returns_non_mem_with_same_typed_memory_arg.fe"
            ),
        );

        let recv = module
            .functions
            .iter()
            .find(|func| func.symbol_name == "__C_recv_0_0")
            .expect("expected recv helper");
        let target_local_idx = recv
            .body
            .locals
            .iter()
            .position(|local| local.name == "target")
            .expect("expected target local");
        let call_dest = recv
            .body
            .blocks
            .iter()
            .flat_map(|block| block.insts.iter())
            .find_map(|inst| match inst {
                MirInst::Assign {
                    dest: Some(dest),
                    rvalue: Rvalue::Call(call),
                    ..
                } if call
                    .resolved_name
                    .as_deref()
                    .is_some_and(|name| name.contains("choose_self_stor_arg0_root_stor")) =>
                {
                    Some(*dest)
                }
                _ => None,
            })
            .expect("expected choose_self call");

        assert_eq!(
            call_dest.index(),
            target_local_idx,
            "choose_self call should assign directly into the `target` local",
        );

        let target = recv.body.local(call_dest);
        assert_eq!(
            target.address_space,
            AddressSpaceKind::Storage,
            "receiver-field returns should keep storage address space even when another arg matches the return type",
        );
        assert!(
            matches!(target.runtime_shape, RuntimeShape::AddressWord(info) if info.address_space == AddressSpaceKind::Storage),
            "receiver-field returns should stay storage-backed handles, got {:?}",
            target.runtime_shape,
        );
    }

    #[test]
    fn borrowed_reader_args_keep_reader_leaf_infos() {
        let mut db = DriverDataBase::default();
        let url = Url::parse("file:///borrowed_reader_args_keep_reader_leaf_infos.fe").unwrap();
        let file = db.workspace().touch(
            &mut db,
            url,
            Some(include_str!("../../fe/tests/fixtures/fe_test/ref_scalar_nested.fe").to_owned()),
        );
        let top_mod = db.top_mod(file);
        let module = lower_module(&db, top_mod).expect("module should lower");

        let test_func = module
            .functions
            .iter()
            .find(|func| func.symbol_name == "test_ref_scalar_nested")
            .expect("expected ref_scalar_nested test");
        let sum_reader = module
            .functions
            .iter()
            .find(|func| func.symbol_name == "sum_reader")
            .expect("expected sum_reader helper");
        let reader_local = test_func
            .body
            .locals
            .iter()
            .position(|local| local.name == "reader")
            .map(|idx| crate::LocalId(idx as u32))
            .expect("expected reader local");
        let call_arg = test_func
            .body
            .blocks
            .iter()
            .flat_map(|block| block.insts.iter())
            .find_map(|inst| match inst {
                MirInst::Assign {
                    rvalue: Rvalue::Call(call),
                    ..
                } if call.resolved_name.as_deref() == Some("sum_reader") => {
                    call.args.first().copied()
                }
                _ => None,
            })
            .expect("expected sum_reader call arg");
        let arg_pointer_infos = crate::repr::pointer_leaf_infos_for_value(
            &db,
            &CoreLib::new(&db, top_mod.scope()),
            &test_func.body.values,
            &test_func.body.locals,
            call_arg,
        );

        assert_eq!(
            arg_pointer_infos,
            test_func.body.local(reader_local).pointer_leaf_infos,
            "borrowed Reader args should preserve the reader local's nested handle leaf infos",
        );
        assert!(
            sum_reader
                .body
                .local(sum_reader.body.param_locals[0])
                .pointer_leaf_infos
                .iter()
                .any(|(path, _)| {
                    matches!(path.iter().next(), Some(Projection::Field(0))) && path.len() == 1
                }),
            "ref Reader params should retain nested `.source` handle leaf infos",
        );
    }

    #[test]
    fn ref_call_args_materialize_plain_scalar_place_roots_as_object_refs() {
        let src = include_str!("../../fe/tests/fixtures/fe_test/ref_scalar_local_place_root.fe");
        let mut db = DriverDataBase::default();
        let module = lower_inline_module(&mut db, "file:///ref_scalar_local_place_root.fe", src);

        let test_func = module
            .functions
            .iter()
            .find(|func| func.symbol_name == "test_ref_scalar_local_place_root")
            .expect("expected ref_scalar_local_place_root test");
        let call_arg = test_func
            .body
            .blocks
            .iter()
            .flat_map(|block| block.insts.iter())
            .find_map(|inst| match inst {
                MirInst::Assign {
                    rvalue: Rvalue::Call(call),
                    ..
                } if call.resolved_name.as_deref() == Some("read_value") => {
                    call.args.first().copied()
                }
                _ => None,
            })
            .expect("expected read_value call arg");
        let crate::ValueOrigin::PlaceRef(place) = &test_func.body.value(call_arg).origin else {
            panic!(
                "expected read_value arg to materialize through a PlaceRef, got {:?}",
                test_func.body.value(call_arg).origin
            );
        };
        let crate::ValueOrigin::PlaceRoot(local) = test_func.body.value(place.base).origin else {
            panic!(
                "expected read_value PlaceRef base to be a PlaceRoot, got {:?}",
                test_func.body.value(place.base).origin
            );
        };
        let root_local = test_func.body.local(local);

        assert!(
            root_local.pointer_leaf_infos.is_empty(),
            "plain scalar temp PlaceRoot should not need pointer metadata: {:?}",
            root_local.pointer_leaf_infos,
        );
        assert_eq!(
            root_local.place_root_layout,
            crate::ir::LocalPlaceRootLayout::ObjectRootStorage {
                target_ty: root_local.ty,
                source: crate::ir::ObjectRootSource::MaterializedScalarBorrow,
            },
            "plain scalar PlaceRoots should now be classified explicitly during MIR lowering",
        );
        assert_eq!(
            test_func.body.value(place.base).runtime_shape,
            RuntimeShape::ObjectRef {
                target_ty: root_local.ty,
            },
            "plain scalar PlaceRoot backing a `ref u256` call arg should currently refine to ObjectRef",
        );
    }

    #[test]
    fn explicit_memory_slot_place_roots_stay_memory_ptrs_for_object_compatible_types() {
        let mut db = DriverDataBase::default();
        let url = Url::parse(
            "file:///explicit_memory_slot_place_roots_stay_memory_ptrs_for_object_compatible_types.fe",
        )
        .expect("test url should be valid");
        let file = db.workspace().touch(
            &mut db,
            url,
            Some(
                r#"
struct Pair {
    a: u256,
    b: u256,
}

fn marker() {
    let pair = Pair { a: 1, b: 2 }
}
"#
                .to_owned(),
            ),
        );
        let top_mod = db.top_mod(file);
        let core = CoreLib::new(&db, top_mod.scope());
        let module = lower_module(&db, top_mod).expect("module should lower");
        let pair_ty = module
            .functions
            .iter()
            .find(|func| func.symbol_name == "marker")
            .and_then(|func| func.body.locals.iter().find(|local| local.name == "pair"))
            .expect("expected `pair` local")
            .ty;
        let locals = vec![LocalData {
            name: "pair_root".to_owned(),
            ty: pair_ty,
            is_mut: true,
            source: SourceInfoId::SYNTHETIC,
            address_space: AddressSpaceKind::Memory,
            pointer_leaf_infos: Vec::new(),
            place_root_layout: LocalPlaceRootLayout::MemorySlot,
            const_backing: crate::ir::LocalConstBacking::Unknown,
            runtime_shape: RuntimeShape::Unresolved,
        }];
        let values = vec![ValueData {
            ty: pair_ty,
            origin: ValueOrigin::PlaceRoot(LocalId(0)),
            source: SourceInfoId::SYNTHETIC,
            repr: ValueRepr::Word,
            pointer_info: None,
            runtime_shape: RuntimeShape::Unresolved,
        }];

        assert_eq!(
            crate::repr::place_root_runtime_shape_for_local(&locals[0]),
            Some(RuntimeShape::MemoryPtr {
                target_ty: Some(pair_ty),
            }),
            "explicit MemorySlot provenance should classify the place root as a memory slot",
        );
        assert_eq!(
            crate::repr::runtime_shape_for_value(&db, &core, &values, &locals, ValueId(0)),
            Some(RuntimeShape::MemoryPtr {
                target_ty: Some(pair_ty),
            }),
            "explicit MemorySlot provenance must not refine object-compatible aggregate roots to ObjectRef",
        );
    }

    #[test]
    fn view_call_args_for_non_copy_structs_stay_by_ref_without_place_root_materialization() {
        let src = include_str!("../../fe/tests/fixtures/fe_test/view_pair_param.fe");
        let mut db = DriverDataBase::default();
        let module = lower_inline_module(&mut db, "file:///view_pair_param.fe", src);

        let first = module
            .functions
            .iter()
            .find(|func| func.symbol_name == "first")
            .expect("expected first helper");
        let first_param_ty = first.body.local(first.body.param_locals[0]).ty;
        let test_func = module
            .functions
            .iter()
            .find(|func| func.symbol_name == "test_view_pair_param")
            .expect("expected view_pair_param test");
        let call_arg = test_func
            .body
            .blocks
            .iter()
            .flat_map(|block| block.insts.iter())
            .find_map(|inst| match inst {
                MirInst::Assign {
                    rvalue: Rvalue::Call(call),
                    ..
                } if call.resolved_name.as_deref() == Some("first") => call.args.first().copied(),
                _ => None,
            })
            .expect("expected first call arg");
        let crate::ValueOrigin::TransparentCast { value: inner } =
            test_func.body.value(call_arg).origin
        else {
            panic!(
                "expected first arg to lower through a view TransparentCast, got {:?}",
                test_func.body.value(call_arg).origin,
            );
        };
        let crate::ValueOrigin::Local(local) = test_func.body.value(inner).origin else {
            panic!(
                "expected view Pair arg to reuse a by-ref local instead of a PlaceRoot, got {:?}",
                test_func.body.value(inner).origin,
            );
        };
        let local_data = test_func.body.local(local);

        assert_eq!(
            test_func.body.value(call_arg).ty,
            first_param_ty,
            "the call arg should lower to the callee's internal view param type",
        );
        assert_ne!(
            first_param_ty, local_data.ty,
            "the internal view type should stay distinct from the aggregate local type",
        );
        assert_eq!(
            test_func.body.value(inner).runtime_shape,
            RuntimeShape::ObjectRef {
                target_ty: local_data.ty,
            },
            "non-Copy aggregate locals passed to bare params should stay object-backed",
        );
        assert_eq!(
            local_data.place_root_layout,
            crate::ir::LocalPlaceRootLayout::ObjectRootValue {
                target_ty: local_data.ty,
                source: crate::ir::ObjectRootSource::DeclaredByRefAggregate,
            },
            "aggregate locals should carry explicit object-root provenance from initial lowering",
        );
        assert_eq!(
            test_func.body.value(call_arg).runtime_shape,
            RuntimeShape::ObjectRef {
                target_ty: local_data.ty,
            },
            "the view cast should preserve the aggregate's object-backed runtime shape",
        );
    }

    #[test]
    fn explicit_object_root_spill_provenance_survives_joined_scalar_ref_flow() {
        let mut db = DriverDataBase::default();
        let url = Url::parse(
            "file:///explicit_object_root_spill_provenance_survives_joined_scalar_ref_flow.fe",
        )
        .expect("test url should be valid");
        let file = db.workspace().touch(
            &mut db,
            url,
            Some(
                r#"
fn smoke() {}
"#
                .to_owned(),
            ),
        );
        let top_mod = db.top_mod(file);
        let mut module = lower_module(&db, top_mod).expect("module should lower");
        let smoke = module
            .functions
            .iter_mut()
            .find(|func| func.symbol_name == "smoke")
            .expect("expected smoke helper");

        let mut body = MirBody::new();
        body.stage = crate::ir::MirStage::Repr;
        body.blocks.push(BasicBlock {
            insts: Vec::new(),
            terminator: Terminator::Return {
                source: SourceInfoId::SYNTHETIC,
                value: None,
            },
        });

        let spill_target_ty = TyId::u256(&db);
        let owner_local = body.alloc_local(LocalData {
            name: "owner".to_owned(),
            ty: spill_target_ty,
            is_mut: false,
            source: SourceInfoId::SYNTHETIC,
            address_space: AddressSpaceKind::Memory,
            pointer_leaf_infos: Vec::new(),
            place_root_layout: LocalPlaceRootLayout::Direct,
            const_backing: crate::ir::LocalConstBacking::Unknown,
            runtime_shape: RuntimeShape::Unresolved,
        });
        let spill_local = body.alloc_local(LocalData {
            name: "spill0".to_owned(),
            ty: spill_target_ty,
            is_mut: false,
            source: SourceInfoId::SYNTHETIC,
            address_space: AddressSpaceKind::Memory,
            pointer_leaf_infos: Vec::new(),
            place_root_layout: LocalPlaceRootLayout::ObjectRootValue {
                target_ty: spill_target_ty,
                source: ObjectRootSource::SpillOf(owner_local),
            },
            const_backing: crate::ir::LocalConstBacking::Unknown,
            runtime_shape: RuntimeShape::Unresolved,
        });
        body.spill_slots.insert(owner_local, spill_local);

        let spill_value = body.alloc_value(ValueData {
            ty: spill_target_ty,
            origin: ValueOrigin::Local(spill_local),
            source: SourceInfoId::SYNTHETIC,
            repr: ValueRepr::Word,
            pointer_info: None,
            runtime_shape: RuntimeShape::Unresolved,
        });
        let spill_place_root = body.alloc_value(ValueData {
            ty: spill_target_ty,
            origin: ValueOrigin::PlaceRoot(spill_local),
            source: SourceInfoId::SYNTHETIC,
            repr: ValueRepr::Word,
            pointer_info: None,
            runtime_shape: RuntimeShape::Unresolved,
        });
        body.blocks[0].insts.push(MirInst::BindValue {
            source: SourceInfoId::SYNTHETIC,
            value: spill_value,
        });
        body.blocks[0].insts.push(MirInst::BindValue {
            source: SourceInfoId::SYNTHETIC,
            value: spill_place_root,
        });
        smoke.body = body;
        smoke.runtime_abi = crate::ir::RuntimeAbi::source_shaped(0, Vec::new());

        super::normalize_runtime_shapes(&db, &mut module);
        let smoke = module
            .functions
            .iter()
            .find(|func| func.symbol_name == "smoke")
            .expect("expected smoke helper after normalization");
        let spill_local_data = smoke.body.local(spill_local);

        assert_eq!(
            spill_local_data.runtime_shape,
            RuntimeShape::ObjectRef {
                target_ty: spill_target_ty,
            },
            "spill locals with explicit ObjectRoot provenance should normalize as object refs",
        );
        assert!(
            smoke.body.values.iter().any(|value| {
                matches!(
                    value.origin,
                    ValueOrigin::Local(local) | ValueOrigin::PlaceRoot(local) if local == spill_local
                ) && matches!(
                    value.runtime_shape,
                    RuntimeShape::ObjectRef { target_ty }
                        if target_ty == spill_target_ty
                )
            }),
            "joined scalar-ref spills should keep object-ref runtime shapes on values that reload from the spill",
        );
    }

    fn normalize_rebound_handle_copy_probe<'db>(
        db: &'db mut DriverDataBase,
        seed_source_memory_info: bool,
        seed_dest_memory_info: bool,
    ) -> (
        crate::MirModule<'db>,
        LocalId,
        LocalId,
        RuntimeShape<'db>,
        Vec<(MirProjectionPath<'db>, PointerInfo<'db>)>,
    ) {
        let url =
            Url::parse("file:///normalize_rebound_handle_copy_probe.fe").expect("valid test url");
        let file = db
            .workspace()
            .touch(db, url, Some("fn smoke() {}".to_owned()));
        let top_mod = db.top_mod(file);
        let mut module = lower_module(db, top_mod).expect("module should lower");
        let smoke = module
            .functions
            .iter_mut()
            .find(|func| func.symbol_name == "smoke")
            .expect("expected smoke helper");
        let core = super::function_core_lib(db, smoke);

        let handle_inner_ty = TyId::u256(db);
        let handle_ty = TyId::borrow_mut_of(db, handle_inner_ty);
        let root = MirProjectionPath::new();
        let memory_info = PointerInfo {
            address_space: AddressSpaceKind::Memory,
            target_ty: Some(handle_inner_ty),
        };
        let storage_info = PointerInfo {
            address_space: AddressSpaceKind::Storage,
            target_ty: Some(handle_inner_ty),
        };

        let mut body = MirBody::new();
        body.stage = crate::ir::MirStage::Repr;
        body.blocks.push(BasicBlock {
            insts: Vec::new(),
            terminator: Terminator::Return {
                source: SourceInfoId::SYNTHETIC,
                value: None,
            },
        });

        let memory_root_infos = || vec![(MirProjectionPath::new(), memory_info)];
        let source_local = body.alloc_local(LocalData {
            name: "source".to_owned(),
            ty: handle_ty,
            is_mut: true,
            source: SourceInfoId::SYNTHETIC,
            address_space: AddressSpaceKind::Memory,
            pointer_leaf_infos: seed_source_memory_info
                .then(memory_root_infos)
                .unwrap_or_default(),
            place_root_layout: LocalPlaceRootLayout::Direct,
            const_backing: crate::ir::LocalConstBacking::Unknown,
            runtime_shape: RuntimeShape::Unresolved,
        });
        let dest_local = body.alloc_local(LocalData {
            name: "dest".to_owned(),
            ty: handle_ty,
            is_mut: true,
            source: SourceInfoId::SYNTHETIC,
            address_space: AddressSpaceKind::Memory,
            pointer_leaf_infos: seed_dest_memory_info
                .then(memory_root_infos)
                .unwrap_or_default(),
            place_root_layout: LocalPlaceRootLayout::Direct,
            const_backing: crate::ir::LocalConstBacking::Unknown,
            runtime_shape: RuntimeShape::Unresolved,
        });
        let other_local = body.alloc_local(LocalData {
            name: "other".to_owned(),
            ty: handle_ty,
            is_mut: true,
            source: SourceInfoId::SYNTHETIC,
            address_space: AddressSpaceKind::Storage,
            pointer_leaf_infos: vec![(root, storage_info)],
            place_root_layout: LocalPlaceRootLayout::Direct,
            const_backing: crate::ir::LocalConstBacking::Unknown,
            runtime_shape: RuntimeShape::Unresolved,
        });

        let source_value = body.alloc_value(ValueData {
            ty: handle_ty,
            origin: ValueOrigin::Local(source_local),
            source: SourceInfoId::SYNTHETIC,
            repr: ValueRepr::Ptr(AddressSpaceKind::Memory),
            pointer_info: Some(memory_info),
            runtime_shape: RuntimeShape::Unresolved,
        });
        let other_value = body.alloc_value(ValueData {
            ty: handle_ty,
            origin: ValueOrigin::Local(other_local),
            source: SourceInfoId::SYNTHETIC,
            repr: ValueRepr::Ptr(AddressSpaceKind::Storage),
            pointer_info: Some(storage_info),
            runtime_shape: RuntimeShape::Unresolved,
        });

        body.blocks[0].insts.push(MirInst::Assign {
            source: SourceInfoId::SYNTHETIC,
            dest: Some(dest_local),
            rvalue: Rvalue::Value(source_value),
        });
        body.blocks[0].insts.push(MirInst::Assign {
            source: SourceInfoId::SYNTHETIC,
            dest: Some(dest_local),
            rvalue: Rvalue::Value(other_value),
        });
        body.blocks[0].insts.push(MirInst::BindValue {
            source: SourceInfoId::SYNTHETIC,
            value: source_value,
        });
        smoke.body = body;
        smoke.runtime_abi = crate::ir::RuntimeAbi::source_shaped(0, Vec::new());

        let expected_source_shape =
            crate::repr::runtime_shape_for_local(db, &core, smoke.body.local(source_local));
        let expected_source_infos = smoke.body.local(source_local).pointer_leaf_infos.clone();

        super::normalize_runtime_shapes(db, &mut module);
        (
            module,
            source_local,
            dest_local,
            expected_source_shape,
            expected_source_infos,
        )
    }

    #[test]
    fn source_local_pointer_infos_do_not_absorb_rebound_dest_infos() {
        let mut db = DriverDataBase::default();
        let (module, source_local, _, _, expected_source_infos) =
            normalize_rebound_handle_copy_probe(&mut db, true, false);
        let smoke = module
            .functions
            .iter()
            .find(|func| func.symbol_name == "smoke")
            .expect("expected smoke helper");
        let source = smoke.body.local(source_local);

        assert_eq!(
            source.pointer_leaf_infos, expected_source_infos,
            "copying `source` into `dest` must not let later `dest` assignments rewrite `source` leaf metadata",
        );
    }

    #[test]
    fn source_local_runtime_shape_does_not_change_after_dest_rebind() {
        let mut db = DriverDataBase::default();
        let (module, source_local, _, expected_source_shape, _) =
            normalize_rebound_handle_copy_probe(&mut db, true, false);
        let smoke = module
            .functions
            .iter()
            .find(|func| func.symbol_name == "smoke")
            .expect("expected smoke helper");
        let source = smoke.body.local(source_local);

        assert_eq!(
            source.runtime_shape, expected_source_shape,
            "copying a handle local into `dest` must not let later `dest` rebinds reclassify `source`",
        );
    }

    #[test]
    fn less_informed_source_local_backfills_same_site_dest_infos_only() {
        let mut db = DriverDataBase::default();
        let (module, source_local, _, _, _) =
            normalize_rebound_handle_copy_probe(&mut db, false, true);
        let smoke = module
            .functions
            .iter()
            .find(|func| func.symbol_name == "smoke")
            .expect("expected smoke helper");
        let source = smoke.body.local(source_local);

        assert!(
            matches!(
                source.pointer_leaf_infos.as_slice(),
                [(
                    path,
                    PointerInfo {
                        address_space: AddressSpaceKind::Memory,
                        target_ty: Some(_),
                    },
                )] if path.is_empty()
            ),
            "copying a less-informed `source` into a pre-specialized `dest` should backfill only the same-site memory handle metadata, got {:?}",
            source.pointer_leaf_infos,
        );
    }

    #[test]
    fn less_informed_source_runtime_shape_stays_stable_after_dest_rebind() {
        let mut db = DriverDataBase::default();
        let (module, source_local, _, expected_source_shape, _) =
            normalize_rebound_handle_copy_probe(&mut db, false, true);
        let smoke = module
            .functions
            .iter()
            .find(|func| func.symbol_name == "smoke")
            .expect("expected smoke helper");
        let source = smoke.body.local(source_local);

        assert_eq!(
            source.runtime_shape, expected_source_shape,
            "copying a less-informed `source` into a pre-specialized `dest` must not let later `dest` assignments reclassify `source`",
        );
    }

    #[test]
    fn incompatible_whole_local_rebinds_do_not_partially_merge_dest_layout_facts() {
        let mut db = DriverDataBase::default();
        let (module, _, dest_local, _, _) =
            normalize_rebound_handle_copy_probe(&mut db, true, false);
        let smoke = module
            .functions
            .iter()
            .find(|func| func.symbol_name == "smoke")
            .expect("expected smoke helper");
        let dest = smoke.body.local(dest_local);

        assert!(
            matches!(
                dest.pointer_leaf_infos.as_slice(),
                [(
                    path,
                    PointerInfo {
                        address_space: AddressSpaceKind::Memory,
                        target_ty: Some(_),
                    },
                )] if path.is_empty()
            ),
            "incompatible whole-local rebinds should not partially rewrite the assignee leaf metadata, got {:?}",
            dest.pointer_leaf_infos,
        );
        assert!(
            matches!(dest.runtime_shape, RuntimeShape::ObjectRef { .. }),
            "incompatible whole-local rebinds should not partially rewrite the assignee runtime shape, got {:?}",
            dest.runtime_shape,
        );
    }

    #[test]
    fn selected_handle_fixtures_do_not_leave_memory_capability_locals_without_leaf_infos() {
        let fixtures = [
            (
                "file:///borrow_handle_forwarding.fe",
                include_str!("../../fe/tests/fixtures/fe_test/borrow_handle_forwarding.fe"),
            ),
            (
                "file:///method_forwards_memory_mut_borrow.fe",
                include_str!(
                    "../../fe/tests/fixtures/fe_test/method_forwards_memory_mut_borrow.fe"
                ),
            ),
            (
                "file:///receiver_returns_non_mem_with_same_typed_memory_arg.fe",
                include_str!(
                    "../../fe/tests/fixtures/fe_test/receiver_returns_non_mem_with_same_typed_memory_arg.fe"
                ),
            ),
            (
                "file:///ref_scalar_nested.fe",
                include_str!("../../fe/tests/fixtures/fe_test/ref_scalar_nested.fe"),
            ),
            (
                "file:///contract_field_mut_borrow_matrix.fe",
                include_str!("../../fe/tests/fixtures/fe_test/contract_field_mut_borrow_matrix.fe"),
            ),
            (
                "file:///option_mut_scalar_match_regression.fe",
                include_str!(
                    "../../fe/tests/fixtures/fe_test/option_mut_scalar_match_regression.fe"
                ),
            ),
        ];

        for (path, src) in fixtures {
            let mut db = DriverDataBase::default();
            let module = lower_inline_module(&mut db, path, src);
            for func in &module.functions {
                for (idx, local) in func.body.locals.iter().enumerate() {
                    if local.address_space == AddressSpaceKind::Memory
                        && matches!(
                            local.runtime_shape,
                            RuntimeShape::MemoryPtr { .. } | RuntimeShape::AddressWord(_)
                        )
                    {
                        assert!(
                            !local.pointer_leaf_infos.is_empty(),
                            "memory pointer-like local v{idx} `{}` in `{}` from `{path}` should carry pointer leaf infos",
                            local.name,
                            func.symbol_name,
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn poseidon_helpers_specialize_array_params_by_root_carrier_space() {
        let mut db = DriverDataBase::default();
        let module = lower_inline_module(
            &mut db,
            "file:///poseidon_mock.fe",
            include_str!("../../fe/tests/fixtures/fe_test/poseidon_mock.fe"),
        );

        let ark_memory = module
            .functions
            .iter()
            .find(|func| func.symbol_name == "ark")
            .expect("expected default memory-backed ark instance");
        let ark_code = module
            .functions
            .iter()
            .find(|func| func.symbol_name.starts_with("ark_arg0_root_code"))
            .expect("expected code-specialized ark instance");

        let memory_param = ark_memory.body.local(ark_memory.body.param_locals[0]);
        assert_eq!(
            memory_param.address_space,
            AddressSpaceKind::Memory,
            "default ark param should stay memory-backed, got {:?}",
            memory_param,
        );
        assert!(
            matches!(memory_param.runtime_shape, RuntimeShape::ObjectRef { .. }),
            "default ark param should stay object-backed, got {:?}",
            memory_param.runtime_shape,
        );

        let code_param = ark_code.body.local(ark_code.body.param_locals[0]);
        assert_eq!(
            code_param.address_space,
            AddressSpaceKind::Code,
            "code-specialized ark param should stay code-backed, got {:?}",
            code_param,
        );
        assert!(
            matches!(code_param.runtime_shape, RuntimeShape::ConstRef { .. }),
            "code-specialized ark param should stay const-backed, got {:?}",
            code_param.runtime_shape,
        );
    }

    #[test]
    fn runtime_shapes_refine_loaded_object_backed_scalar_handles() {
        let mut db = DriverDataBase::default();
        let module = lower_inline_module(
            &mut db,
            "file:///runtime_shapes_refine_loaded_object_backed_scalar_handles.fe",
            include_str!("../../codegen/tests/fixtures/ref_is_not_a_copy.fe"),
        );

        let test_func = module
            .functions
            .iter()
            .find(|func| func.symbol_name == "test_ref_is_not_a_copy")
            .expect("expected ref_is_not_a_copy test");
        let live_view = module
            .functions
            .iter()
            .find(|func| func.symbol_name == "live_view")
            .expect("expected live_view helper");
        assert!(
            matches!(
                live_view.runtime_return_shape,
                RuntimeShape::ObjectRef { .. }
            ),
            "live_view should return an object-backed live view, got {:?}",
            live_view.runtime_return_shape,
        );
        assert_eq!(
            live_view.runtime_return_pointer_leaf_infos.len(),
            2,
            "live_view should preserve field-level return pointer metadata",
        );
        let live_view_ty = test_func
            .body
            .locals
            .iter()
            .find(|local| local.name == "live")
            .expect("expected live local")
            .ty;
        let mut saw_live_view_local = false;
        for func in &module.functions {
            for local in func
                .body
                .locals
                .iter()
                .filter(|local| local.ty == live_view_ty)
            {
                saw_live_view_local = true;
                assert!(
                    matches!(local.runtime_shape, RuntimeShape::ObjectRef { .. }),
                    "{}::{} should stay object-backed, got {:?}",
                    func.symbol_name,
                    local.name,
                    local.runtime_shape,
                );
                assert_eq!(
                    local.pointer_leaf_infos.len(),
                    3,
                    "{}::{} should preserve root and field pointer metadata",
                    func.symbol_name,
                    local.name,
                );
            }
        }
        assert!(saw_live_view_local, "expected at least one LiveView local");
        for local_name in ["live_x", "live_y"] {
            let local = test_func
                .body
                .locals
                .iter()
                .find(|local| local.name == local_name)
                .unwrap_or_else(|| panic!("expected {local_name} local"));
            assert!(
                matches!(local.runtime_shape, RuntimeShape::ObjectRef { .. }),
                "{local_name} should refine to an object ref after loading from an object-backed handle field, got {:?}",
                local.runtime_shape,
            );
        }
    }

    #[test]
    fn runtime_shapes_trace_free_function_local_alias_returns_to_const_sources() {
        let mut db = DriverDataBase::default();
        let module = lower_inline_module(
            &mut db,
            "file:///runtime_shapes_trace_free_function_local_alias_returns_to_const_sources.fe",
            r#"
fn choose_alias(values: [u256; 4], fallback: [u256; 4]) -> [u256; 4] {
    let alias = values
    alias
}

fn read() -> u256 {
    let values: [u256; 4] = [1, 2, 3, 4]
    let mut fallback: [u256; 4] = [5, 6, 7, 8]
    fallback[0] = 9
    let chosen = choose_alias(values, fallback)
    chosen[0]
}
"#,
        );

        let read = module
            .functions
            .iter()
            .find(|func| func.symbol_name == "read")
            .expect("expected read helper");
        let chosen_local_idx = read
            .body
            .locals
            .iter()
            .position(|local| local.name == "chosen")
            .expect("expected chosen local");
        let call_dest = read
            .body
            .blocks
            .iter()
            .flat_map(|block| block.insts.iter())
            .find_map(|inst| match inst {
                MirInst::Assign {
                    dest: Some(dest),
                    rvalue: Rvalue::Call(call),
                    ..
                } if call
                    .resolved_name
                    .as_deref()
                    .is_some_and(|name| name.contains("choose_alias")) =>
                {
                    Some(*dest)
                }
                _ => None,
            })
            .expect("expected choose_alias call");

        assert_eq!(
            call_dest.index(),
            chosen_local_idx,
            "choose_alias call should assign directly into the `chosen` local",
        );

        let chosen = read.body.local(call_dest);
        assert_eq!(
            chosen.address_space,
            AddressSpaceKind::Code,
            "local aliases of proven const-backed args should stay code-backed even with same-typed memory fallbacks",
        );
        assert!(
            matches!(chosen.runtime_shape, RuntimeShape::ConstRef { .. }),
            "free-function local alias returns should stay const-backed, got {:?}",
            chosen.runtime_shape,
        );
        assert!(
            matches!(chosen.const_backing, crate::ir::LocalConstBacking::Const),
            "free-function local alias returns should keep const backing, got {:?}",
            chosen.const_backing,
        );
    }

    #[test]
    fn runtime_shapes_preserve_scalar_param_shapes() {
        let src = include_str!("../../codegen/tests/fixtures/array_mut.fe");

        {
            let mut db = DriverDataBase::default();
            let url =
                Url::parse("file:///runtime_shapes_preserve_scalar_param_shapes_pre.fe").unwrap();
            let file = db.workspace().touch(&mut db, url, Some(src.to_owned()));
            let top_mod = db.top_mod(file);
            let hir_func = top_mod
                .all_funcs(&db)
                .iter()
                .copied()
                .find(|func| {
                    func.name(&db)
                        .to_opt()
                        .is_some_and(|name| name.data(&db) == "array_mut")
                })
                .expect("expected array_mut function");
            let source_arg_tys = hir_func.arg_tys(&db);
            let source_arg_ty = source_arg_tys[0].skip_binder();
            let (diags, typed_body) = check_func_body(&db, hir_func);
            assert!(diags.is_empty(), "unexpected diagnostics: {diags:#?}");
            let core = crate::CoreLib::new(&db, top_mod.scope());
            assert_eq!(
                crate::repr::runtime_shape_for_ty(
                    &db,
                    &core,
                    *source_arg_ty,
                    AddressSpaceKind::Memory,
                ),
                RuntimeShape::Word(RuntimeWordKind::I8),
                "source arg runtime shape should preserve the scalar width",
            );
            let lowered = lower_function(
                &db,
                hir_func,
                typed_body.clone(),
                None,
                Vec::new(),
                Vec::new(),
                Vec::new(),
            )
            .expect("array_mut should lower");
            assert_eq!(
                lowered
                    .body
                    .local(lowered.body.param_locals[0])
                    .runtime_shape,
                RuntimeShape::Unresolved,
                "raw lower_function output should leave local runtime shapes unresolved until normalization",
            );
        }

        {
            let mut db = DriverDataBase::default();
            let module = lower_inline_module(
                &mut db,
                "file:///runtime_shapes_preserve_scalar_param_shapes.fe",
                src,
            );

            let array_mut = module
                .functions
                .iter()
                .find(|func| func.symbol_name == "array_mut")
                .expect("expected array_mut function");
            let param_local = *array_mut
                .body
                .param_locals
                .first()
                .expect("array_mut should have one param");
            let local = array_mut.body.local(param_local);
            assert_eq!(
                local.runtime_shape,
                RuntimeShape::Word(RuntimeWordKind::I8),
                "scalar param runtime shape regressed after the full MIR pipeline",
            )
        }
    }

    #[test]
    fn checked_add_u8_helpers_keep_scalar_runtime_shapes() {
        let mut db = DriverDataBase::default();
        let url = Url::parse("file:///position_lifecycle.fe").expect("valid test url");
        let file = db.workspace().touch(
            &mut db,
            url,
            Some(include_str!("../../fe/tests/fixtures/fe_test/position_lifecycle.fe").to_owned()),
        );
        let top_mod = db.top_mod(file);
        let ingot = top_mod.ingot(&db);
        let mut module = crate::lower_ingot(&db, ingot).expect("ingot should lower");
        let prep = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            crate::prepare_module_for_evm_yul_codegen(&db, &mut module);
        }));

        for func in module
            .functions
            .iter()
            .filter(|func| {
                func.symbol_name.contains("u8")
                    && (func.symbol_name.contains("checked_add")
                        || func.symbol_name.contains("saturating_add")
                        || func.symbol_name.contains("wrappingadd"))
            })
        {
            eprintln!(
                "helper {} ret={:?} locals={:?}",
                func.symbol_name,
                func.runtime_return_shape,
                func.body
                    .locals
                    .iter()
                    .map(|local| (&local.name, local.ty.pretty_print(&db), local.runtime_shape))
                    .collect::<Vec<_>>()
            );
            eprintln!("body {:#?}", func.body.blocks);
        }

        let checked_add = module
            .functions
            .iter()
            .find(|func| func.symbol_name.contains("checked_add_unsigned_impl__u8"))
            .expect("expected checked_add<u8> helper");
        assert!(
            prep.is_ok(),
            "yul prep should not panic before checked_add<u8> inspection"
        );
        assert_eq!(
            checked_add.runtime_return_shape,
            RuntimeShape::Word(RuntimeWordKind::I8),
            "checked_add<u8> return shape must stay scalar, got {:?}",
            checked_add.runtime_return_shape,
        );
        for &param_local in &checked_add.body.param_locals {
            let param = checked_add.body.local(param_local);
            assert_eq!(
                param.runtime_shape,
                RuntimeShape::Word(RuntimeWordKind::I8),
                "checked_add<u8> params must stay scalar, got {:?}",
                param.runtime_shape,
            );
        }
        if let Some(result_local) = checked_add.body.locals.iter().find(|local| local.name == "result") {
            assert_eq!(
                result_local.runtime_shape,
                RuntimeShape::Word(RuntimeWordKind::I8),
                "checked_add<u8>::result must stay scalar, got {:?}",
                result_local.runtime_shape,
            );
        }
    }

    #[test]
    fn runtime_shapes_refine_storage_handle_enum_returns_from_aggregate_stores() {
        let src =
            include_str!("../../fe/tests/fixtures/fe_test/option_mut_scalar_match_regression.fe");
        let mut db = DriverDataBase::default();
        let module = lower_inline_module(
            &mut db,
            "file:///option_mut_scalar_match_regression.fe",
            src,
        );

        let maybe_value_mut = module
            .functions
            .iter()
            .find(|func| func.symbol_name.contains("maybe_value_mut_stor"))
            .expect("expected storage-specialized maybe_value_mut");
        let payload_info = maybe_value_mut
            .runtime_return_pointer_leaf_infos
            .iter()
            .find(|(path, _)| {
                path.iter().any(|projection| {
                    matches!(
                        projection,
                        Projection::VariantField { field_idx, .. } if *field_idx == 0
                    )
                })
            })
            .map(|(_, info)| *info)
            .expect("expected return payload pointer leaf info");

        assert_eq!(
            payload_info.address_space,
            AddressSpaceKind::Storage,
            "storage-specialized enum return payload should preserve storage handle space",
        );
    }

    #[test]
    fn runtime_shapes_preserve_owner_leaf_infos_for_enum_tag_temps() {
        let src =
            include_str!("../../fe/tests/fixtures/fe_test/option_mut_scalar_match_regression.fe");
        let mut db = DriverDataBase::default();
        let module = lower_inline_module(
            &mut db,
            "file:///option_mut_scalar_match_regression.fe",
            src,
        );

        let recv = module
            .functions
            .iter()
            .find(|func| func.symbol_name == "__C_recv_0_0")
            .expect("expected recv helper");
        let discr_local = recv
            .body
            .locals
            .iter()
            .find(|local| matches!(local.runtime_shape, RuntimeShape::EnumTag { .. }))
            .expect("expected enum-tag temp local in recv helper");
        let payload_info = discr_local
            .pointer_leaf_infos
            .iter()
            .find(|(path, _)| {
                path.iter().any(|projection| {
                    matches!(
                        projection,
                        Projection::VariantField { field_idx, .. } if *field_idx == 0
                    )
                })
            })
            .map(|(_, info)| *info)
            .expect("expected enum-tag temp to preserve owner payload leaf info");

        assert_eq!(
            payload_info.address_space,
            AddressSpaceKind::Storage,
            "enum-tag temp should preserve the storage-specialized owner layout",
        );
    }

    #[test]
    fn runtime_shapes_keep_storage_scalar_payload_loads_as_address_words() {
        let src =
            include_str!("../../fe/tests/fixtures/fe_test/option_mut_scalar_match_regression.fe");
        let mut db = DriverDataBase::default();
        let module = lower_inline_module(
            &mut db,
            "file:///option_mut_scalar_match_regression.fe",
            src,
        );

        let recv = module
            .functions
            .iter()
            .find(|func| func.symbol_name == "__C_recv_0_0")
            .expect("expected recv helper");
        let value_local = recv
            .body
            .locals
            .iter()
            .find(|local| local.name == "value")
            .expect("expected Option::Some payload binding local");

        let target_ty = match value_local.runtime_shape {
            RuntimeShape::AddressWord(PointerInfo {
                address_space: AddressSpaceKind::Storage,
                target_ty,
            }) => target_ty,
            other => panic!("unexpected payload binding runtime shape: {other:?}"),
        };
        assert!(
            target_ty.is_some(),
            "storage-backed scalar payload bindings should stay typed storage address words",
        );
    }

    #[test]
    fn runtime_shapes_refine_object_backed_match_discriminant_temps() {
        let src = include_str!("../../fe/tests/fixtures/fe_test/if_let_while_let.fe");
        let mut db = DriverDataBase::default();
        let module = lower_inline_module(&mut db, "file:///if_let_while_let.fe", src);

        for func in &module.functions {
            for block in &func.body.blocks {
                for inst in &block.insts {
                    let MirInst::Assign {
                        dest: Some(dest),
                        rvalue: Rvalue::Load { place },
                        ..
                    } = inst
                    else {
                        continue;
                    };
                    if !matches!(
                        place.projection.iter().last(),
                        Some(Projection::Discriminant)
                    ) {
                        continue;
                    }
                    let local = func.body.local(*dest);
                    assert!(
                        matches!(local.runtime_shape, RuntimeShape::EnumTag { .. }),
                        "discriminant temp v{} `{}` in `{}` should lower as EnumTag, got {:?} for {:?}",
                        dest.index(),
                        local.name,
                        func.symbol_name,
                        local.runtime_shape,
                        place,
                    );
                }
            }
        }
    }

    #[test]
    fn runtime_shapes_keep_projected_scalar_handle_match_bindings_object_backed() {
        let src = include_str!("../../fe/tests/fixtures/fe_test/if_let_while_let.fe");
        let mut db = DriverDataBase::default();
        let module = lower_inline_module(&mut db, "file:///if_let_while_let.fe", src);

        let func = module
            .functions
            .iter()
            .find(|func| func.symbol_name == "sum_if_let")
            .expect("expected sum_if_let");
        let x_local = func
            .body
            .locals
            .iter()
            .enumerate()
            .find(|(_, local)| local.name == "x")
            .expect("expected match binding local `x`");
        assert!(
            matches!(x_local.1.runtime_shape, RuntimeShape::ObjectRef { .. }),
            "match binding local `x` should stay object-backed, got {:?}",
            x_local.1.runtime_shape,
        );
        let value = func
            .body
            .values
            .get(3)
            .expect("expected field projection value for match binding");
        assert!(
            matches!(value.runtime_shape, RuntimeShape::ObjectRef { .. }),
            "projected scalar-handle match binding should normalize to ObjectRef, got {:?} for {:?}",
            value.runtime_shape,
            value.origin,
        );
        assert!(
            matches!(value.origin, ValueOrigin::PlaceRef(_)),
            "expected projected binding value to be a PlaceRef, got {:?}",
            value.origin,
        );
    }

    #[test]
    fn zero_sized_contract_init_params_are_not_lowered_as_handler_params() {
        let src = r#"
msg ZeroMsg {
    #[selector = 0]
    Ping -> u256,
}

pub contract ZeroPayloadContractArgs {
    mut value: u256

    init(dummy: ()) uses (mut value) {
        let kept = dummy
        value = 1
    }

    recv ZeroMsg {
        Ping -> u256 uses (value) {
            value
        }
    }
}
"#;
        let mut db = DriverDataBase::default();
        let module = lower_inline_module(&mut db, "file:///zero_payload_contract_args.fe", src);

        let init_handler = module
            .functions
            .iter()
            .find(|func| func.symbol_name == "__ZeroPayloadContractArgs_init_contract")
            .expect("expected zero-payload init handler");
        assert!(
            init_handler.body.param_locals.is_empty(),
            "zero-sized init params should not remain in the lowered handler signature",
        );
        assert!(
            init_handler
                .body
                .locals
                .iter()
                .any(|local| local.name == "dummy"),
            "zero-sized init param should still be materialized as a body local when referenced",
        );
    }

    #[test]
    fn contract_init_that_only_restores_fresh_storage_zero_is_elided() {
        let src = r#"
msg ZeroStoreMsg {
    #[selector = 0]
    Ping -> u256,
}

struct Store {
    x: u256,
    y: u256,
}

pub contract RedundantZeroInit {
    mut store: Store

    init() uses (mut store) {
        store.x = 0
        store.y = 0
    }

    recv ZeroStoreMsg {
        Ping -> u256 uses (store) {
            store.x + store.y
        }
    }
}
"#;
        let mut db = DriverDataBase::default();
        let module = lower_inline_module(&mut db, "file:///redundant_zero_init.fe", src);

        assert!(
            module
                .functions
                .iter()
                .all(|func| func.symbol_name != "__RedundantZeroInit_init_contract"),
            "redundant zero-only init handlers should be dropped entirely",
        );

        let init_entry = module
            .functions
            .iter()
            .find(|func| func.symbol_name == "__RedundantZeroInit_init")
            .expect("expected init entrypoint");
        assert!(
            init_entry
                .body
                .blocks
                .iter()
                .all(|block| block.insts.iter().all(|inst| match inst {
                    MirInst::Assign {
                        rvalue:
                            Rvalue::Intrinsic {
                                op:
                                    IntrinsicOp::CodeRegionOffset
                                    | IntrinsicOp::CodeRegionLen
                                    | IntrinsicOp::Codecopy
                                    | IntrinsicOp::Callvalue,
                                ..
                            },
                        ..
                    } => true,
                    MirInst::Assign { .. } => false,
                    MirInst::Store { .. }
                    | MirInst::InitAggregate { .. }
                    | MirInst::SetDiscriminant { .. }
                    | MirInst::BindValue { .. } => false,
                })),
            "zero-only init should collapse to the minimal runtime codecopy/return sequence",
        );
    }
}
