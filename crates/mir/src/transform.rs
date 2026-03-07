use hir::analysis::HirAnalysisDb;
use hir::analysis::ty::simplified_pattern::ConstructorKind;
use hir::analysis::ty::ty_def::{PrimTy, TyBase, TyData, TyId};
use hir::hir_def::EnumVariant;
use hir::projection::{IndexSource, Projection};
use rustc_hash::FxHashSet;

use crate::ir::{
    LocalData, MirBody, MirInst, MirProjectionPath, Place, Rvalue, SourceInfoId, TerminatingCall,
    Terminator, ValueData, ValueId, ValueOrigin, ValueRepr,
};
use crate::layout;

mod lower_capability_to_repr;

pub(crate) use lower_capability_to_repr::lower_capability_to_repr;

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
            // Bound the walk defensively (cycles should be impossible).
            for _ in 0..values.len() {
                match values.get(inner.index()).map(|v| &v.origin) {
                    Some(ValueOrigin::TransparentCast { value }) => inner = *value,
                    _ => break,
                }
            }
            values[idx].origin = ValueOrigin::TransparentCast { value: inner };
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
                        pointer_info: crate::ir::try_value_pointer_info_in(
                            values,
                            locals,
                            base_at_point,
                        ),
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
                | ValueOrigin::FuncItem(..)
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
        ValueOrigin::Expr(..)
        | ValueOrigin::ControlFlowResult { .. }
        | ValueOrigin::Unit
        | ValueOrigin::Synthetic(..)
        | ValueOrigin::Local(..)
        | ValueOrigin::PlaceRoot(..)
        | ValueOrigin::FuncItem(..) => Vec::new(),
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
                    Rvalue::ZeroInit | Rvalue::Alloc { .. } | Rvalue::ConstAggregate { .. } => {}
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
    use hir::analysis::ty::ty_def::TyId;
    use url::Url;

    use crate::{
        ir::{
            AddressSpaceKind, BasicBlock, LocalData, MirBody, MirProjectionPath, Place,
            SourceInfoId, Terminator, ValueData, ValueOrigin, ValueRepr,
        },
        lower_module,
    };

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
        });
        let base = body.alloc_value(ValueData {
            ty: wrap_inner_ty,
            origin: ValueOrigin::Local(wrap_local),
            source: SourceInfoId::SYNTHETIC,
            repr: ValueRepr::Ref(AddressSpaceKind::Storage),
            pointer_info: None,
        });
        let value = body.alloc_value(ValueData {
            ty: inner_ty,
            origin: ValueOrigin::Synthetic(crate::ir::SyntheticValue::Int(1u8.into())),
            source: SourceInfoId::SYNTHETIC,
            repr: ValueRepr::Word,
            pointer_info: None,
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
        });
        let root = body.alloc_value(ValueData {
            ty: unit_ty,
            origin: ValueOrigin::Local(local),
            source: SourceInfoId::SYNTHETIC,
            repr: ValueRepr::Word,
            pointer_info: None,
        });
        let place = Place::new(root, MirProjectionPath::new());
        let place_ref = body.alloc_value(ValueData {
            ty: unit_ty,
            origin: ValueOrigin::PlaceRef(place.clone()),
            source: SourceInfoId::SYNTHETIC,
            repr: ValueRepr::Word,
            pointer_info: None,
        });
        let move_out = body.alloc_value(ValueData {
            ty: unit_ty,
            origin: ValueOrigin::MoveOut { place },
            source: SourceInfoId::SYNTHETIC,
            repr: ValueRepr::Word,
            pointer_info: None,
        });

        super::canonicalize_zero_sized(&db, &mut body);

        assert!(matches!(body.value(root).origin, ValueOrigin::Unit));
        assert!(matches!(body.value(place_ref).origin, ValueOrigin::Unit));
        assert!(matches!(body.value(move_out).origin, ValueOrigin::Unit));
    }
}
