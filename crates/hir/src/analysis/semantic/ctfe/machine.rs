use cranelift_entity::EntityRef;
use num_bigint::BigInt;
use num_traits::{One, ToPrimitive, Zero};
use salsa::Update;
use tiny_keccak::{Hasher, Keccak};

use crate::{
    analysis::{
        HirAnalysisDb,
        semantic::instance::{
            GenericSubst, ImplEnv, SemanticInstance, SemanticInstanceKey,
            get_or_build_semantic_instance, instantiate_with_generic_args,
        },
        semantic::{
            FieldIndex, SConst, SExpr, SLocalId, SOperand, SPlace, SStmt, SStmtKind,
            STerminatorKind, SemConstId, SemConstScalar, SemConstValue, SemOrigin, SemanticBody,
            SemanticConstRef, VariantIndex, array_const, bool_const, bytes_const, enum_const,
            int_const, int_ty_shape, normalize_int_to_shape, runtime_size_bytes, sem_const_eq,
            sem_const_from_ty, struct_const, tuple_const, unit_const,
        },
        ty::{
            const_expr::{ConstExpr, ConstExprId},
            const_ty::{ConstTyData, ConstTyId, EvaluatedConstTy, const_ty_from_sem_const},
            corelib::{PrimitiveWrapperCallKind, core_primitive_wrapper_call_kind},
            normalize::normalize_ty,
            ty_check::BodyOwner,
            ty_def::{PrimTy, TyBase, TyData, TyId},
        },
    },
    core::hir_def::expr::LogicalBinOp,
    hir_def::{ArithBinOp, BinOp, CompBinOp, Func, UnOp, attr::ArithmeticMode},
    projection::{IndexSource, Projection},
};

use super::ops::{project_const, store_const};

#[derive(Clone, Debug, PartialEq, Eq, Hash, Update)]
pub struct CtfeConfig {
    pub step_limit: usize,
    pub recursion_limit: usize,
}

impl Default for CtfeConfig {
    fn default() -> Self {
        Self {
            step_limit: 10_000,
            recursion_limit: 64,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Update)]
pub enum CtfeError<'db> {
    NotConstEvaluable {
        origin: SemOrigin<'db>,
    },
    InvalidOperation {
        origin: SemOrigin<'db>,
        message: String,
    },
    InvalidBorrow {
        origin: SemOrigin<'db>,
    },
    InvalidProviderUse {
        origin: SemOrigin<'db>,
    },
    NonConstCall {
        origin: SemOrigin<'db>,
    },
    DivisionByZero {
        origin: SemOrigin<'db>,
    },
    ArithmeticOverflow {
        origin: SemOrigin<'db>,
    },
    NegativeExponent {
        origin: SemOrigin<'db>,
    },
    OutOfBounds {
        origin: SemOrigin<'db>,
    },
    VariantMismatch {
        origin: SemOrigin<'db>,
    },
    UninitializedLocal {
        origin: SemOrigin<'db>,
    },
    StepLimitExceeded {
        origin: SemOrigin<'db>,
    },
    RecursionLimitExceeded {
        origin: SemOrigin<'db>,
    },
    CalleeError {
        origin: SemOrigin<'db>,
        callee: SemanticInstance<'db>,
        source: Box<CtfeError<'db>>,
    },
}

#[salsa::tracked]
pub fn eval_const_instance<'db>(
    db: &'db dyn HirAnalysisDb,
    instance: SemanticInstance<'db>,
) -> Result<SemConstId<'db>, CtfeError<'db>> {
    let mut machine = CtfeMachine::new(db, CtfeConfig::default());
    machine.eval_root(
        instance,
        Vec::new(),
        SemOrigin::Body(instance.key(db).owner(db)),
    )
}

#[salsa::tracked]
pub fn eval_const_ref<'db>(
    db: &'db dyn HirAnalysisDb,
    cref: SemanticConstRef<'db>,
) -> Result<SemConstId<'db>, CtfeError<'db>> {
    let mut machine = CtfeMachine::new(db, CtfeConfig::default());
    machine.eval_root(
        SemanticInstance::new(db, cref.instance(db)),
        Vec::new(),
        cref.origin(db),
    )
}

#[salsa::tracked]
pub fn eval_body_owner_const<'db>(
    db: &'db dyn HirAnalysisDb,
    owner: BodyOwner<'db>,
    generic_args: Vec<crate::analysis::ty::ty_def::TyId<'db>>,
) -> Result<SemConstId<'db>, CtfeError<'db>> {
    let key = SemanticInstanceKey::new(
        db,
        owner,
        GenericSubst::new(db, generic_args),
        crate::analysis::semantic::EffectProviderSubst::empty(db),
        ImplEnv::empty(db, owner.scope()),
    );
    eval_const_instance(db, get_or_build_semantic_instance(db, key))
}

#[salsa::tracked]
pub fn eval_body_owner_const_with_args<'db>(
    db: &'db dyn HirAnalysisDb,
    owner: BodyOwner<'db>,
    generic_args: Vec<crate::analysis::ty::ty_def::TyId<'db>>,
    args: Vec<SemConstId<'db>>,
) -> Result<SemConstId<'db>, CtfeError<'db>> {
    let key = SemanticInstanceKey::new(
        db,
        owner,
        GenericSubst::new(db, generic_args),
        crate::analysis::semantic::EffectProviderSubst::empty(db),
        ImplEnv::empty(db, owner.scope()),
    );
    let instance = get_or_build_semantic_instance(db, key);
    let mut machine = CtfeMachine::new(db, CtfeConfig::default());
    machine.eval_root(
        instance,
        args.into_iter().map(CtfeValue::concrete).collect(),
        SemOrigin::Body(owner),
    )
}

pub(super) fn try_eval_expr_to_const<'db>(
    db: &'db dyn HirAnalysisDb,
    instance: SemanticInstance<'db>,
    result_ty: TyId<'db>,
    expr: &SExpr<'db>,
    locals: &[Option<SemConstId<'db>>],
    origin: SemOrigin<'db>,
) -> Option<SemConstId<'db>> {
    let mut machine = CtfeMachine::new(db, CtfeConfig::default());
    machine
        .eval_expr_with_locals(instance, result_ty, expr.clone(), locals, origin)
        .ok()
}

struct CtfeMachine<'db> {
    db: &'db dyn HirAnalysisDb,
    config: CtfeConfig,
    steps: usize,
    frames: Vec<CtfeFrame<'db>>,
}

struct CtfeFrame<'db> {
    body: SemanticBody<'db>,
    locals: Vec<CtfeSlot<'db>>,
    current: usize,
}

#[derive(Clone)]
enum CtfeSlot<'db> {
    Uninit,
    Init(CtfeValue<'db>),
}

#[derive(Clone)]
enum CtfeValue<'db> {
    Value(CtfeConstValue<'db>),
    Ref(CtfeRef),
}

#[derive(Clone)]
struct CtfeConstValue<'db> {
    value: SemConstId<'db>,
    deferred_origin: Option<SemOrigin<'db>>,
}

impl<'db> CtfeConstValue<'db> {
    fn concrete(value: SemConstId<'db>) -> Self {
        Self {
            value,
            deferred_origin: None,
        }
    }

    fn with_deferred_origin(
        value: SemConstId<'db>,
        deferred_origin: Option<SemOrigin<'db>>,
    ) -> Self {
        Self {
            value,
            deferred_origin,
        }
    }

    fn error_origin(&self, origin: SemOrigin<'db>) -> SemOrigin<'db> {
        self.deferred_origin.unwrap_or(origin)
    }
}

impl<'db> CtfeValue<'db> {
    fn concrete(value: SemConstId<'db>) -> Self {
        Self::Value(CtfeConstValue::concrete(value))
    }

    fn deferred(value: SemConstId<'db>, origin: SemOrigin<'db>) -> Self {
        Self::Value(CtfeConstValue::with_deferred_origin(value, Some(origin)))
    }
}

#[derive(Clone)]
struct CtfeRef {
    frame: usize,
    root: SLocalId,
    path: Box<[CtfePathElem]>,
}

#[derive(Clone)]
pub(super) enum CtfePathElem {
    Field(FieldIndex),
    VariantField {
        variant: VariantIndex,
        field: FieldIndex,
    },
    Index(usize),
}

#[derive(Clone, Copy)]
enum CtfePrimitiveCall {
    Unary(UnOp),
    Binary(BinOp),
}

impl<'db> CtfeMachine<'db> {
    fn new(db: &'db dyn HirAnalysisDb, config: CtfeConfig) -> Self {
        Self {
            db,
            config,
            steps: 0,
            frames: Vec::new(),
        }
    }

    fn eval_root(
        &mut self,
        instance: SemanticInstance<'db>,
        args: Vec<CtfeValue<'db>>,
        origin: SemOrigin<'db>,
    ) -> Result<SemConstId<'db>, CtfeError<'db>> {
        let value = self.eval_instance(instance, args, origin)?;
        let CtfeValue::Value(value) = value else {
            return Err(CtfeError::InvalidBorrow { origin });
        };
        Ok(value.value)
    }

    fn eval_expr_with_locals(
        &mut self,
        instance: SemanticInstance<'db>,
        result_ty: TyId<'db>,
        expr: SExpr<'db>,
        locals: &[Option<SemConstId<'db>>],
        origin: SemOrigin<'db>,
    ) -> Result<SemConstId<'db>, CtfeError<'db>> {
        let body = instance.body(self.db);
        let mut frame_locals = vec![CtfeSlot::Uninit; body.locals.len()];
        for (idx, value) in locals.iter().copied().enumerate() {
            if let Some(value) = value
                && let Some(slot) = frame_locals.get_mut(idx)
            {
                *slot = CtfeSlot::Init(CtfeValue::concrete(value));
            }
        }
        let frame_idx = self.frames.len();
        self.frames.push(CtfeFrame {
            body,
            locals: frame_locals,
            current: 0,
        });
        let result = match self.eval_expr(frame_idx, result_ty, expr, origin)? {
            CtfeValue::Value(value) => Ok(value.value),
            CtfeValue::Ref(_) => Err(CtfeError::InvalidBorrow { origin }),
        };
        self.frames.pop();
        result
    }

    fn eval_instance(
        &mut self,
        instance: SemanticInstance<'db>,
        args: Vec<CtfeValue<'db>>,
        origin: SemOrigin<'db>,
    ) -> Result<CtfeValue<'db>, CtfeError<'db>> {
        self.ensure_const_evaluable(instance, origin)?;
        if self.frames.len() >= self.config.recursion_limit {
            return Err(CtfeError::RecursionLimitExceeded { origin });
        }

        let body = instance.body(self.db);
        let mut locals = vec![CtfeSlot::Uninit; body.locals.len()];
        for (idx, arg) in args.into_iter().enumerate() {
            let Some(slot) = locals.get_mut(idx) else {
                return Err(CtfeError::InvalidOperation {
                    origin,
                    message: "CTFE call arity mismatch".into(),
                });
            };
            *slot = CtfeSlot::Init(arg);
        }
        let frame_idx = self.frames.len();
        self.frames.push(CtfeFrame {
            body,
            locals,
            current: 0,
        });
        let result = self.run_frame(frame_idx);
        self.frames.pop();
        result
    }

    fn ensure_const_evaluable(
        &self,
        instance: SemanticInstance<'db>,
        origin: SemOrigin<'db>,
    ) -> Result<(), CtfeError<'db>> {
        match instance.key(self.db).owner(self.db) {
            BodyOwner::Func(func) if !func.is_const(self.db) => {
                Err(CtfeError::NonConstCall { origin })
            }
            BodyOwner::Func(_) | BodyOwner::Const(_) | BodyOwner::AnonConstBody { .. } => Ok(()),
            BodyOwner::ContractInit { .. } | BodyOwner::ContractRecvArm { .. } => {
                Err(CtfeError::NotConstEvaluable { origin })
            }
        }
    }

    fn run_frame(&mut self, frame_idx: usize) -> Result<CtfeValue<'db>, CtfeError<'db>> {
        loop {
            let block = self.frames[frame_idx].body.blocks[self.frames[frame_idx].current].clone();
            for stmt in block.stmts {
                self.bump(stmt.origin)?;
                self.exec_stmt(frame_idx, stmt)?;
            }
            let term_origin = block.terminator.origin;
            self.bump(term_origin)?;
            match block.terminator.kind {
                STerminatorKind::Goto(bb) => self.frames[frame_idx].current = bb.index(),
                STerminatorKind::Branch {
                    cond,
                    then_bb,
                    else_bb,
                } => {
                    let cond = self.load_value(frame_idx, cond, term_origin)?;
                    let cond = self.expect_bool(frame_idx, cond, term_origin)?;
                    self.frames[frame_idx].current = if cond {
                        then_bb.index()
                    } else {
                        else_bb.index()
                    };
                }
                STerminatorKind::MatchEnum {
                    value,
                    cases,
                    default,
                    ..
                } => {
                    let value = self.load_value(frame_idx, value, term_origin)?;
                    let tag = self.load_enum_variant(value, term_origin)?;
                    self.frames[frame_idx].current = cases
                        .iter()
                        .find(|(variant, _)| *variant == tag)
                        .map_or_else(|| default.map_or(0, |bb| bb.index()), |(_, bb)| bb.index());
                }
                STerminatorKind::Return(Some(value)) => {
                    return self.read_operand(frame_idx, value, term_origin);
                }
                STerminatorKind::Return(None) => {
                    return Ok(CtfeValue::concrete(unit_const(self.db)));
                }
            }
        }
    }

    fn exec_stmt(&mut self, frame_idx: usize, stmt: SStmt<'db>) -> Result<(), CtfeError<'db>> {
        let origin = stmt.origin;
        match stmt.kind {
            SStmtKind::Assign { dst, expr } => {
                let ty = self.frames[frame_idx].body.locals[dst.index()].ty;
                let value = self.eval_expr(frame_idx, ty, expr, origin)?;
                self.frames[frame_idx].locals[dst.index()] = CtfeSlot::Init(value);
            }
            SStmtKind::Store { dst, src } => {
                let place = self.resolve_place(frame_idx, &dst, origin)?;
                let CtfeValue::Value(value) = self.read_operand(frame_idx, src, origin)? else {
                    return Err(CtfeError::InvalidBorrow { origin });
                };
                self.store_place(place, value, origin)?;
            }
        }
        Ok(())
    }

    fn eval_expr(
        &mut self,
        frame_idx: usize,
        result_ty: TyId<'db>,
        expr: SExpr<'db>,
        origin: SemOrigin<'db>,
    ) -> Result<CtfeValue<'db>, CtfeError<'db>> {
        self.bump(origin)?;
        match expr {
            SExpr::Forward(value) | SExpr::UseValue(value) => {
                self.read_operand(frame_idx, value, origin)
            }
            SExpr::CodeRegionRef { .. } => Err(CtfeError::NotConstEvaluable { origin }),
            SExpr::Const(SConst::Value(value)) => Ok(CtfeValue::concrete(value)),
            SExpr::Const(SConst::Ref(cref)) => eval_const_ref(self.db, cref)
                .map(CtfeValue::concrete)
                .map_err(|err| CtfeError::CalleeError {
                    origin: cref.origin(self.db),
                    callee: SemanticInstance::new(self.db, cref.instance(self.db)),
                    source: Box::new(err),
                }),
            SExpr::Unary { op, value } => {
                let value = self.load_value(frame_idx, value, origin)?;
                self.eval_unary(frame_idx, result_ty, op, value, origin)
            }
            SExpr::Binary { op, lhs, rhs } => {
                let lhs = self.load_value(frame_idx, lhs, origin)?;
                let rhs = self.load_value(frame_idx, rhs, origin)?;
                self.eval_binary(frame_idx, result_ty, op, lhs, rhs, origin)
            }
            SExpr::Cast { value, .. } => {
                let value = self.load_value(frame_idx, value, origin)?;
                self.eval_cast(result_ty, value, origin)
            }
            SExpr::AggregateMake { fields, .. } => {
                let fields = self.eval_value_args(frame_idx, &fields, origin)?;
                Ok(self.make_aggregate_value(result_ty, fields))
            }
            SExpr::EnumMake {
                variant, fields, ..
            } => {
                let fields = self.eval_value_args(frame_idx, &fields, origin)?;
                Ok(CtfeValue::concrete(enum_const(
                    self.db,
                    result_ty,
                    variant,
                    fields
                        .into_iter()
                        .map(|field| field.value)
                        .collect::<Vec<_>>()
                        .into_boxed_slice(),
                )))
            }
            SExpr::ReadPlace { place } => {
                let place = self.resolve_place(frame_idx, &place, origin)?;
                let r#ref = CtfeRef {
                    frame: place.frame,
                    root: place.root,
                    path: place.path.into_boxed_slice(),
                };
                self.load_ref_value(&r#ref, origin).map(CtfeValue::Value)
            }
            SExpr::Field { base, field } => {
                let value = self.load_value(frame_idx, base, origin)?;
                self.project_field(value, field, origin)
                    .map(CtfeValue::Value)
            }
            SExpr::Index { base, index } => {
                let value = self.load_value(frame_idx, base, origin)?;
                let index_value = self.load_value(frame_idx, index, origin)?;
                let index = self.index_from_value(frame_idx, index_value, origin)?;
                self.project_index(value, index, origin)
                    .map(CtfeValue::Value)
            }
            SExpr::Borrow {
                place: _,
                provider: Some(_),
                ..
            } => Err(CtfeError::InvalidProviderUse { origin }),
            SExpr::Borrow {
                place,
                provider: None,
                ..
            } => {
                let place = self.resolve_place(frame_idx, &place, origin)?;
                Ok(CtfeValue::Ref(CtfeRef {
                    frame: place.frame,
                    root: place.root,
                    path: place.path.into_boxed_slice(),
                }))
            }
            SExpr::GetEnumTag { value } => {
                let value = self.load_value(frame_idx, value, origin)?;
                let variant = self.load_enum_variant(value, origin)?;
                Ok(CtfeValue::concrete(int_const(
                    self.db,
                    result_ty,
                    BigInt::from(variant.0),
                )))
            }
            SExpr::IsEnumVariant { value, variant } => {
                let value = self.load_value(frame_idx, value, origin)?;
                let actual = self.load_enum_variant(value, origin)?;
                Ok(CtfeValue::concrete(bool_const(self.db, actual == variant)))
            }
            SExpr::ExtractEnumField {
                value,
                variant,
                field,
            } => {
                let value = self.load_value(frame_idx, value, origin)?;
                self.enum_extract(value, variant, field, origin)
                    .map(CtfeValue::Value)
            }
            SExpr::Call {
                callee,
                args,
                effect_args,
            } => {
                if !effect_args.is_empty() {
                    return Err(CtfeError::NotConstEvaluable { origin });
                }
                let args = self
                    .eval_args(frame_idx, &args, origin)?
                    .into_iter()
                    .collect::<Vec<_>>();
                let instance = SemanticInstance::new(self.db, callee.key);
                if let Some(value) =
                    self.try_eval_primitive_call(frame_idx, instance, result_ty, &args, origin)?
                {
                    return Ok(CtfeValue::concrete(value));
                }
                if let BodyOwner::Func(func) = instance.key(self.db).owner(self.db)
                    && func.is_extern(self.db)
                {
                    if !func.is_const(self.db) {
                        return Err(CtfeError::NonConstCall { origin });
                    }
                    let deferred_origin = self.first_deferred_origin(&args).unwrap_or(origin);
                    let value_args = self.materialize_args(args, origin)?;
                    return match self.eval_extern_const_fn(
                        instance,
                        func,
                        result_ty,
                        &value_args,
                        origin,
                    ) {
                        Ok(value) => Ok(CtfeValue::concrete(value)),
                        Err(CtfeError::NotConstEvaluable { .. }) => Ok(CtfeValue::deferred(
                            self.abstract_const_call(
                                ConstExpr::ExternConstFnCall {
                                    func,
                                    generic_args: instance
                                        .key(self.db)
                                        .subst(self.db)
                                        .generic_args(self.db)
                                        .clone(),
                                    args: value_args
                                        .iter()
                                        .copied()
                                        .map(|arg| {
                                            TyId::const_ty(
                                                self.db,
                                                const_ty_from_sem_const(self.db, arg),
                                            )
                                        })
                                        .collect(),
                                },
                                result_ty,
                            ),
                            deferred_origin,
                        )),
                        Err(err) => Err(err),
                    };
                }
                if let BodyOwner::Func(func) = instance.key(self.db).owner(self.db)
                    && !func.is_const(self.db)
                {
                    return Err(CtfeError::NonConstCall { origin });
                }
                match self.eval_instance(instance, args.clone(), origin) {
                    Ok(value) => Ok(value),
                    Err(CtfeError::NotConstEvaluable { .. }) if matches!(instance.key(self.db).owner(self.db), BodyOwner::Func(func) if func.is_const(self.db)) =>
                    {
                        let BodyOwner::Func(func) = instance.key(self.db).owner(self.db) else {
                            unreachable!();
                        };
                        let deferred_origin = self.first_deferred_origin(&args).unwrap_or(origin);
                        let value_args = self.materialize_args(args, origin)?;
                        Ok(CtfeValue::deferred(
                            self.abstract_const_call(
                                ConstExpr::UserConstFnCall {
                                    func,
                                    generic_args: instance
                                        .key(self.db)
                                        .subst(self.db)
                                        .generic_args(self.db)
                                        .clone(),
                                    args: value_args
                                        .iter()
                                        .copied()
                                        .map(|arg| {
                                            TyId::const_ty(
                                                self.db,
                                                const_ty_from_sem_const(self.db, arg),
                                            )
                                        })
                                        .collect(),
                                },
                                result_ty,
                            ),
                            deferred_origin,
                        ))
                    }
                    Err(err) => Err(match err {
                        CtfeError::NotConstEvaluable { .. } => {
                            CtfeError::NotConstEvaluable { origin }
                        }
                        _ => CtfeError::CalleeError {
                            origin,
                            callee: instance,
                            source: Box::new(err),
                        },
                    }),
                }
            }
            SExpr::CodeRegionOffset { .. } | SExpr::CodeRegionLen { .. } => {
                Err(CtfeError::NotConstEvaluable { origin })
            }
        }
    }

    fn try_eval_primitive_call(
        &self,
        frame_idx: usize,
        instance: SemanticInstance<'db>,
        result_ty: TyId<'db>,
        args: &[CtfeValue<'db>],
        origin: SemOrigin<'db>,
    ) -> Result<Option<SemConstId<'db>>, CtfeError<'db>> {
        let BodyOwner::Func(func) = instance.key(self.db).owner(self.db) else {
            return Ok(None);
        };
        let Some(call) = self.primitive_call_kind(func, result_ty) else {
            return Ok(None);
        };
        let value_args = self.materialize_args(args.to_vec(), origin)?;
        if !value_args
            .iter()
            .all(|arg| matches!(arg.value(self.db), SemConstValue::Scalar { .. }))
        {
            return Ok(None);
        }

        let value = match call {
            CtfePrimitiveCall::Unary(op) => {
                let [value] = value_args.as_slice() else {
                    return Ok(None);
                };
                let CtfeValue::Value(value) = self.eval_unary(
                    frame_idx,
                    result_ty,
                    op,
                    CtfeConstValue::concrete(*value),
                    origin,
                )?
                else {
                    return Err(CtfeError::InvalidBorrow { origin });
                };
                value
            }
            CtfePrimitiveCall::Binary(op) => {
                let [lhs, rhs] = value_args.as_slice() else {
                    return Ok(None);
                };
                let CtfeValue::Value(value) = self.eval_binary(
                    frame_idx,
                    result_ty,
                    op,
                    CtfeConstValue::concrete(*lhs),
                    CtfeConstValue::concrete(*rhs),
                    origin,
                )?
                else {
                    return Err(CtfeError::InvalidBorrow { origin });
                };
                value
            }
        };
        Ok(Some(value.value))
    }

    fn primitive_call_kind(
        &self,
        func: Func<'db>,
        result_ty: TyId<'db>,
    ) -> Option<CtfePrimitiveCall> {
        if func.is_extern(self.db) {
            return self.extern_primitive_call_kind(func);
        }
        self.core_ops_wrapper_call_kind(func, result_ty)
    }

    fn extern_primitive_call_kind(&self, func: Func<'db>) -> Option<CtfePrimitiveCall> {
        fn has_numeric_suffix(name: &str) -> bool {
            matches!(
                name,
                "u8" | "u16"
                    | "u32"
                    | "u64"
                    | "u128"
                    | "u256"
                    | "usize"
                    | "i8"
                    | "i16"
                    | "i32"
                    | "i64"
                    | "i128"
                    | "i256"
                    | "isize"
            )
        }

        let name = func.name(self.db).to_opt()?.data(self.db);
        Some(match name.as_str() {
            "__checked_add" => CtfePrimitiveCall::Binary(BinOp::Arith(ArithBinOp::Add)),
            "__checked_sub" => CtfePrimitiveCall::Binary(BinOp::Arith(ArithBinOp::Sub)),
            "__checked_mul" => CtfePrimitiveCall::Binary(BinOp::Arith(ArithBinOp::Mul)),
            "__checked_div" => CtfePrimitiveCall::Binary(BinOp::Arith(ArithBinOp::Div)),
            "__checked_rem" => CtfePrimitiveCall::Binary(BinOp::Arith(ArithBinOp::Rem)),
            "__checked_pow" => CtfePrimitiveCall::Binary(BinOp::Arith(ArithBinOp::Pow)),
            "__checked_neg" => CtfePrimitiveCall::Unary(UnOp::Minus),
            "__not_bool" => CtfePrimitiveCall::Unary(UnOp::Not),
            _ => {
                let suffix = |prefix| {
                    name.strip_prefix(prefix)
                        .filter(|suffix| has_numeric_suffix(suffix))
                };
                if suffix("__shl_").is_some() {
                    CtfePrimitiveCall::Binary(BinOp::Arith(ArithBinOp::LShift))
                } else if suffix("__shr_").is_some() {
                    CtfePrimitiveCall::Binary(BinOp::Arith(ArithBinOp::RShift))
                } else if suffix("__bitand_").is_some() || name == "__bitand_bool" {
                    CtfePrimitiveCall::Binary(BinOp::Arith(ArithBinOp::BitAnd))
                } else if suffix("__bitor_").is_some() || name == "__bitor_bool" {
                    CtfePrimitiveCall::Binary(BinOp::Arith(ArithBinOp::BitOr))
                } else if suffix("__bitxor_").is_some() || name == "__bitxor_bool" {
                    CtfePrimitiveCall::Binary(BinOp::Arith(ArithBinOp::BitXor))
                } else if suffix("__eq_").is_some() || name == "__eq_bool" {
                    CtfePrimitiveCall::Binary(BinOp::Comp(CompBinOp::Eq))
                } else if suffix("__ne_").is_some() || name == "__ne_bool" {
                    CtfePrimitiveCall::Binary(BinOp::Comp(CompBinOp::NotEq))
                } else if suffix("__lt_").is_some() {
                    CtfePrimitiveCall::Binary(BinOp::Comp(CompBinOp::Lt))
                } else if suffix("__le_").is_some() {
                    CtfePrimitiveCall::Binary(BinOp::Comp(CompBinOp::LtEq))
                } else if suffix("__gt_").is_some() {
                    CtfePrimitiveCall::Binary(BinOp::Comp(CompBinOp::Gt))
                } else if suffix("__ge_").is_some() {
                    CtfePrimitiveCall::Binary(BinOp::Comp(CompBinOp::GtEq))
                } else if suffix("__bitnot_").is_some() {
                    CtfePrimitiveCall::Unary(UnOp::BitNot)
                } else {
                    return None;
                }
            }
        })
    }

    fn core_ops_wrapper_call_kind(
        &self,
        func: Func<'db>,
        result_ty: TyId<'db>,
    ) -> Option<CtfePrimitiveCall> {
        match core_primitive_wrapper_call_kind(self.db, func, result_ty)? {
            PrimitiveWrapperCallKind::Unary(op) => Some(CtfePrimitiveCall::Unary(op)),
            PrimitiveWrapperCallKind::Binary(op) => Some(CtfePrimitiveCall::Binary(op)),
            PrimitiveWrapperCallKind::Assign(_) => None,
        }
    }

    fn materialize_args(
        &self,
        args: Vec<CtfeValue<'db>>,
        origin: SemOrigin<'db>,
    ) -> Result<Vec<SemConstId<'db>>, CtfeError<'db>> {
        args.into_iter()
            .map(|arg| match arg {
                CtfeValue::Value(value) => Ok(value.value),
                CtfeValue::Ref(r#ref) => self.load_ref(&r#ref, origin),
            })
            .collect()
    }

    fn first_deferred_origin(&self, args: &[CtfeValue<'db>]) -> Option<SemOrigin<'db>> {
        args.iter().find_map(|arg| match arg {
            CtfeValue::Value(value) => value.deferred_origin,
            CtfeValue::Ref(_) => None,
        })
    }

    fn eval_extern_const_fn(
        &self,
        instance: SemanticInstance<'db>,
        func: crate::hir_def::Func<'db>,
        result_ty: TyId<'db>,
        args: &[SemConstId<'db>],
        origin: SemOrigin<'db>,
    ) -> Result<SemConstId<'db>, CtfeError<'db>> {
        let Some(name) = func.name(self.db).to_opt() else {
            return Err(CtfeError::NotConstEvaluable { origin });
        };

        match name.data(self.db).as_str() {
            "size_of" => self.eval_intrinsic_size_of(instance, result_ty, args, origin),
            "__as_bytes" => self.eval_intrinsic_as_bytes(result_ty, args, origin),
            "__keccak256" => self.eval_intrinsic_keccak(result_ty, args, origin),
            _ => Err(CtfeError::NotConstEvaluable { origin }),
        }
    }

    fn eval_intrinsic_size_of(
        &self,
        instance: SemanticInstance<'db>,
        result_ty: TyId<'db>,
        args: &[SemConstId<'db>],
        origin: SemOrigin<'db>,
    ) -> Result<SemConstId<'db>, CtfeError<'db>> {
        if !args.is_empty() {
            return Err(CtfeError::NotConstEvaluable { origin });
        }
        let ty = *instance
            .key(self.db)
            .subst(self.db)
            .generic_args(self.db)
            .first()
            .ok_or(CtfeError::NotConstEvaluable { origin })?;
        let ty = normalize_ty(
            self.db,
            ty,
            instance.key(self.db).owner(self.db).scope(),
            instance
                .key(self.db)
                .instantiate_typed_body(self.db)
                .assumptions(),
        );
        let size =
            runtime_size_bytes(self.db, ty).ok_or(CtfeError::NotConstEvaluable { origin })?;
        Ok(int_const(self.db, result_ty, BigInt::from(size)))
    }

    fn abstract_const_call(&self, expr: ConstExpr<'db>, result_ty: TyId<'db>) -> SemConstId<'db> {
        let const_ty = ConstTyId::new(
            self.db,
            ConstTyData::Abstract(ConstExprId::new(self.db, expr), result_ty),
        );
        SemConstId::new(
            self.db,
            SemConstValue::TypeLevel {
                ty: result_ty,
                const_ty: TyId::const_ty(self.db, const_ty),
            },
        )
    }

    fn eval_intrinsic_as_bytes(
        &self,
        result_ty: TyId<'db>,
        args: &[SemConstId<'db>],
        origin: SemOrigin<'db>,
    ) -> Result<SemConstId<'db>, CtfeError<'db>> {
        if !is_u8_array_ty(self.db, result_ty) {
            return Err(CtfeError::NotConstEvaluable { origin });
        }
        let [value] = args else {
            return Err(CtfeError::NotConstEvaluable { origin });
        };
        let bytes = self.const_as_bytes(*value, origin)?;
        if let Some(len) = array_len(self.db, result_ty)
            && bytes.len() != len
        {
            return Err(CtfeError::NotConstEvaluable { origin });
        }
        Ok(bytes_const(self.db, result_ty, bytes))
    }

    fn eval_intrinsic_keccak(
        &self,
        result_ty: TyId<'db>,
        args: &[SemConstId<'db>],
        origin: SemOrigin<'db>,
    ) -> Result<SemConstId<'db>, CtfeError<'db>> {
        let [value] = args else {
            return Err(CtfeError::NotConstEvaluable { origin });
        };
        let bytes = self.const_as_bytes(*value, origin)?;
        let mut hasher = Keccak::v256();
        hasher.update(&bytes);
        let mut out = [0u8; 32];
        hasher.finalize(&mut out);
        Ok(int_const(
            self.db,
            result_ty,
            BigInt::from_bytes_be(num_bigint::Sign::Plus, &out),
        ))
    }

    fn eval_args(
        &mut self,
        frame_idx: usize,
        args: &[SOperand],
        origin: SemOrigin<'db>,
    ) -> Result<Vec<CtfeValue<'db>>, CtfeError<'db>> {
        args.iter()
            .map(|arg| self.read_operand(frame_idx, *arg, origin))
            .collect()
    }

    fn eval_value_args(
        &mut self,
        frame_idx: usize,
        args: &[SOperand],
        origin: SemOrigin<'db>,
    ) -> Result<Vec<CtfeConstValue<'db>>, CtfeError<'db>> {
        args.iter()
            .map(|arg| {
                let CtfeValue::Value(value) = self.read_operand(frame_idx, *arg, origin)? else {
                    return Err(CtfeError::InvalidBorrow { origin });
                };
                Ok(value)
            })
            .collect()
    }

    fn read_slot(
        &self,
        frame_idx: usize,
        local: SLocalId,
        origin: SemOrigin<'db>,
    ) -> Result<CtfeValue<'db>, CtfeError<'db>> {
        match self.frames[frame_idx].locals.get(local.index()) {
            Some(CtfeSlot::Init(value)) => Ok(value.clone()),
            Some(CtfeSlot::Uninit) | None => Err(CtfeError::UninitializedLocal { origin }),
        }
    }

    fn read_operand(
        &self,
        frame_idx: usize,
        operand: SOperand,
        origin: SemOrigin<'db>,
    ) -> Result<CtfeValue<'db>, CtfeError<'db>> {
        self.read_slot(frame_idx, operand.value, operand.sem_origin(origin))
    }

    fn load_value(
        &mut self,
        frame_idx: usize,
        operand: SOperand,
        origin: SemOrigin<'db>,
    ) -> Result<CtfeConstValue<'db>, CtfeError<'db>> {
        let origin = operand.sem_origin(origin);
        match self.read_slot(frame_idx, operand.value, origin)? {
            CtfeValue::Value(value) => Ok(value),
            CtfeValue::Ref(r#ref) => self.load_ref_value(&r#ref, origin),
        }
    }

    fn resolve_place(
        &mut self,
        frame_idx: usize,
        place: &SPlace<'db>,
        origin: SemOrigin<'db>,
    ) -> Result<ResolvedPlace, CtfeError<'db>> {
        let mut resolved = match self.read_slot(frame_idx, place.local, origin)? {
            CtfeValue::Ref(r#ref) => ResolvedPlace {
                frame: r#ref.frame,
                root: r#ref.root,
                path: r#ref.path.into_vec(),
            },
            CtfeValue::Value(_) => ResolvedPlace {
                frame: frame_idx,
                root: place.local,
                path: Vec::new(),
            },
        };
        for elem in place.path.iter() {
            resolved.path.push(match elem {
                Projection::Field(field) => {
                    CtfePathElem::Field(FieldIndex((*field).try_into().map_err(|_| {
                        CtfeError::InvalidOperation {
                            origin,
                            message: "field index does not fit in semantic field index".into(),
                        }
                    })?))
                }
                Projection::VariantField {
                    variant, field_idx, ..
                } => CtfePathElem::VariantField {
                    variant: *variant,
                    field: FieldIndex((*field_idx).try_into().map_err(|_| {
                        CtfeError::InvalidOperation {
                            origin,
                            message: "variant field index does not fit in semantic field index"
                                .into(),
                        }
                    })?),
                },
                Projection::Index(IndexSource::Dynamic(index)) => {
                    let index = self.load_value(frame_idx, SOperand::synthetic(*index), origin)?;
                    CtfePathElem::Index(self.index_from_value(frame_idx, index, origin)?)
                }
                Projection::Index(IndexSource::Constant(index)) => CtfePathElem::Index(*index),
                Projection::Deref | Projection::Discriminant => {
                    return Err(CtfeError::InvalidOperation {
                        origin,
                        message: "invalid CTFE place projection".into(),
                    });
                }
            });
        }
        Ok(resolved)
    }

    fn load_ref(
        &self,
        r#ref: &CtfeRef,
        origin: SemOrigin<'db>,
    ) -> Result<SemConstId<'db>, CtfeError<'db>> {
        Ok(self.load_ref_value(r#ref, origin)?.value)
    }

    fn load_ref_value(
        &self,
        r#ref: &CtfeRef,
        origin: SemOrigin<'db>,
    ) -> Result<CtfeConstValue<'db>, CtfeError<'db>> {
        let root = match self.frames[r#ref.frame].locals.get(r#ref.root.index()) {
            Some(CtfeSlot::Init(CtfeValue::Value(value))) => value.clone(),
            Some(CtfeSlot::Init(CtfeValue::Ref(_))) | Some(CtfeSlot::Uninit) | None => {
                return Err(CtfeError::InvalidBorrow { origin });
            }
        };
        self.project_value(root, &r#ref.path, origin)
    }

    fn store_place(
        &mut self,
        place: ResolvedPlace,
        value: CtfeConstValue<'db>,
        origin: SemOrigin<'db>,
    ) -> Result<(), CtfeError<'db>> {
        let root = match self.frames[place.frame].locals.get(place.root.index()) {
            Some(CtfeSlot::Init(CtfeValue::Value(value))) => value.clone(),
            _ => return Err(CtfeError::InvalidBorrow { origin }),
        };
        let updated = store_const(self.db, root.value, &place.path, value.value, origin)?;
        self.frames[place.frame].locals[place.root.index()] = CtfeSlot::Init(CtfeValue::Value(
            self.value_with_origin(updated, value.deferred_origin.or(root.deferred_origin)),
        ));
        Ok(())
    }

    fn project_field(
        &self,
        value: CtfeConstValue<'db>,
        field: FieldIndex,
        origin: SemOrigin<'db>,
    ) -> Result<CtfeConstValue<'db>, CtfeError<'db>> {
        self.project_value(value, &[CtfePathElem::Field(field)], origin)
    }

    fn project_index(
        &self,
        value: CtfeConstValue<'db>,
        index: usize,
        origin: SemOrigin<'db>,
    ) -> Result<CtfeConstValue<'db>, CtfeError<'db>> {
        self.project_value(value, &[CtfePathElem::Index(index)], origin)
    }

    fn enum_extract(
        &self,
        value: CtfeConstValue<'db>,
        variant: VariantIndex,
        field: FieldIndex,
        origin: SemOrigin<'db>,
    ) -> Result<CtfeConstValue<'db>, CtfeError<'db>> {
        let SemConstValue::Enum {
            variant: actual,
            fields,
            ..
        } = value.value.value(self.db)
        else {
            return Err(CtfeError::VariantMismatch {
                origin: value.error_origin(origin),
            });
        };
        if actual != variant {
            return Err(CtfeError::VariantMismatch {
                origin: value.error_origin(origin),
            });
        }
        fields
            .get(field.0 as usize)
            .copied()
            .map(CtfeConstValue::concrete)
            .ok_or(CtfeError::OutOfBounds {
                origin: value.error_origin(origin),
            })
    }

    fn load_enum_variant(
        &self,
        value: CtfeConstValue<'db>,
        origin: SemOrigin<'db>,
    ) -> Result<VariantIndex, CtfeError<'db>> {
        let SemConstValue::Enum { variant, .. } = value.value.value(self.db) else {
            return Err(CtfeError::VariantMismatch {
                origin: value.error_origin(origin),
            });
        };
        Ok(variant)
    }

    fn expect_bool(
        &self,
        frame_idx: usize,
        value: CtfeConstValue<'db>,
        origin: SemOrigin<'db>,
    ) -> Result<bool, CtfeError<'db>> {
        match value.value.value(self.db) {
            SemConstValue::Scalar {
                value: SemConstScalar::Bool(value),
                ..
            } => Ok(value),
            SemConstValue::TypeLevel { ty, const_ty } if ty == TyId::bool(self.db) => {
                let TyData::ConstTy(const_ty) = const_ty.data(self.db) else {
                    return Err(CtfeError::InvalidOperation {
                        origin: value.error_origin(origin),
                        message: format!("expected bool, got {:?}", value.value.value(self.db)),
                    });
                };
                let mut const_ty = const_ty.evaluate(self.db, Some(ty));
                if matches!(const_ty.data(self.db), ConstTyData::Abstract(..)) {
                    let subst = self.frames[frame_idx]
                        .body
                        .owner
                        .key(self.db)
                        .subst(self.db);
                    let instantiated = instantiate_with_generic_args(
                        self.db,
                        TyId::const_ty(self.db, const_ty),
                        subst.generic_args(self.db),
                    );
                    let TyData::ConstTy(instantiated) = instantiated.data(self.db) else {
                        unreachable!("instantiating a const ty must yield a const ty");
                    };
                    const_ty = instantiated.evaluate(self.db, Some(ty));
                }
                let ConstTyData::Evaluated(EvaluatedConstTy::LitBool(value), _) =
                    const_ty.data(self.db)
                else {
                    return Err(CtfeError::InvalidOperation {
                        origin: value.error_origin(origin),
                        message: "expected bool".into(),
                    });
                };
                Ok(*value)
            }
            _ => Err(CtfeError::InvalidOperation {
                origin: value.error_origin(origin),
                message: "expected bool".into(),
            }),
        }
    }

    fn is_bool_like(&self, value: &CtfeConstValue<'db>) -> bool {
        match value.value.value(self.db) {
            SemConstValue::Scalar {
                value: SemConstScalar::Bool(_),
                ..
            } => true,
            SemConstValue::TypeLevel { ty, .. } => ty == TyId::bool(self.db),
            _ => false,
        }
    }

    fn is_int_like(&self, value: &CtfeConstValue<'db>) -> bool {
        match value.value.value(self.db) {
            SemConstValue::Scalar {
                value: SemConstScalar::Int { .. },
                ..
            } => true,
            SemConstValue::TypeLevel { ty, .. } => int_ty_shape(self.db, ty).is_some(),
            _ => false,
        }
    }

    fn expect_int(
        &self,
        frame_idx: usize,
        value: CtfeConstValue<'db>,
        origin: SemOrigin<'db>,
    ) -> Result<BigInt, CtfeError<'db>> {
        match value.value.value(self.db) {
            SemConstValue::Scalar {
                value: SemConstScalar::Int { value },
                ..
            } => Ok(value.clone()),
            SemConstValue::TypeLevel { ty, const_ty } => {
                let TyData::ConstTy(const_ty) = const_ty.data(self.db) else {
                    return Err(CtfeError::InvalidOperation {
                        origin: value.error_origin(origin),
                        message: format!("expected int, got {:?}", value.value.value(self.db)),
                    });
                };
                let mut const_ty = const_ty.evaluate(self.db, Some(ty));
                if matches!(const_ty.data(self.db), ConstTyData::Abstract(..)) {
                    let subst = self.frames[frame_idx]
                        .body
                        .owner
                        .key(self.db)
                        .subst(self.db);
                    let instantiated = instantiate_with_generic_args(
                        self.db,
                        TyId::const_ty(self.db, const_ty),
                        subst.generic_args(self.db),
                    );
                    let TyData::ConstTy(instantiated) = instantiated.data(self.db) else {
                        unreachable!("instantiating a const ty must yield a const ty");
                    };
                    const_ty = instantiated.evaluate(self.db, Some(ty));
                }
                let ConstTyData::Evaluated(EvaluatedConstTy::LitInt(int_id), _) =
                    const_ty.data(self.db)
                else {
                    return Err(CtfeError::InvalidOperation {
                        origin: value.error_origin(origin),
                        message: "expected int".into(),
                    });
                };
                Ok(BigInt::from(int_id.data(self.db).clone()))
            }
            _ => Err(CtfeError::InvalidOperation {
                origin: value.error_origin(origin),
                message: "expected int".into(),
            }),
        }
    }

    fn index_from_value(
        &self,
        frame_idx: usize,
        value: CtfeConstValue<'db>,
        origin: SemOrigin<'db>,
    ) -> Result<usize, CtfeError<'db>> {
        let error_origin = value.error_origin(origin);
        let index = self.expect_int(frame_idx, value, origin)?;
        index.to_usize().ok_or(CtfeError::OutOfBounds {
            origin: error_origin,
        })
    }

    fn eval_unary(
        &self,
        frame_idx: usize,
        result_ty: TyId<'db>,
        op: UnOp,
        value: CtfeConstValue<'db>,
        origin: SemOrigin<'db>,
    ) -> Result<CtfeValue<'db>, CtfeError<'db>> {
        match op {
            UnOp::Plus => Ok(CtfeValue::Value(value)),
            UnOp::Minus => {
                let value = self.expect_int(frame_idx, value, origin)?;
                let value = match self.frames[frame_idx]
                    .body
                    .template_owner
                    .arithmetic_mode(self.db)
                {
                    ArithmeticMode::Checked => {
                        let value = -value;
                        if !self.int_in_range(result_ty, &value) {
                            return Err(CtfeError::ArithmeticOverflow { origin });
                        }
                        value
                    }
                    ArithmeticMode::Unchecked => -value,
                };
                Ok(CtfeValue::concrete(int_const(self.db, result_ty, value)))
            }
            UnOp::Not => Ok(CtfeValue::concrete(bool_const(
                self.db,
                !self.expect_bool(frame_idx, value, origin)?,
            ))),
            UnOp::BitNot => {
                let int = self.expect_int(frame_idx, value, origin)?;
                Ok(CtfeValue::concrete(int_const(
                    self.db,
                    result_ty,
                    -int - BigInt::one(),
                )))
            }
            UnOp::Mut | UnOp::Ref => Err(CtfeError::InvalidOperation {
                origin,
                message: "unexpected borrow operator in CTFE unary evaluation".into(),
            }),
        }
    }

    fn eval_binary(
        &self,
        frame_idx: usize,
        result_ty: TyId<'db>,
        op: BinOp,
        lhs: CtfeConstValue<'db>,
        rhs: CtfeConstValue<'db>,
        origin: SemOrigin<'db>,
    ) -> Result<CtfeValue<'db>, CtfeError<'db>> {
        match op {
            BinOp::Comp(comp) => self.eval_compare(frame_idx, comp, lhs, rhs, origin),
            BinOp::Logical(logical) => {
                let lhs = self.expect_bool(frame_idx, lhs, origin)?;
                let rhs = self.expect_bool(frame_idx, rhs, origin)?;
                let value = match logical {
                    LogicalBinOp::And => lhs && rhs,
                    LogicalBinOp::Or => lhs || rhs,
                };
                Ok(CtfeValue::concrete(bool_const(self.db, value)))
            }
            BinOp::Index => Err(CtfeError::InvalidOperation {
                origin,
                message: "invalid binary op in CTFE expression".into(),
            }),
            BinOp::Arith(ArithBinOp::Range) => Err(CtfeError::NotConstEvaluable { origin }),
            BinOp::Arith(arith) => {
                if (matches!(lhs.value.value(self.db), SemConstValue::TypeLevel { .. })
                    || matches!(rhs.value.value(self.db), SemConstValue::TypeLevel { .. }))
                    && let Some(value) = self.eval_type_level_binary(
                        result_ty,
                        arith,
                        lhs.clone(),
                        rhs.clone(),
                        origin,
                    )
                {
                    return Ok(CtfeValue::Value(value));
                }
                if self.is_bool_like(&lhs) && self.is_bool_like(&rhs) {
                    return match arith {
                        ArithBinOp::BitAnd => Ok(CtfeValue::concrete(bool_const(
                            self.db,
                            self.expect_bool(frame_idx, lhs, origin)?
                                & self.expect_bool(frame_idx, rhs, origin)?,
                        ))),
                        ArithBinOp::BitOr => Ok(CtfeValue::concrete(bool_const(
                            self.db,
                            self.expect_bool(frame_idx, lhs, origin)?
                                | self.expect_bool(frame_idx, rhs, origin)?,
                        ))),
                        ArithBinOp::BitXor => Ok(CtfeValue::concrete(bool_const(
                            self.db,
                            self.expect_bool(frame_idx, lhs, origin)?
                                ^ self.expect_bool(frame_idx, rhs, origin)?,
                        ))),
                        _ => Err(CtfeError::InvalidOperation {
                            origin,
                            message: "expected int".into(),
                        }),
                    };
                }
                let lhs = self.expect_int(frame_idx, lhs, origin)?;
                let rhs = self.expect_int(frame_idx, rhs, origin)?;
                let arithmetic_mode = self.frames[frame_idx]
                    .body
                    .template_owner
                    .arithmetic_mode(self.db);
                let value = match arith {
                    ArithBinOp::Add => {
                        let value = lhs + rhs;
                        if arithmetic_mode == ArithmeticMode::Checked
                            && !self.int_in_range(result_ty, &value)
                        {
                            return Err(CtfeError::ArithmeticOverflow { origin });
                        }
                        value
                    }
                    ArithBinOp::Sub => {
                        let value = lhs - rhs;
                        if arithmetic_mode == ArithmeticMode::Checked
                            && !self.int_in_range(result_ty, &value)
                        {
                            return Err(CtfeError::ArithmeticOverflow { origin });
                        }
                        value
                    }
                    ArithBinOp::Mul => {
                        let value = lhs * rhs;
                        if arithmetic_mode == ArithmeticMode::Checked
                            && !self.int_in_range(result_ty, &value)
                        {
                            return Err(CtfeError::ArithmeticOverflow { origin });
                        }
                        value
                    }
                    ArithBinOp::Div => {
                        if rhs.is_zero() {
                            return Err(CtfeError::DivisionByZero { origin });
                        }
                        if arithmetic_mode == ArithmeticMode::Checked
                            && let Some((bits, true)) = int_ty_shape(self.db, result_ty)
                            && lhs == -(BigInt::one() << (usize::from(bits) - 1))
                            && rhs == -BigInt::one()
                        {
                            return Err(CtfeError::ArithmeticOverflow { origin });
                        }
                        lhs / rhs
                    }
                    ArithBinOp::Rem => {
                        if rhs.is_zero() {
                            return Err(CtfeError::DivisionByZero { origin });
                        }
                        lhs % rhs
                    }
                    ArithBinOp::Pow => {
                        if rhs.sign() == num_bigint::Sign::Minus {
                            return Err(CtfeError::NegativeExponent { origin });
                        }
                        let Some(exp) = rhs.to_biguint() else {
                            return Err(CtfeError::NegativeExponent { origin });
                        };
                        if arithmetic_mode == ArithmeticMode::Checked {
                            let mut acc = BigInt::one();
                            let mut base = lhs;
                            let mut exp = exp;
                            while !exp.is_zero() {
                                if (&exp & num_bigint::BigUint::one()) == num_bigint::BigUint::one()
                                {
                                    acc *= base.clone();
                                    if !self.int_in_range(result_ty, &acc) {
                                        return Err(CtfeError::ArithmeticOverflow { origin });
                                    }
                                }
                                exp >>= 1usize;
                                if exp.is_zero() {
                                    break;
                                }
                                base = base.clone() * base;
                                if !self.int_in_range(result_ty, &base) {
                                    return Err(CtfeError::ArithmeticOverflow { origin });
                                }
                            }
                            acc
                        } else {
                            lhs.pow(exp.to_u32().ok_or(CtfeError::InvalidOperation {
                                origin,
                                message: "invalid power exponent".into(),
                            })?)
                        }
                    }
                    ArithBinOp::LShift => {
                        lhs << rhs.to_usize().ok_or(CtfeError::InvalidOperation {
                            origin,
                            message: "invalid left shift amount".into(),
                        })?
                    }
                    ArithBinOp::RShift => {
                        lhs >> rhs.to_usize().ok_or(CtfeError::InvalidOperation {
                            origin,
                            message: "invalid right shift amount".into(),
                        })?
                    }
                    ArithBinOp::BitAnd => {
                        self.bitwise(result_ty, lhs, rhs, |lhs, rhs| lhs & rhs)?
                    }
                    ArithBinOp::BitOr => self.bitwise(result_ty, lhs, rhs, |lhs, rhs| lhs | rhs)?,
                    ArithBinOp::BitXor => {
                        self.bitwise(result_ty, lhs, rhs, |lhs, rhs| lhs ^ rhs)?
                    }
                    ArithBinOp::Range => unreachable!(),
                };
                Ok(CtfeValue::concrete(int_const(self.db, result_ty, value)))
            }
        }
    }

    fn eval_compare(
        &self,
        frame_idx: usize,
        op: CompBinOp,
        lhs: CtfeConstValue<'db>,
        rhs: CtfeConstValue<'db>,
        origin: SemOrigin<'db>,
    ) -> Result<CtfeValue<'db>, CtfeError<'db>> {
        let result = if self.is_bool_like(&lhs) && self.is_bool_like(&rhs) {
            let lhs = self.expect_bool(frame_idx, lhs, origin)?;
            let rhs = self.expect_bool(frame_idx, rhs, origin)?;
            match op {
                CompBinOp::Eq => lhs == rhs,
                CompBinOp::NotEq => lhs != rhs,
                CompBinOp::Lt => !lhs && rhs,
                CompBinOp::LtEq => !lhs || rhs,
                CompBinOp::Gt => lhs && !rhs,
                CompBinOp::GtEq => lhs || !rhs,
            }
        } else if self.is_int_like(&lhs) && self.is_int_like(&rhs) {
            let lhs = self.expect_int(frame_idx, lhs, origin)?;
            let rhs = self.expect_int(frame_idx, rhs, origin)?;
            match op {
                CompBinOp::Eq => lhs == rhs,
                CompBinOp::NotEq => lhs != rhs,
                CompBinOp::Lt => lhs < rhs,
                CompBinOp::LtEq => lhs <= rhs,
                CompBinOp::Gt => lhs > rhs,
                CompBinOp::GtEq => lhs >= rhs,
            }
        } else {
            match op {
                CompBinOp::Eq => sem_const_eq(self.db, lhs.value, rhs.value),
                CompBinOp::NotEq => !sem_const_eq(self.db, lhs.value, rhs.value),
                CompBinOp::Lt => {
                    self.expect_int(frame_idx, lhs, origin)?
                        < self.expect_int(frame_idx, rhs, origin)?
                }
                CompBinOp::LtEq => {
                    self.expect_int(frame_idx, lhs, origin)?
                        <= self.expect_int(frame_idx, rhs, origin)?
                }
                CompBinOp::Gt => {
                    self.expect_int(frame_idx, lhs, origin)?
                        > self.expect_int(frame_idx, rhs, origin)?
                }
                CompBinOp::GtEq => {
                    self.expect_int(frame_idx, lhs, origin)?
                        >= self.expect_int(frame_idx, rhs, origin)?
                }
            }
        };
        Ok(CtfeValue::concrete(bool_const(self.db, result)))
    }

    fn eval_type_level_binary(
        &self,
        result_ty: TyId<'db>,
        op: ArithBinOp,
        lhs: CtfeConstValue<'db>,
        rhs: CtfeConstValue<'db>,
        origin: SemOrigin<'db>,
    ) -> Option<CtfeConstValue<'db>> {
        let lhs_ty = TyId::const_ty(self.db, const_ty_from_sem_const(self.db, lhs.value));
        let rhs_ty = TyId::const_ty(self.db, const_ty_from_sem_const(self.db, rhs.value));
        let expr = ConstExprId::new(
            self.db,
            ConstExpr::ArithBinOp {
                op,
                lhs: lhs_ty,
                rhs: rhs_ty,
            },
        );
        let const_ty = ConstTyId::new(self.db, ConstTyData::Abstract(expr, result_ty))
            .evaluate(self.db, Some(result_ty));
        sem_const_from_ty(self.db, TyId::const_ty(self.db, const_ty)).map(|value| {
            self.value_with_origin(
                value,
                lhs.deferred_origin.or(rhs.deferred_origin).or(Some(origin)),
            )
        })
    }

    fn bitwise(
        &self,
        result_ty: TyId<'db>,
        lhs: BigInt,
        rhs: BigInt,
        op: impl Fn(BigInt, BigInt) -> BigInt,
    ) -> Result<BigInt, CtfeError<'db>> {
        let Some((bits, signed)) = int_ty_shape(self.db, result_ty) else {
            return Ok(op(lhs, rhs));
        };
        let lhs = normalize_int_to_shape(lhs, bits, false);
        let rhs = normalize_int_to_shape(rhs, bits, false);
        Ok(normalize_int_to_shape(op(lhs, rhs), bits, signed))
    }

    fn int_in_range(&self, result_ty: TyId<'db>, value: &BigInt) -> bool {
        let Some((bits, signed)) = int_ty_shape(self.db, result_ty) else {
            return true;
        };
        if signed {
            let half = BigInt::one() << (usize::from(bits) - 1);
            let min = -half.clone();
            let max = half - BigInt::one();
            value >= &min && value <= &max
        } else {
            value >= &BigInt::zero() && value < &(BigInt::one() << usize::from(bits))
        }
    }

    fn eval_cast(
        &self,
        result_ty: TyId<'db>,
        value: CtfeConstValue<'db>,
        origin: SemOrigin<'db>,
    ) -> Result<CtfeValue<'db>, CtfeError<'db>> {
        match value.value.value(self.db) {
            SemConstValue::Scalar {
                value: SemConstScalar::Bool(value),
                ..
            } if int_ty_shape(self.db, result_ty).is_some() => Ok(CtfeValue::concrete(int_const(
                self.db,
                result_ty,
                if value { BigInt::one() } else { BigInt::zero() },
            ))),
            SemConstValue::Scalar {
                value: SemConstScalar::Int { value },
                ..
            } if result_ty == TyId::bool(self.db) => {
                Ok(CtfeValue::concrete(bool_const(self.db, !value.is_zero())))
            }
            SemConstValue::Scalar {
                value: SemConstScalar::Int { value },
                ..
            } if int_ty_shape(self.db, result_ty).is_some() => Ok(CtfeValue::concrete(int_const(
                self.db,
                result_ty,
                value.clone(),
            ))),
            SemConstValue::Scalar {
                value: SemConstScalar::Bytes(bytes),
                ..
            } => Ok(CtfeValue::concrete(bytes_const(
                self.db,
                result_ty,
                bytes.clone(),
            ))),
            _ => Err(CtfeError::InvalidOperation {
                origin: value.error_origin(origin),
                message: "unsupported cast in CTFE".into(),
            }),
        }
    }

    fn make_aggregate_value(
        &self,
        result_ty: TyId<'db>,
        fields: Vec<CtfeConstValue<'db>>,
    ) -> CtfeValue<'db> {
        let deferred_origin = fields.iter().find_map(|field| field.deferred_origin);
        let fields = fields
            .into_iter()
            .map(|field| field.value)
            .collect::<Vec<_>>()
            .into_boxed_slice();
        let value = if result_ty.is_tuple(self.db) {
            tuple_const(self.db, result_ty, fields)
        } else if result_ty.is_array(self.db) {
            array_const(self.db, result_ty, fields)
        } else {
            struct_const(self.db, result_ty, fields)
        };
        CtfeValue::Value(self.value_with_origin(value, deferred_origin))
    }

    fn project_value(
        &self,
        value: CtfeConstValue<'db>,
        path: &[CtfePathElem],
        origin: SemOrigin<'db>,
    ) -> Result<CtfeConstValue<'db>, CtfeError<'db>> {
        if matches!(value.value.value(self.db), SemConstValue::TypeLevel { .. }) {
            return Err(CtfeError::InvalidOperation {
                origin: value.error_origin(origin),
                message: "invalid const projection".into(),
            });
        }
        project_const(self.db, value.value, path, origin)
            .map(|projected| self.value_with_origin(projected, value.deferred_origin))
    }

    fn value_with_origin(
        &self,
        value: SemConstId<'db>,
        deferred_origin: Option<SemOrigin<'db>>,
    ) -> CtfeConstValue<'db> {
        CtfeConstValue::with_deferred_origin(
            value,
            self.sem_const_contains_type_level(value)
                .then_some(())
                .and(deferred_origin),
        )
    }

    fn sem_const_contains_type_level(&self, value: SemConstId<'db>) -> bool {
        match value.value(self.db) {
            SemConstValue::TypeLevel { .. } => true,
            SemConstValue::Tuple { elems, .. }
            | SemConstValue::Struct { fields: elems, .. }
            | SemConstValue::Array { elems, .. }
            | SemConstValue::Enum { fields: elems, .. } => elems
                .iter()
                .copied()
                .any(|elem| self.sem_const_contains_type_level(elem)),
            SemConstValue::Unit | SemConstValue::Scalar { .. } => false,
        }
    }

    fn bump(&mut self, origin: SemOrigin<'db>) -> Result<(), CtfeError<'db>> {
        self.steps += 1;
        if self.steps > self.config.step_limit {
            return Err(CtfeError::StepLimitExceeded { origin });
        }
        Ok(())
    }

    fn const_as_bytes(
        &self,
        value: SemConstId<'db>,
        origin: SemOrigin<'db>,
    ) -> Result<Vec<u8>, CtfeError<'db>> {
        match value.value(self.db) {
            SemConstValue::Scalar {
                value: SemConstScalar::Bool(flag),
                ..
            } => Ok(vec![u8::from(flag)]),
            SemConstValue::Scalar {
                ty,
                value: SemConstScalar::Int { value },
            } => {
                let Some((bits, _)) = int_ty_shape(self.db, ty) else {
                    return Err(CtfeError::NotConstEvaluable { origin });
                };
                let width = usize::from(bits / 8);
                let (_, bytes) = normalize_int_to_shape(value.clone(), bits, false).to_bytes_be();
                if bytes.len() > width {
                    return Err(CtfeError::NotConstEvaluable { origin });
                }
                let mut out = vec![0u8; width];
                let offset = width - bytes.len();
                out[offset..].copy_from_slice(&bytes);
                Ok(out)
            }
            SemConstValue::Scalar {
                value: SemConstScalar::Bytes(bytes),
                ..
            } => Ok(bytes.clone()),
            SemConstValue::Tuple { elems, .. }
            | SemConstValue::Struct { fields: elems, .. }
            | SemConstValue::Array { elems, .. } => {
                let mut out = Vec::new();
                for elem in elems.iter().copied() {
                    out.extend(self.const_as_bytes(elem, origin)?);
                }
                Ok(out)
            }
            SemConstValue::Enum { ty, variant, .. } if ty.is_unit_variant_only_enum(self.db) => {
                let width = 32;
                let (_, bytes) = BigInt::from(variant.0).to_bytes_be();
                let mut out = vec![0u8; width];
                let offset = width - bytes.len();
                out[offset..].copy_from_slice(&bytes);
                Ok(out)
            }
            SemConstValue::Unit | SemConstValue::TypeLevel { .. } | SemConstValue::Enum { .. } => {
                Err(CtfeError::NotConstEvaluable { origin })
            }
        }
    }
}

#[derive(Clone)]
struct ResolvedPlace {
    frame: usize,
    root: SLocalId,
    path: Vec<CtfePathElem>,
}

fn is_u8_array_ty<'db>(db: &'db dyn HirAnalysisDb, ty: TyId<'db>) -> bool {
    if !ty.is_array(db) {
        return false;
    }
    let (_, args) = ty.decompose_ty_app(db);
    matches!(
        args.first().copied().map(|ty| ty.base_ty(db).data(db)),
        Some(TyData::TyBase(TyBase::Prim(PrimTy::U8)))
    )
}

fn array_len<'db>(db: &'db dyn HirAnalysisDb, ty: TyId<'db>) -> Option<usize> {
    let (_, args) = ty.decompose_ty_app(db);
    let TyData::ConstTy(const_ty) = args.get(1)?.data(db) else {
        return None;
    };
    match const_ty.data(db) {
        ConstTyData::Evaluated(EvaluatedConstTy::LitInt(int_id), _) => int_id.data(db).to_usize(),
        _ => None,
    }
}
