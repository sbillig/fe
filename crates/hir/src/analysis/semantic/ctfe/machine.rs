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
            FieldIndex, SConst, SExpr, SLocalId, SPlace, SPlaceElem, SStmt, SStmtKind,
            STerminatorKind, SemConstId, SemConstScalar, SemConstValue, SemOrigin, SemanticBody,
            SemanticConstRef, VariantIndex, array_const, bool_const, bytes_const, enum_const,
            int_const, int_ty_shape, normalize_int_to_shape, runtime_size_bytes, sem_const_from_ty,
            struct_const, tuple_const, unit_const,
        },
        ty::{
            const_expr::{ConstExpr, ConstExprId},
            const_ty::{ConstTyData, ConstTyId, EvaluatedConstTy, const_ty_from_sem_const},
            normalize::normalize_ty,
            ty_check::BodyOwner,
            ty_def::{PrimTy, TyBase, TyData, TyId},
        },
    },
    hir_def::{ArithBinOp, BinOp, CompBinOp, UnOp},
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
    DivisionByZero {
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
        ImplEnv::empty(db, owner.scope()),
    );
    eval_const_instance(db, get_or_build_semantic_instance(db, key))
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
    Value(SemConstId<'db>),
    Ref(CtfeRef),
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
    Index(usize),
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
        Ok(value)
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
                *slot = CtfeSlot::Init(CtfeValue::Value(value));
            }
        }
        let frame_idx = self.frames.len();
        self.frames.push(CtfeFrame {
            body,
            locals: frame_locals,
            current: 0,
        });
        let result = match self.eval_expr(frame_idx, result_ty, expr, origin)? {
            CtfeValue::Value(value) => Ok(value),
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
        let result = self.run_frame(frame_idx, origin);
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
                Err(CtfeError::NotConstEvaluable { origin })
            }
            BodyOwner::Func(_) | BodyOwner::Const(_) | BodyOwner::AnonConstBody { .. } => Ok(()),
            BodyOwner::ContractInit { .. } | BodyOwner::ContractRecvArm { .. } => {
                Err(CtfeError::NotConstEvaluable { origin })
            }
        }
    }

    fn run_frame(
        &mut self,
        frame_idx: usize,
        origin: SemOrigin<'db>,
    ) -> Result<CtfeValue<'db>, CtfeError<'db>> {
        loop {
            let block = self.frames[frame_idx].body.blocks[self.frames[frame_idx].current].clone();
            for stmt in block.stmts {
                self.bump(origin)?;
                self.exec_stmt(frame_idx, stmt, origin)?;
            }
            self.bump(origin)?;
            match block.terminator.kind {
                STerminatorKind::Goto(bb) => self.frames[frame_idx].current = bb.index(),
                STerminatorKind::Branch {
                    cond,
                    then_bb,
                    else_bb,
                } => {
                    let cond = self.load_value(frame_idx, cond, origin)?;
                    let cond = self.expect_bool(cond, origin)?;
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
                    let value = self.load_value(frame_idx, value, origin)?;
                    let tag = self.load_enum_variant(value, origin)?;
                    self.frames[frame_idx].current = cases
                        .iter()
                        .find(|(variant, _)| *variant == tag)
                        .map_or_else(|| default.map_or(0, |bb| bb.index()), |(_, bb)| bb.index());
                }
                STerminatorKind::Return(Some(value)) => {
                    return self.read_slot(frame_idx, value, origin);
                }
                STerminatorKind::Return(None) => return Ok(CtfeValue::Value(unit_const(self.db))),
            }
        }
    }

    fn exec_stmt(
        &mut self,
        frame_idx: usize,
        stmt: SStmt<'db>,
        origin: SemOrigin<'db>,
    ) -> Result<(), CtfeError<'db>> {
        match stmt.kind {
            SStmtKind::Assign { dst, expr } => {
                let ty = self.frames[frame_idx].body.locals[dst.index()].ty;
                let value = self.eval_expr(frame_idx, ty, expr, origin)?;
                self.frames[frame_idx].locals[dst.index()] = CtfeSlot::Init(value);
            }
            SStmtKind::Store { dst, src } => {
                let place = self.resolve_place(frame_idx, &dst, origin)?;
                let CtfeValue::Value(value) = self.read_slot(frame_idx, src, origin)? else {
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
            SExpr::Use(value) => self.read_slot(frame_idx, value, origin),
            SExpr::Const(SConst::Value(value)) => Ok(CtfeValue::Value(value)),
            SExpr::Const(SConst::Ref(cref)) => eval_const_ref(self.db, cref)
                .map(CtfeValue::Value)
                .map_err(|_| CtfeError::CalleeError {
                    origin: cref.origin(self.db),
                    callee: SemanticInstance::new(self.db, cref.instance(self.db)),
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
                Ok(CtfeValue::Value(self.make_aggregate(result_ty, fields)))
            }
            SExpr::EnumMake {
                variant, fields, ..
            } => {
                let fields = self.eval_value_args(frame_idx, &fields, origin)?;
                Ok(CtfeValue::Value(enum_const(
                    self.db,
                    result_ty,
                    variant,
                    fields.into_boxed_slice(),
                )))
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
                Ok(CtfeValue::Value(int_const(
                    self.db,
                    result_ty,
                    BigInt::from(variant.0),
                )))
            }
            SExpr::IsEnumVariant { value, variant } => {
                let value = self.load_value(frame_idx, value, origin)?;
                let actual = self.load_enum_variant(value, origin)?;
                Ok(CtfeValue::Value(bool_const(self.db, actual == variant)))
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
                if let BodyOwner::Func(func) = instance.key(self.db).owner(self.db)
                    && func.is_extern(self.db)
                {
                    let value_args = self.materialize_args(args, origin)?;
                    return match self.eval_extern_const_fn(
                        instance,
                        func,
                        result_ty,
                        &value_args,
                        origin,
                    ) {
                        Ok(value) => Ok(CtfeValue::Value(value)),
                        Err(CtfeError::NotConstEvaluable { .. }) => Ok(CtfeValue::Value(
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
                        )),
                        Err(err) => Err(err),
                    };
                }
                match self.eval_instance(instance, args.clone(), origin) {
                    Ok(value) => Ok(value),
                    Err(CtfeError::NotConstEvaluable { .. })
                        if matches!(instance.key(self.db).owner(self.db), BodyOwner::Func(_)) =>
                    {
                        let BodyOwner::Func(func) = instance.key(self.db).owner(self.db) else {
                            unreachable!();
                        };
                        let value_args = self.materialize_args(args, origin)?;
                        Ok(CtfeValue::Value(
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
                        ))
                    }
                    Err(err) => Err(match err {
                        CtfeError::NotConstEvaluable { .. } => {
                            CtfeError::NotConstEvaluable { origin }
                        }
                        _ => CtfeError::CalleeError {
                            origin,
                            callee: instance,
                        },
                    }),
                }
            }
            SExpr::CodeRegionOffset { .. } | SExpr::CodeRegionLen { .. } => {
                Err(CtfeError::NotConstEvaluable { origin })
            }
        }
    }

    fn materialize_args(
        &self,
        args: Vec<CtfeValue<'db>>,
        origin: SemOrigin<'db>,
    ) -> Result<Vec<SemConstId<'db>>, CtfeError<'db>> {
        args.into_iter()
            .map(|arg| match arg {
                CtfeValue::Value(value) => Ok(value),
                CtfeValue::Ref(r#ref) => self.load_ref(&r#ref, origin),
            })
            .collect()
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
        args: &[SLocalId],
        origin: SemOrigin<'db>,
    ) -> Result<Vec<CtfeValue<'db>>, CtfeError<'db>> {
        args.iter()
            .map(|arg| self.read_slot(frame_idx, *arg, origin))
            .collect()
    }

    fn eval_value_args(
        &mut self,
        frame_idx: usize,
        args: &[SLocalId],
        origin: SemOrigin<'db>,
    ) -> Result<Vec<SemConstId<'db>>, CtfeError<'db>> {
        args.iter()
            .map(|arg| {
                let CtfeValue::Value(value) = self.read_slot(frame_idx, *arg, origin)? else {
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

    fn load_value(
        &mut self,
        frame_idx: usize,
        local: SLocalId,
        origin: SemOrigin<'db>,
    ) -> Result<SemConstId<'db>, CtfeError<'db>> {
        match self.read_slot(frame_idx, local, origin)? {
            CtfeValue::Value(value) => Ok(value),
            CtfeValue::Ref(r#ref) => self.load_ref(&r#ref, origin),
        }
    }

    fn resolve_place(
        &mut self,
        frame_idx: usize,
        place: &SPlace,
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
                SPlaceElem::Field(field) => CtfePathElem::Field(*field),
                SPlaceElem::Index(index) => {
                    let index = self.load_value(frame_idx, *index, origin)?;
                    CtfePathElem::Index(self.index_from_value(frame_idx, index, origin)?)
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
        let root = match self.frames[r#ref.frame].locals.get(r#ref.root.index()) {
            Some(CtfeSlot::Init(CtfeValue::Value(value))) => *value,
            Some(CtfeSlot::Init(CtfeValue::Ref(_))) | Some(CtfeSlot::Uninit) | None => {
                return Err(CtfeError::InvalidBorrow { origin });
            }
        };
        project_const(self.db, root, &r#ref.path, origin)
    }

    fn store_place(
        &mut self,
        place: ResolvedPlace,
        value: SemConstId<'db>,
        origin: SemOrigin<'db>,
    ) -> Result<(), CtfeError<'db>> {
        let root = match self.frames[place.frame].locals.get(place.root.index()) {
            Some(CtfeSlot::Init(CtfeValue::Value(value))) => *value,
            _ => return Err(CtfeError::InvalidBorrow { origin }),
        };
        let updated = store_const(self.db, root, &place.path, value, origin)?;
        self.frames[place.frame].locals[place.root.index()] =
            CtfeSlot::Init(CtfeValue::Value(updated));
        Ok(())
    }

    fn project_field(
        &self,
        value: SemConstId<'db>,
        field: FieldIndex,
        origin: SemOrigin<'db>,
    ) -> Result<SemConstId<'db>, CtfeError<'db>> {
        project_const(self.db, value, &[CtfePathElem::Field(field)], origin)
    }

    fn project_index(
        &self,
        value: SemConstId<'db>,
        index: usize,
        origin: SemOrigin<'db>,
    ) -> Result<SemConstId<'db>, CtfeError<'db>> {
        project_const(self.db, value, &[CtfePathElem::Index(index)], origin)
    }

    fn enum_extract(
        &self,
        value: SemConstId<'db>,
        variant: VariantIndex,
        field: FieldIndex,
        origin: SemOrigin<'db>,
    ) -> Result<SemConstId<'db>, CtfeError<'db>> {
        let SemConstValue::Enum {
            variant: actual,
            fields,
            ..
        } = value.value(self.db)
        else {
            return Err(CtfeError::VariantMismatch { origin });
        };
        if actual != variant {
            return Err(CtfeError::VariantMismatch { origin });
        }
        fields
            .get(field.0 as usize)
            .copied()
            .ok_or(CtfeError::OutOfBounds { origin })
    }

    fn load_enum_variant(
        &self,
        value: SemConstId<'db>,
        origin: SemOrigin<'db>,
    ) -> Result<VariantIndex, CtfeError<'db>> {
        let SemConstValue::Enum { variant, .. } = value.value(self.db) else {
            return Err(CtfeError::VariantMismatch { origin });
        };
        Ok(variant)
    }

    fn expect_bool(
        &self,
        value: SemConstId<'db>,
        origin: SemOrigin<'db>,
    ) -> Result<bool, CtfeError<'db>> {
        let SemConstValue::Scalar {
            value: SemConstScalar::Bool(value),
            ..
        } = value.value(self.db)
        else {
            return Err(CtfeError::InvalidOperation {
                origin,
                message: "expected bool".into(),
            });
        };
        Ok(value)
    }

    fn expect_int(
        &self,
        frame_idx: usize,
        value: SemConstId<'db>,
        origin: SemOrigin<'db>,
    ) -> Result<BigInt, CtfeError<'db>> {
        match value.value(self.db) {
            SemConstValue::Scalar {
                value: SemConstScalar::Int { value },
                ..
            } => Ok(value.clone()),
            SemConstValue::TypeLevel { ty, const_ty } => {
                let TyData::ConstTy(const_ty) = const_ty.data(self.db) else {
                    return Err(CtfeError::InvalidOperation {
                        origin,
                        message: format!("expected int, got {:?}", value.value(self.db)),
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
                        origin,
                        message: "expected int".into(),
                    });
                };
                Ok(BigInt::from(int_id.data(self.db).clone()))
            }
            _ => Err(CtfeError::InvalidOperation {
                origin,
                message: "expected int".into(),
            }),
        }
    }

    fn index_from_value(
        &self,
        frame_idx: usize,
        value: SemConstId<'db>,
        origin: SemOrigin<'db>,
    ) -> Result<usize, CtfeError<'db>> {
        let index = self.expect_int(frame_idx, value, origin)?;
        index.to_usize().ok_or(CtfeError::OutOfBounds { origin })
    }

    fn eval_unary(
        &self,
        frame_idx: usize,
        result_ty: TyId<'db>,
        op: UnOp,
        value: SemConstId<'db>,
        origin: SemOrigin<'db>,
    ) -> Result<CtfeValue<'db>, CtfeError<'db>> {
        match op {
            UnOp::Plus => Ok(CtfeValue::Value(value)),
            UnOp::Minus => Ok(CtfeValue::Value(int_const(
                self.db,
                result_ty,
                -self.expect_int(frame_idx, value, origin)?,
            ))),
            UnOp::Not => Ok(CtfeValue::Value(bool_const(
                self.db,
                !self.expect_bool(value, origin)?,
            ))),
            UnOp::BitNot => {
                let int = self.expect_int(frame_idx, value, origin)?;
                Ok(CtfeValue::Value(int_const(
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
        lhs: SemConstId<'db>,
        rhs: SemConstId<'db>,
        origin: SemOrigin<'db>,
    ) -> Result<CtfeValue<'db>, CtfeError<'db>> {
        match op {
            BinOp::Comp(comp) => self.eval_compare(frame_idx, comp, lhs, rhs, origin),
            BinOp::Logical(_) | BinOp::Index => Err(CtfeError::InvalidOperation {
                origin,
                message: "invalid binary op in CTFE expression".into(),
            }),
            BinOp::Arith(ArithBinOp::Range) => Err(CtfeError::NotConstEvaluable { origin }),
            BinOp::Arith(arith) => {
                if (matches!(lhs.value(self.db), SemConstValue::TypeLevel { .. })
                    || matches!(rhs.value(self.db), SemConstValue::TypeLevel { .. }))
                    && let Some(value) = self.eval_type_level_binary(result_ty, arith, lhs, rhs)
                {
                    return Ok(CtfeValue::Value(value));
                }
                let lhs = self.expect_int(frame_idx, lhs, origin)?;
                let rhs = self.expect_int(frame_idx, rhs, origin)?;
                let value = match arith {
                    ArithBinOp::Add => lhs + rhs,
                    ArithBinOp::Sub => lhs - rhs,
                    ArithBinOp::Mul => lhs * rhs,
                    ArithBinOp::Div => {
                        if rhs.is_zero() {
                            return Err(CtfeError::DivisionByZero { origin });
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
                        let exp = rhs.to_u32().ok_or(CtfeError::InvalidOperation {
                            origin,
                            message: "invalid power exponent".into(),
                        })?;
                        lhs.pow(exp)
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
                Ok(CtfeValue::Value(int_const(self.db, result_ty, value)))
            }
        }
    }

    fn eval_compare(
        &self,
        frame_idx: usize,
        op: CompBinOp,
        lhs: SemConstId<'db>,
        rhs: SemConstId<'db>,
        origin: SemOrigin<'db>,
    ) -> Result<CtfeValue<'db>, CtfeError<'db>> {
        let result = match op {
            CompBinOp::Eq => lhs == rhs,
            CompBinOp::NotEq => lhs != rhs,
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
        };
        Ok(CtfeValue::Value(bool_const(self.db, result)))
    }

    fn eval_type_level_binary(
        &self,
        result_ty: TyId<'db>,
        op: ArithBinOp,
        lhs: SemConstId<'db>,
        rhs: SemConstId<'db>,
    ) -> Option<SemConstId<'db>> {
        let lhs = TyId::const_ty(self.db, const_ty_from_sem_const(self.db, lhs));
        let rhs = TyId::const_ty(self.db, const_ty_from_sem_const(self.db, rhs));
        let expr = ConstExprId::new(self.db, ConstExpr::ArithBinOp { op, lhs, rhs });
        let const_ty = ConstTyId::new(self.db, ConstTyData::Abstract(expr, result_ty))
            .evaluate(self.db, Some(result_ty));
        sem_const_from_ty(self.db, TyId::const_ty(self.db, const_ty))
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

    fn eval_cast(
        &self,
        result_ty: TyId<'db>,
        value: SemConstId<'db>,
        origin: SemOrigin<'db>,
    ) -> Result<CtfeValue<'db>, CtfeError<'db>> {
        match value.value(self.db) {
            SemConstValue::Scalar {
                value: SemConstScalar::Bool(value),
                ..
            } if int_ty_shape(self.db, result_ty).is_some() => Ok(CtfeValue::Value(int_const(
                self.db,
                result_ty,
                if value { BigInt::one() } else { BigInt::zero() },
            ))),
            SemConstValue::Scalar {
                value: SemConstScalar::Int { value },
                ..
            } if result_ty == TyId::bool(self.db) => {
                Ok(CtfeValue::Value(bool_const(self.db, !value.is_zero())))
            }
            SemConstValue::Scalar {
                value: SemConstScalar::Int { value },
                ..
            } if int_ty_shape(self.db, result_ty).is_some() => Ok(CtfeValue::Value(int_const(
                self.db,
                result_ty,
                value.clone(),
            ))),
            SemConstValue::Scalar {
                value: SemConstScalar::Bytes(bytes),
                ..
            } => Ok(CtfeValue::Value(bytes_const(
                self.db,
                result_ty,
                bytes.clone(),
            ))),
            _ => Err(CtfeError::InvalidOperation {
                origin,
                message: "unsupported cast in CTFE".into(),
            }),
        }
    }

    fn make_aggregate(
        &self,
        result_ty: TyId<'db>,
        fields: Vec<SemConstId<'db>>,
    ) -> SemConstId<'db> {
        if result_ty.is_tuple(self.db) {
            tuple_const(self.db, result_ty, fields.into_boxed_slice())
        } else if result_ty.is_array(self.db) {
            array_const(self.db, result_ty, fields.into_boxed_slice())
        } else {
            struct_const(self.db, result_ty, fields.into_boxed_slice())
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
