use cranelift_entity::EntityRef;
use hir::{
    analysis::{
        semantic::{
            GenericSubst, ImplEnv, SemConstScalar, SemConstValue, SemanticInstance,
            SemanticInstanceKey, check_semantic_borrows, eval_const_instance,
            get_or_build_semantic_instance,
        },
        ty::{
            corelib::{resolve_core_trait, resolve_lib_type_path},
            trait_def::{
                TraitInstId, assoc_const_body_and_impl_args_for_trait_inst,
                resolve_trait_method_instance,
            },
            trait_resolution::{PredicateListId, TraitSolveCx},
            ty_check::BodyOwner,
            ty_def::{InvalidCause, TyId},
        },
    },
    hir_def::{
        IdentId,
        expr::{ArithBinOp, BinOp, CompBinOp},
    },
};
use num_traits::ToPrimitive;
use salsa::Update;

use crate::runtime::lower::{
    class::{runtime_signature_for_key, semantic_return_ty, top_level_class_for_ty_in_env},
    layout::RuntimeTypeEnv,
};
use crate::{
    db::MirDb,
    runtime::{
        AddressSpaceKind, ConstScalar, ContractEffectArgPlan, ContractInitAbiPlan,
        ContractRecvAbiPlan, DispatchDefault, HandleKind, HandleView, InitArgsPlan, PlaceElem,
        PlaceRoot, RBlock, RBlockId, RExpr, RLocal, RLocalId, RStmt, RTerminator, RuntimeBody,
        RuntimeBuiltin, RuntimeCallEdge, RuntimeCarrier, RuntimeClass, RuntimeInputPlan,
        RuntimeLocalRoot, RuntimePlace, RuntimeReturnPlan, RuntimeSignature, RuntimeSyntheticSpec,
        ScalarClass, ScalarRepr, ScalarRole,
        lower::{
            body::lower_to_rmir, call::collect_runtime_calls as collect_runtime_calls_lowered,
        },
        package::runtime_instance_for_semantic,
        pretty::format_runtime_verify_failure,
    },
    verify::verify_runtime_body_detailed,
};

#[salsa::interned]
#[derive(Debug)]
pub struct RuntimeSyntheticInstance<'db> {
    pub spec: RuntimeSyntheticSpec<'db>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Update)]
pub enum RuntimeInstanceSource<'db> {
    Semantic(SemanticInstance<'db>),
    Synthetic(RuntimeSyntheticInstance<'db>),
}

#[salsa::interned]
#[derive(Debug)]
pub struct RuntimeInstanceKey<'db> {
    pub source: RuntimeInstanceSource<'db>,
    #[return_ref]
    pub params: Vec<RuntimeClass<'db>>,
}

impl<'db> RuntimeInstanceKey<'db> {
    pub fn semantic(self, db: &'db dyn MirDb) -> Option<SemanticInstance<'db>> {
        match self.source(db) {
            RuntimeInstanceSource::Semantic(semantic) => Some(semantic),
            RuntimeInstanceSource::Synthetic(_) => None,
        }
    }
}

#[salsa::tracked]
#[derive(Debug)]
pub struct RuntimeInstance<'db> {
    pub key: RuntimeInstanceKey<'db>,
}

#[salsa::tracked]
impl<'db> RuntimeInstance<'db> {
    #[salsa::tracked]
    pub fn signature(self, db: &'db dyn MirDb) -> RuntimeSignature<'db> {
        match self.key(db).source(db) {
            RuntimeInstanceSource::Semantic(semantic) => {
                runtime_signature_for_key(db, semantic, self.key(db).params(db))
            }
            RuntimeInstanceSource::Synthetic(_) => self.body(db).signature,
        }
    }

    #[salsa::tracked]
    pub fn body(self, db: &'db dyn MirDb) -> RuntimeBody<'db> {
        lower_runtime_body(db, self)
    }

    #[salsa::tracked(return_ref)]
    pub fn calls(self, db: &'db dyn MirDb) -> Vec<RuntimeCallEdge<'db>> {
        collect_runtime_calls(db, self)
    }
}

#[salsa::tracked]
pub fn get_or_build_runtime_instance<'db>(
    db: &'db dyn MirDb,
    key: RuntimeInstanceKey<'db>,
) -> RuntimeInstance<'db> {
    RuntimeInstance::new(db, key)
}

fn lower_runtime_body<'db>(db: &'db dyn MirDb, instance: RuntimeInstance<'db>) -> RuntimeBody<'db> {
    let body = match instance.key(db).source(db) {
        RuntimeInstanceSource::Semantic(semantic) => {
            if let Err(diag) = check_semantic_borrows(db, semantic) {
                panic!(
                    "semantic borrow checking failed for {:?}: {}",
                    semantic.key(db),
                    diag.message,
                );
            }
            lower_to_rmir(db, instance)
        }
        RuntimeInstanceSource::Synthetic(synthetic) => {
            lower_synthetic_runtime_body(db, instance, synthetic.spec(db).clone())
        }
    };
    if let Err(failure) = verify_runtime_body_detailed(db, &db, &body) {
        panic!(
            "runtime body verification failed for {:?}\n{}",
            instance.key(db),
            format_runtime_verify_failure(db, &body, &failure),
        );
    }
    body
}

fn collect_runtime_calls<'db>(
    db: &'db dyn MirDb,
    instance: RuntimeInstance<'db>,
) -> Vec<RuntimeCallEdge<'db>> {
    match instance.key(db).source(db) {
        RuntimeInstanceSource::Semantic(_) => collect_runtime_calls_lowered(db, instance),
        RuntimeInstanceSource::Synthetic(synthetic) => {
            collect_synthetic_runtime_calls(db, &synthetic.spec(db))
        }
    }
}

fn collect_synthetic_runtime_calls<'db>(
    db: &'db dyn MirDb,
    spec: &RuntimeSyntheticSpec<'db>,
) -> Vec<RuntimeCallEdge<'db>> {
    let mut calls = Vec::new();
    match spec {
        RuntimeSyntheticSpec::MainRoot { callee }
        | RuntimeSyntheticSpec::TestRoot { callee, .. }
        | RuntimeSyntheticSpec::CodeRegionRoot { callee, .. } => {
            calls.push(RuntimeCallEdge { callee: *callee });
        }
        RuntimeSyntheticSpec::ContractInitAbi { plan } => {
            if let InitArgsPlan::DecodeInitTail { decode_fn, .. } = plan.init_args {
                if let Some(decoder_new) = resolve_sol_decoder_new(db, plan.contract.scope()) {
                    calls.push(RuntimeCallEdge {
                        callee: decoder_new,
                    });
                }
                calls.push(RuntimeCallEdge { callee: decode_fn });
            }
            if let Some(user_init) = plan.user_init {
                calls.push(RuntimeCallEdge { callee: user_init });
            }
        }
        RuntimeSyntheticSpec::ContractRecvAbi { plan } => {
            if let RuntimeInputPlan::DecodeCalldataPayload { decode_fn, .. } = plan.input {
                if let Some(decoder_new) = resolve_sol_decoder_new(db, plan.contract.scope()) {
                    calls.push(RuntimeCallEdge {
                        callee: decoder_new,
                    });
                }
                calls.push(RuntimeCallEdge { callee: decode_fn });
            }
            calls.push(RuntimeCallEdge {
                callee: plan.user_recv,
            });
            if let RuntimeReturnPlan::Value { encode_fn, .. } = plan.ret {
                let scope = plan.contract.scope();
                if let Some(encoder_new) = resolve_sol_encoder_new(db, scope) {
                    calls.push(RuntimeCallEdge {
                        callee: encoder_new,
                    });
                }
                if let Some(reserve_head) = resolve_sol_encoder_reserve_head(db, scope) {
                    calls.push(RuntimeCallEdge {
                        callee: reserve_head,
                    });
                }
                calls.push(RuntimeCallEdge { callee: encode_fn });
                if let Some(finish) = resolve_sol_encoder_finish(db, scope) {
                    calls.push(RuntimeCallEdge { callee: finish });
                }
            }
        }
        RuntimeSyntheticSpec::ContractInitRoot { init_abi, .. } => {
            if let Some(init_abi) = init_abi {
                calls.push(RuntimeCallEdge { callee: *init_abi });
            }
        }
        RuntimeSyntheticSpec::ContractRuntimeRoot { dispatch, .. } => {
            for arm in dispatch.iter() {
                calls.push(RuntimeCallEdge {
                    callee: arm.wrapper,
                });
            }
        }
    }
    calls
}

fn lower_synthetic_runtime_body<'db>(
    db: &'db dyn MirDb,
    instance: RuntimeInstance<'db>,
    spec: RuntimeSyntheticSpec<'db>,
) -> RuntimeBody<'db> {
    let mut builder = SyntheticBodyBuilder::new(db, instance);
    match spec {
        RuntimeSyntheticSpec::MainRoot { callee }
        | RuntimeSyntheticSpec::TestRoot { callee, .. }
        | RuntimeSyntheticSpec::CodeRegionRoot { callee, .. } => {
            builder.build_passthrough_root(callee);
        }
        RuntimeSyntheticSpec::ContractInitAbi { plan } => builder.build_contract_init_abi(plan),
        RuntimeSyntheticSpec::ContractRecvAbi { plan } => builder.build_contract_recv_abi(plan),
        RuntimeSyntheticSpec::ContractInitRoot {
            init_abi,
            runtime_region,
            ..
        } => builder.build_contract_init_root(init_abi, runtime_region),
        RuntimeSyntheticSpec::ContractRuntimeRoot {
            dispatch, default, ..
        } => builder.build_contract_runtime_root(&dispatch, default),
    }
    builder.finish()
}

struct SyntheticBodyBuilder<'db> {
    db: &'db dyn MirDb,
    instance: RuntimeInstance<'db>,
    locals: Vec<RLocal<'db>>,
    blocks: Vec<RBlock<'db>>,
}

impl<'db> SyntheticBodyBuilder<'db> {
    fn new(db: &'db dyn MirDb, instance: RuntimeInstance<'db>) -> Self {
        Self {
            db,
            instance,
            locals: Vec::new(),
            blocks: vec![RBlock {
                stmts: Vec::new(),
                terminator: RTerminator::Stop,
            }],
        }
    }

    fn finish(self) -> RuntimeBody<'db> {
        RuntimeBody {
            owner: self.instance,
            key: self.instance.key(self.db),
            signature: RuntimeSignature {
                params: Vec::new(),
                ret: None,
            },
            semantic_locals: Vec::new(),
            provider_bindings: Vec::new(),
            locals: self.locals,
            blocks: self.blocks,
        }
    }

    fn build_passthrough_root(&mut self, callee: RuntimeInstance<'db>) {
        let callee_body = callee.body(self.db);
        let mut args = Vec::with_capacity(callee_body.signature.params.len());
        for param in &callee_body.signature.params {
            let semantic_ty = callee_body
                .local(param.local)
                .map(|local| local.semantic_ty)
                .unwrap_or_else(|| TyId::invalid(self.db, InvalidCause::Other));
            let local = self.push_local(
                semantic_ty,
                RuntimeCarrier::Value(param.class.clone()),
                RuntimeLocalRoot::None,
            );
            self.push_stmt(
                RBlockId::from_u32(0),
                RStmt::Assign {
                    dst: local,
                    expr: RExpr::Placeholder {
                        class: param.class.clone(),
                    },
                },
            );
            args.push(local);
        }

        if let Some(class) = callee_body.signature.ret.clone() {
            let dst = self.push_local(
                callee_body
                    .locals
                    .first()
                    .map(|local| local.semantic_ty)
                    .unwrap_or_else(|| TyId::invalid(self.db, InvalidCause::Other)),
                RuntimeCarrier::Value(class),
                RuntimeLocalRoot::None,
            );
            self.push_stmt(
                RBlockId::from_u32(0),
                RStmt::Assign {
                    dst,
                    expr: RExpr::Call {
                        callee,
                        args: args.into_boxed_slice(),
                    },
                },
            );
        } else {
            let dst = self.push_erased_local(TyId::unit(self.db));
            self.push_stmt(
                RBlockId::from_u32(0),
                RStmt::Assign {
                    dst,
                    expr: RExpr::Call {
                        callee,
                        args: args.into_boxed_slice(),
                    },
                },
            );
        }
        self.blocks[0].terminator = RTerminator::Stop;
    }

    fn build_contract_init_abi(&mut self, plan: ContractInitAbiPlan<'db>) {
        let zero = self.push_const_word(RBlockId::from_u32(0), 0);
        let cont_bb = if plan.payable {
            RBlockId::from_u32(0)
        } else {
            self.emit_nonpayable_guard(zero)
        };

        let mut call_args = Vec::new();
        if let InitArgsPlan::DecodeInitTail {
            tuple_ty,
            decode_fn,
        } = plan.init_args
        {
            let current_region = crate::runtime::RuntimeCodeRegion::new(
                self.db,
                crate::runtime::RuntimeCodeRegionKey::ContractInit {
                    contract: plan.contract,
                },
            );
            let self_len = self.push_builtin_value(
                cont_bb,
                TyId::u256(self.db),
                RuntimeClass::Scalar(word_scalar_class()),
                RuntimeBuiltin::CodeRegionLen {
                    region: current_region,
                },
            );
            let code_size = self.push_builtin_value(
                cont_bb,
                TyId::u256(self.db),
                RuntimeClass::Scalar(word_scalar_class()),
                RuntimeBuiltin::CodeSize,
            );
            let tail_len = self.push_binary_word(cont_bb, ArithBinOp::Sub, code_size, self_len);
            let tail_ptr = self.push_builtin_value(
                cont_bb,
                TyId::u256(self.db),
                RuntimeClass::Scalar(word_scalar_class()),
                RuntimeBuiltin::Malloc { size: tail_len },
            );
            self.push_side_effect_builtin(
                cont_bb,
                RuntimeBuiltin::CodeCopy {
                    dst: tail_ptr,
                    offset: self_len,
                    len: tail_len,
                },
            );
            let input =
                self.push_memory_bytes_value(cont_bb, plan.contract.scope(), tail_ptr, tail_len);
            let decoder_new =
                resolve_sol_decoder_new(self.db, plan.contract.scope()).expect("decoder_new");
            let decoder = self.push_call_result(cont_bb, decoder_new, vec![input]);
            if let Some(decoded) = self.push_call(cont_bb, decode_fn, vec![decoder]) {
                call_args.extend(self.extract_tuple_fields(
                    cont_bb,
                    decoded,
                    tuple_ty,
                    plan.contract.scope(),
                ));
            }
        }

        if let Some(user_init) = plan.user_init {
            call_args.extend(self.owner_effect_call_args(
                cont_bb,
                user_init,
                call_args.len(),
                &plan.owner_effect_args,
            ));
            let _ = self.push_ignored_call(cont_bb, user_init, call_args);
        }
        self.blocks[cont_bb.index()].terminator = RTerminator::Return(None);
    }

    fn build_contract_recv_abi(&mut self, plan: ContractRecvAbiPlan<'db>) {
        let zero = self.push_const_word(RBlockId::from_u32(0), 0);
        let four = self.push_const_word(RBlockId::from_u32(0), 4);
        let cont_bb = if plan.payable {
            RBlockId::from_u32(0)
        } else {
            self.emit_nonpayable_guard(zero)
        };

        let mut call_args = Vec::new();
        if let RuntimeInputPlan::DecodeCalldataPayload { msg_ty, decode_fn } = plan.input {
            let size = self.push_builtin_value(
                cont_bb,
                TyId::u256(self.db),
                RuntimeClass::Scalar(word_scalar_class()),
                RuntimeBuiltin::CallDataSize,
            );
            let payload_len = self.push_binary_word(cont_bb, ArithBinOp::Sub, size, four);
            let payload_ptr = self.push_builtin_value(
                cont_bb,
                TyId::u256(self.db),
                RuntimeClass::Scalar(word_scalar_class()),
                RuntimeBuiltin::Malloc { size: payload_len },
            );
            self.push_side_effect_builtin(
                cont_bb,
                RuntimeBuiltin::CallDataCopy {
                    dst: payload_ptr,
                    offset: four,
                    len: payload_len,
                },
            );
            let input = self.push_memory_bytes_value(
                cont_bb,
                plan.contract.scope(),
                payload_ptr,
                payload_len,
            );
            let decoder_new =
                resolve_sol_decoder_new(self.db, plan.contract.scope()).expect("decoder_new");
            let decoder = self.push_call_result(cont_bb, decoder_new, vec![input]);
            if let Some(decoded) = self.push_call(cont_bb, decode_fn, vec![decoder]) {
                call_args.extend(self.extract_tuple_fields(
                    cont_bb,
                    decoded,
                    msg_ty,
                    plan.contract.scope(),
                ));
            }
        }
        call_args.extend(self.owner_effect_call_args(
            cont_bb,
            plan.user_recv,
            call_args.len(),
            &plan.owner_effect_args,
        ));

        let ret = self.push_call(cont_bb, plan.user_recv, call_args);
        match plan.ret {
            RuntimeReturnPlan::Unit => {
                self.blocks[cont_bb.index()].terminator = RTerminator::ReturnData {
                    offset: zero,
                    len: zero,
                };
            }
            RuntimeReturnPlan::Value { encode_fn, .. } => {
                let ret_value = ret.expect("value-returning recv wrapper should produce a value");
                let scope = plan.contract.scope();
                let encoder_new = resolve_sol_encoder_new(self.db, scope).expect("encoder_new");
                let reserve_head =
                    resolve_sol_encoder_reserve_head(self.db, scope).expect("reserve_head");
                let finish = resolve_sol_encoder_finish(self.db, scope).expect("finish");
                let encoded_size =
                    encoded_size_for_ty(self.db, scope, self.locals[ret_value.index()].semantic_ty)
                        .expect("encoded size");
                let encoder = self.push_call_result(cont_bb, encoder_new, Vec::new());
                let head_size = self.push_const_u256(cont_bb, encoded_size);
                let _ = self.push_ignored_call(cont_bb, reserve_head, vec![encoder, head_size]);
                let _ = self.push_ignored_call(cont_bb, encode_fn, vec![ret_value, encoder]);
                let encoded = self.push_call_result(cont_bb, finish, vec![encoder]);
                let fields = self.extract_tuple_fields(
                    cont_bb,
                    encoded,
                    self.locals[encoded.index()].semantic_ty,
                    scope,
                );
                let [offset, len]: [RLocalId; 2] = fields
                    .try_into()
                    .expect("encoded return should expose ptr/len");
                self.blocks[cont_bb.index()].terminator = RTerminator::ReturnData { offset, len };
            }
        }
    }

    fn build_contract_init_root(
        &mut self,
        init_abi: Option<RuntimeInstance<'db>>,
        runtime_region: crate::runtime::RuntimeCodeRegion<'db>,
    ) {
        let zero = self.push_const_word(RBlockId::from_u32(0), 0);
        if let Some(init_abi) = init_abi {
            let _ = self.push_ignored_call(RBlockId::from_u32(0), init_abi, Vec::new());
        }
        let runtime_offset = self.push_builtin_value(
            RBlockId::from_u32(0),
            TyId::u256(self.db),
            RuntimeClass::Scalar(word_scalar_class()),
            RuntimeBuiltin::CodeRegionOffset {
                region: runtime_region,
            },
        );
        let runtime_len = self.push_builtin_value(
            RBlockId::from_u32(0),
            TyId::u256(self.db),
            RuntimeClass::Scalar(word_scalar_class()),
            RuntimeBuiltin::CodeRegionLen {
                region: runtime_region,
            },
        );
        self.push_side_effect_builtin(
            RBlockId::from_u32(0),
            RuntimeBuiltin::CodeCopy {
                dst: zero,
                offset: runtime_offset,
                len: runtime_len,
            },
        );
        self.blocks[0].terminator = RTerminator::ReturnData {
            offset: zero,
            len: runtime_len,
        };
    }

    fn build_contract_runtime_root(
        &mut self,
        dispatch: &[crate::runtime::DispatchArm<'db>],
        default: DispatchDefault,
    ) {
        let zero = self.push_const_word(RBlockId::from_u32(0), 0);
        let selector = self.push_builtin_value(
            RBlockId::from_u32(0),
            TyId::u256(self.db),
            RuntimeClass::Scalar(selector_scalar_class()),
            RuntimeBuiltin::CallDataSelector,
        );
        let default_bb = self.new_block();
        let mut cases = Vec::with_capacity(dispatch.len());
        for arm in dispatch {
            let block = self.new_block();
            cases.push((u32_scalar(arm.selector), block));
            self.blocks[block.index()].terminator = RTerminator::TerminalCall {
                callee: arm.wrapper,
                args: Box::default(),
            };
        }
        self.blocks[default_bb.index()].terminator = match default {
            DispatchDefault::RevertEmpty => RTerminator::Revert {
                offset: zero,
                len: zero,
            },
        };
        self.blocks[0].terminator = RTerminator::SwitchScalar {
            discr: selector,
            cases: cases.into_boxed_slice(),
            default: default_bb,
        };
    }

    fn emit_nonpayable_guard(&mut self, zero: RLocalId) -> RBlockId {
        let entry = RBlockId::from_u32(0);
        let revert_bb = self.new_block();
        let cont_bb = self.new_block();
        let callvalue = self.push_builtin_value(
            entry,
            TyId::u256(self.db),
            RuntimeClass::Scalar(word_scalar_class()),
            RuntimeBuiltin::CallValue,
        );
        let cond = self.push_bool_binary(entry, CompBinOp::NotEq, callvalue, zero);
        self.blocks[entry.index()].terminator = RTerminator::Branch {
            cond,
            then_bb: revert_bb,
            else_bb: cont_bb,
        };
        self.blocks[revert_bb.index()].terminator = RTerminator::Revert {
            offset: zero,
            len: zero,
        };
        cont_bb
    }

    fn emit_contract_effect_args(
        &mut self,
        bb: RBlockId,
        bindings: &[ContractEffectArgPlan<'db>],
    ) -> Vec<RLocalId> {
        bindings
            .iter()
            .map(|binding| match binding {
                ContractEffectArgPlan::ContractField(binding) => self.push_builtin_value(
                    bb,
                    binding.declared_ty,
                    binding.class.clone(),
                    RuntimeBuiltin::MakeContractFieldHandle {
                        slot: binding.slot,
                        class: binding.class.clone(),
                        kind: binding.kind.clone(),
                    },
                ),
                ContractEffectArgPlan::Placeholder { declared_ty, class } => {
                    let local = self.push_local(
                        *declared_ty,
                        RuntimeCarrier::Value(class.clone()),
                        RuntimeLocalRoot::None,
                    );
                    self.push_stmt(
                        bb,
                        RStmt::Assign {
                            dst: local,
                            expr: RExpr::Placeholder {
                                class: class.clone(),
                            },
                        },
                    );
                    local
                }
            })
            .collect()
    }

    fn owner_effect_call_args(
        &mut self,
        bb: RBlockId,
        callee: RuntimeInstance<'db>,
        provided_prefix: usize,
        bindings: &[ContractEffectArgPlan<'db>],
    ) -> Vec<RLocalId> {
        let needed = callee
            .body(self.db)
            .signature
            .params
            .len()
            .saturating_sub(provided_prefix);
        let args = self.emit_contract_effect_args(bb, bindings);
        assert_eq!(
            args.len(),
            needed,
            "synthetic owner-effect arg count mismatch for {callee:?}"
        );
        args
    }

    fn runtime_type_env(
        &self,
        scope: hir::hir_def::scope_graph::ScopeId<'db>,
    ) -> RuntimeTypeEnv<'db> {
        RuntimeTypeEnv::new(Some(scope), PredicateListId::empty_list(self.db))
    }

    fn extract_tuple_fields(
        &mut self,
        bb: RBlockId,
        tuple: RLocalId,
        tuple_ty: TyId<'db>,
        scope: hir::hir_def::scope_graph::ScopeId<'db>,
    ) -> Vec<RLocalId> {
        let Some(tuple_class) = self.locals[tuple.index()].carrier.value_class().cloned() else {
            return Vec::new();
        };
        let env = self.runtime_type_env(scope);
        let tuple_root = match tuple_class {
            RuntimeClass::Handle { .. } => PlaceRoot::Handle(tuple),
            RuntimeClass::AggregateValue { layout } => match &self.locals[tuple.index()].root {
                RuntimeLocalRoot::Slot(_) => PlaceRoot::Slot(tuple),
                RuntimeLocalRoot::Handle(_) => PlaceRoot::Handle(tuple),
                RuntimeLocalRoot::Ptr { .. } | RuntimeLocalRoot::None => {
                    let handle = self.push_local(
                        tuple_ty,
                        RuntimeCarrier::Value(RuntimeClass::Handle {
                            layout,
                            kind: HandleKind::ObjectValue,
                            view: HandleView::Whole,
                        }),
                        RuntimeLocalRoot::None,
                    );
                    self.push_stmt(
                        bb,
                        RStmt::Assign {
                            dst: handle,
                            expr: RExpr::MaterializeToObject { src: tuple },
                        },
                    );
                    PlaceRoot::Handle(handle)
                }
            },
            RuntimeClass::Scalar(_) | RuntimeClass::RawAddr { .. } => {
                panic!("tuple extraction requires aggregate carrier")
            }
        };
        tuple_ty
            .field_types(self.db)
            .into_iter()
            .enumerate()
            .map(|(idx, field_ty)| {
                let class =
                    top_level_class_for_ty_in_env(self.db, env, field_ty, AddressSpaceKind::Memory)
                        .unwrap_or(RuntimeClass::RawAddr {
                            space: AddressSpaceKind::Memory,
                            target: None,
                        });
                let dst = self.push_local(
                    field_ty,
                    RuntimeCarrier::Value(class),
                    RuntimeLocalRoot::None,
                );
                let place = RuntimePlace {
                    root: tuple_root.clone(),
                    path: vec![PlaceElem::Field(hir::analysis::semantic::FieldIndex(
                        idx as u16,
                    ))]
                    .into_boxed_slice(),
                };
                self.push_stmt(
                    bb,
                    RStmt::Assign {
                        dst,
                        expr: RExpr::Load { place },
                    },
                );
                dst
            })
            .collect()
    }

    fn push_call(
        &mut self,
        bb: RBlockId,
        callee: RuntimeInstance<'db>,
        args: Vec<RLocalId>,
    ) -> Option<RLocalId> {
        let sig = callee.body(self.db).signature.clone();
        let args = self.coerce_call_args(bb, callee, args);
        sig.ret.map(|class| {
            let semantic_ty = callee
                .key(self.db)
                .semantic(self.db)
                .map(|semantic| semantic_return_ty(self.db, semantic))
                .unwrap_or_else(|| TyId::invalid(self.db, InvalidCause::Other));
            let dst = self.push_local(
                semantic_ty,
                RuntimeCarrier::Value(class),
                RuntimeLocalRoot::None,
            );
            self.push_stmt(
                bb,
                RStmt::Assign {
                    dst,
                    expr: RExpr::Call {
                        callee,
                        args: args.into_boxed_slice(),
                    },
                },
            );
            dst
        })
    }

    fn push_ignored_call(
        &mut self,
        bb: RBlockId,
        callee: RuntimeInstance<'db>,
        args: Vec<RLocalId>,
    ) -> RLocalId {
        let args = self.coerce_call_args(bb, callee, args);
        let dst = self.push_erased_local(TyId::unit(self.db));
        self.push_stmt(
            bb,
            RStmt::Assign {
                dst,
                expr: RExpr::Call {
                    callee,
                    args: args.into_boxed_slice(),
                },
            },
        );
        dst
    }

    fn push_call_result(
        &mut self,
        bb: RBlockId,
        callee: RuntimeInstance<'db>,
        args: Vec<RLocalId>,
    ) -> RLocalId {
        self.push_call(bb, callee, args)
            .expect("synthetic helper call should produce a value")
    }

    fn coerce_call_args(
        &mut self,
        bb: RBlockId,
        callee: RuntimeInstance<'db>,
        args: Vec<RLocalId>,
    ) -> Vec<RLocalId> {
        let body = callee.body(self.db);
        assert_eq!(
            args.len(),
            body.signature.params.len(),
            "synthetic call arg count mismatch for {callee:?}"
        );
        args.into_iter()
            .zip(body.signature.params.iter())
            .map(|(arg, param)| {
                let semantic_ty = body
                    .local(param.local)
                    .map(|local| local.semantic_ty)
                    .unwrap_or_else(|| self.locals[arg.index()].semantic_ty);
                self.coerce_runtime_value(bb, arg, &param.class, semantic_ty)
            })
            .collect()
    }

    fn coerce_runtime_value(
        &mut self,
        bb: RBlockId,
        src: RLocalId,
        target: &RuntimeClass<'db>,
        semantic_ty: TyId<'db>,
    ) -> RLocalId {
        let source = self.locals[src.index()]
            .carrier
            .value_class()
            .cloned()
            .unwrap_or_else(|| panic!("cannot coerce erased runtime value {src:?} to {target:?}"));
        if &source == target {
            return src;
        }

        match (source, target.clone()) {
            (
                RuntimeClass::Handle {
                    layout,
                    kind: HandleKind::ObjectValue | HandleKind::ConstValue,
                    view: crate::runtime::HandleView::Whole,
                },
                RuntimeClass::Handle {
                    layout: target_layout,
                    kind: HandleKind::Provider { provider_ty, space },
                    view: crate::runtime::HandleView::Whole,
                },
            ) if layout == target_layout => {
                let dst = self.push_local(
                    semantic_ty,
                    RuntimeCarrier::Value(RuntimeClass::Handle {
                        layout,
                        kind: HandleKind::Provider { provider_ty, space },
                        view: crate::runtime::HandleView::Whole,
                    }),
                    RuntimeLocalRoot::None,
                );
                self.push_stmt(
                    bb,
                    RStmt::Assign {
                        dst,
                        expr: RExpr::AddrOf {
                            place: RuntimePlace {
                                root: PlaceRoot::Handle(src),
                                path: Box::default(),
                            },
                        },
                    },
                );
                dst
            }
            (
                RuntimeClass::Handle {
                    layout,
                    kind: HandleKind::ObjectValue | HandleKind::ConstValue,
                    view: crate::runtime::HandleView::Whole,
                },
                RuntimeClass::RawAddr {
                    space,
                    target: target_layout,
                },
            ) if target_layout.is_none_or(|target_layout| target_layout == layout) => {
                let dst = self.push_local(
                    semantic_ty,
                    RuntimeCarrier::Value(RuntimeClass::RawAddr {
                        space,
                        target: Some(layout),
                    }),
                    RuntimeLocalRoot::None,
                );
                self.push_stmt(
                    bb,
                    RStmt::Assign {
                        dst,
                        expr: RExpr::AddrOf {
                            place: RuntimePlace {
                                root: PlaceRoot::Handle(src),
                                path: Box::default(),
                            },
                        },
                    },
                );
                dst
            }
            (
                RuntimeClass::Handle {
                    layout,
                    kind: _,
                    view: _,
                },
                RuntimeClass::AggregateValue {
                    layout: target_layout,
                },
            ) if layout == target_layout => {
                let dst = self.push_local(
                    semantic_ty,
                    RuntimeCarrier::Value(RuntimeClass::AggregateValue { layout }),
                    RuntimeLocalRoot::None,
                );
                self.push_stmt(
                    bb,
                    RStmt::Assign {
                        dst,
                        expr: RExpr::Load {
                            place: RuntimePlace {
                                root: PlaceRoot::Handle(src),
                                path: Box::default(),
                            },
                        },
                    },
                );
                dst
            }
            (
                RuntimeClass::AggregateValue { layout },
                RuntimeClass::Handle {
                    layout: target_layout,
                    kind: HandleKind::ObjectValue,
                    view: crate::runtime::HandleView::Whole,
                },
            ) if layout == target_layout => {
                let dst = self.push_local(
                    semantic_ty,
                    RuntimeCarrier::Value(RuntimeClass::Handle {
                        layout,
                        kind: HandleKind::ObjectValue,
                        view: crate::runtime::HandleView::Whole,
                    }),
                    RuntimeLocalRoot::None,
                );
                self.push_stmt(
                    bb,
                    RStmt::Assign {
                        dst,
                        expr: RExpr::MaterializeToObject { src },
                    },
                );
                dst
            }
            (
                RuntimeClass::Handle {
                    layout,
                    kind: HandleKind::ConstValue,
                    view: crate::runtime::HandleView::Whole,
                },
                RuntimeClass::Handle {
                    layout: target_layout,
                    kind: HandleKind::ObjectValue,
                    view: crate::runtime::HandleView::Whole,
                },
            ) if layout == target_layout => {
                let dst = self.push_local(
                    semantic_ty,
                    RuntimeCarrier::Value(RuntimeClass::Handle {
                        layout,
                        kind: HandleKind::ObjectValue,
                        view: crate::runtime::HandleView::Whole,
                    }),
                    RuntimeLocalRoot::None,
                );
                self.push_stmt(
                    bb,
                    RStmt::Assign {
                        dst,
                        expr: RExpr::MaterializeToObject { src },
                    },
                );
                dst
            }
            (
                RuntimeClass::RawAddr { space, .. },
                RuntimeClass::Handle {
                    layout,
                    kind:
                        HandleKind::Provider {
                            provider_ty,
                            space: provider_space,
                        },
                    view: crate::runtime::HandleView::Whole,
                },
            ) if space == provider_space => {
                let dst = self.push_local(
                    semantic_ty,
                    RuntimeCarrier::Value(RuntimeClass::Handle {
                        layout,
                        kind: HandleKind::Provider { provider_ty, space },
                        view: crate::runtime::HandleView::Whole,
                    }),
                    RuntimeLocalRoot::None,
                );
                self.push_stmt(
                    bb,
                    RStmt::Assign {
                        dst,
                        expr: RExpr::ProviderFromRaw {
                            raw: src,
                            provider_ty,
                            space,
                            layout,
                        },
                    },
                );
                dst
            }
            (
                RuntimeClass::AggregateValue { layout },
                RuntimeClass::Handle {
                    layout: target_layout,
                    kind: HandleKind::ObjectValue,
                    view: HandleView::Whole,
                },
            ) if layout == target_layout => {
                let dst = self.push_local(
                    semantic_ty,
                    RuntimeCarrier::Value(RuntimeClass::Handle {
                        layout,
                        kind: HandleKind::ObjectValue,
                        view: HandleView::Whole,
                    }),
                    RuntimeLocalRoot::None,
                );
                self.push_stmt(
                    bb,
                    RStmt::Assign {
                        dst,
                        expr: RExpr::MaterializeToObject { src },
                    },
                );
                dst
            }
            (
                RuntimeClass::AggregateValue { layout },
                RuntimeClass::Handle {
                    layout: target_layout,
                    kind: HandleKind::Provider { provider_ty, space },
                    view: HandleView::Whole,
                },
            ) if layout == target_layout => {
                let object = self.coerce_runtime_value(
                    bb,
                    src,
                    &RuntimeClass::Handle {
                        layout,
                        kind: HandleKind::ObjectValue,
                        view: HandleView::Whole,
                    },
                    semantic_ty,
                );
                self.coerce_runtime_value(
                    bb,
                    object,
                    &RuntimeClass::Handle {
                        layout,
                        kind: HandleKind::Provider { provider_ty, space },
                        view: HandleView::Whole,
                    },
                    semantic_ty,
                )
            }
            (
                RuntimeClass::Scalar(ScalarClass {
                    repr:
                        ScalarRepr::Int {
                            bits: 256,
                            signed: false,
                        },
                    role: ScalarRole::Plain,
                }),
                RuntimeClass::Handle {
                    layout,
                    kind: HandleKind::Provider { provider_ty, space },
                    view: HandleView::Whole,
                },
            ) => {
                let raw = self.push_local(
                    semantic_ty,
                    RuntimeCarrier::Value(RuntimeClass::RawAddr {
                        space,
                        target: Some(layout),
                    }),
                    RuntimeLocalRoot::None,
                );
                self.push_stmt(
                    bb,
                    RStmt::Assign {
                        dst: raw,
                        expr: RExpr::WordToRawAddr {
                            value: src,
                            space,
                            target: Some(layout),
                        },
                    },
                );
                self.coerce_runtime_value(
                    bb,
                    raw,
                    &RuntimeClass::Handle {
                        layout,
                        kind: HandleKind::Provider { provider_ty, space },
                        view: HandleView::Whole,
                    },
                    semantic_ty,
                )
            }
            (
                RuntimeClass::Handle {
                    kind: HandleKind::Provider { .. },
                    ..
                },
                RuntimeClass::RawAddr { .. },
            ) => {
                let dst = self.push_local(
                    semantic_ty,
                    RuntimeCarrier::Value(target.clone()),
                    RuntimeLocalRoot::None,
                );
                self.push_stmt(
                    bb,
                    RStmt::Assign {
                        dst,
                        expr: RExpr::ProviderToRaw { value: src },
                    },
                );
                dst
            }
            (
                RuntimeClass::RawAddr { .. },
                RuntimeClass::Scalar(ScalarClass {
                    repr:
                        ScalarRepr::Int {
                            bits: 256,
                            signed: false,
                        },
                    ..
                }),
            ) => {
                let dst = self.push_local(
                    semantic_ty,
                    RuntimeCarrier::Value(target.clone()),
                    RuntimeLocalRoot::None,
                );
                self.push_stmt(
                    bb,
                    RStmt::Assign {
                        dst,
                        expr: RExpr::Cast {
                            value: src,
                            to: word_scalar_class(),
                        },
                    },
                );
                dst
            }
            (source, target) => {
                panic!("unsupported synthetic runtime coercion from {source:?} to {target:?}")
            }
        }
    }

    fn push_builtin_value(
        &mut self,
        bb: RBlockId,
        semantic_ty: TyId<'db>,
        class: RuntimeClass<'db>,
        builtin: RuntimeBuiltin<'db>,
    ) -> RLocalId {
        let dst = self.push_local(
            semantic_ty,
            RuntimeCarrier::Value(class),
            RuntimeLocalRoot::None,
        );
        self.push_stmt(
            bb,
            RStmt::Assign {
                dst,
                expr: RExpr::Builtin(builtin),
            },
        );
        dst
    }

    fn push_side_effect_builtin(&mut self, bb: RBlockId, builtin: RuntimeBuiltin<'db>) -> RLocalId {
        let dst = self.push_erased_local(TyId::unit(self.db));
        self.push_stmt(
            bb,
            RStmt::Assign {
                dst,
                expr: RExpr::Builtin(builtin),
            },
        );
        dst
    }

    fn push_const_word(&mut self, bb: RBlockId, value: u32) -> RLocalId {
        self.push_const_scalar(
            bb,
            ConstScalar::Int {
                bits: 256,
                signed: false,
                words: if value == 0 {
                    Vec::new()
                } else {
                    value
                        .to_be_bytes()
                        .into_iter()
                        .skip_while(|byte| *byte == 0)
                        .collect()
                },
            },
        )
    }

    fn push_const_u256(&mut self, bb: RBlockId, value: u64) -> RLocalId {
        self.push_const_scalar(
            bb,
            ConstScalar::Int {
                bits: 256,
                signed: false,
                words: if value == 0 {
                    Vec::new()
                } else {
                    value
                        .to_be_bytes()
                        .into_iter()
                        .skip_while(|byte| *byte == 0)
                        .collect()
                },
            },
        )
    }

    fn push_const_scalar(&mut self, bb: RBlockId, value: ConstScalar) -> RLocalId {
        let dst = self.push_local(
            TyId::u256(self.db),
            RuntimeCarrier::Value(RuntimeClass::Scalar(word_scalar_class())),
            RuntimeLocalRoot::None,
        );
        self.push_stmt(
            bb,
            RStmt::Assign {
                dst,
                expr: RExpr::ConstScalar(value),
            },
        );
        dst
    }

    fn push_memory_bytes_value(
        &mut self,
        bb: RBlockId,
        scope: hir::hir_def::scope_graph::ScopeId<'db>,
        ptr: RLocalId,
        len: RLocalId,
    ) -> RLocalId {
        let ptr = if matches!(
            self.locals[ptr.index()].carrier.value_class(),
            Some(RuntimeClass::RawAddr { .. })
        ) {
            let casted = self.push_local(
                TyId::u256(self.db),
                RuntimeCarrier::Value(RuntimeClass::Scalar(word_scalar_class())),
                RuntimeLocalRoot::None,
            );
            self.push_stmt(
                bb,
                RStmt::Assign {
                    dst: casted,
                    expr: RExpr::Cast {
                        value: ptr,
                        to: word_scalar_class(),
                    },
                },
            );
            casted
        } else {
            ptr
        };
        let ty = memory_bytes_ty(self.db, scope).expect("MemoryBytes");
        let class = top_level_class_for_ty_in_env(
            self.db,
            self.runtime_type_env(scope),
            ty,
            AddressSpaceKind::Memory,
        )
        .expect("memory bytes runtime class");
        let root = match &class {
            RuntimeClass::Handle { .. } => RuntimeLocalRoot::Handle(class.clone()),
            _ => RuntimeLocalRoot::Slot(class.clone()),
        };
        let local = self.push_local(ty, RuntimeCarrier::Value(class.clone()), root);
        let root = match class {
            RuntimeClass::Handle { layout, .. } => {
                self.push_stmt(
                    bb,
                    RStmt::Assign {
                        dst: local,
                        expr: RExpr::AllocObject { layout },
                    },
                );
                PlaceRoot::Handle(local)
            }
            RuntimeClass::Scalar(_)
            | RuntimeClass::RawAddr { .. }
            | RuntimeClass::AggregateValue { .. } => PlaceRoot::Slot(local),
        };
        self.push_stmt(
            bb,
            RStmt::Store {
                dst: RuntimePlace {
                    root: root.clone(),
                    path: vec![PlaceElem::Field(hir::analysis::semantic::FieldIndex(0))]
                        .into_boxed_slice(),
                },
                src: ptr,
            },
        );
        self.push_stmt(
            bb,
            RStmt::Store {
                dst: RuntimePlace {
                    root: root.clone(),
                    path: vec![PlaceElem::Field(hir::analysis::semantic::FieldIndex(1))]
                        .into_boxed_slice(),
                },
                src: len,
            },
        );
        local
    }

    fn push_binary_word(
        &mut self,
        bb: RBlockId,
        op: ArithBinOp,
        lhs: RLocalId,
        rhs: RLocalId,
    ) -> RLocalId {
        let dst = self.push_local(
            TyId::u256(self.db),
            RuntimeCarrier::Value(RuntimeClass::Scalar(word_scalar_class())),
            RuntimeLocalRoot::None,
        );
        self.push_stmt(
            bb,
            RStmt::Assign {
                dst,
                expr: RExpr::Binary {
                    op: BinOp::Arith(op),
                    lhs,
                    rhs,
                },
            },
        );
        dst
    }

    fn push_bool_binary(
        &mut self,
        bb: RBlockId,
        op: CompBinOp,
        lhs: RLocalId,
        rhs: RLocalId,
    ) -> RLocalId {
        let dst = self.push_local(
            TyId::bool(self.db),
            RuntimeCarrier::Value(RuntimeClass::Scalar(bool_scalar_class())),
            RuntimeLocalRoot::None,
        );
        self.push_stmt(
            bb,
            RStmt::Assign {
                dst,
                expr: RExpr::Binary {
                    op: BinOp::Comp(op),
                    lhs,
                    rhs,
                },
            },
        );
        dst
    }

    fn push_local(
        &mut self,
        semantic_ty: TyId<'db>,
        carrier: RuntimeCarrier<'db>,
        root: RuntimeLocalRoot<'db>,
    ) -> RLocalId {
        let id = RLocalId::from_u32(self.locals.len() as u32);
        self.locals.push(RLocal {
            semantic_ty,
            carrier,
            root,
        });
        id
    }

    fn push_erased_local(&mut self, semantic_ty: TyId<'db>) -> RLocalId {
        self.push_local(semantic_ty, RuntimeCarrier::Erased, RuntimeLocalRoot::None)
    }

    fn push_stmt(&mut self, bb: RBlockId, stmt: RStmt<'db>) {
        self.blocks[bb.index()].stmts.push(stmt);
    }

    fn new_block(&mut self) -> RBlockId {
        let id = RBlockId::from_u32(self.blocks.len() as u32);
        self.blocks.push(RBlock {
            stmts: Vec::new(),
            terminator: RTerminator::Trap,
        });
        id
    }
}

trait RuntimeCarrierExt<'db> {
    fn value_class(&self) -> Option<&RuntimeClass<'db>>;
}

impl<'db> RuntimeCarrierExt<'db> for RuntimeCarrier<'db> {
    fn value_class(&self) -> Option<&RuntimeClass<'db>> {
        match self {
            RuntimeCarrier::Erased => None,
            RuntimeCarrier::Value(class) => Some(class),
        }
    }
}

fn bool_scalar_class<'db>() -> ScalarClass<'db> {
    ScalarClass {
        repr: ScalarRepr::Bool,
        role: ScalarRole::Plain,
    }
}

fn word_scalar_class<'db>() -> ScalarClass<'db> {
    ScalarClass {
        repr: ScalarRepr::Int {
            bits: 256,
            signed: false,
        },
        role: ScalarRole::Plain,
    }
}

fn selector_scalar_class<'db>() -> ScalarClass<'db> {
    ScalarClass {
        repr: ScalarRepr::Int {
            bits: 32,
            signed: false,
        },
        role: ScalarRole::Plain,
    }
}

fn u32_scalar(value: u32) -> ConstScalar {
    ConstScalar::Int {
        bits: 32,
        signed: false,
        words: if value == 0 {
            Vec::new()
        } else {
            value
                .to_be_bytes()
                .into_iter()
                .skip_while(|byte| *byte == 0)
                .collect()
        },
    }
}

fn resolve_sol_decoder_new<'db>(
    db: &'db dyn MirDb,
    scope: hir::hir_def::scope_graph::ScopeId<'db>,
) -> Option<RuntimeInstance<'db>> {
    let abi_ty = sol_abi_ty(db, scope)?;
    let input_ty = memory_bytes_ty(db, scope)?;
    let abi_trait = resolve_core_trait(db, scope, &["abi", "Abi"])?;
    let inst = TraitInstId::new_simple(db, abi_trait, vec![abi_ty]);
    resolve_trait_runtime_instance(db, scope, inst, "decoder_new", vec![input_ty]).ok()
}

fn resolve_sol_encoder_new<'db>(
    db: &'db dyn MirDb,
    scope: hir::hir_def::scope_graph::ScopeId<'db>,
) -> Option<RuntimeInstance<'db>> {
    let abi_ty = sol_abi_ty(db, scope)?;
    let abi_trait = resolve_core_trait(db, scope, &["abi", "Abi"])?;
    let inst = TraitInstId::new_simple(db, abi_trait, vec![abi_ty]);
    resolve_trait_runtime_instance(db, scope, inst, "encoder_new", Vec::new()).ok()
}

fn resolve_sol_encoder_reserve_head<'db>(
    db: &'db dyn MirDb,
    scope: hir::hir_def::scope_graph::ScopeId<'db>,
) -> Option<RuntimeInstance<'db>> {
    let abi_ty = sol_abi_ty(db, scope)?;
    let encoder_ty = sol_encoder_ty(db, scope)?;
    let encoder_trait = resolve_core_trait(db, scope, &["abi", "AbiEncoder"])?;
    let inst = TraitInstId::new_simple(db, encoder_trait, vec![encoder_ty, abi_ty]);
    resolve_trait_runtime_instance(db, scope, inst, "reserve_head", Vec::new()).ok()
}

fn resolve_sol_encoder_finish<'db>(
    db: &'db dyn MirDb,
    scope: hir::hir_def::scope_graph::ScopeId<'db>,
) -> Option<RuntimeInstance<'db>> {
    let abi_ty = sol_abi_ty(db, scope)?;
    let encoder_ty = sol_encoder_ty(db, scope)?;
    let encoder_trait = resolve_core_trait(db, scope, &["abi", "AbiEncoder"])?;
    let inst = TraitInstId::new_simple(db, encoder_trait, vec![encoder_ty, abi_ty]);
    resolve_trait_runtime_instance(db, scope, inst, "finish", Vec::new()).ok()
}

fn encoded_size_for_ty<'db>(
    db: &'db dyn MirDb,
    scope: hir::hir_def::scope_graph::ScopeId<'db>,
    ty: TyId<'db>,
) -> Option<u64> {
    let abi_size_trait = resolve_core_trait(db, scope, &["abi", "AbiSize"])?;
    let assumptions = hir::analysis::ty::trait_resolution::PredicateListId::empty_list(db);
    let inst = TraitInstId::new_simple(db, abi_size_trait, vec![ty]);
    let (body, impl_args) = assoc_const_body_and_impl_args_for_trait_inst(
        db,
        TraitSolveCx::new(db, scope).with_assumptions(assumptions),
        inst,
        IdentId::new(db, "ENCODED_SIZE".to_string()),
    )?;
    let key = SemanticInstanceKey::new(
        db,
        BodyOwner::AnonConstBody {
            body,
            expected: TyId::u256(db),
        },
        GenericSubst::new(db, impl_args),
        ImplEnv::new(db, scope, assumptions, vec![inst]),
    );
    let value = eval_const_instance(db, get_or_build_semantic_instance(db, key)).ok()?;
    match value.value(db) {
        SemConstValue::Scalar {
            value: SemConstScalar::Int { value },
            ..
        } => value.to_u64(),
        _ => None,
    }
}

fn resolve_trait_runtime_instance<'db>(
    db: &'db dyn MirDb,
    scope: hir::hir_def::scope_graph::ScopeId<'db>,
    inst: TraitInstId<'db>,
    method: &str,
    extra_generic_args: Vec<TyId<'db>>,
) -> Result<RuntimeInstance<'db>, ()> {
    let assumptions = hir::analysis::ty::trait_resolution::PredicateListId::empty_list(db);
    let (func, mut impl_args) = resolve_trait_method_instance(
        db,
        TraitSolveCx::new(db, scope).with_assumptions(assumptions),
        inst,
        IdentId::new(db, method.to_string()),
    )
    .ok_or(())?;
    impl_args.extend(extra_generic_args);
    let key = SemanticInstanceKey::new(
        db,
        BodyOwner::Func(func),
        GenericSubst::new(db, impl_args),
        ImplEnv::new(db, scope, assumptions, vec![inst]),
    );
    Ok(runtime_instance_for_semantic(
        db,
        get_or_build_semantic_instance(db, key),
    ))
}

fn sol_abi_ty<'db>(
    db: &'db dyn MirDb,
    scope: hir::hir_def::scope_graph::ScopeId<'db>,
) -> Option<TyId<'db>> {
    resolve_lib_type_path(db, scope, "std::abi::Sol")
}

fn memory_bytes_ty<'db>(
    db: &'db dyn MirDb,
    scope: hir::hir_def::scope_graph::ScopeId<'db>,
) -> Option<TyId<'db>> {
    resolve_lib_type_path(db, scope, "std::evm::memory_input::MemoryBytes")
}

fn sol_encoder_ty<'db>(
    db: &'db dyn MirDb,
    scope: hir::hir_def::scope_graph::ScopeId<'db>,
) -> Option<TyId<'db>> {
    resolve_lib_type_path(db, scope, "std::abi::sol::SolEncoder")
}
