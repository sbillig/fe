use common::indexmap::IndexMap;
use hir::{
    analysis::{
        diagnostics::SpannedHirAnalysisDb,
        ty::{trait_def::TraitInstId, ty_def::TyId},
    },
    hir_def::{
        CallableDef,
        expr::{BinOp, CompBinOp},
    },
    semantic::EffectSource,
};
use num_bigint::BigUint;

use crate::{
    core_lib::CoreLib,
    ir::{
        AddressSpaceKind, BodyBuilder, CallOrigin, CallTargetRef, CodeRegionRef, ContractFunction,
        ContractFunctionKind, HirCallTarget, IntrinsicOp, MirFunction, MirFunctionOrigin,
        RuntimeAbi, RuntimeShape, Rvalue, SourceInfoId, SwitchTarget, SwitchValue, SymbolSource,
        SyntheticId, TerminatingCall, Terminator, ValueId, ValueOrigin, ValueRepr,
    },
    layout, repr,
};

use super::{
    ContractLoweringConfig, MirLowerResult, handlers,
    plan::{
        CodeRegionQueryKind, CodeRegionQueryPlan, ContractPlan, DefaultAction, FieldBindingMode,
        InitArgsPlan, InitEntrypointPlan, InitFinishPlan, RuntimeArgsPlan, RuntimeEntrypointPlan,
        RuntimeReturnPlan, SyntheticFnPlan,
    },
    symbols::SymbolMangler,
    target::{AbiContext, TargetContext, TargetHostContext},
};

pub(super) struct ContractEmitter<'db, 'a> {
    db: &'db dyn SpannedHirAnalysisDb,
    target: &'a TargetContext<'db>,
    config: &'a ContractLoweringConfig<'a>,
}

impl<'db, 'a> ContractEmitter<'db, 'a> {
    pub(super) fn new(
        db: &'db dyn SpannedHirAnalysisDb,
        target: &'a TargetContext<'db>,
        config: &'a ContractLoweringConfig<'a>,
    ) -> Self {
        Self { db, target, config }
    }

    pub(super) fn emit_program(
        &self,
        plan: &ContractPlan<'db>,
    ) -> MirLowerResult<Vec<MirFunction<'db>>> {
        let mangler = SymbolMangler::new(plan.display_name.clone());
        let mut out = Vec::with_capacity(plan.functions.len());
        let mut observable_init_handler = false;
        for fn_plan in &plan.functions {
            match fn_plan {
                SyntheticFnPlan::InitHandler(handler) => {
                    let Some(mut function) =
                        handlers::emit_init_handler(self.db, plan.contract, handler, &mangler)?
                    else {
                        continue;
                    };
                    observable_init_handler = true;
                    function.defer_root =
                        self.config.defer_all_roots || fn_plan.always_defer_root();
                    out.push(function);
                }
                SyntheticFnPlan::InitEntrypoint(entry) => {
                    let mut function =
                        self.emit_init_entrypoint(plan, entry, &mangler, observable_init_handler);
                    function.defer_root =
                        self.config.defer_all_roots || fn_plan.always_defer_root();
                    out.push(function);
                }
                _ => {
                    let mut function = self.emit_fn(plan, fn_plan, &mangler)?;
                    function.defer_root =
                        self.config.defer_all_roots || fn_plan.always_defer_root();
                    out.push(function);
                }
            }
        }
        Ok(out)
    }

    fn emit_fn(
        &self,
        plan: &ContractPlan<'db>,
        fn_plan: &SyntheticFnPlan<'db>,
        mangler: &SymbolMangler,
    ) -> MirLowerResult<MirFunction<'db>> {
        let mut function = match fn_plan {
            SyntheticFnPlan::InitHandler(handler) => {
                handlers::emit_init_handler(self.db, plan.contract, handler, mangler)?
                    .expect("init handler emission should be handled by emit_program")
            }
            SyntheticFnPlan::RecvHandler(handler) => {
                handlers::emit_recv_handler(self.db, plan.contract, handler, mangler)?
            }
            SyntheticFnPlan::InitEntrypoint(entry) => {
                self.emit_init_entrypoint(plan, entry, mangler, true)
            }
            SyntheticFnPlan::RuntimeEntrypoint(entry) => {
                self.emit_runtime_entrypoint(plan, entry, mangler)
            }
            SyntheticFnPlan::CodeRegionQuery(query) => self.emit_code_region_query(query, mangler),
        };
        function.symbol_name = mangler.symbol_for(fn_plan.id());
        Ok(function)
    }

    fn emit_init_entrypoint(
        &self,
        plan: &ContractPlan<'db>,
        entry: &InitEntrypointPlan<'db>,
        mangler: &SymbolMangler,
        emit_init_call: bool,
    ) -> MirFunction<'db> {
        let mut emitter = SyntheticFnEmitter::new(self.db, self.target, plan.contract);
        let root = emitter.root_effect();
        let zero = emitter.zero_u256();
        let InitFinishPlan::ReturnCodeRegion { target } = entry.finish;

        if !entry.is_payable {
            emitter.emit_callvalue_guard("callvalue_init", zero, root);
        }

        let runtime_offset = emitter.code_region_offset(target, "runtime_offset");
        let runtime_len = emitter.code_region_len(target, "runtime_len");

        if emit_init_call && let Some(call) = &entry.init_call {
            let fields = emitter.bind_all_fields(plan, entry.field_mode, root);
            let args = match call.args {
                InitArgsPlan::Empty => Vec::new(),
                InitArgsPlan::DecodeInitTailTupleElements { tuple_ty } => {
                    emitter.decode_init_tail_tuple_elements(tuple_ty, root)
                }
            };
            let effect_args = emitter.realize_effect_args(&call.effects, zero, &fields);
            let _ = emitter.call_synthetic(call.callee, args, effect_args, TyId::unit(self.db));
        }

        emitter.return_code_region(runtime_offset, runtime_len, zero);

        emitter.finish_synthetic_function(SyntheticFnMeta {
            id: entry.id,
            ret_ty: TyId::unit(self.db),
            returns_value: false,
            runtime_return_shape: RuntimeShape::Erased,
            contract_function: Some(ContractFunction {
                contract_name: plan.display_name.clone(),
                kind: ContractFunctionKind::Init,
            }),
            runtime_abi: RuntimeAbi::source_shaped(0, Vec::new()),
            symbol_name: mangler.symbol_for(entry.id),
        })
    }

    fn emit_runtime_entrypoint(
        &self,
        plan: &ContractPlan<'db>,
        entry: &RuntimeEntrypointPlan<'db>,
        mangler: &SymbolMangler,
    ) -> MirFunction<'db> {
        let mut emitter = SyntheticFnEmitter::new(self.db, self.target, plan.contract);
        let root = emitter.root_effect();
        let zero = emitter.zero_u256();
        let fields = emitter.bind_all_fields(plan, entry.field_mode, root);

        let selector = emitter.assign_runtime_local(
            "selector",
            self.target.abi.selector_ty,
            false,
            AddressSpaceKind::Memory,
            Rvalue::Call(emitter.host_runtime_selector(root)),
        );

        if entry
            .arms
            .iter()
            .any(|arm| !matches!(arm.call.args, RuntimeArgsPlan::Empty))
        {
            emitter.ensure_runtime_decoder(root);
        }

        let default_block = emitter.body.make_block();
        let mut targets = Vec::with_capacity(entry.arms.len());
        for arm in &entry.arms {
            let block = emitter.body.make_block();
            targets.push(SwitchTarget {
                value: SwitchValue::Int(BigUint::from(arm.selector)),
                block,
            });

            emitter.body.move_to_block(block);
            if !arm.is_payable {
                emitter.emit_callvalue_guard(
                    format!("callvalue_{}_{}", arm.recv_idx, arm.arm_idx),
                    zero,
                    root,
                );
            }
            let args = match arm.call.args {
                RuntimeArgsPlan::Empty => Vec::new(),
                RuntimeArgsPlan::DecodeRuntimeInput { ty } => {
                    emitter.decode_runtime_payload(ty).into_iter().collect()
                }
            };
            let effect_args = emitter.realize_effect_args(&arm.call.effects, zero, &fields);
            match arm.ret {
                RuntimeReturnPlan::Unit => {
                    let _ = emitter.call_synthetic(
                        arm.call.callee,
                        args,
                        effect_args,
                        TyId::unit(self.db),
                    );
                    emitter.body.terminate_current(Terminator::TerminatingCall {
                        source: SourceInfoId::SYNTHETIC,
                        call: TerminatingCall::Call(emitter.host_return_unit(root)),
                    });
                }
                RuntimeReturnPlan::Value { ty } => {
                    let result = emitter
                        .call_synthetic(arm.call.callee, args, effect_args, ty)
                        .expect("value-returning synthetic call should materialize a value");
                    emitter.body.terminate_current(Terminator::TerminatingCall {
                        source: SourceInfoId::SYNTHETIC,
                        call: TerminatingCall::Call(emitter.host_return_value(root, result, ty)),
                    });
                }
            }
        }

        emitter.body.move_to_block(default_block);
        match entry.default {
            DefaultAction::Abort => {
                emitter.body.terminate_current(Terminator::TerminatingCall {
                    source: SourceInfoId::SYNTHETIC,
                    call: TerminatingCall::Call(emitter.host_abort(root)),
                });
            }
        }

        let entry_block = emitter.body.entry_block();
        emitter.body.move_to_block(entry_block);
        emitter.body.switch(selector, targets, default_block);

        emitter.finish_synthetic_function(SyntheticFnMeta {
            id: entry.id,
            ret_ty: TyId::unit(self.db),
            returns_value: false,
            runtime_return_shape: RuntimeShape::Erased,
            contract_function: Some(ContractFunction {
                contract_name: plan.display_name.clone(),
                kind: ContractFunctionKind::Runtime,
            }),
            runtime_abi: RuntimeAbi::source_shaped(0, Vec::new()),
            symbol_name: mangler.symbol_for(entry.id),
        })
    }

    fn emit_code_region_query(
        &self,
        query: &CodeRegionQueryPlan<'db>,
        mangler: &SymbolMangler,
    ) -> MirFunction<'db> {
        let mut emitter = SyntheticFnEmitter::new(self.db, self.target, query.id.contract());
        let code_region = emitter.code_region_ref(query.target);
        let value = emitter.assign_runtime_local(
            "ret",
            TyId::u256(self.db),
            false,
            AddressSpaceKind::Memory,
            Rvalue::Intrinsic {
                op: match query.kind {
                    CodeRegionQueryKind::Offset => IntrinsicOp::CodeRegionOffset,
                    CodeRegionQueryKind::Len => IntrinsicOp::CodeRegionLen,
                },
                args: vec![code_region],
            },
        );
        emitter.body.return_value(value);
        emitter.finish_synthetic_function(SyntheticFnMeta {
            id: query.id,
            ret_ty: TyId::u256(self.db),
            returns_value: true,
            runtime_return_shape: RuntimeShape::Word(crate::ir::RuntimeWordKind::I256),
            contract_function: None,
            runtime_abi: RuntimeAbi::source_shaped(0, Vec::new()),
            symbol_name: mangler.symbol_for(query.id),
        })
    }
}

struct SyntheticFnEmitter<'db, 'a> {
    db: &'db dyn SpannedHirAnalysisDb,
    host: &'a TargetHostContext<'db>,
    abi: &'a AbiContext<'db>,
    core: CoreLib<'db>,
    body: BodyBuilder<'db>,
    runtime_decoder: Option<ValueId>,
}

impl<'db, 'a> SyntheticFnEmitter<'db, 'a> {
    fn new(
        db: &'db dyn SpannedHirAnalysisDb,
        target: &'a TargetContext<'db>,
        contract: hir::hir_def::Contract<'db>,
    ) -> Self {
        Self {
            db,
            host: &target.host,
            abi: &target.abi,
            core: CoreLib::new(db, contract.scope()),
            body: BodyBuilder::new(),
            runtime_decoder: None,
        }
    }

    fn root_effect(&mut self) -> ValueId {
        self.body.unit_value(self.host.root_effect_ty)
    }

    fn zero_u256(&mut self) -> ValueId {
        self.body
            .const_int_value(TyId::u256(self.db), BigUint::from(0u8))
    }

    fn bind_all_fields(
        &mut self,
        plan: &ContractPlan<'db>,
        mode: FieldBindingMode,
        root: ValueId,
    ) -> Vec<ValueId> {
        let host_field_func = match mode {
            FieldBindingMode::Init => self.host.init_field_fn,
            FieldBindingMode::Runtime => self.host.field_fn,
        };
        let u256_ty = TyId::u256(self.db);
        plan.fields
            .iter()
            .map(|field| {
                let slot_value = self.body.const_int_value(u256_ty, field.slot.clone());
                let call = self.field_value_call(
                    host_field_func,
                    root,
                    slot_value,
                    field.declared_ty,
                    field.is_provider,
                );
                self.assign_runtime_local(
                    format!("field{}", field.index),
                    u256_ty,
                    true,
                    AddressSpaceKind::Memory,
                    Rvalue::Call(call),
                )
            })
            .collect()
    }

    fn code_region_ref(&mut self, target: SyntheticId<'db>) -> ValueId {
        self.body.code_region_value(
            self.db,
            CodeRegionRef {
                origin: MirFunctionOrigin::Synthetic(target),
                generic_args: Vec::new(),
                symbol: None,
            },
        )
    }

    fn code_region_offset(&mut self, target: SyntheticId<'db>, local_name: &str) -> ValueId {
        let code_region = self.code_region_ref(target);
        self.assign_runtime_local(
            local_name,
            TyId::u256(self.db),
            false,
            AddressSpaceKind::Memory,
            Rvalue::Intrinsic {
                op: IntrinsicOp::CodeRegionOffset,
                args: vec![code_region],
            },
        )
    }

    fn code_region_len(&mut self, target: SyntheticId<'db>, local_name: &str) -> ValueId {
        let code_region = self.code_region_ref(target);
        self.assign_runtime_local(
            local_name,
            TyId::u256(self.db),
            false,
            AddressSpaceKind::Memory,
            Rvalue::Intrinsic {
                op: IntrinsicOp::CodeRegionLen,
                args: vec![code_region],
            },
        )
    }

    fn return_code_region(&mut self, offset: ValueId, len: ValueId, zero: ValueId) {
        self.body.assign(
            None,
            Rvalue::Intrinsic {
                op: IntrinsicOp::Codecopy,
                args: vec![zero, offset, len],
            },
        );
        self.body.terminate_current(Terminator::TerminatingCall {
            source: SourceInfoId::SYNTHETIC,
            call: TerminatingCall::Intrinsic {
                op: IntrinsicOp::ReturnData,
                args: vec![zero, len],
            },
        });
    }

    fn decode_init_tail_tuple_elements(
        &mut self,
        tuple_ty: TyId<'db>,
        root: ValueId,
    ) -> Vec<ValueId> {
        let args_offset = self.assign_runtime_local(
            "init_code_len",
            TyId::u256(self.db),
            false,
            AddressSpaceKind::Memory,
            Rvalue::Intrinsic {
                op: IntrinsicOp::CurrentCodeRegionLen,
                args: Vec::new(),
            },
        );
        let code_size = self.assign_runtime_local(
            "code_size",
            TyId::u256(self.db),
            false,
            AddressSpaceKind::Memory,
            Rvalue::Intrinsic {
                op: IntrinsicOp::Codesize,
                args: Vec::new(),
            },
        );
        let cond_value = self.body.alloc_value(
            TyId::bool(self.db),
            ValueOrigin::Binary {
                op: hir::hir_def::expr::BinOp::Comp(hir::hir_def::expr::CompBinOp::Lt),
                lhs: code_size,
                rhs: args_offset,
            },
            ValueRepr::Word,
        );

        let abort_block = self.body.make_block();
        let cont_block = self.body.make_block();
        self.body.branch(cond_value, abort_block, cont_block);

        self.body.move_to_block(abort_block);
        self.body.terminate_current(Terminator::TerminatingCall {
            source: SourceInfoId::SYNTHETIC,
            call: TerminatingCall::Call(self.host_abort(root)),
        });

        self.body.move_to_block(cont_block);
        let args_len = self.body.alloc_value(
            TyId::u256(self.db),
            ValueOrigin::Binary {
                op: hir::hir_def::expr::BinOp::Arith(hir::hir_def::expr::ArithBinOp::Sub),
                lhs: code_size,
                rhs: args_offset,
            },
            ValueRepr::Word,
        );
        let args_ptr = self.assign_runtime_local(
            "args_ptr",
            TyId::u256(self.db),
            false,
            AddressSpaceKind::Memory,
            Rvalue::Intrinsic {
                op: IntrinsicOp::Alloc,
                args: vec![args_len],
            },
        );
        self.body.assign(
            None,
            Rvalue::Intrinsic {
                op: IntrinsicOp::Codecopy,
                args: vec![args_ptr, args_offset, args_len],
            },
        );

        let input = self.assign_runtime_local(
            "input",
            self.host.init_input_ty,
            false,
            AddressSpaceKind::Memory,
            Rvalue::Alloc {
                address_space: AddressSpaceKind::Memory,
            },
        );
        self.body.store_field(input, 0, args_ptr);
        self.body.store_field(input, 1, args_len);

        let decoder = self.assign_runtime_local(
            "decoder",
            self.abi.init_decoder_ty,
            false,
            AddressSpaceKind::Memory,
            Rvalue::Call(self.abi_decoder_new(input, self.host.init_input_ty)),
        );

        tuple_ty
            .field_types(self.db)
            .iter()
            .copied()
            .enumerate()
            .filter(|(_, ty)| !layout::is_zero_sized_ty(self.db, *ty))
            .map(|(idx, ty)| {
                self.assign_runtime_local(
                    format!("init_arg{idx}"),
                    ty,
                    false,
                    AddressSpaceKind::Memory,
                    Rvalue::Call(self.decode_decode(decoder, self.abi.init_decoder_ty, ty)),
                )
            })
            .collect()
    }

    fn ensure_runtime_decoder(&mut self, root: ValueId) -> ValueId {
        if let Some(decoder) = self.runtime_decoder {
            return decoder;
        }
        let decoder = self.assign_runtime_local(
            "decoder",
            self.abi.runtime_decoder_ty,
            false,
            AddressSpaceKind::Memory,
            Rvalue::Call(self.host_runtime_decoder(root)),
        );
        self.runtime_decoder = Some(decoder);
        decoder
    }

    fn decode_runtime_payload(&mut self, ty: TyId<'db>) -> Option<ValueId> {
        let decoder = self
            .runtime_decoder
            .expect("runtime decoder should be initialized");
        if layout::is_zero_sized_ty(self.db, ty) {
            self.body.assign(
                None,
                Rvalue::Call(self.decode_decode(decoder, self.abi.runtime_decoder_ty, ty)),
            );
            return None;
        }
        Some(self.assign_runtime_local(
            "args",
            ty,
            false,
            AddressSpaceKind::Memory,
            Rvalue::Call(self.decode_decode(decoder, self.abi.runtime_decoder_ty, ty)),
        ))
    }

    fn realize_effect_args(
        &self,
        effects: &[EffectSource],
        zero_u256: ValueId,
        bound_fields: &[ValueId],
    ) -> Vec<ValueId> {
        effects
            .iter()
            .map(|effect| match effect {
                EffectSource::Root => zero_u256,
                EffectSource::Field(idx) => bound_fields
                    .get(*idx as usize)
                    .copied()
                    .unwrap_or(zero_u256),
            })
            .collect()
    }

    fn call_synthetic(
        &mut self,
        callee: SyntheticId<'db>,
        args: Vec<ValueId>,
        effect_args: Vec<ValueId>,
        ret_ty: TyId<'db>,
    ) -> Option<ValueId> {
        let call = CallOrigin {
            expr: None,
            target: Some(CallTargetRef::Synthetic(callee)),
            args,
            effect_args,
            resolved_name: None,
            checked_intrinsic: None,
            builtin_terminator: None,
            receiver_space: None,
        };
        if layout::is_zero_sized_ty(self.db, ret_ty) {
            self.body.assign(None, Rvalue::Call(call));
            return None;
        }
        Some(self.assign_runtime_local(
            "result",
            ret_ty,
            false,
            AddressSpaceKind::Memory,
            Rvalue::Call(call),
        ))
    }

    fn host_abort(&self, root: ValueId) -> CallOrigin<'db> {
        self.call_hir(
            CallableDef::Func(self.host.abort_fn),
            self.host.contract_host_inst.args(self.db).to_vec(),
            Some(self.host.contract_host_inst),
            vec![root],
        )
    }

    fn emit_callvalue_guard(&mut self, label: impl Into<String>, zero: ValueId, root: ValueId) {
        let callvalue = self.assign_runtime_local(
            label,
            TyId::u256(self.db),
            false,
            AddressSpaceKind::Memory,
            Rvalue::Intrinsic {
                op: IntrinsicOp::Callvalue,
                args: Vec::new(),
            },
        );
        let is_nonzero = self.body.alloc_value(
            TyId::bool(self.db),
            ValueOrigin::Binary {
                op: BinOp::Comp(CompBinOp::NotEq),
                lhs: callvalue,
                rhs: zero,
            },
            ValueRepr::Word,
        );
        let abort_block = self.body.make_block();
        let cont_block = self.body.make_block();
        self.body.branch(is_nonzero, abort_block, cont_block);
        self.body.move_to_block(abort_block);
        self.body.terminate_current(Terminator::TerminatingCall {
            source: SourceInfoId::SYNTHETIC,
            call: TerminatingCall::Call(self.host_abort(root)),
        });
        self.body.move_to_block(cont_block);
    }

    fn host_return_unit(&self, root: ValueId) -> CallOrigin<'db> {
        self.call_hir(
            CallableDef::Func(self.host.return_unit_fn),
            self.host.contract_host_inst.args(self.db).to_vec(),
            Some(self.host.contract_host_inst),
            vec![root],
        )
    }

    fn host_return_value(
        &self,
        root: ValueId,
        result: ValueId,
        ret_ty: TyId<'db>,
    ) -> CallOrigin<'db> {
        let mut generic_args = self.host.contract_host_inst.args(self.db).to_vec();
        generic_args.push(self.abi.abi_ty);
        generic_args.push(ret_ty);
        self.call_hir(
            CallableDef::Func(self.host.return_value_fn),
            generic_args,
            Some(self.host.contract_host_inst),
            vec![root, result],
        )
    }

    fn host_runtime_selector(&self, root: ValueId) -> CallOrigin<'db> {
        let mut generic_args = self.host.contract_host_inst.args(self.db).to_vec();
        generic_args.push(self.abi.abi_ty);
        self.call_hir(
            CallableDef::Func(self.host.runtime_selector_fn),
            generic_args,
            Some(self.host.contract_host_inst),
            vec![root],
        )
    }

    fn host_runtime_decoder(&self, root: ValueId) -> CallOrigin<'db> {
        let mut generic_args = self.host.contract_host_inst.args(self.db).to_vec();
        generic_args.push(self.abi.abi_ty);
        self.call_hir(
            CallableDef::Func(self.host.runtime_decoder_fn),
            generic_args,
            Some(self.host.contract_host_inst),
            vec![root],
        )
    }

    fn abi_decoder_new(&self, input_value: ValueId, input_ty: TyId<'db>) -> CallOrigin<'db> {
        let mut generic_args = self.abi.abi_inst.args(self.db).to_vec();
        generic_args.push(input_ty);
        self.call_hir(
            CallableDef::Func(self.abi.abi_decoder_new),
            generic_args,
            Some(self.abi.abi_inst),
            vec![input_value],
        )
    }

    fn decode_decode(
        &self,
        decoder_value: ValueId,
        decoder_ty: TyId<'db>,
        target_ty: TyId<'db>,
    ) -> CallOrigin<'db> {
        let inst = TraitInstId::new(
            self.db,
            self.abi.decode_trait,
            vec![target_ty, self.abi.abi_ty],
            IndexMap::new(),
        );
        let mut generic_args = inst.args(self.db).to_vec();
        generic_args.push(decoder_ty);
        self.call_hir(
            CallableDef::Func(self.abi.decode_decode),
            generic_args,
            Some(inst),
            vec![decoder_value],
        )
    }

    fn field_value_call(
        &self,
        host_field_func: hir::hir_def::Func<'db>,
        root: ValueId,
        slot: ValueId,
        declared_ty: TyId<'db>,
        is_provider: bool,
    ) -> CallOrigin<'db> {
        if is_provider {
            let inst = TraitInstId::new(
                self.db,
                self.host.effect_handle_trait,
                vec![declared_ty],
                IndexMap::new(),
            );
            return self.call_hir(
                CallableDef::Func(self.host.effect_handle_from_raw_fn),
                inst.args(self.db).to_vec(),
                Some(inst),
                vec![slot],
            );
        }
        let mut generic_args = self.host.contract_host_inst.args(self.db).to_vec();
        generic_args.push(declared_ty);
        self.call_hir(
            CallableDef::Func(host_field_func),
            generic_args,
            Some(self.host.contract_host_inst),
            vec![root, slot],
        )
    }

    fn call_hir(
        &self,
        callable_def: CallableDef<'db>,
        generic_args: Vec<TyId<'db>>,
        trait_inst: Option<TraitInstId<'db>>,
        args: Vec<ValueId>,
    ) -> CallOrigin<'db> {
        CallOrigin {
            expr: None,
            target: Some(CallTargetRef::Hir(HirCallTarget {
                callable_def,
                generic_args,
                trait_inst,
            })),
            args,
            effect_args: Vec::new(),
            resolved_name: None,
            checked_intrinsic: None,
            builtin_terminator: None,
            receiver_space: None,
        }
    }

    fn assign_runtime_local(
        &mut self,
        name: impl Into<String>,
        ty: TyId<'db>,
        is_mut: bool,
        address_space: AddressSpaceKind,
        rvalue: Rvalue<'db>,
    ) -> ValueId {
        let repr = self.value_repr_for_ty(ty, address_space);
        self.body
            .assign_to_new_local(name, ty, is_mut, address_space, repr, rvalue)
            .value
    }

    fn value_repr_for_ty(&self, ty: TyId<'db>, space: AddressSpaceKind) -> ValueRepr {
        if ty.as_capability(self.db).is_some() {
            return match repr::repr_kind_for_ty(self.db, &self.core, ty) {
                repr::ReprKind::Ptr(_) => ValueRepr::Ptr(space),
                repr::ReprKind::Ref => ValueRepr::Ref(space),
                repr::ReprKind::Zst | repr::ReprKind::Word => ValueRepr::Word,
            };
        }
        match repr::repr_kind_for_ty(self.db, &self.core, ty) {
            repr::ReprKind::Ptr(space) => ValueRepr::Ptr(space),
            repr::ReprKind::Ref => ValueRepr::Ref(space),
            repr::ReprKind::Zst | repr::ReprKind::Word => ValueRepr::Word,
        }
    }

    fn finish_synthetic_function(self, meta: SyntheticFnMeta<'db>) -> MirFunction<'db> {
        MirFunction {
            origin: MirFunctionOrigin::Synthetic(meta.id),
            body: self.body.build(),
            typed_body: None,
            generic_args: Vec::new(),
            ret_ty: meta.ret_ty,
            returns_value: meta.returns_value,
            runtime_abi: meta.runtime_abi,
            runtime_return_shape: meta.runtime_return_shape,
            contract_function: meta.contract_function,
            inline_hint: None,
            symbol_name: meta.symbol_name,
            symbol_source: SymbolSource::Internal,
            receiver_space: None,
            defer_root: false,
        }
    }
}

struct SyntheticFnMeta<'db> {
    id: SyntheticId<'db>,
    ret_ty: TyId<'db>,
    returns_value: bool,
    runtime_return_shape: RuntimeShape<'db>,
    contract_function: Option<ContractFunction>,
    runtime_abi: RuntimeAbi<'db>,
    symbol_name: String,
}
