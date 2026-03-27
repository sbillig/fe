use std::{
    cell::RefCell,
    collections::VecDeque,
    hash::{Hash, Hasher},
};

use common::indexmap::IndexMap;
use hir::analysis::ty::corelib::{
    resolve_core_trait, resolve_lib_func_path, resolve_lib_type_path,
};
use hir::analysis::{
    HirAnalysisDb,
    diagnostics::SpannedHirAnalysisDb,
    diagnostics::format_diags,
    ty::{
        const_ty::ConstTyData,
        fold::{TyFoldable, TyFolder},
        normalize::normalize_ty,
        trait_def::{TraitInstId, resolve_trait_method_instance},
        trait_resolution::{PredicateListId, TraitSolveCx, is_goal_satisfiable},
        ty_check::check_func_body,
        ty_def::{TyBase, TyData, TyId},
    },
};
use hir::hir_def::{
    CallableDef, Func, HirIngot, IdentId, PathId, PathKind, item::ItemKind, scope_graph::ScopeId,
};
use rustc_hash::{FxHashMap, FxHashSet};

use crate::{
    CallOrigin, MirFunction,
    capability_space::{pointer_leaf_infos_for_ty_with_default, pointer_leaf_paths_for_ty},
    core_lib::CoreLib,
    dedup::deduplicate_mir,
    ir::{AddressSpaceKind, RuntimeShape, ValueId},
    lower::{MirLowerError, MirLowerResult, lower_function},
};

/// Walks generic MIR templates, cloning them per concrete substitution so
/// downstream passes only ever see monomorphic MIR.
///
/// Create monomorphic MIR instances for every reachable generic instantiation.
///
/// The input `templates` are lowered once from HIR and may contain generic
/// placeholders. This routine discovers all concrete substitutions reachable
/// from `main`/exported roots, clones the required templates, and performs the
/// type substitution directly on MIR so later passes do not need to reason
/// about generics.
pub(crate) fn monomorphize_functions<'db>(
    db: &'db dyn SpannedHirAnalysisDb,
    templates: Vec<MirFunction<'db>>,
) -> MirLowerResult<Vec<MirFunction<'db>>> {
    let mut monomorphizer = Monomorphizer::new(db, templates);
    monomorphizer.seed_roots();
    monomorphizer.process_worklist()?;
    Ok(deduplicate_mir(db, monomorphizer.into_instances()))
}

/// Worklist-driven builder that instantiates concrete MIR bodies on demand.
struct Monomorphizer<'db> {
    db: &'db dyn SpannedHirAnalysisDb,
    templates: Vec<MirFunction<'db>>,
    func_index: FxHashMap<TemplateKey<'db>, usize>,
    func_defs: FxHashMap<Func<'db>, CallableDef<'db>>,
    instances: Vec<MirFunction<'db>>,
    instance_map: FxHashMap<InstanceKey<'db>, usize>,
    worklist: VecDeque<usize>,
    current_symbol: Option<String>,
    ambiguous_bases: FxHashSet<String>,
    deferred_error: RefCell<Option<MirLowerError>>,
}

#[derive(Clone, PartialEq, Eq, Hash)]
struct InstanceKey<'db> {
    origin: crate::ir::MirFunctionOrigin<'db>,
    args: Vec<TyId<'db>>,
    receiver_space: Option<AddressSpaceKind>,
    effect_param_space_overrides: Vec<Option<AddressSpaceKind>>,
    param_capability_space_overrides: Vec<Vec<(crate::MirProjectionPath<'db>, AddressSpaceKind)>>,
}
#[derive(Clone, PartialEq, Eq, Hash)]
struct TemplateKey<'db> {
    origin: crate::ir::MirFunctionOrigin<'db>,
    receiver_space: Option<AddressSpaceKind>,
}

/// How a call target should be handled during monomorphization.
#[derive(Clone, Copy)]
enum CallTarget<'db> {
    /// The callee has a body and should be instantiated like any other template.
    Template(Func<'db>),
    /// The callee is a declaration only (e.g. `extern`); no MIR body exists.
    Decl(Func<'db>),
    /// The callee is a MIR-synthetic function.
    Synthetic(crate::ir::MirFunctionOrigin<'db>),
}

type ParamCapabilitySpaceOverrides<'db> =
    Vec<Vec<(crate::MirProjectionPath<'db>, AddressSpaceKind)>>;
type NormalizedCallInstanceInputs<'db> = (
    Vec<TyId<'db>>,
    Vec<Option<AddressSpaceKind>>,
    ParamCapabilitySpaceOverrides<'db>,
);

struct CallSite<'db> {
    bb_idx: usize,
    inst_idx: Option<usize>,
    target: CallTarget<'db>,
    args: Vec<TyId<'db>>,
    receiver_space: Option<AddressSpaceKind>,
    effect_param_space_overrides: Vec<Option<AddressSpaceKind>>,
    param_capability_space_overrides: ParamCapabilitySpaceOverrides<'db>,
}

/// An `AbortWithValue` call site: either a regular instruction or a terminator.
enum AbortWithValueSite<'db> {
    Inst {
        bb_idx: usize,
        inst_idx: usize,
        concrete_t: TyId<'db>,
    },
    Term {
        bb_idx: usize,
        concrete_t: TyId<'db>,
    },
}

fn resolve_default_root_effect_ty<'db>(
    db: &'db dyn HirAnalysisDb,
    scope: ScopeId<'db>,
    assumptions: PredicateListId<'db>,
) -> Option<TyId<'db>> {
    let target_ty = resolve_lib_type_path(db, scope, "std::evm::EvmTarget")?;
    let target_trait = resolve_core_trait(db, scope, &["contracts", "Target"])?;
    let inst_target = TraitInstId::new(db, target_trait, vec![target_ty], IndexMap::new());
    let root_ident = IdentId::new(db, "RootEffect".to_owned());
    Some(normalize_ty(
        db,
        TyId::assoc_ty(db, inst_target, root_ident),
        scope,
        assumptions,
    ))
}

impl<'db> InstanceKey<'db> {
    /// Pack a function and its (possibly empty) substitution list for hashing.
    fn new(
        origin: crate::ir::MirFunctionOrigin<'db>,
        args: &[TyId<'db>],
        receiver_space: Option<AddressSpaceKind>,
        effect_param_space_overrides: &[Option<AddressSpaceKind>],
        param_capability_space_overrides: &[Vec<(
            crate::MirProjectionPath<'db>,
            AddressSpaceKind,
        )>],
    ) -> Self {
        Self {
            origin,
            args: args.to_vec(),
            receiver_space,
            effect_param_space_overrides: effect_param_space_overrides.to_vec(),
            param_capability_space_overrides: param_capability_space_overrides.to_vec(),
        }
    }
}

impl<'db> Monomorphizer<'db> {
    /// Find the std ingot by searching all known templates for one whose ingot
    /// has std as a dependency (or is std itself). Returns `None` only if std
    /// is genuinely absent (e.g. a bare-core compilation).
    fn find_std_ingot(&self, _func_idx: usize) -> Option<hir::Ingot<'db>> {
        for template in &self.templates {
            let ingot = match template.origin {
                crate::ir::MirFunctionOrigin::Hir(func) => func.scope().ingot(self.db),
                crate::ir::MirFunctionOrigin::Synthetic(synth) => {
                    synth.contract().scope().ingot(self.db)
                }
            };
            if ingot.kind(self.db) == common::ingot::IngotKind::Std {
                return Some(ingot);
            }
            if let Some((_, std_dep)) = ingot
                .resolved_external_ingots(self.db)
                .iter()
                .find(|(_, dep)| dep.kind(self.db) == common::ingot::IngotKind::Std)
            {
                return Some(*std_dep);
            }
        }
        None
    }

    fn core_for_origin(&self, origin: crate::ir::MirFunctionOrigin<'db>) -> CoreLib<'db> {
        let scope = match origin {
            crate::ir::MirFunctionOrigin::Hir(func) => func.scope(),
            crate::ir::MirFunctionOrigin::Synthetic(synth) => synth.contract().scope(),
        };
        CoreLib::new(self.db, scope)
    }

    /// Build the bookkeeping structures (template lookup + lowered FuncDef cache).
    fn new(db: &'db dyn SpannedHirAnalysisDb, templates: Vec<MirFunction<'db>>) -> Self {
        let func_index = templates
            .iter()
            .enumerate()
            .map(|(idx, func)| {
                (
                    TemplateKey {
                        origin: func.origin,
                        receiver_space: func.receiver_space,
                    },
                    idx,
                )
            })
            .collect();
        let mut func_defs = FxHashMap::default();
        for func in templates.iter().filter_map(|f| match f.origin {
            crate::ir::MirFunctionOrigin::Hir(func) => Some(func),
            crate::ir::MirFunctionOrigin::Synthetic(_) => None,
        }) {
            if let Some(def) = func.as_callable(db) {
                func_defs.insert(func, def);
            }
        }

        let mut monomorphizer = Self {
            db,
            templates,
            func_index,
            func_defs,
            instances: Vec::new(),
            instance_map: FxHashMap::default(),
            worklist: VecDeque::new(),
            current_symbol: None,
            ambiguous_bases: FxHashSet::default(),
            deferred_error: RefCell::new(None),
        };
        monomorphizer.ambiguous_bases = monomorphizer.compute_ambiguous_bases();
        monomorphizer
    }

    fn compute_ambiguous_bases(&self) -> FxHashSet<String> {
        let mut qualifiers_by_base: FxHashMap<String, FxHashSet<String>> = FxHashMap::default();
        for template in &self.templates {
            let crate::ir::MirFunctionOrigin::Hir(func) = template.origin else {
                continue;
            };
            let receiver_space = canonicalize_receiver_space(template.receiver_space);
            let base = self.base_name_root_without_disambiguation(func, receiver_space);
            let qualifier = self.function_qualifier(func);
            qualifiers_by_base
                .entry(base)
                .or_default()
                .insert(qualifier);
        }

        qualifiers_by_base
            .into_iter()
            .filter_map(|(base, qualifiers)| (qualifiers.len() > 1).then_some(base))
            .collect()
    }

    fn defer_error(&self, err: MirLowerError) {
        let mut deferred_error = self.deferred_error.borrow_mut();
        if deferred_error.is_none() {
            *deferred_error = Some(err);
        }
    }

    fn take_deferred_error(&self) -> Option<MirLowerError> {
        self.deferred_error.borrow_mut().take()
    }

    /// Instantiate all non-generic templates up front so they are always emitted
    /// (even if they are never referenced by another generic instantiation).
    fn seed_roots(&mut self) {
        for idx in 0..self.templates.len() {
            let origin = self.templates[idx].origin;
            let receiver_space = canonicalize_receiver_space(self.templates[idx].receiver_space);

            if let crate::ir::MirFunctionOrigin::Synthetic(_) = origin {
                if !self.templates[idx].defer_root {
                    let _ = self.ensure_synthetic_instance(origin, receiver_space, &[], &[]);
                }
                continue;
            }

            let crate::ir::MirFunctionOrigin::Hir(func) = origin else {
                continue;
            };
            let Some(def) = self.func_defs.get(&func).copied() else {
                continue;
            };
            if def.receiver_ty(self.db).is_some() {
                continue;
            }

            // Seed non-generic functions immediately so we always emit them.
            let params = def.params(self.db);
            if params.is_empty() {
                let _ = self.ensure_instance(func, &[], receiver_space, &[], &[]);
                continue;
            }

            let provider_param_count = params
                .iter()
                .filter(
                    |ty| matches!(ty.data(self.db), TyData::TyParam(p) if p.is_effect_provider()),
                )
                .count();
            let is_test = ItemKind::from(func)
                .attrs(self.db)
                .is_some_and(|attrs| attrs.has_attr(self.db, "test"));
            let top_mod_has_contracts = !func.top_mod(self.db).all_contracts(self.db).is_empty();
            if provider_param_count == 0
                || provider_param_count != params.len()
                || (!is_test && top_mod_has_contracts)
            {
                continue;
            }

            let mem_ptr_ctor =
                resolve_lib_type_path(self.db, func.scope(), "core::effect_ref::MemPtr")
                    .unwrap_or_else(|| panic!("missing core type `core::effect_ref::MemPtr`"));
            let assumptions = PredicateListId::empty_list(self.db);
            let root_effect_ty = resolve_default_root_effect_ty(self.db, func.scope(), assumptions);
            let effect_ref_trait =
                resolve_core_trait(self.db, func.scope(), &["effect_ref", "EffectRef"]);
            let effect_ref_mut_trait =
                resolve_core_trait(self.db, func.scope(), &["effect_ref", "EffectRefMut"]);
            let mut args = Vec::with_capacity(provider_param_count);
            let mut can_seed = true;

            for binding in func.effect_bindings(self.db) {
                match binding.key_kind {
                    hir::analysis::ty::effects::EffectKeyKind::Type => {
                        let Some(ty) = binding.key_ty else {
                            continue;
                        };
                        if !ty.is_star_kind(self.db) {
                            continue;
                        }
                        let root_provider_ty = root_effect_ty.filter(|root_effect_ty| {
                            let Some(effect_ref_trait) = effect_ref_trait else {
                                return false;
                            };
                            let goal = TraitInstId::new(
                                self.db,
                                effect_ref_trait,
                                vec![*root_effect_ty, ty],
                                IndexMap::new(),
                            );
                            if !matches!(
                                is_goal_satisfiable(
                                    self.db,
                                    TraitSolveCx::new(self.db, func.scope())
                                        .with_assumptions(assumptions),
                                    goal,
                                ),
                                hir::analysis::ty::trait_resolution::GoalSatisfiability::Satisfied(
                                    _
                                )
                            ) {
                                return false;
                            }
                            if !binding.is_mut {
                                return true;
                            }
                            let Some(effect_ref_mut_trait) = effect_ref_mut_trait else {
                                return false;
                            };
                            let goal = TraitInstId::new(
                                self.db,
                                effect_ref_mut_trait,
                                vec![*root_effect_ty, ty],
                                IndexMap::new(),
                            );
                            matches!(
                                is_goal_satisfiable(
                                    self.db,
                                    TraitSolveCx::new(self.db, func.scope())
                                        .with_assumptions(assumptions),
                                    goal,
                                ),
                                hir::analysis::ty::trait_resolution::GoalSatisfiability::Satisfied(
                                    _
                                )
                            )
                        });
                        args.push(
                            root_provider_ty
                                .unwrap_or_else(|| TyId::app(self.db, mem_ptr_ctor, ty)),
                        );
                    }
                    hir::analysis::ty::effects::EffectKeyKind::Trait => {
                        let Some(root_effect_ty) = root_effect_ty else {
                            can_seed = false;
                            break;
                        };
                        let Some(key_trait) = binding.key_trait else {
                            can_seed = false;
                            break;
                        };
                        let mut trait_args = key_trait.args(self.db).to_vec();
                        if trait_args.is_empty() {
                            can_seed = false;
                            break;
                        }
                        trait_args[0] = root_effect_ty;
                        let goal = TraitInstId::new(
                            self.db,
                            key_trait.def(self.db),
                            trait_args,
                            key_trait.assoc_type_bindings(self.db).clone(),
                        );
                        if !matches!(
                            is_goal_satisfiable(
                                self.db,
                                TraitSolveCx::new(self.db, func.scope())
                                    .with_assumptions(assumptions),
                                goal,
                            ),
                            hir::analysis::ty::trait_resolution::GoalSatisfiability::Satisfied(_)
                        ) {
                            can_seed = false;
                            break;
                        }
                        args.push(root_effect_ty);
                    }
                    hir::analysis::ty::effects::EffectKeyKind::Other => {}
                }
            }

            if can_seed && args.len() == provider_param_count {
                let _ = self.ensure_instance(func, &args, receiver_space, &[], &[]);
            }
        }
    }

    /// Drain the worklist by resolving calls in each newly-created instance.
    fn process_worklist(&mut self) -> MirLowerResult<()> {
        let mut iterations: usize = 0;
        while let Some(func_idx) = self.worklist.pop_front() {
            self.current_symbol = Some(self.instances[func_idx].symbol_name.clone());
            iterations += 1;
            if iterations > 100_000 {
                panic!("monomorphization worklist exceeded 100k iterations; possible cycle");
            }
            self.resolve_calls(func_idx);
            if let Some(err) = self.take_deferred_error() {
                return Err(err);
            }
        }
        if let Some(err) = self.take_deferred_error() {
            return Err(err);
        }
        Ok(())
    }

    /// Inspect every call inside the function at `func_idx` and enqueue its targets.
    fn resolve_calls(&mut self, func_idx: usize) {
        let call_sites: Vec<CallSite<'db>> = {
            let function = &self.instances[func_idx];
            let solve_cx = TraitSolveCx::new(
                self.db,
                match function.origin {
                    crate::ir::MirFunctionOrigin::Hir(func) => func.scope(),
                    crate::ir::MirFunctionOrigin::Synthetic(synth) => match synth {
                        crate::ir::SyntheticId::ContractInitEntrypoint(contract)
                        | crate::ir::SyntheticId::ContractRuntimeEntrypoint(contract)
                        | crate::ir::SyntheticId::ContractInitHandler(contract)
                        | crate::ir::SyntheticId::ContractInitCodeOffset(contract)
                        | crate::ir::SyntheticId::ContractInitCodeLen(contract) => contract.scope(),
                        crate::ir::SyntheticId::ContractRuntimeDispatchArm { contract, .. }
                        | crate::ir::SyntheticId::ContractRecvArmHandler { contract, .. } => {
                            contract.scope()
                        }
                    },
                },
            );
            let mut sites = Vec::new();
            for (bb_idx, block) in function.body.blocks.iter().enumerate() {
                for (inst_idx, inst) in block.insts.iter().enumerate() {
                    if let crate::MirInst::Assign {
                        rvalue: crate::ir::Rvalue::Call(call),
                        ..
                    } = inst
                        && let Some((target_func, args)) = self.resolve_call_target(solve_cx, call)
                    {
                        let receiver_space = canonicalize_receiver_space(call.receiver_space);
                        let effect_param_space_overrides = self.call_effect_param_space_overrides(
                            function,
                            call,
                            target_func,
                            &args,
                            receiver_space,
                        );
                        let param_capability_space_overrides = self
                            .call_param_capability_space_overrides(
                                function,
                                call,
                                target_func,
                                &args,
                                receiver_space,
                            );
                        sites.push(CallSite {
                            bb_idx,
                            inst_idx: Some(inst_idx),
                            target: target_func,
                            args,
                            receiver_space,
                            effect_param_space_overrides,
                            param_capability_space_overrides,
                        });
                    }
                }

                if let crate::Terminator::TerminatingCall {
                    call: crate::ir::TerminatingCall::Call(call),
                    ..
                } = &block.terminator
                    && let Some((target_func, args)) = self.resolve_call_target(solve_cx, call)
                {
                    let receiver_space = canonicalize_receiver_space(call.receiver_space);
                    let effect_param_space_overrides = self.call_effect_param_space_overrides(
                        function,
                        call,
                        target_func,
                        &args,
                        receiver_space,
                    );
                    let param_capability_space_overrides = self
                        .call_param_capability_space_overrides(
                            function,
                            call,
                            target_func,
                            &args,
                            receiver_space,
                        );
                    sites.push(CallSite {
                        bb_idx,
                        inst_idx: None,
                        target: target_func,
                        args,
                        receiver_space,
                        effect_param_space_overrides,
                        param_capability_space_overrides,
                    });
                }
            }
            sites
        };

        let func_item_sites = {
            let function = &self.instances[func_idx];
            function
                .body
                .values
                .iter()
                .enumerate()
                .filter_map(|(value_idx, value)| {
                    if let crate::ValueOrigin::CodeRegionRef(root) = &value.origin {
                        Some((value_idx, root.clone()))
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>()
        };

        for CallSite {
            bb_idx,
            inst_idx,
            target,
            args,
            receiver_space,
            effect_param_space_overrides,
            param_capability_space_overrides,
        } in call_sites
        {
            let resolved_name = match target {
                CallTarget::Template(func) => {
                    let Some((_, symbol)) = self.ensure_instance(
                        func,
                        &args,
                        receiver_space,
                        &effect_param_space_overrides,
                        &param_capability_space_overrides,
                    ) else {
                        if self.deferred_error.borrow().is_some() {
                            return;
                        }

                        let name = func.pretty_print_signature(self.db);
                        self.defer_error(MirLowerError::Unsupported {
                            func_name: name,
                            message: "failed to instantiate MIR".to_string(),
                        });
                        return;
                    };
                    Some(symbol)
                }
                CallTarget::Decl(func) => {
                    let (normalized_args, normalized_effect_param_space_overrides, _) = self
                        .normalize_call_instance_inputs(
                            func,
                            &args,
                            &effect_param_space_overrides,
                            &param_capability_space_overrides,
                        );
                    Some(self.mangled_name(
                        func,
                        &normalized_args,
                        receiver_space,
                        &normalized_effect_param_space_overrides,
                        &[],
                    ))
                }
                CallTarget::Synthetic(origin) => {
                    let (_, symbol) = self
                        .ensure_synthetic_instance(
                            origin,
                            receiver_space,
                            &effect_param_space_overrides,
                            &param_capability_space_overrides,
                        )
                        .unwrap_or_else(|| {
                            panic!("failed to instantiate synthetic MIR for `{origin:?}`")
                        });
                    Some(symbol)
                }
            };

            if let Some(name) = resolved_name {
                match inst_idx {
                    Some(inst_idx) => {
                        let inst =
                            &mut self.instances[func_idx].body.blocks[bb_idx].insts[inst_idx];
                        if let crate::MirInst::Assign {
                            rvalue: crate::ir::Rvalue::Call(call),
                            ..
                        } = inst
                        {
                            call.resolved_name = Some(name);
                        }
                    }
                    None => {
                        let term = &mut self.instances[func_idx].body.blocks[bb_idx].terminator;
                        if let crate::Terminator::TerminatingCall {
                            call: crate::ir::TerminatingCall::Call(call),
                            ..
                        } = term
                        {
                            call.resolved_name = Some(name);
                        }
                    }
                }
            }
        }

        for (value_idx, target) in func_item_sites {
            let symbol = match target.origin {
                crate::ir::MirFunctionOrigin::Hir(func) => {
                    let Some((_, symbol)) =
                        self.ensure_instance(func, &target.generic_args, None, &[], &[])
                    else {
                        if self.deferred_error.borrow().is_some() {
                            return;
                        }

                        let name = func.pretty_print(self.db);
                        panic!("failed to instantiate MIR for `{name}`");
                    };
                    symbol
                }
                crate::ir::MirFunctionOrigin::Synthetic(_) => {
                    self.ensure_synthetic_instance(target.origin, None, &[], &[])
                        .unwrap_or_else(|| {
                            panic!("failed to instantiate synthetic MIR for `{target:?}`")
                        })
                        .1
                }
            };
            if let crate::ValueOrigin::CodeRegionRef(target) =
                &mut self.instances[func_idx].body.values[value_idx].origin
            {
                target.symbol = Some(symbol);
            }
        }

        // Rewrite AbortWithValue terminators: either redirect to std::evm::effects::revert<T>
        // when T satisfies Encode<Sol> + AbiSize, or emit a compile error.
        self.rewrite_abort_with_value(func_idx);
    }

    /// Scan the function at `func_idx` for `AbortWithValue` calls (both regular instructions
    /// and terminators) and either redirect them to `std::evm::effects::revert<T>()` (when
    /// the error type is ABI-encodable) or emit a compile error.
    fn rewrite_abort_with_value(&mut self, func_idx: usize) {
        let function = &self.instances[func_idx];

        let mut sites: Vec<AbortWithValueSite<'db>> = Vec::new();

        for (bb_idx, block) in function.body.blocks.iter().enumerate() {
            // Check regular call instructions
            for (inst_idx, inst) in block.insts.iter().enumerate() {
                if let crate::MirInst::Assign {
                    rvalue: crate::ir::Rvalue::Call(call),
                    ..
                } = inst
                    && call.builtin_terminator
                        == Some(crate::ir::BuiltinTerminatorKind::AbortWithValue)
                    && let Some(crate::ir::CallTargetRef::Hir(hir_target)) = call.target.as_ref()
                    && let Some(&concrete_t) = hir_target.generic_args.first()
                {
                    sites.push(AbortWithValueSite::Inst {
                        bb_idx,
                        inst_idx,
                        concrete_t,
                    });
                }
            }

            // Check terminating calls
            if let crate::Terminator::TerminatingCall {
                call: crate::ir::TerminatingCall::Call(call),
                ..
            } = &block.terminator
                && call.builtin_terminator == Some(crate::ir::BuiltinTerminatorKind::AbortWithValue)
                && let Some(crate::ir::CallTargetRef::Hir(hir_target)) = call.target.as_ref()
                && let Some(&concrete_t) = hir_target.generic_args.first()
            {
                sites.push(AbortWithValueSite::Term { bb_idx, concrete_t });
            }
        }

        for site in sites {
            let concrete_t = match &site {
                AbortWithValueSite::Inst { concrete_t, .. } => *concrete_t,
                AbortWithValueSite::Term { concrete_t, .. } => *concrete_t,
            };

            // We need a scope that can see both core traits (Encode, AbiSize)
            // and std types (Sol, revert). The concrete type's ingot works for
            // user-defined types, but primitives and core types return
            // `None` from `ingot()`. In that case we fall back to the std
            // ingot's root scope, which can see everything we need.
            //
            // If std is not available at all (bare-core compilation),
            // `std_aware_scope` is `None` and we skip the trait check
            // entirely, falling back to a plain empty revert.
            let std_aware_scope = concrete_t
                .ingot(self.db)
                .filter(|ingot| {
                    // Core ingot can't see std; skip it so we hit the
                    // std-ingot fallback below.
                    ingot.kind(self.db) != common::ingot::IngotKind::Core
                })
                .or_else(|| self.find_std_ingot(func_idx))
                .map(|ingot| ingot.root_mod(self.db).scope());

            // Check if T: Encode<Sol> + AbiSize (only possible when std is present).
            let can_encode = std_aware_scope.and_then(|scope| {
                let solve_cx = TraitSolveCx::new(self.db, scope);

                let encode_trait = resolve_core_trait(self.db, scope, &["abi", "Encode"])?;
                let abi_size_trait = resolve_core_trait(self.db, scope, &["abi", "AbiSize"])?;
                let sol_ty = resolve_lib_type_path(self.db, scope, "std::abi::Sol")?;

                let encode_inst = TraitInstId::new(
                    self.db,
                    encode_trait,
                    vec![concrete_t, sol_ty],
                    IndexMap::new(),
                );
                if !is_goal_satisfiable(self.db, solve_cx, encode_inst).is_satisfied() {
                    return None;
                }

                let abi_size_inst =
                    TraitInstId::new(self.db, abi_size_trait, vec![concrete_t], IndexMap::new());
                if !is_goal_satisfiable(self.db, solve_cx, abi_size_inst).is_satisfied() {
                    return None;
                }

                Some(scope)
            });

            // Helper: promote an instruction-level call to a TerminatingCall
            // by removing the instruction and all subsequent ones, then setting
            // the block terminator.
            let promote_inst_to_terminator =
                |instances: &mut Vec<MirFunction<'db>>,
                 bb_idx: usize,
                 inst_idx: usize,
                 mut call: CallOrigin<'db>,
                 source: crate::ir::SourceInfoId,
                 resolved_name: Option<String>| {
                    if let Some(name) = resolved_name {
                        call.resolved_name = Some(name);
                        call.builtin_terminator = None;
                    } else {
                        call.builtin_terminator = Some(crate::ir::BuiltinTerminatorKind::Abort);
                    }
                    let block = &mut instances[func_idx].body.blocks[bb_idx];
                    block.insts.truncate(inst_idx);
                    block.terminator = crate::Terminator::TerminatingCall {
                        source,
                        call: crate::ir::TerminatingCall::Call(call),
                    };
                };

            if let Some(scope) = can_encode {
                let revert_symbol =
                    resolve_lib_func_path(self.db, scope, "std::evm::effects::revert").and_then(
                        |revert_func| {
                            self.ensure_instance(revert_func, &[concrete_t], None, &[], &[])
                                .map(|(_, symbol)| symbol)
                        },
                    );

                let Some(symbol) = revert_symbol else {
                    let ty_name = concrete_t.pretty_print(self.db);
                    let func_name = self
                        .current_symbol
                        .clone()
                        .unwrap_or_else(|| self.instances[func_idx].symbol_name.clone());
                    self.defer_error(MirLowerError::Unsupported {
                        func_name,
                        message: format!(
                            "failed to instantiate `revert<{ty_name}>()` for `unwrap()`"
                        ),
                    });
                    return;
                };

                match site {
                    AbortWithValueSite::Inst {
                        bb_idx, inst_idx, ..
                    } => {
                        // Extract the call from the instruction, then promote
                        // it to a TerminatingCall.
                        let inst =
                            self.instances[func_idx].body.blocks[bb_idx].insts[inst_idx].clone();
                        if let crate::MirInst::Assign {
                            rvalue: crate::ir::Rvalue::Call(call),
                            source,
                            ..
                        } = inst
                        {
                            promote_inst_to_terminator(
                                &mut self.instances,
                                bb_idx,
                                inst_idx,
                                call,
                                source,
                                Some(symbol),
                            );
                        }
                    }
                    AbortWithValueSite::Term { bb_idx, .. } => {
                        let term = &mut self.instances[func_idx].body.blocks[bb_idx].terminator;
                        if let crate::Terminator::TerminatingCall {
                            call: crate::ir::TerminatingCall::Call(call),
                            ..
                        } = term
                        {
                            call.resolved_name = Some(symbol);
                            call.builtin_terminator = None;
                        }
                    }
                }
            } else if std_aware_scope.is_some() {
                // std is available but the type doesn't satisfy the bounds —
                // this is a real user error.
                let ty_name = concrete_t.pretty_print(self.db);
                let func_name = self
                    .current_symbol
                    .clone()
                    .unwrap_or_else(|| self.instances[func_idx].symbol_name.clone());
                self.defer_error(MirLowerError::Unsupported {
                    func_name,
                    message: format!(
                        "`unwrap()` requires the error type `{ty_name}` to implement `Encode<Sol>` and `AbiSize`"
                    ),
                });
                return;
            } else {
                // std is not available (bare-core compilation) — no ABI
                // encoding infrastructure exists, so fall back to empty revert.
                self.downgrade_abort_with_value_to_abort(func_idx, site);
            }
        }
    }

    /// Downgrade an `AbortWithValue` call to a plain `Abort` (empty revert).
    /// Used when std is not available and ABI encoding is impossible.
    fn downgrade_abort_with_value_to_abort(
        &mut self,
        func_idx: usize,
        site: AbortWithValueSite<'db>,
    ) {
        match site {
            AbortWithValueSite::Inst {
                bb_idx, inst_idx, ..
            } => {
                let inst = self.instances[func_idx].body.blocks[bb_idx].insts[inst_idx].clone();
                if let crate::MirInst::Assign {
                    rvalue: crate::ir::Rvalue::Call(mut call),
                    source,
                    ..
                } = inst
                {
                    call.builtin_terminator = Some(crate::ir::BuiltinTerminatorKind::Abort);
                    let block = &mut self.instances[func_idx].body.blocks[bb_idx];
                    block.insts.truncate(inst_idx);
                    block.terminator = crate::Terminator::TerminatingCall {
                        source,
                        call: crate::ir::TerminatingCall::Call(call),
                    };
                }
            }
            AbortWithValueSite::Term { bb_idx, .. } => {
                let term = &mut self.instances[func_idx].body.blocks[bb_idx].terminator;
                if let crate::Terminator::TerminatingCall {
                    call: crate::ir::TerminatingCall::Call(call),
                    ..
                } = term
                {
                    call.builtin_terminator = Some(crate::ir::BuiltinTerminatorKind::Abort);
                }
            }
        }
    }

    fn ensure_synthetic_instance(
        &mut self,
        origin: crate::ir::MirFunctionOrigin<'db>,
        receiver_space: Option<AddressSpaceKind>,
        effect_param_space_overrides: &[Option<AddressSpaceKind>],
        param_capability_space_overrides: &[Vec<(
            crate::MirProjectionPath<'db>,
            AddressSpaceKind,
        )>],
    ) -> Option<(usize, String)> {
        let receiver_space = canonicalize_receiver_space(receiver_space);
        debug_assert!(
            matches!(origin, crate::ir::MirFunctionOrigin::Synthetic(_)),
            "ensure_synthetic_instance called with non-synthetic origin"
        );

        let template_idx = *self.func_index.get(&TemplateKey {
            origin,
            receiver_space,
        })?;
        let template = &self.templates[template_idx];
        let normalized_effect_param_space_overrides = self
            .normalize_effect_param_space_overrides_for_len(
                template.body.effect_param_locals.len(),
                effect_param_space_overrides,
            );
        let normalized_param_capability_space_overrides = self
            .normalize_param_capability_space_overrides_for_len(
                template.body.param_locals.len(),
                param_capability_space_overrides,
            );

        let key = InstanceKey::new(
            origin,
            &[],
            receiver_space,
            &normalized_effect_param_space_overrides,
            &normalized_param_capability_space_overrides,
        );
        if let Some(&idx) = self.instance_map.get(&key) {
            let symbol = self.instances[idx].symbol_name.clone();
            return Some((idx, symbol));
        }

        let mut instance = template.clone();
        self.apply_synthetic_param_capability_space_overrides(
            &mut instance,
            &normalized_param_capability_space_overrides,
        );
        self.apply_synthetic_effect_param_space_overrides(
            &mut instance,
            &normalized_effect_param_space_overrides,
        );
        let mut symbol = instance.symbol_name.clone();
        let effect_suffix = effect_param_space_suffix(&normalized_effect_param_space_overrides);
        if !effect_suffix.is_empty() {
            symbol = format!("{symbol}_{effect_suffix}");
        }
        let param_cap_suffix =
            param_capability_space_suffix(&normalized_param_capability_space_overrides);
        if !param_cap_suffix.is_empty() {
            symbol = format!("{symbol}_{param_cap_suffix}");
        }
        instance.symbol_name = symbol.clone();

        let idx = self.instances.len();
        self.instances.push(instance);
        self.instance_map.insert(key, idx);
        self.worklist.push_back(idx);

        Some((idx, symbol))
    }

    /// Ensure a `(func, args)` instance exists, cloning and substituting if needed.
    fn ensure_instance(
        &mut self,
        func: Func<'db>,
        args: &[TyId<'db>],
        receiver_space: Option<AddressSpaceKind>,
        effect_param_space_overrides: &[Option<AddressSpaceKind>],
        param_capability_space_overrides: &[Vec<(
            crate::MirProjectionPath<'db>,
            AddressSpaceKind,
        )>],
    ) -> Option<(usize, String)> {
        let receiver_space = canonicalize_receiver_space(receiver_space);
        let (
            normalized_args,
            normalized_effect_param_space_overrides,
            normalized_param_capability_space_overrides,
        ) = self.normalize_call_instance_inputs(
            func,
            args,
            effect_param_space_overrides,
            param_capability_space_overrides,
        );
        let norm_scope = crate::ty::normalization_scope_for_args(self.db, func, &normalized_args);
        let mut assumptions = PredicateListId::empty_list(self.db);

        let key = InstanceKey::new(
            crate::ir::MirFunctionOrigin::Hir(func),
            &normalized_args,
            receiver_space,
            &normalized_effect_param_space_overrides,
            &normalized_param_capability_space_overrides,
        );
        if let Some(&idx) = self.instance_map.get(&key) {
            let symbol = self.instances[idx].symbol_name.clone();
            return Some((idx, symbol));
        }

        let symbol_name = self.mangled_name(
            func,
            &normalized_args,
            receiver_space,
            &normalized_effect_param_space_overrides,
            &normalized_param_capability_space_overrides,
        );
        let mut instance = if args.is_empty()
            && normalized_effect_param_space_overrides.is_empty()
            && normalized_param_capability_space_overrides.is_empty()
        {
            let template_idx = self.ensure_template(func, receiver_space)?;
            let mut instance = self.templates[template_idx].clone();
            instance.receiver_space = receiver_space;
            instance.symbol_name = symbol_name.clone();
            self.apply_substitution(&mut instance);
            instance
        } else {
            let (diags, typed_body) = check_func_body(self.db, func);
            if !diags.is_empty() {
                let func_name = func.pretty_print_signature(self.db);
                let diagnostics = format_diags(self.db, diags);
                self.defer_error(MirLowerError::AnalysisDiagnostics {
                    func_name,
                    diagnostics,
                });
                return None;
            }
            let mut folder = ParamSubstFolder {
                args: &normalized_args,
            };
            let typed_body = typed_body.clone().fold_with(self.db, &mut folder);
            assumptions = typed_body.assumptions();

            // After substitution, normalize any remaining associated types.
            let mut normalizer = NormalizeFolder {
                scope: norm_scope,
                assumptions,
            };
            let typed_body = typed_body.fold_with(self.db, &mut normalizer);
            let mut instance = match lower_function(
                self.db,
                func,
                typed_body,
                receiver_space,
                normalized_args.clone(),
                normalized_effect_param_space_overrides.clone(),
                normalized_param_capability_space_overrides.clone(),
            ) {
                Ok(instance) => instance,
                Err(err) => {
                    self.defer_error(err);
                    return None;
                }
            };
            instance.receiver_space = receiver_space;
            instance.symbol_name = symbol_name.clone();
            instance
        };

        let ret_ty = CallableDef::Func(func)
            .ret_ty(self.db)
            .instantiate(self.db, &normalized_args);
        let ret_ty = normalize_ty(self.db, ret_ty, norm_scope, assumptions);
        instance.ret_ty = ret_ty;
        instance.returns_value = !crate::layout::is_zero_sized_ty(self.db, ret_ty);
        instance.generic_args = normalized_args;

        let idx = self.instances.len();
        let symbol = instance.symbol_name.clone();
        self.instances.push(instance);
        self.instance_map.insert(key, idx);
        self.worklist.push_back(idx);
        Some((idx, symbol))
    }

    fn normalize_call_instance_inputs(
        &self,
        func: Func<'db>,
        args: &[TyId<'db>],
        effect_param_space_overrides: &[Option<AddressSpaceKind>],
        param_capability_space_overrides: &[Vec<(
            crate::MirProjectionPath<'db>,
            AddressSpaceKind,
        )>],
    ) -> NormalizedCallInstanceInputs<'db> {
        let norm_scope = crate::ty::normalization_scope_for_args(self.db, func, args);
        let assumptions = PredicateListId::empty_list(self.db);
        let normalized_args: Vec<_> = args
            .iter()
            .copied()
            .map(|ty| {
                let normalized = normalize_ty(self.db, ty, norm_scope, assumptions);
                if !ty.has_invalid(self.db) && normalized.has_invalid(self.db) {
                    ty
                } else {
                    normalized
                }
            })
            .collect();
        let normalized_effect_param_space_overrides =
            self.normalize_effect_param_space_overrides(func, effect_param_space_overrides);
        let normalized_param_capability_space_overrides =
            self.normalize_param_capability_space_overrides(func, param_capability_space_overrides);
        (
            normalized_args,
            normalized_effect_param_space_overrides,
            normalized_param_capability_space_overrides,
        )
    }

    fn apply_synthetic_param_capability_space_overrides(
        &self,
        instance: &mut MirFunction<'db>,
        param_capability_space_overrides: &[Vec<(
            crate::MirProjectionPath<'db>,
            AddressSpaceKind,
        )>],
    ) {
        let core = self.core_for_origin(instance.origin);
        for (idx, entries) in param_capability_space_overrides.iter().enumerate() {
            let Some(&param_local) = instance.body.param_locals.get(idx) else {
                break;
            };
            let local = &mut instance.body.locals[param_local.index()];
            local.pointer_leaf_infos = pointer_leaf_infos_for_ty_with_default(
                self.db,
                &core,
                local.ty,
                local.address_space,
            );
            for (path, space) in entries {
                crate::repr::apply_param_capability_space_override(
                    self.db, &core, local, path, *space,
                );
            }
        }
    }

    fn apply_synthetic_effect_param_space_overrides(
        &self,
        instance: &mut MirFunction<'db>,
        effect_param_space_overrides: &[Option<AddressSpaceKind>],
    ) {
        let core = self.core_for_origin(instance.origin);
        for (idx, space) in effect_param_space_overrides.iter().enumerate() {
            let Some(space) = *space else {
                continue;
            };
            let Some(&effect_local) = instance.body.effect_param_locals.get(idx) else {
                break;
            };
            let local = &mut instance.body.locals[effect_local.index()];
            crate::repr::set_declared_local_address_space(self.db, &core, local, space);
        }
    }

    /// Returns the concrete HIR function targeted by the given call, accounting for trait impls.
    fn resolve_call_target(
        &self,
        solve_cx: TraitSolveCx<'db>,
        call: &CallOrigin<'db>,
    ) -> Option<(CallTarget<'db>, Vec<TyId<'db>>)> {
        let target = call.target.as_ref()?;
        let crate::ir::CallTargetRef::Hir(hir_target) = target else {
            let crate::ir::CallTargetRef::Synthetic(id) = target else {
                unreachable!();
            };
            return Some((
                CallTarget::Synthetic(crate::ir::MirFunctionOrigin::Synthetic(*id)),
                Vec::new(),
            ));
        };
        let base_args = hir_target.generic_args.clone();
        if let Some(inst) = hir_target.trait_inst {
            let method_name = hir_target
                .callable_def
                .name(self.db)
                .expect("trait method call missing name");
            if let Some(origin) = self.resolve_contract_metadata_call(inst, method_name) {
                return Some((CallTarget::Synthetic(origin), Vec::new()));
            }
            let trait_arg_len = inst.args(self.db).len();
            if base_args.len() < trait_arg_len {
                let inst_desc = inst.pretty_print(self.db, false);
                let name = method_name.data(self.db);
                self.defer_error(MirLowerError::Unsupported {
                    func_name: self
                        .current_symbol
                        .clone()
                        .unwrap_or_else(|| "<unknown function>".to_string()),
                    message: format!(
                        "trait method `{name}` args too short for `{inst_desc}`: got {}, expected at least {}",
                        base_args.len(),
                        trait_arg_len
                    ),
                });
                return None;
            }
            if let Some((func, impl_args)) =
                resolve_trait_method_instance(self.db, solve_cx, inst, method_name)
            {
                let mut resolved_args = impl_args;
                resolved_args.extend_from_slice(&base_args[trait_arg_len..]);
                return Some((CallTarget::Template(func), resolved_args));
            }

            if let CallableDef::Func(func) = hir_target.callable_def
                && func.body(self.db).is_some()
            {
                return Some((CallTarget::Template(func), base_args));
            }

            let inst_desc = inst.pretty_print(self.db, true);
            let name = method_name.data(self.db);
            let current = self
                .current_symbol
                .as_deref()
                .unwrap_or("<unknown function>");
            self.defer_error(MirLowerError::Unsupported {
                func_name: current.to_string(),
                message: format!(
                    "failed to resolve trait method `{name}` for `{inst_desc}` (no impl and no default)"
                ),
            });
            return None;
        }

        if let CallableDef::Func(func) = hir_target.callable_def {
            if func.body(self.db).is_some() {
                return Some((CallTarget::Template(func), base_args));
            }
            if let Some(parent) = func.scope().parent(self.db)
                && let ScopeId::Item(item) = parent
                && matches!(item, ItemKind::Trait(_) | ItemKind::ImplTrait(_))
            {
                let name = func.pretty_print_signature(self.db);
                let current = self
                    .current_symbol
                    .as_deref()
                    .unwrap_or("<unknown function>");
                self.defer_error(MirLowerError::Unsupported {
                    func_name: current.to_string(),
                    message: format!("unresolved trait method `{name}` during monomorphization"),
                });
                return None;
            }
            return Some((CallTarget::Decl(func), base_args));
        }
        None
    }

    fn resolve_contract_metadata_call(
        &self,
        inst: hir::analysis::ty::trait_def::TraitInstId<'db>,
        method_name: hir::hir_def::IdentId<'db>,
    ) -> Option<crate::ir::MirFunctionOrigin<'db>> {
        let trait_def = inst.def(self.db);
        let trait_name = trait_def.name(self.db).to_opt()?;
        if trait_name.data(self.db) != "Contract" {
            return None;
        }
        if trait_def.top_mod(self.db).ingot(self.db).kind(self.db) != common::ingot::IngotKind::Std
        {
            return None;
        }

        let contract = inst.args(self.db).first().copied()?.as_contract(self.db)?;
        let synth = match method_name.data(self.db).as_str() {
            "init_code_offset" => crate::ir::SyntheticId::ContractInitCodeOffset(contract),
            "init_code_len" => crate::ir::SyntheticId::ContractInitCodeLen(contract),
            _ => return None,
        };
        Some(crate::ir::MirFunctionOrigin::Synthetic(synth))
    }

    /// Substitute concrete type arguments directly into the MIR body values.
    fn apply_substitution(&self, function: &mut MirFunction<'db>) {
        let mut folder = ParamSubstFolder {
            args: &function.generic_args,
        };

        for value in &mut function.body.values {
            value.ty = value.ty.fold_with(self.db, &mut folder);
            if let crate::ValueOrigin::CodeRegionRef(target) = &mut value.origin {
                target.generic_args = target
                    .generic_args
                    .iter()
                    .map(|ty| ty.fold_with(self.db, &mut folder))
                    .collect();
                target.symbol = match target.origin {
                    crate::ir::MirFunctionOrigin::Hir(func) => {
                        Some(self.mangled_name(func, &target.generic_args, None, &[], &[]))
                    }
                    crate::ir::MirFunctionOrigin::Synthetic(_) => self
                        .func_index
                        .get(&TemplateKey {
                            origin: target.origin,
                            receiver_space: None,
                        })
                        .map(|idx| self.templates[*idx].symbol_name.clone()),
                };
            }
        }

        for block in &mut function.body.blocks {
            for inst in &mut block.insts {
                if let crate::MirInst::Assign {
                    rvalue: crate::ir::Rvalue::Call(call),
                    ..
                } = inst
                    && let Some(crate::ir::CallTargetRef::Hir(target)) = &mut call.target
                {
                    target.generic_args = target
                        .generic_args
                        .iter()
                        .map(|ty| ty.fold_with(self.db, &mut folder))
                        .collect();
                    if let Some(inst) = target.trait_inst {
                        target.trait_inst = Some(inst.fold_with(self.db, &mut folder));
                    }
                    // Clear resolved_name so it will be re-resolved after
                    // type substitution. Non-HIR calls keep `resolved_name`
                    // as-is; typed checked intrinsics are tracked via
                    // `call.checked_intrinsic`.
                    call.resolved_name = None;
                }
            }

            if let crate::Terminator::TerminatingCall {
                call: crate::ir::TerminatingCall::Call(call),
                ..
            } = &mut block.terminator
                && let Some(crate::ir::CallTargetRef::Hir(target)) = &mut call.target
            {
                target.generic_args = target
                    .generic_args
                    .iter()
                    .map(|ty| ty.fold_with(self.db, &mut folder))
                    .collect();
                if let Some(inst) = target.trait_inst {
                    target.trait_inst = Some(inst.fold_with(self.db, &mut folder));
                }
                call.resolved_name = None;
            }
        }
    }

    /// Produce a globally unique (yet mostly readable) symbol name per instance.
    fn mangled_name(
        &self,
        func: Func<'db>,
        args: &[TyId<'db>],
        receiver_space: Option<AddressSpaceKind>,
        effect_param_space_overrides: &[Option<AddressSpaceKind>],
        param_capability_space_overrides: &[Vec<(
            crate::MirProjectionPath<'db>,
            AddressSpaceKind,
        )>],
    ) -> String {
        let receiver_space = canonicalize_receiver_space(receiver_space);
        let mut base = self.base_name_root_without_disambiguation(func, receiver_space);
        if self.ambiguous_bases.contains(&base) {
            let qualifier = self.function_qualifier(func);
            base = format!("{qualifier}_{base}");
        }

        let effect_suffix = effect_param_space_suffix(effect_param_space_overrides);
        if !effect_suffix.is_empty() {
            base = format!("{base}_{effect_suffix}");
        }
        let param_cap_suffix = param_capability_space_suffix(param_capability_space_overrides);
        if !param_cap_suffix.is_empty() {
            base = format!("{base}_{param_cap_suffix}");
        }

        if args.is_empty() {
            return base;
        }

        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        let mut parts = Vec::with_capacity(args.len());
        for ty in args {
            let pretty = self.type_mangle_component(*ty);
            pretty.hash(&mut hasher);
            parts.push(sanitize_symbol_component(&pretty));
        }
        let hash = hasher.finish();
        let suffix = parts.join("_");
        format!("{base}__{suffix}__{hash:08x}")
    }

    fn module_path_segments(&self, mut scope: ScopeId<'db>) -> Vec<String> {
        let ingot = scope.ingot(self.db);
        let root_mod = ingot.root_mod(self.db);

        let mut parts = Vec::new();
        while let Some(parent) = scope.parent_module(self.db) {
            match parent {
                ScopeId::Item(ItemKind::Mod(mod_)) => {
                    if let Some(name) = mod_.name(self.db).to_opt() {
                        parts.push(name.data(self.db).to_string());
                    }
                }
                ScopeId::Item(ItemKind::TopMod(top_mod)) => {
                    if top_mod != root_mod {
                        parts.push(top_mod.name(self.db).data(self.db).to_string());
                    }
                }
                _ => {}
            }
            scope = parent;
        }

        parts.reverse();
        parts
    }

    fn qualified_path_hash(&self, parts: &[String]) -> u64 {
        stable_hash(parts)
    }

    fn ty_identity_hash(&self, ty: TyId<'db>) -> u64 {
        match ty.data(self.db) {
            TyData::TyBase(TyBase::Adt(adt)) => {
                let name = adt
                    .adt_ref(self.db)
                    .name(self.db)
                    .map(|id| id.data(self.db).to_string())
                    .unwrap_or_else(|| "<unknown>".to_string());
                let mut parts = self.module_path_segments(adt.scope(self.db));
                parts.push(name);
                self.qualified_path_hash(&parts)
            }
            TyData::TyBase(TyBase::Contract(contract)) => {
                let name = contract
                    .name(self.db)
                    .to_opt()
                    .map(|id| id.data(self.db).to_string())
                    .unwrap_or_else(|| "<unknown>".to_string());
                let mut parts = self.module_path_segments(contract.scope());
                parts.push(name);
                self.qualified_path_hash(&parts)
            }
            _ => stable_hash(&ty.pretty_print(self.db).to_string()),
        }
    }

    fn trait_ref_identity_hash(&self, trait_ref: hir::hir_def::TraitRefId<'db>) -> Option<u64> {
        let path = trait_ref.path(self.db).to_opt()?;
        let parts = self.path_ident_segments(path);
        Some(self.qualified_path_hash(&parts))
    }

    fn path_ident_segments(&self, path: PathId<'db>) -> Vec<String> {
        let mut parts = Vec::new();
        let mut current = Some(path);
        while let Some(path) = current {
            match path.kind(self.db) {
                PathKind::Ident { ident, .. } => {
                    if let Some(ident) = ident.to_opt() {
                        parts.push(ident.data(self.db).to_string());
                    }
                }
                PathKind::QualifiedType { .. } => {}
            }
            current = path.parent(self.db);
        }
        parts.reverse();
        parts
    }

    fn type_mangle_component(&self, ty: TyId<'db>) -> String {
        match ty.data(self.db) {
            TyData::TyBase(TyBase::Func(callable)) => match callable {
                CallableDef::Func(func) => {
                    let name = func
                        .name(self.db)
                        .to_opt()
                        .map(|ident| ident.data(self.db).to_string())
                        .unwrap_or_else(|| "<unknown>".to_string());
                    let mut parts = self.module_path_segments(func.scope());
                    parts.push(name.clone());
                    let hash = self.qualified_path_hash(&parts);
                    format!("fn {name}_h{hash:08x}")
                }
                CallableDef::VariantCtor(_) => ty.pretty_print(self.db).to_string(),
            },
            TyData::TyBase(TyBase::Adt(adt)) => {
                let name = adt
                    .adt_ref(self.db)
                    .name(self.db)
                    .map(|id| id.data(self.db).to_string())
                    .unwrap_or_else(|| "<unknown>".to_string());
                let hash = self.ty_identity_hash(ty);
                format!("{name}_h{hash:08x}")
            }
            TyData::TyBase(TyBase::Contract(contract)) => {
                let name = contract
                    .name(self.db)
                    .to_opt()
                    .map(|id| id.data(self.db).to_string())
                    .unwrap_or_else(|| "<unknown>".to_string());
                let hash = self.ty_identity_hash(ty);
                format!("{name}_h{hash:08x}")
            }
            _ => ty.pretty_print(self.db).to_string(),
        }
    }

    fn base_name_root_without_disambiguation(
        &self,
        func: Func<'db>,
        receiver_space: Option<AddressSpaceKind>,
    ) -> String {
        let mut base = func
            .name(self.db)
            .to_opt()
            .map(|ident| ident.data(self.db).to_string())
            .unwrap_or_else(|| "<anonymous>".into());
        if let Some(prefix) = self.associated_prefix(func) {
            base = format!("{prefix}_{base}");
        }
        if let Some(space) = receiver_space {
            let suffix = match space {
                AddressSpaceKind::Memory => "mem",
                AddressSpaceKind::Calldata => "calldata",
                AddressSpaceKind::Storage => "stor",
                AddressSpaceKind::TransientStorage => "tstor",
                AddressSpaceKind::Code => "code",
            };
            base = format!("{base}_{suffix}");
        }
        base
    }

    fn function_qualifier(&self, func: Func<'db>) -> String {
        let parts = self.module_path_segments(func.scope());
        if parts.is_empty() {
            return "root".to_string();
        }

        let human = sanitize_symbol_component(&parts.join("_")).to_lowercase();
        let hash = self.qualified_path_hash(&parts);
        format!("{human}_h{hash:08x}")
    }

    /// Returns a sanitized prefix for associated functions/methods based on their owner.
    fn associated_prefix(&self, func: Func<'db>) -> Option<String> {
        let parent = func.scope().parent(self.db)?;
        let ScopeId::Item(item) = parent else {
            return None;
        };
        if let ItemKind::Impl(impl_block) = item {
            let ty = impl_block.ty(self.db);
            if ty.has_invalid(self.db) {
                return None;
            }
            let ty_name = sanitize_symbol_component(ty.pretty_print(self.db)).to_lowercase();
            let hash = self.ty_identity_hash(ty);
            Some(format!("{ty_name}_h{hash:08x}"))
        } else if let ItemKind::ImplTrait(impl_trait) = item {
            let ty = impl_trait.ty(self.db);
            if ty.has_invalid(self.db) {
                return None;
            }
            let self_name = sanitize_symbol_component(ty.pretty_print(self.db)).to_lowercase();
            let self_hash = self.ty_identity_hash(ty);
            let self_part = format!("{self_name}_h{self_hash:08x}");

            let trait_name = impl_trait
                .hir_trait_ref(self.db)
                .to_opt()
                .and_then(|trait_ref| trait_ref.path(self.db).to_opt())
                .and_then(|path| path.segment(self.db, path.segment_index(self.db)))
                .and_then(|segment| match segment.kind(self.db) {
                    PathKind::Ident { ident, .. } => ident.to_opt(),
                    PathKind::QualifiedType { .. } => None,
                })
                .map(|ident| sanitize_symbol_component(ident.data(self.db)).to_lowercase())
                .unwrap_or_else(|| "trait".to_string());

            let trait_hash = impl_trait
                .hir_trait_ref(self.db)
                .to_opt()
                .and_then(|trait_ref| self.trait_ref_identity_hash(trait_ref))
                .unwrap_or(0);
            let trait_part = format!("{trait_name}_h{trait_hash:08x}");

            Some(format!("{self_part}_{trait_part}"))
        } else {
            None
        }
    }

    fn into_instances(self) -> Vec<MirFunction<'db>> {
        self.instances
    }

    /// Ensure we have lowered MIR for `func`, lowering on demand for dependency ingots.
    fn ensure_template(
        &mut self,
        func: Func<'db>,
        receiver_space: Option<AddressSpaceKind>,
    ) -> Option<usize> {
        let receiver_space = canonicalize_receiver_space(receiver_space);
        let key = TemplateKey {
            origin: crate::ir::MirFunctionOrigin::Hir(func),
            receiver_space,
        };
        if let Some(&idx) = self.func_index.get(&key) {
            return Some(idx);
        }

        let (diags, typed_body) = check_func_body(self.db, func);
        if !diags.is_empty() {
            let func_name = func.pretty_print_signature(self.db);
            let diagnostics = format_diags(self.db, diags);
            self.defer_error(MirLowerError::AnalysisDiagnostics {
                func_name,
                diagnostics,
            });
            return None;
        }
        let lowered = lower_function(
            self.db,
            func,
            typed_body.clone(),
            receiver_space,
            Vec::new(),
            Vec::new(),
            Vec::new(),
        );
        let lowered = match lowered {
            Ok(lowered) => lowered,
            Err(err) => {
                self.defer_error(err);
                return None;
            }
        };
        let idx = self.templates.len();
        self.templates.push(lowered);
        self.func_index.insert(key, idx);
        if let Some(def) = func.as_callable(self.db) {
            self.func_defs.insert(func, def);
        }
        Some(idx)
    }

    fn arg_capability_space_at_path(
        &self,
        caller: &MirFunction<'db>,
        arg_value: crate::ValueId,
        path: &crate::MirProjectionPath<'db>,
    ) -> Option<AddressSpaceKind> {
        if let Some((local, prefix)) =
            crate::ir::resolve_local_projection_root(&caller.body.values, arg_value)
        {
            let lookup = prefix.concat(path);
            if let Some(info) =
                crate::ir::lookup_local_pointer_leaf_info(&caller.body.locals, local, &lookup)
            {
                return Some(info.address_space);
            }
        }

        if path.is_empty() {
            return crate::ir::try_value_address_space_in(
                &caller.body.values,
                &caller.body.locals,
                arg_value,
            );
        }

        None
    }

    fn call_param_capability_space_overrides(
        &self,
        caller: &MirFunction<'db>,
        call: &CallOrigin<'db>,
        target: CallTarget<'db>,
        args: &[TyId<'db>],
        receiver_space: Option<AddressSpaceKind>,
    ) -> Vec<Vec<(crate::MirProjectionPath<'db>, AddressSpaceKind)>> {
        let func = match target {
            CallTarget::Template(func) | CallTarget::Decl(func) => func,
            CallTarget::Synthetic(origin) => {
                return self.call_synthetic_param_capability_space_overrides(
                    caller,
                    call,
                    origin,
                    receiver_space,
                );
            }
        };

        let param_count = func.params(self.db).count();
        let mut overrides: Vec<Vec<(crate::MirProjectionPath<'db>, AddressSpaceKind)>> =
            vec![Vec::new(); param_count];
        let core = CoreLib::new(self.db, func.scope());
        let caller_core = self.core_for_origin(caller.origin);

        for (param_idx, param) in func.params(self.db).enumerate() {
            let Some(&arg_value) = call.args.get(param_idx) else {
                continue;
            };

            let mut folder = ParamSubstFolder { args };
            let param_ty = param.ty(self.db).fold_with(self.db, &mut folder);
            let preserve_root = self.should_preserve_root_param_override(
                caller,
                &caller_core,
                arg_value,
                &core,
                param_ty,
            );
            let mut paths = self.param_capability_space_paths_for_ty(&core, param_ty);
            if preserve_root && !paths.iter().any(crate::MirProjectionPath::is_empty) {
                paths.insert(0, crate::MirProjectionPath::new());
            }
            for path in paths {
                if let Some(space) = self.arg_capability_space_at_path(caller, arg_value, &path)
                    && (!matches!(space, AddressSpaceKind::Memory)
                        || (path.is_empty() && preserve_root))
                {
                    overrides[param_idx].push((path, space));
                }
            }
        }

        self.normalize_param_capability_space_overrides(func, &overrides)
    }

    fn call_effect_param_space_overrides(
        &self,
        caller: &MirFunction<'db>,
        call: &CallOrigin<'db>,
        target: CallTarget<'db>,
        args: &[TyId<'db>],
        receiver_space: Option<AddressSpaceKind>,
    ) -> Vec<Option<AddressSpaceKind>> {
        let func = match target {
            CallTarget::Template(func) | CallTarget::Decl(func) => func,
            CallTarget::Synthetic(origin) => {
                return self.call_synthetic_effect_param_space_overrides(
                    caller,
                    call,
                    origin,
                    receiver_space,
                );
            }
        };

        if call.effect_args.is_empty() {
            return Vec::new();
        }

        let default_spaces =
            self.default_effect_param_spaces_for_call(target, func, args, receiver_space);
        let effect_count = default_spaces.len();
        let mut overrides = vec![None; effect_count];

        for (param_ord, override_space) in overrides.iter_mut().enumerate() {
            let Some(&effect_arg_value) = call.effect_args.get(param_ord) else {
                continue;
            };
            let Some(space) = crate::ir::try_value_address_space_in(
                &caller.body.values,
                &caller.body.locals,
                effect_arg_value,
            ) else {
                continue;
            };

            if default_spaces.get(param_ord).copied() != Some(space) {
                *override_space = Some(space);
            }
        }

        overrides
    }

    fn default_effect_param_spaces_for_call(
        &self,
        target: CallTarget<'db>,
        func: Func<'db>,
        args: &[TyId<'db>],
        receiver_space: Option<AddressSpaceKind>,
    ) -> Vec<AddressSpaceKind> {
        match target {
            CallTarget::Template(_) | CallTarget::Decl(_) => {
                self.default_effect_param_spaces_for_hir_func(func, args)
            }
            CallTarget::Synthetic(origin) => self
                .func_index
                .get(&TemplateKey {
                    origin,
                    receiver_space,
                })
                .and_then(|idx| self.templates.get(*idx))
                .map(|template| {
                    template
                        .body
                        .effect_param_locals
                        .iter()
                        .map(|local| template.body.local(*local).address_space)
                        .collect()
                })
                .unwrap_or_default(),
        }
    }

    fn default_effect_param_spaces_for_hir_func(
        &self,
        func: Func<'db>,
        args: &[TyId<'db>],
    ) -> Vec<AddressSpaceKind> {
        let core = CoreLib::new(self.db, func.scope());
        let assumptions = PredicateListId::empty_list(self.db);
        let provider_arg_idx_by_effect =
            hir::analysis::ty::effects::place_effect_provider_param_index_map(self.db, func);
        let mut spaces = vec![AddressSpaceKind::Storage; func.effect_params(self.db).count()];

        for effect in func.effect_params(self.db) {
            let idx = effect.index();
            let provider_ty = provider_arg_idx_by_effect
                .get(idx)
                .copied()
                .flatten()
                .and_then(|provider_idx| {
                    args.get(provider_idx).copied().or_else(|| {
                        CallableDef::Func(func)
                            .params(self.db)
                            .get(provider_idx)
                            .copied()
                    })
                });
            if let Some(provider_ty) = provider_ty
                && let Some(space) =
                    self.default_effect_space_for_provider_ty(func, &core, provider_ty, None)
            {
                spaces[idx] = space;
                continue;
            }

            if let Some(key_path) = effect.key_path(self.db)
                && let Some(provider_ty) =
                    hir::analysis::ty::effects::resolve_normalized_type_effect_key(
                        self.db,
                        key_path,
                        func.scope(),
                        assumptions,
                    )
                && let Some(space) =
                    self.default_effect_space_for_provider_ty(func, &core, provider_ty, None)
            {
                spaces[idx] = space;
                continue;
            }

            if let Some(provider_ty) = effect.key_path(self.db).and_then(|key_path| {
                match hir::analysis::name_resolution::path_resolver::resolve_path(
                    self.db,
                    key_path,
                    func.scope(),
                    assumptions,
                    false,
                )
                .ok()?
                {
                    hir::analysis::name_resolution::path_resolver::PathRes::Ty(ty)
                    | hir::analysis::name_resolution::path_resolver::PathRes::TyAlias(_, ty) => {
                        Some(ty)
                    }
                    _ => None,
                }
            }) && let Some(space) =
                self.default_effect_space_for_provider_ty(func, &core, provider_ty, None)
            {
                spaces[idx] = space;
                continue;
            }

            if let Some(provider_ty) = self.contract_field_provider_ty_for_effect(func, idx)
                && let Some(space) = self.default_effect_space_for_provider_ty(
                    func,
                    &core,
                    provider_ty,
                    Some(AddressSpaceKind::Storage),
                )
            {
                spaces[idx] = space;
            }
        }

        spaces
    }

    fn default_effect_space_for_provider_ty(
        &self,
        func: Func<'db>,
        core: &CoreLib<'db>,
        provider_ty: TyId<'db>,
        by_ref_space: Option<AddressSpaceKind>,
    ) -> Option<AddressSpaceKind> {
        if let Some(space) = raw_effect_space_for_provider_ty(self.db, func.scope(), provider_ty) {
            return Some(space);
        }
        if let Some(space) = crate::repr::effect_provider_space_for_ty(self.db, core, provider_ty) {
            return Some(space);
        }
        matches!(
            crate::repr::repr_kind_for_ty(self.db, core, provider_ty),
            crate::repr::ReprKind::Ref
        )
        .then_some(by_ref_space.unwrap_or(AddressSpaceKind::Memory))
    }

    fn contract_field_provider_ty_for_effect(
        &self,
        func: Func<'db>,
        idx: usize,
    ) -> Option<TyId<'db>> {
        let ItemKind::Contract(contract) = func.scope().parent_item(self.db)? else {
            return None;
        };
        let key_path = func
            .effect_params(self.db)
            .nth(idx)
            .and_then(|effect| effect.key_path(self.db))?;
        if key_path.len(self.db) != 1 {
            return None;
        }
        let field_name = key_path.ident(self.db).to_opt()?;
        let field = contract.fields(self.db).get(&field_name)?;
        field.is_provider.then_some(field.declared_ty)
    }

    fn call_synthetic_param_capability_space_overrides(
        &self,
        caller: &MirFunction<'db>,
        call: &CallOrigin<'db>,
        origin: crate::ir::MirFunctionOrigin<'db>,
        receiver_space: Option<AddressSpaceKind>,
    ) -> Vec<Vec<(crate::MirProjectionPath<'db>, AddressSpaceKind)>> {
        let Some(template_idx) = self.func_index.get(&TemplateKey {
            origin,
            receiver_space,
        }) else {
            return Vec::new();
        };
        let Some(template) = self.templates.get(*template_idx) else {
            return Vec::new();
        };

        let param_count = template.body.param_locals.len();
        let mut overrides: Vec<Vec<(crate::MirProjectionPath<'db>, AddressSpaceKind)>> =
            vec![Vec::new(); param_count];
        let core = self.core_for_origin(template.origin);
        let caller_core = self.core_for_origin(caller.origin);

        for (param_idx, param_local) in template.body.param_locals.iter().enumerate() {
            let Some(&arg_value) = call.args.get(param_idx) else {
                continue;
            };
            let param_ty = template.body.local(*param_local).ty;
            let preserve_root = self.should_preserve_root_param_override(
                caller,
                &caller_core,
                arg_value,
                &core,
                param_ty,
            );
            let mut paths = self.param_capability_space_paths_for_ty(&core, param_ty);
            if preserve_root && !paths.iter().any(crate::MirProjectionPath::is_empty) {
                paths.insert(0, crate::MirProjectionPath::new());
            }
            for path in paths {
                if let Some(space) = self.arg_capability_space_at_path(caller, arg_value, &path)
                    && (!matches!(space, AddressSpaceKind::Memory)
                        || (path.is_empty() && preserve_root))
                {
                    overrides[param_idx].push((path, space));
                }
            }
        }

        self.normalize_param_capability_space_overrides_for_len(param_count, &overrides)
    }

    fn call_synthetic_effect_param_space_overrides(
        &self,
        caller: &MirFunction<'db>,
        call: &CallOrigin<'db>,
        origin: crate::ir::MirFunctionOrigin<'db>,
        receiver_space: Option<AddressSpaceKind>,
    ) -> Vec<Option<AddressSpaceKind>> {
        let Some(template_idx) = self.func_index.get(&TemplateKey {
            origin,
            receiver_space,
        }) else {
            return Vec::new();
        };
        let Some(template) = self.templates.get(*template_idx) else {
            return Vec::new();
        };

        let effect_count = template.body.effect_param_locals.len();
        if effect_count == 0 || call.effect_args.is_empty() {
            return Vec::new();
        }

        let mut overrides = vec![None; effect_count];
        let template_effect_spaces: Vec<_> = template
            .body
            .effect_param_locals
            .iter()
            .map(|local| template.body.local(*local).address_space)
            .collect();
        for (idx, &effect_arg_value) in call.effect_args.iter().take(effect_count).enumerate() {
            let Some(space) = crate::ir::try_value_address_space_in(
                &caller.body.values,
                &caller.body.locals,
                effect_arg_value,
            ) else {
                continue;
            };
            if template_effect_spaces
                .get(idx)
                .copied()
                .unwrap_or(AddressSpaceKind::Memory)
                != space
            {
                overrides[idx] = Some(space);
            }
        }

        self.normalize_effect_param_space_overrides_for_len(effect_count, &overrides)
    }

    fn normalize_effect_param_space_overrides(
        &self,
        func: Func<'db>,
        overrides: &[Option<AddressSpaceKind>],
    ) -> Vec<Option<AddressSpaceKind>> {
        self.normalize_effect_param_space_overrides_for_len(
            func.effect_params(self.db).count(),
            overrides,
        )
    }

    fn normalize_effect_param_space_overrides_for_len(
        &self,
        effect_count: usize,
        overrides: &[Option<AddressSpaceKind>],
    ) -> Vec<Option<AddressSpaceKind>> {
        if overrides.is_empty() || overrides.iter().all(Option::is_none) {
            return Vec::new();
        }

        let mut normalized = vec![None; effect_count];
        let len = std::cmp::min(effect_count, overrides.len());
        normalized[..len].copy_from_slice(&overrides[..len]);
        normalized
    }

    fn normalize_param_capability_space_overrides(
        &self,
        func: Func<'db>,
        overrides: &[Vec<(crate::MirProjectionPath<'db>, AddressSpaceKind)>],
    ) -> Vec<Vec<(crate::MirProjectionPath<'db>, AddressSpaceKind)>> {
        self.normalize_param_capability_space_overrides_for_len(
            func.params(self.db).count(),
            overrides,
        )
    }

    fn normalize_param_capability_space_overrides_for_len(
        &self,
        param_count: usize,
        overrides: &[Vec<(crate::MirProjectionPath<'db>, AddressSpaceKind)>],
    ) -> Vec<Vec<(crate::MirProjectionPath<'db>, AddressSpaceKind)>> {
        if overrides.is_empty() || overrides.iter().all(|entries| entries.is_empty()) {
            return Vec::new();
        }

        let mut normalized: Vec<Vec<(crate::MirProjectionPath<'db>, AddressSpaceKind)>> =
            vec![Vec::new(); param_count];
        let len = std::cmp::min(param_count, overrides.len());
        for (idx, entries) in overrides.iter().take(len).enumerate() {
            let mut merged: FxHashMap<crate::MirProjectionPath<'db>, AddressSpaceKind> =
                FxHashMap::default();
            let mut conflict = None;
            for (path, space) in entries.iter().filter_map(|(path, space)| {
                (!matches!(space, AddressSpaceKind::Memory) || path.is_empty())
                    .then_some((path.clone(), *space))
            }) {
                let Some(existing) = merged.get(&path).copied() else {
                    merged.insert(path, space);
                    continue;
                };
                let Some(merged_space) = (match (existing, space) {
                    (lhs, rhs) if lhs == rhs => Some(lhs),
                    (AddressSpaceKind::Memory, rhs) => Some(rhs),
                    (lhs, AddressSpaceKind::Memory) => Some(lhs),
                    _ => None,
                }) else {
                    conflict = Some((path, existing, space));
                    break;
                };
                merged.insert(path, merged_space);
            }
            normalized[idx] = if let Some((path, existing, incoming)) = conflict {
                let current = self
                    .current_symbol
                    .as_deref()
                    .unwrap_or("<unknown function>");
                self.defer_error(MirLowerError::Unsupported {
                    func_name: current.to_owned(),
                    message: format!(
                        "conflicting capability-space override for param {idx} path `{:?}`: `{:?}` vs `{:?}`",
                        path, existing, incoming
                    ),
                });
                Vec::new()
            } else {
                let mut entries: Vec<_> = merged.into_iter().collect();
                entries.sort_by_cached_key(|(path, _)| format!("{path:?}"));
                entries
            };
        }

        if normalized.iter().all(|entries| entries.is_empty()) {
            Vec::new()
        } else {
            normalized
        }
    }

    fn should_preserve_root_param_override(
        &self,
        caller: &MirFunction<'db>,
        caller_core: &CoreLib<'db>,
        arg_value: ValueId,
        callee_core: &CoreLib<'db>,
        param_ty: TyId<'db>,
    ) -> bool {
        let arg_shape = crate::repr::runtime_shape_for_value(
            self.db,
            caller_core,
            &caller.body.values,
            &caller.body.locals,
            arg_value,
        );
        if matches!(arg_shape, Some(RuntimeShape::MemoryPtr { .. })) {
            return true;
        }

        let Some(arg_space) = crate::ir::try_value_address_space_in(
            &caller.body.values,
            &caller.body.locals,
            arg_value,
        ) else {
            return false;
        };
        if matches!(arg_space, AddressSpaceKind::Memory) {
            return false;
        }

        let default_shape = crate::repr::runtime_shape_for_ty(
            self.db,
            callee_core,
            param_ty,
            AddressSpaceKind::Memory,
        );
        let specialized_shape =
            crate::repr::runtime_shape_for_ty(self.db, callee_core, param_ty, arg_space);
        !matches!(
            specialized_shape,
            RuntimeShape::Erased | RuntimeShape::Word(_)
        ) && specialized_shape != default_shape
    }

    fn param_capability_space_paths_for_ty(
        &self,
        core: &CoreLib<'db>,
        param_ty: TyId<'db>,
    ) -> Vec<crate::MirProjectionPath<'db>> {
        pointer_leaf_paths_for_ty(self.db, core, param_ty)
    }
}

/// Simple folder that replaces `TyParam` occurrences with the concrete args for
/// the instance under construction.
struct ParamSubstFolder<'db, 'a> {
    args: &'a [TyId<'db>],
}

impl<'db> TyFolder<'db> for ParamSubstFolder<'db, '_> {
    fn fold_ty(&mut self, db: &'db dyn HirAnalysisDb, ty: TyId<'db>) -> TyId<'db> {
        match ty.data(db) {
            TyData::TyParam(param) => self.args.get(param.idx).copied().unwrap_or(ty),
            TyData::ConstTy(const_ty) => {
                if let ConstTyData::TyParam(param, _) = const_ty.data(db) {
                    return self.args.get(param.idx).copied().unwrap_or(ty);
                }
                ty.super_fold_with(db, self)
            }
            _ => ty.super_fold_with(db, self),
        }
    }
}

struct NormalizeFolder<'db> {
    scope: ScopeId<'db>,
    assumptions: PredicateListId<'db>,
}

impl<'db> TyFolder<'db> for NormalizeFolder<'db> {
    fn fold_ty(&mut self, db: &'db dyn HirAnalysisDb, ty: TyId<'db>) -> TyId<'db> {
        let normalized = normalize_ty(db, ty, self.scope, self.assumptions);
        if !ty.has_invalid(db) && normalized.has_invalid(db) {
            ty
        } else {
            normalized
        }
    }
}

/// Replace any non-alphanumeric characters with `_` so the mangled symbol is a
/// valid Yul identifier while remaining somewhat recognizable.
fn sanitize_symbol_component(component: &str) -> String {
    component
        .chars()
        .map(|ch| if ch.is_ascii_alphanumeric() { ch } else { '_' })
        .collect()
}

fn stable_hash<T: Hash + ?Sized>(value: &T) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    value.hash(&mut hasher);
    hasher.finish()
}

fn canonicalize_receiver_space(
    receiver_space: Option<AddressSpaceKind>,
) -> Option<AddressSpaceKind> {
    match receiver_space {
        None | Some(AddressSpaceKind::Memory) => None,
        Some(space) => Some(space),
    }
}

fn raw_effect_space_for_provider_ty(
    db: &dyn HirAnalysisDb,
    scope: ScopeId<'_>,
    provider_ty: TyId<'_>,
) -> Option<AddressSpaceKind> {
    let raw_mem = resolve_lib_type_path(db, scope, "std::evm::RawMem")?;
    if provider_ty == raw_mem {
        return Some(AddressSpaceKind::Memory);
    }

    let raw_storage = resolve_lib_type_path(db, scope, "std::evm::RawStorage")?;
    (provider_ty == raw_storage).then_some(AddressSpaceKind::Storage)
}

fn effect_param_space_suffix(spaces: &[Option<AddressSpaceKind>]) -> String {
    spaces
        .iter()
        .enumerate()
        .filter_map(|(idx, space)| space.map(|space| (idx, space)))
        .map(|(idx, space)| {
            let suffix = match space {
                AddressSpaceKind::Memory => "mem",
                AddressSpaceKind::Calldata => "calldata",
                AddressSpaceKind::Storage => "stor",
                AddressSpaceKind::TransientStorage => "tstor",
                AddressSpaceKind::Code => "code",
            };
            format!("eff{idx}_{suffix}")
        })
        .collect::<Vec<_>>()
        .join("_")
}

fn projection_path_suffix(path: &crate::MirProjectionPath<'_>) -> String {
    if path.is_empty() {
        return "root".to_string();
    }
    path.iter()
        .map(|proj| match proj {
            hir::projection::Projection::Field(idx) => format!("f{idx}"),
            hir::projection::Projection::VariantField { field_idx, .. } => format!("vf{field_idx}"),
            hir::projection::Projection::Index(hir::projection::IndexSource::Constant(idx)) => {
                format!("i{idx}")
            }
            hir::projection::Projection::Index(hir::projection::IndexSource::Dynamic(_)) => {
                "idyn".to_string()
            }
            hir::projection::Projection::Discriminant => "discr".to_string(),
            hir::projection::Projection::Deref => "deref".to_string(),
        })
        .collect::<Vec<_>>()
        .join("_")
}

fn param_capability_space_suffix(
    overrides: &[Vec<(crate::MirProjectionPath<'_>, AddressSpaceKind)>],
) -> String {
    overrides
        .iter()
        .enumerate()
        .flat_map(|(param_idx, entries)| {
            entries.iter().map(move |(path, space)| {
                let space_suffix = match space {
                    AddressSpaceKind::Memory => "mem",
                    AddressSpaceKind::Calldata => "calldata",
                    AddressSpaceKind::Storage => "stor",
                    AddressSpaceKind::TransientStorage => "tstor",
                    AddressSpaceKind::Code => "code",
                };
                let path_suffix = projection_path_suffix(path);
                format!("arg{param_idx}_{path_suffix}_{space_suffix}")
            })
        })
        .collect::<Vec<_>>()
        .join("_")
}

#[cfg(test)]
mod tests {
    use std::{
        fs,
        time::{SystemTime, UNIX_EPOCH},
    };

    use common::InputDb;
    use driver::DriverDataBase;
    use url::Url;

    use super::*;

    #[test]
    fn conflicting_non_memory_param_overrides_report_error() {
        let db = DriverDataBase::default();
        let mut monomorphizer = Monomorphizer::new(&db, Vec::new());
        monomorphizer.current_symbol = Some("test_symbol".to_owned());

        let root = crate::MirProjectionPath::new();
        let overrides = vec![vec![
            (root.clone(), AddressSpaceKind::Storage),
            (root, AddressSpaceKind::Calldata),
        ]];

        let _ = monomorphizer.normalize_param_capability_space_overrides_for_len(1, &overrides);
        let err = monomorphizer
            .take_deferred_error()
            .expect("conflict should defer an error");
        let MirLowerError::Unsupported { func_name, message } = err else {
            panic!("unexpected error kind");
        };
        assert_eq!(func_name, "test_symbol");
        assert!(message.contains("conflicting capability-space override"));
    }

    #[test]
    fn root_memory_param_overrides_are_preserved() {
        let db = DriverDataBase::default();
        let monomorphizer = Monomorphizer::new(&db, Vec::new());

        let root = crate::MirProjectionPath::new();
        let overrides = vec![vec![(root.clone(), AddressSpaceKind::Memory)]];

        assert_eq!(
            monomorphizer.normalize_param_capability_space_overrides_for_len(1, &overrides),
            overrides
        );
    }

    #[test]
    fn deferred_error_is_reported_with_empty_worklist() {
        let db = DriverDataBase::default();
        let mut monomorphizer = Monomorphizer::new(&db, Vec::new());
        monomorphizer.defer_error(MirLowerError::Unsupported {
            func_name: "seed_roots".to_owned(),
            message: "boom".to_owned(),
        });
        assert!(
            monomorphizer.worklist.is_empty(),
            "test assumes no worklist entries"
        );

        let err = monomorphizer
            .process_worklist()
            .expect_err("deferred errors should be reported even with empty worklists");
        let MirLowerError::Unsupported { func_name, message } = err else {
            panic!("expected Unsupported, got {err:?}");
        };
        assert_eq!(func_name, "seed_roots");
        assert_eq!(message, "boom");
    }

    #[test]
    fn synthetic_param_space_overrides_refresh_declared_place_root_layout() {
        let mut db = DriverDataBase::default();
        let url = Url::parse("file:///synthetic_param_place_root_layout.fe").unwrap();
        let src = r#"
struct Pair {
    left: u256,
    right: u256,
}

fn takes_pair(pair: own Pair) -> u256 {
    pair.left
}

pub contract C {
    init() {}
}
"#;

        let file = db.workspace().touch(&mut db, url, Some(src.to_owned()));
        let top_mod = db.top_mod(file);
        let module = crate::lower::lower_module(&db, top_mod).expect("lowered MIR");
        let pair_func = module
            .functions
            .iter()
            .find(|func| {
                matches!(func.origin, crate::ir::MirFunctionOrigin::Hir(_))
                    && func.symbol_name.starts_with("takes_pair")
            })
            .expect("takes_pair function");
        let pair_ty = pair_func.body.locals[pair_func.body.param_locals[0].index()].ty;
        let synthetic_origin = module
            .functions
            .iter()
            .find_map(|func| match func.origin {
                crate::ir::MirFunctionOrigin::Synthetic(id) => {
                    Some(crate::ir::MirFunctionOrigin::Synthetic(id))
                }
                crate::ir::MirFunctionOrigin::Hir(_) => None,
            })
            .expect("synthetic function origin");

        let monomorphizer = Monomorphizer::new(&db, Vec::new());
        let core = monomorphizer.core_for_origin(synthetic_origin);
        let mut body = crate::ir::MirBody::new();
        let local = body.alloc_local(crate::ir::LocalData {
            name: "pair".to_owned(),
            ty: pair_ty,
            is_mut: false,
            source: crate::ir::SourceInfoId::SYNTHETIC,
            address_space: AddressSpaceKind::Memory,
            pointer_leaf_infos: pointer_leaf_infos_for_ty_with_default(
                &db,
                &core,
                pair_ty,
                AddressSpaceKind::Memory,
            ),
            place_root_layout: crate::repr::declared_local_place_root_layout(
                &db,
                &core,
                pair_ty,
                AddressSpaceKind::Memory,
            ),
            const_backing: crate::ir::LocalConstBacking::Unknown,
            runtime_shape: crate::ir::RuntimeShape::Unresolved,
        });
        assert!(matches!(
            body.locals[local.index()].place_root_layout,
            crate::ir::LocalPlaceRootLayout::ObjectRootValue { .. }
        ));
        body.param_locals.push(local);
        let mut instance = MirFunction {
            origin: synthetic_origin,
            body,
            typed_body: None,
            generic_args: Vec::new(),
            ret_ty: pair_ty,
            returns_value: true,
            runtime_abi: crate::ir::RuntimeAbi {
                value_params: vec![true],
                effect_params: Vec::new(),
                effect_param_provider_tys: Vec::new(),
            },
            runtime_return_shape: crate::ir::RuntimeShape::Unresolved,
            runtime_return_pointer_leaf_infos: Vec::new(),
            contract_function: None,
            inline_hint: None,
            symbol_name: "synthetic_override_test".to_owned(),
            symbol_source: crate::ir::SymbolSource::Internal,
            receiver_space: None,
            defer_root: false,
        };

        monomorphizer.apply_synthetic_param_capability_space_overrides(
            &mut instance,
            &[vec![(
                crate::MirProjectionPath::new(),
                AddressSpaceKind::Storage,
            )]],
        );

        let local = &instance.body.locals[local.index()];
        assert_eq!(local.address_space, AddressSpaceKind::Storage);
        assert_eq!(
            local.place_root_layout,
            crate::ir::LocalPlaceRootLayout::Direct
        );
    }

    #[test]
    fn synthetic_memory_root_overrides_preserve_aggregate_memory_ptr_runtime_shape() {
        let mut db = DriverDataBase::default();
        let url = Url::parse("file:///synthetic_memory_root_override.fe").unwrap();
        let src = r#"
struct Pair {
    left: u256,
    right: u256,
}

fn takes_pair(pair: own Pair) -> u256 {
    pair.left
}

pub contract C {
    init() {}
}
"#;

        let file = db.workspace().touch(&mut db, url, Some(src.to_owned()));
        let top_mod = db.top_mod(file);
        let module = crate::lower::lower_module(&db, top_mod).expect("lowered MIR");
        let pair_func = module
            .functions
            .iter()
            .find(|func| {
                matches!(func.origin, crate::ir::MirFunctionOrigin::Hir(_))
                    && func.symbol_name.starts_with("takes_pair")
            })
            .expect("takes_pair function");
        let pair_ty = pair_func.body.locals[pair_func.body.param_locals[0].index()].ty;
        let synthetic_origin = module
            .functions
            .iter()
            .find_map(|func| match func.origin {
                crate::ir::MirFunctionOrigin::Synthetic(id) => {
                    Some(crate::ir::MirFunctionOrigin::Synthetic(id))
                }
                crate::ir::MirFunctionOrigin::Hir(_) => None,
            })
            .expect("synthetic function origin");

        let monomorphizer = Monomorphizer::new(&db, Vec::new());
        let core = monomorphizer.core_for_origin(synthetic_origin);
        let mut body = crate::ir::MirBody::new();
        let local = body.alloc_local(crate::ir::LocalData {
            name: "pair".to_owned(),
            ty: pair_ty,
            is_mut: false,
            source: crate::ir::SourceInfoId::SYNTHETIC,
            address_space: AddressSpaceKind::Memory,
            pointer_leaf_infos: pointer_leaf_infos_for_ty_with_default(
                &db,
                &core,
                pair_ty,
                AddressSpaceKind::Memory,
            ),
            place_root_layout: crate::repr::declared_local_place_root_layout(
                &db,
                &core,
                pair_ty,
                AddressSpaceKind::Memory,
            ),
            const_backing: crate::ir::LocalConstBacking::Unknown,
            runtime_shape: crate::ir::RuntimeShape::Unresolved,
        });
        body.param_locals.push(local);
        let mut instance = MirFunction {
            origin: synthetic_origin,
            body,
            typed_body: None,
            generic_args: Vec::new(),
            ret_ty: pair_ty,
            returns_value: true,
            runtime_abi: crate::ir::RuntimeAbi {
                value_params: vec![true],
                effect_params: Vec::new(),
                effect_param_provider_tys: Vec::new(),
            },
            runtime_return_shape: crate::ir::RuntimeShape::Unresolved,
            runtime_return_pointer_leaf_infos: Vec::new(),
            contract_function: None,
            inline_hint: None,
            symbol_name: "synthetic_memory_root_override_test".to_owned(),
            symbol_source: crate::ir::SymbolSource::Internal,
            receiver_space: None,
            defer_root: false,
        };

        monomorphizer.apply_synthetic_param_capability_space_overrides(
            &mut instance,
            &[vec![(
                crate::MirProjectionPath::new(),
                AddressSpaceKind::Memory,
            )]],
        );

        let local = &instance.body.locals[local.index()];
        assert_eq!(
            local.place_root_layout,
            crate::ir::LocalPlaceRootLayout::MemorySlot
        );
        assert!(
            local
                .pointer_leaf_infos
                .iter()
                .any(|(path, info)| path.is_empty()
                    && info.address_space == AddressSpaceKind::Memory
                    && info.target_ty == Some(pair_ty))
        );
        assert_eq!(
            crate::repr::runtime_shape_for_local(&db, &core, local),
            crate::ir::RuntimeShape::MemoryPtr {
                target_ty: Some(pair_ty),
            }
        );
    }

    #[test]
    fn repeated_field_types_collect_all_capability_paths() {
        let mut db = DriverDataBase::default();
        let url = url::Url::parse("file:///repeated_cap_paths.fe").unwrap();
        let src = r#"
msg Msg {
    #[selector = 1]
    Run -> u256
}

struct Wrapper {
    value: mut u256
}

struct Pair {
    left: Wrapper,
    right: Wrapper
}

fn bump(mut pair: own Pair) -> u256 {
    pair.left.value += 1
    pair.right.value += 10
    pair.left.value * 100 + pair.right.value
}

pub contract C {
    mut a: u256,
    mut b: u256,

    init() uses (mut a, mut b) {
        a = 1
        b = 2
    }

    recv Msg {
        Run -> u256 uses (mut a, mut b) {
            let pair = Pair {
                left: Wrapper { value: mut a },
                right: Wrapper { value: mut b }
            }
            bump(pair)
        }
    }
}
"#;

        let file = db.workspace().touch(&mut db, url, Some(src.to_owned()));
        let top_mod = db.top_mod(file);
        let module = crate::lower::lower_module(&db, top_mod).expect("lowered MIR");

        let bump_symbols = module
            .functions
            .iter()
            .map(|func| func.symbol_name.as_str())
            .filter(|name| name.starts_with("bump"))
            .collect::<Vec<_>>();

        assert!(
            bump_symbols
                .iter()
                .any(|name| { name.contains("arg0_f0_stor") && name.contains("arg0_f1_stor") }),
            "expected bump specialization to carry both repeated-field paths, got: {bump_symbols:?}",
        );
    }

    #[test]
    fn root_seeding_uses_canonical_effect_key_instances() {
        let mut db = DriverDataBase::default();
        let src = fs::read_to_string(format!(
            "{}/../codegen/tests/fixtures/newtype_storage_byplace_effect_arg.fe",
            env!("CARGO_MANIFEST_DIR")
        ))
        .expect("read newtype fixture");
        let url = Url::parse("file:///root_seed_canonical_effect_key.fe").unwrap();
        let file = db.workspace().touch(&mut db, url, Some(src));
        let top_mod = db.top_mod(file);
        let module = crate::lower::lower_module(&db, top_mod).expect("lowered MIR");

        let bump_symbols = module
            .functions
            .iter()
            .map(|func| func.symbol_name.as_str())
            .filter(|name| name.starts_with("bump"))
            .collect::<Vec<_>>();

        assert_eq!(
            bump_symbols.len(),
            1,
            "expected one bump instance, got: {bump_symbols:?}"
        );
        assert!(
            bump_symbols[0].contains("bump__Wrap"),
            "expected canonical key-type bump instance, got: {bump_symbols:?}",
        );
        assert!(
            !bump_symbols[0].contains("StorPtr"),
            "legacy provider-wrapper root seed should be gone: {bump_symbols:?}",
        );
    }

    #[test]
    fn root_seeding_uses_storage_overrides_instead_of_provider_wrapper_args() {
        let mut db = DriverDataBase::default();
        let src = fs::read_to_string(format!(
            "{}/../codegen/tests/fixtures/erc20.fe",
            env!("CARGO_MANIFEST_DIR")
        ))
        .expect("read erc20 fixture");
        let url = Url::parse("file:///root_seed_storage_overrides.fe").unwrap();
        let file = db.workspace().touch(&mut db, url, Some(src));
        let top_mod = db.top_mod(file);
        let module = crate::lower::lower_module(&db, top_mod).expect("lowered MIR");

        let helper_symbols = module
            .functions
            .iter()
            .map(|func| func.symbol_name.as_str())
            .filter(|name| {
                ["mint", "transfer", "approve", "spend_allowance", "burn"]
                    .iter()
                    .any(|prefix| name.starts_with(prefix))
            })
            .collect::<Vec<_>>();

        assert!(
            helper_symbols
                .iter()
                .any(|name| name.contains("eff0_stor__0_1_TokenStore_0__1__Evm")),
            "expected storage-specialized helper names, got: {helper_symbols:?}",
        );
        assert!(
            !helper_symbols
                .iter()
                .any(|name| name.contains("StorPtr_TokenStore")),
            "legacy provider-wrapper helper names should be gone: {helper_symbols:?}",
        );
    }

    #[test]
    fn test_entrypoints_with_only_effect_provider_params_are_root_seeded() {
        let mut db = DriverDataBase::default();
        let url = Url::parse("file:///test_root_seed_effect_only.fe").unwrap();
        let src = r#"
use std::evm::{Evm, assert}

fn helper() uses (evm: mut Evm) {
    assert(true)
}

#[test]
fn smoke() uses (evm: mut Evm) {
    assert(true)
}
"#;

        let file = db.workspace().touch(&mut db, url, Some(src.to_owned()));
        let top_mod = db.top_mod(file);
        let module = crate::lower::lower_module(&db, top_mod).expect("lowered MIR");

        assert!(
            module
                .functions
                .iter()
                .any(|func| func.symbol_name.starts_with("smoke__Evm")),
            "expected #[test] entrypoint to be root-seeded: {:?}",
            module
                .functions
                .iter()
                .map(|func| func.symbol_name.as_str())
                .collect::<Vec<_>>(),
        );
        assert!(
            !module
                .functions
                .iter()
                .any(|func| func.symbol_name == "helper"),
            "ordinary effect-only helpers should still instantiate on demand only",
        );
    }

    #[test]
    fn test_entrypoints_with_trait_effects_root_seed_concrete_provider_args() {
        let mut db = DriverDataBase::default();
        let src = fs::read_to_string(format!(
            "{}/../fe/tests/fixtures/fe_test/address_call_method.fe",
            env!("CARGO_MANIFEST_DIR")
        ))
        .expect("read address_call_method fixture");
        let url = Url::parse("file:///test_root_seed_trait_effects.fe").unwrap();
        let file = db.workspace().touch(&mut db, url, Some(src));
        let top_mod = db.top_mod(file);
        let module = crate::lower::lower_module(&db, top_mod).expect("lowered MIR");

        assert!(
            module.functions.iter().any(|func| func
                .symbol_name
                .starts_with("test_address_call_method__Evm")),
            "expected test root to monomorphize concrete Evm providers: {:?}",
            module
                .functions
                .iter()
                .map(|func| func.symbol_name.as_str())
                .collect::<Vec<_>>(),
        );
    }

    #[test]
    fn test_entrypoints_with_type_key_effects_default_to_memory_providers() {
        let mut db = DriverDataBase::default();
        let url = Url::parse("file:///test_root_seed_type_key_effects.fe").unwrap();
        let src = r#"
type Words = [u256; 4]

fn write_effect(idx: usize, value: u256) uses (words: mut Words) {
    words[idx] = value
}

#[test]
fn smoke() uses (words: mut Words) {
    write_effect(0, 7)
}
"#;

        let file = db.workspace().touch(&mut db, url, Some(src.to_owned()));
        let top_mod = db.top_mod(file);
        let module = crate::lower::lower_module(&db, top_mod).expect("lowered MIR");

        let symbols = module
            .functions
            .iter()
            .map(|func| func.symbol_name.as_str())
            .collect::<Vec<_>>();

        assert!(
            symbols
                .iter()
                .any(|name| name.starts_with("smoke__MemPtr__u256__4__")),
            "expected #[test] type-key effect entrypoint to root-seed a memory provider: {symbols:?}",
        );
        assert!(
            symbols
                .iter()
                .any(|name| name.starts_with("write_effect__MemPtr__u256__4__")),
            "expected helper to specialize to a memory provider: {symbols:?}",
        );
        assert!(
            !symbols.iter().any(|name| name.contains("StorPtr__u256__4")),
            "type-key free-function effects must not default to storage providers: {symbols:?}",
        );
    }

    #[test]
    fn explicit_mem_ptr_effect_args_keep_memory_specializations() {
        let mut db = DriverDataBase::default();
        let url = Url::parse("file:///explicit_raw_boundaries_effect_override.fe").unwrap();
        let file = db.workspace().touch(
            &mut db,
            url,
            Some(include_str!("../../codegen/tests/fixtures/explicit_raw_boundaries.fe").into()),
        );
        let top_mod = db.top_mod(file);
        let module = crate::lower::lower_module(&db, top_mod).expect("lowered MIR");

        let raw_store_symbols = module
            .functions
            .iter()
            .map(|func| func.symbol_name.as_str())
            .filter(|name| name.starts_with("raw_store"))
            .collect::<Vec<_>>();

        assert!(
            raw_store_symbols
                .iter()
                .any(|name| name.contains("MemPtr_Data__Evm")),
            "expected a MemPtr-specialized raw_store instance, got {raw_store_symbols:?}",
        );
        assert!(
            !raw_store_symbols
                .iter()
                .any(|name| name.starts_with("raw_store__StorPtr_Data__Evm")),
            "legacy storage-specialized raw_store instance should be gone: {raw_store_symbols:?}",
        );
    }

    #[test]
    fn by_ref_storage_effect_providers_keep_storage_specializations() {
        let mut db = DriverDataBase::default();
        let url = Url::parse("file:///by_ref_trait_provider_storage_bug.fe").unwrap();
        let file = db.workspace().touch(
            &mut db,
            url,
            Some(
                include_str!("../../codegen/tests/fixtures/by_ref_trait_provider_storage_bug.fe")
                    .into(),
            ),
        );
        let top_mod = db.top_mod(file);
        let module = crate::lower::lower_module(&db, top_mod).expect("lowered MIR");

        let symbols = module
            .functions
            .iter()
            .map(|func| func.symbol_name.as_str())
            .collect::<Vec<_>>();

        let recv = module
            .functions
            .iter()
            .find(|func| func.symbol_name == "__ByRefTraitProviderStorageBug_recv_0_0")
            .expect("expected monomorphized recv instance");
        let use_ctx_call = recv
            .body
            .blocks
            .iter()
            .flat_map(|block| block.insts.iter())
            .find_map(|inst| match inst {
                crate::MirInst::Assign {
                    rvalue: crate::Rvalue::Call(call),
                    ..
                } if call
                    .resolved_name
                    .as_deref()
                    .is_some_and(|name| name.starts_with("use_ctx")) =>
                {
                    Some(call)
                }
                _ => None,
            })
            .expect("expected recv to call use_ctx");
        let effect_arg_spaces = use_ctx_call
            .effect_args
            .iter()
            .map(|value| {
                crate::ir::try_value_address_space_in(&recv.body.values, &recv.body.locals, *value)
            })
            .collect::<Vec<_>>();

        assert_eq!(
            effect_arg_spaces,
            vec![Some(AddressSpaceKind::Storage)],
            "expected recv to forward ctx as a storage effect arg, got {effect_arg_spaces:?}",
        );

        let use_ctx = module
            .functions
            .iter()
            .find(|func| {
                func.symbol_name.starts_with("use_ctx")
                    && func.symbol_name.contains("Pair_h956dff41e88ee341")
            })
            .expect("expected instantiated use_ctx");
        let effect_local = use_ctx.body.effect_param_locals[0];
        assert_eq!(
            use_ctx.body.local(effect_local).address_space,
            AddressSpaceKind::Storage,
            "expected use_ctx effect local to stay storage-backed",
        );
        let pair_sum_call = use_ctx
            .body
            .blocks
            .iter()
            .flat_map(|block| block.insts.iter())
            .find_map(|inst| match inst {
                crate::MirInst::Assign {
                    rvalue: crate::Rvalue::Call(call),
                    ..
                } if call
                    .resolved_name
                    .as_deref()
                    .is_some_and(|name| name.starts_with("pair_h956dff41e88ee341")) =>
                {
                    Some(call)
                }
                _ => None,
            })
            .expect("expected use_ctx to call pair::sum");
        let arg_spaces = pair_sum_call
            .args
            .iter()
            .map(|value| {
                crate::ir::try_value_address_space_in(
                    &use_ctx.body.values,
                    &use_ctx.body.locals,
                    *value,
                )
            })
            .collect::<Vec<_>>();
        assert_eq!(
            arg_spaces,
            vec![Some(AddressSpaceKind::Storage)],
            "expected use_ctx to pass a storage-backed receiver, got {arg_spaces:?}",
        );
        assert!(
            use_ctx.symbol_name.contains("eff0_stor"),
            "expected storage-specialized use_ctx symbol, got `{}`",
            use_ctx.symbol_name,
        );
        assert!(
            pair_sum_call
                .resolved_name
                .as_deref()
                .is_some_and(|name| name.ends_with("_stor")),
            "expected storage-specialized impl call target, got {symbols:?}",
        );
    }

    #[test]
    fn deferred_dependency_create2_reaches_synthetic_contract_edges() {
        let mut db = DriverDataBase::default();
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time")
            .as_nanos();
        let root = std::env::temp_dir().join(format!("fe_mir_create2_{nonce}"));
        fs::create_dir_all(root.join("ingots/consumer/src")).expect("consumer dir");
        fs::create_dir_all(root.join("ingots/provider/src")).expect("provider dir");
        fs::write(
            root.join("fe.toml"),
            "[workspace]\nname = \"cross_ingot_create2\"\nversion = \"0.1.0\"\nmembers = [\"ingots/*\"]\n",
        )
        .expect("workspace config");
        fs::write(
            root.join("ingots/consumer/fe.toml"),
            "[ingot]\nname = \"consumer\"\nversion = \"0.1.0\"\n\n[dependencies]\nprovider = true\n",
        )
        .expect("consumer config");
        fs::write(
            root.join("ingots/provider/fe.toml"),
            "[ingot]\nname = \"provider\"\nversion = \"0.1.0\"\n",
        )
        .expect("provider config");
        fs::write(
            root.join("ingots/consumer/src/lib.fe"),
            r#"
use std::evm::Evm
use provider::Greeter

fn deploy() uses (evm: mut Evm) {
    let _ = evm.create2<Greeter>(value: 0, args: (42,), salt: 1)
}
"#,
        )
        .expect("consumer source");
        fs::write(
            root.join("ingots/provider/src/lib.fe"),
            r#"
use std::abi::sol

pub msg GreetMsg {
    #[selector = sol("greet()")]
    Greet -> u256,
}

pub contract Greeter {
    mut value: u256,

    init(initial_value: u256) uses (mut value) {
        value = initial_value
    }

    recv GreetMsg {
        Greet -> u256 uses (value) {
            value
        }
    }
}
"#,
        )
        .expect("provider source");

        let root_url = Url::from_directory_path(&root).expect("root url");
        let _ = driver::init_ingot(&mut db, &root_url);
        let consumer_url =
            Url::from_directory_path(root.join("ingots/consumer")).expect("consumer url");
        let consumer_ingot = db
            .workspace()
            .containing_ingot(&db, consumer_url)
            .expect("consumer ingot");

        let module = crate::lower::lower_ingot(&db, consumer_ingot).expect("lowered ingot");
        let symbols = module
            .functions
            .iter()
            .map(|func| func.symbol_name.as_str())
            .filter(|name| name.contains("provider__Greeter"))
            .collect::<Vec<_>>();

        assert!(
            symbols.contains(&"__provider__Greeter_init")
                && symbols.contains(&"__provider__Greeter_runtime")
                && symbols.contains(&"__provider__Greeter_init_code_offset")
                && symbols.contains(&"__provider__Greeter_init_code_len")
                && symbols
                    .iter()
                    .any(|name| name.starts_with("__provider__Greeter_init_contract"))
                && symbols
                    .iter()
                    .any(|name| name.starts_with("__provider__Greeter_recv_0_0")),
            "expected typed synthetic call/code-region edges to instantiate dependency contract templates, got: {symbols:?}",
        );
    }

    #[test]
    fn monomorphized_code_backed_take_wrappers_keep_nested_base_leaf_metadata() {
        let mut db = DriverDataBase::default();
        let url = Url::parse(
            "file:///monomorphized_code_backed_take_wrappers_keep_nested_base_leaf_metadata.fe",
        )
        .unwrap();
        let file = db.workspace().touch(
            &mut db,
            url,
            Some(
                include_str!(
                    "../../fe/tests/fixtures/fe_test/view_param_local_ref_take_reverse.fe"
                )
                .to_owned(),
            ),
        );
        let top_mod = db.top_mod(file);
        let mut templates = Vec::new();
        for func in top_mod.all_funcs(&db).iter().copied() {
            if func.body(&db).is_none() {
                continue;
            }
            let (diags, typed_body) = check_func_body(&db, func);
            assert!(diags.is_empty(), "expected no diagnostics, got {diags:?}");
            templates.push(
                crate::lower::lower_function(
                    &db,
                    func,
                    typed_body.clone(),
                    None,
                    Vec::new(),
                    Vec::new(),
                    Vec::new(),
                )
                .expect("function should lower"),
            );
        }

        let functions =
            monomorphize_functions(&db, templates).expect("functions should monomorphize");
        let sum_first4 = functions
            .iter()
            .find(|func| func.symbol_name == "sum_first4_arg0_root_code")
            .expect("expected code-specialized sum_first4 helper");
        let (head_local, take_call) = sum_first4
            .body
            .blocks
            .iter()
            .flat_map(|block| block.insts.iter())
            .find_map(|inst| match inst {
                crate::MirInst::Assign {
                    dest: Some(local),
                    rvalue: crate::Rvalue::Call(call),
                    ..
                } if call
                    .resolved_name
                    .as_deref()
                    .is_some_and(|name| name.contains("take_u256_arg1_root_code___u256__8")) =>
                {
                    Some((*local, call))
                }
                _ => None,
            })
            .expect("expected take_u256 call result local");
        let core = CoreLib::new(&db, top_mod.scope());
        let source_infos = crate::repr::pointer_leaf_infos_for_value(
            &db,
            &core,
            &sum_first4.body.values,
            &sum_first4.body.locals,
            take_call.args[1],
        );
        let forwarded_infos = crate::lower::call_return_pointer_leaf_infos(
            &db,
            &core,
            &sum_first4.body.values,
            &sum_first4.body.locals,
            take_call,
            sum_first4.body.local(head_local).ty,
        );

        assert_eq!(
            sum_first4
                .body
                .local(head_local)
                .pointer_leaf_infos
                .iter()
                .find_map(|(path, info)| {
                    (path
                        == &crate::MirProjectionPath::from_projection(crate::MirProjection::Field(
                            1,
                        )))
                        .then_some(info.address_space)
                }),
            Some(AddressSpaceKind::Code),
            "monomorphized take wrapper results should already preserve the code-backed base field leaf; head_local={:?}; arg_value={:?}; arg_origin={:?}; arg_ptr={:?}; param_local={:?}; source_infos={source_infos:?}; forwarded_infos={forwarded_infos:?}; take_call={take_call:?}",
            sum_first4.body.local(head_local),
            take_call.args[1],
            sum_first4.body.value(take_call.args[1]).origin,
            sum_first4.body.value_pointer_info(take_call.args[1]),
            sum_first4.body.local(sum_first4.body.param_locals[0]),
        );
    }

    #[test]
    fn code_backed_take_array_calls_compute_nested_base_param_overrides() {
        let mut db = DriverDataBase::default();
        let url = Url::parse(
            "file:///code_backed_take_array_calls_compute_nested_base_param_overrides.fe",
        )
        .unwrap();
        let file = db.workspace().touch(
            &mut db,
            url,
            Some(
                include_str!(
                    "../../fe/tests/fixtures/fe_test/view_param_local_ref_take_reverse.fe"
                )
                .to_owned(),
            ),
        );
        let top_mod = db.top_mod(file);
        let mut templates = Vec::new();
        for func in top_mod.all_funcs(&db).iter().copied() {
            if func.body(&db).is_none() {
                continue;
            }
            let (diags, typed_body) = check_func_body(&db, func);
            assert!(diags.is_empty(), "expected no diagnostics, got {diags:?}");
            templates.push(
                crate::lower::lower_function(
                    &db,
                    func,
                    typed_body.clone(),
                    None,
                    Vec::new(),
                    Vec::new(),
                    Vec::new(),
                )
                .expect("function should lower"),
            );
        }

        let mut monomorphizer = Monomorphizer::new(&db, templates);
        monomorphizer.seed_roots();
        monomorphizer
            .process_worklist()
            .expect("monomorphization should succeed");
        let sum_first4 = monomorphizer
            .instances
            .iter()
            .find(|func| func.symbol_name == "sum_first4_arg0_root_code")
            .expect("expected code-specialized sum_first4 helper");
        let take_get_call = sum_first4
            .body
            .blocks
            .iter()
            .flat_map(|block| block.insts.iter())
            .find_map(|inst| match inst {
                crate::MirInst::Assign {
                    rvalue: crate::Rvalue::Call(call),
                    ..
                } if call
                    .resolved_name
                    .as_deref()
                    .is_some_and(|name| name.contains("take_i__t__") && name.contains("get")) =>
                {
                    Some(call)
                }
                _ => None,
            })
            .expect("expected Take::get call in sum_first4");

        let solve_cx = TraitSolveCx::new(&db, top_mod.scope());
        let (target, args) = monomorphizer
            .resolve_call_target(solve_cx, take_get_call)
            .expect("take get call should resolve");
        let overrides = monomorphizer.call_param_capability_space_overrides(
            sum_first4,
            take_get_call,
            target,
            &args,
            canonicalize_receiver_space(take_get_call.receiver_space),
        );

        assert_eq!(
            overrides.first(),
            Some(&vec![(
                crate::MirProjectionPath::from_projection(crate::MirProjection::Field(1)),
                AddressSpaceKind::Code,
            )]),
            "code-backed Take<[u256; 8]> calls should preserve `.base` code-space param overrides; caller_arg={:?}; arg_local={:?}; arg_infos={:?}; call={take_get_call:?}; overrides={overrides:?}",
            take_get_call
                .args
                .first()
                .map(|arg| sum_first4.body.value(*arg)),
            take_get_call
                .args
                .first()
                .and_then(|arg| match sum_first4.body.value(*arg).origin {
                    crate::ValueOrigin::Local(local) => Some(sum_first4.body.local(local)),
                    _ => None,
                }),
            take_get_call
                .args
                .first()
                .map(|arg| crate::repr::pointer_leaf_infos_for_value(
                    &db,
                    &CoreLib::new(&db, top_mod.scope()),
                    &sum_first4.body.values,
                    &sum_first4.body.locals,
                    *arg,
                )),
        );
    }
}
