use crate::{
    analysis::{
        HirAnalysisDb,
        semantic::{
            SEffectArg, SEffectArgValue, SPlace, SValueId,
            resolved_provider_binding_for_instance_effect,
        },
        ty::{
            ProviderAddressSpace,
            effects::EffectKeyKind,
            ty_check::{
                BodyOwner, EffectArg, EffectParamSite, EffectPassMode, LocalBinding,
                ResolvedEffectArg,
            },
        },
    },
    hir_def::ExprId,
    semantic::EffectEnvView,
};

use super::body::SmirLowerCtxt;

impl<'db> SmirLowerCtxt<'db> {
    pub(super) fn lower_effect_args(&mut self, call_expr: ExprId) -> Box<[SEffectArg<'db>]> {
        let args = self
            .typed_body
            .call_effect_args(call_expr)
            .into_iter()
            .flatten()
            .cloned()
            .collect::<Vec<_>>();
        args.into_iter()
            .map(|arg| SEffectArg {
                binding_idx: arg.binding_idx,
                arg: match &arg.arg {
                    EffectArg::Place(place) => {
                        SEffectArgValue::Place(self.lower_place_data(place.clone()))
                    }
                    EffectArg::Value(expr) => SEffectArgValue::Value(
                        self.with_binding_values
                            .get(expr)
                            .copied()
                            .unwrap_or_else(|| self.lower_expr(*expr)),
                    ),
                    EffectArg::Binding(binding) => {
                        let local = self.alloc_binding_local(*binding);
                        if matches!(arg.pass_mode, EffectPassMode::ByPlace) {
                            SEffectArgValue::Place(SPlace {
                                local,
                                path: Box::default(),
                            })
                        } else {
                            SEffectArgValue::Value(local)
                        }
                    }
                    EffectArg::Unknown => SEffectArgValue::Value(self.unit_value()),
                },
                pass_mode: arg.pass_mode,
                target_ty: arg.provider_target_ty,
                provider: self.effect_arg_provider_space(&arg),
            })
            .collect()
    }

    pub(super) fn lower_with_expr(
        &mut self,
        bindings: &[crate::hir_def::expr::WithBinding<'db>],
        body: ExprId,
    ) -> SValueId {
        let mut saved = Vec::with_capacity(bindings.len());
        for binding in bindings {
            let value_expr = binding.value;
            let value = self.lower_expr(value_expr);
            saved.push((
                value_expr,
                self.with_binding_values.insert(value_expr, value),
            ));
        }

        let body_value = self.lower_expr(body);
        for (expr, previous) in saved.into_iter().rev() {
            if let Some(previous) = previous {
                self.with_binding_values.insert(expr, previous);
            } else {
                self.with_binding_values.remove(&expr);
            }
        }
        body_value
    }

    pub(super) fn lower_seq_effect_args(
        &mut self,
        args: &[ResolvedEffectArg<'db>],
    ) -> Box<[SEffectArg<'db>]> {
        args.iter()
            .map(|arg| SEffectArg {
                binding_idx: arg.binding_idx,
                arg: match &arg.arg {
                    EffectArg::Place(place) => {
                        SEffectArgValue::Place(self.lower_place_data(place.clone()))
                    }
                    EffectArg::Value(expr) => SEffectArgValue::Value(
                        self.with_binding_values
                            .get(expr)
                            .copied()
                            .unwrap_or_else(|| self.lower_expr(*expr)),
                    ),
                    EffectArg::Binding(binding) => {
                        let local = self.alloc_binding_local(*binding);
                        if matches!(arg.pass_mode, EffectPassMode::ByPlace) {
                            SEffectArgValue::Place(SPlace {
                                local,
                                path: Box::default(),
                            })
                        } else {
                            SEffectArgValue::Value(local)
                        }
                    }
                    EffectArg::Unknown => SEffectArgValue::Value(self.unit_value()),
                },
                pass_mode: arg.pass_mode,
                target_ty: arg.provider_target_ty,
                provider: self.effect_arg_provider_space(arg),
            })
            .collect()
    }

    fn effect_arg_provider_space(
        &self,
        arg: &ResolvedEffectArg<'db>,
    ) -> Option<ProviderAddressSpace> {
        arg.provider.or_else(|| match &arg.arg {
            EffectArg::Place(place) => {
                let crate::analysis::place::PlaceBase::Binding(binding) = place.base;
                self.binding_provider_space(binding)
            }
            EffectArg::Binding(binding) => self.binding_provider_space(*binding),
            EffectArg::Value(_) | EffectArg::Unknown => None,
        })
    }

    fn binding_provider_space(&self, binding: LocalBinding<'db>) -> Option<ProviderAddressSpace> {
        resolved_provider_binding_for_instance_effect(self.db, self.instance, binding)
            .and_then(|provider| provider.semantics.address_space)
    }
}

pub fn owner_effect_bindings<'db>(
    db: &'db dyn HirAnalysisDb,
    owner: BodyOwner<'db>,
) -> Vec<LocalBinding<'db>> {
    effect_param_site(owner)
        .into_iter()
        .flat_map(|site| {
            let view = EffectEnvView::new(site);
            view.requirements(db)
                .into_iter()
                .filter(|binding| {
                    matches!(
                        binding.key.kind(),
                        EffectKeyKind::Type | EffectKeyKind::Trait
                    )
                })
                .filter_map(move |binding| {
                    view.resolved_binding(db, binding.binding_idx as usize)
                        .map(|binding| LocalBinding::effect_param(&binding))
                })
        })
        .collect()
}

pub fn effect_param_site<'db>(owner: BodyOwner<'db>) -> Option<EffectParamSite<'db>> {
    match owner {
        BodyOwner::Func(func) => Some(EffectParamSite::Func(func)),
        BodyOwner::Const(_) | BodyOwner::AnonConstBody { .. } => None,
        BodyOwner::ContractInit { contract } => Some(EffectParamSite::ContractInit { contract }),
        BodyOwner::ContractRecvArm {
            contract,
            recv_idx,
            arm_idx,
        } => Some(EffectParamSite::ContractRecvArm {
            contract,
            recv_idx,
            arm_idx,
        }),
    }
}

pub fn same_owner_effect_binding<'db>(lhs: LocalBinding<'db>, rhs: LocalBinding<'db>) -> bool {
    match (lhs, rhs) {
        (
            LocalBinding::EffectParam {
                site: lhs_site,
                idx: lhs_idx,
                ..
            },
            LocalBinding::EffectParam {
                site: rhs_site,
                idx: rhs_idx,
                ..
            },
        ) => lhs_site == rhs_site && lhs_idx == rhs_idx,
        (
            LocalBinding::Param {
                site: crate::analysis::ty::ty_check::ParamSite::EffectField(_),
                idx: lhs_idx,
                ty: lhs_ty,
                ..
            },
            LocalBinding::Param {
                site: crate::analysis::ty::ty_check::ParamSite::EffectField(_),
                idx: rhs_idx,
                ty: rhs_ty,
                ..
            },
        ) => lhs_idx == rhs_idx && lhs_ty == rhs_ty,
        _ => lhs == rhs,
    }
}
