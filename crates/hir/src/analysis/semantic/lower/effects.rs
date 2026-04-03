use crate::{
    analysis::{
        HirAnalysisDb,
        semantic::{SEffectArg, SEffectArgValue, SPlace, SValueId},
        ty::{
            effects::EffectKeyKind,
            ty_check::{
                BodyOwner, EffectArg, EffectParamSite, EffectPassMode, LocalBinding,
                ResolvedEffectArg,
            },
        },
    },
    hir_def::ExprId,
    semantic::{EffectEnvView, ProviderBinding},
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
                target_ty: arg.instantiated_target_ty,
                provider: arg.provider,
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
}

pub fn owner_effect_bindings<'db>(
    db: &'db dyn HirAnalysisDb,
    owner: BodyOwner<'db>,
) -> Vec<LocalBinding<'db>> {
    effect_param_site(owner)
        .into_iter()
        .flat_map(|site| EffectEnvView::new(site).requirements(db))
        .filter(|binding| {
            matches!(
                binding.key.kind(),
                EffectKeyKind::Type | EffectKeyKind::Trait
            )
        })
        .map(|binding| LocalBinding::EffectParam {
            site: binding.binding_site,
            idx: binding.binding_idx as usize,
            key_path: binding.binding_path,
            is_mut: binding.is_mut,
        })
        .collect()
}

pub fn resolved_provider_binding_for_owner_effect<'db>(
    db: &'db dyn HirAnalysisDb,
    owner: BodyOwner<'db>,
    binding: LocalBinding<'db>,
) -> Option<ProviderBinding<'db>> {
    let binding_idx = match binding {
        LocalBinding::EffectParam { idx, .. }
        | LocalBinding::Param {
            site: crate::analysis::ty::ty_check::ParamSite::EffectField(_),
            idx,
            ..
        } => idx,
        LocalBinding::Local { .. } | LocalBinding::Param { .. } => return None,
    };
    let site = effect_param_site(owner)?;
    let view = EffectEnvView::new(site);
    let provider_idx = view
        .resolutions(db)
        .into_iter()
        .find(|resolution| resolution.requirement_idx as usize == binding_idx)?
        .provider_idx;
    view.providers(db)
        .into_iter()
        .find(|provider| provider.provider_idx == provider_idx)
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
                idx: lhs_idx,
                key_path: lhs_key,
                ..
            },
            LocalBinding::EffectParam {
                idx: rhs_idx,
                key_path: rhs_key,
                ..
            },
        ) => lhs_idx == rhs_idx && lhs_key == rhs_key,
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

pub(super) fn lower_seq_effect_args<'db>(
    cx: &mut SmirLowerCtxt<'db>,
    args: &[ResolvedEffectArg<'db>],
) -> Box<[SEffectArg<'db>]> {
    args.iter()
        .map(|arg| SEffectArg {
            arg: match &arg.arg {
                EffectArg::Place(place) => {
                    SEffectArgValue::Place(cx.lower_place_data(place.clone()))
                }
                EffectArg::Value(expr) => SEffectArgValue::Value(
                    cx.with_binding_values
                        .get(expr)
                        .copied()
                        .unwrap_or_else(|| cx.lower_expr(*expr)),
                ),
                EffectArg::Binding(binding) => {
                    let local = cx.alloc_binding_local(*binding);
                    if matches!(arg.pass_mode, EffectPassMode::ByPlace) {
                        SEffectArgValue::Place(SPlace {
                            local,
                            path: Box::default(),
                        })
                    } else {
                        SEffectArgValue::Value(local)
                    }
                }
                EffectArg::Unknown => SEffectArgValue::Value(cx.unit_value()),
            },
            pass_mode: arg.pass_mode,
            target_ty: arg.instantiated_target_ty,
            provider: arg.provider,
        })
        .collect()
}
