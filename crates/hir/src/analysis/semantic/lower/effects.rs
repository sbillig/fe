use crate::{
    analysis::{
        HirAnalysisDb,
        semantic::{SEffectArg, SEffectArgValue, SPlace, SValueId},
        ty::{
            effects::EffectKeyKind,
            ty_check::{
                BodyOwner, EffectArg, EffectParamSite, EffectPassMode, LocalBinding, ParamSite,
                ResolvedEffectArg,
            },
            ty_def::{InvalidCause, TyId},
        },
    },
    hir_def::{ExprId, params::FuncParamMode},
    semantic::{EffectEnvView, EffectSource},
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

pub(super) fn owner_effect_bindings<'db>(
    db: &'db dyn HirAnalysisDb,
    owner: BodyOwner<'db>,
) -> Vec<LocalBinding<'db>> {
    effect_param_site(owner)
        .into_iter()
        .flat_map(|site| EffectEnvView::new(site).bindings(db).iter())
        .filter(|binding| matches!(binding.key_kind, EffectKeyKind::Type | EffectKeyKind::Trait))
        .map(|binding| match binding.source {
            EffectSource::Root => LocalBinding::EffectParam {
                site: binding.binding_site,
                idx: binding.binding_idx as usize,
                key_path: binding.binding_path,
                is_mut: binding.is_mut,
            },
            EffectSource::Field(_) => LocalBinding::Param {
                site: ParamSite::EffectField(binding.binding_site),
                idx: binding.binding_idx as usize,
                mode: FuncParamMode::View,
                ty: binding
                    .key_ty
                    .unwrap_or_else(|| TyId::invalid(db, InvalidCause::Other)),
                is_mut: binding.is_mut,
            },
        })
        .collect()
}

fn effect_param_site<'db>(owner: BodyOwner<'db>) -> Option<EffectParamSite<'db>> {
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
