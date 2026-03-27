use rustc_hash::FxHashSet;

use crate::{
    db::MirDb,
    instance::RuntimeInstance,
    runtime::{
        HandleKind, HandleView, Layout, LocalSlotKind, RExpr, RStmt, RTerminator, RuntimeBody,
        RuntimeCarrier, RuntimeClass, RuntimeProgramView, RuntimeSignature, ScalarClass,
        ScalarRepr, ScalarRole,
    },
    verify::VerifyError,
};

use super::{
    consts::verify_const_region,
    layout::verify_class_layouts,
    place::{
        enum_extract_class, enum_tag_class, enum_tag_class_from_value, project_place,
        runtime_value_class, scalar_class_from_const, verify_enum_handle,
        verify_enum_write_variant, verify_value_enum_variant,
    },
};

pub fn verify_runtime_body<'db>(
    db: &'db dyn MirDb,
    program: &impl RuntimeProgramView<'db>,
    body: &RuntimeBody<'db>,
) -> Result<(), VerifyError<'db>> {
    verify_signature(body)?;

    let mut visited_layouts = FxHashSet::default();
    for (local_idx, local) in body.locals.iter().enumerate() {
        if let LocalSlotKind::Slot(class) = &local.slot
            && !matches!(&local.carrier, RuntimeCarrier::Value(carrier) if carrier == class)
        {
            return Err(VerifyError::SlotCarrierMismatch(
                crate::runtime::RLocalId::from_u32(local_idx as u32),
            ));
        }

        if let RuntimeCarrier::Value(class) = &local.carrier {
            verify_class_layouts(db, program, class, &mut visited_layouts)?;
        }
    }

    for block in &body.blocks {
        for stmt in &block.stmts {
            match stmt {
                RStmt::Assign { dst, expr } => verify_assign(db, program, body, *dst, expr)?,
                RStmt::Store { dst, src } => verify_store(db, program, body, dst, *src)?,
                RStmt::CopyInto { dst, src } => verify_copy_into(db, program, body, dst, *src)?,
                RStmt::EnumSetTag { root, variant } => {
                    verify_enum_handle(body, *root, *variant, program)?;
                }
                RStmt::EnumWriteVariant {
                    root,
                    variant,
                    fields,
                } => verify_enum_write_variant(program, body, *root, *variant, fields)?,
            }
        }

        verify_terminator(db, program, body, &block.terminator)?;
    }

    Ok(())
}

fn verify_signature<'db>(body: &RuntimeBody<'db>) -> Result<(), VerifyError<'db>> {
    for param in &body.signature.params {
        let local = body
            .local(param.local)
            .ok_or(VerifyError::MissingRuntimeLocal(param.local))?;
        if local.carrier != RuntimeCarrier::Value(param.class.clone()) {
            return Err(VerifyError::SlotCarrierMismatch(param.local));
        }
    }

    Ok(())
}

fn verify_call<'db>(
    _db: &'db dyn MirDb,
    program: &impl RuntimeProgramView<'db>,
    body: &RuntimeBody<'db>,
    callee: RuntimeInstance<'db>,
    args: &[crate::runtime::RValueId],
) -> Result<(), VerifyError<'db>> {
    let RuntimeSignature { params, .. } = program.body(callee).signature;
    if params.len() != args.len() {
        return Err(VerifyError::CallArgCountMismatch(callee));
    }

    for (idx, (arg, param)) in args.iter().zip(params.iter()).enumerate() {
        let Some(class) = body.value_class(*arg) else {
            return Err(VerifyError::ErasedRuntimeValue(*arg));
        };
        if class != &param.class {
            return Err(VerifyError::CallArgClassMismatch(callee, idx));
        }
    }

    Ok(())
}

fn verify_assign<'db>(
    db: &'db dyn MirDb,
    program: &impl RuntimeProgramView<'db>,
    body: &RuntimeBody<'db>,
    dst: crate::runtime::RLocalId,
    expr: &RExpr<'db>,
) -> Result<(), VerifyError<'db>> {
    let local = body
        .local(dst)
        .ok_or(VerifyError::MissingRuntimeLocal(dst))?;
    let dst_class = match &local.carrier {
        RuntimeCarrier::Erased => None,
        RuntimeCarrier::Value(class) => Some(class.clone()),
    };

    let expr_class = match expr {
        RExpr::Use(value) => Some(runtime_value_class(body, *value)?.clone()),
        RExpr::ConstScalar(value) => Some(RuntimeClass::Scalar(scalar_class_from_const(value))),
        RExpr::Unary { value, .. } => {
            if !matches!(
                (runtime_value_class(body, *value)?, &dst_class),
                (RuntimeClass::Scalar(_), Some(RuntimeClass::Scalar(_)))
            ) {
                return Err(VerifyError::InvalidExprClass(dst));
            }
            dst_class.clone()
        }
        RExpr::Binary { lhs, rhs, .. } => {
            if !matches!(
                (
                    runtime_value_class(body, *lhs)?,
                    runtime_value_class(body, *rhs)?,
                    &dst_class,
                ),
                (
                    RuntimeClass::Scalar(_),
                    RuntimeClass::Scalar(_),
                    Some(RuntimeClass::Scalar(_)),
                )
            ) {
                return Err(VerifyError::InvalidExprClass(dst));
            }
            dst_class.clone()
        }
        RExpr::Cast { value, to } => {
            let _ = runtime_value_class(body, *value)?;
            Some(RuntimeClass::Scalar(to.clone()))
        }
        RExpr::ConstHandle { region, layout } => {
            let region_id = *region;
            let region = program.const_region(region_id);
            verify_const_region(db, program, region.clone())?;
            if region.layout != *layout {
                return Err(VerifyError::InvalidConstRegion(region_id));
            }
            Some(RuntimeClass::Handle {
                layout: *layout,
                kind: HandleKind::ConstValue,
                view: HandleView::Whole,
            })
        }
        RExpr::AllocObject { layout } => Some(RuntimeClass::Handle {
            layout: *layout,
            kind: HandleKind::ObjectValue,
            view: HandleView::Whole,
        }),
        RExpr::MaterializeToObject { src } => {
            let src_class = runtime_value_class(body, *src)?;
            let Some(RuntimeClass::Handle {
                layout,
                kind: HandleKind::ObjectValue,
                view: HandleView::Whole,
            }) = &dst_class
            else {
                return Err(VerifyError::InvalidExprClass(dst));
            };
            match src_class {
                RuntimeClass::AggregateValue { layout: src_layout } if *src_layout == *layout => {}
                RuntimeClass::Handle {
                    layout: src_layout,
                    kind: HandleKind::ConstValue,
                    view: HandleView::Whole,
                } if *src_layout == *layout => {}
                _ => return Err(VerifyError::InvalidExprClass(dst)),
            }
            dst_class.clone()
        }
        RExpr::ProviderFromRaw {
            raw,
            provider_ty,
            space,
            layout,
        } => {
            let RuntimeClass::RawAddr {
                space: raw_space, ..
            } = runtime_value_class(body, *raw)?
            else {
                return Err(VerifyError::InvalidExprClass(dst));
            };
            if *raw_space != *space {
                return Err(VerifyError::InvalidExprClass(dst));
            }
            Some(RuntimeClass::Handle {
                layout: *layout,
                kind: HandleKind::Provider {
                    provider_ty: *provider_ty,
                    space: *space,
                },
                view: HandleView::Whole,
            })
        }
        RExpr::AddrOf { place } => {
            let _ = project_place(db, program, body, place)?;
            if !matches!(
                &dst_class,
                Some(
                    RuntimeClass::Handle {
                        kind: HandleKind::Provider { .. },
                        ..
                    } | RuntimeClass::RawAddr { .. }
                )
            ) {
                return Err(VerifyError::InvalidExprClass(dst));
            }
            dst_class.clone()
        }
        RExpr::Load { place } => Some(project_place(db, program, body, place)?),
        RExpr::Call { callee, args } => {
            verify_call(db, program, body, *callee, args)?;
            program.body(*callee).signature.ret.clone()
        }
        RExpr::ProviderToRaw { value } => {
            if !matches!(
                (runtime_value_class(body, *value)?, &dst_class),
                (
                    RuntimeClass::Handle {
                        kind: HandleKind::Provider { .. },
                        ..
                    },
                    Some(RuntimeClass::RawAddr { .. }),
                )
            ) {
                return Err(VerifyError::InvalidExprClass(dst));
            }
            dst_class.clone()
        }
        RExpr::EnumMake {
            layout,
            variant,
            fields,
        } => {
            let expected = RuntimeClass::AggregateValue { layout: *layout };
            verify_value_enum_variant(program, body, expected.clone(), *variant, fields)?;
            Some(expected)
        }
        RExpr::EnumTagOfValue { value } => Some(enum_tag_class_from_value(db, body, *value)?),
        RExpr::EnumIsVariant { value, variant } => {
            verify_value_enum_variant(
                program,
                body,
                runtime_value_class(body, *value)?.clone(),
                *variant,
                &[],
            )?;
            Some(RuntimeClass::Scalar(ScalarClass {
                repr: ScalarRepr::Bool,
                role: ScalarRole::Plain,
            }))
        }
        RExpr::EnumExtract {
            value,
            variant,
            field,
        } => Some(enum_extract_class(db, body, *value, *variant, *field)?),
        RExpr::EnumGetTag { root } => {
            let RuntimeClass::Handle { layout, .. } = runtime_value_class(body, *root)?.clone()
            else {
                return Err(VerifyError::InvalidExprClass(dst));
            };
            if !matches!(program.layout(layout), Layout::Enum(_)) {
                return Err(VerifyError::InvalidEnumTag(layout));
            }
            Some(RuntimeClass::Scalar(enum_tag_class(layout, program)))
        }
        RExpr::EnumAssertVariantRef { root, variant } => {
            let class = verify_enum_handle(body, *root, *variant, program)?;
            let RuntimeClass::Handle { layout, kind, .. } = class else {
                unreachable!();
            };
            Some(RuntimeClass::Handle {
                layout,
                kind,
                view: HandleView::EnumVariant(*variant),
            })
        }
    };

    if let Some(dst_class) = &dst_class
        && expr_class.as_ref() != Some(dst_class)
    {
        return Err(VerifyError::InvalidExprClass(dst));
    }
    if dst_class.is_none() && matches!(expr, RExpr::Call { .. }) {
        return Ok(());
    }
    if dst_class.is_none() && expr_class.is_some() {
        return Err(VerifyError::InvalidExprClass(dst));
    }
    Ok(())
}

fn verify_store<'db>(
    db: &'db dyn MirDb,
    program: &impl RuntimeProgramView<'db>,
    body: &RuntimeBody<'db>,
    dst: &crate::runtime::RuntimePlace,
    src: crate::runtime::RValueId,
) -> Result<(), VerifyError<'db>> {
    let target = project_place(db, program, body, dst)?;
    let source = runtime_value_class(body, src)?;
    if &target != source {
        return Err(VerifyError::InvalidStoreClass);
    }
    Ok(())
}

fn verify_copy_into<'db>(
    db: &'db dyn MirDb,
    program: &impl RuntimeProgramView<'db>,
    body: &RuntimeBody<'db>,
    dst: &crate::runtime::RuntimePlace,
    src: crate::runtime::RValueId,
) -> Result<(), VerifyError<'db>> {
    let target = project_place(db, program, body, dst)?;
    let source = runtime_value_class(body, src)?;
    if &target != source {
        return Err(VerifyError::InvalidCopyClass);
    }
    Ok(())
}

fn verify_terminator<'db>(
    _db: &'db dyn MirDb,
    _program: &impl RuntimeProgramView<'db>,
    body: &RuntimeBody<'db>,
    terminator: &RTerminator<'db>,
) -> Result<(), VerifyError<'db>> {
    match terminator {
        RTerminator::Goto(_) => Ok(()),
        RTerminator::Branch { cond, .. } => {
            let RuntimeClass::Scalar(ScalarClass {
                repr: ScalarRepr::Bool,
                role: ScalarRole::Plain,
            }) = runtime_value_class(body, *cond)?
            else {
                return Err(VerifyError::InvalidExprClass(*cond));
            };
            Ok(())
        }
        RTerminator::MatchEnumTag {
            tag,
            enum_layout,
            cases,
            ..
        } => {
            let class = runtime_value_class(body, *tag)?;
            let RuntimeClass::Scalar(scalar) = class else {
                return Err(VerifyError::InvalidEnumTag(*enum_layout));
            };
            if !matches!(
                scalar.role,
                ScalarRole::EnumTag { enum_layout: tag_layout } if tag_layout == *enum_layout
            ) {
                return Err(VerifyError::InvalidEnumTag(*enum_layout));
            }
            let mut seen = FxHashSet::default();
            for (variant, _) in cases {
                if variant.enum_layout != *enum_layout || !seen.insert(variant.index) {
                    return Err(VerifyError::InvalidVariant(*enum_layout, variant.index));
                }
            }
            Ok(())
        }
        RTerminator::Return(value) => {
            let class = value
                .map(|value| runtime_value_class(body, value).cloned())
                .transpose()?;
            if class != body.signature.ret {
                return Err(VerifyError::InvalidReturnClass);
            }
            Ok(())
        }
    }
}
