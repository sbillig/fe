use crate::{
    analysis::{
        HirAnalysisDb,
        semantic::{SemOrigin, SemanticConstRef},
        ty::{
            assoc_const::AssocConstUse,
            trait_def::{
                assoc_const_body_and_impl_args_for_trait_inst, resolve_trait_method_instance,
            },
            trait_resolution::TraitSolveCx,
            ty_check::{BodyOwner, Callable, ConstRef, TypedBody},
            ty_def::TyId,
        },
    },
    hir_def::{CallableDef, Const},
};

use super::{GenericSubst, ImplEnv, SemanticInstanceKey};

pub(crate) fn semantic_callee_key<'db>(
    db: &'db dyn HirAnalysisDb,
    caller_key: SemanticInstanceKey<'db>,
    typed_body: &TypedBody<'db>,
    callable: &Callable<'db>,
) -> Option<SemanticInstanceKey<'db>> {
    let (owner, subst_args) = match callable.callable_def() {
        CallableDef::Func(func) => {
            let mut subst_args = callable.generic_args().to_vec();
            let owner = if let Some(inst) = callable.trait_inst()
                && let Some(name) = func.name(db).to_opt()
                && let Some((impl_func, impl_args)) = resolve_trait_method_instance(
                    db,
                    TraitSolveCx::new(db, caller_key.owner(db).scope())
                        .with_assumptions(typed_body.assumptions()),
                    inst,
                    name,
                ) {
                let trait_arg_len = inst.args(db).len();
                if subst_args.len() >= trait_arg_len {
                    let mut resolved_args = impl_args;
                    resolved_args.extend_from_slice(&subst_args[trait_arg_len..]);
                    subst_args = resolved_args;
                }
                BodyOwner::Func(impl_func)
            } else {
                BodyOwner::Func(func)
            };
            (owner, subst_args)
        }
        CallableDef::VariantCtor(_) => return None,
    };

    let impl_env = if let Some(witness) = callable.trait_inst() {
        ImplEnv::new(
            db,
            caller_key.impl_env(db).normalization_scope(db),
            typed_body.assumptions(),
            vec![witness],
        )
    } else {
        ImplEnv::new(
            db,
            caller_key.impl_env(db).normalization_scope(db),
            typed_body.assumptions(),
            Vec::new(),
        )
    };

    Some(SemanticInstanceKey::new(
        db,
        owner,
        GenericSubst::new(db, subst_args),
        impl_env,
    ))
}

pub(crate) fn resolve_semantic_const_ref<'db>(
    db: &'db dyn HirAnalysisDb,
    const_ref: ConstRef<'db>,
    ty: TyId<'db>,
    origin: SemOrigin<'db>,
) -> Option<SemanticConstRef<'db>> {
    let instance = match const_ref {
        ConstRef::Const(const_) => semantic_const_key_for_const(db, const_),
        ConstRef::TraitConst(assoc) => semantic_const_key_for_assoc_const(db, assoc, ty),
    }?;
    Some(SemanticConstRef::new(db, instance, ty, origin))
}

fn semantic_const_key_for_const<'db>(
    db: &'db dyn HirAnalysisDb,
    const_: Const<'db>,
) -> Option<SemanticInstanceKey<'db>> {
    let owner = BodyOwner::Const(const_);
    Some(SemanticInstanceKey::new(
        db,
        owner,
        GenericSubst::empty(db),
        ImplEnv::empty(db, owner.scope()),
    ))
}

fn semantic_const_key_for_assoc_const<'db>(
    db: &'db dyn HirAnalysisDb,
    assoc: AssocConstUse<'db>,
    ty: TyId<'db>,
) -> Option<SemanticInstanceKey<'db>> {
    let (body, impl_args) = assoc_const_body_and_impl_args_for_trait_inst(
        db,
        assoc.solve_cx(db),
        assoc.inst(),
        assoc.name(),
    )?;
    Some(SemanticInstanceKey::new(
        db,
        BodyOwner::AnonConstBody { body, expected: ty },
        GenericSubst::new(db, impl_args),
        ImplEnv::new(
            db,
            assoc.origin_scope(),
            assoc.assumptions(),
            vec![assoc.inst()],
        ),
    ))
}
