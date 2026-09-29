//! Const declaration requirements are discharged after inference, for every body
//! owner. Concrete discharge uses ordinary CTFE. Symbolic forwarding compares
//! resolved, typed expressions after substitution, without evaluating
//! unknown parameters or assuming the obligation being checked.
use super::*;
use crate::analysis::name_resolution::resolve_path_with_minter;
use crate::analysis::ty::{subst::substitute_complete, ty_lower::CompleteSubst};
use crate::hir_def::{GenericArg, GenericArgListId, ItemKind, UnOp, scope_graph::ScopeId};

#[derive(Debug, Clone, PartialEq, Eq)]
struct PredicateKey<'db> {
    ty: TyId<'db>,
    arithmetic: crate::hir_def::attr::ArithmeticMode,
    operation: Option<Callable<'db>>,
    term: PredicateTerm<'db>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum PredicateTerm<'db> {
    Literal(LitKind<'db>),
    TypeConst(TyId<'db>),
    Const(Const<'db>),
    TraitConst(TraitInstId<'db>, crate::hir_def::IdentId<'db>),
    InherentConst(
        crate::hir_def::Impl<'db>,
        TyId<'db>,
        crate::hir_def::IdentId<'db>,
    ),
    Binary(BinOp, Box<PredicateKey<'db>>, Box<PredicateKey<'db>>),
    Unary(UnOp, Box<PredicateKey<'db>>),
    Cast(Box<PredicateKey<'db>>),
    Call(Vec<PredicateKey<'db>>),
}

// A declaration's full parameter schema also resolves the parameters that
// lexical lookup keeps owned by an enclosing impl, so one simultaneous
// substitution instantiates a method's requirement with its callable arguments.
fn requirement_subst<'db>(
    db: &'db dyn HirAnalysisDb,
    scope: ScopeId<'db>,
    args: &[TyId<'db>],
) -> Option<CompleteSubst<'db>> {
    let owner = GenericParamOwner::from_item_opt(scope.item())?;
    CompleteSubst::for_owner(db, owner, args.to_vec()).ok()
}

// Arguments inferred in a method body can mention its impl's parameters as the
// impl owns them. Rebase them onto the method's own formals, which is how the
// caller's premises are instantiated, so both sides compare in one basis.
fn caller_args<'db>(
    db: &'db dyn HirAnalysisDb,
    caller: WhereClauseOwner<'db>,
    args: Vec<TyId<'db>>,
) -> Vec<TyId<'db>> {
    let Some(owner) = GenericParamOwner::from_item_opt(caller.into()) else {
        return args;
    };
    let identity = collect_generic_params(db, owner).params(db).to_vec();
    CompleteSubst::for_owner(db, owner, identity)
        .ok()
        .and_then(|subst| substitute_complete(db, args.clone(), &subst).ok())
        .unwrap_or(args)
}

fn predicate_key<'db>(
    db: &'db dyn HirAnalysisDb,
    body: Body<'db>,
    typed: &TypedBody<'db>,
    expr: ExprId,
    subst: &CompleteSubst<'db>,
) -> Option<PredicateKey<'db>> {
    let instantiate = |ty| substitute_complete(db, ty, subst).ok();
    let child = |expr| predicate_key(db, body, typed, expr, subst);
    let term = match expr.data(db, body).borrowed().to_opt()? {
        Expr::Lit(lit) => PredicateTerm::Literal(*lit),
        Expr::Path(_) => {
            if let Some(ValuePathRef::TypeConst(ty)) = typed.value_path_ref(expr) {
                PredicateTerm::TypeConst(instantiate(ty)?)
            } else {
                // A reference's lookup scope is provenance, not its identity.
                // Keep the resolved declaration and substituted receiver/goal.
                match typed.expr_const_ref(expr)? {
                    ConstRef::Const(constant) => PredicateTerm::Const(constant),
                    ConstRef::TraitConst(reference) => PredicateTerm::TraitConst(
                        substitute_complete(db, reference.inst(), subst).ok()?,
                        reference.name(),
                    ),
                    ConstRef::InherentConst(reference) => PredicateTerm::InherentConst(
                        reference.impl_(),
                        instantiate(reference.receiver_ty())?,
                        reference.name(),
                    ),
                }
            }
        }
        Expr::Bin(lhs, rhs, op) => {
            PredicateTerm::Binary(*op, Box::new(child(*lhs)?), Box::new(child(*rhs)?))
        }
        Expr::Un(value, op) => PredicateTerm::Unary(*op, Box::new(child(*value)?)),
        Expr::Cast(value, _) => PredicateTerm::Cast(Box::new(child(*value)?)),
        Expr::Call(_, call_args) => {
            typed.callable_expr(expr)?;
            PredicateTerm::Call(
                call_args
                    .iter()
                    .map(|arg| child(arg.expr))
                    .collect::<Option<_>>()?,
            )
        }
        _ => return None,
    };
    Some(PredicateKey {
        ty: instantiate(typed.expr_ty(db, expr))?,
        arithmetic: BodyOwner::AnonConstBody {
            body,
            expected: TyId::bool(db),
        }
        .arithmetic_mode(db),
        operation: match typed.callable_expr(expr) {
            Some(callable) => Some(substitute_complete(db, callable.clone(), subst).ok()?),
            None => None,
        },
        term,
    })
}

fn predicate_flags<'db>(db: &'db dyn HirAnalysisDb, mut typed: TypedBody<'db>) -> TyFlags {
    // Ambient trait assumptions are not dependencies of the expression itself.
    typed.assumptions = PredicateListId::empty_list(db);
    collect_flags(db, typed)
}

#[salsa::tracked(return_ref, cycle_initial=formation_cycle_initial, cycle_fn=formation_cycle_recover)]
pub(super) fn check_predicate_formation<'db>(
    db: &'db dyn HirAnalysisDb,
    body: Body<'db>,
) -> (Vec<FuncBodyDiag<'db>>, TypedBody<'db>) {
    let owner = BodyOwner::AnonConstBody {
        body,
        expected: TyId::bool(db),
    };
    let (mut diags, typed) = infer_body(db, owner).clone();
    if diags.is_empty() || static_assert_ignorable_type_diags(db, &diags) {
        diags.extend(
            crate::analysis::ty::const_check::check_const_body_expressions(db, body, &typed),
        );
        diags.extend(check_body_requirements(db, owner, &typed));
    }
    if has_recursive_requirement(&diags) {
        diags = vec![BodyDiag::RecursiveConstRequirement(body.span().into()).into()];
    }
    (diags, typed)
}

pub(super) fn predicate_may_depend_on_params<'db>(
    db: &'db dyn HirAnalysisDb,
    body: Body<'db>,
) -> bool {
    let typed = &infer_body(
        db,
        BodyOwner::AnonConstBody {
            body,
            expected: TyId::bool(db),
        },
    )
    .1;
    predicate_flags(db, typed.clone()).contains(TyFlags::HAS_PARAM)
}

// Inherent calls keep ordinary method resolution. Requirements constrain the
// resolved call; they do not participate in candidate selection.
pub(super) fn function_requirements_supported(db: &dyn HirAnalysisDb, func: Func<'_>) -> bool {
    !func.is_associated_func(db) || matches!(func.scope().parent_item(db), Some(ItemKind::Impl(_)))
}

// Requirements scope over function signatures/bodies and ADT fields, but
// their formation must be checked without those assumptions. In particular,
// nested anonymous constants inside a predicate are part of its formation.
fn requirement_premise_owner<'db>(
    db: &'db dyn HirAnalysisDb,
    owner: BodyOwner<'db>,
) -> Option<WhereClauseOwner<'db>> {
    if !matches!(owner, BodyOwner::Func(_) | BodyOwner::AnonConstBody { .. }) {
        return None;
    }
    premise_owner_in_scope(db, owner.scope())
}

fn premise_owner_in_scope<'db>(
    db: &'db dyn HirAnalysisDb,
    origin: ScopeId<'db>,
) -> Option<WhereClauseOwner<'db>> {
    let mut current = Some(origin);
    while let Some(scope) = current {
        match scope.item() {
            item @ (ItemKind::Func(_) | ItemKind::Struct(_) | ItemKind::Enum(_)) => {
                if matches!(item, ItemKind::Func(func) if !function_requirements_supported(db, func))
                {
                    return None;
                }
                let candidate = WhereClauseOwner::from_item_opt(item)?;
                if candidate
                    .where_clause(db)
                    .const_predicates(db)
                    .iter()
                    .any(|predicate| origin.is_transitive_child_of(db, predicate.scope()))
                {
                    return None;
                }
                return Some(candidate);
            }
            ItemKind::Body(_) => current = scope.parent(db),
            // A nested declaration does not inherit function premises.
            _ => return None,
        }
    }
    None
}

pub(super) fn check_body_requirements<'db>(
    db: &'db dyn HirAnalysisDb,
    owner: BodyOwner<'db>,
    typed: &TypedBody<'db>,
) -> Vec<FuncBodyDiag<'db>> {
    let Some(body) = typed.body() else {
        return Vec::new();
    };
    let caller = requirement_premise_owner(db, owner);
    let direct_callees: FxHashSet<_> = body
        .exprs(db)
        .values()
        .filter_map(|expr| match expr.borrowed().to_opt()? {
            Expr::Call(callee, _) => Some(*callee),
            _ => None,
        })
        .collect();
    let mut diags = Vec::new();
    // Requirements are checked where a type enters the body: an authored type
    // (checked by `check_declared_type_requirements`) or an instantiating
    // expression. A pattern, a binding use, or a block or branch only carries a
    // type that entered elsewhere, so checking it again repeats that report.
    // A function-typed expression is not checked for its type arguments: they
    // are written types, checked where they are written, or inferred from the
    // argument and result expressions, which are checked themselves. The type
    // before `::` in a path is neither, so an item reached through it checks
    // that type itself (`check_entered_header`).
    for (expr, data) in body.exprs(db).iter() {
        if typed.expr_binding(expr).is_some()
            || matches!(
                data.borrowed().to_opt(),
                Some(Expr::Block(..) | Expr::If(..) | Expr::Match(..) | Expr::With(..))
            )
        {
            continue;
        }
        let ty = typed.expr_ty(db, expr);
        if !matches!(ty.base_ty(db).data(db), TyData::TyBase(TyBase::Func(_)))
            && let Some((_, diag)) =
                check_type_requirements(db, ty, owner.scope(), expr.span(body).into(), &[])
        {
            diags.push(diag.into());
        }
        if let Some(headers) = const_ref_headers(db, typed, expr)
            && let Some(diag) = check_entered_header(db, typed, expr, headers, owner.scope())
        {
            diags.push(diag.into());
        }
        if direct_callees.contains(&expr) && typed.callable_expr(expr).is_none() {
            continue;
        }
        let (definition, args) = if let Some(callable) = typed.callable_expr(expr) {
            (callable.callable_def(), callable.generic_args())
        } else {
            let ty = typed.expr_ty(db, expr);
            let (base, args) = ty.decompose_ty_app(db);
            let TyData::TyBase(TyBase::Func(definition)) = base.data(db) else {
                continue;
            };
            (*definition, args)
        };
        let func = match definition {
            CallableDef::Func(func) => func,
            // A constructor used as a value enters its enum type here, since no
            // call expression instantiates it. A called constructor's enum type
            // enters at the call, which is checked above.
            CallableDef::VariantCtor(_) => {
                if matches!(data.borrowed().to_opt(), Some(Expr::Path(..)))
                    && !direct_callees.contains(&expr)
                    && let Some((_, diag)) = check_type_requirements(
                        db,
                        definition.ret_ty(db).instantiate(db, args),
                        owner.scope(),
                        expr.span(body).into(),
                        &[],
                    )
                {
                    diags.push(diag.into());
                }
                continue;
            }
        };
        if let Some(diag) = check_entered_header(
            db,
            typed,
            expr,
            callee_headers(db, func, args),
            owner.scope(),
        ) {
            diags.push(diag.into());
        }
        // Ground clauses are already mandatory declaration checks. Unsupported
        // associated/generic owner contexts are rejected at their declarations.
        let predicates = WhereClauseOwner::Func(func)
            .where_clause(db)
            .const_predicates(db);
        if predicates.is_empty()
            || !function_requirements_supported(db, func)
            || collect_generic_params(db, func.into())
                .params(db)
                .is_empty()
        {
            continue;
        }
        let caller = caller.filter(|_| args.iter().any(|ty| ty.has_param(db)));
        for &predicate in predicates {
            let failures = discharge_requirement(
                db,
                WhereClauseOwner::Func(func),
                predicate,
                args.to_vec(),
                caller,
            );
            if !failures.is_empty() {
                diags.push(
                    BodyDiag::ConstRequirementNotSatisfied {
                        primary: expr.span(body).into(),
                        predicate: predicate.span().into(),
                        reason: requirement_reason(failures),
                    }
                    .into(),
                );
                // The diagnostic above already names the failure. The raw
                // failures point at the callee's predicate, which is not at
                // fault, so only a recursion marker is kept: predicate
                // formation needs it to absorb a cycle.
                diags.extend(
                    failures
                        .iter()
                        .filter(|diag| {
                            matches!(
                                diag,
                                FuncBodyDiag::Body(BodyDiag::RecursiveConstRequirement(_))
                            )
                        })
                        .cloned(),
                );
            }
        }
    }
    for (nested, expected) in expression_const_bodies(db, body, typed) {
        diags.extend(anon_const_requirements(db, nested, expected));
    }
    diags
}

/// The unmet requirements of an anonymous constant checked against
/// `expected`. Type lowering evaluates such a constant but does not check its
/// requirements: checking them evaluates code, which can lower the type the
/// constant is part of. So the constant's position owns them, and its owner
/// calls this: `check_body_requirements` for expression positions and
/// `check_declared_type_requirements` for types. An inference failure is
/// reported by type lowering, so it has no requirements here.
fn anon_const_requirements<'db>(
    db: &'db dyn HirAnalysisDb,
    body: Body<'db>,
    expected: TyId<'db>,
) -> Vec<FuncBodyDiag<'db>> {
    let owner = BodyOwner::AnonConstBody { body, expected };
    let (diags, typed) = infer_body(db, owner);
    if diags.is_empty() || static_assert_ignorable_type_diags(db, diags) {
        check_body_requirements(db, owner, typed)
    } else {
        Vec::new()
    }
}

/// The anonymous constants written directly in `path`'s segments, such as
/// `{ n + 1 }` in `Bounded<{ n + 1 }>::helper`.
fn path_const_bodies<'db>(db: &'db dyn HirAnalysisDb, path: PathId<'db>) -> Vec<Body<'db>> {
    let mut bodies = Vec::new();
    let mut segment = Some(path);
    while let Some(current) = segment {
        bodies.extend(generic_arg_const_bodies(db, current.generic_args(db)));
        segment = current.parent(db);
    }
    bodies
}

fn generic_arg_const_bodies<'db>(
    db: &'db dyn HirAnalysisDb,
    args: GenericArgListId<'db>,
) -> impl Iterator<Item = Body<'db>> + 'db {
    args.data(db).iter().filter_map(|arg| match arg {
        GenericArg::Const(arg) => match arg.value {
            ConstGenericArgValue::Expr(Partial::Present(body)) => Some(body),
            _ => None,
        },
        _ => None,
    })
}

/// Pairs each of `bodies` with the type lowering checks it against, read
/// from `lowered`, a lowering that deferred its anonymous constants.
fn positioned_const_bodies<'db>(
    db: &'db dyn HirAnalysisDb,
    lowered: &[TyId<'db>],
    bodies: &[Body<'db>],
) -> Vec<(Body<'db>, TyId<'db>)> {
    unevaluated_const_bodies(db, lowered)
        .into_iter()
        .filter(|(body, _)| bodies.contains(body))
        .collect()
}

/// The types a path resolution carries, including trait arguments.
fn path_res_tys<'db>(db: &'db dyn HirAnalysisDb, res: PathRes<'db>) -> Vec<TyId<'db>> {
    let mut tys = Vec::new();
    match res {
        PathRes::Trait(inst) | PathRes::TraitMethod(inst, _) => {
            tys.extend(inst.args(db).iter().copied());
        }
        res => {
            res.map_over_ty(|ty| {
                tys.push(ty);
                ty
            });
        }
    }
    tys
}

/// The anonymous constants a body writes in expression and pattern
/// positions, with the type each position checks it against: array repeat
/// lengths, explicit const arguments of calls, method calls and function
/// values, and const arguments of the other path segments. Types written in
/// the body are covered by `check_declared_type_requirements`.
fn expression_const_bodies<'db>(
    db: &'db dyn HirAnalysisDb,
    body: Body<'db>,
    typed: &TypedBody<'db>,
) -> Vec<(Body<'db>, TyId<'db>)> {
    let checked_const_ty = |ty: TyId<'db>| match ty.data(db) {
        TyData::ConstTy(const_ty) if !ty.has_invalid(db) => Some(const_ty.ty(db)),
        _ => None,
    };
    // Explicit arguments of a callable's own parameters, which the type
    // checker applies after the path resolves.
    let explicit_args = |expr: ExprId, args: GenericArgListId<'db>| {
        let (definition, generic_args) = match typed.callable_expr(expr) {
            Some(callable) => (callable.callable_def(), callable.generic_args().to_vec()),
            None => {
                let (base, generic_args) = typed.expr_ty(db, expr).decompose_ty_app(db);
                let TyData::TyBase(TyBase::Func(definition)) = base.data(db) else {
                    return Vec::new();
                };
                (*definition, generic_args.to_vec())
            }
        };
        let offset = definition.offset_to_explicit_params_position(db);
        args.data(db)
            .iter()
            .enumerate()
            .filter_map(|(idx, arg)| {
                let GenericArg::Const(arg) = arg else {
                    return None;
                };
                let ConstGenericArgValue::Expr(Partial::Present(body)) = arg.value else {
                    return None;
                };
                Some((body, checked_const_ty(*generic_args.get(offset + idx)?)?))
            })
            .collect::<Vec<_>>()
    };
    // Arguments of the other segments, which path resolution applies. Each
    // segment's arguments are read from the resolution of the path up to that
    // segment: a qualifier such as `Holder<{ n }>` in `Holder<{ n }>::make` is
    // a type, whatever the full path names.
    let resolved_args = |path: PathId<'db>, value: bool| {
        let mut entries = Vec::new();
        let mut segment = Some((path, value));
        while let Some((prefix, value)) = segment {
            let bodies: Vec<_> = generic_arg_const_bodies(db, prefix.generic_args(db)).collect();
            if !bodies.is_empty() {
                let minter = LoweringContext::deferred(HoleAnchor::TemplatePath {
                    path: prefix,
                    scope: body.scope(),
                    assumptions: typed.assumptions(),
                });
                let resolve = |value| {
                    resolve_path_with_minter(
                        db,
                        prefix,
                        body.scope(),
                        typed.assumptions(),
                        value,
                        &minter,
                    )
                };
                if let Ok(res) = resolve(value).or_else(|_| resolve(!value)) {
                    entries.extend(positioned_const_bodies(db, &path_res_tys(db, res), &bodies));
                }
            }
            segment = prefix.parent(db).map(|parent| (parent, false));
        }
        entries
    };

    let mut found: Vec<(Body<'db>, TyId<'db>)> = Vec::new();
    for (expr, data) in body.exprs(db).iter() {
        let entries = match data.borrowed().to_opt() {
            Some(Expr::ArrayRep(_, len)) => len
                .to_opt()
                .and_then(|len| {
                    let len_ty = *typed.expr_ty(db, expr).decompose_ty_app(db).1.get(1)?;
                    Some(vec![(len, checked_const_ty(len_ty)?)])
                })
                .unwrap_or_default(),
            Some(Expr::Path(Partial::Present(path))) => {
                let mut entries = resolved_args(*path, true);
                entries.extend(explicit_args(expr, path.generic_args(db)));
                entries
            }
            Some(Expr::RecordInit(Partial::Present(path), _)) => resolved_args(*path, false),
            Some(Expr::MethodCall(_, _, args, _)) => explicit_args(expr, *args),
            _ => Vec::new(),
        };
        found.extend(entries);
    }
    for pat in body.pats(db).values() {
        if let Some(
            Pat::Path(Partial::Present(path), _)
            | Pat::PathTuple(Partial::Present(path), _)
            | Pat::Record(Partial::Present(path), _),
        ) = pat.borrowed().to_opt()
        {
            found.extend(resolved_args(*path, true));
        }
    }
    // A body appears in one position, but two lookups can both find it.
    let mut seen = FxHashSet::default();
    found.retain(|(body, _)| seen.insert(*body));
    found
}

/// The types an impl's header instantiates: its self type and, for a trait
/// impl, the trait's arguments, in the impl's declaration coordinates.
fn header_types<'db>(db: &'db dyn HirAnalysisDb, item: ItemKind<'db>) -> Vec<TyId<'db>> {
    match item {
        ItemKind::Impl(impl_) => vec![impl_.ty(db)],
        ItemKind::ImplTrait(impl_trait) => {
            let mut tys = vec![impl_trait.ty(db)];
            if let Some(inst) = impl_trait.trait_inst(db) {
                tys.extend(inst.args(db).iter().copied());
            }
            tys
        }
        _ => Vec::new(),
    }
}

/// The header types a use of `func` with `args` instantiates: its impl's
/// header, or a trait method's `Self` and trait arguments.
fn callee_headers<'db>(
    db: &'db dyn HirAnalysisDb,
    func: Func<'db>,
    args: &[TyId<'db>],
) -> Vec<TyId<'db>> {
    match func.scope().parent_item(db) {
        Some(ItemKind::Trait(trait_)) => args
            .get(..trait_.params(db).len())
            .map(<[_]>::to_vec)
            .unwrap_or_default(),
        Some(item @ (ItemKind::Impl(_) | ItemKind::ImplTrait(_))) => {
            let Ok(subst) = CompleteSubst::for_owner(db, func.into(), args.to_vec()) else {
                return Vec::new();
            };
            header_types(db, item)
                .into_iter()
                .filter_map(|ty| substitute_complete(db, ty, &subst).ok())
                .collect()
        }
        _ => Vec::new(),
    }
}

/// The header types an associated constant path instantiates.
fn const_ref_headers<'db>(
    db: &'db dyn HirAnalysisDb,
    typed: &TypedBody<'db>,
    expr: ExprId,
) -> Option<Vec<TyId<'db>>> {
    match typed.expr_const_ref(expr)? {
        ConstRef::Const(_) => None,
        ConstRef::TraitConst(reference) => Some(reference.inst().args(db).to_vec()),
        ConstRef::InherentConst(reference) => Some(vec![reference.receiver_ty()]),
    }
}

/// A use of an associated item enters the header types it instantiates, such
/// as `Bounded<0>` in `Bounded<0>::helper()`, which no written type or checked
/// expression may carry. A header type that a checked receiver, argument,
/// result, or called function value already carries was checked where that
/// value entered; any other is checked here.
fn check_entered_header<'db>(
    db: &'db dyn HirAnalysisDb,
    typed: &TypedBody<'db>,
    expr: ExprId,
    headers: Vec<TyId<'db>>,
    scope: ScopeId<'db>,
) -> Option<crate::analysis::ty::diagnostics::TyDiagCollection<'db>> {
    let body = typed.body()?;
    let is_function =
        |ty: TyId<'db>| matches!(ty.base_ty(db).data(db), TyData::TyBase(TyBase::Func(_)));
    let mut carried = vec![expr];
    match expr.data(db, body).borrowed().to_opt() {
        Some(Expr::Call(callee, call_args)) => {
            if typed.expr_binding(*callee).is_some() {
                carried.push(*callee);
            }
            carried.extend(call_args.iter().map(|arg| arg.expr));
        }
        Some(Expr::MethodCall(receiver, _, _, call_args)) => {
            carried.push(*receiver);
            carried.extend(call_args.iter().map(|arg| arg.expr));
        }
        _ => {}
    }
    // A function value's own type is not checked (see `check_body_requirements`),
    // so it carries its header types only once bound and used again.
    let carried: Vec<_> = carried
        .into_iter()
        .map(|carrier| (carrier, typed.expr_ty(db, carrier)))
        .filter(|&(carrier, ty)| carrier != expr || !is_function(ty))
        .map(|(_, ty)| ty)
        .collect();
    headers
        .into_iter()
        .filter(|&header| !carried.iter().any(|&ty| ty_mentions(db, ty, header)))
        .find_map(|header| check_type_requirements(db, header, scope, expr.span(body).into(), &[]))
        .map(|(_, diag)| diag)
}

/// Whether `needle` occurs in `ty`.
fn ty_mentions<'db>(db: &'db dyn HirAnalysisDb, ty: TyId<'db>, needle: TyId<'db>) -> bool {
    ty == needle
        || ty
            .decompose_ty_app(db)
            .1
            .iter()
            .any(|&arg| ty_mentions(db, arg, needle))
}

/// Returns the first unmet requirement in `ty`, with the type application
/// that fails it. Failures of the applications in `reported` were already
/// reported where those types entered, so the search continues past them.
fn check_type_requirements<'db>(
    db: &'db dyn HirAnalysisDb,
    ty: TyId<'db>,
    scope: ScopeId<'db>,
    span: crate::span::DynLazySpan<'db>,
    reported: &[TyId<'db>],
) -> Option<(
    TyId<'db>,
    crate::analysis::ty::diagnostics::TyDiagCollection<'db>,
)> {
    let (base, args) = ty.decompose_ty_app(db);
    for &arg in args {
        if let Some(failure) = check_type_requirements(db, arg, scope, span.clone(), reported) {
            return Some(failure);
        }
    }
    if reported.contains(&ty) {
        return None;
    }
    let TyData::TyBase(TyBase::Adt(adt)) = base.data(db) else {
        return None;
    };
    use crate::analysis::ty::adt_def::AdtRef;
    let (declaration, generic_owner, kind) = match adt.adt_ref(db) {
        AdtRef::Struct(record) => (
            WhereClauseOwner::Struct(record),
            GenericParamOwner::Struct(record),
            "records",
        ),
        AdtRef::Enum(enum_) => (
            WhereClauseOwner::Enum(enum_),
            GenericParamOwner::Enum(enum_),
            "enums",
        ),
    };
    let predicates = declaration.where_clause(db).const_predicates(db);
    if predicates.is_empty() {
        return None;
    }
    if args.len() != collect_generic_params(db, generic_owner).params(db).len() {
        return Some((
            ty,
            crate::analysis::ty::diagnostics::TyLowerDiag::ConstRequirementNotSatisfied {
                primary: span,
                predicate: predicates[0].span().into(),
                reason: format!(
                    "partially applied {kind} with const requirements are not supported; \
                     supply all arguments"
                ),
            }
            .into(),
        ));
    }
    if args.iter().any(|arg| arg.has_var(db)) {
        return None;
    }
    let caller = premise_owner_in_scope(db, scope);
    for &predicate in predicates {
        let failures = discharge_requirement(db, declaration, predicate, args.to_vec(), caller);
        if !failures.is_empty() {
            return Some((
                ty,
                crate::analysis::ty::diagnostics::TyLowerDiag::ConstRequirementNotSatisfied {
                    primary: span,
                    predicate: predicate.span().into(),
                    reason: requirement_reason(failures),
                }
                .into(),
            ));
        }
    }
    None
}

fn requirement_reason(diags: &[FuncBodyDiag<'_>]) -> String {
    for diag in diags {
        match diag {
            FuncBodyDiag::Body(BodyDiag::RecursiveConstRequirement(_)) => {
                return "recursive const requirement cannot establish itself".into();
            }
            FuncBodyDiag::Body(BodyDiag::WhereConstPredicateFailed(_)) => {
                return "condition evaluated to `false`".into();
            }
            FuncBodyDiag::Body(BodyDiag::ConstRequirementNotSatisfied { reason, .. }) => {
                return reason.clone();
            }
            FuncBodyDiag::Ty(TyDiagCollection::Ty(diag)) => {
                let reason = match diag {
                    TyLowerDiag::ConstEvalDivisionByZero(_) => {
                        "constant evaluation encountered division by zero"
                    }
                    TyLowerDiag::ConstEvalArithmeticOverflow(_) => "constant evaluation overflowed",
                    TyLowerDiag::ConstEvalStepLimitExceeded(_) => {
                        "constant evaluation exceeded its step limit"
                    }
                    TyLowerDiag::ConstEvalRecursionLimitExceeded(_) => {
                        "constant evaluation exceeded its recursion limit"
                    }
                    TyLowerDiag::ConstEvalRecursiveConst(_) => "recursive constant evaluation",
                    _ => continue,
                };
                return reason.into();
            }
            _ => {}
        }
    }
    "condition could not be established; the predicate must be a well-formed, evaluable bool".into()
}

#[salsa::tracked(return_ref, cycle_initial=requirement_cycle_initial, cycle_fn=requirement_cycle_recover)]
fn discharge_requirement<'db>(
    db: &'db dyn HirAnalysisDb,
    declaration: WhereClauseOwner<'db>,
    predicate: Body<'db>,
    args: Vec<TyId<'db>>,
    caller: Option<WhereClauseOwner<'db>>,
) -> Vec<FuncBodyDiag<'db>> {
    let expected = TyId::bool(db);
    let (diags, typed) = check_predicate_formation(db, predicate);
    if has_recursive_requirement(diags) {
        return vec![BodyDiag::RecursiveConstRequirement(predicate.span().into()).into()];
    }
    if !diags.is_empty() && !static_assert_ignorable_type_diags(db, diags) {
        return diags.clone();
    }
    let args = match caller {
        Some(caller) => caller_args(db, caller, args),
        None => args,
    };
    let unsubstitutable = || {
        vec![
            BodyDiag::ConstRequirementNotSatisfied {
                primary: predicate.span().into(),
                predicate: predicate.span().into(),
                reason: "the condition could not be instantiated with these generic arguments"
                    .into(),
            }
            .into(),
        ]
    };
    let Some(subst) = requirement_subst(db, ItemKind::from(declaration).scope(), &args) else {
        return unsubstitutable();
    };
    let Ok(mut instantiated) = substitute_complete(db, typed.clone(), &subst) else {
        return unsubstitutable();
    };
    // TypedBody deliberately preserves formal TypeConst paths for runtime ABI
    // selection. Substitute these references only in this dependency view.
    for reference in instantiated.value_path_refs.values_mut().flatten() {
        if let ValuePathRef::TypeConst(ty) = reference {
            let Ok(substituted) = substitute_complete(db, *ty, &subst) else {
                return unsubstitutable();
            };
            *ty = substituted;
        }
    }
    let symbolic = predicate_flags(db, instantiated).contains(TyFlags::HAS_PARAM);
    if symbolic {
        let key = predicate_key(db, predicate, typed, predicate.expr(db), &subst);
        let caller_subst = caller.and_then(|caller| {
            let owner = GenericParamOwner::from_item_opt(caller.into())?;
            let identity = collect_generic_params(db, owner).params(db);
            requirement_subst(db, ItemKind::from(caller).scope(), identity)
                .map(|subst| (caller, subst))
        });
        if let (Some(key), Some((caller, caller_subst))) = (&key, &caller_subst) {
            for &premise in caller.where_clause(db).const_predicates(db) {
                let (diags, typed) = check_predicate_formation(db, premise);
                if (!diags.is_empty() && !static_assert_ignorable_type_diags(db, diags))
                    || predicate_key(db, premise, typed, premise.expr(db), caller_subst).as_ref()
                        != Some(key)
                {
                    continue;
                }
                return Vec::new();
            }
        }
        // An unused type parameter does not make a ground predicate unknown.
        if predicate_may_depend_on_params(db, predicate) {
            return vec![
                BodyDiag::ConstRequirementNotSatisfied {
                    primary: predicate.span().into(),
                    predicate: predicate.span().into(),
                    reason: if key.is_some() {
                        "no matching const requirement in the caller after substitution".into()
                    } else {
                        "symbolic forwarding of this expression is not supported; \
                         concrete evaluation is required"
                            .into()
                    },
                }
                .into(),
            ];
        }
    }
    let owner = BodyOwner::AnonConstBody {
        body: predicate,
        expected,
    };
    let outcome = eval_body_owner_const(db, owner, GenericSubst::for_body_owner(db, owner, args));
    const_predicate_outcome_diags(db, predicate, owner, outcome)
}

fn requirement_cycle_initial<'db>(
    _db: &'db dyn HirAnalysisDb,
    _declaration: WhereClauseOwner<'db>,
    predicate: Body<'db>,
    _args: Vec<TyId<'db>>,
    _caller: Option<WhereClauseOwner<'db>>,
) -> Vec<FuncBodyDiag<'db>> {
    vec![BodyDiag::RecursiveConstRequirement(predicate.span().into()).into()]
}

fn requirement_cycle_recover<'db>(
    _db: &'db dyn HirAnalysisDb,
    _value: &[FuncBodyDiag<'db>],
    _count: u32,
    _declaration: WhereClauseOwner<'db>,
    _predicate: Body<'db>,
    _args: Vec<TyId<'db>>,
    _caller: Option<WhereClauseOwner<'db>>,
) -> salsa::CycleRecoveryAction<Vec<FuncBodyDiag<'db>>> {
    salsa::CycleRecoveryAction::Iterate
}

// A recursive failure is absorbing. Canonicalize it to this query's own
// predicate instead of growing a diagnostic stack on each fixpoint iteration.
fn has_recursive_requirement(diags: &[FuncBodyDiag<'_>]) -> bool {
    diags.iter().any(|diag| {
        matches!(
            diag,
            FuncBodyDiag::Body(BodyDiag::RecursiveConstRequirement(_))
        )
    })
}

fn formation_cycle_initial<'db>(
    db: &'db dyn HirAnalysisDb,
    body: Body<'db>,
) -> (Vec<FuncBodyDiag<'db>>, TypedBody<'db>) {
    let typed = infer_body(
        db,
        BodyOwner::AnonConstBody {
            body,
            expected: TyId::bool(db),
        },
    )
    .1
    .clone();
    (
        vec![BodyDiag::RecursiveConstRequirement(body.span().into()).into()],
        typed,
    )
}
fn formation_cycle_recover<'db>(
    _db: &'db dyn HirAnalysisDb,
    _value: &(Vec<FuncBodyDiag<'db>>, TypedBody<'db>),
    _count: u32,
    _body: Body<'db>,
) -> salsa::CycleRecoveryAction<(Vec<FuncBodyDiag<'db>>, TypedBody<'db>)> {
    salsa::CycleRecoveryAction::Iterate
}

/// Check every authored type position, including unused defaults and aliases,
/// and the anonymous constants written directly in types and trait
/// references. Inferred expression types are checked separately after body
/// inference.
pub(crate) fn check_declared_type_requirements<'db>(
    db: &'db dyn HirAnalysisDb,
    top_mod: crate::hir_def::TopLevelMod<'db>,
) -> Vec<FuncBodyDiag<'db>> {
    use crate::hir_def::{TraitRefId, TypeKind};
    use crate::span::types::LazyTySpan;
    use crate::visitor::{
        Visitor, VisitorCtxt, prelude::LazyTraitRefSpan, walk_trait_ref, walk_type,
    };
    struct Checker<'db> {
        db: &'db dyn HirAnalysisDb,
        diags: Vec<FuncBodyDiag<'db>>,
        /// Failing type applications, in report order.
        reported: Vec<TyId<'db>>,
        /// Type parameter defaults, and how deep the walk is inside one.
        /// `check_generic_default_bodies` owns the anonymous constants there.
        defaults: FxHashSet<crate::hir_def::TypeId<'db>>,
        default_depth: usize,
    }
    fn assumptions_at<'db>(
        db: &'db dyn HirAnalysisDb,
        scope: ScopeId<'db>,
    ) -> PredicateListId<'db> {
        let mut enclosing = scope;
        while matches!(enclosing.item(), ItemKind::Body(_)) {
            let Some(parent) = enclosing.parent(db) else {
                break;
            };
            enclosing = parent;
        }
        crate::semantic::constraints_for(db, enclosing.item())
    }
    impl<'db> Checker<'db> {
        fn check_const_bodies(&mut self, lowered: &[TyId<'db>], bodies: &[Body<'db>]) {
            for (body, expected) in positioned_const_bodies(self.db, lowered, bodies) {
                self.diags
                    .extend(anon_const_requirements(self.db, body, expected));
            }
        }
    }
    impl<'db> Visitor<'db> for Checker<'db> {
        fn visit_generic_param(
            &mut self,
            ctxt: &mut VisitorCtxt<'db, crate::visitor::prelude::LazyGenericParamSpan<'db>>,
            param: &crate::hir_def::GenericParam<'db>,
        ) {
            if let crate::hir_def::GenericParam::Type(param) = param
                && let Some(default) = param.default_ty
            {
                self.defaults.insert(default);
            }
            crate::visitor::walk_generic_param(self, ctxt, param);
        }

        // The default owner checks the constants of a default type, not the
        // types written inside those constants' bodies.
        fn visit_body(
            &mut self,
            ctxt: &mut VisitorCtxt<'db, crate::visitor::prelude::LazyBodySpan<'db>>,
            body: Body<'db>,
        ) {
            let depth = std::mem::take(&mut self.default_depth);
            crate::visitor::walk_body(self, ctxt, body);
            self.default_depth = depth;
        }

        fn visit_trait_ref(
            &mut self,
            ctxt: &mut VisitorCtxt<'db, LazyTraitRefSpan<'db>>,
            trait_ref: TraitRefId<'db>,
        ) {
            if let Some(path) = trait_ref.path(self.db).to_opt()
                && self.default_depth == 0
            {
                let bodies = path_const_bodies(self.db, path);
                if !bodies.is_empty() {
                    let scope = ctxt.scope();
                    let assumptions = assumptions_at(self.db, scope);
                    let minter = LoweringContext::deferred(HoleAnchor::TemplatePath {
                        path,
                        scope,
                        assumptions,
                    });
                    if let Ok(res) =
                        resolve_path_with_minter(self.db, path, scope, assumptions, false, &minter)
                    {
                        let lowered = path_res_tys(self.db, res);
                        self.check_const_bodies(&lowered, &bodies);
                    }
                }
            }
            walk_trait_ref(self, ctxt, trait_ref);
        }

        fn visit_ty(
            &mut self,
            ctxt: &mut VisitorCtxt<'db, LazyTySpan<'db>>,
            hir_ty: crate::hir_def::TypeId<'db>,
        ) {
            let scope = ctxt.scope();
            let assumptions = assumptions_at(self.db, scope);
            let bodies = match hir_ty.data(self.db) {
                TypeKind::Array(_, Partial::Present(len)) => vec![*len],
                TypeKind::Path(Partial::Present(path)) => path_const_bodies(self.db, *path),
                _ => Vec::new(),
            };
            let in_default = self.defaults.contains(&hir_ty);
            self.default_depth += usize::from(in_default);
            if !bodies.is_empty() && self.default_depth == 0 {
                let lowered = lower_hir_ty_deferred(self.db, hir_ty, scope, assumptions);
                self.check_const_bodies(&[lowered], &bodies);
            }
            let ty = lower_hir_ty(self.db, hir_ty, scope, assumptions);
            let span = ctxt.span();
            // Nested authored types are checked first. A failure one of them
            // reported entered there, so the enclosing type does not repeat it;
            // a failure reachable only through an alias expansion is reported here.
            let nested = self.reported.len();
            walk_type(self, ctxt, hir_ty);
            if !ty.has_invalid(self.db)
                && let Some(span) = span
                && let Some((failing, diag)) = check_type_requirements(
                    self.db,
                    ty,
                    scope,
                    span.into(),
                    &self.reported[nested..],
                )
            {
                self.reported.push(failing);
                self.diags.push(diag.into());
            }
            self.default_depth -= usize::from(in_default);
        }
    }
    let mut checker = Checker {
        db,
        diags: Vec::new(),
        reported: Vec::new(),
        defaults: FxHashSet::default(),
        default_depth: 0,
    };
    let mut ctxt = VisitorCtxt::new(db, top_mod.scope(), top_mod.span());
    checker.visit_top_mod(&mut ctxt, top_mod);
    checker.diags
}
