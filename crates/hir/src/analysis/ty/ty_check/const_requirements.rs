//! Const declaration requirements are discharged after inference, for every body
//! owner. Concrete discharge uses ordinary CTFE. Symbolic forwarding compares
//! resolved, typed expressions after substitution, without evaluating
//! unknown parameters or assuming the obligation being checked.
use super::*;
use crate::analysis::name_resolution::{
    method_selection::{MethodCandidate, select_method_candidate},
    resolve_path_with_minter,
};
use crate::analysis::ty::canonical::Canonicalized;
use crate::analysis::ty::{
    subst::substitute_complete,
    ty_lower::{CompleteSubst, lower_type_alias},
};
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
    caller: GenericParamOwner<'db>,
    args: Vec<TyId<'db>>,
) -> Vec<TyId<'db>> {
    let identity = collect_generic_params(db, caller).params(db).to_vec();
    CompleteSubst::for_owner(db, caller, identity)
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

/// Why a const requirement does not hold at a use. Discharge decides it, and
/// the use's diagnostic renders it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Update)]
pub enum RequirementFailure {
    /// The condition evaluated to `false`.
    False,
    /// The requirement depends on itself.
    Recursive,
    /// Compile-time evaluation of the condition stopped.
    Evaluation(EvaluationStop),
    /// The condition is ill-formed, or its evaluation gave no `bool`.
    NotEstablished,
    /// The condition could not be instantiated with the use's arguments.
    NotInstantiable,
    /// A generic use states no identical condition.
    NoMatchingPremise,
    /// A generic use cannot forward the condition's expression.
    NotForwardable,
    /// A record with conditions that is not fully applied.
    PartiallyAppliedRecord,
    /// An enum with conditions that is not fully applied.
    PartiallyAppliedEnum,
}

/// Why compile-time evaluation of a condition stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Update)]
pub enum EvaluationStop {
    DivisionByZero,
    Overflow,
    StepLimit,
    RecursionLimit,
    RecursiveConst,
}

impl RequirementFailure {
    fn from_evaluation(cause: &InvalidCause<'_>) -> Self {
        Self::Evaluation(match cause {
            InvalidCause::ConstEvalDivisionByZero { .. } => EvaluationStop::DivisionByZero,
            InvalidCause::ConstEvalArithmeticOverflow { .. } => EvaluationStop::Overflow,
            InvalidCause::ConstEvalStepLimitExceeded { .. } => EvaluationStop::StepLimit,
            InvalidCause::ConstEvalRecursionLimitExceeded { .. } => EvaluationStop::RecursionLimit,
            InvalidCause::ConstEvalRecursiveConst { .. } => EvaluationStop::RecursiveConst,
            _ => return Self::NotEstablished,
        })
    }
}

/// Whether a use's const requirement holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Update)]
pub(super) enum Discharge {
    Holds,
    Fails(RequirementFailure),
}

/// Whether a predicate can be discharged: it type checks as a `bool` const
/// body and does not depend on itself.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Update)]
pub(super) enum FormationStatus {
    WellFormed,
    IllFormed,
    Recursive,
}

#[derive(Debug, Clone, PartialEq, Eq, Update)]
pub(super) struct PredicateFormation<'db> {
    /// The predicate's own diagnostics, reported at its declaration.
    pub(super) diags: Vec<FuncBodyDiag<'db>>,
    pub(super) typed: TypedBody<'db>,
    pub(super) status: FormationStatus,
}

impl PredicateFormation<'_> {
    pub(super) fn is_well_formed(&self) -> bool {
        self.status == FormationStatus::WellFormed
    }
}

/// A body's requirement check: its diagnostics, and whether a requirement it
/// uses depends on itself, which makes a predicate's formation recursive.
#[derive(Default)]
pub(super) struct RequirementCheck<'db> {
    pub(super) diags: Vec<FuncBodyDiag<'db>>,
    recursive: bool,
}

impl<'db> RequirementCheck<'db> {
    fn unmet(
        &mut self,
        primary: DynLazySpan<'db>,
        predicate: Body<'db>,
        failure: RequirementFailure,
    ) {
        self.recursive |= failure == RequirementFailure::Recursive;
        self.diags
            .push(unmet_requirement_diag(primary, predicate, failure));
    }

    fn extend(&mut self, other: Self) {
        self.recursive |= other.recursive;
        self.diags.extend(other.diags);
    }
}

fn unmet_requirement_diag<'db>(
    primary: DynLazySpan<'db>,
    predicate: Body<'db>,
    failure: RequirementFailure,
) -> FuncBodyDiag<'db> {
    BodyDiag::ConstRequirementNotSatisfied {
        primary,
        predicate: predicate.span().into(),
        reason: failure,
    }
    .into()
}

#[salsa::tracked(return_ref, cycle_initial=formation_cycle_initial, cycle_fn=formation_cycle_recover)]
pub(super) fn check_predicate_formation<'db>(
    db: &'db dyn HirAnalysisDb,
    body: Body<'db>,
) -> PredicateFormation<'db> {
    let owner = BodyOwner::AnonConstBody {
        body,
        expected: TyId::bool(db),
    };
    let (mut diags, typed) = infer_body(db, owner).clone();
    // A condition the parser could not read has no expression to check or
    // evaluate. The parser reported it, at its position.
    if matches!(body.expr(db).data(db, body), Partial::Absent) {
        return PredicateFormation {
            diags,
            typed,
            status: FormationStatus::IllFormed,
        };
    }
    let mut recursive = false;
    if diags.is_empty() || static_assert_ignorable_type_diags(db, &diags) {
        diags.extend(
            crate::analysis::ty::const_check::check_const_body_expressions(db, body, &typed),
        );
        let requirements = check_body_requirements(db, owner, &typed);
        recursive = requirements.recursive;
        diags.extend(requirements.diags);
    }
    // A recursive failure is absorbing, and is reported once, at this predicate.
    let status = if recursive {
        diags = vec![BodyDiag::RecursiveConstRequirement(body.span().into()).into()];
        FormationStatus::Recursive
    } else if diags.is_empty() || static_assert_ignorable_type_diags(db, &diags) {
        FormationStatus::WellFormed
    } else {
        FormationStatus::IllFormed
    };
    PredicateFormation {
        diags,
        typed,
        status,
    }
}

/// Whether uses of a declaration check `predicate`. A ground predicate, one
/// that mentions no generic parameter, is checked once where it is declared
/// (`check_where_const_predicates`), whether or not anything uses the item,
/// so a use does not report it again.
fn checked_at_uses<'db>(db: &'db dyn HirAnalysisDb, predicate: Body<'db>) -> bool {
    predicate_may_depend_on_params(db, predicate)
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
) -> Option<GenericParamOwner<'db>> {
    if !matches!(owner, BodyOwner::Func(_) | BodyOwner::AnonConstBody { .. }) {
        return None;
    }
    premise_owner_in_scope(db, owner.scope())
}

/// The declaration whose generic parameters the premises at `origin` are
/// stated over (see `caller_premises`).
fn premise_owner_in_scope<'db>(
    db: &'db dyn HirAnalysisDb,
    origin: ScopeId<'db>,
) -> Option<GenericParamOwner<'db>> {
    let mut current = Some(origin);
    while let Some(scope) = current {
        match scope.item() {
            // A trait method can neither state conditions nor assume a header's.
            ItemKind::Func(func)
                if matches!(func.scope().parent_item(db), Some(ItemKind::Trait(_))) =>
            {
                return None;
            }
            item @ (ItemKind::Func(_) | ItemKind::Struct(_) | ItemKind::Enum(_)) => {
                let candidate = WhereClauseOwner::from_item_opt(item)?;
                if candidate
                    .where_clause(db)
                    .const_predicates(db)
                    .iter()
                    .any(|predicate| origin.is_transitive_child_of(db, predicate.scope()))
                {
                    return None;
                }
                return GenericParamOwner::from_item_opt(item);
            }
            item @ (ItemKind::Impl(_) | ItemKind::ImplTrait(_) | ItemKind::TypeAlias(_)) => {
                return GenericParamOwner::from_item_opt(item);
            }
            ItemKind::Body(_) => current = scope.parent(db),
            // A nested declaration does not inherit function premises.
            _ => return None,
        }
    }
    None
}

/// The conditions that hold where premises are stated over `caller`'s
/// generic parameters, each with the substitution that states it there:
/// - the caller's own conditions, when its uses check them;
/// - inside an impl, the conditions of the records and enums in its header,
///   since every use of the impl instantiates the header with checked types
///   (`check_entered_header`, and the receiver and argument types);
/// - in a type alias, the conditions of the aliased type, since every
///   application of the alias is checked after expansion.
///
/// An impl's own `where` conditions are not premises: they are rejected at
/// the impl, and no use checks them.
fn caller_premises<'db>(
    db: &'db dyn HirAnalysisDb,
    caller: GenericParamOwner<'db>,
) -> Vec<(Body<'db>, CompleteSubst<'db>)> {
    let item = ItemKind::from(caller);
    let mut premises = Vec::new();
    let own = match caller {
        GenericParamOwner::Func(func) => function_requirements_supported(db, func),
        GenericParamOwner::Struct(_) | GenericParamOwner::Enum(_) => true,
        _ => false,
    };
    if own
        && let Some(owner) = WhereClauseOwner::from_item_opt(item)
        && let Some(subst) = requirement_subst(
            db,
            item.scope(),
            collect_generic_params(db, caller).params(db),
        )
    {
        premises.extend(
            owner
                .where_clause(db)
                .const_predicates(db)
                .iter()
                .map(|&predicate| (predicate, subst.clone())),
        );
    }
    let implied = match caller {
        GenericParamOwner::Func(func) => func
            .scope()
            .parent_item(db)
            .map(|parent| header_types(db, parent))
            .unwrap_or_default(),
        GenericParamOwner::Impl(_) | GenericParamOwner::ImplTrait(_) => header_types(db, item),
        GenericParamOwner::TypeAlias(alias) => {
            vec![lower_type_alias(db, alias).alias_to.instantiate_identity()]
        }
        _ => Vec::new(),
    };
    for ty in caller_args(db, caller, implied) {
        type_conditions(db, ty, &mut premises);
    }
    premises
}

/// The conditions of each fully applied record or enum in `ty`, instantiated
/// with its arguments.
fn type_conditions<'db>(
    db: &'db dyn HirAnalysisDb,
    ty: TyId<'db>,
    conditions: &mut Vec<(Body<'db>, CompleteSubst<'db>)>,
) {
    use crate::analysis::ty::adt_def::AdtRef;
    let (base, args) = ty.decompose_ty_app(db);
    for &arg in args {
        type_conditions(db, arg, conditions);
    }
    let TyData::TyBase(TyBase::Adt(adt)) = base.data(db) else {
        return;
    };
    let declaration = match adt.adt_ref(db) {
        AdtRef::Struct(record) => WhereClauseOwner::Struct(record),
        AdtRef::Enum(enum_) => WhereClauseOwner::Enum(enum_),
    };
    let predicates = declaration.where_clause(db).const_predicates(db);
    if predicates.is_empty() {
        return;
    }
    let Some(subst) = requirement_subst(db, ItemKind::from(declaration).scope(), args) else {
        return;
    };
    conditions.extend(
        predicates
            .iter()
            .map(|&predicate| (predicate, subst.clone())),
    );
}

pub(super) fn check_body_requirements<'db>(
    db: &'db dyn HirAnalysisDb,
    owner: BodyOwner<'db>,
    typed: &TypedBody<'db>,
) -> RequirementCheck<'db> {
    let mut check = RequirementCheck::default();
    let Some(body) = typed.body() else {
        return check;
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
    let written = written_type_applications(db, owner);
    // Requirements are checked where a type enters the body: an authored type
    // (checked by `check_declared_type_requirements`) or an instantiating
    // expression. An expression does not report an application that a
    // written type carries, since that type reports it. A pattern, a binding use, or a block or branch only carries a
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
            && let Some(unmet) = check_type_requirements(db, ty, owner.scope(), &written)
        {
            check.unmet(expr.span(body).into(), unmet.predicate, unmet.failure);
        }
        if let Some(headers) = const_ref_headers(db, typed, expr)
            && let Some(unmet) =
                check_entered_header(db, typed, expr, headers, owner.scope(), &written)
        {
            check.unmet(expr.span(body).into(), unmet.predicate, unmet.failure);
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
                    && let Some(unmet) = check_type_requirements(
                        db,
                        definition.ret_ty(db).instantiate(db, args),
                        owner.scope(),
                        &written,
                    )
                {
                    check.unmet(expr.span(body).into(), unmet.predicate, unmet.failure);
                }
                continue;
            }
        };
        if let Some(unmet) = check_entered_header(
            db,
            typed,
            expr,
            callee_headers(db, func, args),
            owner.scope(),
            &written,
        ) {
            check.unmet(expr.span(body).into(), unmet.predicate, unmet.failure);
        }
        // Unsupported associated or generic owner contexts are rejected at
        // their declarations.
        let predicates = WhereClauseOwner::Func(func)
            .where_clause(db)
            .const_predicates(db);
        if predicates.is_empty() || !function_requirements_supported(db, func) {
            continue;
        }
        let caller = caller.filter(|_| args.iter().any(|ty| ty.has_param(db)));
        for &predicate in predicates
            .iter()
            .filter(|&&predicate| checked_at_uses(db, predicate))
        {
            if let Discharge::Fails(failure) = discharge_requirement(
                db,
                WhereClauseOwner::Func(func),
                predicate,
                args.to_vec(),
                caller,
            ) {
                check.unmet(expr.span(body).into(), predicate, failure);
            }
        }
    }
    for (nested, expected) in expression_const_bodies(db, body, typed) {
        check.extend(anon_const_position_check(db, nested, expected));
    }
    check
}

/// What an anonymous constant checked against `expected` reports at its
/// position: its unmet requirements, or the inference failures that its
/// type's error cannot show. Type lowering evaluates such a constant but does
/// not check its requirements, since checking them evaluates code, which can
/// lower the type the constant is part of. So the constant's position owns
/// them, and its owner calls this: `check_body_requirements` for expression
/// positions and `check_declared_type_requirements` for types.
fn anon_const_position_check<'db>(
    db: &'db dyn HirAnalysisDb,
    body: Body<'db>,
    expected: TyId<'db>,
) -> RequirementCheck<'db> {
    use crate::analysis::ty::const_ty::{ConstBodyFailure, const_body_failure};
    let owner = BodyOwner::AnonConstBody { body, expected };
    let (diags, typed) = infer_body(db, owner);
    if diags.is_empty() || static_assert_ignorable_type_diags(db, diags) {
        return check_body_requirements(db, owner, typed);
    }
    // A lone path to a constant is read as that constant, not checked as a
    // body, so its failures are reported where the constant is.
    let position = body.scope().parent(db).unwrap_or(body.scope());
    let mut check = RequirementCheck::default();
    if !crate::analysis::ty::ty_lower::const_body_names_a_constant(
        db,
        body,
        position,
        typed.assumptions(),
    ) && let ConstBodyFailure::AtPosition(_) = const_body_failure(db, body, diags, typed)
    {
        check.diags = diags.clone();
    }
    check
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
    // The callable an expression calls or names, with the generic arguments
    // inference gave it. A call whose explicit arguments failed has no
    // checked callable, so its definition is found again from its path or
    // receiver, without arguments.
    let callee = |expr: ExprId| -> Option<(CallableDef<'db>, Vec<TyId<'db>>)> {
        if let Some(callable) = typed.callable_expr(expr) {
            return Some((callable.callable_def(), callable.generic_args().to_vec()));
        }
        let (base, generic_args) = typed.expr_ty(db, expr).decompose_ty_app(db);
        if let TyData::TyBase(TyBase::Func(definition)) = base.data(db) {
            return Some((*definition, generic_args.to_vec()));
        }
        let candidate_def = |candidate: MethodCandidate<'db>| match candidate {
            MethodCandidate::InherentMethod(method) => method.def,
            MethodCandidate::TraitMethod(method) | MethodCandidate::NeedsConfirmation(method) => {
                CallableDef::Func(method.method)
            }
        };
        let definition = match expr.data(db, body).borrowed().to_opt()? {
            Expr::Path(Partial::Present(path)) => {
                let minter = LoweringContext::deferred(HoleAnchor::TemplatePath {
                    path: *path,
                    scope: body.scope(),
                    assumptions: typed.assumptions(),
                });
                match resolve_path_with_minter(
                    db,
                    *path,
                    body.scope(),
                    typed.assumptions(),
                    true,
                    &minter,
                )
                .ok()?
                {
                    PathRes::Func(ty) => match ty.base_ty(db).data(db) {
                        TyData::TyBase(TyBase::Func(definition)) => *definition,
                        _ => return None,
                    },
                    PathRes::Method(_, candidate) => candidate_def(candidate),
                    PathRes::TraitMethod(_, method) => CallableDef::Func(method),
                    _ => return None,
                }
            }
            Expr::MethodCall(receiver, Partial::Present(name), _, _) => {
                let receiver_ty = typed.expr_ty(db, *receiver);
                if receiver_ty.has_invalid(db) {
                    return None;
                }
                candidate_def(
                    select_method_candidate(
                        db,
                        &Canonicalized::new(db, receiver_ty),
                        *name,
                        body.scope(),
                        typed.assumptions(),
                        None,
                    )
                    .ok()?,
                )
            }
            _ => return None,
        };
        Some((definition, Vec::new()))
    };
    // Explicit arguments of a callable's own parameters, which the type
    // checker applies after the path resolves. Each is checked against its
    // parameter's type: the one its checked argument carries, or, for an
    // argument that failed, the parameter's declared type.
    let explicit_args = |expr: ExprId, args: GenericArgListId<'db>| {
        let Some((definition, generic_args)) = callee(expr) else {
            return Vec::new();
        };
        let offset = definition.offset_to_explicit_params_position(db);
        let expected = |index: usize| {
            generic_args
                .get(index)
                .and_then(|&arg| checked_const_ty(arg))
                .or_else(|| {
                    let CallableDef::Func(func) = definition else {
                        return None;
                    };
                    collect_generic_params(db, func.into())
                        .params(db)
                        .get(index)?
                        .const_ty_ty(db)
                })
        };
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
                Some((body, expected(offset + idx)?))
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
            // A length is checked against the array's length parameter type.
            Some(Expr::ArrayRep(_, len)) => len
                .to_opt()
                .and_then(|len| {
                    let expected = TyId::array(db, TyId::unit(db))
                        .applicable_ty(db)?
                        .const_ty?;
                    Some(vec![(len, expected)])
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
    written: &[TyId<'db>],
) -> Option<TypeRequirementFailure<'db>> {
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
        .find_map(|header| check_type_requirements(db, header, scope, written))
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

/// An unmet requirement of a type application.
pub(super) struct TypeRequirementFailure<'db> {
    /// The failing application.
    ty: TyId<'db>,
    predicate: Body<'db>,
    failure: RequirementFailure,
}

/// Returns the first unmet requirement in `ty`. Failures of the
/// applications in `reported` were already reported where those types
/// entered, so the search continues past them.
fn check_type_requirements<'db>(
    db: &'db dyn HirAnalysisDb,
    ty: TyId<'db>,
    scope: ScopeId<'db>,
    reported: &[TyId<'db>],
) -> Option<TypeRequirementFailure<'db>> {
    let (base, args) = ty.decompose_ty_app(db);
    for &arg in args {
        if let Some(failure) = check_type_requirements(db, arg, scope, reported) {
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
    let (declaration, generic_owner, partially_applied) = match adt.adt_ref(db) {
        AdtRef::Struct(record) => (
            WhereClauseOwner::Struct(record),
            GenericParamOwner::Struct(record),
            RequirementFailure::PartiallyAppliedRecord,
        ),
        AdtRef::Enum(enum_) => (
            WhereClauseOwner::Enum(enum_),
            GenericParamOwner::Enum(enum_),
            RequirementFailure::PartiallyAppliedEnum,
        ),
    };
    let predicates = declaration.where_clause(db).const_predicates(db);
    if predicates.is_empty() {
        return None;
    }
    if args.len() != collect_generic_params(db, generic_owner).params(db).len() {
        return Some(TypeRequirementFailure {
            ty,
            predicate: predicates[0],
            failure: partially_applied,
        });
    }
    if args.iter().any(|arg| arg.has_var(db)) {
        return None;
    }
    let caller = premise_owner_in_scope(db, scope);
    predicates
        .iter()
        .filter(|&&predicate| checked_at_uses(db, predicate))
        .find_map(|&predicate| {
            match discharge_requirement(db, declaration, predicate, args.to_vec(), caller) {
                Discharge::Holds => None,
                Discharge::Fails(failure) => Some(TypeRequirementFailure {
                    ty,
                    predicate,
                    failure,
                }),
            }
        })
}

#[salsa::tracked(cycle_initial=requirement_cycle_initial, cycle_fn=requirement_cycle_recover)]
fn discharge_requirement<'db>(
    db: &'db dyn HirAnalysisDb,
    declaration: WhereClauseOwner<'db>,
    predicate: Body<'db>,
    args: Vec<TyId<'db>>,
    caller: Option<GenericParamOwner<'db>>,
) -> Discharge {
    let formation = check_predicate_formation(db, predicate);
    match formation.status {
        FormationStatus::WellFormed => {}
        FormationStatus::IllFormed => {
            return Discharge::Fails(RequirementFailure::NotEstablished);
        }
        FormationStatus::Recursive => return Discharge::Fails(RequirementFailure::Recursive),
    }
    let not_instantiable = Discharge::Fails(RequirementFailure::NotInstantiable);
    let args = match caller {
        Some(caller) => caller_args(db, caller, args),
        None => args,
    };
    let Some(subst) = requirement_subst(db, ItemKind::from(declaration).scope(), &args) else {
        return not_instantiable;
    };
    let Ok(mut instantiated) = substitute_complete(db, formation.typed.clone(), &subst) else {
        return not_instantiable;
    };
    // TypedBody deliberately preserves formal TypeConst paths for runtime ABI
    // selection. Substitute these references only in this dependency view.
    for reference in instantiated.value_path_refs.values_mut().flatten() {
        if let ValuePathRef::TypeConst(ty) = reference {
            let Ok(substituted) = substitute_complete(db, *ty, &subst) else {
                return not_instantiable;
            };
            *ty = substituted;
        }
    }
    let symbolic = predicate_flags(db, instantiated).contains(TyFlags::HAS_PARAM);
    if symbolic {
        let key = predicate_key(db, predicate, &formation.typed, predicate.expr(db), &subst);
        if let (Some(key), Some(caller)) = (&key, caller) {
            for (premise, premise_subst) in caller_premises(db, caller) {
                let premise_formation = check_predicate_formation(db, premise);
                if premise_formation.is_well_formed()
                    && predicate_key(
                        db,
                        premise,
                        &premise_formation.typed,
                        premise.expr(db),
                        &premise_subst,
                    )
                    .as_ref()
                        == Some(key)
                {
                    return Discharge::Holds;
                }
            }
        }
        // An unused type parameter does not make a ground predicate unknown.
        if predicate_may_depend_on_params(db, predicate) {
            return Discharge::Fails(if key.is_some() {
                RequirementFailure::NoMatchingPremise
            } else {
                RequirementFailure::NotForwardable
            });
        }
    }
    let owner = BodyOwner::AnonConstBody {
        body: predicate,
        expected: TyId::bool(db),
    };
    match condition_outcome(db, owner, GenericSubst::for_body_owner(db, owner, args)) {
        ConditionOutcome::True => Discharge::Holds,
        ConditionOutcome::False => Discharge::Fails(RequirementFailure::False),
        ConditionOutcome::NotBool | ConditionOutcome::Blocked(_) => {
            Discharge::Fails(RequirementFailure::NotEstablished)
        }
        ConditionOutcome::Failed(cause) => {
            Discharge::Fails(RequirementFailure::from_evaluation(&cause))
        }
    }
}

// A requirement reached again while it is being discharged depends on itself.
fn requirement_cycle_initial<'db>(
    _db: &'db dyn HirAnalysisDb,
    _declaration: WhereClauseOwner<'db>,
    _predicate: Body<'db>,
    _args: Vec<TyId<'db>>,
    _caller: Option<GenericParamOwner<'db>>,
) -> Discharge {
    Discharge::Fails(RequirementFailure::Recursive)
}

fn requirement_cycle_recover<'db>(
    _db: &'db dyn HirAnalysisDb,
    _value: &Discharge,
    _count: u32,
    _declaration: WhereClauseOwner<'db>,
    _predicate: Body<'db>,
    _args: Vec<TyId<'db>>,
    _caller: Option<GenericParamOwner<'db>>,
) -> salsa::CycleRecoveryAction<Discharge> {
    salsa::CycleRecoveryAction::Iterate
}

fn formation_cycle_initial<'db>(
    db: &'db dyn HirAnalysisDb,
    body: Body<'db>,
) -> PredicateFormation<'db> {
    let typed = infer_body(
        db,
        BodyOwner::AnonConstBody {
            body,
            expected: TyId::bool(db),
        },
    )
    .1
    .clone();
    PredicateFormation {
        diags: vec![BodyDiag::RecursiveConstRequirement(body.span().into()).into()],
        typed,
        status: FormationStatus::Recursive,
    }
}
fn formation_cycle_recover<'db>(
    _db: &'db dyn HirAnalysisDb,
    _value: &PredicateFormation<'db>,
    _count: u32,
    _body: Body<'db>,
) -> salsa::CycleRecoveryAction<PredicateFormation<'db>> {
    salsa::CycleRecoveryAction::Iterate
}

/// The assumptions a type written at `scope` is lowered with.
fn assumptions_at<'db>(db: &'db dyn HirAnalysisDb, scope: ScopeId<'db>) -> PredicateListId<'db> {
    let mut enclosing = scope;
    while matches!(enclosing.item(), ItemKind::Body(_)) {
        let Some(parent) = enclosing.parent(db) else {
            break;
        };
        enclosing = parent;
    }
    crate::semantic::constraints_for(db, enclosing.item())
}

/// The type applications in the types written in `owner`'s signature and
/// body, lowered as `check_declared_type_requirements` lowers them. That check
/// reports a written type's unmet requirement where the type is written, so
/// an expression that carries the same application does not report it again.
/// A `where` clause is left out: its conditions are checked in their own
/// context.
fn written_type_applications<'db>(
    db: &'db dyn HirAnalysisDb,
    owner: BodyOwner<'db>,
) -> Vec<TyId<'db>> {
    use crate::hir_def::{TypeId, WhereClauseId};
    use crate::visitor::{
        Visitor, VisitorCtxt,
        prelude::{LazyTySpan, LazyWhereClauseSpan},
        walk_type,
    };
    struct Collector<'db> {
        db: &'db dyn HirAnalysisDb,
        applications: FxHashSet<TyId<'db>>,
    }
    impl<'db> Collector<'db> {
        fn collect(&mut self, ty: TyId<'db>) {
            if self.applications.insert(ty) {
                for &arg in ty.decompose_ty_app(self.db).1 {
                    self.collect(arg);
                }
            }
        }
    }
    impl<'db> Visitor<'db> for Collector<'db> {
        fn visit_where_clause(
            &mut self,
            _: &mut VisitorCtxt<'db, LazyWhereClauseSpan<'db>>,
            _: WhereClauseId<'db>,
        ) {
        }

        fn visit_ty(&mut self, ctxt: &mut VisitorCtxt<'db, LazyTySpan<'db>>, hir_ty: TypeId<'db>) {
            let scope = ctxt.scope();
            let ty = lower_hir_ty(self.db, hir_ty, scope, assumptions_at(self.db, scope));
            // The same types `check_declared_type_requirements` checks.
            if !ty.has_invalid(self.db) && ctxt.span().is_some() {
                self.collect(ty);
            }
            walk_type(self, ctxt, hir_ty);
        }
    }
    let mut collector = Collector {
        db,
        applications: FxHashSet::default(),
    };
    let item = match owner {
        BodyOwner::Func(func) => Some(ItemKind::Func(func)),
        BodyOwner::Const(const_) => Some(ItemKind::Const(const_)),
        _ => None,
    };
    if let Some(item) = item {
        collector.visit_item(&mut VisitorCtxt::with_item(db, item), item);
    } else if let Some(body) = owner.body(db) {
        collector.visit_body(&mut VisitorCtxt::with_body(db, body), body);
    }
    collector.applications.into_iter().collect()
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
    impl<'db> Checker<'db> {
        fn check_const_bodies(&mut self, lowered: &[TyId<'db>], bodies: &[Body<'db>]) {
            for (body, expected) in positioned_const_bodies(self.db, lowered, bodies) {
                self.diags
                    .extend(anon_const_position_check(self.db, body, expected).diags);
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
                && let Some(unmet) =
                    check_type_requirements(self.db, ty, scope, &self.reported[nested..])
            {
                self.reported.push(unmet.ty);
                self.diags.push(unmet_requirement_diag(
                    span.into(),
                    unmet.predicate,
                    unmet.failure,
                ));
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
