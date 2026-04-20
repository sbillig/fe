use std::borrow::Cow;

use common::indexmap::IndexSet;
use cranelift_entity::EntityRef;
use hir::analysis::{
    semantic::{
        FieldIndex, GenericSubst, ImplEnv, NEffectArg, NEffectArgValue, SConst, SLocalId,
        SemanticCalleeRef, SemanticInstance, SemanticInstanceKey, ValueProvenance, VariantIndex,
        borrowck::{
            NBorrowRoot, NExpr, NLocalInterface, NLocalOrigin, NOperand, NSPlace, NSPlaceRoot,
            NormalizedBindingLowering, NormalizedSemanticBody,
        },
        get_or_build_semantic_instance, sem_const_ty, semantic_binding_lowering,
        semantic_binding_ty, semantic_instance_assumptions,
    },
    ty::{
        ProviderKind,
        corelib::runtime_builtin_func_kind,
        normalize::normalize_ty,
        provider::registered_root_providers,
        trait_def::{TraitInstId, resolve_trait_method_instance},
        trait_resolution::{PredicateListId, TraitSolveCx},
        ty_check::{BodyOwner, EffectParamSite, EffectPassMode, LocalBinding, ParamSite},
        ty_def::{TyData, TyId, strip_derived_adt_layout_args},
    },
};
use hir::hir_def::ArithBinOp;
use hir::projection::Projection;
use hir::semantic::ProviderBinding;
use rustc_hash::FxHashMap;
use salsa::Update;

use crate::{
    db::MirDb,
    instance::{RuntimeInstanceKey, RuntimeInstanceSource},
    runtime::{
        AddressSpaceKind, BorrowAccess, Layout, LayoutId, RefKind, RefView, RuntimeBoundarySpec,
        RuntimeCarrier, RuntimeClass, RuntimeCodeRegion, RuntimeCodeRegionKey, RuntimeParam,
        RuntimeParamPlan, RuntimeSignature, SaturatingBinOp, ScalarClass, ScalarRepr, ScalarRole,
        VariantId,
    },
};

use super::{
    call_input::{
        CompiledCallInputPlan, CompiledMaterializationPlan, RuntimeValueEvaluator,
        compile_call_input_plan_for_semantic, compile_value_pass_plan,
    },
    infer::{fallback_root_transport_class, local_place_root_class},
    interface::{runtime_param_locals, runtime_visible_binding_plans},
    layout::{
        layout_for_aggregate_instance_in_context, layout_for_enum_variant_instance_in_context,
        layout_for_ty_in_context,
    },
    place::{
        address_space_from_provider, project_field_class, project_index_class,
        project_variant_field_class,
    },
    returns::RuntimeReturnAnalysisCx,
    type_info::{
        RuntimeTypeEnv, aggregate_transport_depends_on_runtime_source,
        boundary_source_uses_transport_sensitive_aggregate, boundary_spec_for_ty_in_env,
        default_borrow_transport_set, provider_address_space_to_runtime,
        provider_class_for_target_in_context, provider_class_for_target_in_env,
        runtime_repr_ty_in_context, scalar_class_for_ty_in_env, stored_class_for_ty_in_context,
        top_level_class_for_ty_in_context, top_level_class_for_ty_in_env,
    },
};

#[derive(Clone)]
pub(crate) struct BodyStaticFacts<'db> {
    local_facts: Vec<LocalStaticFacts<'db>>,
    assignments: Vec<AssignStaticFacts<'db>>,
    stmt_assignments: Vec<Vec<Option<usize>>>,
    assignments_by_local: Vec<Vec<usize>>,
    dynamic_dependents_by_local: Vec<Vec<SLocalId>>,
    root_provider_locals: FxHashMap<ProviderBinding<'db>, SLocalId>,
}

#[derive(Clone)]
pub(super) struct AssignStaticFacts<'db> {
    pub(super) block_idx: usize,
    pub(super) stmt_idx: usize,
    pub(super) dst: SLocalId,
    uses: Box<[SLocalId]>,
    expr: Option<ExprStaticFacts<'db>>,
}

impl<'db> AssignStaticFacts<'db> {
    pub(crate) fn uses(&self) -> &[SLocalId] {
        &self.uses
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct BoundarySiteId(u32);

#[derive(Clone, Debug)]
pub(super) struct StagedBoundary<'db> {
    site: BoundarySiteId,
    pub(super) boundary: RuntimeBoundarySpec<'db>,
    matcher: CompiledBoundaryMatcher<'db>,
}

#[derive(Clone, Copy)]
pub(super) struct BoundaryRef<'a, 'db> {
    site: Option<BoundarySiteId>,
    boundary: &'a RuntimeBoundarySpec<'db>,
    matcher: Option<&'a CompiledBoundaryMatcher<'db>>,
}

impl<'a, 'db> BoundaryRef<'a, 'db> {
    pub(super) fn unstaged(boundary: &'a RuntimeBoundarySpec<'db>) -> Self {
        Self {
            site: None,
            boundary,
            matcher: None,
        }
    }

    pub(super) fn staged(boundary: &'a StagedBoundary<'db>) -> Self {
        Self {
            site: Some(boundary.site),
            boundary: &boundary.boundary,
            matcher: Some(&boundary.matcher),
        }
    }
}

#[derive(Default)]
pub(super) struct BoundarySiteAllocator {
    next: u32,
}

impl BoundarySiteAllocator {
    pub(super) fn stage<'db>(&mut self, boundary: RuntimeBoundarySpec<'db>) -> StagedBoundary<'db> {
        let site = BoundarySiteId(self.next);
        self.next += 1;
        let matcher = CompiledBoundaryMatcher::for_boundary(&boundary);
        StagedBoundary {
            site,
            boundary,
            matcher,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct BoundarySpecializationCacheKey<'db> {
    local: SLocalId,
    site: BoundarySiteId,
    aggregate_layout: Option<LayoutId<'db>>,
}

#[derive(Clone, Debug)]
enum BoundarySpecializationCacheValue<'db> {
    Unchanged,
    Specialized {
        boundary: RuntimeBoundarySpec<'db>,
        matcher: CompiledBoundaryMatcher<'db>,
    },
}

type BoundarySpecializationCache<'db> =
    FxHashMap<BoundarySpecializationCacheKey<'db>, BoundarySpecializationCacheValue<'db>>;

#[derive(Default)]
pub(super) struct InferClassCache<'db> {
    boundary_specializations: BoundarySpecializationCache<'db>,
    local_versions: Vec<u32>,
    local_dynamic_facts: Vec<CachedLocalDynamicFacts<'db>>,
}

impl<'db> InferClassCache<'db> {
    pub(super) fn new(local_count: usize) -> Self {
        Self {
            boundary_specializations: BoundarySpecializationCache::default(),
            local_versions: vec![0; local_count],
            local_dynamic_facts: vec![CachedLocalDynamicFacts::default(); local_count],
        }
    }

    pub(super) fn note_carrier_changed(&mut self, local: SLocalId) {
        let version = self
            .local_versions
            .get_mut(local.index())
            .unwrap_or_else(|| panic!("missing local version slot for {local:?}"));
        *version += 1;
        if let Some(entry) = self.local_dynamic_facts.get_mut(local.index()) {
            entry.facts = None;
        }
    }

    pub(super) fn invalidate_local_dynamic_facts(&mut self, local: SLocalId) {
        if let Some(entry) = self.local_dynamic_facts.get_mut(local.index()) {
            entry.facts = None;
        }
    }

    fn local_dynamic_facts(
        &mut self,
        env: BodyEnv<'_, 'db>,
        local: SLocalId,
        carriers: &[RuntimeCarrier<'db>],
    ) -> Option<LocalDynamicFacts<'db>> {
        let local_static = env.local_facts(local)?;
        let self_version = *self.local_versions.get(local.index())?;
        let entry = self
            .local_dynamic_facts
            .get_mut(local.index())
            .unwrap_or_else(|| panic!("missing dynamic fact cache entry for {local:?}"));
        if let Some(facts) = entry.facts.as_ref()
            && entry.self_version == self_version
            && local_static
                .source_locals
                .iter()
                .zip(entry.source_versions.iter())
                .all(|(dep, version)| self.local_versions[dep.index()] == *version)
        {
            return Some(facts.clone());
        }
        let facts = LocalDynamicFacts::compute(env, local, carriers);
        entry.self_version = self_version;
        entry.source_versions = local_static
            .source_locals
            .iter()
            .map(|dep| self.local_versions[dep.index()])
            .collect();
        entry.facts = facts.clone();
        facts
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum RuntimeClassShape<'db> {
    Scalar(ScalarClass<'db>),
    AggregateValue {
        layout: LayoutId<'db>,
    },
    Ref {
        pointee: Box<RuntimeClassShape<'db>>,
        kind: RefShapeKind,
        view: RefView<'db>,
    },
    RawAddr {
        space: AddressSpaceKind,
        target: Option<LayoutId<'db>>,
    },
}

impl<'db> RuntimeClassShape<'db> {
    fn from_class(class: &RuntimeClass<'db>) -> Self {
        match class {
            RuntimeClass::Scalar(class) => Self::Scalar(class.clone()),
            RuntimeClass::AggregateValue { layout } => Self::AggregateValue { layout: *layout },
            RuntimeClass::Ref {
                pointee,
                kind,
                view,
            } => Self::Ref {
                pointee: Box::new(Self::from_class(pointee)),
                kind: RefShapeKind::from_kind(kind),
                view: view.clone(),
            },
            RuntimeClass::RawAddr { space, target } => Self::RawAddr {
                space: *space,
                target: *target,
            },
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum RefShapeKind {
    Const,
    Object,
    Provider(AddressSpaceKind),
}

impl RefShapeKind {
    fn from_kind(kind: &RefKind<'_>) -> Self {
        match kind {
            RefKind::Const => Self::Const,
            RefKind::Object => Self::Object,
            RefKind::Provider { space, .. } => Self::Provider(*space),
        }
    }
}

#[derive(Clone, Debug)]
enum CompiledBoundaryMatcher<'db> {
    Exact(CompiledExactBoundaryMatcher<'db>),
    BorrowLike {
        pointee: RuntimeClassShape<'db>,
        allow_object: bool,
        allow_const: bool,
        provider_spaces: Box<[AddressSpaceKind]>,
        allow_raw_addr: bool,
    },
}

impl<'db> CompiledBoundaryMatcher<'db> {
    fn for_boundary(boundary: &RuntimeBoundarySpec<'db>) -> Self {
        match boundary {
            RuntimeBoundarySpec::ExactTransport(class) | RuntimeBoundarySpec::ExactShape(class) => {
                Self::Exact(CompiledExactBoundaryMatcher::for_class(class))
            }
            RuntimeBoundarySpec::BorrowLike { pointee, allow, .. } => Self::BorrowLike {
                pointee: RuntimeClassShape::from_class(pointee),
                allow_object: allow.allow_object,
                allow_const: allow.allow_const,
                provider_spaces: allow.provider_spaces.clone(),
                allow_raw_addr: allow.allow_raw_addr,
            },
        }
    }

    fn matches_shape(&self, actual: &RuntimeClassShape<'db>) -> bool {
        match self {
            CompiledBoundaryMatcher::Exact(matcher) => matcher.matches_shape(actual),
            CompiledBoundaryMatcher::BorrowLike {
                pointee,
                allow_object,
                allow_const,
                provider_spaces,
                allow_raw_addr,
            } => match actual {
                RuntimeClassShape::Ref {
                    pointee: actual_pointee,
                    kind: RefShapeKind::Object,
                    view: RefView::Whole,
                } => *allow_object && **actual_pointee == *pointee,
                RuntimeClassShape::Ref {
                    pointee: actual_pointee,
                    kind: RefShapeKind::Const,
                    view: RefView::Whole,
                } => *allow_const && **actual_pointee == *pointee,
                RuntimeClassShape::Ref {
                    pointee: actual_pointee,
                    kind: RefShapeKind::Provider(space),
                    view: RefView::Whole,
                } => provider_spaces.contains(space) && **actual_pointee == *pointee,
                RuntimeClassShape::RawAddr { .. } => *allow_raw_addr,
                RuntimeClassShape::Scalar(_)
                | RuntimeClassShape::AggregateValue { .. }
                | RuntimeClassShape::Ref {
                    view: RefView::EnumVariant(_),
                    ..
                } => false,
            },
        }
    }
}

#[derive(Clone, Debug)]
enum CompiledExactBoundaryMatcher<'db> {
    Scalar(ScalarClass<'db>),
    AggregateValue(LayoutId<'db>),
    Ref {
        pointee: RuntimeClassShape<'db>,
        view: RefView<'db>,
        raw_addr_target: Option<LayoutId<'db>>,
    },
    RawAddr {
        target: Option<LayoutId<'db>>,
    },
}

impl<'db> CompiledExactBoundaryMatcher<'db> {
    fn for_class(class: &RuntimeClass<'db>) -> Self {
        match class {
            RuntimeClass::Scalar(class) => Self::Scalar(class.clone()),
            RuntimeClass::AggregateValue { layout } => Self::AggregateValue(*layout),
            RuntimeClass::Ref { pointee, view, .. } => Self::Ref {
                pointee: RuntimeClassShape::from_class(pointee),
                view: view.clone(),
                raw_addr_target: pointee.aggregate_layout(),
            },
            RuntimeClass::RawAddr { target, .. } => Self::RawAddr { target: *target },
        }
    }

    fn matches_shape(&self, actual: &RuntimeClassShape<'db>) -> bool {
        match (self, actual) {
            (CompiledExactBoundaryMatcher::Scalar(expected), RuntimeClassShape::Scalar(actual)) => {
                actual == expected
            }
            (
                CompiledExactBoundaryMatcher::AggregateValue(expected),
                RuntimeClassShape::AggregateValue { layout },
            ) => layout == expected,
            (
                CompiledExactBoundaryMatcher::Ref { pointee, view, .. },
                RuntimeClassShape::Ref {
                    pointee: actual_pointee,
                    view: actual_view,
                    ..
                },
            ) => **actual_pointee == *pointee && actual_view == view,
            (
                CompiledExactBoundaryMatcher::Ref {
                    raw_addr_target, ..
                },
                RuntimeClassShape::RawAddr { target, .. },
            ) => target == raw_addr_target,
            (
                CompiledExactBoundaryMatcher::RawAddr { target: expected },
                RuntimeClassShape::RawAddr { target, .. },
            ) => target == expected,
            _ => false,
        }
    }
}

#[derive(Clone, Debug)]
struct LocalDynamicFacts<'db> {
    exact_source_shape: Option<RuntimeClassShape<'db>>,
    aggregate_layout: Option<LayoutId<'db>>,
}

impl<'db> LocalDynamicFacts<'db> {
    fn compute(
        env: BodyEnv<'_, 'db>,
        local: SLocalId,
        carriers: &[RuntimeCarrier<'db>],
    ) -> Option<Self> {
        let local_static = env.local_facts(local)?;
        let exact_source_shape = carrier_value_class_ref(local, carriers)
            .map(RuntimeClassShape::from_class)
            .or_else(|| {
                env.semantic_value_class(carriers, local)
                    .as_ref()
                    .map(RuntimeClassShape::from_class)
            });
        let aggregate_layout = local_static
            .boundary_source_transport_sensitive
            .then(|| env.actual_aggregate_class_for_source(carriers, local))
            .flatten()
            .and_then(|class| class.aggregate_layout());
        Some(Self {
            exact_source_shape,
            aggregate_layout,
        })
    }
}

#[derive(Clone, Debug, Default)]
struct CachedLocalDynamicFacts<'db> {
    facts: Option<LocalDynamicFacts<'db>>,
    self_version: u32,
    source_versions: Box<[u32]>,
}

#[derive(Clone)]
struct LocalStaticFacts<'db> {
    boundary_source_transport_sensitive: bool,
    semantic_fallback_class: Option<RuntimeClass<'db>>,
    root_place_fallback_class: Option<RuntimeClass<'db>>,
    root_transport_fallback_class: Option<RuntimeClass<'db>>,
    pub(super) materialization_plan: CompiledMaterializationPlan<'db>,
    source_locals: Box<[SLocalId]>,
}

#[derive(Clone)]
enum ExprStaticFacts<'db> {
    Const(Option<RuntimeClass<'db>>),
    DirectClass(Option<RuntimeClass<'db>>),
    AggregateMake(AggregateMakeStaticFacts<'db>),
    Borrow {
        provider_fallback: Option<RuntimeClass<'db>>,
    },
    Call(CallStaticFacts<'db>),
}

#[derive(Clone)]
struct AggregateMakeStaticFacts<'db> {
    direct_class: Option<RuntimeClass<'db>>,
    ctor: AggregateCtorKind<'db>,
    fields: Vec<AggregateMakeFieldStaticFacts<'db>>,
}

#[derive(Clone)]
enum AggregateCtorKind<'db> {
    Aggregate(TyId<'db>),
    EnumVariant {
        enum_ty: TyId<'db>,
        variant: VariantIndex,
    },
}

#[derive(Clone)]
struct AggregateMakeFieldStaticFacts<'db> {
    boundary: Option<StagedBoundary<'db>>,
    stored_class: RuntimeClass<'db>,
}

#[derive(Clone)]
struct CallStaticFacts<'db> {
    semantic: SemanticInstance<'db>,
    builtin_return_class: Option<Option<RuntimeClass<'db>>>,
    input_plan: CompiledCallInputPlan<'db>,
}

impl<'db> BodyStaticFacts<'db> {
    pub(crate) fn new(db: &'db dyn MirDb, body: &NormalizedSemanticBody<'db>) -> Self {
        let typed_body = body.owner.key(db).typed_body(db);
        let type_env = RuntimeTypeEnv::new(
            typed_body.body().map(|body| body.scope()),
            typed_body.assumptions(),
        );
        Self::new_in_context(db, body, typed_body, type_env)
    }

    pub(super) fn new_in_context(
        db: &'db dyn MirDb,
        body: &NormalizedSemanticBody<'db>,
        typed_body: &hir::analysis::ty::ty_check::TypedBody<'db>,
        type_env: RuntimeTypeEnv<'db>,
    ) -> Self {
        let mut boundary_sites = BoundarySiteAllocator::default();
        let local_facts: Vec<_> = body
            .locals
            .iter()
            .enumerate()
            .map(|(idx, local_data)| {
                let local = SLocalId::from_u32(idx as u32);
                build_local_static_facts(db, type_env, local, local_data, body)
            })
            .collect();
        let mut assignments = Vec::new();
        let stmt_assignments = body
            .blocks
            .iter()
            .enumerate()
            .map(|(block_idx, block)| {
                block
                    .stmts
                    .iter()
                    .enumerate()
                    .map(|(stmt_idx, stmt)| {
                        let hir::analysis::semantic::NSStmtKind::Assign { dst, expr } = &stmt.kind
                        else {
                            return None;
                        };
                        let result_ty = body
                            .locals
                            .get(dst.index())
                            .unwrap_or_else(|| panic!("missing assignment local for {dst:?}"))
                            .ty;
                        let expr_facts = build_expr_static_facts(
                            db,
                            body,
                            typed_body,
                            type_env,
                            expr,
                            result_ty,
                            &mut boundary_sites,
                        );
                        let assignment = AssignStaticFacts {
                            block_idx,
                            stmt_idx,
                            dst: *dst,
                            uses: expr_used_locals(body, expr),
                            expr: expr_facts,
                        };
                        let assign_id = assignments.len();
                        assignments.push(assignment);
                        Some(assign_id)
                    })
                    .collect()
            })
            .collect();
        let mut assignments_by_local = vec![Vec::new(); body.locals.len()];
        for (assign_id, assignment) in assignments.iter().enumerate() {
            for local in assignment.uses.iter().copied() {
                assignments_by_local[local.index()].push(assign_id);
            }
        }
        let mut dynamic_dependents_by_local = vec![Vec::new(); body.locals.len()];
        for (local_idx, facts) in local_facts.iter().enumerate() {
            for dependency in facts.source_locals.iter().copied() {
                dynamic_dependents_by_local[dependency.index()]
                    .push(SLocalId::from_u32(local_idx as u32));
            }
        }
        let root_provider_locals = build_runtime_visible_root_provider_locals(db, body.owner);
        Self {
            local_facts,
            assignments,
            stmt_assignments,
            assignments_by_local,
            dynamic_dependents_by_local,
            root_provider_locals,
        }
    }

    fn local(&self, local: SLocalId) -> Option<&LocalStaticFacts<'db>> {
        self.local_facts.get(local.index())
    }

    fn expr(&self, block_idx: usize, stmt_idx: usize) -> Option<&ExprStaticFacts<'db>> {
        let assign_id = self
            .stmt_assignments
            .get(block_idx)?
            .get(stmt_idx)?
            .as_ref()?;
        self.assignments.get(*assign_id)?.expr.as_ref()
    }

    pub(super) fn assignment(&self, assign_id: usize) -> Option<&AssignStaticFacts<'db>> {
        self.assignments.get(assign_id)
    }

    pub(super) fn assignments(&self) -> &[AssignStaticFacts<'db>] {
        &self.assignments
    }

    pub(super) fn source_locals(&self, local: SLocalId) -> &[SLocalId] {
        self.local(local)
            .map(|facts| facts.source_locals.as_ref())
            .unwrap_or(&[])
    }

    pub(super) fn boundary_source_transport_sensitive(&self, local: SLocalId) -> bool {
        self.local(local)
            .is_some_and(|facts| facts.boundary_source_transport_sensitive)
    }

    fn assignments_using_local(&self, local: SLocalId) -> &[usize] {
        self.assignments_by_local
            .get(local.index())
            .map(Vec::as_slice)
            .unwrap_or(&[])
    }

    fn dynamic_dependents(&self, local: SLocalId) -> &[SLocalId] {
        self.dynamic_dependents_by_local
            .get(local.index())
            .map(Vec::as_slice)
            .unwrap_or(&[])
    }

    fn root_provider_local(&self, provider: &ProviderBinding<'db>) -> Option<SLocalId> {
        self.root_provider_locals.get(provider).copied()
    }
}

#[derive(Clone, Copy)]
pub(crate) struct BodyEnv<'a, 'db> {
    db: &'db dyn MirDb,
    body: &'a NormalizedSemanticBody<'db>,
    type_env: RuntimeTypeEnv<'db>,
    facts: &'a BodyStaticFacts<'db>,
}

impl<'a, 'db> BodyEnv<'a, 'db> {
    pub(crate) fn new(
        db: &'db dyn MirDb,
        body: &'a NormalizedSemanticBody<'db>,
        typed_body: &'a hir::analysis::ty::ty_check::TypedBody<'db>,
        facts: &'a BodyStaticFacts<'db>,
    ) -> Self {
        let type_env = RuntimeTypeEnv::new(
            typed_body.body().map(|body| body.scope()),
            typed_body.assumptions(),
        );
        Self::from_parts(db, body, type_env, facts)
    }

    pub(super) fn from_parts(
        db: &'db dyn MirDb,
        body: &'a NormalizedSemanticBody<'db>,
        type_env: RuntimeTypeEnv<'db>,
        facts: &'a BodyStaticFacts<'db>,
    ) -> Self {
        Self {
            db,
            body,
            type_env,
            facts,
        }
    }

    pub(super) fn db(self) -> &'db dyn MirDb {
        self.db
    }

    pub(super) fn body(self) -> &'a NormalizedSemanticBody<'db> {
        self.body
    }

    pub(super) fn scope(self) -> Option<hir::hir_def::scope_graph::ScopeId<'db>> {
        self.type_env.scope
    }

    pub(super) fn assumptions(self) -> PredicateListId<'db> {
        self.type_env.assumptions
    }

    fn local_facts(self, local: SLocalId) -> Option<&'a LocalStaticFacts<'db>> {
        self.facts.local(local)
    }

    pub(super) fn materialization_plan(
        self,
        local: SLocalId,
    ) -> Option<&'a CompiledMaterializationPlan<'db>> {
        self.local_facts(local)
            .map(|facts| &facts.materialization_plan)
    }

    fn expr_facts(self, block_idx: usize, stmt_idx: usize) -> Option<&'a ExprStaticFacts<'db>> {
        self.facts.expr(block_idx, stmt_idx)
    }

    pub(super) fn assignment(self, assign_id: usize) -> Option<&'a AssignStaticFacts<'db>> {
        self.facts.assignment(assign_id)
    }

    pub(super) fn assignment_count(self) -> usize {
        self.facts.assignments.len()
    }

    pub(super) fn assignments_using_local(self, local: SLocalId) -> &'a [usize] {
        self.facts.assignments_using_local(local)
    }

    pub(super) fn dynamic_dependents(self, local: SLocalId) -> &'a [SLocalId] {
        self.facts.dynamic_dependents(local)
    }

    fn actual_runtime_visible_root_provider_local(
        self,
        provider: &ProviderBinding<'db>,
    ) -> Option<SLocalId> {
        self.facts.root_provider_local(provider)
    }

    pub(super) fn actual_runtime_visible_root_provider_class(
        self,
        carriers: &[RuntimeCarrier<'db>],
        provider: &ProviderBinding<'db>,
    ) -> Option<(SLocalId, RuntimeClass<'db>)> {
        let local = self.actual_runtime_visible_root_provider_local(provider)?;
        carrier_value_class(local, carriers).map(|class| (local, class))
    }

    pub(super) fn boundary_source_transport_sensitive(self, local: SLocalId) -> bool {
        self.local_facts(local)
            .is_some_and(|facts| facts.boundary_source_transport_sensitive)
    }

    pub(super) fn root_place_fallback_class(self, local: SLocalId) -> Option<RuntimeClass<'db>> {
        self.local_facts(local)
            .and_then(|facts| facts.root_place_fallback_class.clone())
    }

    pub(super) fn root_transport_fallback_class(
        self,
        local: SLocalId,
    ) -> Option<RuntimeClass<'db>> {
        self.local_facts(local)
            .and_then(|facts| facts.root_transport_fallback_class.clone())
    }

    pub(super) fn with_carriers<'carriers>(
        self,
        carriers: &'carriers [RuntimeCarrier<'db>],
    ) -> RuntimeBodyCx<'a, 'carriers, 'db> {
        RuntimeBodyCx {
            env: self,
            carriers,
        }
    }

    pub(super) fn expr_direct_class(
        self,
        carriers: &[RuntimeCarrier<'db>],
        block_idx: usize,
        stmt_idx: usize,
        expr: &NExpr<'db>,
        mut class_cache: Option<&mut InferClassCache<'db>>,
        returns: &mut RuntimeReturnAnalysisCx<'db>,
    ) -> Option<RuntimeClass<'db>> {
        let expr_facts = self.expr_facts(block_idx, stmt_idx);
        Some(match expr {
            NExpr::Use(value) => {
                RuntimeValueEvaluator::new(self, carriers, class_cache).materialize(value.local)?
            }
            NExpr::Const(_)
            | NExpr::Unary { .. }
            | NExpr::Binary { .. }
            | NExpr::Cast { .. }
            | NExpr::CodeRegionOffset { .. }
            | NExpr::CodeRegionLen { .. }
            | NExpr::IsEnumVariant { .. } => match expr_facts {
                Some(ExprStaticFacts::Const(class) | ExprStaticFacts::DirectClass(class)) => {
                    class.clone()?
                }
                _ => panic!(
                    "missing staged runtime class facts: owner={:?}; expr={expr:?}",
                    self.body.owner.key(self.db),
                ),
            },
            NExpr::CodeRegionRef { .. } => return None,
            NExpr::GetEnumTag { value } => {
                let enum_layout = self
                    .semantic_value_class(carriers, value.local)?
                    .aggregate_layout()
                    .expect("enum tag source should have aggregate layout");
                RuntimeClass::Scalar(ScalarClass {
                    repr: match enum_layout.data(self.db) {
                        Layout::Enum(layout) => layout.tag.repr,
                        Layout::Struct(_) | Layout::Array(_) => {
                            panic!("enum tag source should lower as enum layout: {enum_layout:?}")
                        }
                    },
                    role: ScalarRole::EnumTag { enum_layout },
                })
            }
            NExpr::AggregateMake { fields, .. } | NExpr::EnumMake { fields, .. } => {
                let Some(ExprStaticFacts::AggregateMake(facts)) = expr_facts else {
                    panic!(
                        "missing staged aggregate facts: owner={:?}; expr={expr:?}",
                        self.body.owner.key(self.db),
                    );
                };
                aggregate_make_class_from_facts(
                    self,
                    facts,
                    fields,
                    carriers,
                    class_cache.as_deref_mut(),
                )?
            }
            NExpr::ExtractEnumField {
                value,
                variant,
                field,
            } => {
                let value_class = self.semantic_value_class(carriers, value.local)?;
                let variant = VariantId {
                    enum_layout: value_class
                        .aggregate_layout()
                        .expect("enum extract source should have aggregate layout"),
                    index: variant.0,
                };
                project_variant_field_class(self.db, value_class, variant, FieldIndex(field.0))
            }
            NExpr::ReadPlace { place, .. } => self
                .normalized_place_class(carriers, place)
                .or_else(|| match expr_facts {
                    Some(ExprStaticFacts::DirectClass(class)) => class.clone(),
                    _ => None,
                })?,
            NExpr::Borrow { place, .. } => self
                .normalized_place_address_class(carriers, place)
                .or_else(|| match expr_facts {
                    Some(ExprStaticFacts::Borrow { provider_fallback }) => {
                        provider_fallback.clone()
                    }
                    _ => None,
                })?,
            NExpr::Call {
                args, effect_args, ..
            } => {
                let Some(ExprStaticFacts::Call(facts)) = expr_facts else {
                    panic!(
                        "missing staged runtime call facts: owner={:?}; expr={expr:?}",
                        self.body.owner.key(self.db),
                    );
                };
                if let Some(class) = facts.builtin_return_class.clone() {
                    return class;
                }
                let param_classes: Vec<_> = RuntimeValueEvaluator::new(self, carriers, class_cache)
                    .selected_call_inputs(args, effect_args, &facts.input_plan)
                    .into_iter()
                    .map(|arg| arg.class)
                    .collect();
                return returns.return_class_for_key(RuntimeInstanceKey::new(
                    self.db,
                    RuntimeInstanceSource::Semantic(facts.semantic),
                    param_classes,
                ));
            }
        })
    }
}

impl<'a, 'db> BodyEnv<'a, 'db> {
    pub(crate) fn normalized_place_class(
        self,
        carriers: &[RuntimeCarrier<'db>],
        place: &NSPlace<'db>,
    ) -> Option<RuntimeClass<'db>> {
        let mut current =
            normalized_place_root_class_in_context(self, place.root.clone(), carriers)?;
        for projection in place.path.iter() {
            current = match projection {
                Projection::Field(field) => project_field_class(
                    self.db,
                    current,
                    FieldIndex((*field).try_into().expect("field index fits")),
                ),
                Projection::Index(_) => project_index_class(self.db, current),
                Projection::Deref => match current {
                    RuntimeClass::Ref { pointee, .. } => *pointee,
                    RuntimeClass::RawAddr {
                        target: Some(layout),
                        ..
                    } => RuntimeClass::AggregateValue { layout },
                    RuntimeClass::AggregateValue { .. }
                    | RuntimeClass::Scalar(_)
                    | RuntimeClass::RawAddr { target: None, .. } => {
                        panic!("invalid deref projection class")
                    }
                },
                Projection::VariantField {
                    variant, field_idx, ..
                } => project_variant_field_place_class(
                    self.db,
                    current,
                    *variant,
                    FieldIndex((*field_idx).try_into().expect("field index fits")),
                ),
                Projection::Discriminant => match current {
                    RuntimeClass::Ref { pointee, .. } => match pointee.aggregate_layout() {
                        Some(layout) => match layout.data(self.db) {
                            Layout::Enum(layout) => RuntimeClass::Scalar(layout.tag),
                            Layout::Struct(_) | Layout::Array(_) => {
                                panic!("invalid discriminant projection class")
                            }
                        },
                        None => panic!("invalid discriminant projection class"),
                    },
                    RuntimeClass::AggregateValue { layout }
                    | RuntimeClass::RawAddr {
                        target: Some(layout),
                        ..
                    } => match layout.data(self.db) {
                        Layout::Enum(layout) => RuntimeClass::Scalar(layout.tag),
                        Layout::Struct(_) | Layout::Array(_) => {
                            panic!("invalid discriminant projection class")
                        }
                    },
                    RuntimeClass::Scalar(_) | RuntimeClass::RawAddr { target: None, .. } => {
                        panic!("invalid discriminant projection class")
                    }
                },
            };
        }
        Some(current)
    }

    pub(crate) fn normalized_place_address_class(
        self,
        carriers: &[RuntimeCarrier<'db>],
        place: &NSPlace<'db>,
    ) -> Option<RuntimeClass<'db>> {
        let value_class = self.normalized_place_class(carriers, place)?;
        let root_class =
            normalized_place_root_transport_class_in_context(self, place.root.clone(), carriers)?;
        let (root_space, force_raw) = match place.root {
            NSPlaceRoot::CarrierDerefLocal(_) => (AddressSpaceKind::Memory, false),
            NSPlaceRoot::Root(root) => match self.body.root(root)? {
                NBorrowRoot::Param { .. } | NBorrowRoot::LocalSlot { .. } => {
                    (AddressSpaceKind::Memory, false)
                }
                NBorrowRoot::Provider { binding } => {
                    (provider_root_space(binding, &root_class), false)
                }
            },
        };
        Some(ref_class_for_place_result(
            &root_class,
            &value_class,
            root_space,
            force_raw,
        ))
    }

    pub(crate) fn specialize_boundary_for_source(
        self,
        carriers: &[RuntimeCarrier<'db>],
        local: SLocalId,
        boundary: &RuntimeBoundarySpec<'db>,
    ) -> RuntimeBoundarySpec<'db> {
        specialize_boundary_for_runtime_source_in_context(
            self,
            local,
            BoundaryRef::unstaged(boundary),
            carriers,
            None,
        )
        .boundary
        .into_owned()
    }

    pub(crate) fn actual_aggregate_class_for_source(
        self,
        carriers: &[RuntimeCarrier<'db>],
        local: SLocalId,
    ) -> Option<RuntimeClass<'db>> {
        snapshot_source_place(self.body, local)
            .and_then(|place| self.normalized_place_class(carriers, place))
            .and_then(|class| actual_aggregate_class_from_runtime_source(&class))
            .or_else(|| {
                self.semantic_value_class(carriers, local)
                    .and_then(|class| actual_aggregate_class_from_runtime_source(&class))
            })
    }

    pub(super) fn semantic_value_class(
        self,
        carriers: &[RuntimeCarrier<'db>],
        local: SLocalId,
    ) -> Option<RuntimeClass<'db>> {
        let local_data = self.body.locals.get(local.index())?;
        let local_facts = self.local_facts(local)?;
        match local_data.facts.interface {
            NLocalInterface::Erased => None,
            NLocalInterface::DirectValue | NLocalInterface::DirectCarrier => {
                carrier_value_class(local, carriers)
            }
            NLocalInterface::PlaceCarrier => carrier_value_class(local, carriers)
                .or_else(|| local_facts.semantic_fallback_class.clone()),
            NLocalInterface::PlaceBoundValue => self
                .normalized_place_class(
                    carriers,
                    self.body.locals.get(local.index())?.backing_place()?,
                )
                .or_else(|| local_facts.semantic_fallback_class.clone()),
        }
    }
}

fn root_provider_for_runtime_visible_binding<'db>(
    db: &'db dyn MirDb,
    semantic: SemanticInstance<'db>,
    binding: LocalBinding<'db>,
) -> Option<ProviderBinding<'db>> {
    match semantic_binding_lowering(db, semantic, binding) {
        hir::analysis::semantic::SemanticBindingLowering::DirectValue {
            provenance: ValueProvenance::RootProvider(provider),
        }
        | hir::analysis::semantic::SemanticBindingLowering::DirectCarrier {
            provider: Some(provider),
            ..
        }
        | hir::analysis::semantic::SemanticBindingLowering::PlaceBoundValue {
            provenance: hir::analysis::semantic::PlaceProvenance::RootProvider(provider),
            ..
        } => Some(provider),
        hir::analysis::semantic::SemanticBindingLowering::Erased
        | hir::analysis::semantic::SemanticBindingLowering::DirectValue { .. }
        | hir::analysis::semantic::SemanticBindingLowering::DirectCarrier {
            provider: None, ..
        }
        | hir::analysis::semantic::SemanticBindingLowering::PlaceCarrier { .. }
        | hir::analysis::semantic::SemanticBindingLowering::PlaceBoundValue {
            provenance: hir::analysis::semantic::PlaceProvenance::Derived { .. },
            ..
        } => None,
    }
}

fn build_runtime_visible_root_provider_locals<'db>(
    db: &'db dyn MirDb,
    semantic: SemanticInstance<'db>,
) -> FxHashMap<ProviderBinding<'db>, SLocalId> {
    let mut locals = FxHashMap::default();
    for entry in runtime_visible_binding_plans(db, semantic) {
        let Some(provider) = root_provider_for_runtime_visible_binding(db, semantic, entry.binding)
        else {
            continue;
        };
        locals.entry(provider).or_insert(entry.local);
    }
    locals
}

fn build_local_static_facts<'db>(
    db: &'db dyn MirDb,
    type_env: RuntimeTypeEnv<'db>,
    local: SLocalId,
    local_data: &hir::analysis::semantic::borrowck::NSLocal<'db>,
    body: &NormalizedSemanticBody<'db>,
) -> LocalStaticFacts<'db> {
    let scope = type_env.scope;
    let assumptions = type_env.assumptions;
    let lowered_ty = lowered_place_like_ty(local, local_data);
    let semantic_fallback_class = match local_data.facts.interface {
        NLocalInterface::PlaceCarrier | NLocalInterface::PlaceBoundValue => {
            lowered_ty.map(|ty| stored_class_for_ty_in_context(db, ty, scope, assumptions))
        }
        NLocalInterface::Erased | NLocalInterface::DirectValue | NLocalInterface::DirectCarrier => {
            None
        }
    };
    let root_place_fallback_class = match local_data.facts.interface {
        NLocalInterface::Erased => None,
        NLocalInterface::DirectValue => Some(stored_class_for_ty_in_context(
            db,
            local_data.ty,
            scope,
            assumptions,
        )),
        NLocalInterface::PlaceCarrier
        | NLocalInterface::DirectCarrier
        | NLocalInterface::PlaceBoundValue => {
            lowered_ty.map(|ty| stored_class_for_ty_in_context(db, ty, scope, assumptions))
        }
    };
    LocalStaticFacts {
        boundary_source_transport_sensitive: boundary_source_uses_transport_sensitive_aggregate(
            db,
            local_data.ty,
            scope,
            assumptions,
        ),
        semantic_fallback_class,
        root_place_fallback_class,
        root_transport_fallback_class: fallback_root_transport_class(
            db,
            local_data,
            scope,
            assumptions,
        ),
        materialization_plan: if matches!(local_data.facts.interface, NLocalInterface::Erased) {
            CompiledMaterializationPlan::Erased
        } else {
            match local_data.facts.interface {
                NLocalInterface::DirectValue => top_level_class_for_ty_in_context(
                    db,
                    local_data.ty,
                    AddressSpaceKind::Memory,
                    scope,
                    assumptions,
                )
                .map_or(
                    CompiledMaterializationPlan::AggregateFromSource,
                    |fallback| CompiledMaterializationPlan::AggregateFromSourceOrFallback {
                        fallback,
                    },
                ),
                NLocalInterface::PlaceCarrier
                | NLocalInterface::DirectCarrier
                | NLocalInterface::PlaceBoundValue => CompiledMaterializationPlan::SemanticValue,
                NLocalInterface::Erased => unreachable!(),
            }
        },
        source_locals: local_source_locals(body, local_data),
    }
}

fn lowered_place_like_ty<'db>(
    local: SLocalId,
    local_data: &hir::analysis::semantic::borrowck::NSLocal<'db>,
) -> Option<TyId<'db>> {
    match (&local_data.facts.interface, &local_data.lowering) {
        (
            NLocalInterface::PlaceCarrier | NLocalInterface::DirectCarrier,
            NormalizedBindingLowering::CarrierLocal { target_ty, .. },
        ) => Some(*target_ty),
        (
            NLocalInterface::PlaceBoundValue,
            NormalizedBindingLowering::PlaceBoundValue { value_ty, .. },
        ) => Some(*value_ty),
        (NLocalInterface::PlaceCarrier | NLocalInterface::DirectCarrier, _) => {
            panic!("carrier local missing carrier lowering: {local:?}")
        }
        (NLocalInterface::PlaceBoundValue, _) => {
            panic!("place-bound local missing place-bound lowering: {local:?}")
        }
        (NLocalInterface::Erased | NLocalInterface::DirectValue, _) => None,
    }
}

fn local_source_locals<'db>(
    body: &NormalizedSemanticBody<'db>,
    local_data: &hir::analysis::semantic::borrowck::NSLocal<'db>,
) -> Box<[SLocalId]> {
    let mut uses = IndexSet::new();
    if let Some(place) = local_data.backing_place() {
        uses.extend(place_used_locals(body, place));
    }
    if let Some(place) = local_data.snapshot_source_place() {
        uses.extend(place_used_locals(body, place));
    }
    uses.into_iter().collect()
}

fn expr_used_locals<'db>(body: &NormalizedSemanticBody<'db>, expr: &NExpr<'db>) -> Box<[SLocalId]> {
    let mut uses = IndexSet::new();
    match expr {
        NExpr::Use(value)
        | NExpr::Unary { value, .. }
        | NExpr::Cast { value, .. }
        | NExpr::GetEnumTag { value }
        | NExpr::IsEnumVariant { value, .. }
        | NExpr::ExtractEnumField { value, .. } => {
            uses.insert(value.local);
        }
        NExpr::Binary { lhs, rhs, .. } => {
            uses.insert(lhs.local);
            uses.insert(rhs.local);
        }
        NExpr::AggregateMake { fields, .. } | NExpr::EnumMake { fields, .. } => {
            uses.extend(fields.iter().map(|field| field.local));
        }
        NExpr::ReadPlace { place, .. } | NExpr::Borrow { place, .. } => {
            uses.extend(place_used_locals(body, place));
        }
        NExpr::Call {
            args, effect_args, ..
        } => {
            uses.extend(args.iter().map(|arg| arg.local));
            for effect_arg in effect_args {
                match &effect_arg.arg {
                    NEffectArgValue::Place(place) => uses.extend(place_used_locals(body, place)),
                    NEffectArgValue::Value(value) => {
                        uses.insert(value.local);
                    }
                }
            }
        }
        NExpr::Const(_)
        | NExpr::CodeRegionRef { .. }
        | NExpr::CodeRegionOffset { .. }
        | NExpr::CodeRegionLen { .. } => {}
    }
    uses.into_iter().collect()
}

fn place_used_locals<'db>(
    body: &NormalizedSemanticBody<'db>,
    place: &NSPlace<'db>,
) -> Box<[SLocalId]> {
    let mut uses = IndexSet::new();
    match place.root {
        NSPlaceRoot::Root(root) => match body.root(root) {
            Some(NBorrowRoot::Param { local, .. }) | Some(NBorrowRoot::LocalSlot { local }) => {
                uses.insert(*local);
            }
            Some(NBorrowRoot::Provider { .. }) | None => {}
        },
        NSPlaceRoot::CarrierDerefLocal(local) => {
            uses.insert(local);
        }
    }
    uses.extend(place.path.iter().filter_map(|projection| match projection {
        Projection::Index(hir::projection::IndexSource::Dynamic(local)) => Some(*local),
        _ => None,
    }));
    uses.into_iter().collect()
}

fn build_expr_static_facts<'db>(
    db: &'db dyn MirDb,
    body: &NormalizedSemanticBody<'db>,
    typed_body: &hir::analysis::ty::ty_check::TypedBody<'db>,
    type_env: RuntimeTypeEnv<'db>,
    expr: &NExpr<'db>,
    result_ty: TyId<'db>,
    boundary_sites: &mut BoundarySiteAllocator,
) -> Option<ExprStaticFacts<'db>> {
    Some(match expr {
        NExpr::Use(_) | NExpr::CodeRegionRef { .. } => return None,
        NExpr::Const(const_) => ExprStaticFacts::Const(match const_ {
            SConst::Value(value) => {
                let ty = sem_const_ty(db, *value);
                if ty == TyId::unit(db) {
                    None
                } else {
                    top_level_class_for_ty_in_env(db, type_env, ty, AddressSpaceKind::Memory)
                }
            }
            SConst::Ref(cref) => {
                panic!("unresolved const ref reached runtime class inference: {cref:?}")
            }
        }),
        NExpr::Unary { .. }
        | NExpr::Binary { .. }
        | NExpr::Cast { .. }
        | NExpr::CodeRegionOffset { .. }
        | NExpr::CodeRegionLen { .. } => ExprStaticFacts::DirectClass(
            scalar_class_for_ty_in_env(db, type_env, result_ty).map(RuntimeClass::Scalar),
        ),
        NExpr::GetEnumTag { .. } => return None,
        NExpr::AggregateMake { ty, fields } => {
            let direct_class =
                top_level_class_for_ty_in_env(db, type_env, *ty, AddressSpaceKind::Memory)
                    .filter(|class| !matches!(class, RuntimeClass::AggregateValue { .. }));
            let field_tys = if ty.is_array(db) {
                let (_, args) = ty.decompose_ty_app(db);
                let elem_ty = args.first().copied().expect("array element type");
                vec![elem_ty; fields.len()]
            } else {
                ty.field_types(db)
            };
            if field_tys.len() != fields.len() {
                return None;
            }
            let fields = field_tys
                .into_iter()
                .map(|field_ty| AggregateMakeFieldStaticFacts {
                    boundary: boundary_spec_for_ty_in_env(
                        db,
                        type_env,
                        field_ty,
                        AddressSpaceKind::Memory,
                    )
                    .map(|boundary| boundary_sites.stage(boundary)),
                    stored_class: stored_class_for_ty_in_context(
                        db,
                        field_ty,
                        type_env.scope,
                        type_env.assumptions,
                    ),
                })
                .collect();
            ExprStaticFacts::AggregateMake(AggregateMakeStaticFacts {
                direct_class,
                ctor: AggregateCtorKind::Aggregate(*ty),
                fields,
            })
        }
        NExpr::EnumMake {
            enum_ty,
            variant,
            fields,
        } => {
            let enum_ = enum_ty
                .as_enum(db)
                .unwrap_or_else(|| panic!("enum construction reached non-enum type"));
            let args = enum_ty.generic_args(db);
            let enum_variant = enum_.variants(db).nth(variant.0 as usize)?;
            let field_tys = enum_variant
                .field_tys(db)
                .into_iter()
                .map(|field| field.instantiate(db, args))
                .collect::<Vec<_>>();
            if field_tys.len() != fields.len() {
                return None;
            }
            let fields = field_tys
                .into_iter()
                .map(|field_ty| AggregateMakeFieldStaticFacts {
                    boundary: boundary_spec_for_ty_in_env(
                        db,
                        type_env,
                        field_ty,
                        AddressSpaceKind::Memory,
                    )
                    .map(|boundary| boundary_sites.stage(boundary)),
                    stored_class: stored_class_for_ty_in_context(
                        db,
                        field_ty,
                        type_env.scope,
                        type_env.assumptions,
                    ),
                })
                .collect();
            ExprStaticFacts::AggregateMake(AggregateMakeStaticFacts {
                direct_class: None,
                ctor: AggregateCtorKind::EnumVariant {
                    enum_ty: *enum_ty,
                    variant: *variant,
                },
                fields,
            })
        }
        NExpr::ReadPlace { .. } => ExprStaticFacts::DirectClass(top_level_class_for_ty_in_env(
            db,
            type_env,
            result_ty,
            AddressSpaceKind::Memory,
        )),
        NExpr::ExtractEnumField { .. } => return None,
        NExpr::Borrow { provider, .. } => ExprStaticFacts::Borrow {
            provider_fallback: provider.map(|provider| RuntimeClass::RawAddr {
                space: address_space_from_provider(provider),
                target: None,
            }),
        },
        NExpr::IsEnumVariant { .. } => {
            ExprStaticFacts::DirectClass(Some(RuntimeClass::Scalar(ScalarClass {
                repr: ScalarRepr::Bool,
                role: ScalarRole::Plain,
            })))
        }
        NExpr::Call {
            callee,
            args,
            effect_args,
        } => {
            let caller_key = body.owner.key(db);
            let callee_key = resolve_runtime_call_key(
                db, caller_key, typed_body, body, *callee, args,
            )
            .unwrap_or_else(|err| {
                panic!(
                    "runtime call resolution failed during return-class inference for {:?}: {err}",
                    caller_key,
                )
            });
            let semantic = get_or_build_semantic_instance(db, callee_key);
            let callee_typed_body = semantic.key(db).typed_body(db);
            ExprStaticFacts::Call(CallStaticFacts {
                semantic,
                builtin_return_class: extern_builtin_return_class(db, semantic, result_ty),
                input_plan: compile_call_input_plan_for_semantic(
                    db,
                    body,
                    semantic,
                    RuntimeTypeEnv::new(
                        callee_typed_body.body().map(|body| body.scope()),
                        callee_typed_body.assumptions(),
                    ),
                    effect_args,
                    boundary_sites,
                ),
            })
        }
    })
}

#[derive(Clone, Copy)]
pub(crate) struct RuntimeBodyCx<'a, 'carriers, 'db> {
    pub(crate) env: BodyEnv<'a, 'db>,
    pub(crate) carriers: &'carriers [RuntimeCarrier<'db>],
}

impl<'a, 'carriers, 'db> RuntimeBodyCx<'a, 'carriers, 'db> {
    pub(crate) fn materialized_value_class(self, local: SLocalId) -> Option<RuntimeClass<'db>> {
        RuntimeValueEvaluator::new(self.env, self.carriers, None).materialize(local)
    }

    pub(crate) fn normalized_place_class(self, place: &NSPlace<'db>) -> Option<RuntimeClass<'db>> {
        self.env.normalized_place_class(self.carriers, place)
    }

    pub(crate) fn normalized_place_address_class(
        self,
        place: &NSPlace<'db>,
    ) -> Option<RuntimeClass<'db>> {
        self.env
            .normalized_place_address_class(self.carriers, place)
    }

    pub(crate) fn actual_aggregate_class_for_source(
        self,
        local: SLocalId,
    ) -> Option<RuntimeClass<'db>> {
        self.env
            .actual_aggregate_class_for_source(self.carriers, local)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Update)]
pub(crate) struct RuntimeVisibleBindingPlan<'db> {
    pub(crate) binding: LocalBinding<'db>,
    pub(crate) local: SLocalId,
    pub(crate) semantic_ty: TyId<'db>,
    pub(crate) plan: RuntimeParamPlan<'db>,
}

#[derive(Clone, Debug)]
pub(crate) struct RuntimeEffectBindingPlan<'db> {
    pub(crate) class: RuntimeClass<'db>,
    pub(crate) boundary: RuntimeBoundarySpec<'db>,
}

#[derive(Clone, Debug)]
pub(crate) enum RuntimeVisibleReturnPlan<'db> {
    Erased,
    Exact(RuntimeClass<'db>),
    Constrained(RuntimeBoundarySpec<'db>),
    PassActual,
}

pub(crate) fn runtime_address_space(class: &RuntimeClass<'_>) -> Option<AddressSpaceKind> {
    match class {
        RuntimeClass::Ref {
            kind: RefKind::Provider { space, .. },
            ..
        }
        | RuntimeClass::RawAddr { space, .. } => Some(*space),
        RuntimeClass::Scalar(_)
        | RuntimeClass::AggregateValue { .. }
        | RuntimeClass::Ref {
            kind: RefKind::Const | RefKind::Object,
            ..
        } => None,
    }
}

pub(super) fn provider_root_space<'db>(
    binding: &ProviderBinding<'db>,
    root_class: &RuntimeClass<'db>,
) -> AddressSpaceKind {
    runtime_address_space(root_class).unwrap_or_else(|| match binding.semantics.kind {
        ProviderKind::RootObject => AddressSpaceKind::Memory,
        ProviderKind::Handle | ProviderKind::RawAddress => address_space_from_provider(
            binding
                .semantics
                .address_space
                .unwrap_or_else(|| panic!("provider binding missing resolved space")),
        ),
    })
}

pub(crate) fn ref_class_for_place_result<'db>(
    root_class: &RuntimeClass<'db>,
    value_class: &RuntimeClass<'db>,
    root_space: AddressSpaceKind,
    force_raw: bool,
) -> RuntimeClass<'db> {
    if !force_raw {
        match root_class {
            RuntimeClass::Ref { kind, .. } => {
                return RuntimeClass::Ref {
                    pointee: Box::new(value_class.clone()),
                    kind: kind.clone(),
                    view: RefView::Whole,
                };
            }
            RuntimeClass::AggregateValue { .. } => {
                return RuntimeClass::Ref {
                    pointee: Box::new(value_class.clone()),
                    kind: RefKind::Object,
                    view: RefView::Whole,
                };
            }
            RuntimeClass::Scalar(_) | RuntimeClass::RawAddr { .. } => {}
        }
    }
    RuntimeClass::RawAddr {
        space: runtime_address_space(root_class).unwrap_or(root_space),
        target: value_class.aggregate_layout(),
    }
}

pub(crate) fn runtime_signature_for_key_with_returns<'db>(
    db: &'db dyn MirDb,
    semantic: SemanticInstance<'db>,
    params: &[RuntimeClass<'db>],
    returns: &mut RuntimeReturnAnalysisCx<'db>,
) -> RuntimeSignature<'db> {
    let key = RuntimeInstanceKey::new(
        db,
        RuntimeInstanceSource::Semantic(semantic),
        params.to_vec(),
    );
    RuntimeSignature {
        params: params
            .iter()
            .zip(runtime_param_locals(db, semantic, params))
            .map(|(class, local)| RuntimeParam {
                local: crate::runtime::RLocalId::from_u32(local.index() as u32),
                class: class.clone(),
            })
            .collect(),
        ret: returns.return_class_for_key(key),
    }
}

fn provider_root_place_class<'db>(
    db: &'db dyn MirDb,
    value_ty: TyId<'db>,
    provider_class: &RuntimeClass<'db>,
    scope: Option<hir::hir_def::scope_graph::ScopeId<'db>>,
    assumptions: PredicateListId<'db>,
) -> RuntimeClass<'db> {
    match provider_class {
        RuntimeClass::Ref { pointee, .. } => *pointee.clone(),
        RuntimeClass::RawAddr {
            target: Some(layout),
            ..
        } => RuntimeClass::AggregateValue { layout: *layout },
        RuntimeClass::Scalar(_)
        | RuntimeClass::AggregateValue { .. }
        | RuntimeClass::RawAddr { target: None, .. } => {
            stored_class_for_ty_in_context(db, value_ty, scope, assumptions)
        }
    }
}

pub(crate) fn runtime_class_for_provider_binding<'db>(
    db: &'db dyn MirDb,
    provider: &ProviderBinding<'db>,
    scope: Option<hir::hir_def::scope_graph::ScopeId<'db>>,
    assumptions: PredicateListId<'db>,
) -> Option<RuntimeClass<'db>> {
    match provider.semantics.kind {
        ProviderKind::RootObject => top_level_class_for_ty_in_context(
            db,
            provider.provider_ty,
            AddressSpaceKind::Memory,
            scope,
            assumptions,
        ),
        ProviderKind::Handle | ProviderKind::RawAddress => {
            Some(provider_class_for_target_in_context(
                db,
                provider.semantics.target_ty,
                provider_address_space_to_runtime(provider.semantics.address_space?),
                scope,
                assumptions,
            ))
        }
    }
}

pub(crate) fn runtime_class_for_effect_binding_provider_in_context<'db>(
    db: &'db dyn MirDb,
    provider: &ProviderBinding<'db>,
    scope: Option<hir::hir_def::scope_graph::ScopeId<'db>>,
    assumptions: PredicateListId<'db>,
) -> Option<RuntimeClass<'db>> {
    match provider.semantics.kind {
        ProviderKind::RootObject => Some(provider_class_for_target_in_context(
            db,
            Some(provider.semantics.target_ty.unwrap_or(provider.provider_ty)),
            provider
                .semantics
                .address_space
                .map_or(AddressSpaceKind::Memory, provider_address_space_to_runtime),
            scope,
            assumptions,
        )),
        ProviderKind::Handle | ProviderKind::RawAddress => {
            runtime_class_for_provider_binding(db, provider, scope, assumptions)
        }
    }
}

pub(crate) fn runtime_class_for_direct_value_provider_in_context<'db>(
    db: &'db dyn MirDb,
    provider: &ProviderBinding<'db>,
    scope: Option<hir::hir_def::scope_graph::ScopeId<'db>>,
    assumptions: PredicateListId<'db>,
) -> Option<RuntimeClass<'db>> {
    runtime_class_for_effect_binding_provider_in_context(db, provider, scope, assumptions)
}

fn runtime_class_for_provider_value_ty_in_context<'db>(
    db: &'db dyn MirDb,
    provider: &ProviderBinding<'db>,
    value_ty: TyId<'db>,
    scope: Option<hir::hir_def::scope_graph::ScopeId<'db>>,
    assumptions: PredicateListId<'db>,
) -> Option<RuntimeClass<'db>> {
    let space = match provider.semantics.kind {
        ProviderKind::RootObject => provider
            .semantics
            .address_space
            .map_or(AddressSpaceKind::Memory, provider_address_space_to_runtime),
        ProviderKind::Handle | ProviderKind::RawAddress => {
            provider_address_space_to_runtime(provider.semantics.address_space?)
        }
    };
    Some(provider_class_for_target_in_context(
        db,
        Some(value_ty),
        space,
        scope,
        assumptions,
    ))
}

fn effect_binding_borrow_boundary<'db>(
    db: &'db dyn MirDb,
    binding: LocalBinding<'db>,
    pointee_ty: TyId<'db>,
    scope: Option<hir::hir_def::scope_graph::ScopeId<'db>>,
    assumptions: PredicateListId<'db>,
) -> RuntimeBoundarySpec<'db> {
    let access = if binding.is_mut() {
        BorrowAccess::ReadWrite
    } else {
        BorrowAccess::ReadOnly
    };
    RuntimeBoundarySpec::BorrowLike {
        pointee: stored_class_for_ty_in_context(db, pointee_ty, scope, assumptions),
        access,
        allow: default_borrow_transport_set(access, AddressSpaceKind::Memory),
    }
}

fn specialize_effect_binding_boundary_for_class<'db>(
    boundary: RuntimeBoundarySpec<'db>,
    class: &RuntimeClass<'db>,
) -> RuntimeBoundarySpec<'db> {
    specialize_boundary_for_aggregate_layout(&boundary, class.aggregate_layout()).into_owned()
}

pub(crate) fn runtime_effect_binding_plan<'db>(
    db: &'db dyn MirDb,
    semantic: SemanticInstance<'db>,
    binding: LocalBinding<'db>,
) -> Option<RuntimeEffectBindingPlan<'db>> {
    if !matches!(binding, LocalBinding::EffectParam { .. }) {
        return None;
    }
    let owner = semantic.key(db).owner(db);
    let env = RuntimeTypeEnv::new(
        Some(owner.scope()),
        semantic_instance_assumptions(db, semantic),
    );
    let binding_ty = semantic_binding_ty(db, semantic, binding);
    match semantic_binding_lowering(db, semantic, binding) {
        hir::analysis::semantic::SemanticBindingLowering::Erased => None,
        hir::analysis::semantic::SemanticBindingLowering::DirectValue {
            provenance: ValueProvenance::RootProvider(provider),
        } => {
            let class = runtime_class_for_provider_value_ty_in_context(
                db,
                &provider,
                binding_ty,
                env.scope,
                env.assumptions,
            )?;
            let boundary = specialize_effect_binding_boundary_for_class(
                effect_binding_borrow_boundary(db, binding, binding_ty, env.scope, env.assumptions),
                &class,
            );
            Some(RuntimeEffectBindingPlan { class, boundary })
        }
        hir::analysis::semantic::SemanticBindingLowering::DirectValue { .. } => {
            let class =
                runtime_class_for_explicit_root_provider_param(db, env, binding, binding_ty)
                    .or_else(|| {
                        top_level_class_for_ty_in_env(db, env, binding_ty, AddressSpaceKind::Memory)
                    })?;
            Some(RuntimeEffectBindingPlan {
                class: class.clone(),
                boundary: RuntimeBoundarySpec::exact_for_class(class),
            })
        }
        hir::analysis::semantic::SemanticBindingLowering::DirectCarrier {
            provider: Some(provider),
            ..
        } => {
            let class =
                runtime_class_for_provider_binding(db, &provider, env.scope, env.assumptions)?;
            Some(RuntimeEffectBindingPlan {
                class: class.clone(),
                boundary: RuntimeBoundarySpec::exact_for_class(class),
            })
        }
        hir::analysis::semantic::SemanticBindingLowering::DirectCarrier {
            provider: None,
            target_ty,
        } => {
            let class =
                top_level_class_for_ty_in_env(db, env, binding_ty, AddressSpaceKind::Memory)
                    .or_else(|| {
                        Some(provider_class_for_target_in_env(
                            db,
                            env,
                            Some(target_ty),
                            AddressSpaceKind::Memory,
                        ))
                    })?;
            Some(RuntimeEffectBindingPlan {
                class: class.clone(),
                boundary: RuntimeBoundarySpec::exact_for_class(class),
            })
        }
        hir::analysis::semantic::SemanticBindingLowering::PlaceCarrier { value_ty } => {
            let class =
                provider_class_for_target_in_env(db, env, Some(value_ty), AddressSpaceKind::Memory);
            Some(RuntimeEffectBindingPlan {
                class: class.clone(),
                boundary: RuntimeBoundarySpec::exact_for_class(class),
            })
        }
        hir::analysis::semantic::SemanticBindingLowering::PlaceBoundValue {
            provenance: hir::analysis::semantic::PlaceProvenance::RootProvider(provider),
            value_ty,
        } => {
            let class = runtime_class_for_provider_value_ty_in_context(
                db,
                &provider,
                value_ty,
                env.scope,
                env.assumptions,
            )?;
            let boundary = specialize_effect_binding_boundary_for_class(
                effect_binding_borrow_boundary(db, binding, value_ty, env.scope, env.assumptions),
                &class,
            );
            Some(RuntimeEffectBindingPlan { class, boundary })
        }
        hir::analysis::semantic::SemanticBindingLowering::PlaceBoundValue {
            provenance: hir::analysis::semantic::PlaceProvenance::Derived { .. },
            ..
        } => None,
    }
}

fn runtime_exact_class_for_ordinary_binding_in_env<'db>(
    db: &'db dyn MirDb,
    env: RuntimeTypeEnv<'db>,
    binding: LocalBinding<'db>,
    binding_ty: TyId<'db>,
) -> Option<RuntimeClass<'db>> {
    runtime_class_for_explicit_root_provider_param(db, env, binding, binding_ty).or_else(|| {
        match boundary_spec_for_ty_in_env(db, env, binding_ty, AddressSpaceKind::Memory) {
            Some(RuntimeBoundarySpec::ExactTransport(class))
            | Some(RuntimeBoundarySpec::ExactShape(class)) => Some(class),
            Some(RuntimeBoundarySpec::BorrowLike { .. }) | None => None,
        }
    })
}

fn runtime_exact_class_for_visible_binding_in_env<'db>(
    db: &'db dyn MirDb,
    semantic: SemanticInstance<'db>,
    env: RuntimeTypeEnv<'db>,
    binding: LocalBinding<'db>,
    binding_ty: TyId<'db>,
) -> Option<RuntimeClass<'db>> {
    match semantic_binding_lowering(db, semantic, binding) {
        hir::analysis::semantic::SemanticBindingLowering::Erased => None,
        hir::analysis::semantic::SemanticBindingLowering::DirectValue {
            provenance: ValueProvenance::RootProvider(provider),
        } => runtime_class_for_provider_value_ty_in_context(
            db,
            &provider,
            binding_ty,
            env.scope,
            env.assumptions,
        ),
        hir::analysis::semantic::SemanticBindingLowering::DirectValue { .. } => {
            runtime_exact_class_for_ordinary_binding_in_env(db, env, binding, binding_ty).or_else(
                || top_level_class_for_ty_in_env(db, env, binding_ty, AddressSpaceKind::Memory),
            )
        }
        hir::analysis::semantic::SemanticBindingLowering::DirectCarrier {
            provider: Some(provider),
            ..
        } => runtime_class_for_provider_binding(db, &provider, env.scope, env.assumptions),
        hir::analysis::semantic::SemanticBindingLowering::DirectCarrier {
            provider: None,
            target_ty,
        } => top_level_class_for_ty_in_env(db, env, binding_ty, AddressSpaceKind::Memory).or_else(
            || {
                Some(provider_class_for_target_in_env(
                    db,
                    env,
                    Some(target_ty),
                    AddressSpaceKind::Memory,
                ))
            },
        ),
        hir::analysis::semantic::SemanticBindingLowering::PlaceCarrier { value_ty } => Some(
            provider_class_for_target_in_env(db, env, Some(value_ty), AddressSpaceKind::Memory),
        ),
        hir::analysis::semantic::SemanticBindingLowering::PlaceBoundValue {
            provenance: hir::analysis::semantic::PlaceProvenance::RootProvider(provider),
            value_ty,
        } => runtime_class_for_provider_value_ty_in_context(
            db,
            &provider,
            value_ty,
            env.scope,
            env.assumptions,
        ),
        hir::analysis::semantic::SemanticBindingLowering::PlaceBoundValue {
            provenance: hir::analysis::semantic::PlaceProvenance::Derived { .. },
            ..
        } => None,
    }
}

pub(crate) fn runtime_effect_binding_plan_for_binding_idx<'db>(
    db: &'db dyn MirDb,
    semantic: SemanticInstance<'db>,
    binding_idx: u32,
) -> Option<RuntimeEffectBindingPlan<'db>> {
    let BodyOwner::Func(func) = semantic.key(db).owner(db) else {
        return None;
    };
    let resolved = hir::semantic::EffectEnvView::new(EffectParamSite::Func(func))
        .resolved_binding(db, binding_idx as usize)?;
    runtime_effect_binding_plan(
        db,
        semantic,
        LocalBinding::EffectParam {
            site: resolved.requirement.binding_site,
            idx: resolved.requirement.binding_idx as usize,
            binding_name: resolved.requirement.binding_name,
            provider_idx: resolved.provider.provider_idx,
            key_path: resolved.requirement.binding_path,
            is_mut: resolved.requirement.is_mut,
        },
    )
}

pub(crate) fn runtime_visible_binding_class<'db>(
    db: &'db dyn MirDb,
    semantic: SemanticInstance<'db>,
    binding: LocalBinding<'db>,
) -> Option<RuntimeClass<'db>> {
    if let Some(plan) = runtime_effect_binding_plan(db, semantic, binding) {
        return Some(plan.class);
    }
    let owner = semantic.key(db).owner(db);
    let typed_body = semantic.key(db).typed_body(db);
    let env = RuntimeTypeEnv::new(Some(owner.scope()), typed_body.assumptions());
    let binding_ty = semantic_binding_ty(db, semantic, binding);
    runtime_exact_class_for_visible_binding_in_env(db, semantic, env, binding, binding_ty)
}

pub(crate) fn owner_effect_binding_boundary<'db>(
    db: &'db dyn MirDb,
    semantic: SemanticInstance<'db>,
    binding: LocalBinding<'db>,
) -> Option<RuntimeBoundarySpec<'db>> {
    runtime_effect_binding_plan(db, semantic, binding).map(|plan| plan.boundary)
}

fn runtime_class_for_explicit_root_provider_param<'db>(
    db: &'db dyn MirDb,
    env: RuntimeTypeEnv<'db>,
    binding: LocalBinding<'db>,
    binding_ty: TyId<'db>,
) -> Option<RuntimeClass<'db>> {
    let LocalBinding::Param {
        site: ParamSite::Func(func),
        idx,
        ..
    } = binding
    else {
        return None;
    };
    if !func
        .params(db)
        .nth(idx)
        .is_some_and(|param| param.is_self_param(db))
    {
        return None;
    }
    let canonical = |ty| {
        strip_derived_adt_layout_args(
            db,
            runtime_repr_ty_in_context(
                db,
                env.scope
                    .map_or(ty, |scope| normalize_ty(db, ty, scope, env.assumptions)),
                env.scope,
                env.assumptions,
            ),
        )
    };
    let binding_ty = canonical(binding_ty);
    let binding_ty = binding_ty
        .as_capability(db)
        .map_or(binding_ty, |(_, inner)| canonical(inner));
    registered_root_providers(db, EffectParamSite::Func(func))
        .into_iter()
        .find(|provider| canonical(provider.provider_ty) == binding_ty)
        .map(|provider| {
            provider_class_for_target_in_env(
                db,
                env,
                Some(provider.provider_ty),
                AddressSpaceKind::Memory,
            )
        })
}

fn aggregate_make_class_from_facts<'db>(
    env: BodyEnv<'_, 'db>,
    facts: &AggregateMakeStaticFacts<'db>,
    fields: &[NOperand],
    carriers: &[RuntimeCarrier<'db>],
    class_cache: Option<&mut InferClassCache<'db>>,
) -> Option<RuntimeClass<'db>> {
    if let Some(class) = facts.direct_class.clone() {
        return Some(class);
    }
    if facts.fields.len() != fields.len() {
        return None;
    }
    let mut field_classes = Vec::with_capacity(fields.len());
    let mut evaluator = RuntimeValueEvaluator::new(env, carriers, class_cache);
    for (field, field_facts) in fields.iter().copied().zip(facts.fields.iter()) {
        let class = field_facts
            .boundary
            .as_ref()
            .and_then(|boundary| {
                let mut boundary_sites = BoundarySiteAllocator::default();
                evaluator
                    .selected_value_pass_plan(
                        field,
                        &compile_value_pass_plan(
                            RuntimeParamPlan::Boundary(boundary.boundary.clone()),
                            &mut boundary_sites,
                        ),
                    )
                    .map(|arg| arg.class)
            })
            .or_else(|| evaluator.materialize(field.local))
            .unwrap_or_else(|| field_facts.stored_class.clone());
        field_classes.push(class);
    }
    Some(RuntimeClass::AggregateValue {
        layout: match facts.ctor {
            AggregateCtorKind::Aggregate(ty) => layout_for_aggregate_instance_in_context(
                env.db,
                ty,
                &field_classes,
                env.scope(),
                env.assumptions(),
            ),
            AggregateCtorKind::EnumVariant { enum_ty, variant } => {
                layout_for_enum_variant_instance_in_context(
                    env.db,
                    enum_ty,
                    variant.0 as usize,
                    &field_classes,
                    env.scope(),
                    env.assumptions(),
                )
            }
        },
    })
}

pub(crate) fn visible_return_class_for_local<'db>(
    env: BodyEnv<'_, 'db>,
    local: SLocalId,
    plan: &RuntimeVisibleReturnPlan<'db>,
    carriers: &[RuntimeCarrier<'db>],
) -> Option<RuntimeClass<'db>> {
    let mut evaluator = RuntimeValueEvaluator::new(env, carriers, None);
    match plan {
        RuntimeVisibleReturnPlan::Erased => None,
        RuntimeVisibleReturnPlan::Exact(class) => {
            evaluator.materialize(local).or_else(|| Some(class.clone()))
        }
        RuntimeVisibleReturnPlan::Constrained(boundary) => {
            let mut boundary_sites = BoundarySiteAllocator::default();
            evaluator.selected_value_class_for_local(
                local,
                &compile_value_pass_plan(
                    RuntimeParamPlan::Boundary(boundary.clone()),
                    &mut boundary_sites,
                ),
            )
        }
        RuntimeVisibleReturnPlan::PassActual => evaluator.actual_value(local),
    }
}

pub(super) struct SpecializedBoundary<'a, 'db> {
    pub(super) boundary: Cow<'a, RuntimeBoundarySpec<'db>>,
    matcher: CompiledBoundaryMatcher<'db>,
}

pub(super) fn specialize_boundary_for_runtime_source_in_context<'a, 'db>(
    env: BodyEnv<'_, 'db>,
    local: SLocalId,
    boundary: BoundaryRef<'a, 'db>,
    carriers: &[RuntimeCarrier<'db>],
    mut class_cache: Option<&mut InferClassCache<'db>>,
) -> SpecializedBoundary<'a, 'db> {
    let aggregate_layout = class_cache
        .as_deref_mut()
        .and_then(|cache| cache.local_dynamic_facts(env, local, carriers))
        .and_then(|facts| facts.aggregate_layout)
        .or_else(|| {
            env.local_facts(local)
                .filter(|facts| facts.boundary_source_transport_sensitive)
                .and_then(|_| env.actual_aggregate_class_for_source(carriers, local))
                .and_then(|class| class.aggregate_layout())
        });
    if let (Some(site), Some(cache)) = (
        boundary.site,
        class_cache
            .as_deref_mut()
            .map(|cache| &mut cache.boundary_specializations),
    ) {
        let key = BoundarySpecializationCacheKey {
            local,
            site,
            aggregate_layout,
        };
        if let Some(cached) = cache.get(&key) {
            let specialized = match cached {
                BoundarySpecializationCacheValue::Unchanged => SpecializedBoundary {
                    boundary: Cow::Borrowed(boundary.boundary),
                    matcher: boundary.matcher.cloned().unwrap_or_else(|| {
                        CompiledBoundaryMatcher::for_boundary(boundary.boundary)
                    }),
                },
                BoundarySpecializationCacheValue::Specialized { boundary, matcher } => {
                    SpecializedBoundary {
                        boundary: Cow::Owned(boundary.clone()),
                        matcher: matcher.clone(),
                    }
                }
            };
            return preserve_actual_shape_boundary_for_runtime_source(
                env,
                local,
                specialized,
                carriers,
                class_cache,
            );
        }
        let specialized_boundary =
            specialize_boundary_for_aggregate_layout(boundary.boundary, aggregate_layout);
        let specialized_matcher = match &specialized_boundary {
            Cow::Borrowed(_) => boundary
                .matcher
                .cloned()
                .unwrap_or_else(|| CompiledBoundaryMatcher::for_boundary(boundary.boundary)),
            Cow::Owned(boundary) => CompiledBoundaryMatcher::for_boundary(boundary),
        };
        cache.insert(
            key,
            match &specialized_boundary {
                Cow::Borrowed(_) => BoundarySpecializationCacheValue::Unchanged,
                Cow::Owned(boundary) => BoundarySpecializationCacheValue::Specialized {
                    boundary: boundary.clone(),
                    matcher: specialized_matcher.clone(),
                },
            },
        );
        return preserve_actual_shape_boundary_for_runtime_source(
            env,
            local,
            SpecializedBoundary {
                boundary: specialized_boundary,
                matcher: specialized_matcher,
            },
            carriers,
            class_cache,
        );
    }
    preserve_actual_shape_boundary_for_runtime_source(
        env,
        local,
        SpecializedBoundary {
            matcher: boundary
                .matcher
                .cloned()
                .unwrap_or_else(|| CompiledBoundaryMatcher::for_boundary(boundary.boundary)),
            boundary: specialize_boundary_for_aggregate_layout(boundary.boundary, aggregate_layout),
        },
        carriers,
        class_cache,
    )
}

pub(super) fn nonself_backing_value_place<'a, 'db>(
    body: &'a NormalizedSemanticBody<'db>,
    local: SLocalId,
) -> Option<&'a NSPlace<'db>> {
    let place = body.local(local)?.backing_place()?;
    (!is_self_rooted_value_place(body, local, place)).then_some(place)
}

fn is_self_rooted_value_place<'db>(
    body: &NormalizedSemanticBody<'db>,
    local: SLocalId,
    place: &NSPlace<'db>,
) -> bool {
    if !place.path.is_empty() {
        return false;
    }
    match place.root {
        NSPlaceRoot::CarrierDerefLocal(root_local) => root_local == local,
        NSPlaceRoot::Root(root) => matches!(
            body.root(root),
            Some(NBorrowRoot::Param { local: root_local, .. } | NBorrowRoot::LocalSlot { local: root_local })
                if *root_local == local
        ),
    }
}

pub(super) fn snapshot_source_place<'a, 'db>(
    body: &'a NormalizedSemanticBody<'db>,
    local: SLocalId,
) -> Option<&'a NSPlace<'db>> {
    body.local(local)?.snapshot_source_place()
}

pub(crate) fn desired_runtime_param_plan<'db>(
    db: &'db dyn MirDb,
    semantic: SemanticInstance<'db>,
    typed_body: &hir::analysis::ty::ty_check::TypedBody<'db>,
    idx: usize,
) -> RuntimeParamPlan<'db> {
    let Some(binding) = typed_body.param_binding(idx) else {
        return RuntimeParamPlan::Erased;
    };
    let binding_ty = typed_body.binding_ty(db, binding);
    let scope = typed_body.body().map(|body| body.scope());
    let assumptions = typed_body.assumptions();
    let env = RuntimeTypeEnv::new(scope, assumptions);
    let repr_ty = runtime_repr_ty_in_context(db, binding_ty, scope, assumptions);
    if runtime_abstract_param_ty(db, binding_ty, scope, assumptions)
        || matches!(
            repr_ty.base_ty(db).data(db),
            TyData::TyParam(param) if param.is_effect() || param.is_effect_provider()
        )
    {
        return RuntimeParamPlan::PassActual;
    }
    if let Some(class) =
        runtime_exact_class_for_visible_binding_in_env(db, semantic, env, binding, binding_ty)
        && (!matches!(class, RuntimeClass::AggregateValue { .. })
            || !aggregate_transport_depends_on_runtime_source(db, binding_ty, scope, assumptions))
    {
        return RuntimeParamPlan::Boundary(runtime_param_boundary(
            db,
            typed_body,
            binding,
            RuntimeBoundarySpec::exact_for_class(class),
        ));
    }
    let Some(boundary) = boundary_spec_for_ty_in_env(db, env, binding_ty, AddressSpaceKind::Memory)
    else {
        return RuntimeParamPlan::Erased;
    };
    if matches!(
        boundary,
        RuntimeBoundarySpec::ExactTransport(RuntimeClass::AggregateValue { .. })
            | RuntimeBoundarySpec::ExactShape(RuntimeClass::AggregateValue { .. })
    ) && aggregate_transport_depends_on_runtime_source(db, binding_ty, scope, assumptions)
    {
        return RuntimeParamPlan::PassActual;
    }
    RuntimeParamPlan::Boundary(runtime_param_boundary(db, typed_body, binding, boundary))
}

pub(crate) fn resolve_runtime_call_key<'db>(
    db: &'db dyn MirDb,
    caller_key: SemanticInstanceKey<'db>,
    caller_typed_body: &hir::analysis::ty::ty_check::TypedBody<'db>,
    body: &NormalizedSemanticBody<'db>,
    callee: SemanticCalleeRef<'db>,
    args: &[NOperand],
) -> Result<SemanticInstanceKey<'db>, crate::runtime::LowerError> {
    let callee_key = callee.key;
    let callee_semantic = get_or_build_semantic_instance(db, callee_key);
    if contract_metadata_builtin(db, callee_semantic).is_some() {
        return Ok(callee_key);
    }
    let BodyOwner::Func(func) = callee_key.owner(db) else {
        return Ok(callee_key);
    };
    let Some(trait_) = func.containing_trait(db) else {
        return Ok(callee_key);
    };
    if func.body(db).is_some() {
        return Ok(callee_key);
    }
    let Some(method_name) = func.name(db).to_opt() else {
        return Err(crate::runtime::LowerError::Unsupported(format!(
            "runtime trait-call resolution reached an unnamed declaration-only method: caller={caller_key:?} callee={callee_key:?}"
        )));
    };
    let impl_env = callee_key.impl_env(db);
    let original_inst: Option<TraitInstId<'db>> = impl_env
        .witnesses(db)
        .iter()
        .find(|inst| inst.def(db) == trait_)
        .copied();
    let concrete_inst = if func
        .params(db)
        .next()
        .is_some_and(|param| param.is_self_param(db))
    {
        let Some(arg) = args.first() else {
            return Err(crate::runtime::LowerError::Unsupported(format!(
                "runtime trait-call resolution is missing a self argument: caller={caller_key:?} callee={callee_key:?}"
            )));
        };
        let Some(self_ty) =
            concrete_runtime_self_ty_for_call_arg(db, caller_typed_body, body, arg.local)
        else {
            return Err(crate::runtime::LowerError::Unsupported(format!(
                "runtime trait-call resolution could not infer the concrete self type: caller={caller_key:?} callee={callee_key:?} local={:?}",
                arg.local,
            )));
        };
        let mut inst_args = original_inst
            .map(|inst| inst.args(db).to_vec())
            .unwrap_or_else(|| vec![self_ty]);
        let Some(first) = inst_args.first_mut() else {
            return Err(crate::runtime::LowerError::Unsupported(format!(
                "runtime trait-call resolution produced an empty trait-inst arg list: caller={caller_key:?} callee={callee_key:?}"
            )));
        };
        *first = self_ty;
        TraitInstId::new(
            db,
            trait_,
            inst_args,
            original_inst
                .map(|inst| inst.assoc_type_bindings(db).clone())
                .unwrap_or_default(),
        )
    } else {
        let Some(original_inst) = original_inst else {
            return Err(crate::runtime::LowerError::Unsupported(format!(
                "runtime trait-call resolution is missing a trait witness for a declaration-only method: caller={caller_key:?} callee={callee_key:?}"
            )));
        };
        original_inst
    };
    let assumptions = runtime_callee_assumptions(db, caller_key, caller_typed_body);
    let Some((impl_func, mut impl_args)) = resolve_trait_method_instance(
        db,
        TraitSolveCx::new(db, caller_key.impl_env(db).normalization_scope(db))
            .with_assumptions(assumptions),
        concrete_inst,
        method_name,
    ) else {
        return Err(crate::runtime::LowerError::Unsupported(format!(
            "runtime trait-call resolution failed to resolve a concrete impl body: caller={caller_key:?} decl={callee_key:?} method={} concrete_inst={} original_inst={}",
            method_name.data(db),
            concrete_inst.pretty_print(db, false),
            original_inst
                .map(|inst| inst.pretty_print(db, false))
                .unwrap_or_else(|| "<none>".to_string()),
        )));
    };
    let trait_arg_len = concrete_inst.args(db).len();
    let tail = callee_key
        .subst(db)
        .generic_args(db)
        .get(trait_arg_len..)
        .unwrap_or(callee_key.subst(db).generic_args(db).as_slice());
    impl_args.extend_from_slice(tail);
    let mut witnesses = IndexSet::new();
    witnesses.extend(caller_key.impl_env(db).witnesses(db).iter().copied());
    witnesses.extend(impl_env.witnesses(db).iter().copied());
    witnesses.insert(concrete_inst);
    Ok(SemanticInstanceKey::new(
        db,
        BodyOwner::Func(impl_func),
        GenericSubst::new(db, impl_args),
        hir::analysis::semantic::EffectProviderSubst::empty(db),
        ImplEnv::new(
            db,
            caller_key.impl_env(db).normalization_scope(db),
            assumptions,
            witnesses.into_iter().collect::<Vec<_>>(),
        ),
    ))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum GenericNumericIntrinsicKind {
    Bitcast,
    Saturating(SaturatingBinOp),
    CheckedBinary(ArithBinOp),
    CheckedNeg,
}

fn runtime_callee_assumptions<'db>(
    db: &'db dyn MirDb,
    caller_key: SemanticInstanceKey<'db>,
    caller_typed_body: &hir::analysis::ty::ty_check::TypedBody<'db>,
) -> PredicateListId<'db> {
    let impl_env = caller_key.impl_env(db);
    let mut predicates: IndexSet<_> = caller_typed_body
        .assumptions()
        .list(db)
        .iter()
        .copied()
        .collect();
    predicates.extend(impl_env.assumptions(db).list(db).iter().copied());
    predicates.extend(impl_env.witnesses(db).iter().copied());
    PredicateListId::new(db, predicates.into_iter().collect::<Vec<_>>())
}

fn concrete_runtime_self_ty_for_call_arg<'db>(
    db: &'db dyn MirDb,
    caller_typed_body: &hir::analysis::ty::ty_check::TypedBody<'db>,
    body: &NormalizedSemanticBody<'db>,
    local: SLocalId,
) -> Option<TyId<'db>> {
    let scope = caller_typed_body.body().map(|body| body.scope());
    let assumptions = caller_typed_body.assumptions();
    let normalized = |ty| normalize_runtime_self_ty(db, ty, scope, assumptions);
    let local_data = body.locals.get(local.index())?;
    match (
        &local_data.facts.interface,
        &local_data.facts.origin,
        &local_data.lowering,
    ) {
        (NLocalInterface::Erased, _, _) => None,
        (
            NLocalInterface::DirectValue | NLocalInterface::DirectCarrier,
            NLocalOrigin::RootProvider(provider),
            _,
        ) => Some(normalized(provider.provider_ty)),
        (
            NLocalInterface::PlaceBoundValue,
            NLocalOrigin::RootProvider(provider),
            NormalizedBindingLowering::PlaceBoundValue { value_ty, .. },
        ) => Some(normalized(
            provider.semantics.target_ty.unwrap_or(*value_ty),
        )),
        (
            NLocalInterface::PlaceBoundValue,
            NLocalOrigin::SelfRooted | NLocalOrigin::AliasedPlace,
            NormalizedBindingLowering::PlaceBoundValue { value_ty, .. },
        ) => Some(normalized(*value_ty)),
        (NLocalInterface::DirectValue, _, _) => Some(normalized(local_data.ty)),
        (
            NLocalInterface::PlaceCarrier | NLocalInterface::DirectCarrier,
            _,
            NormalizedBindingLowering::CarrierLocal { target_ty, .. },
        ) => Some(normalized(*target_ty)),
        _ => None,
    }
}

fn normalize_runtime_self_ty<'db>(
    db: &'db dyn MirDb,
    ty: TyId<'db>,
    scope: Option<hir::hir_def::scope_graph::ScopeId<'db>>,
    assumptions: PredicateListId<'db>,
) -> TyId<'db> {
    let ty = runtime_repr_ty_in_context(db, ty, scope, assumptions);
    if let Some((_, inner)) = ty.as_borrow(db) {
        return scope.map_or(inner, |scope| normalize_ty(db, inner, scope, assumptions));
    }
    scope.map_or(ty, |scope| normalize_ty(db, ty, scope, assumptions))
}

pub(super) fn carrier_value_class<'db>(
    local: SLocalId,
    carriers: &[RuntimeCarrier<'db>],
) -> Option<RuntimeClass<'db>> {
    carrier_value_class_ref(local, carriers).cloned()
}

fn carrier_value_class_ref<'a, 'db>(
    local: SLocalId,
    carriers: &'a [RuntimeCarrier<'db>],
) -> Option<&'a RuntimeClass<'db>> {
    match carriers.get(local.index())? {
        RuntimeCarrier::Erased => None,
        RuntimeCarrier::Value(class) => Some(class),
    }
}

fn normalized_place_root_transport_class_in_context<'db>(
    env: BodyEnv<'_, 'db>,
    root: NSPlaceRoot,
    carriers: &[RuntimeCarrier<'db>],
) -> Option<RuntimeClass<'db>> {
    match root {
        NSPlaceRoot::CarrierDerefLocal(local) => {
            carrier_value_class(local, carriers).or_else(|| {
                env.local_facts(local)?
                    .root_transport_fallback_class
                    .clone()
            })
        }
        NSPlaceRoot::Root(root) => match env.body.root(root)? {
            NBorrowRoot::Param { local, .. } | NBorrowRoot::LocalSlot { local } => {
                carrier_value_class(*local, carriers).or_else(|| {
                    env.local_facts(*local)?
                        .root_transport_fallback_class
                        .clone()
                })
            }
            NBorrowRoot::Provider { binding } => env
                .actual_runtime_visible_root_provider_class(carriers, binding)
                .map(|(_, class)| class)
                .or_else(|| {
                    runtime_class_for_effect_binding_provider_in_context(
                        env.db,
                        binding,
                        env.scope(),
                        env.assumptions(),
                    )
                    .or_else(|| {
                        runtime_class_for_direct_value_provider_in_context(
                            env.db,
                            binding,
                            env.scope(),
                            env.assumptions(),
                        )
                    })
                }),
        },
    }
}

fn normalized_place_root_class_in_context<'db>(
    env: BodyEnv<'_, 'db>,
    root: NSPlaceRoot,
    carriers: &[RuntimeCarrier<'db>],
) -> Option<RuntimeClass<'db>> {
    let cx = env.with_carriers(carriers);
    match root {
        NSPlaceRoot::CarrierDerefLocal(local) => {
            let local_data = env.body.locals.get(local.index())?;
            local_place_root_class(cx, local, local_data, carriers.get(local.index())?)
        }
        NSPlaceRoot::Root(root) => match env.body.root(root)? {
            NBorrowRoot::Param { local, .. } | NBorrowRoot::LocalSlot { local } => {
                local_place_root_class(
                    cx,
                    *local,
                    env.body.locals.get(local.index())?,
                    carriers.get(local.index())?,
                )
            }
            NBorrowRoot::Provider { binding } => {
                let provider_class = env
                    .actual_runtime_visible_root_provider_class(carriers, binding)
                    .map(|(_, class)| class)
                    .or_else(|| {
                        runtime_class_for_effect_binding_provider_in_context(
                            env.db,
                            binding,
                            env.scope(),
                            env.assumptions(),
                        )
                        .or_else(|| {
                            runtime_class_for_direct_value_provider_in_context(
                                env.db,
                                binding,
                                env.scope(),
                                env.assumptions(),
                            )
                        })
                    })?;
                Some(provider_root_place_class(
                    env.db,
                    binding.provider_ty,
                    &provider_class,
                    env.scope(),
                    env.assumptions(),
                ))
            }
        },
    }
}

fn project_variant_field_place_class<'db>(
    db: &'db dyn MirDb,
    class: RuntimeClass<'db>,
    variant: VariantIndex,
    field: FieldIndex,
) -> RuntimeClass<'db> {
    let layout = class
        .aggregate_layout()
        .unwrap_or_else(|| panic!("invalid variant-field projection class"));
    match layout.data(db) {
        Layout::Enum(layout) => {
            layout.variants[variant.0 as usize].fields[field.0 as usize].clone()
        }
        Layout::Struct(_) | Layout::Array(_) => panic!("invalid variant-field projection layout"),
    }
}

fn specialize_boundary_for_aggregate_layout<'a, 'db>(
    boundary: &'a RuntimeBoundarySpec<'db>,
    aggregate_layout: Option<LayoutId<'db>>,
) -> Cow<'a, RuntimeBoundarySpec<'db>> {
    match boundary {
        RuntimeBoundarySpec::ExactTransport(desired) => {
            match specialize_exact_boundary_for_aggregate_layout(desired, aggregate_layout) {
                Cow::Borrowed(_) => Cow::Borrowed(boundary),
                Cow::Owned(class) => Cow::Owned(RuntimeBoundarySpec::ExactTransport(class)),
            }
        }
        RuntimeBoundarySpec::ExactShape(desired) => {
            match specialize_exact_boundary_for_aggregate_layout(desired, aggregate_layout) {
                Cow::Borrowed(_) => Cow::Borrowed(boundary),
                Cow::Owned(class) => Cow::Owned(RuntimeBoundarySpec::ExactShape(class)),
            }
        }
        RuntimeBoundarySpec::BorrowLike {
            pointee:
                RuntimeClass::AggregateValue {
                    layout: desired_layout,
                },
            access,
            allow,
        } => match aggregate_layout {
            Some(layout) if layout != *desired_layout => {
                Cow::Owned(RuntimeBoundarySpec::BorrowLike {
                    pointee: RuntimeClass::AggregateValue { layout },
                    access: *access,
                    allow: allow.clone(),
                })
            }
            Some(_) | None => Cow::Borrowed(boundary),
        },
        RuntimeBoundarySpec::BorrowLike { .. } => Cow::Borrowed(boundary),
    }
}

fn specialize_exact_boundary_for_aggregate_layout<'a, 'db>(
    desired: &'a RuntimeClass<'db>,
    aggregate_layout: Option<LayoutId<'db>>,
) -> Cow<'a, RuntimeClass<'db>> {
    match (desired, aggregate_layout) {
        (_, None) => Cow::Borrowed(desired),
        (
            RuntimeClass::AggregateValue {
                layout: desired_layout,
            },
            Some(layout),
        ) if layout == *desired_layout => Cow::Borrowed(desired),
        (RuntimeClass::AggregateValue { .. }, Some(layout)) => {
            Cow::Owned(RuntimeClass::AggregateValue { layout })
        }
        (
            RuntimeClass::Ref {
                pointee,
                kind,
                view,
            },
            Some(layout),
        ) if pointee.aggregate_layout().is_some() && pointee.aggregate_layout() != Some(layout) => {
            Cow::Owned(RuntimeClass::Ref {
                pointee: Box::new(RuntimeClass::AggregateValue { layout }),
                kind: kind.clone(),
                view: view.clone(),
            })
        }
        (RuntimeClass::Ref { .. }, Some(_)) => Cow::Borrowed(desired),
        (
            RuntimeClass::RawAddr {
                space,
                target: Some(desired_target),
            },
            Some(layout),
        ) if layout != *desired_target => Cow::Owned(RuntimeClass::RawAddr {
            space: *space,
            target: Some(layout),
        }),
        (RuntimeClass::Scalar(_) | RuntimeClass::RawAddr { target: None, .. }, Some(_))
        | (
            RuntimeClass::RawAddr {
                target: Some(_), ..
            },
            Some(_),
        ) => Cow::Borrowed(desired),
    }
}

fn preserve_actual_shape_boundary_for_runtime_source<'a, 'db>(
    env: BodyEnv<'_, 'db>,
    local: SLocalId,
    boundary: SpecializedBoundary<'a, 'db>,
    carriers: &[RuntimeCarrier<'db>],
    class_cache: Option<&mut InferClassCache<'db>>,
) -> SpecializedBoundary<'a, 'db> {
    if !matches!(
        boundary.boundary.as_ref(),
        RuntimeBoundarySpec::ExactShape(_)
    ) {
        return boundary;
    }
    let actual_matches = if let Some(class_cache) = class_cache {
        class_cache
            .local_dynamic_facts(env, local, carriers)
            .and_then(|facts| facts.exact_source_shape)
            .is_some_and(|shape| boundary.matcher.matches_shape(&shape))
    } else if let Some(actual) = carrier_value_class_ref(local, carriers) {
        boundary
            .matcher
            .matches_shape(&RuntimeClassShape::from_class(actual))
    } else {
        env.semantic_value_class(carriers, local)
            .as_ref()
            .map(RuntimeClassShape::from_class)
            .is_some_and(|shape| boundary.matcher.matches_shape(&shape))
    };
    if !actual_matches {
        return boundary;
    }
    if let Some(actual) = carrier_value_class_ref(local, carriers) {
        return SpecializedBoundary {
            matcher: CompiledBoundaryMatcher::for_boundary(&RuntimeBoundarySpec::ExactShape(
                actual.clone(),
            )),
            boundary: Cow::Owned(RuntimeBoundarySpec::ExactShape(actual.clone())),
        };
    }
    let Some(actual) = env.semantic_value_class(carriers, local) else {
        return boundary;
    };
    SpecializedBoundary {
        matcher: CompiledBoundaryMatcher::for_boundary(&RuntimeBoundarySpec::ExactShape(
            actual.clone(),
        )),
        boundary: Cow::Owned(RuntimeBoundarySpec::ExactShape(actual)),
    }
}

pub(crate) fn desired_runtime_effect_arg_boundary<'db>(
    db: &'db dyn MirDb,
    env: RuntimeTypeEnv<'db>,
    arg: &NEffectArg<'db>,
    plan: Option<&RuntimeEffectBindingPlan<'db>>,
    effect_space: AddressSpaceKind,
) -> Option<RuntimeBoundarySpec<'db>> {
    if let Some(plan) = plan {
        return Some(plan.boundary.clone());
    }
    arg.target_ty.map(|target_ty| match arg.pass_mode {
        EffectPassMode::ByPlace | EffectPassMode::ByTempPlace => RuntimeBoundarySpec::BorrowLike {
            pointee: stored_class_for_ty_in_context(db, target_ty, env.scope, env.assumptions),
            access: BorrowAccess::ReadWrite,
            allow: default_borrow_transport_set(BorrowAccess::ReadWrite, effect_space),
        },
        EffectPassMode::ByValue | EffectPassMode::Unknown => boundary_spec_for_ty_in_env(
            db,
            env,
            target_ty,
            effect_space,
        )
        .unwrap_or(RuntimeBoundarySpec::ExactShape(
            provider_class_for_target_in_env(db, env, Some(target_ty), effect_space),
        )),
    })
}

pub(crate) enum ContractMetadataBuiltin<'db> {
    InitCodeOffset(RuntimeCodeRegion<'db>),
    InitCodeLen(RuntimeCodeRegion<'db>),
}

pub(crate) fn contract_metadata_builtin<'db>(
    db: &'db dyn MirDb,
    semantic: SemanticInstance<'db>,
) -> Option<ContractMetadataBuiltin<'db>> {
    let BodyOwner::Func(func) = semantic.key(db).owner(db) else {
        return None;
    };
    let name = func.name(db).to_opt()?.data(db);
    let trait_ = func.containing_trait(db)?;
    if trait_.name(db).to_opt()?.data(db) != "Contract" {
        return None;
    }
    let contract = semantic
        .key(db)
        .subst(db)
        .generic_args(db)
        .iter()
        .find_map(|ty| ty.as_contract(db))?;
    let region = RuntimeCodeRegion::new(db, RuntimeCodeRegionKey::ContractInit { contract });
    match name.as_str() {
        "init_code_offset" => Some(ContractMetadataBuiltin::InitCodeOffset(region)),
        "init_code_len" => Some(ContractMetadataBuiltin::InitCodeLen(region)),
        _ => None,
    }
}

#[salsa::tracked]
fn runtime_extern_builtin_return_class<'db>(
    db: &'db dyn MirDb,
    semantic: SemanticInstance<'db>,
    result_ty: TyId<'db>,
) -> Option<Option<RuntimeClass<'db>>> {
    let typed_body = semantic.key(db).typed_body(db);
    let env = RuntimeTypeEnv::new(
        typed_body.body().map(|body| body.scope()),
        typed_body.assumptions(),
    );
    if contract_metadata_builtin(db, semantic).is_some() {
        return Some(top_level_class_for_ty_in_env(
            db,
            env,
            result_ty,
            AddressSpaceKind::Memory,
        ));
    }
    let hir::analysis::ty::ty_check::BodyOwner::Func(func) = semantic.key(db).owner(db) else {
        return None;
    };
    if func.body(db).is_none()
        && func
            .name(db)
            .to_opt()
            .is_some_and(|name| is_runtime_intrinsic_name(name.data(db).as_str()))
    {
        return Some(top_level_class_for_ty_in_env(
            db,
            env,
            result_ty,
            AddressSpaceKind::Memory,
        ));
    }
    runtime_builtin_func_kind(db, func)
        .is_some()
        .then(|| top_level_class_for_ty_in_env(db, env, result_ty, AddressSpaceKind::Memory))
}

fn extern_builtin_return_class<'db>(
    db: &'db dyn MirDb,
    semantic: SemanticInstance<'db>,
    result_ty: TyId<'db>,
) -> Option<Option<RuntimeClass<'db>>> {
    runtime_extern_builtin_return_class(db, semantic, result_ty)
}

fn is_runtime_intrinsic_name(name: &str) -> bool {
    if matches!(name, "alloc") || generic_numeric_intrinsic_kind(name).is_some() {
        return true;
    }
    intrinsic_numeric_name_parts(name).is_some()
}

pub(super) fn generic_numeric_intrinsic_kind(name: &str) -> Option<GenericNumericIntrinsicKind> {
    Some(match name {
        "__bitcast" => GenericNumericIntrinsicKind::Bitcast,
        "__saturating_add" => GenericNumericIntrinsicKind::Saturating(SaturatingBinOp::Add),
        "__saturating_sub" => GenericNumericIntrinsicKind::Saturating(SaturatingBinOp::Sub),
        "__saturating_mul" => GenericNumericIntrinsicKind::Saturating(SaturatingBinOp::Mul),
        "__checked_add" => GenericNumericIntrinsicKind::CheckedBinary(ArithBinOp::Add),
        "__checked_sub" => GenericNumericIntrinsicKind::CheckedBinary(ArithBinOp::Sub),
        "__checked_mul" => GenericNumericIntrinsicKind::CheckedBinary(ArithBinOp::Mul),
        "__checked_div" => GenericNumericIntrinsicKind::CheckedBinary(ArithBinOp::Div),
        "__checked_rem" => GenericNumericIntrinsicKind::CheckedBinary(ArithBinOp::Rem),
        "__checked_pow" => GenericNumericIntrinsicKind::CheckedBinary(ArithBinOp::Pow),
        "__checked_neg" => GenericNumericIntrinsicKind::CheckedNeg,
        _ => return None,
    })
}

fn intrinsic_numeric_name_parts(name: &str) -> Option<(&str, &str)> {
    let op = name.strip_prefix("__")?;
    [
        "_u8", "_u16", "_u32", "_u64", "_u128", "_u256", "_usize", "_i8", "_i16", "_i32", "_i64",
        "_i128", "_i256", "_isize", "_bool",
    ]
    .iter()
    .find_map(|suffix| op.strip_suffix(suffix).map(|prefix| (prefix, *suffix)))
}

pub(crate) fn runtime_param_class<'db>(
    db: &'db dyn MirDb,
    typed_body: &hir::analysis::ty::ty_check::TypedBody<'db>,
    binding: hir::analysis::ty::ty_check::LocalBinding<'db>,
    actual: RuntimeClass<'db>,
) -> RuntimeClass<'db> {
    let ty = runtime_repr_ty_in_context(
        db,
        typed_body.binding_ty(db, binding),
        typed_body.body().map(|body| body.scope()),
        typed_body.assumptions(),
    );
    if runtime_abstract_param_ty(
        db,
        typed_body.binding_ty(db, binding),
        typed_body.body().map(|body| body.scope()),
        typed_body.assumptions(),
    ) || matches!(
        ty.base_ty(db).data(db),
        TyData::TyParam(param) if param.is_effect() || param.is_effect_provider()
    ) {
        return actual;
    }
    if binding.is_mut() && ty.as_enum(db).is_some() {
        return RuntimeClass::object_ref(layout_for_ty_in_context(
            db,
            ty,
            typed_body.body().map(|body| body.scope()),
            typed_body.assumptions(),
        ));
    }
    actual
}

pub(crate) fn runtime_param_boundary<'db>(
    db: &'db dyn MirDb,
    typed_body: &hir::analysis::ty::ty_check::TypedBody<'db>,
    binding: hir::analysis::ty::ty_check::LocalBinding<'db>,
    boundary: RuntimeBoundarySpec<'db>,
) -> RuntimeBoundarySpec<'db> {
    match boundary {
        RuntimeBoundarySpec::ExactTransport(actual) => RuntimeBoundarySpec::ExactTransport(
            runtime_param_class(db, typed_body, binding, actual),
        ),
        RuntimeBoundarySpec::ExactShape(actual) => {
            RuntimeBoundarySpec::ExactShape(runtime_param_class(db, typed_body, binding, actual))
        }
        RuntimeBoundarySpec::BorrowLike {
            pointee,
            access,
            allow,
        } => {
            let ty = runtime_repr_ty_in_context(
                db,
                typed_body.binding_ty(db, binding),
                typed_body.body().map(|body| body.scope()),
                typed_body.assumptions(),
            );
            if binding.is_mut() && ty.as_enum(db).is_some() {
                return RuntimeBoundarySpec::ExactTransport(RuntimeClass::object_ref(
                    layout_for_ty_in_context(
                        db,
                        ty,
                        typed_body.body().map(|body| body.scope()),
                        typed_body.assumptions(),
                    ),
                ));
            }
            RuntimeBoundarySpec::BorrowLike {
                pointee,
                access,
                allow,
            }
        }
    }
}

pub(crate) fn semantic_return_ty<'db>(
    db: &'db dyn MirDb,
    semantic: SemanticInstance<'db>,
) -> TyId<'db> {
    semantic.key(db).typed_body(db).result_ty()
}

pub(crate) fn default_return_class<'db>(
    db: &'db dyn MirDb,
    typed_body: &hir::analysis::ty::ty_check::TypedBody<'db>,
) -> Option<RuntimeClass<'db>> {
    let env = RuntimeTypeEnv::new(
        typed_body.body().map(|body| body.scope()),
        typed_body.assumptions(),
    );
    let return_borrow_provider = typed_body.return_borrow_provider();
    let default_space =
        return_borrow_provider.map_or(AddressSpaceKind::Memory, address_space_from_provider);
    if return_borrow_provider.is_some() {
        return Some(provider_class_for_target_in_env(
            db,
            env,
            Some(typed_body.result_ty()),
            default_space,
        ));
    }
    top_level_class_for_ty_in_env(db, env, typed_body.result_ty(), default_space)
}

pub(crate) fn desired_runtime_return_plan<'db>(
    db: &'db dyn MirDb,
    typed_body: &hir::analysis::ty::ty_check::TypedBody<'db>,
) -> RuntimeVisibleReturnPlan<'db> {
    let env = RuntimeTypeEnv::new(
        typed_body.body().map(|body| body.scope()),
        typed_body.assumptions(),
    );
    let return_borrow_provider = typed_body.return_borrow_provider();
    let default_space =
        return_borrow_provider.map_or(AddressSpaceKind::Memory, address_space_from_provider);
    let ty = typed_body.result_ty();
    if return_borrow_provider.is_some() {
        return RuntimeVisibleReturnPlan::PassActual;
    }
    let repr_ty = runtime_repr_ty_in_context(db, ty, env.scope, env.assumptions);
    if runtime_abstract_param_ty(db, ty, env.scope, env.assumptions)
        || matches!(
            repr_ty.base_ty(db).data(db),
            TyData::TyParam(param) if param.is_effect() || param.is_effect_provider()
        )
    {
        return RuntimeVisibleReturnPlan::PassActual;
    }
    let Some(boundary) = boundary_spec_for_ty_in_env(db, env, ty, default_space) else {
        return RuntimeVisibleReturnPlan::Erased;
    };
    match &boundary {
        RuntimeBoundarySpec::ExactTransport(class @ RuntimeClass::Scalar(_))
        | RuntimeBoundarySpec::ExactTransport(class @ RuntimeClass::Ref { .. })
        | RuntimeBoundarySpec::ExactTransport(class @ RuntimeClass::RawAddr { .. }) => {
            RuntimeVisibleReturnPlan::Exact(class.clone())
        }
        RuntimeBoundarySpec::ExactTransport(class @ RuntimeClass::AggregateValue { .. })
            if !aggregate_transport_depends_on_runtime_source(
                db,
                ty,
                env.scope,
                env.assumptions,
            ) =>
        {
            RuntimeVisibleReturnPlan::Exact(class.clone())
        }
        RuntimeBoundarySpec::ExactTransport(RuntimeClass::AggregateValue { .. })
        | RuntimeBoundarySpec::ExactShape(_)
        | RuntimeBoundarySpec::BorrowLike { .. } => RuntimeVisibleReturnPlan::Constrained(boundary),
    }
}

pub(crate) fn actual_aggregate_class_from_runtime_source<'db>(
    class: &RuntimeClass<'db>,
) -> Option<RuntimeClass<'db>> {
    match class {
        RuntimeClass::AggregateValue { .. } => Some(class.clone()),
        RuntimeClass::Ref { pointee, .. } => pointee
            .aggregate_layout()
            .map(|layout| RuntimeClass::AggregateValue { layout }),
        RuntimeClass::RawAddr {
            target: Some(layout),
            ..
        } => Some(RuntimeClass::AggregateValue { layout: *layout }),
        RuntimeClass::Scalar(_) | RuntimeClass::RawAddr { target: None, .. } => None,
    }
}

fn runtime_abstract_param_ty<'db>(
    db: &'db dyn MirDb,
    ty: TyId<'db>,
    scope: Option<hir::hir_def::scope_graph::ScopeId<'db>>,
    assumptions: PredicateListId<'db>,
) -> bool {
    let ty = runtime_repr_ty_in_context(db, ty, scope, assumptions);
    ty.has_param(db) || ty.contains_assoc_ty_of_param(db)
}

#[cfg(test)]
mod tests {
    use common::InputDb;
    use driver::DriverDataBase;
    use hir::{
        analysis::semantic::{
            SemanticInstance, borrowck::normalize_semantic_body, get_or_build_semantic_instance,
            owner_effect_bindings, root_semantic_instance_key,
        },
        analysis::ty::ty_check::BodyOwner,
    };
    use url::Url;

    use super::super::call_input::RuntimeValueEvaluator;
    use super::*;
    use crate::runtime::lower::realize::RuntimeBoundaryMatcher;
    use crate::runtime::{
        lower::{
            infer::LocalStateInferer,
            interface::{runtime_param_locals, runtime_visible_binding_plans},
            returns::RuntimeReturnAnalysisCx,
        },
        package::runtime_instance_for_semantic,
    };

    fn runtime_signature_for_named_func<'db>(
        db: &'db DriverDataBase,
        top_mod: hir::hir_def::TopLevelMod<'db>,
        name: &str,
    ) -> RuntimeSignature<'db> {
        runtime_instance_for_semantic(db, semantic_instance_for_named_func(db, top_mod, name))
            .signature(db)
    }

    fn func_by_name<'db>(
        db: &'db DriverDataBase,
        top_mod: hir::hir_def::TopLevelMod<'db>,
        name: &str,
    ) -> hir::core::hir_def::item::Func<'db> {
        top_mod
            .all_funcs(db)
            .iter()
            .copied()
            .find(|func| {
                func.name(db)
                    .to_opt()
                    .is_some_and(|func_name| func_name.data(db) == name)
            })
            .unwrap_or_else(|| panic!("missing function `{name}`"))
    }

    fn semantic_instance_for_named_func<'db>(
        db: &'db DriverDataBase,
        top_mod: hir::hir_def::TopLevelMod<'db>,
        name: &str,
    ) -> SemanticInstance<'db> {
        let func = func_by_name(db, top_mod, name);
        let key = root_semantic_instance_key(db, BodyOwner::Func(func)).unwrap_or_else(|err| {
            panic!("failed to build root semantic key for `{name}`: {err:?}")
        });
        get_or_build_semantic_instance(db, key)
    }

    fn contract_by_name<'db>(
        db: &'db DriverDataBase,
        top_mod: hir::hir_def::TopLevelMod<'db>,
        name: &str,
    ) -> hir::hir_def::item::Contract<'db> {
        top_mod
            .all_contracts(db)
            .iter()
            .copied()
            .find(|contract| {
                contract
                    .name(db)
                    .to_opt()
                    .is_some_and(|contract_name| contract_name.data(db) == name)
            })
            .unwrap_or_else(|| panic!("missing contract `{name}`"))
    }

    #[test]
    fn poseidon_helpers_keep_visible_by_value_array_returns() {
        let mut db = DriverDataBase::default();
        let file_url =
            Url::parse("file:///poseidon_helpers_keep_visible_by_value_array_returns.fe").unwrap();
        db.workspace().touch(
            &mut db,
            file_url.clone(),
            Some(
                include_str!("../../../../fe/tests/fixtures/fe_test/poseidon_mock.fe").to_string(),
            ),
        );
        let file = db
            .workspace()
            .get(&db, &file_url)
            .expect("file should be loaded");
        let top_mod = db.top_mod(file);
        let signatures = ["ark", "sigma_full", "mix"]
            .into_iter()
            .map(|name| (name, runtime_signature_for_named_func(&db, top_mod, name)))
            .collect::<Vec<_>>();

        assert!(
            signatures.iter().all(|(_, signature)| matches!(
                signature.ret,
                Some(RuntimeClass::AggregateValue { .. })
            )),
            "Poseidon helpers should keep by-value aggregate return signatures:\n{}",
            signatures
                .iter()
                .map(|(name, signature)| format!("{name}: {signature:#?}"))
                .collect::<Vec<_>>()
                .join("\n\n")
        );
        assert!(
            signatures.iter().all(|(_, signature)| !matches!(
                signature.ret,
                Some(RuntimeClass::Ref {
                    kind: RefKind::Object,
                    ..
                })
            )),
            "Poseidon helpers must not leak internal object-backed carriers into visible return contracts:\n{}",
            signatures
                .iter()
                .map(|(name, signature)| format!("{name}: {signature:#?}"))
                .collect::<Vec<_>>()
                .join("\n\n")
        );
    }

    #[test]
    fn transport_shaped_returns_remain_visible_transport_returns() {
        let mut db = DriverDataBase::default();
        let file_url =
            Url::parse("file:///transport_shaped_returns_remain_visible_transport_returns.fe")
                .unwrap();
        db.workspace().touch(
            &mut db,
            file_url.clone(),
            Some(
                include_str!(
                    "../../../../fe/tests/fixtures/fe_test/mut_self_storage_receiver_regression.fe"
                )
                .to_string(),
            ),
        );
        let file = db
            .workspace()
            .get(&db, &file_url)
            .expect("file should be loaded");
        let top_mod = db.top_mod(file);
        let signature = runtime_signature_for_named_func(&db, top_mod, "value_mut");

        assert!(
            matches!(
                signature.ret,
                Some(RuntimeClass::Ref { .. } | RuntimeClass::RawAddr { .. })
            ),
            "transport-shaped returns must remain visible transport returns, not be normalized to by-value aggregates:\n{signature:#?}"
        );
        assert!(
            !matches!(signature.ret, Some(RuntimeClass::AggregateValue { .. })),
            "transport-shaped returns must not be normalized to by-value aggregate contracts:\n{signature:#?}"
        );
    }

    #[test]
    fn specialized_grant_callee_keeps_self_runtime_visible() {
        let mut db = DriverDataBase::default();
        let file_url =
            Url::parse("file:///specialized_grant_callee_keeps_self_runtime_visible.fe").unwrap();
        db.workspace().touch(
            &mut db,
            file_url.clone(),
            Some(include_str!("../../../../codegen/tests/fixtures/erc20.fe").to_string()),
        );
        let file = db
            .workspace()
            .get(&db, &file_url)
            .expect("file should be loaded");
        let top_mod = db.top_mod(file);
        let contract = top_mod
            .all_contracts(&db)
            .first()
            .copied()
            .expect("erc20 fixture should define a contract");
        let init_key = root_semantic_instance_key(&db, BodyOwner::ContractInit { contract })
            .unwrap_or_else(|err| panic!("failed to build root init semantic key: {err:?}"));
        let init =
            runtime_instance_for_semantic(&db, get_or_build_semantic_instance(&db, init_key));
        let grant = init
            .calls(&db)
            .iter()
            .find_map(|call| {
                let semantic = call.callee.key(&db).semantic(&db)?;
                match semantic.key(&db).owner(&db) {
                    BodyOwner::Func(func)
                        if func
                            .name(&db)
                            .to_opt()
                            .is_some_and(|name| name.data(&db) == "grant") =>
                    {
                        Some((semantic, call.callee))
                    }
                    _ => None,
                }
            })
            .expect("init should call grant");
        let (semantic, callee) = grant;
        let typed_body = semantic.key(&db).instantiate_typed_body(&db);
        let self_binding = typed_body
            .param_binding(0)
            .expect("grant typed body should keep self as the first param binding");
        let self_lowering = semantic_binding_lowering(&db, semantic, self_binding);
        let plans = runtime_visible_binding_plans(&db, semantic);
        let signature = callee.signature(&db);

        assert_eq!(
            plans.len(),
            3,
            "specialized grant callee should keep self + 2 explicit args runtime-visible:\nself_lowering={self_lowering:#?}\nplans={plans:#?}\nsignature={signature:#?}"
        );
        assert!(
            matches!(
                plans[0].plan,
                RuntimeParamPlan::Boundary(RuntimeBoundarySpec::ExactShape(RuntimeClass::Ref {
                    kind: RefKind::Provider { .. },
                    ..
                }))
            ),
            "specialized grant receiver should remain a visible provider transport:\nself_lowering={self_lowering:#?}\nplans={plans:#?}\nsignature={signature:#?}"
        );
        assert_eq!(
            signature.params.len(),
            3,
            "specialized grant runtime signature should keep self + 2 explicit args:\nself_lowering={self_lowering:#?}\nplans={plans:#?}\nsignature={signature:#?}"
        );
    }

    #[test]
    fn provider_backed_effect_bindings_keep_actualized_borrow_boundaries() {
        let mut db = DriverDataBase::default();
        let file_url = Url::parse(
            "file:///provider_backed_effect_bindings_keep_actualized_borrow_boundaries.fe",
        )
        .unwrap();
        db.workspace().touch(
            &mut db,
            file_url.clone(),
            Some(
                include_str!(
                    "../../../../fe/tests/fixtures/fe_test/reentrancy_mutex_storage_map.fe"
                )
                .to_string(),
            ),
        );
        let file = db
            .workspace()
            .get(&db, &file_url)
            .expect("file should be loaded");
        let top_mod = db.top_mod(file);
        let contract = contract_by_name(&db, top_mod, "B");
        for arm_idx in [1, 2, 4] {
            let semantic = get_or_build_semantic_instance(
                &db,
                root_semantic_instance_key(
                    &db,
                    BodyOwner::ContractRecvArm {
                        contract,
                        recv_idx: 0,
                        arm_idx,
                    },
                )
                .unwrap_or_else(|err| panic!("failed to build recv-arm semantic key: {err:?}")),
            );
            let binding = owner_effect_bindings(&db, semantic.key(&db).owner(&db))
                .into_iter()
                .next()
                .expect("Protected arm should keep one owner effect binding");
            let binding_ty = semantic_binding_ty(&db, semantic, binding);
            let plan = runtime_effect_binding_plan(&db, semantic, binding)
                .expect("guarded_balances should lower to a runtime effect binding plan");
            let RuntimeClass::Ref {
                kind: RefKind::Provider { space, .. },
                pointee,
                ..
            } = &plan.class
            else {
                panic!(
                    "guarded_balances should lower as a provider ref for arm {arm_idx}:\n{:#?}",
                    plan.class
                );
            };
            assert_eq!(*space, AddressSpaceKind::Storage);
            let RuntimeClass::AggregateValue { layout } = **pointee else {
                panic!(
                    "guarded_balances provider ref should point at its semantic value aggregate for arm {arm_idx}:\n{:#?}",
                    plan.class
                );
            };
            let Layout::Struct(layout_data) = layout.data(&db) else {
                panic!(
                    "guarded_balances provider pointee should use struct layout for arm {arm_idx}:\n{:#?}",
                    layout.data(&db)
                );
            };
            assert_eq!(layout_data.source_ty, binding_ty);
            assert_eq!(
                layout_data.fields.len(),
                1,
                "guarded_balances Mutex layout should expose the wrapped value field for arm {arm_idx}"
            );

            assert!(
                RuntimeBoundaryMatcher::class_satisfies_boundary(&plan.class, &plan.boundary),
                "provider-backed effect binding plan should keep an actualized boundary matching its chosen runtime class:\nplan={plan:#?}"
            );
            let _ = runtime_instance_for_semantic(&db, semantic).body(&db);
        }
    }

    #[test]
    fn provider_backed_method_call_inputs_keep_storage_receiver_transport() {
        let mut db = DriverDataBase::default();
        let file_url = Url::parse(
            "file:///provider_backed_method_call_inputs_keep_storage_receiver_transport.fe",
        )
        .unwrap();
        db.workspace().touch(
            &mut db,
            file_url.clone(),
            Some(
                include_str!(
                    "../../../../fe/tests/fixtures/fe_test/reentrancy_mutex_storage_map.fe"
                )
                .to_string(),
            ),
        );
        let file = db
            .workspace()
            .get(&db, &file_url)
            .expect("file should be loaded");
        let top_mod = db.top_mod(file);
        let contract = contract_by_name(&db, top_mod, "B");
        let semantic = get_or_build_semantic_instance(
            &db,
            root_semantic_instance_key(
                &db,
                BodyOwner::ContractRecvArm {
                    contract,
                    recv_idx: 0,
                    arm_idx: 4,
                },
            )
            .unwrap_or_else(|err| panic!("failed to build recv-arm semantic key: {err:?}")),
        );
        let instance = runtime_instance_for_semantic(&db, semantic);
        let typed_body = semantic.key(&db).typed_body(&db);
        let normalized = normalize_semantic_body(&db, semantic)
            .unwrap_or_else(|err| panic!("failed to normalize LockAndCheck: {err:?}"));
        let facts = BodyStaticFacts::new(&db, &normalized);
        let env = BodyEnv::new(&db, &normalized, typed_body, &facts);
        let params = instance.key(&db).params(&db);
        let mut returns = RuntimeReturnAnalysisCx::new(&db);
        let inferred = LocalStateInferer::new(
            env,
            params,
            &runtime_param_locals(&db, semantic, params),
            &mut returns,
        )
        .run();
        let mut checked_calls = Vec::new();
        for (block_idx, block) in normalized.blocks.iter().enumerate() {
            for (stmt_idx, stmt) in block.stmts.iter().enumerate() {
                let hir::analysis::semantic::NSStmtKind::Assign { expr, .. } = &stmt.kind else {
                    continue;
                };
                let NExpr::Call {
                    callee,
                    args,
                    effect_args,
                } = expr
                else {
                    continue;
                };
                let BodyOwner::Func(func) = callee.key.owner(&db) else {
                    continue;
                };
                let Some(name) = func.name(&db).to_opt().map(|name| name.data(&db)) else {
                    continue;
                };
                if !matches!(name.as_str(), "lock" | "is_locked" | "unlock") {
                    continue;
                }
                let ExprStaticFacts::Call(call_facts) =
                    facts.expr(block_idx, stmt_idx).unwrap_or_else(|| {
                        panic!("missing staged call facts for {block_idx}:{stmt_idx}")
                    })
                else {
                    panic!("{name} expression should keep staged call facts");
                };
                let receiver = args.first().map(|arg| arg.local);
                let (receiver_actual, receiver_materialized, selected, selected_classes) = {
                    let mut class_cache = InferClassCache::new(normalized.locals.len());
                    let mut evaluator =
                        RuntimeValueEvaluator::new(env, &inferred.carriers, Some(&mut class_cache));
                    let receiver_actual = receiver.and_then(|local| evaluator.actual_value(local));
                    let receiver_materialized =
                        receiver.and_then(|local| evaluator.materialize(local));
                    let selected =
                        evaluator.selected_call_inputs(args, effect_args, &call_facts.input_plan);
                    let selected_classes = selected
                        .iter()
                        .map(|arg| arg.class.clone())
                        .collect::<Vec<_>>();
                    (
                        receiver_actual,
                        receiver_materialized,
                        selected,
                        selected_classes,
                    )
                };
                let selected_return = returns.return_class_for_key(RuntimeInstanceKey::new(
                    &db,
                    RuntimeInstanceSource::Semantic(call_facts.semantic),
                    selected_classes,
                ));
                if !selected.is_empty() {
                    assert!(
                        matches!(
                            selected.first().map(|arg| &arg.class),
                            Some(
                                RuntimeClass::Ref {
                                    kind: RefKind::Provider {
                                        space: AddressSpaceKind::Storage,
                                        ..
                                    },
                                    ..
                                } | RuntimeClass::RawAddr {
                                    space: AddressSpaceKind::Storage,
                                    ..
                                }
                            )
                        ),
                        "provider-backed mutex receiver call input should preserve storage transport for `{name}`:\nreceiver_actual={receiver_actual:#?}\nreceiver_materialized={receiver_materialized:#?}\nselected_return={selected_return:#?}\nselected={selected:#?}",
                    );
                }
                if name == "lock" {
                    assert!(
                        matches!(
                            selected_return,
                            Some(
                                RuntimeClass::Ref {
                                    kind: RefKind::Provider {
                                        space: AddressSpaceKind::Storage,
                                        ..
                                    },
                                    ..
                                } | RuntimeClass::RawAddr {
                                    space: AddressSpaceKind::Storage,
                                    ..
                                }
                            )
                        ),
                        "storage-specialized `lock` should keep a storage transport return:\nselected_return={selected_return:#?}\nselected={selected:#?}",
                    );
                }
                checked_calls.push(name.to_string());
            }
        }

        assert_eq!(
            checked_calls,
            ["lock", "is_locked", "unlock"],
            "LockAndCheck should contain the expected mutex method calls",
        );
        let _ = instance.body(&db);
    }

    #[test]
    fn inner_take_call_uses_the_same_runtime_key_and_return_class_as_lowering() {
        let mut db = DriverDataBase::default();
        let file_url = Url::parse(
            "file:///inner_take_call_uses_the_same_runtime_key_and_return_class_as_lowering.fe",
        )
        .unwrap();
        db.workspace().touch(
            &mut db,
            file_url.clone(),
            Some(
                include_str!(
                    "../../../../fe/tests/fixtures/fe_test/view_param_local_ref_take_reverse.fe"
                )
                .to_string(),
            ),
        );
        let file = db
            .workspace()
            .get(&db, &file_url)
            .expect("file should be loaded");
        let top_mod = db.top_mod(file);
        let sum_last4 = semantic_instance_for_named_func(&db, top_mod, "sum_last4");
        let (semantic, instance) = runtime_instance_for_semantic(&db, sum_last4)
            .calls(&db)
            .iter()
            .find_map(|call| {
                let semantic = call.callee.key(&db).semantic(&db)?;
                match semantic.key(&db).owner(&db) {
                    BodyOwner::Func(func)
                        if func
                            .name(&db)
                            .to_opt()
                            .is_some_and(|name| name.data(&db) == "take_u256") =>
                    {
                        Some((semantic, call.callee))
                    }
                    _ => None,
                }
            })
            .expect("sum_last4 should call specialized take_u256");
        let typed_body = semantic.key(&db).typed_body(&db);
        let normalized = normalize_semantic_body(&db, semantic)
            .unwrap_or_else(|err| panic!("failed to normalize specialized take_u256: {err:?}"));
        let facts = BodyStaticFacts::new(&db, &normalized);
        let env = BodyEnv::new(&db, &normalized, typed_body, &facts);
        let params = instance.key(&db).params(&db);
        let mut returns = RuntimeReturnAnalysisCx::new(&db);
        let inferred = LocalStateInferer::new(
            env,
            params,
            &runtime_param_locals(&db, semantic, params),
            &mut returns,
        )
        .run();
        let (call_dst, args, effect_args, call_facts) = normalized
            .blocks
            .iter()
            .enumerate()
            .find_map(|(block_idx, block)| {
                block.stmts.iter().enumerate().find_map(|(stmt_idx, stmt)| {
                    let hir::analysis::semantic::NSStmtKind::Assign { dst, expr } = &stmt.kind
                    else {
                        return None;
                    };
                    let NExpr::Call {
                        callee,
                        args,
                        effect_args,
                    } = expr
                    else {
                        return None;
                    };
                    let BodyOwner::Func(func) = callee.key.owner(&db) else {
                        return None;
                    };
                    if func
                        .name(&db)
                        .to_opt()
                        .is_none_or(|name| name.data(&db) != "take")
                    {
                        return None;
                    }
                    let ExprStaticFacts::Call(call_facts) =
                        facts.expr(block_idx, stmt_idx).unwrap_or_else(|| {
                            panic!("missing staged call facts for {block_idx}:{stmt_idx}")
                        })
                    else {
                        panic!("inner take expression should keep staged call facts");
                    };
                    Some((*dst, args.clone(), effect_args.clone(), call_facts.clone()))
                })
            })
            .expect("specialized take_u256 should contain an inner call to take");
        let mut class_cache = InferClassCache::new(normalized.locals.len());
        let inferred_param_classes =
            RuntimeValueEvaluator::new(env, &inferred.carriers, Some(&mut class_cache))
                .selected_call_inputs(&args, &effect_args, &call_facts.input_plan)
                .into_iter()
                .map(|arg| arg.class)
                .collect::<Vec<_>>();
        let lowered_take = instance
            .calls(&db)
            .iter()
            .find_map(|call| {
                let semantic = call.callee.key(&db).semantic(&db)?;
                match semantic.key(&db).owner(&db) {
                    BodyOwner::Func(func)
                        if func
                            .name(&db)
                            .to_opt()
                            .is_some_and(|name| name.data(&db) == "take") =>
                    {
                        Some(call.callee)
                    }
                    _ => None,
                }
            })
            .expect("specialized take_u256 should lower an inner call to take");
        let inferred_dst_class = match inferred.carriers.get(call_dst.index()) {
            Some(RuntimeCarrier::Value(class)) => Some(class.clone()),
            Some(RuntimeCarrier::Erased) | None => None,
        };
        let lowered_dst_class = match instance.body(&db).locals.get(call_dst.index()) {
            Some(local) => match &local.carrier {
                RuntimeCarrier::Value(class) => Some(class.clone()),
                RuntimeCarrier::Erased => None,
            },
            None => None,
        };
        let lowered_return_class = returns.return_class_for_key(lowered_take.key(&db));

        assert_eq!(
            inferred_param_classes,
            *lowered_take.key(&db).params(&db),
            "infer-time call classification should build the same runtime key as lowering for take_u256 -> take:\ninferred_param_classes={inferred_param_classes:#?}\nlowered_key={:#?}",
            lowered_take.key(&db),
        );
        assert_eq!(
            inferred_dst_class, lowered_dst_class,
            "infer-time dst carrier should match the lowered call-result carrier for take_u256 -> take:\ninferred_dst_class={inferred_dst_class:#?}\nlowered_dst_class={lowered_dst_class:#?}",
        );
        assert_eq!(
            inferred_dst_class, lowered_return_class,
            "infer-time dst carrier should match the specialized callee return class for take_u256 -> take:\ninferred_dst_class={inferred_dst_class:#?}\nlowered_return_class={lowered_return_class:#?}",
        );
    }

    #[test]
    fn own_scalar_call_inputs_materialize_provider_backed_direct_values() {
        let mut db = DriverDataBase::default();
        let file_url = Url::parse(
            "file:///own_scalar_call_inputs_materialize_provider_backed_direct_values.fe",
        )
        .unwrap();
        db.workspace().touch(
            &mut db,
            file_url.clone(),
            Some(
                include_str!(
                    "../../../../fe/tests/fixtures/fe_test/contract_field_mut_borrow_matrix.fe"
                )
                .to_string(),
            ),
        );
        let file = db
            .workspace()
            .get(&db, &file_url)
            .expect("file should be loaded");
        let top_mod = db.top_mod(file);
        let contract = contract_by_name(&db, top_mod, "C");
        let semantic = get_or_build_semantic_instance(
            &db,
            root_semantic_instance_key(
                &db,
                BodyOwner::ContractRecvArm {
                    contract,
                    recv_idx: 0,
                    arm_idx: 2,
                },
            )
            .unwrap_or_else(|err| panic!("failed to build recv-arm semantic key: {err:?}")),
        );
        let instance = runtime_instance_for_semantic(&db, semantic);
        let typed_body = semantic.key(&db).typed_body(&db);
        let normalized = normalize_semantic_body(&db, semantic)
            .unwrap_or_else(|err| panic!("failed to normalize SelectAndMutate: {err:?}"));
        let facts = BodyStaticFacts::new(&db, &normalized);
        let env = BodyEnv::new(&db, &normalized, typed_body, &facts);
        let params = instance.key(&db).params(&db);
        let mut returns = RuntimeReturnAnalysisCx::new(&db);
        let inferred = LocalStateInferer::new(
            env,
            params,
            &runtime_param_locals(&db, semantic, params),
            &mut returns,
        )
        .run();
        let (args, effect_args, call_facts) = normalized
            .blocks
            .iter()
            .enumerate()
            .find_map(|(block_idx, block)| {
                block.stmts.iter().enumerate().find_map(|(stmt_idx, stmt)| {
                    let hir::analysis::semantic::NSStmtKind::Assign { expr, .. } = &stmt.kind
                    else {
                        return None;
                    };
                    let NExpr::Call {
                        callee,
                        args,
                        effect_args,
                    } = expr
                    else {
                        return None;
                    };
                    let BodyOwner::Func(func) = callee.key.owner(&db) else {
                        return None;
                    };
                    if func
                        .name(&db)
                        .to_opt()
                        .is_none_or(|name| name.data(&db) != "set_scaled")
                    {
                        return None;
                    }
                    let ExprStaticFacts::Call(call_facts) =
                        facts.expr(block_idx, stmt_idx).unwrap_or_else(|| {
                            panic!("missing staged call facts for {block_idx}:{stmt_idx}")
                        })
                    else {
                        panic!("set_scaled expression should keep staged call facts");
                    };
                    Some((args.clone(), effect_args.clone(), call_facts.clone()))
                })
            })
            .expect("SelectAndMutate should call set_scaled");
        let mut class_cache = InferClassCache::new(normalized.locals.len());
        let inferred_param_classes =
            RuntimeValueEvaluator::new(env, &inferred.carriers, Some(&mut class_cache))
                .selected_call_inputs(&args, &effect_args, &call_facts.input_plan)
                .into_iter()
                .map(|arg| arg.class)
                .collect::<Vec<_>>();
        let lowered_set_scaled = instance
            .calls(&db)
            .iter()
            .find_map(|call| {
                let semantic = call.callee.key(&db).semantic(&db)?;
                match semantic.key(&db).owner(&db) {
                    BodyOwner::Func(func)
                        if func
                            .name(&db)
                            .to_opt()
                            .is_some_and(|name| name.data(&db) == "set_scaled") =>
                    {
                        Some(call.callee)
                    }
                    _ => None,
                }
            })
            .expect("SelectAndMutate should lower a call to set_scaled");

        assert!(
            matches!(
                inferred_param_classes.first(),
                Some(RuntimeClass::Scalar(_))
            ),
            "provider-backed direct values passed to own u256 params must materialize by value:\nclasses={inferred_param_classes:#?}",
        );
        assert!(
            matches!(
                lowered_set_scaled.key(&db).params(&db).first(),
                Some(RuntimeClass::Scalar(_))
            ),
            "specialized set_scaled runtime key must keep its first param by-value:\nkey={:#?}",
            lowered_set_scaled.key(&db),
        );
    }

    #[test]
    fn transport_shaped_method_return_keeps_runtime_visible_signature() {
        let mut db = DriverDataBase::default();
        let file_url =
            Url::parse("file:///transport_shaped_method_return_keeps_runtime_visible_signature.fe")
                .unwrap();
        db.workspace().touch(
            &mut db,
            file_url.clone(),
            Some(
                include_str!(
                    "../../../../fe/tests/fixtures/fe_test/contract_field_mut_borrow_matrix.fe"
                )
                .to_string(),
            ),
        );
        let file = db
            .workspace()
            .get(&db, &file_url)
            .expect("file should be loaded");
        let top_mod = db.top_mod(file);
        let signature = runtime_signature_for_named_func(&db, top_mod, "pick_ac_mut");

        assert!(
            matches!(
                signature.ret,
                Some(
                    RuntimeClass::Ref {
                        kind: RefKind::Provider { .. },
                        ..
                    } | RuntimeClass::RawAddr {
                        space: AddressSpaceKind::Storage,
                        ..
                    }
                )
            ),
            "mut-returning method signatures must keep provider/storage transport, not degrade to object refs:\n{signature:#?}"
        );
    }
}
