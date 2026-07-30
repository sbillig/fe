use camino::Utf8PathBuf;
use common::indexmap::IndexMap;
use fe_hir::{
    analysis::{
        semantic::{
            NBorrowRoot, NSPlaceRoot, NSStmtKind, NormalizedBindingLowering, NormalizedBodyFacts,
            SCallReturnProjectionStep, SExpr, SStmtKind, get_or_build_semantic_instance,
            identity_semantic_instance_key, normalize_semantic_body,
        },
        ty::{
            closure::implemented_closure_call_trait,
            const_ty::CallableInputLayoutHoleOrigin,
            corelib::resolve_core_trait,
            trait_def::TraitInstId,
            trait_resolution::{GoalSatisfiability, TraitSolveCx, is_goal_satisfiable},
            ty_check::{
                BodyOwner, ClosureCaptureAccess, LocalBinding, ParamSite, ReturnIndexSource,
                ReturnProjectionStep, ReturnProvenance, ReturnSource, TypedCallableBody,
                check_func_body,
            },
            ty_def::{ClosureCallMode, TyId},
            ty_is_copy,
        },
    },
    hir_def::{Func, ItemKind, TopLevelMod},
    projection::Projection,
    test_db::{HirAnalysisTestDb, format_diagnostics},
};

fn find_func<'db>(db: &'db HirAnalysisTestDb, top_mod: TopLevelMod<'db>, name: &str) -> Func<'db> {
    top_mod
        .all_items(db)
        .iter()
        .find_map(|item| match item {
            ItemKind::Func(func)
                if func
                    .name(db)
                    .to_opt()
                    .is_some_and(|func_name| func_name.data(db) == name) =>
            {
                Some(*func)
            }
            _ => None,
        })
        .unwrap_or_else(|| panic!("missing function `{name}`"))
}

#[test]
fn unpacked_mut_closure_param_retains_binding_and_lowers_writes_as_stores() {
    let mut db = HirAnalysisTestDb::default();
    let file = db.new_stand_alone(
        Utf8PathBuf::from("unpacked_mut_closure_param.fe"),
        r#"
fn probe() {
    let write = |target: mut| {
        target = 42
    }
    let mut value: u256 = 0
    write.call(mut value)
}
"#,
    );
    let (top_mod, _) = db.top_mod(file);
    db.assert_no_diags(top_mod);

    let probe = find_func(&db, top_mod, "probe");
    let probe = get_or_build_semantic_instance(
        &db,
        identity_semantic_instance_key(&db, BodyOwner::Func(probe)),
    );
    let closure = probe
        .callees(&db)
        .iter()
        .find_map(|callee| {
            matches!(callee.key.owner(&db), BodyOwner::Closure { .. })
                .then(|| get_or_build_semantic_instance(&db, callee.key))
        })
        .expect("specialized closure instance");
    let normalized = normalize_semantic_body(&db, closure).expect("normalized closure body");
    let (local_id, local) = normalized
        .locals
        .iter()
        .enumerate()
        .find(|(_, local)| {
            matches!(
                local.source,
                Some(LocalBinding::Param {
                    site: ParamSite::Closure(_),
                    idx: 0,
                    ..
                })
            )
        })
        .map(|(idx, local)| {
            (
                fe_hir::analysis::semantic::SLocalId::from_u32(idx as u32),
                local,
            )
        })
        .expect("unpacked logical closure parameter");
    let NormalizedBindingLowering::CarrierLocal {
        root: Some(root), ..
    } = local.lowering
    else {
        panic!("mut closure parameter must be a rooted carrier: {local:?}");
    };
    assert!(
        matches!(
            normalized.root(root),
            Some(NBorrowRoot::LocalSlot { local }) if *local == local_id
        ),
        "an unpacked logical parameter is not a physical ABI parameter",
    );
    assert!(
        normalized.blocks.iter().any(|block| {
            block.stmts.iter().any(|stmt| {
                matches!(
                    &stmt.kind,
                    NSStmtKind::Store {
                        dst:
                            fe_hir::analysis::semantic::NSPlace {
                                root: NSPlaceRoot::CarrierDerefLocal(local),
                                ..
                            },
                        ..
                    } if *local == local_id
                )
            })
        }),
        "assignment through a mut closure parameter must lower as a store",
    );
}

fn source(
    origin: CallableInputLayoutHoleOrigin,
    projection: Vec<ReturnProjectionStep>,
) -> ReturnSource {
    ReturnSource {
        result_projection: Vec::new(),
        origin,
        projection,
    }
}

#[test]
fn closure_return_provenance_uses_its_own_inputs() {
    let mut db = HirAnalysisTestDb::default();
    let file = db.new_stand_alone(
        Utf8PathBuf::from("closure_return_provenance.fe"),
        r#"
struct Boxed {
    value: u256,
}

fn make(_ captured: own Boxed) {
    let from_param = |value: own Boxed| -> Boxed { value }
    let from_capture = |_ unit: own ()| -> Boxed { captured }
}
"#,
    );
    let (top_mod, _) = db.top_mod(file);
    db.assert_no_diags(top_mod);

    let func = find_func(&db, top_mod, "make");
    let (_, typed_body) = check_func_body(&db, func);
    let mut saw_param = false;
    let mut saw_capture = false;
    for (_, info) in typed_body.closure_infos() {
        let callable = TypedCallableBody::new(BodyOwner::closure(&db, info.ty), typed_body);
        let ReturnProvenance::Forwarded(sources) = callable.return_provenance(&db) else {
            panic!(
                "closure return should be forwarded: {:?}",
                callable.return_provenance(&db)
            );
        };
        match info.captures.len() {
            0 => {
                assert_eq!(
                    sources,
                    vec![source(
                        CallableInputLayoutHoleOrigin::ValueParam(1),
                        vec![ReturnProjectionStep::Field(0)]
                    )]
                );
                saw_param = true;
            }
            1 => {
                assert_eq!(
                    sources,
                    vec![source(
                        CallableInputLayoutHoleOrigin::Receiver,
                        vec![ReturnProjectionStep::Field(0)]
                    )]
                );
                saw_capture = true;
            }
            count => panic!("unexpected capture count: {count}"),
        }
    }
    assert!(saw_param && saw_capture);
}

#[test]
fn nested_closure_does_not_change_parent_return_provenance() {
    let mut db = HirAnalysisTestDb::default();
    let file = db.new_stand_alone(
        Utf8PathBuf::from("parent_return_provenance.fe"),
        r#"
struct Boxed {
    value: u256,
}

fn passthrough(_ value: own Boxed) -> Boxed {
    let unused = |_ unit: own ()| -> u256 { 1 }
    value
}
"#,
    );
    let (top_mod, _) = db.top_mod(file);
    db.assert_no_diags(top_mod);

    let func = find_func(&db, top_mod, "passthrough");
    let (_, typed_body) = check_func_body(&db, func);
    let callable = TypedCallableBody::new(BodyOwner::Func(func), typed_body);
    assert_eq!(
        callable.return_provenance(&db),
        ReturnProvenance::Forwarded(vec![source(
            CallableInputLayoutHoleOrigin::ValueParam(0),
            Vec::new()
        )])
    );
}

#[test]
fn nested_noncopy_capture_makes_every_owning_closure_consuming() {
    let mut db = HirAnalysisTestDb::default();
    let file = db.new_stand_alone(
        Utf8PathBuf::from("nested_noncopy_capture.fe"),
        r#"
struct Boxed {
    value: u256,
}

fn make() {
    let boxed = Boxed { value: 42 }
    let outer = |_ unit: own ()| -> Boxed {
        let inner = |_ unit: own ()| -> Boxed { boxed }
        inner.call_once(())
    }
}
"#,
    );
    let (top_mod, _) = db.top_mod(file);
    db.assert_no_diags(top_mod);

    let func = find_func(&db, top_mod, "make");
    let (_, typed_body) = check_func_body(&db, func);
    let call_modes = typed_body
        .closure_infos()
        .map(|(_, info)| info.ty.call_mode(&db))
        .collect::<Vec<_>>();
    assert_eq!(
        call_modes,
        vec![ClosureCallMode::Consuming, ClosureCallMode::Consuming]
    );
}

#[test]
fn nested_copy_capture_resolves_reusable_after_outer_parameter_inference() {
    let mut db = HirAnalysisTestDb::default();
    let file = db.new_stand_alone(
        Utf8PathBuf::from("nested_inferred_copy_capture.fe"),
        r#"
fn probe() -> u256 {
    let make = |value: own| {
        let get = || value
        get
    }
    let get = make.call_once(21 as u256)
    get.call() + get.call()
}
"#,
    );
    let (top_mod, _) = db.top_mod(file);
    db.assert_no_diags(top_mod);

    let probe = find_func(&db, top_mod, "probe");
    let (_, typed_body) = check_func_body(&db, probe);
    let call_modes = typed_body
        .closure_infos()
        .map(|(_, info)| info.ty.call_mode(&db))
        .collect::<Vec<_>>();
    assert_eq!(
        call_modes,
        vec![ClosureCallMode::Reusable, ClosureCallMode::Reusable]
    );
}

#[test]
fn nested_copy_capture_call_waits_for_outer_parameter_inference() {
    let mut db = HirAnalysisTestDb::default();
    let file = db.new_stand_alone(
        Utf8PathBuf::from("nested_deferred_copy_capture_call.fe"),
        r#"
fn probe() -> u256 {
    let make = |value: own| {
        let get = || value
        get.call() + get.call()
    }
    make.call_once(21 as u256)
}
"#,
    );
    let (top_mod, _) = db.top_mod(file);
    db.assert_no_diags(top_mod);

    let probe = find_func(&db, top_mod, "probe");
    let (_, typed_body) = check_func_body(&db, probe);
    assert!(
        typed_body
            .closure_infos()
            .all(|(_, info)| info.ty.call_mode(&db) == ClosureCallMode::Reusable)
    );
}

#[test]
fn nested_copy_pattern_capture_resolves_reusable_after_outer_inference() {
    let mut db = HirAnalysisTestDb::default();
    let file = db.new_stand_alone(
        Utf8PathBuf::from("nested_inferred_copy_pattern_capture.fe"),
        r#"
fn probe() -> u256 {
    let make = |pair: own| {
        let get = || {
            let (value,) = pair
            value
        }
        get
    }
    let get = make.call_once((21 as u256,))
    get.call() + get.call()
}
"#,
    );
    let (top_mod, _) = db.top_mod(file);
    db.assert_no_diags(top_mod);

    let probe = find_func(&db, top_mod, "probe");
    let (_, typed_body) = check_func_body(&db, probe);
    assert!(
        typed_body
            .closure_infos()
            .all(|(_, info)| info.ty.call_mode(&db) == ClosureCallMode::Reusable)
    );
}

#[test]
fn inferred_copy_construction_propagates_reusability_through_nested_closures() {
    let mut db = HirAnalysisTestDb::default();
    let file = db.new_stand_alone(
        Utf8PathBuf::from("nested_inferred_copy_capture_propagation.fe"),
        r#"
fn probe() -> u256 {
    let make = |value: own| {
        let outer = || {
            let inner = || value
            inner
        }
        outer
    }
    let outer = make.call_once(21 as u256)
    let left = outer.call()
    let right = outer.call()
    left.call() + right.call()
}
"#,
    );
    let (top_mod, _) = db.top_mod(file);
    db.assert_no_diags(top_mod);

    let probe = find_func(&db, top_mod, "probe");
    let (_, typed_body) = check_func_body(&db, probe);
    assert!(
        typed_body
            .closure_infos()
            .all(|(_, info)| info.ty.call_mode(&db) == ClosureCallMode::Reusable)
    );
}

#[test]
fn nested_inferred_copy_capture_fn_obligation_waits_for_outer_inference() {
    let mut db = HirAnalysisTestDb::default();
    let file = db.new_stand_alone(
        Utf8PathBuf::from("nested_inferred_copy_capture_fn_bound.fe"),
        r#"
use core::functional::Fn

fn invoke<T, F: Fn<(), T>>(_ function: F) -> T {
    function.call()
}

fn probe() -> u256 {
    let make = |value: own| {
        let get = || value
        invoke(get)
    }
    make.call_once(42 as u256)
}
"#,
    );
    let (top_mod, _) = db.top_mod(file);
    db.assert_no_diags(top_mod);
}

#[test]
fn nested_inferred_noncopy_capture_does_not_satisfy_fn_obligation() {
    let mut db = HirAnalysisTestDb::default();
    let file = db.new_stand_alone(
        Utf8PathBuf::from("nested_inferred_noncopy_capture_fn_bound.fe"),
        r#"
use core::functional::Fn

struct Boxed {
    value: u256,
}

fn invoke<T, F: Fn<(), T>>(_ function: F) -> T {
    function.call()
}

fn probe() -> Boxed {
    let make = |value: own| {
        let take = || value
        invoke(take)
    }
    make.call_once(Boxed { value: 42 })
}
"#,
    );
    let (top_mod, _) = db.top_mod(file);
    let rendered = format_diagnostics(&db, &db.run_on_top_mod(top_mod));
    assert!(
        rendered.contains("trait bound is not satisfied") && rendered.contains("Fn"),
        "{rendered}"
    );
}

#[test]
fn concrete_specialization_does_not_reclassify_an_unbounded_generic_capture() {
    let mut db = HirAnalysisTestDb::default();
    let file = db.new_stand_alone(
        Utf8PathBuf::from("generic_capture_specialization.fe"),
        r#"
fn generic<T>(_ value: own T) -> T {
    let take = || value
    take.call()
}

fn probe() -> u256 {
    generic(42 as u256)
}
"#,
    );
    let (top_mod, _) = db.top_mod(file);
    let rendered = format_diagnostics(&db, &db.run_on_top_mod(top_mod));
    assert!(
        rendered.contains("no method named `call` found"),
        "{rendered}"
    );
}

#[test]
fn copy_bound_makes_a_generic_capture_reusable() {
    let mut db = HirAnalysisTestDb::default();
    let file = db.new_stand_alone(
        Utf8PathBuf::from("copy_bound_generic_capture.fe"),
        r#"
fn generic<T: Copy>(_ value: own T) -> T {
    let read = || value
    read.call()
    read.call()
}

fn probe() -> u256 {
    generic(42 as u256)
}
"#,
    );
    let (top_mod, _) = db.top_mod(file);
    db.assert_no_diags(top_mod);

    let generic = find_func(&db, top_mod, "generic");
    let (_, typed_body) = check_func_body(&db, generic);
    assert!(
        typed_body
            .closure_infos()
            .all(|(_, info)| info.ty.call_mode(&db) == ClosureCallMode::Reusable)
    );
}

#[test]
fn nested_inferred_noncopy_capture_remains_consuming() {
    let mut db = HirAnalysisTestDb::default();
    let file = db.new_stand_alone(
        Utf8PathBuf::from("nested_inferred_noncopy_capture.fe"),
        r#"
struct Boxed {
    value: u256,
}

fn probe() -> u256 {
    let make = |value: own| {
        let take = || value
        take
    }
    let take = make.call_once(Boxed { value: 42 })
    take.call_once().value
}
"#,
    );
    let (top_mod, _) = db.top_mod(file);
    db.assert_no_diags(top_mod);

    let probe = find_func(&db, top_mod, "probe");
    let (_, typed_body) = check_func_body(&db, probe);
    assert_eq!(
        typed_body
            .closure_infos()
            .filter(|(_, info)| !info.ty.captures(&db).is_empty())
            .map(|(_, info)| info.ty.call_mode(&db))
            .collect::<Vec<_>>(),
        vec![ClosureCallMode::Consuming]
    );
}

#[test]
fn closure_copyability_is_structural_over_capture_types() {
    let mut db = HirAnalysisTestDb::default();
    let file = db.new_stand_alone(
        Utf8PathBuf::from("closure_copy_capture.fe"),
        r#"
fn probe() -> u256 {
    let offset = 1
    let add = |value: own u256| value + offset
    let copied = add

    let forty_two = || 42 as u256
    let copied_empty = forty_two

    add.call(20) + copied.call(20) + forty_two.call() + copied_empty.call() - 84
}
"#,
    );
    let (top_mod, _) = db.top_mod(file);
    db.assert_no_diags(top_mod);
    let probe = find_func(&db, top_mod, "probe");
    let (_, typed_body) = check_func_body(&db, probe);
    assert!(
        typed_body.closure_infos().all(|(_, info)| ty_is_copy(
            &db,
            probe.scope(),
            TyId::closure(&db, info.ty),
            typed_body.assumptions(),
        )),
        "empty closures and closures with only Copy captures must be Copy"
    );

    let file = db.new_stand_alone(
        Utf8PathBuf::from("closure_noncopy_capture.fe"),
        r#"
struct Boxed {
    value: u256,
}

fn probe() {
    let boxed = Boxed { value: 42 }
    let read = || boxed.value
    let moved = read
    read.call()
}
"#,
    );
    let (top_mod, _) = db.top_mod(file);
    db.assert_no_diags(top_mod);
    let probe = find_func(&db, top_mod, "probe");
    let (_, typed_body) = check_func_body(&db, probe);
    let closure = typed_body
        .closure_infos()
        .next()
        .expect("non-Copy capture closure")
        .1;
    assert!(!ty_is_copy(
        &db,
        probe.scope(),
        TyId::closure(&db, closure.ty),
        typed_body.assumptions(),
    ));
}

#[test]
fn discarded_noncopy_capture_makes_closure_consuming() {
    let mut db = HirAnalysisTestDb::default();
    let file = db.new_stand_alone(
        Utf8PathBuf::from("discarded_noncopy_capture.fe"),
        r#"
struct Boxed {
    value: u256,
}

fn make() {
    let boxed = Boxed { value: 42 }
    let consume = |_ unit: own ()| -> () {
        boxed
        ()
    }
}
"#,
    );
    let (top_mod, _) = db.top_mod(file);
    db.assert_no_diags(top_mod);

    let func = find_func(&db, top_mod, "make");
    let (_, typed_body) = check_func_body(&db, func);
    let closure = typed_body
        .closure_infos()
        .next()
        .expect("missing closure")
        .1;
    assert_eq!(closure.ty.call_mode(&db), ClosureCallMode::Consuming);
    assert_eq!(closure.captures[0].access, ClosureCaptureAccess::Move);
}

#[test]
fn borrow_parameter_closures_implement_call_traits_with_capability_argument_packs() {
    let mut db = HirAnalysisTestDb::default();
    let file = db.new_stand_alone(
        Utf8PathBuf::from("closure_borrow_param_trait.fe"),
        r#"
fn make() {
    let borrow = |value: mut u256| -> u256 { value }
}
"#,
    );
    let (top_mod, _) = db.top_mod(file);
    let func = find_func(&db, top_mod, "make");
    let (_, typed_body) = check_func_body(&db, func);
    let closure = typed_body
        .closure_infos()
        .next()
        .expect("missing borrow-parameter closure")
        .1
        .ty;
    for trait_name in ["Fn", "FnOnce"] {
        let trait_ = resolve_core_trait(&db, func.scope(), &["functional", trait_name])
            .unwrap_or_else(|| panic!("missing core `{trait_name}` trait"));
        assert!(
            implemented_closure_call_trait(
                &db,
                func.scope(),
                typed_body.assumptions(),
                closure,
                trait_,
            )
            .is_some(),
            "reusable borrow-parameter closure must implement `{trait_name}`",
        );
    }
}

#[test]
fn table_solver_enforces_closure_call_capabilities_and_signature() {
    let mut db = HirAnalysisTestDb::default();
    let file = db.new_stand_alone(
        Utf8PathBuf::from("closure_table_solver.fe"),
        r#"
struct Boxed {
    value: u256,
}

fn make() {
    let read_box = Boxed { value: 1 }
    let take_box = Boxed { value: 2 }
    let reusable = || -> u256 { read_box.value }
    let consuming = || -> Boxed { take_box }
    let unary = |value: own u256| -> u256 { value }
}
"#,
    );
    let (top_mod, _) = db.top_mod(file);
    let func = find_func(&db, top_mod, "make");
    let (_, typed_body) = check_func_body(&db, func);
    let closures = typed_body
        .closure_infos()
        .map(|(_, info)| info.ty)
        .collect::<Vec<_>>();
    let reusable = closures
        .iter()
        .copied()
        .find(|closure| {
            closure.call_mode(&db) == ClosureCallMode::Reusable && closure.params(&db).is_empty()
        })
        .expect("reusable closure");
    let consuming = closures
        .iter()
        .copied()
        .find(|closure| closure.call_mode(&db) == ClosureCallMode::Consuming)
        .expect("consuming closure");
    let unary = closures
        .iter()
        .copied()
        .find(|closure| closure.params(&db).len() == 1)
        .expect("unary closure");
    let assumptions = typed_body.assumptions();
    let solve_cx = TraitSolveCx::new(&db, func.scope()).with_assumptions(assumptions);

    let solve = |closure, trait_name, args, ret| {
        let trait_ = resolve_core_trait(&db, func.scope(), &["functional", trait_name])
            .unwrap_or_else(|| panic!("missing core `{trait_name}` trait"));
        let goal = TraitInstId::new(
            &db,
            trait_,
            vec![TyId::closure(&db, closure), args, ret],
            IndexMap::new(),
        );
        is_goal_satisfiable(&db, solve_cx, goal)
    };

    for (closure, trait_name, expected) in [
        (reusable, "Fn", true),
        (reusable, "FnOnce", true),
        (consuming, "Fn", false),
        (consuming, "FnOnce", true),
    ] {
        let result = solve(
            closure,
            trait_name,
            closure.args_pack_ty(&db),
            closure.ret_ty(&db),
        );
        assert_eq!(
            matches!(result, GoalSatisfiability::Satisfied(_)),
            expected,
            "unexpected solver result for {trait_name}: {result:?}",
        );
    }

    assert!(matches!(
        solve(
            reusable,
            "Fn",
            unary.args_pack_ty(&db),
            reusable.ret_ty(&db),
        ),
        GoalSatisfiability::UnSat(_)
    ));
    assert!(matches!(
        solve(reusable, "Fn", reusable.args_pack_ty(&db), TyId::bool(&db),),
        GoalSatisfiability::UnSat(_)
    ));
}

#[test]
fn deferred_closure_expressions_apply_contextual_result_capabilities() {
    let mut db = HirAnalysisTestDb::default();
    let file = db.new_stand_alone(
        Utf8PathBuf::from("deferred_closure_result_capabilities.fe"),
        r#"
use core::functional::Fn

struct Boxed {
    value: u256,
}

impl Boxed {
    fn get(self) -> u256 {
        self.value
    }
}

struct Choice {}

trait Choose<T> {
    fn choose(self) -> T
}

impl Choose<u256> for Choice {
    fn choose(self) -> u256 {
        42
    }
}

impl Choose<bool> for Choice {
    fn choose(self) -> bool {
        true
    }
}

fn pair<T, F: Fn<(view T,), view u256>, G: Fn<(view T,), u256>>(
    _ first: F,
    _ witness: G,
) {}

fn probe() {
    pair(
        |value| value.value,
        |value: Boxed| value.value,
    )
    pair(
        |value| value.get(),
        |value: Boxed| value.value,
    )
    pair(
        |value| value as u256,
        |value: u8| value as u256,
    )
    let selected: view u256 = Choice {}.choose()
}
"#,
    );
    let (top_mod, _) = db.top_mod(file);
    db.assert_no_diags(top_mod);
}

#[test]
fn contextual_closure_inference_handles_methods_and_view_callable_params() {
    let mut db = HirAnalysisTestDb::default();
    let file = db.new_stand_alone(
        Utf8PathBuf::from("closure_method_context.fe"),
        r#"
use core::functional::Fn

struct Boxed {
    value: u256,
}

struct Apply {}

impl Apply {
    fn call_with<T, F: Fn<(view T,), u256>>(
        self,
        _ value: own T,
        _ function: view F,
    ) -> u256 {
        function.call(value)
    }
}

fn probe() -> u256 {
    Apply {}.call_with(Boxed { value: 42 }, |value| value.value)
}
"#,
    );
    let (top_mod, _) = db.top_mod(file);
    db.assert_no_diags(top_mod);
}

#[test]
fn contextual_closure_inference_flows_through_tuple_arguments() {
    let mut db = HirAnalysisTestDb::default();
    let file = db.new_stand_alone(
        Utf8PathBuf::from("closure_tuple_context.fe"),
        r#"
use core::functional::Fn

struct Boxed {
    value: u256,
}

fn apply_wrapped<F: Fn<(Boxed,), u256>>(_ wrapped: own (F,)) -> u256 {
    wrapped.0.call(Boxed { value: 42 })
}

fn probe() -> u256 {
    apply_wrapped((|value: own| value.value,))
}
"#,
    );
    let (top_mod, _) = db.top_mod(file);
    db.assert_no_diags(top_mod);
}

#[test]
fn contextual_closure_inference_flows_through_struct_arguments() {
    let mut db = HirAnalysisTestDb::default();
    let file = db.new_stand_alone(
        Utf8PathBuf::from("closure_struct_context.fe"),
        r#"
use core::functional::Fn

struct Boxed {
    value: u256,
}

struct Wrapped<F> {
    function: F,
}

fn apply_wrapped<F: Fn<(Boxed,), u256>>(_ wrapped: own Wrapped<F>) -> u256 {
    wrapped.function.call(Boxed { value: 42 })
}

fn probe() -> u256 {
    apply_wrapped(Wrapped { function: |value: own| value.value })
}
"#,
    );
    let (top_mod, _) = db.top_mod(file);
    db.assert_no_diags(top_mod);
}

#[test]
fn contextual_closure_inference_flows_through_array_arguments() {
    let mut db = HirAnalysisTestDb::default();
    let file = db.new_stand_alone(
        Utf8PathBuf::from("closure_array_context.fe"),
        r#"
use core::functional::Fn

struct Boxed {
    value: u256,
}

fn apply_wrapped<F: Fn<(Boxed,), u256>>(_ functions: own [F; 1]) -> u256 {
    functions[0].call(Boxed { value: 42 })
}

fn probe() -> u256 {
    apply_wrapped([|value: own| value.value])
}
"#,
    );
    let (top_mod, _) = db.top_mod(file);
    db.assert_no_diags(top_mod);
}

#[test]
fn contextual_closure_inference_flows_through_enum_arguments() {
    let mut db = HirAnalysisTestDb::default();
    let file = db.new_stand_alone(
        Utf8PathBuf::from("closure_enum_context.fe"),
        r#"
use core::functional::Fn

struct Boxed {
    value: u256,
}

enum Wrapped<F> {
    Function(F),
    Empty,
}

fn apply_wrapped<F: Fn<(Boxed,), u256>>(_ wrapped: own Wrapped<F>) -> u256 {
    match wrapped {
        Wrapped::Function(function) => function.call(Boxed { value: 42 }),
        Wrapped::Empty => 0,
    }
}

fn probe() -> u256 {
    apply_wrapped(Wrapped::Function(|value: own| value.value))
}
"#,
    );
    let (top_mod, _) = db.top_mod(file);
    db.assert_no_diags(top_mod);
}

#[test]
fn contextual_closure_inference_flows_through_block_arguments() {
    let mut db = HirAnalysisTestDb::default();
    let file = db.new_stand_alone(
        Utf8PathBuf::from("closure_block_context.fe"),
        r#"
use core::functional::Fn

struct Boxed {
    value: u256,
}

fn apply<F: Fn<(Boxed,), u256>>(_ function: F) -> u256 {
    function.call(Boxed { value: 42 })
}

fn probe() -> u256 {
    apply({
        |value: own| value.value
    })
}
"#,
    );
    let (top_mod, _) = db.top_mod(file);
    db.assert_no_diags(top_mod);
}

#[test]
fn inferred_mut_closure_parameter_fields_are_assignable_after_call_resolution() {
    let mut db = HirAnalysisTestDb::default();
    let file = db.new_stand_alone(
        Utf8PathBuf::from("closure_inferred_mut_field_assignment.fe"),
        r#"
struct Boxed {
    value: u256,
}

fn probe() {
    let mut boxed = Boxed { value: 0 }
    let set = |target: mut| {
        target.value = 42
    }
    set.call(mut boxed)
}
"#,
    );
    let (top_mod, _) = db.top_mod(file);
    db.assert_no_diags(top_mod);
}

#[test]
fn inferred_owned_closure_parameter_fields_can_be_borrowed_after_call_resolution() {
    let mut db = HirAnalysisTestDb::default();
    let file = db.new_stand_alone(
        Utf8PathBuf::from("closure_inferred_owned_field_borrow.fe"),
        r#"
struct Boxed {
    value: u256,
}

fn probe() {
    let borrow = |boxed: own| -> ref u256 { ref boxed.value }
    let returned = borrow.call_once(Boxed { value: 42 })
}
"#,
    );
    let (top_mod, _) = db.top_mod(file);
    db.assert_no_diags(top_mod);
}

#[test]
fn deferred_closure_place_checks_use_resolved_capability_mutability() {
    let mut db = HirAnalysisTestDb::default();
    let file = db.new_stand_alone(
        Utf8PathBuf::from("closure_deferred_place_capabilities.fe"),
        r#"
struct Boxed {
    value: u256,
}

struct MutHolder {
    value: mut u256,
}

fn probe() {
    let mut boxed = Boxed { value: 0 }
    let borrow = |target: mut| -> mut u256 { mut target.value }
    let returned = borrow.call(mut boxed)
    returned = 42

    let mut value = 0
    let holder = MutHolder { value: mut value }
    let set = |target| {
        target.value = 42
    }
    set.call(holder)

    let mut array_value = 0
    let handles: [mut u256; 1] = [mut array_value]
    let set_index = |items| {
        items[0] = 42
    }
    set_index.call(handles)

    let mut borrowed_value = 0
    let borrowed_handles: [mut u256; 1] = [mut borrowed_value]
    let borrow_index = |items| -> mut u256 { mut items[0] }
    let returned = borrow_index.call(borrowed_handles)
    returned = 42

    let mut incremented_value = 41
    let incremented_handles: [mut u256; 1] = [mut incremented_value]
    let increment_index = |items| {
        items[0] += 1
    }
    increment_index.call(incremented_handles)
}
"#,
    );
    let (top_mod, _) = db.top_mod(file);
    db.assert_no_diags(top_mod);

    let file = db.new_stand_alone(
        Utf8PathBuf::from("closure_deferred_immutable_places.fe"),
        r#"
struct Boxed {
    value: u256,
}

struct RefHolder {
    value: ref u256,
}

fn probe() {
    let set_owned = |target: own| {
        target.value = 42
    }
    set_owned.call_once(Boxed { value: 0 })

    let value = 0
    let mut holder = RefHolder { value: ref value }
    let set_ref = |target: mut| {
        target.value = 42
    }
    set_ref.call(mut holder)

    let array_value = 0
    let mut handles: [ref u256; 1] = [ref array_value]
    let set_index_ref = |items: mut| {
        items[0] = 42
    }
    set_index_ref.call(mut handles)

    let borrowed_array_value = 0
    let mut borrowed_handles: [ref u256; 1] = [ref borrowed_array_value]
    let borrow_index_ref = |items: mut| -> mut u256 { mut items[0] }
    let returned = borrow_index_ref.call(mut borrowed_handles)
}
"#,
    );
    let (top_mod, _) = db.top_mod(file);
    let rendered = format_diagnostics(&db, &db.run_on_top_mod(top_mod));
    assert_eq!(
        rendered
            .matches("left-hand side of assignment is immutable")
            .count(),
        3,
        "{rendered}"
    );
    assert!(
        rendered.contains("mutable borrow requires a mutable place"),
        "{rendered}"
    );
}

#[test]
fn inferred_closure_results_copy_from_borrow_capabilities() {
    let mut db = HirAnalysisTestDb::default();
    let file = db.new_stand_alone(
        Utf8PathBuf::from("closure_copy_borrow_results.fe"),
        r#"
enum Maybe {
    None,
    Some(u256),
}

fn probe() -> u256 {
    let from_pattern = |input| match input {
        Maybe::None => 0
        Maybe::Some(value) => value
    }
    let from_reversed_pattern = |input| match input {
        Maybe::Some(value) => value
        Maybe::None => 0
    }
    let from_ref = |value: ref u256| if true { 0 } else { value }
    let from_reversed_ref = |value: ref u256| if true { value } else { 0 }
    let from_mut = |value: mut u256| if true { 0 } else { value }
    let mut mutable: u256 = 0
    from_pattern.call(Maybe::Some(40))
        + from_reversed_pattern.call(Maybe::Some(0))
        + from_ref.call(ref mutable)
        + from_reversed_ref.call(ref mutable)
        + from_mut.call(mut mutable)
        + 2
}
"#,
    );
    let (top_mod, _) = db.top_mod(file);
    db.assert_no_diags(top_mod);
}

#[test]
fn unconstrained_fn_carriers_infer_closure_parameter_capabilities() {
    let mut db = HirAnalysisTestDb::default();
    let file = db.new_stand_alone(
        Utf8PathBuf::from("closure_generic_carrier_inference.fe"),
        r#"
use core::functional::Fn

fn accept<A, R, F: Fn<(A,), R>>(_ function: F) {}
fn accept_pair<A, B, R, F: Fn<(A, B), R>>(_ function: F) {}
fn read_ref(_ value: ref u256) -> u256 { value }
fn read_mut(_ value: mut u256) -> u256 { value }
fn read_view(_ value: view u256) -> u256 { value }
fn take(_ value: own u256) -> u256 { value }
fn sum(_ left: ref u256, _ right: mut u256) -> u256 { left + right }

fn probe() {
    accept(|value: ref u256| -> u256 { value })
    accept(|value: mut u256| -> u256 { value })
    accept(|value: view u256| -> u256 { value })
    accept(|value: ref| -> u256 { read_ref(value) })
    accept(|value: mut| -> u256 { read_mut(value) })
    accept(|value: view| -> u256 { read_view(value) })
    accept(|value| -> u256 { read_view(value) })
    accept(|value: own| -> u256 { take(value) })
    accept_pair(|left: ref, right: mut| -> u256 { sum(left, right) })
}
"#,
    );
    let (top_mod, _) = db.top_mod(file);
    db.assert_no_diags(top_mod);
}

#[test]
fn closure_parameter_layout_holes_are_inferred_monomorphically() {
    let mut db = HirAnalysisTestDb::default();
    let file = db.new_stand_alone(
        Utf8PathBuf::from("closure_layout_hole_parameter.fe"),
        r#"
use core::functional::Fn

struct Rooted<const ROOT: u256 = _> {
    value: u256,
}

struct Pair<const ROOT: u256 = _> {
    left: Rooted<ROOT>,
    right: Rooted<ROOT>,
}

struct Independent<const LEFT: u256 = _, const RIGHT: u256 = _> {
    left: Rooted<LEFT>,
    right: Rooted<RIGHT>,
}

fn apply_concrete<F: Fn<(view Rooted<9>,), u256>>(_ function: F) -> u256 {
    function.call(Rooted<9> { value: 8 })
}

fn probe() -> u256 {
    let read = |value: Rooted| -> u256 { value.value }
    let read_explicit = |value: Rooted<_>| -> u256 { value.value }
    let read_ref = |value: ref Rooted| -> u256 { value.value }
    let read_mut = |value: mut Rooted| -> u256 { value.value }
    let consume = |value: own Rooted| value
    let sum = |pair: Pair| -> u256 { pair.left.value + pair.right.value }
    let sum_independent =
        |pair: Independent| -> u256 { pair.left.value + pair.right.value }
    let sum_separate =
        |left: Rooted, right: Rooted| -> u256 { left.value + right.value }
    let rooted = Rooted<5> { value: 4 }
    let mut mutable = Rooted<6> { value: 5 }
    read.call(Rooted<7> { value: 2 })
        + read_explicit.call(Rooted<4> { value: 3 })
        + read_ref.call(ref rooted)
        + read_mut.call(mut mutable)
        + consume.call_once(Rooted<10> { value: 9 }).value
        + sum.call(Pair<8> {
            left: Rooted<8> { value: 20 },
            right: Rooted<8> { value: 20 },
        })
        + sum_independent.call(Independent<11, 12> {
            left: Rooted<11> { value: 6 },
            right: Rooted<12> { value: 7 },
        })
        + sum_separate.call(
            Rooted<13> { value: 6 },
            Rooted<14> { value: 7 },
        )
        + apply_concrete(|value: Rooted| value.value)
}
"#,
    );
    let (top_mod, _) = db.top_mod(file);
    db.assert_no_diags(top_mod);

    let file = db.new_stand_alone(
        Utf8PathBuf::from("closure_layout_hole_monomorphism.fe"),
        r#"
struct Rooted<const ROOT: u256 = _> {
    value: u256,
}

fn probe() {
    let read = |value: Rooted| -> u256 { value.value }
    read.call(Rooted<7> { value: 1 })
    read.call(Rooted<8> { value: 2 })
}
"#,
    );
    let (top_mod, _) = db.top_mod(file);
    let rendered = format_diagnostics(&db, &db.run_on_top_mod(top_mod));
    assert!(!rendered.contains("inferred const generic"), "{rendered}");
    assert!(rendered.contains("type mismatch"), "{rendered}");

    let file = db.new_stand_alone(
        Utf8PathBuf::from("closure_layout_hole_unresolved.fe"),
        r#"
struct Rooted<const ROOT: u256 = _> {
    value: u256,
}

fn probe() {
    let read = |value: Rooted| value.value
}
"#,
    );
    let (top_mod, _) = db.top_mod(file);
    let rendered = format_diagnostics(&db, &db.run_on_top_mod(top_mod));
    assert!(rendered.contains("type annotation is needed"), "{rendered}");

    let file = db.new_stand_alone(
        Utf8PathBuf::from("closure_layout_hole_return.fe"),
        r#"
struct Rooted<const ROOT: u256 = _> {
    value: u256,
}

fn probe() {
    let make = || -> Rooted { Rooted<7> { value: 42 } }
}
"#,
    );
    let (top_mod, _) = db.top_mod(file);
    let rendered = format_diagnostics(&db, &db.run_on_top_mod(top_mod));
    assert!(rendered.contains("inferred const generic"), "{rendered}");
}

#[test]
fn inferred_closure_parameter_types_are_monomorphic_across_calls() {
    let mut db = HirAnalysisTestDb::default();
    let file = db.new_stand_alone(
        Utf8PathBuf::from("closure_payload_monomorphism.fe"),
        r#"
fn probe() {
    let identity = |value| value
    identity.call(42 as u256)
    identity.call(true)
}
"#,
    );
    let (top_mod, _) = db.top_mod(file);
    let rendered = format_diagnostics(&db, &db.run_on_top_mod(top_mod));
    assert!(rendered.contains("type mismatch"), "{rendered}");
    assert!(rendered.contains("bool"), "{rendered}");
    assert!(rendered.contains("u256"), "{rendered}");
}

#[test]
fn closure_return_type_is_inferred_from_only_explicit_returns() {
    let mut db = HirAnalysisTestDb::default();
    let file = db.new_stand_alone(
        Utf8PathBuf::from("closure_explicit_return_inference.fe"),
        r#"
fn probe() -> u256 {
    let choose = |flag| {
        if flag {
            return 20 as u256
        }
        return 22 as u256
    }
    choose.call(true)
}
"#,
    );
    let (top_mod, _) = db.top_mod(file);
    db.assert_no_diags(top_mod);

    let func = find_func(&db, top_mod, "probe");
    let (_, typed_body) = check_func_body(&db, func);
    let closure = typed_body
        .closure_infos()
        .next()
        .expect("missing closure")
        .1;
    assert_eq!(closure.ty.ret_ty(&db), TyId::u256(&db));
}

#[test]
fn closure_call_arity_diagnostics_count_only_explicit_arguments() {
    let mut db = HirAnalysisTestDb::default();
    let file = db.new_stand_alone(
        Utf8PathBuf::from("closure_call_arity.fe"),
        r#"
fn probe() {
    let zero = || 42 as u256
    zero.call(())

    let one = |_ unit: own ()| 42 as u256
    one.call()
}
"#,
    );
    let (top_mod, _) = db.top_mod(file);
    let rendered = format_diagnostics(&db, &db.run_on_top_mod(top_mod));
    assert!(
        rendered.contains("expected 0 arguments, but 1 given"),
        "{rendered}"
    );
    assert!(
        rendered.contains("expected 1 arguments, but 0 given"),
        "{rendered}"
    );
}

#[test]
fn duplicate_closure_parameter_names_are_rejected() {
    let mut db = HirAnalysisTestDb::default();
    let file = db.new_stand_alone(
        Utf8PathBuf::from("duplicate_closure_parameter.fe"),
        r#"
fn probe() {
    let choose = |value: own u256, value: own bool| value
    let result: bool = choose.call_once(1, true)
}
"#,
    );
    let (top_mod, _) = db.top_mod(file);
    let rendered = format_diagnostics(&db, &db.run_on_top_mod(top_mod));
    assert!(
        rendered.contains("duplicate closure parameter `value`"),
        "{rendered}"
    );
}

#[test]
fn closure_parameter_annotations_enforce_function_ownership_invariants() {
    let mut db = HirAnalysisTestDb::default();
    let file = db.new_stand_alone(
        Utf8PathBuf::from("invalid_closure_parameter_ownership.fe"),
        r#"
fn probe() {
    let invalid = |item: own ref u256| item
    let invalid_mut = |mut item| item
    let bad_param = |item: MissingParam| item
    let bad_return = || -> MissingReturn { 1 }
}
"#,
    );
    let (top_mod, _) = db.top_mod(file);
    let rendered = format_diagnostics(&db, &db.run_on_top_mod(top_mod));
    assert!(rendered.contains("invalid `own` parameter"), "{rendered}");
    assert!(
        rendered.contains("`own` parameters must have owned types"),
        "{rendered}"
    );
    assert!(
        rendered.contains("invalid `mut` parameter syntax"),
        "{rendered}"
    );
    assert!(
        rendered.contains("`MissingParam` is not found"),
        "{rendered}"
    );
    assert!(
        rendered.contains("`MissingReturn` is not found"),
        "{rendered}"
    );
}

#[test]
fn concrete_closure_callees_do_not_inherit_unrelated_dispatch_witnesses() {
    let mut db = HirAnalysisTestDb::default();
    let file = db.new_stand_alone(
        Utf8PathBuf::from("closure_callee_impl_env.fe"),
        r#"
fn sink(_ value: own u256) {}

fn invoke() {
    let first = |value: own| sink(value)
    let second = |value: own| sink(value)
    first.call(1)
    second.call(2)
}
"#,
    );
    let (top_mod, _) = db.top_mod(file);
    db.assert_no_diags(top_mod);

    let invoke = find_func(&db, top_mod, "invoke");
    let invoke_instance = get_or_build_semantic_instance(
        &db,
        identity_semantic_instance_key(&db, BodyOwner::Func(invoke)),
    );
    let closure_keys = invoke_instance
        .callees(&db)
        .iter()
        .filter(|callee| matches!(callee.key.owner(&db), BodyOwner::Closure { .. }))
        .map(|callee| callee.key)
        .collect::<Vec<_>>();
    assert_eq!(closure_keys.len(), 2);

    let sink_keys = closure_keys
        .into_iter()
        .map(|key| {
            get_or_build_semantic_instance(&db, key)
                .callees(&db)
                .iter()
                .find_map(|callee| {
                    let BodyOwner::Func(func) = callee.key.owner(&db) else {
                        return None;
                    };
                    func.name(&db)
                        .to_opt()
                        .is_some_and(|name| name.data(&db) == "sink")
                        .then_some(callee.key)
                })
                .expect("missing closure call to sink")
        })
        .collect::<Vec<_>>();
    assert_eq!(
        sink_keys[0], sink_keys[1],
        "a concrete callee key must not depend on which closure dispatch witness reached it"
    );
    let env = sink_keys[0].impl_env(&db);
    assert!(env.assumptions(&db).is_empty(&db));
    assert!(env.witnesses(&db).is_empty());
}

#[test]
fn schematic_default_method_callees_retain_their_proof_environment() {
    let mut db = HirAnalysisTestDb::default();
    let file = db.new_stand_alone(
        Utf8PathBuf::from("schematic_default_method_impl_env.fe"),
        r#"
trait Probe {
    fn probe(self) {}
}

fn generic<T: Probe>(_ value: own T) {
    value.probe()
}
"#,
    );
    let (top_mod, _) = db.top_mod(file);
    db.assert_no_diags(top_mod);

    let generic = find_func(&db, top_mod, "generic");
    let instance = get_or_build_semantic_instance(
        &db,
        identity_semantic_instance_key(&db, BodyOwner::Func(generic)),
    );
    let callee = instance
        .callees(&db)
        .iter()
        .find(|callee| {
            matches!(
                callee.key.owner(&db),
                BodyOwner::Func(func)
                    if func
                        .name(&db)
                        .to_opt()
                        .is_some_and(|name| name.data(&db) == "probe")
            )
        })
        .expect("missing generic default-method callee");
    let env = callee.key.impl_env(&db);
    assert!(!env.assumptions(&db).is_empty(&db));
    assert!(!env.witnesses(&db).is_empty());
}

#[test]
fn effect_specialized_closure_retains_instance_capture_metadata() {
    let mut db = HirAnalysisTestDb::default();
    let file = db.new_stand_alone(
        Utf8PathBuf::from("effect_specialized_closure.fe"),
        r#"
use core::effect_ref::MemPtr
use core::functional::Fn

struct Slot<const ROOT: u256 = _> {}

fn capture_target() uses (target: Slot) {
    let handle = ref target
    let f = |_ unit: own ()| -> () {
        let captured = handle
    }
    f.call(())
}

pub fn invoke() {
    let provider: MemPtr<Slot<7>> = MemPtr::from_raw(0)
    with (provider) {
        capture_target()
    }
}
"#,
    );
    let (top_mod, _) = db.top_mod(file);
    db.assert_no_diags(top_mod);

    let invoke = find_func(&db, top_mod, "invoke");
    let invoke_instance = get_or_build_semantic_instance(
        &db,
        identity_semantic_instance_key(&db, BodyOwner::Func(invoke)),
    );
    let capture_key = invoke_instance
        .body(&db)
        .blocks
        .iter()
        .flat_map(|block| &block.stmts)
        .find_map(|stmt| {
            let SStmtKind::Assign {
                expr: SExpr::Call { callee, .. },
                ..
            } = &stmt.kind
            else {
                return None;
            };
            let BodyOwner::Func(func) = callee.key.owner(&db) else {
                return None;
            };
            func.name(&db)
                .to_opt()
                .is_some_and(|name| name.data(&db) == "capture_target")
                .then_some(callee.key)
        })
        .expect("missing specialized capture_target call");
    let capture_instance = get_or_build_semantic_instance(&db, capture_key);
    let closure_key = capture_instance
        .body(&db)
        .blocks
        .iter()
        .flat_map(|block| &block.stmts)
        .find_map(|stmt| {
            let SStmtKind::Assign {
                expr: SExpr::Call { callee, .. },
                ..
            } = &stmt.kind
            else {
                return None;
            };
            matches!(callee.key.owner(&db), BodyOwner::Closure { .. }).then_some(callee.key)
        })
        .expect("missing specialized closure call");
    let BodyOwner::Closure { ty: closure, .. } = closure_key.owner(&db) else {
        unreachable!()
    };
    let callable = closure_key.callable_body(&db);
    let plan = callable
        .closure_capture_plan(&db, closure)
        .expect("specialized closure must retain its capture plan");
    assert_eq!(
        plan.iter()
            .map(|capture| capture.ty)
            .collect::<Vec<_>>()
            .as_slice(),
        closure.captures(&db),
    );
}

#[test]
fn typed_closure_descriptor_preserves_param_and_capture_field_order() {
    let mut db = HirAnalysisTestDb::default();
    let file = db.new_stand_alone(
        Utf8PathBuf::from("closure_descriptor_field_order.fe"),
        r#"
struct Left {
    value: u256,
}

struct Right {
    value: u256,
}

fn make(_ first: own Left, _ second: own Right) {
    let ordered = |alpha: own u256, beta: own u256| -> u256 {
        second.value + first.value + alpha + beta
    }
}
"#,
    );
    let (top_mod, _) = db.top_mod(file);
    db.assert_no_diags(top_mod);

    let func = find_func(&db, top_mod, "make");
    let (_, typed_body) = check_func_body(&db, func);
    let (_, info) = typed_body
        .closure_infos()
        .next()
        .expect("missing closure metadata");
    let callable = TypedCallableBody::new(BodyOwner::closure(&db, info.ty), typed_body);
    let (closure_body, receiver_mode) = callable
        .owner_closure_body(&db)
        .expect("closure metadata must form a coherent descriptor");

    assert_eq!(closure_body.ty(), info.ty);
    assert_eq!(
        closure_body.param_bindings().len(),
        info.ty.params(&db).len()
    );
    for (idx, binding) in closure_body.param_bindings().iter().enumerate() {
        assert!(matches!(
            binding,
            LocalBinding::Param {
                site: ParamSite::Closure(def),
                idx: binding_idx,
                ..
            } if *def == info.def && *binding_idx == idx
        ));
    }

    let captures = closure_body.captures(&db).collect::<Vec<_>>();
    assert_eq!(captures.len(), info.ty.captures(&db).len());
    assert_eq!(
        captures
            .iter()
            .map(|capture| capture.binding)
            .collect::<Vec<_>>(),
        info.captures
            .iter()
            .map(|capture| capture.binding)
            .collect::<Vec<_>>(),
    );
    assert_eq!(
        captures
            .iter()
            .map(|capture| capture.ty)
            .collect::<Vec<_>>()
            .as_slice(),
        info.ty.captures(&db),
    );
    assert_eq!(
        captures
            .iter()
            .map(|capture| capture.access)
            .collect::<Vec<_>>()
            .as_slice(),
        info.ty.capture_accesses(&db),
    );

    let physical_params = closure_body.physical_param_bindings(&db, receiver_mode);
    assert!(matches!(
        physical_params,
        [
            LocalBinding::Param {
                site: ParamSite::ClosureEnv(env_def),
                ..
            },
            LocalBinding::Param {
                site: ParamSite::ClosureArgs(args_def),
                ..
            }
        ] if env_def == info.def && args_def == info.def
    ));

    let closure = get_or_build_semantic_instance(
        &db,
        identity_semantic_instance_key(&db, BodyOwner::closure(&db, info.ty)),
    );
    let semantic_body = closure.body(&db);
    for (expected_field, binding) in closure_body.param_bindings().iter().copied().enumerate() {
        let local = semantic_body
            .locals
            .iter()
            .enumerate()
            .find_map(|(idx, local)| {
                (local.source == Some(binding))
                    .then_some(fe_hir::analysis::semantic::SLocalId::from_u32(idx as u32))
            })
            .expect("logical closure parameter local");
        assert!(semantic_body.blocks.iter().any(|block| {
            block.stmts.iter().any(|stmt| {
                matches!(
                    &stmt.kind,
                    SStmtKind::Assign {
                        dst,
                        expr: SExpr::Field { field, .. },
                    } if *dst == local && usize::from(field.0) == expected_field
                )
            })
        }));
    }
}

#[test]
fn dynamic_return_indices_retain_callable_input_provenance() {
    let mut db = HirAnalysisTestDb::default();
    let file = db.new_stand_alone(
        Utf8PathBuf::from("dynamic_return_index.fe"),
        r#"
struct Boxed {
    value: u256,
}

fn indexed(_ values: own [Boxed; 2], _ index: own usize) -> Boxed {
    let alias = index
    values[alias]
}

fn mutable_index(_ values: own [Boxed; 2], mut index: own usize) -> Boxed {
    index = 1
    values[index]
}

fn branch_index(
    _ values: own [Boxed; 2],
    _ indices: own [usize; 2],
    _ lane: own usize,
    _ left: own bool,
) -> Boxed {
    if left { values[indices[lane]] } else { values[0] }
}

fn set_index(_ index: mut usize) {
    index = 1
}

fn make(_ values: own [Boxed; 2], _ index: own usize) {
    let from_capture = |_ unit: own ()| -> Boxed { values[index] }
}

fn make_mut(_ values: own [Boxed; 2], _ index: mut usize) {
    let from_mut_capture = |_ unit: own ()| -> Boxed {
        set_index(index)
        values[index]
    }
}
"#,
    );
    let (top_mod, _) = db.top_mod(file);
    db.assert_no_diags(top_mod);

    let indexed = find_func(&db, top_mod, "indexed");
    let (_, indexed_body) = check_func_body(&db, indexed);
    assert_eq!(
        TypedCallableBody::new(BodyOwner::Func(indexed), indexed_body).return_provenance(&db),
        ReturnProvenance::Forwarded(vec![source(
            CallableInputLayoutHoleOrigin::ValueParam(0),
            vec![ReturnProjectionStep::DynamicIndex(ReturnIndexSource {
                origin: CallableInputLayoutHoleOrigin::ValueParam(1),
                projection: Vec::new(),
            })],
        )])
    );
    for name in ["mutable_index", "branch_index"] {
        let func = find_func(&db, top_mod, name);
        let (_, typed_body) = check_func_body(&db, func);
        assert_eq!(
            TypedCallableBody::new(BodyOwner::Func(func), typed_body).return_provenance(&db),
            ReturnProvenance::Fresh,
            "{name} must transport the callee's actual carrier instead of replaying an unstable or branch-only index",
        );
    }

    let make = find_func(&db, top_mod, "make");
    let (_, make_body) = check_func_body(&db, make);
    let (_, closure) = make_body
        .closure_infos()
        .next()
        .expect("missing captured-index closure");
    let callable = TypedCallableBody::new(BodyOwner::closure(&db, closure.ty), make_body);
    assert_eq!(
        callable.return_provenance(&db),
        ReturnProvenance::Forwarded(vec![source(
            CallableInputLayoutHoleOrigin::Receiver,
            vec![
                ReturnProjectionStep::Field(0),
                ReturnProjectionStep::DynamicIndex(ReturnIndexSource {
                    origin: CallableInputLayoutHoleOrigin::Receiver,
                    projection: vec![ReturnProjectionStep::Field(1)],
                }),
            ],
        )])
    );

    let make_mut = find_func(&db, top_mod, "make_mut");
    let (_, make_mut_body) = check_func_body(&db, make_mut);
    let (_, closure) = make_mut_body
        .closure_infos()
        .next()
        .expect("missing mutable-index capture closure");
    assert_eq!(
        TypedCallableBody::new(BodyOwner::closure(&db, closure.ty), make_mut_body)
            .return_provenance(&db),
        ReturnProvenance::Fresh,
        "a mutable captured index can change in the callee and must not be replayed from its pre-call value",
    );
}

#[test]
fn mutable_bindings_are_not_forwarded_as_callable_return_provenance() {
    let mut db = HirAnalysisTestDb::default();
    let file = db.new_stand_alone(
        Utf8PathBuf::from("mutable_return_source.fe"),
        r#"
struct Boxed {
    value: u256,
}

fn reassigned_param(mut value: own Boxed, _ replacement: own Boxed) -> Boxed {
    value = replacement
    value
}

fn reassigned_local(_ left: own Boxed, _ right: own Boxed) -> Boxed {
    let mut value = left
    value = right
    value
}
"#,
    );
    let (top_mod, _) = db.top_mod(file);
    db.assert_no_diags(top_mod);

    for name in ["reassigned_param", "reassigned_local"] {
        let func = find_func(&db, top_mod, name);
        let (_, typed_body) = check_func_body(&db, func);
        assert_eq!(
            TypedCallableBody::new(BodyOwner::Func(func), typed_body).return_provenance(&db),
            ReturnProvenance::Fresh,
            "{name} must return its actual carrier instead of replaying a declaration-time source",
        );
    }
}

#[test]
fn closure_call_materializes_captured_return_index_before_consuming_environment() {
    let mut db = HirAnalysisTestDb::default();
    let file = db.new_stand_alone(
        Utf8PathBuf::from("captured_return_index_call.fe"),
        r#"
struct Boxed {
    value: u256,
}

fn invoke(_ values: own [Boxed; 2], _ index: own usize) -> Boxed {
    let get = |_ unit: own ()| -> Boxed { values[index] }
    get.call_once(())
}
"#,
    );
    let (top_mod, _) = db.top_mod(file);
    db.assert_no_diags(top_mod);

    let invoke = find_func(&db, top_mod, "invoke");
    let semantic = get_or_build_semantic_instance(
        &db,
        identity_semantic_instance_key(&db, BodyOwner::Func(invoke)),
    );
    let body = semantic.body(&db);
    let (call_result, index_local) = body
        .blocks
        .iter()
        .flat_map(|block| &block.stmts)
        .find_map(|stmt| {
            let SStmtKind::Assign {
                expr:
                    SExpr::Call {
                        callee,
                        return_sources,
                        ..
                    },
                dst,
            } = &stmt.kind
            else {
                return None;
            };
            if !matches!(callee.key.owner(&db), BodyOwner::Closure { .. }) {
                return None;
            }
            let [source] = return_sources.as_ref() else {
                panic!("closure call should have one forwarded return source");
            };
            let [
                SCallReturnProjectionStep::Field(0),
                SCallReturnProjectionStep::DynamicIndex(index),
            ] = source.projection.as_slice()
            else {
                panic!("unexpected closure call return source: {source:?}");
            };
            Some((*dst, *index))
        })
        .expect("missing closure call return-index materialization");

    let index_source = body
        .blocks
        .iter()
        .flat_map(|block| &block.stmts)
        .find_map(|stmt| {
            let SStmtKind::Assign {
                dst,
                expr: SExpr::ReadPlace { place, .. },
            } = &stmt.kind
            else {
                return None;
            };
            (*dst == index_local).then_some(place)
        })
        .expect("captured return index should be read into a call-site local");
    let mut path = index_source.path.iter();
    assert!(matches!(path.next(), Some(Projection::Field(1))));
    assert!(path.next().is_none());
    let normalized = normalize_semantic_body(&db, semantic)
        .expect("captured-index closure call should normalize");
    let facts = NormalizedBodyFacts::new(&normalized);
    assert!(
        facts
            .local_layout_source_uses(call_result)
            .contains(&index_local),
        "call-result layout metadata must retain the materialized index as a layout source",
    );
    assert!(
        !facts
            .local_value_source_uses(call_result)
            .contains(&index_local),
        "a layout-only index must not become a runtime value-source fallback",
    );
    assert!(
        facts
            .local_dependency_uses(call_result)
            .contains(&index_local),
        "return slicing and carrier invalidation must retain layout-only dependencies",
    );
}

#[test]
fn closure_loop_control_cannot_target_an_enclosing_loop() {
    let mut db = HirAnalysisTestDb::default();
    let file = db.new_stand_alone(
        Utf8PathBuf::from("closure_loop_control_boundary.fe"),
        r#"
fn probe() {
    while true {
        let bad_break = || { break }
        let bad_continue = || { continue }
        let valid = || {
            while true {
                break
            }
        }
        break
    }
}
"#,
    );
    let (top_mod, _) = db.top_mod(file);
    let rendered = format_diagnostics(&db, &db.run_on_top_mod(top_mod));
    assert_eq!(
        rendered.matches("is not allowed outside of a loop").count(),
        2,
        "{rendered}"
    );
}
