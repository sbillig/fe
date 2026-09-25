use camino::Utf8PathBuf;
use common::indexmap::IndexMap;
use fe_hir::{
    analysis::{
        semantic::{
            SemanticBodyAdmission, SemanticDiagnosticKind, check_semantic_borrows,
            check_semantic_boundaries, get_or_build_semantic_instance,
            identity_semantic_instance_key, root_semantic_instance_key, semantic_body_admission,
        },
        ty::{
            binder::Binder,
            normalize::normalize_ty,
            trait_def::TraitInstId,
            trait_resolution::PredicateListId,
            ty_check::{BodyOwner, check_func_body},
            ty_def::{Kind, TyData, TyId, TyVarSort},
            ty_lower::{
                CompleteSubst, LoweredSlot, ParamBasis, ParamDomainId, ParamKey, SourceParamIndex,
                param_schema,
            },
            unify::{InferenceKey, UnificationTableBase},
        },
    },
    hir_def::{CallableDef, IdentId, ItemKind},
    test_db::{HirAnalysisTestDb, find_func},
};

#[test]
fn unused_dependent_default_stays_symbolic_in_semantic_callee_key() {
    for (source, caller_name, invalid_demand) in [
        (
            "fn inner<const N: usize, T = [u8; { 10 / N }]>() {}\nfn concrete() { inner<0>() }",
            "concrete",
            false,
        ),
        (
            "fn inner<const N: usize, T = [u8; { 10 / N }]>() -> u256 { core::size_of<T>() }\nfn concrete() -> u256 { inner<0>() }",
            "concrete",
            true,
        ),
        (
            "fn inner<const N: usize, T = [u8; { 10 / N }]>() -> u256 { core::size_of<T>() }\nfn outer<const M: usize>() -> u256 { inner<M>() }",
            "outer",
            false,
        ),
    ] {
        let mut db = HirAnalysisTestDb::default();
        let file = db.new_stand_alone(Utf8PathBuf::from("dependent_default.fe"), source);
        let (module, _) = db.top_mod(file);
        let concrete = find_func(&db, module, caller_name);
        let instance = get_or_build_semantic_instance(
            &db,
            identity_semantic_instance_key(&db, BodyOwner::Func(concrete)),
        );
        let callee = instance
            .call_sites(&db)
            .iter()
            .flatten()
            .find_map(|site| site.callee)
            .expect("missing semantic callee");
        let args = callee.key.subst(&db).generic_args(&db);
        assert_eq!(args.len(), 2);
        assert!(
            !args[1].has_invalid(&db),
            "default was evaluated into an invalid key: {:?}",
            args[1]
        );
        let callee_instance = get_or_build_semantic_instance(&db, callee.key);
        match (
            invalid_demand,
            semantic_body_admission(&db, callee_instance),
        ) {
            (false, SemanticBodyAdmission::Ready(_)) => {}
            (true, SemanticBodyAdmission::Rejected(diag)) => {
                assert_eq!(
                    diag.diag(&db).kind,
                    SemanticDiagnosticKind::InvalidConcreteType
                );
            }
            (_, result) => panic!("wrong admission for dependent default: {result:?}"),
        }
    }
}

#[test]
fn stored_trait_function_retains_associated_equality_in_semantic_callee() {
    let mut db = HirAnalysisTestDb::default();
    let file = db.new_stand_alone(
        Utf8PathBuf::from("stored_trait_function_evidence.fe"),
        r#"
trait Out {
    type Item
    fn f(_ x: own Self::Item) -> Self::Item { x }
}

fn direct<T: Out<Item = u256>>(_ x: u256) -> u256 { T::f(x) }
fn stored<T: Out<Item = u256>>(_ x: u256) -> u256 {
    let f = T::f
    f(x)
}
fn qualified_stored<T: Out<Item = u256>>(_ x: u256) -> u256 {
    let f = <T as Out>::f
    f(x)
}
fn tuple_stored<T: Out<Item = u256>>(_ x: u256) -> u256 {
    let pair = (T::f,)
    pair.0(x)
}
"#,
    );
    let (module, _) = db.top_mod(file);

    for name in ["direct", "stored", "qualified_stored", "tuple_stored"] {
        let func = find_func(&db, module, name);
        let (diags, _) = check_func_body(&db, func);
        assert!(diags.is_empty(), "{name} type checking failed: {diags:?}");

        let instance = get_or_build_semantic_instance(
            &db,
            identity_semantic_instance_key(&db, BodyOwner::Func(func)),
        );
        let callee = instance
            .call_sites(&db)
            .iter()
            .flatten()
            .find_map(|site| site.callee)
            .unwrap_or_else(|| panic!("missing semantic callee for {name}"));
        let callee_instance = get_or_build_semantic_instance(&db, callee.key);
        assert_eq!(
            callee_instance.normalized_result_ty(&db),
            TyId::u256(&db),
            "{name} lost its associated equality"
        );
    }
}

#[test]
fn concrete_selected_body_does_not_inherit_unrelated_caller_bounds() {
    let mut db = HirAnalysisTestDb::default();
    let file = db.new_stand_alone(
        Utf8PathBuf::from("concrete_selected_callee_evidence.fe"),
        r#"
trait A {}
trait B {}
trait Out { fn f() {} }
impl Out for bool {}
fn first<T: A>() { <bool as Out>::f() }
fn second<U: B>() { <bool as Out>::f() }
"#,
    );
    let (module, _) = db.top_mod(file);
    db.assert_no_diags(module);
    let callees = ["first", "second"].map(|name| {
        let func = find_func(&db, module, name);
        let caller = get_or_build_semantic_instance(
            &db,
            identity_semantic_instance_key(&db, BodyOwner::Func(func)),
        );
        caller
            .call_sites(&db)
            .iter()
            .flatten()
            .find_map(|site| site.callee)
            .expect("missing selected method")
            .key
    });
    assert_eq!(callees[0], callees[1]);
}

#[test]
fn concrete_free_callee_does_not_inherit_unrelated_caller_bounds() {
    let mut db = HirAnalysisTestDb::default();
    let file = db.new_stand_alone(
        Utf8PathBuf::from("concrete_free_callee_evidence.fe"),
        r#"
trait A {}
trait B {}
trait Need {}
impl Need for bool {}
fn target<T: Need>(_ x: T) {}
fn first<X: A>() { target(true) }
fn second<Y: B>() { target(true) }
"#,
    );
    let (module, _) = db.top_mod(file);
    db.assert_no_diags(module);
    let callees = ["first", "second"].map(|name| {
        let func = find_func(&db, module, name);
        let caller = get_or_build_semantic_instance(
            &db,
            identity_semantic_instance_key(&db, BodyOwner::Func(func)),
        );
        caller
            .call_sites(&db)
            .iter()
            .flatten()
            .find_map(|site| site.callee)
            .expect("missing concrete free call")
            .key
    });
    assert_eq!(callees[0], callees[1]);
}

#[test]
fn source_and_full_method_schemas_keep_inherited_keys_in_both_query_orders() {
    for source_first in [true, false] {
        let mut db = HirAnalysisTestDb::default();
        let file = db.new_stand_alone(
            Utf8PathBuf::from("method_schema_basis.fe"),
            r#"
struct Slot<const ROOT: u256 = _> {}
trait T<X> { fn f<Y>(_ value: Slot, _ extra: Y) {} }
fn foreign<Z>() {}
"#,
        );
        let (module, _) = db.top_mod(file);
        db.assert_no_diags(module);
        let func = module
            .all_funcs(&db)
            .iter()
            .copied()
            .find(|func| {
                func.name(&db)
                    .to_opt()
                    .is_some_and(|name| name.data(&db) == "f")
            })
            .expect("missing trait method");
        let (source, full) = if source_first {
            (
                param_schema(&db, func.into(), ParamBasis::Source),
                param_schema(&db, func.into(), ParamBasis::Full),
            )
        } else {
            let full = param_schema(&db, func.into(), ParamBasis::Full);
            let source = param_schema(&db, func.into(), ParamBasis::Source);
            (source, full)
        };
        assert_eq!(source.basis(&db), ParamBasis::Source);
        assert_eq!(full.basis(&db), ParamBasis::Full);
        assert!(
            full.keys(&db)
                .iter()
                .any(|key| matches!(key, ParamKey::CallableLayout { .. })),
            "method signature did not reserve a hidden layout slot"
        );
        let own_key = source
            .source_key(&db, SourceParamIndex(0))
            .expect("missing method source parameter");
        assert_eq!(full.source_key(&db, SourceParamIndex(0)), Some(own_key));
        assert!(
            full.slot_for(&db, own_key).expect("missing full slot").0
                > source
                    .slot_for(&db, own_key)
                    .expect("missing source slot")
                    .0
        );
        assert_eq!(&source.keys(&db)[..2], &full.keys(&db)[..2]);
        let inherited = source.key_at(&db, LoweredSlot(0)).unwrap();
        let inherited_formal = source.formal_at(&db, LoweredSlot(0)).unwrap();
        assert_eq!(source.original_key(&db, inherited_formal), Some(inherited));
        assert_eq!(full.original_key(&db, inherited_formal), Some(inherited));
        let ParamKey::TraitSelf(trait_) = inherited else {
            panic!("first inherited key should be trait Self");
        };
        let parent = param_schema(&db, trait_.into(), ParamBasis::Full);
        let parent_formal = parent.formal_at(&db, LoweredSlot(0)).unwrap();
        assert_ne!(parent_formal, inherited_formal);
        assert_eq!(full.original_key(&db, parent_formal), Some(inherited));
        let TyData::TyParam(param) = inherited_formal.data(&db) else {
            panic!("trait Self should be a parameter");
        };
        assert_eq!(param.owner, func.scope());

        let source_own = source
            .formal_at(&db, source.slot_for(&db, own_key).unwrap())
            .unwrap();
        let full_own = full
            .formal_at(&db, full.slot_for(&db, own_key).unwrap())
            .unwrap();
        assert_eq!(source.original_key(&db, source_own), Some(own_key));
        assert_eq!(full.original_key(&db, full_own), Some(own_key));
        assert_eq!(full.original_key(&db, source_own), None);
        let hidden_key = full.key_at(&db, LoweredSlot(2)).unwrap();
        assert!(matches!(hidden_key, ParamKey::CallableLayout { .. }));
        let hidden = full.formal_at(&db, LoweredSlot(2)).unwrap();
        assert_eq!(source.original_key(&db, hidden), None);

        let foreign = module
            .all_funcs(&db)
            .iter()
            .copied()
            .find(|candidate| {
                candidate
                    .name(&db)
                    .to_opt()
                    .is_some_and(|name| name.data(&db) == "foreign")
            })
            .expect("missing foreign function");
        let foreign_param = CallableDef::Func(foreign).params(&db)[0];
        assert_eq!(source.original_key(&db, foreign_param), None);
        assert_eq!(full.original_key(&db, foreign_param), None);

        let domain = ParamDomainId::full(&db, full);
        let values = domain
            .slots(&db)
            .map(|slot| full.formal_at(&db, slot).unwrap())
            .collect();
        let subst = CompleteSubst::new(domain, &db, values).unwrap();
        assert_eq!(subst.get(&db, own_key), Some(full_own));
        let prefix = source
            .allowed_default_dependencies(&db, SourceParamIndex(0))
            .unwrap();
        assert!(prefix.position_for(&db, inherited).is_some());
        assert_eq!(prefix.position_for(&db, own_key), None);
    }
}

#[test]
fn sealed_effect_identity_retains_bodyless_call_after_semantic_checks() {
    let mut db = HirAnalysisTestDb::default();
    let file = db.new_stand_alone(
        Utf8PathBuf::from("sealed_effect_trait_selection.fe"),
        r#"
use std::evm::RawStorage
fn access() -> u256 uses (raw: mut RawStorage) { raw.sload(0) }
"#,
    );
    let (module, _) = db.top_mod(file);
    db.assert_no_diags(module);
    let func = find_func(&db, module, "access");
    let instance = get_or_build_semantic_instance(
        &db,
        identity_semantic_instance_key(&db, BodyOwner::Func(func)),
    );
    check_semantic_borrows(&db, instance).expect("sealed effect call must pass borrow validation");
    check_semantic_boundaries(&db, instance)
        .expect("sealed effect call must pass boundary validation");
    let callee = instance
        .call_sites(&db)
        .iter()
        .flatten()
        .find_map(|site| site.callee)
        .expect("missing sealed effect callee")
        .key;
    let BodyOwner::Func(selected) = callee.owner(&db) else {
        panic!("sealed effect callee is not a function");
    };
    assert!(
        selected.body(&db).is_none(),
        "sealed effect call unexpectedly selected its runtime body in HIR"
    );

    let root_key = root_semantic_instance_key(&db, BodyOwner::Func(func))
        .expect("standalone root must synthesize the sealed effect provider");
    let root = get_or_build_semantic_instance(&db, root_key);
    let selected = root
        .call_sites(&db)
        .iter()
        .flatten()
        .find_map(|site| site.callee)
        .expect("missing root sealed effect callee")
        .key;
    let BodyOwner::Func(selected) = selected.owner(&db) else {
        panic!("root sealed effect callee is not a function");
    };
    assert!(
        selected.body(&db).is_some(),
        "standalone root did not select a concrete effect method body"
    );
}

#[test]
fn substituting_projection_and_evidence_preserves_both_arguments_and_bindings() {
    let mut db = HirAnalysisTestDb::default();
    let file = db.new_stand_alone(
        Utf8PathBuf::from("projection_binding_substitution.fe"),
        "trait Out { type Item }\nfn f<T>() {}\n",
    );
    let (module, _) = db.top_mod(file);
    db.assert_no_diags(module);
    let (trait_, func) =
        module
            .children_non_nested(&db)
            .fold((None, None), |acc, item| match item {
                ItemKind::Trait(trait_) => (Some(trait_), acc.1),
                ItemKind::Func(func) => (acc.0, Some(func)),
                _ => acc,
            });
    let trait_ = trait_.expect("missing Out");
    let func = func.expect("missing f");
    let param = CallableDef::Func(func).params(&db)[0];
    let item = IdentId::new(&db, "Item");
    let inst = TraitInstId::new(
        &db,
        trait_,
        vec![param],
        IndexMap::from_iter([(item, param)]),
    );
    let projection = TyId::assoc_ty(&db, inst.trait_ref(&db), item);
    let result = Binder::bind(func.into(), projection).instantiate(&db, &[TyId::bool(&db)]);
    let evidence = Binder::bind(func.into(), inst).instantiate(&db, &[TyId::bool(&db)]);
    let TyData::AssocTy(result) = result.data(&db) else {
        panic!("expected projection after substitution");
    };
    assert_eq!(result.trait_.args(&db), &[TyId::bool(&db)]);
    assert_eq!(
        evidence.assoc_type_bindings(&db).get(&item),
        Some(&TyId::bool(&db)),
        "associated binding retained a stale declaration parameter"
    );
}

#[test]
fn projection_normalization_matches_all_trait_arguments() {
    let mut db = HirAnalysisTestDb::default();
    let file = db.new_stand_alone(
        Utf8PathBuf::from("full_trait_reference_matching.fe"),
        "trait Out<A> { type Item }\nfn f<T>() {}\n",
    );
    let (module, _) = db.top_mod(file);
    db.assert_no_diags(module);
    let (trait_, func) =
        module
            .children_non_nested(&db)
            .fold((None, None), |acc, item| match item {
                ItemKind::Trait(trait_) => (Some(trait_), acc.1),
                ItemKind::Func(func) => (acc.0, Some(func)),
                _ => acc,
            });
    let trait_ = trait_.expect("missing Out");
    let func = func.expect("missing f");
    let param = CallableDef::Func(func).params(&db)[0];
    let item = IdentId::new(&db, "Item");
    let bound = TraitInstId::new(
        &db,
        trait_,
        vec![param, TyId::u8(&db)],
        IndexMap::from_iter([(item, TyId::u256(&db))]),
    );
    let assumptions = PredicateListId::new(&db, vec![bound]);
    let different_reference = TraitInstId::new_simple(&db, trait_, vec![param, TyId::bool(&db)]);
    let projection = TyId::assoc_ty(&db, different_reference.trait_ref(&db), item);
    assert_eq!(
        normalize_ty(&db, projection, func.scope(), assumptions),
        projection,
        "equality from Out<u8> was incorrectly used for Out<bool>"
    );
}

#[test]
fn implied_associated_type_bound_instantiates_all_trait_arguments() {
    let mut db = HirAnalysisTestDb::default();
    let file = db.new_stand_alone(
        Utf8PathBuf::from("implied_associated_type_bound_arguments.fe"),
        "trait Bound<X> {}\ntrait Foo<T, U> { type Item: Bound<T> }\n",
    );
    let (module, _) = db.top_mod(file);
    db.assert_no_diags(module);
    let mut traits = module
        .children_non_nested(&db)
        .filter_map(|item| match item {
            ItemKind::Trait(trait_) => Some(trait_),
            _ => None,
        });
    let bound = traits.next().expect("missing Bound");
    let foo = traits.next().expect("missing Foo");
    let premise = TraitInstId::new_simple(
        &db,
        foo,
        vec![TyId::bool(&db), TyId::u256(&db), TyId::u8(&db)],
    );
    let implied = PredicateListId::new(&db, vec![premise]).extend_all_bounds(&db);
    let item_bound = implied
        .list(&db)
        .iter()
        .copied()
        .find(|pred| pred.def(&db) == bound)
        .expect("missing implied Item: Bound<T> predicate");
    assert_eq!(item_bound.args(&db)[1], TyId::u256(&db));

    let formals = foo.params(&db);
    let swapped = TraitInstId::new_simple(&db, foo, vec![TyId::bool(&db), formals[2], formals[1]]);
    let implied = PredicateListId::new(&db, vec![swapped]).extend_all_bounds(&db);
    let item_bound = implied
        .list(&db)
        .iter()
        .copied()
        .find(|pred| pred.def(&db) == bound)
        .expect("missing swapped implied predicate");
    assert_eq!(item_bound.args(&db)[1], formals[2]);
}

#[test]
fn projection_normalization_does_not_commit_speculative_trait_arguments() {
    let mut db = HirAnalysisTestDb::default();
    let file = db.new_stand_alone(
        Utf8PathBuf::from("speculative_trait_reference_matching.fe"),
        "trait Out<A> { type Item }\nimpl Out<u8> for bool { type Item = u256 }\nfn f<T>() {}\n",
    );
    let (module, _) = db.top_mod(file);
    db.assert_no_diags(module);
    let trait_ = module
        .children_non_nested(&db)
        .find_map(|item| match item {
            ItemKind::Trait(trait_) => Some(trait_),
            _ => None,
        })
        .expect("missing Out");
    let item = IdentId::new(&db, "Item");
    let mut table = UnificationTableBase::<ena::unify::InPlace<InferenceKey<'_>>>::new(&db);
    let unknown = table.new_var(TyVarSort::General, &Kind::Star);
    let projection_ref = TraitInstId::new_simple(&db, trait_, vec![TyId::bool(&db), unknown]);
    let projection = TyId::assoc_ty(&db, projection_ref.trait_ref(&db), item);
    let assumption = TraitInstId::new(
        &db,
        trait_,
        vec![TyId::bool(&db), TyId::u8(&db)],
        IndexMap::from_iter([(item, TyId::u256(&db))]),
    );
    let assumptions = PredicateListId::new(&db, vec![assumption]);
    assert_eq!(
        normalize_ty(&db, projection, module.scope(), assumptions),
        projection,
        "possible equality or impl selection cannot assign a caller inference variable"
    );
}
