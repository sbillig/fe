use common::InputDb;
use driver::DriverDataBase;
use fe_codegen::emit_module_sonatina_ir;
use hir::hir_def::{ArithBinOp, BinOp};
use mir2::runtime::RefKind;
use mir2::{Layout, PlaceElem, PlaceRoot, RExpr, RStmt, RuntimeClass, build_runtime_package};
use url::Url;

#[test]
fn transparent_wrapper_returns_preserve_handle_fields_in_rmir() {
    let src = format!(
        "{}\nfn emit_helpers() -> u256 {{\n    let arr: [u256; 8] = [1, 2, 3, 4, 5, 6, 7, 8]\n    sum_last4(arr) + sum_first4(arr)\n}}\n",
        include_str!("../../fe/tests/fixtures/fe_test/view_param_local_ref_take_reverse.fe"),
    );
    let mut db = DriverDataBase::default();
    let file_url =
        Url::parse("file:///transparent_wrapper_returns_preserve_handle_fields_in_rmir.fe")
            .unwrap();
    db.workspace().touch(&mut db, file_url.clone(), Some(src));
    let file = db
        .workspace()
        .get(&db, &file_url)
        .expect("file should be loaded");
    let top_mod = db.top_mod(file);
    let package = build_runtime_package(&db, top_mod).expect("runtime package");
    let take_debug = package
        .functions(&db)
        .iter()
        .copied()
        .filter(|function| function.symbol(&db).contains("take"))
        .map(|function| {
            let body = function.instance(&db).body(&db);
            format!("{}:\n{body:#?}", function.symbol(&db))
        })
        .collect::<Vec<_>>();
    let take_u256 = package
        .functions(&db)
        .iter()
        .copied()
        .find(|function| {
            if !function.symbol(&db).contains("take") {
                return false;
            }
            let body = function.instance(&db).body(&db);
            let [_, param] = body.signature.params.as_slice() else {
                return false;
            };
            if !matches!(param.class, RuntimeClass::Ref { .. } | RuntimeClass::RawAddr { .. }) {
                return false;
            }
            let transported = body
                .blocks
                .iter()
                .flat_map(|block| block.stmts.iter())
                .find_map(|stmt| match stmt {
                    RStmt::Assign {
                        dst,
                        expr: RExpr::AddrOf { place },
                    } if matches!(
                        place.root,
                        PlaceRoot::Ptr { addr, .. } if addr == param.local && place.path.is_empty()
                    ) => Some(*dst),
                    RStmt::Assign {
                        dst,
                        expr: RExpr::RetagRef { value },
                    } if *value == param.local => Some(*dst),
                    RStmt::Assign {
                        dst,
                        expr: RExpr::Use(value),
                    } if *value == param.local => Some(*dst),
                    RStmt::Assign { .. }
                    | RStmt::EnumAssertVariant { .. }
                    | RStmt::Store { .. }
                    | RStmt::CopyInto { .. }
                    | RStmt::EnumSetTag { .. }
                    | RStmt::EnumWriteVariant { .. } => None,
                });
            body.blocks.iter().flat_map(|block| block.stmts.iter()).any(|stmt| {
                matches!(
                    stmt,
                    RStmt::Store { dst, src }
                        if (matches!(transported, Some(local) if *src == local)
                            || (transported.is_none() && *src == param.local))
                            && matches!(dst.root, PlaceRoot::Ref(_))
                            && matches!(dst.path.as_ref(), [PlaceElem::Field(field)] if field.0 == 1)
                )
            })
        })
        .unwrap_or_else(|| panic!(
            "generated take helper that preserves the incoming handle field\n\n{}",
            take_debug.join("\n\n")
        ));
    let body = take_u256.instance(&db).body(&db);
    let seq_param = body.signature.params[1].local;
    let transported = body
        .blocks
        .iter()
        .flat_map(|block| block.stmts.iter())
        .find_map(|stmt| match stmt {
            RStmt::Assign {
                dst,
                expr: RExpr::AddrOf { place },
            } if matches!(
                place.root,
                PlaceRoot::Ptr { addr, .. } if addr == seq_param && place.path.is_empty()
            ) =>
            {
                Some(*dst)
            }
            RStmt::Assign {
                dst,
                expr: RExpr::RetagRef { value },
            } if *value == seq_param => Some(*dst),
            RStmt::Assign {
                dst,
                expr: RExpr::Use(value),
            } if *value == seq_param => Some(*dst),
            RStmt::Assign { .. }
            | RStmt::EnumAssertVariant { .. }
            | RStmt::Store { .. }
            | RStmt::CopyInto { .. }
            | RStmt::EnumSetTag { .. }
            | RStmt::EnumWriteVariant { .. } => None,
        })
        .unwrap_or(seq_param);

    assert!(
        !body
            .blocks
            .iter()
            .flat_map(|block| block.stmts.iter())
            .any(|stmt| matches!(
                stmt,
                RStmt::Assign {
                    expr: RExpr::MaterializeToObject { .. },
                    ..
                }
            )),
        "transparent wrapper returns should not materialize handle fields:\n{body:#?}"
    );
    assert!(
        body.blocks.iter().flat_map(|block| block.stmts.iter()).any(|stmt| {
            matches!(
                stmt,
                RStmt::Store { dst, src }
                    if *src == transported
                        && matches!(dst.root, PlaceRoot::Ref(_))
                        && matches!(dst.path.as_ref(), [PlaceElem::Field(field)] if field.0 == 1)
            )
        }),
        "transparent wrapper returns should store the incoming borrow transport directly into the wrapper field:\n{body:#?}"
    );
}

#[test]
fn provider_root_trait_receivers_preserve_concrete_runtime_layouts() {
    let mut db = DriverDataBase::default();
    let file_url =
        Url::parse("file:///provider_root_trait_receivers_preserve_concrete_runtime_layouts.fe")
            .unwrap();
    db.workspace().touch(
        &mut db,
        file_url.clone(),
        Some(include_str!("fixtures/by_ref_trait_provider_storage_bug.fe").to_string()),
    );
    let file = db
        .workspace()
        .get(&db, &file_url)
        .expect("file should be loaded");
    let top_mod = db.top_mod(file);
    let package = build_runtime_package(&db, top_mod).expect("runtime package");
    let use_ctx = package
        .functions(&db)
        .iter()
        .copied()
        .find(|function| function.symbol(&db).contains("use_ctx"))
        .expect("generated use_ctx runtime function");
    let body = use_ctx.instance(&db).body(&db);
    let ctx_param = body.signature.params[0].local;

    let load = body
        .blocks
        .iter()
        .flat_map(|block| block.stmts.iter())
        .find_map(|stmt| match stmt {
            RStmt::Assign {
                dst,
                expr: RExpr::Load { place },
            } if place.path.is_empty()
                && (place.root == PlaceRoot::Ref(ctx_param)
                    || matches!(place.root, PlaceRoot::Provider(_))) =>
            {
                Some(*dst)
            }
            RStmt::Assign { .. }
            | RStmt::EnumAssertVariant { .. }
            | RStmt::Store { .. }
            | RStmt::CopyInto { .. }
            | RStmt::EnumSetTag { .. }
            | RStmt::EnumWriteVariant { .. } => None,
        })
        .unwrap_or_else(|| {
            panic!("use_ctx should load its provider-bound receiver handle:\n{body:#?}")
        });

    let mir2::RuntimeCarrier::Value(mir2::RuntimeClass::AggregateValue { layout }) =
        &body.locals[load.as_u32() as usize].carrier
    else {
        panic!("root-provider receiver load should produce a concrete aggregate value:\n{body:#?}");
    };
    let Layout::Struct(layout) = layout.data(&db) else {
        panic!("root-provider receiver load should use the concrete Pair layout:\n{body:#?}");
    };
    assert_eq!(layout.source_ty.pretty_print(&db).to_string(), "Pair");
}

#[test]
fn effect_handle_from_raw_helpers_preserve_transport_in_rmir() {
    let mut db = DriverDataBase::default();
    let file_url =
        Url::parse("file:///effect_handle_from_raw_helpers_preserve_transport_in_rmir.fe").unwrap();
    db.workspace().touch(
        &mut db,
        file_url.clone(),
        Some(include_str!("fixtures/effect_handle_field_deref.fe").to_string()),
    );
    let file = db
        .workspace()
        .get(&db, &file_url)
        .expect("file should be loaded");
    let top_mod = db.top_mod(file);
    let package = build_runtime_package(&db, top_mod).expect("runtime package");

    let from_raw_helpers = package
        .functions(&db)
        .iter()
        .copied()
        .filter(|function| function.symbol(&db).contains("from_raw"))
        .collect::<Vec<_>>();
    assert!(
        !from_raw_helpers.is_empty(),
        "expected generated from_raw helpers in runtime package"
    );

    for function in from_raw_helpers {
        let body = function.instance(&db).body(&db);
        assert!(
            body.blocks
                .iter()
                .flat_map(|block| block.stmts.iter())
                .any(|stmt| {
                    matches!(
                        stmt,
                        RStmt::Assign {
                            expr: RExpr::ProviderFromRaw { .. },
                            ..
                        }
                    )
                }),
            "from_raw helper should rebuild the handle from the raw word:\n{body:#?}"
        );
        assert!(
            !body
                .blocks
                .iter()
                .flat_map(|block| block.stmts.iter())
                .any(|stmt| {
                    matches!(
                        stmt,
                        RStmt::Assign {
                            expr: RExpr::AddrOf { .. } | RExpr::MaterializeToObject { .. },
                            ..
                        } | RStmt::CopyInto { .. }
                    )
                }),
            "from_raw helper should not materialize or take the address of the raw slot:\n{body:#?}"
        );
    }
}

#[test]
fn checked_extern_intrinsics_do_not_become_runtime_functions() {
    let mut db = DriverDataBase::default();
    let file_url =
        Url::parse("file:///checked_extern_intrinsics_do_not_become_runtime_functions.fe").unwrap();
    db.workspace().touch(
        &mut db,
        file_url.clone(),
        Some(
            r#"
fn add(a: u256, b: u256) -> u256 {
    a + b
}

fn neg(x: i256) -> i256 {
    -x
}

fn entry_add() -> u256 {
    add(1, 2)
}

fn entry_neg() -> i256 {
    neg(-3)
}
"#
            .to_string(),
        ),
    );
    let file = db
        .workspace()
        .get(&db, &file_url)
        .expect("file should be loaded");
    let top_mod = db.top_mod(file);
    let package = build_runtime_package(&db, top_mod).expect("runtime package");

    assert!(
        package
            .functions(&db)
            .iter()
            .all(|function| !function.symbol(&db).contains("__checked_")),
        "checked extern intrinsics should lower directly through rMIR expressions, not as runtime functions:\n{:#?}",
        package.functions(&db)
    );
}

#[test]
fn whole_handle_loads_materialize_values_before_rebinding_object_locals() {
    let mut db = DriverDataBase::default();
    let file_url = Url::parse(
        "file:///whole_handle_loads_materialize_values_before_rebinding_object_locals.fe",
    )
    .unwrap();
    db.workspace().touch(
        &mut db,
        file_url.clone(),
        Some(include_str!("../../fe/tests/fixtures/fe_test/poseidon_mock.fe").to_string()),
    );
    let file = db
        .workspace()
        .get(&db, &file_url)
        .expect("file should be loaded");
    let top_mod = db.top_mod(file);
    let package = build_runtime_package(&db, top_mod).expect("runtime package");

    for function in package.functions(&db) {
        let body = function.instance(&db).body(&db);
        assert!(
            !body
                .blocks
                .iter()
                .flat_map(|block| block.stmts.iter())
                .any(|stmt| {
                    matches!(
                        stmt,
                        RStmt::Assign {
                            dst,
                            expr:
                                RExpr::Load {
                                    place:
                                        mir2::RuntimePlace {
                                            root: PlaceRoot::Ref(_),
                                            path,
                                        },
                                },
                        } if path.is_empty()
                            && matches!(
                                body.locals[dst.as_u32() as usize].carrier,
                                mir2::RuntimeCarrier::Value(mir2::RuntimeClass::Ref { .. })
                            )
                    )
                }),
            "whole-handle loads must materialize aggregate values before rebinding handle locals:\n{body:#?}"
        );
    }
}

#[test]
fn by_value_enum_constants_do_not_become_const_regions() {
    let mut db = DriverDataBase::default();
    let file_url =
        Url::parse("file:///by_value_enum_constants_do_not_become_const_regions.fe").unwrap();
    db.workspace().touch(
        &mut db,
        file_url.clone(),
        Some(
            r#"enum E { A, B }

fn select(flag: bool) -> E {
    if flag {
        E::A
    } else {
        E::B
    }
}

fn entry() -> u8 {
    match select(true) {
        E::A => 1,
        E::B => 0,
    }
}"#
            .to_string(),
        ),
    );
    let file = db
        .workspace()
        .get(&db, &file_url)
        .expect("file should be loaded");
    let top_mod = db.top_mod(file);
    let package = build_runtime_package(&db, top_mod).expect("runtime package");
    let const_handle_bodies = package
        .functions(&db)
        .iter()
        .filter_map(|function| {
            let body = function.instance(&db).body(&db);
            body.blocks
                .iter()
                .flat_map(|block| block.stmts.iter())
                .any(|stmt| {
                    matches!(
                        stmt,
                        RStmt::Assign {
                            expr: RExpr::ConstRef { .. },
                            ..
                        }
                    )
                })
                .then(|| format!("{}:\n{body:#?}", function.symbol(&db)))
        })
        .collect::<Vec<_>>();

    assert!(
        package.const_regions(&db).is_empty(),
        "by-value enum constants should not create package const regions:\nregions={:#?}\n{}",
        package.const_regions(&db),
        const_handle_bodies.join("\n\n")
    );
    assert!(
        const_handle_bodies.is_empty(),
        "by-value enum constants should lower inline, not through ConstRef:\n{}",
        const_handle_bodies.join("\n\n")
    );
    assert!(
        package.functions(&db).iter().any(|function| {
            let body = function.instance(&db).body(&db);
            body.blocks
                .iter()
                .flat_map(|block| block.stmts.iter())
                .any(|stmt| {
                    matches!(
                        stmt,
                        RStmt::Assign {
                            expr: RExpr::EnumMake { .. },
                            ..
                        }
                    )
                })
        }),
        "enum constants should lower through EnumMake in value position",
    );
}

#[test]
fn by_value_aggregate_constants_do_not_become_const_regions() {
    let mut db = DriverDataBase::default();
    let file_url =
        Url::parse("file:///by_value_aggregate_constants_do_not_become_const_regions.fe").unwrap();
    db.workspace().touch(
        &mut db,
        file_url.clone(),
        Some(
            r#"const C: [u256; 2] = [10, 20]

fn entry() -> u256 {
    let vals = C
    vals[0] + vals[1]
}"#
            .to_string(),
        ),
    );
    let file = db
        .workspace()
        .get(&db, &file_url)
        .expect("file should be loaded");
    let top_mod = db.top_mod(file);
    let package = build_runtime_package(&db, top_mod).expect("runtime package");
    let const_handle_bodies = package
        .functions(&db)
        .iter()
        .filter_map(|function| {
            let body = function.instance(&db).body(&db);
            body.blocks
                .iter()
                .flat_map(|block| block.stmts.iter())
                .any(|stmt| {
                    matches!(
                        stmt,
                        RStmt::Assign {
                            expr: RExpr::ConstRef { .. },
                            ..
                        }
                    )
                })
                .then(|| format!("{}:\n{body:#?}", function.symbol(&db)))
        })
        .collect::<Vec<_>>();

    assert!(
        package.const_regions(&db).is_empty(),
        "by-value aggregate constants should not create package const regions:\nregions={:#?}\n{}",
        package.const_regions(&db),
        const_handle_bodies.join("\n\n")
    );
    assert!(
        const_handle_bodies.is_empty(),
        "by-value aggregate constants should lower inline, not through ConstRef:\n{}",
        const_handle_bodies.join("\n\n")
    );
}

#[test]
fn borrow_typed_aggregate_literals_lower_without_const_shape_mismatch() {
    let mut db = DriverDataBase::default();
    let file_url =
        Url::parse("file:///borrow_typed_aggregate_literals_lower_without_const_shape_mismatch.fe")
            .unwrap();
    db.workspace().touch(
        &mut db,
        file_url.clone(),
        Some(
            r#"fn first(xs: ref [u256; 2]) -> u256 {
    xs[0]
}

fn entry() -> u256 {
    first([10, 20])
}"#
            .to_string(),
        ),
    );
    let file = db
        .workspace()
        .get(&db, &file_url)
        .expect("file should be loaded");
    let top_mod = db.top_mod(file);
    let package = build_runtime_package(&db, top_mod).expect("runtime package");
    let debug = package
        .functions(&db)
        .iter()
        .map(|function| {
            let body = function.instance(&db).body(&db);
            format!("{}:\n{body:#?}", function.symbol(&db))
        })
        .collect::<Vec<_>>();

    assert!(
        package.functions(&db).iter().any(|function| {
            let body = function.instance(&db).body(&db);
            body.blocks
                .iter()
                .flat_map(|block| block.stmts.iter())
                .any(|stmt| {
                    matches!(
                        stmt,
                        RStmt::Assign {
                            expr: RExpr::MaterializeToObject { .. } | RExpr::AllocObject { .. },
                            ..
                        }
                    )
                })
        }),
        "borrow-typed aggregate literals should materialize through normal object/value lowering:\n{}",
        debug.join("\n\n")
    );
}

#[test]
fn sonatina_enum_tag_matches_preserve_typed_tag_values() {
    let mut db = DriverDataBase::default();
    let file_url =
        Url::parse("file:///sonatina_enum_tag_matches_preserve_typed_tag_values.fe").unwrap();
    db.workspace().touch(
        &mut db,
        file_url.clone(),
        Some(
            r#"enum Maybe {
    None,
    Some(u256),
}

pub fn code(x: Maybe) -> u256 {
    match x {
        Maybe::None => 0,
        Maybe::Some(v) => v,
    }
}"#
            .to_string(),
        ),
    );
    let file = db
        .workspace()
        .get(&db, &file_url)
        .expect("file should be loaded");
    let top_mod = db.top_mod(file);
    emit_module_sonatina_ir(&db, top_mod).expect("enum-tag matches should lower in Sonatina");
}

#[test]
fn object_backed_scalar_field_borrows_lower_as_typed_refs() {
    let mut db = DriverDataBase::default();
    let file_url =
        Url::parse("file:///object_backed_scalar_field_borrows_lower_as_typed_refs.fe").unwrap();
    db.workspace().touch(
        &mut db,
        file_url.clone(),
        Some(include_str!("fixtures/effect_handle_field_deref.fe").to_string()),
    );
    let file = db
        .workspace()
        .get(&db, &file_url)
        .expect("file should be loaded");
    let top_mod = db.top_mod(file);
    let package = build_runtime_package(&db, top_mod).expect("runtime package");

    let mut saw_scalar_ref_borrow = false;
    for function in package.functions(&db) {
        let body = function.instance(&db).body(&db);
        for stmt in body.blocks.iter().flat_map(|block| block.stmts.iter()) {
            let RStmt::Assign {
                dst,
                expr: RExpr::AddrOf { place },
            } = stmt
            else {
                continue;
            };
            if !matches!(place.root, PlaceRoot::Ref(_)) {
                continue;
            }
            let Some(RuntimeClass::Ref { pointee, kind, .. }) = body.value_class(*dst) else {
                continue;
            };
            if !matches!(**pointee, RuntimeClass::Scalar(_)) {
                continue;
            }
            assert!(
                matches!(
                    kind,
                    RefKind::Object | RefKind::Provider { .. } | RefKind::Const
                ),
                "scalar field borrow should lower as a typed ref, not a raw address:\n{body:#?}"
            );
            saw_scalar_ref_borrow = true;
        }
    }

    assert!(
        saw_scalar_ref_borrow,
        "expected at least one object-backed scalar field borrow to lower as RuntimeClass::Ref"
    );
}

#[test]
fn checked_overflow_exprs_survive_into_rmir() {
    let mut db = DriverDataBase::default();
    let file_url = Url::parse("file:///debug_checked_add_unused_local_rmir.fe").unwrap();
    db.workspace().touch(
        &mut db,
        file_url.clone(),
        Some(
            r#"
fn test_add_overflow_u8() {
    let x: u8 = 255
    let y: u8 = x + 1
}
"#
            .to_string(),
        ),
    );
    let file = db
        .workspace()
        .get(&db, &file_url)
        .expect("file should be loaded");
    let top_mod = db.top_mod(file);
    let package = build_runtime_package(&db, top_mod).expect("runtime package");
    let func = package
        .functions(&db)
        .iter()
        .copied()
        .find(|function| function.symbol(&db).contains("test_add_overflow_u8"))
        .expect("generated runtime function");
    let body = func.instance(&db).body(&db);
    assert!(
        body.blocks
            .iter()
            .flat_map(|block| block.stmts.iter())
            .any(|stmt| {
                matches!(
                    stmt,
                    RStmt::Assign {
                        expr: RExpr::Binary {
                            op: BinOp::Arith(ArithBinOp::Add),
                            ..
                        } | RExpr::Call { .. },
                        ..
                    }
                )
            }),
        "checked overflow expression should survive semantic const canonicalization and lower to executable rMIR arithmetic:\n{body:#?}"
    );
}
