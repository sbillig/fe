use common::InputDb;
use driver::DriverDataBase;
use fe_codegen::{OptLevel, emit_module_sonatina_ir, emit_runtime_package_sonatina_ir_optimized};
use mir2::runtime::{AddressSpaceKind, RefKind};
use mir2::{
    IntrinsicArithBinOp, Layout, PlaceElem, PlaceRoot, RExpr, RLocalId, RStmt, RuntimeBuiltin,
    RuntimeClass, build_runtime_package, build_test_runtime_package,
};
use url::Url;

fn sonatina_function_body<'a>(ir: &'a str, name: &str) -> &'a str {
    let marker = format!("func private %{name}");
    let start = ir
        .find(&marker)
        .unwrap_or_else(|| panic!("missing function `{name}` in Sonatina IR:\n{ir}"));
    let tail = &ir[start..];
    let end = tail
        .find("\n\nfunc ")
        .or_else(|| tail.find("\n\nobject "))
        .unwrap_or(tail.len());
    &tail[..end]
}

fn sonatina_ops(body: &str) -> Vec<&str> {
    body.lines()
        .filter_map(|line| {
            let line = line.trim();
            let rhs = line.split_once(" = ")?.1;
            rhs.split_whitespace().next()
        })
        .collect()
}

fn contains_op_subsequence(body: &str, expected: &[&str]) -> bool {
    let mut ops = sonatina_ops(body).into_iter();
    expected
        .iter()
        .all(|expected| ops.any(|op| op == *expected))
}

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
    let package = build_test_runtime_package(&db, top_mod, None).expect("runtime test package");
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
fn mem_ptr_from_raw_helpers_use_raw_addr_transport_in_rmir() {
    let mut db = DriverDataBase::default();
    let file_url =
        Url::parse("file:///mem_ptr_from_raw_helpers_use_raw_addr_transport_in_rmir.fe").unwrap();
    db.workspace().touch(
        &mut db,
        file_url.clone(),
        Some(include_str!("fixtures/raw_log_emit.fe").to_string()),
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
        "expected generated MemPtr::from_raw helper in runtime package"
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
                            expr: RExpr::WordToRawAddr {
                                space: AddressSpaceKind::Memory,
                                ..
                            },
                            ..
                        }
                    )
                }),
            "MemPtr::from_raw should keep memory handles as raw-address transport:\n{body:#?}"
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
                            expr: RExpr::ProviderFromRaw {
                                space: AddressSpaceKind::Memory,
                                ..
                            },
                            ..
                        }
                    )
                }),
            "MemPtr::from_raw must not reconstruct memory provider refs:\n{body:#?}"
        );
    }
}

#[test]
fn object_backed_nested_handle_fields_follow_carriers_before_projecting_children() {
    let mut db = DriverDataBase::default();
    let file_url = Url::parse(
        "file:///object_backed_nested_handle_fields_follow_carriers_before_projecting_children.fe",
    )
    .unwrap();
    db.workspace().touch(
        &mut db,
        file_url.clone(),
        Some(
            r#"struct Data {
    x: u256,
}

struct View {
    d: ref Data,
}

fn read(v: own View) -> u256 {
    v.d.x
}

fn entry() -> u256 {
    let data = Data { x: 1 }
    let view = View { d: ref data }
    read(view)
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
    let output = emit_module_sonatina_ir(&db, top_mod).expect("Sonatina IR");
    let read = sonatina_function_body(&output, "read");

    assert!(
        contains_op_subsequence(read, &["obj.proj", "obj.load", "obj.proj"]),
        "nested object-backed handle field access should load/follow the carrier before projecting children:\n{read}"
    );
}

#[test]
fn storage_backed_nested_handle_fields_follow_carriers_before_projecting_children() {
    let mut db = DriverDataBase::default();
    let file_url = Url::parse(
        "file:///storage_backed_nested_handle_fields_follow_carriers_before_projecting_children.fe",
    )
    .unwrap();
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
    let output = emit_module_sonatina_ir(&db, top_mod).expect("Sonatina IR");
    let bump = sonatina_function_body(&output, "bump");

    assert!(
        contains_op_subsequence(bump, &["obj.proj", "obj.load", "evm_sload"]),
        "nested storage-backed handle field access should load/follow the carrier before loading children:\n{bump}"
    );
}

#[test]
fn storage_backed_nested_handle_field_borrows_use_storage_transport() {
    let mut db = DriverDataBase::default();
    let file_url =
        Url::parse("file:///storage_backed_nested_handle_field_borrows_use_storage_transport.fe")
            .unwrap();
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
    let bump = package
        .functions(&db)
        .iter()
        .copied()
        .find(|function| function.symbol(&db).contains("bump"))
        .expect("generated bump runtime function");
    let body = bump.instance(&db).body(&db);

    let nested_field_borrow = body
        .blocks
        .iter()
        .flat_map(|block| block.stmts.iter())
        .find_map(|stmt| match stmt {
            RStmt::Assign {
                dst,
                expr: RExpr::AddrOf { place },
            } if matches!(place.root, PlaceRoot::Ref(_))
                && matches!(place.path.as_ref(), [PlaceElem::Field(field0), PlaceElem::Deref, PlaceElem::Field(field1)] if field0.0 == 0 && field1.0 == 0) =>
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
            panic!("expected nested field borrow in bump runtime body:\n{body:#?}")
        });

    let Some(RuntimeClass::Ref { pointee, kind, .. }) = body.value_class(nested_field_borrow)
    else {
        panic!("nested storage-backed field borrow should lower as a typed ref:\n{body:#?}");
    };
    assert!(
        matches!(**pointee, RuntimeClass::Scalar(_)),
        "nested storage-backed field borrow should point at the scalar field:\n{body:#?}"
    );
    assert!(
        matches!(
            kind,
            RefKind::Provider {
                space: mir2::AddressSpaceKind::Storage,
                ..
            }
        ),
        "nested storage-backed field borrow should use storage transport, not object/memory transport:\n{body:#?}"
    );
}

#[test]
fn projected_enum_field_snapshots_preserve_full_enum_payloads() {
    let mut db = DriverDataBase::default();
    let file_url =
        Url::parse("file:///projected_enum_field_snapshots_preserve_full_enum_payloads.fe")
            .unwrap();
    db.workspace().touch(
        &mut db,
        file_url.clone(),
        Some(
            r#"enum Flag {
    A(u256),
    B(u256),
}

impl Copy for Flag {}

struct Inner {
    flag: Flag,
}

struct Wrapper {
    inner: Inner,
}

fn repack(wrapper: Wrapper) -> Inner {
    let flag = wrapper.inner.flag
    let copy = flag
    Inner { flag: copy }
}

fn entry() -> Inner {
    repack(Wrapper {
        inner: Inner { flag: Flag::A(42) },
    })
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
    let output = emit_module_sonatina_ir(&db, top_mod).expect("Sonatina IR");
    let repack = sonatina_function_body(&output, "repack");

    assert!(
        repack.contains("obj.load") && repack.contains("obj.store"),
        "repack should preserve the full enum payload through object loads/stores:\n{repack}"
    );
    assert!(
        !sonatina_ops(repack)
            .into_iter()
            .any(|op| op == "enum_tag" || op == "enum_tag_of"),
        "projected enum field payload should stay a full enum value, not an enum tag:\n{repack}"
    );
}

#[test]
fn owned_aggregate_values_with_place_style_reads_get_object_backed_runtime_storage() {
    let mut db = DriverDataBase::default();
    let file_url = Url::parse(
        "file:///owned_aggregate_values_with_place_style_reads_get_object_backed_runtime_storage.fe",
    )
    .unwrap();
    db.workspace().touch(
        &mut db,
        file_url.clone(),
        Some(
            r#"struct Table {
    used: [u8; 4],
}

impl Table {
    fn get(self, _ slot: usize) -> u8 {
        let used = self.used
        used[slot]
    }
}

fn entry() -> u8 {
    Table { used: [1, 2, 3, 4] }.get(2)
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
    let get = package
        .functions(&db)
        .iter()
        .copied()
        .find(|function| function.symbol(&db).contains("get"))
        .expect("generated get runtime function");
    let body = get.instance(&db).body(&db);
    let used_local = body
        .locals
        .iter()
        .enumerate()
        .find_map(|(idx, local)| {
            (idx >= body.signature.params.len()
                && local.semantic_ty.pretty_print(&db) == "[u8; 4]"
                && matches!(
                    local.carrier,
                    mir2::RuntimeCarrier::Value(RuntimeClass::Ref {
                        kind: RefKind::Object,
                        ..
                    })
                ))
            .then_some(RLocalId::from_u32(idx as u32))
        })
        .unwrap_or_else(|| {
            panic!(
                "owned aggregate local with later place-style reads should be object-backed:\n{body:#?}"
            )
        });

    assert!(
        body.blocks
            .iter()
            .flat_map(|block| block.stmts.iter())
            .any(|stmt| {
                matches!(
                    stmt,
                    RStmt::Assign {
                        expr: RExpr::MaterializePlaceToObject { .. }
                            | RExpr::MaterializeToObject { .. },
                        ..
                    }
                )
            }),
        "owned aggregate local should materialize through object-backed lowering:\n{body:#?}"
    );
    assert!(
        body.blocks
            .iter()
            .flat_map(|block| block.stmts.iter())
            .any(|stmt| {
                matches!(
                    stmt,
                    RStmt::Assign {
                        expr: RExpr::Load { place },
                        ..
                    } if place.root == PlaceRoot::Ref(used_local)
                        && matches!(place.path.as_ref(), [PlaceElem::Index(_)])
                )
            }),
        "later projections should read from the owned aggregate local itself:\n{body:#?}"
    );
}

#[test]
fn linear_probe_big_struct_keeps_array_projection_reads_place_based() {
    let mut db = DriverDataBase::default();
    let file_url =
        Url::parse("file:///linear_probe_big_struct_keeps_array_projection_reads_place_based.fe")
            .unwrap();
    db.workspace().touch(
        &mut db,
        file_url.clone(),
        Some(
            format!(
                "{}\nfn entry() -> u256 {{\n    let mut table = Table::empty()\n    table.set(17, 99)\n    table.get(17)\n}}\n",
                include_str!("../../fe/tests/fixtures/fe_test/linear_probe_big_struct.fe")
            ),
        ),
    );
    let file = db
        .workspace()
        .get(&db, &file_url)
        .expect("file should be loaded");
    let top_mod = db.top_mod(file);
    emit_module_sonatina_ir(&db, top_mod)
        .expect("linear_probe_big_struct should lower through Sonatina");
    let package = build_runtime_package(&db, top_mod).expect("runtime package");

    for (name, expected_fields) in [("set", [true, true, false]), ("get", [true, true, true])] {
        let function = package
            .functions(&db)
            .iter()
            .copied()
            .find(|function| function.symbol(&db).contains(name))
            .unwrap_or_else(|| panic!("missing `{name}` runtime function"));
        let body = function.instance(&db).body(&db);
        let self_local = body.signature.params[0].local;
        let mut saw_indexed_reads = [false; 3];
        for stmt in body.blocks.iter().flat_map(|block| block.stmts.iter()) {
            let RStmt::Assign {
                expr: RExpr::Load { place },
                ..
            } = stmt
            else {
                continue;
            };
            if place.root != PlaceRoot::Ref(self_local) && place.root != PlaceRoot::Slot(self_local)
            {
                continue;
            }
            match place.path.as_ref() {
                [PlaceElem::Field(field)] if field.0 < 3 => {
                    panic!(
                        "{name} should not load whole array field {} before indexing:\n{body:#?}",
                        field.0
                    );
                }
                [PlaceElem::Field(field), PlaceElem::Index(_)] if field.0 < 3 => {
                    saw_indexed_reads[field.0 as usize] = true;
                }
                _ => {}
            }
        }
        for (idx, expected) in expected_fields.into_iter().enumerate() {
            assert_eq!(
                saw_indexed_reads[idx],
                expected,
                "{name} should {}read field {} through a composite place:\n{body:#?}",
                if expected { "" } else { "not " },
                idx
            );
        }
    }
}

#[test]
fn materialized_scalar_uses_do_not_keep_rawaddr_runtime_carriers() {
    let mut db = DriverDataBase::default();
    let file_url =
        Url::parse("file:///materialized_scalar_uses_do_not_keep_rawaddr_runtime_carriers.fe")
            .unwrap();
    db.workspace().touch(
        &mut db,
        file_url.clone(),
        Some(
            r#"fn bump(_ x: mut u8) -> u8 {
    x += 1
    x
}

fn entry() -> u8 {
    let mut x: u8 = 2
    bump(mut x)
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
    let bump = package
        .functions(&db)
        .iter()
        .copied()
        .find(|function| function.symbol(&db).contains("bump"))
        .expect("generated bump runtime function");
    let body = bump.instance(&db).body(&db);
    let param_count = body.signature.params.len();
    let rawaddr_temps = body
        .locals
        .iter()
        .enumerate()
        .skip(param_count)
        .filter_map(|(idx, local)| {
            matches!(
                local.carrier,
                mir2::RuntimeCarrier::Value(RuntimeClass::RawAddr { target: None, .. })
            )
            .then_some(RLocalId::from_u32(idx as u32))
        })
        .collect::<Vec<_>>();

    assert!(
        body.blocks
            .iter()
            .flat_map(|block| block.stmts.iter())
            .any(|stmt| {
                matches!(
                    stmt,
                    RStmt::Assign {
                        expr: RExpr::Load { .. },
                        ..
                    }
                )
            }),
        "materializing a mutable scalar use should load from its place root:\n{body:#?}"
    );
    assert!(
        rawaddr_temps.iter().all(|temp| {
            body.blocks
                .iter()
                .flat_map(|block| block.stmts.iter())
                .any(|stmt| {
                    matches!(
                        stmt,
                        RStmt::Assign {
                            expr:
                                RExpr::Load {
                                    place:
                                        mir2::RuntimePlace {
                                            root:
                                                PlaceRoot::Ptr {
                                                    addr,
                                                    class: RuntimeClass::Scalar(_),
                                                    ..
                                                },
                                            ..
                                        },
                                },
                            ..
                        } if addr == temp
                    )
                })
        }),
        "raw-address scalar temps must be materialized through scalar loads before use:\n{body:#?}"
    );
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
}

pub fn main() -> u256 {
    code(Maybe::Some(1))
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
                        expr: RExpr::Call { .. }
                            | RExpr::Builtin(RuntimeBuiltin::IntrinsicArith {
                                op: IntrinsicArithBinOp::Add,
                                checked: true,
                                ..
                            }),
                        ..
                    }
                )
            }),
        "checked overflow expression should survive semantic const canonicalization and lower to executable rMIR arithmetic:\n{body:#?}"
    );
}

#[test]
fn unit_branch_mutations_do_not_use_erased_call_results() {
    let mut db = DriverDataBase::default();
    let file_url =
        Url::parse("file:///unit_branch_mutations_do_not_use_erased_call_results.fe").unwrap();
    db.workspace().touch(
        &mut db,
        file_url.clone(),
        Some(
            r#"
#[test]
fn unit_branch_add_assign() {
    let mut x: usize = 0
    if true {
        x += 1
    } else {
        x += 1
    }
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
                            expr: RExpr::Use(value),
                            ..
                        } if body.value_class(*value).is_none()
                    )
                }),
            "unit-valued mutation branches should not try to use erased call results:\n{}:\n{body:#?}",
            function.symbol(&db),
        );
    }
}

#[test]
fn by_value_array_returns_keep_visible_aggregate_signatures() {
    let mut db = DriverDataBase::default();
    let file_url =
        Url::parse("file:///by_value_array_returns_keep_visible_aggregate_signatures.fe").unwrap();
    db.workspace().touch(
        &mut db,
        file_url.clone(),
        Some(
            r#"
fn return_array_after_projection(xs: [u8; 4]) -> [u8; 4] {
    let ys = xs
    let _ = ys[2]
    ys
}

fn use_returned_array() -> u8 {
    let mut xs: [u8; 4] = [1, 2, 3, 4]
    xs = return_array_after_projection(xs)
    xs[0]
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
    let callee = package
        .functions(&db)
        .iter()
        .copied()
        .find(|function| {
            function
                .symbol(&db)
                .contains("return_array_after_projection")
        })
        .expect("return_array_after_projection runtime function");
    let body = callee.instance(&db).body(&db);

    assert!(
        matches!(
            body.signature.ret,
            Some(RuntimeClass::AggregateValue { .. })
        ),
        "by-value aggregate helper should keep a visible aggregate return signature:\n{body:#?}"
    );
    assert!(
        body.locals
            .iter()
            .skip(body.signature.params.len())
            .any(|local| matches!(
                &local.carrier,
                mir2::RuntimeCarrier::Value(RuntimeClass::Ref {
                    kind: RefKind::Object,
                    pointee,
                    ..
                }) if matches!(pointee.as_ref(), RuntimeClass::AggregateValue { layout } if matches!(layout.data(&db), Layout::Array(_)))
            )),
        "helper should still be free to use internal object-backed storage for projectable owned aggregates:\n{body:#?}"
    );
}

#[test]
fn callers_of_by_value_array_returns_do_not_receive_object_ref_results() {
    let mut db = DriverDataBase::default();
    let file_url = Url::parse(
        "file:///callers_of_by_value_array_returns_do_not_receive_object_ref_results.fe",
    )
    .unwrap();
    db.workspace().touch(
        &mut db,
        file_url.clone(),
        Some(
            r#"
fn return_array_after_projection(xs: [u8; 4]) -> [u8; 4] {
    let ys = xs
    let _ = ys[2]
    ys
}

fn use_returned_array() -> u8 {
    let mut xs: [u8; 4] = [1, 2, 3, 4]
    xs = return_array_after_projection(xs)
    xs[0]
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
    let callee = package
        .functions(&db)
        .iter()
        .copied()
        .find(|function| {
            function
                .symbol(&db)
                .contains("return_array_after_projection")
        })
        .expect("return_array_after_projection runtime function")
        .instance(&db);
    let caller = package
        .functions(&db)
        .iter()
        .copied()
        .find(|function| function.symbol(&db).contains("use_returned_array"))
        .expect("use_returned_array runtime function");
    let body = caller.instance(&db).body(&db);

    let call_results = body
        .blocks
        .iter()
        .flat_map(|block| block.stmts.iter())
        .filter_map(|stmt| match stmt {
            RStmt::Assign {
                dst,
                expr: RExpr::Call { callee: target, .. },
            } if *target == callee => Some(*dst),
            _ => None,
        })
        .collect::<Vec<_>>();

    assert!(
        !call_results.is_empty(),
        "caller should contain a direct call to the array-return helper:\n{body:#?}"
    );
    assert!(
        call_results.iter().all(|result| matches!(
            body.value_class(*result),
            Some(RuntimeClass::AggregateValue { .. })
        )),
        "by-value aggregate call results should stay visible aggregate values in callers, not object refs:\n{body:#?}"
    );
}

#[test]
fn by_value_array_return_materialization_structurally_copies_into_object_storage() {
    let mut db = DriverDataBase::default();
    let file_url = Url::parse(
        "file:///by_value_array_return_materialization_structurally_copies_into_object_storage.fe",
    )
    .unwrap();
    db.workspace().touch(
        &mut db,
        file_url.clone(),
        Some(
            r#"
fn return_array_after_projection(xs: [u256; 3]) -> [u256; 3] {
    let ys = xs
    let _ = ys[2]
    ys
}

fn use_returned_array() -> u256 {
    let mut xs: [u256; 3] = [1, 2, 3]
    xs = return_array_after_projection(xs)
    xs[0]
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
    let output = emit_runtime_package_sonatina_ir_optimized(
        &db,
        &package,
        fe_codegen::EVM_LAYOUT,
        OptLevel::O0,
    )
    .expect("Sonatina IR");
    let body = sonatina_function_body(&output, "use_returned_array");

    assert!(
        body.contains("extract_value"),
        "by-value aggregate materialization should structurally extract fields instead of whole-object obj.store:\n{body}"
    );
    assert!(
        body.contains("obj.load"),
        "object-backed aggregate copies should load leaf values from the source object before storing them:\n{body}"
    );
}

#[test]
fn fieldless_enum_fields_copy_into_object_storage_via_enum_ops() {
    let mut db = DriverDataBase::default();
    let file_url =
        Url::parse("file:///fieldless_enum_fields_copy_into_object_storage_via_enum_ops.fe")
            .unwrap();
    db.workspace().touch(
        &mut db,
        file_url.clone(),
        Some(
            r#"
enum E {
    A,
    B,
}

struct Pair {
    xs: [u256; 3],
    e: E,
}

fn build(flag: bool) -> Pair {
    let xs: [u256; 3] = [1, 2, 3]
    let e = if flag { E::A } else { E::B }
    Pair { xs, e }
}

#[test]
fn exercise() {
    let pair = build(true)
    assert(pair.xs[0] == 1)
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
    let package = build_test_runtime_package(&db, top_mod, None).expect("runtime test package");
    let output = emit_runtime_package_sonatina_ir_optimized(
        &db,
        &package,
        fe_codegen::EVM_LAYOUT,
        OptLevel::O0,
    )
    .expect("Sonatina IR");
    assert!(
        output.contains("enum.set_tag"),
        "fieldless enum object copies should set the destination enum tag explicitly:\n{output}"
    );
}
