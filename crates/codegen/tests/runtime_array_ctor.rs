use common::InputDb;
use driver::DriverDataBase;
use hir::{
    analysis::{
        semantic::{get_or_build_semantic_instance, identity_semantic_instance_key},
        ty::ty_check::BodyOwner,
    },
    projection::IndexSource,
};
use mir2::{
    Layout, PlaceElem, PlaceRoot, RExpr, RStmt, RuntimeInstanceKey, get_or_build_runtime_instance,
    instance::RuntimeInstanceSource,
};
use url::Url;

#[test]
fn array_literals_store_each_constant_index_before_load() {
    let mut db = DriverDataBase::default();
    let file_url =
        Url::parse("file:///array_literals_store_each_constant_index_before_load.fe").unwrap();
    db.workspace().touch(
        &mut db,
        file_url.clone(),
        Some(
            r#"fn read() -> u256 {
    let x: [u256; 4] = [1, 2, 3, 4]
    return x[0]
}"#
            .to_string(),
        ),
    );
    let file = db
        .workspace()
        .get(&db, &file_url)
        .expect("file should be loaded");
    let top_mod = db.top_mod(file);
    let [func] = top_mod.all_funcs(&db).as_slice() else {
        panic!("expected exactly one function");
    };

    let semantic = get_or_build_semantic_instance(
        &db,
        identity_semantic_instance_key(&db, BodyOwner::Func(*func)),
    );
    let runtime = get_or_build_runtime_instance(
        &db,
        RuntimeInstanceKey::new(&db, RuntimeInstanceSource::Semantic(semantic), vec![]),
    );
    let body = runtime.body(&db);

    let array_alloc = body
        .blocks
        .iter()
        .flat_map(|block| block.stmts.iter())
        .find_map(|stmt| match stmt {
            RStmt::Assign {
                dst,
                expr: RExpr::AllocObject { layout },
            } => match layout.data(&db) {
                Layout::Array(array) if array.len == 4 => Some(*dst),
                Layout::Struct(_) | Layout::Array(_) | Layout::Enum(_) => None,
            },
            RStmt::Assign { .. }
            | RStmt::EnumAssertVariant { .. }
            | RStmt::Store { .. }
            | RStmt::CopyInto { .. }
            | RStmt::EnumSetTag { .. }
            | RStmt::EnumWriteVariant { .. } => None,
        })
        .expect("expected array object allocation");

    let mut indices = body
        .blocks
        .iter()
        .flat_map(|block| block.stmts.iter())
        .filter_map(|stmt| match stmt {
            RStmt::Store { dst, .. } if dst.root == PlaceRoot::Ref(array_alloc) => {
                match dst.path.as_ref() {
                    [PlaceElem::Index(IndexSource::Constant(idx))] => Some(*idx),
                    _ => None,
                }
            }
            RStmt::Assign { .. }
            | RStmt::EnumAssertVariant { .. }
            | RStmt::Store { .. }
            | RStmt::CopyInto { .. }
            | RStmt::EnumSetTag { .. }
            | RStmt::EnumWriteVariant { .. } => None,
        })
        .collect::<Vec<_>>();
    indices.sort_unstable();
    assert_eq!(indices, vec![0, 1, 2, 3]);

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
                    } if place.root == PlaceRoot::Ref(array_alloc) && place.path.is_empty()
                )
            }),
        "expected array aggregate lowering to load the populated object back into the value local",
    );
}
