use fe_hir::{
    analysis::{
        semantic::{
            EntryLocate, NDataProjection, NStatementKind, SemanticInstance,
            identity_semantic_instance_key, normalize_semantic_body,
        },
        ty::{AddressSpace, trait_def::MethodArgMapError, ty_check::BodyOwner},
    },
    hir_def::ItemKind,
    test_db::HirAnalysisTestDb,
};

const SOURCE: &str = r#"
use core::marker::{Pinned, Storage, Transient}
use core::ops::{LaneCodec, Location, StateIndex}
use std::evm::StorageMap

struct Words {}

impl Pinned for Words {
    type Space = Transient
}

unsafe impl StateIndex for Words {
    type Key = u256
    type Entry = u256

    fn locate(self, _ root: u256, _ key: u256) -> Location {
        Location { slot: root + key, offset: 0 }
    }
}

struct U32Lanes {}

impl LaneCodec<u32> for U32Lanes {
    const BITS: u256 = 32

    fn get(_ word: u256, offset: u256) -> u32 {
        ((word >> offset) & 0xffffffff).downcast_truncate()
    }

    fn set(_ word: u256, offset: u256, value: own u32) -> u256 {
        (word & !(0xffffffff << offset)) | ((value as u256) << offset)
    }
}

struct Counters {}

impl Pinned for Counters {
    type Space = Storage
}

unsafe impl StateIndex for Counters {
    type Key = u256
    type Entry = u32
    type Lanes = U32Lanes

    fn locate(self, _ root: u256, _ key: u256) -> Location {
        Location { slot: root + key / 8, offset: (key % 8) * 32 }
    }
}

struct Cell<P> {}

impl<P: Pinned> Pinned for Cell<P> {
    type Space = P::Space
}

unsafe impl<P: Pinned> StateIndex for Cell<P> {
    type Key = u256
    type Entry = u256

    fn locate(self, _ root: u256, _ key: u256) -> Location {
        Location { slot: root + key, offset: 0 }
    }
}

fn word(_ w: Words, _ k: u256) -> u256 {
    w[k]
}

fn counter(_ c: Counters, _ k: u256) -> u32 {
    c[k]
}

fn nested(_ m: StorageMap<u256, StorageMap<bool, u8>>, _ a: u256, _ b: bool) -> u8 {
    m[a][b]
}

fn any_space<P: Pinned>(_ c: Cell<P>, _ k: u256) -> u256 {
    c[k]
}

fn any_collection<C: StateIndex<Entry = u256>>(_ c: C, _ k: C::Key) -> u256 {
    c[k]
}
"#;

/// Calls `check` with `entry_locate` of each entry step of the places the
/// body of `name` in `SOURCE` reads or writes, in order.
fn with_entries(
    name: &str,
    check: impl FnOnce(&HirAnalysisTestDb, &[Result<EntryLocate<'_>, MethodArgMapError<'_>>]),
) {
    let mut db = HirAnalysisTestDb::default();
    let file = db.new_stand_alone("entry_locate.fe".into(), SOURCE);
    let (top_mod, _) = db.top_mod(file);
    db.assert_no_diags(top_mod);
    let db = &db;
    let func = top_mod
        .all_items(db)
        .iter()
        .find_map(|item| match item {
            ItemKind::Func(func)
                if func
                    .name(db)
                    .to_opt()
                    .is_some_and(|ident| ident.data(db) == name) =>
            {
                Some(*func)
            }
            _ => None,
        })
        .unwrap_or_else(|| panic!("missing function `{name}`"));
    let instance = SemanticInstance::new(
        db,
        identity_semantic_instance_key(db, BodyOwner::Func(func, None)),
    );
    let body = &normalize_semantic_body(db, instance)
        .expect("normalized body")
        .body;
    let mut places = Vec::new();
    for statement in body.blocks.iter().flat_map(|block| &block.statements) {
        match &statement.kind {
            NStatementKind::Define { expr, .. } => {
                expr.for_each_place_operand(|place| places.push(place.clone()))
            }
            NStatementKind::Store { destination, .. } => places.push(destination.clone()),
            _ => {}
        }
    }
    let entries: Vec<_> = places
        .iter()
        .flat_map(|place| {
            place
                .path
                .iter()
                .enumerate()
                .filter(|(_, projection)| matches!(projection, NDataProjection::Entry(_)))
                .map(move |(idx, _)| {
                    let collection = body
                        .place_prefix_ty(db, place, idx)
                        .expect("collection type");
                    instance.entry_locate(db, collection)
                })
        })
        .collect();
    check(db, &entries);
}

/// The key type, entry type and `locate` implementor of `entry`.
fn summary(db: &HirAnalysisTestDb, entry: &EntryLocate<'_>) -> (String, String, String) {
    let BodyOwner::Func(locate, _) = entry.callee.owner(db) else {
        panic!("`locate` is a function: {entry:?}");
    };
    assert_eq!(locate.name(db).to_opt().unwrap().data(db), "locate");
    let implementor = locate
        .containing_impl_trait(db)
        .expect("`locate` of an implementation")
        .ty(db)
        .pretty_print(db)
        .to_string();
    (
        entry.key_ty.pretty_print(db).to_string(),
        entry.entry_ty.pretty_print(db).to_string(),
        implementor,
    )
}

#[test]
fn entry_names_its_collections_locate() {
    with_entries("word", |db, entries| {
        let [Ok(entry)] = entries else {
            panic!("expected one entry: {entries:?}");
        };
        assert_eq!(
            summary(db, entry),
            ("u256".into(), "u256".into(), "Words".into())
        );
        assert_eq!(entry.space, Some(AddressSpace::Transient));
        assert_eq!(entry.lanes, None);
    });
}

#[test]
fn packed_entry_names_its_codec() {
    with_entries("counter", |db, entries| {
        let [Ok(entry)] = entries else {
            panic!("expected one entry: {entries:?}");
        };
        assert_eq!(
            summary(db, entry),
            ("u256".into(), "u32".into(), "Counters".into())
        );
        assert_eq!(entry.space, Some(AddressSpace::Storage));
        let lanes = entry.lanes.expect("lanes");
        assert_eq!(lanes.codec.pretty_print(db), "U32Lanes");
        assert_eq!(lanes.bits, 32);
    });
}

#[test]
fn nested_entries_locate_each_collection() {
    with_entries("nested", |db, entries| {
        let [Ok(outer), Ok(inner)] = entries else {
            panic!("expected two entries: {entries:?}");
        };
        assert_eq!(
            summary(db, outer),
            (
                "u256".into(),
                "StorageMap<bool, u8>".into(),
                "StorageMap<K, V>".into()
            )
        );
        assert_eq!(
            summary(db, inner),
            ("bool".into(), "u8".into(), "StorageMap<K, V>".into())
        );
        assert_ne!(outer.callee, inner.callee);
        for entry in [outer, inner] {
            assert_eq!(entry.space, Some(AddressSpace::Storage));
            assert_eq!(entry.lanes, None);
        }
    });
}

#[test]
fn generic_space_is_unknown() {
    with_entries("any_space", |db, entries| {
        let [Ok(entry)] = entries else {
            panic!("expected one entry: {entries:?}");
        };
        assert_eq!(
            summary(db, entry),
            ("u256".into(), "u256".into(), "Cell<P>".into())
        );
        assert_eq!(entry.space, None);
    });
}

#[test]
fn generic_collection_has_no_locate_body() {
    with_entries("any_collection", |_, entries| {
        assert!(
            matches!(entries, [Err(MethodArgMapError::MissingBody)]),
            "{entries:?}"
        );
    });
}
