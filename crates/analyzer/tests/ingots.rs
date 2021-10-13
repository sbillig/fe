use fe_analyzer::namespace::items::{Global, Ingot};
use fe_analyzer::{AnalyzerDb, TestDb};
use fe_parser::parse_file;
use std::rc::Rc;

// TODO: move to analysis.rs and print snapshots of each module
#[test]
fn basic_ingot() {
    let files = test_files::build_filestore("ingots/basic_ingot");

    let db = TestDb::default();

    let global = Global::default();
    let global_id = db.intern_global(Rc::new(global));

    let ingot = Ingot {
        name: "basic_ingot".to_string(),
        global: global_id,
        fe_files: files
            .files
            .values()
            .into_iter()
            .map(|file| {
                (
                    file.id,
                    (file.clone(), parse_file(file.id, &file.content).unwrap().0),
                )
            })
            .collect(),
    };
    let ingot_id = db.intern_ingot(Rc::new(ingot));

    fe_analyzer::analyze_ingot(&db, ingot_id).expect("compilation failed");
}
