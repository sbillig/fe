use crate::namespace::items::{IngotId, Module, ModuleContext, ModuleFileContent, ModuleId};
use crate::AnalyzerDb;
use fe_parser::ast;
use indexmap::set::IndexSet;
use std::path::Path;
use std::rc::Rc;

pub fn ingot_all_modules(db: &dyn AnalyzerDb, ingot_id: IngotId) -> Rc<Vec<ModuleId>> {
    let ingot = &ingot_id.data(db);

    let file_modules = ingot
        .fe_files
        .values()
        .into_iter()
        .map(|(file, ast)| {
            let module = Module {
                name: Path::new(&file.name)
                    .file_stem()
                    .unwrap()
                    .to_str()
                    .unwrap()
                    .to_string(),
                ast: ast.clone(),
                file_content: ModuleFileContent::File { file: file.id },
                context: ModuleContext::Ingot(ingot_id),
            };

            db.intern_module(Rc::new(module))
        })
        .collect::<Vec<_>>();

    let dir_modules = ingot
        .fe_files
        .values()
        .into_iter()
        .map(|(file, ast)| {
            Path::new(&file.name)
                .parent()
                .unwrap()
                .to_str()
                .unwrap()
                .to_string()
        })
        .collect::<IndexSet<_>>()
        .into_iter()
        .map(|dir| {
            let module = Module {
                name: dir
                    .clone()
                    .split("/")
                    .collect::<Vec<_>>()
                    .last()
                    .unwrap()
                    .to_string(),
                ast: ast::Module { body: vec![] },
                context: ModuleContext::Ingot(ingot_id),
                file_content: ModuleFileContent::Dir { dir_path: dir },
            };

            db.intern_module(Rc::new(module))
        })
        .collect::<Vec<ModuleId>>();

    let modules = [file_modules, dir_modules].concat();
    // panic!("{:?}", modules.iter().map(|module| module.name(db)).collect::<Vec<_>>());
    Rc::new(modules)
}
