use crate::diagnostics::CsDbWrapper;
use camino::{Utf8Path, Utf8PathBuf};
use codespan_reporting::term::{
    self,
    termcolor::{BufferWriter, ColorChoice},
};
use common::{
    diagnostics::CompleteDiagnostic,
    indexmap::IndexSet,
    input::{IngotDependency, IngotKind, Version},
    InputFile, InputIngot,
};
use hir::{
    hir_def::TopLevelMod,
    lower::{map_file_to_mod, module_tree},
};
use hir_analysis::{
    analysis_pass::{AnalysisPassManager, ParsingPass},
    diagnostics::DiagnosticVoucher,
    name_resolution::ImportAnalysisPass,
    ty::{
        AdtDefAnalysisPass, BodyAnalysisPass, FuncAnalysisPass, ImplAnalysisPass,
        ImplTraitAnalysisPass, TraitAnalysisPass, TypeAliasAnalysisPass,
    },
};

use crate::diagnostics::ToCsDiag;

#[derive(Default, Clone)]
#[salsa::db]
pub struct DriverDataBase {
    storage: salsa::Storage<Self>,
}
#[salsa::db]
impl salsa::Database for DriverDataBase {
    fn salsa_event(&self, _event: &dyn Fn() -> salsa::Event) {}
}

impl DriverDataBase {
    // TODO: An temporary implementation for ui testing.
    pub fn run_on_top_mod<'db>(&'db self, top_mod: TopLevelMod<'db>) -> DiagnosticsCollection<'db> {
        self.run_on_file_with_pass_manager(top_mod, initialize_analysis_pass)
    }

    pub fn run_on_file_with_pass_manager<'db, F>(
        &'db self,
        top_mod: TopLevelMod<'db>,
        pm_builder: F,
    ) -> DiagnosticsCollection<'db>
    where
        F: FnOnce(&'db DriverDataBase) -> AnalysisPassManager<'db>,
    {
        let mut pass_manager = pm_builder(self);
        DiagnosticsCollection(pass_manager.run_on_module(top_mod))
    }

    pub fn run_on_ingot(&self, ingot: InputIngot) -> DiagnosticsCollection {
        self.run_on_ingot_with_pass_manager(ingot, initialize_analysis_pass)
    }

    pub fn run_on_ingot_with_pass_manager<'db, F>(
        &'db self,
        ingot: InputIngot,
        pm_builder: F,
    ) -> DiagnosticsCollection<'db>
    where
        F: FnOnce(&'db DriverDataBase) -> AnalysisPassManager<'db>,
    {
        let tree = module_tree(self, ingot);
        let mut pass_manager = pm_builder(self);
        DiagnosticsCollection(pass_manager.run_on_module_tree(tree))
    }

    pub fn standalone(
        &mut self,
        file_path: &Utf8Path,
        source: &str,
        core_ingot: InputIngot,
    ) -> (InputIngot, InputFile) {
        let kind = IngotKind::StandAlone;

        // We set the ingot version to 0.0.0 for stand-alone file.
        let version = Version::new(0, 0, 0);
        let root_file = file_path;
        let core_dependency = IngotDependency::new("core", core_ingot);
        let mut external_ingots = IndexSet::default();
        external_ingots.insert(core_dependency);

        let ingot = InputIngot::new(
            self,
            file_path.parent().unwrap().as_str(),
            kind,
            version,
            external_ingots,
        );

        let file_name = root_file.file_name().unwrap();
        let input_file = InputFile::new(self, file_name.into(), source.to_string());
        ingot.set_root_file(self, input_file);
        ingot.set_files(self, [input_file].into_iter().collect());
        (ingot, input_file)
    }

    pub fn standalone_no_core(
        &mut self,
        file_path: &Utf8Path,
        source: &str,
    ) -> (InputIngot, InputFile) {
        let kind = IngotKind::StandAlone;

        // We set the ingot version to 0.0.0 for stand-alone file.
        let version = Version::new(0, 0, 0);
        let root_file = file_path;

        let ingot = InputIngot::new(
            self,
            file_path.parent().unwrap().as_str(),
            kind,
            version,
            IndexSet::default(),
        );

        let file_name = root_file.file_name().unwrap();
        let input_file = InputFile::new(self, file_name.into(), source.to_string());
        ingot.set_root_file(self, input_file);
        ingot.set_files(self, [input_file].into_iter().collect());
        (ingot, input_file)
    }

    pub fn local_ingot(
        &mut self,
        path: &Utf8Path,
        version: &Version,
        source_root: &Utf8Path,
        source_files: Vec<(Utf8PathBuf, String)>,
        core_ingot: InputIngot,
    ) -> (InputIngot, IndexSet<InputFile>) {
        let core_dependency = IngotDependency::new("core", core_ingot);
        let mut external_ingots = IndexSet::default();
        external_ingots.insert(core_dependency);
        let input_ingot = InputIngot::new(
            self,
            path.as_str(),
            IngotKind::Local,
            version.clone(),
            external_ingots,
        );

        let input_files = self.set_ingot_source_files(input_ingot, source_root, source_files);
        (input_ingot, input_files)
    }

    pub fn core_ingot(
        &mut self,
        path: &Utf8Path,
        version: &Version,
        source_root: &Utf8Path,
        source_files: Vec<(Utf8PathBuf, String)>,
    ) -> (InputIngot, IndexSet<InputFile>) {
        let input_ingot = InputIngot::new(
            self,
            path.as_str(),
            IngotKind::Core,
            version.clone(),
            IndexSet::default(),
        );

        let input_files = self.set_ingot_source_files(input_ingot, source_root, source_files);
        (input_ingot, input_files)
    }

    fn set_ingot_source_files(
        &mut self,
        ingot: InputIngot,
        root: &Utf8Path,
        files: Vec<(Utf8PathBuf, String)>,
    ) -> IndexSet<InputFile> {
        let input_files = files
            .into_iter()
            .map(|(path, content)| InputFile::new(self, path, content))
            .collect::<IndexSet<_>>();

        let root_file = *input_files
            .iter()
            .find(|input_file| {
                let input_path = input_file.path(self);
                if let (Ok(input_abs), Ok(root_abs)) = (
                    std::path::PathBuf::from(input_path).canonicalize(),
                    std::path::PathBuf::from(root).canonicalize(),
                ) {
                    input_abs == root_abs
                } else {
                    input_path == root
                }
            })
            .expect("missing root source file");

        ingot.set_files(self, input_files.clone());
        ingot.set_root_file(self, root_file);

        input_files
    }

    pub fn top_mod(&self, ingot: InputIngot, input: InputFile) -> TopLevelMod {
        map_file_to_mod(self, ingot, input)
    }
}

pub struct DiagnosticsCollection<'db>(Vec<Box<dyn DiagnosticVoucher<'db> + 'db>>);
impl<'db> DiagnosticsCollection<'db> {
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn emit(&self, db: &'db DriverDataBase) {
        let writer = BufferWriter::stderr(ColorChoice::Auto);
        let mut buffer = writer.buffer();
        let config = term::Config::default();

        for diag in self.finalize(db) {
            term::emit(&mut buffer, &config, &CsDbWrapper(db), &diag.to_cs(db)).unwrap();
        }

        eprintln!("{}", std::str::from_utf8(buffer.as_slice()).unwrap());
    }

    /// Format the accumulated diagnostics to a string.
    pub fn format_diags(&self, db: &'db DriverDataBase) -> String {
        let writer = BufferWriter::stderr(ColorChoice::Never);
        let mut buffer = writer.buffer();
        let config = term::Config::default();

        for diag in self.finalize(db) {
            term::emit(&mut buffer, &config, &CsDbWrapper(db), &diag.to_cs(db)).unwrap();
        }

        std::str::from_utf8(buffer.as_slice()).unwrap().to_string()
    }

    fn finalize(&self, db: &'db DriverDataBase) -> Vec<CompleteDiagnostic> {
        let mut diags: Vec<_> = self.0.iter().map(|d| d.to_complete(db)).collect();
        diags.sort_by(|lhs, rhs| match lhs.error_code.cmp(&rhs.error_code) {
            std::cmp::Ordering::Equal => lhs.primary_span().cmp(&rhs.primary_span()),
            ord => ord,
        });
        diags
    }
}

fn initialize_analysis_pass(db: &DriverDataBase) -> AnalysisPassManager<'_> {
    let mut pass_manager = AnalysisPassManager::new();
    pass_manager.add_module_pass(Box::new(ParsingPass::new(db)));
    // xxx pass_manager.add_module_pass(Box::new(DefConflictAnalysisPass::new(db)));
    pass_manager.add_module_pass(Box::new(ImportAnalysisPass::new(db)));
    // xxx pass_manager.add_module_pass(Box::new(PathAnalysisPass::new(db)));
    pass_manager.add_module_pass(Box::new(AdtDefAnalysisPass::new(db)));
    pass_manager.add_module_pass(Box::new(TypeAliasAnalysisPass::new(db)));
    pass_manager.add_module_pass(Box::new(TraitAnalysisPass::new(db)));
    pass_manager.add_module_pass(Box::new(ImplAnalysisPass::new(db)));
    pass_manager.add_module_pass(Box::new(ImplTraitAnalysisPass::new(db)));
    pass_manager.add_module_pass(Box::new(FuncAnalysisPass::new(db)));
    pass_manager.add_module_pass(Box::new(BodyAnalysisPass::new(db)));
    pass_manager
}
