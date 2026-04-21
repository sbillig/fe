use std::collections::{HashMap, HashSet};

use driver::DriverDataBase;

use crate::{
    TestMetadata, TestModuleOutput,
    test_output::{TestRootMetadataError, runtime_test_root_metadata},
    yul::{
        doc::{YulDoc, render_docs},
        errors::YulError,
        legalize::{YFunctionId, YulFunctionPlan, YulObjectPlan, YulPackage, YulSectionPlan},
    },
};

use super::{
    function::{FunctionEmitter, render_function_doc},
    util::section_object_label,
};

pub(super) struct PackageIndex<'a, 'db> {
    pub(super) db: &'db DriverDataBase,
    functions: HashMap<YFunctionId, &'a YulFunctionPlan<'db>>,
    sections: HashMap<(String, mir2::RuntimeSectionName), &'a YulSectionPlan<'db>>,
    const_region_labels: HashMap<mir2::ConstRegionId<'db>, &'a str>,
    code_region_labels: HashMap<mir2::RuntimeCodeRegion<'db>, &'a str>,
}

impl<'a, 'db> PackageIndex<'a, 'db> {
    fn new(db: &'db DriverDataBase, package: &'a YulPackage<'db>) -> Self {
        let functions = package
            .functions
            .iter()
            .map(|function| (function.id, function))
            .collect();
        let sections = package
            .objects
            .iter()
            .flat_map(|object| {
                object
                    .sections
                    .iter()
                    .map(move |section| ((object.name.clone(), section.name.clone()), section))
            })
            .collect();
        let const_region_labels = package
            .const_region_labels
            .iter()
            .map(|(region, label)| (*region, label.as_str()))
            .collect();
        let code_region_labels = package
            .code_region_labels
            .iter()
            .map(|(region, label)| (*region, label.as_str()))
            .collect();
        Self {
            db,
            functions,
            sections,
            const_region_labels,
            code_region_labels,
        }
    }

    pub(super) fn function(
        &self,
        function: YFunctionId,
    ) -> Result<&'a YulFunctionPlan<'db>, YulError> {
        self.functions.get(&function).copied().ok_or_else(|| {
            YulError::InvalidYulPackage(format!(
                "missing legalized function plan for `{:?}`",
                function
            ))
        })
    }

    pub(super) fn section(
        &self,
        object: &str,
        section: &mir2::RuntimeSectionName,
    ) -> Result<&'a YulSectionPlan<'db>, YulError> {
        self.sections
            .get(&(object.to_string(), section.clone()))
            .copied()
            .ok_or_else(|| {
                YulError::InvalidYulPackage(format!(
                    "missing legalized section `{section:?}` for object `{object}`"
                ))
            })
    }

    pub(super) fn const_label(
        &self,
        region: mir2::ConstRegionId<'db>,
    ) -> Result<&'a str, YulError> {
        self.const_region_labels
            .get(&region)
            .copied()
            .ok_or_else(|| {
                YulError::InvalidYulPackage(format!("missing const label for region `{region:?}`"))
            })
    }

    pub(super) fn code_region_label(
        &self,
        region: mir2::RuntimeCodeRegion<'db>,
    ) -> Result<&'a str, YulError> {
        self.code_region_labels
            .get(&region)
            .copied()
            .ok_or_else(|| {
                YulError::InvalidYulPackage(format!("missing code region label for `{region:?}`"))
            })
    }

    pub(super) fn package_layout(&self) -> crate::TargetDataLayout {
        crate::EVM_LAYOUT
    }
}

pub fn emit_runtime_package_yul<'db>(
    db: &'db DriverDataBase,
    package: &YulPackage<'db>,
) -> Result<String, YulError> {
    let index = PackageIndex::new(db, package);
    let root_objects = root_objects_in_emit_order(package);
    let docs = root_objects
        .into_iter()
        .map(|object| {
            let mut rendered_sections = HashSet::default();
            render_root_object(&index, object, &mut rendered_sections, &mut Vec::new())
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut lines = Vec::new();
    render_docs(&docs, 0, &mut lines);
    Ok(lines.join("\n"))
}

pub fn emit_runtime_package_object_yul<'db>(
    db: &'db DriverDataBase,
    package: &YulPackage<'db>,
    object_name: &str,
) -> Result<String, YulError> {
    let index = PackageIndex::new(db, package);
    let object = package
        .objects
        .iter()
        .find(|object| object.root && object.name == object_name)
        .ok_or_else(|| {
            YulError::InvalidYulPackage(format!(
                "missing root object `{object_name}` in Yul package"
            ))
        })?;
    let mut rendered_sections = HashSet::default();
    let doc = render_root_object(&index, object, &mut rendered_sections, &mut Vec::new())?;
    let mut lines = Vec::new();
    render_docs(&[doc], 0, &mut lines);
    Ok(lines.join("\n"))
}

pub fn emit_test_runtime_package_yul<'db>(
    db: &'db DriverDataBase,
    package: &YulPackage<'db>,
    filter: Option<&str>,
) -> Result<TestModuleOutput, YulError> {
    let index = PackageIndex::new(db, package);
    let root_objects = root_objects_in_emit_order(package);
    let mut tests = Vec::new();
    for object in root_objects {
        let Some(section) = object
            .sections
            .iter()
            .find(|section| matches!(section.name, mir2::RuntimeSectionName::Test(_)))
        else {
            continue;
        };
        let metadata = test_metadata_for_section(&index, section)?;
        if let Some(filter) = filter
            && !metadata.display_name.contains(filter)
            && !metadata.hir_name.contains(filter)
            && !index.function(section.entry)?.symbol.contains(filter)
        {
            continue;
        }
        let mut rendered_sections = HashSet::default();
        let mut stack = Vec::new();
        let wrapper_name = format!("test_{}", object.name);
        let doc = render_test_root_object(
            &index,
            object,
            &wrapper_name,
            &mut rendered_sections,
            &mut stack,
        )?;
        let mut lines = Vec::new();
        render_docs(&[doc], 0, &mut lines);
        tests.push(TestMetadata {
            display_name: metadata.display_name,
            hir_name: metadata.hir_name,
            symbol_name: index.function(section.entry)?.symbol.clone(),
            object_name: wrapper_name,
            yul: lines.join("\n"),
            bytecode: Vec::new(),
            sonatina_observability_json: None,
            value_param_count: 0,
            effect_param_count: 0,
            init_bytecode: Vec::new(),
            expected_revert: metadata.expected_revert,
            initial_balance: metadata.initial_balance,
        });
    }
    Ok(TestModuleOutput { tests })
}

fn root_objects_in_emit_order<'a, 'db>(
    package: &'a YulPackage<'db>,
) -> Vec<&'a YulObjectPlan<'db>> {
    let mut roots = package
        .objects
        .iter()
        .filter(|object| object.root)
        .collect::<Vec<_>>();
    roots.sort_by(|lhs, rhs| lhs.name.cmp(&rhs.name));
    if let Some(primary) = &package.primary_object
        && let Some(idx) = roots.iter().position(|object| &object.name == primary)
    {
        let primary = roots.remove(idx);
        roots.insert(0, primary);
    }
    roots
}

fn render_root_object<'a, 'db>(
    index: &PackageIndex<'a, 'db>,
    object: &'a YulObjectPlan<'db>,
    rendered_sections: &mut HashSet<(String, mir2::RuntimeSectionName)>,
    stack: &mut Vec<(String, mir2::RuntimeSectionName)>,
) -> Result<YulDoc, YulError> {
    let body = render_root_object_body(index, object, &object.name, rendered_sections, stack)?;
    Ok(YulDoc::block(format!("object \"{}\" ", object.name), body))
}

fn render_test_root_object<'a, 'db>(
    index: &PackageIndex<'a, 'db>,
    object: &'a YulObjectPlan<'db>,
    wrapper_name: &str,
    rendered_sections: &mut HashSet<(String, mir2::RuntimeSectionName)>,
    stack: &mut Vec<(String, mir2::RuntimeSectionName)>,
) -> Result<YulDoc, YulError> {
    let runtime = YulDoc::block(
        "object \"runtime\" ",
        render_root_object_body(index, object, "runtime", rendered_sections, stack)?,
    );
    let init = YulDoc::block(
        "code ",
        vec![
            YulDoc::line("datacopy(0, dataoffset(\"runtime\"), datasize(\"runtime\"))"),
            YulDoc::line("return(0, datasize(\"runtime\"))"),
        ],
    );
    Ok(YulDoc::block(
        format!("object \"{wrapper_name}\" "),
        vec![init, runtime],
    ))
}

fn render_root_object_body<'a, 'db>(
    index: &PackageIndex<'a, 'db>,
    object: &'a YulObjectPlan<'db>,
    object_label: &str,
    rendered_sections: &mut HashSet<(String, mir2::RuntimeSectionName)>,
    stack: &mut Vec<(String, mir2::RuntimeSectionName)>,
) -> Result<Vec<YulDoc>, YulError> {
    let primary = object.sections.first().ok_or_else(|| {
        YulError::InvalidYulPackage(format!("object `{}` has no sections", object.name))
    })?;
    rendered_sections.insert((object.name.clone(), primary.name.clone()));
    let mut body = render_section_body(index, primary, object_label, rendered_sections, stack)?;
    for section in &object.sections {
        let key = (object.name.clone(), section.name.clone());
        if rendered_sections.contains(&key) {
            continue;
        }
        rendered_sections.insert(key);
        body.push(render_nested_section(
            index,
            section,
            section_object_label(&section.name),
            rendered_sections,
            stack,
        )?);
    }
    Ok(body)
}

fn render_nested_section<'a, 'db>(
    index: &PackageIndex<'a, 'db>,
    section: &'a YulSectionPlan<'db>,
    label: String,
    rendered_sections: &mut HashSet<(String, mir2::RuntimeSectionName)>,
    stack: &mut Vec<(String, mir2::RuntimeSectionName)>,
) -> Result<YulDoc, YulError> {
    let key = (section.object_name.clone(), section.name.clone());
    if stack.contains(&key) {
        return Err(YulError::InvalidYulPackage(format!(
            "cyclic Yul section embed detected for `{}` / `{:?}`",
            section.object_name, section.name
        )));
    }
    rendered_sections.insert(key.clone());
    stack.push(key);
    let body = render_section_body(index, section, &label, rendered_sections, stack)?;
    let _ = stack.pop();
    Ok(YulDoc::block(format!("object \"{label}\" "), body))
}

fn render_section_body<'a, 'db>(
    index: &PackageIndex<'a, 'db>,
    section: &'a YulSectionPlan<'db>,
    object_label: &str,
    rendered_sections: &mut HashSet<(String, mir2::RuntimeSectionName)>,
    stack: &mut Vec<(String, mir2::RuntimeSectionName)>,
) -> Result<Vec<YulDoc>, YulError> {
    let mut body = vec![YulDoc::block(
        "code ",
        render_section_code(index, section, object_label)?,
    )];
    for embed in &section.embeds {
        let source = index.section(&embed.source_object, &embed.source_section)?;
        body.push(render_nested_section(
            index,
            source,
            embed.label.clone(),
            rendered_sections,
            stack,
        )?);
    }
    for region in &section.const_regions {
        body.push(YulDoc::line(format!(
            "data \"{}\" hex\"{}\"",
            region.label,
            hex::encode(&region.bytes)
        )));
    }
    Ok(body)
}

fn render_section_code<'a, 'db>(
    index: &PackageIndex<'a, 'db>,
    section: &'a YulSectionPlan<'db>,
    object_label: &str,
) -> Result<Vec<YulDoc>, YulError> {
    let mut docs = Vec::new();
    for function in &section.functions {
        docs.push(render_function_doc(
            index,
            index.function(*function)?,
            object_label,
        )?);
    }
    docs.extend(render_section_entry(index, section)?);
    Ok(docs)
}

fn render_section_entry<'a, 'db>(
    index: &PackageIndex<'a, 'db>,
    section: &'a YulSectionPlan<'db>,
) -> Result<Vec<YulDoc>, YulError> {
    let entry = index.function(section.entry)?;
    let mut docs = Vec::new();
    let args = entry
        .params
        .iter()
        .zip(&entry.param_kinds)
        .map(|(class, kind)| match kind {
            crate::yul::legalize::YulParamKind::Effect(_) => {
                Ok(FunctionEmitter::zero_for_class(class))
            }
            crate::yul::legalize::YulParamKind::Visible(_) => {
                Err(YulError::InvalidYulPackage(format!(
                    "root section entry `{}` unexpectedly requires visible arguments",
                    entry.symbol
                )))
            }
        })
        .collect::<Result<Vec<_>, _>>()?;
    let call = format!(
        "{}({})",
        super::util::prefix_yul_name(&entry.symbol),
        args.join(", ")
    );
    match section.name {
        mir2::RuntimeSectionName::Init
        | mir2::RuntimeSectionName::Runtime
        | mir2::RuntimeSectionName::Main
        | mir2::RuntimeSectionName::Test(_)
        | mir2::RuntimeSectionName::CodeRegion(_) => {
            docs.push(YulDoc::line(call));
        }
    }
    Ok(docs)
}

type TestSectionMetadata = crate::test_output::TestRootMetadata;

fn test_metadata_for_section<'a, 'db>(
    index: &PackageIndex<'a, 'db>,
    section: &YulSectionPlan<'db>,
) -> Result<TestSectionMetadata, YulError> {
    let entry = index.function(section.entry)?;
    runtime_test_root_metadata(
        index.db,
        &entry.runtime_function.owner(index.db),
        &section.name,
    )
    .map_err(|err| match err {
        TestRootMetadataError::InvalidPackage(message) => YulError::InvalidYulPackage(message),
        TestRootMetadataError::Unsupported(message) => YulError::Unsupported(message),
    })
}
