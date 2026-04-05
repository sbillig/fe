use driver::DriverDataBase;
use hir::analysis::semantic::FieldIndex;
use mir2::{
    LayoutId, ScalarClass, ScalarRepr, VariantId, array_elem_size_bytes, enum_tag_size_bytes,
    enum_variant_field_offset_bytes, layout_size_bytes, struct_field_offset_bytes,
};

use crate::yul::{
    doc::YulDoc,
    errors::YulError,
    legalize::{
        YBlockId, YLocalId, YulAddressSpace, YulFunctionPlan, YulLocal, YulLocalRoot, YulPlace,
        YulPlaceRoot, YulValueClass,
    },
    state::FunctionState,
};

use super::{module::PackageIndex, util::prefix_yul_name};

#[derive(Clone)]
pub(super) struct RenderedValue<'db> {
    pub(super) setup: Vec<YulDoc>,
    pub(super) value: String,
    pub(super) class: YulValueClass<'db>,
}

pub(super) struct FunctionEmitter<'a, 'db> {
    pub(super) db: &'db DriverDataBase,
    pub(super) index: &'a PackageIndex<'a, 'db>,
    pub(super) plan: &'a YulFunctionPlan<'db>,
    pub(super) state: FunctionState,
}

pub(super) fn render_function_doc<'a, 'db>(
    index: &'a PackageIndex<'a, 'db>,
    plan: &'a YulFunctionPlan<'db>,
) -> Result<YulDoc, YulError> {
    FunctionEmitter::new(index.db, index, plan).render()
}

impl<'a, 'db> FunctionEmitter<'a, 'db> {
    fn new(
        db: &'db DriverDataBase,
        index: &'a PackageIndex<'a, 'db>,
        plan: &'a YulFunctionPlan<'db>,
    ) -> Self {
        Self {
            db,
            index,
            plan,
            state: FunctionState::new(plan),
        }
    }

    fn render(mut self) -> Result<YulDoc, YulError> {
        let signature_params = self.bind_params();
        let mut docs = self.render_prologue(&signature_params)?;
        docs.extend(self.render_body()?);
        let ret_suffix = self.plan.ret.as_ref().map(|_| " -> ret").unwrap_or("");
        let params = signature_params.join(", ");
        let caption = format!(
            "function {}({params}){ret_suffix} ",
            prefix_yul_name(&self.plan.symbol)
        );
        Ok(YulDoc::block(caption, docs))
    }

    fn bind_params(&mut self) -> Vec<String> {
        let mut out = Vec::with_capacity(self.plan.param_locals.len());
        for (idx, local) in self.plan.param_locals.iter().enumerate() {
            let name = format!("${}", self.plan.param_names[idx]);
            if !matches!(
                self.plan.locals[local.index()].root,
                YulLocalRoot::MemorySlot { .. }
            ) && self.plan.locals[local.index()].class.is_some()
            {
                self.state.assign_param_name(*local, name.clone());
            }
            out.push(name);
        }
        out
    }

    fn render_prologue(&mut self, param_inputs: &[String]) -> Result<Vec<YulDoc>, YulError> {
        let mut docs = Vec::new();
        for (idx, local) in self.plan.locals.iter().enumerate() {
            let YulLocalRoot::MemorySlot { class } = &local.root else {
                continue;
            };
            let root_name = self.state.root_name(YLocalId(idx as u32)).ok_or_else(|| {
                YulError::InvalidYulPackage(format!("missing root slot for local {idx}"))
            })?;
            docs.extend(self.alloc_memory_root_slot(root_name, class)?);
        }
        for (idx, local) in self.plan.locals.iter().enumerate() {
            let local_id = YLocalId(idx as u32);
            if self.plan.param_locals.contains(&local_id) {
                continue;
            }
            let Some(name) = self.state.value_name(local_id) else {
                continue;
            };
            let Some(class) = &local.class else {
                continue;
            };
            docs.push(YulDoc::line(format!(
                "let {name} := {}",
                Self::zero_for_class(class)
            )));
        }

        for (param_idx, local) in self.plan.param_locals.iter().enumerate() {
            let Some(class) = self.plan.locals[local.index()].class.as_ref() else {
                continue;
            };
            let input = param_inputs.get(param_idx).cloned().unwrap_or_default();
            let value = RenderedValue {
                setup: Vec::new(),
                value: input,
                class: class.clone(),
            };
            if matches!(
                self.plan.locals[local.index()].root,
                YulLocalRoot::MemorySlot { .. }
            ) {
                docs.extend(self.write_local_storage(*local, value)?);
            }
        }
        Ok(docs)
    }

    fn render_body(&mut self) -> Result<Vec<YulDoc>, YulError> {
        if self.plan.blocks.len() <= 1 {
            return self.render_linear_block(YBlockId(0));
        }
        self.render_pc_dispatch()
    }

    pub(super) fn render_linear_block(&mut self, block: YBlockId) -> Result<Vec<YulDoc>, YulError> {
        let block = self.plan.blocks.get(block.index()).ok_or_else(|| {
            YulError::InvalidYulPackage(format!("missing block {}", block.index()))
        })?;
        let mut docs = Vec::new();
        for stmt in &block.stmts {
            docs.extend(self.render_stmt(stmt)?);
        }
        docs.extend(self.render_terminator(&block.terminator)?);
        Ok(docs)
    }

    pub(super) fn local(&self, local: YLocalId) -> Result<&YulLocal<'db>, YulError> {
        self.plan
            .locals
            .get(local.index())
            .ok_or_else(|| YulError::InvalidYulPackage(format!("missing local {}", local.index())))
    }

    pub(super) fn local_class(&self, local: YLocalId) -> Result<YulValueClass<'db>, YulError> {
        self.local(local)?
            .class
            .clone()
            .ok_or_else(|| YulError::InvalidYulPackage(format!("local {local:?} is erased")))
    }

    pub(super) fn root_slot_name(&self, local: YLocalId) -> Result<&str, YulError> {
        self.state.root_name(local).ok_or_else(|| {
            let local_info = self.plan.locals.get(local.index());
            YulError::InvalidYulPackage(format!(
                "function `{}` local {local:?} does not have a root slot (local={local_info:?})",
                self.plan.symbol
            ))
        })
    }

    pub(super) fn alloc_memory_slot(
        &self,
        root_name: &str,
        class: &YulValueClass<'db>,
    ) -> Result<Vec<YulDoc>, YulError> {
        let bytes = self.class_size_bytes(class)?;
        Ok(vec![
            YulDoc::line(format!("let {root_name} := mload(0x40)")),
            YulDoc::block(
                format!("if iszero({root_name}) "),
                vec![YulDoc::line(format!("{root_name} := 0x80"))],
            ),
            YulDoc::line(format!("mstore(0x40, add({root_name}, {bytes}))")),
        ])
    }

    pub(super) fn alloc_memory_root_slot(
        &self,
        root_name: &str,
        class: &mir2::RuntimeClass<'db>,
    ) -> Result<Vec<YulDoc>, YulError> {
        let bytes = self.runtime_class_size_bytes(class)?;
        Ok(vec![
            YulDoc::line(format!("let {root_name} := mload(0x40)")),
            YulDoc::block(
                format!("if iszero({root_name}) "),
                vec![YulDoc::line(format!("{root_name} := 0x80"))],
            ),
            YulDoc::line(format!("mstore(0x40, add({root_name}, {bytes}))")),
        ])
    }

    pub(super) fn class_size_bytes(&self, class: &YulValueClass<'db>) -> Result<usize, YulError> {
        Ok(match class {
            YulValueClass::Word(_) => self.index.package_layout().word_size_bytes,
            YulValueClass::MemoryPtr { layout }
            | YulValueClass::CodePtr { layout }
            | YulValueClass::StoragePtr { layout }
            | YulValueClass::TransientPtr { layout }
            | YulValueClass::CalldataPtr { layout } => {
                layout_size_bytes(self.db, *layout, self.index.package_layout())
            }
        })
    }

    pub(super) fn runtime_class_size_bytes(
        &self,
        class: &mir2::RuntimeClass<'db>,
    ) -> Result<usize, YulError> {
        Ok(match class {
            mir2::RuntimeClass::Scalar(_) | mir2::RuntimeClass::RawAddr { .. } => {
                self.index.package_layout().word_size_bytes
            }
            mir2::RuntimeClass::AggregateValue { layout }
            | mir2::RuntimeClass::Handle { layout, .. } => {
                layout_size_bytes(self.db, *layout, self.index.package_layout())
            }
        })
    }

    pub(super) fn class_layout(
        &self,
        class: &YulValueClass<'db>,
    ) -> Result<LayoutId<'db>, YulError> {
        match class {
            YulValueClass::Word(_) => Err(YulError::Layout("word class has no layout".to_string())),
            YulValueClass::MemoryPtr { layout }
            | YulValueClass::CodePtr { layout }
            | YulValueClass::StoragePtr { layout }
            | YulValueClass::TransientPtr { layout }
            | YulValueClass::CalldataPtr { layout } => Ok(*layout),
        }
    }

    pub(super) fn field_offset_bytes(&self, layout: LayoutId<'db>, field: FieldIndex) -> usize {
        struct_field_offset_bytes(self.db, layout, field, self.index.package_layout())
    }

    pub(super) fn variant_field_offset_bytes(
        &self,
        layout: LayoutId<'db>,
        variant: VariantId<'db>,
        field: FieldIndex,
    ) -> usize {
        enum_tag_size_bytes(self.db, layout, self.index.package_layout())
            + enum_variant_field_offset_bytes(
                self.db,
                layout,
                variant,
                field,
                self.index.package_layout(),
            )
    }

    pub(super) fn index_stride_bytes(&self, layout: LayoutId<'db>) -> usize {
        array_elem_size_bytes(self.db, layout, self.index.package_layout())
    }

    pub(super) fn storage_word_offset_bytes(&self, offset: usize) -> usize {
        offset / self.index.package_layout().word_size_bytes
    }

    pub(super) fn zero_for_class(class: &YulValueClass<'db>) -> String {
        match class {
            YulValueClass::Word(_)
            | YulValueClass::MemoryPtr { .. }
            | YulValueClass::CodePtr { .. }
            | YulValueClass::StoragePtr { .. }
            | YulValueClass::TransientPtr { .. }
            | YulValueClass::CalldataPtr { .. } => "0".to_string(),
        }
    }

    pub(super) fn raw_word_mask(class: &ScalarClass<'db>) -> Option<String> {
        match class.repr {
            ScalarRepr::Bool => Some("0xff".to_string()),
            ScalarRepr::Int { bits, .. } | ScalarRepr::Address { bits } if bits < 256 => {
                Some(format!("0x{}", "ff".repeat(bits.div_ceil(8) as usize)))
            }
            ScalarRepr::FixedBytes { len } if len < 32 => {
                Some(format!("0x{}", "ff".repeat(len as usize)))
            }
            ScalarRepr::Int { .. } | ScalarRepr::FixedBytes { .. } | ScalarRepr::Address { .. } => {
                None
            }
        }
    }

    pub(super) fn scalar_word_expr(&self, local: YLocalId) -> Result<String, YulError> {
        let class = self.local_class(local)?;
        match &class {
            YulValueClass::Word(_) => {
                if let Some(name) = self.state.value_name(local) {
                    Ok(name.to_string())
                } else {
                    let root = self.root_slot_name(local)?;
                    Ok(format!("mload({root})"))
                }
            }
            _ => Err(YulError::InvalidYulPackage(format!(
                "local {local:?} is not word-valued: {class:?}"
            ))),
        }
    }

    pub(super) fn local_value(&self, local: YLocalId) -> Result<RenderedValue<'db>, YulError> {
        let class = self.local_class(local)?;
        let value = if let Some(name) = self.state.value_name(local) {
            name.to_string()
        } else {
            let root = self.root_slot_name(local)?;
            match class {
                YulValueClass::Word(_) => format!("mload({root})"),
                YulValueClass::MemoryPtr { .. } => root.to_string(),
                YulValueClass::CodePtr { .. }
                | YulValueClass::StoragePtr { .. }
                | YulValueClass::TransientPtr { .. }
                | YulValueClass::CalldataPtr { .. } => root.to_string(),
            }
        };
        Ok(RenderedValue {
            setup: Vec::new(),
            value,
            class,
        })
    }

    pub(super) fn root_space_for_class(
        class: &YulValueClass<'db>,
    ) -> Result<YulAddressSpace, YulError> {
        match class {
            YulValueClass::MemoryPtr { .. } => Ok(YulAddressSpace::Memory),
            YulValueClass::CodePtr { .. } => Ok(YulAddressSpace::Code),
            YulValueClass::StoragePtr { .. } => Ok(YulAddressSpace::Storage),
            YulValueClass::TransientPtr { .. } => Ok(YulAddressSpace::Transient),
            YulValueClass::CalldataPtr { .. } => Ok(YulAddressSpace::Calldata),
            YulValueClass::Word(_) => Err(YulError::Layout(
                "word class cannot act as a place root".to_string(),
            )),
        }
    }

    pub(super) fn write_local_storage(
        &mut self,
        local: YLocalId,
        src: RenderedValue<'db>,
    ) -> Result<Vec<YulDoc>, YulError> {
        let Some(class) = self.plan.locals[local.index()].class.as_ref() else {
            return Ok(src.setup);
        };
        if let Some(name) = self.state.value_name(local) {
            let mut docs = src.setup;
            docs.push(YulDoc::line(format!("{name} := {}", src.value)));
            return Ok(docs);
        }
        let mut docs = src.setup.clone();
        docs.extend(self.write_place_from_value(
            YulPlace {
                root: YulPlaceRoot::Slot(local),
                path: Box::default(),
                storage_kind: match &self.plan.locals[local.index()].root {
                    YulLocalRoot::MemorySlot { class } => match class {
                        mir2::RuntimeClass::AggregateValue { .. } => {
                            crate::yul::legalize::YulStorageKind::Bytes
                        }
                        mir2::RuntimeClass::Scalar(_)
                        | mir2::RuntimeClass::Handle { .. }
                        | mir2::RuntimeClass::RawAddr { .. } => {
                            crate::yul::legalize::YulStorageKind::Cell
                        }
                    },
                    YulLocalRoot::None | YulLocalRoot::PtrRoot { .. } => {
                        crate::yul::legalize::YulStorageKind::Bytes
                    }
                },
                runtime_result_class: match &self.plan.locals[local.index()].root {
                    YulLocalRoot::MemorySlot { class } => class.clone(),
                    YulLocalRoot::None | YulLocalRoot::PtrRoot { .. } => match class {
                        YulValueClass::Word(word) => mir2::RuntimeClass::Scalar(word.clone()),
                        YulValueClass::MemoryPtr { layout }
                        | YulValueClass::CodePtr { layout }
                        | YulValueClass::StoragePtr { layout }
                        | YulValueClass::TransientPtr { layout }
                        | YulValueClass::CalldataPtr { layout } => {
                            mir2::RuntimeClass::AggregateValue { layout: *layout }
                        }
                    },
                },
                result_class: class.clone(),
            },
            src,
        )?);
        Ok(docs)
    }
}
