use crate::yul::{
    doc::YulDoc,
    errors::YulError,
    legalize::{YStmt, YulAddressSpace, YulPlace, YulValueClass},
};

use super::function::{FunctionEmitter, RenderedValue};

#[derive(Clone, Copy)]
enum WordLoadSpace {
    Memory,
    Storage,
    Transient,
}

impl<'a, 'db> FunctionEmitter<'a, 'db> {
    pub(super) fn render_stmt(&mut self, stmt: &YStmt<'db>) -> Result<Vec<YulDoc>, YulError> {
        match stmt {
            YStmt::Assign { dst, expr } => {
                let expected = self.plan.locals[dst.index()].class.as_ref();
                let value = self.render_expr(expr, expected)?;
                self.write_local_storage(*dst, value)
            }
            YStmt::Call { callee, args } => {
                let mut docs = Vec::new();
                let args = args
                    .iter()
                    .map(|arg| self.local_value(*arg))
                    .collect::<Result<Vec<_>, _>>()?;
                for arg in &args {
                    docs.extend(arg.setup.clone());
                }
                let rendered_args = args
                    .into_iter()
                    .map(|arg| arg.value)
                    .collect::<Vec<_>>()
                    .join(", ");
                let callee_plan = self.index.function(*callee)?;
                let call = format!(
                    "{}({rendered_args})",
                    super::util::prefix_yul_name(&callee_plan.symbol)
                );
                docs.push(YulDoc::line(if callee_plan.ret.is_some() {
                    format!("pop({call})")
                } else {
                    call
                }));
                Ok(docs)
            }
            YStmt::Builtin(builtin) => self.render_builtin_stmt(builtin),
            YStmt::Store { dst, src } => {
                let src = self.local_value(*src)?;
                self.write_place_from_value(dst.clone(), src)
            }
            YStmt::CopyInto { dst, src } => {
                let src = self.local_value(*src)?;
                self.write_place_from_value(dst.clone(), src)
            }
            YStmt::EnumAssertVariant {
                value: _,
                variant: _,
            } => Ok(Vec::new()),
            YStmt::EnumSetTag { root, variant } => {
                let root = self.local_value(*root)?;
                let layout = self.class_layout(&root.class)?;
                let tag_class = self.enum_tag_class(layout)?;
                self.write_enum_tag_to_addr(
                    Self::root_space_for_class(&root.class)?,
                    root.value,
                    RenderedValue {
                        setup: root.setup,
                        value: variant.index.to_string(),
                        class: YulValueClass::Word(tag_class),
                    },
                )
            }
            YStmt::EnumWriteVariant {
                root,
                variant,
                fields,
            } => {
                let root = self.local_value(*root)?;
                let layout = self.class_layout(&root.class)?;
                let mut docs = root.setup;
                docs.extend(self.write_enum_variant(
                    &root.value,
                    Self::root_space_for_class(&root.class)?,
                    layout,
                    *variant,
                    fields,
                )?);
                Ok(docs)
            }
        }
    }

    pub(super) fn write_place_from_value(
        &mut self,
        dst: YulPlace<'db>,
        src: RenderedValue<'db>,
    ) -> Result<Vec<YulDoc>, YulError> {
        if let Some(local) = self.direct_word_slot_local(&dst) {
            return self.write_local_storage(local, src);
        }
        let (mut docs, addr, space) = self.address_of_place(&dst)?;
        docs.extend(src.setup.clone());
        if matches!(dst.result_class, YulValueClass::Word(_)) {
            docs.extend(if dst.packed_byte_access {
                self.write_packed_byte_scalar_to_addr(space, addr, src)?
            } else {
                self.write_scalar_to_addr(space, addr, src)?
            });
        } else {
            match dst.storage_kind {
                crate::yul::legalize::YulStorageKind::Cell => {
                    docs.extend(self.write_transport_word_to_addr(space, addr, src)?);
                }
                crate::yul::legalize::YulStorageKind::Bytes => {
                    docs.extend(self.copy_into_addr(dst.result_class.clone(), space, addr, src)?);
                }
            }
        }
        Ok(docs)
    }

    pub(super) fn copy_into_addr(
        &mut self,
        dst_class: YulValueClass<'db>,
        dst_space: YulAddressSpace,
        dst_addr: String,
        src: RenderedValue<'db>,
    ) -> Result<Vec<YulDoc>, YulError> {
        let mut docs = Vec::new();
        let size = self.class_size_bytes(&dst_class)?;
        match (&dst_class, &src.class, dst_space) {
            (_, YulValueClass::MemoryPtr { .. }, YulAddressSpace::Memory) => {
                docs.extend(self.copy_word_chunks(
                    YulAddressSpace::Memory,
                    dst_addr,
                    WordLoadSpace::Memory,
                    src.value,
                    size,
                ));
            }
            (_, YulValueClass::CodePtr { .. }, YulAddressSpace::Memory) => {
                docs.push(YulDoc::line(format!(
                    "datacopy({dst_addr}, {}, {size})",
                    src.value
                )));
            }
            (_, YulValueClass::CalldataPtr { .. }, YulAddressSpace::Memory) => {
                docs.push(YulDoc::line(format!(
                    "calldatacopy({dst_addr}, {}, {size})",
                    src.value
                )));
            }
            (_, YulValueClass::MemoryPtr { .. }, YulAddressSpace::Storage) => {
                docs.extend(self.copy_word_chunks(
                    YulAddressSpace::Storage,
                    dst_addr,
                    WordLoadSpace::Memory,
                    src.value,
                    size,
                ));
            }
            (_, YulValueClass::MemoryPtr { .. }, YulAddressSpace::Transient) => {
                docs.extend(self.copy_word_chunks(
                    YulAddressSpace::Transient,
                    dst_addr,
                    WordLoadSpace::Memory,
                    src.value,
                    size,
                ));
            }
            (_, YulValueClass::StoragePtr { .. }, YulAddressSpace::Memory) => {
                docs.extend(self.copy_word_chunks(
                    YulAddressSpace::Memory,
                    dst_addr,
                    WordLoadSpace::Storage,
                    src.value,
                    size,
                ));
            }
            (_, YulValueClass::StoragePtr { .. }, YulAddressSpace::Storage) => {
                docs.extend(self.copy_word_chunks(
                    YulAddressSpace::Storage,
                    dst_addr,
                    WordLoadSpace::Storage,
                    src.value,
                    size,
                ));
            }
            (_, YulValueClass::CodePtr { .. }, YulAddressSpace::Storage) => {
                docs.extend(self.copy_non_memory_source_via_scratch(
                    &dst_class,
                    YulAddressSpace::Storage,
                    dst_addr,
                    YulAddressSpace::Code,
                    src.value,
                    size,
                )?);
            }
            (_, YulValueClass::CodePtr { .. }, YulAddressSpace::Transient) => {
                docs.extend(self.copy_non_memory_source_via_scratch(
                    &dst_class,
                    YulAddressSpace::Transient,
                    dst_addr,
                    YulAddressSpace::Code,
                    src.value,
                    size,
                )?);
            }
            (_, YulValueClass::TransientPtr { .. }, YulAddressSpace::Memory) => {
                docs.extend(self.copy_word_chunks(
                    YulAddressSpace::Memory,
                    dst_addr,
                    WordLoadSpace::Transient,
                    src.value,
                    size,
                ));
            }
            (_, YulValueClass::TransientPtr { .. }, YulAddressSpace::Transient) => {
                docs.extend(self.copy_word_chunks(
                    YulAddressSpace::Transient,
                    dst_addr,
                    WordLoadSpace::Transient,
                    src.value,
                    size,
                ));
            }
            (_, YulValueClass::CalldataPtr { .. }, YulAddressSpace::Storage) => {
                docs.extend(self.copy_non_memory_source_via_scratch(
                    &dst_class,
                    YulAddressSpace::Storage,
                    dst_addr,
                    YulAddressSpace::Calldata,
                    src.value,
                    size,
                )?);
            }
            (_, YulValueClass::CalldataPtr { .. }, YulAddressSpace::Transient) => {
                docs.extend(self.copy_non_memory_source_via_scratch(
                    &dst_class,
                    YulAddressSpace::Transient,
                    dst_addr,
                    YulAddressSpace::Calldata,
                    src.value,
                    size,
                )?);
            }
            (_, YulValueClass::Word(_), YulAddressSpace::Memory) => {
                docs.push(YulDoc::line(format!("mstore({dst_addr}, {})", src.value)));
            }
            (_, YulValueClass::Word(_), YulAddressSpace::Storage) => {
                docs.push(YulDoc::line(format!("sstore({dst_addr}, {})", src.value)));
            }
            (_, YulValueClass::Word(_), YulAddressSpace::Transient) => {
                docs.push(YulDoc::line(format!("tstore({dst_addr}, {})", src.value)));
            }
            (_, _, YulAddressSpace::Code | YulAddressSpace::Calldata) => {
                return Err(YulError::Unsupported(format!(
                    "cannot write aggregate values into {dst_space:?}"
                )));
            }
            _ => {
                return Err(YulError::Unsupported(format!(
                    "unsupported Yul aggregate copy from {:?} into {:?} in {dst_space:?}",
                    src.class, dst_class
                )));
            }
        }
        Ok(docs)
    }

    fn copy_non_memory_source_via_scratch(
        &mut self,
        dst_class: &YulValueClass<'db>,
        dst_space: YulAddressSpace,
        dst_addr: String,
        src_space: YulAddressSpace,
        src_addr: String,
        size: usize,
    ) -> Result<Vec<YulDoc>, YulError> {
        let scratch = self.state.alloc_temp();
        let mut docs = self.alloc_memory_slot(&scratch, dst_class)?;
        docs.push(YulDoc::line(match src_space {
            YulAddressSpace::Code => format!("datacopy({scratch}, {src_addr}, {size})"),
            YulAddressSpace::Calldata => {
                format!("calldatacopy({scratch}, {src_addr}, {size})")
            }
            YulAddressSpace::Memory | YulAddressSpace::Storage | YulAddressSpace::Transient => {
                return Err(YulError::InvalidYulPackage(format!(
                    "scratch staging is only valid for code/calldata sources, found {src_space:?}"
                )));
            }
        }));
        docs.extend(self.copy_word_chunks(
            dst_space,
            dst_addr,
            WordLoadSpace::Memory,
            scratch,
            size,
        ));
        Ok(docs)
    }

    fn copy_word_chunks(
        &self,
        dst_space: YulAddressSpace,
        dst_addr: String,
        src_space: WordLoadSpace,
        src_addr: String,
        size: usize,
    ) -> Vec<YulDoc> {
        let mut docs = Vec::new();
        let word_size = self.index.package_layout().word_size_bytes;
        let words = size.div_ceil(word_size);
        let chunk_addr = |base: &str, offset| {
            if offset == 0 {
                base.to_string()
            } else {
                format!("add({base}, {offset})")
            }
        };
        for idx in 0..words {
            let dst_offset = match dst_space {
                YulAddressSpace::Memory => idx * word_size,
                YulAddressSpace::Storage | YulAddressSpace::Transient => idx,
                YulAddressSpace::Calldata | YulAddressSpace::Code => unreachable!(),
            };
            let src_offset = match src_space {
                WordLoadSpace::Memory => idx * word_size,
                WordLoadSpace::Storage | WordLoadSpace::Transient => idx,
            };
            let dst = chunk_addr(&dst_addr, dst_offset);
            let src = chunk_addr(&src_addr, src_offset);
            let loaded = match src_space {
                WordLoadSpace::Memory => format!("mload({src})"),
                WordLoadSpace::Storage => format!("sload({src})"),
                WordLoadSpace::Transient => format!("tload({src})"),
            };
            let stored = match dst_space {
                YulAddressSpace::Memory => format!("mstore({dst}, {loaded})"),
                YulAddressSpace::Storage => format!("sstore({dst}, {loaded})"),
                YulAddressSpace::Transient => format!("tstore({dst}, {loaded})"),
                YulAddressSpace::Calldata | YulAddressSpace::Code => unreachable!(),
            };
            docs.push(YulDoc::line(stored));
        }
        docs
    }

    fn write_scalar_to_addr(
        &self,
        space: YulAddressSpace,
        addr: String,
        src: RenderedValue<'db>,
    ) -> Result<Vec<YulDoc>, YulError> {
        let YulValueClass::Word(_) = src.class else {
            return Err(YulError::Unsupported(format!(
                "scalar store requires a word source, found {:?}",
                src.class
            )));
        };
        let YulValueClass::Word(word) = &src.class else {
            unreachable!("checked above")
        };
        let value = self.canonicalize_scalar_expr(src.value, word);
        Ok(match space {
            YulAddressSpace::Memory => vec![YulDoc::line(format!("mstore({addr}, {value})"))],
            YulAddressSpace::Storage => {
                vec![YulDoc::line(format!("sstore({addr}, {value})"))]
            }
            YulAddressSpace::Transient => {
                vec![YulDoc::line(format!("tstore({addr}, {value})"))]
            }
            YulAddressSpace::Calldata | YulAddressSpace::Code => {
                return Err(YulError::Unsupported(format!(
                    "scalar store into {space:?} is not supported"
                )));
            }
        })
    }

    fn write_packed_byte_scalar_to_addr(
        &self,
        space: YulAddressSpace,
        addr: String,
        src: RenderedValue<'db>,
    ) -> Result<Vec<YulDoc>, YulError> {
        let YulValueClass::Word(word) = &src.class else {
            return Err(YulError::Unsupported(format!(
                "packed scalar store requires a word source, found {:?}",
                src.class
            )));
        };
        let value = self.canonicalize_scalar_expr(src.value, word);
        Ok(match space {
            YulAddressSpace::Memory => vec![YulDoc::line(format!("mstore8({addr}, {value})"))],
            YulAddressSpace::Storage
            | YulAddressSpace::Transient
            | YulAddressSpace::Calldata
            | YulAddressSpace::Code => {
                return Err(YulError::Unsupported(format!(
                    "packed byte scalar store into {space:?} is not supported"
                )));
            }
        })
    }

    fn write_transport_word_to_addr(
        &self,
        space: YulAddressSpace,
        addr: String,
        src: RenderedValue<'db>,
    ) -> Result<Vec<YulDoc>, YulError> {
        Ok(match space {
            YulAddressSpace::Memory => vec![YulDoc::line(format!("mstore({addr}, {})", src.value))],
            YulAddressSpace::Storage => {
                vec![YulDoc::line(format!("sstore({addr}, {})", src.value))]
            }
            YulAddressSpace::Transient => {
                vec![YulDoc::line(format!("tstore({addr}, {})", src.value))]
            }
            YulAddressSpace::Calldata | YulAddressSpace::Code => {
                return Err(YulError::Unsupported(format!(
                    "cannot write a one-word transport into {space:?}"
                )));
            }
        })
    }

    fn write_enum_tag_to_addr(
        &self,
        space: YulAddressSpace,
        addr: String,
        src: RenderedValue<'db>,
    ) -> Result<Vec<YulDoc>, YulError> {
        match space {
            YulAddressSpace::Memory => self.write_packed_byte_scalar_to_addr(space, addr, src),
            YulAddressSpace::Storage | YulAddressSpace::Transient => {
                self.write_scalar_to_addr(space, addr, src)
            }
            YulAddressSpace::Code | YulAddressSpace::Calldata => Err(YulError::Unsupported(
                format!("enum tag store into {space:?} is not supported"),
            )),
        }
    }

    pub(super) fn write_enum_variant(
        &mut self,
        root: &str,
        space: YulAddressSpace,
        layout: mir2::LayoutId<'db>,
        variant: mir2::VariantId<'db>,
        fields: &[crate::yul::legalize::YLocalId],
    ) -> Result<Vec<YulDoc>, YulError> {
        let tag_class = self.enum_tag_class(layout)?;
        let mut docs = self.write_enum_tag_to_addr(
            space,
            root.to_string(),
            RenderedValue {
                setup: Vec::new(),
                value: variant.index.to_string(),
                class: YulValueClass::Word(tag_class),
            },
        )?;
        for (idx, field) in fields.iter().enumerate() {
            let src = self.local_value(*field)?;
            let offset = self.variant_field_offset_bytes(
                layout,
                variant,
                hir::analysis::semantic::FieldIndex(idx as u16),
            );
            let addr = match space {
                YulAddressSpace::Memory | YulAddressSpace::Code | YulAddressSpace::Calldata => {
                    format!("add({root}, {offset})")
                }
                YulAddressSpace::Storage | YulAddressSpace::Transient => {
                    format!("add({root}, {})", self.storage_word_offset_bytes(offset))
                }
            };
            if matches!(src.class, YulValueClass::Word(_)) {
                docs.extend(self.write_scalar_to_addr(space, addr, src)?);
            } else {
                docs.extend(self.copy_into_addr(src.class.clone(), space, addr, src)?);
            }
        }
        Ok(docs)
    }
}
