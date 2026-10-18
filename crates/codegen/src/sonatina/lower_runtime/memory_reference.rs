use hir::analysis::{semantic::FieldIndex, ty::AddressSpace};
use mir::{Layout, LayoutId, ResolvedPlaceElem, RuntimeClass, RuntimeMemoryLayout, VariantId};
use sonatina_codegen::transform::aggregate::EnumLoweredLayout;
use sonatina_ir::{
    BlockId, Type, ValueId,
    inst::{
        arith::{Add, Mul, Sub},
        cast::Zext,
        cmp::Eq,
        control_flow::{BrTable, Jump, Phi, PhiArgs, Unreachable},
        data::{ConstLoad, Mstore, ObjLoad, ObjMaterializeHeap},
    },
    types::CompoundType,
};

use super::{
    CopySource, EnumLoad, FunctionLowerer, LowerError, Lowered, LoweringInstSet, scalar_ty,
};

// A descriptor's layout tag says how its address reaches the referent, which
// is always in memory (the runtime verifier admits a native reference to
// memory only): as a raw memory address, or as a Sonatina object's address.
const RAW_MEMORY: u64 = 0;
const NATIVE_MEMORY: u64 = 1;
const REFERENT_LAYOUTS: [(u64, ReferentLayout); 2] = [
    (RAW_MEMORY, ReferentLayout::Raw(AddressSpace::Memory)),
    (NATIVE_MEMORY, ReferentLayout::Object),
];

#[derive(Clone, Copy)]
pub(super) enum ReferentLayout {
    Raw(AddressSpace),
    Object,
}

/// Stored references are one-word pointers to immutable address/layout
/// descriptors. Copying a reference never copies its referent.
#[derive(Clone, Copy)]
pub(super) struct MemoryReference {
    pub(super) addr: ValueId,
    pub(super) layout: ValueId,
}

impl<'db, I: LoweringInstSet + 'static> FunctionLowerer<'_, 'db, '_, I> {
    pub(super) fn store_memory_reference(
        &mut self,
        reference: MemoryReference,
    ) -> Result<ValueId, LowerError> {
        let size = self.index_value(64);
        let descriptor = self.allocate_bytes(size, Type::I256)?;
        let descriptor = self.coerce_value_to_ty(descriptor, Type::I256)?;
        let addr = self.coerce_value_to_ty(reference.addr, Type::I256)?;
        self.fb.insert_inst_no_result(Mstore::new(
            self.module.inst_set(),
            descriptor,
            addr,
            Type::I256,
        ));
        let layout_addr = self.offset_address_unscaled(descriptor, 32)?;
        self.fb.insert_inst_no_result(Mstore::new(
            self.module.inst_set(),
            layout_addr,
            reference.layout,
            Type::I256,
        ));
        Ok(descriptor)
    }

    pub(super) fn load_memory_reference(
        &mut self,
        descriptor: ValueId,
    ) -> Result<MemoryReference, LowerError> {
        let addr = self.load_word(descriptor, AddressSpace::Memory)?;
        let layout_addr = self.offset_address_unscaled(descriptor, 32)?;
        let layout = self.load_word(layout_addr, AddressSpace::Memory)?;
        Ok(MemoryReference { addr, layout })
    }

    pub(super) fn export_object_reference(
        &mut self,
        object: ValueId,
    ) -> Result<ValueId, LowerError> {
        let Some(CompoundType::ObjRef(pointee)) = self
            .fb
            .type_of(object)
            .resolve_compound(&self.fb.module_builder.ctx)
        else {
            return Err(LowerError::Internal(
                "memory-reference export requires object storage".into(),
            ));
        };
        let ptr_ty = self.fb.ptr_type(pointee);
        // This exports the original allocation, including all aliases. It must
        // not be replaced with an allocation followed by a referent copy.
        let addr = self.fb.insert_inst(
            ObjMaterializeHeap::new(self.module.inst_set(), object),
            ptr_ty,
        );
        self.store_native_reference(addr)
    }

    pub(super) fn store_native_reference(&mut self, addr: ValueId) -> Result<ValueId, LowerError> {
        let layout = self.index_value(NATIVE_MEMORY);
        self.store_memory_reference(MemoryReference { addr, layout })
    }

    pub(super) fn store_raw_memory_reference(
        &mut self,
        addr: ValueId,
    ) -> Result<ValueId, LowerError> {
        let layout = self.index_value(RAW_MEMORY);
        self.store_memory_reference(MemoryReference { addr, layout })
    }

    pub(super) fn load_referent(
        &mut self,
        reference: MemoryReference,
        class: &RuntimeClass<'db>,
    ) -> Result<ValueId, LowerError> {
        let ty = match class {
            RuntimeClass::Scalar(scalar) => scalar_ty(scalar),
            _ => self.module.ty_for_class(class)?,
        };
        self.load_reference_value(reference, ty, |this, memory| {
            this.load_memory_value(reference.addr, memory, class)
        })
    }

    pub(super) fn load_referent_enum_tag(
        &mut self,
        reference: MemoryReference,
        layout: LayoutId<'db>,
    ) -> Result<ValueId, LowerError> {
        let Layout::Enum(data) = layout.data(self.module.db) else {
            return Err(LowerError::Internal("enum tag requires enum layout".into()));
        };
        let ty = self.module.enum_tag_ty(layout)?;
        self.load_reference_value(reference, ty, |this, memory| {
            // Validate the discriminant without reading or reconstructing payloads.
            this.load_enum_from_ptr(reference.addr, memory, layout, &data, EnumLoad::Tag)
        })
    }

    fn load_reference_value(
        &mut self,
        reference: MemoryReference,
        ty: Type,
        mut load: impl FnMut(&mut Self, ReferentLayout) -> Result<ValueId, LowerError>,
    ) -> Result<ValueId, LowerError> {
        let done = self.fb.append_block();
        let blocks = self.branch_on_referent_layout(reference);
        let mut values = PhiArgs::with_capacity(blocks.len());
        for (block, memory) in blocks {
            self.fb.switch_to_block(block);
            let value = load(self, memory)?;
            let pred = self
                .fb
                .current_block()
                .expect("referent load has a current block");
            values.push((value, pred));
            self.fb
                .insert_inst_no_result(Jump::new(self.module.inst_set(), done));
        }
        self.fb.switch_to_block(done);
        Ok(self
            .fb
            .insert_inst(Phi::new(self.module.inst_set(), values), ty))
    }

    pub(super) fn store_referent(
        &mut self,
        reference: MemoryReference,
        class: &RuntimeClass<'db>,
        value: ValueId,
    ) -> Result<(), LowerError> {
        let ty = match class {
            RuntimeClass::Scalar(scalar) => scalar_ty(scalar),
            _ => self.module.ty_for_class(class)?,
        };
        let value = self.coerce_value_to_ty(value, ty)?;
        let done = self.fb.append_block();
        for (block, memory) in self.branch_on_referent_layout(reference) {
            self.fb.switch_to_block(block);
            match memory {
                ReferentLayout::Raw(space) => {
                    self.copy_to_ptr(reference.addr, space, class, value)?
                }
                ReferentLayout::Object => {
                    self.copy_memory_value(reference.addr, memory, class, value)?
                }
            }
            self.fb
                .insert_inst_no_result(Jump::new(self.module.inst_set(), done));
        }
        self.fb.switch_to_block(done);
        Ok(())
    }

    /// Branches on `reference`'s layout tag to one block per referent layout,
    /// which the caller fills; any other tag is unreachable.
    fn branch_on_referent_layout(
        &mut self,
        reference: MemoryReference,
    ) -> [(BlockId, ReferentLayout); 2] {
        let invalid = self.fb.append_block();
        let blocks = REFERENT_LAYOUTS.map(|(tag, memory)| (tag, self.fb.append_block(), memory));
        let cases = blocks
            .iter()
            .map(|(tag, block, _)| (self.index_value(*tag), *block))
            .collect::<Vec<_>>();
        self.fb.insert_inst_no_result(BrTable::new(
            self.module.inst_set(),
            reference.layout,
            Some(invalid),
            cases,
        ));
        self.fb.switch_to_block(invalid);
        self.fb
            .insert_inst_no_result(Unreachable::new(self.module.inst_set()));
        blocks.map(|(_, block, memory)| (block, memory))
    }

    pub(super) fn copy_source_value(
        &mut self,
        source: CopySource<'db>,
        target: &RuntimeClass<'db>,
    ) -> Result<ValueId, LowerError> {
        let class = self.copy_source_class(&source).clone();
        let ty = self.module.ty_for_class(&class)?;
        let value = match source {
            CopySource::Value { value, .. } => value,
            CopySource::Object { value, .. } => self
                .fb
                .insert_inst(ObjLoad::new(self.module.inst_set(), value), ty),
            CopySource::Const { value, .. } => self
                .fb
                .insert_inst(ConstLoad::new(self.module.inst_set(), value), ty),
            CopySource::Ptr {
                addr, space, lane, ..
            } => self.load_from_ptr_lane(addr, space, lane, &class)?,
        };
        self.retype_value_for_class(value, &class, target)
    }

    pub(super) fn native_class_layout(
        &mut self,
        class: &RuntimeClass<'db>,
    ) -> Result<EnumLoweredLayout, LowerError> {
        let ty = self.module.ty_for_class(class)?;
        // Stored references address the original Sonatina allocation. Its enum
        // products and target padding must match the eventual legalization,
        // without rewriting types while this module is still being emitted.
        Ok(EnumLoweredLayout::new(&self.fb.module_builder.ctx, ty))
    }

    pub(super) fn referent_field_address(
        &mut self,
        addr: ValueId,
        memory: ReferentLayout,
        class: &RuntimeClass<'db>,
        index: usize,
    ) -> Result<ValueId, LowerError> {
        let offset = match memory {
            ReferentLayout::Object => self
                .native_class_layout(class)?
                .field_offset(index)
                .ok_or_else(|| {
                    LowerError::Unsupported("unrepresentable native field offset".into())
                })? as u64,
            ReferentLayout::Raw(space) => {
                let layout = class
                    .aggregate_layout()
                    .ok_or_else(|| LowerError::Internal("field requires aggregate".into()))?;
                let raw = RuntimeMemoryLayout::for_space(self.module.db, space);
                match layout.data(self.module.db) {
                    Layout::Struct(data) => raw.struct_field_offset(&data, index)?,
                    Layout::Array(data) => raw.array_element_offset(&data, index as u64)?,
                    Layout::Enum(_) => {
                        return Err(LowerError::Internal("enum field requires variant".into()));
                    }
                }
            }
        };
        self.offset_address_unscaled(addr, offset)
    }

    pub(super) fn referent_variant_address(
        &mut self,
        addr: ValueId,
        memory: ReferentLayout,
        variant: VariantId<'db>,
        field: usize,
    ) -> Result<ValueId, LowerError> {
        let offset = match memory {
            ReferentLayout::Object => self
                .native_class_layout(&RuntimeClass::AggregateValue {
                    layout: variant.enum_layout,
                })?
                .variant_field_offset(variant.index as usize, field)
                .ok_or_else(|| {
                    LowerError::Unsupported("unrepresentable native variant offset".into())
                })? as u64,
            ReferentLayout::Raw(space) => RuntimeMemoryLayout::for_space(self.module.db, space)
                .variant_field_offset(variant, FieldIndex(field as u16))?,
        };
        self.offset_address_unscaled(addr, offset)
    }

    /// `raw` for a raw memory referent, `native` for an object.
    fn memory_layout_offset(&mut self, layout: ValueId, raw: u64, native: u64) -> ValueId {
        let native_tag = self.index_value(NATIVE_MEMORY);
        let native_flag = self.fb.insert_inst(
            Eq::new(self.module.inst_set(), layout, native_tag),
            Type::I1,
        );
        let native_flag = self.fb.insert_inst(
            Zext::new(self.module.inst_set(), native_flag, Type::I256),
            Type::I256,
        );
        let raw = self.index_value(raw);
        let native = self.index_value(native);
        let difference = self
            .fb
            .insert_inst(Sub::new(self.module.inst_set(), native, raw), Type::I256);
        let difference = self.fb.insert_inst(
            Mul::new(self.module.inst_set(), native_flag, difference),
            Type::I256,
        );
        self.fb.insert_inst(
            Add::new(self.module.inst_set(), raw, difference),
            Type::I256,
        )
    }

    pub(super) fn project_memory_reference(
        &mut self,
        reference: MemoryReference,
        base: &RuntimeClass<'db>,
        elem: &ResolvedPlaceElem<'db>,
    ) -> Result<Lowered<(MemoryReference, RuntimeClass<'db>)>, LowerError> {
        let raw = RuntimeMemoryLayout::raw(self.module.db);
        let (offset, class) = match elem {
            ResolvedPlaceElem::Field { field, class } => {
                let native = self
                    .native_class_layout(base)?
                    .field_offset(field.0 as usize)
                    .ok_or_else(|| {
                        LowerError::Unsupported("unrepresentable native field offset".into())
                    })?;
                (
                    self.memory_layout_offset(
                        reference.layout,
                        raw.field_offset(base, *field)?,
                        native as u64,
                    ),
                    class,
                )
            }
            ResolvedPlaceElem::Index { index, class } => {
                let index = self.checked_index_value(base, index)?;
                let native = self.native_class_layout(class)?.size().map_err(|err| {
                    LowerError::Unsupported(format!(
                        "unrepresentable native memory layout: {err:?}"
                    ))
                })?;
                let stride = self.memory_layout_offset(
                    reference.layout,
                    raw.index_stride(base)?,
                    native as u64,
                );
                (
                    self.fb
                        .insert_inst(Mul::new(self.module.inst_set(), index, stride), Type::I256),
                    class,
                )
            }
            ResolvedPlaceElem::VariantField {
                variant,
                field,
                class,
            } => {
                let native = self
                    .native_class_layout(base)?
                    .variant_field_offset(variant.index as usize, field.0 as usize)
                    .ok_or_else(|| {
                        LowerError::Unsupported("unrepresentable native variant offset".into())
                    })?;
                (
                    self.memory_layout_offset(
                        reference.layout,
                        raw.variant_field_offset(*variant, *field)?,
                        native as u64,
                    ),
                    class,
                )
            }
            ResolvedPlaceElem::Deref { .. } => {
                return Err(LowerError::Internal(
                    "dereference requires loading the stored carrier".into(),
                ));
            }
        };
        let addr = self.fb.insert_inst(
            Add::new(self.module.inst_set(), reference.addr, offset),
            Type::I256,
        );
        Ok(Lowered::Value((
            MemoryReference {
                addr,
                layout: reference.layout,
            },
            class.clone(),
        )))
    }
}
