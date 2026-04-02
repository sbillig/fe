mod consts;
mod layout;
mod package;
mod place;
mod runtime;

use crate::{
    instance::RuntimeInstance,
    runtime::{ConstRegionId, LayoutId, RuntimeClass},
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VerifyError<'db> {
    MissingRuntimeLocal(crate::runtime::RLocalId),
    MissingRuntimeBlock(crate::runtime::RBlockId),
    ErasedRuntimeValue(crate::runtime::RValueId),
    SlotCarrierMismatch(crate::runtime::RLocalId),
    InvalidLayoutHandleView(LayoutId<'db>),
    InvalidConstRegion(ConstRegionId<'db>),
    InvalidVariant(LayoutId<'db>, u16),
    InvalidPlace(RuntimeClass<'db>),
    InvalidVariantPlace(RuntimeClass<'db>),
    InvalidEnumTag(LayoutId<'db>),
    InvalidReturnClass,
    InvalidExprClass(crate::runtime::RLocalId),
    InvalidStoreClass,
    InvalidCopyClass,
    CallArgCountMismatch(RuntimeInstance<'db>),
    CallArgClassMismatch(RuntimeInstance<'db>, usize),
    InvalidCodeRegion(crate::runtime::RuntimeCodeRegion<'db>),
    InvalidPackageFunction(crate::instance::RuntimeInstance<'db>),
    InvalidPackageObject(crate::runtime::RuntimeObject<'db>),
    InvalidPackageSection(
        crate::runtime::RuntimeObject<'db>,
        crate::runtime::RuntimeSectionName,
    ),
    DuplicateRuntimeSymbol(String),
}

pub use consts::verify_const_region;
pub use package::verify_runtime_package;
pub use place::resolve_runtime_place;
pub use runtime::verify_runtime_body;
