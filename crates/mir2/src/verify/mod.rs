mod consts;
mod layout;
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
}

pub use consts::verify_const_region;
pub use runtime::verify_runtime_body;
