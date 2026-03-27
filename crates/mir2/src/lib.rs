pub mod db;
pub mod instance;
pub mod runtime;
pub mod verify;

pub use db::MirDb;
pub use instance::{RuntimeInstance, RuntimeInstanceKey, get_or_build_runtime_instance};
pub use runtime::{
    AddressSpaceKind, ArrayLayout, ConstNode, ConstRegion, ConstRegionId, ConstScalar, EnumLayout,
    EnumVariantLayout, HandleKind, HandleView, Layout, LayoutId, LocalSlotKind, PlaceElem,
    PlaceRoot, RBlock, RBlockId, RExpr, RLocal, RLocalId, RStmt, RTerminator, RValueId,
    RuntimeBody, RuntimeCallEdge, RuntimeCarrier, RuntimeClass, RuntimeParam, RuntimePlace,
    RuntimeProgramView, RuntimeSignature, ScalarClass, ScalarRepr, ScalarRole, StructLayout,
    VariantId,
};
pub use verify::{VerifyError, verify_const_region, verify_runtime_body};
