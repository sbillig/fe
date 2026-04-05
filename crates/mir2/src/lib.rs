pub mod db;
pub mod instance;
pub mod runtime;
pub mod verify;

pub use db::MirDb;
pub use instance::{RuntimeInstance, RuntimeInstanceKey, get_or_build_runtime_instance};
pub use runtime::{
    AddressSpaceKind, ArrayLayout, ConstNode, ConstRegion, ConstRegionId, ConstScalar, EnumLayout,
    EnumVariantLayout, HandleKind, HandleView, IntrinsicArithBinOp, Layout, LayoutId, LowerError,
    PlaceElem, PlaceRoot, RBlock, RBlockId, RExpr, RLocal, RLocalId, RStmt, RTerminator, RValueId,
    ResolvedCodeRegion, ResolvedPlaceElem, ResolvedPlaceRootKind, ResolvedRuntimePlace,
    RuntimeBody, RuntimeBuiltin, RuntimeCallEdge, RuntimeCarrier, RuntimeClass, RuntimeCodeRegion,
    RuntimeCodeRegionKey, RuntimeEmbed, RuntimeFunction, RuntimeFunctionOwner, RuntimeInlineHint,
    RuntimeLinkage, RuntimeLocalRoot, RuntimeObject, RuntimePackage, RuntimeParam, RuntimePlace,
    RuntimeProgramView, RuntimeReturnPlan, RuntimeSection, RuntimeSectionName, RuntimeSectionRef,
    RuntimeSignature, RuntimeSyntheticSpec, SaturatingBinOp, ScalarClass, ScalarRepr, ScalarRole,
    StructLayout, VariantId, array_elem_size_bytes, build_runtime_package,
    build_test_runtime_package, enum_tag_size_bytes, enum_variant_field_offset_bytes,
    layout_size_bytes, serialize_const_region_bytes, struct_field_offset_bytes,
};
pub use verify::{
    VerifyError, resolve_runtime_place, verify_const_region, verify_runtime_body,
    verify_runtime_package,
};
