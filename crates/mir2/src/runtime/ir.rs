use cranelift_entity::{EntityRef, entity_impl};
use hir::analysis::{
    semantic::{FieldIndex, SemanticInstance},
    ty::ty_def::TyId,
};
use hir::hir_def::{BinOp, Contract, Func, TopLevelMod, UnOp};
use hir::projection::IndexSource;
use hir::semantic::ProviderBinding;
use salsa::Update;

use crate::{
    db::MirDb,
    instance::{RuntimeInstance, RuntimeInstanceKey},
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Update)]
pub enum AddressSpaceKind {
    Memory,
    Storage,
    Transient,
    Calldata,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Update)]
pub enum RuntimeClass<'db> {
    Scalar(ScalarClass<'db>),
    AggregateValue {
        layout: LayoutId<'db>,
    },
    Handle {
        layout: LayoutId<'db>,
        kind: HandleKind<'db>,
        view: HandleView<'db>,
    },
    RawAddr {
        space: AddressSpaceKind,
        target: Option<LayoutId<'db>>,
    },
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Update)]
pub enum RuntimeCarrier<'db> {
    Erased,
    Value(RuntimeClass<'db>),
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Update)]
pub struct ScalarClass<'db> {
    pub repr: ScalarRepr,
    pub role: ScalarRole<'db>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Update)]
pub enum ScalarRepr {
    Bool,
    Int { bits: u16, signed: bool },
    FixedBytes { len: u16 },
    Address { bits: u16 },
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Update)]
pub enum ScalarRole<'db> {
    Plain,
    EnumTag { enum_layout: LayoutId<'db> },
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Update)]
pub enum HandleKind<'db> {
    ConstValue,
    ObjectValue,
    Provider {
        provider_ty: TyId<'db>,
        space: AddressSpaceKind,
    },
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Update)]
pub enum HandleView<'db> {
    Whole,
    EnumVariant(VariantId<'db>),
}

#[salsa::interned]
#[derive(Debug)]
pub struct LayoutId<'db> {
    pub key: LayoutKey<'db>,
}

impl<'db> LayoutId<'db> {
    pub fn data(self, db: &'db dyn MirDb) -> Layout<'db> {
        match self.key(db) {
            LayoutKey::Struct(layout) => Layout::Struct(layout.clone()),
            LayoutKey::Array(layout) => Layout::Array(layout.clone()),
            LayoutKey::Enum(layout) => Layout::Enum(EnumLayout {
                source_ty: layout.source_ty,
                tag: ScalarClass {
                    repr: enum_tag_repr(layout.variants.len()),
                    role: ScalarRole::EnumTag { enum_layout: self },
                },
                variants: layout.variants.clone(),
            }),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Update)]
pub enum LayoutKey<'db> {
    Struct(StructLayout<'db>),
    Array(ArrayLayout<'db>),
    Enum(EnumLayoutKey<'db>),
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Update)]
pub enum Layout<'db> {
    Struct(StructLayout<'db>),
    Array(ArrayLayout<'db>),
    Enum(EnumLayout<'db>),
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Update)]
pub struct StructLayout<'db> {
    pub source_ty: TyId<'db>,
    pub fields: Box<[RuntimeClass<'db>]>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Update)]
pub struct ArrayLayout<'db> {
    pub source_ty: TyId<'db>,
    pub elem: RuntimeClass<'db>,
    pub len: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Update)]
pub struct EnumLayout<'db> {
    pub source_ty: TyId<'db>,
    pub tag: ScalarClass<'db>,
    pub variants: Box<[EnumVariantLayout<'db>]>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Update)]
pub struct EnumLayoutKey<'db> {
    pub source_ty: TyId<'db>,
    pub variants: Box<[EnumVariantLayout<'db>]>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Update)]
pub struct EnumVariantLayout<'db> {
    pub name: String,
    pub fields: Box<[RuntimeClass<'db>]>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Update)]
pub struct VariantId<'db> {
    pub enum_layout: LayoutId<'db>,
    pub index: u16,
}

impl<'db> VariantId<'db> {
    pub fn layout(self, db: &'db dyn MirDb) -> Option<EnumLayout<'db>> {
        match self.enum_layout.data(db) {
            Layout::Enum(layout) => Some(layout),
            Layout::Struct(_) | Layout::Array(_) => None,
        }
    }
}

fn enum_tag_repr(variant_count: usize) -> ScalarRepr {
    let bits = if variant_count <= u8::MAX as usize + 1 {
        8
    } else if variant_count <= u16::MAX as usize + 1 {
        16
    } else {
        32
    };
    ScalarRepr::Int {
        bits,
        signed: false,
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Update)]
pub struct ConstRegion<'db> {
    pub layout: LayoutId<'db>,
    pub value: ConstNode<'db>,
}

#[salsa::interned]
#[derive(Debug)]
pub struct ConstRegionId<'db> {
    pub layout: LayoutId<'db>,
    pub value: ConstNode<'db>,
}

impl<'db> ConstRegionId<'db> {
    pub fn data(self, db: &'db dyn MirDb) -> ConstRegion<'db> {
        ConstRegion {
            layout: self.layout(db),
            value: self.value(db).clone(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Update)]
pub enum ConstNode<'db> {
    Scalar(ConstScalar),
    Aggregate {
        layout: LayoutId<'db>,
        fields: Box<[ConstNode<'db>]>,
    },
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Update)]
pub enum ConstScalar {
    Bool(bool),
    Int {
        bits: u16,
        signed: bool,
        words: Vec<u8>,
    },
    FixedBytes(Vec<u8>),
    Address {
        bits: u16,
        bytes: Vec<u8>,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Update)]
pub struct RLocalId(u32);
entity_impl!(RLocalId);

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Update)]
pub struct RBlockId(u32);
entity_impl!(RBlockId);

pub type RValueId = RLocalId;

#[derive(Clone, Debug, PartialEq, Eq, Hash, Update)]
pub struct RuntimeBody<'db> {
    pub owner: RuntimeInstance<'db>,
    pub key: RuntimeInstanceKey<'db>,
    pub signature: RuntimeSignature<'db>,
    pub semantic_locals: Vec<RuntimeLocalLowering<'db>>,
    pub provider_bindings: Vec<RuntimeProviderBinding<'db>>,
    pub locals: Vec<RLocal<'db>>,
    pub blocks: Vec<RBlock<'db>>,
}

impl<'db> RuntimeBody<'db> {
    pub fn local(&self, id: RLocalId) -> Option<&RLocal<'db>> {
        self.locals.get(id.index())
    }

    pub fn block(&self, id: RBlockId) -> Option<&RBlock<'db>> {
        self.blocks.get(id.index())
    }

    pub fn value_class(&self, value: RValueId) -> Option<&RuntimeClass<'db>> {
        match &self.local(value)?.carrier {
            RuntimeCarrier::Erased => None,
            RuntimeCarrier::Value(class) => Some(class),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Update)]
pub struct RLocal<'db> {
    pub semantic_ty: TyId<'db>,
    pub carrier: RuntimeCarrier<'db>,
    pub root: RuntimeLocalRoot<'db>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Update)]
pub enum RuntimeLocalRoot<'db> {
    None,
    Slot(RuntimeClass<'db>),
    Handle(RuntimeClass<'db>),
    Ptr {
        space: AddressSpaceKind,
        class: RuntimeClass<'db>,
    },
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Update)]
pub enum RuntimeLocalLowering<'db> {
    Erased,
    DirectValue,
    PlaceCarrier {
        place_class: RuntimeClass<'db>,
    },
    PlaceBoundValue {
        provider: RuntimeProviderBindingId,
        place_class: RuntimeClass<'db>,
    },
    DirectCarrier {
        provider: Option<RuntimeProviderBindingId>,
        place_class: RuntimeClass<'db>,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Update)]
pub struct RuntimeProviderBindingId(u32);
entity_impl!(RuntimeProviderBindingId);

#[derive(Clone, Debug, PartialEq, Eq, Hash, Update)]
pub struct RuntimeProviderBinding<'db> {
    pub provider: ProviderBinding<'db>,
    pub value: RLocalId,
    pub provider_class: RuntimeClass<'db>,
    pub place_class: RuntimeClass<'db>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Update)]
pub struct RuntimeSignature<'db> {
    pub params: Vec<RuntimeParam<'db>>,
    pub ret: Option<RuntimeClass<'db>>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Update)]
pub struct RuntimeParam<'db> {
    pub local: RLocalId,
    pub class: RuntimeClass<'db>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Update)]
pub struct RBlock<'db> {
    pub stmts: Vec<RStmt<'db>>,
    pub terminator: RTerminator<'db>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Update)]
pub struct RuntimeCallEdge<'db> {
    pub callee: RuntimeInstance<'db>,
}

#[salsa::interned]
#[derive(Debug)]
pub struct RuntimeCodeRegion<'db> {
    pub key: RuntimeCodeRegionKey<'db>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Update)]
pub enum RuntimeCodeRegionKey<'db> {
    ContractInit {
        contract: Contract<'db>,
    },
    ContractRuntime {
        contract: Contract<'db>,
    },
    ManualContractRoot {
        func: Func<'db>,
    },
    FunctionRoot {
        symbol: String,
        callee: RuntimeInstance<'db>,
    },
}

#[salsa::interned]
#[derive(Debug)]
pub struct ResolvedCodeRegion<'db> {
    pub region: RuntimeCodeRegion<'db>,
    pub symbol: String,
    pub source: RuntimeSectionRef<'db>,
    pub root: RuntimeFunction<'db>,
}

#[salsa::interned]
#[derive(Debug)]
pub struct RuntimePackage<'db> {
    pub top_mod: TopLevelMod<'db>,
    pub functions: Vec<RuntimeFunction<'db>>,
    pub plan: RuntimePackagePlan<'db>,
}

impl<'db> RuntimePackage<'db> {
    pub fn objects(self, db: &'db dyn MirDb) -> Vec<RuntimeObject<'db>> {
        self.plan(db).objects(db)
    }

    pub fn const_regions(self, db: &'db dyn MirDb) -> Vec<ConstRegionId<'db>> {
        self.plan(db).const_regions(db)
    }

    pub fn code_regions(self, db: &'db dyn MirDb) -> Vec<ResolvedCodeRegion<'db>> {
        self.plan(db).code_regions(db)
    }

    pub fn root_objects(self, db: &'db dyn MirDb) -> Vec<RuntimeObject<'db>> {
        self.plan(db).root_objects(db)
    }

    pub fn primary_object(self, db: &'db dyn MirDb) -> Option<RuntimeObject<'db>> {
        self.plan(db).primary_object(db)
    }
}

#[salsa::interned]
#[derive(Debug)]
pub struct RuntimePackagePlan<'db> {
    pub objects: Vec<RuntimeObject<'db>>,
    pub const_regions: Vec<ConstRegionId<'db>>,
    pub code_regions: Vec<ResolvedCodeRegion<'db>>,
    pub root_objects: Vec<RuntimeObject<'db>>,
    pub primary_object: Option<RuntimeObject<'db>>,
}

#[salsa::interned]
#[derive(Debug)]
pub struct RuntimeFunction<'db> {
    pub instance: RuntimeInstance<'db>,
    pub symbol: String,
    pub linkage: RuntimeLinkage,
    pub inline_hint: RuntimeInlineHint,
    pub owner: RuntimeFunctionOwner<'db>,
    pub referenced_const_regions: Vec<ConstRegionId<'db>>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Update)]
pub enum RuntimeLinkage {
    Private,
    Internal,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Update)]
pub enum RuntimeInlineHint {
    Auto,
    Hint,
    Always,
    Never,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Update)]
pub enum RuntimeFunctionOwner<'db> {
    Semantic(SemanticInstance<'db>),
    Synthetic(RuntimeSyntheticSpec<'db>),
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Update)]
pub enum RuntimeSyntheticSpec<'db> {
    MainRoot {
        callee: RuntimeInstance<'db>,
    },
    TestRoot {
        name: String,
        callee: RuntimeInstance<'db>,
    },
    ContractInitAbi {
        plan: ContractInitAbiPlan<'db>,
    },
    ContractRecvAbi {
        plan: ContractRecvAbiPlan<'db>,
    },
    ContractInitRoot {
        contract: Contract<'db>,
        init_abi: Option<RuntimeInstance<'db>>,
        runtime_region: RuntimeCodeRegion<'db>,
    },
    ContractRuntimeRoot {
        contract: Contract<'db>,
        dispatch: Box<[DispatchArm<'db>]>,
        default: DispatchDefault,
    },
    CodeRegionRoot {
        symbol: String,
        callee: RuntimeInstance<'db>,
    },
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Update)]
pub struct ContractInitAbiPlan<'db> {
    pub contract: Contract<'db>,
    pub payable: bool,
    pub user_init: Option<RuntimeInstance<'db>>,
    pub owner_effect_args: Box<[ContractEffectArgPlan<'db>]>,
    pub init_args: InitArgsPlan<'db>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Update)]
pub struct ContractRecvAbiPlan<'db> {
    pub contract: Contract<'db>,
    pub selector: u32,
    pub payable: bool,
    pub user_recv: RuntimeInstance<'db>,
    pub owner_effect_args: Box<[ContractEffectArgPlan<'db>]>,
    pub input: RuntimeInputPlan<'db>,
    pub ret: RuntimeReturnPlan<'db>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Update)]
pub enum ContractEffectArgPlan<'db> {
    ContractField(ContractFieldBinding<'db>),
    Placeholder {
        declared_ty: TyId<'db>,
        class: RuntimeClass<'db>,
    },
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Update)]
pub struct ContractFieldBinding<'db> {
    pub slot: u128,
    pub declared_ty: TyId<'db>,
    pub class: RuntimeClass<'db>,
    pub kind: HandleKind<'db>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Update)]
pub enum InitArgsPlan<'db> {
    None,
    DecodeInitTail {
        tuple_ty: TyId<'db>,
        decode_fn: RuntimeInstance<'db>,
    },
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Update)]
pub enum RuntimeInputPlan<'db> {
    None,
    DecodeCalldataPayload {
        msg_ty: TyId<'db>,
        decode_fn: RuntimeInstance<'db>,
    },
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Update)]
pub enum RuntimeReturnPlan<'db> {
    Unit,
    Value {
        ty: TyId<'db>,
        encode_fn: RuntimeInstance<'db>,
    },
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Update)]
pub struct DispatchArm<'db> {
    pub selector: u32,
    pub wrapper: RuntimeInstance<'db>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Update)]
pub enum DispatchDefault {
    RevertEmpty,
}

#[salsa::interned]
#[derive(Debug)]
pub struct RuntimeObject<'db> {
    pub name: String,
    pub sections: Vec<RuntimeSection<'db>>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Update)]
pub struct RuntimeSection<'db> {
    pub name: RuntimeSectionName,
    pub entry: RuntimeFunction<'db>,
    pub embeds: Vec<RuntimeEmbed<'db>>,
    pub const_regions: Vec<ConstRegionId<'db>>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Update)]
pub enum RuntimeSectionName {
    Init,
    Runtime,
    Main,
    Test(String),
    CodeRegion(String),
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Update)]
pub struct RuntimeEmbed<'db> {
    pub source: RuntimeSectionRef<'db>,
    pub as_symbol: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Update)]
pub enum RuntimeSectionRef<'db> {
    Local {
        object: RuntimeObject<'db>,
        section: RuntimeSectionName,
    },
    External {
        object: RuntimeObject<'db>,
        section: RuntimeSectionName,
    },
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Update)]
pub enum PlaceRoot<'db> {
    Slot(RLocalId),
    Handle(RValueId),
    Provider(RuntimeProviderBindingId),
    Ptr {
        addr: RValueId,
        space: AddressSpaceKind,
        class: RuntimeClass<'db>,
    },
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Update)]
pub struct RuntimePlace<'db> {
    pub root: PlaceRoot<'db>,
    pub path: Box<[PlaceElem]>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Update)]
pub enum PlaceElem {
    Field(FieldIndex),
    Index(IndexSource<RValueId>),
    VariantField(FieldIndex),
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Update)]
pub struct ResolvedRuntimePlace<'db> {
    pub root_kind: ResolvedPlaceRootKind<'db>,
    pub result_class: RuntimeClass<'db>,
    pub path: Box<[ResolvedPlaceElem<'db>]>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Update)]
pub enum ResolvedPlaceRootKind<'db> {
    Slot {
        local: RLocalId,
        class: RuntimeClass<'db>,
    },
    Handle {
        value: RValueId,
        class: RuntimeClass<'db>,
    },
    Provider {
        binding: RuntimeProviderBindingId,
        value: RLocalId,
        provider_class: RuntimeClass<'db>,
        class: RuntimeClass<'db>,
    },
    Ptr {
        addr: RValueId,
        space: AddressSpaceKind,
        class: RuntimeClass<'db>,
    },
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Update)]
pub enum ResolvedPlaceElem<'db> {
    Field {
        field: FieldIndex,
        class: RuntimeClass<'db>,
    },
    Index {
        index: IndexSource<RValueId>,
        class: RuntimeClass<'db>,
    },
    VariantField {
        variant: VariantId<'db>,
        field: FieldIndex,
        class: RuntimeClass<'db>,
    },
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Update)]
pub enum RuntimeBuiltin<'db> {
    Mload {
        addr: RValueId,
    },
    Mstore {
        addr: RValueId,
        value: RValueId,
    },
    Mstore8 {
        addr: RValueId,
        value: RValueId,
    },
    Msize,
    Sload {
        slot: RValueId,
    },
    Sstore {
        slot: RValueId,
        value: RValueId,
    },
    CallValue,
    ReturnDataSize,
    ReturnDataCopy {
        dst: RValueId,
        offset: RValueId,
        len: RValueId,
    },
    CallDataSize,
    CallDataLoad {
        offset: RValueId,
    },
    CallDataCopy {
        dst: RValueId,
        offset: RValueId,
        len: RValueId,
    },
    CodeSize,
    CodeCopy {
        dst: RValueId,
        offset: RValueId,
        len: RValueId,
    },
    Keccak256 {
        offset: RValueId,
        len: RValueId,
    },
    AddMod {
        lhs: RValueId,
        rhs: RValueId,
        modulus: RValueId,
    },
    MulMod {
        lhs: RValueId,
        rhs: RValueId,
        modulus: RValueId,
    },
    IntrinsicArith {
        op: IntrinsicArithBinOp,
        lhs: RValueId,
        rhs: RValueId,
        class: ScalarClass<'db>,
    },
    Saturating {
        op: SaturatingBinOp,
        lhs: RValueId,
        rhs: RValueId,
        class: ScalarClass<'db>,
    },
    Address,
    Caller,
    Origin,
    GasPrice,
    CoinBase,
    Timestamp,
    Number,
    PrevRandao,
    GasLimit,
    ChainId,
    BaseFee,
    SelfBalance,
    BlockHash {
        block: RValueId,
    },
    Gas,
    CodeRegionOffset {
        region: RuntimeCodeRegion<'db>,
    },
    CodeRegionLen {
        region: RuntimeCodeRegion<'db>,
    },
    Malloc {
        size: RValueId,
    },
    Call {
        gas: RValueId,
        addr: RValueId,
        value: RValueId,
        args_offset: RValueId,
        args_len: RValueId,
        ret_offset: RValueId,
        ret_len: RValueId,
    },
    StaticCall {
        gas: RValueId,
        addr: RValueId,
        args_offset: RValueId,
        args_len: RValueId,
        ret_offset: RValueId,
        ret_len: RValueId,
    },
    DelegateCall {
        gas: RValueId,
        addr: RValueId,
        args_offset: RValueId,
        args_len: RValueId,
        ret_offset: RValueId,
        ret_len: RValueId,
    },
    Create {
        value: RValueId,
        offset: RValueId,
        len: RValueId,
    },
    Create2 {
        value: RValueId,
        offset: RValueId,
        len: RValueId,
        salt: RValueId,
    },
    Log0 {
        offset: RValueId,
        len: RValueId,
    },
    Log1 {
        offset: RValueId,
        len: RValueId,
        topic0: RValueId,
    },
    Log2 {
        offset: RValueId,
        len: RValueId,
        topic0: RValueId,
        topic1: RValueId,
    },
    Log3 {
        offset: RValueId,
        len: RValueId,
        topic0: RValueId,
        topic1: RValueId,
        topic2: RValueId,
    },
    Log4 {
        offset: RValueId,
        len: RValueId,
        topic0: RValueId,
        topic1: RValueId,
        topic2: RValueId,
        topic3: RValueId,
    },
    CallDataSelector,
    MakeContractFieldHandle {
        slot: u128,
        class: RuntimeClass<'db>,
        kind: HandleKind<'db>,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Update)]
pub enum SaturatingBinOp {
    Add,
    Sub,
    Mul,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Update)]
pub enum IntrinsicArithBinOp {
    Add,
    Sub,
    Mul,
    Div,
    Rem,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Update)]
pub enum RExpr<'db> {
    Use(RValueId),
    ConstScalar(ConstScalar),
    Placeholder {
        class: RuntimeClass<'db>,
    },
    Builtin(RuntimeBuiltin<'db>),
    Unary {
        op: UnOp,
        value: RValueId,
    },
    Binary {
        op: BinOp,
        lhs: RValueId,
        rhs: RValueId,
    },
    Cast {
        value: RValueId,
        to: ScalarClass<'db>,
    },
    ConstHandle {
        region: ConstRegionId<'db>,
        layout: LayoutId<'db>,
    },
    AllocObject {
        layout: LayoutId<'db>,
    },
    MaterializeToObject {
        src: RValueId,
    },
    ProviderFromRaw {
        raw: RValueId,
        provider_ty: TyId<'db>,
        space: AddressSpaceKind,
        layout: LayoutId<'db>,
    },
    WordToRawAddr {
        value: RValueId,
        space: AddressSpaceKind,
        target: Option<LayoutId<'db>>,
    },
    ProviderToRaw {
        value: RValueId,
    },
    AddrOf {
        place: RuntimePlace<'db>,
    },
    Load {
        place: RuntimePlace<'db>,
    },
    Call {
        callee: RuntimeInstance<'db>,
        args: Box<[RValueId]>,
    },
    EnumMake {
        layout: LayoutId<'db>,
        variant: VariantId<'db>,
        fields: Box<[RValueId]>,
    },
    EnumTagOfValue {
        value: RValueId,
    },
    EnumIsVariant {
        value: RValueId,
        variant: VariantId<'db>,
    },
    EnumExtract {
        value: RValueId,
        variant: VariantId<'db>,
        field: FieldIndex,
    },
    EnumGetTag {
        root: RValueId,
    },
    EnumAssertVariantRef {
        root: RValueId,
        variant: VariantId<'db>,
    },
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Update)]
pub enum RStmt<'db> {
    Assign {
        dst: RLocalId,
        expr: RExpr<'db>,
    },
    Store {
        dst: RuntimePlace<'db>,
        src: RValueId,
    },
    CopyInto {
        dst: RuntimePlace<'db>,
        src: RValueId,
    },
    EnumSetTag {
        root: RValueId,
        variant: VariantId<'db>,
    },
    EnumWriteVariant {
        root: RValueId,
        variant: VariantId<'db>,
        fields: Box<[RValueId]>,
    },
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Update)]
pub enum RTerminator<'db> {
    Goto(RBlockId),
    Branch {
        cond: RValueId,
        then_bb: RBlockId,
        else_bb: RBlockId,
    },
    SwitchScalar {
        discr: RValueId,
        cases: Box<[(ConstScalar, RBlockId)]>,
        default: RBlockId,
    },
    MatchEnumTag {
        tag: RValueId,
        enum_layout: LayoutId<'db>,
        cases: Box<[(VariantId<'db>, RBlockId)]>,
        default: Option<RBlockId>,
    },
    TerminalCall {
        callee: RuntimeInstance<'db>,
        args: Box<[RValueId]>,
    },
    ReturnData {
        offset: RValueId,
        len: RValueId,
    },
    Revert {
        offset: RValueId,
        len: RValueId,
    },
    SelfDestruct {
        beneficiary: RValueId,
    },
    Trap,
    Return(Option<RValueId>),
    Stop,
}

pub trait RuntimeProgramView<'db> {
    fn body(&self, id: RuntimeInstance<'db>) -> RuntimeBody<'db>;
    fn layout(&self, id: LayoutId<'db>) -> Layout<'db>;
    fn const_region(&self, id: ConstRegionId<'db>) -> ConstRegion<'db>;
    fn code_region(&self, id: RuntimeCodeRegion<'db>) -> Option<ResolvedCodeRegion<'db>>;
}

impl<'db> RuntimeProgramView<'db> for &'db dyn MirDb {
    fn body(&self, id: RuntimeInstance<'db>) -> RuntimeBody<'db> {
        id.body(*self).clone()
    }

    fn layout(&self, id: LayoutId<'db>) -> Layout<'db> {
        id.data(*self)
    }

    fn const_region(&self, id: ConstRegionId<'db>) -> ConstRegion<'db> {
        id.data(*self)
    }

    fn code_region(&self, _id: RuntimeCodeRegion<'db>) -> Option<ResolvedCodeRegion<'db>> {
        None
    }
}
