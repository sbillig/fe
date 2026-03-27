use cranelift_entity::{EntityRef, entity_impl};
use hir::analysis::ty::ty_def::TyId;
use hir::hir_def::{BinOp, UnOp};
use salsa::Update;

use hir::analysis::semantic::{FieldIndex, SemanticCalleeRef};

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
    pub slot: LocalSlotKind<'db>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Update)]
pub enum LocalSlotKind<'db> {
    None,
    Slot(RuntimeClass<'db>),
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
    pub semantic_callee: SemanticCalleeRef<'db>,
    pub runtime_arg_classes: Vec<RuntimeClass<'db>>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Update)]
pub enum PlaceRoot {
    Slot(RLocalId),
    Handle(RValueId),
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Update)]
pub struct RuntimePlace {
    pub root: PlaceRoot,
    pub path: Box<[PlaceElem]>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Update)]
pub enum PlaceElem {
    Field(FieldIndex),
    Index(RValueId),
    VariantField(FieldIndex),
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Update)]
pub enum RExpr<'db> {
    Use(RValueId),
    ConstScalar(ConstScalar),
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
    ProviderToRaw {
        value: RValueId,
    },
    AddrOf {
        place: RuntimePlace,
    },
    Load {
        place: RuntimePlace,
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
        dst: RuntimePlace,
        src: RValueId,
    },
    CopyInto {
        dst: RuntimePlace,
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
    MatchEnumTag {
        tag: RValueId,
        enum_layout: LayoutId<'db>,
        cases: Box<[(VariantId<'db>, RBlockId)]>,
        default: Option<RBlockId>,
    },
    Return(Option<RValueId>),
}

pub trait RuntimeProgramView<'db> {
    fn body(&self, id: RuntimeInstance<'db>) -> RuntimeBody<'db>;
    fn layout(&self, id: LayoutId<'db>) -> Layout<'db>;
    fn const_region(&self, id: ConstRegionId<'db>) -> ConstRegion<'db>;
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
}
