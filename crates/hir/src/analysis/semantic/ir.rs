use cranelift_entity::{EntityRef, entity_impl};
use salsa::Update;

use crate::{
    analysis::{
        semantic::instance::{SemanticInstance, SemanticInstanceKey},
        ty::{
            provider::ProviderAddressSpace,
            ty_check::{BodyOwner, EffectPassMode, LocalBinding},
            ty_def::{BorrowKind, TyId},
        },
    },
    hir_def::{BinOp, ExprId, Func, StmtId, UnOp},
    semantic::ProviderBinding,
};

use super::consts::SemConstId;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Update)]
pub enum Mutability {
    Mutable,
    Immutable,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Update)]
pub struct FieldIndex(pub u16);

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Update)]
pub struct VariantIndex(pub u16);

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Update)]
pub struct SLocalId(u32);
entity_impl!(SLocalId);

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Update)]
pub struct SBlockId(u32);
entity_impl!(SBlockId);

pub type SValueId = SLocalId;

#[derive(Clone, Debug, PartialEq, Eq, Hash, Update)]
pub struct SemanticBody<'db> {
    pub owner: SemanticInstance<'db>,
    pub template_owner: BodyOwner<'db>,
    pub locals: Vec<SLocal<'db>>,
    pub blocks: Vec<SBlock<'db>>,
}

impl<'db> SemanticBody<'db> {
    pub fn local(&self, id: SLocalId) -> Option<&SLocal<'db>> {
        self.locals.get(id.index())
    }

    pub fn block(&self, id: SBlockId) -> Option<&SBlock<'db>> {
        self.blocks.get(id.index())
    }

    pub fn callees(&self) -> Vec<SemanticCalleeRef<'db>> {
        let mut callees = Vec::new();
        for block in &self.blocks {
            for stmt in &block.stmts {
                if let SStmtKind::Assign {
                    expr: SExpr::Call { callee, .. },
                    ..
                } = &stmt.kind
                {
                    callees.push(*callee);
                }
            }
        }
        callees
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Update)]
pub struct SLocal<'db> {
    pub ty: TyId<'db>,
    pub mutability: Mutability,
    pub source: Option<LocalBinding<'db>>,
    pub role: SemanticLocalRole<'db>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Update)]
pub enum SemanticProjection<'db> {
    Field(usize),
    VariantField {
        variant: VariantIndex,
        enum_ty: TyId<'db>,
        field_idx: usize,
    },
    Index(SLocalId),
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Update)]
pub enum ValueProvenance<'db> {
    Ordinary,
    RootProvider(ProviderBinding<'db>),
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Update)]
pub enum PlaceProvenance<'db> {
    RootProvider(ProviderBinding<'db>),
    Derived {
        base: SLocalId,
        path: Box<[SemanticProjection<'db>]>,
    },
}

impl<'db> PlaceProvenance<'db> {
    pub fn root_provider(&self, locals: &[SLocal<'db>]) -> Option<ProviderBinding<'db>> {
        match self {
            Self::RootProvider(provider) => Some(provider.clone()),
            Self::Derived { base, .. } => locals
                .get(base.index())
                .and_then(|local| local.role.root_provider(locals)),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Update)]
pub enum SemanticBindingLowering<'db> {
    Erased,
    DirectValue {
        provenance: ValueProvenance<'db>,
    },
    PlaceCarrier {
        value_ty: TyId<'db>,
    },
    PlaceBoundValue {
        provenance: PlaceProvenance<'db>,
        value_ty: TyId<'db>,
    },
    DirectCarrier {
        provider: Option<ProviderBinding<'db>>,
        target_ty: TyId<'db>,
    },
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Update)]
pub enum SemanticLocalRole<'db> {
    Erased,
    DirectValue {
        provenance: ValueProvenance<'db>,
    },
    PlaceCarrier {
        value_ty: TyId<'db>,
    },
    PlaceBoundValue {
        provenance: PlaceProvenance<'db>,
        value_ty: TyId<'db>,
    },
    DirectCarrier {
        provider: Option<ProviderBinding<'db>>,
        target_ty: TyId<'db>,
    },
}

impl<'db> SemanticLocalRole<'db> {
    pub fn root_provider(&self, locals: &[SLocal<'db>]) -> Option<ProviderBinding<'db>> {
        match self {
            Self::DirectValue {
                provenance: ValueProvenance::RootProvider(provider),
            } => Some(provider.clone()),
            Self::PlaceBoundValue { provenance, .. } => provenance.root_provider(locals),
            Self::DirectCarrier {
                provider: Some(provider),
                ..
            } => Some(provider.clone()),
            Self::Erased
            | Self::DirectValue {
                provenance: ValueProvenance::Ordinary,
            }
            | Self::PlaceCarrier { .. }
            | Self::DirectCarrier { provider: None, .. } => None,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Update)]
pub struct SBlock<'db> {
    pub stmts: Vec<SStmt<'db>>,
    pub terminator: STerminator<'db>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Update)]
pub struct SemanticCalleeRef<'db> {
    pub key: SemanticInstanceKey<'db>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Update)]
pub enum SemOrigin<'db> {
    Expr(ExprId),
    Stmt(StmtId),
    Body(BodyOwner<'db>),
    Synthetic,
}

#[salsa::interned]
#[derive(Debug)]
pub struct SemanticConstRef<'db> {
    pub instance: SemanticInstanceKey<'db>,
    pub ty: TyId<'db>,
    pub origin: SemOrigin<'db>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Update)]
pub enum ManualContractSection {
    Init,
    Runtime,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Update)]
pub enum SemanticCodeRegionRef<'db> {
    ManualContractRoot { func: Func<'db> },
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Update)]
pub struct SPlace {
    pub local: SLocalId,
    pub path: Box<[SPlaceElem]>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Update)]
pub enum SPlaceElem {
    Field(FieldIndex),
    Index(SValueId),
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Update)]
pub struct SEffectArg<'db> {
    pub binding_idx: u32,
    pub arg: SEffectArgValue,
    pub pass_mode: EffectPassMode,
    pub target_ty: Option<TyId<'db>>,
    pub provider: Option<ProviderAddressSpace>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Update)]
pub enum SEffectArgValue {
    Place(SPlace),
    Value(SValueId),
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Update)]
pub enum SExpr<'db> {
    Forward(SValueId),
    UseValue(SValueId),
    CodeRegionRef {
        region: SemanticCodeRegionRef<'db>,
    },
    Const(SConst<'db>),
    Unary {
        op: UnOp,
        value: SValueId,
    },
    Binary {
        op: BinOp,
        lhs: SValueId,
        rhs: SValueId,
    },
    Cast {
        value: SValueId,
        to: TyId<'db>,
    },
    AggregateMake {
        ty: TyId<'db>,
        fields: Box<[SValueId]>,
    },
    EnumMake {
        enum_ty: TyId<'db>,
        variant: VariantIndex,
        fields: Box<[SValueId]>,
    },
    Field {
        base: SValueId,
        field: FieldIndex,
    },
    Index {
        base: SValueId,
        index: SValueId,
    },
    Borrow {
        place: SPlace,
        kind: BorrowKind,
        provider: Option<ProviderAddressSpace>,
    },
    GetEnumTag {
        value: SValueId,
    },
    IsEnumVariant {
        value: SValueId,
        variant: VariantIndex,
    },
    ExtractEnumField {
        value: SValueId,
        variant: VariantIndex,
        field: FieldIndex,
    },
    CodeRegionOffset {
        region: SemanticCodeRegionRef<'db>,
    },
    CodeRegionLen {
        region: SemanticCodeRegionRef<'db>,
    },
    Call {
        callee: SemanticCalleeRef<'db>,
        args: Box<[SValueId]>,
        effect_args: Box<[SEffectArg<'db>]>,
    },
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Update)]
pub enum SConst<'db> {
    Value(SemConstId<'db>),
    Ref(SemanticConstRef<'db>),
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Update)]
pub struct SStmt<'db> {
    pub origin: SemOrigin<'db>,
    pub kind: SStmtKind<'db>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Update)]
pub enum SStmtKind<'db> {
    Assign { dst: SLocalId, expr: SExpr<'db> },
    Store { dst: SPlace, src: SValueId },
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Update)]
pub struct STerminator<'db> {
    pub origin: SemOrigin<'db>,
    pub kind: STerminatorKind<'db>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Update)]
pub enum STerminatorKind<'db> {
    Goto(SBlockId),
    Branch {
        cond: SValueId,
        then_bb: SBlockId,
        else_bb: SBlockId,
    },
    MatchEnum {
        value: SValueId,
        enum_ty: TyId<'db>,
        cases: Box<[(VariantIndex, SBlockId)]>,
        default: Option<SBlockId>,
    },
    Return(Option<SValueId>),
}
