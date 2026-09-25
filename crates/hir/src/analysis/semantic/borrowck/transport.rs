//! One input transport policy for entry facts, summary verification, and calls.
use std::collections::BTreeMap;

use crate::analysis::{
    HirAnalysisDb,
    semantic::{
        SemanticInstance,
        capability::{
            external::{ExternalOrigin, ExternalSource, ReferentContract},
            handle::{HandleAddressSpace, OpaqueHandleContract},
            index::IndexExpr,
            path::{Projection, RegionPath, project_referent_ty},
            semantics::{CapabilityClass, CapabilitySemantics, TransportClass},
            shape::{ShapeError, ShapeId, capability_shape},
            source::InputOrigin,
        },
        normalized::{NRootKind, NValueDefinition, NormalizedBody},
    },
    ty::{ProviderAddressSpace, ty_check::BodyOwner, ty_def::TyId},
};

/// Ordinary parameters cannot carry native mutable access to provider storage.
/// A receiver explicitly supports provider transport. Read-only views transport
/// their contents; they do not turn nested native handles into plain values.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum InputTransportMode {
    Ordinary,
    Receiver,
}

impl InputTransportMode {
    pub(super) fn for_param(db: &dyn HirAnalysisDb, owner: BodyOwner<'_>, param: u32) -> Self {
        if param == 0
            && matches!(owner, BodyOwner::Func(function) if function.receiver_ty(db).is_some())
        {
            Self::Receiver
        } else {
            Self::Ordinary
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum TransportObligation {
    Memory,
    Writable,
}

/// Held transport applies while a route follows native borrows and views.
/// Following a raw pointer or nominal handle ends it.
#[derive(Clone, Copy)]
pub(super) struct TransportCursor {
    mode: InputTransportMode,
    held: bool,
}

impl TransportCursor {
    pub(super) fn new(mode: InputTransportMode) -> Self {
        Self { mode, held: true }
    }

    fn after(mut self, semantics: CapabilitySemantics<'_>) -> Self {
        self.held &= matches!(
            semantics.class,
            CapabilityClass::Borrow(_) | CapabilityClass::View
        );
        self
    }

    pub(super) fn obligation(
        self,
        semantics: CapabilitySemantics<'_>,
    ) -> Option<TransportObligation> {
        (self.held && semantics.transport == TransportClass::MemoryBorrow).then_some(
            match self.mode {
                InputTransportMode::Ordinary => TransportObligation::Memory,
                InputTransportMode::Receiver => TransportObligation::Writable,
            },
        )
    }
}

#[derive(Clone)]
pub(super) struct InputTransportContract<'db> {
    instance: SemanticInstance<'db>,
    params: BTreeMap<u32, TyId<'db>>,
}

/// The referent reached by following stored capabilities, and the last
/// capability selected on the way.
#[derive(Clone, Copy)]
pub(super) struct TransportRoute<'db> {
    pub ty: TyId<'db>,
    pub cursor: Option<TransportCursor>,
    pub selected: Option<(CapabilitySemantics<'db>, ReferentContract<'db>)>,
}

impl<'db> TransportRoute<'db> {
    /// A route whose referents carry no input transport refinement.
    pub(super) fn untransported(ty: TyId<'db>) -> Self {
        Self {
            ty,
            cursor: None,
            selected: None,
        }
    }
}

impl<'db> InputTransportContract<'db> {
    pub(super) fn new(body: &NormalizedBody<'db>) -> Self {
        let mut params = BTreeMap::new();
        for root in &body.roots {
            if let NRootKind::ParamPlace { param } = root.kind {
                params.insert(param, root.ty);
            }
        }
        for value in &body.values {
            if let NValueDefinition::EntryParam { param } = value.definition {
                params.insert(param, value.ty);
            }
        }
        Self {
            instance: body.owner,
            params,
        }
    }

    pub(super) fn param_ty(&self, param: u32) -> Option<TyId<'db>> {
        self.params.get(&param).copied()
    }

    pub(super) fn parameter(&self, db: &dyn HirAnalysisDb, param: u32) -> TransportCursor {
        TransportCursor::new(InputTransportMode::for_param(
            db,
            self.instance.key(db).owner(db),
            param,
        ))
    }

    pub(super) fn shape(
        &self,
        db: &'db dyn HirAnalysisDb,
        ty: TyId<'db>,
    ) -> Result<ShapeId<'db>, ShapeError<'db>> {
        capability_shape(
            db,
            self.instance.key(db).impl_env(db).normalization_scope(db),
            self.instance.assumptions(db),
            ty,
        )
    }

    pub(super) fn referent(
        &self,
        db: &'db dyn HirAnalysisDb,
        semantics: CapabilitySemantics<'db>,
        cursor: Option<TransportCursor>,
    ) -> Result<ReferentContract<'db>, ShapeError<'db>> {
        let mut contract = referent_contract(db, self.instance, semantics)?;
        if cursor.and_then(|cursor| cursor.obligation(semantics))
            == Some(TransportObligation::Memory)
        {
            contract.address_space = HandleAddressSpace::Known(ProviderAddressSpace::Memory);
        }
        Ok(contract)
    }

    /// Follow the capability stored at each path from the route's referent.
    pub(super) fn follow<'a>(
        &self,
        db: &'db dyn HirAnalysisDb,
        mut route: TransportRoute<'db>,
        paths: impl IntoIterator<Item = &'a [Projection<IndexExpr<'db>>]>,
    ) -> Result<Option<TransportRoute<'db>>, ShapeError<'db>>
    where
        'db: 'a,
    {
        for path in paths {
            let Some(slot) = project_referent_ty(db, self.instance, route.ty, path) else {
                return Ok(None);
            };
            let Some(semantics) = self.shape(db, slot)?.direct(db) else {
                return Ok(None);
            };
            let contract = self.referent(db, semantics, route.cursor)?;
            route = TransportRoute {
                ty: contract.ty,
                cursor: route.cursor.map(|cursor| cursor.after(semantics)),
                selected: Some((semantics, contract)),
            };
        }
        Ok(Some(route))
    }

    /// The transport route of an exact input-derived source from its parameter.
    pub(super) fn route(
        &self,
        db: &'db dyn HirAnalysisDb,
        source: &ExternalSource<'db>,
    ) -> Result<Option<TransportRoute<'db>>, ShapeError<'db>> {
        let ExternalOrigin::Input(input) = &source.origin else {
            return Ok(None);
        };
        let Some(param_ty) = self
            .param_ty(input.param())
            .filter(|_| !source.is_reachable())
        else {
            return Ok(None);
        };
        let start = TransportRoute {
            ty: param_ty,
            cursor: Some(self.parameter(db, input.param())),
            selected: None,
        };
        let start = match input.origin() {
            // A parameter place need not hold a capability itself.
            InputOrigin::Place(_) if self.shape(db, param_ty)?.direct(db).is_none() => Some(start),
            InputOrigin::Place(_) => self.follow(db, start, [&[][..]])?,
            InputOrigin::Slot { slot, .. } => self.follow(
                db,
                TransportRoute {
                    ty: param_ty.as_view(db).unwrap_or(param_ty),
                    ..start
                },
                [slot.as_slice()],
            )?,
        };
        let Some(start) = start else {
            return Ok(None);
        };
        self.follow(
            db,
            start,
            input
                .dereferences()
                .iter()
                .chain(source.dereferences())
                .map(RegionPath::as_slice),
        )
    }
}

pub(super) fn referent_contract<'db>(
    db: &'db dyn HirAnalysisDb,
    instance: SemanticInstance<'db>,
    semantics: CapabilitySemantics<'db>,
) -> Result<ReferentContract<'db>, ShapeError<'db>> {
    let space = if matches!(
        semantics.class,
        CapabilityClass::Handle | CapabilityClass::Pointer
    ) {
        OpaqueHandleContract::for_ty(
            db,
            instance.key(db).impl_env(db).normalization_scope(db),
            instance.assumptions(db),
            semantics.representation_ty,
        )
        .map_err(|error| ShapeError::UnresolvedCapability(error.0))?
        .ok_or(ShapeError::UnresolvedCapability(
            semantics.representation_ty,
        ))?
        .address_space
    } else {
        HandleAddressSpace::Unspecified
    };
    Ok(ReferentContract::new(db, semantics.target_ty, space))
}
