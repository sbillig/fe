//! Arbitrary replacement contents after a possible byte-level overwrite.
//! Entry reads have a different denotation and must never be used as havoc.
use super::{
    external::{ClobberCondition, ExternalSource, ReferentContract},
    footprint::AccessFootprint,
    guard::Guard,
    handle::{
        AddressOccurrence, OpaqueContentsId, OpaqueHandleContract, OpaqueHandleRef, OpaqueWriteSite,
    },
    index::{BinderScope, IndexNamespace},
    loan::CapabilityRef,
    path::RegionPath,
    region::{OverlapResult, RegionRoot, RegionSet, SymbolicPlace, substitute_clause},
    semantics::{CapabilityClass, UnresolvedCapability},
    shape::ShapeId,
    source::SourceExpr,
    state::{CapabilityValue, CapabilityValues},
    value::{Guarded, IndexPayload},
};
use crate::{
    analysis::ty::{trait_resolution::PredicateListId, ty_def::TyId},
    hir_def::scope_graph::ScopeId,
};

#[derive(Clone, Copy)]
pub struct OpaqueWrite<'db> {
    pub site: OpaqueWriteSite<'db>,
    pub scope: ScopeId<'db>,
    pub assumptions: PredicateListId<'db>,
}

impl<'db> OpaqueWrite<'db> {
    pub fn contents(
        self,
        values: &mut CapabilityValues<'db>,
        shape: ShapeId<'db>,
        scope: &BinderScope,
        clobber: Option<(&RegionRoot<'db>, AccessFootprint<'_, 'db>)>,
    ) -> Result<CapabilityValue<'db>, UnresolvedCapability<'db>> {
        let db = values.db;
        // Keep conditional replacements for identifiable cells. Seeds and
        // replacements inside already arbitrary memory share one closed heap
        // family; retaining their cell parameters would grow at every load.
        let saturated = matches!(self.site, OpaqueWriteSite::Seed)
            || clobber.is_some_and(
                |(root, _)| matches!(root, RegionRoot::External(source) if source.is_arbitrary()),
            );
        let mut failure = None;
        let value = values.from_shape(shape, scope, |semantics, path, scope| {
            let native = matches!(
                semantics.class,
                CapabilityClass::Borrow(_) | CapabilityClass::View
            );
            // Native bytes establish no valid target or authority. This address
            // is only a stable summary identity; typed use of the marker rejects.
            let ty = if native {
                TyId::ptr_to(db, semantics.target_ty)
            } else {
                semantics.representation_ty
            };
            let contract = match OpaqueHandleContract::for_ty(db, self.scope, self.assumptions, ty)
            {
                Ok(Some(contract)) => contract,
                Ok(None) | Err(_) => {
                    failure = Some(UnresolvedCapability(ty));
                    return Vec::new();
                }
            };
            let (witness_scope, source) = if saturated {
                (
                    scope.clone(),
                    ExternalSource::opaque_memory(ReferentContract::new(
                        db,
                        contract.target_ty,
                        contract.address_space,
                    )),
                )
            } else {
                // Independent cells may receive different addresses even at the
                // same write site. Lexical witnesses keep replay deterministic.
                let (witness_scope, witness) = scope.bind(IndexNamespace::Existential);
                (
                    witness_scope,
                    ExternalSource::opaque(
                        db,
                        OpaqueHandleRef {
                            contract,
                            occurrence: AddressOccurrence::Overwrite(OpaqueContentsId::new(
                                db,
                                self.site,
                                semantics.representation_ty,
                                path.clone(),
                            )),
                            arguments: scope.variables().chain([witness]).collect(),
                        },
                    ),
                )
            };
            let alternatives = if !saturated && let Some((target, written)) = clobber {
                let extent = written.extent;
                let target = SymbolicPlace {
                    root: target.clone(),
                    path: RegionPath::new(path.as_slice()),
                    views: Default::default(),
                };
                let target_region =
                    RegionSet::singleton(&witness_scope, target.root.clone(), target.path.clone());
                written
                    .region
                    .clauses()
                    .iter()
                    .filter_map(|clause| {
                        let fresh = clause.guard.scope().freshening(&witness_scope);
                        let guard = clause.guard.substitute(&fresh)?;
                        let written_region =
                            RegionSet::new(&witness_scope, vec![substitute_clause(clause, &fresh)]);
                        if matches!(
                            AccessFootprint::typed(&target_region).overlap(
                                db,
                                AccessFootprint {
                                    region: &written_region,
                                    extent: extent.substitute(&fresh),
                                }
                            ),
                            OverlapResult::Disjoint
                        ) {
                            return None;
                        }
                        let condition = SourceExpr::from_place(&target)
                            .zip(SourceExpr::from_place(&clause.payload))
                            .map(|(target, written)| {
                                Box::new(ClobberCondition::new(
                                    target,
                                    written.substitute(db, &fresh),
                                    extent.substitute(&fresh),
                                ))
                            });
                        Some((guard, condition))
                    })
                    .collect::<Vec<_>>()
            } else {
                vec![(Guard::always(&witness_scope), None)]
            };
            alternatives
                .into_iter()
                .map(|(guard, clobber)| {
                    let mut source = source.clone();
                    source.clobber = clobber;
                    let region = RegionSet::singleton(
                        guard.scope(),
                        RegionRoot::External(source),
                        RegionPath::default(),
                    );
                    let payload = if native {
                        CapabilityRef::Invalidated {
                            class: semantics.class,
                            region,
                        }
                    } else {
                        CapabilityRef::Address(region)
                    };
                    Guarded { guard, payload }
                })
                .collect()
        });
        failure.map_or(Ok(value), Err)
    }
}
