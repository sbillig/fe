//! Structural holders and the contents of explicitly inventoried storage.
//!
//! A carrier describes its referent region. Loading that region reads a separate
//! structural value; updating it never changes the carrier or its loan identity.
use std::collections::{BTreeMap, BTreeSet};

use super::{
    birth::AllocationBirth,
    external::StorageMatch,
    footprint::AccessFootprint,
    guard::{Guard, ValueOccurrence},
    handle::AddressOccurrence,
    index::{BinderScope, IndexExpr, IndexNamespace, IndexSubst},
    loan::{CapabilityRef, LoanDef},
    opaque::OpaqueWrite,
    path::{RegionPath, StructuralPath},
    region::{OverlapResult, RegionRoot, RegionSet},
    repack::RepackError,
    semantics::UnresolvedCapability,
    shape::ShapeId,
    value::{Guarded, IndexPayload, ValueId, ValueInterner, ValueLimits},
};
use crate::analysis::{HirAnalysisDb, semantic::normalized::NValueId};

pub type CapabilityValue<'db> = ValueId<'db, CapabilityRef<'db>>;
pub type CapabilityValues<'db> = ValueInterner<'db, CapabilityRef<'db>>;

/// Typed-cell matches for one request, whose guards are in the request's
/// binder scope. Their union is the represented part of a demand.
pub struct StorageCoverage<'a, 'db> {
    pub matches: Vec<(
        &'a RegionRoot<'db>,
        &'a CapabilityValue<'db>,
        StorageMatch<'db>,
    )>,
}

impl<'db> StorageCoverage<'_, 'db> {
    fn covered(&self) -> Option<Guard<'db>> {
        self.matches
            .iter()
            .map(|(_, _, matched)| matched.guard.clone())
            .reduce(|left, right| left.or(&right))
    }

    pub fn complete(&self, demand: &Guard<'db>) -> bool {
        self.covered()
            .is_some_and(|covered| demand.implies(&covered))
    }

    pub fn uncovered(&self, demand: &Guard<'db>) -> Option<Guard<'db>> {
        self.covered().map_or_else(
            || Some(demand.clone()),
            |covered| demand.difference(&covered),
        )
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StateError<'db> {
    OpaqueContents(UnresolvedCapability<'db>),
    MissingStorage(Box<RegionRoot<'db>>),
    Repack(RepackError<'db>),
    UnrepresentableWrite(Box<RegionRoot<'db>>),
}

/// The keys and shapes are the immutable inventory shared by every block state.
/// Keeping empty holders in the inventory makes all updates shape-checked and
/// prevents a missing referent from being mistaken for capability-free storage.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BorrowState<'db> {
    guard: Guard<'db>,
    values: BTreeMap<NValueId, CapabilityValue<'db>>,
    contents: BTreeMap<RegionRoot<'db>, CapabilityValue<'db>>,
    certified_contents: Vec<CertifiedContents<'db>>,
    scalar_cells: BTreeMap<RegionRoot<'db>, Vec<Guarded<'db, Option<IndexExpr<'db>>>>>,
}

/// A verified must range over a typed family and its possible contents.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CertifiedContents<'db> {
    pub family: RegionRoot<'db>,
    pub scope: BinderScope,
    pub coverage: Guard<'db>,
    pub contents: CapabilityValue<'db>,
}

impl<'db> BorrowState<'db> {
    pub fn new(
        values: &mut CapabilityValues<'db>,
        holders: impl IntoIterator<Item = (NValueId, ShapeId<'db>)>,
        storage: impl IntoIterator<Item = (RegionRoot<'db>, CapabilityValue<'db>)>,
    ) -> Self {
        let scope = BinderScope::default();
        let mut state = Self {
            guard: Guard::always(&scope),
            values: BTreeMap::new(),
            contents: BTreeMap::new(),
        };
        for (id, shape) in holders {
            assert!(
                state
                    .values
                    .insert(id, values.empty(shape, &scope))
                    .is_none()
            );
        }
        for (root, value) in storage {
            assert!(
                !matches!(root, RegionRoot::Value(_)),
                "an SSA value is not an inventoried storage root"
            );
            assert!(
                value
                    .scope()
                    .variables()
                    .all(|variable| root.indices().any(|index| index == variable)),
                "storage binders must occur in its root"
            );
            for index in root.indices() {
                value.scope().validate(index).expect("free storage binder");
            }
            assert!(state.contents.insert(root, value).is_none());
        }
        state
    }

    pub fn extend_storage(&mut self, inventory: &Self) {
        for (root, initial) in &inventory.contents {
            self.contents
                .entry(root.clone())
                .or_insert_with(|| initial.clone());
        }
    }

    /// Inventory can precede execution. Each allocation occurrence starts with
    /// its byte seed again; older loop-family members retain their own contents.
    /// A call's typed poststates are applied only after this birth.
    pub fn birth_allocations(
        &mut self,
        values: &mut CapabilityValues<'db>,
        inventory: &Self,
        families: &BTreeMap<AddressOccurrence<'db>, Vec<RegionRoot<'db>>>,
        births: &[AllocationBirth<'db>],
    ) {
        for birth in births {
            let Some(roots) = families.get(&birth.allocation.occurrence) else {
                continue;
            };
            for root in roots {
                let initial = &inventory.contents[root];
                let Some(born) = birth.selector(root, initial.scope()) else {
                    continue;
                };
                let old = self.contents.get_mut(root).expect("inventoried allocation");
                let kept =
                    values.map_guards(old, |guard| guard.difference(&born.in_scope(guard.scope())));
                let initial = values.with_guard(initial, &born);
                *old = values.join(&kept, &initial);
            }
        }
    }

    /// SSA representation roots are covered by their structural value.
    pub fn storage_coverage<'a>(
        &'a self,
        root: &'a RegionRoot<'db>,
        scope: &BinderScope,
    ) -> StorageCoverage<'a, 'db> {
        let representation = match root {
            RegionRoot::Value(value) => Some((root, self.value(*value))),
            _ => None,
        };
        StorageCoverage {
            matches: self
                .contents
                .iter()
                .chain(representation)
                .filter_map(|(candidate, contents)| {
                    let matched = storage_instance(candidate, contents.scope(), root, scope)?;
                    Some((candidate, contents, matched))
                })
                .collect(),
        }
    }

    pub fn guard(&self) -> &Guard<'db> {
        &self.guard
    }

    /// Restrict every holder and storage alternative to a feasible control-flow edge.
    pub fn constrain(&mut self, guard: &Guard<'db>, values: &mut CapabilityValues<'db>) -> bool {
        let Some(guard) = self.guard.and(guard) else {
            return false;
        };
        for value in self.values.values_mut().chain(self.contents.values_mut()) {
            *value = values.with_guard(value, &guard.in_scope(value.scope()));
        }
        self.guard = guard;
        true
    }

    pub fn forget_iteration(
        &mut self,
        values: &mut CapabilityValues<'db>,
        repeated: impl Fn(IndexExpr<'db>) -> bool + Copy,
        occurrence: impl Fn(ValueOccurrence) -> bool + Copy,
    ) {
        self.guard = self
            .guard
            .forget_occurrences(occurrence)
            .forget_indices(repeated);
        let mut destination = CapabilityValues::new(values.db, ValueLimits::default());
        for value in self.values.values_mut().chain(self.contents.values_mut()) {
            let mapped = values.map_payloads(value, &mut destination, |_, _, entry, domain| {
                let guard = domain.forget_occurrences(occurrence);
                let payload = entry.payload.forget_occurrences(occurrence);
                let mut scope = guard.scope().clone();
                let indices: BTreeSet<_> = guard
                    .indices()
                    .into_iter()
                    .chain(payload.indices())
                    .filter(|index| repeated(*index))
                    .collect();
                let bindings: Vec<_> = indices
                    .into_iter()
                    .map(|index| {
                        let (nested, witness) = scope.bind(IndexNamespace::Existential);
                        scope = nested;
                        (index, witness)
                    })
                    .collect();
                let subst = IndexSubst::new(guard.scope(), &scope, bindings)
                    .expect("previous value occurrence witnesses");
                guard
                    .substitute(&subst)
                    .map(|guard| Guarded {
                        guard,
                        payload: payload.substitute(values.db, &subst),
                    })
                    .into_iter()
                    .collect()
            });
            *value = destination.widen(&mapped);
        }
    }

    pub fn value(&self, id: NValueId) -> &CapabilityValue<'db> {
        self.values.get(&id).expect("inventoried SSA value")
    }

    pub fn set_value(&mut self, id: NValueId, value: CapabilityValue<'db>) {
        let old = self.values.get_mut(&id).expect("inventoried SSA value");
        assert_eq!(old.shape(), value.shape(), "SSA capability shape mismatch");
        assert_eq!(old.scope(), value.scope(), "SSA capability scope mismatch");
        *old = value;
    }

    pub fn holders(&self) -> impl Iterator<Item = (NValueId, &CapabilityValue<'db>)> {
        self.values.iter().map(|(id, value)| (*id, value))
    }

    pub fn storage(&self) -> impl Iterator<Item = (&RegionRoot<'db>, &CapabilityValue<'db>)> {
        self.contents.iter()
    }

    pub fn certified(&self) -> &[CertifiedContents<'db>] {
        &self.certified_contents
    }

    fn matching_certified_contents<'a>(
        &'a self,
        clause: &Guarded<'db, SymbolicPlace<'db>>,
    ) -> Vec<(&'a CertifiedContents<'db>, IndexSubst<'db>, Guard<'db>)> {
        self.certified_contents
            .iter()
            .filter_map(|certificate| {
                let (RegionRoot::External(family), RegionRoot::External(instance)) =
                    (&certificate.family, &clause.payload.root)
                else {
                    return None;
                };
                if instance.uncertain() || instance.is_reachable() {
                    return None;
                }
                let matched =
                    family.match_instance(&certificate.scope, instance, clause.guard.scope())?;
                let write = matched.write?;
                let guard = certificate
                    .coverage
                    .substitute(&matched.substitution)?
                    .and(&matched.guard)?
                    .and(&write.guard.substitute(&matched.substitution)?)?
                    .and(&clause.guard)?
                    .and(&self.guard.in_scope(clause.guard.scope()))?;
                Some((certificate, matched.substitution, guard))
            })
            .collect()
    }

    /// Definite typed initialization supplied by a verified whole-cell range.
    /// The selected request path is retained, so this is also valid for a leaf.
    pub fn certified_initialized_region(&self, region: &RegionSet<'db>) -> RegionSet<'db> {
        RegionSet::new(
            region.scope(),
            region.clauses().iter().flat_map(|clause| {
                self.matching_certified_contents(clause)
                    .into_iter()
                    .map(|(_, _, guard)| Guarded {
                        guard,
                        payload: clause.payload.clone(),
                    })
            }),
        )
    }

    /// Keep a proved range separate from possible contents, whose join may
    /// widen guards. Reads use this must fact to exclude old possibilities only
    /// on covered members of the same typed storage family.
    pub fn certify_family_contents(
        &mut self,
        family: &RegionRoot<'db>,
        family_scope: &BinderScope,
        coverage: &Guard<'db>,
        replacement: &CapabilityValue<'db>,
    ) -> bool {
        assert_eq!(coverage.scope(), family_scope, "certificate guard scope");
        assert_eq!(
            replacement.scope(),
            family_scope,
            "certificate content scope"
        );
        assert!(
            coverage.implies(&self.guard.in_scope(family_scope)),
            "certificate must be restricted to its state"
        );
        if !matches!(family, RegionRoot::External(source) if !source.uncertain() && !source.is_reachable())
            || !self.contents.get(family).is_some_and(|old| {
                old.scope() == family_scope && old.shape() == replacement.shape()
            })
        {
            return false;
        }
        self.certified_contents.push(CertifiedContents {
            family: family.clone(),
            scope: family_scope.clone(),
            coverage: coverage.clone(),
            contents: replacement.clone(),
        });
        true
    }

    pub fn join(&mut self, other: &Self, values: &mut CapabilityValues<'db>) -> bool {
        assert!(
            self.values.keys().eq(other.values.keys()),
            "holder inventory mismatch"
        );
        assert!(
            self.contents.keys().eq(other.contents.keys()),
            "storage inventory mismatch"
        );
        let joined_guard = self.guard.or(&other.guard);
        let mut changed = joined_guard != self.guard;
        self.guard = joined_guard;
        for (old, incoming) in self
            .values
            .values_mut()
            .zip(other.values.values())
            .chain(self.contents.values_mut().zip(other.contents.values()))
        {
            let joined = values.join(old, incoming);
            let joined = values.widen(&joined);
            changed |= joined != *old;
            *old = joined;
        }
        changed
    }

    /// Resolve only the direct carrier. Nested handles are contents of its target,
    /// not additional destinations of the outer borrow.
    pub fn referent_region(
        &self,
        db: &'db dyn HirAnalysisDb,
        carrier: NValueId,
        path: &RegionPath<IndexExpr<'db>>,
        loans: &[LoanDef<'db>],
    ) -> RegionSet<'db> {
        let value = self.value(carrier);
        value
            .direct()
            .iter()
            .filter(|entry| !matches!(entry.payload, CapabilityRef::Invalidated { .. }))
            .fold(RegionSet::empty(value.scope()), |region, entry| {
                region.union(
                    &entry
                        .payload
                        .region(db, loans, entry.guard.scope())
                        .with_guard(&entry.guard)
                        .close_existentials(value.scope()),
                )
            })
            .project(path)
    }

    /// Load structural contents from the selected storage, preserving guards and
    /// array-family arguments. The caller supplies the snapshot's enum identity.
    pub fn read_region(
        &self,
        db: &'db dyn HirAnalysisDb,
        values: &mut CapabilityValues<'db>,
        region: &RegionSet<'db>,
        shape: ShapeId<'db>,
        occurrence: ValueOccurrence,
    ) -> Result<CapabilityValue<'db>, StateError<'db>> {
        let mut result = values.empty(shape, region.scope());
        if !shape.contains_capability(db) {
            return Ok(result);
        }
        for clause in region.clauses() {
            let coverage = self.storage_coverage(&clause.payload.root, clause.guard.scope());
            if !coverage.complete(&clause.guard) {
                return Err(StateError::MissingStorage(Box::new(
                    clause.payload.root.clone(),
                )));
            }
            // A certified range replaces every older possibility on its members.
            let certified = self.matching_certified_contents(clause);
            let certified_coverage = certified
                .iter()
                .map(|(_, _, guard)| guard.clone())
                .reduce(|left, right| left.or(&right));
            let possible = coverage
                .matches
                .into_iter()
                .filter_map(|(_, contents, matched)| {
                    let guard = matched.guard.and(&clause.guard)?;
                    let guard = match &certified_coverage {
                        Some(covered) => guard.difference(covered)?,
                        None => guard,
                    };
                    Some((contents, matched.substitution, guard))
                });
            let certified = certified
                .into_iter()
                .map(|(certificate, substitution, guard)| {
                    (&certificate.contents, substitution, guard)
                });
            for (contents, substitution, guard) in possible.chain(certified) {
                let contents = values.substitute(contents, &substitution);
                let path = StructuralPath::new(clause.payload.path.as_slice());
                let Some(selected) = values.project(&contents, &path, occurrence) else {
                    continue;
                };
                let selected = clause
                    .payload
                    .views
                    .apply(db, values, &selected, clause.payload.path.as_slice(), false)
                    .map_err(StateError::Repack)?;
                assert_eq!(selected.shape(), shape, "referent load shape mismatch");
                let selected = values.with_guard(&selected, &guard);
                let selected = values.close_existentials(&selected, region.scope());
                result = values.join(&result, &selected);
            }
        }
        Ok(result)
    }

    /// A singleton destination replaces its selected member under its guard.
    /// Ambiguous alternatives retain old contents and weakly add the new ones.
    /// All replacements are prepared before changing state, so failure is atomic.
    pub fn write_region(
        &mut self,
        overwrite: OpaqueWrite<'db>,
        values: &mut CapabilityValues<'db>,
        region: &RegionSet<'db>,
        replacement: &CapabilityValue<'db>,
    ) -> Result<(), StateError<'db>> {
        self.write_regions(overwrite, values, &[(region, replacement)])
    }

    /// Byte writes may destroy every overlapping typed interpretation, including
    /// an exact cell when an intrinsic supplies no structural poststate.
    pub fn invalidate_memory(
        &mut self,
        values: &mut CapabilityValues<'db>,
        footprint: AccessFootprint<'_, 'db>,
        overwrite: OpaqueWrite<'db>,
    ) -> Result<(), StateError<'db>> {
        let region = footprint.region;
        let mut updates = BTreeMap::new();
        for (root, contents) in &self.contents {
            if !contents.shape().contains_capability(values.db) {
                continue;
            }
            let candidate =
                RegionSet::singleton(contents.scope(), root.clone(), RegionPath::default())
                    .substitute(values.db, &contents.scope().freshening(region.scope()))
                    .close_existentials(region.scope());
            if !matches!(
                AccessFootprint::typed(&candidate).overlap(values.db, footprint),
                OverlapResult::Disjoint
            ) {
                let unknown = overwrite
                    .contents(
                        values,
                        contents.shape(),
                        contents.scope(),
                        Some((root, footprint)),
                    )
                    .map_err(StateError::OpaqueContents)?;
                updates.insert(root.clone(), values.join(contents, &unknown));
            }
        }
        self.contents.extend(updates);
        Ok(())
    }

    /// Apply one call's complete poststate without imposing an order on aliased
    /// inputs. Independent destinations replace exactly; possibly overlapping
    /// destinations weakly retain every candidate. Failure leaves state intact.
    pub fn write_regions(
        &mut self,
        overwrite: OpaqueWrite<'db>,
        values: &mut CapabilityValues<'db>,
        replacements: &[(&RegionSet<'db>, &CapabilityValue<'db>)],
    ) -> Result<(), StateError<'db>> {
        let scope = BinderScope::default();
        let independent: Vec<_> = replacements
            .iter()
            .map(|(region, replacement)| {
                assert_eq!(region.scope(), replacement.scope(), "store scope mismatch");
                region
                    .substitute(values.db, &region.scope().freshening(&scope))
                    .close_existentials(&scope)
            })
            .collect();
        let mut updates = BTreeMap::new();
        // Family members each destination clause reaches as the same typed cell,
        // and those among them the typed update wrote.
        let mut typed = Vec::new();
        let mut written = Vec::new();
        for (index, (region, replacement)) in replacements.iter().enumerate() {
            let interferes = independent.iter().enumerate().any(|(other, region)| {
                index != other
                    && !matches!(
                        independent[index].overlap(values.db, region),
                        OverlapResult::Disjoint
                    )
            });
            for clause in region.clauses() {
                let coverage = self.storage_coverage(&clause.payload.root, clause.guard.scope());
                if matches!(clause.payload.root, RegionRoot::Value(_))
                    || !coverage.complete(&clause.guard)
                {
                    return Err(StateError::MissingStorage(Box::new(
                        clause.payload.root.clone(),
                    )));
                }
                let mut domains = BTreeMap::new();
                let mut applied = BTreeMap::new();
                for (root, contents, matched) in coverage.matches {
                    if let Some(domain) = &matched.typed {
                        domains.insert(root, domain.clone());
                    }
                    let (Some(write_guard), Some(embedding)) =
                        (clause.guard.and(&matched.guard), matched.write)
                    else {
                        continue;
                    };
                    if let Some(domain) = matched.typed {
                        applied.insert(root, domain);
                    }
                    let path = StructuralPath::new(clause.payload.path.as_slice());
                    let old = updates.get(root).unwrap_or(contents);
                    let lift = IndexSubst::new(replacement.scope(), clause.guard.scope(), [])
                        .expect("store witness scope");
                    let replacement = values.substitute(replacement, &lift);
                    let replacement = clause
                        .payload
                        .views
                        .apply(
                            values.db,
                            values,
                            &replacement,
                            clause.payload.path.as_slice(),
                            true,
                        )
                        .map_err(StateError::Repack)?;
                    let changed = values
                        .replace_family(old, &path, &replacement, &write_guard, &embedding)
                        .map_err(|_| {
                            StateError::UnrepresentableWrite(Box::new(clause.payload.root.clone()))
                        })?;
                    let updated = if !interferes && region.definite_write().is_some() {
                        changed
                    } else {
                        values.join(old, &changed)
                    };
                    updates.insert(root.clone(), updated);
                }
                typed.push(domains);
                written.push(applied);
            }
        }
        let clauses = replacements
            .iter()
            .flat_map(|(region, _)| region.clauses().iter().map(move |clause| (*region, clause)));
        for ((region, clause), domains) in clauses.zip(&typed) {
            for (root, contents) in &self.contents {
                if !contents.shape().contains_capability(values.db)
                    || !root.may_alias_unknown(&clause.payload.root)
                {
                    continue;
                }
                // The typed update accounts for members reached as the same
                // typed cell. Other members may still share written bytes.
                let residual = match domains.get(root) {
                    Some(domain) => Guard::always(contents.scope()).difference(domain),
                    None => Some(Guard::always(contents.scope())),
                };
                let Some(residual) = residual else {
                    continue;
                };
                let candidate =
                    RegionSet::singleton(contents.scope(), root.clone(), RegionPath::default())
                        .with_guard(&residual)
                        .substitute(values.db, &contents.scope().freshening(region.scope()))
                        .close_existentials(region.scope());
                if matches!(
                    candidate.overlap(values.db, region),
                    OverlapResult::Disjoint
                ) {
                    continue;
                }
                // Unknown bases may overlap at a byte offset, even when
                // their typed contents have the same shape. A partial store
                // can corrupt a native capability rather than copy it intact.
                let old = updates.get(root).unwrap_or(contents);
                let unknown = overwrite
                    .contents(
                        values,
                        contents.shape(),
                        contents.scope(),
                        Some((root, AccessFootprint::typed(region))),
                    )
                    .map_err(StateError::OpaqueContents)?;
                let unknown = values.with_guard(&unknown, &residual);
                updates.insert(root.clone(), values.join(old, &unknown));
            }
        }
        // Unknown bases may overlap at different physical offsets. Retain every
        // compatible stored capability, with independent witnesses for the source
        // and destination families; an unknown write never removes prior contents.
        // Members the typed update wrote already hold the replacement.
        let mut written = written.into_iter();
        for (region, replacement) in replacements {
            for clause in region.clauses() {
                let applied = written.next().expect("typed domains for every clause");
                let lift = IndexSubst::new(replacement.scope(), clause.guard.scope(), [])
                    .expect("unknown store witness scope");
                let replacement = values.substitute(replacement, &lift);
                let replacement = clause
                    .payload
                    .views
                    .apply(
                        values.db,
                        values,
                        &replacement,
                        clause.payload.path.as_slice(),
                        true,
                    )
                    .map_err(StateError::Repack)?;
                let replacement = values.with_guard(&replacement, &clause.guard);
                let leaves = values.leaves(&replacement, ValueOccurrence::Summary);
                if leaves.is_empty() {
                    continue;
                }
                let db = values.db;
                for (root, contents) in &self.contents {
                    if !root.may_alias_unknown(&clause.payload.root)
                        || (root == &clause.payload.root
                            && !root.is_reachable()
                            && root.indices().next().is_none())
                    {
                        continue;
                    }
                    let residual = match applied.get(root) {
                        Some(domain) => Guard::always(contents.scope()).difference(domain),
                        None => Some(Guard::always(contents.scope())),
                    };
                    let Some(residual) = residual else {
                        continue;
                    };
                    let candidate =
                        RegionSet::singleton(contents.scope(), root.clone(), RegionPath::default())
                            .with_guard(&residual)
                            .substitute(values.db, &contents.scope().freshening(region.scope()))
                            .close_existentials(region.scope());
                    if matches!(
                        candidate.overlap(values.db, region),
                        OverlapResult::Disjoint
                    ) {
                        continue;
                    }
                    let added = values.from_shape(
                        contents.shape(),
                        contents.scope(),
                        |semantics, _, scope| {
                            leaves
                                .iter()
                                .filter(|leaf| leaf.semantics == semantics)
                                .filter_map(|leaf| {
                                    let subst = leaf.guard.scope().freshening(scope);
                                    Some(Guarded {
                                        guard: leaf.guard.substitute(&subst)?,
                                        payload: leaf.payload.substitute(db, &subst),
                                    })
                                })
                                .collect()
                        },
                    );
                    let added = values.with_guard(&added, &residual);
                    let old = updates.get(root).unwrap_or(contents);
                    updates.insert(root.clone(), values.join(old, &added));
                }
            }
        }
        self.contents.extend(updates);
        Ok(())
    }
}

fn storage_instance<'db>(
    root: &RegionRoot<'db>,
    scope: &BinderScope,
    instance: &RegionRoot<'db>,
    instance_scope: &BinderScope,
) -> Option<StorageMatch<'db>> {
    match (root, instance) {
        (RegionRoot::External(root), RegionRoot::External(instance)) => {
            root.match_instance(scope, instance, instance_scope)
        }
        _ if root == instance => Some(StorageMatch {
            substitution: IndexSubst::new(scope, instance_scope, []).ok()?,
            guard: Guard::always(instance_scope),
            typed: Some(Guard::always(scope)),
            write: Some(Guarded {
                guard: Guard::always(scope),
                payload: BTreeMap::new(),
            }),
        }),
        _ => None,
    }
}

#[cfg(test)]
mod tests;
