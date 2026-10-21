//! This module implements import and export resolution for HIR.
//!
//! Imports are resolved to a fixed point: each pass resolves uses from scratch
//! against the resolutions that the others had after the last pass, until a
//! pass changes nothing. A use's own resolution is withdrawn while it is
//! resolved, so it can't name itself. Because nothing is kept from an earlier
//! pass, a resolution that a later import shadows is replaced, and because no
//! use sees what another resolved in the same pass, the result doesn't depend
//! on the order of the uses.
use std::{cell::RefCell, hash::BuildHasher, mem};

use common::{indexmap::IndexMap, ingot::Ingot};
use rustc_hash::{FxBuildHasher, FxHashMap, FxHashSet};
use salsa::Update;
use thin_vec::ThinVec;

use super::{
    EarlyNameQueryId,
    diagnostics::ImportDiag,
    name_resolver::{
        NameDerivation, NameDomain, NameRes, NameResBucket, NameResKind, NameResolutionError,
        NameResolver, QueryDirective,
    },
};
use crate::{
    analysis::HirAnalysisDb,
    hir_def::{
        HirIngot, IdentId, Partial, Use, UsePathSegment, prim_ty::PrimTy, scope_graph::ScopeId,
    },
};

/// The names a glob import brings into its scope.
type GlobNames<'db> = FxHashMap<IdentId<'db>, Vec<NameRes<'db>>>;

/// A failed lookup: the index of the path segment, and the error.
type SegmentError<'db> = (usize, NameResolutionError<'db>);

pub(crate) struct ImportResolver<'db> {
    db: &'db dyn HirAnalysisDb,

    /// The ingot that is being resolved.
    ingot: Ingot<'db>,

    /// The current resolutions of the uses, which lookups see.
    imports: ResolvedImports<'db>,

    /// The bucket that each use other than a glob imports.
    buckets: FxHashMap<Use<'db>, NameResBucket<'db>>,

    /// The named uses that import each name into each scope.
    named_uses: IndexMap<(ScopeId<'db>, IdentId<'db>), Vec<Use<'db>>>,

    /// The scopes of the ingot whose imports lookups have read since this was
    /// last taken.
    reads: RefCell<FxHashSet<ScopeId<'db>>>,
}

/// What a use imports.
#[derive(PartialEq, Eq)]
enum Imported<'db> {
    Glob(GlobNames<'db>),
    Bucket(NameResBucket<'db>),
}

impl<'db> Imported<'db> {
    fn none(db: &'db dyn HirAnalysisDb, use_: Use<'db>) -> Self {
        if use_.is_glob(db) {
            Self::Glob(GlobNames::default())
        } else {
            Self::Bucket(NameResBucket::default())
        }
    }

    fn hash(&self, use_: Use<'db>) -> u64 {
        match self {
            Self::Glob(names) => names.iter().fold(0u64, |hash, entry| {
                hash.wrapping_add(FxBuildHasher.hash_one((use_, entry)))
            }),
            Self::Bucket(bucket) => FxBuildHasher.hash_one((use_, bucket)),
        }
    }
}

impl<'db> ImportResolver<'db> {
    pub(crate) fn new(db: &'db dyn HirAnalysisDb, ingot: Ingot<'db>) -> Self {
        Self {
            db,
            ingot,
            imports: ResolvedImports::default(),
            buckets: FxHashMap::default(),
            named_uses: IndexMap::default(),
            reads: RefCell::default(),
        }
    }

    pub(crate) fn resolve_imports(mut self) -> (ResolvedImports<'db>, Vec<ImportDiag<'db>>) {
        let db = self.db;
        let uses: Vec<_> = self
            .ingot
            .module_tree(db)
            .all_modules()
            .flat_map(|top_mod| top_mod.scope_graph(db).unresolved_uses.iter().copied())
            .collect();
        for &use_ in &uses {
            let scope = use_scope(db, use_);
            if use_.is_glob(db) {
                // Making every glob's entry up front fixes the order in which
                // lookups visit globs, so a pass that changes nothing finds
                // the same names in the same order.
                self.imports
                    .glob_resolved
                    .entry(scope)
                    .or_default()
                    .imported
                    .insert(use_, GlobNames::default());
            } else if let Some(name) = use_.imported_name(db) {
                self.named_uses.entry((scope, name)).or_default().push(use_);
            }
        }

        // A pass only resolves the uses that read the imports of a scope that
        // the last pass changed; the others would resolve as before.
        //
        // `state` sums the hashes of all resolutions. A state that repeats
        // means the passes cycle without settling: the uses that changed in
        // the cycle depend on each other in a way that no resolution
        // satisfies, like `use x::y as z` and `use z::x` where the outer scope
        // has an `x`, or in more than one way, like `use x as y` and
        // `use y as x` where the outer scope has both. A pass that changes
        // nothing repeats the state of the last one, which ends the iteration
        // without marking any use.
        let mut outcomes = vec![Ok(()); uses.len()];
        let mut changed_in = vec![0; uses.len()];
        let mut readers: FxHashMap<ScopeId<'db>, FxHashSet<usize>> = FxHashMap::default();
        let mut dirty: FxHashSet<usize> = (0..uses.len()).collect();
        let mut state = 0u64;
        let mut states = FxHashMap::from_iter([(state, 0)]);
        for pass in 1.. {
            let resolved: Vec<_> = dirty
                .into_iter()
                .map(|idx| {
                    let resolved = self.resolve(uses[idx]);
                    for scope in self.reads.take() {
                        readers.entry(scope).or_default().insert(idx);
                    }
                    (idx, resolved)
                })
                .collect();
            dirty = FxHashSet::default();
            for (idx, (imported, outcome)) in resolved {
                outcomes[idx] = outcome;
                let Some(imported) = imported else {
                    continue;
                };
                let use_ = uses[idx];
                let hash = imported.hash(use_);
                let old = self.replace(use_, imported);
                state = state.wrapping_add(hash).wrapping_sub(old.hash(use_));
                changed_in[idx] = pass;
                let scope_readers = readers.get(&use_scope(db, use_)).into_iter().flatten();
                dirty.extend(scope_readers.filter(|&&reader| reader != idx));
            }
            if let Some(start) = states.insert(state, pass) {
                for (outcome, &changed) in outcomes.iter_mut().zip(&changed_in) {
                    if changed > start {
                        *outcome = Err((0, NameResolutionError::Ambiguous(ThinVec::new())));
                    }
                }
                break;
            }
        }

        let mut diags: Vec<_> = uses
            .iter()
            .zip(outcomes)
            .filter(|(use_, _)| !use_.is_synthetic_use(db))
            .filter_map(|(&use_, outcome)| {
                let (idx, err) = outcome.err()?;
                self.diag(use_, idx, err)
            })
            .collect();
        diags.extend(self.conflicts());

        for &use_ in &uses {
            if !use_.is_glob(db)
                && use_.imported_name(db).is_none()
                && let Some(bucket) = self.buckets.remove(&use_)
                && !bucket.is_empty()
            {
                let unnamed = &mut self.imports.unnamed_resolved;
                unnamed.entry(use_scope(db, use_)).or_default().push(bucket);
            }
        }

        (self.imports, diags)
    }

    /// Resolves `use_` against the current resolutions of the others. Returns
    /// what it imports if that changed, and the error if the resolution failed.
    fn resolve(
        &mut self,
        use_: Use<'db>,
    ) -> (Option<Imported<'db>>, Result<(), SegmentError<'db>>) {
        let db = self.db;
        let scope = use_scope(db, use_);
        let own = self.replace(use_, Imported::none(db, use_));
        let resolved = self.resolve_base(use_, scope).and_then(|target| {
            if use_.is_glob(db) {
                let mut names =
                    NameResolver::new(db, &*self).collect_all_resolutions_for_glob(target, scope);
                for res in names.values_mut().flatten() {
                    res.derivation = NameDerivation::GlobImported(use_);
                }
                return Ok(Imported::Glob(names));
            }
            let segments = use_path(db, use_)?;
            let idx = segments.len() - 1;
            let mut bucket = self.lookup(scope, target, idx, &segments[idx])?;
            bucket.set_derivation(NameDerivation::NamedImported(use_));
            Ok(Imported::Bucket(bucket))
        });
        let (imported, outcome) = match resolved {
            Ok(imported) => (
                imported,
                self.glob_shadows_external(use_).map_or(Ok(()), Err),
            ),
            Err(err) => (Imported::none(db, use_), Err(err)),
        };
        let changed = imported != own;
        self.replace(use_, own);
        (changed.then_some(imported), outcome)
    }

    /// Replaces what `use_` imports, and returns what it imported before.
    fn replace(&mut self, use_: Use<'db>, imported: Imported<'db>) -> Imported<'db> {
        let scope = use_scope(self.db, use_);
        match imported {
            Imported::Glob(names) => {
                let old = &mut self.imports.glob_resolved[&scope].imported[&use_];
                Imported::Glob(mem::replace(old, names))
            }
            Imported::Bucket(bucket) => {
                let old = self.buckets.insert(use_, bucket).unwrap_or_default();
                if let Some(name) = use_.imported_name(self.db) {
                    self.merge_named(scope, name);
                }
                Imported::Bucket(old)
            }
        }
    }

    /// Merges the buckets of the named uses importing `name` into `scope`.
    fn merge_named(&mut self, scope: ScopeId<'db>, name: IdentId<'db>) {
        let mut merged = NameResBucket::default();
        for use_ in &self.named_uses[&(scope, name)] {
            if let Some(bucket) = self.buckets.get(use_) {
                merged.merge(bucket);
            }
        }
        let named = self.imports.named_resolved.entry(scope).or_default();
        if merged.bucket.is_empty() {
            named.remove(&name);
        } else {
            named.insert(name, merged);
        }
    }

    /// Resolves all segments of the path of `use_` but the last one, and
    /// returns the scope that the last segment is resolved in.
    fn resolve_base(
        &self,
        use_: Use<'db>,
        scope: ScopeId<'db>,
    ) -> Result<ScopeId<'db>, SegmentError<'db>> {
        let segments = use_path(self.db, use_)?;
        segments[..segments.len() - 1].iter().enumerate().try_fold(
            scope,
            |target, (idx, segment)| {
                let bucket = self.lookup(scope, target, idx, segment)?;
                let res = bucket
                    .pick(NameDomain::TYPE)
                    .as_ref()
                    .unwrap_or_else(|_| bucket.iter_ok().next().unwrap());
                if res.is_mod(self.db) || res.is_enum(self.db) {
                    Ok(res.scope().unwrap())
                } else {
                    Err((idx, NameResolutionError::InvalidPathSegment(res.clone())))
                }
            },
        )
    }

    /// Looks up the `idx`th segment of the path of a use in `use_scope`, in
    /// `scope`. Only the first segment is looked up in lexical parents and
    /// external ingots.
    fn lookup(
        &self,
        use_scope: ScopeId<'db>,
        scope: ScopeId<'db>,
        idx: usize,
        segment: &Partial<UsePathSegment<'db>>,
    ) -> Result<NameResBucket<'db>, SegmentError<'db>> {
        let db = self.db;
        let name = segment
            .to_opt()
            .and_then(UsePathSegment::ident)
            .ok_or((idx, NameResolutionError::Invalid))?;
        let directive = if idx == 0 {
            QueryDirective::new()
        } else {
            QueryDirective::new().disallow_lex().disallow_external()
        };
        let query = EarlyNameQueryId::new(db, name, scope, directive);
        let mut bucket = NameResolver::new(db, self).resolve_query(query);

        let mut invisible = None;
        bucket.bucket.retain(|(_, res)| match res {
            Ok(res) if !res.is_importable() => false,
            Ok(res) if !res.is_visible(db, use_scope) => {
                if invisible.is_none() {
                    invisible = res.derived_from(db);
                }
                false
            }
            _ => true,
        });
        if let Some((_, err)) = bucket.errors().find(|(_, err)| {
            !matches!(
                err,
                NameResolutionError::NotFound | NameResolutionError::Invalid
            )
        }) {
            return Err((idx, err.clone()));
        }
        if bucket.is_empty() {
            let err = invisible.map_or(NameResolutionError::NotFound, |span| {
                NameResolutionError::Invisible(Some(span))
            });
            return Err((idx, err));
        }
        Ok(bucket)
    }

    /// A name that a glob imports shadows an external ingot or a builtin type
    /// of the same name, so the first segment of a use path that resolves to
    /// it is ambiguous.
    fn glob_shadows_external(&self, use_: Use<'db>) -> Option<SegmentError<'db>> {
        let db = self.db;
        let scope = use_scope(db, use_);
        let segment = use_path(db, use_).ok()?.first()?;
        let name = segment.to_opt()?.ident()?;
        let bucket = self.lookup(scope, scope, 0, segment).ok()?;
        let res = bucket.pick(NameDomain::TYPE).as_ref().ok()?;
        let shadows_external = self
            .ingot
            .resolved_external_ingots(db)
            .iter()
            .any(|(ingot_name, _)| *ingot_name == name)
            || PrimTy::from_name(db, name).is_some();
        (matches!(res.derivation.origin(), NameDerivation::GlobImported(_))
            && !res.is_external(db, self.ingot)
            && shadows_external)
            .then(|| (0, NameResolutionError::Ambiguous(ThinVec::new())))
    }

    /// Returns a diagnostic for each name that several named uses import into
    /// the same scope and domain.
    fn conflicts(&self) -> impl Iterator<Item = ImportDiag<'db>> + '_ {
        self.named_uses
            .iter()
            .filter(|(_, uses)| uses.len() > 1)
            .filter_map(|((scope, name), _)| {
                let bucket = self.imports.named_resolved.get(scope)?.get(name)?;
                let mut spans = ThinVec::new();
                for (_, err) in bucket.errors() {
                    let NameResolutionError::Ambiguous(cands) = err else {
                        continue;
                    };
                    for span in cands.iter().filter_map(|cand| cand.derived_from(self.db)) {
                        if !spans.contains(&span) {
                            spans.push(span);
                        }
                    }
                }
                (!spans.is_empty()).then(|| ImportDiag::Conflict(*name, spans))
            })
    }

    fn diag(
        &self,
        use_: Use<'db>,
        idx: usize,
        err: NameResolutionError<'db>,
    ) -> Option<ImportDiag<'db>> {
        let db = self.db;
        let span = use_.span().path().segment(idx).into();
        let ident = use_path(db, use_).ok()?.get(idx)?.to_opt()?.ident()?;
        Some(match err {
            NameResolutionError::NotFound => ImportDiag::NotFound(span, ident),
            // The parser has already reported an invalid segment.
            NameResolutionError::Invalid => return None,
            NameResolutionError::Ambiguous(cands) => ImportDiag::ambiguous(db, span, ident, cands),
            NameResolutionError::InvalidPathSegment(res) => {
                ImportDiag::InvalidPathSegment(span, ident, res.kind.name_span(db))
            }
            NameResolutionError::Invisible(def_span) => {
                ImportDiag::Invisible(span, ident, def_span)
            }
            NameResolutionError::Conflict(ident, spans) => ImportDiag::Conflict(ident, spans),
        })
    }
}

/// The scope that `use_` imports into.
fn use_scope<'db>(db: &'db dyn HirAnalysisDb, use_: Use<'db>) -> ScopeId<'db> {
    ScopeId::from_item(use_.into()).lex_parent(db).unwrap()
}

/// The segments of the path of `use_`, of which there is at least one.
fn use_path<'db>(
    db: &'db dyn HirAnalysisDb,
    use_: Use<'db>,
) -> Result<&'db [Partial<UsePathSegment<'db>>], SegmentError<'db>> {
    use_.path(db)
        .to_opt()
        .map(|path| path.data(db).as_slice())
        .filter(|segments| !segments.is_empty())
        .ok_or((0, NameResolutionError::Invalid))
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Update)]
pub struct ResolvedImports<'db> {
    pub named_resolved: IndexMap<ScopeId<'db>, NamedImportSet<'db>>,
    pub glob_resolved: IndexMap<ScopeId<'db>, GlobImportSet<'db>>,
    pub unnamed_resolved: IndexMap<ScopeId<'db>, Vec<NameResBucket<'db>>>,
}

pub(super) trait Importer<'db> {
    /// Returns the resolved imports of the ingot containing `scope`.
    fn imports<'a>(
        &'a self,
        db: &'db dyn HirAnalysisDb,
        scope: ScopeId<'db>,
    ) -> &'a ResolvedImports<'db>;

    fn named_imports<'a>(
        &'a self,
        db: &'db dyn HirAnalysisDb,
        scope: ScopeId<'db>,
    ) -> Option<&'a NamedImportSet<'db>> {
        self.imports(db, scope).named_resolved.get(&scope)
    }

    fn glob_imports<'a>(
        &'a self,
        db: &'db dyn HirAnalysisDb,
        scope: ScopeId<'db>,
    ) -> Option<&'a GlobImportSet<'db>> {
        self.imports(db, scope).glob_resolved.get(&scope)
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub(super) struct DefaultImporter;

impl<'db> Importer<'db> for DefaultImporter {
    fn imports<'a>(
        &'a self,
        db: &'db dyn HirAnalysisDb,
        scope: ScopeId<'db>,
    ) -> &'a ResolvedImports<'db> {
        &super::resolve_imports(db, scope.ingot(db)).1
    }
}

impl<'db> Importer<'db> for ImportResolver<'db> {
    fn imports<'a>(
        &'a self,
        db: &'db dyn HirAnalysisDb,
        scope: ScopeId<'db>,
    ) -> &'a ResolvedImports<'db> {
        if scope.ingot(db) == self.ingot {
            self.reads.borrow_mut().insert(scope);
            &self.imports
        } else {
            DefaultImporter.imports(db, scope)
        }
    }
}

pub type NamedImportSet<'db> = FxHashMap<IdentId<'db>, NameResBucket<'db>>;

#[derive(Clone, Debug, Default, PartialEq, Eq, Update)]
pub struct GlobImportSet<'db> {
    imported: IndexMap<Use<'db>, GlobNames<'db>>,
}
impl<'db> GlobImportSet<'db> {
    /// Returns imported resolutions for the given `name`.
    pub fn name_res_for(&self, name: IdentId<'db>) -> impl Iterator<Item = &NameRes<'db>> {
        self.imported
            .values()
            .flat_map(move |v| v.get(&name).into_iter().flatten())
    }

    pub fn iter(&self) -> impl Iterator<Item = (&Use<'db>, &GlobNames<'db>)> {
        self.imported.iter()
    }
}

impl NameRes<'_> {
    /// Returns true if the resolution is a builtin or is in another ingot.
    fn is_external(&self, db: &dyn HirAnalysisDb, ingot: Ingot) -> bool {
        match self.kind {
            NameResKind::Scope(scope) => scope.ingot(db) != ingot,
            NameResKind::Prim(_) => true,
        }
    }
}
