//! Declaration-owned generic defaults and their application-site instantiation.

use crate::{
    analysis::{
        HirAnalysisDb,
        name_resolution::{EarlyNameQueryId, NameResKind, QueryDirective, resolve_query},
    },
    hir_def::{
        ConstGenericArgValue, GenericParam, GenericParamOwner, Partial, PathId,
        scope_graph::ScopeId,
    },
    span::path::LazyPathSpan,
    visitor::{Visitor, VisitorCtxt, walk_path},
};

/// Parameter dependencies are a property of the source, not of successful type
/// lowering. In particular, an unresolved `T::Item` still refers to `T`.
#[salsa::tracked(return_ref)]
pub(crate) fn default_dependencies<'db>(
    db: &'db dyn HirAnalysisDb,
    owner: GenericParamOwner<'db>,
    param_idx: usize,
) -> Vec<usize> {
    let view = owner.param_view(db, param_idx);
    let mut collector = DefaultDependencies {
        db,
        owner,
        indices: Vec::new(),
    };
    match view.param {
        GenericParam::Type(param) => {
            if let Some(ty) = param.default_ty {
                let mut ctxt = VisitorCtxt::new(
                    db,
                    owner.scope(),
                    view.span().into_type_param().default_ty(),
                );
                collector.visit_ty(&mut ctxt, ty);
            }
        }
        GenericParam::Const(param) => {
            if let Some(ConstGenericArgValue::Expr(Partial::Present(body))) = param.default {
                collector.visit_body(&mut VisitorCtxt::with_body(db, body), body);
            }
        }
    }
    collector.indices.sort_unstable();
    collector.indices.dedup();
    collector.indices
}

struct DefaultDependencies<'db> {
    db: &'db dyn HirAnalysisDb,
    owner: GenericParamOwner<'db>,
    indices: Vec<usize>,
}

impl<'db> Visitor<'db> for DefaultDependencies<'db> {
    fn visit_path(&mut self, ctxt: &mut VisitorCtxt<'db, LazyPathSpan<'db>>, path: PathId<'db>) {
        if let Some(name) = path.root_ident(self.db) {
            let query = EarlyNameQueryId::new(self.db, name, ctxt.scope(), QueryDirective::new());
            for res in resolve_query(self.db, query).iter_ok() {
                if let NameResKind::Scope(ScopeId::GenericParam(item, idx)) = res.kind
                    && GenericParamOwner::from_item_opt(item) == Some(self.owner)
                {
                    self.indices.push(usize::from(idx));
                }
            }
        }
        // Includes qualified receivers and generic arguments on every segment,
        // as well as const bodies nested inside a type.
        walk_path(self, ctxt, path);
    }
}
