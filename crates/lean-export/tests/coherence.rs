//! Instance coherence: the normalized body of each function instance the
//! fixtures' contracts reach, against the normalized body of the generic
//! function with the instance's types substituted. Normalization makes some
//! choices by type, which a difference shows.
//!
//! One kind of difference is expected: a drop that is opaque in the generic
//! body, of a value whose type has a type parameter, and that the instance
//! destroys by its type: it runs a `Drop` impl's code, or, for a type without
//! drop glue, is not there.

mod load;

use std::{collections::BTreeSet, fmt::Write};

use common::ingot::IngotKind;
use cranelift_entity::EntityRef;
use hir::{
    analysis::{
        HirAnalysisDb,
        semantic::{
            NBlock, NDropTarget, NEffectArg, NEffectArgValue, NExpr, NPlace, NStatementKind,
            NTerminatorKind, NValueDefinition, NValueId, NormalizedBody, SConst, SemanticInstance,
            SemanticInstanceKey, identity_semantic_instance_key, normalize_semantic_body,
        },
        ty::{
            binder::Binder,
            const_ty::{ConstCanonEnv, ConstCanonMode, canonicalize_ty_for_mode},
            drop_glue::ty_has_drop_glue,
            fold::TyFoldable,
            ty_check::BodyOwner,
            ty_def::TyId,
            ty_is_copy,
        },
    },
    hir_def::{Contract, GenericParamOwner, ItemKind},
};
use load::{FIXTURES, compare, with_fixture};

/// How an instance's body differs from its generic body with its types.
#[derive(Default)]
struct Differences {
    /// The drops opaque in the generic body.
    expected: Vec<String>,
    /// The parameters whose places lie in another space.
    spaces: Vec<String>,
    /// The first other difference.
    unexpected: Option<String>,
}

/// The comparison of a generic body, `generic`, with an instance's body,
/// `instance`, under the instance's substitution `args` of `owner`'s
/// parameters.
struct Compare<'a, 'db> {
    db: &'db dyn HirAnalysisDb,
    owner: GenericParamOwner<'db>,
    args: &'db [TyId<'db>],
    instance: SemanticInstance<'db>,
    generic: &'a NormalizedBody<'db>,
    body: &'a NormalizedBody<'db>,
}

impl<'db> Compare<'_, 'db> {
    /// `ty` of the generic body with the instance's types, normalized.
    fn subst<T: TyFoldable<'db>>(&self, value: T) -> T {
        Binder::bind(self.owner, value).instantiate(self.db, self.args)
    }

    /// `ty` of the instance, normalized and with its constants in the form
    /// identity compares.
    fn canonical(&self, ty: TyId<'db>) -> TyId<'db> {
        let db = self.db;
        let env = ConstCanonEnv::new(
            self.instance.key(db).owner(db).scope(),
            self.instance.assumptions(db),
            None,
        );
        let ty = self.instance.normalized_ty(db, ty);
        canonicalize_ty_for_mode(db, ty, env, ConstCanonMode::Identity)
    }

    fn ty(&self, generic: TyId<'db>, ty: TyId<'db>) -> Result<(), String> {
        let (substituted, ty) = (self.canonical(self.subst(generic)), self.canonical(ty));
        if substituted == ty {
            return Ok(());
        }
        Err(format!(
            "type {} where the instance has {}",
            show(self.db, substituted),
            show(self.db, ty)
        ))
    }

    fn eq<T: PartialEq + std::fmt::Debug>(what: &str, generic: T, ty: T) -> Result<(), String> {
        if generic == ty {
            return Ok(());
        }
        Err(format!("{what} {generic:?} where the instance has {ty:?}"))
    }

    /// The callees agree: the same function with the same arguments, or a
    /// trait's method in the generic body that the instance resolves to the
    /// method of the same name of an impl of the trait.
    fn callee(
        &self,
        generic: SemanticInstanceKey<'db>,
        key: SemanticInstanceKey<'db>,
    ) -> Result<(), String> {
        let db = self.db;
        let substituted = self.subst(generic);
        if substituted.owner(db) == key.owner(db) {
            let args = |key: SemanticInstanceKey<'db>| -> Vec<_> {
                let args = key.subst(db).generic_args(db);
                args.iter().map(|ty| self.canonical(*ty)).collect()
            };
            if args(substituted) == args(key) {
                return Ok(());
            }
        }
        if let (BodyOwner::Func(method, _), BodyOwner::Func(resolved, _)) =
            (generic.owner(db), key.owner(db))
            && let Some(def) = method.containing_trait(db)
            && let Some(ItemKind::ImplTrait(item)) = resolved.scope().parent_item(db)
            && item.trait_def(db) == Some(def)
            && method.name(db) == resolved.name(db)
        {
            return Ok(());
        }
        Err(format!(
            "callee {} where the instance has {}",
            describe(db, substituted),
            describe(db, key)
        ))
    }

    fn place(&self, generic: &NPlace<'db>, place: &NPlace<'db>) -> Result<(), String> {
        Self::eq("place base", generic.base, place.base)?;
        Self::eq("place path", &generic.path, &place.path)?;
        self.ty(generic.ty, place.ty)
    }

    fn effect_arg(&self, generic: &NEffectArg<'db>, arg: &NEffectArg<'db>) -> Result<(), String> {
        Self::eq("effect", generic.binding_idx, arg.binding_idx)?;
        Self::eq("effect pass mode", generic.pass_mode, arg.pass_mode)?;
        Self::eq("effect `mut`", generic.required_mut, arg.required_mut)?;
        Self::eq("effect space", generic.provider, arg.provider)?;
        match (generic.provider_target_ty, arg.provider_target_ty) {
            (Some(generic), Some(ty)) => self.ty(generic, ty)?,
            (generic, ty) => Self::eq("effect target", generic.is_some(), ty.is_some())?,
        }
        match (&generic.arg, &arg.arg) {
            (NEffectArgValue::Place(generic), NEffectArgValue::Place(place)) => {
                self.place(generic, place)
            }
            (NEffectArgValue::Value(generic), NEffectArgValue::Value(value)) => {
                Self::eq("effect operand", generic, value)
            }
            _ => Err("effect argument of another kind".to_string()),
        }
    }

    fn constant(&self, generic: &SConst<'db>, constant: &SConst<'db>) -> Result<(), String> {
        match (generic, constant) {
            (SConst::Ref(generic), SConst::Ref(constant)) => {
                self.ty(generic.ty(self.db), constant.ty(self.db))?;
                self.callee(generic.instance(self.db), constant.instance(self.db))
            }
            // A constant the generic body describes or takes from its
            // evidence, such as an associated constant of a type parameter
            // or a constant parameter, which the instance resolves.
            (SConst::Description(_) | SConst::Evidence(_), SConst::Ref(_) | SConst::Value(_)) => {
                Ok(())
            }
            (generic, constant) => Self::eq("constant", generic, constant),
        }
    }

    fn expr(&self, generic: &NExpr<'db>, expr: &NExpr<'db>) -> Result<(), String> {
        match (generic, expr) {
            (NExpr::Load { place: g, mode: gm }, NExpr::Load { place, mode }) => {
                Self::eq("read mode", gm, mode)?;
                self.place(g, place)
            }
            (
                NExpr::Borrow {
                    place: g,
                    kind: gk,
                    provider: gp,
                },
                NExpr::Borrow {
                    place,
                    kind,
                    provider,
                },
            ) => {
                Self::eq("borrow", (gk, gp), (kind, provider))?;
                self.place(g, place)
            }
            (NExpr::MakeView { place: g, .. }, NExpr::MakeView { place, .. }) => {
                self.place(g, place)
            }
            (NExpr::Const(g), NExpr::Const(constant)) => self.constant(g, constant),
            (NExpr::PointerCast { value: g, to: gt }, NExpr::PointerCast { value, to })
            | (NExpr::ScalarCast { value: g, to: gt }, NExpr::ScalarCast { value, to })
            | (NExpr::ArrayRepeat { value: g, ty: gt }, NExpr::ArrayRepeat { value, ty: to }) => {
                Self::eq("operand", g, value)?;
                self.ty(*gt, *to)
            }
            (NExpr::AggregateMake { ty: gt, fields: g }, NExpr::AggregateMake { ty, fields }) => {
                Self::eq("fields", g, fields)?;
                self.ty(*gt, *ty)
            }
            (
                NExpr::EnumMake {
                    enum_ty: gt,
                    variant: gv,
                    fields: g,
                },
                NExpr::EnumMake {
                    enum_ty,
                    variant,
                    fields,
                },
            ) => {
                Self::eq("variant", (gv, g), (variant, fields))?;
                self.ty(*gt, *enum_ty)
            }
            (
                NExpr::Call {
                    callee: gc,
                    args: g,
                    effect_args: ge,
                    ..
                },
                NExpr::Call {
                    callee,
                    args,
                    effect_args,
                    ..
                },
            ) => {
                Self::eq("arguments", g, args)?;
                Self::eq("effect arguments", ge.len(), effect_args.len())?;
                for (g, arg) in ge.iter().zip(effect_args) {
                    self.effect_arg(g, arg)?;
                }
                self.callee(gc.key, callee.key)
            }
            // The other expressions hold no types.
            (generic, expr) => Self::eq("expression", generic, expr),
        }
    }

    /// The drops of `block` of each body, by the place among its other
    /// statements they are at, with their targets' types.
    fn drops<'b>(
        body: &'b NormalizedBody<'db>,
        block: &'b NBlock<'db>,
    ) -> Vec<(usize, &'b NDropTarget<'db>, TyId<'db>)> {
        let mut at = 0;
        let mut drops = Vec::new();
        for statement in &block.statements {
            match &statement.kind {
                NStatementKind::Drop { target } => {
                    let ty = match target {
                        NDropTarget::Value(value) => body.values[value.index()].ty,
                        NDropTarget::Place(place) => place.ty,
                    };
                    drops.push((at, target, ty));
                }
                _ => at += 1,
            }
        }
        drops
    }

    /// The differences of the bodies: the expected ones, those in the space
    /// of a parameter's place, and the first other.
    fn bodies(&self) -> Differences {
        let mut differences = Differences::default();
        differences.unexpected = self.blocks(&mut differences).err();
        differences
    }

    fn blocks(&self, differences: &mut Differences) -> Result<(), String> {
        let (generic, body) = (self.generic, self.body);
        // The first value that differs, by how it is defined, then the
        // number of values.
        for (index, (g, value)) in generic.values.iter().zip(&body.values).enumerate() {
            let defined = |body: &NormalizedBody<'db>| {
                let id = NValueId::new(index);
                body.defining_expr(id).map(|(_, expr)| expr_kind(expr))
            };
            Self::eq(
                &format!("v{index}"),
                (definition(generic, g.definition), defined(generic)),
                (definition(body, value.definition), defined(body)),
            )?;
            Self::eq(
                &format!("v{index} mutability"),
                g.mutability,
                value.mutability,
            )?;
            self.ty(g.ty, value.ty)
                .map_err(|error| format!("v{index}: {error}"))?;
        }
        Self::eq("values", generic.values.len(), body.values.len())?;
        Self::eq("roots", generic.roots.len(), body.roots.len())?;
        for (index, (g, root)) in generic.roots.iter().zip(&body.roots).enumerate() {
            // An instance takes a place parameter in the space its caller
            // has it in, which is not one of its types.
            if g.address_space != root.address_space {
                differences.spaces.push(format!(
                    "root {index} in {} where the generic body has {}",
                    root.address_space.pretty(),
                    g.address_space.pretty()
                ));
            }
            Self::eq(
                &format!("root {index}"),
                std::mem::discriminant(&g.kind),
                std::mem::discriminant(&root.kind),
            )?;
            self.ty(g.ty, root.ty)
                .map_err(|error| format!("root {index}: {error}"))?;
        }
        Self::eq("blocks", generic.blocks.len(), body.blocks.len())?;
        Self::eq("entry", generic.entry, body.entry)?;
        let db = self.db;
        let scope = self.instance.key(db).owner(db).scope();
        let destroys = |ty| {
            ty_has_drop_glue(db, ty) && !ty_is_copy(db, scope, ty, self.instance.assumptions(db))
        };
        for (index, (g, block)) in generic.blocks.iter().zip(&body.blocks).enumerate() {
            let at = |error| format!("block {index}: {error}");
            Self::eq("parameters", &g.params, &block.params).map_err(at)?;
            let statements = |block: &'_ NBlock<'db>| {
                let kinds = block.statements.iter().map(|statement| &statement.kind);
                kinds
                    .filter(|kind| !matches!(kind, NStatementKind::Drop { .. }))
                    .cloned()
                    .collect::<Vec<_>>()
            };
            let (gs, statements) = (statements(g), statements(block));
            Self::eq("statements", gs.len(), statements.len()).map_err(at)?;
            for (n, (g, statement)) in gs.iter().zip(&statements).enumerate() {
                let at = |error| format!("block {index} statement {n}: {error}");
                match (g, statement) {
                    (
                        NStatementKind::Define {
                            result: gr,
                            expr: g,
                        },
                        NStatementKind::Define { result, expr },
                    ) => {
                        Self::eq("result", gr, result).map_err(at)?;
                        self.expr(g, expr).map_err(at)?;
                    }
                    (
                        NStatementKind::Store {
                            destination: g,
                            value: gv,
                        },
                        NStatementKind::Store { destination, value },
                    ) => {
                        Self::eq("stored", gv, value).map_err(at)?;
                        self.place(g, destination).map_err(at)?;
                    }
                    (
                        NStatementKind::CheckIndex {
                            index: gi,
                            array: g,
                        },
                        NStatementKind::CheckIndex { index, array },
                    ) => {
                        Self::eq("index", gi, index).map_err(at)?;
                        self.ty(*g, *array).map_err(at)?;
                    }
                    (g, statement) => Self::eq("statement", g, statement).map_err(at)?,
                }
            }
            let mut drops = Self::drops(body, block);
            for (position, g, ty) in Self::drops(generic, g) {
                let substituted = self.instance.normalized_ty(db, self.subst(ty));
                let found = drops.iter().position(|(at, target, _)| {
                    *at == position
                        && match (g, target) {
                            (NDropTarget::Value(g), NDropTarget::Value(value)) => g == value,
                            (NDropTarget::Place(g), NDropTarget::Place(place)) => {
                                self.place(g, place).is_ok()
                            }
                            _ => false,
                        }
                });
                let opaque = ty.has_param(db);
                let shown = substituted.pretty_print(db);
                match found {
                    Some(found) => {
                        drops.remove(found);
                        if opaque && destroys(substituted) {
                            differences
                                .expected
                                .push(format!("drop of {shown} runs its type's code"));
                        }
                    }
                    None if opaque && !destroys(substituted) => {
                        differences.expected.push(format!(
                            "drop of {shown}, which destroys nothing, is not there"
                        ));
                    }
                    None => return Err(at(format!("drop of {shown} is not there"))),
                }
            }
            if let Some((_, _, ty)) = drops.first() {
                return Err(at(format!(
                    "drop of {} only in the instance",
                    ty.pretty_print(db)
                )));
            }
            let terminator = (&g.terminator.kind, &block.terminator.kind);
            match terminator {
                (
                    NTerminatorKind::MatchEnum {
                        value: gv,
                        enum_ty: g,
                        cases: gc,
                        default: gd,
                    },
                    NTerminatorKind::MatchEnum {
                        value,
                        enum_ty,
                        cases,
                        default,
                    },
                ) => {
                    Self::eq("match", (gv, gc, gd), (value, cases, default)).map_err(at)?;
                    self.ty(*g, *enum_ty).map_err(at)?;
                }
                (g, terminator) => Self::eq("terminator", g, terminator).map_err(at)?,
            }
        }
        Ok(())
    }
}

/// `definition` with the place of a statement among the block's statements
/// other than drops.
fn definition(body: &NormalizedBody<'_>, definition: NValueDefinition) -> NValueDefinition {
    match definition {
        NValueDefinition::Statement { block, statement } => {
            let before = &body.blocks[block.index()].statements[..statement as usize];
            let drops = before
                .iter()
                .filter(|statement| matches!(statement.kind, NStatementKind::Drop { .. }))
                .count();
            NValueDefinition::Statement {
                block,
                statement: statement - drops as u32,
            }
        }
        definition => definition,
    }
}

/// The kind of `expr`, without its operands.
fn expr_kind(expr: &NExpr<'_>) -> String {
    let debug = format!("{expr:?}");
    debug[..debug.find([' ', '(', '{']).unwrap_or(debug.len())].to_string()
}

/// `ty` as Fe writes it, with the capability it is.
fn show(db: &dyn HirAnalysisDb, ty: TyId<'_>) -> String {
    match ty.as_capability(db) {
        Some((kind, inner)) => format!("{kind:?} {}", inner.pretty_print(db)),
        None => ty.pretty_print(db).clone(),
    }
}

/// `key`'s function and type arguments, as Fe writes them.
fn describe(db: &dyn HirAnalysisDb, key: SemanticInstanceKey<'_>) -> String {
    let name = match key.owner(db) {
        BodyOwner::Func(func, _) => {
            let parent = func.scope().parent_item(db).and_then(|item| match item {
                ItemKind::Impl(item) => Some(item.ty(db).pretty_print(db).clone()),
                ItemKind::ImplTrait(item) => Some(item.ty(db).pretty_print(db).clone()),
                _ => None,
            });
            let name = func.name(db).to_opt().map(|name| name.data(db).clone());
            match parent {
                Some(parent) => format!("{parent}::{}", name.unwrap_or_default()),
                None => {
                    let path = func.scope().pretty_path(db).or(name).unwrap_or_default();
                    // The compiler calls the root module of core and std `lib`.
                    match func.top_mod(db).ingot(db).kind(db) {
                        IngotKind::Core => path.replacen("lib::", "core::", 1),
                        IngotKind::Std => path.replacen("lib::", "std::", 1),
                        _ => path,
                    }
                }
            }
        }
        owner => format!("{owner:?}"),
    };
    let args: Vec<_> = key
        .subst(db)
        .generic_args(db)
        .iter()
        .map(|ty| ty.pretty_print(db).clone())
        .collect();
    format!("{name}<{}>", args.join(", "))
}

/// The function instances the entries of `contract` call, by the calls and
/// constants of their bodies, each with its normalized body.
fn instances<'db>(
    db: &'db dyn HirAnalysisDb,
    contract: Contract<'db>,
) -> Vec<SemanticInstanceKey<'db>> {
    let mut pending = vec![identity_semantic_instance_key(
        db,
        BodyOwner::ContractInit { contract },
    )];
    for (recv_idx, recv) in contract.recv_views(db).enumerate() {
        for (arm_idx, _) in recv.arms(db).enumerate() {
            pending.push(identity_semantic_instance_key(
                db,
                BodyOwner::ContractRecvArm {
                    contract,
                    recv_idx: recv_idx as u32,
                    arm_idx: arm_idx as u32,
                },
            ));
        }
    }
    let mut seen = Vec::new();
    while let Some(key) = pending.pop() {
        if seen.contains(&key) {
            continue;
        }
        seen.push(key);
        let Ok(normalized) = normalize_semantic_body(db, SemanticInstance::new(db, key)) else {
            continue;
        };
        for statement in normalized
            .body
            .blocks
            .iter()
            .flat_map(|block| &block.statements)
        {
            match &statement.kind {
                NStatementKind::Define {
                    expr: NExpr::Call { callee, .. },
                    ..
                } => pending.push(callee.key),
                NStatementKind::Define {
                    expr: NExpr::Const(SConst::Ref(constant)),
                    ..
                } => pending.push(constant.instance(db)),
                _ => {}
            }
        }
    }
    seen
}

/// Each instance of a generic function the fixtures reach, against its
/// generic body with the instance's types: the expected differences, the
/// spaces of parameters, and the first other difference of each instance,
/// which `golden/coherence.txt` records.
#[test]
fn coherence() {
    let mut sections: [BTreeSet<String>; 3] = Default::default();
    let mut compared = 0;
    for (name, _) in FIXTURES {
        with_fixture(name, |db, contract| {
            for key in instances(db, contract) {
                let BodyOwner::Func(func, _) = key.owner(db) else {
                    continue;
                };
                let args = key.subst(db).generic_args(db);
                let generic = identity_semantic_instance_key(db, key.owner(db));
                if args.is_empty() || generic == key {
                    continue;
                }
                let instance = SemanticInstance::new(db, key);
                let shown = describe(db, key);
                let (Ok(body), Ok(generic)) = (
                    normalize_semantic_body(db, instance),
                    normalize_semantic_body(db, SemanticInstance::new(db, generic)),
                ) else {
                    sections[2].insert(format!("{shown}: a body does not normalize"));
                    continue;
                };
                compared += 1;
                let compare = Compare {
                    db,
                    owner: func.into(),
                    args,
                    instance,
                    generic: &generic.body,
                    body: &body.body,
                };
                let differences = compare.bodies();
                let found = [
                    differences.expected,
                    differences.spaces,
                    differences.unexpected.into_iter().collect(),
                ];
                for (section, found) in sections.iter_mut().zip(found) {
                    section.extend(found.into_iter().map(|found| format!("{shown}: {found}")));
                }
            }
        });
    }
    let mut report = format!("instances compared: {compared}\n");
    let titles = [
        "expected: drops opaque in the generic body",
        "places of parameters in another space",
        "unexpected: the first other difference",
    ];
    for (title, section) in titles.iter().zip(&sections) {
        writeln!(report, "\n{title}: {}", section.len()).unwrap();
        for line in section {
            writeln!(report, "  {line}").unwrap();
        }
    }
    compare("coherence.txt", &report);
}
