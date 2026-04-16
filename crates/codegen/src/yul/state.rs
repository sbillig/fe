use std::{cell::Cell, rc::Rc};

use super::legalize::{
    YExpr, YLocalId, YStmt, YulFunctionPlan, YulLocal, YulLocalRoot, YulPlace, YulPlaceRoot,
    YulValueClass,
};

#[derive(Clone)]
pub(super) struct FunctionState {
    value_names: Vec<Option<String>>,
    declared_values: Vec<bool>,
    root_names: Vec<Option<String>>,
    declared_roots: Vec<bool>,
    value_backed_locals: Vec<bool>,
    next_temp: Rc<Cell<usize>>,
}

impl FunctionState {
    pub(super) fn new<'db>(func: &YulFunctionPlan<'db>) -> Self {
        let value_backed_locals = compute_value_backed_locals(func);
        let mut value_names = vec![None; func.locals.len()];
        let declared_values = vec![false; func.locals.len()];
        let mut root_names = vec![None; func.locals.len()];
        let declared_roots = vec![false; func.locals.len()];
        for (idx, local) in func.locals.iter().enumerate() {
            let local_id = YLocalId(idx as u32);
            if matches!(local.root, YulLocalRoot::MemorySlot { .. }) {
                root_names[idx] = Some(format!("r{idx}"));
            }
            if value_backed_locals[idx] && !func.param_locals.contains(&local_id) {
                value_names[idx] = Some(format!("v{idx}"));
            }
        }
        Self {
            value_names,
            declared_values,
            root_names,
            declared_roots,
            value_backed_locals,
            next_temp: Rc::new(Cell::new(func.locals.len())),
        }
    }

    pub(super) fn assign_param_name(&mut self, local: YLocalId, name: String) {
        self.value_names[local.index()] = Some(name);
        self.declared_values[local.index()] = true;
    }

    pub(super) fn value_name(&self, local: YLocalId) -> Option<&str> {
        self.is_declared(local)
            .then(|| self.local_name(local))
            .flatten()
    }

    pub(super) fn local_name(&self, local: YLocalId) -> Option<&str> {
        self.value_names[local.index()].as_deref()
    }

    pub(super) fn uses_value_name(&self, local: YLocalId) -> bool {
        self.value_backed_locals[local.index()]
    }

    pub(super) fn is_declared(&self, local: YLocalId) -> bool {
        self.declared_values[local.index()]
    }

    pub(super) fn mark_declared(&mut self, local: YLocalId) {
        self.declared_values[local.index()] = true;
    }

    pub(super) fn root_name(&self, local: YLocalId) -> Option<&str> {
        self.root_names[local.index()].as_deref()
    }

    pub(super) fn is_root_declared(&self, local: YLocalId) -> bool {
        self.declared_roots[local.index()]
    }

    pub(super) fn mark_root_declared(&mut self, local: YLocalId) {
        self.declared_roots[local.index()] = true;
    }

    pub(super) fn alloc_temp(&mut self) -> String {
        let next = self.next_temp.get();
        let temp = format!("t{next}");
        self.next_temp.set(next + 1);
        temp
    }
}

fn local_supports_value_name(local: &YulLocal<'_>) -> bool {
    local.class.is_some()
        && (!matches!(local.root, YulLocalRoot::MemorySlot { .. })
            || matches!(local.class, Some(YulValueClass::Word(_))))
}

fn compute_value_backed_locals<'db>(func: &YulFunctionPlan<'db>) -> Vec<bool> {
    let mut value_backed = func
        .locals
        .iter()
        .map(local_supports_value_name)
        .collect::<Vec<_>>();
    let mut slot_word_roots = vec![false; func.locals.len()];
    for block in &func.blocks {
        for stmt in &block.stmts {
            mark_stmt_slot_word_roots(func, stmt, &mut slot_word_roots);
        }
    }
    for (idx, needs_root) in slot_word_roots.into_iter().enumerate() {
        if needs_root {
            value_backed[idx] = false;
        }
    }
    value_backed
}

fn mark_stmt_slot_word_roots<'db>(
    func: &YulFunctionPlan<'db>,
    stmt: &YStmt<'db>,
    slot_word_roots: &mut [bool],
) {
    match stmt {
        YStmt::Assign { expr, .. } => {
            mark_expr_slot_word_roots(func, expr, slot_word_roots);
        }
        YStmt::Store { dst, .. } | YStmt::CopyInto { dst, .. } => {
            if !is_direct_word_slot_place(func, dst) {
                mark_place_slot_word_root(func, dst, slot_word_roots);
            }
        }
        YStmt::Call { .. }
        | YStmt::Builtin(_)
        | YStmt::EnumAssertVariant { .. }
        | YStmt::EnumSetTag { .. }
        | YStmt::EnumWriteVariant { .. } => {}
    }
}

fn mark_expr_slot_word_roots<'db>(
    func: &YulFunctionPlan<'db>,
    expr: &YExpr<'db>,
    slot_word_roots: &mut [bool],
) {
    match expr {
        YExpr::MaterializePlaceToObject { place, .. } | YExpr::AddrOf { place } => {
            mark_place_slot_word_root(func, place, slot_word_roots);
        }
        YExpr::Load { place } => {
            if !is_direct_word_slot_place(func, place) {
                mark_place_slot_word_root(func, place, slot_word_roots);
            }
        }
        YExpr::Placeholder { .. }
        | YExpr::Use(_)
        | YExpr::ConstWord(_)
        | YExpr::Builtin(_)
        | YExpr::Unary { .. }
        | YExpr::Binary { .. }
        | YExpr::Cast { .. }
        | YExpr::ConstRef { .. }
        | YExpr::AllocObject { .. }
        | YExpr::MaterializeToObject { .. }
        | YExpr::ProviderFromRaw { .. }
        | YExpr::WordToRawAddr { .. }
        | YExpr::ProviderToRaw { .. }
        | YExpr::Call { .. }
        | YExpr::EnumTagOfValue { .. }
        | YExpr::EnumGetTag { .. }
        | YExpr::EnumAssertVariantRef { .. }
        | YExpr::EnumIsVariant { .. }
        | YExpr::EnumMake { .. }
        | YExpr::EnumExtract { .. } => {}
    }
}

fn mark_place_slot_word_root<'db>(
    func: &YulFunctionPlan<'db>,
    place: &YulPlace<'db>,
    slot_word_roots: &mut [bool],
) {
    let YulPlaceRoot::Slot(local) = place.root else {
        return;
    };
    if !matches!(
        func.locals[local.index()].class,
        Some(YulValueClass::Word(_))
    ) || !matches!(
        func.locals[local.index()].root,
        YulLocalRoot::MemorySlot { .. }
    ) {
        return;
    }
    slot_word_roots[local.index()] = true;
}

fn is_direct_word_slot_place<'db>(func: &YulFunctionPlan<'db>, place: &YulPlace<'db>) -> bool {
    let YulPlaceRoot::Slot(local) = place.root else {
        return false;
    };
    place.path.is_empty()
        && matches!(place.result_class, YulValueClass::Word(_))
        && matches!(
            func.locals[local.index()].class,
            Some(YulValueClass::Word(_))
        )
        && matches!(
            func.locals[local.index()].root,
            YulLocalRoot::MemorySlot { .. }
        )
}
