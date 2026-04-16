use std::{cell::Cell, rc::Rc};

use super::legalize::{YLocalId, YulFunctionPlan, YulLocal, YulLocalRoot, YulValueClass};

#[derive(Clone)]
pub(super) struct FunctionState {
    value_names: Vec<Option<String>>,
    declared_values: Vec<bool>,
    root_names: Vec<Option<String>>,
    declared_roots: Vec<bool>,
    next_temp: Rc<Cell<usize>>,
}

impl FunctionState {
    pub(super) fn new<'db>(func: &YulFunctionPlan<'db>) -> Self {
        let mut value_names = vec![None; func.locals.len()];
        let declared_values = vec![false; func.locals.len()];
        let mut root_names = vec![None; func.locals.len()];
        let declared_roots = vec![false; func.locals.len()];
        for (idx, local) in func.locals.iter().enumerate() {
            let local_id = YLocalId(idx as u32);
            if matches!(local.root, YulLocalRoot::MemorySlot { .. }) {
                root_names[idx] = Some(format!("r{idx}"));
            }
            if local_uses_value_name(local) && !func.param_locals.contains(&local_id) {
                value_names[idx] = Some(format!("v{idx}"));
            }
        }
        Self {
            value_names,
            declared_values,
            root_names,
            declared_roots,
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

pub(super) fn local_uses_value_name(local: &YulLocal<'_>) -> bool {
    local.class.is_some()
        && (!matches!(local.root, YulLocalRoot::MemorySlot { .. })
            || matches!(local.class, Some(YulValueClass::Word(_))))
}
