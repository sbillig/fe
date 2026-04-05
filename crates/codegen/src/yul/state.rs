use crate::yul::legalize::{YLocalId, YulFunctionPlan, YulLocalRoot};

#[derive(Clone)]
pub(super) struct FunctionState {
    value_names: Vec<Option<String>>,
    root_names: Vec<Option<String>>,
    next_temp: usize,
}

impl FunctionState {
    pub(super) fn new<'db>(func: &YulFunctionPlan<'db>) -> Self {
        let mut value_names = vec![None; func.locals.len()];
        let mut root_names = vec![None; func.locals.len()];
        for (idx, local) in func.locals.iter().enumerate() {
            let local_id = YLocalId(idx as u32);
            if matches!(local.root, YulLocalRoot::MemorySlot { .. }) {
                root_names[idx] = Some(format!("r{idx}"));
            } else if local.class.is_some() && !func.param_locals.contains(&local_id) {
                value_names[idx] = Some(format!("v{idx}"));
            }
        }
        Self {
            value_names,
            root_names,
            next_temp: func.locals.len(),
        }
    }

    pub(super) fn assign_param_name(&mut self, local: YLocalId, name: String) {
        self.value_names[local.index()] = Some(name);
    }

    pub(super) fn value_name(&self, local: YLocalId) -> Option<&str> {
        self.value_names[local.index()].as_deref()
    }

    pub(super) fn root_name(&self, local: YLocalId) -> Option<&str> {
        self.root_names[local.index()].as_deref()
    }

    pub(super) fn alloc_temp(&mut self) -> String {
        let temp = format!("t{}", self.next_temp);
        self.next_temp += 1;
        temp
    }
}
