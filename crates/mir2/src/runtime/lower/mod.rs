pub mod body;
pub(crate) mod call;
pub(crate) mod classify;
pub(crate) mod coerce;
mod consts;
pub(crate) mod infer;
pub(crate) mod interface;
pub(crate) mod layout;
mod place;
pub(crate) mod returns;
pub(crate) mod type_info;
pub(crate) mod value_eval;

pub use body::lower_to_rmir;
pub use call::collect_runtime_calls;
