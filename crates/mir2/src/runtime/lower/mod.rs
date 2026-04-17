pub mod body;
pub(crate) mod call;
pub(crate) mod classify;
mod consts;
pub(crate) mod interface;
pub(crate) mod layout;
mod place;
pub(crate) mod type_info;

pub use body::lower_to_rmir;
pub use call::collect_runtime_calls;
