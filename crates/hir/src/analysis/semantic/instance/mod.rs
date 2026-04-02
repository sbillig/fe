mod const_ref;
mod semantic;
mod template;

pub(crate) use const_ref::{resolve_semantic_const_ref, semantic_callee_key};
pub use semantic::{
    SemanticInstance, SemanticInstanceKey, get_or_build_semantic_instance,
    semantic_may_return_normally,
};
pub use template::{
    GenericSubst, ImplEnv, TypedBodyTemplate, instantiate_typed_body,
    instantiate_with_generic_args, typed_body_template,
};
