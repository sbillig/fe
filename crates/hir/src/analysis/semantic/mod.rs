pub mod borrowck;
pub mod consts;
pub mod ctfe;
pub mod instance;
pub mod ir;
pub mod lower;
mod verify;

pub use borrowck::*;
pub use consts::*;
pub use ctfe::*;
pub use instance::{
    GenericSubst, ImplEnv, SemanticInstance, SemanticInstanceKey, TypedBodyTemplate,
    get_or_build_semantic_instance, instantiate_typed_body, instantiate_with_generic_args,
    semantic_binding_lowering, semantic_may_return_normally, typed_body_template,
};
pub use ir::*;
pub use lower::{
    effect_param_site, lower_to_smir, owner_effect_bindings,
    resolved_provider_binding_for_owner_effect, same_owner_effect_binding,
};
pub use verify::{SemanticVerifyError, verify_semantic_body};
