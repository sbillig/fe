A contract field whose type names an associated type through a generic argument or tuple element, such as `mut g: G<<Nar as Tr>::Out>`, no longer fails to build with `ContractFieldTypeMismatch`.
