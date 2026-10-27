A qualified associated type can begin a value path, as in `<S as Tr>::Item::new()`; the function used to be looked up on the unresolved projection and not found.
