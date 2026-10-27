A bound may name a concrete type's associated type, such as `F: Fn(<Two as HasItem>::Item) -> u256`; such a bound used to fail with "trait bound is not satisfied" at its own use.
