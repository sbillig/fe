`core::size_of<T>()` of a generic type can now be used in a constant that depends on `T`, such as an array length, and is computed once `T` is known.
