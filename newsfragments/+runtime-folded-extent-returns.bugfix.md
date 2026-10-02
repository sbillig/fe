Runtime generic functions that return a constant-foldable array whose length is not a bare const parameter, such as `[3; T::N]` or `[0; word_len(N)]`, no longer crash code generation.
