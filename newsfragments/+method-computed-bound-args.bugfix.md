A method in an `impl` or trait whose bound has a computed const argument, such as `fn need<const N: usize, U: Has<{ N + 1 }>>()`, no longer crashes the compiler.
