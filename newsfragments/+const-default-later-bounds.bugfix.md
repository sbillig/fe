A bound or `where` clause that names a const generic parameter with a default, such as `T: Has<1>` for `trait Has<const N: usize, const L: usize = { N }>`, no longer crashes the compiler.
