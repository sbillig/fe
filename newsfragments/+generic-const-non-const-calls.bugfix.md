A non-const function call in a constant that depends on generic parameters, such as an array length `[u8; f(N)]` or a generic impl constant, is now reported where it is written.
