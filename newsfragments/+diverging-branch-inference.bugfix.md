A conditional or `match` with a diverging branch, such as `if c { return 1 } else { zero() }`, no longer gives its other branches the type `!`; their type is inferred from how the result is used.
