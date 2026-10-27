A recv arm may return a zero-sized type, such as a field-less struct with its own `Encode` implementation, instead of crashing the compiler.
