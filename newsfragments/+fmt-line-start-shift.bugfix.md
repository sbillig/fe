`fe fmt` no longer breaks a long expression before `<<` or `<`, which turned valid code into code that fails to parse ("line-start `<<` is ambiguous"); the operator now ends the line.
