Reverts (including `assert!` with a message) now compile for native targets, where they trap. Previously they failed with an unsupported-terminator error.
