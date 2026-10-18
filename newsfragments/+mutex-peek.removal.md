`Mutex::peek` is removed. `let v = guard.value()` copies a `Copy` value out of the lock just as `peek` did, so it can be kept across an external call.
