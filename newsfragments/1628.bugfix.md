Fix `fe check` and `fe build` hanging forever on a loop that reassigns a raw pointer from an offset of itself, such as `p = ptr::offset_bytes(p, 32)`.
