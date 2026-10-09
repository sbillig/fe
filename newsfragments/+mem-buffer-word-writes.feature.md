Add `MemBuffer.write_word` for checked, unaligned word writes without growing the allocation; writes extend the logical extent and zero any intervening gap.
