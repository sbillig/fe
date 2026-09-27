Comparison operators such as `==` and `<` no longer move a non-Copy right-hand operand; `a == b` now borrows `b` as `a.eq(b)` does.
