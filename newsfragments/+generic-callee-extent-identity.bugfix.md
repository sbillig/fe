Computed array lengths that call a generic `const fn`, such as `[u8; len<N>()]`, now have the same type wherever they are written, including when the function has trait bounds.
