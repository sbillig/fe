#include <stdint.h>

// Compiled separately from Fe with LTO disabled. The compiler sees an external
// call with an unknown result; the C optimizer must keep this scalar opaque too.
uint64_t __fe_native_black_box_u64(uint64_t value) {
    __asm__ __volatile__("" : "+r"(value) : : "memory");
    return value;
}
