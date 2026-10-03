// Interposed only for the typed-box O1 regression. Watched sizes distinguish
// explicit owner slots from native reference descriptors.
#define _DEFAULT_SOURCE
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
static unsigned allocations, releases, failures, units, unit_releases;
typedef union {
    max_align_t alignment;
    struct { size_t extent, size; } info;
} Header;
void *malloc(size_t size) {
    if (size == 4097) { failures++; return NULL; }
    if (size > SIZE_MAX - sizeof(Header)) return NULL;
    size_t extent = sizeof(Header) + size;
    Header *header = mmap(NULL, extent, PROT_READ | PROT_WRITE,
                          MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (header == MAP_FAILED) return NULL;
    header->info.extent = extent;
    header->info.size = size;
    allocations += size == 257;
    units += size == 1;
    memset(header + 1, 0xa5, size);
    return header + 1;
}
void free(void *pointer) {
    if (!pointer) return;
    Header *header = (Header *)pointer - 1;
    releases += header->info.size == 257;
    unit_releases += header->info.size == 1;
    munmap(header, header->info.extent);
}
__attribute__((destructor)) static void check_releases(void) {
    if (allocations != 3 || releases != 3 || units != 1 || unit_releases != 1 || failures != 1) _Exit(101);
}
