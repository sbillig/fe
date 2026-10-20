// Relinked only into the fallible-allocation regression executable. It refuses
// 4096-byte requests and counts the 1024-byte and 1-byte blocks the test owns;
// a zero-byte request reaches the host as one byte.
#define _DEFAULT_SOURCE
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>

static unsigned allocations;
static unsigned releases;
static unsigned failures;

typedef union {
    max_align_t alignment;
    struct { size_t extent; int watched; } info;
} Header;

void *malloc(size_t size) {
    if (size == 4096) {
        failures++;
        return NULL;
    }
    if (size > SIZE_MAX - sizeof(Header)) return NULL;
    size_t extent = sizeof(Header) + size;
    Header *header = mmap(NULL, extent, PROT_READ | PROT_WRITE,
                          MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (header == MAP_FAILED) return NULL;
    header->info.extent = extent;
    header->info.watched = size == 1024 || size == 1;
    allocations += header->info.watched;
    return header + 1;
}

void free(void *pointer) {
    if (pointer == NULL) return;
    Header *header = (Header *)pointer - 1;
    releases += header->info.watched;
    munmap(header, header->info.extent);
}

__attribute__((destructor)) static void check_releases(void) {
    if (allocations != 2 || releases != 2 || failures != 1) _Exit(101);
}
