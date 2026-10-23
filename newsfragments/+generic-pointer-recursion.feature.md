A type may hold itself through a generic type that keeps its argument behind a pointer, as in `struct Tree { children: MemVec<Tree> }`. Such fields were rejected as recursive type definitions.
