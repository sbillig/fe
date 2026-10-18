`size_of` counts a contract value as its 32-byte address, as memory stores it, so memory arrays, slices and allocations of a type holding a contract no longer overlap their elements.
