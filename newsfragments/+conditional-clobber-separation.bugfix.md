Preserve caller-checkable separation conditions when a possible raw overwrite
makes an input pointer appear to alias fresh local storage. Independent native
buffers can now be read after journal growth through forwarding and recursive
calls, while callers with concrete aliases remain rejected.
