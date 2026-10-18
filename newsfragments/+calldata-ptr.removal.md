`RawOps::calldata_ptr` is removed. Cast the offset instead, `offset as CalldataPtr<T>`; reading through the pointer still needs `unsafe`.
