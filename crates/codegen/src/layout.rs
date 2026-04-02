#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TargetDataLayout {
    pub word_size_bytes: usize,
    pub discriminant_size_bytes: usize,
}

impl TargetDataLayout {
    pub const fn evm() -> Self {
        Self {
            word_size_bytes: 32,
            discriminant_size_bytes: 1,
        }
    }
}

pub const EVM_LAYOUT: TargetDataLayout = TargetDataLayout::evm();
pub const WORD_SIZE_BYTES: usize = EVM_LAYOUT.word_size_bytes;
pub const DISCRIMINANT_SIZE_BYTES: usize = EVM_LAYOUT.discriminant_size_bytes;
