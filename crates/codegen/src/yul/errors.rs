use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum YulError {
    Unsupported(String),
    InvalidYulPackage(String),
    Layout(String),
    ConstSerialization(String),
}

impl fmt::Display for YulError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            YulError::Unsupported(message)
            | YulError::InvalidYulPackage(message)
            | YulError::Layout(message)
            | YulError::ConstSerialization(message) => write!(f, "{message}"),
        }
    }
}

impl std::error::Error for YulError {}
