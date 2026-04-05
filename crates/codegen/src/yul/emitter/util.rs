pub(super) fn prefix_yul_name(name: &str) -> String {
    if name.starts_with('$') {
        sanitize_yul_ident(name)
    } else {
        format!("${}", sanitize_yul_ident(name))
    }
}

pub(super) fn sanitize_yul_ident(value: &str) -> String {
    value
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || ch == '_' || ch == '$' {
                ch
            } else {
                '_'
            }
        })
        .collect()
}

pub(super) fn section_object_label(name: &mir2::RuntimeSectionName) -> String {
    match name {
        mir2::RuntimeSectionName::Init => "init".to_string(),
        mir2::RuntimeSectionName::Runtime => "runtime".to_string(),
        mir2::RuntimeSectionName::Main => "main".to_string(),
        mir2::RuntimeSectionName::Test(name) => sanitize_yul_ident(name),
        mir2::RuntimeSectionName::CodeRegion(symbol) => sanitize_yul_ident(symbol),
    }
}
