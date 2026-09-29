use fe_parser::{RecoveryMode, SyntaxKind, parse_source_file, syntax_node::SyntaxNode};

/// Each level of `Wrap<<T as Model>::Point>` probed the positions inside it to
/// tell generic arguments from a shift, and every enclosing level probed them
/// all again, so parsing cost grew by roughly 5x per level: depth 12 took about
/// 96 seconds. This depth is unreachable that way and immediate with probe
/// outcomes reused per position, so the test bounds the speculative work as much
/// as it checks the tree.
#[test]
fn deeply_nested_qualified_generic_args_parse_without_repeating_probes() {
    const DEPTH: usize = 32;

    let mut ty = "T".to_string();
    for _ in 0..DEPTH {
        ty = format!("Wrap<<{ty} as Model>::Point>");
    }
    let source = format!("type A = {ty}\n");

    let (green, errors) = parse_source_file(&source, RecoveryMode::new(false));
    assert!(errors.is_empty(), "{errors:#?}");

    let cst = SyntaxNode::new_root(green);
    assert_eq!(cst.text().to_string(), source);
    assert_eq!(
        cst.descendants()
            .filter(|node| node.kind() == SyntaxKind::QualifiedType)
            .count(),
        DEPTH
    );
}
