use fe_parser::{RecoveryMode, SyntaxKind, parse_source_file, syntax_node::SyntaxNode};

fn nest(depth: usize, innermost: &str) -> String {
    let mut ty = innermost.to_string();
    for _ in 0..depth {
        ty = format!("Wrap<<{ty} as Model>::Point>");
    }
    ty
}

/// Each level of `Wrap<<T as Model>::Point>` probed the positions inside it to
/// tell generic arguments from a shift, and every enclosing level probed them
/// all again, so parsing cost grew by roughly 5x per level: depth 12 took about
/// 96 seconds. This depth is unreachable that way and immediate with probe
/// outcomes reused per position, so the test bounds the speculative work as much
/// as it checks the tree.
#[test]
fn deeply_nested_qualified_generic_args_parse_without_repeating_probes() {
    const DEPTH: usize = 32;

    let source = format!("type A = {}\n", nest(DEPTH, "T"));

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

/// Malformed nesting must be bounded too, which means reusing probe outcomes
/// that recovered. Excluding those instead, as an earlier revision did, left
/// each enclosing level re-parsing everything below it: depth 10 cost about
/// 13 ms and grew over twofold per level, putting this depth out of reach.
/// Reuse is sound only because `recover` is confined to the scopes the probe
/// itself opened, so how far it consumes cannot depend on the caller. The
/// confinement has no effect on timing, so this test pins the reuse; the suite
/// as a whole pins that confining recovery changed no tree or diagnostic.
#[test]
fn deeply_nested_malformed_generic_args_do_not_repeat_probes() {
    const DEPTH: usize = 32;

    // A qualified type missing its `as`, innermost, and an argument list that
    // never closes: both force recovery inside the probes at every level.
    for innermost in ["Wrap<<T Model>::Point>", "Wrap<<T as Model>::Point"] {
        for recover in [false, true] {
            let source = format!("type A = {}\n", nest(DEPTH, innermost));
            let (green, errors) = parse_source_file(&source, RecoveryMode::new(recover));
            assert!(!errors.is_empty(), "{innermost:?} recover={recover}");
            // Recovery keeps consuming, so the tree still spans the input; a
            // `NoRecover` parse stops at the error and is not expected to.
            if recover {
                assert_eq!(
                    SyntaxNode::new_root(green).text().to_string(),
                    source,
                    "{innermost:?} recover={recover}",
                );
            }
        }
    }
}
