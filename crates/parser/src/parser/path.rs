use crate::{ParseError, SyntaxKind, TextRange, TextSize};

use super::{
    Parser, ProbeKind, define_scope,
    expr::{is_lshift, is_lt_eq},
    param::{GenericArgListScope, TraitRefScope},
    token_stream::TokenStream,
    type_::parse_type,
};

define_scope! {
    #[doc(hidden)]
    pub PathScope { is_expr: bool },
    Path,
    (Colon2)
}
impl super::Parse for PathScope {
    type Error = ParseError;

    fn parse<S: TokenStream>(&mut self, parser: &mut Parser<S>) -> Result<(), Self::Error> {
        parser.set_newline_as_trivia(false);
        parser.parse(PathSegmentScope::new(self.is_expr))?;
        while parser.bump_if(SyntaxKind::Colon2) {
            parser.parse(PathSegmentScope::default())?;
        }
        Ok(())
    }
}

define_scope! { PathSegmentScope { is_expr: bool }, PathSegment }
impl super::Parse for PathSegmentScope {
    type Error = ParseError;

    fn parse<S: TokenStream>(&mut self, parser: &mut Parser<S>) -> Result<(), Self::Error> {
        match parser.current_kind() {
            Some(SyntaxKind::Lt) if is_qualified_type(parser) => {
                parser.parse(QualifiedTypeScope::default())
            }
            Some(kind) if is_path_segment(kind) => {
                parser.bump();

                if parser.current_kind_same_line() == Some(SyntaxKind::Lt) && is_lshift(parser) {
                    // `<<` is a left shift unless it opens generic arguments
                    // whose first argument is a qualified path, as in
                    // `Wrapped<<T as Trait>::Item>`. No shift operand continues
                    // with `>::`, so that prefix settles it, and a cast such as
                    // `value << bits as u256 >> 1` stays a shift.
                    if parser.probe(ProbeKind::LShiftOpensGenericArgs, |parser| {
                        parser.bump();
                        parser.parses_without_error(QualifiedTypeScope::default())
                            && parser.current_kind() == Some(SyntaxKind::Colon2)
                    }) {
                        // Errors inside the arguments are reported as they are parsed.
                        let _ = parser.parse(GenericArgListScope::new(self.is_expr));
                    }
                    return Ok(());
                }

                let is_turbofish = parser.current_kind_same_line() == Some(SyntaxKind::Colon2)
                    && parser.peek_two() == (Some(SyntaxKind::Colon2), Some(SyntaxKind::Lt));

                if (is_turbofish
                    || (parser.current_kind_same_line() == Some(SyntaxKind::Lt)
                        && !is_lt_eq(parser)))
                    && parser.probe(
                        ProbeKind::GenericArgList {
                            is_expr: self.is_expr,
                        },
                        |parser| {
                            parser.bump_if(SyntaxKind::Colon2);
                            parser.parses_without_error(GenericArgListScope::new(self.is_expr))
                        },
                    )
                {
                    if is_turbofish {
                        parser.bump_trivias();
                        parser.add_error(ParseError::Unexpected(
                            "unexpected turbofish syntax `::<`; remove the double colons".into(),
                            TextRange::at(parser.current_pos, TextSize::from(3)),
                        ));
                        parser.bump_expected(SyntaxKind::Colon2);
                    }
                    parser
                        .parse(GenericArgListScope::new(self.is_expr))
                        .expect("the probe suggests this will succeed");
                }
                Ok(())
            }
            _ => Err(ParseError::expected(
                &[SyntaxKind::PathSegment],
                None,
                parser.end_of_prev_token,
            )),
        }
    }
}

define_scope! { QualifiedTypeScope, QualifiedType }
impl super::Parse for QualifiedTypeScope {
    type Error = ParseError;

    fn parse<S: TokenStream>(&mut self, parser: &mut Parser<S>) -> Result<(), Self::Error> {
        parser.bump_expected(SyntaxKind::Lt);

        match parse_type(parser, None) {
            Ok(_) => {}
            Err(_) => {
                return Err(ParseError::expected(
                    &[SyntaxKind::PathType],
                    None,
                    parser.end_of_prev_token,
                ));
            }
        }
        if !parser.bump_if(SyntaxKind::AsKw) {
            return Err(ParseError::expected(
                &[SyntaxKind::AsKw],
                None,
                parser.end_of_prev_token,
            ));
        }
        parser.parse(TraitRefScope::default())?;
        if parser.bump_if(SyntaxKind::Gt) {
            Ok(())
        } else {
            Err(ParseError::expected(
                &[SyntaxKind::Gt],
                None,
                parser.end_of_prev_token,
            ))
        }
    }
}

pub(super) fn is_qualified_type<S: TokenStream>(parser: &mut Parser<S>) -> bool {
    parser.probe(ProbeKind::QualifiedType, |parser| {
        parser.bump_if(SyntaxKind::Lt)
            && parse_type(parser, None).is_ok()
            && parser.current_kind() == Some(SyntaxKind::AsKw)
    })
}

pub(super) fn is_path_segment(kind: SyntaxKind) -> bool {
    matches!(
        kind,
        SyntaxKind::SelfTypeKw
            | SyntaxKind::SelfKw
            | SyntaxKind::IngotKw
            | SyntaxKind::SuperKw
            | SyntaxKind::Ident
            | SyntaxKind::Lt
    )
}
