use crate::yul::{
    doc::YulDoc,
    errors::YulError,
    legalize::{YBlockId, YTerminator},
};

use super::function::FunctionEmitter;

impl<'a, 'db> FunctionEmitter<'a, 'db> {
    pub(super) fn render_pc_dispatch(&mut self) -> Result<Vec<YulDoc>, YulError> {
        let mut cases = Vec::with_capacity(self.plan.blocks.len() + 1);
        for (idx, _) in self.plan.blocks.iter().enumerate() {
            cases.push(YulDoc::wide_block(
                format!("case {idx} "),
                self.render_linear_block(YBlockId(idx as u32))?,
            ));
        }
        cases.push(YulDoc::wide_block(
            "default ".to_string(),
            vec![YulDoc::line("invalid()")],
        ));
        Ok(vec![
            YulDoc::line("let pc := 0"),
            YulDoc::block(
                "for {} 1 {} ",
                vec![YulDoc::wide_block("switch pc ", cases)],
            ),
        ])
    }

    pub(super) fn render_terminator(
        &mut self,
        terminator: &YTerminator<'db>,
    ) -> Result<Vec<YulDoc>, YulError> {
        Ok(match terminator {
            YTerminator::Goto(target) => {
                vec![
                    YulDoc::line(format!("pc := {}", target.index())),
                    YulDoc::line("continue"),
                ]
            }
            YTerminator::Branch {
                cond,
                then_bb,
                else_bb,
            } => vec![
                YulDoc::block(
                    format!("if {} ", self.scalar_word_expr(*cond)?),
                    vec![
                        YulDoc::line(format!("pc := {}", then_bb.index())),
                        YulDoc::line("continue"),
                    ],
                ),
                YulDoc::line(format!("pc := {}", else_bb.index())),
                YulDoc::line("continue"),
            ],
            YTerminator::SwitchWord {
                discr,
                cases,
                default,
            } => {
                let mut docs = Vec::with_capacity(cases.len() + 1);
                for (value, block) in cases.iter() {
                    docs.push(YulDoc::wide_block(
                        format!("case {} ", self.const_scalar_expr(value)),
                        vec![
                            YulDoc::line(format!("pc := {}", block.index())),
                            YulDoc::line("continue"),
                        ],
                    ));
                }
                docs.push(YulDoc::wide_block(
                    "default ".to_string(),
                    vec![
                        YulDoc::line(format!("pc := {}", default.index())),
                        YulDoc::line("continue"),
                    ],
                ));
                vec![YulDoc::wide_block(
                    format!("switch {} ", self.scalar_word_expr(*discr)?),
                    docs,
                )]
            }
            YTerminator::MatchEnumTag {
                tag,
                cases,
                default,
                ..
            } => {
                let mut docs = Vec::with_capacity(cases.len() + usize::from(default.is_some()));
                for (variant, block) in cases.iter() {
                    docs.push(YulDoc::wide_block(
                        format!("case {} ", variant.index),
                        vec![
                            YulDoc::line(format!("pc := {}", block.index())),
                            YulDoc::line("continue"),
                        ],
                    ));
                }
                if let Some(default) = default {
                    docs.push(YulDoc::wide_block(
                        "default ".to_string(),
                        vec![
                            YulDoc::line(format!("pc := {}", default.index())),
                            YulDoc::line("continue"),
                        ],
                    ));
                } else {
                    docs.push(YulDoc::wide_block(
                        "default ".to_string(),
                        vec![YulDoc::line("invalid()")],
                    ));
                }
                vec![YulDoc::wide_block(
                    format!("switch {} ", self.scalar_word_expr(*tag)?),
                    docs,
                )]
            }
            YTerminator::TerminalCall { callee, args } => {
                let args = args
                    .iter()
                    .map(|arg| self.local_value(*arg))
                    .collect::<Result<Vec<_>, _>>()?;
                let callee_plan = self.index.function(*callee)?;
                let mut docs = Vec::new();
                let rendered_args = args
                    .iter()
                    .map(|arg| arg.value.as_str())
                    .collect::<Vec<_>>()
                    .join(", ");
                docs.push(YulDoc::line(format!(
                    "{}({rendered_args})",
                    super::util::prefix_yul_name(&callee_plan.symbol)
                )));
                docs.push(YulDoc::line("invalid()"));
                docs
            }
            YTerminator::ReturnData { offset, len } => vec![YulDoc::line(format!(
                "return({}, {})",
                self.scalar_word_expr(*offset)?,
                self.scalar_word_expr(*len)?
            ))],
            YTerminator::Revert { offset, len } => vec![YulDoc::line(format!(
                "revert({}, {})",
                self.scalar_word_expr(*offset)?,
                self.scalar_word_expr(*len)?
            ))],
            YTerminator::SelfDestruct { beneficiary } => vec![YulDoc::line(format!(
                "selfdestruct({})",
                self.scalar_word_expr(*beneficiary)?
            ))],
            YTerminator::Trap => vec![YulDoc::line("invalid()")],
            YTerminator::Return(Some(value)) => {
                let value = self.local_value(*value)?;
                let mut docs = value.setup;
                docs.push(YulDoc::line(format!("ret := {}", value.value)));
                docs.push(YulDoc::line("leave"));
                docs
            }
            YTerminator::Return(None) => vec![YulDoc::line("leave")],
            YTerminator::Stop => vec![YulDoc::line("stop()")],
        })
    }
}
