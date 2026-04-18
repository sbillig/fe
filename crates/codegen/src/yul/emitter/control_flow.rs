use crate::yul::{
    doc::YulDoc,
    errors::YulError,
    legalize::{YBlockId, YLocalId, YTerminator},
};

use super::function::{FunctionEmitter, YLoopInfo};

#[derive(Clone, Copy, Debug)]
pub(super) struct LoopEmitCtx {
    header: YBlockId,
    break_target: Option<YBlockId>,
}

#[derive(Clone, Copy)]
pub(super) struct EmitCtx<'a> {
    loop_ctx: Option<LoopEmitCtx>,
    stop_blocks: &'a [YBlockId],
}

impl<'a, 'db> FunctionEmitter<'a, 'db> {
    pub(super) fn emit_block(&mut self, block: YBlockId) -> Result<Vec<YulDoc>, YulError> {
        let mut path = Vec::new();
        self.emit_block_internal(
            block,
            EmitCtx {
                loop_ctx: None,
                stop_blocks: &[],
            },
            &mut path,
        )
    }

    pub(super) fn emit_block_internal(
        &mut self,
        block: YBlockId,
        ctx: EmitCtx<'_>,
        path: &mut Vec<YBlockId>,
    ) -> Result<Vec<YulDoc>, YulError> {
        if ctx.stop_blocks.contains(&block) {
            return Ok(Vec::new());
        }
        if !matches!(ctx.loop_ctx, Some(LoopEmitCtx { header, .. }) if header == block)
            && let Some(loop_info) = self.loop_info(block).cloned()
        {
            let (loop_doc, exit) = self.emit_loop(block, loop_info, ctx.stop_blocks, path)?;
            let mut docs = vec![loop_doc];
            if let Some(exit) = exit {
                docs.extend(self.emit_edge(exit, ctx, path)?);
            }
            return Ok(docs);
        }
        if path.contains(&block) {
            return Err(YulError::Unsupported(format!(
                "structured Yul emission requires a reducible CFG; block {} re-entered on the active path",
                block.index()
            )));
        }

        let block_data = self
            .plan
            .blocks
            .get(block.index())
            .ok_or_else(|| YulError::InvalidYulPackage(format!("missing block {}", block.index())))?
            .clone();
        path.push(block);

        let mut docs = Vec::new();
        for stmt in &block_data.stmts {
            docs.extend(self.render_stmt(stmt)?);
        }
        docs.extend(self.emit_terminator(block, &block_data.terminator, ctx, path)?);
        path.pop();
        Ok(docs)
    }

    fn emit_edge(
        &mut self,
        target: YBlockId,
        ctx: EmitCtx<'_>,
        path: &mut Vec<YBlockId>,
    ) -> Result<Vec<YulDoc>, YulError> {
        if let Some(loop_ctx) = ctx.loop_ctx {
            if target == loop_ctx.header {
                return Ok(vec![YulDoc::line("continue")]);
            }
            if loop_ctx.break_target == Some(target) {
                return Ok(vec![YulDoc::line("break")]);
            }
            if !self
                .loop_info(loop_ctx.header)
                .expect("active loop header should have loop info")
                .blocks
                .contains(&target)
            {
                let mut exit_stops = ctx.stop_blocks.to_vec();
                if let Some(break_target) = loop_ctx.break_target
                    && !exit_stops.contains(&break_target)
                {
                    exit_stops.push(break_target);
                }
                let mut docs = self.emit_block_internal(
                    target,
                    EmitCtx {
                        loop_ctx: None,
                        stop_blocks: &exit_stops,
                    },
                    path,
                )?;
                if loop_ctx.break_target.is_some() {
                    docs.push(YulDoc::line("break"));
                }
                return Ok(docs);
            }
        }
        if ctx.stop_blocks.contains(&target) {
            return Ok(Vec::new());
        }
        self.emit_block_internal(target, ctx, path)
    }

    fn emit_terminator(
        &mut self,
        block: YBlockId,
        terminator: &YTerminator<'db>,
        ctx: EmitCtx<'_>,
        path: &mut Vec<YBlockId>,
    ) -> Result<Vec<YulDoc>, YulError> {
        Ok(match terminator {
            YTerminator::Goto(target) => self.emit_edge(*target, ctx, path)?,
            YTerminator::Branch {
                cond,
                then_bb,
                else_bb,
            } => self.emit_branch(block, *cond, *then_bb, *else_bb, ctx, path)?,
            YTerminator::SwitchWord {
                discr,
                cases,
                default,
            } => self.emit_switch(
                block,
                self.scalar_word_expr(*discr)?,
                cases
                    .iter()
                    .map(|(value, target)| (self.const_scalar_expr(value), *target))
                    .collect(),
                Some(*default),
                ctx,
                path,
            )?,
            YTerminator::MatchEnumTag {
                tag,
                cases,
                default,
                ..
            } => self.emit_switch(
                block,
                self.scalar_word_expr(*tag)?,
                cases
                    .iter()
                    .map(|(variant, target)| (variant.index.to_string(), *target))
                    .collect(),
                *default,
                ctx,
                path,
            )?,
            YTerminator::TerminalCall { callee, args } => {
                let args = args
                    .iter()
                    .map(|arg| self.local_value(*arg))
                    .collect::<Result<Vec<_>, _>>()?;
                let mut docs = Vec::new();
                for arg in &args {
                    docs.extend(arg.setup.clone());
                }
                let rendered_args = args
                    .iter()
                    .map(|arg| arg.value.as_str())
                    .collect::<Vec<_>>()
                    .join(", ");
                let callee_plan = self.index.function(*callee)?;
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

    fn emit_branch(
        &mut self,
        block: YBlockId,
        cond: YLocalId,
        then_bb: YBlockId,
        else_bb: YBlockId,
        ctx: EmitCtx<'_>,
        path: &mut Vec<YBlockId>,
    ) -> Result<Vec<YulDoc>, YulError> {
        let cond = self.scalar_word_expr(cond)?;
        let join = self.join_candidate(block, ctx.loop_ctx, ctx.stop_blocks);
        let mut branch_stops = ctx.stop_blocks.to_vec();
        if let Some(join) = join
            && !branch_stops.contains(&join)
        {
            branch_stops.push(join);
        }

        let mut then_state = self.state.clone();
        let then_docs = self.with_state(&mut then_state, |this| {
            this.emit_edge(
                then_bb,
                EmitCtx {
                    loop_ctx: ctx.loop_ctx,
                    stop_blocks: &branch_stops,
                },
                path,
            )
        })?;
        let mut else_state = self.state.clone();
        let else_docs = self.with_state(&mut else_state, |this| {
            this.emit_edge(
                else_bb,
                EmitCtx {
                    loop_ctx: ctx.loop_ctx,
                    stop_blocks: &branch_stops,
                },
                path,
            )
        })?;

        let mut docs = Vec::new();
        if !then_docs.is_empty() {
            docs.push(YulDoc::block(format!("if {cond} "), then_docs));
        }
        if !else_docs.is_empty() {
            docs.push(YulDoc::block(format!("if iszero({cond}) "), else_docs));
        }
        if let Some(join) = join {
            docs.extend(self.emit_block_internal(join, ctx, path)?);
        }
        Ok(docs)
    }

    fn emit_switch(
        &mut self,
        block: YBlockId,
        discr: String,
        cases: Vec<(String, YBlockId)>,
        default: Option<YBlockId>,
        ctx: EmitCtx<'_>,
        path: &mut Vec<YBlockId>,
    ) -> Result<Vec<YulDoc>, YulError> {
        let join = self.join_candidate(block, ctx.loop_ctx, ctx.stop_blocks);
        let mut switch_stops = ctx.stop_blocks.to_vec();
        if let Some(join) = join
            && !switch_stops.contains(&join)
        {
            switch_stops.push(join);
        }

        let mut docs = vec![YulDoc::line(format!("switch {discr}"))];
        for (literal, target) in cases {
            let mut case_state = self.state.clone();
            let case_docs = self.with_state(&mut case_state, |this| {
                this.emit_edge(
                    target,
                    EmitCtx {
                        loop_ctx: ctx.loop_ctx,
                        stop_blocks: &switch_stops,
                    },
                    path,
                )
            })?;
            docs.push(YulDoc::wide_block(format!("case {literal} "), case_docs));
        }

        let default_docs = if let Some(default) = default {
            let mut default_state = self.state.clone();
            self.with_state(&mut default_state, |this| {
                this.emit_edge(
                    default,
                    EmitCtx {
                        loop_ctx: ctx.loop_ctx,
                        stop_blocks: &switch_stops,
                    },
                    path,
                )
            })?
        } else {
            vec![YulDoc::line("invalid()")]
        };
        docs.push(YulDoc::wide_block("default ", default_docs));

        if let Some(join) = join {
            docs.extend(self.emit_block_internal(join, ctx, path)?);
        }
        Ok(docs)
    }

    fn join_candidate(
        &self,
        block: YBlockId,
        loop_ctx: Option<LoopEmitCtx>,
        stop_blocks: &[YBlockId],
    ) -> Option<YBlockId> {
        let join = self.ipdom.get(block.index()).copied().flatten()?;
        if stop_blocks.contains(&join)
            || matches!(loop_ctx, Some(LoopEmitCtx { header, break_target }) if join == header || break_target == Some(join))
        {
            None
        } else {
            Some(join)
        }
    }

    fn emit_loop(
        &mut self,
        header: YBlockId,
        loop_info: YLoopInfo,
        stop_blocks: &[YBlockId],
        path: &mut Vec<YBlockId>,
    ) -> Result<(YulDoc, Option<YBlockId>), YulError> {
        let mut loop_stops = stop_blocks.to_vec();
        if let Some(exit) = loop_info.exit
            && !loop_stops.contains(&exit)
        {
            loop_stops.push(exit);
        }
        let body_docs = self.emit_block_internal(
            header,
            EmitCtx {
                loop_ctx: Some(LoopEmitCtx {
                    header,
                    break_target: loop_info.exit,
                }),
                stop_blocks: &loop_stops,
            },
            path,
        )?;
        Ok((YulDoc::block("for {} 1 {} ", body_docs), loop_info.exit))
    }
}
