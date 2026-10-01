# Logs Directory

The complete pipeline writes four JSONL report streams. New records use the
canonical schemas described below.

- `guardgpt_audit.jsonl` - compact security events containing identifiers,
	session turn metadata when available, intent, confidence, risk, action, final
	status, allowed state, and reason codes. Raw prompts and generated responses
	are not written here.
- `guardgpt_prompt.jsonl` - detailed per-request prompt reports containing
	prompt information, security analysis, dataset evidence, input decision,
	generation metadata, output-audit status, final result, and optional session
	metadata.
- `guardgpt_session.jsonl` - session reports containing session ID, turn count,
	intent history, current and previous intent, transitions, the existing
	session risk score, and the session summary.
- `guardgpt_complete.jsonl` - comprehensive reports linking the prompt report,
	optional session report, and final execution result by request and audit IDs.

The compact audit stream is written by the complete pipeline. The registered
MCP `audit_logger` tool is retained as a compatibility stub and is not the
canonical writer.

Detailed prompt reports may contain raw prompt text. Treat the logs directory
as runtime data and protect it accordingly. Existing historical records are not
silently migrated or deleted, so older entries may use earlier schemas.
