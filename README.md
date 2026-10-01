# GuardGPT

GuardGPT is a local prompt-safety and response-auditing system. It combines a
Sentence-Transformers intent classifier, a validated dataset/FAISS evidence
layer, deterministic jailbreak patterns, a decision engine, optional
multi-turn history, Ollama generation, and a structured output audit.

The reporting layer is observational. It does not change classification,
thresholds, temporal state, generation, output auditing, or the final
`ALLOW`/`SANITIZE`/`BLOCK` decision.

## Current Architecture

The supported end-to-end request path is `complete_request`:

```text
User or application
        |
        v
GuardGPT host / MCP client
        |
        v
MCP server (streamable HTTP)
        |
        v
CompletePipeline
        |
        +--> IntentClassifier
        +--> DatasetLoader + normalized FAISS evidence
        +--> deterministic jailbreak-pattern checks
        +--> ConversationGuard history and temporal state, when a session exists
        |
        v
DecisionEngine
        |
        +--> ALLOW
        +--> SANITIZE
        +--> BLOCK
        |
        +--> Ollama/Llama generation when allowed
        +--> OutputAuditor for generated candidates
        |
        v
Final response and observational reports
```

The normal `main.py` CLI starts or reuses an MCP server and calls
`complete_request`. The server delegates to `CompletePipeline`, which is the
canonical path for input checks, optional rewriting, generation, output review,
and report persistence.

### LangGraph compatibility path

`agent/graph.py` defines a separate LangGraph workflow with these nodes:

```text
ReceivePrompt
  -> PromptAnalysis
  -> JailbreakDetection
  -> ContentModeration
  -> CombineResults
  -> Decision
  -> AuditLog
  -> BuildReport
```

The nodes are MCP-only adapters. They do not implement a second classifier or
decision engine. This graph is retained for compatibility and agent-oriented
tests. The CLI's canonical `complete_request` path uses the complete pipeline
directly rather than this report-only graph.

## Repository Layout

```text
GuardGPT/
├── main.py                         CLI entry point
├── requirements.txt                Python dependencies
├── run_tests.py                    focused unittest runner
├── verify_augmented_dataset.py     dataset/index alignment check
├── agent/                          LangGraph compatibility workflow and MCP client
├── core/                           safety, classification, pipeline, and reporting logic
├── mcp_server/                     MCP server, schemas, and tool adapters
├── data/                           JSONL dataset, FAISS index, and ID map
├── intent_classifier/              supplied classifier checkpoint and tokenizer
├── logs/                           generated JSONL reports and audit events
└── tests/                          regression and integration tests
```

## Setup

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

The supplied runtime artifacts are expected at:

- `data/guardgpt_dataset.jsonl`
- `data/guardgpt_faiss.index`
- `data/guardgpt_id_map.json`
- `intent_classifier/best_model.pt`
- `intent_classifier/tokenizer.json`

The embedding model is `all-MiniLM-L6-v2`. It is loaded locally by default.
Set `GUARDGPT_ALLOW_MODEL_DOWNLOAD=1` to explicitly permit a missing embedding
model to be downloaded.

### Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `GUARDGPT_MCP_URL` | `http://127.0.0.1:8000/mcp` | MCP endpoint used by the client |
| `GUARDGPT_PROJECT_ROOT` | auto-detected | Root used for data and logs |
| `GUARDGPT_DATASET` | `data/guardgpt_dataset.jsonl` | Optional dataset path override |
| `GUARDGPT_ALLOW_MODEL_DOWNLOAD` | disabled | Permit embedding-model download |
| `GUARDGPT_INTENT_MODEL` | `intent_classifier/best_model.pt` | Optional classifier checkpoint override |
| `OLLAMA_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `llama3` | Generation model |
| `OLLAMA_AUDIT_MODEL` | generation model | Optional separate audit model |
| `OLLAMA_TIMEOUT` | `120` | Ollama request timeout in seconds |
| `OLLAMA_TEMPERATURE` | `0.2` | Generation temperature |

Ollama is required for answer generation, sanitize rewrites, and output
auditing. Input-only checks do not call Ollama.

## Running GuardGPT

### Single prompt

```powershell
python main.py --prompt "Explain how Python lists work."
```

Use `--check` with a prompt to run input checks without generation:

```powershell
python main.py --check --prompt "Explain web authentication."
```

### Interactive chat

```powershell
python main.py --chat
```

The chat loop also starts when no prompt or demo flag is supplied. Within chat:

- `/new` starts a new session.
- `/session` requests current session analytics.
- `/status` checks local dependencies and Ollama.
- `/help` shows commands.
- `/exit` leaves chat.

Each session has its own `ConversationGuard`, history, and temporal state.

### Demo matrix and status

```powershell
python main.py --pipeline
python main.py --demo
python main.py --status
```

`--pipeline` runs the current built-in matrix: safe general, coding, prompt
injection, harmful, self-harm support, educational, and empty-input cases.

### MCP server

The CLI auto-starts the server when the configured endpoint is unavailable. To
start it manually:

```powershell
python -m mcp_server.server
```

The server binds to a loopback address only.

## Intent Classifier

The existing classifier uses `sentence-transformers/all-MiniLM-L6-v2` embeddings
with dimension 384. It builds prototype embeddings for the semantic intent
taxonomy and can load the supplied checkpoint:

```text
intent_classifier/best_model.pt
```

The checkpoint is a classifier/encoder artifact, not a temporal model. The
security labels used by the trained safety output and temporal state are:

1. `safe`
2. `prompt_injection`
3. `jailbreak`
4. `harmful_instructions`
5. `manipulation`
6. `self_harm_risk`

The classifier also has operational semantic intents such as `coding`,
`educational`, `benign`, `creative`, `personal_advice`, `harmful`, `illegal`,
`account_recovery`, `cyber_abuse`, and `self_harm`. These are existing
classification outputs used by the safety pipeline; they are not additional
temporal labels.

## Dataset and FAISS Evidence

`data/guardgpt_dataset.jsonl` is the active dataset. The current file contains
19,200 records: 3,200 records for each of the six security labels listed above.
Records currently contain `input_text`, `intent`, `confidence`, and `reason`;
the loader normalizes records and supplies operational defaults such as
`request_id`, `target_verdict`, and category scores when needed.

The supplied vector artifacts are:

- `data/guardgpt_faiss.index`
- `data/guardgpt_id_map.json`

The current index is a FAISS `IndexFlatIP` with 19,200 normalized vectors of
dimension 384. `DatasetLoader` validates that the JSONL records, index, and ID
map have matching counts and records, then queries the nearest normalized
embedding. The resulting cosine-style inner-product similarity is exposed as
`dataset_match_confidence` and the nearest record supplies dataset evidence.

Dataset evidence is contextual evidence. It does not replace the
`IntentClassifier` and does not independently define the final safety action.

Run the artifact alignment check with:

```powershell
python verify_augmented_dataset.py
```

## Safety Decision Pipeline

`SafetyService.analyze()` obtains the classifier result, queries the dataset,
applies the existing risk estimator and deterministic jailbreak patterns, and
passes the resulting signal to `DecisionEngine`. The service also reuses the
classifier's normalized embedding for the observational temporal layer.

`DecisionEngine` is the authority for the final action:

- `ALLOW`: continue with the original request.
- `SANITIZE`: use the existing rewrite path, recheck the rewritten prompt, and
  generate only if the recheck allows it.
- `BLOCK`: do not generate an answer.

`CompletePipeline` then:

1. Rejects empty or oversized input.
2. Runs input analysis and optional session-history evaluation.
3. Handles high-confidence self-harm blocks with the existing fixed supportive
   response and does not call the generator for that response.
4. Rewrites and rechecks `SANITIZE` requests.
5. Generates through `LlamaBackend`/Ollama only after input checks pass.
6. Sends each candidate to `OutputAuditor`.
7. Allows one bounded regeneration attempt if the candidate is unsafe or
   irrelevant.
8. Releases only an audited response.

Final statuses used by the complete pipeline are:

- `SAFE`
- `CAUTION`
- `UNSAFE`
- `SUPPORT`
- `INVALID_INPUT`
- `ERROR`

Risk levels are `safe`, `low`, `medium`, `high`, and `critical`. Their
thresholds are defined by `core/risk_estimator.py`; this README does not
duplicate or reinterpret those thresholds.

Self-harm is handled as a support need. A sufficiently confident
`self_harm`/`self_harm_risk` block returns the existing support response with
status `SUPPORT`; it is not represented as a separate `FLAG` action.

`OutputAuditor` validates a strict structured verdict containing `safe`,
`relevant`, and allowed audit categories. Malformed or contradictory verdicts
fail closed.

## Temporal Intent State

`core/temporal_intent.py` implements a deterministic mathematical layer for
session observation. It is not trained and does not replace the existing
classifier or decision engine.

For an embedding $c_t$, previous hidden state $H_{t-1}$, dataset evidence
$R_t$, and previous intent distribution $I_{t-1}$, it applies:

$$
H_t = \lambda H_{t-1} + (1-\lambda)c_t
$$

$$
\hat I_t = \operatorname{Softmax}(
W_c c_t + W_h H_{t-1} + W_r R_t + W_I I_{t-1} + b)
$$

The historical recurrence term $W_I I_{t-1}$ is part of the executable
candidate-intent calculation.

$$
I_t = (1-g_t)I_{t-1} + g_t\hat I_t
$$

$$
g_t = \sigma(W_g[c_t;H_{t-1};R_t;I_{t-1}] + b_g)
$$

Dimensions:

| Value | Dimension |
|---|---:|
| `c_t` | 384 |
| `H_t` | 384 |
| `R_t` | 1 |
| `I_t` | 6 |
| `W_c` | `[6, 384]` |
| `W_h` | `[6, 384]` |
| `W_r` | `[6, 1]` |
| `W_I` | `[6, 6]` |
| `b` | `[6]` |
| gate input | `384 + 384 + 1 + 6 = 775` |
| `W_g` | `[1, 775]` |
| `b_g` | `[1]` |

The temporal labels are exactly:

```text
safe, prompt_injection, jailbreak, harmful_instructions,
manipulation, self_harm_risk
```

Each `ConversationGuard` owns independent `H_t` and `I_t` state. `reset()`
restores the zero hidden state and deterministic uniform six-dimensional
probability distribution. `R_t` is the existing `dataset_match_confidence`.

The temporal update is observational and session-scoped. It does not alter
classifier output, risk thresholds, `DecisionEngine` inputs, safety actions,
generation, or output auditing.

## MCP Interface

The loopback MCP server registers:

| Tool | Responsibility |
|---|---|
| `health` | Reports server version and complete-pipeline availability |
| `complete_request` | Canonical end-to-end check, generation, audit, and reporting path |
| `prompt_analysis` | Classifier, dataset evidence, risk, and reason-code adapter |
| `jailbreak_detection` | Classifier plus deterministic jailbreak-pattern adapter |
| `content_moderation` | Classifier and dataset-category moderation adapter |
| `decision` | Thin `DecisionEngine` wrapper |
| `audit_logger` | Backward-compatible stub; the complete pipeline writes the audit event |

The five report-only analysis tools remain available for compatibility. They
are not separate safety engines and are not all executed by the canonical
`complete_request` flow.

## Reports and Logs

The complete pipeline writes four JSONL files under `logs/`.

### `guardgpt_audit.jsonl`

Compact security event containing identifiers, session turn metadata when
available, intent, confidence, risk, action, final status, allowed state, and
reason codes. It does not store the raw prompt or generated response.

### `guardgpt_prompt.jsonl`

One detailed prompt report per processed request. It contains `report_type:
"prompt"`, request/audit identifiers, prompt text, intent information,
security analysis, dataset evidence, input decision, generation metadata,
output-audit status, final result, and optional session metadata.

### `guardgpt_session.jsonl`

One session report for each processed request with a session. It contains
`session_id`, request/audit identifiers, turn count, intent history, current and
previous intent, transitions, the existing session risk score, and the
deterministic summary text from `ConversationGuard`.

### `guardgpt_complete.jsonl`

One comprehensive report containing the canonical prompt report, the canonical
session report or `null`, and an `execution` object with final action, final
status, and allowed state. `request_id`, `audit_id`, and `session_id` link the
records across files.

New records use these canonical schemas. Existing historical JSONL records are
not silently migrated or deleted, so older entries may retain earlier schemas.

## Tests and Validation

Run the focused offline suite:

```powershell
python run_tests.py
```

Run the full pytest suite:

```powershell
python -m pytest tests/ -v
```

The repository currently contains a known collection inconsistency in
`tests/test_pipeline_with_conversation.py`: it expects
`main.run_agent_with_history`, which is not part of the current `main.py`.
That helper is not added by the current architecture.

The test suite covers classifier/dataset integration, MCP tools and client
behavior, LangGraph node shape, complete HTTP transport, safety decisions,
self-harm gates, session history, reporting, and temporal equations.

## Operational Notes

- The MCP server is restricted to loopback addresses.
- Local embedding-model download is opt-in.
- Invalid MCP calls and backend failures fail closed or surface typed client
  errors; raw stack traces are not presented as normal user responses.
- Do not expose the local Ollama or unauthenticated loopback MCP service
  publicly.
- Generated logs are runtime artifacts and may contain sensitive prompt data in
  detailed prompt reports. The compact audit log intentionally remains
  redacted.