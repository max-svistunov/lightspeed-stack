# Pydantic AI ↔ Llama Stack: Concept Mapping for a Backend-Agnostic YAML Schema

## 1. Pydantic AI core concepts and configuration surface

**What is an Agent.** In Pydantic AI an `Agent` is a generic, type-parameterised container that owns: a default model, instructions / system prompts, tools (and toolsets), capabilities (composable behavior units), an output type / output validators, retry budgets, model settings, dependency type, and instrumentation settings. From the official Agents API reference (ai.pydantic.dev/api/agent/): `Agent` is generic in `(AgentDepsT, OutputDataT)` and "by default, if neither generic parameter is customised, agents have type `Agent[None, str]`" (https://ai.pydantic.dev/api/agent/).

Canonical construction (from the project README / overview at ai.pydantic.dev):
```python
from pydantic_ai import Agent
agent = Agent(
    'anthropic:claude-sonnet-4-6',
    instructions='Be concise, reply with one sentence.',
)
result = agent.run_sync('Where does "hello world" come from?')
```
(https://ai.pydantic.dev/)

The Agent also owns tools registered via `@agent.tool` / `@agent.tool_plain` or via `tools=[...]`, dependencies via `deps_type=...`, structured outputs via `output_type=...`, and capabilities via `capabilities=[...]` (https://ai.pydantic.dev/api/agent/, https://ai.pydantic.dev/capabilities/).

**How the model/provider is declared.** Pydantic AI is **model-string + client-object based**, not Llama-Stack-style named provider entries. The simplest form is a string `'<provider>:<model>'`:

> "When you instantiate an Agent with just a name formatted as `<provider>:<model>`, e.g. `openai:gpt-5.2` or `openrouter:google/gemini-3-pro-preview`, Pydantic AI will automatically select the appropriate model class, provider, and profile."
> — https://ai.pydantic.dev/models/overview/

For non-default endpoints, auth, or AI-gateway use, you instantiate a `Model` class and pass a `Provider`:
```python
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.azure import AzureProvider
agent = Agent(OpenAIChatModel('gpt-5.2', provider=AzureProvider(...)))
```
(https://ai.pydantic.dev/models/overview/, https://ai.pydantic.dev/api/providers/)

There is **no native concept of named provider entries with type/id/config that get looked up by name at runtime**, the way Llama Stack does. The `Provider` in Pydantic AI is a Python class with constructor args (`api_key`, `base_url`, `openai_client`, `http_client`), not a registry entry referenced from a YAML file by id (https://pydantic.dev/docs/ai/api/pydantic-ai/providers/).

**Multiple models / providers in one app.** Per-agent: each `Agent` instance owns its own model. Globally there is no "default provider list"; the closest thing is the `gateway/...` prefix (Pydantic AI Gateway) or the `FallbackModel` wrapper that takes multiple models and falls back on failure (https://ai.pydantic.dev/models/overview/). Multi-model applications usually create multiple `Agent` instances and pass them around, optionally via dependency injection (`deps_type` carries a `RunContext` with whatever shared clients/configs you want).

**File-based config — yes, natively.** Since the introduction of `AgentSpec` and `Agent.from_file` / `Agent.from_spec`, Pydantic AI supports declarative YAML/JSON agent definitions:

```python
from pydantic_ai import Agent
agent = Agent.from_file('agent.yaml')
```

```yaml
# agent.yaml
model: anthropic:claude-opus-4-6
instructions: "You are a helpful assistant."
capabilities:
  - WebSearch: {local: duckduckgo}
  - Thinking: {effort: high}
```
(https://ai.pydantic.dev/core-concepts/agent-spec/, https://ai.pydantic.dev/api/agent/)

`AgentSpec.to_file('agent.yaml')` can also emit a companion `agent_schema.json` for editor autocompletion. The spec is **per-agent**, not a server-wide config — there is no equivalent to Llama Stack's single `run.yaml` describing the whole runtime, APIs, and provider registry. (See section 5 for what this implies for a single-file operator config.)

**Credentials / API keys at runtime.** Three mechanisms, in order of expressiveness:
1. **Environment variables** — each `Provider` class reads a conventional env var if no explicit `api_key=` is passed (e.g., `OLLAMA_API_KEY`, `OPENAI_API_KEY`, documented in https://pydantic.dev/docs/ai/api/pydantic-ai/providers/).
2. **Explicit provider client construction** — pass `api_key=`, `base_url=`, or a fully-constructed vendor SDK client (e.g., `openai_client=AsyncOpenAI(...)`) to the `Provider`.
3. **AgentSpec/from_file** — the YAML spec does **not** declare credentials; it declares model name and capabilities, and credentials still come from env vars or from Python code that wires the `Provider`.

There is no built-in `${env.VAR}` interpolation in `AgentSpec` YAML. Template strings (`{{user_name}}`) exist but resolve against `deps`, not environment variables (https://ai.pydantic.dev/core-concepts/agent-spec/).

## 2. RAG / retrieval / vector stores

**No built-in RAG abstraction. No vector-store abstraction.** Pydantic AI's official docs are explicit:

> "The main semantic difference between Pydantic AI Tools and RAG is RAG is synonymous with vector search, while Pydantic AI tools are more general-purpose. For vector search, you can use our embeddings support to generate embeddings across multiple providers."
> — https://ai.pydantic.dev/tools/

RAG is implemented as a user-written tool that calls whatever vector DB the user has chosen. The official "RAG" example (https://ai.pydantic.dev/examples/rag/, https://github.com/pydantic/pydantic-ai/blob/main/docs/examples/rag.md) uses **PostgreSQL + pgvector** directly via `asyncpg` and the **OpenAI SDK for embeddings** — Pydantic AI itself isn't involved in indexing:

```python
@rag_agent.tool
async def retrieve(context: RunContext[Deps], search_query: str) -> str:
    embedding = await context.deps.openai.embeddings.create(
        input=search_query, model='text-embedding-3-small',
    )
    rows = await context.deps.pool.fetch(
        'SELECT chunk FROM text_chunks ORDER BY embedding <-> $1 LIMIT 5',
        pydantic_core.to_json(embedding.data[0].embedding).decode(),
    )
    return '\n\n'.join(f'# Chunk:\n{row["chunk"]}\n' for row in rows)
```
(verbatim from the official example)

The docs even note: *"Note building the database doesn't use Pydantic AI right now, instead it uses the OpenAI SDK directly."* (https://github.com/pydantic/pydantic-ai/blob/main/docs/examples/rag.md)

**Canonical community patterns.** From observed community projects and Pydantic AI's own example:
- **pgvector + OpenAI / Voyage embeddings via raw SDK calls** — the pattern in the official example, and in projects such as github.com/serkanyasr/agentic_rag_project (Pydantic AI + FastAPI + pgvector) and github.com/cskwork/pydantic-rag-ollama (Ollama embeddings + pgvector).
- **MCP server for retrieval** — exposing a vector DB through an MCP server and consuming it via Pydantic AI's `MCP(url=...)` capability (https://ai.pydantic.dev/capabilities/).
- **LlamaIndex / LangChain as a retrieval backend** — used purely as a library inside a Pydantic AI tool; no first-class integration is documented in ai.pydantic.dev.

There is no `pydantic_ai.vector_store` module, no `VectorStore` protocol, and no roadmap entry that surfaced in my search for adding one (caveat: I did not find a public roadmap document, only the version policy and release notes — see §7).

## 3. Safety / guardrails

**No built-in safety / guardrails / shield API in Pydantic AI core.** GitHub Issue #1197 ("Guardrails") is the open feature request, with a working-design proposal that mirrors the OpenAI Agents SDK's `@input_guardrail` / `@output_guardrail` decorators (https://github.com/pydantic/pydantic-ai/issues/1197). As of `pydantic-ai-slim 1.97.0` (May 15, 2026, https://pypi.org/project/pydantic-ai-slim/) it has not been merged into core.

**What Pydantic AI *does* provide is validation-as-correctness, not safety:**
- `output_type=` enforces the response shape via Pydantic validation (https://ai.pydantic.dev/output/).
- `@agent.output_validator` lets you raise `ModelRetry` and force the model to try again — but it operates only on the structured/typed output and is enforced via a per-run retry budget (https://ai.pydantic.dev/output/, https://ai.pydantic.dev/api/agent/).
- Tools can raise `ModelRetry` for argument-level checks (https://ai.pydantic.dev/tools-advanced/).
- Tool-call approval (`requires_approval=`) is a deterministic human-in-the-loop gate on a per-tool basis (https://ai.pydantic.dev/api/tools/, https://ai.pydantic.dev/).

These are **schema enforcement and retry control**, not content-safety in the Llama Guard / Prompt Guard sense. They don't see the user prompt before the model does, and they don't classify content categories.

**Typical user patterns for actual safety:**
1. **Custom output validator** that calls an external moderation API (OpenAI moderations, Lakera, LLM Guard) — minimal but only catches output, runs after the model call.
2. **Tool gating** — a `prepare` function on `Tool` that filters tool availability based on `RunContext` (https://ai.pydantic.dev/api/tools/).
3. **Third-party capability packages built on the Capabilities API:**
   - `pydantic-ai-guardrails` (https://pypi.org/project/pydantic-ai-guardrails/, https://github.com/jagreehal/pydantic-ai-guardrails) — `GuardedAgent` wrapper with input/output guardrails, llm-guard + autoevals + Guardrails Hub integrations, OpenAI Guardrails-UI config loading, parallel execution.
   - `pydantic-ai-shields` (https://github.com/vstorm-co/pydantic-ai-shields) — `PromptInjection`, `PiiDetector`, `SecretRedaction`, `BlockedKeywords`, `NoRefusals`, `OutputGuard`, `AsyncGuardrail` capabilities passed via `capabilities=[...]`.
4. **NeMo Guardrails / Guardrails AI / Llama Guard via tool call** — wrap the entire model call in your own pipeline outside Pydantic AI.

None of these are first-party. None are mentioned in the official Pydantic AI docs as the canonical answer. *I am unsure* whether any will be brought into core before V2.

## 4. Tools / function calling

**Declaration / registration** (https://ai.pydantic.dev/tools/, https://ai.pydantic.dev/toolsets/):
- `@agent.tool` — decorator, function receives `RunContext` as first arg.
- `@agent.tool_plain` — decorator, no `RunContext`.
- `tools=[fn1, fn2, Tool(fn3, name=..., description=...)]` on the `Agent` constructor.
- `FunctionToolset(tools=[...])` + `toolsets=[...]` on the constructor — first-class collections of tools, can be combined dynamically with `@agent.toolset`.
- Dynamic registration inside a running tool: `toolset.add_function(...)` / `toolset.add_tool(...)`.

Schema is **auto-generated** from function signatures and docstrings via griffe (https://ai.pydantic.dev/tools/). Args validated by Pydantic; `ModelRetry` triggers a retry with feedback to the model.

**Comparison to Llama Stack's `tool_runtime` / `registered_resources.tool_groups`.** Llama Stack treats tool *implementations* as plug-in providers under `providers.tool_runtime` (e.g., `remote::model-context-protocol`, `remote::tavily-search`, `inline::rag-runtime`), and the *named tool groups* the agent can invoke as a separate registered-resources list (`tool_groups:` or `registered_resources.tool_groups:`), each referencing a runtime by `provider_id` (e.g., MCP endpoint URI). The split is intentional: it lets the operator add or remove tool backends without touching application code (https://llamastack.github.io/docs).

Pydantic AI has no equivalent split. Tools are **Python objects/callables**, not configuration. There is one exception that brings configuration-driven extensibility: **capability packages** (`AbstractCapability` subclasses) can be referenced from `AgentSpec` YAML by class name and registered via `custom_capability_types=` on `Agent.from_spec` / `Agent.from_file` (https://ai.pydantic.dev/capabilities/). MCP is exposed this way: in `agent.yaml` you can write `capabilities: [{MCP: {url: https://mcp.example.com/api}}]` — that's the closest analogue to Llama Stack's `tool_groups` entry for an MCP endpoint, and it's the surface you would use if you want a YAML-only tool declaration.

There is no general plugin discovery via entry-points documented in ai.pydantic.dev; you must `pip install` the capability package and pass its class to `custom_capability_types`.

## 5. Mapping table (Llama Stack ↔ Pydantic AI)

| Llama Stack concept | Pydantic AI equivalent / pattern |
|---|---|
| `providers.inference` (named entry with id+type+config) | No direct equivalent. Pattern: `Agent('openai:gpt-5.2')` model string, or explicit `Provider(api_key=…, base_url=…)` + `Model` instance per agent. |
| `providers.safety` / `shields` | No direct equivalent in core. Pattern: `@agent.output_validator` for shape, third-party capability packages (`pydantic-ai-shields`, `pydantic-ai-guardrails`) for content safety. |
| `providers.vector_io` / vector stores | No direct equivalent. Pattern: tool that calls pgvector/Milvus/Qdrant via the user's chosen client; embeddings via Pydantic AI's `embeddings` support or a raw SDK. |
| `providers.tool_runtime` | No direct equivalent as a provider registry. Pattern: tools registered as Python callables; MCP endpoints declared via the `MCP` capability in code or `AgentSpec` YAML. |
| `providers.agents` (Agents API as a provider) | The `Agent` class itself; not a server provider — it's a Python object. No equivalent of swapping the agent runtime via config. |
| `apis: [agents, inference, safety, …]` | No direct equivalent. Pydantic AI has no notion of selectively enabling capability *APIs* — every Agent always supports tools, output validation, etc., as Python APIs. |
| `registered_resources` (models, shields, vector stores) | Partially: model is declared per-Agent in `AgentSpec` (`model:`). Shields/vector stores have no equivalent registry — they're code. |
| `storage` (sqlstore / kvstore) | No equivalent. Pydantic AI itself is stateless per run; durable execution is delegated to **Temporal / DBOS / Prefect** integrations (https://ai.pydantic.dev/api/agent/). Conversation state is the caller's responsibility. |
| `${env.VAR}` env-ref resolution in config | No equivalent in `AgentSpec` YAML. Env vars are read by `Provider` classes at construction time. For YAML-side interpolation you must layer your own loader (e.g., `pydantic-settings` or a manual `os.path.expandvars` pre-pass). |

## 6. Implications for a backend-agnostic operator-facing schema

Replaying the user's example schema:
```yaml
inference:
  providers:
    - type: openai
      api_key_env: OPENAI_API_KEY
      allowed_models: [gpt-4o-mini]
rag:
  providers:
    - type: faiss
      embedding_model: sentence-transformers
safety:
  default_shield: llama-guard
```

### `inference` block — **STABLE-ish across both backends. ~75% confidence.**

This concept maps cleanly to Llama Stack today (`providers.inference` with `provider_type: remote::openai`, `api_key: ${env.OPENAI_API_KEY}`, optionally a `registered_resources.models` allow-list) and to Pydantic AI tomorrow (synthesize a Python `OpenAIProvider(api_key=os.environ["OPENAI_API_KEY"])` and use model strings constrained to `allowed_models`). The single-item-list shape (`providers: [...]`) on the Llama Stack side preserves the model that Llama Stack uses today; on the Pydantic AI side a synthesizer picks the first matching entry per agent. Operators describe "what credentials, what endpoints, what models are allowed" — a vocabulary stable in both worlds.

**Where it breaks:** Llama Stack allows multiple named provider entries of the same API serving different models (e.g., `vllm-inference` and `vllm-safety`); Pydantic AI has no global registry, so naming providers is meaningless until your synthesizer assigns them to specific Agents. Decision: keep the list but treat the entries as available-clients, not as a global registry. Don't expose `provider_id` in the abstract schema unless you also expose agent-to-provider binding.

### `rag` block — **NOT STABLE. ~25% confidence it survives.**

Llama Stack has `providers.vector_io` (with `inline::faiss`, `remote::milvus`, `remote::pgvector`, etc.) plus `registered_resources.vector_stores` and a first-class `/v1/vector_stores` API. Pydantic AI has **none of this**. The official RAG example wires pgvector directly via `asyncpg` (https://github.com/pydantic/pydantic-ai/blob/main/docs/examples/rag.md). A backend-agnostic synthesizer can take your `rag.providers[].type=faiss` and produce a Llama Stack provider entry, but for Pydantic AI the synthesizer would have to *generate code or instantiate a vector-DB client and a tool that uses it* — a much larger gap.

Worse, vocabulary diverges: `embedding_model: sentence-transformers` is a model identifier in Llama Stack (registered under `registered_resources.models` with `model_type: embedding`); in Pydantic AI it would be a parameter to `pydantic_ai.embeddings.SentenceTransformersEmbedder` or similar (the `sentence-transformers` package extra is in `pydantic-ai-slim[sentence-transformers]` — https://pypi.org/project/pydantic-ai-slim/).

**Recommendation:** Until Pydantic AI ships a built-in vector-store abstraction (no public signal it's coming in the next 6–12 months — see §7), keep RAG configuration **under a Llama-Stack-specific subtree**, and on the Pydantic AI side require operators to declare which tool implements retrieval. A minimal portable surface might be `rag.embedding_model:` only — both backends understand "which model to embed with."

### `safety` block — **NOT STABLE. ~20% confidence it survives.**

`default_shield: llama-guard` translates 1:1 to Llama Stack — `providers.safety: - provider_type: inline::llama-guard` + `registered_resources.shields: - shield_id: llama-guard, provider_id: llama-guard`, plus per-agent `input_shields` / `output_shields`. On Pydantic AI, there is no shield concept; the closest path is a third-party capability (`pydantic-ai-shields` provides a `PromptInjection` capability you'd add to `agent.yaml`'s `capabilities:` list), but the actual *model used* is hard-coded inside each capability, and Llama Guard specifically is not a first-class Pydantic AI option.

The vocabulary `default_shield: <name>` makes sense in Llama Stack (where shields are registered resources you can name); it makes no sense in Pydantic AI without inventing a registry layer.

**Recommendation:** Keep `safety.default_shield` under a Llama-Stack-specific subtree. The portable surface is essentially nothing today — at best `safety.enabled: bool` and `safety.fail_closed: bool`.

### Minimum YAML surface stable across both backends

```yaml
# This much survives a Llama Stack → Pydantic AI migration, with caveats:
inference:
  providers:
    - type: openai                      # vendor identifier (stable)
      api_key_env: OPENAI_API_KEY       # env-var name only (synthesizer reads value)
      base_url: https://...             # optional override
      allowed_models: [gpt-4o-mini]
# everything else (rag, safety, storage, tool_runtime, shields, vector_stores)
# goes under a backend-specific block:
backend_specific:
  llama_stack:
    rag: {...}
    safety: {...}
    storage: {...}
  pydantic_ai:
    capabilities: [...]
    spec_overrides: {...}
```

## 7. Pydantic AI's own roadmap / stability signals

**Stability today.** Pydantic AI reached **V1.0.0 on September 4, 2025** with an explicit API-stability commitment: *"V1 means we're committing to API stability: we will not break your code for at least 6 months."* (https://pydantic.dev/articles/pydantic-ai-v1). The version policy adds: *"We will not intentionally make breaking changes in minor releases of V1. V2 will be released in April 2026 at the earliest, 6 months after the release of V1 in September 2025."* (https://ai.pydantic.dev/version-policy/). Current PyPI version is **`pydantic-ai 1.98.0` (May 19, 2026)** and **`pydantic-ai-slim 1.97.0` (May 15, 2026)** (https://pypi.org/project/pydantic-ai/, https://pypi.org/project/pydantic-ai-slim/) — production/stable classification.

As of this report (May 20, 2026), **V2 has not shipped** — the upgrade guide at https://ai.pydantic.dev/changelog/ lists only V1.x breaking changes (which the team explicitly notes were *accidental* leftovers from pre-V1 work, e.g., the Python evaluator removal in #2808 left out of v1.0.0).

**Release cadence.** Weekly to bi-weekly minor releases since V1 (e.g., v1.90.0 on May 4, 2026; v1.91.0; v1.93.0 on May 9; v1.94.0 on May 12; v1.95.0; v1.97.0 on May 15; v1.98.0 on May 19) — https://github.com/pydantic/pydantic-ai/releases. No breaking changes to agent construction, model strings, or `@agent.tool` since 1.0; recent breaking-change entries in the changelog are pre-V1 (the upgrade guide is filtered to historical pre-1.0 churn).

**Recent changes touching agent / provider / tool surface in the last 12 months** (https://ai.pydantic.dev/changelog/, https://github.com/pydantic/pydantic-ai/releases):
- `AgentStreamEvent` expanded to a union — backward compatible (#2689).
- `format_as_xml` import path moved (#2446/#1484) — minor.
- Removal of deprecated `Agent.result_validator`, `AgentRunResult.data`, `Agent.last_run_messages` (#2451).
- `TenacityTransport` now requires `RetryConfig` TypedDict (#2670, #2717).
- v1.94.0 (May 12, 2026): "Drop mistralai as dependency from pydantic-ai by @Kludex in #5384"; OpenAI profile flag for multi-system messages (https://github.com/pydantic/pydantic-ai/releases).
- v1.95.0 / v1.97.0 / v1.98.0: incremental fixes; deprecation of `OutlinesModel` / `OutlinesProvider`, `AGUIApp` / `Agent.to_ag_ui()` in favor of `AGUIAdapter`.

None touched the public Agent constructor signature, the `<provider>:<model>` string convention, the `@agent.tool` decorator, or `AgentSpec`/`from_file`.

**Roadmap signals for built-in safety / RAG / registered-resources.**
- **Guardrails (Issue #1197)** is open with an OpenAI-Agents-SDK-style proposal; not on a stated milestone. *I am unsure* whether it will land before V2.
- No public roadmap document at ai.pydantic.dev mentions a built-in vector-store / RAG abstraction or a Llama-Stack-style provider registry. Thoughtworks Technology Radar **Volume 33** (published November 5, 2025, per PRNewswire) confirms the framework is intentionally narrow: *"Rather than trying to be a Swiss Army knife, PydanticAI offers a lightweight yet powerful approach."* (https://www.thoughtworks.com/radar/languages-and-frameworks/pydantic-ai) — and the same page notes "This blip is not on the current edition of the Radar," indicating the entry was Volume 33 and not Volume 34.
- The big roadmap themes per the v1 launch article (https://pydantic.dev/articles/pydantic-ai-v1) are: durable execution (Temporal/DBOS/Prefect), human-in-the-loop tool approval, MCP/A2A/AG-UI interop, and the Pydantic AI Gateway. **Not** safety shields, not vector stores, not server-side configuration.

**Conclusion (§7):** API stability is high (V1, ~9 months in production, weekly minors, no agent-surface breakage). The library is **intentionally not growing into Llama Stack's territory** on a 6–12 month horizon.

## 8. Recommendation

**Abstract only the inference vocabulary today. Keep RAG, safety, storage, and tool-runtime under backend-specific subtrees. Confidence: ~80%.** Build the single-file schema around an `inference:` block of vendor + endpoint + env-var-name + allow-listed models — that vocabulary maps cleanly to Llama Stack today and to Pydantic AI's Provider + model-string surface tomorrow, and a thin synthesizer per backend covers the gap. Do **not** abstract `rag.*` or `safety.*` into a portable vocabulary right now: Pydantic AI has no built-in vector-store or shield concept, no public roadmap signal that either is coming before V2, and the third-party capability packages that fill the gap (pydantic-ai-shields, pydantic-ai-guardrails) have incompatible vocabularies with each other and with Llama Guard. Park them under `backend_specific.llama_stack.{rag,safety,storage}` and a parallel `backend_specific.pydantic_ai.{capabilities,spec_overrides}`. Treat MCP endpoints as the one tool-runtime concept worth abstracting (~60% confidence) — both backends support MCP natively and the URI + auth-token + allowed-tools fields are stable on both sides. Re-evaluate at every Pydantic AI minor release for: (a) a built-in guardrails API merging Issue #1197, (b) any vector-store abstraction, (c) a server-side / multi-agent config concept. If any of those land, the safety or RAG vocabulary becomes worth abstracting; until then, premature abstraction will cost you more than the duplication.

## Caveats

- **Llama Stack rebrand.** On **April 28, 2026**, the Llama Stack project rebranded to **OGX**, per the official announcement blog post (https://ogx-ai.github.io/blog/from-llama-stack-to-ogx): *"Llama Stack is now OGX. The name changed, but more importantly, so did the mission."* The repo `github.com/llamastack/llama-stack` and the mirror `github.com/meta-llama/llama-stack` redirect there. Latest release tag observed: **v1.0.2 (May 13, 2026)**. The OGX rebrand post also states that **"The project supports 23 inference providers. You can run GPT-4, Claude, Gemini, Mistral, or any model you want behind OGX."** Some templates still ship `image_name:` (legacy) while newer ones use `distro_name:` (PR #4396). The `registered_resources:` block introduced in PR #4600 is the new canonical home for `models / shields / vector_stores / tool_groups / datasets / benchmarks`; older templates with these as bare top-level keys still load.
- **Llama Stack env-var syntax.** Confirmed forms: `${env.VAR}` (required), `${env.VAR:=default}` (with default), `${env.VAR:+value}` (conditional, e.g., enable a provider only when a key is set). The single-colon form `${env.VAR:default}` seen in some older blog posts is **not** the current canonical syntax.
- **Llama Stack source-file paths in the schema reconstruction could not be directly verified** because raw GitHub fetches were blocked during research. The schema sketch is reconstructed from third-party verbatim quotes (Cerebras, Red Hat, Medium) and PR titles. Before locking your schema, fetch `src/llama_stack/core/datatypes.py` (look for `StackRunConfig`) and a current `src/llama_stack/distributions/starter/run.yaml` to confirm.
- **`pydantic-ai-shields` and `pydantic-ai-guardrails` are third-party**, maintained by independent authors (vstorm-co and jagreehal respectively). They are not part of pydantic/pydantic-ai. Treat them as community packages whose APIs may diverge from anything the Pydantic team eventually ships.
- **No Pydantic AI public roadmap doc** was found; conclusions about "not on the roadmap" are inferred from the version-policy doc, the v1 launch post, recent release notes, and the absence of relevant milestones — not from a positive statement that these features are out of scope. Flagged as inference, not fact.
- **Pydantic AI V2 timing.** The version policy says *"V2 will be released in April 2026 at the earliest"*; as of May 20, 2026, V2 has not shipped and the changelog page lists only V1.x entries. The user's stated migration window (2026 or Q1 2027) likely overlaps the V2 release; plan to re-validate this report when V2 ships.