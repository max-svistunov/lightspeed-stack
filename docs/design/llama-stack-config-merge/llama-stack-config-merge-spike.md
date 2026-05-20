# Spike: Llama Stack config merge (unified `lightspeed-stack.yaml`)

## Overview

**The problem**: Operators today must maintain two configuration files —
`lightspeed-stack.yaml` (LCORE settings) and `run.yaml` (Llama Stack
operational config: providers, storage, APIs, safety, registered resources).
This split increases the chance of misconfiguration, makes downstream
deployment templates larger, and forces every Lightspeed team to understand
Llama Stack's internal schema. LCORE-836 asks for a single source of truth.

**The recommendation**: A layered approach — Option C (high-level keys +
`native_override` escape hatch) as the base structure, with Option D
(profiles) enabled as an optional layer on top. See
[Design options A–E](#design-options-ae) for the short names of each
option and [Design alternatives considered](#design-alternatives-considered)
for the scoring.

- **High-level keys** in `lightspeed-stack.yaml` under a new `llama_stack.config`
  section (inference, later storage/safety/...). Most downstream teams write
  only these.
- **`native_override`** escape hatch under the same section — raw Llama Stack
  schema, deep-merged last. Covers anything the high-level schema doesn't
  express.
- **`profile`** field that points to a YAML file used as the baseline — the
  "profiles" feature is mechanism-only; LCORE ships no profiles of its own
  beyond one or two reference examples under `examples/profiles/`.
- **`baseline: default | empty`** selects whether the synthesis starts from
  LCORE's built-in baseline or a blank slate.
- **Legacy mode preserved**: existing `llama_stack.library_client_config_path`
  works unchanged through a deprecation window. Mutual exclusion with the new
  `llama_stack.config` block is enforced at load time.
- **Migration tool**: `lightspeed-stack --migrate-config` produces a unified
  single-file config from an existing (`run.yaml` + `lightspeed-stack.yaml`)
  pair, lossless round-trip.

**PoC validation**: A library-mode PoC proves the mechanism end-to-end.
A unified `lightspeed-stack.yaml` containing only `llama_stack.config`
(no external `run.yaml`) successfully drives LCORE:
liveness/readiness green, `/v1/query` returns a real model response,
`native_override` demonstrably takes effect. Full unit-test suite passes
(2098 tests), including a lossless migrate-then-synthesize round-trip.
Server-mode end-to-end was not re-run through docker-compose — the container
rebuild time was impractical and unrelated to PoC quality (the container image
is ~2 GB of LS dependencies); the same synthesis code path is exercised by the
library-mode PoC and unit tests.

---

## Design options A–E

- **A (Embedded native)** — `llama_stack.config` is the raw Llama Stack
  schema, verbatim. Same surface area downstream teams see today, just
  moved into one file. No abstraction win.
- **B (High-level only)** — `llama_stack.config` exposes only LCORE-defined
  high-level keys (e.g. `inference.providers`). Best UX when every operator
  intent maps cleanly; painful at the edges where the high-level schema
  doesn't yet cover a need (no escape hatch).
- **C (B + `native_override`)** — high-level keys for the common path, plus
  a raw-LS `native_override` block deep-merged last as an escape hatch.
  Combines B's UX with A's flexibility. **Recommended (Decision S1).**
- **D (Profiles)** — a user-authored YAML file pointed to by
  `llama_stack.config.profile: <path>`, used as the synthesis baseline
  instead of LCORE's built-in default. A composable *layer* on top of
  A/B/C, not a standalone shape. LCORE ships the mechanism; downstream
  teams (or operators) author the YAML.
- **E (Kustomize-style patches)** — ship a default baseline; the operator
  writes JSON-Patch-like overlays against it. Viable alternative to C;
  strongest for backward compat with existing `run.yaml` files, weakest
  on validation rigor and dynamic-reconfig fit.

---

## Strategic decisions — for @sbunciak (PM) and @tisnik

These set scope, approach, and rollout shape. Each has a recommendation —
please confirm or override.

### Decision S1: Overall shape

See [Design alternatives considered](#design-alternatives-considered)
for the scoring.

| Option | Standalone shape |
|---|---|
| A (Embedded native) | `llama_stack.config` is raw LS schema, verbatim |
| B (High-level only) | LCORE-defined high-level keys; no escape hatch |
| **C (B + `native_override`)** | High-level keys + raw-LS escape hatch |
| E (Kustomize-style patches) | Default baseline + JSON-Patch-like overlays |

D is not listed because it's a layer that composes on top of any of
A/B/C/E, not a standalone shape — the decision on whether to enable that
layer is Decision T6.

**Recommendation**: **C** as the base structure, with **D** enabled as
an optional layer (feature only, no shipped profiles — see Decision T6).
Best balance of UX, escape-hatch power, validation rigor, and
dynamic-reconfig fit for the broader feature roadmap (LCORE-777/781).

### Decision S2: Deprecation timeline for the legacy path

**Recommendation**: deprecate the legacy two-file path fully by end of Q4;
emit startup deprecation warnings during Q3 and Q4. @sbunciak to confirm
or override the calendar.

### Decision S3: Downstream implications we may not have seen

**Ask**: do we need to account for anything apart from Konflux?
Reviewers from downstream teams should flag any deployment surface that
treats `run.yaml` as a separate artifact (ConfigMap, templated file,
build-time asset) that the unified design would need to accommodate.

### Decision S4: Scope of this spike — what is deliberately left out

The following related work streams are **not** included in this spike and
should be tracked as separate future JIRAs:

- **Llama Stack process supervision** from LCORE (restart-on-crash, signal
  propagation, merged logs). Orthogonal to config merging; covered by
  LCORE-777 / LCORE-778.
- **Hot-reload / dynamic reconfig** (e.g., live `POST /v1/rag` that adds a
  BYOK RAG without restart). Llama Stack does not natively support
  hot-reload; achieving it would require supervision + restart flows.
  Covered by LCORE-781.

**Recommendation**: confirm this scope split. If reviewers want any of the
above pulled in, this spike's JIRAs grow accordingly.

### Decision S5: Where do backend-agnostic high-level keys sit?

**Context**: S1 places the unified config's high-level keys
(`inference.providers` today; later `rag.providers`, etc.) inside the
LS-specific subtree at `llama_stack.config.inference`. LCORE will migrate
from Llama Stack to Pydantic AI over time. Under S1's layout, that
transition would force every downstream team to relearn the config schema —
the `llama_stack` subtree name becomes a lie, and high-level keys would
have to move.

**Recommendation**: lift the backend-agnostic keys to the top level of
`lightspeed-stack.yaml` now. Leave LS-specific knobs under
`llama_stack.config`. Extends S1; does not replace it (Option C + optional
D recommendation stands).

| Today (per S1) | Proposed |
|---|---|
| `llama_stack.config.inference.providers: …` | `inference.providers: …` |
| (future) `llama_stack.config.rag.providers: …` | `rag.providers: …` |
| `llama_stack.config.native_override: …` | unchanged — LS-specific |
| `llama_stack.config.profile: …` | unchanged — LS-specific (points at LS run.yaml shape) |
| `llama_stack.config.baseline: …` | unchanged — LS-specific |

The synthesizer reads `inference.providers` from the top level and emits LS
provider entries exactly as today — only the input node moves. When the
Pydantic AI transition lands, the same top-level schema is consumed by a
new synthesizer that emits Pydantic AI's shape; downstream operators see no
change.

**Scope discipline — what stays under `llama_stack.config`**: anything
whose vocabulary is genuinely LS-specific and unlikely to translate across
backends. Today that's `native_override` (raw LS schema by definition),
`profile` (a YAML file in LS run.yaml shape), `baseline`, and any future
`apis` / `registered_resources` / `vector_io` / `safety`-shield knobs if
they ship as high-level keys before the cutover. We do **not** preemptively
lift these.

**Confidence**: 70%. Reservation: we don't yet know Pydantic AI's provider
vocabulary, so the recommendation will be updated once a research pass on
Pydantic AI's actual config surface lands. Until then, abstract only what
is clearly stable across backends (`inference.providers`: type + creds +
allowed_models; later `rag.providers`); defer broader abstraction.

> **TODO (spike author, post-research)**: determine whether the
> `inference.providers[].type` Literal vocabulary (`openai`, `azure`,
> `sentence_transformers`, …) maps cleanly to Pydantic AI's provider
> declaration, or whether LCORE should define its own canonical type
> vocabulary that each backend-specific synthesizer translates.

**Implementation impact**: if adopted, this changes the scope of the
existing **Unified `llama_stack.config` schema + synthesizer** JIRA — it
ships the top-level shape from day one. No new JIRA is needed.

---

## Technical decisions — for @tisnik and team leads

Architecture-level and implementation-level. Each has a recommendation
grounded in the PoC.

### Decision T1: Format detection (shape vs version field vs both)

How does LCORE tell unified-mode configs from legacy-mode configs?

| Option | Works by |
|---|---|
| Shape only | Presence of `llama_stack.config` → unified; else legacy |
| Version field only | Explicit `config_format_version: 2` required |
| **Both (soft-coupled)** | Shape decides; version field optional but must agree when present |

**Recommendation**: **both, soft-coupled**. Gives a cheap upgrade path for
future real schema bumps without forcing every existing user to add a
version field today. Confidence: 75%.

### Decision T2: Override precedence (inside Option C)

When `llama_stack.config.native_override` overlaps with a high-level key,
what semantics?

| Strategy | Example: `safety: {excluded_categories: [a, b]}` vs override `{excluded_categories: [c]}` |
|---|---|
| Deep-merge, append lists | result: `[a, b, c]` |
| **Deep-merge, replace lists** | result: `[c]`; other keys in `safety` preserved |
| Whole-key override | result: whole `safety` replaced; lose `default_shield_id` unless restated |
| JSON Patch (ops) | explicit — `{op: replace, path: /safety/excluded_categories, value: [c]}` |

**Recommendation**: **deep-merge with list replacement**. Simple mental model,
no list-merge tarpit, keeps scalar + map overrides minimal. Implemented in
`deep_merge_list_replace()`. Confidence: 70%.

See [Merge semantics — worked examples](#merge-semantics--worked-examples).

### Decision T3: Secrets in synthesized files

The synthesized run.yaml lives on disk (library mode: `$TMPDIR`; server mode:
inside the LS container). Option space:

| Option | On-disk content |
|---|---|
| **Keep env-var refs verbatim** | `api_key: ${env.OPENAI_API_KEY}` (resolved by LS at start) |
| Resolve before writing | `api_key: sk-...` |

**Recommendation**: **keep env-var refs verbatim**. Security-leaning default;
resolved secrets never touch the disk. Implemented in
`apply_high_level_inference` (emits `${env.<NAME>}` strings). Confidence: 95%.

### Decision T4: Synthesized file location

Where the synthesized `run.yaml` goes at runtime:

| Option | Path |
|---|---|
| Temp file | `$TMPDIR/llama_stack_synthesized_config.yaml` |
| **Persistent known path** | Local: `./.generated/run.yaml` or `~/.local/state/lightspeed-stack/run.yaml`; Container: `/app-root/.generated/run.yaml`. Overwrite on each boot. |

**Recommendation**: **persistent known path, overwrite on boot**. Debuggable,
no stale-file risk (always overwritten before LS starts). The PoC used
`$TMPDIR` for expediency; production should use the persistent path. CLI flag
`--synthesized-config-output <path>` for debugging. Confidence: 85%.

### Decision T5: Migration tool invocation

How operators invoke the migration tool:

| Option | Example |
|---|---|
| Separate script under `scripts/` | `uv run python scripts/migrate-config.py ...` |
| **Flag on main entry point** | `lightspeed-stack --migrate-config --run-yaml X -c Y --migrate-output Z` |
| Subcommand refactor | `lightspeed-stack migrate-config ...` (BREAKS existing invocations) |

**Recommendation**: **flag on main entry point**. Parallels the existing
`--dump-configuration` / `--dump-schema` flags; zero breaking change to
existing invocations. Implemented in `src/lightspeed_stack.py` + a
companion `migrate_config_dumb()` function. Confidence: 90%.

### Decision T6: Profile distribution

How profiles (Option D layer) reach downstream teams:

| Option | Details |
|---|---|
| Ship named profiles in `src/profiles/` | LCORE ships a pre-curated set; `profile: openai-remote` resolves |
| **Feature only, no shipped profiles** | `profile: <path-to-file>` is the only invocation; teams author their own; LCORE ships 1–2 reference examples under `examples/profiles/` |

**Recommendation**: **feature only, no shipped profiles**. Avoids
profile-sprawl and the burden of keeping "blessed" profiles in sync with
downstream products. 1–2 reference examples in `examples/profiles/` are
documentation, not shipped runtime assets. Confidence: 85%.

### Decision T7: The `baseline` field (added during PoC)

The migration tool must be lossless: migrate an existing `run.yaml` into a
unified config, then synthesize it back to a `run.yaml`, and the result
must match the original byte-for-byte. The PoC surfaced a leak: when
`native_override` contains the entire `run.yaml` body, LCORE's built-in
baseline still deep-merges underneath and adds keys that weren't in the
original. Fix: a `baseline: "default" | "empty"` field that lets the
caller pick the synthesis starting point.

- `baseline: default` (default value) — start from LCORE's built-in baseline.
- `baseline: empty` — start from `{}`. Used by the dumb migration tool, so
  that `native_override` is the only thing the synthesizer sees.

**Recommendation**: **accept this field, with `default` as the default value**.
That preserves the zero-config "fresh user authors `llama_stack.config` and
gets a working LS baseline" UX; the migration tool sets `baseline: empty`
explicitly so the migrate-then-synthesize loop above matches the original
`run.yaml`. Alternatives (`inherit_defaults: bool`, `starting_point: ...`)
are cosmetic. Confidence: 80%.

### Decision T8: Konflux pipelines — for @radofuchs

The `.tekton/` directory in this repo holds Konflux build-pipeline
definitions. If any pipeline template mounts `run.yaml` separately, unified
mode needs that pipeline to either (a) keep using legacy mode during the
deprecation window, or (b) mount the unified `lightspeed-stack.yaml` and
drop the `run.yaml` mount.

**Ask**: @radofuchs to confirm current Konflux pipeline shape and plan
migration.

---

## Proposed JIRAs

Each JIRA's agentic-tool instruction points to the spec doc
(`llama-stack-config-merge.md`), the permanent reference. The first JIRA
(authoring e2e feature files) is the intentional kickoff — it happens
before feature implementation so the test shape is not influenced by
implementation choices.

<!-- type: Story -->
<!-- key: LCORE-???? -->
### LCORE-???? E2E feature files for unified mode (no step implementation)

**User story**: As a Lightspeed Core e2e engineer, I want the behave
feature files for unified-mode scenarios written before the feature
implementation lands, so that the test shape reflects the feature's
intended behavior rather than the chosen implementation, and any
architectural gaps surface early.

**Description**: Author behave `.feature` files under `tests/e2e/features/`
that describe the behaviors required of unified mode. Step definitions
(Python glue) are explicitly **not** part of this ticket — they are
covered by a later sibling ticket (LCORE-???? — Implement step
definitions). The feature files can be submitted for review and land
before implementation of the feature itself begins.

**Scope**:

- `.feature` files covering, at minimum, these R1–R11 surfaces from the
  spec doc:
  - Boot LCORE with unified `lightspeed-stack.yaml` (no external
    `run.yaml`); `/liveness`, `/readiness`, and `/v1/query` succeed.
  - Boot LCORE with legacy config
    (`library_client_config_path` + external `run.yaml`); same result.
  - Setting both `llama_stack.config` and
    `llama_stack.library_client_config_path` fails at config-load time
    with a clear error that mentions `--migrate-config`.
  - Migration tool: `lightspeed-stack --migrate-config ...` produces a
    unified file that drives equivalent Llama Stack behavior.
  - `native_override` deep-merges onto the baseline with list
    replacement (tested on a scalar key and a list key).
  - `profile:` path (absolute and relative-to-config-dir) loads the
    referenced baseline.
  - Secrets appear as `${env.FOO}` references in the synthesized
    `run.yaml` on disk; never resolved to raw values.
  - Legacy mode emits a one-line deprecation WARN at startup; unified
    mode does not.
- Additions to `tests/e2e/test_list.txt` so behave discovers the new
  files.
- Gherkin scenarios authored from the spec doc (`R1..R11`) only; author
  must avoid reading the implementation JIRAs' scope sections while
  drafting scenarios.

**Acceptance criteria**:

- behave parses every new `.feature` file without syntax errors.
- behave marks all new scenario steps as `undefined` (step definitions
  land in LCORE-????).
- `uv run make test-e2e` remains green (new scenarios are skipped or
  reported undefined, not failing).
- Any ambiguity or architectural tension uncovered while authoring is
  captured either as a comment in the spec doc or as a new sub-JIRA.

**Blocks**: LCORE-???? (Implement behave step definitions for unified
mode).

**Agentic tool instruction**:

```text
Read "Requirements" (R1..R11) and "Use Cases" in
docs/design/llama-stack-config-merge/llama-stack-config-merge.md.
Do NOT read the other JIRAs' scope sections or the synthesizer/schema
implementation code while authoring; the point of this ticket is to
produce feature files uncontaminated by implementation detail.
Key files to create: tests/e2e/features/unified-mode-*.feature plus
additions to tests/e2e/test_list.txt. Do NOT create step definitions in
tests/e2e/features/steps/.
To verify: `uv run behave --dry-run tests/e2e/features/unified-mode-*.feature`
parses successfully; `uv run make test-e2e` still green with the new
scenarios reported as undefined.
```

<!-- type: Task -->
<!-- key: LCORE-???? -->
### LCORE-???? Unified `llama_stack.config` schema + synthesizer

**Description**: Implement the unified-mode config schema
(`UnifiedLlamaStackConfig`, `UnifiedInferenceSection`,
`UnifiedInferenceProvider`) and the synthesizer that produces a full Llama
Stack `run.yaml` from it. Wire library mode to the synthesizer. Preserve
legacy mode through mutual-exclusion validation.

**Scope**:

- New Pydantic classes in `src/models/config.py`.
- New functions in `src/llama_stack_configuration.py`:
  `synthesize_configuration`, `deep_merge_list_replace`,
  `apply_high_level_inference`, `load_default_baseline`, `synthesize_to_file`.
- A shipped default baseline at `src/data/default_run.yaml`.
- Library-mode wiring in `src/client.py`: detect unified vs legacy, write
  synthesized file, pass path to library client.
- Cross-field validation: reject both `config` and
  `library_client_config_path` set simultaneously.
- Legacy behavior (`llama_stack.library_client_config_path` path) unchanged.

**Acceptance criteria**:

- Unified `lightspeed-stack.yaml` (no external `run.yaml`) boots LCORE in
  library mode and serves `/v1/query`.
- Legacy configs continue to work with no change.
- Mutual-exclusion error message fires cleanly when both forms are set.
- Unit tests for synthesizer, merge semantics, schema validation.

**Agentic tool instruction**:

```text
Read the "Architecture" and "Implementation Suggestions" sections of
docs/design/llama-stack-config-merge/llama-stack-config-merge.md.
Key files to create or modify:
  src/models/config.py  (new classes; modify LlamaStackConfiguration)
  src/llama_stack_configuration.py  (synthesize_configuration + helpers)
  src/data/default_run.yaml  (new)
  src/client.py  (library-mode wiring)
To verify: run a unified-mode config end-to-end via `uv run lightspeed-stack -c <config>` and confirm /v1/query succeeds.
```

<!-- type: Task -->
<!-- key: LCORE-???? -->
### LCORE-???? Migration tool — dumb-mode lift-and-shift

**Description**: Implement `--migrate-config` on the `lightspeed-stack` CLI
that produces a unified single-file config from an existing
(`run.yaml` + `lightspeed-stack.yaml`) pair. Dumb mode places the entire
`run.yaml` body under `llama_stack.config.native_override` with
`baseline: empty`, removes `library_client_config_path`.

**Scope**:

- `migrate_config_dumb()` function in `src/llama_stack_configuration.py`.
- `--migrate-config`, `--run-yaml`, `--migrate-output` flags in
  `src/lightspeed_stack.py`.
- Round-trip test: migrate → synthesize → byte-identical to original
  `run.yaml`.

**Acceptance criteria**:

- `lightspeed-stack --migrate-config --run-yaml X -c Y --migrate-output Z`
  produces a unified config that boots LCORE in library mode to the same
  Llama Stack behavior as the original pair.
- Round-trip unit test passes.
- `--help` describes the flag clearly.

**Agentic tool instruction**:

```text
Read "Migration tool" in docs/design/llama-stack-config-merge/llama-stack-config-merge.md.
Key files: src/lightspeed_stack.py, src/llama_stack_configuration.py,
tests/unit/test_llama_stack_synthesize.py.
To verify: migrate the repo's root run.yaml + lightspeed-stack.yaml, then
start LCORE with the output; confirm /v1/query works.
```

<!-- type: Task -->
<!-- key: LCORE-???? -->
### LCORE-???? LS container entrypoint + deployment artifacts for unified mode

**Description**: Update the Llama Stack container entrypoint and deployment
manifests so server mode works end-to-end from a unified
`lightspeed-stack.yaml`. Rebuild guidance for container images that bundle
the synthesizer script and default baseline.

**Scope**:

- Update `scripts/llama-stack-entrypoint.sh` — the existing script already
  defers to the Python CLI for auto-detection; document that behavior.
- Update `test.containerfile` to copy `src/data/` into the LS container so
  `load_default_baseline()` resolves.
- Provide a unified-mode `docker-compose.yaml` (or update the existing one)
  that mounts only `lightspeed-stack.yaml` into the LS container.
- Update `.tekton/` pipelines as needed (coordinate with pipeline owner,
  see Decision T8).

**Acceptance criteria**:

- `docker compose up` with a unified `lightspeed-stack.yaml` starts both
  containers healthy; `/v1/query` works through LCORE → LS.
- Legacy docker-compose layout (with external `run.yaml` mount) still works.

**Agentic tool instruction**:

```text
Read "Architecture → Server mode" in docs/design/llama-stack-config-merge/llama-stack-config-merge.md.
Key files: scripts/llama-stack-entrypoint.sh, test.containerfile,
docker-compose.yaml, .tekton/*.yaml.
To verify: docker compose up with the unified config; curl LCORE /v1/query.
```

<!-- type: Story -->
<!-- key: LCORE-???? -->
### LCORE-???? Migrate in-repo e2e / integration test configurations

**User story**: As a Lightspeed Core maintainer, I want the in-repo e2e and
integration tests to use the unified-mode config format, so that the
reference configuration shapes downstream teams see are the new ones.

**Description**: Convert `tests/e2e/configs/run-*.yaml` and
`tests/e2e/configuration/**/lightspeed-stack*.yaml` into unified form
(or delete the `run-*.yaml` side and fold the content into the
corresponding `lightspeed-stack*.yaml`). Migrate `tests/e2e-prow/rhoai/`
configs similarly.

**Scope**:

- Identify every test config that references `run.yaml`.
- Mechanically migrate using the migration tool (dumb mode).
- Re-run the full e2e suite and resolve any differences.

**Acceptance criteria**:

- No in-repo test config references an external `run.yaml`.
- `uv run make test-e2e` passes.
- Existing test coverage is preserved (no tests deleted solely to make the
  migration pass).

**Agentic tool instruction**:

```text
Read "Migration paths" in docs/design/llama-stack-config-merge/llama-stack-config-merge.md.
Key files: tests/e2e/configs/, tests/e2e/configuration/, tests/e2e-prow/rhoai/.
To verify: `uv run make test-e2e` green.
```

<!-- type: Task -->
<!-- key: LCORE-???? -->
### LCORE-???? Implement behave step definitions for unified-mode feature files

**Description**: Implement the Python step definitions
(`@given`/`@when`/`@then` functions) under `tests/e2e/features/steps/`
for the `.feature` files authored in LCORE-???? (E2E feature files
kickoff). After this ticket lands, the scenarios transition from
`undefined` to fully executing.

The feature files are taken as-is — do not modify the Gherkin to make
implementation easier. If a scenario cannot be implemented faithfully,
raise it against the spec doc (and possibly back to LCORE-???? kickoff)
rather than quietly weakening the test.

**Scope**:

- Step definitions for every step pattern in the new `.feature` files.
- Fixtures or helpers under `tests/e2e/features/steps/` as needed
  (e.g., temp-dir config authoring, subprocess start/stop for LCORE,
  HTTP client helpers reusing existing `tests/e2e/` patterns).
- CI wiring so the new scenarios run as part of `uv run make test-e2e`.

**Acceptance criteria**:

- behave reports zero `undefined` steps across the new `.feature`
  files.
- `uv run make test-e2e` runs the new scenarios and they pass.
- No Gherkin edit was made to accommodate implementation constraints
  (or if any edit was made, it is documented in a PR comment with
  explicit rationale).

**Blocked by**:

- LCORE-???? (E2E feature files for unified mode — the `.feature`
  files being implemented against).
- LCORE-???? (Unified schema + synthesizer), LCORE-????
  (Migration tool), LCORE-???? (LS container entrypoint + deployment)
  — the feature under test must exist.

**Agentic tool instruction**:

```text
Read "Architecture" and "Requirements" in
docs/design/llama-stack-config-merge/llama-stack-config-merge.md.
Key files to create: tests/e2e/features/steps/unified-mode*.py (or
extend existing step-definition modules if patterns reuse cleanly).
Do not modify tests/e2e/features/unified-mode-*.feature — take the
Gherkin as-is. If a scenario genuinely cannot be implemented faithfully,
file a sub-ticket rather than changing the Gherkin quietly.
To verify: `uv run make test-e2e` runs every new scenario green and
behave reports zero undefined steps.
```

<!-- type: Story -->
<!-- key: LCORE-???? -->
### LCORE-???? Docs migration to unified mode as primary

**User story**: As an operator reading Lightspeed Core docs, I want the
single-file unified configuration to be the primary way documented, with
legacy mode clearly marked as a deprecation path.

**Description**: Update
`docs/deployment_guide.md`, `docs/byok_guide.md`, `docs/okp_guide.md`,
`docs/rag_guide.md`, `docs/providers.md`, `docs/config.md`, `README.md`,
`docs/local-stack-testing.md` to document unified mode as primary. Add a
migration section with the migration tool command. Clean up the stale
`create_argument_parser` docstring in `src/lightspeed_stack.py` that still
mentions the removed `-g/-i/-o` flags.

**Scope**:

- Each doc file touched.
- A new migration section (step-by-step).
- Update the `create_argument_parser` docstring in
  `src/lightspeed_stack.py`.

**Acceptance criteria**:

- Every doc page that showed a two-file setup also shows the unified-mode
  equivalent.
- Migration tool invocation documented with a worked example.
- `docs/openapi.md` / `docs/config.html` regenerated.

**Agentic tool instruction**:

```text
Read "Deprecation timeline" and "Migration paths" in docs/design/llama-stack-config-merge/llama-stack-config-merge.md.
Key files: docs/*.md, docs/*.html, docs/*.json, README.md, src/lightspeed_stack.py docstring.
To verify: rendered docs present the unified mode first; legacy mode is visibly deprecated.
```

<!-- type: Task -->
<!-- key: LCORE-???? -->
### LCORE-???? Reference profile examples and profile-path doc

**Description**: Add `examples/profiles/` with two reference profile YAML
files — one remote-provider (OpenAI) and one inline-provider (sentence-
transformers + FAISS) — purely as reference material. Document how operators
write and reference their own profiles via
`llama_stack.config.profile: <path>`.

**Scope**:

- `examples/profiles/openai-remote.yaml`
- `examples/profiles/inline-faiss.yaml`
- Docs section: how to author a profile, where to place it, how to
  reference it from `lightspeed-stack.yaml`.

**Acceptance criteria**:

- Both examples load cleanly via the synthesizer (sanity test).
- A docs section titled "Profiles" exists and has a worked example.

**Agentic tool instruction**:

```text
Read "Profiles" in docs/design/llama-stack-config-merge/llama-stack-config-merge.md.
Key files to create: examples/profiles/*.yaml, a "Profiles" section in docs/config.md or docs/deployment_guide.md.
To verify: load the example via `uv run lightspeed-stack -c <wrapper.yaml>` referencing the profile; confirm LS boots.
```

<!-- type: Task -->
<!-- key: LCORE-???? -->
### LCORE-???? Deprecation warning for legacy mode

**Description**: After the unified-mode feature lands (one release later),
emit a one-line startup WARN when `library_client_config_path` is set. Link
to the migration doc. Legacy mode continues to fully function.

**Scope**:

- Warning emission point: on load in `LlamaStackConfiguration`
  `check_llama_stack_model` validator, or at LCORE startup.
- Log line format includes a stable URL fragment to the migration doc.

**Acceptance criteria**:

- Legacy configs still load and run.
- A single WARN line appears at startup when legacy fields are used.
- The warning is not emitted in unified mode.

**Agentic tool instruction**:

```text
Read "Deprecation timeline" in docs/design/llama-stack-config-merge/llama-stack-config-merge.md.
Key files: src/models/config.py (or src/lightspeed_stack.py startup).
To verify: run LCORE with a legacy config; confirm WARN line; run with unified config; confirm no WARN.
```

---

## PoC results

### What the PoC does

The PoC proves the mechanism end-to-end in library mode: a unified config
works with `native_override` and a `profile:` baseline. Server-mode
end-to-end validation was skipped — same synthesis code path, container
rebuild time was impractical.

**Important**: The PoC diverges from the production design in these ways:

- Uses `$TMPDIR` for the synthesized `run.yaml` instead of the persistent
  known path recommended in Decision T4.
- No `--synthesized-config-output` CLI flag yet.
- The migration tool ships only the "dumb" mode (lift the whole `run.yaml`
  into `native_override`). The "smart" mode that factors an existing
  `run.yaml` into high-level keys is deliberately deferred to future work;
  it is captured under the spec doc's "Open Questions for Future Work" and
  is not part of the proposed implementation JIRAs.
- No deprecation warning yet (that's its own JIRA).
- The high-level inference parser writes `provider_id` straight from the
  `type:` Literal value (e.g. `sentence_transformers`, with an underscore).
  The shipped baseline `run.yaml` and the wider LS ecosystem refer to that
  same provider by the hyphenated name (`sentence-transformers`). When both
  are present the two IDs don't match, so baseline references to the
  embedder break. The PoC sidestepped the collision by using
  `baseline: default` plus a `native_override` block — not high-level
  inference — for the validation run. Fix before production: hyphenate the
  emitted `provider_id` so it matches the ecosystem convention used in
  baselines (or, equivalently, alias the Literal value at emit time).

### Results

Full evidence bundle for the library-mode PoC (paths relative to this doc):

- [`poc-results/lightspeed-stack-unified-library.yaml`](poc-results/lightspeed-stack-unified-library.yaml)
  — the unified-mode config used.
- [`poc-results/library-mode/synthesized-run.yaml`](poc-results/library-mode/synthesized-run.yaml)
  — what LCORE produced (3.7 KB).
- [`poc-results/library-mode/query-response.json`](poc-results/library-mode/query-response.json)
  — a real `/v1/query` round-trip.
- [`poc-results/library-mode/README.md`](poc-results/library-mode/README.md)
  — walkthrough.

Summary of validation:

| Check | Evidence |
|---|---|
| Liveness 200 | `curl /liveness` → `{"alive":true}` |
| Readiness 200 | `curl /readiness` → `{"ready":true,"reason":"All providers are healthy","providers":[]}` |
| `/v1/query` works | `{"response":"The three primary colors are red, blue, and yellow.",...}` |
| Profile loaded | `profile: /.../tests/e2e/configs/run-ci.yaml` resolved |
| `native_override` took effect | `safety.default_shield_id: llama-guard` in synthesized output |
| No external `run.yaml` needed | No `library_client_config_path` in config |
| Secrets preserved as env refs | `api_key: ${env.OPENAI_API_KEY}` in synthesized file |
| Full unit suite | 2098 passed, 1 skipped, 0 failed |
| Round-trip lossless | `test_migrate_then_synthesize_reproduces_run_yaml` green |

### Findings discovered during PoC

- **`AsyncLlamaStackAsLibraryClient` takes a file path, not a dict.** The
  initial design assumed we could pass the synthesized configuration to the
  library client in memory and avoid touching the filesystem. In practice
  `llama_stack.core.library_client.AsyncLlamaStackAsLibraryClient` accepts
  only a string path (or, in newer versions, a `StackRunConfig` object that
  is itself built from a parsed YAML file). There is no dict-only entry
  point in the public API. Consequences for the implementation:
  - Library mode **must** write the synthesized `run.yaml` to disk before
    constructing the client (R10 in the spec doc — persistent known path,
    overwritten each boot).
  - The disk-write step is the same shape as server mode's, so the two
    paths can share `synthesize_to_file()`.
  - Any future "dict-only" optimization would require an upstream
    Llama Stack API addition; not worth pursuing.
- **`profile:` path resolution** uses the directory of the
  `lightspeed-stack.yaml`. Relative paths work only when the profile is
  co-located with the LCORE config. Absolute paths always work. Spec doc
  recommends documenting this clearly.
- **Default baseline requires `EXTERNAL_PROVIDERS_DIR`**. `src/data/default_run.yaml`
  (copied from the repo's `run.yaml`) references `${env.EXTERNAL_PROVIDERS_DIR}`
  without a default. Either ship a thinner default baseline, or change the
  reference to `${env.EXTERNAL_PROVIDERS_DIR:=~/.llama/providers.d}`. Flagging
  for the implementation JIRA.
- **High-level inference naming collision** (described above in "divergence
  from production design").

---

## Background sections

### Current architecture

Two files:

- **`lightspeed-stack.yaml`** — LCORE settings: service host/port, auth,
  conversation cache, user data collection, MCP servers, authentication,
  authorization, quota, etc. Also contains `llama_stack:` with
  connection-to-LS settings (URL/api_key or library-client mode with a path
  to an external `run.yaml`).
- **`run.yaml`** — Llama Stack operational config: `apis`, `providers`
  (inference, safety, tool_runtime, vector_io, agents, ...), `storage`,
  `registered_resources`, `vector_stores`, `safety`.

**Existing enrichment** (`src/llama_stack_configuration.py`):

- LCORE already enriches an input `run.yaml` with dynamic values from
  `lightspeed-stack.yaml`: Azure Entra ID tokens (side-effect to `.env`),
  BYOK RAG entries, Solr/OKP provider/store/model registration. Output is
  an enriched `run.yaml`.
- Called in two places: `scripts/llama-stack-entrypoint.sh` at LS container
  boot (server mode) and `src/client.py:_enrich_library_config()` (library
  mode).
- LCORE-779 made this automatic; LCORE-518 (closed spike) proved (re)generation
  feasibility. Both are the groundwork the current spike builds on.

The new synthesizer *subsumes* the enrichment: it builds the full run.yaml
(baseline + enrichment + high-level + native_override) rather than
incrementally enriching an existing one.

### Design alternatives considered

This section scores the five design alternatives (A, B, C, D, E)
against the attributes that matter for LCORE-836. Each cell is a 1–5
rating; higher is better for that attribute. Cells marked with **★** in
the attribute name carry more weight in the final choice. The
recommendation that comes out of these scores is C as the base shape
with D enabled as an optional layer (Decision S1 + Decision T6). For the
option short names see [Design options A–E](#design-options-ae).

Attribute definitions (★ = high-weight for LCORE-836):

- **★ Operator UX** — how little raw LS schema a typical operator must
  read or write to express common intents (one provider, one safety
  filter, default storage). High = the high-level keys cover almost
  everything; low = operators must hand-author LS provider blocks.
- **Abstraction cleanliness** — how well the LCORE-facing schema hides
  internal LS shape. High = LCORE owns a stable surface that survives
  LS schema bumps; low = LCORE just relays LS schema verbatim.
- **LS schema resilience** — how exposed downstream operators are to
  Llama Stack schema churn. High = high-level keys absorb upstream
  renames/restructures inside LCORE; low = every LS change is a
  breaking change downstream.
- **★ Escape-hatch power** — coverage when the high-level schema
  doesn't yet express something the operator needs (e.g. an obscure
  provider config). High = the operator can drop in raw LS YAML without
  blocking; low = the operator is stuck waiting for LCORE to add
  first-class support.
- **Implementation cost** — engineering work to ship the option (one
  release scope). High = small change; low = significant new code +
  tests + docs.
- **Maintenance load** — ongoing burden after ship (per release).
  High = touches one place; low = many surfaces to keep in sync
  (high-level keys, baselines, examples, migration tool).
- **★ Backward compatibility** — how cleanly the option lets legacy
  two-file configs keep working through a deprecation window without
  duplicate code paths. High = legacy path stays intact while unified
  path adds on; low = the option forces an early breaking change.
- **Validation rigor** — strength of static + load-time checks LCORE
  can run against the operator's config. High = Pydantic + cross-field
  validators catch most mistakes; low = errors only surface when LS
  itself fails to start.
- **★ Dynamic-reconfig fit** — how well the option composes with the
  feature roadmap that wants to change LS config at runtime
  (LCORE-777/781, BYOK RAG additions). High = the synthesized config is
  a single dict the supervisor can recompute and reload; low = the
  shape forces in-place file edits.
- **★ Library+server parity** — whether the same operator-facing
  config drives both library-mode and server-mode LS without separate
  configurations. High = one file, two modes; low = needs mode-specific
  variants.
- **Provider plurality** — how many of the LS provider types the option
  covers without an escape hatch. High = all common types reachable via
  the option's normal surface; low = the option only covers one or two.
- **Testability** — ease of writing automated tests against the
  option's surface. High = small, deterministic inputs map to a single
  dict output; low = templated/inherited shapes that need integration
  tests to exercise.

| Attribute | A | B | C | D | E |
|---|---|---|---|---|---|
| ★ Operator UX | 2 | 5 | **4** | 5 | 3 |
| Abstraction cleanliness | 1 | 4 | 3 | 4 | 2 |
| LS schema resilience | 1 | 4 | 3 | 3 | 2 |
| ★ Escape-hatch power | 5 | 1 | 5 | 5 | 5 |
| Implementation cost | 4 | 2 | 2 | 3 | 3 |
| Maintenance load | 2 | 3 | 3 | 2 | 3 |
| ★ Backward compatibility | 3 | 3 | 3 | 3 | 4 |
| Validation rigor | 2 | 5 | 4 | 3 | 2 |
| ★ Dynamic-reconfig fit | 2 | 5 | 4 | 4 | 2 |
| ★ Library+server parity | 5 | 4 | 4 | 5 | 5 |
| Provider plurality | 5 | 4 | 5 | 4 | 5 |
| Testability | 3 | 4 | 3 | 5 | 3 |

The recommendation column is **C**: it ties or beats every other
standalone option on the high-weight attributes except Operator UX,
where it costs one point against B (because the escape hatch adds
schema surface area) — a trade we accept to keep escape-hatch power at
5. D layered on top of C adds testability and a clean path for
deployment-team-authored baselines without changing C's structure.

### Merge semantics — worked examples

Given the baseline:

```yaml
safety:
  default_shield_id: llama-guard
  excluded_categories: [violence, sexual_content]
providers:
  inference:
    - provider_id: openai
      provider_type: remote::openai
      config:
        api_key: ${env.OPENAI_API_KEY}
```

And `native_override`:

```yaml
safety:
  excluded_categories: [spam]
```

**Deep-merge-with-list-replacement (chosen)** produces:

```yaml
safety:
  default_shield_id: llama-guard          # preserved (not in override)
  excluded_categories: [spam]             # list replaced
providers:                                # not in override — preserved
  inference:
    - provider_id: openai
      ...
```

The recommendation's appeal: to keep `default_shield_id`, the user doesn't
have to restate it. To replace `excluded_categories`, the user provides the
new list — they don't need to know a patch syntax.

### Process-model recap (no LCORE supervision of LS)

**Library mode**: LCORE process embeds the Llama Stack library client. LCORE
synthesizes `run.yaml` to a file, calls `AsyncLlamaStackAsLibraryClient(path)`,
initializes, serves. One process.

**Server mode**: Llama Stack runs as a separate process (container). LCORE
connects to it over HTTP. Under unified mode, the LS container's entrypoint
reads the mounted `lightspeed-stack.yaml`, the Python CLI auto-detects
unified mode, synthesizes `run.yaml`, then `exec llama stack run` with it.
LCORE container reads the same `lightspeed-stack.yaml`, ignores the
`config` sub-block (server mode — only connection fields matter), connects.
Two processes. LCORE does **not** start, monitor, or supervise the LS
process — the orchestrator (docker-compose, systemd, k8s) does. Supervision
is out of scope for this spike (see Decision S4).

### What must not break during rollout

See [Backward compatibility scope](#backward-compatibility-scope). The four
must-not-break surfaces:

1. Existing `lightspeed-stack.yaml` with `library_client_config_path`.
2. Existing `run.yaml` content, including fields LCORE doesn't model.
3. Existing CI/CD templating that treats `run.yaml` as a separate artifact.
4. Existing enrichment behavior (Azure Entra ID, BYOK RAG, Solr/OKP).

### Backward compatibility scope

Detection rule at load time:

| `lightspeed-stack.yaml` shape | Interpretation |
|---|---|
| `llama_stack.library_client_config_path` set, no `llama_stack.config` | **Legacy** — today's behavior |
| `llama_stack.config.*` present | **Unified** — new path |
| Both present | Error at load time — clear message |
| Neither (remote URL only, no config) | Existing remote mode — unchanged |

Three migration paths operators can choose:

| Path | Effort | Result |
|---|---|---|
| Do nothing | 0 | Legacy keeps working until deprecation window closes |
| Lift-and-shift (via migration tool) | seconds | Unified single file, zero semantic change |
| Re-express | hours+ | Unified single file, fully adopts the high-level schema |

---

## Appendix A — Files changed in the PoC

Relative to `upstream/main`:

| File | Purpose |
|---|---|
| `src/models/config.py` | New classes: `UnifiedInferenceProvider`, `UnifiedInferenceSection`, `UnifiedLlamaStackConfig`; modified `LlamaStackConfiguration` (adds `config` field + mutual-exclusion validator) |
| `src/llama_stack_configuration.py` | New: `synthesize_configuration`, `deep_merge_list_replace`, `apply_high_level_inference`, `load_default_baseline`, `synthesize_to_file`, `migrate_config_dumb`. CLI `main()` auto-detects unified vs legacy. |
| `src/data/default_run.yaml` | Built-in default baseline (copied from repo root `run.yaml` for the PoC — implementation JIRA should slim it down; see PoC surprise about `EXTERNAL_PROVIDERS_DIR`) |
| `src/client.py` | Library-mode path picks synthesis for unified configs, enrichment for legacy |
| `src/lightspeed_stack.py` | `--migrate-config`, `--run-yaml`, `--migrate-output` flags |
| `scripts/llama-stack-entrypoint.sh` | Comment updated — script itself needs no change (Python CLI auto-detects) |
| `test.containerfile` | Copies `src/data/` into the LS container |
| `tests/unit/test_llama_stack_synthesize.py` | 22 new tests: merge semantics, high-level inference, synthesize pipeline, migration round-trip |
| `tests/unit/models/config/test_llama_stack_configuration.py` | 3 new tests: unified/legacy mutual exclusion |
| `tests/unit/models/config/test_dump_configuration.py` | 5 expected-dict updates (new `config: None` field appears in dumps) |
| `tests/unit/test_client.py` | Error-message regex updated |
| `docs/design/llama-stack-config-merge/` | Spike doc, spec doc, PoC evidence, proposed JIRAs |

## Appendix B — Commands to reproduce the library-mode PoC

```bash
# 1. Start LCORE in library mode with a unified config
export OPENAI_API_KEY=<your-key>
export E2E_OPENAI_MODEL=gpt-4o-mini
mkdir -p /tmp/lcore-836-poc
uv run lightspeed-stack \
  -c docs/design/llama-stack-config-merge/poc-results/lightspeed-stack-unified-library.yaml

# 2. In another shell — query
curl -s http://localhost:8080/liveness
curl -s http://localhost:8080/readiness
curl -s -X POST http://localhost:8080/v1/query \
  -H 'Content-Type: application/json' \
  -d '{"query": "Name three primary colors. One sentence."}'

# 3. Inspect what was synthesized
cat /tmp/llama_stack_synthesized_config.yaml
```
