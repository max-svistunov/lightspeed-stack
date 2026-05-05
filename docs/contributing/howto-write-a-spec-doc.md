# How to write a spec doc

A spec doc is the permanent in-repo feature specification.  It is the single
source of truth for what the feature does and how it works.  All implementation
JIRAs reference it.  Agentic coding tools read it for guidance.

**Claude Code shortcut**: `/spec-doc` creates one interactively.

## Configuration

Team-policy conventions for the spec doc shape (which sections are required,
which are `if_applicable`, etc.) live in
[`feature-design.config`](feature-design.config). Personal overrides go in
`.feature-design.config.local` at the repo root (gitignored).

## When to write one

- As part of a spike (see [howto-run-a-spike.md](howto-run-a-spike.md), step 6).
- When a feature is well-understood but not yet documented.
- When an existing feature needs a retroactive spec.

## How to write one

Use [spec-doc-template.md](templates/spec-doc-template.md).

### Location

Place the spec doc at `docs/design/<feature>/<feature>.md`.

### Filling in the template

**What**: Describes the feature.

**Why**: The problem it solves.

**Requirements (Rx)**: Numbered requirements.  For each requirement it should be
easy to provide clear acceptance criteria.

**Use Cases (Ux)**: "As a [role], I want [X], so that [Y]."

**Architecture**: Flow diagram, then subsections for each component.  Include
where things live (file paths), function signatures, schemas, configuration.
Architecture sub-sections marked `if_applicable` in
[`feature-design.config`](feature-design.config) (Trigger mechanism, Storage
/ data model, API changes, etc.) are present only when the feature actually
has the concern. Delete unused sub-sections.

**Acceptance test surface**: Maps each requirement (R1..Rn) to an observable
behavior. This section drives the e2e-kickoff JIRA's feature files: the
person writing `.feature` files reads this section to author Gherkin
scenarios.

**Aspect-specific concerns**: Sections like Latency and Cost, Observability,
Capacity planning, Failure modes, Security considerations,
Migration / backwards compatibility, Telemetry / data privacy, Feature
flags / rollout, Runbook / oncall, Internationalization, API versioning,
Rate limiting / quotas. All `if_applicable` — include only when the
feature genuinely has the concern. Delete sub-sections that don't apply.

**Implementation Suggestions**: File paths, insertion points, code patterns,
test patterns.  Be specific — this section is read by both humans and agentic
coding tools.

**Open Questions**: Things explicitly deferred, and why.  Each item must
trace back to its origin (a spike decision, a PoC finding, or a reviewer
comment) so the rationale survives over time.

**Changelog**: Record significant changes after initial creation.  Date, what
changed, why.

**Appendices**: PoC evidence, API comparisons, reference sources.

### Relationship to the spike doc

The spike doc records everything that was considered.

The spec doc records the approved decisions.

### Keeping it up to date

The spec doc is a living document.  Update it when:
- A decision is changed.
- Implementation reveals something the spec didn't anticipate.
- A reviewer raises a point that changes the design.
