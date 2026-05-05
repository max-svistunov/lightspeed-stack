TODO (example, delete): [conversation-compaction-spike.md](../../design/conversation-compaction/conversation-compaction-spike.md) (LCORE-1314)

# Spike for TODO: feature name

## Overview

**The problem**: TODO

**The recommendation**: TODO

**PoC validation**: TODO

## Strategic decisions — for TODO specify reviewer(s) (default: @sbunciak)

High-level decisions that determine scope, approach, and cost. Each has a
recommendation — please confirm or override.

### Decision S1: TODO title

TODO: Context, options table, recommendation. Link to the relevant background section(s) below.

| Option | Description |
|--------|-------------|
| A      |             |
| B      |             |

**Recommendation**: TODO

**Confidence**: TODO (e.g., 75% — see `confidence_format` in
`docs/contributing/feature-design.config`)

## Technical decisions — for TODO specify reviewer(s) (default: @tisnik)

Architecture-level and implementation-level decisions.

### Decision T1: TODO title

TODO: Context, options table, recommendation. Link to the relevant background section(s) below.

| Option | Description |
|--------|-------------|
| A      |             |
| B      |             |

**Recommendation**: TODO

**Confidence**: TODO

## Stakeholder decisions — for TODO specify reviewer(s) from requesting team

REMOVE THIS WHOLE SECTION IF NOT APPLICABLE. Include this section only when
the feature originated from a specific team or stakeholder (controlled by
`include_stakeholder_decisions_section = if_applicable` in
`docs/contributing/feature-design.config`). Decisions that the requesting
team is uniquely positioned to weigh in on go here.

### Decision SH1: TODO title

TODO: Context, options table, recommendation.

**Recommendation**: TODO

**Confidence**: TODO

## Out of scope

What this spike deliberately does *not* address. Each item should explain
why it's deferred and (where possible) reference a follow-up ticket or
existing JIRA that will pick it up.

- TODO: deferred item — why
- TODO: deferred item — why

## Proposed JIRAs

Order tickets to reflect dependency / kickoff sequence: the first ticket is
what kicks off the work; later tickets follow the implementation order.
Each JIRA's agentic tool instruction should point to the **spec doc** (the
permanent reference), not this spike doc.

The first two stubs below are required-by-default per project convention
(`require_e2e_kickoff_jira = yes`,
`require_e2e_step_definitions_jira = yes`). Customize them for the feature
or remove if explicitly not applicable; document the removal rationale in
`Out of scope` above.

<!-- type: Story -->
<!-- key: LCORE-???? -->
### LCORE-???? E2E feature files for TODO feature (no step implementation)

**User story**: As a Lightspeed Core e2e engineer, I want the behave
feature files for TODO feature scenarios written before the feature
implementation lands, so that the test shape reflects the feature's
intended behavior rather than the chosen implementation, and any
architectural gaps surface early.

**Description**: Author behave `.feature` files under `tests/e2e/features/`
that describe the behaviors required of TODO feature. Step definitions
(Python glue) are explicitly **not** part of this ticket — they are
covered by a later sibling ticket (LCORE-????).

**Scope**:
- `.feature` files covering R1..Rn from the spec doc
- Additions to `tests/e2e/test_list.txt`
- Author from spec doc requirements only; do not read implementation code

**Acceptance criteria**:
- behave parses every new `.feature` file without syntax errors
- behave marks all new scenario steps as `undefined`
- `uv run make test-e2e` remains green (new scenarios skipped/undefined, not failing)

**Blocks**: LCORE-???? (step-definitions counterpart)

**Agentic tool instruction**:

```text
Read "Requirements" and "Acceptance test surface" in
docs/design/<feature>/<feature>.md.
Do NOT read other JIRAs' scope sections or the implementation code while
authoring; the point of this ticket is feature files uncontaminated by
implementation detail.
Key files to create: tests/e2e/features/<feature>-*.feature plus
additions to tests/e2e/test_list.txt. Do NOT create step definitions.
```

<!-- type: Task -->
<!-- key: LCORE-???? -->
### LCORE-???? Implement behave step definitions for TODO feature files

**Description**: Implement Python step definitions under
`tests/e2e/features/steps/` for the `.feature` files authored in
LCORE-???? (kickoff). Take the Gherkin as-is; if a scenario cannot be
implemented faithfully, raise it against the spec doc rather than
quietly weakening the test.

**Blocked by**:
- LCORE-???? (E2E feature files kickoff)
- TODO list implementation tickets that must exist before step defs run

**Agentic tool instruction**:

```text
Read "Architecture" and "Requirements" in
docs/design/<feature>/<feature>.md.
Take feature files from tests/e2e/features/<feature>-*.feature as-is;
do not modify Gherkin to accommodate implementation constraints.
To verify: `uv run make test-e2e` runs every new scenario green.
```

<!-- type: Task -->
<!-- key: LCORE-???? -->
### LCORE-???? TODO fill in title

TODO: Use the format from `docs/contributing/templates/jira-ticket-template.md`. Change type above to Story if user-facing.

**Description**: TODO

**Scope**:

- TODO

**Acceptance criteria**:

- TODO

**Agentic tool instruction**:

```text
Read the "[section]" section in docs/design/<feature>/<feature>.md.
Key files: [files].
```

## Proposed incidental JIRAs

REMOVE THIS WHOLE SECTION IF NOT APPLICABLE (controlled by
`include_proposed_incidental_jiras_section = if_applicable`). Tickets for
unrelated bugs or improvements noticed *during* the spike that aren't part
of the feature itself but should be tracked.

<!-- type: Task -->
<!-- key: LCORE-???? -->
### LCORE-???? TODO incidental finding title

**Description**: TODO

## PoC results

REMOVE THIS WHOLE SECTION IF NO PoC WAS BUILT. If a PoC was built, document
what it does, what it proved, and how it diverges from the production
design. PoC code and evidence live under
`docs/design/<feature>/poc-results/` and are removed before merge per
`docs/contributing/howto-organize-poc-output.md`.

### What the PoC does

**Important**: The PoC diverges from the production design in these ways:
- TODO

### Results

TODO

### Findings discovered during the PoC

REMOVE THIS SUBSECTION IF NO INTERESTING FINDINGS. Surprises uncovered
while building the PoC that the implementation JIRAs need to address.

- TODO: finding — implication for the design — JIRA reference

## Incidental findings

REMOVE THIS WHOLE SECTION IF NO INCIDENTAL FINDINGS. Unrelated bugs or
improvements noticed during this spike. File these as separate JIRAs (see
`Proposed incidental JIRAs` above).

- TODO

## External input needed

REMOVE THIS WHOLE SECTION IF NOT APPLICABLE. Items where reviewers need to
get input from people outside the immediate team (other teams, external
maintainers, downstream consumers). Each item should name who to ask.

- TODO: item — who to ask

## Background sections

TODO: Research and analysis that supports the decisions above. These sections are linked from the decisions, not read front-to-back. Common topics: current architecture, existing approaches, design alternatives. Add or remove as needed.

## Glossary

REMOVE THIS WHOLE SECTION IF NOT NEEDED. Define terms specific to this
spike that a reader unfamiliar with the feature would benefit from.

## Appendix A

TODO: Supporting material — external references, responses to suggestions from team members. Add appendices as needed.
