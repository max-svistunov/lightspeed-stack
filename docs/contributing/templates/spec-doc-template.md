TODO (example, delete): [conversation-compaction.md](../../design/conversation-compaction/conversation-compaction.md) (LCORE-1311)

# Feature design for TODO: feature name

|                    |                                           |
|--------------------|-------------------------------------------|
| **Date**           | TODO                                      |
| **Component**      | TODO                                      |
| **Authors**        | TODO                                      |
| **Feature**        | TODO: [LCORE-XXXX](https://redhat.atlassian.net/browse/LCORE-XXXX) |
| **Spike**          | TODO: [LCORE-XXXX](https://redhat.atlassian.net/browse/LCORE-XXXX) |
| **Links**          | TODO                                      |

## What

TODO: What does this feature do?

## Why

TODO: What problem does this solve? What happens today without it?

## Requirements

TODO: Numbered, testable requirements. For each, it should be easy to provide clear acceptance criteria.

- **R1:**
- **R2:**

## Use Cases

TODO: User stories in "As a [role], I want [X], so that [Y]" format.

- **U1:**
- **U2:**

## Architecture

### Overview

TODO: Flow diagram showing the request/response path with the new feature.

```text
TODO: flow diagram
```

TODO: Add subsections below for each relevant component. Architecture
sub-sections marked `if_applicable` in
`docs/contributing/feature-design.config` should be present only when the
feature actually has that concern. Delete unused sub-sections; add
feature-specific ones.

### Trigger mechanism

REMOVE IF NOT APPLICABLE. When and how the feature activates.

### Storage / data model changes

REMOVE IF NOT APPLICABLE. Schema changes, which backends need updates.

### Configuration

REMOVE IF NOT APPLICABLE. YAML config example and configuration class.

``` yaml
TODO: config example
```

``` python
TODO: configuration class
```

### API changes

REMOVE IF NOT APPLICABLE. New or changed fields in request/response models.

### Error handling

REMOVE IF NOT APPLICABLE. How errors are surfaced — new error types, HTTP status codes, recovery behavior.

### Security considerations

REMOVE IF NOT APPLICABLE. Auth, access control, data sensitivity implications.

### Migration / backwards compatibility

REMOVE IF NOT APPLICABLE. Schema migrations, API versioning, feature flags for gradual rollout.

## Acceptance test surface

Maps each requirement (R1..Rn) to one or more observable behaviors. This
section is the source-of-truth that drives the e2e-kickoff JIRA's feature
files. Authors of `.feature` files read this section to write Gherkin
scenarios.

| Req | Observable behavior | Verified by |
|-----|---------------------|-------------|
| R1  | TODO                | TODO        |

## Aspect-specific concerns

REMOVE ANY SUB-SECTION THAT DOES NOT APPLY. These sections cover concerns
that may or may not be relevant to a given feature. The defaults in
`docs/contributing/feature-design.config` are `if_applicable` — include
only when the feature genuinely has the concern.

### Latency and Cost

How the feature affects per-request performance and cost.

### Observability

What is logged, what is measured, what is traced. New dashboards / alerts.

### Capacity planning

How much load the feature handles before degrading; what scaling decisions
are tied to it.

### Failure modes

Non-obvious ways the feature can fail, and what happens when it does.

### Telemetry / data privacy

What data is collected, where it goes, how user privacy is preserved.

### Feature flags / rollout

Gradual rollout strategy if this feature lands behind a flag.

### Runbook / oncall implications

What oncall needs to know; new alerts, new failure modes, recovery
procedures.

### Internationalization

i18n / l10n implications, if any.

### API versioning

If the feature changes a public API, how the version bump is handled.

### Rate limiting / quotas

If the feature introduces or interacts with rate limits or quotas.

## Implementation Suggestions

### Key files and insertion points

TODO: Table of files to create or modify.

| File | What to do |
|------|------------|
| TODO | TODO       |

### Insertion point detail

TODO: Where the feature hooks into existing code — function name, what's available at that point, what the code should do.

### Config pattern

All config classes extend `ConfigurationBase` which sets `extra="forbid"`.
Use `Field()` with defaults, title, and description.  Add
`@model_validator(mode="after")` for cross-field validation if needed.

Example config files go in `examples/`.

### Test patterns

- Framework: pytest + pytest-asyncio + pytest-mock.  unittest is banned by ruff.
- Mock Llama Stack client: `mocker.AsyncMock(spec=AsyncLlamaStackClient)`.
- Patch at module level: `mocker.patch("utils.module.function_name", ...)`.
- Async mocking pattern: see `tests/unit/utils/test_shields.py`.
- Config validation tests: see `tests/unit/models/config/`.

TODO: Describe any feature-specific test considerations (e.g., tests that need a running service, special fixtures, concurrency testing).

## Open Questions for Future Work

Things explicitly deferred and why. **Each item must trace back to its
origin**: a spike decision, a PoC finding, or a reviewer comment, so that
the rationale survives over time.

- TODO: question — origin (e.g., "Deferred from spike Decision T7" or
  "Surfaced during PoC, blocked on external dependency on TODO")

## Changelog

TODO: Record significant changes after initial creation.

| Date | Change | Reason |
|------|--------|--------|
|      | Initial version |        |

## Appendix A

TODO: Supporting material — PoC evidence, API comparisons, reference sources. Add appendices as needed.
