# How to organize PoC output

When a spike includes a proof-of-concept, the validation results should be
structured so that reviewers can quickly understand what was tested and what
was found. This guide is intentionally **principles-first**, not
prescriptive about filenames — different PoC shapes warrant different
artifacts. Use the ones that fit; ignore the ones that don't.

## Where it goes

Place results in `docs/design/<feature>/poc-results/`. The directory is
expected to be removed before the spike PR is merged
(see [howto-run-a-spike.md](howto-run-a-spike.md), step 10); reviewers
read it during review and the spike doc preserves the salient findings.

## Principles

1. **A reviewer reading nothing but a top-level report should grasp the
   essentials.** Whatever you call this file (`README.md`, `01-poc-report.md`,
   `summary.txt` — pick what fits), it must be the first thing a reviewer
   opens, and it must stand alone.

2. **Order files by usefulness for the human reviewer**, not by the order
   you produced them. If naming conventions help (numeric prefixes,
   descriptive names), use them — but don't reorder for the sake of
   uniformity.

3. **Separate human-readable summary from machine-readable raw data.** A
   summary that interleaves prose findings with 50KB of raw JSON is
   unreviewable.

4. **Plain text where humans read; structured formats where machines read.**
   Markdown / `.txt` for prose; `.json` / `.yaml` / `.csv` for data.

5. **Show, don't just claim.** When a PoC proves "X works", include
   evidence: command output, response payload, before/after diff,
   screenshot, etc.

6. **Cross-reference the spike doc.** Each significant finding in the
   spike doc's "PoC results" or "Findings discovered during the PoC"
   section should link back to a specific file in `poc-results/`.

## Two common PoC shapes

### Code PoC (mechanism validation)

The PoC is code that demonstrates a mechanism end-to-end (e.g., a
synthesizer that produces a config file consumed by a downstream service).

Typical artifacts:

- **Top-level report** — what was tested, what worked, what surprised you
- **Sample input(s)** — the config file(s) the PoC was driven with
- **Sample output(s)** — what the code produced (synthesized files, API
  responses, log excerpts)
- **Repro instructions** — exact commands to reproduce the PoC, with the
  environment variables required

### Data / experimental PoC

The PoC runs a controlled experiment and produces measurements (e.g.,
quality of summarization across N queries with thresholds at
{3, 5, 10}).

Typical artifacts:

- **Top-level report** — methodology, results, findings, implications for
  the production design
- **Conversation / interaction log** — human-readable record of what
  happened during the run
- **Quantitative data** — numbers (token usage, latency, accuracy)
- **Structured event data** — `.json` or `.csv` for downstream analysis
- **Extracted outputs** — model outputs, summaries, etc., as text

## What a good top-level report contains

Whatever its filename, it should include:

- **Glossary**: terms specific to this PoC (omit if not needed).
- **Design**: what was tested, how, with what parameters.
- **Results**: what happened, with concrete numbers / outputs.
- **Findings**: what the results mean for the production design — what
  was proved, disproved, or surprising. Each finding worth carrying
  forward should also appear in the spike doc's "Findings discovered
  during the PoC" section.
- **Implications**: how the findings change the recommended design or
  proposed JIRAs. If a finding doesn't change anything, say so explicitly.

## Multiple PoC runs

If the PoC was run multiple times with different parameters, give each run
its own descriptive sub-directory:

```
poc-results/
├── README.md             # one report for the whole spike
├── run-5-queries/
└── run-50-queries/
```

Avoid timestamp-named sub-directories — they don't tell a reader what
varied between runs.

## Removed before merge

PoC results are removed from the branch before the spike PR is merged. They
serve their purpose during review and persist in git history (or a tag, if
explicitly preserved). The spike doc's "PoC results" and "Findings
discovered during the PoC" sections capture what survives the merge.
