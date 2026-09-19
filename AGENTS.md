# Research collaboration rules

The canonical research-method and AI-collaboration rules are defined in
`../../RESEARCH_WORKFLOW.md`. Read and follow that document before starting
research discussion, experiment design, or implementation.

## Discussion order

For research discussions, progress through these layers in order:

1. research background;
2. the paper's core problem;
3. the original design trade-offs;
4. a concrete possible limitation;
5. evidence needed to confirm the limitation;
6. experimental observations;
7. a narrowed research question;
8. an improvement direction;
9. experiment design;
10. implementation.

Do not jump to functions, CUDA code, or buffer layout before the research
question, hypothesis, and experiment purpose are established.

Before implementation, distinguish:

- confirmed facts;
- observed experimental phenomena;
- unverified hypotheses;
- experiments still needed;
- the next research decision.

## Weekly progress reports

The reporting week uses Asia/Taipei time and starts every Wednesday. It ends on
the following Tuesday. Reports are stored in `reports/weekly/` and named by
their Wednesday start date, for example `2026-07-22.md`.

After every material, verified research milestone, update the active weekly
report in the same turn. Material milestones include:

- confirming or rejecting a research hypothesis;
- identifying a serious correctness or performance problem;
- implementing and verifying a fix;
- completing a benchmark or ablation that changes the research conclusion;
- reaching a scope or research-direction decision.

If work begins on or after a new Wednesday and that week's report does not
exist, create it from `reports/weekly/TEMPLATE.md`. Do not continue adding new
work to the previous week's file.

Reports are factual source material for weekly meeting slides. Each progress
item should contain:

1. the problem or question;
2. prior status or hypothesis;
3. evidence or experiment;
4. method or fix, if any;
5. quantitative result;
6. conclusion and limitations;
7. relevant artifact paths;
8. the next research decision.

Do not record a plausible explanation as a confirmed cause. Label preliminary
single-run measurements and explicitly retain failed experiments when they
help explain a limitation.

Write weekly progress reports in Traditional Chinese. Keep model names, API
names, code identifiers, units, and technical terms such as `Pinned Slab`,
`CUDA stream`, `torch.cat`, `Prefill`, and `Decode` in English when that is
clearer than translation.


## GitHub weekly experiment-report mirror

This repository is the readable GitHub mirror for standalone Twilight/HeadInfer
experiment reports. The local `reports/weekly/YYYY-MM-DD.md` remains canonical;
this mirror does not replace it.

After a verified material experiment creates or updates a standalone Markdown
report, sync a GitHub-readable copy to:

```text
reports/<MMDD>/
```

`<MMDD>` is the Wednesday immediately after the local reporting week closes.
For example, local week 2026-09-16 through 2026-09-22 maps to `reports/0923/`.
On each new Wednesday, create or confirm the new folder before adding that
week's experiment reports.

Only publish self-contained Markdown: state experimental conditions,
quantitative observations, limitations, and relevant local artifact/source
references. Diagnostic and single-run evidence must retain its scope and
preliminary limitations. Do not mirror raw JSON/CSV/logs/checkpoints/logits,
timelines, large traces, or sensitive data. Avoid duplicate byte-identical
uploads; revise the existing Markdown when evidence changes. An explicit user
request not to publish, to delay publication, or to use another repository
overrides this rule.
