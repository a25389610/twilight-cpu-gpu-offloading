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
