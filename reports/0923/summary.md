# Twilight GQA union temporal-reuse diagnostic

Diagnostic-only trace run; its TPOT is intentionally not a performance result.
All ratios are row-weighted over layer × KV head × valid decode-step rows.
Payload refers to history union rows only; it excludes the direct-layout current-token slot.

## Overall row-weighted results

| scope | 1-step reuse | new/miss | 2-step coverage | 4-step coverage | 8-step coverage | current payload | ideal previous-token miss payload | ideal byte reduction |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| full | 90.78% | 9.22% | 92.36% | 94.58% | 96.46% | 507.805 MiB/token | 46.807 MiB/token | 90.78% |
| remote | 90.13% | 9.87% | 91.82% | 94.20% | 96.21% | 472.915 MiB/token | 46.700 MiB/token | 90.13% |

## Per-request row-weighted results

### full

| request | 1-step | new/miss | 2-step | 4-step | 8-step | current MiB/token | miss MiB/token |
|---|---:|---:|---:|---:|---:|---:|---:|
| 001_niah_multikey_3_i011 | 91.18% | 8.82% | 92.61% | 94.76% | 96.64% | 512.222 MiB/token | 45.175 MiB/token |
| 002_vt_i002 | 89.93% | 10.07% | 91.65% | 94.04% | 96.03% | 487.907 MiB/token | 49.113 MiB/token |
| 003_qa_1_i011 | 91.18% | 8.82% | 92.78% | 94.91% | 96.68% | 523.286 MiB/token | 46.134 MiB/token |

### remote

| request | 1-step | new/miss | 2-step | 4-step | 8-step | current MiB/token | miss MiB/token |
|---|---:|---:|---:|---:|---:|---:|---:|
| 001_niah_multikey_3_i011 | 90.56% | 9.44% | 92.09% | 94.39% | 96.41% | 477.332 MiB/token | 45.070 MiB/token |
| 002_vt_i002 | 89.18% | 10.82% | 91.03% | 93.60% | 95.74% | 453.016 MiB/token | 49.005 MiB/token |
| 003_qa_1_i011 | 90.58% | 9.42% | 92.29% | 94.56% | 96.46% | 488.395 MiB/token | 46.026 MiB/token |

## Ideal resident-cache working set

Mean distinct history rows in the prior-window union. This is a residency estimate, not an allocated-buffer-size claim.

| scope | previous 1 step | previous 2 steps | previous 4 steps | previous 8 steps |
|---|---:|---:|---:|---:|
| full | 508.113 MiB/token | 555.493 MiB/token | 628.763 MiB/token | 722.135 MiB/token |
| remote | 473.222 MiB/token | 520.495 MiB/token | 593.559 MiB/token | 686.549 MiB/token |

## Request and layer spread

Request median/min/max (three row-weighted request ratios):

| scope | metric | median | min | max |
|---|---|---:|---:|---:|
| full | 1-step coverage | 91.18% | 89.93% | 91.18% |
| full | 2-step coverage | 92.61% | 91.65% | 92.78% |
| full | 4-step coverage | 94.76% | 94.04% | 94.91% |
| full | 8-step coverage | 96.64% | 96.03% | 96.68% |
| remote | 1-step coverage | 90.56% | 89.18% | 90.58% |
| remote | 2-step coverage | 92.09% | 91.03% | 92.29% |
| remote | 4-step coverage | 94.39% | 93.60% | 94.56% |
| remote | 8-step coverage | 96.41% | 95.74% | 96.46% |

## Common-horizon comparison

All lags below use the same eligible target steps `t=9..32`, so incremental coverage is directly comparable.

| scope | 1-step | 2-step | 4-step | 8-step |
|---|---:|---:|---:|---:|
| full | 92.00% | 92.75% | 94.62% | 96.46% |
| remote | 91.43% | 92.24% | 94.24% | 96.21% |

Layer summary is row-weighted across all three requests; full per-layer values are in `layer_summary.csv`.

| scope | metric | layer median | layer min | layer max |
|---|---|---:|---:|---:|
| full | 1-step coverage | 90.90% | 88.17% | 93.66% |
| full | 2-step coverage | 92.60% | 88.96% | 94.49% |
| full | 4-step coverage | 94.74% | 91.47% | 95.81% |
| full | 8-step coverage | 96.49% | 94.80% | 97.43% |
| remote | 1-step coverage | 90.20% | 87.50% | 93.32% |
| remote | 2-step coverage | 92.04% | 88.34% | 94.15% |
| remote | 4-step coverage | 94.34% | 90.99% | 95.59% |
| remote | 8-step coverage | 96.22% | 94.51% | 97.30% |
