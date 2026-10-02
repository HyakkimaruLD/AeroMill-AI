# Acceptance evidence

| Split / mode | Controllable ≥27/30 | Stable ≥29/30 | Timely ≥90% | Recover A RMS ≥40% | False/unverified =0/0 | Max run p95 <100 ms | Misses | False incidents/1000 s | Extra traversal time | Completed | HOLD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| validation / baseline | 0/30 | 20/30 | 86.7% | unavailable | 0/0 | 100.47 | 4 | 18.12 | unavailable | 23/60 | 37 |
| final / baseline | 0/30 | 18/30 | 83.3% | unavailable | 0/0 | 31.38 | 5 | 25.74 | unavailable | 20/60 | 40 |
| validation / ml_agent | 30/30 | 30/30 | 100.0% | 43.6% | 0/0 | 78.99 | 0 | 0.00 | 28.8% | 60/60 | 0 |
| final / ml_agent | 30/30 | 30/30 | 100.0% | 27.9% | 0/0 | 122.17 | 0 | 0.00 | 29.9% | 60/60 | 0 |
| validation / no_adaptation | 0/30 | 30/30 | 0.0% | unavailable | 0/0 | 72.99 | 30 | 0.00 | 0.0% | 60/60 | 0 |
| final / no_adaptation | 0/30 | 30/30 | 0.0% | unavailable | 0/0 | 18.46 | 30 | 0.00 | 0.0% | 60/60 | 0 |
| validation / threshold_search | 24/30 | 20/30 | 78.0% | 16.3% | 1/2 | 77.07 | 9 | 16.10 | 28.1% | 59/60 | 1 |
| final / threshold_search | 20/30 | 18/30 | 75.6% | 15.0% | 3/6 | 22.36 | 10 | 20.11 | 28.5% | 59/60 | 1 |

| ML target | Validation | Final/test | Result |
|---|---:|---:|---|
| Window recall ≥90% | 100.0% | 100.0% | PASS |
| Window FPR ≤5% | 0.0% | 0.0% | PASS |
| Controllable ≥27/30 | 30/30 | 30/30 | PASS |
| Stable ≥29/30 | 30/30 | 30/30 | PASS |
| False/unverified recovery =0/0 | 0/0 | 0/0 | PASS |
| Timely detection ≥90% | 100.0% | 100.0% | PASS |
| Recover A RMS ≥40% | 43.6% | 27.9% | MISS |
| Processing p95 <100 ms | 78.99 | 122.17 | MISS |
| Mandatory stable demos: ten each | 50/50 previously recorded | Carried forward; not final-set runs | PASS, prefreeze |
| Ten consecutive main demos | 10/10 previously recorded | Checked in prefreeze suite | PASS, prefreeze |
| Mandatory failure cases | Passed previously | Checked in prefreeze suite | PASS, prefreeze |

Prefreeze suite: 246 passed in 187.72s (0:03:07). These demonstration/failure checks are separate from the 240 final runs.

Processing is the maximum of per-run window-processing p95, matching validation aggregation. Final ran serially; validation used four concurrent mode workers. Timing differences are not attributed to model quality. Acquisition spans 250 ms; first available window is at 0.3 s. ACK, spindle slew, settling and verification use simulation time, not processing milliseconds; see the prior latency-components.json and final event logs.

RMS is X-axis total RMS, last second before first command versus second after independently confirmed recovery. The table uses the minimum measurable Recover A reduction; other scenarios have no universal 40% target. Extra traversal time averages completed controllable pairs relative to no_adaptation on the identical family; HOLDs are failures and have no traversal time.

Raw rows, event logs, attempts, window arrays and commands are retained. No final run was dropped or repeated. Test metrics exclude label -1. Family/regime/engagement tables contain both class counts; unavailable means no examples of that class.

Synthetic simulator; fixed A/B/C commands. Residual normalization may cause an extra retry after genuine recovery (2/40 earlier random demos). Baseline stays resonant by construction; threshold_search is the policy-matched comparator. No claim against a spectral-rule detector.
