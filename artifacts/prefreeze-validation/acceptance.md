# Prefreeze validation acceptance

Validation only. The ML verification residual ratio limit changed from 2.0 to 2.2; the model, detection threshold, deadlines, simulator and grading are unchanged.

Targets: success ≥27/30, zero false recoveries; stable ≥29/30; detection within 1 s ≥90%; Recover A RMS reduction ≥40%; processing p95 <100 ms.

| Mode | Success /30 | Stable /30 | Timely detection | RMS min | False / unverified recovery | p95 ms | False interventions | False incidents /1000 s | Extra time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| no_adaptation | 0 | 30 | 0.0% | unavailable | 0 / 0 | 72.99 | 0 | 0.00 | 0.0% |
| baseline | 0 | 20 | 86.7% | unavailable | 0 / 0 | 100.47 | 14 | 18.12 | unavailable |
| threshold_search | 24 | 20 | 78.0% | 16.3% | 1 / 2 | 77.07 | 39 | 16.10 | 28.1% |
| ml_agent | 30 | 30 | 100.0% | 43.6% | 0 / 0 | 78.99 | 0 | 0.00 | 28.8% |

| Other §4 target | Evidence |
|---|---|
| Window recall ≥90%, FPR ≤5% | Validation RF 100.0%, 0.0%; 5110 chatter / 7890 stable windows. Transitional windows excluded. |
| Mandatory stable demos: 10 seeds each | 50/50, stable/engagement/impact/harmonic/noisy. |
| Failure scenarios and ten consecutive Recover A runs | Full suite: 239 passed; includes test_recover_a_ten_consecutive_distinct_seeds and mandatory failure cases. |

Four concurrent mode workers, cold memory per run; maximum of per-run processing p95. Tests and diagnostic simulations ran separately.
Before-change concurrent ML p95: 127.85 ms (FAIL); the complete before summaries and behavioral delta are retained in acceptance.json.
Extra time uses completed controllable pairs; HOLD remains failure. Window metrics are freshly recomputed without fitting. See results.json and runs.csv for every run; window-metrics.json for family/regime/engagement breakdowns.

Separate clocks, validation seed 200005: 250 ms window, first available at the 0.3 s tick; ACK 0.1 s; spindle slew 0.3/0.6 s; settle 1 s; VERIFY 1.5 s; truth confirmation 3 s. These are simulation times, separate from processing milliseconds in latency-components.json.

The three original false commands are verification retries after true detections. See false-interventions.json for truth envelopes, score and residual series, engagement/impact state and full planning records. harmonic-band-check.json isolates the RPM-dependent residual filter.

Synthetic simulation; not validated on real CNC.
Validation is not independent final-set certification.
Residual normalization still depends on RPM; demo false retries remain.
Baseline remains resonant by construction; RF is compared only against total RMS, not a spectral rule.
