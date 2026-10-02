# Final evaluation

One frozen evaluation retained all 240 runs: 60 fixed families in each of four modes. The held-out test windows were evaluated once. Reserve was not generated or evaluated.

The ML agent completed 30/30 controllable runs successfully and left 30/30 stable runs without intervention. Timely detection was 100.0%, with 0 missed incidents. False/unverified recoveries: 0/0. Minimum measurable Recover A RMS reduction: 27.9%. Test recall/FPR: 100.0%/0.0%.

Targets met: Window recall ≥90%, Window FPR ≤5%, Controllable ≥27/30, Stable ≥29/30, False/unverified recovery =0/0, Timely detection ≥90%. Targets missed: Recover A RMS ≥40%, Processing p95 <100 ms. Stable demonstrations, ten main demonstrations, and mandatory failure checks passed before freezing; these are separate evidence.

Validation achieved 30/30 controllable, 30/30 stable, 100.0% timely detection, and 43.6% RMS reduction. Maximum per-run processing p95 was 78.99 ms versus 122.17 ms final. Final was serial; validation used four workers, so timing is not directly comparable.

All modes, productivity costs and failed IDs are in acceptance.md and failed-runs.md. This is a synthetic simulator with fixed A/B/C candidates, not real CNC validation. Residual normalization can trigger an extra retry after genuine recovery (2/40 earlier random demos). Baseline is weak by construction because it remains resonant; threshold_search is the policy-matched comparator. No spectral-rule advantage is established.
