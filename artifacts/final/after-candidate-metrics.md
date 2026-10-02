# Test metrics after each candidate

Reporting from saved episode arrays, with no inference or simulation. The frozen generator emits one window per 0.1-second tick starting at tick 3 (0.3 s). For saved regime=after windows, assign the latest manifest command whose scheduled time is strictly less than the window-end time. Transitional labels remain excluded from recall/FPR; before and during groups remain in window-metrics.json.

| Candidate | All windows | Chatter | Stable | Transitional | Recall | FPR |
|---|---:|---:|---:|---:|---:|---:|
| A | 2770 | 1635 | 1080 | 55 | 100.0% | 0.0% |
| B | 3485 | 1315 | 1970 | 200 | 100.0% | 0.0% |
| C | 3615 | 1005 | 2435 | 175 | 100.0% | 0.0% |
