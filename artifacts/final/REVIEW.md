# Evidence review

An independent adversarial review checked frozen hashes first, then attempt counts,
metric definitions, split integrity, summary claims and text hygiene. A subsequent
own review checked the saved rows, windows, raw digests and unchanged source.

All 26 frozen files match the implementation and marker commits. The 314 raw
evidence digests match. There are 240 unique final attempts, result rows and run
IDs, plus 60 test episodes; all match the frozen manifest. The original evaluator
bytes and summary definitions match validation. All acceptance-table arithmetic
and saved window metric groups were independently recounted.

The review found a reporting gap: settled windows were not separated after A/B/C.
The added candidate tables use retained window order and frozen schedules, without
model inference or simulation. Their disjoint union exactly matches the saved
“after” group. No original metric or frozen file changed. No substantive findings
remain beyond the disclosed target misses and limitations.

The ML agent misses the minimum Recover A RMS target and processing-latency target.
Those findings remain reported, not repaired. No further tuning or evaluation was
performed. Reserve was not generated or evaluated.

Receipts and hashes attest the recorded execution; they are not an external access
audit and cannot exclude unlogged execution or transient changes. Saved per-run
p95 values permit aggregation checks, not reconstruction of individual processing
durations. Truth grades remain the original independent-evaluator outputs; they
were not regenerated. Historical demonstrations are separate prefreeze evidence.
