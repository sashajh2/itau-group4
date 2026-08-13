# Refactoring records

Historical context: restructures that have already happened. Useful for understanding
why the code looks the way it does, and for spotting descriptions that predate the
current layout.

| Document | Covers |
|---|---|
| [`refactoring_summary.md`](refactoring_summary.md) | The move from a flat `scripts/` directory to the current semantic package layout (`data/`, `training/`, `evaluation/`, …) — before/after trees and the rationale |
| [`improvements.md`](improvements.md) | Changes made to the disentangled training loop to address instability and loss imbalance: content-group filtering, loss rebalancing, and related fixes |

> **These describe past states.** Where a path in them no longer exists, treat the
> document as a record rather than a bug. The current layout is mapped in
> [`../README.md`](../README.md).
