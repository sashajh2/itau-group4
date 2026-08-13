# Planning

Proposals and open questions — work that is intended, not yet work that is done. For
what has already been run, see [`../analysis/`](../analysis/).

| Document | Covers |
|---|---|
| [`hybrid_anomaly_detection_plan.md`](hybrid_anomaly_detection_plan.md) | Proposal for a temporal encoder trained with combined reconstruction + contrastive objectives to learn the "real manifold", aiming to generalize to unseen fake types (Sora, future methods) |
| [`PIPELINE_REFACTORING_PLAN.md`](PIPELINE_REFACTORING_PLAN.md) | The plan behind the current unified train+eval pipeline: shared data loading, hyperparameter sweeps, variance-regularization focus. Largely carried out — see [`../refactoring/`](../refactoring/). |

## `ideas/` — the open backlog

Short exploratory notes. These are the most useful place to look for "what should I work
on next".

| Document | Covers |
|---|---|
| [`ideas/embedding_combinations.md`](ideas/embedding_combinations.md) | Ways to combine the three per-segment embeddings (HuBERT 768-d, OpenL3, SENet) — concatenation, fusion, and alternatives |
| [`ideas/dataset_combinations.md`](ideas/dataset_combinations.md) | Inventory of the available datasets with real/fake counts and manipulation types, and strategies for combining them. Note the class imbalance: ShareVeo3 contributes 1460 fakes and 0 reals. |
| [`ideas/bag_of_embeddings.md`](ideas/bag_of_embeddings.md) | Order-free alternatives to sequence models, motivated by transformer and LSTM results suggesting temporal order is not being exploited |
| [`ideas/disentanglement_fixes.md`](ideas/disentanglement_fixes.md) | Failure analysis across the checkpointed disentanglement runs, with a per-run metrics table and candidate fixes. The `fix1`/`fix1a` experiments in `experiments/` come from this. |
