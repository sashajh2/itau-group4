# Architecture

How the system and the models are built. Read these to understand *what* exists and
*why* it is shaped that way, before reading [`../guides/`](../guides/) to learn how to
run it.

| Document | Covers |
|---|---|
| [`branch_documentation.md`](branch_documentation.md) | **Start here.** Overview of the deepfake detection pipeline: orthogonal embedding heads, the heterogeneous (identity) head and its two training stages, episodic learning, and evaluation on raw embeddings without training. |
| [`parameter_breakdown.md`](parameter_breakdown.md) | Where the ~7.5M parameters of `AVTemporalModel` actually go, layer by layer — useful when deciding what to shrink. |

## `time_series/`

Three explainers for the temporal models in `models/time_series/`. They describe
`time_series_model.py` (`AVTemporalModel`), a PatchTST-style temporal Transformer over
audio+video embedding sequences.

| Document | Covers |
|---|---|
| [`time_series/time_series_model_explanation.md`](time_series/time_series_model_explanation.md) | The model end to end: problem statement, architecture, data flow |
| [`time_series/transformer_walkthrough.md`](time_series/transformer_walkthrough.md) | A step-by-step walkthrough of the same model, tensor shape by tensor shape. Note it is an AV *fusion* model — it expects both modalities. |
| [`time_series/patch_tokenization_explained.md`](time_series/patch_tokenization_explained.md) | Deep dive on patch tokenization: why it is used here, its cost/benefit, and the alternatives |

## `disentangled/`

| Document | Covers |
|---|---|
| [`disentangled/loss_balancing_alternatives.md`](disentangled/loss_balancing_alternatives.md) | Why the additive `L_proto + λ_var·L_var + λ_orth·L_orth` combination is fragile, and what could replace it |

The module's own README — losses, equations, and usage — lives with the code at
[`../../training/disentangled/README.md`](../../training/disentangled/README.md).
