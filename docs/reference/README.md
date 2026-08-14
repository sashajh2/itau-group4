# Reference

Background material that is neither project documentation nor a plan.

| File | What it is |
|---|---|
| [`machine_learning_for_visual_fraud_detection.pdf`](machine_learning_for_visual_fraud_detection.pdf) | The background paper this project builds on. Full title: *Machine Learning for Visual Fraud Detection — Homoglyph Spoofing and Deepfake Identification*. |
| [`cursor_prompt_evaluation_metrics.md`](cursor_prompt_evaluation_metrics.md) | An LLM prompt, kept for provenance: the spec used to generate the evaluation metrics in the disentangled training loop. It documents which metrics were asked for and when they are computed (on input embeddings before training, then after each epoch on the `z^auth` projections). |

Note that `cursor_prompt_evaluation_metrics.md` is a *prompt*, not a specification of
current behavior — read the code in `training/disentangled/metrics.py` for what is
actually computed.
