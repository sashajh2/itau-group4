# Reference

Background material that is neither project documentation nor a plan.

| File | What it is |
|---|---|
| [`Machine_Learning_for_Visual_Fraud_Detection___Homoglyph_Spoofing_and_Deepfake_Identification-6.pdf`](Machine_Learning_for_Visual_Fraud_Detection___Homoglyph_Spoofing_and_Deepfake_Identification-6.pdf) | The background paper this project builds on — visual fraud detection covering homoglyph spoofing and deepfake identification |
| [`Cursor_Prompt_Evaluation_Metrics.md`](Cursor_Prompt_Evaluation_Metrics.md) | An LLM prompt, kept for provenance: the spec used to generate the evaluation metrics in the disentangled training loop. It documents which metrics were asked for and when they are computed (on input embeddings before training, then after each epoch on the `z^auth` projections). |

Note that `Cursor_Prompt_Evaluation_Metrics.md` is a *prompt*, not a specification of
current behavior — read the code in `training/disentangled/metrics.py` for what is
actually computed.
