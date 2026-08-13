# Analysis

Findings: what the experiments showed and how to read them. The corresponding output
files (plots, CSVs, metrics JSON) live in [`../../results/`](../../results/).

| Document | Covers |
|---|---|
| [`experiment_summary.md`](experiment_summary.md) | **Start here.** Phase-by-phase log of everything tried so far and how each attempt turned out. The fastest way to catch up. |
| [`local_global_embedding_analysis.md`](local_global_embedding_analysis.md) | The most complete write-up: local and global structure of pretrained embeddings for audiovisual deepfake detection. See the caveats below. |
| [`COLLAPSE_ANALYSIS_AND_FIXES.md`](COLLAPSE_ANALYSIS_AND_FIXES.md) | Diagnosis of embedding collapse during disentangled training — the symptoms (Wasserstein distance 0.533 → 0.002, intra-group variance −99.8%), the causes, and proposed fixes |
| [`RESULTS_INTERPRETATION.md`](RESULTS_INTERPRETATION.md) | Reading of the "conservative" hyperparameter config results, which still collapsed. A worked example of how to interpret the metrics the pipeline emits. |

## Caveats on `local_global_embedding_analysis.md`

The document states these itself, but they are worth surfacing before anyone builds on it:

- It is a **first draft**, written to be edited rather than shipped.
- Every citation is a placeholder of the form `[CITE: Author, Title, Venue Year]` and
  **none have been verified**.
- Claims not yet supported by the evidence are tagged `[SPECULATIVE]`; questions that
  would change the conclusions are tagged `[OPEN]`.
- Its Appendix A cross-references every number back to the file in this repository it
  came from, so results can be re-derived.

## Related

- The proposed follow-ups to the collapse findings are in
  [`../planning/ideas/disentanglement_fixes.md`](../planning/ideas/disentanglement_fixes.md).
- What was changed in response is recorded in
  [`../refactoring/IMPROVEMENTS.md`](../refactoring/IMPROVEMENTS.md).
