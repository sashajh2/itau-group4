Interpretation of baseline result files (four experiments)

1) avdf_hubert_original_split.json
- Significance: HuBERT-only baseline on AVDeepfake1M/AV1M HDF5 via scripts/avdf_baselines.py; large-scale reference split (val_split=0.2, random_state=42).
- Data type/shape: ~230,178 segment embeddings in HDF5 (train 184,142 | val 46,036); HuBERT dim per segment (768).
- Structure/processing: Filters videos with augmentation info, includes av1m/avdeepfake sources, splits stratified; reports MLPs (with embedding metrics), RandomForest, XGBoost; confusion counts on val set.
- Results (counts, %): RF TP=1,142 TN=43,940 FP=353 FN=601 (acc 97.9%); XGB TP=1,358 TN=43,183 FP=1,110 FN=385 (acc 96.8%); MLPs also reported (see file).
- Interpretation: Strong performance on this large dataset; indicates HuBERT embeddings are effective when trained/evaluated on full AVDF/AV1M distribution.
- Suggested changes: For fair comparison to smaller precomputed sets, rerun this pipeline on the same 941-sample NPZs or rerun combined embeddings on the HDF5 if available.

2) hubert_only_same_split.json
- Significance: HuBERT-only on the 941 precomputed segments (NPY + unified labels) with the same 80/20 split (random_state=42) used for combined runs.
- Data type/shape: 941 samples, embedding dim 768.
- Structure/processing: train_test_split(test_size=0.2, stratified); models RandomForest (800 trees, balanced) and XGBoost (1000 trees, tuned params, scale_pos_weight).
- Results (counts, %): RF TP=17 TN=64 FP=41 FN=67 (acc 42.9%); XGB TP=29 TN=54 FP=51 FN=55 (acc 43.9%).
- Interpretation: Weak performance on this small set; recall and specificity both low, suggesting HuBERT alone underfits or data domain differs from AVDF baseline.
- Suggested changes: Try richer models or augment data; or prefer concatenated embeddings which perform better on the same split.

3) untuned_combined_openl3_hubert_senet.json
- Significance: First successful concat of OpenL3+HuBERT+SeNet on 941 samples; default RF/XGB settings.
- Data type/shape: 941 samples, embedding dim 3,328 (512+768+2,048).
- Structure/processing: Concatenate embeddings; same 80/20 stratified split; RF default-ish, XGB default-ish.
- Results (counts, %): RF TP=29 TN=71 FP=34 FN=55 (acc 52.9%); XGB TP=34 TN=69 FP=36 FN=50 (acc 54.5%).
- Interpretation: Clear improvement over HuBERT-only on this split; better recall and specificity.
- Suggested changes: Hyperparameter tuning to boost recall without excessive FP (see tuned run).

4) tuned_combined_openl3_hubert_senet.json
- Significance: Concat embeddings with stronger RF/XGB hyperparameters (more trees, class balancing, tweaked depths/cols).
- Data type/shape: 941 samples, embedding dim 3,328.
- Structure/processing: Same 80/20 stratified split; RF 800 trees balanced; XGB 1000 trees, depth 5, lr 0.05, subsample/colsample 0.9, scale_pos_weight.
- Results (counts, %): RF TP=30 TN=72 FP=33 FN=54 (acc 54.0%); XGB TP=37 TN=64 FP=41 FN=47 (acc 53.4%).
- Interpretation: Best recall on this split; RF slightly improves balance; XGB trades some specificity for higher TP.
- Suggested changes: Explore additional tuning (depth/min_child_weight) or cross-validation; consider class-weighted threshold adjustment if FP cost is high.

Comparison / best choice for your 941-sample dataset
- Best overall on this small set: tuned_combined_openl3_hubert_senet (highest TP/recall with reasonable TN vs other 941 runs).
- HuBERT-only on 941 samples performs worst; concatenation helps.
- The AVDF HuBERT baseline shows much higher metrics but on a much larger/different dataset; not directly comparable to the 941-set results.
- If your goal is better embeddings for this 941-sample dataset, prefer the combined embeddings (tuned). For alignment with the large AVDF setting, rerun combined embeddings on the larger HDF5 or rerun HuBERT-only on exactly the same data/split as combined.

5) combined_openl3_hubert_senet_full_2026-01-20T18:12:32.706941+00:00_tree_results.json
- Significance: Full-dataset concat (OpenL3+HuBERT+SeNet) directly from deepfake_embeddings_2.h5; evaluated on the same 80/20 stratified split as avdf_hubert_original.
- Data type/shape: 230,178 segment samples; combined dim 3,328; train 184,142 | test 46,036.
- Structure/processing: Filter av1m/avdeepfake videos; require all three embeddings; concatenate per segment; split stratified (random_state=42); RF (lighter, 100 trees, balanced); XGB (200 trees, depth 4, lr 0.1, subsample/colsample 0.8, scale_pos_weight=25.4).
- Results (counts, %): RF TP=615 TN=44,292 FP=1 FN=1,128 (acc 97.6%, prec 99.8%, rec 35.3%, spec ≈100%); XGB TP=1,516 TN=40,719 FP=3,574 FN=227 (acc 91.7%, prec 29.8%, rec 87.0%, spec 91.9%).
- Interpretation: RF is extremely conservative (near-perfect specificity/precision, low recall); XGB is recall-heavy with meaningful accuracy but many FPs. Shows concatenation scales to the full dataset with competitive aggregate accuracy; choice depends on precision/recall trade-off.
- Suggested changes: Tune thresholds or class weights depending on FP tolerance; expand trees/regularization sweep to lift XGB precision without crushing recall; consider calibrating decision threshold for RF to recover recall.

Comparison including full-dataset run
- On the full dataset, concatenated embeddings with XGB give strong recall (87%) but low precision (29.8%); RF gives high precision/specificity (99.8%/≈100%) but low recall (35.3%), overall acc 97.6%.
- Compared to the large HuBERT-only AVDF baseline (acc 96–98% with better precision/recall balance), full concatenation improves recall (XGB) but hurts precision; RF preserves precision but recalls less than HuBERT-only.
- For broad deployment on the full set: use the full concatenated embeddings; pick RF if you need ultra-low FP, or XGB if you need high recall and can tolerate more FP. Further tuning/thresholding can push toward a better balance. The 941-sample results remain for small-set comparison only.
