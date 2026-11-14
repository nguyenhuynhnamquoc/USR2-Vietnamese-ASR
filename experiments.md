# Vietnamese ASR — Experiment Log

## EXP-001: Pilot Fine-tuning with 100 Training Samples

**Date:** October 28, 2026  
**Status:** Completed
**Objective:** Verify the Vietnamese ASR fine-tuning pipeline and assess initial model behavior using a small subset of VIVOS.

### 1. Experimental Setup

| Parameter | Configuration |
|---|---|
| Base model | USR 2.0 Base+ |
| Pretrained checkpoint | Base+ High-resource (LRS3 + VoxCeleb2) |
| Loaded encoder tensors | 376/376 |
| Dataset | VIVOS Vietnamese Speech Corpus |
| Available training samples | 10,260 |
| Available validation samples | 1,400 |
| Training batches processed | 100 |
| Validation batches processed | 20 |
| Batch size | 1 |
| Epochs | 1 limited epoch |
| Learning rate | 1e-4 (default) |
| Optimizer | AdamW |
| Loss function | CTC Loss |
| Precision | CUDA mixed precision (FP16) |
| GPU | NVIDIA GeForce RTX 4060 Laptop 8 GB |
| Output directory | `output/pilot_100/` |

### 2. Experimental Results

| Metric | Training | Validation |
|---|---:|---:|
| CTC Loss | 15.5535 | 6.6613 |
| WER | 171.45% | 100.00% |

The model successfully completed 100 training batches and 20 validation batches without reporting CUDA out-of-memory errors.

The run produced `best.pt` and `last.pt` checkpoints.

### 3. Results Analysis

**Training behavior:** The pipeline successfully loaded the pretrained USR 2.0 audio encoder and executed forward propagation, CTC loss computation, backpropagation, and optimizer updates.

**CTC Loss:** Both training and validation losses were finite. However, the two losses cannot be interpreted as evidence of improvement without a comparable baseline or learning curve.

**Word Error Rate:** The training WER was 171.45%, while the validation WER reached 100%. WER can exceed 100% when insertion errors are numerous. These values indicate poor recognition performance at this early training stage.

**Important limitation:** Because only 100 training samples and 20 validation samples were processed, the results do not represent performance on the full VIVOS dataset.

### 4. Conclusions

The experiment confirms that the Vietnamese ASR training pipeline can execute successfully on the available GPU hardware.

However, it does not yet demonstrate effective Vietnamese speech recognition or model convergence.

### 5. Next Experiments

- Inspect reference transcripts and predicted transcriptions to identify CTC blank collapse or excessive insertions.
- Track loss and WER across multiple epochs.
- Evaluate on a fixed validation subset for comparable results.
- Investigate learning-rate adjustments and encoder-freezing strategies.
- Report full validation WER and CER after sufficient training.

**Research note:** This experiment is a pipeline-validation pilot, not a final ASR benchmark.


---

## EXP-002: Three-Epoch Pilot Fine-tuning

**Date:** October 28, 2026  
**Status:** Completed  
**Objective:** Investigate training-loss convergence and recognition performance across three limited epochs using USR 2.0 Base+ on Vietnamese VIVOS.

### 1. Experimental Setup

| Parameter | Configuration |
|---|---|
| Base model | USR 2.0 Base+ |
| Pretrained checkpoint | Base+ High-resource (LRS3 + VoxCeleb2) |
| Loaded encoder tensors | 376/376 |
| Dataset | VIVOS Vietnamese Speech Corpus |
| Training batches per epoch | 100 |
| Validation batches per epoch | 20 |
| Epochs | 3 limited epochs |
| Batch size | 1 |
| Learning rate | 1e-4 |
| Optimizer | AdamW |
| Loss function | CTC Loss |
| Precision | CUDA mixed precision (FP16) |
| GPU | NVIDIA RTX 4060 Laptop 8 GB |
| Output directory | `output/pilot_3epochs/` |

Training samples were shuffled each epoch. Therefore, the 100 batches did not necessarily contain the same utterances across epochs. Validation used the first 20 batches consistently.

### 2. Experimental Results

| Epoch | Train Loss | Train WER | Validation Loss | Validation WER |
|---|---:|---:|---:|---:|
| 1 | 15.5535 | 171.45% | 6.6613 | 100.00% |
| 2 | 6.7272 | 100.00% | 6.4742 | 100.00% |
| 3 | 6.6180 | 100.00% | 6.4635 | 100.00% |

The experiment completed successfully without reported CUDA out-of-memory errors.

### 3. Results Analysis

**Training loss convergence:** Training loss decreased from 15.5535 to 6.6180, representing a reduction of approximately 57.4%. This indicates that the optimization process reduced the CTC objective over the three limited epochs.

**Validation loss:** Validation loss decreased from 6.6613 to 6.4635, approximately 3.0%. However, the decrease was relatively small compared with the training loss.

**Recognition performance:** Validation WER remained at 100% throughout all three epochs. Therefore, the observed reduction in CTC loss has not yet translated into measurable recognition improvement on the selected validation subset.

**Potential CTC blank collapse:** The model may be producing predominantly blank outputs or incorrect token sequences. This remains a hypothesis and requires inspection of decoded predictions.

**Comparison with EXP-001:** The first epoch reproduced the EXP-001 metrics exactly. Both experiments used the same initialization, seed, training batch limit, and validation batch limit.

### 4. Checkpoint Analysis

The training script selects the best checkpoint using validation WER.

Because validation WER remained at 100% throughout all epochs, `best.pt` corresponds to Epoch 1, while `last.pt` contains the model state from Epoch 3.

This distinction is important because validation loss improved after Epoch 1 even though validation WER did not.

### 5. Limitations

- Only 100 training batches were processed per epoch.
- Only 20 validation samples were evaluated.
- The training subset was reshuffled between epochs.
- No character error rate (CER) was reported.
- No decoded transcript examples were inspected.
- The official VIVOS test set was not evaluated.

These results should not be treated as a final Vietnamese ASR benchmark.

### 6. Conclusions

The model successfully completed three limited fine-tuning epochs with decreasing CTC loss.

However, recognition performance remained unsatisfactory, with validation WER fixed at 100%.

The next priority should be diagnosing prediction behavior before increasing the training duration.

### 7. Next Steps

1. Decode several validation samples and compare predictions with reference transcripts.
2. Measure the proportion of blank CTC predictions.
3. Verify CTC decoding and tokenizer mapping.
4. Investigate learning-rate and encoder-freezing strategies.
5. Continue training only after confirming that the model can produce meaningful Vietnamese predictions.

**Research note:** EXP-002 demonstrates optimization progress but does not yet establish effective Vietnamese speech recognition.

---

## EXP-003: CTC Prediction and Blank Collapse Diagnostics

**Date:** October 28, 2026  
**Status:** Completed  
**Objective:** Diagnose the persistent 100% validation WER observed in EXP-002.

### 1. Experimental Setup

| Parameter | Configuration |
|---|---|
| Model | USR 2.0 Base+ Vietnamese CTC |
| Checkpoint | `output/pilot_3epochs/last.pt` |
| Trained epoch | 3 |
| Dataset | VIVOS validation |
| Evaluated samples | 5 |
| Decoding | CTC greedy decoding |
| Device | NVIDIA RTX 4060 (CUDA) |
| Additional training | None |

### 2. Results

| Sample | Prediction | CTC Blank Ratio |
|---|---|---:|
| 1 | Empty string | 100.00% |
| 2 | Empty string | 100.00% |
| 3 | Empty string | 100.00% |
| 4 | Empty string | 100.00% |
| 5 | Empty string | 100.00% |

All five reference transcripts contained valid Vietnamese text, while the model produced empty predictions.

### 3. Analysis

The model exhibited complete greedy-decoding blank dominance across all five examined validation utterances.

This behavior explains the observed 100% WER on these samples: every reference token was omitted.

Despite decreasing CTC loss during EXP-002, the model had not learned to generate meaningful Vietnamese transcriptions for the inspected utterances.

The blank ratio measures the proportion of frames where the blank class has the highest logit, not its predicted probability.

Potential contributing factors include insufficient training iterations, an untrained CTC projection layer, and unsuitable fine-tuning hyperparameters.

These causes have not yet been isolated experimentally.

### 4. Conclusion

EXP-003 confirms CTC blank-dominated predictions on five validation utterances after three limited training epochs.

The current checkpoint is not yet suitable for Vietnamese ASR deployment or performance benchmarking.

### 5. Recommended Next Experiment

**EXP-004: Single-Sample Overfitting Test**

- Select one fixed training utterance.
- Train on that same utterance repeatedly.
- Track CTC loss, decoded transcription, and blank ratio.
- Compare frozen-encoder training against full fine-tuning if needed.
- Verify whether the model can memorize a single Vietnamese sentence.

A successful single-sample overfitting test would support the correctness of the training and decoding pipeline, though it would not establish generalization.



## EXP-004: Single-Sample Overfitting — Frozen Encoder

**Date:** October 28, 2026  
**Status:** Completed — exact-match criterion not reached  
**Objective:** Test whether a newly initialized Vietnamese CTC head can memorize a single VIVOS utterance using frozen USR 2.0 Base+ encoder features.

### Experimental Configuration

| Parameter | Value |
|---|---|
| Model | USR 2.0 Base+ |
| Pretrained encoder tensors | 376 loaded |
| Dataset | VIVOS training |
| Training utterance | KHÁCH SẠN |
| Audio duration | 3.0 seconds |
| Training samples | 1 |
| Encoder | Frozen, evaluation mode |
| CTC head | Trainable, newly initialized |
| Optimizer | AdamW |
| Learning rate | 0.001 |
| Weight decay | 0 |
| Training steps | 200 |
| GPU | NVIDIA RTX 4060 Laptop |

### Experimental Results

| Step | CTC Loss | Blank Ratio | Decoded Prediction |
|---|---:|---:|---|
| 1 | 96.8517 | 0.00% | Long, incorrect transcription |
| 10 | 61.6775 | 18.67% | KHÁCH N NSẠN |
| 50 | 13.5414 | 12.00% | KHÁCH N NSẠN |
| 100 | 9.6489 | 12.00% | KHÁCH N SẠN |
| 150 | 6.7398 | 12.00% | KHÁCH N SẠN |
| 200 | 4.4338 | 12.00% | KHÁCH N SẠN |

### Analysis

CTC loss decreased by approximately 95.42% over 200 optimization steps.

The decoded output progressively became more similar to the reference transcription. At step 200, the model generated both reference words but retained an extra token, `N`.

The resulting utterance-level WER was 50%, calculated using whitespace-separated words.

The experiment provides evidence that a trainable CTC projection can learn from frozen pretrained audio features. However, the model did not achieve exact-match memorization within 200 steps.

Importantly, this experiment trained a newly initialized CTC head. It did not continue training the EXP-002 checkpoint and therefore does not establish that the previous checkpoint recovered from blank-dominated predictions.

### Conclusion

The frozen-encoder single-sample training test showed substantial optimization progress and increasingly accurate transcriptions, but did not reach the exact-match success criterion.

### Next Action

Extend the single-sample training experiment to 1,000 steps while preserving the same model configuration, sample, and learning rate. Evaluate whether exact-match decoding is achieved.


### EXP-004 Extension: 1,000-Step Single-Sample Overfitting

**Status:** Completed — Exact-match criterion not reached

**Objective:** Extend the frozen-encoder, single-utterance training test from 200 to 1,000 optimization steps.

| Step | CTC Loss | Blank Ratio | Prediction |
|---|---:|---:|---|
| 1 | 96.8517 | 0.00% | Long incorrect transcription |
| 100 | 9.6489 | 12.00% | KHÁCH N SẠN |
| 200 | 4.4338 | 12.00% | KHÁCH N SẠN |
| 500 | 0.9090 | 12.00% | KHÁCH N SẠN |
| 750 | 0.6607 | 12.00% | KHÁCH N SẠN |
| 1,000 | 0.5523 | 12.00% | KHÁCH N SẠN |

**Reference:** KHÁCH SẠN

**Final utterance-level WER:** 50%.

**Analysis:** CTC loss decreased by approximately 99.43% from its initial value. However, the greedy-decoded transcription retained an additional `N` token after 1,000 steps. The prediction remained unchanged from step 100 onwards, despite continued loss reduction.

The results demonstrate substantial optimization progress without exact-match memorization. This does not yet establish a decoder implementation bug or a model architecture defect.

**Next experiment:** Inspect SentencePiece token IDs, greedy-decoded token sequences, and CTC alignment behavior using the trained projection head.

**Conclusion:** Further diagnostics are recommended before increasing the number of training steps or expanding the dataset.

---

## EXP-005: CTC Tokenization and Alignment Diagnostics

**Status:** Completed  
**Objective:** Identify the source of the additional `N` token observed in EXP-004.

### Experimental Setup

- Model: USR 2.0 Base+ with frozen encoder
- CTC head: Trained for 1,000 steps in EXP-004
- Checkpoint: `output/exp004/ctc_head_1000.pt`
- Evaluation utterance: `KHÁCH SẠN`
- Decoding: Greedy CTC
- Additional training: None

### Results

**Reference token sequence:**

`['▁KHÁCH', '▁', 'S', 'Ạ', 'N']`

**Predicted token sequence:**

`['▁KHÁCH', '▁', 'N', '▁', 'S', 'Ạ', 'N']`

| Frame | Predicted Token | Top-1 Probability |
|---|---|---:|
| 0 | ▁KHÁCH | 0.9432 |
| 4 | ▁ | 0.9823 |
| 13 | N | 0.7520 |
| 14 | ▁ | 0.9969 |
| 15 | S | 0.8919 |
| 23 | Ạ | 0.9868 |
| 39 | N | 0.9305 |

**CTC loss:** 0.551988  
**Blank argmax ratio:** 12.00%  
**Mean blank probability:** 0.1193

### Analysis

The additional `N` originates from a CTC non-blank prediction at frame 13. A subsequent word-boundary token at frame 14 produces the extra word in the final transcription.

The SentencePiece decoder correctly converts the predicted token sequence into `KHÁCH N SẠN`. Therefore, the insertion is not introduced by SentencePiece text decoding.

The CTC objective and greedy decoding follow different optimization criteria. A decreasing CTC loss does not guarantee that the greedy-decoded sequence matches the reference.

### Conclusion

The unexpected `N` is attributable to the model's predicted CTC token sequence.

The experiment does not establish an implementation defect in the tokenizer or decoder.

### Next Experiment

**EXP-006:** Compare greedy decoding against CTC prefix beam search using the same checkpoint and audio sample, without additional training.

---

## EXP-006: Greedy Decoding vs. CTC Prefix Beam Search

**Status:** Completed  
**Objective:** Determine whether CTC prefix beam search can eliminate the extra `N` token observed in EXP-004 and EXP-005.

### Experimental Configuration

| Parameter | Configuration |
|---|---|
| Model | USR 2.0 Base+ with frozen encoder |
| Checkpoint | `output/exp004/ctc_head_1000.pt` |
| Evaluation sample | KHÁCH SẠN |
| Decoding methods | Greedy CTC and prefix beam search |
| Beam width | 50 |
| Top-k candidates per frame | 30 |
| Additional training | None |

### Results

| Decoding Method | Predicted Transcript | WER | Exact Match |
|---|---|---:|---|
| Greedy | KHÁCH N SẠN | 50% | No |
| CTC Prefix Beam Search | KHÁCH N SẠN | 50% | No |

**Beam search token IDs:** `[260, 3, 4, 3, 988, 986, 4]`

### Analysis

Both decoding strategies produced identical transcriptions with an additional `N` token.

The tested beam-search configuration did not improve recognition accuracy over greedy decoding.

The result suggests that the insertion error cannot be resolved merely by switching between these two decoder configurations.

However, the beam search is approximate due to beam-width and per-frame candidate pruning. The experiment does not rule out improvements from alternative decoding configurations or language-model integration.

### Conclusion

No decoding improvement was observed on the examined utterance.

Further investigation should focus on CTC model training and alignment behavior.

### Next Experiment

**EXP-007:** Continue training the saved CTC projection head on the same utterance for up to 500 additional optimization steps, without updating the pretrained encoder.


---

## EXP-007: Continued Single-Sample CTC Head Training

**Date:** October 28, 2026  
**Status:** Completed — Exact-match memorization achieved

### 1. Objective

Determine whether continued optimization of the CTC projection head can eliminate the additional `N` token observed in previous experiments.

### 2. Experimental Configuration

| Parameter | Configuration |
|---|---|
| Base model | USR 2.0 Base+ |
| Dataset | VIVOS |
| Training sample | KHÁCH SẠN |
| Audio duration | 3.0 seconds |
| Encoder | Frozen |
| CTC head | Trainable |
| Initial CTC weights | EXP-004, 1,000-step checkpoint |
| Optimizer | AdamW, newly initialized |
| Learning rate | 0.001 |
| Additional training steps | 149 |
| Device | NVIDIA RTX 4060 Laptop GPU |

The experiment reused the CTC head weights learned in EXP-004 but initialized a new optimizer.

### 3. Experimental Results

| Step | CTC Loss | Blank Ratio | Prediction |
|---|---:|---:|---|
| 1 | 0.5520 | 12.00% | KHÁCH N SẠN |
| 50 | 0.3943 | 12.00% | KHÁCH N SẠN |
| 100 | 0.3083 | 12.00% | KHÁCH N SẠN |
| 140 | 0.2563 | 12.00% | KHÁCH N SẠN |
| 149 | 0.2462 | 12.00% | KHÁCH SẠN |

**Final exact match:** Yes  
**Final training-utterance WER:** 0%

### 4. Results Analysis

The additional `N` token persisted during the first 148 steps but disappeared at step 149.

The reported CTC loss decreased from 0.5520 to 0.2462, a reduction of approximately 55.4%.

The model successfully memorized the selected Vietnamese training utterance using a frozen pretrained audio encoder and an optimized CTC projection head.

These results demonstrate that the tested architecture can produce an exact reference transcription for one training sample.

However, successful single-utterance memorization does not establish acoustic generalization or reliable recognition of unseen Vietnamese speech.

### 5. Conclusion

The single-sample overfitting test achieved its exact-match criterion.

The next priority is to investigate whether the model can learn multiple different utterances and distinguish their corresponding audio inputs.

### 6. Next Experiment

**EXP-008: Small-Subset Overfitting and Generalization Test**

Train a frozen-encoder CTC head on a fixed subset of 5–10 different Vietnamese utterances.

Evaluate training recognition accuracy and compare with a separate, unseen validation subset.

The experiment should assess whether the model learns useful audio-to-text mappings beyond memorizing a single training example.

## EXP-008: Small-Subset Overfitting and Generalization Test

**Status:** Completed
**Objective:** Test whether a newly initialized CTC head can learn multiple VIVOS utterances with the encoder frozen, and compare performance on separate validation utterances.

### 1. Experimental Setup

| Parameter | Configuration |
|---|---|
| Encoder | USR 2.0 Base+ pretrained checkpoint, frozen |
| CTC head | Newly initialized and trainable |
| Training subset | 10 utterances sampled deterministically from the training manifest |
| Validation subset | 10 separate utterances sampled deterministically from the validation manifest |
| Random seed | 42 |
| Optimizer | AdamW |
| Learning rate | 0.001 |
| Training steps | 1,000 |
| Loss | CTC |
| Evaluation metrics | CTC loss, word error rate, character error rate, exact-match rate, blank ratio |
| Default device | CUDA when available; otherwise CPU |
| Script | `scripts/exp008_multi_sample.py` |
| Output directory | `output/exp008/` |

The script selects the training and validation subsets independently, caches the frozen encoder features, and records the selected manifest indices for reproducibility. The validation subset is not used for optimization.

Run from the repository root in PowerShell:

```powershell
python scripts\exp008_multi_sample.py
```

To explicitly run on the GPU or adjust the experiment size:

```powershell
python scripts\exp008_multi_sample.py `
  --device cuda `
  --train-samples 10 `
  --validation-samples 10 `
  --steps 1000 `
  --seed 42
```

The script writes the trained CTC head to `output/exp008/ctc_head.pt`, summary metrics and selected indices to `output/exp008/summary.json`, and decoded train/validation examples to `output/exp008/predictions.csv`.

### 2. Experimental Results

The experiment ran on CPU for 1,000 steps. The pretrained encoder loaded 376 tensors. The fixed subsets were selected with seed 42.

| Split | Step | CTC loss | WER | CER | Exact match | Blank ratio |
|---|---:|---:|---:|---:|---:|---:|
| Train | 0 | 50.3706 | 559.04% | 520.11% | 0% | 0.26% |
| Train | 1,000 | 0.2335 | 8.43% | 6.06% | 50% | 73.39% |
| Validation | 0 | 35.6673 | 484.44% | 455.35% | 0% | 0.72% |
| Validation | 1,000 | 8.5081 | 100.00% | 95.45% | 0% | 98.99% |

The run saved the CTC head, metrics, and decoded examples under `output/exp008/`.

### 3. Results Analysis

Training metrics improved substantially: WER fell from 559.04% to 8.43%, and 5 of 10 training utterances reached exact match. This shows the newly initialized CTC head can fit the selected training examples with the encoder frozen.

Validation did not improve to usable recognition: WER remained 100%, no utterance reached exact match, and the final blank-frame ratio was 98.99%. The CTC loss decreased, but validation predictions remained mostly blank or very short. The large train/validation gap indicates the observed training fit did not generalize to this small held-out subset.

### 4. Conclusion

EXP-008 demonstrates partial memorization of the 10 training utterances, not generalization to unseen VIVOS utterances. These 10 validation examples are diagnostic only and do not replace evaluation on the official test split.

### 5. Follow-up

EXP-009 tests whether reducing the learning rate improves held-out decoding while keeping the subsets, seed, architecture, and step count fixed.

---

## EXP-009: Lower Learning Rate on the EXP-008 Subsets

**Status:** Completed
**Objective:** Test whether reducing the CTC-head learning rate from `1e-3` to `1e-4` reduces validation blank collapse.

### 1. Experimental Setup

EXP-009 reused `scripts/exp008_multi_sample.py` with the same 10 training utterances, 10 validation utterances, seed 42, frozen encoder, and 1,000 training steps as EXP-008. The only changed training parameter was the learning rate (`1e-4`). The run used CPU and saved outputs under `output/exp009_lr1e4/`.

### 2. Experimental Results

| Split | Step | CTC loss | WER | CER | Exact match | Blank ratio |
|---|---:|---:|---:|---:|---:|---:|
| Train | 0 | 50.3706 | 559.04% | 520.11% | 0% | 0.26% |
| Train | 1,000 | 4.0166 | 54.22% | 48.76% | 0% | 89.72% |
| Validation | 0 | 35.6673 | 484.44% | 455.35% | 0% | 0.72% |
| Validation | 1,000 | 7.4697 | 100.00% | 100.00% | 0% | 100.00% |

The run produced `ctc_head.pt`, `summary.json`, and `predictions.csv`. The summary JSON identifies itself as `EXP-008` because it was generated by the EXP-008 script; the run configuration and output directory identify this lower-learning-rate follow-up.

### 3. Results Analysis

Reducing the learning rate did not resolve validation blank collapse: validation WER remained 100%, exact match remained 0%, and blank ratio reached 100%. The validation CTC loss was slightly lower than EXP-008's final loss (7.4697 vs. 8.5081), but decoded recognition did not improve.

Training also fit less well than EXP-008: final training WER was 54.22% and exact match was 0%, compared with 8.43% WER and 50% exact match at `1e-3`. On these fixed 10-example subsets, the lower learning rate reduced training fit without improving validation decoding.

### 4. Conclusion and Next Experiment

The learning-rate reduction alone is insufficient to improve generalization. EXP-010 will increase the fixed training and validation subsets while keeping the encoder frozen and learning rate at `1e-3`. The official test split remains reserved for final evaluation.

---

## EXP-010: Larger-Subset Frozen-Encoder CTC Test

**Status:** Completed
**Objective:** Test whether more training utterances improve held-out decoding under the EXP-008 learning rate, using a larger validation subset for a more informative estimate.

### 1. Experimental Setup

| Parameter | Configuration |
|---|---|
| Encoder | USR 2.0 Base+ pretrained checkpoint, frozen |
| CTC head | Newly initialized and trainable |
| Training subset | 50 VIVOS training utterances |
| Validation subset | 50 separate VIVOS validation utterances |
| Random seed | 42 |
| Optimizer | AdamW |
| Learning rate | 0.001 |
| Training steps | 5,000 (100 updates per training utterance) |
| Device | CPU, matching EXP-008 and EXP-009 |
| Output directory | `output/exp010/` |

This run keeps the model and learning rate from EXP-008, but increases both subsets. The 50 validation utterances are used only for evaluation, not optimization.

Run from the repository root in PowerShell:

```powershell
.venv\Scripts\python.exe scripts\exp008_multi_sample.py `
  --experiment-id EXP-010 `
  --train-samples 50 `
  --validation-samples 50 `
  --steps 5000 `
  --learning-rate 0.001 `
  --log-every 250 `
  --seed 42 `
  --device cpu `
  --output-dir output\exp010
```

The script saves `ctc_head.pt`, `summary.json` (labeled `EXP-010`), and `predictions.csv` under `output/exp010/`. The selected manifest indices are recorded in the summary for reproducibility.

### 2. Experimental Results

The run completed on CPU, loaded 376 pretrained encoder tensors, and saved its outputs under `output/exp010/`.

| Split | Step | CTC loss | WER | CER | Exact match | Blank ratio |
|---|---:|---:|---:|---:|---:|---:|
| Train | 0 | 47.2045 | 511.36% | 478.05% | 0% | 0.16% |
| Train | 5,000 | 0.4021 | 13.25% | 11.19% | 22% | 77.91% |
| Validation | 0 | 38.6935 | 490.09% | 462.42% | 0% | 0.57% |
| Validation | 5,000 | 10.4737 | 99.57% | 94.62% | 0% | 98.00% |

### 3. Results Analysis

The CTC head fit the training subset to some extent: 11 of 50 training utterances reached exact match. However, the validation predictions remained mostly empty or very short and unrelated to the references. Validation WER stayed near 100%, with no exact matches, despite the larger validation set.

The blank ratio is high on validation, but should be interpreted together with decoded outputs and error rates because CTC commonly assigns many frames to the blank class. Here, the near-100% WER and predominantly empty/short predictions confirm that recognition did not generalize.

The EXP-008 and EXP-010 subsets are not identical: changing `sample_count` changes the deterministic sample selected by `random.sample`, even with the same seed. Therefore, their metrics are useful for diagnosing the overall pattern but are not a paired comparison on identical utterances.

### 4. Conclusion and Next Experiment

Increasing the number of examples to 50 and providing 100 updates per training utterance did not produce useful held-out recognition with the encoder frozen. The next experiment should compare the frozen encoder with partial encoder fine-tuning on a fixed train/validation subset, keeping the data split and evaluation procedure identical between runs. Continue reserving the official test split for final evaluation.

---

## EXP-011: Partial Fine-Tuning of the Last Encoder Block

**Status:** Completed
**Objective:** Test whether adapting the final transformer encoder block improves validation recognition compared with the frozen-encoder EXP-010 run.

### 1. Experimental Setup

| Parameter | Configuration |
|---|---|
| Encoder initialization | USR 2.0 Base+ pretrained checkpoint |
| Trainable encoder layers | Final Transformer block only |
| CTC head | Newly initialized and trainable |
| Training/validation subsets | Exact 50/50 indices recorded by EXP-010 |
| Encoder learning rate | 0.00001 |
| CTC-head learning rate | 0.001 |
| Training steps | 1,000 (20 updates per training utterance) |
| Random seed | 42 |
| Device | Auto-detect CUDA; otherwise CPU |
| Output directory | `output/exp011/` |

This is a compute-limited pilot because encoder fine-tuning is more expensive than the cached-feature runs. It trains the final encoder block with a lower learning rate than the new CTC projection and evaluates on the same held-out utterances as EXP-010.

Run from the repository root in PowerShell:

```powershell
.venv\Scripts\python.exe scripts\exp011_partial_finetune.py `
  --steps 1000 `
  --log-every 250 `
  --encoder-learning-rate 0.00001 `
  --head-learning-rate 0.001 `
  --seed 42 `
  --device auto `
  --output-dir output\exp011
```

By default, the script reads the fixed indices from `output/exp010/summary.json`. It saves the fine-tuned final block and CTC projection to `partial_finetune.pt`, plus `summary.json` and `predictions.csv`.

### 2. Experimental Results

The run completed on CPU, loaded 376 pretrained encoder tensors, and saved its outputs under `output/exp011/`.

| Split | Step | CTC loss | WER | CER | Exact match | Blank ratio |
|---|---:|---:|---:|---:|---:|---:|
| Train | 0 | 47.2045 | 511.36% | 478.05% | 0% | 0.16% |
| Train | 1,000 | 1.6072 | 36.28% | 33.83% | 8% | 85.53% |
| Validation | 0 | 38.6935 | 490.09% | 462.42% | 0% | 0.57% |
| Validation | 1,000 | 8.1626 | 99.57% | 96.39% | 0% | 98.82% |

### 3. Results Analysis

EXP-011 used the same train and validation utterances as EXP-010. Validation WER remained 99.57% with no exact matches, and the decoded validation predictions were still mostly empty or very short. The validation CTC loss decreased to 8.1626, but this did not improve word-level recognition.

EXP-011 ran for 1,000 steps, compared with 5,000 in EXP-010, so the training-fit metrics are not a convergence-matched comparison. EXP-011's training WER was 36.28% with 8% exact match at step 1,000; these results alone cannot establish that partial fine-tuning is worse or better after equal training exposure.

### 4. Conclusion and Next Experiment

Fine-tuning the final encoder block for 1,000 steps did not yet improve validation WER on the fixed EXP-010 split. The lower validation loss is not sufficient evidence of improved decoding. EXP-012 compared decoders and checked CTC alignment feasibility before another training run.

---

## EXP-012: CTC Decoding and Alignment Diagnostics

**Status:** Completed
**Objective:** Determine whether EXP-010/011 validation errors are sensitive to CTC decoding and check for blank-dominated outputs or impossible target alignments without training.

### 1. Experimental Setup

| Parameter | Configuration |
|---|---|
| Checkpoints | EXP-010 frozen encoder and EXP-011 partial fine-tune |
| Validation subset | Same 50 validation utterances used by EXP-010/011 |
| Decoders | Greedy and CTC prefix beam search |
| Beam width / per-frame top-k | 20 / 20 |
| Training | None |
| Output directory | `output/exp012/` |

Run from the repository root in PowerShell:

```powershell
.venv\Scripts\python.exe scripts\exp012_decode_diagnostics.py `
  --beam-width 20 `
  --top-k 20 `
  --device auto `
  --output-dir output\exp012
```

The script loads the checkpoint summaries and weights from EXP-010/011, verifies that their recorded splits match, and writes greedy/beam predictions and CTC diagnostics to `predictions.csv` and `summary.json`.

### 2. Experimental Results

The diagnostic ran on CPU without training. No validation sample violated the minimum number of CTC frames required by its target sequence.

| Checkpoint | Decoder | Mean CTC loss | WER | CER | Exact match | Empty predictions |
|---|---|---:|---:|---:|---:|---:|
| EXP-010 frozen encoder | Greedy | 10.4737 | 99.57% | 94.62% | 0% | 17/50 |
| EXP-010 frozen encoder | Beam (20/20) | 10.4737 | 99.35% | 94.16% | 0% | 14/50 |
| EXP-011 partial fine-tune | Greedy | 8.1626 | 99.57% | 96.39% | 0% | 24/50 |
| EXP-011 partial fine-tune | Beam (20/20) | 8.1626 | 99.57% | 96.04% | 0% | 21/50 |

| Checkpoint | Blank argmax ratio | Mean blank probability | CTC minimum-frame violations |
|---|---:|---:|---:|
| EXP-010 frozen encoder | 98.00% | 94.24% | 0 |
| EXP-011 partial fine-tune | 98.82% | 90.53% | 0 |

The diagnostic corrected CTC loss normalization to match the per-utterance, target-length-normalized loss reported by EXP-010/011.

### 3. Results Analysis

Beam search made only a small difference: EXP-010 WER changed from 99.57% to 99.35%, while EXP-011 remained at 99.57%. Neither decoder produced an exact match. The tested beam configuration alone does not explain the high WER.

All target sequences had enough encoder frames for a valid CTC alignment, so insufficient frame length is not the cause on this validation subset. Both checkpoints assigned most frames to blank and produced many empty or very short hypotheses. This points to poor learned token evidence on unseen utterances rather than a CTC length constraint. The result is limited to beam width/top-k 20 and this fixed 50-example validation subset.

### 4. Conclusion and Next Experiment

EXP-012 found no evidence that the tested decoding change or impossible CTC alignments caused the validation failure. EXP-013 uses a larger training subset and full validation with best-WER checkpointing before moving to a full-dataset run. Do not treat the 50-example diagnostic as official test performance.

---

## EXP-013: End-to-End Fine-Tuning on a 1,000-Utterance Subset

**Status:** Completed
**Objective:** Test whether full encoder adaptation on substantially more labeled VIVOS data improves WER, while keeping the official test split untouched.

### 1. Experimental Setup

| Parameter | Configuration |
|---|---|
| Encoder initialization | USR 2.0 Base+ pretrained checkpoint |
| Trainable parameters | Full audio encoder and CTC projection |
| Training subset | 1,000 utterances selected deterministically from the 10,260-sample train split |
| Validation | Full 1,400-utterance validation split each epoch |
| Epochs | 3 |
| Batch size | 2 |
| Encoder learning rate | 0.00001 |
| CTC-head learning rate | 0.001 |
| Weight decay | Encoder: 0.0001; CTC head: 0 |
| Random seed | 42 |
| Output directory | `output/exp013/` |

This is a three-epoch pilot on a fixed 1,000-example subset, not a full pass over all 10,260 training utterances. The script saves `best.pt` based on full-validation WER, `last.pt` with optimizer state for resuming, per-epoch metrics in `summary.json`, and greedy-decoded train/validation examples in `predictions.csv`. Predictions are generated from `best.pt`; the CSV contains `split`, `dataset_index`, `reference`, and `prediction`. Batch-limit smoke tests only export predictions for the batches they process.

Run from the repository root in PowerShell:

```powershell
.venv\Scripts\python.exe scripts\exp013_subset_finetune.py `
  --train-samples 1000 `
  --epochs 3 `
  --batch-size 2 `
  --encoder-learning-rate 0.00001 `
  --head-learning-rate 0.001 `
  --seed 42 `
  --device auto `
  --output-dir output\exp013
```

For a quick pipeline smoke test before the full pilot:

```powershell
.venv\Scripts\python.exe scripts\exp013_subset_finetune.py `
  --train-samples 20 `
  --epochs 1 `
  --limit-train-batches 2 `
  --limit-validation-batches 2 `
  --device auto `
  --output-dir output\exp013_smoke
```

### 2. Experimental Results

The run completed on a CUDA device for all three epochs. The full 1,400-utterance validation set was evaluated after each epoch; the official test set was not used.

| Epoch | Train loss | Train WER | Validation loss | Validation WER |
|---|---:|---:|---:|---:|
| 1 | 8.6582 | 103.70% | 4.2648 | 71.89% |
| 2 | 3.1051 | 58.10% | 2.6924 | 56.22% |
| 3 | 2.0125 | 44.81% | 2.3331 | 50.73% |

The best checkpoint is epoch 3 with validation WER 50.73%. `predictions.csv` contains 1,000 train-subset and 1,400 validation predictions. Exact matches were 95/1,000 (9.5%) on train and 83/1,400 (5.9%) on validation; neither split had empty predictions.

### 3. Interpretation and Next Step

Both training and validation WER improved at every epoch, and the final train/validation gap was about 5.9 percentage points. This is a substantial improvement over the near-100% WER in earlier experiments, although those used only 50 validation utterances, so the results are not a strict like-for-like comparison. Validation WER remains high and the sample predictions still contain many substitutions and omissions.

Continue from `last.pt` to five total epochs on the same fixed subset before increasing the training set. Keep the official VIVOS test split untouched until model selection is complete.