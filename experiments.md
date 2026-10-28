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