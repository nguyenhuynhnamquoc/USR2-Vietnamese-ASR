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