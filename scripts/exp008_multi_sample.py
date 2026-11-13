import argparse
import csv
import json
import random
import sys
from pathlib import Path

import sentencepiece as spm
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from data.vivos_asr import VIVOSASRDataset
from models.vietnamese_ctc import load_pretrained_audio_encoder
from train_asr import decode_greedy, make_model, edit_distance


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a frozen-encoder CTC head on a VIVOS subset."
    )
    parser.add_argument(
        "--experiment-id",
        default="EXP-008",
        help="Experiment label recorded in the output summary.",
    )
    parser.add_argument("--train-samples", type=int, default=10)
    parser.add_argument("--validation-samples", type=int, default=10)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--train-manifest",
        type=Path,
        default=PROJECT_ROOT / "dataset" / "manifests" / "train.tsv",
    )
    parser.add_argument(
        "--validation-manifest",
        type=Path,
        default=PROJECT_ROOT / "dataset" / "manifests" / "validation.tsv",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=PROJECT_ROOT / "dataset" / "vivos",
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=PROJECT_ROOT / "dataset" / "tokenizer" / "vivos_vi.model",
    )
    parser.add_argument(
        "--backbone-config",
        type=Path,
        default=PROJECT_ROOT
        / "conf"
        / "model"
        / "backbone"
        / "resnet_transformer_baseplus.yaml",
    )
    parser.add_argument(
        "--pretrained-checkpoint",
        type=Path,
        default=PROJECT_ROOT / "path" / "baseplus_high_resource_lrs3vox2.pth",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "output" / "exp008",
    )
    return parser.parse_args()


def normalize_text(text):
    return " ".join(text.upper().split())


def character_edit_distance(prediction, reference):
    previous = list(range(len(reference) + 1))
    for row, predicted_char in enumerate(prediction, start=1):
        current = [row]
        for column, reference_char in enumerate(reference, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (predicted_char != reference_char),
                )
            )
        previous = current
    return previous[-1]


def choose_indices(dataset_size, sample_count, seed):
    if not 1 <= sample_count <= dataset_size:
        raise ValueError(
            f"Requested {sample_count} samples, but the split contains "
            f"{dataset_size} utterances."
        )
    return sorted(random.Random(seed).sample(range(dataset_size), sample_count))


def cache_features(model, dataset, indices, device):
    cached = []
    model.encoder.eval()
    for index in indices:
        sample = dataset[index]
        audio = sample["audio"].unsqueeze(0).to(device)
        frame_count = audio.shape[1] // 640
        if frame_count <= 0:
            raise ValueError(f"Audio at dataset index {index} is too short.")
        mask = torch.ones((1, 1, frame_count), dtype=torch.bool, device=device)

        with torch.no_grad(), torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            features = model.encoder(
                xs_a=audio.unsqueeze(-1),
                masks=mask,
            ).float()

        targets = sample["tokens"].to(device)
        if targets.numel() > features.shape[1]:
            raise ValueError(
                f"Transcript at dataset index {index} has more tokens than "
                "the encoder produced frames."
            )
        cached.append(
            {
                "index": index,
                "features": features.detach(),
                "targets": targets,
                "input_length": torch.tensor(
                    [features.shape[1]], dtype=torch.long
                ),
                "target_length": torch.tensor(
                    [targets.numel()], dtype=torch.long
                ),
                "text": sample["text"],
            }
        )
    return cached


def evaluate(head, examples, tokenizer, criterion):
    head.eval()
    total_loss = 0.0
    word_errors = 0
    reference_words = 0
    char_errors = 0
    reference_chars = 0
    exact_matches = 0
    blank_frames = 0
    total_frames = 0
    results = []

    with torch.inference_mode():
        for example in examples:
            logits = head(example["features"])
            log_probs = logits.float().log_softmax(dim=-1).transpose(0, 1)
            loss = criterion(
                log_probs,
                example["targets"],
                example["input_length"],
                example["target_length"],
            )
            prediction = decode_greedy(
                logits, example["input_length"], tokenizer
            )[0]
            reference = example["text"]
            normalized_prediction = normalize_text(prediction)
            normalized_reference = normalize_text(reference)
            predicted_words = normalized_prediction.split()
            reference_word_list = normalized_reference.split()
            word_errors += edit_distance(predicted_words, reference_word_list)
            reference_words += len(reference_word_list)
            char_errors += character_edit_distance(
                normalized_prediction, normalized_reference
            )
            reference_chars += len(normalized_reference)
            exact_matches += normalized_prediction == normalized_reference

            frame_classes = logits.argmax(dim=-1)
            blank_frames += int((frame_classes == 0).sum().item())
            total_frames += frame_classes.numel()
            total_loss += float(loss.item())
            results.append(
                {
                    "dataset_index": example["index"],
                    "reference": reference,
                    "prediction": prediction,
                }
            )

    if reference_words == 0 or reference_chars == 0 or total_frames == 0:
        raise ValueError("Cannot calculate metrics for an empty evaluation set.")
    count = len(examples)
    return {
        "loss": total_loss / count,
        "wer": word_errors / reference_words,
        "cer": char_errors / reference_chars,
        "exact_match": exact_matches / count,
        "blank_ratio": blank_frames / total_frames,
        "predictions": results,
    }


def print_metrics(step, training, validation):
    print(
        f"Step {step:4d} | "
        f"Train loss {training['loss']:.4f}, WER {training['wer']:.2%}, "
        f"CER {training['cer']:.2%}, exact {training['exact_match']:.0%} | "
        f"Validation loss {validation['loss']:.4f}, "
        f"WER {validation['wer']:.2%}, CER {validation['cer']:.2%}, "
        f"exact {validation['exact_match']:.0%}, "
        f"blank {validation['blank_ratio']:.2%}",
        flush=True,
    )


def main():
    args = parse_args()
    if args.steps <= 0 or args.log_every <= 0 or args.learning_rate <= 0:
        raise ValueError("--steps, --log-every, and --learning-rate must be positive.")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    device = torch.device(
        "cuda" if args.device == "cuda" or (
            args.device == "auto" and torch.cuda.is_available()
        ) else "cpu"
    )

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(4)

    tokenizer = spm.SentencePieceProcessor(model_file=str(args.tokenizer))
    train_dataset = VIVOSASRDataset(
        args.train_manifest, args.data_root, args.tokenizer
    )
    validation_dataset = VIVOSASRDataset(
        args.validation_manifest, args.data_root, args.tokenizer
    )
    train_indices = choose_indices(
        len(train_dataset), args.train_samples, args.seed
    )
    validation_indices = choose_indices(
        len(validation_dataset), args.validation_samples, args.seed + 1
    )

    model = make_model(tokenizer, args.backbone_config)
    matched, _ = load_pretrained_audio_encoder(
        model, args.pretrained_checkpoint
    )
    model.to(device)
    model.encoder.eval()
    model.ctc_projection.train()

    print(f"Device: {device}; loaded encoder tensors: {matched}")
    print(
        f"Caching frozen encoder features for {len(train_indices)} train and "
        f"{len(validation_indices)} validation utterances..."
    )
    training_examples = cache_features(
        model, train_dataset, train_indices, device
    )
    validation_examples = cache_features(
        model, validation_dataset, validation_indices, device
    )

    criterion = torch.nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = torch.optim.AdamW(
        model.ctc_projection.parameters(),
        lr=args.learning_rate,
        weight_decay=0,
    )
    training_order = list(range(len(training_examples)))

    initial_train = evaluate(
        model.ctc_projection, training_examples, tokenizer, criterion
    )
    initial_validation = evaluate(
        model.ctc_projection, validation_examples, tokenizer, criterion
    )
    print_metrics(0, initial_train, initial_validation)

    for step in range(1, args.steps + 1):
        if (step - 1) % len(training_order) == 0:
            random.shuffle(training_order)
        example = training_examples[training_order[(step - 1) % len(training_order)]]

        model.ctc_projection.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model.ctc_projection(example["features"])
        log_probs = logits.float().log_softmax(dim=-1).transpose(0, 1)
        loss = criterion(
            log_probs,
            example["targets"],
            example["input_length"],
            example["target_length"],
        )
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite CTC loss at step {step}.")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.ctc_projection.parameters(), max_norm=5.0
        )
        optimizer.step()

        if step % args.log_every == 0 or step == args.steps:
            training_metrics = evaluate(
                model.ctc_projection, training_examples, tokenizer, criterion
            )
            validation_metrics = evaluate(
                model.ctc_projection, validation_examples, tokenizer, criterion
            )
            print_metrics(step, training_metrics, validation_metrics)

    final_training = evaluate(
        model.ctc_projection, training_examples, tokenizer, criterion
    )
    final_validation = evaluate(
        model.ctc_projection, validation_examples, tokenizer, criterion
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "ctc_head.pt"
    torch.save(model.ctc_projection.state_dict(), checkpoint_path)

    predictions_path = args.output_dir / "predictions.csv"
    with predictions_path.open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=("split", "dataset_index", "reference", "prediction"),
        )
        writer.writeheader()
        for split, results in (
            ("train", final_training["predictions"]),
            ("validation", final_validation["predictions"]),
        ):
            for result in results:
                writer.writerow({"split": split, **result})

    summary = {
        "experiment": args.experiment_id,
        "seed": args.seed,
        "device": str(device),
        "loaded_encoder_tensors": matched,
        "steps": args.steps,
        "learning_rate": args.learning_rate,
        "train_indices": train_indices,
        "validation_indices": validation_indices,
        "initial": {
            "train": {key: value for key, value in initial_train.items()
                      if key != "predictions"},
            "validation": {key: value for key, value in initial_validation.items()
                           if key != "predictions"},
        },
        "final": {
            "train": {key: value for key, value in final_training.items()
                      if key != "predictions"},
            "validation": {key: value for key, value in final_validation.items()
                           if key != "predictions"},
        },
        "checkpoint": str(checkpoint_path),
        "predictions": str(predictions_path),
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Saved CTC head: {checkpoint_path}")
    print(f"Saved metrics: {summary_path}")
    print(f"Saved decoded examples: {predictions_path}")


if __name__ == "__main__":
    main()
