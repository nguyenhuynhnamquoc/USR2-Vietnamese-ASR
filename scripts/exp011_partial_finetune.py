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
from train_asr import decode_greedy, edit_distance, make_model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune the last encoder block and CTC head on the EXP-010 split."
    )
    parser.add_argument(
        "--reference-summary",
        type=Path,
        default=PROJECT_ROOT / "output" / "exp010" / "summary.json",
    )
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--log-every", type=int, default=250)
    parser.add_argument("--encoder-learning-rate", type=float, default=1e-5)
    parser.add_argument("--head-learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "output" / "exp011",
    )
    return parser.parse_args()


def load_indices(summary_path, train_size, validation_size):
    if not summary_path.is_file():
        raise FileNotFoundError(
            f"EXP-010 summary not found: {summary_path}. "
            "Run EXP-010 first or pass --reference-summary."
        )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    indices = {}
    for key, dataset_size in (
        ("train_indices", train_size),
        ("validation_indices", validation_size),
    ):
        values = summary.get(key)
        if not isinstance(values, list) or not values:
            raise ValueError(f"{summary_path} must contain a non-empty {key} list.")
        if any(type(index) is not int or not 0 <= index < dataset_size for index in values):
            raise ValueError(f"{key} contains indices outside its dataset.")
        if len(set(values)) != len(values):
            raise ValueError(f"{key} contains duplicate indices.")
        indices[key] = values
    return indices["train_indices"], indices["validation_indices"]


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


def prepare_examples(dataset, indices):
    return [
        {
            "index": index,
            "sample": dataset[index],
        }
        for index in indices
    ]


def set_mode(model, training):
    model.encoder.eval()
    model.ctc_projection.train(training)
    if training:
        model.encoder.encoders[-1].train()


def forward_sample(model, sample, device):
    audio = sample["audio"].unsqueeze(0).to(device)
    frame_count = audio.shape[1] // 640
    if frame_count <= 0:
        raise ValueError("Audio sample is too short to encode.")
    mask = torch.ones((1, 1, frame_count), dtype=torch.bool, device=device)
    features = model.encoder(xs_a=audio.unsqueeze(-1), masks=mask)
    return model.ctc_projection(features), features.shape[1]


def evaluate(model, examples, tokenizer, criterion, device):
    set_mode(model, training=False)
    total_loss = 0.0
    word_errors = 0
    reference_words = 0
    char_errors = 0
    reference_chars = 0
    exact_matches = 0
    blank_frames = 0
    total_frames = 0
    predictions = []

    with torch.inference_mode():
        for example in examples:
            sample = example["sample"]
            logits, frame_count = forward_sample(model, sample, device)
            targets = sample["tokens"].to(device)
            input_length = torch.tensor([frame_count], dtype=torch.long)
            target_length = torch.tensor([targets.numel()], dtype=torch.long)
            log_probs = logits.float().log_softmax(dim=-1).transpose(0, 1)
            loss = criterion(log_probs, targets, input_length, target_length)
            prediction = decode_greedy(logits, input_length, tokenizer)[0]
            reference = sample["text"]
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
            predictions.append(
                {
                    "dataset_index": example["index"],
                    "reference": reference,
                    "prediction": prediction,
                }
            )

    if not examples or reference_words == 0 or reference_chars == 0 or total_frames == 0:
        raise ValueError("Cannot calculate metrics for an empty evaluation set.")
    count = len(examples)
    return {
        "loss": total_loss / count,
        "wer": word_errors / reference_words,
        "cer": char_errors / reference_chars,
        "exact_match": exact_matches / count,
        "blank_ratio": blank_frames / total_frames,
        "predictions": predictions,
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
    if args.steps <= 0 or args.log_every <= 0:
        raise ValueError("--steps and --log-every must be positive.")
    if args.encoder_learning_rate <= 0 or args.head_learning_rate <= 0:
        raise ValueError("Learning rates must be positive.")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    device = torch.device(
        "cuda"
        if args.device == "cuda"
        or (args.device == "auto" and torch.cuda.is_available())
        else "cpu"
    )

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(4)

    tokenizer_path = PROJECT_ROOT / "dataset" / "tokenizer" / "vivos_vi.model"
    tokenizer = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
    train_dataset = VIVOSASRDataset(
        PROJECT_ROOT / "dataset" / "manifests" / "train.tsv",
        PROJECT_ROOT / "dataset" / "vivos",
        tokenizer_path,
    )
    validation_dataset = VIVOSASRDataset(
        PROJECT_ROOT / "dataset" / "manifests" / "validation.tsv",
        PROJECT_ROOT / "dataset" / "vivos",
        tokenizer_path,
    )
    train_indices, validation_indices = load_indices(
        args.reference_summary, len(train_dataset), len(validation_dataset)
    )
    training_examples = prepare_examples(train_dataset, train_indices)
    validation_examples = prepare_examples(validation_dataset, validation_indices)

    backbone_config = (
        PROJECT_ROOT
        / "conf"
        / "model"
        / "backbone"
        / "resnet_transformer_baseplus.yaml"
    )
    pretrained_checkpoint = (
        PROJECT_ROOT / "path" / "baseplus_high_resource_lrs3vox2.pth"
    )
    model = make_model(tokenizer, backbone_config)
    matched, _ = load_pretrained_audio_encoder(model, pretrained_checkpoint)
    model.to(device)

    for parameter in model.encoder.parameters():
        parameter.requires_grad_(False)
    last_block = model.encoder.encoders[-1]
    for parameter in last_block.parameters():
        parameter.requires_grad_(True)
    for parameter in model.ctc_projection.parameters():
        parameter.requires_grad_(True)

    optimizer = torch.optim.AdamW(
        [
            {
                "params": last_block.parameters(),
                "lr": args.encoder_learning_rate,
            },
            {
                "params": model.ctc_projection.parameters(),
                "lr": args.head_learning_rate,
            },
        ],
        weight_decay=0,
    )
    criterion = torch.nn.CTCLoss(blank=0, zero_infinity=True)
    print(
        f"Device: {device}; loaded encoder tensors: {matched}; "
        "trainable encoder blocks: 1/12"
    )
    print(
        f"Using EXP-010 split: {len(train_indices)} train and "
        f"{len(validation_indices)} validation utterances"
    )
    initial_train = evaluate(
        model, training_examples, tokenizer, criterion, device
    )
    initial_validation = evaluate(
        model, validation_examples, tokenizer, criterion, device
    )
    print_metrics(0, initial_train, initial_validation)

    training_order = list(range(len(training_examples)))
    for step in range(1, args.steps + 1):
        if (step - 1) % len(training_order) == 0:
            random.shuffle(training_order)
        example = training_examples[
            training_order[(step - 1) % len(training_order)]
        ]
        set_mode(model, training=True)
        optimizer.zero_grad(set_to_none=True)
        logits, frame_count = forward_sample(model, example["sample"], device)
        sample = example["sample"]
        targets = sample["tokens"].to(device)
        log_probs = logits.float().log_softmax(dim=-1).transpose(0, 1)
        loss = criterion(
            log_probs,
            targets,
            torch.tensor([frame_count], dtype=torch.long),
            torch.tensor([targets.numel()], dtype=torch.long),
        )
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite CTC loss at step {step}.")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [
                parameter
                for parameter in model.parameters()
                if parameter.requires_grad
            ],
            max_norm=5.0,
        )
        optimizer.step()

        if step % args.log_every == 0 or step == args.steps:
            train_metrics = evaluate(
                model, training_examples, tokenizer, criterion, device
            )
            validation_metrics = evaluate(
                model, validation_examples, tokenizer, criterion, device
            )
            print_metrics(step, train_metrics, validation_metrics)

    final_train = evaluate(model, training_examples, tokenizer, criterion, device)
    final_validation = evaluate(
        model, validation_examples, tokenizer, criterion, device
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "partial_finetune.pt"
    torch.save(
        {
            "encoder_last_block": last_block.state_dict(),
            "ctc_projection": model.ctc_projection.state_dict(),
        },
        checkpoint_path,
    )

    predictions_path = args.output_dir / "predictions.csv"
    with predictions_path.open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=("split", "dataset_index", "reference", "prediction"),
        )
        writer.writeheader()
        for split, results in (
            ("train", final_train["predictions"]),
            ("validation", final_validation["predictions"]),
        ):
            for result in results:
                writer.writerow({"split": split, **result})

    summary = {
        "experiment": "EXP-011",
        "reference_summary": str(args.reference_summary),
        "seed": args.seed,
        "device": str(device),
        "loaded_encoder_tensors": matched,
        "trainable_encoder_blocks": 1,
        "steps": args.steps,
        "encoder_learning_rate": args.encoder_learning_rate,
        "head_learning_rate": args.head_learning_rate,
        "train_indices": train_indices,
        "validation_indices": validation_indices,
        "initial": {
            "train": {
                key: value
                for key, value in initial_train.items()
                if key != "predictions"
            },
            "validation": {
                key: value
                for key, value in initial_validation.items()
                if key != "predictions"
            },
        },
        "final": {
            "train": {
                key: value
                for key, value in final_train.items()
                if key != "predictions"
            },
            "validation": {
                key: value
                for key, value in final_validation.items()
                if key != "predictions"
            },
        },
        "checkpoint": str(checkpoint_path),
        "predictions": str(predictions_path),
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Saved fine-tuned weights: {checkpoint_path}")
    print(f"Saved metrics: {summary_path}")
    print(f"Saved decoded examples: {predictions_path}")


if __name__ == "__main__":
    main()
