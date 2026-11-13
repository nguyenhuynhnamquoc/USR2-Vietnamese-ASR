import argparse
import csv
import json
import sys
from pathlib import Path

import sentencepiece as spm
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from data.vivos_asr import VIVOSASRDataset
from models.vietnamese_ctc import load_pretrained_audio_encoder
from train_asr import decode_greedy, edit_distance, make_model
from utils.ctc_decode import beam_search


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare CTC decoding and alignment diagnostics without training."
    )
    parser.add_argument(
        "--exp010-summary",
        type=Path,
        default=PROJECT_ROOT / "output" / "exp010" / "summary.json",
    )
    parser.add_argument(
        "--exp011-summary",
        type=Path,
        default=PROJECT_ROOT / "output" / "exp011" / "summary.json",
    )
    parser.add_argument(
        "--exp010-checkpoint",
        type=Path,
        default=PROJECT_ROOT / "output" / "exp010" / "ctc_head.pt",
    )
    parser.add_argument(
        "--exp011-checkpoint",
        type=Path,
        default=PROJECT_ROOT / "output" / "exp011" / "partial_finetune.pt",
    )
    parser.add_argument("--beam-width", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "output" / "exp012",
    )
    return parser.parse_args()


def read_split(exp010_summary_path, exp011_summary_path):
    for path in (exp010_summary_path, exp011_summary_path):
        if not path.is_file():
            raise FileNotFoundError(f"Experiment summary not found: {path}")
    exp010 = json.loads(exp010_summary_path.read_text(encoding="utf-8"))
    exp011 = json.loads(exp011_summary_path.read_text(encoding="utf-8"))
    for key in ("train_indices", "validation_indices"):
        if exp010.get(key) != exp011.get(key):
            raise ValueError(
                f"EXP-010 and EXP-011 have different {key}; "
                "this comparison requires the same fixed split."
            )
    return exp010["validation_indices"]


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


def decode_beam(class_ids, tokenizer):
    token_ids = [
        class_id - 1
        for class_id in class_ids
        if class_id > 0
        and class_id - 1 not in (tokenizer.bos_id(), tokenizer.eos_id())
    ]
    return tokenizer.decode(token_ids).strip()


def validate_indices(indices, dataset_size):
    if not isinstance(indices, list) or not indices:
        raise ValueError("The EXP-010 validation index list is empty or invalid.")
    if any(type(index) is not int or not 0 <= index < dataset_size for index in indices):
        raise ValueError("The EXP-010 validation indices are outside the dataset.")
    if len(set(indices)) != len(indices):
        raise ValueError("The EXP-010 validation indices contain duplicates.")


def evaluate_checkpoint(
    model,
    checkpoint_path,
    checkpoint_kind,
    examples,
    tokenizer,
    device,
    beam_width,
    top_k,
):
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint_kind == "partial":
        model.encoder.encoders[-1].load_state_dict(state["encoder_last_block"])
        model.ctc_projection.load_state_dict(state["ctc_projection"])
    else:
        model.ctc_projection.load_state_dict(state)

    model.encoder.eval()
    model.ctc_projection.eval()
    metrics = {
        "greedy": {"word_errors": 0, "char_errors": 0, "exact_matches": 0, "empty": 0},
        "beam": {"word_errors": 0, "char_errors": 0, "exact_matches": 0, "empty": 0},
    }
    reference_words = 0
    reference_chars = 0
    total_loss = 0.0
    blank_argmax = 0
    blank_probability = 0.0
    total_frames = 0
    infeasible_alignments = 0
    output_rows = []
    criterion = torch.nn.CTCLoss(blank=0, reduction="none", zero_infinity=False)

    with torch.inference_mode():
        for example in examples:
            sample = example["sample"]
            audio = sample["audio"].unsqueeze(0).to(device)
            input_mask_frames = audio.shape[1] // 640
            if input_mask_frames <= 0:
                raise ValueError(
                    f"Audio at validation index {example['index']} is too short."
                )
            mask = torch.ones(
                (1, 1, input_mask_frames), dtype=torch.bool, device=device
            )
            features = model.encoder(xs_a=audio.unsqueeze(-1), masks=mask)
            logits = model.ctc_projection(features)[0, : features.shape[1]].float()
            input_length = features.shape[1]
            targets = sample["tokens"].to(device)
            target_ids = targets.tolist()
            repeated_adjacent_tokens = sum(
                previous == current
                for previous, current in zip(target_ids, target_ids[1:])
            )
            minimum_frames = len(target_ids) + repeated_adjacent_tokens
            if input_length < minimum_frames:
                infeasible_alignments += 1
            else:
                loss = criterion(
                    logits.log_softmax(dim=-1).unsqueeze(1),
                    targets,
                    torch.tensor([input_length], dtype=torch.long),
                    torch.tensor([targets.numel()], dtype=torch.long),
                )[0]
                if not torch.isfinite(loss):
                    raise RuntimeError(
                        f"Non-finite CTC loss at validation index {example['index']}."
                    )
                total_loss += float(loss.item()) / targets.numel()

            probabilities = logits.softmax(dim=-1)
            blank_argmax += int((logits.argmax(dim=-1) == 0).sum().item())
            blank_probability += float(probabilities[:, 0].sum().item())
            total_frames += input_length
            greedy = decode_greedy(
                logits.unsqueeze(0),
                torch.tensor([input_length], dtype=torch.long),
                tokenizer,
            )[0]
            beam_classes = beam_search(
                logits, beam_width=beam_width, top_k=top_k
            )
            beam = decode_beam(beam_classes, tokenizer)
            reference = sample["text"]
            normalized_reference = normalize_text(reference)
            reference_word_list = normalized_reference.split()
            reference_words += len(reference_word_list)
            reference_chars += len(normalized_reference)

            for name, prediction in (("greedy", greedy), ("beam", beam)):
                normalized_prediction = normalize_text(prediction)
                word_list = normalized_prediction.split()
                metrics[name]["word_errors"] += edit_distance(
                    word_list, reference_word_list
                )
                metrics[name]["char_errors"] += character_edit_distance(
                    normalized_prediction, normalized_reference
                )
                metrics[name]["exact_matches"] += (
                    normalized_prediction == normalized_reference
                )
                metrics[name]["empty"] += not normalized_prediction

            output_rows.append(
                {
                    "dataset_index": example["index"],
                    "reference": reference,
                    "greedy_prediction": greedy,
                    "beam_prediction": beam,
                    "input_frames": input_length,
                    "target_tokens": len(target_ids),
                    "minimum_ctc_frames": minimum_frames,
                }
            )

    if reference_words == 0 or reference_chars == 0 or total_frames == 0:
        raise ValueError("Cannot calculate diagnostics for an empty validation set.")
    count = len(examples)
    metric_summary = {
        name: {
            "wer": values["word_errors"] / reference_words,
            "cer": values["char_errors"] / reference_chars,
            "exact_match": values["exact_matches"] / count,
            "empty_predictions": values["empty"],
        }
        for name, values in metrics.items()
    }
    return {
        "checkpoint": str(checkpoint_path),
        "mean_ctc_loss": total_loss / max(1, count - infeasible_alignments),
        "blank_argmax_ratio": blank_argmax / total_frames,
        "mean_blank_probability": blank_probability / total_frames,
        "ctc_minimum_frame_violations": infeasible_alignments,
        "decoding": metric_summary,
        "predictions": output_rows,
    }


def main():
    args = parse_args()
    if args.beam_width <= 0 or args.top_k <= 0:
        raise ValueError("--beam-width and --top-k must be positive.")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    device = torch.device(
        "cuda"
        if args.device == "cuda"
        or (args.device == "auto" and torch.cuda.is_available())
        else "cpu"
    )
    torch.set_num_threads(4)

    tokenizer_path = PROJECT_ROOT / "dataset" / "tokenizer" / "vivos_vi.model"
    tokenizer = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
    validation_dataset = VIVOSASRDataset(
        PROJECT_ROOT / "dataset" / "manifests" / "validation.tsv",
        PROJECT_ROOT / "dataset" / "vivos",
        tokenizer_path,
    )
    validation_indices = read_split(
        args.exp010_summary, args.exp011_summary
    )
    validate_indices(validation_indices, len(validation_dataset))
    examples = [
        {"index": index, "sample": validation_dataset[index]}
        for index in validation_indices
    ]

    model = make_model(
        tokenizer,
        PROJECT_ROOT
        / "conf"
        / "model"
        / "backbone"
        / "resnet_transformer_baseplus.yaml",
    )
    matched, _ = load_pretrained_audio_encoder(
        model, PROJECT_ROOT / "path" / "baseplus_high_resource_lrs3vox2.pth"
    )
    model.to(device)

    print(
        f"Device: {device}; loaded encoder tensors: {matched}; "
        f"validation utterances: {len(examples)}"
    )
    results = {}
    rows = []
    for label, checkpoint_path, checkpoint_kind in (
        ("EXP-010 frozen encoder", args.exp010_checkpoint, "frozen"),
        ("EXP-011 partial fine-tune", args.exp011_checkpoint, "partial"),
    ):
        result = evaluate_checkpoint(
            model,
            checkpoint_path,
            checkpoint_kind,
            examples,
            tokenizer,
            device,
            args.beam_width,
            args.top_k,
        )
        results[label] = {
            key: value for key, value in result.items() if key != "predictions"
        }
        rows.extend(
            {"checkpoint": label, **prediction}
            for prediction in result["predictions"]
        )
        for decoder_name, decoder_metrics in result["decoding"].items():
            print(
                f"{label} | {decoder_name} | "
                f"WER {decoder_metrics['wer']:.2%}, "
                f"CER {decoder_metrics['cer']:.2%}, "
                f"exact {decoder_metrics['exact_match']:.0%}"
            )
        print(
            f"{label} | blank argmax {result['blank_argmax_ratio']:.2%}, "
            f"mean blank probability {result['mean_blank_probability']:.2%}, "
            f"CTC minimum-frame violations "
            f"{result['ctc_minimum_frame_violations']}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = args.output_dir / "predictions.csv"
    with predictions_path.open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=(
                "checkpoint",
                "dataset_index",
                "reference",
                "greedy_prediction",
                "beam_prediction",
                "input_frames",
                "target_tokens",
                "minimum_ctc_frames",
            ),
        )
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "experiment": "EXP-012",
        "training_performed": False,
        "device": str(device),
        "validation_indices": validation_indices,
        "beam_width": args.beam_width,
        "top_k": args.top_k,
        "results": results,
        "predictions": str(predictions_path),
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Saved diagnostics: {summary_path}")
    print(f"Saved decoded examples: {predictions_path}")


if __name__ == "__main__":
    main()
