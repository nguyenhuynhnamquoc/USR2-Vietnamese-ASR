import argparse
import json
import random
import sys
from pathlib import Path

import sentencepiece as spm
import torch
from torch.utils.data import Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from data.vivos_asr import VIVOSASRDataset
from models.vietnamese_ctc import load_pretrained_audio_encoder
from train_asr import make_loader, make_model, run_epoch


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune a pretrained Vietnamese CTC ASR model on a fixed "
            "VIVOS subset and evaluate on the full validation split."
        )
    )
    parser.add_argument("--train-samples", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--encoder-learning-rate", type=float, default=1e-5)
    parser.add_argument("--head-learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--limit-train-batches", type=int, default=0)
    parser.add_argument("--limit-validation-batches", type=int, default=0)
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "output" / "exp013",
    )
    return parser.parse_args()


def choose_indices(dataset_size, sample_count, seed):
    if not 1 <= sample_count <= dataset_size:
        raise ValueError(
            f"Requested {sample_count} train samples, but the split contains "
            f"{dataset_size} utterances."
        )
    return sorted(random.Random(seed).sample(range(dataset_size), sample_count))


def save_checkpoint(
    path,
    model,
    optimizer,
    scaler,
    epoch,
    best_validation_wer,
    train_indices,
    history,
    args,
    include_optimizer=False,
):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        "best_validation_wer": best_validation_wer,
        "train_indices": train_indices,
        "history": history,
        "encoder_learning_rate": args.encoder_learning_rate,
        "head_learning_rate": args.head_learning_rate,
        "seed": args.seed,
    }
    if include_optimizer:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
        checkpoint["scaler_state_dict"] = scaler.state_dict()
    torch.save(checkpoint, path)


def write_history(output_dir, history, train_indices, args, device, best_epoch, best_wer):
    summary = {
        "experiment": "EXP-013",
        "device": str(device),
        "seed": args.seed,
        "train_samples": args.train_samples,
        "train_indices": train_indices,
        "validation_samples": "full",
        "epochs_requested": args.epochs,
        "batch_size": args.batch_size,
        "encoder_learning_rate": args.encoder_learning_rate,
        "head_learning_rate": args.head_learning_rate,
        "weight_decay": args.weight_decay,
        "best_epoch": best_epoch,
        "best_validation_wer": best_wer,
        "history": history,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main():
    args = parse_args()
    if args.train_samples <= 0 or args.epochs <= 0 or args.batch_size <= 0:
        raise ValueError("--train-samples, --epochs, and --batch-size must be positive.")
    if args.num_workers < 0:
        raise ValueError("--num-workers cannot be negative.")
    if args.encoder_learning_rate <= 0 or args.head_learning_rate <= 0:
        raise ValueError("Learning rates must be positive.")
    if args.weight_decay < 0:
        raise ValueError("--weight-decay cannot be negative.")
    if args.limit_train_batches < 0 or args.limit_validation_batches < 0:
        raise ValueError("Batch limits cannot be negative.")
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
    train_indices = choose_indices(
        len(train_dataset), args.train_samples, args.seed
    )
    train_subset = Subset(train_dataset, train_indices)
    train_loader = make_loader(
        train_subset,
        args.batch_size,
        True,
        args.num_workers,
        device,
    )
    validation_loader = make_loader(
        validation_dataset,
        args.batch_size,
        False,
        args.num_workers,
        device,
    )

    model = make_model(tokenizer, backbone_config)
    matched, _ = load_pretrained_audio_encoder(model, pretrained_checkpoint)
    model.to(device)
    encoder_parameters = list(model.encoder.parameters())
    head_parameters = list(model.ctc_projection.parameters())
    optimizer = torch.optim.AdamW(
        [
            {
                "params": encoder_parameters,
                "lr": args.encoder_learning_rate,
                "weight_decay": args.weight_decay,
            },
            {
                "params": head_parameters,
                "lr": args.head_learning_rate,
                "weight_decay": 0.0,
            },
        ]
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    criterion = torch.nn.CTCLoss(blank=0, zero_infinity=True)
    start_epoch = 0
    best_validation_wer = float("inf")
    best_epoch = None
    history = []

    if args.resume:
        if not args.resume.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {args.resume}")
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True)
        if "optimizer_state_dict" not in checkpoint:
            raise ValueError(
                f"Checkpoint does not contain optimizer state: {args.resume}"
            )
        if checkpoint.get("train_indices") != train_indices:
            raise ValueError(
                "Resume checkpoint uses a different training subset. "
                "Use the same --seed and --train-samples."
            )
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if checkpoint.get("scaler_state_dict"):
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_validation_wer = checkpoint["best_validation_wer"]
        history = checkpoint.get("history", [])
        best_epoch = min(
            history,
            key=lambda row: row["validation_wer"],
        )["epoch"] if history else None

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"Device: {device}; loaded encoder tensors: {matched}; "
        f"train: {len(train_subset)}; validation: {len(validation_dataset)}"
    )
    if device.type == "cpu":
        print("CUDA is unavailable; full encoder fine-tuning on CPU may be slow.")

    for epoch in range(start_epoch, args.epochs):
        train_loss, train_wer = run_epoch(
            model,
            train_loader,
            criterion,
            tokenizer,
            device,
            optimizer=optimizer,
            scaler=scaler,
            max_batches=args.limit_train_batches,
        )
        validation_loss, validation_wer = run_epoch(
            model,
            validation_loader,
            criterion,
            tokenizer,
            device,
            max_batches=args.limit_validation_batches,
        )
        epoch_metrics = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_wer": train_wer,
            "validation_loss": validation_loss,
            "validation_wer": validation_wer,
            "train_batches": min(
                len(train_loader),
                args.limit_train_batches or len(train_loader),
            ),
            "validation_batches": min(
                len(validation_loader),
                args.limit_validation_batches or len(validation_loader),
            ),
        }
        history.append(epoch_metrics)
        print(
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"train loss {train_loss:.4f}, WER {train_wer:.2%} | "
            f"validation loss {validation_loss:.4f}, "
            f"WER {validation_wer:.2%}",
            flush=True,
        )

        if validation_wer < best_validation_wer:
            best_validation_wer = validation_wer
            best_epoch = epoch + 1
            save_checkpoint(
                args.output_dir / "best.pt",
                model,
                optimizer,
                scaler,
                epoch,
                best_validation_wer,
                train_indices,
                history,
                args,
            )
        save_checkpoint(
            args.output_dir / "last.pt",
            model,
            optimizer,
            scaler,
            epoch,
            best_validation_wer,
            train_indices,
            history,
            args,
            include_optimizer=True,
        )
        write_history(
            args.output_dir,
            history,
            train_indices,
            args,
            device,
            best_epoch,
            best_validation_wer,
        )

    print(
        f"Best validation WER: {best_validation_wer:.2%} "
        f"(epoch {best_epoch})"
    )
    print(f"Best checkpoint: {args.output_dir / 'best.pt'}")
    print(f"Latest checkpoint: {args.output_dir / 'last.pt'}")
    print(f"Metrics: {args.output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
