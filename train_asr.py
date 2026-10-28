import argparse
import math
import random
from pathlib import Path

import sentencepiece as spm
import torch
from omegaconf import OmegaConf
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from data.vivos_asr import VIVOSASRDataset, collate_vivos_asr
from models.vietnamese_ctc import (
    VietnameseCTCASR,
    load_pretrained_audio_encoder,
)


def parse_args():
    project_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Train or evaluate Vietnamese VIVOS ASR.")
    parser.add_argument(
        "--train-manifest",
        type=Path,
        default=project_root / "dataset" / "manifests" / "train.tsv",
    )
    parser.add_argument(
        "--validation-manifest",
        type=Path,
        default=project_root / "dataset" / "manifests" / "validation.tsv",
    )
    parser.add_argument(
        "--test-manifest",
        type=Path,
        default=project_root / "dataset" / "manifests" / "test.tsv",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=project_root / "dataset" / "vivos",
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=project_root / "dataset" / "tokenizer" / "vivos_vi.model",
    )
    parser.add_argument(
        "--backbone-config",
        type=Path,
        default=project_root / "conf" / "model" / "backbone" / "resnet_transformer_baseplus.yaml",
    )
    parser.add_argument(
        "--pretrained-checkpoint",
        default=str(project_root / "path" / "baseplus_high_resource_lrs3vox2.pth"),
        help="Set to an empty string to train the encoder from scratch.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "output" / "vivos_asr",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--evaluate-only",
        action="store_true",
        help="Evaluate a saved checkpoint on --eval-manifest without training.",
    )
    parser.add_argument(
        "--eval-manifest",
        type=Path,
        help="Manifest to evaluate; defaults to the official test manifest.",
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--limit-train-batches",
        type=int,
        default=0,
        help="Optional smoke-test limit; 0 means no limit.",
    )
    parser.add_argument(
        "--limit-validation-batches",
        type=int,
        default=0,
        help="Optional smoke-test limit; 0 means no limit.",
    )
    return parser.parse_args()


def edit_distance(prediction, reference):
    previous = list(range(len(reference) + 1))
    for row, predicted_word in enumerate(prediction, start=1):
        current = [row]
        for column, reference_word in enumerate(reference, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (predicted_word != reference_word),
                )
            )
        previous = current
    return previous[-1]


def decode_greedy(logits, lengths, tokenizer):
    predictions = logits.argmax(dim=-1)
    hypotheses = []
    for prediction, length in zip(predictions, lengths):
        token_ids = []
        previous = None
        for raw_id in prediction[: int(length)].tolist():
            if raw_id != 0 and raw_id != previous:
                token_id = raw_id - 1
                if token_id not in (tokenizer.bos_id(), tokenizer.eos_id()):
                    token_ids.append(token_id)
            previous = raw_id
        hypotheses.append(tokenizer.decode(token_ids).strip())
    return hypotheses


def make_model(tokenizer, backbone_config):
    config = OmegaConf.load(backbone_config)
    model = VietnameseCTCASR(
        vocabulary_size=tokenizer.get_piece_size(),
        idim=config.idim,
        attention_dim=config.adim,
        attention_heads=config.aheads,
        linear_units=config.eunits,
        num_blocks=config.elayers,
        gamma_init=config.gamma_init,
    )
    return model


def make_loader(dataset, batch_size, shuffle, num_workers, device):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
        collate_fn=collate_vivos_asr,
    )


def move_batch(batch, device):
    return {
        "audio": batch["audio"].to(device, non_blocking=True),
        "audio_lengths": batch["audio_lengths"],
        "tokens": batch["tokens"].to(device, non_blocking=True),
        "token_lengths": batch["token_lengths"],
        "texts": batch["texts"],
    }


def calculate_metrics(hypotheses, references):
    errors = 0
    words = 0
    for hypothesis, reference in zip(hypotheses, references):
        reference_words = reference.split()
        errors += edit_distance(hypothesis.split(), reference_words)
        words += len(reference_words)
    if words == 0:
        raise ValueError("Cannot calculate WER: validation references contain no words.")
    return errors / words


def run_epoch(model, loader, criterion, tokenizer, device, optimizer=None, scaler=None,
              max_batches=0):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    utterance_count = 0
    errors = 0
    reference_words = 0
    use_amp = device.type == "cuda"

    for batch_index, batch in enumerate(loader):
        if max_batches and batch_index >= max_batches:
            break
        batch = move_batch(batch, device)
        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp,
            ):
                logits, input_lengths = model(batch["audio"], batch["audio_lengths"])
                token_lengths = batch["token_lengths"]
                if torch.any(token_lengths > input_lengths.cpu()):
                    raise ValueError(
                        "A transcript has more tokens than the audio encoder frames."
                    )
                log_probs = logits.float().log_softmax(dim=-1).transpose(0, 1)
                loss = criterion(
                    log_probs,
                    batch["tokens"],
                    input_lengths.cpu(),
                    token_lengths.cpu(),
                )

            if training:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()

        hypotheses = decode_greedy(logits.detach(), input_lengths, tokenizer)
        for hypothesis, reference in zip(hypotheses, batch["texts"]):
            reference_split = reference.split()
            errors += edit_distance(hypothesis.split(), reference_split)
            reference_words += len(reference_split)
        count = len(batch["texts"])
        total_loss += float(loss.detach()) * count
        utterance_count += count

    if utterance_count == 0:
        raise ValueError("No batches were processed.")
    wer = errors / reference_words if reference_words else math.inf
    return total_loss / utterance_count, wer


def save_checkpoint(path, model, optimizer, epoch, best_wer, arguments,
                    include_optimizer=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        "best_validation_wer": best_wer,
        "tokenizer_path": str(arguments.tokenizer.resolve()),
        "backbone_config": str(arguments.backbone_config.resolve()),
    }
    if include_optimizer:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    torch.save(checkpoint, path)


def main():
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.num_workers < 0:
        raise ValueError("Epochs and batch size must be positive; workers cannot be negative.")
    if args.learning_rate <= 0:
        raise ValueError("Learning rate must be positive.")
    if args.limit_train_batches < 0 or args.limit_validation_batches < 0:
        raise ValueError("Batch limits cannot be negative.")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if not args.tokenizer.is_file():
        raise FileNotFoundError(f"Tokenizer model not found: {args.tokenizer}")
    if not args.backbone_config.is_file():
        raise FileNotFoundError(f"Backbone config not found: {args.backbone_config}")
    tokenizer = spm.SentencePieceProcessor(model_file=str(args.tokenizer))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("CUDA is unavailable; training on CPU will be very slow.")
    model = make_model(tokenizer, args.backbone_config)
    criterion = torch.nn.CTCLoss(blank=0, zero_infinity=True)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    if args.evaluate_only:
        if args.checkpoint is None or not args.checkpoint.is_file():
            raise FileNotFoundError("--evaluate-only requires an existing --checkpoint.")
        manifest_path = args.eval_manifest or args.test_manifest
        dataset = VIVOSASRDataset(manifest_path, args.data_root, args.tokenizer)
        loader = make_loader(dataset, args.batch_size, False, args.num_workers, device)
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        loss, wer = run_epoch(model, loader, criterion, tokenizer, device)
        print(f"Evaluation manifest: {manifest_path}")
        print(f"Loss: {loss:.4f} | WER: {wer:.4f} ({wer * 100:.2f}%)")
        return

    train_dataset = VIVOSASRDataset(args.train_manifest, args.data_root, args.tokenizer)
    validation_dataset = VIVOSASRDataset(
        args.validation_manifest, args.data_root, args.tokenizer
    )
    train_loader = make_loader(
        train_dataset, args.batch_size, True, args.num_workers, device
    )
    validation_loader = make_loader(
        validation_dataset, args.batch_size, False, args.num_workers, device
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    start_epoch = 0
    best_wer = math.inf

    if args.resume:
        if not args.resume.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {args.resume}")
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True)
        if "optimizer_state_dict" not in checkpoint:
            raise ValueError(
                f"Checkpoint cannot resume optimizer state: {args.resume}. "
                "Use the last.pt checkpoint."
            )
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_wer = checkpoint["best_validation_wer"]
    elif args.pretrained_checkpoint.strip():
        pretrained_checkpoint = Path(args.pretrained_checkpoint)
        if not pretrained_checkpoint.is_file():
            raise FileNotFoundError(
                f"Pretrained checkpoint not found: {pretrained_checkpoint}"
            )
        matched, incompatible = load_pretrained_audio_encoder(
            model, pretrained_checkpoint
        )
        print(
            f"Loaded {matched} audio encoder tensors from "
            f"{pretrained_checkpoint}."
        )
        if incompatible.missing_keys:
            print(f"Encoder tensors not initialized from checkpoint: {len(incompatible.missing_keys)}")

    model.to(device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}; train: {len(train_dataset)}; validation: {len(validation_dataset)}")

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
        print(
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"train loss {train_loss:.4f}, WER {train_wer:.4f} | "
            f"validation loss {validation_loss:.4f}, WER {validation_wer:.4f}"
        )

        if validation_wer < best_wer:
            best_wer = validation_wer
            save_checkpoint(
                args.output_dir / "best.pt",
                model,
                optimizer,
                epoch,
                best_wer,
                args,
            )
        save_checkpoint(
            args.output_dir / "last.pt",
            model,
            optimizer,
            epoch,
            best_wer,
            args,
            include_optimizer=True,
        )

    if not args.evaluate_only:
        print(f"Best validation WER: {best_wer:.4f}")
        print(f"Best checkpoint: {args.output_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
