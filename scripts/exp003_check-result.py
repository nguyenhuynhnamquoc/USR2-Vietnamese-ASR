import argparse
from pathlib import Path
import sys

import sentencepiece as spm
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from data.vivos_asr import VIVOSASRDataset, collate_vivos_asr
from train_asr import decode_greedy, make_model


DEFAULT_CHECKPOINT = PROJECT_ROOT / "output" / "pilot_3epochs" / "last.pt"
VALIDATION_MANIFEST = PROJECT_ROOT / "dataset" / "manifests" / "validation.tsv"
DATA_ROOT = PROJECT_ROOT / "dataset" / "vivos"
TOKENIZER_PATH = PROJECT_ROOT / "dataset" / "tokenizer" / "vivos_vi.model"
BACKBONE_CONFIG = (
    PROJECT_ROOT
    / "conf"
    / "model"
    / "backbone"
    / "resnet_transformer_baseplus.yaml"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inspect CTC blank predictions on five VIVOS validation samples."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="Checkpoint to inspect (defaults to EXP-002 last.pt).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {args.checkpoint}. "
            "Run scripts\\exp002_3-epoch-test.py first."
        )

    tokenizer = spm.SentencePieceProcessor(model_file=str(TOKENIZER_PATH))
    dataset = VIVOSASRDataset(
        VALIDATION_MANIFEST,
        DATA_ROOT,
        TOKENIZER_PATH,
    )
    model = make_model(tokenizer, BACKBONE_CONFIG)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {device}; validation samples: {min(5, len(dataset))}\n")

    with torch.inference_mode():
        for index in range(min(5, len(dataset))):
            batch = collate_vivos_asr([dataset[index]])
            logits, lengths = model(
                batch["audio"].to(device),
                batch["audio_lengths"],
            )
            frame_count = int(lengths[0])
            frame_predictions = logits[0, :frame_count].argmax(dim=-1)
            blank_ratio = (frame_predictions == 0).float().mean().item()
            prediction = decode_greedy(logits, lengths, tokenizer)[0]
            reference = batch["texts"][0]
            print(
                f"Sample {index + 1} | blank ratio: {blank_ratio:.2%}\n"
                f"  Reference : {reference}\n"
                f"  Prediction: {prediction or '<empty>'}"
            )


if __name__ == "__main__":
    main()
