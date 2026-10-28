import argparse
import csv
import tempfile
from pathlib import Path

import sentencepiece as spm


def main():
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Train a Vietnamese SentencePiece tokenizer from the VIVOS train split."
    )
    parser.add_argument(
        "--train-manifest",
        type=Path,
        default=project_root / "dataset" / "manifests" / "train.tsv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "dataset" / "tokenizer",
    )
    parser.add_argument("--vocab-size", type=int, default=1000)
    args = parser.parse_args()

    manifest_path = args.train_manifest.resolve()
    output_dir = args.output_dir.resolve()
    if not manifest_path.is_file():
        parser.error(f"Training manifest not found: {manifest_path}")
    if args.vocab_size < 16:
        parser.error("--vocab-size must be at least 16")

    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as temp_dir:
        text_path = Path(temp_dir) / "train.txt"
        with manifest_path.open("r", encoding="utf-8", newline="") as manifest:
            reader = csv.DictReader(manifest, delimiter="\t")
            if not reader.fieldnames or "text" not in reader.fieldnames:
                parser.error(f"Manifest must contain a 'text' column: {manifest_path}")

            sentence_count = 0
            with text_path.open("w", encoding="utf-8", newline="\n") as text_file:
                for row in reader:
                    transcript = row["text"].strip()
                    if transcript:
                        text_file.write(transcript + "\n")
                        sentence_count += 1

        if sentence_count == 0:
            parser.error(f"No transcripts found in {manifest_path}")

        model_prefix = output_dir / "vivos_vi"
        spm.SentencePieceTrainer.train(
            input=str(text_path),
            model_prefix=str(model_prefix),
            model_type="unigram",
            vocab_size=args.vocab_size,
            character_coverage=1.0,
            normalization_rule_name="identity",
            unk_id=0,
            bos_id=1,
            eos_id=2,
            pad_id=-1,
            hard_vocab_limit=False,
        )

    print(f"Trained on {sentence_count} training transcripts.")
    print(f"Tokenizer model: {model_prefix}.model")
    print(f"Tokenizer vocabulary: {model_prefix}.vocab")
    print("Special token IDs: <unk>=0, <s>=1, </s>=2")


if __name__ == "__main__":
    main()
