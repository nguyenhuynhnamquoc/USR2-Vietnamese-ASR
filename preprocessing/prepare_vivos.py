import argparse
import csv
import random
import unicodedata
import wave
from pathlib import Path


def read_split(vivos_root, split):
    prompts_path = vivos_root / split / "prompts.txt"
    if not prompts_path.is_file():
        raise FileNotFoundError(f"Transcript file not found: {prompts_path}")

    records = []
    seen_ids = set()
    with prompts_path.open("r", encoding="utf-8") as prompts:
        for line_number, line in enumerate(prompts, start=1):
            line = line.strip()
            if not line:
                continue

            parts = line.split(maxsplit=1)
            if len(parts) != 2:
                raise ValueError(f"Invalid transcript at {prompts_path}:{line_number}")

            utterance_id, transcript = parts
            if utterance_id in seen_ids:
                raise ValueError(f"Duplicate utterance ID {utterance_id} in {prompts_path}")
            seen_ids.add(utterance_id)

            speaker_id = utterance_id.rsplit("_", maxsplit=1)[0]
            audio_path = Path(split) / "waves" / speaker_id / f"{utterance_id}.wav"
            full_audio_path = vivos_root / audio_path
            if not full_audio_path.is_file():
                raise FileNotFoundError(f"Audio file not found: {full_audio_path}")

            with wave.open(str(full_audio_path), "rb") as audio:
                if audio.getframerate() != 16000 or audio.getnchannels() != 1:
                    raise ValueError(
                        f"Expected mono 16 kHz audio, got {audio.getnchannels()} channel(s) "
                        f"at {audio.getframerate()} Hz: {full_audio_path}"
                    )

            transcript = unicodedata.normalize("NFC", " ".join(transcript.split()))
            records.append(
                {
                    "audio_path": audio_path.as_posix(),
                    "text": transcript,
                    "speaker_id": speaker_id,
                }
            )

    if not records:
        raise ValueError(f"No utterances found in {prompts_path}")
    return records


def write_manifest(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as manifest:
        writer = csv.DictWriter(
            manifest,
            fieldnames=("audio_path", "text", "speaker_id"),
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(records)


def main():
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Validate VIVOS and create audio/transcript TSV manifests."
    )
    parser.add_argument(
        "--vivos-root",
        type=Path,
        default=project_root / "dataset" / "vivos",
        help="Path to the extracted VIVOS directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "dataset" / "manifests",
        help="Directory where train.tsv, validation.tsv, and test.tsv are written.",
    )
    parser.add_argument(
        "--validation-speakers",
        type=int,
        default=5,
        help="Number of training speakers held out for validation.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    vivos_root = args.vivos_root.resolve()
    train_records = read_split(vivos_root, "train")
    test_records = read_split(vivos_root, "test")
    speakers = sorted({record["speaker_id"] for record in train_records})
    if not 1 <= args.validation_speakers < len(speakers):
        parser.error(
            f"--validation-speakers must be between 1 and {len(speakers) - 1}"
        )

    validation_speakers = set(
        random.Random(args.seed).sample(speakers, args.validation_speakers)
    )
    training = [
        record for record in train_records
        if record["speaker_id"] not in validation_speakers
    ]
    validation = [
        record for record in train_records
        if record["speaker_id"] in validation_speakers
    ]

    output_dir = args.output_dir.resolve()
    write_manifest(output_dir / "train.tsv", training)
    write_manifest(output_dir / "validation.tsv", validation)
    write_manifest(output_dir / "test.tsv", test_records)

    print(f"Train:      {len(training)} utterances")
    print(
        f"Validation: {len(validation)} utterances "
        f"({len(validation_speakers)} held-out speakers)"
    )
    print(f"Test:       {len(test_records)} utterances (official VIVOS test split)")
    print(f"Manifests:  {output_dir}")


if __name__ == "__main__":
    main()
