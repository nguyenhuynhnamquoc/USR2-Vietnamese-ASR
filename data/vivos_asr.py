import csv
from pathlib import Path
import wave

import numpy as np
import sentencepiece as spm
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


class VIVOSASRDataset(Dataset):
    def __init__(self, manifest_path, data_root, tokenizer_path):
        self.manifest_path = Path(manifest_path)
        self.data_root = Path(data_root)
        if not self.manifest_path.is_file():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")
        if not self.data_root.is_dir():
            raise NotADirectoryError(f"Audio data directory not found: {self.data_root}")

        self.tokenizer = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
        with self.manifest_path.open("r", encoding="utf-8", newline="") as manifest:
            reader = csv.DictReader(manifest, delimiter="\t")
            required_columns = {"audio_path", "text"}
            if not reader.fieldnames or not required_columns.issubset(reader.fieldnames):
                raise ValueError(
                    f"{self.manifest_path} must contain columns {sorted(required_columns)}"
                )
            self.samples = list(reader)
        if not self.samples:
            raise ValueError(f"No utterances found in {self.manifest_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        audio_path = self.data_root / Path(sample["audio_path"])
        if not audio_path.is_file():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        with wave.open(str(audio_path), "rb") as wav:
            if wav.getframerate() != 16000:
                raise ValueError(
                    f"Expected 16 kHz audio, got {wav.getframerate()} Hz: {audio_path}"
                )
            if wav.getnchannels() != 1:
                raise ValueError(
                    f"Expected mono audio, got {wav.getnchannels()} channels: {audio_path}"
                )
            if wav.getsampwidth() != 2 or wav.getcomptype() != "NONE":
                raise ValueError(f"Expected uncompressed 16-bit PCM audio: {audio_path}")
            audio_data = wav.readframes(wav.getnframes())
        audio = torch.from_numpy(
            np.frombuffer(audio_data, dtype=np.int16).copy()
        ).to(torch.float32) / 32768.0

        tokenizer_ids = self.tokenizer.encode(sample["text"], out_type=int)
        if not tokenizer_ids:
            raise ValueError(f"Transcript produced no tokenizer tokens: {audio_path}")
        # Reserve CTC class 0 for blank; SentencePiece IDs start at class 1.
        token_ids = [token_id + 1 for token_id in tokenizer_ids]

        return {
            "audio": audio,
            "tokens": torch.tensor(token_ids, dtype=torch.long),
            "text": sample["text"],
        }


def collate_vivos_asr(batch):
    audio = [sample["audio"] for sample in batch]
    token_sequences = [sample["tokens"] for sample in batch]
    audio_lengths = torch.tensor([item.numel() for item in audio], dtype=torch.long)
    token_lengths = torch.tensor(
        [item.numel() for item in token_sequences], dtype=torch.long
    )

    return {
        "audio": pad_sequence(audio, batch_first=True),
        "audio_lengths": audio_lengths,
        "tokens": torch.cat(token_sequences),
        "token_lengths": token_lengths,
        "texts": [sample["text"] for sample in batch],
    }
