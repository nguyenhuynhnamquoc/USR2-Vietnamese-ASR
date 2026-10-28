import csv
import os
import wave
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
import sentencepiece as spm

from espnet.nets.pytorch_backend.nets_utils import make_non_pad_mask
from espnet.nets.pytorch_backend.transformer.mask import subsequent_mask


def load_wav_16k_mono(wav_path: str) -> torch.Tensor:
    """Load 16kHz mono WAV file reliably using Python's built-in wave module."""
    with wave.open(wav_path, "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        n_frames = wf.getnframes()
        raw_bytes = wf.readframes(n_frames)

    if sampwidth == 2:
        audio = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    elif sampwidth == 4:
        audio = np.frombuffer(raw_bytes, dtype=np.int32).astype(np.float32) / 2147483648.0
    elif sampwidth == 1:
        audio = (np.frombuffer(raw_bytes, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    else:
        raise ValueError(f"Unsupported sample width {sampwidth} in {wav_path}")

    if n_channels > 1:
        audio = audio.reshape(-1, n_channels).mean(axis=1)

    return torch.from_numpy(audio)


class VIVOSDataset(Dataset):
    def __init__(
        self,
        manifest_path: str,
        vivos_root: str,
        tokenizer_path: str,
        max_duration_sec: float = 20.0,
        min_duration_sec: float = 0.5,
    ):
        super().__init__()
        self.manifest_path = manifest_path
        self.vivos_root = vivos_root
        self.sp = spm.SentencePieceProcessor(model_file=tokenizer_path)
        self.vocab_size = self.sp.get_piece_size()
        self.samples = []

        with open(manifest_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                audio_rel = row["audio_path"]
                text = row["text"].strip()
                if not text:
                    continue
                full_path = os.path.join(vivos_root, audio_rel)
                self.samples.append({
                    "audio_path": full_path,
                    "text": text,
                    "speaker_id": row.get("speaker_id", ""),
                })

        self.max_samples = int(max_duration_sec * 16000)
        self.min_samples = int(min_duration_sec * 16000)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[index]
        audio = load_wav_16k_mono(sample["audio_path"])

        # Truncate if audio is too long
        if audio.shape[0] > self.max_samples:
            audio = audio[:self.max_samples]

        # Shift token IDs by +1 so 0 is reserved for CTC <blank>
        sp_ids = self.sp.encode(sample["text"])
        token_ids = [idx + 1 for idx in sp_ids]

        return {
            "audio": audio,
            "target": torch.tensor(token_ids, dtype=torch.long),
            "text": sample["text"],
            "path": sample["audio_path"],
        }


def vivos_collate_fn(batch: List[Dict], eos_id: int = 1001, sos_id: int = 1001, ignore_id: int = -1):
    # Filter out empty samples
    batch = [s for s in batch if s["audio"].shape[0] >= 640 and len(s["target"]) > 0]
    if not batch:
        return {}

    audio_lens = [s["audio"].shape[0] for s in batch]
    target_lens = [s["target"].shape[0] for s in batch]

    # Align max audio length to multiple of 640 for Conv1dResNet
    max_audio_len = int(np.ceil(max(audio_lens) / 640.0) * 640)
    max_target_len = max(target_lens)

    bsz = len(batch)
    padded_audio = torch.zeros(bsz, max_audio_len, 1, dtype=torch.float32)
    padded_targets = torch.full((bsz, max_target_len), ignore_id, dtype=torch.long)
    flat_targets = []

    # ys_in (prepended with sos), ys_out (appended with eos) for autoregressive attention decoder
    ys_in = torch.full((bsz, max_target_len + 1), ignore_id, dtype=torch.long)
    ys_out = torch.full((bsz, max_target_len + 1), ignore_id, dtype=torch.long)

    for i, s in enumerate(batch):
        alen = s["audio"].shape[0]
        padded_audio[i, :alen, 0] = s["audio"]

        tgt = s["target"]
        tlen = tgt.shape[0]
        padded_targets[i, :tlen] = tgt
        flat_targets.extend(tgt.tolist())

        ys_in[i, 0] = sos_id
        ys_in[i, 1:tlen + 1] = tgt

        ys_out[i, :tlen] = tgt
        ys_out[i, tlen] = eos_id

    audio_lens_tensor = torch.tensor(audio_lens, dtype=torch.long)
    frame_lens_tensor = audio_lens_tensor // 640
    target_lens_tensor = torch.tensor(target_lens, dtype=torch.long)

    # Encoder mask: (B, 1, max_frames)
    max_frames = max_audio_len // 640
    enc_mask = make_non_pad_mask(frame_lens_tensor, maxlen=max_frames).unsqueeze(1)

    # Decoder subsequent mask: (1, L+1, L+1) combined with non-pad
    tgt_sub_mask = subsequent_mask(max_target_len + 1).unsqueeze(0)
    tgt_non_pad = (ys_in != ignore_id).unsqueeze(1)
    tgt_mask = tgt_sub_mask & tgt_non_pad

    return {
        "audio": padded_audio,
        "audio_lens": audio_lens_tensor,
        "frame_lens": frame_lens_tensor,
        "enc_mask": enc_mask,
        "padded_targets": padded_targets,
        "flat_targets": torch.tensor(flat_targets, dtype=torch.long),
        "target_lens": target_lens_tensor,
        "ys_in": ys_in,
        "ys_out": ys_out,
        "tgt_mask": tgt_mask,
        "texts": [s["text"] for s in batch],
        "paths": [s["path"] for s in batch],
    }


class VIVOSDataModule(LightningDataModule):
    def __init__(
        self,
        manifest_dir: str = "dataset/manifests",
        vivos_root: str = "dataset/vivos",
        tokenizer_path: str = "dataset/tokenizer/vivos_vi.model",
        batch_size: int = 4,
        num_workers: int = 2,
        max_duration_sec: float = 20.0,
    ):
        super().__init__()
        self.manifest_dir = Path(manifest_dir)
        self.vivos_root = vivos_root
        self.tokenizer_path = tokenizer_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_duration_sec = max_duration_sec

        self.sp = spm.SentencePieceProcessor(model_file=tokenizer_path)
        # odim = blank (0) + 1000 vocab units (1..1000) + eos/sos (1001)
        self.odim = self.sp.get_piece_size() + 2
        self.eos_id = self.odim - 1
        self.sos_id = self.odim - 1

    def setup(self, stage: Optional[str] = None):
        if stage in (None, "fit"):
            self.train_ds = VIVOSDataset(
                manifest_path=str(self.manifest_dir / "train.tsv"),
                vivos_root=self.vivos_root,
                tokenizer_path=self.tokenizer_path,
                max_duration_sec=self.max_duration_sec,
            )
            self.val_ds = VIVOSDataset(
                manifest_path=str(self.manifest_dir / "validation.tsv"),
                vivos_root=self.vivos_root,
                tokenizer_path=self.tokenizer_path,
                max_duration_sec=self.max_duration_sec,
            )
        if stage in (None, "test"):
            self.test_ds = VIVOSDataset(
                manifest_path=str(self.manifest_dir / "test.tsv"),
                vivos_root=self.vivos_root,
                tokenizer_path=self.tokenizer_path,
                max_duration_sec=self.max_duration_sec,
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=lambda b: vivos_collate_fn(
                b, eos_id=self.eos_id, sos_id=self.sos_id, ignore_id=-1
            ),
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=lambda b: vivos_collate_fn(
                b, eos_id=self.eos_id, sos_id=self.sos_id, ignore_id=-1
            ),
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=lambda b: vivos_collate_fn(
                b, eos_id=self.eos_id, sos_id=self.sos_id, ignore_id=-1
            ),
        )
