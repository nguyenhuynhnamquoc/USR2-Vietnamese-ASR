
from pathlib import Path

import torch
import sentencepiece as spm

from train_asr import make_model, decode_greedy
from data.vivos_asr import VIVOSASRDataset
from models.vietnamese_ctc import load_pretrained_audio_encoder
from utils.ctc_decode import beam_search

PROJECT_ROOT = Path(__file__).resolve().parents[1]


tokenizer = spm.SentencePieceProcessor(
    model_file=str(PROJECT_ROOT / "dataset" / "tokenizer" / "vivos_vi.model")
)

dataset = VIVOSASRDataset(
    str(PROJECT_ROOT / "dataset" / "manifests" / "train.tsv"),
    str(PROJECT_ROOT / "dataset" / "vivos"),
    str(PROJECT_ROOT / "dataset" / "tokenizer" / "vivos_vi.model")
)

sample = dataset[0]

model = make_model(
    tokenizer,
    PROJECT_ROOT / "conf" / "model" / "backbone" / "resnet_transformer_baseplus.yaml"
)

load_pretrained_audio_encoder(
    model, PROJECT_ROOT / "path" / "baseplus_high_resource_lrs3vox2.pth"
)

head = torch.load(
    PROJECT_ROOT / "output" / "exp004" / "ctc_head_1000.pt",
    map_location="cpu",
    weights_only=True,
)

model.ctc_projection.load_state_dict(head)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
model.to(device).eval()

audio = sample["audio"].unsqueeze(0).to(device)

with torch.inference_mode():
    with torch.autocast(
        device_type=device.type,
        dtype=torch.float16,
        enabled=device.type == "cuda",
    ):
        logits, lengths = model(
            audio,
            torch.tensor([audio.shape[1]], dtype=torch.long)
        )

greedy = decode_greedy(logits, lengths, tokenizer)[0]

beam_classes = beam_search(
    logits[0, :int(lengths[0])],
    beam_width=50,
    top_k=30,
)

beam_token_ids = [
    token - 1
    for token in beam_classes
    if token != 0
    and token - 1 not in (
        tokenizer.bos_id(),
        tokenizer.eos_id(),
    )
]

beam_text = tokenizer.decode(beam_token_ids)

print("\n=== EXP-006 RESULTS ===")
print("Reference:", repr(sample["text"]))
print("Greedy:", repr(greedy))
print("Beam search:", repr(beam_text))
print("Beam token IDs:", beam_token_ids)
print("Exact match:", beam_text == sample["text"])
