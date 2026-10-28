
from pathlib import Path

import torch
import sentencepiece as spm

from train_asr import make_model, decode_greedy
from data.vivos_asr import VIVOSASRDataset
from models.vietnamese_ctc import load_pretrained_audio_encoder

torch.manual_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint_path = Path("output/exp004/ctc_head_1000.pt")
if not checkpoint_path.is_file():
    raise FileNotFoundError(
        f"Missing checkpoint: {checkpoint_path}"
    )

tokenizer = spm.SentencePieceProcessor(
    model_file="dataset/tokenizer/vivos_vi.model"
)

dataset = VIVOSASRDataset(
    "dataset/manifests/train.tsv",
    "dataset/vivos",
    "dataset/tokenizer/vivos_vi.model"
)

sample = dataset[0]

model = make_model(
    tokenizer,
    "conf/model/backbone/resnet_transformer_baseplus.yaml"
)

load_pretrained_audio_encoder(
    model,
    "../usr2-original/checkpoints/baseplus.pth"
)

head_state = torch.load(
    checkpoint_path,
    map_location="cpu",
    weights_only=True
)
model.ctc_projection.load_state_dict(head_state)
model.to(device).eval()

audio = sample["audio"].unsqueeze(0).to(device)
frame_count = audio.shape[1] // 640

mask = torch.ones(
    (1, 1, frame_count),
    dtype=torch.bool,
    device=device
)

with torch.inference_mode():
    with torch.autocast(
        device_type=device.type,
        dtype=torch.float16,
        enabled=device.type == "cuda"
    ):
        features = model.encoder(
            xs_a=audio.unsqueeze(-1),
            masks=mask
        )

    logits = model.ctc_projection(features.float())

lengths = torch.tensor([logits.shape[1]], dtype=torch.long)

# Original target token IDs (without CTC offset)
target_ids = [
    int(token_id) - 1
    for token_id in sample["tokens"].tolist()
]

print("\n=== 1. REFERENCE ===")
print("Text:", repr(sample["text"]))
print("Token IDs:", target_ids)
print("Pieces:", tokenizer.id_to_piece(target_ids))

# Greedy CTC prediction
frame_classes = logits[0].argmax(dim=-1).tolist()

decoded_ids = []
previous = None

print("\n=== 2. CTC NON-BLANK EVENTS ===")

probabilities = logits[0].softmax(dim=-1)

for frame, class_id in enumerate(frame_classes):
    if class_id != 0 and class_id != previous:
        token_id = class_id - 1
        if token_id not in (
            tokenizer.bos_id(),
            tokenizer.eos_id()
        ):
            decoded_ids.append(token_id)

            print(
                f"Frame {frame:3d} | "
                f"ID {token_id:4d} | "
                f"Piece {tokenizer.id_to_piece(token_id)!r} | "
                f"Top-1 probability "
                f"{probabilities[frame, class_id].item():.4f}"
            )

    previous = class_id

print("\n=== 3. DECODED RESULT ===")
print("Predicted token IDs:", decoded_ids)
print("Predicted pieces:", tokenizer.id_to_piece(decoded_ids))
print("SentencePiece decoded:", repr(tokenizer.decode(decoded_ids)))
print("Greedy decoded:", repr(
    decode_greedy(logits, lengths, tokenizer)[0]
))

print("\n=== 4. BLANK STATISTICS ===")
blank_ratio = (
    torch.tensor(frame_classes) == 0
).float().mean().item()

blank_prob = probabilities[:, 0]

print(f"Blank argmax ratio: {blank_ratio:.2%}")
print(f"Mean blank probability: {blank_prob.mean().item():.4f}")
print(f"Max blank probability: {blank_prob.max().item():.4f}")

print("\n=== 5. CTC LOSS ===")
loss = torch.nn.CTCLoss(blank=0, zero_infinity=True)(
    logits.log_softmax(-1).transpose(0, 1),
    sample["tokens"].to(device),
    lengths,
    torch.tensor([len(sample["tokens"])], dtype=torch.long)
)
print(f"CTC loss: {loss.item():.6f}")
