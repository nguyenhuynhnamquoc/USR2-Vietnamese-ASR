
import torch
import sentencepiece as spm

from train_asr import make_model, decode_greedy
from data.vivos_asr import VIVOSASRDataset
from models.vietnamese_ctc import load_pretrained_audio_encoder

torch.manual_seed(42)
torch.set_num_threads(4)

device = torch.device("cuda")
tokenizer = spm.SentencePieceProcessor(
    model_file="dataset/tokenizer/vivos_vi.model"
)

dataset = VIVOSASRDataset(
    "dataset/manifests/train.tsv",
    "dataset/vivos",
    "dataset/tokenizer/vivos_vi.model"
)

sample = dataset[0]
reference = sample["text"]

model = make_model(
    tokenizer,
    "conf/model/backbone/resnet_transformer_baseplus.yaml"
)

matched, _ = load_pretrained_audio_encoder(
    model, "../usr2-original/checkpoints/baseplus.pth"
)
model.to(device).eval()

# Cache frozen encoder features once
audio = sample["audio"].unsqueeze(0).to(device)
frames = audio.shape[1] // 640
mask = torch.ones((1, 1, frames), dtype=torch.bool, device=device)

with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
    features = model.encoder(
        xs_a=audio.unsqueeze(-1),
        masks=mask
    ).float().detach()

targets = sample["tokens"].to(device)
input_lengths = torch.tensor([features.shape[1]], dtype=torch.long)
target_lengths = torch.tensor([len(targets)], dtype=torch.long)

criterion = torch.nn.CTCLoss(blank=0, zero_infinity=True)
optimizer = torch.optim.AdamW(
    model.ctc_projection.parameters(),
    lr=1e-3,
    weight_decay=0
)

print("Loaded encoder tensors:", matched)
print("Reference:", reference)
print("Audio duration:", audio.shape[1] / 16000, "seconds")

for step in range(1, 1001):
    optimizer.zero_grad(set_to_none=True)

    logits = model.ctc_projection(features)
    log_probs = logits.log_softmax(-1).transpose(0, 1)

    loss = criterion(
        log_probs, targets, input_lengths, target_lengths
    )

    if not torch.isfinite(loss):
        raise RuntimeError(f"Non-finite loss at step {step}")

    loss.backward()
    optimizer.step()

    with torch.no_grad():
        updated_logits = model.ctc_projection(features)
        prediction = decode_greedy(
            updated_logits, input_lengths, tokenizer
        )[0]
        classes = updated_logits.argmax(dim=-1)
        blank_ratio = (classes == 0).float().mean().item()

    if step == 1 or step % 10 == 0 or prediction == reference:
        print(
            f"Step {step:3d} | Loss {loss.item():.4f} "
            f"| Blank {blank_ratio:.2%} "
            f"| Prediction: {prediction!r}",
            flush=True
        )

    if prediction == reference:
        print("SUCCESS: Sample memorized!")
        break


torch.save(
    model.ctc_projection.state_dict(),
    "output/exp004/ctc_head_1000.pt"
)