
import math
import torch
import sentencepiece as spm

from train_asr import make_model, decode_greedy
from data.vivos_asr import VIVOSASRDataset
from models.vietnamese_ctc import load_pretrained_audio_encoder


NEG_INF = float("-inf")


def logadd(a, b):
    if a == NEG_INF:
        return b
    if b == NEG_INF:
        return a
    if a < b:
        a, b = b, a
    return a + math.log1p(math.exp(b - a))


def beam_search(logits, beam_width=50, top_k=30):
    log_probs = logits.float().log_softmax(-1).cpu()
    beams = {(): (0.0, NEG_INF)}

    for frame_probs in log_probs:
        candidates = frame_probs.topk(
            min(top_k, frame_probs.numel())
        ).indices.tolist()

        if 0 not in candidates:
            candidates.append(0)

        next_beams = {}

        for prefix, (p_blank, p_nonblank) in beams.items():
            total = logadd(p_blank, p_nonblank)

            # CTC blank transition
            old_blank, old_nonblank = next_beams.get(
                prefix, (NEG_INF, NEG_INF)
            )
            next_beams[prefix] = (
                logadd(old_blank, total + frame_probs[0].item()),
                old_nonblank,
            )

            for token in candidates:
                if token == 0:
                    continue

                score = frame_probs[token].item()

                if prefix and token == prefix[-1]:
                    old_blank, old_nonblank = next_beams.get(
                        prefix, (NEG_INF, NEG_INF)
                    )
                    next_beams[prefix] = (
                        old_blank,
                        logadd(old_nonblank, p_nonblank + score),
                    )
                    extended_score = p_blank + score
                else:
                    extended_score = total + score

                extended = prefix + (token,)
                old_blank, old_nonblank = next_beams.get(
                    extended, (NEG_INF, NEG_INF)
                )
                next_beams[extended] = (
                    old_blank,
                    logadd(old_nonblank, extended_score),
                )

        ranked = sorted(
            next_beams.items(),
            key=lambda item: logadd(*item[1]),
            reverse=True,
        )
        beams = dict(ranked[:beam_width])

    return max(beams, key=lambda p: logadd(*beams[p]))


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
    model, "../usr2-original/checkpoints/baseplus.pth"
)

head = torch.load(
    "output/exp004/ctc_head_1000.pt",
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
