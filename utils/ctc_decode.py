import math

import torch


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
    if logits.ndim != 2:
        raise ValueError("Expected logits with shape (time, classes).")
    if beam_width <= 0 or top_k <= 0:
        raise ValueError("beam_width and top_k must be positive.")

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

    return max(beams, key=lambda prefix: logadd(*beams[prefix]))
