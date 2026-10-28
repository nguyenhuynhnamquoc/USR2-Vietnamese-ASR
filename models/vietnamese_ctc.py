import torch
from torch import nn

from espnet.nets.pytorch_backend.transformer.encoder_av import Encoder


class VietnameseCTCASR(nn.Module):
    def __init__(
        self,
        vocabulary_size,
        idim=512,
        attention_dim=512,
        attention_heads=8,
        linear_units=2048,
        num_blocks=12,
        gamma_init=0.1,
    ):
        super().__init__()
        self.encoder = Encoder(
            idim=idim,
            attention_dim=attention_dim,
            attention_heads=attention_heads,
            linear_units=linear_units,
            num_blocks=num_blocks,
            gamma_init=gamma_init,
        )
        del self.encoder.frontend_v
        del self.encoder.linear_v
        del self.encoder.linear_av
        self.ctc_projection = nn.Linear(attention_dim, vocabulary_size + 1)

    def forward(self, audio, audio_lengths):
        frame_lengths = torch.div(audio_lengths, 640, rounding_mode="floor")
        if torch.any(frame_lengths <= 0):
            raise ValueError("Every audio sample must contain at least 640 samples.")
        frame_lengths = frame_lengths.to(device=audio.device)
        max_frames = audio.shape[1] // 640
        if torch.any(frame_lengths > max_frames):
            raise ValueError("Audio lengths exceed the padded batch audio size.")

        frame_positions = torch.arange(max_frames, device=audio.device)
        padding_mask = frame_positions.unsqueeze(0) < frame_lengths.unsqueeze(1)
        encoded = self.encoder(
            xs_a=audio.unsqueeze(-1),
            masks=padding_mask.unsqueeze(1),
        )
        return self.ctc_projection(encoded), frame_lengths


def load_pretrained_audio_encoder(model, checkpoint_path):
    checkpoint_path = str(checkpoint_path)
    state = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    if "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]

    target_state = model.encoder.state_dict()
    prefixes = (
        "_orig_mod.model.backbone.encoder.",
        "model.backbone.encoder.",
        "backbone.encoder.",
        "encoder.",
    )
    encoder_state = {}
    for key, value in state.items():
        normalized_key = key
        for prefix in prefixes:
            if normalized_key.startswith(prefix):
                normalized_key = normalized_key[len(prefix):]
                break
        if normalized_key in target_state and target_state[normalized_key].shape == value.shape:
            encoder_state[normalized_key] = value

    if not encoder_state:
        raise ValueError(
            f"No compatible audio encoder weights found in checkpoint: {checkpoint_path}"
        )

    result = model.encoder.load_state_dict(encoder_state, strict=False)
    matched = len(encoder_state)
    del state
    return matched, result
