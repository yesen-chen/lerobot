from dataclasses import dataclass


@dataclass
class RLTConfig:
    """Configuration for RL Token encoder-decoder and actor training."""

    # Encoder-Decoder transformer
    enc_dec_num_layers: int = 2
    enc_dec_num_heads: int = 4
    enc_dec_hidden_dim: int | None = None  # None = auto from VLA's text_config.hidden_size
    enc_dec_dropout: float = 0.1
    enc_dec_ff_dim_multiplier: int = 4

    # Actor MLP
    actor_hidden_dim: int = 256
    actor_chunk_size: int = 10
    ref_action_dropout: float = 0.5

    # Training
    lr: float = 1e-4
    weight_decay: float = 1e-4
    grad_clip_norm: float = 10.0
    warmup_steps: int = 500
