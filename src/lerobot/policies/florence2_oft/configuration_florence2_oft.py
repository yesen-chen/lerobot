"""Florence2-OFT Policy configuration.

Florence2-OFT uses Florence2 vision-language encoder as backbone with an MLP action head
for continuous action prediction via L1 regression. The decoder is removed for efficiency.

Reference:
  - Florence2: https://huggingface.co/microsoft/Florence-2-large
"""

from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode
from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import DiffuserSchedulerConfig


@PreTrainedConfig.register_subclass("florence2_oft")
@dataclass
class Florence2OFTConfig(PreTrainedConfig):
    """Configuration class for Florence2OFTPolicy.

    This policy uses a Florence2 vision-language model encoder to process images
    and text prompts, then predicts continuous actions via an MLP action head using
    L1 regression on the encoder's hidden states at action token positions.

    Args:
        n_obs_steps: Number of observation steps (fixed to 1 for Florence2).
        chunk_size: Action prediction horizon (number of future action steps to predict).
        n_action_steps: Number of actions to execute per inference call.
        vlm_model_name_or_path: HuggingFace model ID or local path for Florence2 model.
        action_head_hidden_mult: Hidden dimension multiplier for the MLP action head.
        action_head_num_blocks: Number of MLP ResNet blocks in the action head.
        action_token: Special token used to mark action prediction positions.
        image_size: Image size for Florence2 processor [H, W]. None uses model default.
        task_prompt: Text prompt prefix for action prediction.
        freeze_vision_tower: Whether to freeze the vision encoder during training.
        freeze_backbone: Whether to freeze the language encoder backbone during training.
    """

    n_obs_steps: int = 1
    chunk_size: int = 16
    n_action_steps: int = 16

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            # Florence2 processor handles image normalization internally.
            # Using IDENTITY here prevents double-normalization: lerobot normalizes
            # images from [0,1] to ~[-2,2] with MEAN_STD, which then gets fed into
            # Florence2 processor again. With IDENTITY, images stay in [0,1] and
            # Florence2 processor normalizes them correctly.
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MIN_MAX,
            "ACTION": NormalizationMode.MIN_MAX,
        }
    )

    drop_n_last_frames: int = 0

    vlm_model_name_or_path: str = "microsoft/Florence-2-base"
    action_head_hidden_mult: int = 2
    action_head_num_blocks: int = 2
    # Match ILStudio florence2_oft policy defaults. Their config explicitly uses
    # bf16 because fp16 overflows in Florence2's DaViT vision encoder.
    action_token: str = "🔍"
    use_bf16: bool = True
    image_size: list[int] | None = None
    use_multi_view: bool = False
    clip_action: bool = True
    clip_action_range: float = 1.0
    task_prompt: str = "Predict the next robot actions:"

    freeze_vision_tower: bool = False
    freeze_backbone: bool = False

    # Training presets aligned with openpi_lora.yaml
    optimizer_lr: float = 5e-5
    optimizer_betas: tuple = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-10
    optimizer_grad_clip_norm: float = 1.0
    scheduler_name: str = "cosine"
    scheduler_warmup_steps: int = 1000

    def __post_init__(self):
        super().__post_init__()
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"`n_action_steps` ({self.n_action_steps}) must be <= `chunk_size` ({self.chunk_size})."
            )

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self) -> DiffuserSchedulerConfig:
        return DiffuserSchedulerConfig(
            name=self.scheduler_name,
            num_warmup_steps=self.scheduler_warmup_steps,
        )

    def validate_features(self) -> None:
        if not self.image_features:
            raise ValueError("Florence2-OFT requires at least one image input feature.")

    @property
    def observation_delta_indices(self) -> list:
        return list(range(1 - self.n_obs_steps, 1))

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
