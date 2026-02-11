#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode
from lerobot.optim.optimizers import AdamWConfig
from lerobot.utils.constants import ACTION, OBS_IMAGE, OBS_STATE


def is_image_feature(key: str) -> bool:
    """Check if a feature key represents an image feature.

    Args:
        key: The feature key to check

    Returns:
        True if the key represents an image feature, False otherwise
    """
    return key.startswith(OBS_IMAGE)


@dataclass
class MLPNetworkConfig:
    """Configuration for MLP network architecture."""

    hidden_dims: list[int] = field(default_factory=lambda: [256, 256])
    activate_final: bool = True
    dropout_rate: float | None = None


@PreTrainedConfig.register_subclass("mlp")
@dataclass
class MLPConfig(PreTrainedConfig):
    """Configuration class for MLP Policy.

    This policy supports both deterministic and stochastic action prediction,
    and can predict action chunks instead of single actions.

    Args:
        chunk_size: Number of action steps to predict in one forward pass (action chunking).
        n_action_steps: Number of action steps to execute before querying the policy again.
        deterministic: If True, use deterministic policy (direct action prediction).
                      If False, use stochastic policy (predict distribution parameters).
        use_tanh_squash: Whether to use tanh squashing for stochastic actions.
        std_min: Minimum standard deviation for stochastic policy.
        std_max: Maximum standard deviation for stochastic policy.
        fixed_std: If provided, use a fixed standard deviation instead of learning it.
        init_final: Initialization scale for final layer weights.
        latent_dim: Hidden dimension for encoders.
        shared_encoder: Whether to share encoder between different modalities.
        vision_encoder_name: Name of pretrained vision encoder (None for default CNN).
        freeze_vision_encoder: Whether to freeze the vision encoder.
        image_encoder_hidden_dim: Hidden dimension for image encoder.
        image_embedding_pooling_dim: Dimension for image embedding pooling.
    """

    # Input / output structure
    chunk_size: int = 16  # Number of actions to predict at once
    n_action_steps: int = 16  # Number of actions to execute

    # Normalization
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MIN_MAX,
            "ENV": NormalizationMode.MIN_MAX,
            "ACTION": NormalizationMode.MIN_MAX,
        }
    )

    # Policy type
    deterministic: bool = True  # If False, use stochastic policy with distribution

    # Stochastic policy parameters
    use_tanh_squash: bool = True
    std_min: float = 1e-5
    std_max: float = 10.0
    fixed_std: float | None = None
    init_final: float = 0.05

    # Network architecture
    mlp_network_kwargs: MLPNetworkConfig = field(default_factory=MLPNetworkConfig)
    latent_dim: int = 256

    # Vision encoder settings
    vision_encoder_name: str | None = None
    freeze_vision_encoder: bool = False
    image_encoder_hidden_dim: int = 32
    image_embedding_pooling_dim: int = 8
    shared_encoder: bool = True

    # Training
    optimizer_lr: float = 1e-4
    optimizer_weight_decay: float = 1e-4

    def __post_init__(self):
        super().__post_init__()
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"n_action_steps ({self.n_action_steps}) cannot be greater than chunk_size ({self.chunk_size})"
            )

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            weight_decay=self.optimizer_weight_decay,
        )

    def get_scheduler_preset(self) -> None:
        return None

    def validate_features(self) -> None:
        """Validate that required features are present."""
        has_image = any(is_image_feature(key) for key in self.input_features)
        has_state = OBS_STATE in self.input_features

        if not (has_state or has_image):
            raise ValueError(
                "You must provide either 'observation.state' or an image observation "
                "(key starting with 'observation.image') in the input features"
            )

        if ACTION not in self.output_features:
            raise ValueError("You must provide 'action' in the output features")

    @property
    def image_features(self) -> list[str]:
        return [key for key in self.input_features if is_image_feature(key)]

    @property
    def observation_delta_indices(self) -> list:
        return None

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
