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

"""MLP Policy

A simple MLP-based policy that supports:
- Deterministic and stochastic action prediction
- Action chunking for temporal consistency
- Image and state observations
"""

from collections import deque
from dataclasses import asdict

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from torch import Tensor
from torch.distributions import Distribution, MultivariateNormal

from lerobot.policies.mlp.configuration_mlp import MLPConfig
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.diffusion.modeling_diffusion import DiffusionRgbEncoder
from lerobot.policies.sac.modeling_sac import MLP, TanhMultivariateNormalDiag, orthogonal_init
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_STATE


class MLPObservationEncoder(nn.Module):
    """Diffusion-style observation encoder (ResNet + SpatialSoftmax + concat state/env)."""

    def __init__(self, config: MLPConfig) -> None:
        super().__init__()
        self.config = config
        self.image_keys = list(config.image_features.keys())
        self.has_images = bool(self.image_keys)
        self.has_state = config.robot_state_feature is not None
        self.has_env = config.env_state_feature is not None

        if self.has_images:
            if config.use_separate_rgb_encoder_per_camera:
                self.rgb_encoders = nn.ModuleList(
                    [DiffusionRgbEncoder(config) for _ in range(len(self.image_keys))]
                )
                self._image_feature_dim = self.rgb_encoders[0].feature_dim * len(self.image_keys)
            else:
                self.rgb_encoder = DiffusionRgbEncoder(config)
                self._image_feature_dim = self.rgb_encoder.feature_dim * len(self.image_keys)
        else:
            self._image_feature_dim = 0

        output_dim = 0
        if self.has_state:
            output_dim += config.robot_state_feature.shape[0]
        if self.has_env:
            output_dim += config.env_state_feature.shape[0]
        output_dim += self._image_feature_dim
        self.output_dim = output_dim

    def forward(self, obs: dict[str, Tensor]) -> Tensor:
        parts: list[Tensor] = []
        if self.has_state:
            parts.append(obs[OBS_STATE])
        if self.has_env:
            parts.append(obs[OBS_ENV_STATE])
        if self.has_images:
            if self.config.use_separate_rgb_encoder_per_camera:
                img_features = [
                    encoder(obs[key]) for encoder, key in zip(self.rgb_encoders, self.image_keys, strict=True)
                ]
            else:
                img_features = [self.rgb_encoder(obs[key]) for key in self.image_keys]
            parts.append(torch.cat(img_features, dim=-1))

        if not parts:
            raise ValueError("No observation components found to encode.")
        return torch.cat(parts, dim=-1)


class MLPPolicy(PreTrainedPolicy):
    """MLP Policy with support for deterministic/stochastic predictions and action chunking."""

    config_class = MLPConfig
    name = "mlp"

    def __init__(
        self,
        config: MLPConfig | None = None,
        dataset_stats: dict | None = None,
        dataset_meta=None,
        **kwargs,
    ):
        """Initialize MLP Policy.
        
        Args:
            config: Policy configuration
            dataset_stats: Dataset statistics (passed by factory but not used)
            dataset_meta: Dataset metadata (passed by factory but not used)
            **kwargs: Additional arguments (ignored for compatibility)
        """
        super().__init__(config)
        config.validate_features()
        self.config = config

        # Get action dimension (single step)
        self.action_dim = config.output_features[ACTION].shape[0]

        # Initialize diffusion-style encoder
        self.encoder = MLPObservationEncoder(config)

        # Initialize MLP backbone
        # Convert mlp_network_kwargs to dict if it's a dataclass, otherwise use it as is
        network_kwargs = (
            asdict(self.config.mlp_network_kwargs)
            if hasattr(self.config.mlp_network_kwargs, "__dataclass_fields__")
            else self.config.mlp_network_kwargs
        )
        self.network = MLP(input_dim=self.encoder.output_dim, **network_kwargs)

        # Get the output dimension from the last linear layer of the network
        for layer in reversed(self.network.net):
            if isinstance(layer, nn.Linear):
                network_out_dim = layer.out_features
                break

        if self.config.deterministic:
            # Deterministic policy: directly predict actions
            # Output shape: [batch_size, chunk_size * action_dim]
            self.action_head = nn.Linear(network_out_dim, self.config.chunk_size * self.action_dim)
            if self.config.init_final is not None:
                nn.init.uniform_(
                    self.action_head.weight, -self.config.init_final, self.config.init_final
                )
                nn.init.uniform_(self.action_head.bias, -self.config.init_final, self.config.init_final)
            else:
                orthogonal_init()(self.action_head.weight)
        else:
            # Stochastic policy: predict distribution parameters
            # Mean layer
            self.mean_layer = nn.Linear(network_out_dim, self.config.chunk_size * self.action_dim)
            if self.config.init_final is not None:
                nn.init.uniform_(
                    self.mean_layer.weight, -self.config.init_final, self.config.init_final
                )
                nn.init.uniform_(self.mean_layer.bias, -self.config.init_final, self.config.init_final)
            else:
                orthogonal_init()(self.mean_layer.weight)

            # Standard deviation layer (if not using fixed std)
            if self.config.fixed_std is None:
                self.std_layer = nn.Linear(network_out_dim, self.config.chunk_size * self.action_dim)
                if self.config.init_final is not None:
                    nn.init.uniform_(
                        self.std_layer.weight, -self.config.init_final, self.config.init_final
                    )
                    nn.init.uniform_(
                        self.std_layer.bias, -self.config.init_final, self.config.init_final
                    )
                else:
                    orthogonal_init()(self.std_layer.weight)

        # Action queue for sequential execution
        self.reset()

    def reset(self):
        """Reset the policy state (action queue)."""
        self._action_queue = deque([], maxlen=self.config.n_action_steps)

    def get_optim_params(self) -> dict:
        """Return optimizer parameter groups with optional backbone learning rate scaling."""
        if self.config.backbone_lr_scale != 1.0 and self.config.image_features:
            backbone_params = []
            other_params = []
            for name, param in self.named_parameters():
                if "backbone" in name:
                    backbone_params.append(param)
                else:
                    other_params.append(param)
            backbone_lr = self.config.optimizer_lr * self.config.backbone_lr_scale
            return [
                {"params": other_params},
                {"params": backbone_params, "lr": backbone_lr},
            ]
        return self.parameters()

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select a single action given environment observations.

        This method manages an action queue and only calls predict_action_chunk
        when the queue is empty.

        Args:
            batch: Dictionary containing observations

        Returns:
            Single action tensor of shape [batch_size, action_dim]
        """
        self.eval()

        # If action queue is empty, predict a new chunk
        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch)[:, : self.config.n_action_steps]
            # Transpose to get [n_action_steps, batch_size, action_dim]
            self._action_queue.extend(actions.transpose(0, 1))

        return self._action_queue.popleft()

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        """Predict a chunk of actions given environment observations.

        Args:
            batch: Dictionary containing observations

        Returns:
            Action chunk tensor of shape [batch_size, chunk_size, action_dim]
        """
        self.eval()

        if self.config.deterministic:
            # Deterministic prediction
            actions = self._forward_deterministic(batch)
        else:
            # Stochastic prediction: sample from distribution
            actions, _, _ = self._forward_stochastic(batch)

        return actions

    def forward(self, batch: dict[str, Tensor], reduction: str = "mean") -> tuple[Tensor, dict]:
        """Forward pass for training.

        Args:
            batch: Dictionary containing observations and actions
            reduction: Loss reduction method - "mean" for standard training, "none" for per-sample loss.

        Returns:
            loss: Total loss
            loss_dict: Dictionary of individual loss components
        """
        if self.config.deterministic:
            actions_pred = self._forward_deterministic(batch)
            actions_target = batch[ACTION]  # [batch_size, chunk_size, action_dim]

            # Compute loss (L1 or MSE, configurable)
            if self.config.loss_type == "l1":
                loss = F.l1_loss(actions_pred, actions_target, reduction="none")
            else:
                loss = F.mse_loss(actions_pred, actions_target, reduction="none")

            # Mask loss wherever the action is padded with copies (edges of dataset trajectory)
            if self.config.do_mask_loss_for_padding and "action_is_pad" in batch:
                in_episode_bound = ~batch["action_is_pad"]
                loss = loss * in_episode_bound.unsqueeze(-1)

            if reduction == "none":
                loss = loss.mean(dim=(1, 2))  # per-sample loss
            else:
                loss = loss.mean()

            loss_dict = {f"{self.config.loss_type}_loss": loss.item()}
        else:
            # Stochastic policy: use L1/MSE loss on predicted mean
            actions_pred, log_probs, means = self._forward_stochastic(batch)
            actions_target = batch[ACTION]  # [batch_size, chunk_size, action_dim]

            if self.config.loss_type == "l1":
                loss = F.l1_loss(means, actions_target, reduction="none")
            else:
                loss = F.mse_loss(means, actions_target, reduction="none")

            # Mask loss wherever the action is padded
            if self.config.do_mask_loss_for_padding and "action_is_pad" in batch:
                in_episode_bound = ~batch["action_is_pad"]
                loss = loss * in_episode_bound.unsqueeze(-1)

            if reduction == "none":
                loss = loss.mean(dim=(1, 2))
            else:
                loss = loss.mean()

            loss_dict = {f"{self.config.loss_type}_loss": loss.item()}

        return loss, loss_dict

    def _forward_deterministic(self, batch: dict[str, Tensor]) -> Tensor:
        """Forward pass for deterministic policy.

        Args:
            batch: Dictionary containing observations

        Returns:
            Predicted actions of shape [batch_size, chunk_size, action_dim]
        """
        # Encode observations
        obs_enc = self.encoder(batch)

        # Pass through network
        features = self.network(obs_enc)

        # Predict actions
        actions_flat = self.action_head(features)  # [batch_size, chunk_size * action_dim]

        # Reshape to [batch_size, chunk_size, action_dim]
        batch_size = actions_flat.shape[0]
        actions = actions_flat.reshape(batch_size, self.config.chunk_size, self.action_dim)

        return actions

    def _forward_stochastic(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor, Tensor]:
        """Forward pass for stochastic policy.

        Args:
            batch: Dictionary containing observations

        Returns:
            actions: Sampled actions of shape [batch_size, chunk_size, action_dim]
            log_probs: Log probabilities of sampled actions
            means: Mean of the distribution (tanh-transformed when use_tanh_squash=True)
        """
        actions_flat, log_probs, means_flat, dist = self._get_distribution(batch)

        # Reshape actions to [batch_size, chunk_size, action_dim]
        batch_size = actions_flat.shape[0]
        actions = actions_flat.reshape(batch_size, self.config.chunk_size, self.action_dim)
        means = means_flat.reshape(batch_size, self.config.chunk_size, self.action_dim)

        return actions, log_probs, means

    def _get_distribution(
        self, batch: dict[str, Tensor]
    ) -> tuple[Tensor, Tensor, Tensor, Distribution]:
        """Get the action distribution for stochastic policy.

        Args:
            batch: Dictionary containing observations

        Returns:
            actions: Sampled actions (flat)
            log_probs: Log probabilities of sampled actions
            means: Mean of the distribution (flat). When use_tanh_squash=True,
                   this is the tanh-transformed mean for compatibility with action targets.
            dist: The distribution object
        """
        # Encode observations
        obs_enc = self.encoder(batch)

        # Pass through network
        features = self.network(obs_enc)

        # Get mean
        means = self.mean_layer(features)  # [batch_size, chunk_size * action_dim]

        # Get standard deviation
        if self.config.fixed_std is None:
            log_std = self.std_layer(features)
            std = torch.exp(log_std)
            std = torch.clamp(std, self.config.std_min, self.config.std_max)
        else:
            std = torch.full_like(means, self.config.fixed_std)

        # Build transformed distribution
        if self.config.use_tanh_squash:
            dist = TanhMultivariateNormalDiag(loc=means, scale_diag=std)
            # Apply tanh to means so they are in the same space as target actions
            transformed_means = torch.tanh(means)
        else:
            dist = MultivariateNormal(means, torch.diag_embed(std))
            transformed_means = means

        # Sample actions (reparameterized)
        actions = dist.rsample()

        # Compute log probabilities
        log_probs = dist.log_prob(actions)

        return actions, log_probs, transformed_means, dist
