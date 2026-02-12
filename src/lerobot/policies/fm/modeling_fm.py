#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

"""Flow Matching Policy for robot visuomotor learning.

Implements Conditional Flow Matching (Lipman et al., 2023) with:
- Image + state observation encoding (same as Diffusion Policy)
- 1D UNet backbone for velocity field prediction
- PI05-consistent integration direction (t: 1→0) for RTC compatibility
- Support for both Euler and adaptive ODE (dopri5) solvers
"""

"""
lerobot-train     --dataset.repo_id=lerobot/aloha_sim_transfer_cube_human     --policy.type=diffusion     --steps=200000     --batch_size=64     --eval_freq=10000     --save_freq=100000 --dataset.image_transforms.enable=false     --policy.device=cuda     --policy.use_amp=true     --wandb.enable=true     --wandb.project=aloha_dp     --policy.push_to_hub=false     --env.type=aloha     --env.task=AlohaTransferCube-v0 --policy.pretrained_backbone_weights=ResNet18_Weights.IMAGENET1K_V1  --policy.use_group_norm=false --policy.backbone_lr_scale=0.1
"""

import math
from collections import deque
from collections.abc import Callable
from typing import TypedDict

import einops
import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
import torchvision
from diffusers.training_utils import EMAModel
from torch import Tensor, nn
from typing_extensions import Unpack

from lerobot.policies.fm.configuration_fm import FlowMatchingConfig
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.policies.rtc.modeling_rtc import RTCProcessor
from lerobot.policies.utils import (
    get_device_from_parameters,
    get_dtype_from_parameters,
    get_output_shape,
    populate_queues,
)
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE


class ActionSelectKwargs(TypedDict, total=False):
    noise: Tensor | None
    inference_delay: int | None
    prev_chunk_left_over: Tensor | None
    execution_horizon: int | None


class FlowMatchingPolicy(PreTrainedPolicy):
    """Flow Matching Policy for visuomotor robot learning.

    Uses Conditional Flow Matching to learn a velocity field that transports noise to
    expert actions. Supports images + state observations, same as Diffusion Policy.
    Integration direction is compatible with PI05 / RTC (t: 1 → 0).
    """

    config_class = FlowMatchingConfig
    name = "flow_matching"

    def __init__(
        self,
        config: FlowMatchingConfig,
        **kwargs,
    ):
        super().__init__(config)
        config.validate_features()
        self.config = config

        self._queues = None

        # Initialize RTC processor
        self.rtc_processor: RTCProcessor | None = None
        if config.rtc_config is not None:
            self.rtc_processor = RTCProcessor(config.rtc_config)

        self.flow_matching = FlowMatchingModel(config)

        # Initialize EMA if enabled
        self.ema = None
        self._ema_device_set = False
        if config.use_ema:
            self.ema = EMAModel(
                self.flow_matching.parameters(),
                power=config.ema_power,
            )

        self.reset()

    def get_optim_params(self) -> dict:
        """Return optimizer parameter groups with optional backbone learning rate scaling."""
        if self.config.backbone_lr_scale != 1.0 and self.config.image_features:
            # Separate backbone and non-backbone parameters
            backbone_params = []
            other_params = []
            for name, param in self.flow_matching.named_parameters():
                if "backbone" in name:
                    backbone_params.append(param)
                else:
                    other_params.append(param)
            # Calculate actual backbone LR from base LR and scale factor
            backbone_lr = self.config.optimizer_lr * self.config.backbone_lr_scale
            return [
                {"params": other_params},
                {"params": backbone_params, "lr": backbone_lr},
            ]
        return self.flow_matching.parameters()

    def _rtc_enabled(self) -> bool:
        return self.config.rtc_config is not None and self.config.rtc_config.enabled

    def reset(self):
        """Clear observation and action queues. Should be called on `env.reset()`"""
        self._queues = {
            OBS_STATE: deque(maxlen=self.config.n_obs_steps),
            ACTION: deque(maxlen=self.config.n_action_steps),
        }
        if self.config.image_features:
            self._queues[OBS_IMAGES] = deque(maxlen=self.config.n_obs_steps)
        if self.config.env_state_feature:
            self._queues[OBS_ENV_STATE] = deque(maxlen=self.config.n_obs_steps)

    @torch.no_grad()
    def predict_action_chunk(
        self, batch: dict[str, Tensor], **kwargs: Unpack[ActionSelectKwargs]
    ) -> Tensor:
        """Predict a chunk of actions given environment observations."""
        batch = {k: torch.stack(list(self._queues[k]), dim=1) for k in batch if k in self._queues}
        actions = self.flow_matching.generate_actions(
            batch,
            noise=kwargs.get("noise"),
            rtc_processor=self.rtc_processor if self._rtc_enabled() else None,
            inference_delay=kwargs.get("inference_delay"),
            prev_chunk_left_over=kwargs.get("prev_chunk_left_over"),
            execution_horizon=kwargs.get("execution_horizon"),
        )
        return actions

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], **kwargs: Unpack[ActionSelectKwargs]) -> Tensor:
        """Select a single action given environment observations.

        This method handles caching a history of observations and an action trajectory.
        Same scheme as Diffusion Policy:
          - `n_obs_steps` observations are cached.
          - The model generates `horizon` steps worth of actions.
          - `n_action_steps` worth of actions are kept for execution.
        """
        if ACTION in batch:
            batch.pop(ACTION)

        if self.config.image_features:
            batch = dict(batch)
            # At inference, images are (B, C, H, W) = 4D (single timestep)
            # Stack cameras at dim=1 to get (B, n_cameras, C, H, W) per timestep
            # The queue will store these, and predict_action_chunk stacks at dim=1 to get
            # (B, n_obs_steps, n_cameras, C, H, W)
            batch[OBS_IMAGES] = torch.stack([batch[key] for key in self.config.image_features], dim=1)

        self._queues = populate_queues(self._queues, batch)

        if len(self._queues[ACTION]) == 0:
            actions = self.predict_action_chunk(batch, **kwargs)
            self._queues[ACTION].extend(actions.transpose(0, 1))

        action = self._queues[ACTION].popleft()
        return action

    def forward(self, batch: dict[str, Tensor], reduction: str = "mean") -> tuple[Tensor, None]:
        """Run the batch through the model and compute the loss for training or validation."""
        if self.config.image_features:
            batch = dict(batch)
            # Get the first image to determine the shape
            first_key = next(iter(self.config.image_features))
            first_img = batch[first_key]

            # Expected shape is (B, n_obs_steps, C, H, W) = 5D
            # If 4D (B, C, H, W), add the n_obs_steps dimension
            if first_img.dim() == 4:
                images = [batch[key].unsqueeze(1) for key in self.config.image_features]
            else:
                images = [batch[key] for key in self.config.image_features]

            # Stack cameras at dim 2: (B, n_obs_steps, n_cameras, C, H, W)
            batch[OBS_IMAGES] = torch.stack(images, dim=2)
        loss = self.flow_matching.compute_loss(batch, reduction=reduction)
        return loss, None

    def update(self):
        """Update EMA parameters after each training step."""
        if self.ema is not None:
            # Ensure EMA shadow parameters are on the same device as model parameters (only once)
            if not self._ema_device_set:
                device = next(self.flow_matching.parameters()).device
                self.ema.to(device)
                self._ema_device_set = True
            self.ema.step(self.flow_matching.parameters())

    def use_ema_weights(self):
        """Switch to EMA weights for inference."""
        if self.ema is not None:
            # Store current weights
            self.ema.store(self.flow_matching.parameters())
            # Copy EMA weights to model
            self.ema.copy_to(self.flow_matching.parameters())

    def restore_training_weights(self):
        """Restore training weights after inference with EMA."""
        if self.ema is not None:
            self.ema.restore(self.flow_matching.parameters())


# ==============================================================================
# Flow Matching Model
# ==============================================================================


class FlowMatchingModel(nn.Module):
    """Core Flow Matching model with observation encoding and velocity prediction.

    Integration direction follows PI05 convention:
        x_t = t * noise + (1 - t) * action    (t=0 → action, t=1 → noise)
        u_t = noise - action                   (target velocity)
        Sampling: t goes from 1 → 0            (noise → action)

    This is compatible with RTC guidance which expects time going from 1 → 0.
    """

    def __init__(self, config: FlowMatchingConfig):
        super().__init__()
        self.config = config

        # Build observation encoders (same as Diffusion Policy).
        global_cond_dim = self.config.robot_state_feature.shape[0]
        if self.config.image_features:
            num_images = len(self.config.image_features)
            if self.config.use_separate_rgb_encoder_per_camera:
                encoders = [FMRgbEncoder(config) for _ in range(num_images)]
                self.rgb_encoder = nn.ModuleList(encoders)
                global_cond_dim += encoders[0].feature_dim * num_images
            else:
                self.rgb_encoder = FMRgbEncoder(config)
                global_cond_dim += self.rgb_encoder.feature_dim * num_images
        if self.config.env_state_feature:
            global_cond_dim += self.config.env_state_feature.shape[0]

        # Velocity prediction network: 1D UNet conditioned on (time_embed, global_cond).
        self.velocity_net = FMConditionalUnet1d(
            config, global_cond_dim=global_cond_dim * config.n_obs_steps
        )

    # =========================================================================
    # Inference
    # =========================================================================

    def generate_actions(
        self,
        batch: dict[str, Tensor],
        noise: Tensor | None = None,
        rtc_processor: RTCProcessor | None = None,
        inference_delay: int | None = None,
        prev_chunk_left_over: Tensor | None = None,
        execution_horizon: int | None = None,
    ) -> Tensor:
        """Generate actions using Flow Matching sampling.

        Args:
            batch: Observation batch with state and optional images.
            noise: Optional initial noise. Sampled from N(0,I) if None.
            rtc_processor: Optional RTC processor for real-time chunking guidance.
            inference_delay: RTC inference delay parameter.
            prev_chunk_left_over: RTC previous chunk leftover.
            execution_horizon: RTC execution horizon.

        Returns:
            Action tensor of shape (B, n_action_steps, action_dim).
        """
        batch_size, n_obs_steps = batch[OBS_STATE].shape[:2]
        assert n_obs_steps == self.config.n_obs_steps

        # Encode observations.
        global_cond = self._prepare_global_conditioning(batch)

        # Sample actions.
        actions = self._sample_flow(
            batch_size,
            global_cond=global_cond,
            noise=noise,
            rtc_processor=rtc_processor,
            inference_delay=inference_delay,
            prev_chunk_left_over=prev_chunk_left_over,
            execution_horizon=execution_horizon,
        )

        # Extract n_action_steps from the current observation onward.
        start = n_obs_steps - 1
        end = start + self.config.n_action_steps
        actions = actions[:, start:end]

        # Clip actions to prevent physics simulation instabilities
        if self.config.clip_sample:
            actions = actions.clamp(-self.config.clip_sample_range, self.config.clip_sample_range)

        return actions

    def _sample_flow(
        self,
        batch_size: int,
        global_cond: Tensor | None = None,
        noise: Tensor | None = None,
        rtc_processor: RTCProcessor | None = None,
        inference_delay: int | None = None,
        prev_chunk_left_over: Tensor | None = None,
        execution_horizon: int | None = None,
    ) -> Tensor:
        """Sample actions by integrating the velocity field from t=1 (noise) to t=0 (action).

        Supports both Euler and adaptive ODE (dopri5) solvers. When RTC is enabled,
        only Euler is used (RTC requires per-step access to v_t).
        """
        device = get_device_from_parameters(self)
        dtype = get_dtype_from_parameters(self)
        action_dim = self.config.action_feature.shape[0]

        # Initial noise at t=1.
        x_t = (
            noise
            if noise is not None
            else torch.randn(
                size=(batch_size, self.config.horizon, action_dim),
                dtype=dtype,
                device=device,
            )
        )

        use_rtc = rtc_processor is not None
        use_ode = self.config.solver_type == "dopri5" and not use_rtc

        if use_ode:
            x_0 = self._sample_ode(x_t, global_cond)
        else:
            x_0 = self._sample_euler(
                x_t,
                global_cond,
                rtc_processor=rtc_processor,
                inference_delay=inference_delay,
                prev_chunk_left_over=prev_chunk_left_over,
                execution_horizon=execution_horizon,
            )

        return x_0

    def _sample_euler(
        self,
        x_t: Tensor,
        global_cond: Tensor | None,
        num_steps: int | None = None,
        rtc_processor: RTCProcessor | None = None,
        inference_delay: int | None = None,
        prev_chunk_left_over: Tensor | None = None,
        execution_horizon: int | None = None,
    ) -> Tensor:
        """Sample using Euler method, integrating from t=1 to t=0.

        Compatible with RTC guidance (PI05-style direction).
        """
        if num_steps is None:
            num_steps = self.config.num_inference_steps

        dt = -1.0 / num_steps  # Negative because t goes from 1 → 0.

        for step in range(num_steps):
            time = 1.0 + step * dt  # t ∈ {1.0, 1-dt, ..., dt}
            time_tensor = torch.full(
                (x_t.shape[0],), time, dtype=x_t.dtype, device=x_t.device
            )

            def denoise_step_partial(input_x_t, current_time=time_tensor):
                return self.velocity_net(input_x_t, current_time, global_cond=global_cond)

            if rtc_processor is not None:
                v_t = rtc_processor.denoise_step(
                    x_t=x_t,
                    prev_chunk_left_over=prev_chunk_left_over,
                    inference_delay=inference_delay,
                    time=time,
                    original_denoise_step_partial=denoise_step_partial,
                    execution_horizon=execution_horizon,
                )
            else:
                v_t = denoise_step_partial(x_t)

            x_t = x_t + dt * v_t

            if rtc_processor is not None and rtc_processor.is_debug_enabled():
                rtc_processor.track(time=time, x_t=x_t, v_t=v_t)

        return x_t

    def _sample_ode(self, x_t: Tensor, global_cond: Tensor | None) -> Tensor:
        """Sample using adaptive ODE solver (dopri5), integrating from t=1 to t=0.

        More accurate than Euler but slower. Not compatible with RTC (no per-step access).
        """
        from torchdiffeq import odeint

        device = x_t.device

        class ODEFunc(nn.Module):
            def __init__(self, velocity_net, global_cond):
                super().__init__()
                self.velocity_net = velocity_net
                self.global_cond = global_cond

            def forward(self, t, x):
                # t is scalar; expand to batch.
                time_tensor = t.expand(x.shape[0])
                return self.velocity_net(x, time_tensor, global_cond=self.global_cond)

        ode_func = ODEFunc(self.velocity_net, global_cond)

        # Integrate from t=1 to t=0.
        t_span = torch.tensor([1.0, 0.0], device=device)

        solution = odeint(
            ode_func,
            x_t,
            t_span,
            method="dopri5",
            atol=self.config.ode_atol,
            rtol=self.config.ode_rtol,
        )

        # solution[-1] is the state at t=0 (i.e., the action).
        return solution[-1]

    # =========================================================================
    # Training
    # =========================================================================

    def compute_loss(self, batch: dict[str, Tensor], reduction: str = "mean") -> Tensor:
        """Compute Flow Matching loss.

        Interpolation (PI05-consistent):
            x_t = t * noise + (1 - t) * action
            u_t = noise - action    (target velocity)

        Loss:
            MSE(v_pred, u_t)

        Args:
            batch: Input batch containing observations and actions.
            reduction: Loss reduction method. "mean" for average, "none" for per-sample loss.
        """
        assert set(batch).issuperset({OBS_STATE, ACTION, "action_is_pad"})
        assert OBS_IMAGES in batch or OBS_ENV_STATE in batch
        n_obs_steps = batch[OBS_STATE].shape[1]
        horizon = batch[ACTION].shape[1]
        assert horizon == self.config.horizon
        assert n_obs_steps == self.config.n_obs_steps

        # Encode observations.
        global_cond = self._prepare_global_conditioning(batch)

        # Target action trajectory.
        trajectory = batch[ACTION]  # (B, horizon, action_dim)
        batch_size = trajectory.shape[0]

        # Sample noise and time.
        noise = torch.randn_like(trajectory)
        epsilon = 1e-5
        t = torch.rand(batch_size, device=trajectory.device) * (1.0 - epsilon) + epsilon  # t ∈ [ε, 1]

        # Flow Matching interpolation (PI05-consistent direction).
        t_expanded = t[:, None, None]  # (B, 1, 1)
        x_t = t_expanded * noise + (1.0 - t_expanded) * trajectory
        u_t = noise - trajectory  # Target velocity.

        # Predict velocity.
        v_pred = self.velocity_net(x_t, t, global_cond=global_cond)

        # Compute loss.
        loss = F.mse_loss(v_pred, u_t, reduction="none")

        # Mask loss for padded actions.
        if self.config.do_mask_loss_for_padding:
            if "action_is_pad" not in batch:
                raise ValueError(
                    "You need to provide 'action_is_pad' in the batch when "
                    f"{self.config.do_mask_loss_for_padding=}."
                )
            in_episode_bound = ~batch["action_is_pad"]
            loss = loss * in_episode_bound.unsqueeze(-1)

        if reduction == "mean":
            return loss.mean()
        elif reduction == "none":
            # Return per-sample loss (B,) by averaging over action dimensions
            return loss.mean(dim=(1, 2))
        else:
            raise ValueError(f"Unsupported reduction: {reduction}")

    # =========================================================================
    # Observation encoding (same as Diffusion Policy)
    # =========================================================================

    def _prepare_global_conditioning(self, batch: dict[str, Tensor]) -> Tensor:
        """Encode image features and concatenate them with the state vector."""
        batch_size, n_obs_steps = batch[OBS_STATE].shape[:2]
        global_cond_feats = [batch[OBS_STATE]]

        if self.config.image_features:
            if self.config.use_separate_rgb_encoder_per_camera:
                images_per_camera = einops.rearrange(batch[OBS_IMAGES], "b s n ... -> n (b s) ...")
                img_features_list = torch.cat(
                    [
                        encoder(images)
                        for encoder, images in zip(self.rgb_encoder, images_per_camera, strict=True)
                    ]
                )
                img_features = einops.rearrange(
                    img_features_list, "(n b s) ... -> b s (n ...)", b=batch_size, s=n_obs_steps
                )
            else:
                img_features = self.rgb_encoder(
                    einops.rearrange(batch[OBS_IMAGES], "b s n ... -> (b s n) ...")
                )
                img_features = einops.rearrange(
                    img_features, "(b s n) ... -> b s (n ...)", b=batch_size, s=n_obs_steps
                )
            global_cond_feats.append(img_features)

        if self.config.env_state_feature:
            global_cond_feats.append(batch[OBS_ENV_STATE])

        return torch.cat(global_cond_feats, dim=-1).flatten(start_dim=1)


# ==============================================================================
# Vision Encoder (shared with Diffusion Policy pattern)
# ==============================================================================


class SpatialSoftmax(nn.Module):
    """Spatial Soft Argmax operation for extracting image keypoints.

    Same as Diffusion Policy implementation.
    """

    def __init__(self, input_shape, num_kp=None):
        super().__init__()
        assert len(input_shape) == 3
        self._in_c, self._in_h, self._in_w = input_shape

        if num_kp is not None:
            self.nets = torch.nn.Conv2d(self._in_c, num_kp, kernel_size=1)
            self._out_c = num_kp
        else:
            self.nets = None
            self._out_c = self._in_c

        pos_x, pos_y = np.meshgrid(np.linspace(-1.0, 1.0, self._in_w), np.linspace(-1.0, 1.0, self._in_h))
        pos_x = torch.from_numpy(pos_x.reshape(self._in_h * self._in_w, 1)).float()
        pos_y = torch.from_numpy(pos_y.reshape(self._in_h * self._in_w, 1)).float()
        self.register_buffer("pos_grid", torch.cat([pos_x, pos_y], dim=1))

    def forward(self, features: Tensor) -> Tensor:
        if self.nets is not None:
            features = self.nets(features)
        features = features.reshape(-1, self._in_h * self._in_w)
        attention = F.softmax(features, dim=-1)
        expected_xy = attention @ self.pos_grid
        feature_keypoints = expected_xy.view(-1, self._out_c, 2)
        return feature_keypoints


class FMRgbEncoder(nn.Module):
    """Encodes an RGB image into a 1D feature vector using ResNet + SpatialSoftmax.

    Same architecture as DiffusionRgbEncoder.
    """

    def __init__(self, config: FlowMatchingConfig):
        super().__init__()
        if config.crop_shape is not None:
            self.do_crop = True
            self.center_crop = torchvision.transforms.CenterCrop(config.crop_shape)
            if config.crop_is_random:
                self.maybe_random_crop = torchvision.transforms.RandomCrop(config.crop_shape)
            else:
                self.maybe_random_crop = self.center_crop
        else:
            self.do_crop = False

        backbone_model = getattr(torchvision.models, config.vision_backbone)(
            weights=config.pretrained_backbone_weights
        )
        self.backbone = nn.Sequential(*(list(backbone_model.children())[:-2]))
        if config.use_group_norm:
            if config.pretrained_backbone_weights:
                raise ValueError(
                    "You can't replace BatchNorm in a pretrained model without ruining the weights!"
                )
            self.backbone = _replace_submodules(
                root_module=self.backbone,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(num_groups=x.num_features // 16, num_channels=x.num_features),
            )

        images_shape = next(iter(config.image_features.values())).shape
        dummy_shape_h_w = config.crop_shape if config.crop_shape is not None else images_shape[1:]
        dummy_shape = (1, images_shape[0], *dummy_shape_h_w)
        feature_map_shape = get_output_shape(self.backbone, dummy_shape)[1:]

        self.pool = SpatialSoftmax(feature_map_shape, num_kp=config.spatial_softmax_num_keypoints)
        self.feature_dim = config.spatial_softmax_num_keypoints * 2
        self.out = nn.Linear(config.spatial_softmax_num_keypoints * 2, self.feature_dim)
        self.relu = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        if self.do_crop:
            if self.training:
                x = self.maybe_random_crop(x)
            else:
                x = self.center_crop(x)
        x = torch.flatten(self.pool(self.backbone(x)), start_dim=1)
        x = self.relu(self.out(x))
        return x


def _replace_submodules(
    root_module: nn.Module, predicate: Callable[[nn.Module], bool], func: Callable[[nn.Module], nn.Module]
) -> nn.Module:
    """Replace submodules matching a predicate with a replacement function."""
    if predicate(root_module):
        return func(root_module)
    replace_list = [k.split(".") for k, m in root_module.named_modules(remove_duplicate=True) if predicate(m)]
    for *parents, k in replace_list:
        parent_module = root_module
        if len(parents) > 0:
            parent_module = root_module.get_submodule(".".join(parents))
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        tgt_module = func(src_module)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)
    assert not any(predicate(m) for _, m in root_module.named_modules(remove_duplicate=True))
    return root_module


# ==============================================================================
# Velocity Prediction Network (1D UNet, same architecture as Diffusion Policy)
# ==============================================================================


class FMSinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embedding for continuous time t ∈ [0, 1]."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        # x is float in [0, 1], scale to reasonable range for sinusoidal encoding.
        emb = x.unsqueeze(-1) * emb.unsqueeze(0) * 1000.0
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class FMConv1dBlock(nn.Module):
    """Conv1d → GroupNorm → Mish"""

    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


class FMConditionalResidualBlock1d(nn.Module):
    """ResNet-style 1D convolutional block with FiLM modulation for conditioning."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_dim: int,
        kernel_size: int = 3,
        n_groups: int = 8,
        use_film_scale_modulation: bool = False,
    ):
        super().__init__()
        self.use_film_scale_modulation = use_film_scale_modulation
        self.out_channels = out_channels

        self.conv1 = FMConv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups)

        cond_channels = out_channels * 2 if use_film_scale_modulation else out_channels
        self.cond_encoder = nn.Sequential(nn.Mish(), nn.Linear(cond_dim, cond_channels))

        self.conv2 = FMConv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups)

        self.residual_conv = (
            nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        out = self.conv1(x)

        cond_embed = self.cond_encoder(cond).unsqueeze(-1)
        if self.use_film_scale_modulation:
            scale = cond_embed[:, : self.out_channels]
            bias = cond_embed[:, self.out_channels :]
            out = scale * out + bias
        else:
            out = out + cond_embed

        out = self.conv2(out)
        out = out + self.residual_conv(x)
        return out


class FMConditionalUnet1d(nn.Module):
    """1D convolutional UNet for velocity field prediction.

    Predicts v(x_t, t, cond) — the velocity field at time t given noisy trajectory x_t
    and global conditioning (state + image features).

    Architecture is the same as DiffusionConditionalUnet1d but the time input is
    continuous t ∈ [0, 1] rather than discrete timestep indices.
    """

    def __init__(self, config: FlowMatchingConfig, global_cond_dim: int):
        super().__init__()
        self.config = config

        # Time embedding: sinusoidal + MLP.
        self.time_encoder = nn.Sequential(
            FMSinusoidalPosEmb(config.time_embed_dim),
            nn.Linear(config.time_embed_dim, config.time_embed_dim * 4),
            nn.Mish(),
            nn.Linear(config.time_embed_dim * 4, config.time_embed_dim),
        )

        # FiLM conditioning dimension.
        cond_dim = config.time_embed_dim + global_cond_dim

        # Channel configuration for UNet.
        in_out = [(config.action_feature.shape[0], config.down_dims[0])] + list(
            zip(config.down_dims[:-1], config.down_dims[1:], strict=True)
        )

        # UNet encoder.
        common_res_block_kwargs = {
            "cond_dim": cond_dim,
            "kernel_size": config.kernel_size,
            "n_groups": config.n_groups,
            "use_film_scale_modulation": config.use_film_scale_modulation,
        }
        self.down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            self.down_modules.append(
                nn.ModuleList(
                    [
                        FMConditionalResidualBlock1d(dim_in, dim_out, **common_res_block_kwargs),
                        FMConditionalResidualBlock1d(dim_out, dim_out, **common_res_block_kwargs),
                        nn.Conv1d(dim_out, dim_out, 3, 2, 1) if not is_last else nn.Identity(),
                    ]
                )
            )

        # Middle blocks.
        self.mid_modules = nn.ModuleList(
            [
                FMConditionalResidualBlock1d(
                    config.down_dims[-1], config.down_dims[-1], **common_res_block_kwargs
                ),
                FMConditionalResidualBlock1d(
                    config.down_dims[-1], config.down_dims[-1], **common_res_block_kwargs
                ),
            ]
        )

        # UNet decoder.
        self.up_modules = nn.ModuleList([])
        for ind, (dim_out, dim_in) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            self.up_modules.append(
                nn.ModuleList(
                    [
                        FMConditionalResidualBlock1d(dim_in * 2, dim_out, **common_res_block_kwargs),
                        FMConditionalResidualBlock1d(dim_out, dim_out, **common_res_block_kwargs),
                        nn.ConvTranspose1d(dim_out, dim_out, 4, 2, 1) if not is_last else nn.Identity(),
                    ]
                )
            )

        self.final_conv = nn.Sequential(
            FMConv1dBlock(config.down_dims[0], config.down_dims[0], kernel_size=config.kernel_size),
            nn.Conv1d(config.down_dims[0], config.action_feature.shape[0], 1),
        )

    def forward(self, x: Tensor, time: Tensor, global_cond: Tensor | None = None) -> Tensor:
        """
        Args:
            x: (B, T, action_dim) noisy trajectory x_t.
            time: (B,) continuous time values t ∈ [0, 1].
            global_cond: (B, global_cond_dim) observation conditioning.

        Returns:
            (B, T, action_dim) predicted velocity v(x_t, t, cond).
        """
        # For 1D convolutions: (B, T, D) → (B, D, T).
        x = einops.rearrange(x, "b t d -> b d t")

        time_embed = self.time_encoder(time)

        if global_cond is not None:
            global_feature = torch.cat([time_embed, global_cond], dim=-1)
        else:
            global_feature = time_embed

        # Encoder.
        encoder_skip_features: list[Tensor] = []
        for resnet, resnet2, downsample in self.down_modules:
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            encoder_skip_features.append(x)
            x = downsample(x)

        # Middle.
        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        # Decoder.
        for resnet, resnet2, upsample in self.up_modules:
            x = torch.cat((x, encoder_skip_features.pop()), dim=1)
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            x = upsample(x)

        x = self.final_conv(x)

        # (B, D, T) → (B, T, D).
        x = einops.rearrange(x, "b d t -> b t d")
        return x


# ==============================================================================
# Test (run with: python -m lerobot.policies.fm.modeling_fm)
# ==============================================================================

if __name__ == "__main__":
    from lerobot.configs.types import FeatureType, PolicyFeature

    from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE

    print("=" * 60)
    print("Flow Matching Policy - Model build and forward test")
    print("=" * 60)

    batch_size = 4
    n_obs_steps = 2
    horizon = 16
    n_action_steps = 8
    state_dim = 2
    action_dim = 2

    # Test 1: Config with env_state only (no images)
    print("\n[Test 1] Config with env_state only...")
    input_features = {
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(state_dim,)),
        OBS_ENV_STATE: PolicyFeature(type=FeatureType.ENV, shape=(4,)),
    }
    output_features = {
        ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(action_dim,)),
    }
    config = FlowMatchingConfig(
        input_features=input_features,
        output_features=output_features,
        n_obs_steps=n_obs_steps,
        horizon=horizon,
        n_action_steps=n_action_steps,
    )
    config.validate_features()

    policy = FlowMatchingPolicy(config=config)
    policy.eval()

    # Training forward
    batch_train = {
        OBS_STATE: torch.randn(batch_size, n_obs_steps, state_dim),
        OBS_ENV_STATE: torch.randn(batch_size, n_obs_steps, 4),
        ACTION: torch.randn(batch_size, horizon, action_dim),
        "action_is_pad": torch.zeros(batch_size, horizon, dtype=torch.bool),
    }
    loss, _ = policy(batch_train)
    assert loss.dim() == 0 and loss.item() >= 0, "Training loss should be scalar and non-negative"
    print(f"  Training forward OK. Loss: {loss.item():.4f}")

    # Inference: predict_action_chunk (need to populate queues first)
    policy.reset()
    for _ in range(n_obs_steps):
        policy._queues[OBS_STATE].append(batch_train[OBS_STATE][:, -1])
        policy._queues[OBS_ENV_STATE].append(batch_train[OBS_ENV_STATE][:, -1])
    batch_infer = {OBS_STATE: batch_train[OBS_STATE], OBS_ENV_STATE: batch_train[OBS_ENV_STATE]}
    actions = policy.predict_action_chunk(batch_infer)
    assert actions.shape == (batch_size, n_action_steps, action_dim), (
        f"Expected shape ({batch_size}, {n_action_steps}, {action_dim}), got {actions.shape}"
    )
    print(f"  predict_action_chunk OK. Output shape: {actions.shape}")

    # select_action (single step)
    policy.reset()
    single_obs = {
        OBS_STATE: batch_train[OBS_STATE][:, -1],
        OBS_ENV_STATE: batch_train[OBS_ENV_STATE][:, -1],
    }
    policy._queues = populate_queues(policy._queues, single_obs)
    action = policy.select_action(single_obs)
    assert action.shape == (batch_size, action_dim), f"Expected ({batch_size}, {action_dim}), got {action.shape}"
    print(f"  select_action OK. Output shape: {action.shape}")

    print("\n[Test 1] PASSED")

    # Test 2: Config with images (full pipeline)
    print("\n[Test 2] Config with images (full pipeline)...")
    input_features_img = {
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(state_dim,)),
        "observation.image.top": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 96, 96)),
    }
    config_img = FlowMatchingConfig(
        input_features=input_features_img,
        output_features=output_features,
        n_obs_steps=n_obs_steps,
        horizon=horizon,
        n_action_steps=n_action_steps,
    )
    config_img.validate_features()

    policy_img = FlowMatchingPolicy(config=config_img)
    policy_img.eval()

    batch_train_img = {
        OBS_STATE: torch.randn(batch_size, n_obs_steps, state_dim),
        "observation.image.top": torch.rand(batch_size, n_obs_steps, 3, 96, 96),
        ACTION: torch.randn(batch_size, horizon, action_dim),
        "action_is_pad": torch.zeros(batch_size, horizon, dtype=torch.bool),
    }
    loss_img, _ = policy_img(batch_train_img)
    assert loss_img.dim() == 0 and loss_img.item() >= 0
    print(f"  Training forward (with images) OK. Loss: {loss_img.item():.4f}")

    policy_img.reset()
    for i in range(n_obs_steps):
        policy_img._queues[OBS_STATE].append(batch_train_img[OBS_STATE][:, i])
        policy_img._queues[OBS_IMAGES].append(
            batch_train_img["observation.image.top"][:, i].unsqueeze(1)
        )
    batch_infer_img = {
        OBS_STATE: batch_train_img[OBS_STATE],
        OBS_IMAGES: batch_train_img["observation.image.top"].unsqueeze(2),
    }
    actions_img = policy_img.predict_action_chunk(batch_infer_img)
    assert actions_img.shape == (batch_size, n_action_steps, action_dim)
    print(f"  predict_action_chunk (with images) OK. Output shape: {actions_img.shape}")

    print("\n[Test 2] PASSED")

    # Test 3: Euler vs ODE solver (env_state only for speed)
    print("\n[Test 3] Euler vs ODE solver...")
    config_euler = FlowMatchingConfig(
        input_features=input_features,
        output_features=output_features,
        n_obs_steps=n_obs_steps,
        horizon=horizon,
        n_action_steps=n_action_steps,
        num_inference_steps=10,
        solver_type="euler",
    )
    config_euler.validate_features()
    policy_euler = FlowMatchingPolicy(config=config_euler)
    policy_euler.eval()

    model_euler = policy_euler.flow_matching
    global_cond = torch.cat(
        [batch_train[OBS_STATE].flatten(1), batch_train[OBS_ENV_STATE].flatten(1)], dim=1
    )
    noise = torch.randn(batch_size, horizon, action_dim)

    out_euler = model_euler._sample_euler(noise.clone(), global_cond, num_steps=5)
    assert out_euler.shape == noise.shape
    print(f"  Euler solver OK. Output shape: {out_euler.shape}")

    config_ode = FlowMatchingConfig(
        input_features=input_features,
        output_features=output_features,
        n_obs_steps=n_obs_steps,
        horizon=horizon,
        n_action_steps=n_action_steps,
        solver_type="dopri5",
    )
    config_ode.validate_features()
    policy_ode = FlowMatchingPolicy(config=config_ode)
    policy_ode.eval()

    out_ode = policy_ode.flow_matching._sample_ode(noise.clone(), global_cond)
    assert out_ode.shape == noise.shape
    print(f"  ODE solver (dopri5) OK. Output shape: {out_ode.shape}")

    print("\n[Test 3] PASSED")

    print("\n" + "=" * 60)
    print("All tests passed. Model build and forward are correct.")
    print("=" * 60)

