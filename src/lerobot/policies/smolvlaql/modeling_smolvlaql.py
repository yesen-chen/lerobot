#!/usr/bin/env python

# Copyright 2025 HuggingFace Inc. team. All rights reserved.
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

"""
SmolVLA-QL: SmolVLA with QC-FQL (Chunked Flow Q-Learning)

Extends SmolVLA with a "Shared Expert, Dual Head" architecture:
- action_out_proj: velocity field prediction (flow matching, existing)
- value_heads: Q-value prediction (QC-FQL critic, new)

Architecture:
    VLM → prefix KV cache
    ActionExpert(input_actions, cross-attn to KV cache) → hidden states
      ├── action_out_proj → velocity field (flow matching)
      └── value_heads → pool → scalar Q-value (detached from expert)

Gradient flow:
- Critic training: only value_heads updated (expert hidden states detached)
- Policy training: expert backbone + action_out_proj updated (standard flow matching)
- Target value heads: EMA copy, never directly trained
"""

import copy
import math
from collections import deque
from typing import TypedDict

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn
from typing_extensions import Unpack

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.rtc.modeling_rtc import RTCProcessor
from lerobot.policies.smolvlaql.configuration_smolvlaql import SmolVLAQLConfig
from lerobot.policies.smolvlaql.smolvlmql_with_expert import SmolVLMWithExpertModel
from lerobot.policies.utils import (
    populate_queues,
)
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE
from lerobot.utils.utils import get_safe_dtype


class ActionSelectKwargs(TypedDict, total=False):
    inference_delay: int | None
    prev_chunk_left_over: Tensor | None
    execution_horizon: int | None
    use_best_of_n: bool | None


def create_sinusoidal_pos_embedding(
    time: torch.tensor, dimension: int, min_period: float, max_period: float, device="cpu"
) -> Tensor:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    dtype = get_safe_dtype(torch.float64, device.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    pos_emb = torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)
    return pos_emb


def make_att_2d_masks(pad_masks, att_masks):
    """Copied from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` int[B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: int32[B, N] mask that's 1 where previous tokens cannot depend on
        it and 0 where it shares the same attention mask as the previous token.
    """
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    att_2d_masks = att_2d_masks & pad_2d_masks
    return att_2d_masks


def resize_with_pad(img, width, height, pad_value=-1):
    if img.ndim != 4:
        raise ValueError(f"(b,c,h,w) expected, but {img.shape}")

    cur_height, cur_width = img.shape[2:]

    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    resized_img = F.interpolate(
        img, size=(resized_height, resized_width), mode="bilinear", align_corners=False
    )

    pad_height = max(0, int(height - resized_height))
    pad_width = max(0, int(width - resized_width))

    padded_img = F.pad(resized_img, (pad_width, 0, pad_height, 0), value=pad_value)
    return padded_img


def pad_vector(vector, new_dim):
    """Can be (batch_size x sequence_length x features_dimension)
    or (batch_size x features_dimension)
    """
    if vector.shape[-1] == new_dim:
        return vector
    shape = list(vector.shape)
    current_dim = shape[-1]
    shape[-1] = new_dim
    new_vector = torch.zeros(*shape, dtype=vector.dtype, device=vector.device)
    new_vector[..., :current_dim] = vector
    return new_vector


def normalize(x, min_val, max_val):
    return (x - min_val) / (max_val - min_val)


def unnormalize(x, min_val, max_val):
    return x * (max_val - min_val) + min_val


def safe_arcsin(value):
    return torch.arcsin(torch.clamp(value, -1.0, 1.0))


def aloha_gripper_to_angular(value):
    value = unnormalize(value, min_val=0.01844, max_val=0.05800)

    def linear_to_radian(linear_position, arm_length, horn_radius):
        value = (horn_radius**2 + linear_position**2 - arm_length**2) / (2 * horn_radius * linear_position)
        return safe_arcsin(value)

    value = linear_to_radian(value, arm_length=0.036, horn_radius=0.022)
    return normalize(value, min_val=0.4, max_val=1.5)


def aloha_gripper_from_angular(value):
    value = unnormalize(value, min_val=0.4, max_val=1.5)
    return normalize(value, min_val=-0.6213, max_val=1.4910)


def aloha_gripper_from_angular_inv(value):
    value = unnormalize(value, min_val=-0.6213, max_val=1.4910)
    return normalize(value, min_val=0.4, max_val=1.5)


def pad_tensor(tensor, max_len, pad_value=0):
    """Pads a tensor along sequence dimension to match max_len."""
    b, d = tensor.shape[:2]
    padded_tensor = torch.full(
        (b, max_len, *tensor.shape[2:]), pad_value, dtype=tensor.dtype, device=tensor.device
    )
    padded_tensor[:, :d] = tensor
    return padded_tensor


# ==================== QC-FQL Components ====================


class ValueHead(nn.Module):
    """MLP that maps pooled expert hidden states to a scalar Q-value.

    Architecture: mean-pool over chunk tokens → MLP(LayerNorm + SiLU) → scalar.
    """

    def __init__(self, input_dim: int, hidden_dims: tuple[int, ...] = (512, 512)):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.LayerNorm(dim),
                nn.SiLU(),
            ])
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, expert_hidden_states: Tensor) -> Tensor:
        """
        Args:
            expert_hidden_states: (B, chunk_size, expert_hidden_size)
        Returns:
            (B,) scalar Q-values
        """
        pooled = expert_hidden_states.mean(dim=1)  # (B, expert_hidden_size)
        return self.mlp(pooled).squeeze(-1)  # (B,)


class OnestepActionHead(nn.Module):
    """Per-timestep MLP: noise_hidden -> action.

    Takes expert hidden states from processing noise at t=1 (which contain full
    observation context via cross-attention to prefix), and predicts clean actions
    in a single forward pass. The MLP is shared across all chunk timesteps.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: tuple[int, ...] = (512, 512, 512, 512),
    ):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.LayerNorm(dim),
                nn.SiLU(),
            ])
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, noise_hidden: Tensor) -> Tensor:
        """
        Args:
            noise_hidden: (B, chunk_size, input_dim) — expert hidden from noise at t=1.
        Returns:
            (B, chunk_size, output_dim) — predicted action chunk.
        """
        return self.mlp(noise_hidden)


# ==================== Policy ====================


class SmolVLAQLPolicy(PreTrainedPolicy):
    """SmolVLA policy extended with QC-FQL critic for value-guided action selection.

    Dual-head architecture sharing the Action Expert backbone:
    - Flow matching head (action_out_proj): trained with standard flow matching loss
    - Value heads (value_heads): trained with TD loss on detached expert hidden states
    """

    config_class = SmolVLAQLConfig
    name = "smolvlaql"

    def __init__(
        self,
        config: SmolVLAQLConfig,
        **kwargs,
    ):
        super().__init__(config)
        config.validate_features()
        self.config = config
        self.init_rtc_processor()
        self.model = VLAFlowMatching(config, rtc_processor=self.rtc_processor)
        self.reset()

    def reset(self):
        """This should be called whenever the environment is reset."""
        self._queues = {
            ACTION: deque(maxlen=self.config.n_action_steps),
        }

    def init_rtc_processor(self):
        """Initialize RTC processor if RTC is enabled in config."""
        self.rtc_processor = None
        if self.config.rtc_config is not None:
            self.rtc_processor = RTCProcessor(self.config.rtc_config)
            model_value = getattr(self, "model", None)
            if model_value is not None:
                model_value.rtc_processor = self.rtc_processor

    def get_optim_params(self) -> dict:
        """Return separate parameter groups for policy, onestep_head, and critic optimizers.

        Splitting onestep_head from the backbone prevents alpha-amplified distill
        gradients from dominating gradient clipping of the VLM/Expert backbone.

        Returns:
            dict with keys:
                "backbone": VLM + Expert + action projections + state_proj
                "onestep":  onestep_head only (receives distill_loss + q_loss gradients)
                "critic":   value_heads (excluding frozen target_value_heads)
        """
        backbone_params = []
        onestep_params = []
        critic_params = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if "value_heads" in name and "target_value_heads" not in name:
                critic_params.append(param)
            elif "target_value_heads" in name:
                continue
            elif "onestep_head" in name:
                onestep_params.append(param)
            else:
                backbone_params.append(param)
        return {
            "backbone": backbone_params,
            "onestep": onestep_params,
            "critic": critic_params,
        }

    def _get_action_chunk(
        self, batch: dict[str, Tensor], noise: Tensor | None = None, **kwargs: Unpack[ActionSelectKwargs]
    ) -> Tensor:
        for k in batch:
            if k in self._queues and k != ACTION:
                batch[k] = torch.stack(list(self._queues[k]), dim=1)

        images, img_masks = self.prepare_images(batch)
        state = self.prepare_state(batch)
        lang_tokens = batch[f"{OBS_LANGUAGE_TOKENS}"]
        lang_masks = batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]

        use_best_of_n = kwargs.pop("use_best_of_n", False)

        eval_mode = getattr(self.config, "qc_eval_mode", "onestep")
        if self.config.qc_actor_type == "distill-ddpg" and eval_mode == "onestep":
            actions = self.model.sample_actions_onestep(
                images, img_masks, lang_tokens, lang_masks, state, noise=noise
            )
        elif use_best_of_n:
            actions = self.model.sample_actions_best_of_n(
                images, img_masks, lang_tokens, lang_masks, state
            )
        else:
            actions = self.model.sample_actions(
                images, img_masks, lang_tokens, lang_masks, state, noise=noise, **kwargs
            )

        original_action_dim = self.config.action_feature.shape[0]
        actions = actions[:, :, :original_action_dim]

        if self.config.adapt_to_pi_aloha:
            actions = self._pi_aloha_encode_actions(actions)

        return actions

    def _prepare_batch(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        if self.config.adapt_to_pi_aloha:
            batch[OBS_STATE] = self._pi_aloha_decode_state(batch[OBS_STATE])
        return batch

    @torch.no_grad()
    def predict_action_chunk(
        self, batch: dict[str, Tensor], noise: Tensor | None = None, **kwargs: Unpack[ActionSelectKwargs]
    ) -> Tensor:
        self.eval()
        batch = self._prepare_batch(batch)
        self._queues = populate_queues(self._queues, batch, exclude_keys=[ACTION])
        actions = self._get_action_chunk(batch, noise, **kwargs)
        return actions

    @torch.no_grad()
    def select_action(
        self, batch: dict[str, Tensor], noise: Tensor | None = None, **kwargs: Unpack[ActionSelectKwargs]
    ) -> Tensor:
        """Select a single action given environment observations."""
        assert not self._rtc_enabled(), (
            "RTC is not supported for select_action, use it with predict_action_chunk"
        )

        self.eval()
        batch = self._prepare_batch(batch)
        self._queues = populate_queues(self._queues, batch, exclude_keys=[ACTION])

        if self._check_get_actions_condition():
            actions = self._get_action_chunk(batch, noise)
            self._queues[ACTION].extend(actions.transpose(0, 1)[: self.config.n_action_steps])

        return self._queues[ACTION].popleft()

    def _check_get_actions_condition(self) -> bool:
        return len(self._queues[ACTION]) == 0

    def _rtc_enabled(self) -> bool:
        return self.config.rtc_config is not None and self.config.rtc_config.enabled

    def forward(
        self, batch: dict[str, Tensor], noise=None, time=None, reduction: str = "mean",
    ) -> dict[str, Tensor]:
        """Flow matching training forward (actor BC loss).

        关于 action padding 的处理：
        LeRobot 的 _get_query_indices 已将越界 action index clamp 到 episode 内，
        并标记 action_is_pad=True。此处用 action_is_pad 屏蔽 padded action 的 loss，
        无需额外的 valid mask（ACFQL 的 valid 是为了处理跨 episode 采样，
        LeRobot 不存在该问题）。
        """
        if self.config.adapt_to_pi_aloha:
            batch[OBS_STATE] = self._pi_aloha_decode_state(batch[OBS_STATE])
            batch[ACTION] = self._pi_aloha_encode_actions_inv(batch[ACTION])

        images, img_masks = self.prepare_images(batch)
        state = self.prepare_state(batch)
        lang_tokens = batch[f"{OBS_LANGUAGE_TOKENS}"]
        lang_masks = batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]
        actions = self.prepare_action(batch)
        actions_is_pad = batch["action_is_pad"]
        loss_dict = {}
        losses, _fwd_info = self.model.forward(images, img_masks, lang_tokens, lang_masks, state, actions, noise, time)
        loss_dict["losses_after_forward"] = losses.clone().mean().item()

        if actions_is_pad is not None:
            in_episode_bound = ~actions_is_pad
            losses = losses * in_episode_bound.unsqueeze(-1)
            loss_dict["losses_after_in_ep_bound"] = losses.clone().mean().item()

        #actual_action_dim = self.config.action_feature.shape[0]
        losses = losses[:, :, :self.config.max_action_dim]
        loss_dict["losses_after_rm_padding"] = losses.clone().mean().item()

        if reduction == "none":
            per_sample_loss = losses.mean(dim=(1, 2))
            loss_dict["loss"] = per_sample_loss.mean().item()
            return per_sample_loss, loss_dict
        else:
            loss = losses.mean()
            loss_dict["loss"] = loss.item()
            return loss, loss_dict

    def forward_actor(
        self, batch: dict[str, Tensor],
    ) -> tuple[Tensor, dict]:
        """Combined actor loss for distill-ddpg mode.

        actor_loss = bc_flow_loss + alpha * distill_loss + q_loss

        When qc_actor_type != "distill-ddpg", falls back to standard forward().

        Returns:
            total_loss: scalar
            loss_dict: diagnostic info
        """
        if self.config.qc_actor_type != "distill-ddpg":
            return self.forward(batch)

        # --- 1) BC flow loss (reuse existing forward, which also gives us prefix_out) ---
        if self.config.adapt_to_pi_aloha:
            batch[OBS_STATE] = self._pi_aloha_decode_state(batch[OBS_STATE])
            batch[ACTION] = self._pi_aloha_encode_actions_inv(batch[ACTION])

        images, img_masks = self.prepare_images(batch)
        state = self.prepare_state(batch)
        lang_tokens = batch[f"{OBS_LANGUAGE_TOKENS}"]
        lang_masks = batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]
        actions = self.prepare_action(batch)
        actions_is_pad = batch["action_is_pad"]

        losses, fwd_info = self.model.forward(
            images, img_masks, lang_tokens, lang_masks, state, actions
        )

        if actions_is_pad is not None:
            in_episode_bound = ~actions_is_pad
            losses = losses * in_episode_bound.unsqueeze(-1)

        #actual_ad = self.config.action_feature.shape[0]
        losses = losses[:, :, :self.config.max_action_dim]
        bc_flow_loss = losses.mean()

        prefix_embs = fwd_info["prefix_embs"]
        prefix_pad_masks = fwd_info["prefix_pad_masks"]
        prefix_att_masks = fwd_info["prefix_att_masks"]
        prefix_out = fwd_info["prefix_out"]

        # --- 2) Distillation loss: MSE(onestep_actions, flow_actions.detach()) ---
        B = actions.shape[0]
        device = actions.device
        distill_noise = self.model.sample_noise(
            (B, self.config.chunk_size, self.config.max_action_dim), device
        )

        flow_actions = self.model.compute_flow_actions(
            prefix_embs.detach(), prefix_pad_masks, prefix_att_masks, distill_noise
        )

        onestep_actions = self.model.sample_onestep_actions(
            prefix_embs.detach(), prefix_pad_masks, prefix_att_masks, noise=distill_noise
        )
        actual_action_dim = self.config.action_feature.shape[0]
        distill_loss = F.mse_loss(
            onestep_actions[:, :, :actual_action_dim],
            flow_actions[:, :, :actual_action_dim],
            reduction="mean",
        )

        # --- 3) Q loss: maximize Q-value of one-step actions ---
        # We want q_loss to ONLY update onestep_head, not expert/action_in_proj/etc.
        # Strategy: compute dQ/da via torch.autograd.grad (no gradient accumulation
        # on expert params), then build a surrogate loss that only back-props to onestep_head.
        actions_for_q = onestep_actions.detach().clamp(-1, 1).requires_grad_(True)
        expert_hidden = self.model.expert_forward_with_action_grad(
            prefix_embs.detach(), prefix_pad_masks, prefix_att_masks,
            actions_for_q,
        )
        q_values = self.model.compute_q_values(expert_hidden, use_target=False)
        q_scalar = q_values.mean(dim=0).mean()  # scalar Q averaged over critics and batch
        dq_da = torch.autograd.grad(q_scalar, actions_for_q)[0]  # (B, chunk_size, action_dim)
        # Surrogate: dq_da already contains 1/B from q_scalar's batch mean,
        # so .sum() (not .sum()/B) gives the correct -(1/B)*Σ_b ∂Q_b/∂θ gradient.
        q_loss = -(onestep_actions[:, :, :actual_action_dim] * dq_da[:, :, :actual_action_dim].detach()).sum()

        # --- Total actor loss ---
        alpha = self.config.qc_alpha
        total_loss = bc_flow_loss + alpha * distill_loss + q_loss

        loss_dict = {
            "loss": total_loss.item(),
            "bc_flow_loss": bc_flow_loss.item(),
            "distill_loss": distill_loss.item(),
            "q_loss": q_loss.item(),
            "q_actor_mean": q_scalar.item(),
        }
        return total_loss, loss_dict

    def forward_distill(
        self, batch: dict[str, Tensor],
    ) -> tuple[Tensor, dict]:
        """BC flow loss (backbone) + distillation loss (onestep_head), no Q-learning.

        total_loss = bc_flow_loss + alpha * distill_loss

        Returns:
            total_loss: scalar
            loss_dict: diagnostic info
        """
        if self.config.adapt_to_pi_aloha:
            batch[OBS_STATE] = self._pi_aloha_decode_state(batch[OBS_STATE])
            batch[ACTION] = self._pi_aloha_encode_actions_inv(batch[ACTION])

        images, img_masks = self.prepare_images(batch)
        state = self.prepare_state(batch)
        lang_tokens = batch[f"{OBS_LANGUAGE_TOKENS}"]
        lang_masks = batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]
        actions = self.prepare_action(batch)
        actions_is_pad = batch["action_is_pad"]

        losses, fwd_info = self.model.forward(
            images, img_masks, lang_tokens, lang_masks, state, actions
        )

        if actions_is_pad is not None:
            in_episode_bound = ~actions_is_pad
            losses = losses * in_episode_bound.unsqueeze(-1)

        losses = losses[:, :, :self.config.max_action_dim]
        bc_flow_loss = losses.mean()

        prefix_embs = fwd_info["prefix_embs"]
        prefix_pad_masks = fwd_info["prefix_pad_masks"]
        prefix_att_masks = fwd_info["prefix_att_masks"]
        prefix_out = fwd_info["prefix_out"]

        B = actions.shape[0]
        device = actions.device
        distill_noise = self.model.sample_noise(
            (B, self.config.chunk_size, self.config.max_action_dim), device
        )

        flow_actions = self.model.compute_flow_actions(
            prefix_embs.detach(), prefix_pad_masks, prefix_att_masks, distill_noise
        )

        onestep_actions = self.model.sample_onestep_actions(
            prefix_embs.detach(), prefix_pad_masks, prefix_att_masks, noise=distill_noise
        )
        actual_action_dim = self.config.action_feature.shape[0]
        distill_loss = F.mse_loss(
            onestep_actions[:, :, :actual_action_dim],
            flow_actions[:, :, :actual_action_dim],
            reduction="mean",
        )

        alpha = self.config.qc_alpha
        total_loss = bc_flow_loss + alpha * distill_loss

        loss_dict = {
            "loss": total_loss.item(),
            "bc_flow_loss": bc_flow_loss.item(),
            "distill_loss": distill_loss.item(),
        }
        return total_loss, loss_dict

    def forward_critic(
        self,
        batch: dict[str, Tensor],
        next_batch: dict[str, Tensor],
        rewards: Tensor,
        terminateds: Tensor,
        next_obs_is_pad: Tensor | None = None,
    ) -> tuple[Tensor, dict]:
        """Compute QC-FQL critic TD loss.

        Gradient flow:
        - Expert backbone: NO gradient (hidden states detached)
        - Value heads: YES gradient (trained to minimize TD error)
        - Target value heads: NO gradient (EMA copy, used for stable targets)

        Args:
            batch: Current observation batch (standard lerobot format with actions).
            next_batch: Next observation batch (same format, actions not required).
            rewards: Cumulative discounted rewards over the action chunk, shape (B,).
            terminateds: Whether a true terminal state occurred in the chunk, shape (B,).
                Only uses terminated (not truncated). Truncated episodes still bootstrap.
            next_obs_is_pad: (B,) bool from prepare_qc_batch.
                True when abs_idx + chunk_size >= ep_end, meaning the next observation
                is clamped to the episode's last frame (garbage for TD bootstrap).
                Corresponding per-sample critic loss is zeroed out.

                对比 ACFQL 的两层保护：
                - masks[..., -1]: 1 - terminated，用于 TD bootstrap → 我们用 terminateds
                - valid[..., -1]: 1 - done（检测跨 episode）→ LeRobot 无跨 episode 问题，
                  但需检测 next obs 被 padded 的情况 → 用 next_obs_is_pad

        Returns:
            critic_loss: Scalar loss for updating value_heads.
            loss_dict: Diagnostic info dict.
        """
        if self.config.adapt_to_pi_aloha:
            batch[OBS_STATE] = self._pi_aloha_decode_state(batch[OBS_STATE])

        # --- Current Q(s, a) ---
        images, img_masks = self.prepare_images(batch)
        state = self.prepare_state(batch)
        lang_tokens = batch[f"{OBS_LANGUAGE_TOKENS}"]
        lang_masks = batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]
        actions = self.prepare_action(batch)

        prefix = self.model.embed_prefix(images, img_masks, lang_tokens, lang_masks, state=state)
        expert_hidden = self.model.expert_forward_for_value(prefix[0], prefix[1], prefix[2], actions)
        q_values = self.model.compute_q_values(expert_hidden, use_target=False)  # (num_critics, B)

        # --- Target Q(s', a') ---
        with torch.no_grad():
            if self.config.adapt_to_pi_aloha:
                next_batch[OBS_STATE] = self._pi_aloha_decode_state(next_batch[OBS_STATE])

            next_images, next_img_masks = self.prepare_images(next_batch)
            next_state = self.prepare_state(next_batch)
            next_lang_tokens = next_batch.get(
                f"{OBS_LANGUAGE_TOKENS}", lang_tokens
            )
            next_lang_masks = next_batch.get(
                f"{OBS_LANGUAGE_ATTENTION_MASK}", lang_masks
            )

            if self.config.qc_actor_type == "distill-ddpg":
                next_prefix_embs, next_prefix_pad_masks, next_prefix_att_masks = (
                    self.model.embed_prefix(
                        next_images, next_img_masks, next_lang_tokens, next_lang_masks,
                        state=next_state,
                    )
                )
                next_actions = self.model.sample_onestep_actions(
                    next_prefix_embs, next_prefix_pad_masks, next_prefix_att_masks
                )
                expert_hidden_next = self.model.expert_forward_for_value(
                    next_prefix_embs, next_prefix_pad_masks, next_prefix_att_masks, next_actions
                )
                target_q_all = self.model.compute_q_values(expert_hidden_next, use_target=True)
                if self.config.qc_q_agg == "min":
                    target_q_values = target_q_all.min(dim=0).values
                else:
                    target_q_values = target_q_all.mean(dim=0)
            else:
                # Best-of-N: sample N action chunks, pick the best by Q-value
                _best_actions, target_q_values = self.model.sample_best_of_n_with_target_q(
                    next_images, next_img_masks, next_lang_tokens, next_lang_masks, next_state
                )

            # TD target: r + γ^h * (1 - terminated) * Q_target(s', a')
            discount = self.config.qc_discount ** self.config.chunk_size
            td_target = rewards + discount * (1.0 - terminateds.float()) * target_q_values

        # --- Critic loss: per-sample MSE, masked by next_obs_is_pad ---
        # When next obs is padded (clamp to episode end), the TD target bootstraps
        # from garbage observation → zero out these samples.
        # Cf. ACFQL: critic_loss = ((q - target_q)^2 * valid[..., -1]).mean()
        #   where valid[..., -1] detects cross-episode data (same purpose, different mechanism).
        if next_obs_is_pad is not None:
            critic_valid = (~next_obs_is_pad).float()
        else:
            raise ValueError("next_obs_is_pad is required for critic loss")


        loss_dict = {}
        critic_losses = []
        for i, q_i in enumerate(q_values):
            per_sample_loss = (q_i - td_target).pow(2) * critic_valid
            loss_i = per_sample_loss.mean()
            critic_losses.append(loss_i)
            loss_dict[f"critic_q{i}_loss"] = loss_i.item()
            loss_dict[f"critic_q{i}_mean"] = q_i.mean().item()

        critic_loss = sum(critic_losses) / len(critic_losses)
        loss_dict["critic_loss"] = critic_loss.item()
        loss_dict["td_target_mean"] = td_target.mean().item()
        loss_dict["critic_valid_ratio"] = critic_valid.mean().item()

        return critic_loss, loss_dict

    def prepare_images(self, batch):
        """Apply SmolVLA preprocessing to the images."""
        images = []
        img_masks = []
        present_img_keys = [key for key in self.config.image_features if key in batch]
        missing_img_keys = [key for key in self.config.image_features if key not in batch]

        if len(present_img_keys) == 0:
            raise ValueError(
                f"All image features are missing from the batch. At least one expected. "
                f"(batch: {batch.keys()}) (image_features:{self.config.image_features})"
            )
        for key in present_img_keys:
            img = batch[key][:, -1, :, :, :] if batch[key].ndim == 5 else batch[key]
            if self.config.resize_imgs_with_padding is not None:
                img = resize_with_pad(img, *self.config.resize_imgs_with_padding, pad_value=0)

            img = img * 2.0 - 1.0

            bsize = img.shape[0]
            device = img.device
            if f"{key}_padding_mask" in batch:
                mask = batch[f"{key}_padding_mask"].bool()
            else:
                mask = torch.ones(bsize, dtype=torch.bool, device=device)
            images.append(img)
            img_masks.append(mask)

        for num_empty_cameras in range(len(missing_img_keys)):
            if num_empty_cameras >= self.config.empty_cameras:
                break
            img = torch.ones_like(img) * -1
            mask = torch.zeros_like(mask)
            images.append(img)
            img_masks.append(mask)
        return images, img_masks

    def _pi_aloha_decode_state(self, state):
        for motor_idx in [1, 2, 8, 9]:
            state[:, motor_idx] *= -1
        for motor_idx in [6, 13]:
            state[:, motor_idx] = aloha_gripper_to_angular(state[:, motor_idx])
        return state

    def _pi_aloha_encode_actions(self, actions):
        for motor_idx in [1, 2, 8, 9]:
            actions[:, :, motor_idx] *= -1
        for motor_idx in [6, 13]:
            actions[:, :, motor_idx] = aloha_gripper_from_angular(actions[:, :, motor_idx])
        return actions

    def _pi_aloha_encode_actions_inv(self, actions):
        for motor_idx in [1, 2, 8, 9]:
            actions[:, :, motor_idx] *= -1
        for motor_idx in [6, 13]:
            actions[:, :, motor_idx] = aloha_gripper_from_angular_inv(actions[:, :, motor_idx])
        return actions

    def prepare_state(self, batch):
        """Pad state"""
        state = batch[OBS_STATE][:, -1, :] if batch[OBS_STATE].ndim > 2 else batch[OBS_STATE]
        state = pad_vector(state, self.config.max_state_dim)
        return state

    def prepare_action(self, batch):
        """Pad action"""
        actions = pad_vector(batch[ACTION], self.config.max_action_dim)
        return actions

    def _get_default_peft_targets(self) -> dict[str, any]:
        """Return default PEFT target modules for SmolVLA-QL fine-tuning."""
        common_projections = (
            "state_proj|action_in_proj|action_out_proj|action_time_mlp_in|action_time_mlp_out"
        )
        target_modules = rf"(model\.vlm_with_expert\.lm_expert\..*\.(q|v)_proj|model\.({common_projections}))"
        return {
            "target_modules": target_modules,
            "modules_to_save": [],
        }

    def _validate_peft_config(self, peft_config) -> None:
        """Validate PEFT configuration for SmolVLA-QL."""
        super()._validate_peft_config(peft_config)
        if not self.config.load_vlm_weights:
            import logging

            logging.warning(
                "Training SmolVLA-QL from scratch using PEFT. This is unlikely to yield good results. "
                "Set `load_vlm_weights=True` to fine-tune the existing policy."
            )


# ==================== Core Model ====================


class VLAFlowMatching(nn.Module):
    """SmolVLA with QC-FQL Dual-Head Architecture.

    ┌──────────────────────────────────────────┐
    │                 actions    Q-value        │
    │                    ▲          ▲           │
    │ ┌─────────┐      ┌┤────┐   ┌─┤─────┐    │
    │ |         │────► │     │   │ Value  │    │
    │ |         │ kv   │ Act │   │ Head   │    │
    │ |         │────► │ Exp │──►│(detach)│    │
    │ |   VLM   │cache │ ert │   └────────┘    │
    │ │         │────► |     │                 │
    │ │         │      │     │                 │
    │ └▲──▲───▲─┘      └───▲─┘                │
    │  │  |   |            │                   │
    │  |  |   |          noise/actions         │
    │  │  │ state                              │
    │  │ language tokens                       │
    │  image(s)                                │
    └──────────────────────────────────────────┘
    """

    def __init__(self, config: SmolVLAQLConfig, rtc_processor: RTCProcessor | None = None):
        super().__init__()
        self.config = config

        self.vlm_with_expert = SmolVLMWithExpertModel(
            model_id=self.config.vlm_model_name,
            freeze_vision_encoder=self.config.freeze_vision_encoder,
            train_expert_only=self.config.train_expert_only,
            load_vlm_weights=self.config.load_vlm_weights,
            attention_mode=self.config.attention_mode,
            num_expert_layers=self.config.num_expert_layers,
            num_vlm_layers=self.config.num_vlm_layers,
            self_attn_every_n_layers=self.config.self_attn_every_n_layers,
            expert_width_multiplier=self.config.expert_width_multiplier,
            device=self.config.device if self.config.device is not None else "auto",
        )
        self.state_proj = nn.Linear(
            self.config.max_state_dim, self.vlm_with_expert.config.text_config.hidden_size
        )
        self.action_in_proj = nn.Linear(self.config.max_action_dim, self.vlm_with_expert.expert_hidden_size)
        self.action_out_proj = nn.Linear(self.vlm_with_expert.expert_hidden_size, self.config.max_action_dim)

        self.action_time_mlp_in = nn.Linear(
            self.vlm_with_expert.expert_hidden_size * 2, self.vlm_with_expert.expert_hidden_size
        )
        self.action_time_mlp_out = nn.Linear(
            self.vlm_with_expert.expert_hidden_size, self.vlm_with_expert.expert_hidden_size
        )

        # ---- QC-FQL Value Heads ----
        expert_hidden_size = self.vlm_with_expert.expert_hidden_size
        self.value_heads = nn.ModuleList([
            ValueHead(expert_hidden_size, config.qc_value_hidden_dims)
            for _ in range(config.qc_num_critics)
        ])
        self.target_value_heads = nn.ModuleList([
            ValueHead(expert_hidden_size, config.qc_value_hidden_dims)
            for _ in range(config.qc_num_critics)
        ])
        # Initialize targets as exact copies
        for target_head, source_head in zip(self.target_value_heads, self.value_heads):
            target_head.load_state_dict(source_head.state_dict())
        # Freeze target heads — they are updated via EMA only
        for p in self.target_value_heads.parameters():
            p.requires_grad = False

        # ---- One-Step Action Head (distill-ddpg) ----
        if config.qc_actor_type == "distill-ddpg":
            self.onestep_head = OnestepActionHead(
                input_dim=expert_hidden_size,
                output_dim=config.max_action_dim,
                hidden_dims=config.qc_onestep_hidden_dims,
            )

        self.set_requires_grad()
        self.fake_image_token = self.vlm_with_expert.processor.tokenizer.fake_image_token_id
        self.global_image_token = self.vlm_with_expert.processor.tokenizer.global_image_token_id
        self.global_image_start_token = torch.tensor(
            [self.fake_image_token, self.global_image_token], dtype=torch.long
        )

        self.add_image_special_tokens = self.config.add_image_special_tokens
        self.image_end_token = torch.tensor([self.fake_image_token], dtype=torch.long)
        self.prefix_length = self.config.prefix_length
        self.rtc_processor = rtc_processor

    def _rtc_enabled(self):
        return self.config.rtc_config is not None and self.config.rtc_config.enabled

    def set_requires_grad(self):
        for params in self.state_proj.parameters():
            params.requires_grad = self.config.train_state_proj

    def sample_noise(self, shape, device):
        noise = torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=device,
        )
        return noise

    def sample_time(self, bsize, device):
        beta_dist = torch.distributions.Beta(concentration1=1.5, concentration0=1.0)
        time_beta = beta_dist.sample((bsize,)).to(device=device, dtype=torch.float32)
        time = time_beta * 0.999 + 0.001
        return time

    # ==================== One-Step Policy Helpers ====================

    def sample_onestep_actions(
        self,
        prefix_embs: Tensor,
        prefix_pad_masks: Tensor,
        prefix_att_masks: Tensor,
        noise: Tensor | None = None,
    ) -> Tensor:
        """Compute noise_hidden via expert, then predict actions with per-timestep MLP.

        Args:
            prefix_embs: (B, prefix_len, vlm_hidden) from embed_prefix.
            prefix_pad_masks: (B, prefix_len)
            prefix_att_masks: (B, prefix_len)
            noise: (B, chunk_size, max_action_dim). Sampled if None.
        Returns:
            (B, chunk_size, max_action_dim) action chunk.
        """
        B = prefix_embs.shape[0]
        device = prefix_embs.device
        if noise is None:
            noise = self.sample_noise(
                (B, self.config.chunk_size, self.config.max_action_dim), device
            )
        noise_hidden = self.compute_noise_hidden(
            prefix_embs, prefix_pad_masks, prefix_att_masks, noise
        )
        return self.onestep_head(noise_hidden)

    def embed_prefix(
        self, images, img_masks, lang_tokens, lang_masks, state: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embed images with SigLIP and language tokens with embedding layer to prepare
        for SmolVLM transformer processing.
        """
        embs = []
        pad_masks = []
        att_masks = []
        for _img_idx, (
            img,
            img_mask,
        ) in enumerate(zip(images, img_masks, strict=False)):
            if self.add_image_special_tokens:
                image_start_token = (
                    self.vlm_with_expert.embed_language_tokens(
                        self.global_image_start_token.to(device=self.vlm_with_expert.vlm.device)
                    )
                    .unsqueeze(0)
                    .expand(img.shape[0], -1, -1)
                )
                image_start_mask = torch.ones_like(
                    image_start_token[:, :, 0], dtype=torch.bool, device=image_start_token.device
                )
                att_masks += [0] * (image_start_mask.shape[-1])
                embs.append(image_start_token)
                pad_masks.append(image_start_mask)

            img_emb = self.vlm_with_expert.embed_image(img)
            img_emb = img_emb

            img_emb_dim = img_emb.shape[-1]
            img_emb = img_emb * torch.tensor(img_emb_dim**0.5, dtype=img_emb.dtype, device=img_emb.device)

            bsize, num_img_embs = img_emb.shape[:2]
            img_mask = img_mask[:, None].expand(bsize, num_img_embs)

            embs.append(img_emb)
            pad_masks.append(img_mask)

            att_masks += [0] * (num_img_embs)
            if self.add_image_special_tokens:
                image_end_token = (
                    self.vlm_with_expert.embed_language_tokens(
                        self.image_end_token.to(device=self.vlm_with_expert.vlm.device)
                    )
                    .unsqueeze(0)
                    .expand(img.shape[0], -1, -1)
                )
                image_end_mask = torch.ones_like(
                    image_end_token[:, :, 0], dtype=torch.bool, device=image_end_token.device
                )
                embs.append(image_end_token)
                pad_masks.append(image_end_mask)
                att_masks += [0] * (image_end_mask.shape[1])
        lang_emb = self.vlm_with_expert.embed_language_tokens(lang_tokens)
        lang_emb_dim = lang_emb.shape[-1]
        lang_emb = lang_emb * math.sqrt(lang_emb_dim)

        embs.append(lang_emb)
        pad_masks.append(lang_masks)

        num_lang_embs = lang_emb.shape[1]
        att_masks += [0] * num_lang_embs

        state_emb = self.state_proj(state)
        state_emb = state_emb[:, None, :] if state_emb.ndim == 2 else state_emb
        embs.append(state_emb)
        bsize = state_emb.shape[0]
        device = state_emb.device

        states_seq_len = state_emb.shape[1]
        state_mask = torch.ones(bsize, states_seq_len, dtype=torch.bool, device=device)
        pad_masks.append(state_mask)

        att_masks += [1] * (states_seq_len)
        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)
        att_masks = att_masks[None, :]

        seq_len = pad_masks.shape[1]
        if seq_len < self.prefix_length:
            embs = pad_tensor(embs, self.prefix_length, pad_value=0)
            pad_masks = pad_tensor(pad_masks, self.prefix_length, pad_value=0)
            att_masks = pad_tensor(att_masks, self.prefix_length, pad_value=0)

        att_masks = att_masks.expand(bsize, -1)

        return embs, pad_masks, att_masks

    def embed_suffix(self, noisy_actions, timestep):
        """Embed noisy_actions + timestep to prepare for Expert processing."""
        embs = []
        pad_masks = []
        att_masks = []

        action_emb = self.action_in_proj(noisy_actions)
        device = action_emb.device
        bsize = action_emb.shape[0]
        dtype = action_emb.dtype
        time_emb = create_sinusoidal_pos_embedding(
            timestep,
            self.vlm_with_expert.expert_hidden_size,
            self.config.min_period,
            self.config.max_period,
            device=device,
        )
        time_emb = time_emb.type(dtype=dtype)

        time_emb = time_emb[:, None, :].expand_as(action_emb)
        action_time_emb = torch.cat([action_emb, time_emb], dim=2)

        action_time_emb = self.action_time_mlp_in(action_time_emb)
        action_time_emb = F.silu(action_time_emb)
        action_time_emb = self.action_time_mlp_out(action_time_emb)

        embs.append(action_time_emb)

        bsize, action_time_dim = action_time_emb.shape[:2]
        action_time_mask = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=device)
        pad_masks.append(action_time_mask)

        att_masks += [1] * self.config.chunk_size
        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))
        return embs, pad_masks, att_masks

    # ==================== Flow Matching (policy training) ====================

    def forward(
        self, images, img_masks, lang_tokens, lang_masks, state, actions, noise=None, time=None
    ) -> tuple[Tensor, dict]:
        """Flow matching training forward.

        Returns:
            losses: (B, chunk_size, action_dim) per-element MSE losses.
            info: dict with cached intermediate tensors for forward_actor reuse:
                - prefix_embs, prefix_pad_masks, prefix_att_masks
                - prefix_out (from VLM, for one-step head)
        """
        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)

        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=state
        )
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(x_t, time)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        (prefix_out, suffix_out), _ = self.vlm_with_expert.forward(
            attention_mask=att_2d_masks,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, suffix_embs],
            use_cache=False,
            fill_kv_cache=False,
        )
        suffix_out = suffix_out[:, -self.config.chunk_size :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        v_t = self.action_out_proj(suffix_out)
        losses = F.mse_loss(u_t, v_t, reduction="none")

        fwd_info = {
            "prefix_embs": prefix_embs,
            "prefix_pad_masks": prefix_pad_masks,
            "prefix_att_masks": prefix_att_masks,
            "prefix_out": prefix_out,
        }
        return losses, fwd_info

    # ==================== Inference ====================

    def sample_actions(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state,
        noise=None,
        **kwargs: Unpack[ActionSelectKwargs],
    ) -> Tensor:
        """Standard flow matching inference (no Q-value guidance)."""
        bsize = state.shape[0]
        device = state.device

        if noise is None:
            actions_shape = (bsize, self.config.chunk_size, self.config.max_action_dim)
            noise = self.sample_noise(actions_shape, device)

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=state
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        _, past_key_values = self.vlm_with_expert.forward(
            attention_mask=prefix_att_2d_masks,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=self.config.use_cache,
            fill_kv_cache=True,
        )
        num_steps = self.config.num_steps
        dt = -1.0 / num_steps

        x_t = noise
        for step in range(num_steps):
            time = 1.0 + step * dt
            time_tensor = torch.tensor(time, dtype=torch.float32, device=device).expand(bsize)

            def denoise_step_partial_call(input_x_t, current_timestep=time_tensor):
                return self.denoise_step(
                    x_t=input_x_t,
                    prefix_pad_masks=prefix_pad_masks,
                    past_key_values=past_key_values,
                    timestep=current_timestep,
                )

            if self._rtc_enabled():
                inference_delay = kwargs.get("inference_delay")
                prev_chunk_left_over = kwargs.get("prev_chunk_left_over")
                execution_horizon = kwargs.get("execution_horizon")

                v_t = self.rtc_processor.denoise_step(
                    x_t=x_t,
                    prev_chunk_left_over=prev_chunk_left_over,
                    inference_delay=inference_delay,
                    time=time,
                    original_denoise_step_partial=denoise_step_partial_call,
                    execution_horizon=execution_horizon,
                )
            else:
                v_t = denoise_step_partial_call(x_t)

            x_t = x_t + dt * v_t

            if self.rtc_processor is not None and self.rtc_processor.is_debug_enabled():
                self.rtc_processor.track(time=time, x_t=x_t, v_t=v_t)

        return x_t

    def sample_actions_onestep(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state,
        noise=None,
    ) -> Tensor:
        """One-step inference: embed_prefix → noise_hidden via expert → MLP → actions."""
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=state
        )
        return self.sample_onestep_actions(
            prefix_embs, prefix_pad_masks, prefix_att_masks, noise=noise
        )

    @torch.no_grad()
    def compute_flow_actions(
        self,
        prefix_embs: Tensor,
        prefix_pad_masks: Tensor,
        prefix_att_masks: Tensor,
        noise: Tensor,
    ) -> Tensor:
        """Run full iterative flow denoising from precomputed prefix to get clean actions.

        Used as the distillation target for the one-step head (called under no_grad).

        Args:
            prefix_embs: (B, prefix_len, vlm_hidden) precomputed.
            prefix_pad_masks: (B, prefix_len)
            prefix_att_masks: (B, prefix_len)
            noise: (B, chunk_size, max_action_dim) starting noise.
        Returns:
            (B, chunk_size, max_action_dim) denoised actions.
        """
        bsize = noise.shape[0]
        device = noise.device

        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        _, past_key_values = self.vlm_with_expert.forward(
            attention_mask=prefix_att_2d_masks,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=self.config.use_cache,
            fill_kv_cache=True,
        )

        num_steps = self.config.num_steps
        dt = -1.0 / num_steps
        x_t = noise
        for step in range(num_steps):
            t = 1.0 + step * dt
            time_tensor = torch.tensor(t, dtype=torch.float32, device=device).expand(bsize)
            v_t = self.denoise_step(
                x_t=x_t,
                prefix_pad_masks=prefix_pad_masks,
                past_key_values=past_key_values,
                timestep=time_tensor,
            )
            x_t = x_t + dt * v_t
        return x_t

    def denoise_step(
        self,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
    ):
        """Apply one denoising step of the noise `x_t` at a given timestep."""
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(x_t, timestep)

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]
        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)

        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)

        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)
        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        outputs_embeds, _ = self.vlm_with_expert.forward(
            attention_mask=full_att_2d_masks,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=self.config.use_cache,
            fill_kv_cache=False,
        )
        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.chunk_size :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        v_t = self.action_out_proj(suffix_out)
        return v_t

    # ==================== QC-FQL Value Components ====================

    def expert_forward_for_value(
        self,
        prefix_embs: Tensor,
        prefix_pad_masks: Tensor,
        prefix_att_masks: Tensor,
        clean_actions: Tensor,
    ) -> Tensor:
        """Forward clean actions through the shared expert at t=0 for value estimation.

        Uses timestep=0 to signal "fully denoised actions" to the expert.
        The entire VLM+expert forward is run under no_grad and the output is detached,
        ensuring that critic training does NOT update the expert backbone, VLM, or state_proj.

        Args:
            prefix_embs: (B, prefix_len, vlm_hidden) — precomputed VLM prefix embeddings.
            prefix_pad_masks: (B, prefix_len) — padding mask for prefix.
            prefix_att_masks: (B, prefix_len) — attention mask for prefix.
            clean_actions: (B, chunk_size, action_dim) — clean action chunk to evaluate.

        Returns:
            (B, chunk_size, expert_hidden_size) — detached expert hidden states.
        """
        bsize = clean_actions.shape[0]
        device = clean_actions.device

        # Detach prefix to sever any gradient connection to VLM / state_proj
        prefix_embs = prefix_embs.detach()

        timestep = torch.zeros(bsize, device=device, dtype=torch.float32)
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(clean_actions, timestep)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        with torch.no_grad():
            (_, suffix_out), _ = self.vlm_with_expert.forward(
                attention_mask=att_2d_masks,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, suffix_embs],
                use_cache=False,
                fill_kv_cache=False,
            )

        suffix_out = suffix_out[:, -self.config.chunk_size :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        return suffix_out  # detached due to no_grad context

    def expert_forward_with_action_grad(
        self,
        prefix_embs: Tensor,
        prefix_pad_masks: Tensor,
        prefix_att_masks: Tensor,
        clean_actions: Tensor,
    ) -> Tensor:
        """Like expert_forward_for_value but allows gradients through clean_actions.

        Used by forward_actor's q_loss: gradient flows from Q → value_heads →
        expert_hidden → expert layers → embed_suffix → action_in_proj → clean_actions
        → onestep_head. Prefix is detached so VLM/state_proj get no gradient from this path.
        """
        bsize = clean_actions.shape[0]
        device = clean_actions.device

        prefix_embs = prefix_embs.detach()

        timestep = torch.zeros(bsize, device=device, dtype=torch.float32)
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(clean_actions, timestep)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        (_, suffix_out), _ = self.vlm_with_expert.forward(
            attention_mask=att_2d_masks,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, suffix_embs],
            use_cache=False,
            fill_kv_cache=False,
        )

        suffix_out = suffix_out[:, -self.config.chunk_size :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        return suffix_out

    def compute_q_values(self, expert_hidden_states: Tensor, use_target: bool = False) -> Tensor:
        """Compute Q-values from expert hidden states using the value head ensemble.

        Args:
            expert_hidden_states: (B, chunk_size, expert_hidden_size) — detached hidden states.
            use_target: if True, use target_value_heads (for stable TD targets).

        Returns:
            (num_critics, B) tensor of Q-values.
        """
        heads = self.target_value_heads if use_target else self.value_heads
        q_values = torch.stack([head(expert_hidden_states) for head in heads], dim=0)
        return q_values

    @torch.no_grad()
    def update_target_value_heads(self):
        """Polyak (EMA) update: target ← (1 - τ) * target + τ * online."""
        tau = self.config.qc_tau
        for target_head, source_head in zip(self.target_value_heads, self.value_heads):
            for tp, sp in zip(target_head.parameters(), source_head.parameters()):
                tp.data.mul_(1.0 - tau).add_(sp.data, alpha=tau)

    # ==================== Best-of-N Inference ====================

    @torch.no_grad()
    def sample_actions_best_of_n(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state,
        n_samples: int | None = None,
    ) -> Tensor:
        """Sample N action chunks from the flow policy and pick the best by Q-value.

        Steps:
        1. Compute VLM prefix KV cache once.
        2. Run N parallel denoising trajectories (batch dimension = N * B).
        3. Evaluate all N chunks with value heads (via expert_forward_for_value).
        4. Select the chunk with highest aggregated Q-value per batch element.

        Args:
            images, img_masks, lang_tokens, lang_masks, state: observation inputs.
            n_samples: override for config.qc_best_of_n.

        Returns:
            (B, chunk_size, action_dim) — the best action chunk per batch element.
        """
        n = n_samples or self.config.qc_best_of_n
        bsize = state.shape[0]
        device = state.device

        # 1. Compute prefix once
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=state
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        _, past_key_values = self.vlm_with_expert.forward(
            attention_mask=prefix_att_2d_masks,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=self.config.use_cache,
            fill_kv_cache=True,
        )

        # 2. Replicate KV cache and masks for N parallel samples
        rep_prefix_pad_masks = prefix_pad_masks.repeat(n, 1)  # (N*B, prefix_len)
        rep_past_key_values = {}
        for layer_idx, kv in past_key_values.items():
            rep_past_key_values[layer_idx] = {
                "key_states": kv["key_states"].repeat(n, 1, 1, 1),
                "value_states": kv["value_states"].repeat(n, 1, 1, 1),
            }

        # 3. Sample N denoising trajectories in parallel
        actions_shape = (n * bsize, self.config.chunk_size, self.config.max_action_dim)
        noise = self.sample_noise(actions_shape, device)

        num_steps = self.config.num_steps
        dt = -1.0 / num_steps
        x_t = noise

        for step in range(num_steps):
            t = 1.0 + step * dt
            time_tensor = torch.tensor(t, dtype=torch.float32, device=device).expand(n * bsize)
            v_t = self.denoise_step(
                x_t=x_t,
                prefix_pad_masks=rep_prefix_pad_masks,
                past_key_values=rep_past_key_values,
                timestep=time_tensor,
            )
            x_t = x_t + dt * v_t

        # x_t: (N*B, chunk_size, action_dim)

        # 4. Evaluate all chunks with value heads
        rep_prefix_embs = prefix_embs.repeat(n, 1, 1)
        rep_prefix_att_masks = prefix_att_masks.repeat(n, 1)

        expert_hidden = self._expert_forward_for_value_no_grad(
            rep_prefix_embs, rep_prefix_pad_masks, rep_prefix_att_masks, x_t
        )
        q_values = self.compute_q_values(expert_hidden, use_target=False)  # (num_critics, N*B)

        # Aggregate across critics
        if self.config.qc_q_agg == "min":
            q_agg = q_values.min(dim=0).values  # (N*B,)
        else:
            q_agg = q_values.mean(dim=0)  # (N*B,)

        # Reshape to (N, B) and pick argmax per batch element
        q_agg = q_agg.view(n, bsize)
        best_idx = q_agg.argmax(dim=0)  # (B,)

        x_t = x_t.view(n, bsize, self.config.chunk_size, self.config.max_action_dim)
        best_actions = x_t[best_idx, torch.arange(bsize, device=device)]  # (B, chunk_size, action_dim)

        return best_actions

    @torch.no_grad()
    def sample_best_of_n_with_target_q(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state,
        n_samples: int | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Sample Best-of-N actions AND compute target Q-value in one pass.

        Combines action sampling + Q evaluation + target Q evaluation to avoid
        redundant prefix computations. Used during critic training for the TD target.

        Returns:
            best_actions: (B, chunk_size, action_dim)
            target_q: (B,) — aggregated target Q-value for the best actions.
        """
        n = n_samples or self.config.qc_best_of_n
        bsize = state.shape[0]
        device = state.device

        # 1. Compute prefix once
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=state
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        _, past_key_values = self.vlm_with_expert.forward(
            attention_mask=prefix_att_2d_masks,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=self.config.use_cache,
            fill_kv_cache=True,
        )

        # 2. Replicate for N parallel samples
        rep_prefix_pad_masks = prefix_pad_masks.repeat(n, 1)
        rep_past_key_values = {}
        for layer_idx, kv in past_key_values.items():
            rep_past_key_values[layer_idx] = {
                "key_states": kv["key_states"].repeat(n, 1, 1, 1),
                "value_states": kv["value_states"].repeat(n, 1, 1, 1),
            }

        # 3. Denoise N samples
        actions_shape = (n * bsize, self.config.chunk_size, self.config.max_action_dim)
        noise = self.sample_noise(actions_shape, device)

        num_steps = self.config.num_steps
        dt = -1.0 / num_steps
        x_t = noise

        for step in range(num_steps):
            t = 1.0 + step * dt
            time_tensor = torch.tensor(t, dtype=torch.float32, device=device).expand(n * bsize)
            v_t = self.denoise_step(
                x_t=x_t,
                prefix_pad_masks=rep_prefix_pad_masks,
                past_key_values=rep_past_key_values,
                timestep=time_tensor,
            )
            x_t = x_t + dt * v_t

        # 4. Expert forward for all N*B actions
        rep_prefix_embs = prefix_embs.repeat(n, 1, 1)
        rep_prefix_att_masks = prefix_att_masks.repeat(n, 1)

        expert_hidden = self._expert_forward_for_value_no_grad(
            rep_prefix_embs, rep_prefix_pad_masks, rep_prefix_att_masks, x_t
        )

        # 5. Pick best by online Q, then evaluate with target Q
        q_online = self.compute_q_values(expert_hidden, use_target=False)  # (num_critics, N*B)
        if self.config.qc_q_agg == "min":
            q_agg = q_online.min(dim=0).values
        else:
            q_agg = q_online.mean(dim=0)

        q_agg = q_agg.view(n, bsize)
        best_idx = q_agg.argmax(dim=0)  # (B,)

        # Gather best actions
        x_t = x_t.view(n, bsize, self.config.chunk_size, self.config.max_action_dim)
        best_actions = x_t[best_idx, torch.arange(bsize, device=device)]

        # Gather best expert hidden states and compute target Q
        expert_hidden = expert_hidden.view(
            n, bsize, self.config.chunk_size, self.vlm_with_expert.expert_hidden_size
        )
        best_expert_hidden = expert_hidden[best_idx, torch.arange(bsize, device=device)]

        target_q = self.compute_q_values(best_expert_hidden, use_target=True)  # (num_critics, B)
        if self.config.qc_q_agg == "min":
            target_q = target_q.min(dim=0).values
        else:
            target_q = target_q.mean(dim=0)

        return best_actions, target_q  # (B, chunk, dim), (B,)

    def _expert_forward_for_value_no_grad(
        self,
        prefix_embs: Tensor,
        prefix_pad_masks: Tensor,
        prefix_att_masks: Tensor,
        clean_actions: Tensor,
    ) -> Tensor:
        """Internal helper: expert forward under no_grad (already in a no_grad context).

        Same as expert_forward_for_value but assumes caller handles no_grad.
        """
        bsize = clean_actions.shape[0]
        device = clean_actions.device

        timestep = torch.zeros(bsize, device=device, dtype=torch.float32)
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(clean_actions, timestep)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        (_, suffix_out), _ = self.vlm_with_expert.forward(
            attention_mask=att_2d_masks,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, suffix_embs],
            use_cache=False,
            fill_kv_cache=False,
        )
        suffix_out = suffix_out[:, -self.config.chunk_size :]
        return suffix_out.to(dtype=torch.float32)

    @torch.no_grad()
    def compute_noise_hidden(
        self,
        prefix_embs: Tensor,
        prefix_pad_masks: Tensor,
        prefix_att_masks: Tensor,
        noise: Tensor,
    ) -> Tensor:
        """Expert forward on noise at t=1, returning detached expert hidden states.

        The noise is embedded as if it were x_t at timestep=1 (pure noise), then
        processed by the expert with cross-attention to the observation prefix.
        The resulting hidden states carry rich, observation-conditioned information
        about the noise — much richer than a single state_token.

        All inputs are detached internally; no gradient flows back to the backbone.

        Args:
            prefix_embs: (B, prefix_len, vlm_hidden) from embed_prefix.
            prefix_pad_masks: (B, prefix_len)
            prefix_att_masks: (B, prefix_len)
            noise: (B, chunk_size, max_action_dim)
        Returns:
            (B, chunk_size, expert_hidden_size) — detached expert hidden states.
        """
        bsize = noise.shape[0]
        device = noise.device

        prefix_embs = prefix_embs.detach()

        timestep = torch.ones(bsize, device=device, dtype=torch.float32)
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(noise, timestep)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        (_, suffix_out), _ = self.vlm_with_expert.forward(
            attention_mask=att_2d_masks,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, suffix_embs],
            use_cache=False,
            fill_kv_cache=False,
        )
        suffix_out = suffix_out[:, -self.config.chunk_size :]
        return suffix_out.to(dtype=torch.float32)
