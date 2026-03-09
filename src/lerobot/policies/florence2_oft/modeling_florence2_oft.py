"""Florence2-OFT Policy for LeRobot.

Model overview
==============

The policy is a lightweight "vision-language encoder + MLP action head" model:

    observation images
            |
            v
      Florence2 processor
    (PIL conversion, resize,
     tokenize prompt text)
            |
            v
    +-----------------------+
    | Florence2 vision tower|
    |   _encode_image()     |
    +-----------------------+
            |
            | image tokens
            v
    +-----------------------+      prompt + repeated action tokens
    | merge image tokens    | <----------------------------------+
    | with text embeddings  |                                    |
    +-----------------------+                                    |
            |                                                   |
            v                                                   |
    +-----------------------+                                   |
    | Florence2 text encoder|                                   |
    |   language_model      |                                   |
    +-----------------------+                                   |
            |                                                   |
            | hidden states of the final action-token positions |
            v                                                   |
    +-----------------------+                                   |
    |  MLP action head      |                                   |
    | (L1RegressionActionHead)                                  |
    +-----------------------+                                   |
            |                                                   |
            v                                                   |
      predicted action chunk -----------------------------------+

High-level behavior:
  1. Build a text prompt and append `chunk_size` action tokens.
  2. Encode image(s) with Florence2's vision backbone.
  3. Concatenate image features with text embeddings.
  4. Run Florence2's encoder over the merged sequence.
  5. Take the last `chunk_size` hidden states as action queries.
  6. Map them to continuous robot actions with an MLP head.

The Florence2 decoder and LM head are removed because this policy only needs
encoder features for regression, not text generation.

Reference:
  - Florence2: https://huggingface.co/microsoft/Florence-2-large
"""

import logging
from collections import deque
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoProcessor
from typing_extensions import Unpack

from lerobot.configs.types import FeatureType
from lerobot.policies.florence2_oft.configuration_florence2_oft import Florence2OFTConfig
from lerobot.policies.pretrained import ActionSelectKwargs, PreTrainedPolicy
from lerobot.policies.utils import get_device_from_parameters, populate_queues
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

logger = logging.getLogger(__name__)


# =============================================================================
# MLP ResNet Action Head (L1 Regression)
# =============================================================================


class MLPResNetBlock(nn.Module):
    """One MLP ResNet block with a residual connection."""

    def __init__(self, dim: int):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.ReLU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.ffn(x)


class MLPResNet(nn.Module):
    """MLP with residual connection blocks."""

    def __init__(self, num_blocks: int, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.mlp_resnet_blocks = nn.ModuleList([MLPResNetBlock(dim=hidden_dim) for _ in range(num_blocks)])
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.layer_norm1(x)
        x = self.fc1(x)
        x = self.relu(x)
        for block in self.mlp_resnet_blocks:
            x = block(x)
        x = self.layer_norm2(x)
        x = self.fc2(x)
        return x


class L1RegressionActionHead(nn.Module):
    """MLP-based action head that generates continuous actions via L1 regression."""

    def __init__(
        self,
        input_dim: int = 1024,
        hidden_dim: int = 2048,
        action_dim: int = 7,
        num_blocks: int = 2,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.model = MLPResNet(
            num_blocks=num_blocks,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=action_dim,
        )

    def forward(self, actions_hidden_states: Tensor) -> Tensor:
        """
        Args:
            actions_hidden_states: (B, chunk_len, hidden_dim)
        Returns:
            actions: (B, chunk_len, action_dim)
        """
        batch_size, chunk_len, hidden_dim = actions_hidden_states.shape
        x = actions_hidden_states.reshape(batch_size * chunk_len, hidden_dim)
        x = self.model(x)
        return x.view(batch_size, chunk_len, self.action_dim)


# =============================================================================
# Florence2-OFT Policy
# =============================================================================


class Florence2OFTPolicy(PreTrainedPolicy):
    """Florence2-OFT Policy for continuous action prediction.

    Uses Florence2 encoder as backbone and predicts actions via L1 regression
    on encoder hidden states at action token positions.
    """

    config_class = Florence2OFTConfig
    name = "florence2_oft"

    def __init__(self, config: Florence2OFTConfig, **kwargs):
        super().__init__(config)
        config.validate_features()
        self.config = config

        self._queues = None

        # Determine action_dim from output features
        action_dim = config.action_feature.shape[0]

        # Match ILStudio florence2_oft defaults: keep Florence2 in bf16 on CUDA.
        # Their config explicitly notes fp16 can overflow inside the vision encoder.
        if config.device != "cpu":
            torch_dtype = torch.bfloat16 if config.use_bf16 else torch.float16
        else:
            torch_dtype = torch.float32
        self.vlm = AutoModelForCausalLM.from_pretrained(
            config.vlm_model_name_or_path,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            attn_implementation="eager",
        )

        # Get hidden size from VLM
        vlm_hidden_size = self.vlm.config.projection_dim
        self.vlm.config.hidden_size = vlm_hidden_size

        # Remove unused modules to save memory
        if hasattr(self.vlm, "decoder"):
            del self.vlm.decoder
            logger.info("Removed Florence2 decoder to save memory")
        if hasattr(self.vlm, "lm_head"):
            del self.vlm.lm_head
            logger.info("Removed Florence2 lm_head to save memory")

        # Build action head (kept in float32 for numerical stability)
        self.action_head = L1RegressionActionHead(
            input_dim=vlm_hidden_size,
            hidden_dim=vlm_hidden_size * config.action_head_hidden_mult,
            action_dim=action_dim,
            num_blocks=config.action_head_num_blocks,
        ).float()

        self.l1_loss = nn.L1Loss()

        # Load Florence2 processor
        self.multimodal_processor = AutoProcessor.from_pretrained(
            config.vlm_model_name_or_path,
            trust_remote_code=True,
        )

        # Configure image size if specified.
        # Florence2 currently requires square feature maps in _encode_image.
        if config.image_size is not None:
            h, w = config.image_size[0], config.image_size[1] if len(config.image_size) > 1 else config.image_size[0]
            side = min(int(h), int(w))
            if int(h) != int(w):
                logger.warning(
                    "Florence2 expects square image features. Received image_size=%s; using square %sx%s.",
                    config.image_size,
                    side,
                    side,
                )
            if hasattr(self.multimodal_processor, "image_processor"):
                self.multimodal_processor.image_processor.size = {"height": side, "width": side}
                self.multimodal_processor.image_processor.crop_size = {"height": side, "width": side}

        # Setup tokenizer and action token
        self.tokenizer = (
            self.multimodal_processor.tokenizer
            if hasattr(self.multimodal_processor, "tokenizer")
            else self.multimodal_processor
        )
        self._setup_action_token()

        self._vlm_dtype = torch_dtype
        self.chunk_size = config.chunk_size

        # Apply freezing
        self._apply_freeze()

        self.reset()

    def _setup_action_token(self):
        """Add action token as special token if needed for single-token tokenization."""
        action_token = self.config.action_token
        token_ids = self.tokenizer(action_token, add_special_tokens=False)["input_ids"]

        if len(token_ids) != 1 and hasattr(self.tokenizer, "add_tokens"):
            num_added = self.tokenizer.add_tokens([action_token], special_tokens=True)
            if num_added > 0:
                logger.info(f"Added action token '{action_token}' as special token.")
                if hasattr(self.vlm, "resize_token_embeddings"):
                    self.vlm.resize_token_embeddings(len(self.tokenizer))
                elif hasattr(self.vlm, "language_model") and hasattr(
                    self.vlm.language_model, "resize_token_embeddings"
                ):
                    self.vlm.language_model.resize_token_embeddings(len(self.tokenizer))

    def _apply_freeze(self):
        """Apply parameter freezing based on config."""
        if self.config.freeze_vision_tower:
            for name in ["vision_tower", "image_projection"]:
                module = getattr(self.vlm, name, None)
                if module is not None:
                    for p in module.parameters():
                        p.requires_grad = False
            logger.info("Vision components frozen")

        if self.config.freeze_backbone:
            if hasattr(self.vlm, "language_model"):
                for p in self.vlm.language_model.parameters():
                    p.requires_grad = False
                logger.info("Language model backbone frozen")

        self.action_head.requires_grad_(True)

    def _tensor_to_pil(self, image_tensor: Tensor) -> Image.Image:
        """Convert a tensor to PIL Image. Accepts (C, H, W) or (H, W, C) in [0,1] or [0,255]."""
        img = image_tensor.detach().cpu()
        # Reduce to 3D (C, H, W) by taking the first element of any leading dims
        while img.dim() > 3:
            img = img[0]
        # CHW -> HWC
        if img.dim() == 3 and img.shape[0] in [1, 3, 4]:
            img = img.permute(1, 2, 0)
        img_np = img.numpy()
        if img_np.max() <= 1.0:
            img_np = (img_np * 255).clip(0, 255)
        return Image.fromarray(img_np.astype(np.uint8))

    def _build_prompt(self, instruction: str = "") -> str:
        """Build text prompt with action tokens appended."""
        action_tokens = self.config.action_token * self.chunk_size
        return f"{self.config.task_prompt} {instruction} {action_tokens}"

    def _prepare_vlm_inputs(
        self, batch: dict[str, Tensor]
    ) -> dict[str, Tensor]:
        """Convert lerobot batch to Florence2 encoder inputs.

        Supports multi-camera: each camera's image is processed separately, and their
        pixel_values are stacked along the batch dimension for joint encoding.
        The encoder will later reshape features to [B, N_view*N, D].

        Returns dict with 'input_ids', 'attention_mask', 'pixel_values', and 'n_cameras'.
        """
        device = get_device_from_parameters(self)

        if OBS_IMAGES in batch:
            images_tensor = batch[OBS_IMAGES]
        else:
            img_keys = sorted(k for k in batch if k.startswith("observation.image"))
            if not img_keys:
                raise ValueError("No image features found in batch for Florence2-OFT.")
            images_tensor = batch[img_keys[0]]

        # Normalize to (B, N_cameras, C, H, W):
        #   6D (B, T, N, C, H, W) → take last timestep → (B, N, C, H, W)
        #   5D (B, T, C, H, W) or (B, N, C, H, W) → take last timestep → (B, C, H, W) → unsqueeze
        #   4D (B, C, H, W) → unsqueeze to (B, 1, C, H, W)
        if images_tensor.dim() == 6:
            images_tensor = images_tensor[:, -1]  # (B, N, C, H, W)
        elif images_tensor.dim() == 5:
            images_tensor = images_tensor[:, -1]  # (B, C, H, W)
            images_tensor = images_tensor.unsqueeze(1)  # (B, 1, C, H, W)
        elif images_tensor.dim() == 4:
            images_tensor = images_tensor.unsqueeze(1)  # (B, 1, C, H, W)

        # Match processor size to actual tensor size when not explicitly configured.
        # This prevents processor-side upscaling (e.g., to Florence defaults) that can
        # explode visual token count and attention memory.
        # Florence2 requires square feature maps, so use min(H, W) as square side.
        if self.config.image_size is None and hasattr(self.multimodal_processor, "image_processor"):
            h, w = int(images_tensor.shape[-2]), int(images_tensor.shape[-1])
            side = min(h, w)
            self.multimodal_processor.image_processor.size = {"height": side, "width": side}
            self.multimodal_processor.image_processor.crop_size = {"height": side, "width": side}

        # ILStudio data path uses single-view for Florence2-OFT. Keep this as default
        # for memory stability, and allow opt-in multi-view through config.
        if not self.config.use_multi_view:
            images_tensor = images_tensor[:, :1]

        batch_size, n_cameras = images_tensor.shape[0], images_tensor.shape[1]

        # Process the first camera image to get text tokens (shared across all cameras)
        pil_ref = [self._tensor_to_pil(images_tensor[i, 0]) for i in range(batch_size)]
        prompts = [self._build_prompt() for _ in range(batch_size)]
        ref_inputs = self.multimodal_processor(
            text=prompts, images=pil_ref, return_tensors="pt", padding=True, truncation=True
        )

        # Process ALL camera images to get pixel_values: stack into (B*N_cameras, C, H', W')
        all_pixel_values = []
        for cam_idx in range(n_cameras):
            pil_imgs = [self._tensor_to_pil(images_tensor[i, cam_idx]) for i in range(batch_size)]
            cam_inputs = self.multimodal_processor(
                text=prompts, images=pil_imgs, return_tensors="pt", padding=True, truncation=True
            )
            all_pixel_values.append(cam_inputs["pixel_values"])

        # Keep batch-major order before flattening cameras:
        #   stack -> (B, N_cameras, C, H', W') -> reshape -> (B*N_cameras, C, H', W')
        # This matches the later `view(B, -1, D)` in `_forward_encoder`.
        stacked_pv = torch.stack(all_pixel_values, dim=1).reshape(
            batch_size * n_cameras, *all_pixel_values[0].shape[1:]
        )

        result = {
            "input_ids": ref_inputs["input_ids"].to(device),
            "attention_mask": ref_inputs.get("attention_mask", torch.ones_like(ref_inputs["input_ids"])).to(device),
            "pixel_values": stacked_pv.to(device),
            "n_cameras": n_cameras,
        }
        return result

    def _forward_encoder(
        self,
        input_ids: Tensor,
        pixel_values: Tensor,
        attention_mask: Tensor | None = None,
        n_cameras: int = 1,
    ) -> Tensor:
        """Forward pass through Florence2 encoder.

        Encodes all camera views and flattens their features into a single sequence
        [B, N_view*N_tokens, D] before merging with text embeddings, matching the
        ILStudio architecture.
        """
        # Match ILStudio exactly here: keep image tensors in the VLM load dtype
        # and rely on the outer autocast context for compute precision.
        pixel_values = pixel_values.to(dtype=self._vlm_dtype)

        # pixel_values: (B*N_cameras, C, H, W)
        valid_feats = self.vlm._encode_image(pixel_values)  # (B*N_cam, N_tokens, D)
        if not torch.isfinite(valid_feats).all():
            nonfinite = self._count_nonfinite(valid_feats)
            raise RuntimeError(
                f"Non-finite Florence2 image features: {nonfinite} / {valid_feats.numel()}. "
                f"pixel_values[min={pixel_values.min().item():.4f}, max={pixel_values.max().item():.4f}, "
                f"mean={pixel_values.mean().item():.4f}, dtype={pixel_values.dtype}]"
            )

        inputs_embeds = self.vlm.get_input_embeddings()(input_ids)  # (B, L, D)
        B, L, D = inputs_embeds.shape
        if not torch.isfinite(inputs_embeds).all():
            nonfinite = self._count_nonfinite(inputs_embeds)
            raise RuntimeError(f"Non-finite Florence2 text embeddings: {nonfinite} / {inputs_embeds.numel()}.")

        # Reshape: (B*N_cam, N_tokens, D) → (B, N_cam*N_tokens, D)
        image_features = valid_feats.view(B, -1, D)

        merged_embeds, merged_attention_mask = self.vlm._merge_input_ids_with_image_features(
            image_features,
            inputs_embeds,
        )
        if not torch.isfinite(merged_embeds).all():
            nonfinite = self._count_nonfinite(merged_embeds)
            raise RuntimeError(f"Non-finite Florence2 merged embeddings: {nonfinite} / {merged_embeds.numel()}.")

        enc_out = self.vlm.language_model.model.encoder(
            attention_mask=merged_attention_mask,
            inputs_embeds=merged_embeds,
        )

        return enc_out.last_hidden_state

    @staticmethod
    def _count_nonfinite(x: Tensor) -> int:
        return int((~torch.isfinite(x)).sum().item())

    def get_optim_params(self) -> dict:
        """Return optimizer parameters. All trainable parameters in a single group."""
        return self.parameters()

    def reset(self):
        """Clear observation and action queues."""
        self._queues = {
            OBS_STATE: deque(maxlen=self.config.n_obs_steps),
            ACTION: deque(maxlen=self.config.n_action_steps),
        }
        if self.config.image_features:
            self._queues[OBS_IMAGES] = deque(maxlen=self.config.n_obs_steps)

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs: Unpack[ActionSelectKwargs]) -> Tensor:
        """Predict a chunk of actions given observations."""
        batch = {k: torch.stack(list(self._queues[k]), dim=1) for k in batch if k in self._queues}

        vlm_inputs = self._prepare_vlm_inputs(batch)
        n_cameras = vlm_inputs.pop("n_cameras", 1)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            last_hidden = self._forward_encoder(
                input_ids=vlm_inputs["input_ids"],
                pixel_values=vlm_inputs["pixel_values"],
                attention_mask=vlm_inputs.get("attention_mask"),
                n_cameras=n_cameras,
            )

        with torch.autocast("cuda", enabled=False):
            action_queries = last_hidden.float()[:, -self.chunk_size :, :]
            pred_actions = self.action_head(action_queries)
            pred_actions = torch.nan_to_num(pred_actions, nan=0.0, posinf=1.0, neginf=-1.0)
            if self.config.clip_action:
                pred_actions = pred_actions.clamp(
                    -self.config.clip_action_range, self.config.clip_action_range
                )

        return pred_actions[:, : self.config.n_action_steps]

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], **kwargs: Unpack[ActionSelectKwargs]) -> Tensor:
        """Select a single action given observations, with action caching."""
        if ACTION in batch:
            batch.pop(ACTION)

        if self.config.image_features:
            batch = dict(batch)
            batch[OBS_IMAGES] = torch.stack([batch[key] for key in self.config.image_features], dim=1)

        self._queues = populate_queues(self._queues, batch)

        if len(self._queues[ACTION]) == 0:
            actions = self.predict_action_chunk(batch, **kwargs)
            self._queues[ACTION].extend(actions.transpose(0, 1))

        action = self._queues[ACTION].popleft()
        action = torch.nan_to_num(action, nan=0.0, posinf=1.0, neginf=-1.0)
        if self.config.clip_action:
            action = action.clamp(-self.config.clip_action_range, self.config.clip_action_range)
        return action

    def forward(self, batch: dict[str, Tensor], reduction: str = "mean") -> tuple[Tensor, dict | None]:
        """Training forward pass. Computes L1 loss between predicted and target actions."""
        if self.config.image_features:
            batch = dict(batch)
            first_key = next(iter(self.config.image_features))
            first_img = batch[first_key]

            if first_img.dim() == 4:
                images = [batch[key].unsqueeze(1) for key in self.config.image_features]
            else:
                images = [batch[key] for key in self.config.image_features]

            batch[OBS_IMAGES] = torch.stack(images, dim=2)

        vlm_inputs = self._prepare_vlm_inputs(batch)
        n_cameras = vlm_inputs.pop("n_cameras", 1)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            last_hidden = self._forward_encoder(
                input_ids=vlm_inputs["input_ids"],
                pixel_values=vlm_inputs["pixel_values"],
                attention_mask=vlm_inputs.get("attention_mask"),
                n_cameras=n_cameras,
            )

        if not torch.isfinite(last_hidden).all():
            nonfinite = self._count_nonfinite(last_hidden)
            logger.warning(
                "Non-finite Florence2 encoder outputs detected: %s / %s. Skipping this training step.",
                nonfinite,
                last_hidden.numel(),
            )
            zero_loss = torch.zeros((), device=last_hidden.device, dtype=torch.float32, requires_grad=True)
            return zero_loss, {"action_loss": 0.0, "skipped_nonfinite_encoder": 1.0}

        with torch.autocast("cuda", enabled=False):
            last_hidden_f32 = last_hidden.float()
            action_queries = last_hidden_f32[:, -self.chunk_size :, :]
            pred_actions = self.action_head(action_queries)

            # Get target actions
            actions_target = batch[ACTION]
            if actions_target.dim() == 3:
                actions_target = actions_target[:, -self.chunk_size :, :]
            actions_target = actions_target.float()

            if not torch.isfinite(pred_actions).all():
                nonfinite = self._count_nonfinite(pred_actions)
                logger.warning(
                    "Non-finite predicted actions detected: %s / %s. Skipping this training step.",
                    nonfinite,
                    pred_actions.numel(),
                )
                zero_loss = torch.zeros((), device=pred_actions.device, dtype=torch.float32, requires_grad=True)
                return zero_loss, {"action_loss": 0.0, "skipped_nonfinite_pred_actions": 1.0}

            if not torch.isfinite(actions_target).all():
                nonfinite = self._count_nonfinite(actions_target)
                logger.warning(
                    "Non-finite target actions detected: %s / %s. Skipping this training step.",
                    nonfinite,
                    actions_target.numel(),
                )
                zero_loss = torch.zeros((), device=actions_target.device, dtype=torch.float32, requires_grad=True)
                return zero_loss, {"action_loss": 0.0, "skipped_nonfinite_targets": 1.0}

            # Apply padding mask if provided
            if "action_is_pad" in batch:
                is_pad = batch["action_is_pad"][:, -self.chunk_size :]
                valid_mask = ~is_pad
                if valid_mask.any():
                    action_loss = self.l1_loss(pred_actions[valid_mask], actions_target[valid_mask])
                else:
                    action_loss = torch.tensor(0.0, device=pred_actions.device, dtype=torch.float32)
            else:
                action_loss = self.l1_loss(pred_actions, actions_target)

        if not torch.isfinite(action_loss):
            logger.warning("Non-finite action loss detected. Skipping this training step.")
            action_loss = torch.zeros((), device=pred_actions.device, dtype=torch.float32, requires_grad=True)
            return action_loss, {"action_loss": 0.0, "skipped_nonfinite_loss": 1.0}

        info = {"action_loss": action_loss.item()}
        return action_loss, info
