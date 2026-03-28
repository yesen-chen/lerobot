"""
RL Token (RLT) implementation on top of a frozen SmolVLA.

Reference: "RL Token: Bootstrapping Online RL with Vision-Language-Action Models"
           https://www.pi.website/research/rlt

Four core components:
  1. RLTokenEncoderDecoder - encoder-decoder transformer that compresses image
     embeddings into a single RL token via an information bottleneck.
  2. RLTActor - lightweight three-branch MLP that predicts actions from the
     RL token, proprioceptive state, VLA reference actions, and sampling noise.
  3. RLTPolicy - wrapper that holds a frozen SmolVLA plus the above modules
     and exposes two-phase training interfaces.
  4. RLTActorEvalWrapper - thin wrapper making RLTPolicy compatible with
     the LeRobot eval rollout loop (select_action / reset / action queue).
"""

import logging
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F

from lerobot.policies.rlt.configuration_rlt import RLTConfig
from lerobot.policies.utils import populate_queues
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Encoder-Decoder for RL Token extraction
# ---------------------------------------------------------------------------

class RLTokenEncoderDecoder(nn.Module):
    """Encoder-decoder transformer that produces a compact RL token from
    image embeddings and trains via autoregressive reconstruction.

    Encoder: bidirectional self-attention over [sg(z_1), ..., sg(z_N), e_rl].
             The output at the last position (e_rl) is the RL token.
    Decoder: causal self-attention over [z_rl, sg(z_1), ..., sg(z_{N-1})].
             Position j predicts sg(z_{j+1}) through a linear projection.
    """

    def __init__(self, embed_dim: int, config: RLTConfig):
        super().__init__()
        self.embed_dim = embed_dim
        self.config = config

        num_layers = config.enc_dec_num_layers
        num_heads = config.enc_dec_num_heads
        dropout = config.enc_dec_dropout
        ff_dim = embed_dim * config.enc_dec_ff_dim_multiplier

        # Learnable RL token embedding appended to encoder input
        self.rl_token_embedding = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

        # Encoder (bidirectional)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Decoder (causal)
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=num_layers)

        # Normalize RL token output to prevent norm explosion
        self.encoder_out_norm = nn.LayerNorm(embed_dim)

        # Output projection for reconstruction
        self.output_proj = nn.Linear(embed_dim, embed_dim)

    def encode(self, image_embs: torch.Tensor) -> torch.Tensor:
        """Encode image embeddings into a single RL token.

        Args:
            image_embs: (B, N, D) detached image embeddings from vision encoder.

        Returns:
            z_rl: (B, D) the RL token.
        """
        B = image_embs.shape[0]
        # Append learnable e_rl at the end: [sg(z_1), ..., sg(z_N), e_rl]
        e_rl = self.rl_token_embedding.expand(B, -1, -1)
        encoder_input = torch.cat([image_embs, e_rl], dim=1)  # (B, N+1, D)

        encoder_output = self.encoder(encoder_input)  # (B, N+1, D)
        z_rl = encoder_output[:, -1, :]  # (B, D) - last position
        z_rl = self.encoder_out_norm(z_rl)
        return z_rl

    def decode(self, z_rl: torch.Tensor, image_embs: torch.Tensor) -> torch.Tensor:
        """Autoregressively reconstruct image embeddings from the RL token.

        Args:
            z_rl: (B, D) RL token from encoder.
            image_embs: (B, N, D) detached image embeddings (teacher forcing targets).

        Returns:
            predictions: (B, N, D) predicted embeddings, where position j predicts z_{j+1}.
        """
        # Decoder input: [z_rl, sg(z_1), sg(z_2), ..., sg(z_{N-1})]
        z_rl_expanded = z_rl.unsqueeze(1)  # (B, 1, D)
        decoder_input = torch.cat([z_rl_expanded, image_embs[:, :-1, :]], dim=1)  # (B, N, D)

        N = decoder_input.shape[1]
        causal_mask = torch.triu(
            torch.ones(N, N, device=decoder_input.device, dtype=torch.bool), diagonal=1
        )  # True = masked positions

        decoder_output = self.decoder(decoder_input, mask=causal_mask)  # (B, N, D)
        predictions = self.output_proj(decoder_output)  # (B, N, D)
        return predictions

    def forward(self, image_embs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Full forward: encode to RL token, decode to reconstruct, compute loss.

        Args:
            image_embs: (B, N, D) detached image embeddings.

        Returns:
            z_rl: (B, D) RL token.
            recon_loss: scalar reconstruction MSE loss.
        """
        z_rl = self.encode(image_embs)
        predictions = self.decode(z_rl, image_embs)

        # Targets: sg(z_1), sg(z_2), ..., sg(z_N)
        targets = image_embs  # (B, N, D) - already detached
        recon_loss = F.mse_loss(predictions, targets)
        return z_rl, recon_loss


# ---------------------------------------------------------------------------
# 2. Actor MLP (three-branch)
# ---------------------------------------------------------------------------

class RLTActor(nn.Module):
    """Lightweight three-branch MLP actor.

    Branch 1 (perception):   z_rl → Linear → SiLU
    Branch 2 (proprioception): state → Linear → SiLU
    Branch 3 (action+noise): [ref_action, noise] flattened → Linear → SiLU
    Merge: cat(b1, b2, b3) → Linear → SiLU → Linear → reshape (B, C, action_dim)
    """

    def __init__(
        self,
        embed_dim: int,
        state_dim: int,
        action_dim: int,
        config: RLTConfig,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.chunk_size = config.actor_chunk_size
        self.ref_action_dropout = config.ref_action_dropout

        H = config.actor_hidden_dim
        action_noise_dim = self.chunk_size * action_dim * 2  # ref_action + noise flattened

        self.branch_rl = nn.Sequential(nn.Linear(embed_dim, H), nn.SiLU())
        self.branch_state = nn.Sequential(nn.Linear(state_dim, H), nn.SiLU())
        self.branch_action_noise = nn.Sequential(nn.Linear(action_noise_dim, H), nn.SiLU())

        self.merge = nn.Sequential(
            nn.Linear(3 * H, H),
            nn.SiLU(),
            nn.Linear(H, self.chunk_size * action_dim),
        )

    def forward(
        self,
        z_rl: torch.Tensor,
        state: torch.Tensor,
        ref_action: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            z_rl: (B, D) RL token.
            state: (B, state_dim) proprioceptive state.
            ref_action: (B, C, action_dim) VLA reference actions (already sliced).
            noise: (B, C, action_dim) sampling noise (already sliced).

        Returns:
            predicted_action: (B, C, action_dim)
        """
        B = z_rl.shape[0]

        # Ref action dropout: zero out entire samples with probability p
        if self.training and self.ref_action_dropout > 0:
            dropout_mask = (
                torch.rand(B, 1, 1, device=ref_action.device) > self.ref_action_dropout
            ).float()
            ref_action = ref_action * dropout_mask

        b1 = self.branch_rl(z_rl)  # (B, H)
        b2 = self.branch_state(state)  # (B, H)
        action_noise_flat = torch.cat([
            ref_action.reshape(B, -1),
            noise.reshape(B, -1),
        ], dim=-1)  # (B, C*action_dim*2)
        b3 = self.branch_action_noise(action_noise_flat)  # (B, H)

        merged = torch.cat([b1, b2, b3], dim=-1)  # (B, 3*H)
        out = self.merge(merged)  # (B, C*action_dim)
        return out.reshape(B, self.chunk_size, self.action_dim)


# ---------------------------------------------------------------------------
# 3. RLTPolicy wrapper
# ---------------------------------------------------------------------------

class RLTPolicy(nn.Module):
    """Wrapper holding a frozen SmolVLA, an RL token encoder-decoder, and an actor.

    Provides two training phases:
      Phase 1 (encoder_decoder): train the encoder-decoder on reconstruction loss.
      Phase 2 (actor): freeze encoder, train actor to mimic VLA reference actions.
    """

    def __init__(
        self,
        vla_policy,
        config: RLTConfig,
        action_dim: int,
        state_dim: int,
    ):
        super().__init__()
        self.config = config
        self.action_dim = action_dim
        self.state_dim = state_dim

        # Frozen VLA - stored but not registered as submodule to avoid
        # optimizer picking up its parameters
        self.vla_policy = vla_policy
        self._freeze_vla()

        # Determine embed_dim from VLA
        self.embed_dim = vla_policy.model.vlm_with_expert.config.text_config.hidden_size
        logger.info(f"RLT embed_dim={self.embed_dim}, action_dim={action_dim}, state_dim={state_dim}")

        # Encoder-Decoder
        self.encoder_decoder = RLTokenEncoderDecoder(
            embed_dim=self.embed_dim,
            config=config,
        )

        # Actor
        self.actor = RLTActor(
            embed_dim=self.embed_dim,
            state_dim=state_dim,
            action_dim=action_dim,
            config=config,
        )

    def _freeze_vla(self):
        """Freeze all VLA parameters."""
        self.vla_policy.eval()
        for p in self.vla_policy.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def extract_image_embeddings(self, images: list[torch.Tensor]) -> torch.Tensor:
        """Extract raw image embeddings from the frozen VLA's vision encoder.

        Args:
            images: list of (B, C, H, W) tensors, one per camera.

        Returns:
            image_embs: (B, N_total, D) detached image embeddings.
        """
        all_img_embs = []
        for img in images:
            img_emb = self.vla_policy.model.vlm_with_expert.embed_image(img)
            all_img_embs.append(img_emb)
        return torch.cat(all_img_embs, dim=1).detach().float()

    @torch.no_grad()
    def get_vla_reference_actions(
        self,
        images: list[torch.Tensor],
        img_masks: list[torch.Tensor],
        lang_tokens: torch.Tensor,
        lang_masks: torch.Tensor,
        state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run full VLA inference to get reference actions and the noise used.

        Returns:
            ref_actions: (B, chunk_size, action_dim) unpadded VLA actions.
            noise: (B, chunk_size, action_dim) the noise used during sampling, also unpadded.
        """
        vla_model = self.vla_policy.model
        bsize = state.shape[0]
        device = state.device

        # Explicitly sample noise so we can pass the same noise to the actor
        actions_shape = (bsize, vla_model.config.chunk_size, vla_model.config.max_action_dim)
        noise = vla_model.sample_noise(actions_shape, device)

        # Run VLA inference
        ref_actions = vla_model.sample_actions(
            images, img_masks, lang_tokens, lang_masks, state, noise=noise,
        )

        # Unpad to real action dimensions
        ref_actions = ref_actions[:, :, :self.action_dim]
        noise = noise[:, :, :self.action_dim]
        return ref_actions.detach(), noise.detach()

    def forward_encoder_decoder(
        self,
        images: list[torch.Tensor],
    ) -> tuple[torch.Tensor, dict]:
        """Phase 1 forward: compute reconstruction loss.

        Args:
            images: list of preprocessed (B, C, H, W) image tensors.

        Returns:
            recon_loss: scalar loss.
            info_dict: logging metrics.
        """
        image_embs = self.extract_image_embeddings(images)
        z_rl, recon_loss = self.encoder_decoder(image_embs)
        return recon_loss, {"recon_loss": recon_loss.item(), "z_rl_norm": z_rl.norm(dim=-1).mean().item()}

    def forward_actor(
        self,
        images: list[torch.Tensor],
        img_masks: list[torch.Tensor],
        lang_tokens: torch.Tensor,
        lang_masks: torch.Tensor,
        state_raw: torch.Tensor,
        state_padded: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """Phase 2 forward: compute actor loss (MSE vs VLA reference actions).

        Args:
            images: list of preprocessed image tensors.
            img_masks: list of image masks.
            lang_tokens: (B, L) language token ids.
            lang_masks: (B, L) language attention masks.
            state_raw: (B, state_dim) raw proprioceptive state for actor input.
            state_padded: (B, max_state_dim) padded state for VLA inference.

        Returns:
            actor_loss: scalar loss.
            info_dict: logging metrics.
        """
        C = self.config.actor_chunk_size

        # Get RL token (frozen encoder)
        image_embs = self.extract_image_embeddings(images)
        with torch.no_grad():
            z_rl = self.encoder_decoder.encode(image_embs)
            z_rl = z_rl.detach()

        # Get VLA reference actions and noise
        ref_actions, noise = self.get_vla_reference_actions(
            images, img_masks, lang_tokens, lang_masks, state_padded,
        )

        # Slice to actor chunk size
        ref_actions_sliced = ref_actions[:, :C, :]
        noise_sliced = noise[:, :C, :]

        # Actor forward
        predicted_actions = self.actor(z_rl, state_raw, ref_actions_sliced, noise_sliced)

        # MSE loss vs reference actions
        actor_loss = F.mse_loss(predicted_actions, ref_actions_sliced)
        return actor_loss, {
            "actor_loss": actor_loss.item(),
            "ref_action_mean": ref_actions_sliced.abs().mean().item(),
            "pred_action_mean": predicted_actions.abs().mean().item(),
        }


# ---------------------------------------------------------------------------
# 4. Eval wrapper - makes RLTPolicy usable in the LeRobot rollout loop
# ---------------------------------------------------------------------------

class RLTActorEvalWrapper(nn.Module):
    """Thin wrapper that gives RLTPolicy a select_action / reset interface
    compatible with the LeRobot eval rollout loop.

    Two inference modes:
      "with_vla"   - Run full VLA to get ref_action + noise, then actor edits them.
                     Slower but leverages VLA guidance. (default)
      "independent" - ref_action = zeros, noise = random. Actor generates actions
                     purely from z_rl + state, no VLA forward needed.
                     Much faster, tests the actor's standalone capability.
    """

    MODES = ("with_vla", "independent")

    def __init__(self, rlt_policy: RLTPolicy, mode: str = "with_vla"):
        super().__init__()
        assert mode in self.MODES, f"mode must be one of {self.MODES}, got '{mode}'"
        self.rlt_policy = rlt_policy
        self.vla = rlt_policy.vla_policy
        self.mode = mode
        self.n_action_steps = rlt_policy.config.actor_chunk_size
        self._queues: dict[str, deque] = {}
        self.reset()

        from lerobot.policies.pretrained import PreTrainedPolicy
        PreTrainedPolicy.register(type(self))

    # ---- interface expected by eval_policy / rollout ----

    @property
    def config(self):
        return self.vla.config

    def reset(self):
        self._queues = {ACTION: deque(maxlen=self.n_action_steps)}

    @torch.no_grad()
    def select_action(self, batch, **kwargs):
        batch = self.vla._prepare_batch(batch)
        self._queues = populate_queues(self._queues, batch, exclude_keys=[ACTION])

        if len(self._queues[ACTION]) == 0:
            actions = self._get_action_chunk(batch)
            self._queues[ACTION].extend(actions.transpose(0, 1)[:self.n_action_steps])

        return self._queues[ACTION].popleft()

    def _get_action_chunk(self, batch) -> torch.Tensor:
        """Run actor inference to produce an action chunk."""
        C = self.rlt_policy.config.actor_chunk_size
        action_dim = self.rlt_policy.action_dim

        images, img_masks = self.vla.prepare_images(batch)
        state_padded = self.vla.prepare_state(batch)

        # RL token (always needed)
        image_embs = self.rlt_policy.extract_image_embeddings(images)
        z_rl = self.rlt_policy.encoder_decoder.encode(image_embs)

        B = z_rl.shape[0]
        device = z_rl.device

        if self.mode == "with_vla":
            lang_tokens = batch[OBS_LANGUAGE_TOKENS]
            lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
            ref_actions, noise = self.rlt_policy.get_vla_reference_actions(
                images, img_masks, lang_tokens, lang_masks, state_padded,
            )
            ref_sliced = ref_actions[:, :C, :]
            noise_sliced = noise[:, :C, :]
        else:
            # Independent mode: zero ref_action (mimics dropout=1.0), random noise
            ref_sliced = torch.zeros(B, C, action_dim, device=device)
            noise_sliced = torch.randn(B, C, action_dim, device=device)

        predicted = self.rlt_policy.actor(z_rl, state_padded, ref_sliced, noise_sliced)
        return predicted
