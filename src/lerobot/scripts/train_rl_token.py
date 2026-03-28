#!/usr/bin/env python
"""
Two-phase RL Token training script.

Phase 1 (encoder_decoder): Train encoder-decoder on reconstruction loss.
  - Loads frozen SmolVLA, creates RLTokenEncoderDecoder.
  - Optimizes only encoder-decoder params.
  - Loss = MSE reconstruction of image embeddings through the RL token bottleneck.

Phase 2 (actor): Train actor to mimic VLA reference actions using RL token.
  - Loads frozen SmolVLA + frozen trained encoder.
  - Optimizes only actor params.
  - Loss = MSE between actor predictions and VLA reference action chunks.

Usage:
    # Phase 1
    python src/lerobot/scripts/train_rl_token.py \
      --phase=encoder_decoder \
      --vla_path=./libero_ckpt/smolvla_xxx \
      --dataset.repo_id=lerobot/aloha_sim_transfer_cube_scripted \
      --steps=20000 --batch_size=16

    # Phase 2
    python src/lerobot/scripts/train_rl_token.py \
      --phase=actor \
      --vla_path=./libero_ckpt/smolvla_xxx \
      --encoder_path=./output/rlt_encoder/best.pt \
      --dataset.repo_id=lerobot/aloha_sim_transfer_cube_scripted \
      --steps=20000 --batch_size=16

    # With pre-training VLA eval (requires --env config)
    python src/lerobot/scripts/train_rl_token.py \
      --phase=encoder_decoder \
      --vla_path=./libero_ckpt/smolvla_xxx \
      --eval_vla_episodes=10 \
      --dataset.repo_id=lerobot/aloha_sim_transfer_cube_scripted \
      --env.type=libero --env.task=... \
      --steps=20000 --batch_size=16
"""
#pretrain vla
#lerobot-train   --dataset.repo_id=lerobot/aloha_sim_transfer_cube_scripted   --policy.type=smolvla   --policy.load_vlm_weights=true   --policy.vlm_model_name=HuggingFaceTB/SmolVLM2-256M-Video-Instruct   --policy.chunk_size=50   --policy.n_action_steps=50   --steps=10000   --batch_size=16   --eval_freq=10000   --save_freq=10000   --env.type=aloha   --env.task=AlohaTransferCube-v0   --eval.batch_size=5   --eval.n_episodes=20   --wandb.enable=true   --wandb.project=aloha_smolvla   --policy.push_to_hub=false   --output_dir=./libero_ckpt/smolvla_aloha_v3_50_nofreeze_masknorm10_256M   --policy.optimizer_weight_decay=1e-4   --policy.scheduler_decay_steps=10000   --policy.scheduler_warmup_steps=500   --dataset.image_transforms.center_crop_to_square=true   --policy.freeze_vision_encoder=false --policy.train_expert_only=false

# train encoder decoder
#python src/lerobot/scripts/train_rl_token.py   --phase=encoder_decoder   --vla_path=./libero_ckpt/smolvla_aloha_v3_50_nofreeze_masknorm10_256M/checkpoints/010000/pretrained_model   --rlt_output_dir=./output/rlt_aloha   --eval_vla_episodes=10   --rlt_lr=1e-4   --enc_dec_num_layers=2   --enc_dec_num_heads=4   --dataset.repo_id=lerobot/aloha_sim_transfer_cube_scripted   --dataset.image_transforms.center_crop_to_square=true   --policy.type=smolvla   --policy.vlm_model_name=HuggingFaceTB/SmolVLM2-256M-Video-Instruct   --policy.chunk_size=50   --policy.n_action_steps=50   --policy.freeze_vision_encoder=false   --policy.train_expert_only=false   --policy.load_vlm_weights=true   --steps=20000   --batch_size=16   --log_freq=100   --save_freq=5000   --env.type=aloha   --env.task=AlohaTransferCube-v0   --eval.batch_size=5   --eval.n_episodes=10  --policy.push_to_hub=false

# train actor
#python src/lerobot/scripts/train_rl_token.py   --phase=actor   --vla_path=./libero_ckpt/smolvla_aloha_v3_50_nofreeze_masknorm10_256M/checkpoints/010000/pretrained_model   --encoder_path=./output/rlt_aloha/encoder_decoder/encoder_decoder_step20000.pt   --rlt_output_dir=./output/rlt_aloha   --actor_chunk_size=10   --dataset.repo_id=lerobot/aloha_sim_transfer_cube_scripted   --dataset.image_transforms.center_crop_to_square=true   --policy.type=smolvla   --policy.vlm_model_name=HuggingFaceTB/SmolVLM2-256M-Video-Instruct   --policy.chunk_size=50   --policy.n_action_steps=50   --policy.freeze_vision_encoder=false   --policy.train_expert_only=false   --policy.load_vlm_weights=true   --steps=20000   --batch_size=16   --log_freq=100   --save_freq=5000   --eval_freq=5000   --env.type=aloha   --env.task=AlohaTransferCube-v0   --eval.batch_size=5   --eval.n_episodes=10 --policy.push_to_hub=false



import argparse
import logging
import sys
import time
from pathlib import Path
from pprint import pformat

import torch
from termcolor import colored

from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_dataset
from lerobot.datasets.sampler import EpisodeAwareSampler
from lerobot.datasets.utils import cycle
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.rlt.configuration_rlt import RLTConfig
from lerobot.policies.rlt.modeling_rlt import RLTActorEvalWrapper, RLTPolicy
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS
from lerobot.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import format_big_number, init_logging


def _parse_rlt_args():
    """Extract RLT-specific args before draccus sees them."""
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--phase", type=str, required=True, choices=["encoder_decoder", "actor"])
    pre.add_argument("--vla_path", type=str, required=True)
    pre.add_argument("--encoder_path", type=str, default=None)
    pre.add_argument("--rlt_output_dir", type=str, default="./output/rlt")
    # VLA pre-eval: set >0 to run VLA eval before training
    pre.add_argument("--eval_vla_episodes", type=int, default=0)
    # RLT config overrides
    pre.add_argument("--enc_dec_num_layers", type=int, default=2)
    pre.add_argument("--enc_dec_num_heads", type=int, default=4)
    pre.add_argument("--enc_dec_dropout", type=float, default=0.1)
    pre.add_argument("--actor_hidden_dim", type=int, default=256)
    pre.add_argument("--actor_chunk_size", type=int, default=10)
    pre.add_argument("--ref_action_dropout", type=float, default=0.5)
    pre.add_argument("--rlt_lr", type=float, default=1e-4)

    known, remaining = pre.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    return known


def build_rlt_config(args) -> RLTConfig:
    return RLTConfig(
        enc_dec_num_layers=args.enc_dec_num_layers,
        enc_dec_num_heads=args.enc_dec_num_heads,
        enc_dec_dropout=args.enc_dec_dropout,
        actor_hidden_dim=args.actor_hidden_dim,
        actor_chunk_size=args.actor_chunk_size,
        ref_action_dropout=args.ref_action_dropout,
        lr=args.rlt_lr,
    )


def _get_fallback_task(dataset) -> str | None:
    """Extract the first task name from the dataset for envs without language instructions."""
    try:
        if dataset.meta.tasks is not None and len(dataset.meta.tasks) > 0:
            task = dataset.meta.tasks.index[0]
            logging.info(f"Loaded fallback_task from dataset: '{task}'")
            return task
    except Exception as e:
        logging.warning(f"Could not load fallback task from dataset: {e}")
    return None


def run_vla_eval(
    vla_policy,
    cfg: TrainPipelineConfig,
    preprocessor,
    postprocessor,
    n_episodes: int,
    device,
    output_dir: Path,
    fallback_task: str | None = None,
    wandb_logger=None,
):
    """Run evaluation of the base VLA to verify it loads and works correctly."""
    if cfg.env is None:
        logging.warning("--eval_vla_episodes > 0 but no --env config provided; skipping VLA eval.")
        return

    from lerobot.envs.factory import make_env, make_env_pre_post_processors
    from lerobot.envs.utils import close_envs
    from lerobot.scripts.lerobot_eval import eval_policy_all

    logging.info(colored(
        f"Running pre-training VLA eval ({n_episodes} episodes) to verify model loading...",
        "cyan", attrs=["bold"]
    ))
    if fallback_task:
        logging.info(f"Using fallback_task: '{fallback_task}'")

    eval_batch_size = min(n_episodes, cfg.eval.batch_size)
    eval_env = make_env(cfg.env, n_envs=eval_batch_size, use_async_envs=cfg.eval.use_async_envs)

    env_preprocessor, env_postprocessor = make_env_pre_post_processors(
        env_cfg=cfg.env,
        policy_cfg=cfg.policy,
        image_transforms=cfg.dataset.image_transforms,
    )

    vla_policy.eval()
    with torch.no_grad():
        eval_results = eval_policy_all(
            envs=eval_env,
            policy=vla_policy,
            env_preprocessor=env_preprocessor,
            env_postprocessor=env_postprocessor,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            n_episodes=n_episodes,
            videos_dir=output_dir / "eval" / "vla_pretrain",
            max_episodes_rendered=min(4, n_episodes),
            start_seed=cfg.seed,
            max_parallel_tasks=cfg.env.max_parallel_tasks,
            fallback_task=fallback_task,
        )

    agg = eval_results["overall"]
    logging.info(
        colored("VLA pre-training eval:", "cyan", attrs=["bold"])
        + f" success={agg['pc_success']:.1f}%"
        + f" reward={agg['avg_sum_reward']:.3f}"
    )

    for suite, suite_info in eval_results.items():
        if suite != "overall":
            logging.info(f"  [{suite}]: {suite_info}")

    if wandb_logger:
        wandb_logger.log_dict({
            "eval/vla_pretrain_success": agg["pc_success"],
            "eval/vla_pretrain_reward": agg["avg_sum_reward"],
        }, step=0, mode="eval")

    close_envs(eval_env)
    return eval_results


def run_actor_vs_vla_eval(
    rlt_policy: RLTPolicy,
    cfg: TrainPipelineConfig,
    preprocessor,
    postprocessor,
    step: int,
    output_dir: Path,
    fallback_task: str | None = None,
    wandb_logger=None,
):
    """Eval both the RLT actor and base VLA in the environment for comparison."""
    if cfg.env is None:
        logging.warning("eval_freq > 0 but no --env config provided; skipping eval.")
        return

    from lerobot.envs.factory import make_env, make_env_pre_post_processors
    from lerobot.envs.utils import close_envs
    from lerobot.scripts.lerobot_eval import eval_policy_all

    n_episodes = cfg.eval.n_episodes
    eval_batch_size = min(n_episodes, cfg.eval.batch_size)
    eval_env = make_env(cfg.env, n_envs=eval_batch_size, use_async_envs=cfg.eval.use_async_envs)

    env_preprocessor, env_postprocessor = make_env_pre_post_processors(
        env_cfg=cfg.env,
        policy_cfg=cfg.policy,
        image_transforms=cfg.dataset.image_transforms,
    )

    # ── Eval Actor ──
    logging.info(colored(f"[Eval step {step}] Running RLT actor eval...", "cyan", attrs=["bold"]))
    actor_wrapper = RLTActorEvalWrapper(rlt_policy)
    actor_wrapper.eval()
    with torch.no_grad():
        actor_results = eval_policy_all(
            envs=eval_env,
            policy=actor_wrapper,
            env_preprocessor=env_preprocessor,
            env_postprocessor=env_postprocessor,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            n_episodes=n_episodes,
            videos_dir=output_dir / "eval" / f"actor_step{step}",
            max_episodes_rendered=min(4, n_episodes),
            start_seed=cfg.seed,
            max_parallel_tasks=cfg.env.max_parallel_tasks,
            fallback_task=fallback_task,
        )

    # ── Eval VLA ──
    logging.info(colored(f"[Eval step {step}] Running VLA baseline eval...", "cyan", attrs=["bold"]))
    rlt_policy.vla_policy.eval()
    with torch.no_grad():
        vla_results = eval_policy_all(
            envs=eval_env,
            policy=rlt_policy.vla_policy,
            env_preprocessor=env_preprocessor,
            env_postprocessor=env_postprocessor,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            n_episodes=n_episodes,
            videos_dir=output_dir / "eval" / f"vla_step{step}",
            max_episodes_rendered=min(4, n_episodes),
            start_seed=cfg.seed,
            max_parallel_tasks=cfg.env.max_parallel_tasks,
            fallback_task=fallback_task,
        )

    actor_agg = actor_results["overall"]
    vla_agg = vla_results["overall"]

    logging.info(
        colored(f"[Eval step {step}]", "cyan", attrs=["bold"])
        + f" Actor: success={actor_agg['pc_success']:.1f}% reward={actor_agg['avg_sum_reward']:.3f}"
        + f" | VLA: success={vla_agg['pc_success']:.1f}% reward={vla_agg['avg_sum_reward']:.3f}"
        + f" | Delta: {actor_agg['pc_success'] - vla_agg['pc_success']:+.1f}%"
    )

    if wandb_logger:
        wandb_logger.log_dict({
            "eval/actor_success": actor_agg["pc_success"],
            "eval/actor_reward": actor_agg["avg_sum_reward"],
            "eval/vla_success": vla_agg["pc_success"],
            "eval/vla_reward": vla_agg["avg_sum_reward"],
            "eval/actor_vs_vla_delta": actor_agg["pc_success"] - vla_agg["pc_success"],
        }, step=step, mode="eval")

    close_envs(eval_env)
    return actor_results, vla_results


@parser.wrap()
def train_rlt(cfg: TrainPipelineConfig):
    rlt_args = _RLT_ARGS
    phase = rlt_args.phase
    vla_path = rlt_args.vla_path
    encoder_path = rlt_args.encoder_path
    output_dir = Path(rlt_args.rlt_output_dir) / phase

    cfg.validate()
    init_logging()

    logging.info(colored(f"RLT Phase: {phase}", "green", attrs=["bold"]))
    logging.info(f"VLA path: {vla_path}")
    if encoder_path:
        logging.info(f"Encoder path: {encoder_path}")
    logging.info(pformat(cfg.to_dict()))

    wandb_logger = None
    if cfg.wandb.enable and cfg.wandb.project:
        from lerobot.rl.wandb_utils import WandBLogger
        wandb_logger = WandBLogger(cfg)
    else:
        logging.info(colored("W&B disabled, logging locally.", "yellow"))

    if cfg.seed is not None:
        set_seed(cfg.seed)

    device = torch.device(cfg.policy.device if cfg.policy.device else "cuda")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    # ── Dataset ──
    logging.info("Creating dataset")
    dataset = make_dataset(cfg)

    # ── VLA Policy ──
    logging.info(f"Loading VLA from {vla_path}")
    cfg.policy.pretrained_path = vla_path
    vla_policy = make_policy(
        cfg=cfg.policy,
        ds_meta=dataset.meta,
        rename_map=cfg.rename_map,
        dataset_config=cfg.dataset,
    )
    vla_policy.to(device)

    # ── Processors ──
    processor_kwargs = {"dataset_stats": dataset.meta.stats}
    if cfg.policy.pretrained_path is not None:
        processor_kwargs["preprocessor_overrides"] = {
            "device_processor": {"device": device.type},
            "normalizer_processor": {
                "stats": dataset.meta.stats,
                "features": {**vla_policy.config.input_features, **vla_policy.config.output_features},
                "norm_map": vla_policy.config.normalization_mapping,
            },
        }
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        **processor_kwargs,
    )

    # ── Determine dimensions ──
    action_dim = vla_policy.config.action_feature.shape[0]
    state_dim = vla_policy.config.max_state_dim
    logging.info(f"action_dim={action_dim}, state_dim={state_dim}")

    # ── Output dir ──
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info(colored(f"Output dir: {output_dir}", "yellow", attrs=["bold"]))

    # ── Fallback task (for envs without language instructions, e.g. ALOHA) ──
    fallback_task = _get_fallback_task(dataset)

    # ── Pre-training VLA eval ──
    if rlt_args.eval_vla_episodes > 0:
        run_vla_eval(
            vla_policy=vla_policy,
            cfg=cfg,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            n_episodes=rlt_args.eval_vla_episodes,
            device=device,
            output_dir=output_dir,
            fallback_task=fallback_task,
            wandb_logger=wandb_logger,
        )

    # ── RLT Config & Policy ──
    rlt_config = build_rlt_config(rlt_args)
    rlt_policy = RLTPolicy(
        vla_policy=vla_policy,
        config=rlt_config,
        action_dim=action_dim,
        state_dim=state_dim,
    )
    rlt_policy.to(device)

    num_enc_dec_params = sum(p.numel() for p in rlt_policy.encoder_decoder.parameters())
    num_actor_params = sum(p.numel() for p in rlt_policy.actor.parameters())
    logging.info(f"Encoder-Decoder params: {num_enc_dec_params} ({format_big_number(num_enc_dec_params)})")
    logging.info(f"Actor params: {num_actor_params} ({format_big_number(num_actor_params)})")

    # ── Phase-specific setup ──
    if phase == "encoder_decoder":
        trainable_params = list(rlt_policy.encoder_decoder.parameters())
        for p in rlt_policy.actor.parameters():
            p.requires_grad = False
    elif phase == "actor":
        if encoder_path is None:
            raise ValueError("--encoder_path is required for phase=actor")
        logging.info(f"Loading encoder-decoder from {encoder_path}")
        enc_dec_state = torch.load(encoder_path, map_location=device, weights_only=True)
        rlt_policy.encoder_decoder.load_state_dict(enc_dec_state)
        for p in rlt_policy.encoder_decoder.parameters():
            p.requires_grad = False
        rlt_policy.encoder_decoder.eval()
        trainable_params = list(rlt_policy.actor.parameters())

    num_trainable = sum(p.numel() for p in trainable_params)
    logging.info(
        colored(f"Trainable parameters ({phase}):", "green")
        + f" {num_trainable} ({format_big_number(num_trainable)})"
    )

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=rlt_config.lr,
        weight_decay=rlt_config.weight_decay,
    )

    def lr_lambda(current_step):
        if current_step < rlt_config.warmup_steps:
            return current_step / max(1, rlt_config.warmup_steps)
        return 1.0
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ── DataLoader ──
    if hasattr(cfg.policy, "drop_n_last_frames"):
        sampler = EpisodeAwareSampler(
            dataset.meta.episodes["dataset_from_index"],
            dataset.meta.episodes["dataset_to_index"],
            episode_indices_to_use=dataset.episodes,
            drop_n_last_frames=cfg.policy.drop_n_last_frames,
            shuffle=True,
        )
        shuffle = False
    else:
        sampler = None
        shuffle = True

    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.num_workers,
        batch_size=cfg.batch_size,
        shuffle=shuffle and not cfg.dataset.streaming,
        sampler=sampler,
        pin_memory=device.type == "cuda",
        drop_last=False,
        prefetch_factor=2 if cfg.num_workers > 0 else None,
    )
    dl_iter = cycle(dataloader)

    # ── Metrics ──
    metric_names = {
        "encoder_decoder": {"recon_loss": AverageMeter("recon", ":.4f"),
                            "z_rl_norm": AverageMeter("z_norm", ":.3f")},
        "actor": {"actor_loss": AverageMeter("actor", ":.4f")},
    }
    train_metrics = {
        **metric_names[phase],
        "grad_norm": AverageMeter("grdn", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
        "update_s": AverageMeter("updt_s", ":.3f"),
        "dataloading_s": AverageMeter("data_s", ":.3f"),
    }
    train_tracker = MetricsTracker(
        cfg.batch_size, dataset.num_frames, dataset.num_episodes, train_metrics,
    )

    logging.info(f"Starting {phase} training for {cfg.steps} steps")

    # ── Training loop ──
    if phase == "encoder_decoder":
        rlt_policy.encoder_decoder.train()
    else:
        rlt_policy.actor.train()

    best_loss = float("inf")
    step = 0

    for _ in range(cfg.steps):
        t0 = time.perf_counter()
        batch = next(dl_iter)
        batch = preprocessor(batch)
        t1 = time.perf_counter()

        images, img_masks = vla_policy.prepare_images(batch)

        if phase == "encoder_decoder":
            loss, info = rlt_policy.forward_encoder_decoder(images)
            train_tracker.recon_loss = loss.item()
            train_tracker.z_rl_norm = info["z_rl_norm"]
        else:
            lang_tokens = batch[OBS_LANGUAGE_TOKENS]
            lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
            state_padded = vla_policy.prepare_state(batch)

            loss, info = rlt_policy.forward_actor(
                images, img_masks, lang_tokens, lang_masks,
                state_raw=state_padded,
                state_padded=state_padded,
            )
            train_tracker.actor_loss = loss.item()

        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, rlt_config.grad_clip_norm)
        optimizer.step()
        lr_scheduler.step()

        t2 = time.perf_counter()
        train_tracker.grad_norm = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
        train_tracker.lr = optimizer.param_groups[0]["lr"]
        train_tracker.update_s = t2 - t1
        train_tracker.dataloading_s = t1 - t0

        step += 1
        train_tracker.step()

        # ── Logging ──
        is_log_step = cfg.log_freq > 0 and step % cfg.log_freq == 0
        if is_log_step:
            logging.info(train_tracker)
            if wandb_logger:
                log_dict = train_tracker.to_dict()
                log_dict.update(info)
                wandb_logger.log_dict(log_dict, step)
            train_tracker.reset_averages()

        # ── Save best + periodic ──
        is_save_step = step % cfg.save_freq == 0 or step == cfg.steps
        if is_save_step:
            current_loss = loss.item()
            if phase == "encoder_decoder":
                save_path = output_dir / f"encoder_decoder_step{step}.pt"
                torch.save(rlt_policy.encoder_decoder.state_dict(), save_path)
                if current_loss < best_loss:
                    best_loss = current_loss
                    best_path = output_dir / "encoder_decoder_best.pt"
                    torch.save(rlt_policy.encoder_decoder.state_dict(), best_path)
                    logging.info(f"New best encoder-decoder (loss={best_loss:.6f}) saved to {best_path}")
            else:
                save_path = output_dir / f"actor_step{step}.pt"
                torch.save(rlt_policy.actor.state_dict(), save_path)
                if current_loss < best_loss:
                    best_loss = current_loss
                    best_path = output_dir / "actor_best.pt"
                    torch.save(rlt_policy.actor.state_dict(), best_path)
                    logging.info(f"New best actor (loss={best_loss:.6f}) saved to {best_path}")
            logging.info(f"Checkpoint saved: {save_path}")

        # ── Eval: actor vs VLA comparison ──
        is_eval_step = cfg.eval_freq > 0 and step % cfg.eval_freq == 0 and phase == "actor"
        if is_eval_step and cfg.env is not None:
            rlt_policy.actor.eval()
            run_actor_vs_vla_eval(
                rlt_policy=rlt_policy,
                cfg=cfg,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                step=step,
                output_dir=output_dir,
                fallback_task=fallback_task,
                wandb_logger=wandb_logger,
            )
            rlt_policy.actor.train()

    logging.info(f"Training complete. Best loss: {best_loss:.6f}")


_RLT_ARGS = None


def main():
    global _RLT_ARGS
    _RLT_ARGS = _parse_rlt_args()

    from lerobot.utils.import_utils import register_third_party_plugins
    register_third_party_plugins()
    train_rlt()


if __name__ == "__main__":
    main()
