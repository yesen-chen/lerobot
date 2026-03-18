#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
QC-FQL (Chunked Flow Q-Learning) training script for SmolVLA-QL.

Dual-optimizer training loop:
  1. Critic update:  prepare_qc_batch → forward_critic → backward → optimizer_critic.step()
  2. EMA target:     model.update_target_value_heads()
  3. Actor update:   forward (flow matching BC) → backward → optimizer_policy.step()

Usage:
    python src/lerobot/scripts/lerobot_train_qc.py \
        --policy.type=smolvlaql \
        --policy.path=lerobot/smolvla_base \
        --dataset.repo_id=<your_dataset> \
        --batch_size=32 \
        --steps=100000
"""

import logging
import time
from pathlib import Path
from pprint import pformat

import torch
from termcolor import colored
from torch.optim.lr_scheduler import LRScheduler

from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_dataset
from lerobot.datasets.sampler import EpisodeAwareSampler
from lerobot.datasets.utils import cycle
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.envs.utils import close_envs
from lerobot.optim.optimizers import load_optimizer_state, save_optimizer_state
from lerobot.optim.schedulers import load_scheduler_state, save_scheduler_state
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.smolvlaql.utils import prepare_qc_batch
from lerobot.rl.wandb_utils import WandBLogger
from lerobot.scripts.lerobot_eval import eval_policy_all
from lerobot.utils.constants import PRETRAINED_MODEL_DIR, TRAINING_STATE_DIR
from lerobot.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.utils.random_utils import load_rng_state, save_rng_state, set_seed
from lerobot.utils.train_utils import (
    get_step_checkpoint_dir,
    get_step_identifier,
    load_training_step,
    save_training_step,
    update_last_checkpoint,
)
from lerobot.utils.utils import format_big_number, init_logging


def save_qc_checkpoint(
    checkpoint_dir: Path,
    step: int,
    cfg: TrainPipelineConfig,
    policy,
    optimizers: dict[str, torch.optim.Optimizer],
    scheduler: LRScheduler | None,
    preprocessor,
    postprocessor,
) -> None:
    pretrained_dir = checkpoint_dir / PRETRAINED_MODEL_DIR
    policy.save_pretrained(pretrained_dir)
    cfg.save_pretrained(pretrained_dir)
    if preprocessor is not None:
        preprocessor.save_pretrained(pretrained_dir)
    if postprocessor is not None:
        postprocessor.save_pretrained(pretrained_dir)

    train_dir = checkpoint_dir / TRAINING_STATE_DIR
    train_dir.mkdir(parents=True, exist_ok=True)
    save_training_step(step, train_dir)
    save_rng_state(train_dir)
    save_optimizer_state(optimizers, train_dir)
    if scheduler is not None:
        save_scheduler_state(scheduler, train_dir)


def load_qc_training_state(
    checkpoint_dir: Path,
    optimizers: dict[str, torch.optim.Optimizer],
    scheduler: LRScheduler | None,
) -> tuple[int, dict[str, torch.optim.Optimizer], LRScheduler | None]:
    train_dir = checkpoint_dir / TRAINING_STATE_DIR
    if not train_dir.is_dir():
        raise NotADirectoryError(train_dir)

    load_rng_state(train_dir)
    step = load_training_step(train_dir)
    optimizers = load_optimizer_state(optimizers, train_dir)
    if scheduler is not None:
        scheduler = load_scheduler_state(scheduler, train_dir)
    return step, optimizers, scheduler


@parser.wrap()
def train_qc(cfg: TrainPipelineConfig):
    cfg.validate()
    init_logging()

    logging.info(pformat(cfg.to_dict()))

    if cfg.wandb.enable and cfg.wandb.project:
        wandb_logger = WandBLogger(cfg)
    else:
        wandb_logger = None
        logging.info(colored("Logs will be saved locally.", "yellow", attrs=["bold"]))

    if cfg.seed is not None:
        set_seed(cfg.seed)

    device = torch.device(cfg.policy.device if cfg.policy.device else "cuda")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    # ── Dataset ──
    logging.info("Creating dataset")
    dataset = make_dataset(cfg)

    # ── Eval env ──
    eval_env = None
    env_preprocessor = None
    env_postprocessor = None
    if cfg.eval_freq > 0 and cfg.env is not None:
        logging.info("Creating eval env")
        eval_env = make_env(cfg.env, n_envs=cfg.eval.batch_size, use_async_envs=cfg.eval.use_async_envs)

    # ── Policy ──
    logging.info("Creating policy")
    policy = make_policy(
        cfg=cfg.policy,
        ds_meta=dataset.meta,
        rename_map=cfg.rename_map,
        dataset_config=cfg.dataset,
    )
    policy.to(device)

    # ── Processors ──
    processor_kwargs = {}
    postprocessor_kwargs = {}
    if (cfg.policy.pretrained_path and not cfg.resume) or not cfg.policy.pretrained_path:
        processor_kwargs["dataset_stats"] = dataset.meta.stats

    if cfg.policy.pretrained_path is not None:
        processor_kwargs["preprocessor_overrides"] = {
            "device_processor": {"device": device.type},
            "normalizer_processor": {
                "stats": dataset.meta.stats,
                "features": {**policy.config.input_features, **policy.config.output_features},
                "norm_map": policy.config.normalization_mapping,
            },
        }
        if cfg.rename_map:
            processor_kwargs["preprocessor_overrides"]["rename_observations_processor"] = {
                "rename_map": cfg.rename_map
            }
        postprocessor_kwargs["postprocessor_overrides"] = {
            "unnormalizer_processor": {
                "stats": dataset.meta.stats,
                "features": policy.config.output_features,
                "norm_map": policy.config.normalization_mapping,
            },
        }

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        **processor_kwargs,
        **postprocessor_kwargs,
    )

    if cfg.rename_map:
        from lerobot.processor.rename_processor import RenameObservationsProcessorStep

        for step_obj in preprocessor.steps:
            if isinstance(step_obj, RenameObservationsProcessorStep):
                step_obj.rename_map = cfg.rename_map
                break

    # ── Triple optimizers: backbone / onestep_head / critic ──
    logging.info("Creating triple optimizers (backbone + onestep + critic)")
    param_groups = policy.get_optim_params()

    optim_kwargs = {"lr": cfg.optimizer.lr, "weight_decay": cfg.optimizer.weight_decay}
    if hasattr(cfg.optimizer, "betas"):
        optim_kwargs["betas"] = tuple(cfg.optimizer.betas)
    if hasattr(cfg.optimizer, "eps"):
        optim_kwargs["eps"] = cfg.optimizer.eps

    optimizer_backbone = torch.optim.AdamW(param_groups["backbone"], **optim_kwargs)
    optimizer_onestep = torch.optim.AdamW(param_groups["onestep"], **optim_kwargs)
    optimizer_critic = torch.optim.AdamW(
        param_groups["critic"],
        lr=policy.config.qc_critic_lr,
        weight_decay=0.0,
    )
    optimizers = {"backbone": optimizer_backbone, "onestep": optimizer_onestep, "critic": optimizer_critic}

    lr_scheduler = None
    if cfg.scheduler is not None:
        lr_scheduler = cfg.scheduler.build(optimizer_backbone, cfg.steps)

    grad_clip = cfg.optimizer.grad_clip_norm

    # ── Resume ──
    step = 0
    if cfg.resume:
        step, optimizers, lr_scheduler = load_qc_training_state(
            cfg.checkpoint_path, optimizers, lr_scheduler
        )
        optimizer_backbone = optimizers["backbone"]
        optimizer_onestep = optimizers["onestep"]
        optimizer_critic = optimizers["critic"]

    # ── Info ──
    num_learnable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    num_total_params = sum(p.numel() for p in policy.parameters())
    num_critic_params = sum(p.numel() for p in param_groups["critic"])
    num_backbone_params = sum(p.numel() for p in param_groups["backbone"])
    num_onestep_params = sum(p.numel() for p in param_groups["onestep"])

    logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")
    if cfg.env is not None:
        logging.info(f"{cfg.env.task=}")
        logging.info("Creating environment processors")
        env_preprocessor, env_postprocessor = make_env_pre_post_processors(
            env_cfg=cfg.env,
            policy_cfg=cfg.policy,
            image_transforms=cfg.dataset.image_transforms,
        )
    logging.info(f"{cfg.steps=} ({format_big_number(cfg.steps)})")
    logging.info(f"{dataset.num_frames=} ({format_big_number(dataset.num_frames)})")
    logging.info(f"{dataset.num_episodes=}")
    logging.info(f"num_backbone_params={num_backbone_params} ({format_big_number(num_backbone_params)})")
    logging.info(f"num_onestep_params={num_onestep_params} ({format_big_number(num_onestep_params)})")
    logging.info(f"num_critic_params={num_critic_params} ({format_big_number(num_critic_params)})")
    logging.info(f"{num_learnable_params=} ({format_big_number(num_learnable_params)})")
    logging.info(f"{num_total_params=} ({format_big_number(num_total_params)})")

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
    train_metrics = {
        "actor_loss": AverageMeter("a_loss", ":.4f"),
        "critic_loss": AverageMeter("c_loss", ":.4f"),
        "bc_flow_loss": AverageMeter("bc", ":.4f"),
        "distill_loss": AverageMeter("dist", ":.4f"),
        "q_loss": AverageMeter("ql", ":.4f"),
        "grad_norm_backbone": AverageMeter("gn_bb", ":.3f"),
        "grad_norm_onestep": AverageMeter("gn_os", ":.3f"),
        "grad_norm_critic": AverageMeter("gn_c", ":.3f"),
        "lr_backbone": AverageMeter("lr_bb", ":0.1e"),
        "lr_critic": AverageMeter("lr_c", ":0.1e"),
        "critic_valid_ratio": AverageMeter("c_val", ":.2f"),
        "update_s": AverageMeter("updt_s", ":.3f"),
        "dataloading_s": AverageMeter("data_s", ":.3f"),
        "preprocess_s": AverageMeter("prep_s", ":.3f"),
        "prepare_batch_s": AverageMeter("qcb_s", ":.3f"),
        "critic_fwd_s": AverageMeter("cf_s", ":.3f"),
        "critic_bwd_s": AverageMeter("cb_s", ":.3f"),
        "ema_s": AverageMeter("ema_s", ":.3f"),
        "actor_fwd_s": AverageMeter("af_s", ":.3f"),
        "actor_bwd_s": AverageMeter("ab_s", ":.3f"),
    }
    train_tracker = MetricsTracker(
        cfg.batch_size,
        dataset.num_frames,
        dataset.num_episodes,
        train_metrics,
        initial_step=step,
    )

    logging.info(
        f"Start QC-FQL training | actor_type={policy.config.qc_actor_type} "
        f"| actor_lr={cfg.optimizer.lr} | critic_lr={policy.config.qc_critic_lr} "
        f"| tau={policy.config.qc_tau} | discount={policy.config.qc_discount} "
        f"| chunk_size={policy.config.chunk_size} | num_critics={policy.config.qc_num_critics}"
        + (f" | alpha={policy.config.qc_alpha}" if policy.config.qc_actor_type == "distill-ddpg" else "")
    )

    policy.train()

    for _ in range(step, cfg.steps):
        t0 = time.perf_counter()
        batch = next(dl_iter)
        t1 = time.perf_counter()
        batch = preprocessor(batch)
        t2 = time.perf_counter()

        # ── prepare_qc_batch ──
        current_batch, next_batch, rewards, terminateds, next_obs_is_pad = prepare_qc_batch(
            batch,
            chunk_size=policy.config.chunk_size,
            discount=policy.config.qc_discount,
        )
        t3 = time.perf_counter()

        # ── Step 1: Critic forward ──
        critic_loss, critic_info = policy.forward_critic(
            current_batch, next_batch, rewards, terminateds,
            next_obs_is_pad=next_obs_is_pad,
        )
        t4 = time.perf_counter()

        # ── Step 1b: Critic backward ──
        optimizer_critic.zero_grad()
        critic_loss.backward()
        grad_norm_critic = torch.nn.utils.clip_grad_norm_(
            param_groups["critic"], grad_clip, error_if_nonfinite=False
        )
        optimizer_critic.step()
        t5 = time.perf_counter()

        # ── Step 2: EMA target update ──
        policy.model.update_target_value_heads()
        t6 = time.perf_counter()

        # ── Step 3: Actor forward ──
        actor_loss, actor_info = policy.forward_actor(current_batch)
        t7 = time.perf_counter()

        # ── Step 3b: Actor backward (separate clipping for backbone vs onestep) ──
        optimizer_backbone.zero_grad()
        optimizer_onestep.zero_grad()
        actor_loss.backward()
        grad_norm_backbone = torch.nn.utils.clip_grad_norm_(
            param_groups["backbone"], grad_clip, error_if_nonfinite=False
        )
        grad_norm_onestep = torch.nn.utils.clip_grad_norm_(
            param_groups["onestep"], grad_clip, error_if_nonfinite=False
        )
        optimizer_backbone.step()
        optimizer_onestep.step()
        t8 = time.perf_counter()

        if lr_scheduler is not None:
            lr_scheduler.step()

        # ── Track metrics ──
        train_tracker.actor_loss = actor_loss.item()
        train_tracker.critic_loss = critic_loss.item()
        train_tracker.bc_flow_loss = actor_info.get("bc_flow_loss", actor_loss.item())
        train_tracker.distill_loss = actor_info.get("distill_loss", 0.0)
        train_tracker.q_loss = actor_info.get("q_loss", 0.0)
        train_tracker.grad_norm_backbone = (
            grad_norm_backbone.item() if isinstance(grad_norm_backbone, torch.Tensor) else grad_norm_backbone
        )
        train_tracker.grad_norm_onestep = (
            grad_norm_onestep.item() if isinstance(grad_norm_onestep, torch.Tensor) else grad_norm_onestep
        )
        train_tracker.grad_norm_critic = (
            grad_norm_critic.item() if isinstance(grad_norm_critic, torch.Tensor) else grad_norm_critic
        )
        train_tracker.lr_backbone = optimizer_backbone.param_groups[0]["lr"]
        train_tracker.lr_critic = optimizer_critic.param_groups[0]["lr"]
        train_tracker.critic_valid_ratio = critic_info.get("critic_valid_ratio", 1.0)
        train_tracker.update_s = t8 - t2
        train_tracker.dataloading_s = t1 - t0

        if step % 1000 == 0:
            logging.info(
                f"[timing] data={t1-t0:.3f} prep={t2-t1:.3f} qcb={t3-t2:.4f} "
                f"cf={t4-t3:.3f} cb={t5-t4:.3f} ema={t6-t5:.4f} "
                f"af={t7-t6:.3f} ab={t8-t7:.3f} total={t8-t0:.3f}"
            )

        step += 1
        train_tracker.step()

        is_log_step = cfg.log_freq > 0 and step % cfg.log_freq == 0
        is_saving_step = step % cfg.save_freq == 0 or step == cfg.steps
        is_eval_step = cfg.eval_freq > 0 and step % cfg.eval_freq == 0

        # ── Logging ──
        if is_log_step:
            logging.info(train_tracker)
            if wandb_logger:
                wandb_log = train_tracker.to_dict()
                wandb_log.update({f"critic/{k}": v for k, v in critic_info.items()})
                wandb_log.update({f"actor/{k}": v for k, v in actor_info.items()})
                wandb_logger.log_dict(wandb_log, step)
            train_tracker.reset_averages()

        # ── Checkpoint ──
        if cfg.save_checkpoint and is_saving_step:
            logging.info(f"Checkpoint policy after step {step}")
            checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, cfg.steps, step)
            save_qc_checkpoint(
                checkpoint_dir=checkpoint_dir,
                step=step,
                cfg=cfg,
                policy=policy,
                optimizers=optimizers,
                scheduler=lr_scheduler,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
            )
            update_last_checkpoint(checkpoint_dir)
            if wandb_logger:
                wandb_logger.log_policy(checkpoint_dir)

        # ── Eval ──
        if cfg.env and is_eval_step:
            step_id = get_step_identifier(step, cfg.steps)
            logging.info(f"Eval policy at step {step}")
            with torch.no_grad():
                eval_info = eval_policy_all(
                    envs=eval_env,
                    policy=policy,
                    env_preprocessor=env_preprocessor,
                    env_postprocessor=env_postprocessor,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    n_episodes=cfg.eval.n_episodes,
                    videos_dir=cfg.output_dir / "eval" / f"videos_step_{step_id}",
                    max_episodes_rendered=4,
                    start_seed=cfg.seed,
                    max_parallel_tasks=cfg.env.max_parallel_tasks,
                )
            aggregated = eval_info["overall"]

            for suite, suite_info in eval_info.items():
                logging.info("Suite %s: %s", suite, suite_info)

            eval_metrics = {
                "avg_sum_reward": AverageMeter("∑rwrd", ":.3f"),
                "pc_success": AverageMeter("success", ":.1f"),
                "eval_s": AverageMeter("eval_s", ":.3f"),
            }
            eval_tracker = MetricsTracker(
                cfg.batch_size, dataset.num_frames, dataset.num_episodes,
                eval_metrics, initial_step=step,
            )
            eval_tracker.eval_s = aggregated.pop("eval_s")
            eval_tracker.avg_sum_reward = aggregated.pop("avg_sum_reward")
            eval_tracker.pc_success = aggregated.pop("pc_success")
            if wandb_logger:
                wandb_log = {**eval_tracker.to_dict(), **eval_info}
                wandb_logger.log_dict(wandb_log, step, mode="eval")
                wandb_logger.log_video(eval_info["overall"]["video_paths"][0], step, mode="eval")

    if eval_env:
        close_envs(eval_env)

    logging.info("End of QC-FQL training")


def main():
    from lerobot.utils.import_utils import register_third_party_plugins

    register_third_party_plugins()
    train_qc()


if __name__ == "__main__":
    main()
