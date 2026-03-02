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

"""
Async inference with dataset recording.

Decouples policy inference from action execution via a background inference thread
and an AsyncActionQueue. The inference thread continuously fills the queue while
the execution thread dequeues actions at fixed FPS and records to the dataset.

Architecture:
    ┌──────────────────────────────────────┐
    │         Main Thread (episode loop)   │
    │  manages episode lifecycle, keyboard │
    └──────────┬───────────────────────────┘
               │
       ┌───────┴────────┐
       ▼                ▼
┌─────────────┐  ┌─────────────────┐
│  Inference  │  │   Execution     │
│   Thread    │  │    Thread       │
│             │  │                 │
│ get obs     │  │ get obs (record)│
│ → policy    │  │ ← ActionQueue   │
│ → put_chunk │  │ → send_action   │
│             │  │ → dataset.add   │
└──────┬──────┘  └────────┬────────┘
       │                  │
       └──── AsyncActionQueue ────┘

Usage (same args as lerobot_infer.py, plus --action_queue.*):
    python -m src.lerobot.scripts.lerobot_infer_async \\
        --robot.type=so101_follower \\
        --robot.port=/dev/ttyACM0 \\
        --robot.id=my_follower_arm \\
        "--robot.cameras={front: {type: opencv, index_or_path: /dev/video0, width: 640, height: 480, fps: 30}, wrist: {type: opencv, index_or_path: /dev/video2, width: 640, height: 480, fps: 30}}" \\
        --dataset.repo_id=zhang/eval_pick_doll_place_async \\
        --dataset.num_episodes=1 \\
        --dataset.single_task=pick_doll_place \\
        --dataset.push_to_hub=false \\
        --dataset.episode_time_s=120 \\
        --image_transforms.resize_to=[240,320] \\
        --initial_position="[-5.71, -99.32, 99.64, 75.61, -45.98, 2.03]" \\
        --policy_training_dataset_root=/home/zhang/.cache/huggingface/lerobot/zhang/pick_doll_place_merged_09_13/ \\
        --policy.path=/home/zhang/robot/lerobot/outputs/train/2.14_307/cs64/fm/pretrained_model \\
        --policy.n_action_steps=64 \\
        --action_queue.type=threshold \\
        --action_queue.refill_threshold=4

    # Or with drop_delay queue:
    #   --action_queue.type=drop_delay \\
    #   --action_queue.refill_threshold=4 \\
    #   --action_queue.drop_delay_n_steps=3
"""

import logging
import shutil
import sys
import tempfile
import time
import traceback
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from pathlib import Path
from pprint import pformat
from threading import Event, Lock, Thread
from typing import Any

import numpy as np
import torch
from torch import Tensor

from lerobot.action_queues import (
    ActionQueueConfig,
    BaseActionQueue,
    DropDelayActionQueueConfig,  # noqa: F401 — triggers draccus registration
    ThresholdActionQueueConfig,
    make_action_queue_from_config,
)
from lerobot.cameras import CameraConfig  # noqa: F401
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.configs import parser
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.train import TRAIN_CONFIG_NAME, TrainPipelineConfig
from lerobot.datasets.image_writer import safe_stop_image_writer
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.pipeline_features import aggregate_pipeline_dataset_features, create_initial_features
from lerobot.datasets.utils import INFO_PATH, build_dataset_frame, combine_feature_dicts, load_json, load_stats
from lerobot.datasets.video_utils import VideoEncodingManager
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import build_inference_frame, make_robot_action, populate_queues
from lerobot.processor import (
    InferenceImageTransformProcessorStep,
    PolicyProcessorPipeline,
    RobotProcessorPipeline,
    make_default_processors,
)
from lerobot.processor.rename_processor import rename_stats
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    bi_openarm_follower,
    bi_so_follower,
    earthrover_mini_plus,
    hope_jr,
    koch_follower,
    make_robot_from_config,
    omx_follower,
    openarm_follower,
    reachy2,
    so_follower,
    unitree_g1 as unitree_g1_robot,
)
from lerobot.teleoperators import (  # noqa: F401
    TeleoperatorConfig,
    make_teleoperator_from_config,
)
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STR
from lerobot.utils.control_utils import (
    init_keyboard_listener,
    is_headless,
    sanity_check_dataset_name,
    sanity_check_dataset_robot_compatibility,
)
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import get_safe_torch_device, init_logging, log_say
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data
from lerobot.utils.constants import HF_LEROBOT_HOME

# Re-use helper functions from lerobot_infer
from lerobot.scripts.lerobot_infer import (
    DatasetRecordConfig,
    InferenceImageTransformConfig,
    _get_training_dataset_root,
    _load_training_dataset_features_for_inference,
    _load_training_dataset_stats_for_inference,
    _validate_inference_feature_alignment,
    move_to_initial_position,
    parse_initial_position,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Policies whose predict_action_chunk reads from self._queues
# and require populate_queues to be called before predict_action_chunk
POLICIES_WITH_QUEUES = {"flow_matching", "diffusion"}


# ─────────────────────────────────────────────
#  Thread-safe robot wrapper
# ─────────────────────────────────────────────

class RobotWrapper:
    """Wraps a Robot with a mutex so two threads can safely call get_observation
    and send_action concurrently."""

    def __init__(self, robot: Robot):
        self.robot = robot
        self._lock = Lock()

    def get_observation(self) -> dict:
        with self._lock:
            return self.robot.get_observation()

    def send_action(self, action):
        with self._lock:
            return self.robot.send_action(action)

    @property
    def observation_features(self):
        return self.robot.observation_features

    @property
    def action_features(self):
        return self.robot.action_features

    @property
    def name(self):
        return self.robot.name

    @property
    def robot_type(self):
        return getattr(self.robot, "robot_type", self.robot.name)


# ─────────────────────────────────────────────
#  Config dataclasses
# ─────────────────────────────────────────────

@dataclass
class AsyncRecordConfig:
    """Configuration for async record.

    Almost identical to RecordConfig in lerobot_infer.py with an
    ``action_queue`` field that selects the queue type from the CLI::

        --action_queue.type=threshold --action_queue.refill_threshold=4
        --action_queue.type=drop_delay --action_queue.drop_delay_n_steps=3
    """
    robot: RobotConfig
    dataset: DatasetRecordConfig

    # Policy (optional – if None the robot won't move)
    policy: PreTrainedConfig | None = None

    # Action queue strategy (resolved via --action_queue.type=...)
    action_queue: ActionQueueConfig = field(default_factory=lambda: ThresholdActionQueueConfig())

    # Local path to training dataset (avoids network calls for stats/features)
    policy_training_dataset_root: str | Path | None = None

    # Image transforms to match training preprocessing
    image_transforms: InferenceImageTransformConfig = field(
        default_factory=InferenceImageTransformConfig
    )

    # Initial robot position before recording starts
    initial_position: str | None = None
    move_to_initial_time_s: float = 3.0

    display_data: bool = False
    display_ip: str | None = None
    display_port: int | None = None
    display_compressed_images: bool = False
    play_sounds: bool = True
    resume: bool = False
    log_latency: bool = False

    def __post_init__(self):
        policy_path = parser.get_path_arg("policy")
        if policy_path:
            cli_overrides = parser.get_cli_overrides("policy")
            self.policy = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
            self.policy.pretrained_path = policy_path

        if self.policy is None:
            raise ValueError("--policy.path is required for async inference.")

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        return ["policy"]


# ─────────────────────────────────────────────
#  Observation preprocessing for inference
# ─────────────────────────────────────────────

def _populate_queue_policy_batch(batch: dict[str, Tensor], policy) -> dict[str, Tensor]:
    """Stack per-camera images and populate the policy's internal observation queues.

    Required for FM / Diffusion policies whose ``predict_action_chunk`` reads
    from ``policy._queues`` which must be filled before each call.
    """
    if ACTION in batch:
        batch = {k: v for k, v in batch.items() if k != ACTION}

    if policy.config.image_features:
        batch = dict(batch)
        images = [batch[key] for key in policy.config.image_features]
        batch[OBS_IMAGES] = torch.stack(images, dim=1)  # (B, n_cams, C, H, W)

    policy._queues = populate_queues(policy._queues, batch)
    return batch


# ─────────────────────────────────────────────
#  Worker threads
# ─────────────────────────────────────────────

def inference_worker(
    robot_wrapper: RobotWrapper,
    action_queue: BaseActionQueue,
    policy: PreTrainedPolicy,
    preprocessor,
    postprocessor,
    robot_observation_processor,
    dataset: LeRobotDataset,
    task: str,
    episode_active: Event,
    shutdown_event: Event,
    cfg: AsyncRecordConfig,
) -> None:
    """Background thread: runs policy forward pass and fills the action queue.

    Delegates *all* trigger/merge decisions to ``action_queue``; this function
    only drives the inference pipeline and calls the queue's lifecycle hooks.
    """
    try:
        logger.info("[INFERENCE] Thread started")
        policy_device = get_safe_torch_device(policy.config.device)
        needs_queue_populate = policy.name in POLICIES_WITH_QUEUES
        robot_type = robot_wrapper.robot_type

        while not shutdown_event.is_set():
            if not episode_active.wait(timeout=0.05):
                continue

            # ── Trigger check: fully delegated to the queue ──────
            if not action_queue.should_request_inference():
                time.sleep(0.002)
                continue

            # ── Lifecycle hook: inference about to start ─────────
            action_queue.on_inference_start()
            t0 = time.perf_counter()

            # ── Observation fetch (may block waiting for camera frame) ──
            t_obs_start = time.perf_counter()
            obs = robot_wrapper.get_observation()
            obs_fetch_s = time.perf_counter() - t_obs_start

            obs_processed = robot_observation_processor(obs)

            # Build feature-aligned observation frame and convert to tensors
            t_preprocess_start = time.perf_counter()
            obs_tensors = build_inference_frame(
                obs_processed,
                device=policy_device,
                ds_features=dataset.features,
                task=task,
                robot_type=robot_type,
            )

            # Preprocess (normalise, rename, etc.)
            preprocessed = preprocessor(obs_tensors)

            # FM / Diffusion: populate internal observation queues
            if needs_queue_populate:
                preprocessed = _populate_queue_policy_batch(preprocessed, policy)
            preprocess_s = time.perf_counter() - t_preprocess_start

            use_amp = getattr(policy.config, "use_amp", False)
            amp_ctx = (
                torch.autocast(device_type=policy_device.type)
                if policy_device.type == "cuda" and use_amp
                else nullcontext()
            )
            # RTC requires enable_grad; plain async uses inference_mode
            grad_ctx = torch.inference_mode()

            t_forward_start = time.perf_counter()
            with grad_ctx, amp_ctx:
                # Returns (1, T, action_dim)
                actions = policy.predict_action_chunk(preprocessed)
            forward_s = time.perf_counter() - t_forward_start

            # Denormalise and strip batch dimension → (T, action_dim)
            t_post_start = time.perf_counter()
            postprocessed = postprocessor(actions).squeeze(0)
            post_s = time.perf_counter() - t_post_start

            latency_s = time.perf_counter() - t0

            # ── Lifecycle hook: inference finished ───────────────
            action_queue.on_inference_done(latency_s)

            # ── Chunk integration: fully delegated to the queue ──
            action_queue.put_chunk(postprocessed)

            # ── Latency breakdown log ─────────────────────────────
            # obs_fetch: time waiting for camera + reading obs (camera latency proxy)
            # preprocess: obs → tensor conversion + normalisation
            # forward: policy model forward pass
            # post: action denormalisation
            # Steps are relative to execution FPS (how many robot steps each phase "costs")
            fps = cfg.dataset.fps
            logger.info(
                f"[INFERENCE] total={latency_s*1000:.0f}ms ({latency_s*fps:.1f}steps) | "
                f"obs_fetch={obs_fetch_s*1000:.0f}ms ({obs_fetch_s*fps:.1f}steps) | "
                f"preprocess={preprocess_s*1000:.0f}ms ({preprocess_s*fps:.1f}steps) | "
                f"forward={forward_s*1000:.0f}ms ({forward_s*fps:.1f}steps) | "
                f"post={post_s*1000:.0f}ms ({post_s*fps:.1f}steps) | "
                f"queue={action_queue.qsize()}"
            )

        logger.info("[INFERENCE] Thread shutting down")

    except Exception:
        logger.error("[INFERENCE] Fatal exception:\n" + traceback.format_exc())
        sys.exit(1)


def execution_worker(
    robot_wrapper: RobotWrapper,
    action_queue: BaseActionQueue,
    robot_action_processor,
    robot_observation_processor,
    dataset: LeRobotDataset | None,
    task: str,
    fps: float,
    episode_active: Event,
    shutdown_event: Event,
    events: dict,
    cfg: AsyncRecordConfig,
    display_data: bool = False,
    display_compressed_images: bool = False,
) -> None:
    """Execution thread: dequeues actions at fixed FPS, sends to robot, records data.

    If the queue is empty, the last sent action is repeated to keep the robot
    stable while waiting for the next inference chunk.
    """
    try:
        logger.info("[EXECUTION] Thread started")
        action_interval = 1.0 / fps
        last_action_dict: dict | None = None
        action_features = list(robot_wrapper.action_features.keys())

        while not shutdown_event.is_set():
            if not episode_active.is_set():
                time.sleep(0.01)
                continue

            if events["exit_early"]:
                events["exit_early"] = False
                episode_active.clear()
                break

            loop_start = time.perf_counter()

            # Read current observation (for recording)
            obs = robot_wrapper.get_observation()
            obs_processed = robot_observation_processor(obs)

            # Dequeue next action (tensor in robot space)
            action_tensor = action_queue.get()

            if action_tensor is not None:
                action_dict = {
                    key: action_tensor[i].item()
                    for i, key in enumerate(action_features)
                }
                last_action_dict = action_dict
            elif last_action_dict is not None:
                # Repeat last action – keeps robot position while inference runs
                action_dict = last_action_dict
                logger.debug("[EXECUTION] Queue empty – repeating last action")
            else:
                # No action available at all yet; skip this step
                dt = time.perf_counter() - loop_start
                precise_sleep(max(0.0, action_interval - dt))
                continue

            # Apply robot-level post-processing (safety clipping, etc.)
            action_to_send = robot_action_processor((action_dict, obs))
            robot_wrapper.send_action(action_to_send)

            # Record to dataset
            if dataset is not None and episode_active.is_set():
                obs_frame = build_dataset_frame(dataset.features, obs_processed, prefix=OBS_STR)
                action_frame = build_dataset_frame(dataset.features, action_dict, prefix=ACTION)
                frame = {**obs_frame, **action_frame, "task": task}
                dataset.add_frame(frame)

            if display_data:
                log_rerun_data(
                    observation=obs_processed,
                    action=action_dict,
                    compress_images=display_compressed_images,
                )

            if cfg.log_latency:
                loop_dt = time.perf_counter() - loop_start
                logger.info(f"[EXECUTION] loop_dt={loop_dt*1000:.1f}ms  queue={action_queue.qsize()}")

            dt = time.perf_counter() - loop_start
            precise_sleep(max(0.0, action_interval - dt))

        logger.info("[EXECUTION] Thread shutting down")

    except Exception:
        logger.error("[EXECUTION] Fatal exception:\n" + traceback.format_exc())
        sys.exit(1)


# ─────────────────────────────────────────────
#  Main entry point
# ─────────────────────────────────────────────

@safe_stop_image_writer
def _run_episodes(
    robot_wrapper: RobotWrapper,
    policy: PreTrainedPolicy,
    preprocessor,
    postprocessor,
    robot_action_processor,
    robot_observation_processor,
    dataset: LeRobotDataset,
    cfg: AsyncRecordConfig,
    events: dict,
):
    """Run all recording episodes with async inference threads."""

    action_queue = make_action_queue_from_config(cfg.action_queue)

    episode_active = Event()
    shutdown_event = Event()
    display_compressed_images = (
        True
        if (cfg.display_data and cfg.display_ip is not None and cfg.display_port is not None)
        else cfg.display_compressed_images
    )

    # Start long-running threads (they wait on episode_active between episodes)
    infer_thread = Thread(
        target=inference_worker,
        args=(
            robot_wrapper, action_queue, policy,
            preprocessor, postprocessor, robot_observation_processor,
            dataset, cfg.dataset.single_task,
            episode_active, shutdown_event, cfg,
        ),
        daemon=True,
        name="AsyncInference",
    )
    exec_thread = Thread(
        target=execution_worker,
        args=(
            robot_wrapper, action_queue,
            robot_action_processor, robot_observation_processor,
            dataset, cfg.dataset.single_task,
            cfg.dataset.fps, episode_active, shutdown_event,
            events, cfg,
            cfg.display_data, display_compressed_images,
        ),
        daemon=True,
        name="AsyncExecution",
    )
    infer_thread.start()
    exec_thread.start()
    logger.info("Async inference and execution threads started")

    try:
        with VideoEncodingManager(dataset):
            recorded_episodes = 0
            while recorded_episodes < cfg.dataset.num_episodes and not events["stop_recording"]:
                log_say(f"Recording episode {dataset.num_episodes}", cfg.play_sounds)
                logger.info(
                    f"[MAIN] Starting episode {recorded_episodes + 1}/{cfg.dataset.num_episodes} "
                    f"(duration={cfg.dataset.episode_time_s}s)"
                )

                # ── Reset state for new episode ──────────────────────
                policy.reset()
                preprocessor.reset()
                postprocessor.reset()
                action_queue.clear()

                episode_active.set()  # Let threads run
                episode_start = time.perf_counter()

                while (
                    not shutdown_event.is_set()
                    and not events["stop_recording"]
                    and (time.perf_counter() - episode_start) < cfg.dataset.episode_time_s
                ):
                    if events["exit_early"] or not episode_active.is_set():
                        events["exit_early"] = False
                        break
                    time.sleep(0.1)

                episode_active.clear()  # Pause threads for cleanup
                logger.info(
                    f"[MAIN] Episode ended after "
                    f"{time.perf_counter() - episode_start:.1f}s"
                )

                if events["rerecord_episode"]:
                    log_say("Re-record episode", cfg.play_sounds)
                    events["rerecord_episode"] = False
                    events["exit_early"] = False
                    dataset.clear_episode_buffer()
                    continue

                dataset.save_episode()
                recorded_episodes += 1
                logger.info(f"[MAIN] Saved episode {recorded_episodes}/{cfg.dataset.num_episodes}")

                # ── Reset phase (not after the last episode) ─────────
                if not events["stop_recording"] and recorded_episodes < cfg.dataset.num_episodes:
                    log_say("Reset the environment", cfg.play_sounds)
                    reset_start = time.perf_counter()
                    while (time.perf_counter() - reset_start) < cfg.dataset.reset_time_s:
                        if events["exit_early"] or events["stop_recording"]:
                            events["exit_early"] = False
                            break
                        time.sleep(0.1)

    finally:
        # Always stop threads before returning, even on KeyboardInterrupt or
        # other exceptions, so robot.disconnect() is never called while
        # inference/execution threads still hold the RobotWrapper lock.
        logger.info("[MAIN] Stopping async threads...")
        shutdown_event.set()
        episode_active.clear()

        if infer_thread.is_alive():
            infer_thread.join(timeout=3.0)
            if infer_thread.is_alive():
                logger.warning("[MAIN] Inference thread did not stop in time")
        if exec_thread.is_alive():
            exec_thread.join(timeout=3.0)
            if exec_thread.is_alive():
                logger.warning("[MAIN] Execution thread did not stop in time")

        logger.info("[MAIN] All threads joined")


@parser.wrap()
def async_record(cfg: AsyncRecordConfig) -> LeRobotDataset:
    init_logging()
    logger.info(pformat(asdict(cfg)))

    if cfg.display_data:
        init_rerun(session_name="async_recording", ip=cfg.display_ip, port=cfg.display_port)

    robot = make_robot_from_config(cfg.robot)

    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

    dataset_features = combine_feature_dicts(
        aggregate_pipeline_dataset_features(
            pipeline=teleop_action_processor,
            initial_features=create_initial_features(action=robot.action_features),
            use_videos=cfg.dataset.video,
        ),
        aggregate_pipeline_dataset_features(
            pipeline=robot_observation_processor,
            initial_features=create_initial_features(observation=robot.observation_features),
            use_videos=cfg.dataset.video,
        ),
    )

    dataset: LeRobotDataset | None = None
    listener = None
    rollout_temp_root: Path | None = None

    try:
        if cfg.resume:
            dataset = LeRobotDataset(
                cfg.dataset.repo_id,
                root=cfg.dataset.root,
                batch_encoding_size=cfg.dataset.video_encoding_batch_size,
                vcodec=cfg.dataset.vcodec,
            )
            if hasattr(robot, "cameras") and len(robot.cameras) > 0:
                dataset.start_image_writer(
                    num_processes=cfg.dataset.num_image_writer_processes,
                    num_threads=cfg.dataset.num_image_writer_threads_per_camera * len(robot.cameras),
                )
            sanity_check_dataset_robot_compatibility(dataset, robot, cfg.dataset.fps, dataset_features)
        else:
            sanity_check_dataset_name(cfg.dataset.repo_id, cfg.policy)
            dataset_root = cfg.dataset.root
            if not cfg.dataset.save_rollout:
                rollout_temp_root = Path(tempfile.mkdtemp(prefix="lerobot_async_rollout_"))
                dataset_root = rollout_temp_root / "dataset"
                logger.info(f"save_rollout=False: using temp dir {rollout_temp_root}")
            dataset = LeRobotDataset.create(
                cfg.dataset.repo_id,
                cfg.dataset.fps,
                root=dataset_root,
                robot_type=robot.name,
                features=dataset_features,
                use_videos=cfg.dataset.video,
                image_writer_processes=cfg.dataset.num_image_writer_processes,
                image_writer_threads=cfg.dataset.num_image_writer_threads_per_camera * len(robot.cameras),
                batch_encoding_size=cfg.dataset.video_encoding_batch_size,
                vcodec=cfg.dataset.vcodec,
            )

        # ── Load policy ──────────────────────────────────────────
        policy = make_policy(cfg.policy, ds_meta=dataset.meta)

        # ── Feature alignment + stats ────────────────────────────
        training_dataset_stats = None
        if cfg.policy.pretrained_path:
            policy_obs_features = _load_training_dataset_features_for_inference(
                cfg.policy.pretrained_path,
                training_dataset_root=cfg.policy_training_dataset_root,
            )
            if policy_obs_features is not None:
                logger.info("Running feature alignment validation...")
                _validate_inference_feature_alignment(
                    policy=policy,
                    robot_observation_features=dataset_features,
                    training_observation_features=policy_obs_features,
                )

            training_dataset_stats = _load_training_dataset_stats_for_inference(
                cfg.policy.pretrained_path,
                training_dataset_root=cfg.policy_training_dataset_root,
            )
            if training_dataset_stats is not None:
                logger.info("Loaded training dataset stats for normalisation")
            else:
                logger.warning(
                    "Could not load training dataset stats – falling back to inference dataset stats. "
                    "Specify --policy_training_dataset_root to avoid action offset errors."
                )

        stats_to_use = training_dataset_stats if training_dataset_stats is not None else dataset.meta.stats
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=cfg.policy,
            pretrained_path=cfg.policy.pretrained_path,
            dataset_stats=rename_stats(stats_to_use, cfg.dataset.rename_map),
            preprocessor_overrides={
                "device_processor": {"device": cfg.policy.device},
                "rename_observations_processor": {"rename_map": cfg.dataset.rename_map},
            },
        )

        # Optionally inject image transform step (centre-crop / resize)
        if cfg.image_transforms.center_crop_to_square or cfg.image_transforms.resize_to is not None:
            img_step = InferenceImageTransformProcessorStep(
                center_crop_to_square=cfg.image_transforms.center_crop_to_square,
                resize_to=cfg.image_transforms.resize_to,
            )
            preprocessor.steps.insert(0, img_step)
            logger.info(
                f"Added InferenceImageTransformProcessorStep: "
                f"center_crop={cfg.image_transforms.center_crop_to_square}, "
                f"resize_to={cfg.image_transforms.resize_to}"
            )

        # ── Connect hardware ─────────────────────────────────────
        robot.connect()

        # Move to initial position if given
        initial_position = parse_initial_position(cfg.initial_position)
        if initial_position is not None:
            log_say("Moving to initial position", cfg.play_sounds)
            move_to_initial_position(
                robot=robot,
                target_position=initial_position,
                duration_s=cfg.move_to_initial_time_s,
                fps=cfg.dataset.fps,
            )
            time.sleep(0.5)
            log_say("Ready", cfg.play_sounds)

        robot_wrapper = RobotWrapper(robot)

        listener, events = init_keyboard_listener()

        # ── Run episodes ─────────────────────────────────────────
        _run_episodes(
            robot_wrapper=robot_wrapper,
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            robot_action_processor=robot_action_processor,
            robot_observation_processor=robot_observation_processor,
            dataset=dataset,
            cfg=cfg,
            events=events,
        )

    finally:
        log_say("Stop recording", cfg.play_sounds, blocking=True)

        if dataset:
            dataset.finalize()

        if robot.is_connected:
            robot.disconnect()

        if not is_headless() and listener:
            listener.stop()

        if cfg.dataset.push_to_hub and cfg.dataset.save_rollout:
            dataset.push_to_hub(tags=cfg.dataset.tags, private=cfg.dataset.private)

        if rollout_temp_root is not None and rollout_temp_root.exists():
            logger.info(f"Deleting temp rollout directory: {rollout_temp_root}")
            shutil.rmtree(rollout_temp_root, ignore_errors=True)

        log_say("Exiting", cfg.play_sounds)

    return dataset


def main():
    from lerobot.utils.import_utils import register_third_party_plugins
    register_third_party_plugins()
    async_record()


if __name__ == "__main__":
    main()
