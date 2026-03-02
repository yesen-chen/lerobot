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
Async inference script with optional RTC for action chunking policies on real robots.

Supports:
- SmolVLA, Pi0, Pi0.5 (with or without RTC)
- Flow Matching Policy (with or without RTC)
- Diffusion Policy (without RTC, async only)

Usage:
    # Flow Matching with RTC
    python examples/rtc/eval_with_real_robot.py \
        --policy.path=/path/to/fm/pretrained_model \
        --policy.device=cuda \
        --rtc.enabled=true \
        --rtc.execution_horizon=20 \
        --robot.type=so101_follower \
        --robot.port=/dev/ttyACM0 \
        --robot.id=my_follower_arm \
        --robot.cameras="{front: {type: opencv, index_or_path: /dev/video0, width: 640, height: 480, fps: 30}, wrist: {type: opencv, index_or_path: /dev/video2, width: 640, height: 480, fps: 30}}" \
        --task="pick_doll_place" \
        --duration=120

    # Flow Matching / Diffusion without RTC (pure async)
    python examples/rtc/eval_with_real_robot.py \
        --policy.path=/path/to/pretrained_model \
        --policy.device=cuda \
        --rtc.enabled=false \
        --robot.type=so101_follower \
        --robot.port=/dev/ttyACM0 \
        --robot.id=my_follower_arm \
        --robot.cameras="{front: {type: opencv, index_or_path: /dev/video0, width: 640, height: 480, fps: 30}, wrist: {type: opencv, index_or_path: /dev/video2, width: 640, height: 480, fps: 30}}" \
        --task="pick_doll_place" \
        --duration=120

    # SmolVLA with RTC
    python examples/rtc/eval_with_real_robot.py \
        --policy.path=helper2424/smolvla_check_rtc_last3 \
        --policy.device=mps \
        --rtc.enabled=true \
        --rtc.execution_horizon=20 \
        --robot.type=so100_follower \
        --robot.port=/dev/tty.usbmodem58FA0834591 \
        --robot.id=so100_follower \
        --robot.cameras="{ gripper: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30}, front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
        --task="Move green small object into the purple platform" \
        --duration=120
"""

import logging
import math
import sys
import time
import traceback
from dataclasses import dataclass, field
from threading import Event, Lock, Thread

import numpy as np
import torch
from torch import Tensor

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.configs import parser
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import RTCAttentionSchedule
from lerobot.datasets.utils import build_dataset_frame, hw_to_dataset_features
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.policies.rtc.action_queue import ActionQueue
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.policies.rtc.latency_tracker import LatencyTracker
from lerobot.processor import InferenceImageTransformProcessorStep
from lerobot.processor.factory import (
    make_default_robot_action_processor,
    make_default_robot_observation_processor,
)
from lerobot.rl.process import ProcessSignalHandler
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    bi_so_follower,
    koch_follower,
    so_follower,
)
from lerobot.robots.utils import make_robot_from_config
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE
from lerobot.utils.hub import HubMixin
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import init_logging

POLICIES_WITH_QUEUES = ["flow_matching", "diffusion"]
RTC_NATIVE_POLICIES = ["smolvla", "pi05", "pi0"]

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_initial_position(position_str: str | None) -> list[float] | None:
    """Parse initial position string like '[-5.71, -99.32, 99.64, 75.61, -45.98, 2.03]' to list of floats."""
    if position_str is None:
        return None
    position_str = position_str.strip()
    if position_str.startswith("["):
        position_str = position_str[1:]
    if position_str.endswith("]"):
        position_str = position_str[:-1]
    try:
        return [float(x.strip()) for x in position_str.split(",")]
    except ValueError as e:
        logger.error(f"Failed to parse initial position: {e}")
        return None


def move_to_initial_position(
    robot: Robot,
    target_position: list[float],
    duration_s: float = 3.0,
    fps: int = 30,
):
    """Smoothly move the robot to the target position using cosine interpolation."""
    logger.info(f"Moving to initial position: {target_position}")

    obs = robot.get_observation()

    if hasattr(robot, "action_features"):
        motor_names = list(robot.action_features.keys())
    else:
        motor_names = sorted([k for k in obs if ".pos" in k])

    if not motor_names:
        logger.warning("Could not find motor names, skipping move to initial position")
        return

    current_position = []
    for name in motor_names:
        if name in obs:
            current_position.append(float(obs[name]))
        else:
            logger.warning(f"Motor {name} not found in observation")
            return

    current_position = np.array(current_position)
    target_position_arr = np.array(target_position)

    if len(current_position) != len(target_position_arr):
        logger.error(
            f"Position dimension mismatch: current has {len(current_position)} joints, "
            f"target has {len(target_position_arr)} joints. Motor names: {motor_names}"
        )
        return

    logger.info(f"Motor names: {motor_names}")
    logger.info(f"Current position: {current_position}")
    logger.info(f"Target position: {target_position_arr}")

    num_steps = int(duration_s * fps)
    for step in range(num_steps):
        t = (step + 1) / num_steps
        t_smooth = 0.5 * (1 - np.cos(np.pi * t))
        interpolated = current_position + t_smooth * (target_position_arr - current_position)

        action = {}
        for i, name in enumerate(motor_names):
            action[name] = float(interpolated[i])
        robot.send_action(action)
        precise_sleep(1.0 / fps)

    logger.info("Reached initial position")


class RobotWrapper:
    def __init__(self, robot: Robot):
        self.robot = robot
        self.lock = Lock()

    def get_observation(self) -> dict[str, Tensor]:
        with self.lock:
            return self.robot.get_observation()

    def send_action(self, action: Tensor):
        with self.lock:
            self.robot.send_action(action)

    def observation_features(self) -> list[str]:
        with self.lock:
            return self.robot.observation_features

    def action_features(self) -> list[str]:
        with self.lock:
            return self.robot.action_features


@dataclass
class RTCDemoConfig(HubMixin):
    """Configuration for RTC demo with action chunking policies and real robots."""

    # Policy configuration
    policy: PreTrainedConfig | None = None

    # Robot configuration
    robot: RobotConfig | None = None

    # RTC configuration
    rtc: RTCConfig = field(
        default_factory=lambda: RTCConfig(
            execution_horizon=40,
            max_guidance_weight=1.0,
            prefix_attention_schedule=RTCAttentionSchedule.EXP,
        )
    )

    # Demo parameters
    duration: float = 30.0  # Duration to run the demo (seconds)
    fps: float = 30.0  # Action execution frequency (Hz)

    # Compute device
    device: str | None = None  # Device to run on (cuda, cpu, auto)

    # Get new actions horizon. The amount of executed steps after which will be requested new actions.
    # It should be higher than inference delay + execution horizon.
    action_queue_size_to_get_new_actions: int = 24

    # Task to execute
    task: str = field(default="", metadata={"help": "Task to execute"})

    # Image transforms for inference (to match training preprocessing)
    center_crop_to_square: bool = field(
        default=False,
        metadata={"help": "Center crop images to square before resizing"},
    )
    resize_to: str | None = field(
        default=None,
        metadata={"help": "Resize images to [H,W], e.g. '[240,320]'"},
    )

    # Initial position for the robot before starting (list of joint positions as string)
    initial_position: str | None = field(
        default=None,
        metadata={"help": "Initial joint positions, e.g. '[-5.71, -99.32, 99.64, 75.61, -45.98, 2.03]'"},
    )
    move_to_initial_time_s: float = 3.0

    # Torch compile configuration
    use_torch_compile: bool = field(
        default=False,
        metadata={"help": "Use torch.compile for faster inference (PyTorch 2.0+)"},
    )

    torch_compile_backend: str = field(
        default="inductor",
        metadata={"help": "Backend for torch.compile (inductor, aot_eager, cudagraphs)"},
    )

    torch_compile_mode: str = field(
        default="default",
        metadata={"help": "Compilation mode (default, reduce-overhead, max-autotune)"},
    )

    torch_compile_disable_cudagraphs: bool = field(
        default=True,
        metadata={
            "help": "Disable CUDA graphs in torch.compile. Required due to in-place tensor "
            "operations in denoising loop (x_t += dt * v_t) which cause tensor aliasing issues."
        },
    )

    def __post_init__(self):
        # HACK: We parse again the cli args here to get the pretrained path if there was one.
        policy_path = parser.get_path_arg("policy")
        if policy_path:
            cli_overrides = parser.get_cli_overrides("policy")
            self.policy = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
            self.policy.pretrained_path = policy_path
        else:
            raise ValueError("Policy path is required")

        # Validate that robot configuration is provided
        if self.robot is None:
            raise ValueError("Robot configuration must be provided")

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        """This enables the parser to load config from the policy using `--policy.path=local/dir`"""
        return ["policy"]


def is_image_key(k: str) -> bool:
    return k.startswith(OBS_IMAGES)


def _prepare_obs_for_queue_policy(batch: dict[str, Tensor], policy) -> dict[str, Tensor]:
    """Prepare observation and populate internal queues for FM/Diffusion policies.

    These policies' predict_action_chunk reads from self._queues, so we must
    stack images and call populate_queues before calling predict_action_chunk.
    """
    from lerobot.policies.utils import populate_queues

    if ACTION in batch:
        batch.pop(ACTION)

    if policy.config.image_features:
        batch = dict(batch)
        images = [batch[key] for key in policy.config.image_features]
        batch[OBS_IMAGES] = torch.stack(images, dim=1)  # (B, n_cameras, C, H, W)

    policy._queues = populate_queues(policy._queues, batch)
    return batch


def get_actions(
    policy,
    robot: RobotWrapper,
    robot_observation_processor,
    action_queue: ActionQueue,
    shutdown_event: Event,
    cfg: RTCDemoConfig,
):
    """Thread function to request action chunks from the policy.

    Supports SmolVLA/Pi0/Pi05 (native RTC) and FM/Diffusion (queue-based) policies.
    """
    try:
        logger.info("[GET_ACTIONS] Starting get actions thread")

        latency_tracker = LatencyTracker()
        fps = cfg.fps
        time_per_chunk = 1.0 / fps
        policy_name = policy.name

        dataset_features = hw_to_dataset_features(robot.observation_features(), "observation")
        policy_device = policy.config.device

        logger.info(f"[GET_ACTIONS] Loading preprocessor/postprocessor from {cfg.policy.pretrained_path}")

        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=cfg.policy,
            pretrained_path=cfg.policy.pretrained_path,
            dataset_stats=None,
            preprocessor_overrides={
                "device_processor": {"device": cfg.policy.device},
            },
        )

        if cfg.center_crop_to_square or cfg.resize_to is not None:
            resize_to_val = None
            if cfg.resize_to is not None:
                resize_str = cfg.resize_to.strip().strip("[]")
                resize_to_val = tuple(int(x.strip()) for x in resize_str.split(","))
            image_transform_step = InferenceImageTransformProcessorStep(
                center_crop_to_square=cfg.center_crop_to_square,
                resize_to=resize_to_val,
            )
            preprocessor.steps.insert(0, image_transform_step)
            logger.info(
                f"[GET_ACTIONS] Added InferenceImageTransformProcessorStep: "
                f"center_crop={cfg.center_crop_to_square}, resize_to={resize_to_val}"
            )

        logger.info("[GET_ACTIONS] Preprocessor/postprocessor loaded successfully with embedded stats")

        get_actions_threshold = cfg.action_queue_size_to_get_new_actions

        if not cfg.rtc.enabled:
            get_actions_threshold = 0

        needs_queue_populate = policy_name in POLICIES_WITH_QUEUES

        while not shutdown_event.is_set():
            if action_queue.qsize() <= get_actions_threshold:
                current_time = time.perf_counter()
                action_index_before_inference = action_queue.get_action_index()
                prev_actions = action_queue.get_left_over()

                inference_latency = latency_tracker.max()
                inference_delay = math.ceil(inference_latency / time_per_chunk)

                obs = robot.get_observation()

                obs_processed = robot_observation_processor(obs)

                obs_with_policy_features = build_dataset_frame(
                    dataset_features, obs_processed, prefix="observation"
                )

                for name in obs_with_policy_features:
                    obs_with_policy_features[name] = torch.from_numpy(obs_with_policy_features[name])
                    if "image" in name:
                        obs_with_policy_features[name] = (
                            obs_with_policy_features[name].type(torch.float32) / 255
                        )
                        obs_with_policy_features[name] = (
                            obs_with_policy_features[name].permute(2, 0, 1).contiguous()
                        )
                    obs_with_policy_features[name] = obs_with_policy_features[name].unsqueeze(0)
                    obs_with_policy_features[name] = obs_with_policy_features[name].to(policy_device)

                obs_with_policy_features["task"] = [cfg.task]
                obs_with_policy_features["robot_type"] = (
                    robot.robot.name if hasattr(robot.robot, "name") else ""
                )

                preprocessed_obs = preprocessor(obs_with_policy_features)

                # FM/Diffusion: must populate internal _queues before predict_action_chunk
                if needs_queue_populate:
                    preprocessed_obs = _prepare_obs_for_queue_policy(preprocessed_obs, policy)

                # RTC needs torch.autograd.grad inside denoise_step, which
                # requires enable_grad() to work. inference_mode() cannot be
                # overridden by enable_grad(), so fall back to no_grad().
                grad_ctx = torch.no_grad() if cfg.rtc.enabled else torch.inference_mode()
                with grad_ctx:
                    actions = policy.predict_action_chunk(
                        preprocessed_obs,
                        inference_delay=inference_delay,
                        prev_chunk_left_over=prev_actions,
                    )

                original_actions = actions.squeeze(0).clone()

                postprocessed_actions = postprocessor(actions)
                postprocessed_actions = postprocessed_actions.squeeze(0)

                new_latency = time.perf_counter() - current_time
                new_delay = math.ceil(new_latency / time_per_chunk)
                latency_tracker.add(new_latency)

                if cfg.rtc.enabled:
                    if cfg.action_queue_size_to_get_new_actions < cfg.rtc.execution_horizon + new_delay:
                        logger.warning(
                            "[GET_ACTIONS] action_queue_size_to_get_new_actions too small. "
                            "Should be higher than inference delay + execution horizon."
                        )
                    if cfg.rtc.execution_horizon <= new_delay:
                        logger.warning(
                            f"[GET_ACTIONS] RTC MISCONFIGURED: execution_horizon={cfg.rtc.execution_horizon} "
                            f"<= inference_delay={new_delay}. "
                            f"RTC guidance covers new_chunk[0:{cfg.rtc.execution_horizon}] but robot executes "
                            f"from new_chunk[{new_delay}] — the guided region is entirely skipped! "
                            f"Set execution_horizon > {new_delay} to fix violent jerking."
                        )

                logger.info(f"merge_before{action_queue.qsize()}")
                action_queue.merge(
                    original_actions, postprocessed_actions, new_delay, action_index_before_inference
                )

                logger.info(
                    f"[GET_ACTIONS] Inference {new_latency*1000:.0f}ms | "
                    f"delay={new_delay} | queue={action_queue.qsize()}"
                )
            else:
                time.sleep(0.01)

        logger.info("[GET_ACTIONS] get actions thread shutting down")
    except Exception as e:
        logger.error(f"[GET_ACTIONS] Fatal exception in get_actions thread: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)


def actor_control(
    robot: RobotWrapper,
    robot_action_processor,
    action_queue: ActionQueue,
    shutdown_event: Event,
    cfg: RTCDemoConfig,
):
    """Thread function to execute actions on the robot.

    Args:
        robot: The robot instance
        action_queue: Queue to get actions from
        shutdown_event: Event to signal shutdown
        cfg: Demo configuration
    """
    try:
        logger.info("[ACTOR] Starting actor thread")

        action_count = 0
        action_interval = 1.0 / cfg.fps

        while not shutdown_event.is_set():
            start_time = time.perf_counter()

            # Try to get an action from the queue with timeout
            action = action_queue.get()

            if action is not None:
                action = action.cpu()
                action_dict = {key: action[i].item() for i, key in enumerate(robot.action_features())}
                action_processed = robot_action_processor((action_dict, None))
                robot.send_action(action_processed)

                action_count += 1

            dt_s = time.perf_counter() - start_time
            time.sleep(max(0, (action_interval - dt_s) - 0.001))

        logger.info(f"[ACTOR] Actor thread shutting down. Total actions executed: {action_count}")
    except Exception as e:
        logger.error(f"[ACTOR] Fatal exception in actor_control thread: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)


def _apply_torch_compile(policy, cfg: RTCDemoConfig):
    """Apply torch.compile to the policy's predict_action_chunk method.

    Args:
        policy: Policy instance to compile
        cfg: Configuration containing torch compile settings

    Returns:
        Policy with compiled predict_action_chunk method
    """

    # PI models handle their own compilation
    if policy.type == "pi05" or policy.type == "pi0":
        return policy

    try:
        # Check if torch.compile is available (PyTorch 2.0+)
        if not hasattr(torch, "compile"):
            logger.warning(
                f"torch.compile is not available. Requires PyTorch 2.0+. "
                f"Current version: {torch.__version__}. Skipping compilation."
            )
            return policy

        logger.info("Applying torch.compile to predict_action_chunk...")
        logger.info(f"  Backend: {cfg.torch_compile_backend}")
        logger.info(f"  Mode: {cfg.torch_compile_mode}")
        logger.info(f"  Disable CUDA graphs: {cfg.torch_compile_disable_cudagraphs}")

        # Compile the predict_action_chunk method
        # - CUDA graphs disabled to prevent tensor aliasing from in-place ops (x_t += dt * v_t)
        compile_kwargs = {
            "backend": cfg.torch_compile_backend,
            "mode": cfg.torch_compile_mode,
        }

        # Disable CUDA graphs if requested (prevents tensor aliasing issues)
        if cfg.torch_compile_disable_cudagraphs:
            compile_kwargs["options"] = {"triton.cudagraphs": False}

        original_method = policy.predict_action_chunk
        compiled_method = torch.compile(original_method, **compile_kwargs)
        policy.predict_action_chunk = compiled_method
        logger.info("✓ Successfully compiled predict_action_chunk")

    except Exception as e:
        logger.error(f"Failed to apply torch.compile: {e}")
        logger.warning("Continuing without torch.compile")

    return policy


@parser.wrap()
def demo_cli(cfg: RTCDemoConfig):
    """Main entry point for RTC demo with draccus configuration."""

    # Initialize logging
    init_logging()

    logger.info(f"Using device: {cfg.device}")

    # Setup signal handler for graceful shutdown
    signal_handler = ProcessSignalHandler(use_threads=True, display_pid=False)
    shutdown_event = signal_handler.shutdown_event

    policy = None
    robot = None
    get_actions_thread = None
    actor_thread = None

    policy_class = get_policy_class(cfg.policy.type)

    # Load config and set compile_model for pi0/pi05 models
    config = PreTrainedConfig.from_pretrained(cfg.policy.pretrained_path)

    if cfg.policy.type == "pi05" or cfg.policy.type == "pi0":
        config.compile_model = cfg.use_torch_compile

    if config.use_peft:
        from peft import PeftConfig, PeftModel

        peft_pretrained_path = cfg.policy.pretrained_path
        peft_config = PeftConfig.from_pretrained(peft_pretrained_path)

        policy = policy_class.from_pretrained(
            pretrained_name_or_path=peft_config.base_model_name_or_path, config=config
        )
        policy = PeftModel.from_pretrained(policy, peft_pretrained_path, config=peft_config)
    else:
        policy = policy_class.from_pretrained(cfg.policy.pretrained_path, config=config)

    # Configure RTC
    policy.config.rtc_config = cfg.rtc

    if policy.name in RTC_NATIVE_POLICIES:
        policy.init_rtc_processor()
    elif policy.name in POLICIES_WITH_QUEUES:
        # FM/Diffusion: RTC processor is initialized in __init__ when rtc_config is set
        if hasattr(policy, 'rtc_processor') and policy.rtc_processor is None and cfg.rtc.enabled:
            from lerobot.policies.rtc.modeling_rtc import RTCProcessor
            policy.rtc_processor = RTCProcessor(cfg.rtc)
        logger.info(f"Policy '{policy.name}' configured for async inference (RTC={cfg.rtc.enabled})")
    else:
        logger.warning(f"Policy '{policy.name}' may not fully support RTC. Proceeding anyway.")

    policy = policy.to(cfg.device)
    policy.eval()

    # Apply torch.compile to predict_action_chunk method if enabled
    if cfg.use_torch_compile:
        policy = _apply_torch_compile(policy, cfg)

    # Create robot
    logger.info(f"Initializing robot: {cfg.robot.type}")
    robot = make_robot_from_config(cfg.robot)
    robot.connect()

    # Move to initial position if specified
    initial_position = parse_initial_position(cfg.initial_position)
    if initial_position is not None:
        move_to_initial_position(
            robot=robot,
            target_position=initial_position,
            duration_s=cfg.move_to_initial_time_s,
            fps=int(cfg.fps),
        )
        time.sleep(0.5)
        logger.info("Ready after moving to initial position")

    robot_wrapper = RobotWrapper(robot)

    # Create robot observation processor
    robot_observation_processor = make_default_robot_observation_processor()
    robot_action_processor = make_default_robot_action_processor()

    # Create action queue for communication between threads
    action_queue = ActionQueue(cfg.rtc)

    # Start chunk requester thread
    get_actions_thread = Thread(
        target=get_actions,
        args=(policy, robot_wrapper, robot_observation_processor, action_queue, shutdown_event, cfg),
        daemon=True,
        name="GetActions",
    )
    get_actions_thread.start()
    logger.info("Started get actions thread")

    # Start action executor thread
    actor_thread = Thread(
        target=actor_control,
        args=(robot_wrapper, robot_action_processor, action_queue, shutdown_event, cfg),
        daemon=True,
        name="Actor",
    )
    actor_thread.start()
    logger.info("Started actor thread")

    logger.info("Started stop by duration thread")

    # Main thread monitors for duration or shutdown
    logger.info(f"Running demo for {cfg.duration} seconds...")
    start_time = time.time()

    while not shutdown_event.is_set() and (time.time() - start_time) < cfg.duration:
        time.sleep(10)

        # Log queue status periodically
        if int(time.time() - start_time) % 5 == 0:
            logger.info(f"[MAIN] Action queue size: {action_queue.qsize()}")

        if time.time() - start_time > cfg.duration:
            break

    logger.info("Demo duration reached or shutdown requested")

    # Signal shutdown
    shutdown_event.set()

    # Wait for threads to finish
    if get_actions_thread and get_actions_thread.is_alive():
        logger.info("Waiting for chunk requester thread to finish...")
        get_actions_thread.join()

    if actor_thread and actor_thread.is_alive():
        logger.info("Waiting for action executor thread to finish...")
        actor_thread.join()

    # Cleanup robot
    if robot:
        robot.disconnect()
        logger.info("Robot disconnected")

    logger.info("Cleanup completed")


if __name__ == "__main__":
    demo_cli()
    logging.info("RTC demo finished")
