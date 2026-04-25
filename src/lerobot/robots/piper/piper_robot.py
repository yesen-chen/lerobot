import logging
import time
from functools import cached_property

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.motors.piper import PiperMotorsBus, PiperMotorsBusConfig
from lerobot.motors.piper.piper import GRIPPER_FACTOR, JOINT_FACTOR
from lerobot.processor import RobotAction, RobotObservation

from ..robot import Robot
from .config_piper import PiperRobotConfig

logger = logging.getLogger(__name__)

MOTOR_NAMES = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "gripper"]


class PiperRobot(Robot):
    """单臂 Piper 机器人：用于回放与推理时的主动控制。"""

    config_class = PiperRobotConfig
    name = "piper"

    def __init__(self, config: PiperRobotConfig):
        super().__init__(config)
        self.config = config
        self.bus = PiperMotorsBus(
            PiperMotorsBusConfig(
                can_name=config.can_name,
            )
        )
        self._is_connected = False
        self._is_calibrated = False
        self.cameras = make_cameras_from_configs(config.cameras)

    # ------------------------------------------------------------------
    # features
    # ------------------------------------------------------------------

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {f"{m}.pos": float for m in MOTOR_NAMES}

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            f"observation.images.{cam_key}": (cam.height, cam.width, 3)
            for cam_key, cam in self.cameras.items()
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft

    @property
    def motor_features(self) -> dict:
        action_names = list(self._motors_ft.keys())
        state_names = list(self._motors_ft.keys())
        return {
            "action": {
                "dtype": "float32",
                "shape": (len(action_names),),
                "names": action_names,
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (len(state_names),),
                "names": state_names,
            },
        }

    @property
    def camera_features(self) -> dict:
        cam_ft = {}
        for cam_key, cam in self.cameras.items():
            key = f"observation.images.{cam_key}"
            cam_ft[key] = {
                "shape": (cam.height, cam.width, cam.channels),
                "names": ["height", "width", "channels"],
                "info": None,
            }
        return cam_ft

    # ------------------------------------------------------------------
    # connection
    # ------------------------------------------------------------------

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    @property
    def is_calibrated(self) -> bool:
        return self._is_calibrated

    def connect(self, calibrate: bool = True) -> None:
        if self._is_connected:
            raise DeviceAlreadyConnectedError("Piper 已连接，请勿重复 connect()")

        if not self.bus.connect(enable=True):
            raise ConnectionError("Piper 使能失败")
        logger.info("Piper 机械臂已使能")

        for name, cam in self.cameras.items():
            cam.connect()
            logger.info("相机 %s 已连接", name)

        self._is_connected = True

        if calibrate:
            self.calibrate()

    def calibrate(self) -> None:
        self.bus.apply_calibration()
        self._is_calibrated = True
        logger.info("Piper 已回到初始位")

    def configure(self) -> None:
        pass

    # ------------------------------------------------------------------
    # observation / action
    # ------------------------------------------------------------------

    def get_observation(self) -> RobotObservation:
        if not self._is_connected:
            raise DeviceNotConnectedError("Piper 未连接")

        raw = self.bus.read()
        obs: RobotObservation = {}
        for joint in ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]:
            obs[f"{joint}.pos"] = raw[joint] / JOINT_FACTOR
        obs["gripper.pos"] = raw["gripper"] / GRIPPER_FACTOR

        for cam_key, cam in self.cameras.items():
            obs[f"observation.images.{cam_key}"] = cam.async_read()

        return obs

    def send_action(self, action: RobotAction) -> RobotAction:
        if not self._is_connected:
            raise DeviceNotConnectedError("Piper 未连接")

        target = [float(action[f"{m}.pos"]) for m in MOTOR_NAMES]
        self.bus.write(target)
        return action

    # ------------------------------------------------------------------
    # disconnect
    # ------------------------------------------------------------------

    def disconnect(self) -> None:
        if not self._is_connected:
            return

        logger.info("保持当前姿态...")
        try:
            self.bus.hold_current_position()
            time.sleep(1.0)
        except Exception as e:
            logger.warning("保位失败: %s", e)

        if self.config.disable_on_disconnect:
            logger.info("正在下使能 Piper...")
            self.bus.connect(enable=False)
        else:
            logger.info(
                "Piper 保持使能并锁在当前姿态。"
                "如需移动请按机械臂示教键切柔顺，或从电源开关断电。"
            )

        for cam in self.cameras.values():
            cam.disconnect()

        self._is_connected = False
        logger.info("Piper 已断开 (motors_enabled=%s)", not self.config.disable_on_disconnect)
