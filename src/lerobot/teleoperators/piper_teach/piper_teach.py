import logging
import time
from functools import cached_property
from typing import Any

from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.motors.piper.piper import GRIPPER_FACTOR, JOINT_FACTOR
from lerobot.processor import RobotAction

from ..teleoperator import Teleoperator
from .config_piper_teach import PiperTeachConfig

logger = logging.getLogger(__name__)

MOTOR_NAMES = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "gripper"]


class PiperTeach(Teleoperator):
    """单臂示教遥操作器：读取同一 Piper 臂的关节状态作为 action。

    在 button 模式下，用户通过物理按键进入示教；
    在 software 模式下，connect() 时自动通过 SDK 进入拖动示教。
    """

    config_class = PiperTeachConfig
    name = "piper_teach"

    def __init__(self, config: PiperTeachConfig):
        super().__init__(config)
        self.config = config
        self._is_connected = False
        self._is_calibrated = True  # Piper 无需外部标定
        self.piper = None

    # ------------------------------------------------------------------
    # features
    # ------------------------------------------------------------------

    @cached_property
    def action_features(self) -> dict[str, type]:
        return {f"{m}.pos": float for m in MOTOR_NAMES}

    @property
    def feedback_features(self) -> dict:
        return {}

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
            raise DeviceAlreadyConnectedError("PiperTeach 已连接")

        from piper_sdk import C_PiperInterface_V2

        self.piper = C_PiperInterface_V2(self.config.can_name)
        self.piper.ConnectPort()

        if self.config.teach_mode == "software":
            self.piper.MotionCtrl_1(0, 0, 0x01)
            logger.info("PiperTeach: 通过 SDK 进入拖动示教")

        self._is_connected = True
        logger.info("PiperTeach 已连接 (模式: %s)", self.config.teach_mode)

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    # ------------------------------------------------------------------
    # action
    # ------------------------------------------------------------------

    def get_action(self) -> RobotAction:
        if not self._is_connected:
            raise DeviceNotConnectedError("PiperTeach 未连接")

        joint_state = self.piper.GetArmJointMsgs().joint_state
        gripper_state = self.piper.GetArmGripperMsgs().gripper_state

        return {
            "joint_1.pos": joint_state.joint_1 / JOINT_FACTOR,
            "joint_2.pos": joint_state.joint_2 / JOINT_FACTOR,
            "joint_3.pos": joint_state.joint_3 / JOINT_FACTOR,
            "joint_4.pos": joint_state.joint_4 / JOINT_FACTOR,
            "joint_5.pos": joint_state.joint_5 / JOINT_FACTOR,
            "joint_6.pos": joint_state.joint_6 / JOINT_FACTOR,
            "gripper.pos": gripper_state.grippers_angle / GRIPPER_FACTOR,
        }

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        pass

    # ------------------------------------------------------------------
    # disconnect
    # ------------------------------------------------------------------

    def disconnect(self) -> None:
        if not self._is_connected:
            return

        if self.config.teach_mode == "software" and self.piper is not None:
            self.piper.MotionCtrl_1(0, 0, 0x02)
            logger.info("PiperTeach: 退出拖动示教")

        self._is_connected = False
        logger.info("PiperTeach 已断开")
