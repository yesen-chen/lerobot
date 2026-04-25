import logging
import time
from dataclasses import dataclass, field

from piper_sdk import C_PiperInterface_V2

logger = logging.getLogger(__name__)

PIPER_MOTORS = {
    "joint_1": (1, "agilex_piper"),
    "joint_2": (2, "agilex_piper"),
    "joint_3": (3, "agilex_piper"),
    "joint_4": (4, "agilex_piper"),
    "joint_5": (5, "agilex_piper"),
    "joint_6": (6, "agilex_piper"),
    "gripper": (7, "agilex_piper"),
}

# 1000 * 180 / pi  —  SDK 单位 0.001° 与弧度的转换因子
JOINT_FACTOR = 57324.840764
# 夹爪 SDK 单位 0.001° 到米的转换：gripper_m * 1e6 = SDK 值
GRIPPER_FACTOR = 1_000_000


@dataclass
class PiperMotorsBusConfig:
    can_name: str = "can0"
    motors: dict[str, tuple[int, str]] = field(default_factory=lambda: dict(PIPER_MOTORS))


class PiperMotorsBus:
    """封装 piper_sdk C_PiperInterface_V2，提供关节读写与示教模式控制。"""

    def __init__(self, config: PiperMotorsBusConfig):
        self.config = config
        self.motors = config.motors
        self._is_connected = False

        self.piper = C_PiperInterface_V2(config.can_name)
        self.piper.ConnectPort()

        self.init_joint_position = [0.0, 0.0, 0.0, 0.0, 0.52, 0.0, 0.0]
        self.safe_disable_position = [0.0, 0.0, 0.0, 0.0, 0.52, 0.0, 0.0]

    # ------------------------------------------------------------------
    # properties
    # ------------------------------------------------------------------

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    @property
    def motor_names(self) -> list[str]:
        return list(self.motors.keys())

    @property
    def motor_models(self) -> list[str]:
        return [model for _, model in self.motors.values()]

    @property
    def motor_indices(self) -> list[int]:
        return [idx for idx, _ in self.motors.values()]

    # ------------------------------------------------------------------
    # connect / disconnect
    # ------------------------------------------------------------------

    def connect(self, enable: bool = True) -> bool:
        """使能/下使能机械臂。

        - enable=True：轮询等待全部电机使能成功（最多 10s）
        - enable=False：只下发一次下使能指令，等待固定时间稳定（避免循环刷指令
          导致与示教模式反复拉扯而抖动）
        """
        if not enable:
            self.piper.DisableArm(7)
            self.piper.GripperCtrl(0, 1000, 0x00, 0)
            time.sleep(1.0)
            self._is_connected = False
            logger.info("Piper 下使能指令已发送")
            return True

        timeout = 10.0
        start = time.time()
        while True:
            elapsed = time.time() - start
            statuses = [
                self.piper.GetArmLowSpdInfoMsgs().motor_1.foc_status.driver_enable_status,
                self.piper.GetArmLowSpdInfoMsgs().motor_2.foc_status.driver_enable_status,
                self.piper.GetArmLowSpdInfoMsgs().motor_3.foc_status.driver_enable_status,
                self.piper.GetArmLowSpdInfoMsgs().motor_4.foc_status.driver_enable_status,
                self.piper.GetArmLowSpdInfoMsgs().motor_5.foc_status.driver_enable_status,
                self.piper.GetArmLowSpdInfoMsgs().motor_6.foc_status.driver_enable_status,
            ]
            self.piper.EnableArm(7)
            self.piper.GripperCtrl(0, 1000, 0x01, 0)
            if all(statuses):
                self._is_connected = True
                logger.info("Piper 使能成功")
                return True

            if elapsed > timeout:
                logger.error("Piper 使能超时")
                self._is_connected = False
                return False

            time.sleep(0.5)

    def safe_disconnect(self):
        """运动到安全位后等待到位。"""
        self.write(self.safe_disable_position)

    def hold_current_position(self):
        """把「当前关节位姿」作为目标下发，用于下使能前保位，避免失电后机械臂掉落。"""
        raw = self.read()
        target = [
            raw["joint_1"] / JOINT_FACTOR,
            raw["joint_2"] / JOINT_FACTOR,
            raw["joint_3"] / JOINT_FACTOR,
            raw["joint_4"] / JOINT_FACTOR,
            raw["joint_5"] / JOINT_FACTOR,
            raw["joint_6"] / JOINT_FACTOR,
            raw["gripper"] / GRIPPER_FACTOR,
        ]
        self.write(target)

    # ------------------------------------------------------------------
    # teach mode (software drag-teach via MotionCtrl_1)
    # ------------------------------------------------------------------

    def enter_teach_mode(self):
        """通过 MotionCtrl_1 进入拖动示教（0x01 = 开始示教录制）。"""
        self.piper.MotionCtrl_1(0, 0, 0x01)
        logger.info("进入软件拖动示教模式")

    def exit_teach_mode(self):
        """退出拖动示教（0x02 = 结束示教录制）。"""
        self.piper.MotionCtrl_1(0, 0, 0x02)
        logger.info("退出软件拖动示教模式")

    # ------------------------------------------------------------------
    # calibration stubs (Piper 无外部标定文件)
    # ------------------------------------------------------------------

    def apply_calibration(self):
        """回到初始位。"""
        self.write(self.init_joint_position)

    def set_calibration(self):
        pass

    def revert_calibration(self):
        pass

    # ------------------------------------------------------------------
    # read / write
    # ------------------------------------------------------------------

    def read(self) -> dict:
        """读取关节（SDK 单位 0.001°）与夹爪状态。"""
        joint_state = self.piper.GetArmJointMsgs().joint_state
        gripper_state = self.piper.GetArmGripperMsgs().gripper_state
        return {
            "joint_1": joint_state.joint_1,
            "joint_2": joint_state.joint_2,
            "joint_3": joint_state.joint_3,
            "joint_4": joint_state.joint_4,
            "joint_5": joint_state.joint_5,
            "joint_6": joint_state.joint_6,
            "gripper": gripper_state.grippers_angle,
        }

    def write(self, target_joint: list):
        """下发关节控制指令。

        target_joint: 长度 7 的列表
            [0..5] 关节 1-6，单位弧度
            [6]    夹爪，单位米 (0 ~ 0.08)
        """
        joints_sdk = [round(target_joint[i] * JOINT_FACTOR) for i in range(6)]
        gripper_sdk = round(target_joint[6] * GRIPPER_FACTOR)

        self.piper.MotionCtrl_2(0x01, 0x01, 100, 0x00)
        self.piper.JointCtrl(*joints_sdk)
        self.piper.GripperCtrl(abs(gripper_sdk), 1000, 0x01, 0)
