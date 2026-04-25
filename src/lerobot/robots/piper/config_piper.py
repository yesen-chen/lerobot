from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig

from ..config import RobotConfig


@RobotConfig.register_subclass("piper")
@dataclass
class PiperRobotConfig(RobotConfig):
    can_name: str = "can0"
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
    # 断开连接时是否下使能机械臂
    #   False（默认，推荐）：保持使能并锁在最后姿态，避免失电后重力掉落；
    #                        需要移动时按物理示教键切柔顺，或从电源开关断电
    #   True：发送 DisableArm 指令，电机会失电；臂可能因重力下坠
    disable_on_disconnect: bool = False
