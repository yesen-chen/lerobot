from dataclasses import dataclass

from ..config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("piper_teach")
@dataclass
class PiperTeachConfig(TeleoperatorConfig):
    can_name: str = "can0"
    teach_mode: str = "button"  # "button" | "software"
