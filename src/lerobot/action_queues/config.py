"""ActionQueueConfig base class using draccus.ChoiceRegistry.

Follows the same pattern as ``RobotConfig``: subclasses register
themselves via ``@ActionQueueConfig.register_subclass("name")`` and
draccus automatically resolves ``--action_queue.type=name`` on the CLI.
"""

from dataclasses import dataclass

import draccus


@dataclass
class ActionQueueConfig(draccus.ChoiceRegistry):
    """Base configuration for all action queue types.

    Subclasses must register with
    ``@ActionQueueConfig.register_subclass("my_queue_type")``.
    """

    @property
    def type(self) -> str:
        return self.get_choice_name(self.__class__)
