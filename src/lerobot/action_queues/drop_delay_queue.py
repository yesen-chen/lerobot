"""Threshold queue that discards the first N actions of every new chunk.

Motivation
~~~~~~~~~~
Policy inference is not instantaneous.  While the inference thread is
running a forward pass (taking, say, ``D`` execution steps worth of
time), the robot keeps executing actions from the *old* chunk.  When the
new chunk finally arrives its first ``D`` actions describe what the robot
*should have been doing* during inference — they are already stale.

By dropping ``drop_delay_n_steps`` actions from the front of each chunk
we skip directly to the actions that are relevant *now*, reducing the
lag between observation and execution.

Typical usage
~~~~~~~~~~~~~
Measure the average inference latency in steps
(``latency_s * fps``), then set ``drop_delay_n_steps`` to that value::

    queue = DropDelayActionQueue(
        drop_delay_n_steps=4,   # ~133 ms at 30 fps
        refill_threshold=4,
    )

.. note::
    If ``drop_delay_n_steps >= len(actions)`` the entire chunk is
    discarded and a warning is logged.  Tune the value so it stays
    well below ``n_action_steps``.
"""

import logging
from dataclasses import dataclass

from torch import Tensor

from lerobot.action_queues.config import ActionQueueConfig
from lerobot.action_queues.threshold_queue import ThresholdActionQueue

logger = logging.getLogger(__name__)


@ActionQueueConfig.register_subclass("drop_delay")
@dataclass
class DropDelayActionQueueConfig(ActionQueueConfig):
    """Config for :class:`DropDelayActionQueue`.

    CLI example::

        --action_queue.type=drop_delay \\
        --action_queue.refill_threshold=4 \\
        --action_queue.drop_delay_n_steps=3
    """
    refill_threshold: int = 0
    drop_delay_n_steps: int = 0


class DropDelayActionQueue(ThresholdActionQueue):
    """ThresholdActionQueue variant that drops leading stale actions."""

    def __init__(self, drop_delay_n_steps: int = 0, **kwargs):
        """
        Args:
            drop_delay_n_steps: Number of leading actions to drop from each
                new chunk.  ``0`` disables the feature (identical to
                ``ThresholdActionQueue``).
                The **first** chunk of every episode is never trimmed regardless
                of this value, because no inference delay has accumulated yet.
            **kwargs: Forwarded to :class:`ThresholdActionQueue`
                (e.g. ``refill_threshold``).
        """
        super().__init__(**kwargs)
        if drop_delay_n_steps < 0:
            raise ValueError(f"drop_delay_n_steps must be >= 0, got {drop_delay_n_steps}")
        self.drop_delay_n_steps = drop_delay_n_steps
        self._is_first_chunk = True

    def put_chunk(self, actions: Tensor) -> None:
        """Store chunk, skipping the delay-drop for the very first chunk."""
        if self._is_first_chunk:
            self._is_first_chunk = False
        elif self.drop_delay_n_steps > 0:
            if self.drop_delay_n_steps >= len(actions):
                logger.warning(
                    f"[DropDelayActionQueue] drop_delay_n_steps={self.drop_delay_n_steps} "
                    f">= chunk length={len(actions)}; entire chunk discarded. "
                    "Reduce drop_delay_n_steps or increase n_action_steps."
                )
                return
            actions = actions[self.drop_delay_n_steps:]

        super().put_chunk(actions)

    def clear(self) -> None:
        """Discard all actions and reset first-chunk flag (episode boundary)."""
        super().clear()
        self._is_first_chunk = True
