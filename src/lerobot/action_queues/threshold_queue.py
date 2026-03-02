"""Simple FIFO queue that triggers inference when depth falls to a threshold.

This is the default, minimal-overhead implementation.

Trigger:   ``qsize() <= refill_threshold``
Integrate: replace – discard any remaining old actions, load fresh chunk.

To extend behaviour without touching the rest of the pipeline, subclass
this (or :class:`BaseActionQueue`) and swap it in.

Example subclasses
~~~~~~~~~~~~~~~~~~
* **FixedIntervalQueue** – override ``should_request_inference`` to fire
  every ``1 / inference_fps`` seconds regardless of queue depth.
* **RTCQueue** – override ``put_chunk`` to blend old and new chunks with
  temporal interpolation instead of hard replacing.
* **ConfidenceQueue** – override ``should_request_inference`` to also
  accept a policy uncertainty score stored via ``on_inference_done``.
"""

from collections import deque
from dataclasses import dataclass
from threading import Lock

from torch import Tensor

from lerobot.action_queues.base import BaseActionQueue
from lerobot.action_queues.config import ActionQueueConfig


@ActionQueueConfig.register_subclass("threshold")
@dataclass
class ThresholdActionQueueConfig(ActionQueueConfig):
    """Config for :class:`ThresholdActionQueue`.

    CLI example::

        --action_queue.type=threshold --action_queue.refill_threshold=4
    """
    refill_threshold: int = 0


class ThresholdActionQueue(BaseActionQueue):
    """FIFO queue that fires inference when ``qsize() <= refill_threshold``."""

    def __init__(self, refill_threshold: int = 0):
        """
        Args:
            refill_threshold: Fire inference when ``qsize() <= refill_threshold``.
                ``0``  → only when queue is empty (may stutter if inference is slow).
                ``>0`` → overlap execution with inference; set to roughly the number
                         of steps inference takes at the target FPS.
        """
        self._deque: deque[Tensor] = deque()
        self._lock = Lock()
        self.refill_threshold = refill_threshold

    def put_chunk(self, actions: Tensor) -> None:
        """Replace queue contents with a new chunk (hard reset, no blending)."""
        actions = actions.detach().cpu()
        with self._lock:
            self._deque.clear()
            for i in range(len(actions)):
                self._deque.append(actions[i].clone())

    def get(self) -> Tensor | None:
        with self._lock:
            return self._deque.popleft() if self._deque else None

    def qsize(self) -> int:
        with self._lock:
            return len(self._deque)

    def clear(self) -> None:
        with self._lock:
            self._deque.clear()

    def should_request_inference(self) -> bool:
        return self.qsize() <= self.refill_threshold
