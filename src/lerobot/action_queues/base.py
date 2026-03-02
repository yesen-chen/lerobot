"""Abstract base class for async inference action queues.

Separates two orthogonal concerns so each can be extended independently:

1. **Inference trigger** — *when* to fire a new policy forward pass.
   Override :meth:`should_request_inference` with any logic:
   queue-depth threshold, fixed-FPS clock, policy confidence score, etc.

2. **Chunk integration** — *how* a new chunk is merged into the queue.
   Override :meth:`put_chunk` for custom strategies: simple replace,
   RTC-style temporal blending, confidence-weighted merge, etc.
"""

from abc import ABC, abstractmethod

from torch import Tensor


class BaseActionQueue(ABC):
    """Abstract base class for all async inference action queues.

    Optional lifecycle hooks
    ~~~~~~~~~~~~~~~~~~~~~~~~
    :meth:`on_inference_start` and :meth:`on_inference_done` are called by
    the inference thread around each forward pass so subclasses can update
    internal state (e.g. a latency estimate used inside
    ``should_request_inference``).
    """

    @abstractmethod
    def put_chunk(self, actions: Tensor) -> None:
        """Integrate a new action chunk produced by the inference thread.

        Args:
            actions: ``(T, action_dim)`` tensor in *robot space*
                     (already postprocessed / denormalised).
        """
        raise NotImplementedError

    @abstractmethod
    def get(self) -> Tensor | None:
        """Consume and return the next action step, or ``None`` if empty.

        Called by the execution thread at fixed FPS.
        """
        raise NotImplementedError

    @abstractmethod
    def qsize(self) -> int:
        """Number of unconsumed action steps currently held."""
        raise NotImplementedError

    @abstractmethod
    def clear(self) -> None:
        """Discard all pending actions (e.g. at episode boundaries)."""
        raise NotImplementedError

    @abstractmethod
    def should_request_inference(self) -> bool:
        """Return ``True`` when the inference thread should start a new forward pass.

        Called continuously (in a tight loop) by the inference thread.
        Implementations may inspect queue depth, wall-clock time, policy
        internal state, or any other signal.
        """
        raise NotImplementedError

    # ── Optional lifecycle hooks (may override) ──

    def on_inference_start(self) -> None:
        """Hook called by the inference thread *before* each forward pass."""

    def on_inference_done(self, latency_s: float) -> None:
        """Hook called by the inference thread *after* each forward pass.

        Args:
            latency_s: Wall-clock duration of the forward pass in seconds.
                       Useful for adaptive trigger strategies.
        """
