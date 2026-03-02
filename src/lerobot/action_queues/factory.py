"""Factory function that creates an action queue instance from its config."""

import logging
from dataclasses import asdict

from lerobot.action_queues.base import BaseActionQueue
from lerobot.action_queues.config import ActionQueueConfig

logger = logging.getLogger(__name__)

# Maps config type name → (queue class, relevant config fields).
# Lazy imports keep the module lightweight when only a subset is used.
_REGISTRY: dict[str, tuple] = {}


def _ensure_registry():
    if _REGISTRY:
        return
    from lerobot.action_queues.threshold_queue import ThresholdActionQueue, ThresholdActionQueueConfig
    from lerobot.action_queues.drop_delay_queue import DropDelayActionQueue, DropDelayActionQueueConfig

    _REGISTRY["threshold"] = (ThresholdActionQueue, ThresholdActionQueueConfig)
    _REGISTRY["drop_delay"] = (DropDelayActionQueue, DropDelayActionQueueConfig)


def make_action_queue_from_config(config: ActionQueueConfig) -> BaseActionQueue:
    """Instantiate the action queue corresponding to *config*.

    Args:
        config: An ``ActionQueueConfig`` subclass instance (e.g.
            ``ThresholdActionQueueConfig`` or ``DropDelayActionQueueConfig``).

    Returns:
        A fully-initialised :class:`BaseActionQueue`.

    Example::

        cfg = ThresholdActionQueueConfig(refill_threshold=4)
        queue = make_action_queue_from_config(cfg)
    """
    _ensure_registry()

    queue_type = config.type
    if queue_type not in _REGISTRY:
        raise ValueError(
            f"Unknown action queue type '{queue_type}'. "
            f"Available: {sorted(_REGISTRY.keys())}"
        )

    queue_cls, _ = _REGISTRY[queue_type]

    kwargs = {k: v for k, v in asdict(config).items() if k != "type"}
    queue = queue_cls(**kwargs)

    logger.info(f"Created {queue_cls.__name__} with {kwargs}")
    return queue
