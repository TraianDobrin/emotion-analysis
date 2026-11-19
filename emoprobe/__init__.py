"""emoprobe: emotion vectors / linear probes for open-weights LLMs."""

from .emotions import CORE_EMOTIONS, EMOTIONS, TOPICS
from .probe import EmotionProbe

__all__ = ["EMOTIONS", "CORE_EMOTIONS", "TOPICS", "EmotionProbe"]
