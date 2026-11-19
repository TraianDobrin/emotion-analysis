"""emoprobe: emotion vectors / linear probes for open-weights LLMs."""

from .emotions import CORE_EMOTIONS, EMOTIONS, HAPPY_MIN, TOPICS

__all__ = ["EMOTIONS", "CORE_EMOTIONS", "HAPPY_MIN", "TOPICS", "EmotionProbe"]


def __getattr__(name):
    # Import torch-dependent code lazily, so data generation works without torch.
    if name == "EmotionProbe":
        from .probe import EmotionProbe

        return EmotionProbe
    raise AttributeError(name)
