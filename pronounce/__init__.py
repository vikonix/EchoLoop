"""EchoLoop pronunciation analysis package.

Reuses the OpenPronounce (MIT) acoustic/phoneme comparison core as a library.
The single entry point is ``analyze``; ``load_models`` / ``warm_up`` manage the
Wav2Vec2 model lifecycle (call them in a background thread at mode startup).
"""

from .speech import (
    analyze,
    load_models,
    warm_up,
    PronunciationResult,
)

__all__ = ["analyze", "load_models", "warm_up", "PronunciationResult"]
