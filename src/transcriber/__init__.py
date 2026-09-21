"""Hinglish call transcriber built on faster-whisper.

Public API:
    Settings      - all tunable options (transcriber.config)
    Transcriber   - loads the model once and transcribes recordings (transcriber.engine)
    to_hinglish   - Devanagari -> Hinglish converter (transcriber.hinglish)
"""

from transcriber.config import Settings
from transcriber.engine import Segment, Transcriber, Transcript
from transcriber.hinglish import to_hinglish

__all__ = ["Segment", "Settings", "Transcriber", "Transcript", "to_hinglish"]
__version__ = "1.0.0"
