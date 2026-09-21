"""All tunable settings in one place.

Change defaults here, or override them per run with command-line flags (see cli.py).
"""

from dataclasses import dataclass, field
from pathlib import Path

# Default input/output folders are relative to where the command is run, so running
# from the project folder uses ./recordings and ./transcripts.
DEFAULT_RECORDINGS_DIR = Path("recordings")
DEFAULT_TRANSCRIPTS_DIR = Path("transcripts")

# Whisper works on 16 kHz mono audio; every input is converted to this.
SAMPLE_RATE = 16_000

# Multilingual models only. English-only models ("medium.en", ...) cannot handle Hindi/Urdu.
#   "large-v3"  best accuracy, ~3 GB download, 2-3x slower than turbo
#   "turbo"     large-v3-turbo: near large-v3 accuracy, much faster, ~1.6 GB (default)
#   "medium" / "small" / "base"  smaller and faster, noticeably less accurate
MODEL_CHOICES = ("turbo", "large-v3", "medium", "small", "base")

LANGUAGE_CHOICES = ("auto", "hi", "en")
FORMAT_CHOICES = ("txt", "srt", "json")


@dataclass(frozen=True)
class Settings:
    """Options that control model loading, decoding and output."""

    # --- model -------------------------------------------------------------
    model_size: str = "turbo"
    device: str = "auto"  # "auto" picks cuda when available, otherwise cpu
    compute_type: str = "auto"  # "auto" picks float16 on cuda and int8 on cpu
    cpu_threads: int = 0  # 0 = let CTranslate2 decide; try the number of physical cores

    # --- decoding ----------------------------------------------------------
    beam_size: int = 5
    # Speech is cut at pauses into chunks of at most this many seconds, each decoded on
    # its own. Devanagari needs many tokens, so 30 s windows get cut off mid-sentence;
    # 15 s chunks avoid that and give readable lines.
    chunk_seconds: int = 15
    batch_size: int = 8

    # --- language ----------------------------------------------------------
    # "auto": transcribe as Hindi (then convert to Hinglish) unless Whisper is at least
    # english_threshold sure the audio is English. Hindi with many English words still
    # counts as Hindi. "hi" / "en" force the language.
    language: str = "auto"
    english_threshold: float = 0.8

    # --- input / output ----------------------------------------------------
    recordings_dir: Path = field(default_factory=lambda: DEFAULT_RECORDINGS_DIR.resolve())
    transcripts_dir: Path = field(default_factory=lambda: DEFAULT_TRANSCRIPTS_DIR.resolve())
    formats: tuple[str, ...] = ("txt",)

    # Only transcribe the first N seconds (0 = whole file). Handy for quick checks.
    limit_seconds: float = 0.0

    def __post_init__(self) -> None:
        if self.language not in LANGUAGE_CHOICES:
            raise ValueError(f"language must be one of {LANGUAGE_CHOICES}, got {self.language!r}")
        unknown = set(self.formats) - set(FORMAT_CHOICES)
        if unknown:
            raise ValueError(f"unknown output format(s): {sorted(unknown)}")
        if self.chunk_seconds <= 0 or self.chunk_seconds > 30:
            raise ValueError("chunk_seconds must be between 1 and 30")
        if not 0.0 <= self.english_threshold <= 1.0:
            raise ValueError("english_threshold must be between 0 and 1")
