"""All tunable settings in one place.

Change defaults here, or override them per run with command-line flags (see cli.py).
"""

from dataclasses import dataclass, field
from pathlib import Path

# Default input/output folders are relative to where the command is run, so running
# from the project folder uses ./recordings and ./transcripts.
DEFAULT_RECORDINGS_DIR = Path("recordings")
DEFAULT_TRANSCRIPTS_DIR = Path("transcripts")

# Optional files in the working directory, picked up automatically when they exist.
DEFAULT_GLOSSARY = Path("glossary.txt")
DEFAULT_CUSTOM_WORDS = Path("custom_words.json")

# Whisper works on 16 kHz mono audio; every input is converted to this.
SAMPLE_RATE = 16_000

# Multilingual models only. English-only models ("medium.en", ...) cannot handle Hindi/Urdu.
#   "large-v3"  best accuracy, ~3 GB download, 2-3x slower than turbo
#   "turbo"     large-v3-turbo: near large-v3 accuracy, much faster, ~1.6 GB (default)
#   "medium" / "small" / "base"  smaller and faster, noticeably less accurate
MODEL_CHOICES = ("turbo", "large-v3", "medium", "small", "base")

LANGUAGE_CHOICES = ("auto", "hi", "en")
FORMAT_CHOICES = ("txt", "srt", "json")
TIMESTAMP_CHOICES = ("clock", "seconds")


def _optional_file(explicit: Path | None, default: Path) -> Path | None:
    """An explicitly chosen file, else the default one if it exists, else None."""
    if explicit is not None:
        return explicit.expanduser().resolve()
    default = default.resolve()
    return default if default.is_file() else None


@dataclass(frozen=True)
class Settings:
    """Options that control model loading, decoding and output."""

    # --- model -------------------------------------------------------------
    model_size: str = "turbo"
    device: str = "auto"  # "auto" picks cuda when available, otherwise cpu
    compute_type: str = "auto"  # "auto" picks float16 on cuda and int8 on cpu
    cpu_threads: int = 0  # 0 = let CTranslate2 decide

    # --- decoding ----------------------------------------------------------
    beam_size: int = 5
    batch_size: int = 8

    # --- chunking (see chunking.py) ----------------------------------------
    # Speech is decoded in chunks of at most this many seconds, cut at the quietest
    # moment near an even split so no word is cut in half. Devanagari needs many tokens,
    # so 30 s windows overflow the decoder; 15 s chunks fit and give readable lines.
    chunk_seconds: int = 15
    # Silero VAD speech probability above which a moment counts as speech. Low on
    # purpose: a quiet caller must not be dropped; a false alarm only costs decoding time.
    speech_threshold: float = 0.3
    # Only silences at least this long are skipped. Shorter pauses are decoded together
    # with the speech around them, so nothing quiet is lost.
    skip_silence_seconds: float = 3.0

    # --- language ----------------------------------------------------------
    # "auto": transcribe as Hindi (then convert to Hinglish) unless Whisper is at least
    # english_threshold sure the audio is English. Hindi with many English words still
    # counts as Hindi. "hi" / "en" force the language.
    language: str = "auto"
    english_threshold: float = 0.8

    # --- vocabulary --------------------------------------------------------
    # Names Whisper should recognise (products, people), one per line, written in the
    # script of the audio. None = use ./glossary.txt when it exists.
    glossary_path: Path | None = None
    # Extra Devanagari -> Hinglish spellings as a JSON object {"देवनागरी": "hinglish"}.
    # None = use ./custom_words.json when it exists.
    custom_words_path: Path | None = None

    # --- input / output ----------------------------------------------------
    recordings_dir: Path = field(default_factory=lambda: DEFAULT_RECORDINGS_DIR.resolve())
    transcripts_dir: Path = field(default_factory=lambda: DEFAULT_TRANSCRIPTS_DIR.resolve())
    formats: tuple[str, ...] = ("txt",)
    timestamps: str = "clock"  # "clock" = [00:15 -> 00:31], "seconds" = [15.57s -> 31.66s]
    force: bool = False  # re-transcribe even when the transcript already exists

    # Only transcribe the first N seconds (0 = whole file). Handy for quick checks.
    limit_seconds: float = 0.0

    def __post_init__(self) -> None:
        if self.language not in LANGUAGE_CHOICES:
            raise ValueError(f"language must be one of {LANGUAGE_CHOICES}, got {self.language!r}")
        if self.timestamps not in TIMESTAMP_CHOICES:
            raise ValueError(f"timestamps must be one of {TIMESTAMP_CHOICES}, got {self.timestamps!r}")
        unknown = set(self.formats) - set(FORMAT_CHOICES)
        if unknown:
            raise ValueError(f"unknown output format(s): {sorted(unknown)}")
        if self.chunk_seconds <= 0 or self.chunk_seconds > 30:
            raise ValueError("chunk_seconds must be between 1 and 30")
        if not 0.0 < self.speech_threshold < 1.0:
            raise ValueError("speech_threshold must be between 0 and 1")
        if self.skip_silence_seconds < 0.5:
            raise ValueError("skip_silence_seconds must be at least 0.5")
        if not 0.0 <= self.english_threshold <= 1.0:
            raise ValueError("english_threshold must be between 0 and 1")

    @property
    def glossary_file(self) -> Path | None:
        return _optional_file(self.glossary_path, DEFAULT_GLOSSARY)

    @property
    def custom_words_file(self) -> Path | None:
        return _optional_file(self.custom_words_path, DEFAULT_CUSTOM_WORDS)
