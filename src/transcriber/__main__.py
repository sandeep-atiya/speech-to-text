"""Allow `python -m transcriber` as an alternative to the hinglish-transcribe command."""

import sys

from transcriber.cli import main

if __name__ == "__main__":
    sys.exit(main())
