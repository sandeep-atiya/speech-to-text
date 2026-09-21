"""Entry point: python transcribe.py [recordings or folders] [options]"""

import sys

from transcriber.cli import main

if __name__ == "__main__":
    sys.exit(main())
