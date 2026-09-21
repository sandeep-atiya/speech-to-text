# Hinglish Call Transcriber

Transcribes Hindi/Urdu call recordings into **Hinglish** (Hindi written in Roman letters,
the way people type in chat) using [faster-whisper](https://github.com/SYSTRAN/faster-whisper).

```
[0.94s -> 15.57s] hello ji namaste ji batayein main hakim sulimaan ji ke yahan se baat kar raha hoon batayein kya madad kar sakta hoon
[16.37s -> 31.66s] ... ghutnon mein donon taraf se jo hota hai ... haan to gap kam ho gaya hai theek hai
```

Whisper cannot write Hinglish by itself, so each recording is transcribed as Hindi
(Devanagari) and converted to Roman letters by a rule-based transliterator with an
editable spelling table. Recordings that are clearly English are transcribed as English.

## Project layout

```
speech-to-text/
├── pyproject.toml           # package metadata, dependencies, console command, tool config
├── uv.lock                  # exact dependency versions (uv)
├── transcribe.py            # convenience launcher:  python transcribe.py [files] [options]
├── src/transcriber/         # application package
│   ├── config.py            #   all settings and their defaults (Settings dataclass)
│   ├── audio.py             #   find recordings, decode audio to 16 kHz mono
│   ├── engine.py            #   model loading, language detection, transcription
│   ├── writer.py            #   txt / srt / json output
│   ├── cli.py               #   command-line parsing and the main loop
│   └── hinglish/            #   Devanagari -> Hinglish converter
│       ├── rules.py         #     transliteration rules (schwa deletion, vowels, nasals)
│       └── words.py         #     spelling table for common words and English loanwords
├── tests/                   # pytest test suite
├── recordings/              # put call recordings here (mp3, wav, m4a, mp4, ...)
├── transcripts/             # transcripts are written here as <name>.hinglish.txt
├── docs/                    # reference material (upstream faster-whisper README)
├── .pre-commit-config.yaml  # ruff + mypy run before every commit
└── .github/workflows/ci.yml # lint, type check and tests on every push
```

## Setup

Python 3.10 or newer. FFmpeg is not needed (audio decoding is bundled).

With [uv](https://docs.astral.sh/uv/) (recommended, uses the lock file):

```bash
uv sync                 # creates .venv with the exact locked versions
uv run hinglish-transcribe --help
```

Or with plain pip into your current Python:

```bash
pip install -e .        # add ".[web]" for the web UI
hinglish-transcribe --help
```

The Whisper model (about 1.6 GB for `turbo`) is downloaded from Hugging Face the first
time it is used and cached in `~/.cache/huggingface`.

## Usage

Run from the project folder so the default `recordings/` and `transcripts/` folders are used.

```bash
hinglish-transcribe                          # every recording in recordings/
hinglish-transcribe call.wav folder/         # specific files or folders
hinglish-transcribe call.wav -f txt -f srt   # several output formats (txt, srt, json)
hinglish-transcribe call.wav --limit-seconds 60   # quick check on the first minute
hinglish-transcribe --help                   # all options
```

`python transcribe.py ...` does the same thing. Each line is printed as soon as it is
ready and the transcript is saved to `transcripts/<recording name>.hinglish.txt`.

## Settings

Defaults live in `src/transcriber/config.py`; every one can be overridden on the command line.

| Setting | Default | Notes |
| --- | --- | --- |
| `--model` | `turbo` | `large-v3` is the most accurate but 2-3x slower and a 3 GB download. `medium`, `small`, `base` are faster and less accurate. English-only `.en` models cannot handle Hindi. |
| `--device` | `auto` | Uses the GPU (CUDA) when available, otherwise the CPU. |
| `--compute-type` | `auto` | `float16` on GPU, `int8` on CPU. |
| `--cpu-threads` | `0` | Library default; measured within 5 % of the best setting on a 6-core CPU. |
| `--language` | `auto` | Hindi unless Whisper is at least 80 % sure the audio is English. Use `hi` or `en` to force it. |
| `--chunk-seconds` | `15` | Speech is cut at pauses into chunks of at most this length, each decoded on its own. Longer chunks can get cut off mid-sentence in Devanagari. |
| `--beam-size` | `5` | Higher is slightly more accurate and slower. |
| `--batch-size` | `8` | Chunks decoded together. Lower it if memory is short. |

## Improving the Hinglish spelling

`src/transcriber/hinglish/words.py` maps Devanagari words to the spelling you want, and is
checked before the rules run. Whisper writes English words in Devanagari (कैप्सूल), so
that table is also where loanwords get their English spelling (`capsule`). Add an entry
whenever a word comes out wrong, then run the tests.

## Development

```bash
uv sync                  # installs the dev tools too (ruff, mypy, pytest, pre-commit)
pre-commit install       # run the checks automatically before each commit
uv run ruff check .      # lint          (ruff format . to format)
uv run mypy              # type check
uv run pytest            # tests with coverage
```

The same checks run in GitHub Actions on every push.

## Accuracy tips

* Recording quality matters most: 8 kHz phone audio is harder than a clean 16 kHz mic.
* `--model large-v3` gives the best Hindi accuracy if you can wait (or have a GPU).
* A GPU with CUDA 12 and cuDNN 9 makes transcription 10x faster; see
  `docs/faster-whisper-README.md` for the required NVIDIA libraries.
