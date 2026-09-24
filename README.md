# Hinglish Call Transcriber

Transcribes Hindi/Urdu call recordings into **Hinglish** (Hindi written in Roman letters,
the way people type in chat) using [faster-whisper](https://github.com/SYSTRAN/faster-whisper).

```
[00:00 -> 00:15] hello ji namaste ji batayein main hakim suleman ji ke yahan se baat kar raha hoon batayein kya madad kar sakta hoon
[00:16 -> 00:31] ... ghutnon mein donon taraf se jo hota hai ... haan to gap kam ho gaya hai theek hai
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
├── glossary.txt             # names Whisper should recognise (products, people)
├── custom_words.json        # your own Devanagari -> Hinglish spellings
├── src/transcriber/         # application package
│   ├── config.py            #   all settings and their defaults (Settings dataclass)
│   ├── audio.py             #   find recordings, decode audio to 16 kHz mono
│   ├── chunking.py          #   cuts a recording into chunks at pauses, skips only long silences
│   ├── engine.py            #   model loading, language detection, transcription
│   ├── cleanup.py           #   removes looping repetitions from lines
│   ├── glossary.py          #   reads glossary.txt and custom_words.json
│   ├── watch.py             #   folder watcher for --watch
│   ├── writer.py            #   txt / srt / json output
│   ├── cli.py               #   command-line parsing and the main loop
│   └── hinglish/            #   Devanagari -> Hinglish converter
│       ├── rules.py         #     transliteration rules (schwa deletion, vowels, nasals)
│       └── words.py         #     built-in spelling table for common words and loanwords
├── tests/                   # pytest test suite
├── recordings/              # put call recordings here (mp3, wav, m4a, mp4, ...)
├── transcripts/             # transcripts are written here as <name>.hinglish.txt
├── docs/                    # reference material (upstream faster-whisper README)
├── .pre-commit-config.yaml  # ruff + mypy run before every commit
└── .github/workflows/ci.yml # lint, type check and tests on every push
```

## Setup

Python 3.12 or newer. FFmpeg is not needed (audio decoding is bundled).

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

## Daily use

Drop recordings into `recordings/` and run from the project folder:

```bash
hinglish-transcribe                 # transcribes every recording that has no transcript yet
hinglish-transcribe --watch         # ...then keeps running and picks up new files as they arrive
```

Each line is printed as soon as it is ready, a progress bar shows the time remaining,
and the transcript is saved to `transcripts/<recording name>.hinglish.txt`. Recordings
that already have a transcript are skipped; add `--force` to redo them.

More options:

```bash
hinglish-transcribe call.wav folder/          # specific files or folders
hinglish-transcribe -f txt -f srt -f json     # several output formats
hinglish-transcribe --timestamps seconds      # [15.57s -> 31.66s] instead of [00:15 -> 00:31]
hinglish-transcribe --limit-seconds 60 call.wav   # quick check on the first minute
hinglish-transcribe --help                    # everything
```

`python transcribe.py ...` and `python -m transcriber ...` do the same thing.

## Web UI

For people who prefer a page to a terminal:

```bash
pip install -e ".[web]"        # once (or: uv sync --extra web)
hinglish-transcribe-web --open # opens http://127.0.0.1:7860
```

Pick a recording from the recordings folder or upload one, press **Transcribe**, and
watch the lines appear. Click any timestamp to play the call from that moment. The
transcript is saved to `transcripts/` like the command line does, with txt, srt and
json downloads. Picking a recording that already has a transcript shows it immediately.
Use `--host 0.0.0.0` to let other machines on the network open the page.

## Getting names and spellings right

Two plain files in the project folder are picked up automatically:

* **`glossary.txt`**: names Whisper should recognise, one per line, written in the
  script of the audio (Devanagari for Hindi/Urdu calls). They are passed to Whisper as
  hotwords. This also makes Whisper add punctuation. If a call comes out with a phrase
  repeated many times, the glossary is the usual cause; shorten it or pass
  `--glossary` with an empty file to switch it off for that run.
* **`custom_words.json`**: your own Devanagari -> Hinglish spellings, for example
  `{"सुलेमान": "suleman"}`. These override the built-in table in
  `src/transcriber/hinglish/words.py`. Whisper writes English words in Devanagari
  (कैप्सूल), so this is also where loanwords get their English spelling.

Whisper occasionally gets stuck repeating a phrase; such runs are trimmed automatically.

## Settings

Defaults live in `src/transcriber/config.py`; every one can be overridden on the command line.

| Setting | Default | Notes |
| --- | --- | --- |
| `--model` | `turbo` | `large-v3` is the most accurate but 2-3x slower and a 3 GB download. `medium`, `small`, `base` are faster and less accurate. English-only `.en` models cannot handle Hindi. |
| `--device` | `auto` | Uses the GPU (CUDA) when available, otherwise the CPU. |
| `--compute-type` | `auto` | `float16` on GPU, `int8` on CPU. |
| `--cpu-threads` | `0` | Library default; measured within 5 % of the best setting on a 6-core CPU. |
| `--language` | `auto` | Hindi unless Whisper is at least 80 % sure the audio is English. Use `hi` or `en` to force it. |
| `--chunk-seconds` | `15` | Speech is decoded in chunks of at most this length. Each cut is placed at the quietest moment near an even split, so words are never cut in half and there are no tiny leftovers. A chunk long enough to overflow the decoder (Devanagari needs many tokens) is decoded again in two halves automatically. |
| `--speech-threshold` | `0.3` | Voice detector sensitivity, 0-1. Lower keeps quieter speech; a false alarm only costs decoding time. |
| `--skip-silence-seconds` | `3` | Only silences at least this long are left out. Shorter pauses are decoded together with the speech around them, so nothing quiet is lost. |
| `--beam-size` | `5` | Higher is slightly more accurate and slower. |
| `--batch-size` | `8` | Chunks decoded together. Lower it if memory is short. |
| `--poll-seconds` | `10` | How often `--watch` looks for new files. |

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

* Every second of speech is decoded. The log line
  `Decoding 17s of speech in 2 chunk(s), skipping 0s of silence` says how much was left
  out; if that looks too high for the call, lower `--speech-threshold` or raise
  `--skip-silence-seconds`.
* Recording quality matters most: 8 kHz phone audio is harder than a clean 16 kHz mic.
* `--model large-v3` gives the best Hindi accuracy if you can wait (or have a GPU).
* A GPU with CUDA 12 and cuDNN 9 makes transcription 10x faster; see
  `docs/faster-whisper-README.md` for the required NVIDIA libraries.
* On a 6-core CPU the `turbo` model runs at about 1.4x realtime: a 20-minute call takes
  about 15 minutes.
