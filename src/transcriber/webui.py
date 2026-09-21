"""Web UI: pick or upload a recording, transcribe it, read it with playback, download it.

    hinglish-transcribe-web            # then open http://127.0.0.1:7860
    hinglish-transcribe-web --open     # opens the browser for you

Needs the "web" extra:  pip install -e ".[web]"   (or: uv sync --extra web)
"""

import argparse
import html
import json
import logging
import queue
import threading
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any

from transcriber.audio import find_recordings
from transcriber.cli import configure_output
from transcriber.config import LANGUAGE_CHOICES, MODEL_CHOICES, Settings
from transcriber.engine import Segment, Transcriber
from transcriber.writer import clock, transcript_paths, write_transcript

log = logging.getLogger("transcriber")

PLAYER_ID = "hinglish-player"
WEB_FORMATS = ("txt", "srt", "json")
CSS = """
.transcript { font-family: ui-monospace, Consolas, monospace; font-size: 15px; line-height: 1.7; }
.transcript .line a { color: #2563eb; text-decoration: none; margin-right: 8px; }
.transcript .line a:hover { text-decoration: underline; }
"""

# Gradio serves local files (from allowed_paths) at this route.
FILE_ROUTE = "/gradio_api/file="


def file_url(path: Path) -> str:
    return FILE_ROUTE + path.resolve().as_posix()


def render_player_html(path: Path | None) -> str:
    if path is None:
        return ""
    return (
        f'<audio id="{PLAYER_ID}" controls preload="metadata" style="width:100%" '
        f'src="{html.escape(file_url(path))}"></audio>'
    )


def render_transcript_html(segments: list[Segment]) -> str:
    """Transcript lines whose timestamps seek the audio player when clicked."""
    if not segments:
        return '<div class="transcript"><i>No speech found.</i></div>'
    lines = []
    for s in segments:
        seek = (
            f"var p=document.getElementById('{PLAYER_ID}');if(p){{p.currentTime={s.start:.2f};p.play();}}return false;"
        )
        lines.append(
            f'<div class="line"><a href="#" onclick="{seek}">[{clock(s.start)}]</a>{html.escape(s.text)}</div>'
        )
    return '<div class="transcript">' + "\n".join(lines) + "</div>"


def load_saved_segments(json_path: Path) -> list[Segment]:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    return [Segment(s["start"], s["end"], s["text"]) for s in data["segments"]]


class WebApp:
    """State behind the UI: settings, the lazily loaded model, and one transcription at a time."""

    def __init__(self, settings: Settings) -> None:
        self.settings = replace(settings, formats=WEB_FORMATS)
        self._transcriber: Transcriber | None = None
        self._lock = threading.Lock()

    @property
    def transcriber(self) -> Transcriber:
        with self._lock:
            if self._transcriber is None:
                self._transcriber = Transcriber(self.settings)
            return self._transcriber

    def recording_names(self) -> list[str]:
        folder = self.settings.recordings_dir
        if not folder.is_dir():
            return []
        return [p.name for p in find_recordings([], folder)]

    def resolve_input(self, chosen: str | None, uploaded: str | None) -> Path:
        if uploaded:
            return Path(uploaded)
        if chosen:
            return self.settings.recordings_dir / chosen
        raise ValueError("Pick a recording from the list or upload one.")

    def existing_outputs(self, path: Path) -> list[Path]:
        paths = transcript_paths(path, self.settings.transcripts_dir, WEB_FORMATS)
        return paths if all(p.is_file() for p in paths) else []

    def show_existing(self, chosen: str | None) -> tuple[str, str, list[str] | None, str]:
        """When a listed recording is picked, show its saved transcript straight away if there is one."""
        if not chosen:
            return "", "", None, ""
        path = self.settings.recordings_dir / chosen
        outputs = self.existing_outputs(path)
        if not outputs:
            return render_player_html(path), "", None, "No transcript yet. Press Transcribe."
        segments = load_saved_segments(outputs[WEB_FORMATS.index("json")])
        return (
            render_player_html(path),
            render_transcript_html(segments),
            [str(p) for p in outputs],
            "Saved transcript.",
        )

    def transcribe(
        self, chosen: str | None, uploaded: str | None, language: str, force: bool
    ) -> Iterator[tuple[str, str, list[str] | None, str]]:
        """Generator so the page updates line by line while the model works."""
        path = self.resolve_input(chosen, uploaded)
        player = render_player_html(path)
        if not force and (outputs := self.existing_outputs(path)):
            saved = load_saved_segments(outputs[WEB_FORMATS.index("json")])
            yield (
                player,
                render_transcript_html(saved),
                [str(p) for p in outputs],
                "Already transcribed (tick Redo to run again).",
            )
            return

        yield player, "", None, "Loading model..."
        transcriber = self.transcriber

        events: queue.Queue[Any] = queue.Queue()
        outcome: dict[str, Any] = {}

        def work() -> None:
            try:
                with self._lock:
                    outcome["transcript"] = transcriber.transcribe(
                        path, on_segment=events.put, on_start=lambda s: events.put(("start", s)), language=language
                    )
            except Exception as exc:  # surfaced to the page below
                outcome["error"] = exc
            finally:
                events.put(None)

        threading.Thread(target=work, daemon=True).start()
        segments: list[Segment] = []
        total = 0.0
        while (event := events.get()) is not None:
            if isinstance(event, tuple):
                total = event[1]
                yield player, "", None, f"Transcribing {clock(total)} of audio..."
                continue
            segments.append(event)
            done = f"{clock(event.end)} / {clock(total)}" if total else f"{len(segments)} lines"
            yield player, render_transcript_html(segments), None, f"Transcribing... {done}"

        if "error" in outcome:
            raise outcome["error"]
        transcript = outcome["transcript"]
        written = write_transcript(transcript, self.settings.transcripts_dir, WEB_FORMATS, self.settings.timestamps)
        speed = transcript.audio_seconds / transcript.elapsed_seconds if transcript.elapsed_seconds else 0.0
        status = (
            f"Done: {clock(transcript.audio_seconds)} of audio in {clock(transcript.elapsed_seconds)} "
            f"({speed:.1f}x realtime), language {transcript.language}. Saved to {self.settings.transcripts_dir}."
        )
        yield player, render_transcript_html(transcript.segments), [str(p) for p in written], status


def build_app(app: WebApp) -> Any:
    import gradio as gr  # imported here so the rest of the package works without the "web" extra

    with gr.Blocks(title="Hinglish Call Transcriber") as demo:
        gr.Markdown(
            "# Hinglish Call Transcriber\n"
            "Pick a recording from the recordings folder or upload one, then press **Transcribe**. "
            "Click a timestamp to play from there."
        )
        with gr.Row():
            with gr.Column(scale=1, min_width=320):
                chosen = gr.Dropdown(choices=app.recording_names(), label="Recording", value=None)
                refresh = gr.Button("Refresh list", size="sm")
                uploaded = gr.Audio(label="...or upload a recording", type="filepath", sources=["upload"])
                language = gr.Radio(choices=list(LANGUAGE_CHOICES), value="auto", label="Language")
                force = gr.Checkbox(label="Redo even if a transcript exists", value=False)
                run = gr.Button("Transcribe", variant="primary")
                status = gr.Markdown("")
                files = gr.File(label="Download", file_count="multiple", interactive=False)
            with gr.Column(scale=2):
                player = gr.HTML()
                transcript = gr.HTML()

        outputs = [player, transcript, files, status]
        refresh.click(lambda: gr.Dropdown(choices=app.recording_names()), outputs=chosen)
        chosen.change(app.show_existing, inputs=chosen, outputs=outputs)
        run.click(app.transcribe, inputs=[chosen, uploaded, language, force], outputs=outputs)
    return demo


def build_parser(defaults: Settings) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hinglish-transcribe-web", description="Web UI for the Hinglish transcriber.")
    parser.add_argument("--host", default="127.0.0.1", help="bind address; use 0.0.0.0 to allow other machines")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--open", action="store_true", help="open the browser automatically")
    parser.add_argument("--share", action="store_true", help="create a temporary public link (gradio.live)")
    parser.add_argument(
        "-m", "--model", default=defaults.model_size, help=f"one of {', '.join(MODEL_CHOICES)} or a path"
    )
    parser.add_argument("--recordings-dir", type=Path, default=defaults.recordings_dir)
    parser.add_argument("-o", "--output-dir", type=Path, default=defaults.transcripts_dir)
    return parser


def main(argv: list[str] | None = None) -> int:
    configure_output()
    defaults = Settings()
    args = build_parser(defaults).parse_args(argv)
    settings = replace(
        defaults,
        model_size=args.model,
        recordings_dir=args.recordings_dir.resolve(),
        transcripts_dir=args.output_dir.resolve(),
    )
    settings.transcripts_dir.mkdir(parents=True, exist_ok=True)
    app = WebApp(settings)
    demo = build_app(app)
    log.info("Recordings: %s | Transcripts: %s", settings.recordings_dir, settings.transcripts_dir)
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        inbrowser=args.open,
        allowed_paths=[str(settings.recordings_dir), str(settings.transcripts_dir)],
        css=CSS,  # Gradio 6 takes styling here rather than in gr.Blocks()
    )
    return 0
