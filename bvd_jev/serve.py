"""Interactive BVD action review; the model loads on first request."""

from __future__ import annotations

import asyncio
import os
import tempfile
import threading
from pathlib import Path

import gradio as gr
from fastapi import FastAPI, File, UploadFile

from jev_batch import Sampling, VideoJevEngine

from .config import DEFAULT_MODEL
from .policy import ACTION_QUESTION, ActionPolicy


_engine: VideoJevEngine | None = None
_lock = threading.Lock()


def classify_file(path: str, fps: float = 1.0, max_frames: int = 16,
                  context: int = 4096, uncertainty_low: float = 0.35,
                  uncertainty_high: float = 0.65) -> dict[str, object]:
    global _engine
    if not path:
        raise ValueError("Upload or select a video")
    sampling = Sampling(fps, int(max_frames), int(context))
    policy = ActionPolicy(uncertainty_low, uncertainty_high)
    with _lock:
        if _engine is None:
            _engine = VideoJevEngine(
                DEFAULT_MODEL,
                cpu_threads=int(os.environ.get("VIDEOJEV_CPU_THREADS", "4")),
            )
        prediction = _engine.predict(path, ACTION_QUESTION, sampling,
                                     guard=policy.guard)
    return policy.format({"path": path}, prediction)


api = FastAPI(title="BVD VideoJev action classifier")


@api.get("/health")
def health() -> dict[str, bool]:
    return {"ok": True, "model_loaded": _engine is not None}


@api.post("/classify")
async def classify_upload(video: UploadFile = File(...), fps: float = 1.0,
                          max_frames: int = 16, context: int = 4096,
                          uncertainty_low: float = 0.35,
                          uncertainty_high: float = 0.65) -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix="bvd-jev-api-") as directory:
        path = Path(directory) / "video.mp4"
        with path.open("wb") as output:
            while chunk := await video.read(4 << 20):
                output.write(chunk)
        result = await asyncio.to_thread(classify_file, str(path), fps, max_frames,
                                         context, uncertainty_low, uncertainty_high)
        result["path"] = video.filename
        return result


root = Path(__file__).resolve().parent.parent
examples = [
    root / "examples/instruction_remove_keyboard.mp4",
    root / "examples/cooking_stir_stew.mp4",
    root / "examples/nonaction_talking.mp4",
    root / "examples/caption_filter_miss_cards.mp4",
    root / "examples/instruction_roll_clay.mp4",
]
with gr.Blocks(title="BVD action review") as ui:
    gr.Markdown("# BVD action review\nVideoJev scores visible hands-on action.")
    video = gr.Video(label="Video", sources=["upload"], format="mp4")
    with gr.Row():
        fps = gr.Slider(0.25, 4.0, value=1.0, step=0.25, label="Sampling FPS")
        frames = gr.Slider(4, 64, value=16, step=2, label="Maximum frames")
        context = gr.Dropdown([4096, 8192, 16384, 32768], value=4096,
                              label="Token budget")
    with gr.Accordion("Review band", open=False):
        low = gr.Slider(0, 1, value=0.35, step=0.05, label="NON_ACTION below")
        high = gr.Slider(0, 1, value=0.65, step=0.05, label="ACTION above")
    output = gr.JSON(label="Decision and timing")
    gr.Button("Classify").click(classify_file, [video, fps, frames, context, low, high], output)
    gr.Examples(examples=[[str(path)] for path in examples if path.is_file()], inputs=video)

app = gr.mount_gradio_app(api, ui, path="/ui")


def main() -> None:
    """Start the interactive review server from the installed CLI."""
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
