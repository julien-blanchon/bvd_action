"""Score BVD videos from an MP4, TAR, directory, or JSON/JSONL manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import asdict
from itertools import islice
from pathlib import Path

from jev_batch import BatchConfig, BatchRunner, Item, Sampling, VideoJevEngine
from jev_batch.types import EncodedVideo, PreparedVideo

from .config import DEFAULT_MODEL
from .policy import ACTION_QUESTION, ActionPolicy
from .sources import input_rows, local_video


def _sha256(path: Path) -> str | None:
    if path.suffix not in (".json", ".jsonl") or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _runtime_sha256() -> str:
    """Pin the two maintained packages for safe resume and provenance."""
    digest = hashlib.sha256()
    root = Path(__file__).resolve().parent.parent
    paths = list((root / "jev_batch").glob("*.py"))
    paths += [root / "bvd_jev" / name for name in
              ("cli.py", "config.py", "policy.py", "sources.py")]
    for path in sorted(paths):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _model_revision(model_path: str) -> str | None:
    metadata = Path(model_path) / ".cache/huggingface/download/config.json.metadata"
    if metadata.is_file():
        revision = metadata.read_text().splitlines()[0].strip()
        if len(revision) == 40 and all(char in "0123456789abcdef" for char in revision):
            return revision
    return None


def _prepare(engine: VideoJevEngine, item: Item,
             sampling: Sampling) -> PreparedVideo | EncodedVideo:
    with local_video(item.source) as path:
        return engine.prepare(path, sampling)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, help="JSONL output; stdout by default")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--max-frames", type=int, default=16)
    parser.add_argument("--context", type=int, default=4096)
    parser.add_argument("--uncertainty-low", type=float, default=0.35)
    parser.add_argument("--uncertainty-high", type=float, default=0.65)
    parser.add_argument("--decision-guard-margin", type=float, default=0.05)
    parser.add_argument("--gpu-batch-size", type=int, default=8)
    parser.add_argument("--vision-batch-size", type=int, default=8)
    parser.add_argument("--batch-visual-token-budget", type=int, default=8192)
    parser.add_argument("--decode-workers", type=int, default=2)
    parser.add_argument("--prefetch", type=int, default=16)
    parser.add_argument("--codec-threads", type=int, default=2)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--attention", choices=("sdpa", "flash_attention_2"), default="sdpa")
    parser.add_argument("--fuse-vision-rope", action="store_true")
    parser.add_argument("--sparse-seek", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    try:
        sampling = Sampling(args.fps, args.max_frames, args.context)
        policy = ActionPolicy(args.uncertainty_low, args.uncertainty_high,
                              args.decision_guard_margin)
        config = BatchConfig(args.vision_batch_size, args.gpu_batch_size,
                             args.batch_visual_token_budget, args.decode_workers,
                             args.prefetch)
        if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
            raise ValueError("Invalid shard settings")
        if args.start < 0 or (args.limit is not None and args.limit < 0):
            raise ValueError("Start and limit must be nonnegative")
        if args.resume and args.output is None:
            raise ValueError("--resume requires --output")
    except ValueError as exc:
        parser.error(str(exc))

    run = {
        "engine": "jev_batch.VideoJevEngine",
        "model": args.model,
        "model_revision": _model_revision(args.model),
        "runtime_sha256": _runtime_sha256(),
        "input": str(args.input.resolve()),
        "input_sha256": _sha256(args.input),
        "question": ACTION_QUESTION.native(),
        "sampling": asdict(sampling),
        "policy": asdict(policy),
        "batch": asdict(config),
        "attention": args.attention,
        "cpu_threads": args.cpu_threads,
        "codec_threads": args.codec_threads,
        "sparse_seek": args.sparse_seek,
        "fuse_vision_rope": args.fuse_vision_rope,
        "start": args.start,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
    }
    completed = 0
    if args.output:
        metadata = Path(str(args.output) + ".run.json")
        if args.output.exists() and not args.resume:
            parser.error(f"Output exists: {args.output}; use --resume or a new path")
        if args.resume and metadata.exists() and json.loads(metadata.read_text()) != run:
            parser.error("Resume settings differ from existing run metadata")
        if args.resume and args.output.exists():
            with args.output.open() as previous:
                completed = sum(bool(line.strip()) for line in previous)
        metadata.write_text(json.dumps(run, indent=2) + "\n")

    rows = (row for index, row in enumerate(input_rows(args.input))
            if index >= args.start and index % args.num_shards == args.shard_index)
    rows = islice(rows, completed, None)
    if args.limit is not None:
        rows = islice(rows, args.limit)
    items = (Item(str(row.get("id", row.get("path", row.get("file", "")))), row)
             for row in rows)

    load_started = time.perf_counter()
    engine = VideoJevEngine(args.model, attention=args.attention,
                            cpu_threads=args.cpu_threads,
                            sparse_seek=args.sparse_seek,
                            codec_threads=args.codec_threads,
                            fuse_vision_rope=args.fuse_vision_rope)
    load_seconds = time.perf_counter() - load_started
    runner = BatchRunner(engine, ACTION_QUESTION, sampling=sampling, config=config,
                         guard=policy.guard,
                         prepare=lambda item, settings: _prepare(engine, item, settings))
    started = time.perf_counter()
    processed = failures = 0
    output = args.output.open("a", buffering=1) if args.output else sys.stdout
    try:
        for item, prediction in runner.run(items):
            try:
                result = policy.format(item.source, prediction)
            except Exception as exc:
                result = {"id": item.id, "error": f"{type(exc).__name__}: {exc}"}
            processed += 1
            failures += "error" in result
            if "error" in result:
                print(f"{item.id}: {result['error']}", file=sys.stderr, flush=True)
            print(json.dumps(result, ensure_ascii=False), file=output, flush=True)
    finally:
        if args.output:
            output.close()
            wall = time.perf_counter() - started
            summary = {
                "processed": processed,
                "failures": failures,
                "model_load_s": load_seconds,
                "processing_wall_s": wall,
                "valid_clips_per_s": (processed - failures) / wall if wall else None,
                "peak_allocated_gib": engine.torch.cuda.max_memory_allocated() / 2**30,
                "gpu": engine.torch.cuda.get_device_name(),
                "torch": engine.torch.__version__,
            }
            Path(str(args.output) + ".summary.json").write_text(
                json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
