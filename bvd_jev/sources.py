"""Read BVD clip manifests, TAR offsets, and local MP4 files."""

from __future__ import annotations

import json
import tarfile
import tempfile
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any


def input_rows(source: Path) -> Iterator[dict[str, Any]]:
    if source.is_dir():
        for archive in sorted(source.rglob("*.tar")):
            yield from input_rows(archive)
        for path in sorted(source.rglob("*.mp4")):
            yield {"path": str(path), "id": str(path)}
    elif source.suffix == ".tar":
        with tarfile.open(source, "r:") as archive:
            for member in archive:
                if member.isfile() and member.name.lower().endswith(".mp4") and member.size > 0:
                    yield {"id": f"{source.name}:{member.name}",
                           "source_tar": str(source), "source_member": member.name,
                           "source_byte_offset": member.offset_data, "bytes": member.size}
    elif source.suffix == ".jsonl":
        with source.open() as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)
    elif source.suffix == ".json":
        content = json.loads(source.read_text())
        yield from content["records"] if isinstance(content, dict) else content
    else:
        yield {"path": str(source), "id": str(source)}


def clip_key(row: Mapping[str, Any]) -> str:
    tar = row.get("source_tar") or row.get("tar_path")
    offset = row.get("source_byte_offset", row.get("byte_offset"))
    if tar is not None and offset is not None:
        return f"tar:{Path(tar).resolve()}@{int(offset)}"
    path = row.get("file") or row.get("path")
    if path:
        return f"file:{Path(path).resolve()}"
    raise ValueError("Clip row needs a path or TAR byte offset")


@contextmanager
def local_video(row: Mapping[str, Any]) -> Iterator[Path]:
    path = row.get("path") or row.get("file")
    if path:
        yield Path(path)
        return
    tar_path = row.get("source_tar") or row.get("tar_path")
    offset = row.get("source_byte_offset", row.get("byte_offset"))
    size = row.get("bytes")
    if tar_path is None or offset is None or size is None:
        raise ValueError("Clip row needs path/file or TAR path, offset, and bytes")
    with tempfile.TemporaryDirectory(prefix="bvd-jev-") as directory:
        clip_path = Path(directory) / "clip.mp4"
        with open(tar_path, "rb") as archive, clip_path.open("wb") as output:
            archive.seek(int(offset))
            remaining = int(size)
            while remaining:
                chunk = archive.read(min(remaining, 4 << 20))
                if not chunk:
                    raise EOFError(f"Short TAR read: {tar_path} at {offset}")
                output.write(chunk)
                remaining -= len(chunk)
        yield clip_path
