from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any


SCHEMA = "navcim.meta-pipeline/v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_reference(path: Path) -> dict[str, str]:
    path = path.resolve()
    return {"path": str(path), "sha256": sha256(path)}


def verify_file(reference: dict[str, str]) -> Path:
    path = Path(reference["path"])
    if not path.is_file():
        raise FileNotFoundError(path)
    if sha256(path) != reference["sha256"]:
        raise ValueError(f"Meta pipeline input changed after registration: {path}")
    return path


def write_manifest(path: Path, stage: str, payload: dict[str, Any], started: float) -> dict[str, Any]:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema": SCHEMA,
        "stage": stage,
        **payload,
        "performance": {
            **payload.get("performance", {}),
            "wall_seconds": time.perf_counter() - started,
        },
    }
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def read_manifest(path: Path, stage: str | None = None) -> dict[str, Any]:
    path = path.resolve()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema") != SCHEMA:
        raise ValueError(f"Expected {SCHEMA} manifest: {path}")
    if stage is not None and manifest.get("stage") != stage:
        raise ValueError(f"Expected {stage!r} manifest: {path}")
    return manifest
