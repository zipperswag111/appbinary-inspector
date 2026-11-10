from __future__ import annotations

import io
import zipfile
from pathlib import Path

from .formats import EMBEDDED_SIGNATURES

def extract_embedded(path: str | Path, output_dir: str | Path, min_offset: int = 1) -> list[dict]:
    """Carve recognizable embedded files without executing or decrypting content."""
    source = Path(path)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    data = source.read_bytes()
    found: list[dict] = []
    seen: set[tuple[int, str]] = set()
    for signature, (kind, extension) in EMBEDDED_SIGNATURES.items():
        start = 0
        while True:
            index = data.find(signature, start)
            if index < 0:
                break
            start = index + 1
            if index < min_offset or (index, kind) in seen:
                continue
            seen.add((index, kind))
            stem = f"{source.stem}_offset_{index:08x}"
            if kind == "zip":
                try:
                    with zipfile.ZipFile(io.BytesIO(data[index:])) as archive:
                        member_dir = destination / stem
                        member_dir.mkdir(exist_ok=True)
                        archive.extractall(member_dir)
                        found.append({"offset": index, "type": kind, "path": str(member_dir), "members": archive.namelist()})
                except (zipfile.BadZipFile, OSError):
                    pass
            else:
                carved = destination / f"{stem}{extension}"
                carved.write_bytes(data[index:])
                found.append({"offset": index, "type": kind, "path": str(carved), "note": "carved from signature through EOF"})
    return found