from __future__ import annotations

import io
import json
import struct
import zipfile
from pathlib import Path

from .formats import EMBEDDED_SIGNATURES


def extract_binary_parts(path: str | Path, output_dir: str | Path) -> dict:
    """Write raw PE sections and a manifest. Reads bytes only; never executes code."""
    source, destination = Path(path), Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    data = source.read_bytes()
    manifest: dict = {"source": str(source), "size_bytes": len(data), "parts": []}
    if not data.startswith(b"MZ") or len(data) < 0x40:
        raise ValueError("binary-parts currently supports Windows PE files beginning with MZ")
    pe_offset = struct.unpack_from("<I", data, 0x3C)[0]
    if data[pe_offset:pe_offset + 4] != b"PE\0\0":
        raise ValueError("invalid PE header")
    _, section_count, _, _, _, optional_size, _ = struct.unpack_from("<HHIIIHH", data, pe_offset + 4)
    header_end = pe_offset + 24 + optional_size + section_count * 40
    header_path = destination / "pe_headers.bin"
    header_path.write_bytes(data[:min(header_end, len(data))])
    manifest["parts"].append({"name": "headers", "offset": 0, "size": header_path.stat().st_size, "path": str(header_path)})
    table_offset = pe_offset + 24 + optional_size
    for index in range(section_count):
        entry = table_offset + index * 40
        name = data[entry:entry + 8].split(b"\0", 1)[0].decode("ascii", "replace") or f"section_{index}"
        raw_size, raw_offset = struct.unpack_from("<II", data, entry + 16)
        content = data[raw_offset:min(raw_offset + raw_size, len(data))] if raw_offset < len(data) else b""
        safe_name = "".join(char if char.isalnum() or char in "._-" else "_" for char in name)
        section_path = destination / f"{index:02d}_{safe_name}.bin"
        section_path.write_bytes(content)
        manifest["parts"].append({"name": name, "offset": raw_offset, "size": len(content), "path": str(section_path)})
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest

def extract_embedded(path: str | Path, output_dir: str | Path, min_offset: int = 1) -> list[dict]:
    """Carve recognizable embedded files without executing or decrypting content.

    ZIP data is extracted only when it starts at a detected signature and passes
    ZIP validation. Other formats are preserved from their signature to EOF for
    analyst review, avoiding guesses about encrypted or packed payload bounds.
    """
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
