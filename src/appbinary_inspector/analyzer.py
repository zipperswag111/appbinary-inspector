from __future__ import annotations

import hashlib
import math
import re
import struct
from pathlib import Path

from .formats import detect_format

PRINTABLE = re.compile(rb"[\x20-\x7e]{4,}")

def _entropy(data: bytes) -> float:
    if not data:
        return 0.0
    counts = [0] * 256
    for value in data:
        counts[value] += 1
    return -sum((n / len(data)) * math.log2(n / len(data)) for n in counts if n)

def _pe_metadata(data: bytes) -> dict:
    if not data.startswith(b"MZ") or len(data) < 0x40:
        return {}
    offset = struct.unpack_from("<I", data, 0x3C)[0]
    if offset + 24 > len(data) or data[offset:offset + 4] != b"PE\0\0":
        return {"valid": False}
    machine, sections, timestamp, _, _, opt_size, characteristics = struct.unpack_from("<HHIIIHH", data, offset + 4)
    machine_names = {0x14C: "x86", 0x8664: "x86-64", 0xAA64: "ARM64"}
    return {"valid": True, "architecture": machine_names.get(machine, hex(machine)), "section_count": sections, "timestamp_utc": timestamp, "optional_header_size": opt_size, "characteristics": hex(characteristics)}

def inspect_file(path: str | Path, string_limit: int = 200) -> dict:
    """Return static metadata. The target file is never executed."""
    target = Path(path)
    data = target.read_bytes()
    report = {"path": str(target), "size_bytes": len(data), "format": detect_format(target), "sha256": hashlib.sha256(data).hexdigest(), "sha1": hashlib.sha1(data).hexdigest(), "entropy": round(_entropy(data), 4), "strings": [s.decode("utf-8", "replace") for s in PRINTABLE.findall(data)[:string_limit]]}
    if data.startswith(b"MZ"):
        report["pe"] = _pe_metadata(data)
    return report