from __future__ import annotations

from pathlib import Path

MAGIC = {
    b"MZ": "PE executable (Windows)",
    b"\x7fELF": "ELF executable (Linux/Unix)",
    b"\xcf\xfa\xed\xfe": "Mach-O executable (macOS, 64-bit)",
    b"\xfe\xed\xfa\xcf": "Mach-O executable (macOS, 64-bit, swapped)",
    b"PK\x03\x04": "ZIP archive",
    b"7z\xbc\xaf'\x1c": "7z archive",
    b"Rar!\x1a\x07": "RAR archive",
}

EMBEDDED_SIGNATURES = {
    b"PK\x03\x04": ("zip", ".zip"),
    b"\x1f\x8b\x08": ("gzip", ".gz"),
    b"7z\xbc\xaf'\x1c": ("7z", ".7z"),
    b"MZ": ("pe", ".exe"),
    b"\x7fELF": ("elf", ".elf"),
    b"%PDF-": ("pdf", ".pdf"),
    b"\x89PNG\r\n\x1a\n": ("png", ".png"),
}

def detect_format(path: Path) -> str:
    header = path.read_bytes()[:16]
    for signature, name in MAGIC.items():
        if header.startswith(signature):
            return name
    return "Unknown binary/data"