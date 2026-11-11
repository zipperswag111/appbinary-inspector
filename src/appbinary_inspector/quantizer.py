"""GGUF conversion helpers for running compatible models with llama.cpp."""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


class QuantizationError(RuntimeError):
    """Raised when required llama.cpp conversion tools are unavailable."""


def _tool_path(value: str | Path | None, names: tuple[str, ...], label: str) -> str:
    if value:
        candidate = Path(value)
        if candidate.is_file():
            return str(candidate)
        resolved = shutil.which(str(value))
        if resolved:
            return resolved
        raise QuantizationError(f"{label} was not found: {value}")
    for name in names:
        resolved = shutil.which(name)
        if resolved:
            return resolved
    raise QuantizationError(f"{label} was not found. Install llama.cpp, then pass its path with the matching option.")


def quantizer_status(converter: str | Path | None = None, llama_quantize: str | Path | None = None) -> dict:
    """Report whether the two llama.cpp tools required for GGUF creation exist."""
    try:
        converter_path = _tool_path(converter, ("convert_hf_to_gguf.py",), "convert_hf_to_gguf.py")
    except QuantizationError:
        converter_path = None
    try:
        quantizer_path = _tool_path(llama_quantize, ("llama-quantize", "llama-quantize.exe", "quantize", "quantize.exe"), "llama-quantize")
    except QuantizationError:
        quantizer_path = None
    return {"ready": bool(converter_path and quantizer_path), "converter": converter_path, "llama_quantize": quantizer_path}


def quantize_hf_to_gguf(model_folder: str | Path, output: str | Path, *, quantization: str = "Q4_K_M", converter: str | Path | None = None, llama_quantize: str | Path | None = None, keep_f16: bool = False) -> dict:
    """Convert a local compatible Hugging Face folder into a quantized GGUF."""
    source = Path(model_folder).expanduser().resolve()
    destination = Path(output).expanduser().resolve()
    if not source.is_dir() or not (source / "config.json").is_file():
        raise QuantizationError("model_folder must be a downloaded Hugging Face model directory containing config.json")
    if destination.suffix.lower() != ".gguf":
        raise QuantizationError("output must end in .gguf")
    converter_path = _tool_path(converter, ("convert_hf_to_gguf.py",), "convert_hf_to_gguf.py")
    quantizer_path = _tool_path(llama_quantize, ("llama-quantize", "llama-quantize.exe", "quantize", "quantize.exe"), "llama-quantize")
    destination.parent.mkdir(parents=True, exist_ok=True)
    f16_output = destination.with_suffix(".f16.gguf")
    try:
        subprocess.run([sys.executable, converter_path, str(source), "--outfile", str(f16_output), "--outtype", "f16"], check=True)
        subprocess.run([quantizer_path, str(f16_output), str(destination), quantization], check=True)
    except subprocess.CalledProcessError as error:
        raise QuantizationError(f"llama.cpp exited with status {error.returncode}") from error
    finally:
        if f16_output.exists() and not keep_f16:
            f16_output.unlink()
    if not destination.is_file():
        raise QuantizationError("llama.cpp finished without creating the GGUF output")
    return {"input": str(source), "output": str(destination), "quantization": quantization, "bytes": destination.stat().st_size, "run_with": "llama.cpp, LM Studio, or another GGUF-compatible runner"}
