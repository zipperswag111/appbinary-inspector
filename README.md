# appbinary-inspector

Static inspection of application binaries, text tokenization, and config-first PyTorch model building without executing analyzed files.

## Features

- Detects PE, ELF, Mach-O, ZIP, 7z, and RAR headers
- Calculates SHA-256 and SHA-1 hashes
- Extracts readable strings and entropy
- Reads basic PE header metadata
- Extracts embedded ZIP archives and carves recognizable binary signatures
- Exports PE headers and raw binary sections for authorized analysis
- Builds a compact tokenizer for source code, logs, or extracted text
- Provides editable PyTorch classifier, autoencoder, and transformer templates

## Install

```bash
pip install appbinary-inspector
```

## Use

```bash
appbin inspect program.exe
appbin extract program.exe --output extracted
appbin binary-extract program.exe --output pe_parts
appbin tokenize source.py --vocab source-tokenizer.json
appbin model-template classifier
```

## PyTorch models

Install optional PyTorch support:

```bash
pip install "appbinary-inspector[torch]"
```

```python
from appbinary_inspector.torch_builder import create_model, preset

config = preset("classifier")
config["layers"][0]["out_features"] = 512
model = create_model(config)
```

## Safety

Targets are read as bytes only and never executed. The package does not defeat encryption, DRM, licensing controls, passwords, or code protections. It can analyze plaintext, export raw PE sections, and extract recognizable embedded content from files you are authorized to inspect.
