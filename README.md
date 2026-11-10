# appbinary-inspector

Static inspection of application binaries without executing them.

## Features

- Detects PE, ELF, Mach-O, ZIP, 7z, and RAR headers
- Calculates SHA-256 and SHA-1 hashes
- Extracts readable strings and entropy
- Reads basic PE header metadata
- Extracts embedded ZIP archives and carves recognizable binary signatures

## Install

```bash
pip install appbinary-inspector
```

## Use

```bash
appbin inspect program.exe
appbin extract program.exe --output extracted
```

## Safety

Targets are read as bytes only and never executed. The package does not defeat encryption, DRM, licensing controls, passwords, or code protections. It can analyze plaintext and extract recognizable embedded content from files you are authorized to inspect.