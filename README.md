# appbinary-inspector

Static inspection tools plus an open-source, disk-paged character-model architecture.

## Features

- Detects PE, ELF, Mach-O, ZIP, 7z, and RAR headers
- Calculates SHA-256 and SHA-1 hashes
- Extracts readable strings and entropy
- Reads basic PE header metadata
- Extracts embedded ZIP archives and carves recognizable binary signatures
- Exports PE headers and raw binary sections for authorized analysis
- Builds a compact tokenizer for source code, logs, or extracted text
- Provides editable PyTorch classifier, autoencoder, and transformer templates
- Keeps CardStack model cards as separately stored 4-bit SSD files
- Loads only router-selected cards into a small LRU cache during generation
- Generates one next byte at a time for true character-level output

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

## CardStack: low-VRAM model experiments

CardStack is a small trainable byte-level language model. Its expert cards are
stored as separate 4-bit files. At inference, inactive cards remain on disk and
the cache retains only a limited number of recently selected cards.

```python
from appbinary_inspector import CardStackConfig, CardStackLM, train_next_character

model = CardStackLM(CardStackConfig(width=128, layers=4, cards_per_layer=8, active_cards=2, cache_cards=4))
losses = train_next_character(model, open("training.txt", encoding="utf-8").read(), steps=500)
model.save("my-card-model")  # writes a base file plus separate 4-bit card files

paged = CardStackLM.load("my-card-model")
print(paged.generate("Hello", new_characters=80))
```

CLI version:

```bash
appbin card-create --output my-card-model --width 128 --layers 4 --cards 8 --active-cards 2
appbin card-train my-card-model training.txt --steps 500
appbin card-generate my-card-model "Hello" --characters 80
```

## Text chat interface

Open a local chat box after training a model. It binds only to your computer,
opens in your browser, and never sends prompts to a remote service.

```bash
appbin chat my-card-model
```

## Python import API

Use the model from normal Python code without a terminal command interface:

```python
from appbinary_inspector import appbincardcr

mycard = appbincardcr("my-card-model")
reply = mycard("Explain gravity", output=True)

# Or call it in one line:
reply = appbincardcr(mycard, "Explain gravity", output=True)
```

## Hugging Face models

Hub IDs and local Transformers directories work through the same API.

```python
from appbinary_inspector import appbincardcr

mycard = appbincardcr(
    "Qwen/Qwen2.5-1.5B-Instruct",
    backend="huggingface",
    device_map="auto",
)
mycard("Explain gravity", output=True)
```

Install support with `pip install "appbinary-inspector[huggingface]"`.
For CUDA 4-bit loading, install `appbinary-inspector[huggingface-4bit]` and
pass `load_in_4bit=True`. Standard Hugging Face models keep their original
dense layout; only models trained or converted for CardStack use disk-card paging.

This reduces active model-card memory. It does not make a dense, unmodified
70B model fit instantly on a low-VRAM GPU: disk bandwidth and CPU/GPU transfer
time become the tradeoff, and models need to be trained for sparse routing.

## Convert a large model for low VRAM

`appbin quantize` creates a 4-bit GGUF for compatible locally downloaded
Hugging Face model folders. GGUF runs in llama.cpp and apps such as LM Studio,
where CPU and SSD offload can be used instead of requiring large VRAM.

Install a current llama.cpp build first. The command needs its
`convert_hf_to_gguf.py` script and `llama-quantize` executable:

```bash
appbin quantize C:\\models\\DeepSeek-R1-Distill-Qwen-32B --output C:\\models\\deepseek-32b-q4.gguf --level Q4_K_M --converter C:\\llama.cpp\\convert_hf_to_gguf.py --llama-quantize C:\\llama.cpp\\build\\bin\\Release\\llama-quantize.exe
```

`Q4_K_M` is the recommended start. A 32B Q4 model still needs roughly 18–22 GB
of disk space and substantial system RAM, but does not require a large GPU.
Conversion temporarily needs extra disk space for an F16 GGUF.

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
