"""Simple Python API for CardStack bundles and Hugging Face causal language models."""
from __future__ import annotations

from pathlib import Path

from .cardstack import CardStackLM


class HuggingFaceCard:
    """Callable wrapper for standard Hub IDs or local Transformers model folders."""

    def __init__(self, model_id: str | Path, *, device_map: str | dict | None = "auto", load_in_4bit: bool = False, trust_remote_code: bool = False, cache_dir: str | Path | None = None, max_memory: dict | None = None) -> None:
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as error:
            raise RuntimeError('Hugging Face support needs: pip install "appbinary-inspector[huggingface]"') from error
        options = {"device_map": device_map, "trust_remote_code": trust_remote_code, "cache_dir": str(cache_dir) if cache_dir else None, "max_memory": max_memory, "low_cpu_mem_usage": True, "dtype": "auto"}
        options = {key: value for key, value in options.items() if value is not None}
        if load_in_4bit:
            try:
                from transformers import BitsAndBytesConfig
            except ImportError as error:
                raise RuntimeError('4-bit loading needs: pip install "appbinary-inspector[huggingface-4bit]"') from error
            options["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code, cache_dir=cache_dir)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **options)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self._input_device = self._find_input_device()
        # Transformers determines ``model.device`` from its first parameter. With
        # Accelerate disk offload that parameter can be a meta tensor even when
        # the embedding layer has a real execution device. Generation then creates
        # its special-token tensors on ``meta`` and fails before the first token.
        if self._input_device is not None and self.model.device.type == "meta":
            original_class = self.model.__class__

            class _AppBinaryOffloadModel(original_class):
                @property
                def device(model_self):
                    return model_self._appbinary_input_device

            self.model.__class__ = _AppBinaryOffloadModel
            self.model._appbinary_input_device = self._input_device

    def _find_input_device(self):
        """Return a real device for token inputs when Accelerate uses disk offload."""
        import torch

        embedding = self.model.get_input_embeddings()
        if embedding is not None and embedding.weight.device.type != "meta":
            return embedding.weight.device
        device_map = getattr(self.model, "hf_device_map", {})
        for name, mapped_device in device_map.items():
            if name.endswith(("embed_tokens", "wte", "word_embeddings")) and mapped_device != "disk":
                return torch.device(mapped_device if isinstance(mapped_device, str) else f"cuda:{mapped_device}")
        for mapped_device in device_map.values():
            if mapped_device != "disk":
                return torch.device(mapped_device if isinstance(mapped_device, str) else f"cuda:{mapped_device}")
        return None

    def __call__(self, text: str, output: bool = False, characters: int = 180, temperature: float = 0.8, system: str | None = None) -> str:
        prompt = text
        if system and getattr(self.tokenizer, "chat_template", None):
            prompt = self.tokenizer.apply_chat_template([{"role": "system", "content": system}, {"role": "user", "content": text}], tokenize=False, add_generation_prompt=True)
        if self._input_device is None:
            raise RuntimeError(
                "This model was entirely disk-offloaded, so it has no device available to run generation. "
                "Use a smaller model, enable a supported quantized GPU setup, or provide enough CPU/GPU memory "
                "for at least the embedding and active layers."
            )
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {name: value.to(self._input_device) for name, value in inputs.items()}
        generated = self.model.generate(**inputs, max_new_tokens=max(1, characters), do_sample=temperature > 0, temperature=max(temperature, 1e-5), pad_token_id=self.tokenizer.pad_token_id)
        reply = self.tokenizer.decode(generated[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        if output:
            print(reply)
        return reply

    def reply(self, text: str, **options) -> str:
        return self(text, **options)


class Card:
    """Callable wrapper for a CardStack local model bundle."""

    def __init__(self, model: str | Path | CardStackLM) -> None:
        self.model = CardStackLM.load(model) if isinstance(model, (str, Path)) else model

    def __call__(self, text: str, output: bool = False, characters: int = 180, temperature: float = 0.8, **_ignored) -> str:
        reply = self.model.generate(text, new_characters=characters, temperature=temperature)
        if output:
            print(reply)
        return reply

    def reply(self, text: str, **options) -> str:
        return self(text, **options)


def _is_cardstack_bundle(value: str | Path) -> bool:
    folder = Path(value)
    return folder.is_dir() and (folder / "base.pt").is_file() and (folder / "config.json").is_file()


def appbincardcr(mycard, text: str | None = None, output: bool = False, backend: str = "auto", **options):
    """Create or call a CardStack or Hugging Face model.

    ``appbincardcr("my-card-model")`` loads CardStack.
    ``appbincardcr("Qwen/Qwen2.5-1.5B-Instruct")`` loads Hugging Face.
    """
    if backend not in {"auto", "cardstack", "huggingface"}:
        raise ValueError("backend must be auto, cardstack, or huggingface")
    if isinstance(mycard, (Card, HuggingFaceCard)):
        card = mycard
    elif backend == "cardstack" or (backend == "auto" and isinstance(mycard, (str, Path)) and _is_cardstack_bundle(mycard)):
        card = Card(mycard)
    else:
        card = HuggingFaceCard(mycard, **options)
        options = {}
    return card if text is None else card(text, output=output, **options)
