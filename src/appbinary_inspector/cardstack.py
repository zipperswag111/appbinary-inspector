"""A compact, character-by-character language model with on-demand expert cards.

This is a research-sized model architecture, not a pre-trained replacement for
large commercial models. Each layer routes work to only ``active_cards`` expert
cards. Serialized cards live in separate files and are loaded into an LRU cache
only when the router selects them.
"""
from __future__ import annotations

import json
import math
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path


def _torch():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as functional
        return torch, nn, functional
    except ImportError as error:
        raise RuntimeError("Install PyTorch first: pip install appbinary-inspector[torch]") from error


@dataclass
class CardStackConfig:
    width: int = 128
    layers: int = 4
    cards_per_layer: int = 8
    active_cards: int = 2
    context_size: int = 128
    cache_cards: int = 4
    card_bits: int = 4


def _pack_int4(values):
    """Pack signed 4-bit values two per byte for SSD storage."""
    torch, _, _ = _torch()
    unsigned = (values.to(torch.int16) + 8).to(torch.uint8).flatten()
    if unsigned.numel() % 2:
        unsigned = torch.cat([unsigned, torch.zeros(1, dtype=torch.uint8)])
    return unsigned[0::2] | (unsigned[1::2] << 4)


def _unpack_int4(packed, count: int):
    torch, _, _ = _torch()
    low = packed & 0x0F
    high = packed >> 4
    return (torch.stack((low, high), dim=1).flatten()[:count].to(torch.int16) - 8).to(torch.float32)


def quantize_card(state: dict) -> dict:
    """Convert a card state dict to independently scaled 4-bit tensors."""
    torch, _, _ = _torch()
    tensors = {}
    for name, tensor in state.items():
        source = tensor.detach().cpu().float()
        scale = max(float(source.abs().max()) / 7.0, 1e-8)
        quantized = torch.round(source / scale).clamp(-8, 7).to("int8")
        tensors[name] = {"shape": list(source.shape), "scale": scale, "count": source.numel(), "packed": _pack_int4(quantized)}
    return {"format": "cardstack-int4-v1", "tensors": tensors}


def dequantize_card(payload: dict) -> dict:
    """Restore a quantized card state dict just before the card is needed."""
    if payload.get("format") != "cardstack-int4-v1":
        return payload
    torch, _, _ = _torch()
    result = {}
    for name, item in payload["tensors"].items():
        values = _unpack_int4(item["packed"], item["count"]) * item["scale"]
        result[name] = values.reshape(item["shape"])
    return result


class CardStackLM:
    """Byte-level LM. A generation step emits exactly one next byte/character."""

    def __init__(self, config: CardStackConfig, card_directory: str | Path | None = None) -> None:
        torch, nn, _ = _torch()
        if not 1 <= config.active_cards <= config.cards_per_layer:
            raise ValueError("active_cards must be between 1 and cards_per_layer")
        self.torch, self.nn, self.config = torch, nn, config
        self.embedding = nn.Embedding(256, config.width)
        self.position = nn.Embedding(config.context_size, config.width)
        self.routers = nn.ModuleList([nn.Linear(config.width, config.cards_per_layer) for _ in range(config.layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(config.width) for _ in range(config.layers)])
        self.head = nn.Linear(config.width, 256)
        self.card_directory = Path(card_directory) if card_directory else None
        self.experts = nn.ModuleList() if not card_directory else None
        if self.experts is not None:
            for _ in range(config.layers):
                self.experts.append(nn.ModuleList([self._new_card() for _ in range(config.cards_per_layer)]))
        self.cache: OrderedDict[tuple[int, int], object] = OrderedDict()

    def _new_card(self):
        nn = self.nn
        return nn.Sequential(nn.Linear(self.config.width, self.config.width * 2), nn.GELU(), nn.Linear(self.config.width * 2, self.config.width))

    def _card(self, layer: int, index: int):
        if self.experts is not None:
            return self.experts[layer][index]
        key = (layer, index)
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        if self.card_directory is None:
            raise RuntimeError("No card storage configured")
        card = self._new_card()
        state = self.torch.load(self.card_directory / f"layer_{layer:02d}_card_{index:02d}.pt", map_location="cpu", weights_only=True)
        state = dequantize_card(state)
        card.load_state_dict(state)
        card.eval()
        self.cache[key] = card
        while len(self.cache) > self.config.cache_cards:
            self.cache.popitem(last=False)
        return card

    def prefetch(self, layer: int, card_indices: list[int]) -> None:
        """Load likely next cards before the caller reaches that layer."""
        if self.experts is None:
            for index in card_indices:
                self._card(layer, int(index))

    def parameters(self):
        modules = [self.embedding, self.position, self.routers, self.norms, self.head]
        if self.experts is not None:
            modules.append(self.experts)
        for module in modules:
            yield from module.parameters()

    def train(self) -> None:
        for module in (self.embedding, self.position, self.routers, self.norms, self.head):
            module.train()
        if self.experts is not None:
            self.experts.train()

    def eval(self) -> None:
        for module in (self.embedding, self.position, self.routers, self.norms, self.head):
            module.eval()
        if self.experts is not None:
            self.experts.eval()

    def forward(self, token_ids):
        torch = self.torch
        _, sequence = token_ids.shape
        if sequence > self.config.context_size:
            token_ids = token_ids[:, -self.config.context_size:]
            sequence = token_ids.shape[1]
        positions = torch.arange(sequence, device=token_ids.device)
        hidden = self.embedding(token_ids) + self.position(positions)[None, :, :]
        for layer, (router, norm) in enumerate(zip(self.routers, self.norms)):
            normalized = norm(hidden)
            scores = router(normalized)
            values, indices = scores.topk(self.config.active_cards, dim=-1)
            # After layer 0 the current representation is a useful cheap hint for
            # the next router, so prefetch its likely cards from SSD into the LRU.
            if self.experts is None and layer + 1 < self.config.layers:
                next_scores = self.routers[layer + 1](normalized)
                next_indices = next_scores.topk(self.config.active_cards, dim=-1).indices.unique().tolist()
                self.prefetch(layer + 1, next_indices)
            weights = values.softmax(dim=-1)
            routed = torch.zeros_like(hidden)
            # Run only cards selected anywhere in this batch, then apply their gate.
            for card_index in indices.unique().tolist():
                card_output = self._card(layer, card_index)(normalized)
                gate = (weights * (indices == card_index)).sum(dim=-1, keepdim=True)
                routed = routed + card_output * gate
            hidden = hidden + routed
        return self.head(hidden)

    def save(self, directory: str | Path, quantize_cards: bool = True) -> None:
        if self.experts is None:
            raise RuntimeError("A paged inference model cannot be saved as a new training bundle")
        destination = Path(directory)
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "config.json").write_text(json.dumps(asdict(self.config), indent=2), encoding="utf-8")
        self.torch.save({"embedding": self.embedding.state_dict(), "position": self.position.state_dict(), "routers": self.routers.state_dict(), "norms": self.norms.state_dict(), "head": self.head.state_dict()}, destination / "base.pt")
        for layer, cards in enumerate(self.experts):
            for index, card in enumerate(cards):
                payload = quantize_card(card.state_dict()) if quantize_cards else card.state_dict()
                self.torch.save(payload, destination / f"layer_{layer:02d}_card_{index:02d}.pt")

    @classmethod
    def load(cls, directory: str | Path) -> "CardStackLM":
        folder = Path(directory)
        config = CardStackConfig(**json.loads((folder / "config.json").read_text(encoding="utf-8")))
        model = cls(config, card_directory=folder)
        state = model.torch.load(folder / "base.pt", map_location="cpu", weights_only=True)
        for name in ("embedding", "position", "routers", "norms", "head"):
            getattr(model, name).load_state_dict(state[name])
        model.eval()
        return model

    @classmethod
    def load_for_training(cls, directory: str | Path) -> "CardStackLM":
        """Load every card into memory for training. Use ``load`` for low-memory inference."""
        folder = Path(directory)
        config = CardStackConfig(**json.loads((folder / "config.json").read_text(encoding="utf-8")))
        model = cls(config)
        state = model.torch.load(folder / "base.pt", map_location="cpu", weights_only=True)
        for name in ("embedding", "position", "routers", "norms", "head"):
            getattr(model, name).load_state_dict(state[name])
        for layer, cards in enumerate(model.experts):
            for index, card in enumerate(cards):
                payload = model.torch.load(folder / f"layer_{layer:02d}_card_{index:02d}.pt", map_location="cpu", weights_only=True)
                card.load_state_dict(dequantize_card(payload))
        return model

    def generate(self, prompt: str, new_characters: int = 80, temperature: float = 0.8) -> str:
        torch = self.torch
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        values = list(prompt.encode("utf-8", "replace")) or [32]
        self.eval()
        with torch.no_grad():
            for _ in range(new_characters):
                input_ids = torch.tensor([values[-self.config.context_size:]], dtype=torch.long)
                logits = self.forward(input_ids)[0, -1] / temperature
                next_value = torch.multinomial(torch.softmax(logits, dim=-1), 1).item()
                values.append(next_value)
        return bytes(values).decode("utf-8", "replace")


def train_next_character(model: CardStackLM, text: str, steps: int = 500, learning_rate: float = 3e-4) -> list[float]:
    """Train a small model from user-supplied text. Returns sampled loss values."""
    torch, _, functional = _torch()
    data = torch.tensor(list(text.encode("utf-8", "replace")), dtype=torch.long)
    if len(data) <= model.config.context_size:
        raise ValueError("Training text must exceed context_size bytes")
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    losses = []
    for step in range(steps):
        start = torch.randint(0, len(data) - model.config.context_size - 1, (1,)).item()
        inputs = data[start:start + model.config.context_size].unsqueeze(0)
        targets = data[start + 1:start + model.config.context_size + 1].unsqueeze(0)
        optimizer.zero_grad()
        loss = functional.cross_entropy(model.forward(inputs).reshape(-1, 256), targets.reshape(-1))
        loss.backward()
        optimizer.step()
        if step % max(1, steps // 10) == 0 or step == steps - 1:
            losses.append(round(float(loss.detach()), 4))
    return losses
