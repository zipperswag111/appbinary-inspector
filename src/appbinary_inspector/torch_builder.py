"""Config-first PyTorch model construction with no hidden training framework."""
from __future__ import annotations

from copy import deepcopy

PRESETS = {
    "classifier": {"input_features": 128, "layers": [{"type": "linear", "out_features": 256}, {"type": "relu"}, {"type": "dropout", "p": 0.2}, {"type": "linear", "out_features": 10}]},
    "autoencoder": {"input_features": 128, "layers": [{"type": "linear", "out_features": 64}, {"type": "relu"}, {"type": "linear", "out_features": 16}, {"type": "relu"}, {"type": "linear", "out_features": 64}, {"type": "relu"}, {"type": "linear", "out_features": 128}]},
    "text_transformer": {"vocab_size": 20_000, "embedding_dim": 128, "sequence_length": 128, "transformer_layers": 2, "heads": 4, "classes": 2},
}


def preset(name: str) -> dict:
    if name not in PRESETS:
        raise ValueError(f"Unknown preset: {name}. Choose from {', '.join(PRESETS)}")
    return deepcopy(PRESETS[name])


def create_model(config: dict):
    """Build an nn.Module from a transparent config. Requires ``pip install appbinary-inspector[torch]``."""
    try:
        import torch.nn as nn
    except ImportError as error:
        raise RuntimeError("PyTorch is optional. Install it with: pip install appbinary-inspector[torch]") from error

    if config.get("kind") == "transformer" or "vocab_size" in config:
        required = ("vocab_size", "embedding_dim", "sequence_length", "transformer_layers", "heads", "classes")
        missing = [key for key in required if key not in config]
        if missing:
            raise ValueError(f"Missing transformer settings: {', '.join(missing)}")
        d_model = config["embedding_dim"]
        encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=config["heads"], batch_first=True),
            num_layers=config["transformer_layers"],
        )

        class TokenClassifier(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.embedding = nn.Embedding(config["vocab_size"], d_model)
                self.encoder = encoder
                self.head = nn.Linear(d_model, config["classes"])

            def forward(self, token_ids):
                encoded = self.encoder(self.embedding(token_ids))
                return self.head(encoded.mean(dim=1))

        return TokenClassifier()

    features = config.get("input_features")
    if not isinstance(features, int) or features < 1:
        raise ValueError("input_features must be a positive integer")
    layers = []
    for spec in config.get("layers", []):
        kind = spec.get("type", "").lower()
        if kind == "linear":
            out_features = spec.get("out_features")
            if not isinstance(out_features, int) or out_features < 1:
                raise ValueError("linear layers require positive out_features")
            layers.append(nn.Linear(features, out_features))
            features = out_features
        elif kind in {"relu", "gelu", "tanh", "sigmoid", "flatten"}:
            layers.append({"relu": nn.ReLU, "gelu": nn.GELU, "tanh": nn.Tanh, "sigmoid": nn.Sigmoid, "flatten": nn.Flatten}[kind]())
        elif kind == "dropout":
            layers.append(nn.Dropout(float(spec.get("p", 0.5))))
        else:
            raise ValueError(f"Unsupported layer type: {kind}")
    if not layers:
        raise ValueError("config requires at least one layer")
    return nn.Sequential(*layers)
