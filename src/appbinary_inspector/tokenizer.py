"""Small, dependency-free tokenizer for source, logs, and extracted text."""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

TOKEN_PATTERN = re.compile(r"[A-Za-z_][A-Za-z_0-9]*|\d+(?:\.\d+)?|[^\s]", re.UNICODE)
SPECIAL_TOKENS = ("<pad>", "<unk>", "<bos>", "<eos>")


class WordTokenizer:
    """Train a compact word/code tokenizer without external model downloads."""

    def __init__(self, vocabulary: dict[str, int] | None = None) -> None:
        self.vocabulary = vocabulary or {token: index for index, token in enumerate(SPECIAL_TOKENS)}
        self.inverse_vocabulary = {index: token for token, index in self.vocabulary.items()}

    @staticmethod
    def tokenize(text: str) -> list[str]:
        return TOKEN_PATTERN.findall(text)

    def fit(self, texts: list[str], max_vocab: int = 20_000, min_frequency: int = 1) -> "WordTokenizer":
        if max_vocab < len(SPECIAL_TOKENS):
            raise ValueError("max_vocab must allow the special tokens")
        counts = Counter(token for text in texts for token in self.tokenize(text))
        self.vocabulary = {token: index for index, token in enumerate(SPECIAL_TOKENS)}
        for token, count in counts.most_common(max_vocab - len(SPECIAL_TOKENS)):
            if count < min_frequency:
                continue
            self.vocabulary[token] = len(self.vocabulary)
        self.inverse_vocabulary = {index: token for token, index in self.vocabulary.items()}
        return self

    def encode(self, text: str, add_boundaries: bool = False) -> list[int]:
        ids = [self.vocabulary.get(token, self.vocabulary["<unk>"]) for token in self.tokenize(text)]
        return ([self.vocabulary["<bos>"]] + ids + [self.vocabulary["<eos>"]]) if add_boundaries else ids

    def decode(self, ids: list[int]) -> list[str]:
        return [self.inverse_vocabulary.get(index, "<unk>") for index in ids]

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({"vocabulary": self.vocabulary}, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "WordTokenizer":
        return cls(json.loads(Path(path).read_text(encoding="utf-8"))["vocabulary"])
