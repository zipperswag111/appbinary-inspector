"""Small Python API for using a disk-paged CardStack model in normal code."""
from __future__ import annotations

from pathlib import Path

from .cardstack import CardStackLM


class Card:
    """A callable local text model.

    Example:
        mycard = appbincardcr("my-card-model")
        reply = mycard("Explain gravity", output=True)
    """

    def __init__(self, model: str | Path | CardStackLM) -> None:
        self.model = CardStackLM.load(model) if isinstance(model, (str, Path)) else model

    def __call__(self, text: str, output: bool = False, characters: int = 180, temperature: float = 0.8) -> str:
        reply = self.model.generate(text, new_characters=characters, temperature=temperature)
        if output:
            print(reply)
        return reply

    def reply(self, text: str, **options) -> str:
        return self(text, **options)


def appbincardcr(mycard: str | Path | CardStackLM | Card, text: str | None = None, output: bool = False, **options):
    """Create or call a CardStack text model with one short function.

    ``appbincardcr("my-card-model")`` returns a reusable card.
    ``appbincardcr(mycard, "Hello", output=True)`` generates and prints text.
    """
    card = mycard if isinstance(mycard, Card) else Card(mycard)
    return card if text is None else card(text, output=output, **options)
