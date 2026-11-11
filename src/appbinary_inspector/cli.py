from __future__ import annotations

import argparse
import json
from pathlib import Path

from .analyzer import inspect_file
from .extract import extract_embedded
from .extract import extract_binary_parts
from .tokenizer import WordTokenizer
from .torch_builder import PRESETS
from .cardstack import CardStackConfig, CardStackLM, train_next_character
from .chat_server import serve_chat

def main() -> None:
    parser = argparse.ArgumentParser(description="Static application-binary inspector. Never executes targets.")
    commands = parser.add_subparsers(dest="command", required=True)
    inspect_cmd = commands.add_parser("inspect", help="inspect a binary")
    inspect_cmd.add_argument("file")
    inspect_cmd.add_argument("--strings", type=int, default=200)
    extract_cmd = commands.add_parser("extract", help="extract recognizable embedded files")
    extract_cmd.add_argument("file")
    extract_cmd.add_argument("--output", required=True)
    parts_cmd = commands.add_parser("binary-extract", help="export PE headers and raw sections")
    parts_cmd.add_argument("file")
    parts_cmd.add_argument("--output", required=True)
    tokenize_cmd = commands.add_parser("tokenize", help="train a tokenizer and encode a text/source file")
    tokenize_cmd.add_argument("file")
    tokenize_cmd.add_argument("--vocab", default="tokenizer.json")
    tokenize_cmd.add_argument("--max-vocab", type=int, default=20_000)
    preset_cmd = commands.add_parser("model-template", help="print a PyTorch model configuration preset")
    preset_cmd.add_argument("name", choices=sorted(PRESETS))
    card_create = commands.add_parser("card-create", help="create a disk-paged character model")
    card_create.add_argument("--output", required=True)
    card_create.add_argument("--width", type=int, default=128)
    card_create.add_argument("--layers", type=int, default=4)
    card_create.add_argument("--cards", type=int, default=8)
    card_create.add_argument("--active-cards", type=int, default=2)
    card_create.add_argument("--cache-cards", type=int, default=4)
    card_train = commands.add_parser("card-train", help="train a CardStack model from text")
    card_train.add_argument("model")
    card_train.add_argument("text_file")
    card_train.add_argument("--steps", type=int, default=500)
    card_generate = commands.add_parser("card-generate", help="generate one character at a time from disk-paged cards")
    card_generate.add_argument("model")
    card_generate.add_argument("prompt")
    card_generate.add_argument("--characters", type=int, default=80)
    chat_cmd = commands.add_parser("chat", help="open a local browser chat box for a trained CardStack model")
    chat_cmd.add_argument("model")
    chat_cmd.add_argument("--port", type=int, default=8765)
    chat_cmd.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    if args.command == "inspect":
        result = inspect_file(args.file, args.strings)
    elif args.command == "extract":
        result = extract_embedded(args.file, args.output)
    elif args.command == "binary-extract":
        result = extract_binary_parts(args.file, args.output)
    elif args.command == "tokenize":
        text = Path(args.file).read_text(encoding="utf-8", errors="replace")
        tokenizer = WordTokenizer().fit([text], max_vocab=args.max_vocab)
        tokenizer.save(args.vocab)
        result = {"vocabulary_path": args.vocab, "vocabulary_size": len(tokenizer.vocabulary), "tokens": tokenizer.tokenize(text), "ids": tokenizer.encode(text)}
    elif args.command == "card-create":
        config = CardStackConfig(width=args.width, layers=args.layers, cards_per_layer=args.cards, active_cards=args.active_cards, cache_cards=args.cache_cards)
        CardStackLM(config).save(args.output)
        result = {"model": args.output, "config": config.__dict__, "note": "Cards were saved in 4-bit files. Train before expecting useful output."}
    elif args.command == "card-train":
        model = CardStackLM.load_for_training(args.model)
        losses = train_next_character(model, Path(args.text_file).read_text(encoding="utf-8", errors="replace"), steps=args.steps)
        model.save(args.model)
        result = {"model": args.model, "steps": args.steps, "loss_samples": losses}
    elif args.command == "card-generate":
        result = {"text": CardStackLM.load(args.model).generate(args.prompt, new_characters=args.characters)}
    elif args.command == "chat":
        serve_chat(args.model, port=args.port, open_browser=not args.no_browser)
        return
    else:
        result = PRESETS[args.name]
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
