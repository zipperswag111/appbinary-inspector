from __future__ import annotations

import argparse
import json
from pathlib import Path

from .analyzer import inspect_file
from .extract import extract_embedded
from .extract import extract_binary_parts
from .tokenizer import WordTokenizer
from .torch_builder import PRESETS

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
    else:
        result = PRESETS[args.name]
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
