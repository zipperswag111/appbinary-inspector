from __future__ import annotations

import argparse
import json

from .analyzer import inspect_file
from .extract import extract_embedded

def main() -> None:
    parser = argparse.ArgumentParser(description="Static application-binary inspector. Never executes targets.")
    commands = parser.add_subparsers(dest="command", required=True)
    inspect_cmd = commands.add_parser("inspect", help="inspect a binary")
    inspect_cmd.add_argument("file")
    inspect_cmd.add_argument("--strings", type=int, default=200)
    extract_cmd = commands.add_parser("extract", help="extract recognizable embedded files")
    extract_cmd.add_argument("file")
    extract_cmd.add_argument("--output", required=True)
    args = parser.parse_args()
    result = inspect_file(args.file, args.strings) if args.command == "inspect" else extract_embedded(args.file, args.output)
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()