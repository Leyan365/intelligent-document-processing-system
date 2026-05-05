"""Run a basic text extraction check for a local PDF or image sample."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from idp_system.pipeline.loader import extract_text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Path to a PDF or image sample")
    args = parser.parse_args()

    result = extract_text(args.source)
    print(f"document_type: {result.doc_type.value}")
    print(f"extraction_method: {result.extraction_method}")
    print(f"characters: {len(result.text)}")
    print("metadata:")
    print(json.dumps(result.metadata, indent=2, default=str))
    print("preview:")
    print(result.text[:500])


if __name__ == "__main__":
    main()
