import json
from pathlib import Path

from sentence_transformers import SentenceTransformer

import argparse

try:
    from .library import SEARCHABLE_TYPES
except ImportError:
    from library import SEARCHABLE_TYPES


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('book_path', type=Path)
    args = parser.parse_args()
    model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2",
        device="cpu",
    )

    book = json.loads(args.book_path.read_text(encoding="utf-8"))

    lengths = []

    for chapter in book["chapters"]:
        for paragraph in chapter["paragraphs"]:
            if paragraph["content_type"] not in SEARCHABLE_TYPES:
                continue

            token_ids = model.tokenizer.encode(
                paragraph["text"],
                add_special_tokens=True,
                truncation=False,
            )

            lengths.append((
                len(token_ids),
                paragraph["paragraph_id"],
            ))

    limit = model.max_seq_length
    too_long = [item for item in lengths if item[0] > limit]

    print(f"Model input limit: {limit} tokens")
    print(f"Searchable paragraphs: {len(lengths)}")
    print(f"Paragraphs exceeding limit: {len(too_long)}")

    print("\nFive longest paragraphs:")
    for length, paragraph_id in sorted(lengths, reverse=True)[:5]:
        print(f"{length:5d} tokens | {paragraph_id}")


if __name__ == "__main__":
    main()
