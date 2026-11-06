import json
from pathlib import Path

from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"

WINDOW = 220
OVERLAP = 40


def main():
    model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2",
        device="cpu",
    )
    tokenizer = model.tokenizer

    book = json.loads(
        (DATA_DIR / "the_prince.json").read_text(encoding="utf-8")
    )

    passages = []

    for chapter in book["chapters"]:
        for paragraph in chapter["paragraphs"]:
            if paragraph["content_type"] == "footnote":
                continue

            text = paragraph["text"]

            encoded = tokenizer(
                text,
                add_special_tokens=False,
                truncation=False,
                return_offsets_mapping=True,
            )
            offsets = encoded["offset_mapping"]

            start = 0
            part = 1

            while start < len(offsets):
                end = min(start + WINDOW, len(offsets))

                char_start = offsets[start][0]
                char_end = offsets[end - 1][1]
                passage_text = text[char_start:char_end]

                # Recheck the actual substring, including special tokens.
                token_count = len(tokenizer.encode(
                    passage_text,
                    add_special_tokens=True,
                    truncation=False,
                ))

                if token_count > model.max_seq_length:
                    raise ValueError(
                        f"Passage exceeds model limit: "
                        f"{paragraph['paragraph_id']}, part {part}"
                    )

                passages.append({
                    "passage_id": (
                        f"{paragraph['paragraph_id']}:part{part:03d}"
                    ),
                    "paragraph_id": paragraph["paragraph_id"],
                    "chapter_id": chapter["chapter_id"],
                    "chapter_title": chapter["title"],
                    "content_type": paragraph["content_type"],
                    "char_start": char_start,
                    "char_end": char_end,
                    "text": passage_text,
                    "token_count": token_count,
                })

                if end == len(offsets):
                    break

                start = end - OVERLAP
                part += 1

    output = {
        "book_id": book["book_id"],
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "chunking_version": "token-window-v1",
        "window_tokens": WINDOW,
        "overlap_tokens": OVERLAP,
        "passages": passages,
    }

    output_path = DATA_DIR / "the_prince_passages.json"
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Created {len(passages)} passages")
    print(f"Largest passage: {max(p['token_count'] for p in passages)} tokens")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()