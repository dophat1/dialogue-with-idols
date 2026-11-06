import json
from pathlib import Path

from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BOOK_PATH = PROJECT_ROOT / "data" / "processed" / "the_prince.json"

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    device="cpu",
)

book = json.loads(BOOK_PATH.read_text(encoding="utf-8"))

lengths = []

for chapter in book["chapters"]:
    for paragraph in chapter["paragraphs"]:
        if paragraph["content_type"] == "footnote":
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