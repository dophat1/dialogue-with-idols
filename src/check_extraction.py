import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BOOK_PATH = PROJECT_ROOT / "data" / "processed" / "the_prince.json"

book = json.loads(BOOK_PATH.read_text(encoding="utf-8"))
chapters = book["chapters"]

# Check chapter count and order.
expected_ids = [f"chap{number:02d}" for number in range(1, 27)]
actual_ids = [chapter["chapter_id"] for chapter in chapters]

assert actual_ids == expected_ids, (
    f"Unexpected chapter order or missing chapters: {actual_ids}"
)

# Check that every chapter contains text and every paragraph ID is unique.
paragraph_ids = []

for chapter in chapters:
    assert chapter["paragraphs"], f"Empty chapter: {chapter['chapter_id']}"

    for paragraph in chapter["paragraphs"]:
        assert paragraph["text"].strip(), "Found an empty paragraph"
        paragraph_ids.append(paragraph["paragraph_id"])

assert len(paragraph_ids) == len(set(paragraph_ids)), (
    "Found duplicate paragraph IDs"
)

print(f"Passed: {len(chapters)} chapters in expected order")
print(f"Passed: {len(paragraph_ids)} nonempty paragraphs with unique IDs")

# Print a sample for manual comparison with the original HTML.
first_chapter = chapters[0]

print(f"\n{first_chapter['title']}")

for paragraph in first_chapter["paragraphs"][:3]:
    print(f"\n[{paragraph['paragraph_id']}]")
    print(paragraph["text"])