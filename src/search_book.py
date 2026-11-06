import json
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BOOK_PATH = PROJECT_ROOT / "data" / "processed" / "the_prince.json"


def tokenize(text):
    return set(re.findall(r"\w+", text.casefold()))


def search(book, query, limit=5):
    query_words = tokenize(query)

    if not query_words:
        return []

    results = []
    

    for chapter in book["chapters"]:
        for index, paragraph in enumerate(chapter["paragraphs"]):
            if paragraph["content_type"] == "footnote":
                continue

            matched_words = query_words & tokenize(paragraph["text"])

            if matched_words:
                results.append({
                    "score": len(matched_words),
                    "matched_words": sorted(matched_words),
                    "chapter": chapter,
                    "paragraph_index": index,
                })

    results.sort(key=lambda result: result["score"], reverse=True)
    return results[:limit]


def display_result(result):
    chapter = result["chapter"]
    index = result["paragraph_index"]
    paragraphs = chapter["paragraphs"]

    print(f"\n{'=' * 70}")
    print(chapter["title"])
    print(f"Matched words: {', '.join(result['matched_words'])}")
    print(
        "Source: https://www.gutenberg.org/cache/epub/1232/"
        f"pg1232-images.html#{chapter['chapter_id']}"
    )

    # Show the matching paragraph and one neighbor on either side.
    start = max(0, index - 1)
    stop = min(len(paragraphs), index + 2)

    for position in range(start, stop):
        paragraph = paragraphs[position]
        label = "MATCH" if position == index else "CONTEXT"

        content_type = paragraph["content_type"]

        print(
            f"\n[{label} | {content_type}] "
            f"{paragraph['paragraph_id']}"
        )
        print(paragraph["text"])


def main():
    book = json.loads(BOOK_PATH.read_text(encoding="utf-8"))

    while True:
        query = input("\nSearch words (or 'quit'): ").strip()

        if query.casefold() == "quit":
            break

        if not query:
            continue

        results = search(book, query)

        if not results:
            print("No matching passages.")
            continue

        for result in results:
            display_result(result)


if __name__ == "__main__":
    main()