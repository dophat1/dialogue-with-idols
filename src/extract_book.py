import json
from pathlib import Path

from bs4 import BeautifulSoup

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_ROOT / "data" / "raw" / "the_prince.html"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "the_prince.json"


def clean_text(element):
    # Collapse HTML formatting whitespace into single spaces.
    return " ".join(element.get_text().split())

def classify_paragraph(paragraph):
    classes = paragraph.get("class", [])

    if "footnote" in classes:
        return "footnote"

    if "poem" in classes:
        return "verse"

    return "body"

def extract_chapters(soup):
    chapters = []

    for container in soup.select("div.chapter"):
        heading = container.find("h2", recursive=False)

        if heading is None:
            continue

        title = clean_text(heading)

        # For this first exercise, include only the main numbered chapters.
        if not title.startswith("CHAPTER "):
            continue

        anchor = heading.find("a", id=True)

        if anchor is None:
            raise ValueError(f"Missing chapter anchor: {title}")

        chapter_id = anchor["id"]
        paragraphs = []

        for paragraph in container.find_all("p", recursive=False):
            text = clean_text(paragraph)

            if not text:
                continue

            position = len(paragraphs) + 1

            paragraphs.append({
                "paragraph_id": f"pg1232:{chapter_id}:p{position:04d}",
                "position": position,
                "content_type": classify_paragraph(paragraph),
                "text": text,
            })
            
        chapters.append({
            "chapter_id": chapter_id,
            "position": len(chapters) + 1,
            "title": title,
            "paragraphs": paragraphs,
        })

    return chapters


def main():
    html = INPUT_PATH.read_text(encoding="utf-8")
    soup = BeautifulSoup(html, "html.parser")

    chapters = extract_chapters(soup)

    book = {
        "book_id": "pg1232",
        "title": "The Prince",
        "author": "Niccolò Machiavelli",
        "translator": "W. K. Marriott",
        "source_url": "https://www.gutenberg.org/ebooks/1232",
        "chapters": chapters,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(book, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    for chapter in chapters:
        print(
            f"{chapter['chapter_id']} | "
            f"{len(chapter['paragraphs'])} paragraphs | "
            f"{chapter['title']}"
        )

    print(f"\nExtracted {len(chapters)} chapters")
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()