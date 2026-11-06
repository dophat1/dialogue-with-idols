from pathlib import Path
from bs4 import BeautifulSoup

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BOOK_PATH = PROJECT_ROOT / "data" / "raw" / "the_prince.html"

html = BOOK_PATH.read_text(encoding="utf-8")
soup = BeautifulSoup(html, "html.parser")

def clean_text(element):
    return " ".join(element.get_text().split())


headings = soup.find_all(["h1", "h2", "h3"])

for heading in headings:
    title = heading.get_text(" ", strip=True)
    anchor = heading.get("id")

    print(f"{heading.name} | id={anchor} | {title}")


first_chapter = next(
    heading
    for heading in soup.find_all("h2")
    if heading.get_text(" ", strip=True).startswith("CHAPTER I.")
)

print("\n--- FIRST CHAPTER HTML ---")
print(first_chapter.prettify())

print("\n--- PARENT ELEMENT ---")
print(first_chapter.parent.name, first_chapter.parent.attrs)

print("\n--- FIRST THREE FOLLOWING ELEMENTS ---")
for element in list(first_chapter.next_siblings):
    # HTML whitespace also appears as a sibling; skip it.
    if not getattr(element, "name", None):
        continue

    print(element.prettify()[:1200])

    # Count only actual HTML elements.
    # We only need a small sample for inspection.
    if element == first_chapter.find_next_siblings(limit=3)[-1]:
        break

chapter_anchor = soup.find(id="chap17")
chapter = chapter_anchor.find_parent("div", class_="chapter")

for paragraph in chapter.find_all("p", recursive=False)[:10]:
    text = " ".join(paragraph.get_text().split())

    print("\nAttributes:", paragraph.attrs)
    print("Text:", text[:250])

print("\n--- ORIGINAL FIRST PARAGRAPH HTML ---")
print(chapter.find("p", recursive=False).prettify())

first_paragraph = chapter.find("p", recursive=False)

print("\n--- RAW PARAGRAPH HTML ---")
print(str(first_paragraph))

print("\n--- TEXT FRAGMENTS ---")
for fragment in first_paragraph.strings:
    print(repr(fragment))