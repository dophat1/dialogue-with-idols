import argparse
from pathlib import Path

from bs4 import BeautifulSoup


def clean_text(element):
    return " ".join(element.get_text().split())


def inspect_heading(heading, sample_size):
    print("\n--- SELECTED HEADING HTML ---")
    print(heading.prettify())

    print("\n--- PARENT ELEMENT ---")
    print(heading.parent.name, heading.parent.attrs)

    print("\n--- FOLLOWING SIBLING ELEMENTS ---")
    for element in heading.find_next_siblings(limit=sample_size):
        print(element.prettify()[:1800])


def inspect_paragraph(paragraph):
    print("\n--- FIRST FOLLOWING PARAGRAPH: RAW HTML ---")
    print(str(paragraph))

    print("\n--- NORMALIZED TEXT ---")
    print(clean_text(paragraph))

    print("\n--- TEXT FRAGMENTS ---")
    for fragment in paragraph.strings:
        print(repr(fragment))


def main():
    parser = argparse.ArgumentParser(
        description="Inspect an HTML document's headings and structure."
    )
    parser.add_argument(
        "html_path",
        type=Path,
        help="Path to a downloaded HTML file",
    )
    parser.add_argument(
        "--heading",
        help="Case-insensitive text to find within a heading",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=5,
        help="Number of following sibling elements to display",
    )
    args = parser.parse_args()

    if not args.html_path.is_file():
        parser.error(f"File not found: {args.html_path}")

    if args.sample < 1:
        parser.error("--sample must be at least 1")

    html = args.html_path.read_text(encoding="utf-8")
    soup = BeautifulSoup(html, "html.parser")
    headings = soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"])

    print(f"File: {args.html_path.resolve()}")
    print(f"Headings found: {len(headings)}")
    print("\n--- HEADINGS ---")

    for number, heading in enumerate(headings, start=1):
        nested_ids = [
            element["id"]
            for element in heading.find_all(id=True)
        ]

        print(
            f"{number}. {heading.name} | "
            f"id={heading.get('id')} | "
            f"nested_ids={nested_ids} | "
            f"{clean_text(heading)}"
        )

    if not args.heading:
        print("\nUse --heading to inspect a heading in more detail.")
        return

    matches = [
        heading
        for heading in headings
        if args.heading.casefold() in clean_text(heading).casefold()
    ]

    if not matches:
        print(f"\nNo heading matched: {args.heading!r}")
        return

    if len(matches) > 1:
        print(
            f"\nFound {len(matches)} matching headings. "
            "Use a more specific --heading value."
        )
        return

    heading = matches[0]
    inspect_heading(heading, args.sample)

    # This is a diagnostic sample, not a chapter-boundary rule.
    paragraph = heading.find_next("p")

    if paragraph is not None:
        inspect_paragraph(paragraph)
    else:
        print("\nNo paragraph follows this heading.")


if __name__ == "__main__":
    main()