"""Validate any extracted book without book-specific chapter assumptions."""
import argparse
import json
from pathlib import Path

try:
    from .extract_book import validate_book
except ImportError:
    from extract_book import validate_book


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('book_path', type=Path)
    parser.add_argument('--expected-chapters', type=int)
    args = parser.parse_args()
    book = json.loads(args.book_path.read_text(encoding='utf-8'))
    sections, paragraphs = validate_book(book, args.expected_chapters)
    print(f'Passed: {sections} ordered sections, {paragraphs} nonempty paragraphs with unique IDs')


if __name__ == '__main__':
    main()
