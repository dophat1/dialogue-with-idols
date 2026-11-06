from pathlib import Path
from urllib.request import urlopen

SOURCE_URL = (
    "https://www.gutenberg.org/cache/epub/1232/pg1232-images.html"
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "data" / "raw" / "the_prince.html"


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with urlopen(SOURCE_URL, timeout=30) as response:
        html_bytes = response.read()

    OUTPUT_PATH.write_bytes(html_bytes)

    print(f"Saved {len(html_bytes):,} bytes")
    print(f"Location: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()