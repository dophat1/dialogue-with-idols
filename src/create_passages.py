import argparse
import json
from pathlib import Path

from sentence_transformers import SentenceTransformer

try:
    from .library import DEFAULT_EMBEDDING_MODEL, SEARCHABLE_TYPES, fingerprint
except ImportError:
    from library import DEFAULT_EMBEDDING_MODEL, SEARCHABLE_TYPES, fingerprint

WINDOW = 220
OVERLAP = 40


def create_passages(book, tokenizer, max_seq_length, model_name=DEFAULT_EMBEDDING_MODEL):
    window = min(WINDOW, max_seq_length - 2)
    if window < 8:
        raise ValueError('Embedding model input limit is too small')
    overlap = min(OVERLAP, window // 3)
    passages = []

    for chapter in book["chapters"]:
        for paragraph in chapter["paragraphs"]:
            if paragraph["content_type"] not in SEARCHABLE_TYPES:
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
                end = min(start + window, len(offsets))

                char_start = offsets[start][0]
                while True:
                    char_end = offsets[end - 1][1]
                    passage_text = text[char_start:char_end]
                    token_count = len(tokenizer.encode(passage_text, add_special_tokens=True, truncation=False))
                    if token_count <= max_seq_length:
                        break
                    end -= 1
                    if end <= start:
                        raise ValueError('A token cannot fit in the embedding model input limit')

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

                start = max(start + 1, end - overlap)
                part += 1

    if not passages:
        raise ValueError("No body or verse passages to index")

    return {
        "book_id": book["book_id"],
        "book_sha256": fingerprint(book),
        "source_sha256": book.get("source_sha256"),
        "indexed_content_types": sorted(SEARCHABLE_TYPES),
        "embedding_model": model_name,
        "chunking_version": "token-window-v1",
        "window_tokens": window,
        "overlap_tokens": overlap,
        "passages": passages,
    }



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('book_path', type=Path)
    parser.add_argument('--model', default=DEFAULT_EMBEDDING_MODEL)
    args = parser.parse_args()
    book_path = args.book_path.resolve()
    book = json.loads(book_path.read_text(encoding='utf-8'))
    model = SentenceTransformer(args.model, device='cpu')
    output = create_passages(book, model.tokenizer, model.max_seq_length, args.model)
    output_path = book_path.with_name(f'{book_path.stem}_passages.json')
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"Created {len(output['passages'])} passages")
    print(f"Largest passage: {max(p['token_count'] for p in output['passages'])} tokens")
    print(f'Saved to: {output_path}')


if __name__ == '__main__':
    main()
