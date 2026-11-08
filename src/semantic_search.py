import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from dataclasses import dataclass

try:
    from .library import DEFAULT_BOOK_ID, SEARCHABLE_TYPES, book_paths, fingerprint, source_link
except ImportError:
    from library import DEFAULT_BOOK_ID, SEARCHABLE_TYPES, book_paths, fingerprint, source_link


@dataclass
class SearchData:
    passages: dict
    chapters: dict
    vectors: np.ndarray
    passage_ids: list
    model_name: str
    book: dict

    def __iter__(self):
        # Preserve the existing five-value unpacking contract.
        return iter((self.passages, self.chapters, self.vectors, self.passage_ids, self.model_name))

    def __getitem__(self, index):
        return tuple(self)[index]


def load_search_data(book_id=DEFAULT_BOOK_ID, data_dir=None):
    paths = book_paths(book_id) if data_dir is None else book_paths(book_id, data_dir)
    book_path, passage_path, vector_path = paths
    book = json.loads(book_path.read_text(encoding='utf-8'))
    passage_data = json.loads(passage_path.read_text(encoding='utf-8'))
    if book['book_id'] != book_id or passage_data['book_id'] != book_id:
        raise ValueError('Artifacts do not belong to the selected book')
    book_hash = fingerprint(book)
    if passage_data.get('book_sha256') != book_hash:
        raise ValueError('Book changed or legacy index detected; rebuild passages and embeddings')
    records = passage_data['passages']
    passages = {p['passage_id']: dict(p) for p in records}
    if not records or len(passages) != len(records):
        raise ValueError('Empty index or duplicate passage IDs')
    with np.load(vector_path, allow_pickle=False) as saved:
        for key, expected in [('book_id', book_id), ('book_sha256', book_hash),
                              ('passages_sha256', fingerprint(passage_data))]:
            if key not in saved or saved[key].item() != expected:
                raise ValueError('Stale or mismatched embeddings; rebuild embeddings')
        vectors = saved['vectors']
        passage_ids = saved['passage_ids'].tolist()
        model_name = saved['embedding_model'].item()
    if model_name != passage_data['embedding_model']:
        raise ValueError('Passages and vectors use different models')
    if vectors.ndim != 2 or vectors.shape[0] != len(passage_ids) or vectors.shape[1] == 0:
        raise ValueError('Invalid vector shape')
    if not np.isfinite(vectors).all():
        raise ValueError('Nonfinite vectors')
    if len(set(passage_ids)) != len(passage_ids) or set(passage_ids) != set(passages):
        raise ValueError('Passage ID mismatch; rebuild embeddings')
    chapters = {}
    paragraph_lookup = {}
    for chapter in book['chapters']:
        key = f"{book_id}:{chapter['chapter_id']}"
        if key in chapters:
            raise ValueError('Duplicate chapter IDs')
        chapters[key] = chapter
        for paragraph in chapter['paragraphs']:
            if paragraph['paragraph_id'] in paragraph_lookup:
                raise ValueError('Duplicate paragraph IDs')
            paragraph_lookup[paragraph['paragraph_id']] = (key, paragraph)
    for passage in passages.values():
        key = f"{book_id}:{passage['chapter_id']}"
        parent_key, paragraph = paragraph_lookup.get(passage['paragraph_id'], (None, None))
        if paragraph is None or parent_key != key or not passage['paragraph_id'].startswith(book_id + ':'):
            raise ValueError('Passage references a missing or foreign paragraph')
        if passage['content_type'] not in SEARCHABLE_TYPES or paragraph['content_type'] not in SEARCHABLE_TYPES:
            raise ValueError('Non-author content found in search index')
        if paragraph['text'][passage['char_start']:passage['char_end']] != passage['text']:
            raise ValueError('Passage text does not match its source offsets')
        passage['chapter_key'] = key
        passage['source_url'] = source_link(book, chapters[key], paragraph)
    return SearchData(passages, chapters, vectors, passage_ids, model_name, book)


def show_result(passage, score, chapters):
    chapter = chapters[passage["chapter_key"]]
    paragraphs = chapter["paragraphs"]

    index = next(
        index
        for index, paragraph in enumerate(paragraphs)
        if paragraph["paragraph_id"] == passage["paragraph_id"]
    )

    print(f"\n{'=' * 70}")
    print(f"Similarity: {score:.3f}")
    print(passage["chapter_title"])
    print(f"Passage: {passage['passage_id']}")
    print(f"Source: {passage.get('source_url')}")

    print("\n[RETRIEVED PASSAGE]")
    print(passage["text"])

    print("\n[SOURCE CONTEXT — includes the full parent paragraph]")

    for position in range(max(0, index - 1), min(len(paragraphs), index + 2)):
        paragraph = paragraphs[position]
        label = "PARENT" if position == index else "NEIGHBOR"

        print(
            f"\n[{label} | {paragraph['content_type']}] "
            f"{paragraph['paragraph_id']}"
        )
        print(paragraph["text"])


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--book', default=DEFAULT_BOOK_ID)
    args = parser.parse_args()
    passages, chapters, vectors, passage_ids, model_name = (
        load_search_data(args.book)
    )

    model = SentenceTransformer(model_name, device="cpu")

    while True:
        query = input("\nSemantic search (or 'quit'): ").strip()

        if query.casefold() == "quit":
            break

        if not query:
            continue

        query_length = len(model.tokenizer.encode(
            query,
            truncation=False,
        ))

        if query_length > model.max_seq_length:
            print("Query is too long. Please shorten it.")
            continue

        query_vector = model.encode(
            query,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        scores = vectors @ query_vector
        ranked_indices = np.argsort(-scores)

        # Show at most one passage per parent paragraph.
        shown_paragraphs = set()

        for index in ranked_indices:
            passage = passages[passage_ids[index]]
            paragraph_id = passage["paragraph_id"]

            if paragraph_id in shown_paragraphs:
                continue

            show_result(passage, float(scores[index]), chapters)
            shown_paragraphs.add(paragraph_id)

            if len(shown_paragraphs) == 3:
                break


if __name__ == "__main__":
    main()