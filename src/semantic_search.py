import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"


def load_search_data():
    passage_data = json.loads(
        (DATA_DIR / "the_prince_passages.json").read_text(
            encoding="utf-8"
        )
    )
    book = json.loads(
        (DATA_DIR / "the_prince.json").read_text(encoding="utf-8")
    )

    passages = {
        passage["passage_id"]: passage
        for passage in passage_data["passages"]
    }

    with np.load(
        DATA_DIR / "the_prince_vectors.npz",
        allow_pickle=False,
    ) as saved:
        vectors = saved["vectors"]
        passage_ids = saved["passage_ids"].tolist()
        model_name = saved["embedding_model"].item()

    if model_name != passage_data["embedding_model"]:
        raise ValueError("Passages and vectors use different models")

    if vectors.shape[0] != len(passage_ids):
        raise ValueError("Vector count does not match passage ID count")

    if set(passage_ids) != set(passages):
        raise ValueError("Passages have changed; rebuild embeddings")

    chapters = {
        chapter["chapter_id"]: chapter
        for chapter in book["chapters"]
    }

    return passages, chapters, vectors, passage_ids, model_name


def show_result(passage, score, chapters):
    chapter = chapters[passage["chapter_id"]]
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
    print(
        "Source: https://www.gutenberg.org/cache/epub/1232/"
        f"pg1232-images.html#{passage['chapter_id']}"
    )

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
    passages, chapters, vectors, passage_ids, model_name = (
        load_search_data()
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