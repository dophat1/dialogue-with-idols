import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

import argparse

try:
    from .library import fingerprint
except ImportError:
    from library import fingerprint

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "passage_path",
        type=Path,
        help="Path to a passages JSON file",
    )
    args = parser.parse_args()

    passage_path = args.passage_path.resolve()

    if not passage_path.stem.endswith("_passages"):
        raise ValueError("Expected a file named <book>_passages.json")

    source_bytes = passage_path.read_bytes()
    data = json.loads(source_bytes)

    model = SentenceTransformer(
        data["embedding_model"],
        device="cpu",
        local_files_only=True,
    )

    passages = data["passages"]
    if not passages:
        raise ValueError("No passages to embed")
    texts = [passage["text"] for passage in passages]

    vectors = model.encode(
        texts,
        batch_size=32,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )

    if not np.isfinite(vectors).all():
        raise ValueError("Embeddings contain invalid numbers")

    book_name = passage_path.stem.removesuffix("_passages")
    output_path = passage_path.with_name(f"{book_name}_vectors.npz")

    np.savez_compressed(
        output_path,
        vectors=vectors,
        passages_sha256=np.array(fingerprint(data)),
        book_sha256=np.array(data["book_sha256"]),
        book_id=np.array(data["book_id"]),
        passage_ids=np.array([
            passage["passage_id"] for passage in passages
        ]),
        embedding_model=np.array(data["embedding_model"]),
    )

    print(f"\nVector shape: {vectors.shape}")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
