import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"


def main():
    passage_path = DATA_DIR / "the_prince_passages.json"
    source_bytes = passage_path.read_bytes()
    data = json.loads(source_bytes)

    model = SentenceTransformer(
        data["embedding_model"],
        device="cpu",
    )

    passages = data["passages"]
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

    output_path = DATA_DIR / "the_prince_vectors.npz"

    np.savez_compressed(
        output_path,
        vectors=vectors,
        passage_ids=np.array([
            passage["passage_id"] for passage in passages
        ]),
        embedding_model=np.array(data["embedding_model"]),
    )

    print(f"\nVector shape: {vectors.shape}")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()