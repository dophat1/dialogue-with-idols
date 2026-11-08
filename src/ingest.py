"""One import command: download/read, extract, chunk, embed, publish a library entry."""
import argparse
import json
import os
import shutil
from uuid import uuid4
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

try:
    from .gutenberg_import import extract_document, fetch_source
    from .create_passages import create_passages
    from .library import DATA_DIR, DEFAULT_EMBEDDING_MODEL, fingerprint
except ImportError:
    from gutenberg_import import extract_document, fetch_source
    from create_passages import create_passages
    from library import DATA_DIR, DEFAULT_EMBEDDING_MODEL, fingerprint


def load_embedder(model_name=DEFAULT_EMBEDDING_MODEL):
    try:
        return SentenceTransformer(model_name, device='cpu', local_files_only=True)
    except OSError:
        return SentenceTransformer(model_name, device='cpu')


def import_bytes(content, filename, *, source_url=None, data_dir=DATA_DIR,
                 embedder=None, model_name=DEFAULT_EMBEDDING_MODEL, progress=None):
    progress = progress or (lambda message: None)
    progress('Extracting metadata and readable text...')
    book = extract_document(content, filename, source_url)
    base_id = book['book_id']
    # Content/metadata-version identities keep editions and old saved citations separate.
    identity = fingerprint({'book': book, 'embedding_model': model_name})[:16]
    book['gutenberg_id'] = base_id if base_id.startswith('pg') and base_id[2:].isdigit() else None
    book['book_id'] = f'{base_id}_{identity}'
    for chapter in book['chapters']:
        for paragraph in chapter['paragraphs']:
            paragraph['paragraph_id'] = book['book_id'] + ':' + paragraph['paragraph_id'].split(':', 1)[1]
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    destination = data_dir / book['book_id']
    if destination.is_dir():
        try:
            from .semantic_search import load_search_data
        except ImportError:
            from semantic_search import load_search_data
        cached = load_search_data(book['book_id'], data_dir)
        progress('This edition is already indexed.')
        return cached.book
    progress('Loading the multilingual search model...')
    embedder = embedder or load_embedder(model_name)
    passages = create_passages(book, embedder.tokenizer, embedder.max_seq_length, model_name)
    progress(f"Indexing {len(passages['passages']):,} passages on CPU...")
    texts = [p['text'] for p in passages['passages']]
    vectors = embedder.encode(texts, batch_size=32, normalize_embeddings=True,
                              convert_to_numpy=True, show_progress_bar=False)
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape[0] != len(texts) or vectors.shape[1] == 0 or not np.isfinite(vectors).all():
        raise ValueError('The embedding model returned invalid vectors; nothing was published.')
    norms = np.linalg.norm(vectors, axis=1)
    if np.any(norms < 1e-8):
        raise ValueError('The embedding model returned empty vectors; nothing was published.')
    vectors = vectors / norms[:, None]
    # Publish only a complete bundle. A failed import cannot replace a working index.
    temporary = data_dir / ('.import-' + uuid4().hex)
    temporary.mkdir()
    try:
        stage = temporary / 'bundle'
        stage.mkdir()
        (stage / 'book.json').write_text(json.dumps(book, ensure_ascii=False, indent=2), encoding='utf-8')
        (stage / 'passages.json').write_text(json.dumps(passages, ensure_ascii=False), encoding='utf-8')
        np.savez_compressed(stage / 'vectors.npz', vectors=vectors,
            passages_sha256=np.array(fingerprint(passages)), book_sha256=np.array(fingerprint(book)),
            book_id=np.array(book['book_id']), passage_ids=np.array([p['passage_id'] for p in passages['passages']]),
            embedding_model=np.array(model_name))
        (stage / 'source.bin').write_bytes(content)
        if not destination.exists():
            try:
                os.rename(stage, destination)
            except OSError:
                if not destination.is_dir():
                    raise
        # A concurrent session may have published the identical edition first.
    finally:
        # Only remove the staging directory created by this call.
        if temporary.resolve().parent != data_dir.resolve() or not temporary.name.startswith('.import-'):
            raise RuntimeError('Invalid staging cleanup path')
        shutil.rmtree(temporary)
    progress('Book is ready to search and discuss.')
    return book


def import_source(value, **kwargs):
    value = value.strip()
    if '://' in value or value.isdigit() or (value.lower().startswith('pg') and value[2:].isdigit()):
        content, filename, source_url = fetch_source(value)
    else:
        path = Path(value)
        if path.stat().st_size > 50 * 1024 * 1024:
            raise ValueError('Book exceeds the 50 MB import limit.')
        content, filename, source_url = path.read_bytes(), path.name, None
    return import_bytes(content, filename, source_url=source_url, **kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', help='Gutenberg book number/URL or local HTML, TXT, EPUB, ZIP')
    parser.add_argument('--model', default=DEFAULT_EMBEDDING_MODEL)
    args = parser.parse_args()
    try:
        book = import_source(args.source, model_name=args.model, progress=print)
    except (OSError, ValueError) as error:
        parser.exit(1, f'Import failed: {error}\n')
    print(f"Ready: {book['title']} ({book['book_id']})")
    for warning in book['extraction_diagnostics']['warnings']:
        print('Note:', warning)


if __name__ == '__main__':
    main()
