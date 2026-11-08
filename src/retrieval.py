"""Hybrid semantic/BM25 retrieval and bounded, source-located evidence."""
import math
import re
from collections import Counter, defaultdict
from functools import lru_cache
from dataclasses import dataclass

import numpy as np

try:
    from .library import source_link
except ImportError:
    from library import source_link

STOPWORDS = set('a an the and or of to in on at for from with by is are was were be been it its this that these those i you he she they we me my your his her their our what which who how why when where does do did can could would should please explain describe tell about according book text author say says said'.split())


MULTILINGUAL_RERANKER_MODEL = 'cross-encoder/mmarco-mMiniLMv2-L12-H384-v1'
ENGLISH_RERANKER_MODEL = 'cross-encoder/ms-marco-MiniLM-L6-v2'


@dataclass
class RetrievalModel:
    embedder: object
    reranker: object = None

    @property
    def tokenizer(self):
        return self.embedder.tokenizer

    @property
    def max_seq_length(self):
        return self.embedder.max_seq_length

    def encode(self, *args, **kwargs):
        return self.embedder.encode(*args, **kwargs)


def load_reranker(language=None):
    model_name = ENGLISH_RERANKER_MODEL if str(language).casefold() in {'en', 'eng', 'english'} else MULTILINGUAL_RERANKER_MODEL
    return _load_reranker_model(model_name)


@lru_cache(maxsize=2)
def _load_reranker_model(model_name):
    from sentence_transformers import CrossEncoder
    try:
        return CrossEncoder(model_name, device='cpu', max_length=512, local_files_only=True)
    except OSError:
        return CrossEncoder(model_name, device='cpu', max_length=512)


def terms(text):
    words = re.findall(r"[^\W_]+(?:['’][^\W_]+)?", text.casefold(), re.UNICODE)
    tokens = [word for word in words if word not in STOPWORDS]
    # Shingles preserve relationships such as names and 'without fighting'.
    tokens.extend('phrase:'+left+'/'+right for left, right in zip(tokens[:], tokens[1:]))
    # Wordless CJK text still gets lexical matches without extra dependencies.
    for word in words:
        if re.search(r'[\u3400-\u9fff\u3040-\u30ff]', word):
            tokens.extend(word[i:i+2] for i in range(len(word)-1))
    return tokens


class BM25:
    def __init__(self, documents):
        self.lengths = np.array([len(doc) for doc in documents], dtype=np.float32)
        self.average = max(float(self.lengths.mean()), 1.0)
        self.postings = defaultdict(list)
        for index, doc in enumerate(documents):
            for term, count in Counter(doc).items():
                self.postings[term].append((index, count))

    def scores(self, query):
        result = np.zeros(len(self.lengths), dtype=np.float32)
        for term in set(query):
            postings = self.postings.get(term, [])
            if not postings:
                continue
            indexes, counts = np.array(postings).T
            indexes = indexes.astype(int)
            idf = math.log(1 + (len(self.lengths)-len(postings)+.5)/(len(postings)+.5))
            result[indexes] += idf * counts * 2.5 / (counts + 1.5*(.25+.75*self.lengths[indexes]/self.average))
        return result


def retrieve(search_data, query, embedder, limit=6):
    query = query.strip()
    if not query:
        raise ValueError('Enter a question or search phrase.')
    if len(query) > 8000:
        raise ValueError('Please shorten the search question.')
    # The full query is used lexically; only the embedding input is bounded.
    tokens = embedder.tokenizer.encode(query, truncation=False)
    embedding_query = query
    if len(tokens) > embedder.max_seq_length:
        if not hasattr(embedder.tokenizer, 'decode'):
            raise ValueError('Please shorten the research query.')
        embedding_query = embedder.tokenizer.decode(tokens[:embedder.max_seq_length-2], skip_special_tokens=True)
    vector = np.asarray(embedder.encode(embedding_query, normalize_embeddings=True, convert_to_numpy=True))
    if vector.ndim != 1 or vector.shape[0] != search_data.vectors.shape[1] or not np.isfinite(vector).all():
        raise ValueError('Query model does not match the selected book index.')
    semantic = search_data.vectors @ vector
    if not hasattr(search_data, '_bm25'):
        documents = [terms(search_data.passages[identity]['chapter_title']+' '+search_data.passages[identity]['text'])
                     for identity in search_data.passage_ids]
        search_data._bm25 = BM25(documents)
    lexical = search_data._bm25.scores(terms(query))
    # Reciprocal-rank fusion combines incomparable cosine and BM25 score scales.
    fused = np.zeros(len(semantic), dtype=np.float32)
    for values, weight in ((semantic, 1.0), (lexical, 1.0)):
        ranked = np.argsort(-values, kind='stable')[:80]
        for rank, index in enumerate(ranked, 1):
            if values is lexical and values[index] <= 0:
                continue
            fused[index] += weight / (60 + rank)
    reranker = getattr(embedder, 'reranker', None)
    candidate_pool = None
    if reranker is not None:
        pool_size = max(limit, 40)
        candidate_pool = set(np.argsort(-semantic)[:pool_size])
        candidate_pool.update(i for i in np.argsort(-lexical)[:pool_size] if lexical[i] > 0)
    results, seen = [], set()
    for index in np.argsort(-fused, kind='stable'):
        if candidate_pool is not None and index not in candidate_pool:
            continue
        passage = search_data.passages[search_data.passage_ids[int(index)]]
        if passage['paragraph_id'] in seen:
            continue
        seen.add(passage['paragraph_id'])
        results.append({'passage_id': passage['passage_id'], 'score': float(fused[index]),
                        'semantic_score': float(semantic[index]), 'lexical_score': float(lexical[index])})
        if candidate_pool is None and len(results) >= limit:
            break
    lexical_best = None
    best_index = int(np.argmax(lexical))
    if lexical[best_index] > 0:
        best_passage = search_data.passages[search_data.passage_ids[best_index]]
        lexical_best = next((r for r in results if search_data.passages[r['passage_id']]['paragraph_id'] == best_passage['paragraph_id']), None)
        if lexical_best is None:
            lexical_best = {'passage_id': best_passage['passage_id'], 'score': float(fused[best_index]),
                            'semantic_score': float(semantic[best_index]), 'lexical_score': float(lexical[best_index])}
            results.append(lexical_best)
    if reranker is not None and results:
        sources = evidence_for_results(search_data, results, max_source_chars=2400)
        positions = {p['paragraph_id']: (c, index) for c in search_data.chapters.values()
                     for index, p in enumerate(c['paragraphs'])}
        pairs = []
        for source in sources:
            chapter, index = positions[source['source_id']]
            paragraphs = chapter['paragraphs']
            def neighbor(position):
                if 0 <= position < len(paragraphs) and paragraphs[position]['content_type'] in {'body', 'verse'}:
                    return paragraphs[position]['text']
                return ''
            context = (source['chapter'] + '\nPrevious paragraph: ' + neighbor(index-1)[-400:]
                       + '\nMatched paragraph: ' + source['text'][:1600]
                       + '\nNext paragraph: ' + neighbor(index+1)[:400])
            pairs.append((query[:1600], context))
        scores = np.asarray(reranker.predict(pairs, batch_size=8, show_progress_bar=False)).reshape(-1)
        if len(scores) != len(results) or not np.isfinite(scores).all():
            raise ValueError('The relevance reranker returned invalid scores.')
        for result, score in zip(results, scores):
            result['rerank_score'] = float(score)
        results = sorted(results, key=lambda item: item['rerank_score'], reverse=True)
    selected = results[:limit]
    # Keep one strong word/phrase match alongside neural choices. Rerankers can
    # lose negation or rare names; retaining that source protects evidence recall.
    if limit >= 3 and lexical_best is not None and lexical_best not in selected:
        selected[-1] = lexical_best
    return selected


def evidence_for_results(search_data, results, max_source_chars=3000):
    book = search_data.book
    lookup = {paragraph['paragraph_id']: (chapter, paragraph)
              for chapter in search_data.chapters.values() for paragraph in chapter['paragraphs']}
    evidence = []
    for result in results:
        passage = search_data.passages[result['passage_id']]
        chapter, paragraph = lookup[passage['paragraph_id']]
        full_text = paragraph['text']
        start, end = 0, len(full_text)
        if len(full_text) > max_source_chars:
            start = max(0, passage['char_start'] - 600)
            end = min(len(full_text), max(start+max_source_chars, passage['char_end']))
            start = max(0, end-max_source_chars)
        evidence.append({'source_id': paragraph['paragraph_id'], 'book_id': book['book_id'],
            'book_title': book['title'], 'author': book['author'], 'translator': book.get('translator'),
            'chapter': chapter['title'], 'content_type': paragraph['content_type'],
            'source_url': source_link(book, chapter, paragraph),
            'source_locator': chapter.get('source_locator'),
            'text': full_text[start:end], 'char_start': start, 'char_end': end,
            'excerpted': start != 0 or end != len(full_text)})
    return evidence


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Prepare the local question/passage relevance reranker.')
    parser.add_argument('--prepare-model', action='store_true', required=True)
    parser.add_argument('--language', default='en')
    args = parser.parse_args()
    load_reranker(args.language)
    print('Reranker ready for language:', args.language)
