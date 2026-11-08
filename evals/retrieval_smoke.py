"""Small source-retrieval regression check; uses existing local indexes, no API."""
import json
from pathlib import Path

from src.ingest import load_embedder
from src.library import catalog
from src.retrieval import RetrievalModel, load_reranker, retrieve, evidence_for_results
from src.semantic_search import load_search_data


def main():
    books = catalog()
    alice = next((key for key, book in books.items() if key.startswith('pg11_') and book.get('format') == 'html'), None)
    raven = next((key for key, book in books.items() if key.startswith('pg1065_')), None)
    cases = [
        ('pg1232', 'Is it better for a ruler to be loved or feared?', ['feared']),
        ('pg1232', 'Why should a prince imitate both the fox and the lion?', ['fox', 'lion']),
        ('pg132', 'How can a general win without fighting a battle?', ['without fighting']),
        (alice, 'Why does Alice follow the White Rabbit?', ['waistcoat']),
        (alice, 'How does eating the mushroom change Alice’s height?', ['mushroom']),
        (raven, 'What word does the raven keep repeating?', ['nevermore']),
        (raven, 'Where does the raven perch inside the room?', ['pallas']),
    ]
    models, report = {}, []
    for book_id, question, needles in cases:
        if book_id is None or book_id not in books:
            continue
        data = load_search_data(book_id)
        language = data.book.get('language') or ('en' if data.model_name.endswith('/all-MiniLM-L6-v2') else None)
        key = (data.model_name, language)
        if key not in models:
            models[key] = RetrievalModel(load_embedder(data.model_name), load_reranker(language))
        results = retrieve(data, question, models[key])
        evidence = evidence_for_results(data, results)
        text = ' '.join(source['text'] for source in evidence).casefold()
        hit = all(needle in text for needle in needles)
        report.append({'book_id': book_id, 'question': question, 'expected_terms': needles,
                       'hit_in_top_6': hit, 'results': results,
                       'evidence': [{'source_id': e['source_id'], 'chapter': e['chapter'], 'text': e['text']} for e in evidence]})
        print(('PASS' if hit else 'MISS') + ' ' + question, flush=True)
    output = Path('docs/retrieval_smoke.json')
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    passed = sum(item['hit_in_top_6'] for item in report)
    print(f'{passed}/{len(report)} checks found expected source terms in the top six. This is not an answer-accuracy benchmark.')
    if passed != len(report):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
