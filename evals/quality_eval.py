"""Evaluate a local, reading-labeled question set with real retrieval and optional answers."""
import argparse
import contextlib
import io
import json
from pathlib import Path

from openai import OpenAIError

from src.answer_question import generate_answer
from src.ingest import load_embedder
from src.retrieval import RetrievalModel, evidence_for_results, load_reranker, retrieve
from src.semantic_search import load_search_data


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GOLD = ROOT / 'evals' / 'quality_gold.json'
DEFAULT_OUTPUT = ROOT / 'docs' / 'quality_eval_results.json'


def summarize(cases):
    relevant = [case for case in cases if case['answerable'] and 'retrieved_source_ids' in case]
    answered = [case for case in cases if 'answer' in case]
    abstention = [case for case in answered if not case['answerable']]
    return {
        'cases': len(cases),
        'answerable_retrieval_cases': len(relevant),
        'hit_at_6': sum(case['any_gold_at_6'] for case in relevant),
        'all_gold_at_6': sum(case['all_gold_at_6'] for case in relevant),
        'mrr': round(sum(1 / case['first_gold_rank'] if case['first_gold_rank'] else 0
                         for case in relevant) / len(relevant), 3) if relevant else None,
        'generated_answers': len(answered),
        'unanswerable_generated': len(abstention),
        'insufficient_on_unanswerable': sum(case['answer']['evidence_status'] == 'insufficient'
                                           and not case['answer']['claims'] for case in abstention),
        'errors': sum('error' in case for case in cases),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gold', type=Path, default=DEFAULT_GOLD)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--split', choices=['development', 'holdout', 'all'], default='all')
    parser.add_argument('--answers', action='store_true', help='Make real OpenAI answer requests; incurs API cost')
    args = parser.parse_args()
    cases = json.loads(args.gold.read_text(encoding='utf-8'))['cases']
    cases = [case for case in cases if args.split == 'all' or case['split'] == args.split]
    books, models, report = {}, {}, []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for case in cases:
        result = {key: case[key] for key in ('id', 'split', 'book_id', 'question', 'answerable',
                                             'expected_answer', 'gold_source_ids')}
        try:
            book_id = case['book_id']
            if book_id not in books:
                books[book_id] = load_search_data(book_id)
            data = books[book_id]
            language = data.book.get('language') or ('en' if data.model_name.endswith('/all-MiniLM-L6-v2') else None)
            model_key = (data.model_name, language)
            if model_key not in models:
                models[model_key] = RetrievalModel(load_embedder(data.model_name), load_reranker(language))
            model = models[model_key]
            ranked = retrieve(data, case['question'], model, limit=6)
            evidence = evidence_for_results(data, ranked, max_source_chars=2500)
            source_ids = [source['source_id'] for source in evidence]
            gold = set(case['gold_source_ids'])
            first = next((i for i, source_id in enumerate(source_ids, 1) if source_id in gold), None)
            result.update(retrieved_source_ids=source_ids, any_gold_at_6=first is not None,
                          all_gold_at_6=bool(gold) and gold.issubset(source_ids), first_gold_rank=first,
                          evidence=evidence)
            if args.answers:
                with contextlib.redirect_stdout(io.StringIO()):
                    run = generate_answer(case['question'], '', model, data, save_run=False)
                result['answer'] = run['answer']
                result['generation_evidence'] = run['evidence']
                result['generation_model'] = run['generation_model_returned']
                result['usage'] = run['usage']
            print(f"{case['id']}: {'hit' if first else 'miss' if gold else 'unanswerable'}"
                  + (' + answer' if 'answer' in result else ''), flush=True)
        except (OSError, ValueError, RuntimeError, OpenAIError) as error:
            # API exceptions may contain private request details; record only their type.
            result['error'] = type(error).__name__
            if isinstance(error, ValueError):
                result['validation_message'] = str(error)
            print(f"{case['id']}: ERROR {type(error).__name__}", flush=True)
        report.append(result)
        args.output.write_text(json.dumps({'summary': summarize(report), 'cases': report},
                                          ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summarize(report), indent=2))
    print(f'Results: {args.output}')


if __name__ == '__main__':
    main()
