# Gutenberg RAG app checkpoint — 2026-11-09

The requested local workflow is implemented: import a Gutenberg URL/number or
book file, extract metadata and readable text, build a persistent index, select
the book in Streamlit, retrieve passages, and generate cited answers. This
supersedes the earlier two-inspected-book-only checkpoint. The full production
roadmap and universal answer accuracy are not claimed complete.

## Implemented changes and why

- `src/gutenberg_import.py` handles Gutenberg downloads plus HTML, TXT, EPUB,
  and text/HTML ZIP adapters. HTML tags, legacy text headers, and EPUB package
  metadata supply bibliographic fields. Unknown books no longer need registry
  entries. Missing metadata and fallback extraction are reported explicitly.
- Download discovery is limited to the requested ebook identity. It verifies
  readable book content and rejects site/help pages and foreign-book redirects.
  A first live test exposed an accidental navigation-page selection; that bug
  was corrected and has a regression test. The failed test bundle was removed
  during repository cleanup.
- EPUB follows its package spine rather than ZIP filename order. Archive size
  and path checks bound processing; archive contents are never extracted onto
  arbitrary filesystem paths. Plain-text and HTML boilerplate are stripped.
- `src/ingest.py` composes extraction, token windows, embedding, and publication.
  A complete bundle is staged before becoming visible. Content/version IDs
  keep changed sources separate; failed imports cannot replace a working book.
- `src/library.py` discovers complete local indexes dynamically. The original
  Prince/Art of War entries remain optional layout/attribution hints, not an
  import allowlist. Existing data and saved Prince history remain supported.
- Token windows respect the actual embedding input limit, including special
  tokens and substring retokenization. New imports use a multilingual model;
  old indexes retain their recorded model. Source offsets and fingerprints
  remain validated throughout the pipeline.
- `src/retrieval.py` combines semantic matching and phrase-aware BM25. Candidate
  generation takes the union of the two rankings, then locally reranks them
  against the question with adjacent paragraph context. Up to six results
  retain the strongest lexical candidate alongside neural choices. This
  protects exact names and relations when a reranker misses them.
- English sources use the English MS MARCO MiniLM reranker; other/unknown
  languages use multilingual mMARCO. Embedding and reranking models are kept
  separate and cached by actual model identity. This avoids accidentally
  inserting a reranker into the embedding model's neural-module sequence.
- Long source paragraphs contribute bounded, character-located excerpts instead
  of being silently skipped. Cited evidence includes document metadata, source
  URLs, and EPUB file locators where applicable.
- `src/library_ui.py` supplies import controls and evidence/context displays for
  the existing Streamlit apps. The discussion page includes both Find passages
  and Generate cited answer. Follow-ups use the prior topic unless a research
  query is supplied. Source switches clear active context/results.
- `requirements.txt`, local-only Streamlit binding, and a 50 MB upload limit
  make setup and usage explicit. Private credentials were not inspected or
  printed, and the configured OpenAI model was not overwritten.

## Verification

The automated suite passed **32 tests** covering parsing, metadata, EPUB order,
archive/file failures, indexing policy, token offsets, stale artifacts, source
boundaries, reranking, lexical evidence retention, and Streamlit import/search/
question flows. Unit/UI tests stub network/model/API dependencies.

Separate real tests downloaded and indexed:

| Source | Format checked | Outcome |
| --- | --- | --- |
| Alice's Adventures in Wonderland, Gutenberg 11 | HTML via ebook page | Imported, indexed, searchable |
| The Raven, Gutenberg 1065 | Plain text | Imported, indexed, searchable |
| Alice's Adventures in Wonderland, Gutenberg 11 | EPUB | Imported in spine order; fallback warning retained |
| Local synthetic and regression fixtures | HTML/text ZIP | Extraction and complete-index tests passed |

Original Prince and Art of War indexes still load with their existing metadata
and source isolation. Art of War's known commentary remains excluded from the
search index. A real-model Streamlit test exercised Alice's passage preview
without a generation request. The local HTTP health endpoint returned `ok`.

`python -m evals.retrieval_smoke` passed **7/7 source-term checks in the top six**
across The Prince, The Art of War, Alice, and The Raven. Development checks cover
love/fear, fox/lion, winning without fighting, following the Rabbit, the mushroom,
Nevermore, and the bird's perch. Early misses informed the contextual reranking
and lexical retention changes. These are development regressions, not held-out
accuracy estimates, and term presence alone does not prove complete relevance.
Detailed results are saved privately in `docs/retrieval_smoke.json`.

Live OpenAI tests generated and saved structurally validated cited answers from
Alice, including the paragraph about the watch, waistcoat, and curiosity. A
separate unrelated PostgreSQL question returned insufficient evidence with no
claims. These live checks verify API operation and selected examples; they do
not establish general semantic correctness. No raw credentials were read by
agent tools; the application loaded its private configuration for those requests.

## Practical limits

- Supported local formats are HTML/HTM/XHTML, text, EPUB, and text/HTML ZIP.
  Image-only scans, audio/video, arbitrary PDFs, and proprietary reader files
  are not decoded. Gutenberg URLs can use a readable alternate edition when
  available. The importer is not a universal media decoder.
- Generic layout and author/editor attribution remain heuristic. Unknown
  editorial classes cannot always be identified. Inspect extraction warnings
  and the original text before trusting attribution.
- Multilingual retrieval quality varies. Scores are not calibrated confidence
  probabilities. Small regression checks are not a representative benchmark.
- Citation checks establish membership, not entailment. Generated interpretations
  can still overreach. The experimental reviewer remains advisory.
- This is a local, one-selected-book app. Authentication, remote deployment,
  durable workers, full conversation memory, and cross-book comparison remain
  separate product work.

The running app is at http://127.0.0.1:8501. See README.md for startup and usage.
Sources, indexes, runs, private handoff files, reports, and logs remain
gitignored. Repository cleanup removed the old embedding experiment,
hard-coded downloader, API ping, keyword-search demo, private-run fixture
generator, generated bytecode, and temporary extraction checks. Publication
contains the application, reusable pipeline tools, tests, and setup documents.
