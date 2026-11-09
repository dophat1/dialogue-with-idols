# Dialogue with Idols

![A meme about Plato writing Socratic dialogues](platodialogue.jpg)

## Preface

I have always admired great thinkers and wished I could sit down and talk with
them. Most lived long before us, so that conversation is impossible in the
literal sense. Their writing, though, still gives us something to respond to.
That led me to wonder whether AI could help me explore their ideas through a
conversation grounded in their own works.

Project Gutenberg makes many of those works freely available in formats that
are practical to process. I started Dialogue with Idols as a way to bring that
personal idea a little closer: choose a book, ask a question, and see the
passages behind the answer. This is an early RAG demo, not a recreation of any
thinker's mind or voice. I plan to keep improving how it reads, retrieves, and
discusses their work.

The name is a playful nod to Nietzsche's *Twilight of the Idols*. If the idols
are gone, perhaps I can bring them back for a conversation—at least through
the words they left behind.

Today, the local Streamlit app lets you import a Project Gutenberg book, search
its text, and ask questions with cited source passages.

## Start the app

From the repository root, using the existing project environment:

```powershell
.\.venv\Scripts\python.exe -m streamlit run src/discussion_app.py --server.address 127.0.0.1
```

Open **http://127.0.0.1:8501**.

## Public portfolio demo

`src/demo_app.py` is a Streamlit Community Cloud entry point for reviewers.
It includes *The Prince* and *The Art of War* as ready-to-search sample books,
so visitors can try **Find passages** immediately. The public demo does not
allow imports or save discussions on the server. Answers remain in the current
browser session, with up to three generations per session. The local app above
keeps the full import and saved-discussion workflow.

To deploy the demo from this GitHub repository:

1. Create a Streamlit Community Cloud app using branch `main` and main file
   `src/demo_app.py`. Choose Python 3.12 in Advanced settings.
2. In the app's private Secrets settings, add root-level `OPENAI_API_KEY` and
   `OPENAI_MODEL` values. The application reads them as environment variables.
   Never add real values to this repository.
3. Open the app, choose a sample book, and try **Find passages** and
   **Generate cited answer**. Check Cloud logs and memory usage on first start;
   the local embedding and reranking models download when first used.

Generation uses the owner's API account. A per-session limit is only a demo
convenience, not an account-wide spending limit; set an OpenAI project budget
before sharing the public link. Streamlit Community Cloud can hibernate or
hit memory limits, so check the live app shortly before sending it to a hirer.

1. In the sidebar, paste a Gutenberg ebook URL or number (for example `11`),
   or upload a downloaded book. Click **Import and index**.
2. Select the book and inspect **Book and extraction details** for metadata
   and extraction warnings.
3. Type your question. **Find passages** searches locally and shows matches,
   surrounding context, and links to the original source.
4. **Generate cited answer** sends the question and selected evidence to OpenAI.
   Inspect the evidence attached to each argument.
5. Reopen saved answers from the sidebar or continue the current discussion.

The original Prince and Art of War indexes and saved answers remain supported.
Switching books clears current results/context. Discussions use one selected
book at a time. Imported editions appear automatically; there is no two-book
allowlist and no per-book configuration file to create.

## Supported imports

- Gutenberg book-page URLs, ebook numbers, or direct download URLs.
- Local HTML/HTM/XHTML, plain text, EPUB, and text/HTML ZIP files.
- UTF-8, UTF-16, and encoding detection for older text files.
- EPUB reading order from the package spine, with internal source locators.
- Metadata from HTML tags, EPUB metadata, or Gutenberg text headers. Missing
  fields are reported instead of inventing an author.

The importer resolves Gutenberg page links to readable downloads. Site/help
pages and redirects to another book are rejected. Standard HTML/text cache
paths provide a fallback when format discovery is unavailable. URLs for some
older binary formats resolve through the ebook page to its readable edition.

The upload limit is 50 MB; archives are limited to 100 MB unpacked and 3000
entries. Image-only scans, audio, video, proprietary reader formats, mathematical
layout, and arbitrary PDFs are not decoded locally. For these, use the Gutenberg
book URL if a readable text/HTML edition exists. No OCR is performed. Gutenberg
publishes a wider range of formats than these text adapters cover; see its
[format documentation](https://www.gutenberg.org/help/file_formats.html).

## How retrieval works

Text is normalized into sections and source-located paragraphs. Token windows
are bounded by the selected embedding model's actual input limit. New imports
use `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`; existing
English indexes retain their recorded model.

Queries combine semantic similarity and phrase-aware BM25 matching using
reciprocal rank fusion. The top results retain the strongest lexical match
alongside neural choices so exact names and relations are not lost. A local
cross-encoder then reranks the union of semantic and lexical candidates with
nearby paragraph context. English books use `cross-encoder/ms-marco-MiniLM-L6-v2`;
other/unknown languages use `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`.
Model resources are cached and reused. Initial model downloads and indexing
can take time; subsequent questions reuse the installed models.

Known commentary/footnote markers are preserved and excluded from author-text
indexing. Generic editions can contain unmarked editorial material; automatic
attribution is not guaranteed. Unusual HTML falls back to continuous text with
an extraction warning. Paragraph structure and source anchors can be lost in
that fallback. Tables of contents and common Gutenberg boilerplate are removed
where identifiable.

Answers use up to six retrieved paragraph excerpts. Very long paragraphs use
bounded excerpts with character offsets rather than being discarded. The app
checks citation membership and basic answer consistency, not whether every
claim is semantically entailed. Retrieval scores are not truth/confidence
probabilities. No system can guarantee the correct answer to every question.

## Local storage and configuration

New books are published as complete bundles under `data/processed/<edition-id>/`:

```text
book.json       metadata, sections, paragraphs, provenance, warnings
passages.json   searchable windows, parent IDs and character offsets
vectors.npz     embeddings, model name and artifact fingerprints
source.bin      exact imported source bytes
```

Content/version identities keep changed editions separate. Failed imports do
not replace working indexes. Book and passage fingerprints reject stale text
and mismatched embeddings. Identical imports reuse their existing complete
index. Sources, indexes, saved runs, and private reports remain gitignored.

Keep `OPENAI_API_KEY` and `OPENAI_MODEL` in the private `.env` file. Importing
and searching do not require an OpenAI key. Generation reads the existing
private configuration only when requested. No model setting is overwritten.
Copy `.env.example` to `.env` and replace its placeholders before generating
answers in a fresh clone. Do not overwrite an existing configured `.env`.

For a new environment, install the tested dependencies and start the app:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run src/discussion_app.py
```

## Command-line tools and checks

```powershell
# Full pipeline, from a URL/number or local file
.\.venv\Scripts\python.exe -m src.ingest 11
.\.venv\Scripts\python.exe -m src.ingest data/raw/my_book.epub

# Extraction only, with optional metadata overrides
.\.venv\Scripts\python.exe src/extract_book.py data/raw/my_book.html

# Search-only interface (no OpenAI requests)
.\.venv\Scripts\python.exe -m streamlit run src/search_app.py

# Automated unit and Streamlit tests
.\.venv\Scripts\python.exe -X utf8 -m unittest discover -s tests -v

# Small retrieval regression check against available local demo books (no API)
.\.venv\Scripts\python.exe -X utf8 -m evals.retrieval_smoke
```

The automated tests stub models/API calls. Separate real-model and live API
smoke checks are documented in [DEVELOPMENT.md](DEVELOPMENT.md). These small
checks are development evidence, not a comprehensive accuracy benchmark.

`src/review_run.py` optionally reviews a saved run against its stored citations.
It makes a separate OpenAI request and remains advisory. Reusable extraction,
inspection, chunking, and embedding commands remain in `src/`; obsolete demo
scripts and generated caches are excluded from the published project.

## First quality evaluation

On 2026-11-10, I ran a first evaluation with 20 questions across *The Prince*,
*The Art of War*, *Alice's Adventures in Wonderland*, and *The Raven*. The
questions and expected source paragraphs were prepared by reading selected
passages before running retrieval. Ten cases were marked for development and
ten for a first holdout check; each split had eight answerable and two
unanswerable questions. The labels and claim review were AI-assisted and still
need independent human review. This small set is a diagnostic check, not a
general accuracy estimate.

| Check | Result |
| --- | ---: |
| Preselected evidence found in the first six results | 14/16 answerable questions (87.5%) |
| All preselected paragraphs found in the first six | 13/16 (81.3%) |
| Mean reciprocal rank of the first preselected paragraph | 0.557 |
| Unanswerable questions answered with insufficient evidence and no claims | 4/4 |
| Answers produced without a validation error | 19/20 |
| Generated claims judged fully / partly / unsupported | 23 / 6 / 1 out of 30 |

Two questions missed their preselected source paragraph. The mercenaries
question in *The Prince* retrieved other passages that answered it, revealing
an incomplete gold label. The *Art of War* question about what is better than
winning every battle missed the passage about breaking resistance without
fighting. Its generated answer discussed preserving an enemy country instead,
which did not answer the question. One other answer was rejected by the app's
validation path; the exact cause was not captured in that run. Some accepted
answers added details or certainty beyond their citations. These are concrete
targets for improving retrieval and claim checking.

The local evaluation runner is
`.\.venv\Scripts\python.exe -m evals.quality_eval` (add `--answers` for paid API
generation). It reads a locally curated question file and writes
full retrieved passages and model outputs to Git-ignored `docs/` files. That
question file and the detailed transcript are local evaluation artifacts and
are not included in a fresh clone. Future portfolio claims should follow
human review of the labels and a larger, more varied set of books and questions.

## Where this could go next

The aim is to move from asking about a book to feeling as though you can talk
with its author. The current app retrieves passages and explains them; the next
steps would build a more convincing conversation around that evidence:

1. **Let the author answer in a grounded voice.** A reader could ask a question
   or challenge an idea and receive a response written from the author's
   perspective. Each substantive point should link to passages from their
   work. When the text leaves room for interpretation, the response should say
   so rather than inventing a view the author never expressed.
2. **Build a fuller picture of each thinker.** Bring together multiple works
   by the same author, with clear edition and translation labels. Let the
   conversation draw on more than one text while keeping every claim traceable
   to the work that supports it.
3. **Invite several thinkers into one discussion.** Give each persona an agent
   that can search and read its own source texts. A reader could pose a topic,
   ask follow-up questions, and watch the personas respond to one another's
   arguments. A moderator could keep the exchange focused and show where their
   documented views agree, differ, or do not address the question.
4. **Make the dialogue inspectable.** Show the passages behind each turn, open
   citations at the relevant place in the book, and distinguish source-based
   claims from the AI's connective interpretation. Let readers save and revisit
   conversations.
5. **Evaluate whether it stays faithful.** Have readers review the initial
   evaluation labels, then test persona answers and multi-persona exchanges for
   citation support, faithful attribution, useful disagreement, and honest
   handling of gaps in the texts.

These are plans, not current features. A persona would be an AI interpretation
grounded in surviving works, not the historical person brought back to life.
