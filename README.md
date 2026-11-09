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
.\.venv\Scripts\python.exe -m streamlit run src/discussion_app.py
```

Open **http://127.0.0.1:8501**.

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

## Where this could go next

The long-term aim is a conversation with great thinkers through their works.
The current app is a single-book RAG demo; these are proposed improvements, in
rough priority order:

1. **Measure quality.** Build a set of questions with human-identified relevant
   passages. Measure whether retrieval finds them and whether each answer claim
   is supported by its citations. Use those results to guide later changes.
2. **Make the exchange feel more like a dialogue.** Offer modes such as
   *explain an idea*, *challenge my view*, and *compare arguments*. Keep the
   book's words distinct from the AI's interpretation, and let the user follow
   each response back to its source.
3. **Add a focused research agent.** For difficult questions, let one agent
   search the selected book, read passages, identify gaps, and search again
   before answering. Show the searches and evidence in the interface, with
   limits on steps, time, and API cost. Compare its results with the existing
   retrieval flow before making it the default.
4. **Compare thinkers across books.** Search several selected works and present
   differences in their arguments, with clear attribution and separate
   citations for each book.
5. **Improve the reading experience.** Open citations at the relevant passage,
   show more surrounding text, and let readers save and revisit a line of
   inquiry.

These ideas are plans, not features in the current release. The guiding rule
is to make the conversation more useful while keeping the underlying texts
visible and the limits of the AI clear.
