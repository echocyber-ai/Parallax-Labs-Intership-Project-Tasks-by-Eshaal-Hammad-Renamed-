# Reddit NLP Pipeline — Week 1: Data Collection + Cleaning

This repo contains my Week 1 deliverable for the internship: collecting one month of Reddit posts and comments from a chosen subreddit, cleaning the data, and running basic tokenization/corpus stats on it.

## Project Overview

The broader goal of this internship (based on the roadmap I was given) is to build toward a semantic search/analysis pipeline over real Reddit discussion data — collect it, clean it, understand it linguistically, and eventually convert it into a searchable format using sentence embeddings and similarity search. Week 1 focuses on the foundation: getting real data and making sure it's clean enough to build on.

## Subreddit & Time Range

- **Subreddit:** r/artificial
- **Date range:** June 1, 2026 – June 30, 2026
- **Data source:** [Arctic Shift API](https://github.com/ArthurHeitmann/arctic_shift) — used instead of PRAW since it allows querying historical posts/comments by exact date range without needing a registered Reddit developer app.

## Environment Setup

Built and run entirely in **Google Colab**, so no local install is strictly required. To run it locally instead, you'll need Python 3.9+ and:

```bash
pip install pandas spacy sentence-transformers faiss-cpu requests
python -m spacy download en_core_web_sm
```

In Colab, the first cell installs everything automatically:

```python
!pip install pandas spacy sentence-transformers faiss-cpu requests
!python -m spacy download en_core_web_sm
```

**Note:** after this cell runs, Colab sometimes shows a "Restart to reload dependencies" warning. This didn't cause any issues in this run, but if `spacy.load()` fails later with an import error, restart the runtime (Runtime → Restart runtime) and re-run the cells from the top.

## How to Run

The notebook (`Task 1`) is organized into 5 sequential cells. Run them in order, top to bottom, in a single Colab session:

1. **Install dependencies** — see above.
2. **Fetch posts** — pulls all posts from r/artificial for June 2026 via the Arctic Shift API. Includes retry logic (up to 5 retries per request) since the API intermittently returns `422` timeout errors under load — this is expected and handled automatically, not a bug. Saves progress to `posts_progress.json` after every batch of 100.
3. **Fetch comments** — same approach for comments. Takes noticeably longer (~15-20 min) since there are far more comments than posts, and hits the same intermittent `422` timeouts more frequently at scale. Saves progress to `comments_progress.json` after every batch.
4. **Clean the data**:
   - Combines post title + body, and comment body, into a single `text` column
   - Removes exact `[deleted]` / `[removed]` placeholder content
   - Removes bot comments (currently filters `AutoModerator` by username)
   - Strips URLs and markdown formatting (`**bold**`, `[text](link)`, `#`, `` ` ``, etc.) using regex
   - Drops rows that are empty after cleaning
   - Drops duplicate rows based on cleaned text
5. **Tokenize with spaCy + generate corpus stats** — lowercases text, removes stopwords and punctuation, and counts word frequency across a random sample of 2,000 cleaned comments (sampling keeps this fast — spaCy processing all ~18k comments individually would take considerably longer, and a 2k sample is plenty for representative stats).

## Results

**Raw collection:**

| Dataset | Rows | Columns |
|---|---|---|
| Posts | 2,317 | 117 |
| Comments | 20,447 | 75 |

**After cleaning:**

| Step | Posts | Comments |
|---|---|---|
| Removed deleted/removed | 2,317 (no change) | 18,791 |
| Removed bot comments | — | 18,791 (no AutoModerator found in this window) |
| Final (after empty-text & duplicate removal) | 2,264 | 18,397 |

**Tokenization (2,000-comment sample):**
- Total tokens generated: 43,125
- Top words: *ai, like, people, use, think, model, time, work, actually, way, good, real, data, need, things, models, thing, human, know, right*

This makes sense for r/artificial — heavy repetition of "ai," "model(s)," and "data" reflects the subreddit's core topic, while words like "think," "people," and "actually" reflect the discussion/opinion-heavy nature of the comments.

## Output Files

- `posts_progress.json` — raw collected posts (before cleaning)
- `comments_progress.json` — raw collected comments (before cleaning)
- Cleaned DataFrames (`posts_clean`, `comments_clean`) and tokenization results are generated inline in the notebook

## Known Limitations

- The Arctic Shift API returns frequent `422 Timeout` errors under sustained load. This is expected behavior on their end (their own docs mention this), and the retry logic handles it, but it does add time to the collection process.
- Bot filtering only excludes `AutoModerator` by exact username. No AutoModerator comments happened to appear in this specific subreddit/date window, so this filter had no visible effect here — it's kept in for robustness on future runs where it may matter more, and other bot accounts could be added to the filter list if spotted.
- Tokenization/corpus stats are computed on a 2,000-comment random sample rather than the full cleaned dataset, for speed. The full dataset can be tokenized later if more precise stats are needed.


## Week 2: Chunking, Embeddings & Semantic Search

This week's goal was to take the cleaned Reddit dataset from Week 1 and turn it into a working semantic search system — one that can find relevant comments based on *meaning*, not just exact keyword matches.

### Pipeline Overview

The system built this week follows this flow: cleaned comments from Week 1 are split into smaller text chunks, each chunk is converted into a numerical embedding, those embeddings are stored in a vector database (ChromaDB), and finally a search function lets you type a query and retrieve the most semantically relevant chunks.

**New dependencies added this week:** `sentence-transformers` and `chromadb`, alongside Week 1's `pandas`, `spacy`, and `requests`.

**Input data:** this week's work builds directly on the cleaned posts and comments produced in Week 1 within the same notebook — 2,264 cleaned posts and 18,397 cleaned comments.

---

### Task 1: Text Chunking Strategy (with unit tests)

**Why chunking matters:** embedding models work best on smaller, focused pieces of text rather than long blocks that might span multiple ideas. Chunking also allows search to return a specific relevant snippet instead of an entire, possibly long, comment.

**Strategy used: recursive splitting.** The approach tries the most natural split point first, and only falls back to a rougher method if a piece of text is still too long:
1. Split on paragraph breaks, if present
2. If a piece is still too long, split on sentence boundaries
3. If a single sentence is still too long, split on word boundaries as a last resort

A small overlap is carried from the end of one chunk into the start of the next, so context isn't abruptly lost at a chunk boundary — a common practice in retrieval systems.

**Parameters used:** a maximum chunk size of 300 characters with a 30-character overlap. These were chosen because Reddit comments are mostly short, so 300 characters keeps most comments as a single chunk while still splitting the minority of long, multi-topic comments into more focused pieces.

**Unit tests** were written to verify: short text returns unmodified as a single chunk, empty text returns an empty list rather than an error, long text correctly splits into multiple chunks, and no resulting chunk wildly exceeds the intended size limit. All tests passed.

**Result on real data:** 18,397 cleaned comments produced **30,507 chunks** — roughly 1.7 chunks per comment on average, reflecting that most comments are short one-liners while a smaller number of longer comments split into 2–3 pieces.

**Bug encountered and fixed:** the first run on real data threw a `TypeError` because some text values had become missing (`NaN`) after the Week 1 CSV was saved and reloaded — empty strings had been reinterpreted as true missing values on reload. This was fixed by explicitly skipping missing values and force-converting all text to string type before chunking.

---

### Task 2: Generating Embeddings (with performance logging)

**What an embedding is:** a list of numbers representing the *meaning* of a piece of text. Texts with similar meaning end up with numerically similar representations even if they share no exact words — this is what makes meaning-based search possible, as opposed to simple keyword matching.

**Model used:** `all-MiniLM-L6-v2`, a small, fast, well-regarded general-purpose sentence embedding model that outputs 384-dimensional vectors per input. Chosen for a good balance of speed and quality, suitable for this scale of prototyping.

**Performance was measured and logged across two environments:**

| Environment | Chunks embedded | Total time | Rate |
|---|---|---|---|
| CPU (Colab default) | 5,000 (sample) | 165.37 seconds | 30.23 chunks/sec |
| GPU (Colab T4) | 5,000 (sample) | 3.18 seconds | 1574.26 chunks/sec |
| GPU (Colab T4) | 30,507 (full dataset) | 20.50 seconds | 1,736.80 chunks/sec |

**Key finding:** switching Colab's runtime to a free T4 GPU gave roughly a **45x speedup** over CPU for the same workload. At this dataset's scale, GPU embedding is effectively instantaneous (under 20 seconds for all 30,507 chunks), while CPU alone would take nearly 3 minutes and would become a real bottleneck at larger scale.

**Final embedding output:** 30,507 chunks, each represented as a 384-number vector.

---

### Task 3: ChromaDB Setup, Ingestion, and Semantic Search

**What ChromaDB is:** a vector database — instead of searching by exact matches like a conventional database, it stores embeddings and retrieves by *similarity*, finding the stored vectors mathematically closest to a query's vector.

**Setup:** an in-memory ChromaDB client was created for this prototyping stage (data persists only within a single Colab session), with a single collection to hold all comment chunks. Each chunk was stored along with a unique ID, its embedding, and its original text, so search results come back human-readable without needing a separate lookup step. Ingestion was done in batches of 5,000 to stay within safe request sizes.

**Result:** all 30,507 chunks were ingested in **23.61 seconds**.

**Semantic search** was implemented so that a typed query is first embedded with the same model used for the stored chunks, then compared against the collection to retrieve the closest matches by similarity.

**Retrieval performance was tested and latency logged across multiple queries:**

| Query | Latency | Result quality |
|---|---|---|
| "is AI going to replace programmers" | 33.20 ms | Relevant — returned comments about AI replacing people and tasks, with no exact keyword overlap |
| "AI is making people lose critical thinking skills" | 8.61 ms | Relevant — returned comments specifically about critical thinking decline |
| "how good is Google's AI model" | 7.74 ms | Relevant — returned comments specifically evaluating Google's AI |

**Why this matters:** none of these queries shared exact wording with their top matches (for example, "programmers" never literally appears in the matched comment about "replacing people"), confirming the system retrieves based on genuine meaning rather than keyword overlap. Latency stayed well under 35ms across all tested queries, fast enough for real-time interactive use.

---

### Task 5: Documenting the Chunking Strategy & Handling ChromaDB Edge Cases

**Chunking strategy documentation:** the reasoning behind the chunking approach (recursive splitting, 300-character max size, 30-character overlap) is detailed in Task 1 above, and is also documented directly inside the notebook as a markdown cell placed alongside the chunking code — so the logic is visible in context, not just in this README.

**ChromaDB edge case handling:** to move beyond a working demo toward a more robust system, the search function was hardened against common failure cases before being considered complete:

| Edge case | Handling approach | Result when tested |
|---|---|---|
| Empty query | Rejected immediately with a clear error, no embedding attempted | Returned a clean error message instead of crashing |
| Whitespace-only query | Treated the same as an empty query | Same graceful error as above |
| Excessively long query | Truncated to a safe maximum length before embedding | Implemented as a safeguard against degraded performance on unbounded input |
| Requesting far more results than exist in the collection | Automatically capped to the actual available count | Correctly returned the full available set instead of erroring, when 999,999 results were requested against a much smaller collection |
| Unexpected internal errors | Caught and returned as a readable error rather than crashing the program | Not triggered during testing, but in place as a safety net |
| Normal query (sanity check) | Confirmed the added safety checks didn't break normal behavior | Worked identically to the unprotected version, with no added latency |

---

### Summary of Week 2 Results

| Metric | Value |
|---|---|
| Comments chunked | 18,397 |
| Chunks generated | 30,507 |
| Embedding dimensions | 384 |
| Full-dataset embedding time (GPU) | 17.57 seconds |
| ChromaDB ingestion time (full dataset) | 23.61 seconds |
| Sample query latency range | 7.74 ms – 33.20 ms |

### Known Limitations

- ChromaDB is currently in-memory only — the collection is wiped when the Colab session ends. Persistent on-disk storage would be needed for a production version.
- Only the comments dataset was chunked and embedded this week; posts are not yet part of the search index.
- The excessively-long-query edge case was implemented but not explicitly exercised with a real oversized input during this week's testing.






