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

---

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

**Result on real data:** 18,397 cleaned comments produced **30,508 chunks** — roughly 1.7 chunks per comment on average, reflecting that most comments are short one-liners while a smaller number of longer comments split into 2–3 pieces.

---

### Task 2: Generating Embeddings (with performance logging)

**What an embedding is:** a list of numbers representing the *meaning* of a piece of text. Texts with similar meaning end up with numerically similar representations even if they share no exact words — this is what makes meaning-based search possible, as opposed to simple keyword matching.

**Model used:** `all-MiniLM-L6-v2`, a small, fast, well-regarded general-purpose sentence embedding model that outputs 384-dimensional vectors per input.

**Performance was measured and logged across multiple configurations, all on a Colab T4 GPU:**

| Configuration | Chunks embedded | Total time | Rate |
|---|---|---|---|
| Baseline, batch size 64 | 5,000 (sample) | 3.18 seconds | 1,574.26 chunks/sec |
| Baseline, batch size 64 | 30,508 (full dataset) | 21.10 seconds | 1,445.94 chunks/sec |
| Optimized: half-precision (fp16) + batch size 256 | 30,508 (full dataset) | 9.97 seconds | 3,059.42 chunks/sec |

**Key finding:** switching the model to half-precision (`model.half()`) and increasing batch size from 64 to 256 roughly **doubled** throughput on the full dataset — cutting embedding time from ~21 seconds down to under 10 seconds, with no meaningful loss in embedding quality for this use case. (A separate CPU-only comparison in an earlier session, on the same 5,000-chunk sample, took 165.37 seconds on CPU versus ~3-4 seconds on GPU — roughly a 45x speedup from GPU alone, before any further optimization.)

**Final embedding output:** 30,508 chunks, each represented as a 384-number vector.

---

### Task 3: ChromaDB Setup, Ingestion, and Semantic Search

**What ChromaDB is:** a vector database — instead of searching by exact matches like a conventional database, it stores embeddings and retrieves by *similarity*, finding the stored vectors mathematically closest to a query's vector.

**Setup:** an in-memory ChromaDB client was created for this prototyping stage (data persists only within a single Colab session), with a single collection (`reddit_comments`) to hold all comment chunks. Each chunk was stored along with a unique ID, its embedding, and its original text, so search results come back human-readable without needing a separate lookup step. Ingestion was done in batches of 5,000 to stay within safe request sizes.

**Result:** all 30,508 chunks were ingested in **30.32 seconds**.

**Semantic search** was implemented so that a typed query is first embedded with the same model used for the stored chunks, then compared against the collection to retrieve the closest matches by similarity.

**Retrieval performance was tested and latency logged across multiple queries:**

| Query | Latency | Result quality |
|---|---|---|
| "is AI going to replace programmers" | 17.29 ms | Relevant — returned comments about AI replacing people and tasks, with no exact keyword overlap |
| "AI is making people lose critical thinking skills" | 18.33 ms | Relevant — returned comments specifically about critical thinking decline |
| "how good is Google's AI model" | 18.19 ms | Relevant — returned comments specifically evaluating Google's AI |

**Why this matters:** none of these queries shared exact wording with their top matches (for example, "programmers" never literally appears in the matched comment about "replacing people"), confirming the system retrieves based on genuine meaning rather than keyword overlap. Latency stayed under 20ms across all tested queries, fast enough for real-time interactive use.

---

### Task 4 & 5: Documenting the Chunking Strategy & Handling ChromaDB Edge Cases

**Chunking strategy documentation** is detailed in Task 1 above, and is also documented directly inside the notebook as a markdown cell placed alongside the chunking code — so the logic is visible in context, not just in this README.

**ChromaDB edge case handling:** the search function was hardened against common failure cases before being considered complete:

| Edge case | Handling approach | Result when tested |
|---|---|---|
| Empty query | Rejected immediately with a clear error, no embedding attempted | Returned `{'error': 'Query cannot be empty'}` |
| Whitespace-only query | Treated the same as an empty query | Same graceful error as above |
| Excessively long query | Truncated to a safe maximum length (1,000 characters) before embedding | Implemented as a safeguard against degraded performance on unbounded input |
| Requesting far more results than exist (`n_results=999999`) | Automatically capped to the actual available count | Correctly returned 30,507 results (essentially the full collection of 30,508 items) instead of erroring |
| Unexpected internal errors | Caught and returned as a readable error rather than crashing the program | Not triggered during testing, but in place as a safety net |
| Normal query (sanity check) | Confirmed the added safety checks didn't break normal behavior | Returned 5 results in 9.59 ms, no regression from the safety additions |

---

### Summary of Week 2 Results

| Metric | Value |
|---|---|
| Comments chunked | 18,397 |
| Chunks generated | 30,508 |
| Embedding dimensions | 384 |
| Full-dataset embedding time (GPU, fp16, batch 256) | 9.97 seconds |
| ChromaDB ingestion time (full dataset) | 30.32 seconds |
| Sample query latency range | 9.59 ms – 18.33 ms |

### Known Limitations

- ChromaDB is currently in-memory only — the collection is wiped when the Colab session ends. Persistent on-disk storage would be needed for a production version.
- Only the comments dataset was chunked and embedded this week; posts are not yet part of the search index.
- The excessively-long-query edge case was implemented but not explicitly exercised with a real oversized input during this week's testing.

---

## Week 3: LLM Integration, Prompt Engineering & Robust RAG

This week's goal was to complete the RAG (Retrieval-Augmented Generation) pipeline started in Week 2 — taking the semantic search system built last week and connecting it to an actual LLM, so the system can generate real, grounded answers from retrieved Reddit comments instead of just returning raw search results.

### Pipeline Overview

Query → semantic search retrieves relevant chunks (Week 2) → domain check confirms the query is actually answerable from this dataset → chunks are injected into a structured prompt → LLM generates an answer → a second LLM call verifies the answer is grounded in the retrieved context → the full process is timed stage-by-stage.

---

### Task 1: Integrating the LLM API (OpenRouter)

**Setup:** created an OpenRouter account and API key. The key was stored securely using Colab's built-in Secrets manager (accessible via the key icon in the sidebar) rather than hardcoded into the notebook, so it's never exposed if the notebook is pushed to a public GitHub repo.

**Error encountered #1 — 404 on the initial model choice:**
The first attempt used `deepseek/deepseek-chat-v3.1:free` as the model, which returned a `404` error. Research confirmed this wasn't a bug on our end — DeepSeek had removed all free-tier models from OpenRouter entirely by the time of this session (a change that happened sometime after many tutorials/guides referencing that model ID were written). This is a good example of why hardcoding a specific "free" model ID long-term is risky — the free model roster on OpenRouter changes frequently.

**Fix:** queried OpenRouter's own live model-listing endpoint directly from the notebook (`GET /api/v1/models`), filtered to models with $0 pricing on both input and output, and picked from that real, current list rather than trusting a hardcoded name from documentation. Landed on **`openai/gpt-oss-20b:free`** — a genuine OpenAI open-weight model, general-purpose (not code- or audio-specialized like several other free options available), and a good balance of quality vs. rate-limit risk.

**Basic API call implementation:** built a `call_llm()` function that sends a list of `{"role": ..., "content": ...}` messages to OpenRouter's chat completions endpoint and returns the raw response. Verified working with a simple test message (status 200, correct reply).

---

### Task 2: Prompt Engineering & Context Injection

**System prompt design:** implemented a system prompt that explicitly instructs the model to:
- Answer using *only* the provided Reddit comments (no outside knowledge)
- Clearly state when the context doesn't contain enough information, rather than guessing
- Cite which comment(s) informed each part of the answer

**Context injection:** built a `build_rag_prompt()` function that takes a user query and a list of retrieved chunks (from Week 2's `semantic_search`), formats the chunks as a bulleted context block, and combines them with the query into a user message — paired with the system prompt above.

**Result:** tested end-to-end with the query *"is AI going to replace programmers"*. The model correctly synthesized an answer using only the retrieved comments, explicitly referencing "Comment 1," "Comment 4," etc. — confirming the context injection and citation instruction both worked as intended, rather than the model falling back on its own general training knowledge.

---

### Task 3: Robust Error Handling for API Calls

Built `call_llm_safe()` to replace the basic `call_llm()`, handling the specific failure modes an LLM API can produce:

| Failure case | Handling |
|---|---|
| Rate limiting (HTTP 429) | Retries with increasing wait time between attempts (5s, 10s, 15s), up to 3 attempts, instead of failing immediately |
| Token/context limit exceeded (HTTP 400) | Detected and surfaced as a clear, readable error message rather than a cryptic failure |
| Network timeout | Caught and retried, rather than crashing the whole pipeline |
| Other unexpected connection errors | Caught and returned as a descriptive error instead of an unhandled exception |
| Max retries exhausted | Returns a clear "giving up" message rather than hanging indefinitely |

**This was validated for real, not just in theory:** during later end-to-end latency testing, a genuine `429` rate-limit response was hit organically (visible in the logs as *"Rate limited. Waiting 5s before retry (1/3)..."*), and the retry logic successfully recovered and completed the request on the next attempt — confirming the error handling works under real conditions, not just hypothetical ones.

---

### Task 4: Hallucination Checks & Out-of-Domain Query Handling

**Out-of-domain detection:** implemented `is_query_in_domain()`, which uses ChromaDB's own distance scores (already available from the retrieval step) to judge whether a query is even relevant to this dataset before attempting to answer it.

**Error encountered #2 — incorrectly rejecting valid queries:**
The first version used a similarity threshold of `0.35`, chosen without evidence. This caused *every* query — including clearly relevant ones like "is AI going to replace programmers" — to be incorrectly flagged as out-of-domain and rejected.

**Fix — calibrated the threshold using real data instead of a guess:** printed the actual ChromaDB distance scores for a known in-domain query versus a known out-of-domain query:
- In-domain query ("AI replace programmers") → distances clustered around **0.55–0.58**
- Out-of-domain query ("capital of France") → distances clustered around **1.17–1.38**

This revealed a clear, wide gap between the two clusters. The threshold was updated to **0.8**, sitting safely in that gap. Re-testing confirmed both cases now behave correctly: the in-domain query is answered normally, and the out-of-domain query is correctly rejected before any LLM call is made — this is documented in the code as a comment explaining exactly where the number came from, rather than leaving it as an unexplained "magic number."

**Hallucination checking:** implemented `check_hallucination()` using an **LLM-as-judge** approach — a second, separate LLM call given the original context and the generated answer, instructed to respond with only `GROUNDED` or `UNGROUNDED` depending on whether every claim in the answer is actually supported by the context. This is a widely used practical technique, though not a perfect guarantee (an LLM checking itself has inherent limits) — noted here as an honest limitation, not oversold as foolproof.

**Full pipeline (`rag_answer`) combining both:** retrieves chunks → checks domain relevance (short-circuits with a clear message if irrelevant, without wasting an LLM call) → generates an answer if relevant → runs the hallucination check on the result.

**Verified results after the threshold fix:**
- In-domain query → real synthesized answer returned, hallucination check independently returned `GROUNDED`
- Out-of-domain query ("what is the capital of France") → correctly rejected with a clear explanation, no LLM call attempted

---

### Task 5: End-to-End Latency Measurement & Logging

Built `rag_answer_with_latency()` to time each stage of the pipeline separately (not just the total), so it's clear *where* time is actually being spent rather than treating the whole pipeline as one black box.

**Real results from testing three queries:**

| Query | Retrieval | Domain check | Generation | Hallucination check | Total |
|---|---|---|---|---|---|
| "is AI going to replace programmers" | 33.64 ms | 11.42 ms | 32,707.18 ms | 6,010.50 ms | 38,762.73 ms |
| "how good is Google's AI model" | 10.33 ms | 7.80 ms | 12,383.10 ms | 26,520.87 ms | 38,922.11 ms |
| "what is the capital of France" (out-of-domain) | 10.51 ms | 7.86 ms | — (skipped) | — (skipped) | 18.37 ms |

**Key finding:** retrieval and domain-checking are essentially instantaneous (under 35ms combined), and are not a bottleneck anywhere in this pipeline. The overwhelming majority of end-to-end latency — often 30+ seconds — comes from the two LLM calls (generation and hallucination checking), which is the expected cost of using a free, shared, rate-limited model tier rather than a paid, dedicated one.

**A genuine rate-limit event was captured live during this test** (visible in the console output as a retry message), demonstrating the Task 3 error handling working under real conditions rather than only in isolated testing.

**Important design observation worth noting:** the hallucination check roughly doubles the slowest part of the pipeline's cost, since it requires a full second LLM call on top of generation. For a production system, this suggests the hallucination check could reasonably be made optional, or only triggered for lower-confidence answers, rather than run unconditionally on every single query — a real tradeoff between reliability and speed that this latency data made visible.

---

### Summary of Week 3 Results

| Metric | Value |
|---|---|
| LLM used | `openai/gpt-oss-20b:free` (via OpenRouter) |
| Domain-check threshold | 0.8 (calibrated from real distance data) |
| Retrieval + domain check latency | ~10-45 ms combined |
| Generation latency (typical) | ~12-33 seconds (free-tier model) |
| Hallucination check latency (typical) | ~6-27 seconds |
| Out-of-domain queries | Correctly short-circuited in under 20ms, no LLM call made |

### Known Limitations

- The free LLM tier used here is noticeably slow (12-33 seconds per generation) and subject to rate limiting; a paid tier or a different provider would be needed for a production-grade response time.
- The hallucination check is an LLM-as-judge heuristic, not a guaranteed detector — it can itself be wrong, and roughly doubles total latency when enabled.
- The domain-check threshold (0.8) was calibrated using only two example queries; a larger, more varied test set would give a more robust threshold.
- API key is currently used via Colab Secrets for local development; a deployed version would need a proper secrets-management setup (e.g. environment variables on the hosting platform).


## Week 4: Topic Modeling, Sentiment Analysis & Metadata-Enhanced Retrieval

This week's goal was to add semantic *understanding* on top of the Week 2/3 retrieval and generation pipeline — discovering themes across the dataset, tagging emotional tone, and using both to enable filtered search.

### Task 1: Topic Modeling (BERTopic)

**Approach:** used BERTopic, reusing the same `all-MiniLM-L6-v2` embedding model from Week 2 so topics live in the same semantic space as the search system, rather than introducing a separate model.

**Initial result:** BERTopic discovered clean, coherent topics (LLM discussion, image/video generation, gaming, AI consciousness debate, stock/finance, math tools, AI-job-replacement, US-China chip competition, Google scam calls, and more) — but **52% of all comments (9,527 of 18,397) fell into the "-1" unclustered/outlier bucket**, more than every real topic combined.

**Diagnosis (validated, not assumed):** the first hypothesis was that outliers were simply short, low-content comments. This was checked directly against the data and **disproven** — outlier comments averaged 282 characters versus 267 for clustered comments, essentially the same length. The real cause is that BERTopic's default clustering algorithm (HDBSCAN) is density-based: it only forms a topic when enough documents pack tightly together, and Reddit's wide topic diversity leaves many legitimate, substantive comments too spread out to meet that density bar, even though they aren't low-quality.

**Fix:** applied `topic_model.reduce_outliers(..., strategy="embeddings")`, which reassigns each outlier to its nearest real topic by embedding similarity rather than leaving it unclustered. This reduced outliers from 9,527 to **0**, with topic sizes becoming much more balanced (roughly 100–450 comments per topic instead of one dominant catch-all bucket).

### Task 2: Sentiment Analysis (with manual accuracy evaluation)

**Model used:** `cardiffnlp/twitter-roberta-base-sentiment-latest`, chosen specifically because it's trained on social-media text, which generalizes better to Reddit's informal style than a model trained on, say, product/movie reviews.

**Full dataset run:** sentiment computed for all 18,397 comments in 447.92 seconds (~41 comments/sec on GPU).

| Sentiment | Count | % |
|---|---|---|
| Neutral | 8,751 | 47.6% |
| Negative | 6,979 | 37.9% |
| Positive | 2,667 | 14.5% |

**Manual accuracy evaluation:** a random sample of 30 comments was manually labeled by hand and compared against the model's predictions.

- **Result: 25/30 correct = 83.3% accuracy**
- **Error pattern (not random):** all 5 misclassifications were the model over-predicting "negative" — 3 cases of true-neutral text called negative, 2 cases of true-positive text called negative. Zero errors in the opposite direction. This suggests the model has a directional bias toward "negative" on text that discusses serious/heavy topics (e.g. AI job anxiety) in an otherwise neutral or even hopeful tone — it appears to be picking up on topic weight rather than actual sentiment in some cases.

### Task 3: Manual Validation & Edge Case Handling

Two specific edge cases were tested directly against real data, as required:

**Short documents (<20 characters, 974 comments in the dataset):** mixed results. Some short comments correctly clustered into sensible micro-topics that emerged naturally from the data itself — e.g. a "yes/nope/uh" affirmation-topic and a "lol/lmao/haha" reaction-topic. However, some short comments (e.g. "You broke the image") were forced into unrelated, nonsensical topics purely because there wasn't enough content to disambiguate meaning. This is an honest, documented limitation rather than a fully solved problem.

**Jargon-heavy technical comments (54 comments containing terms like "VRAM," "hyperparameter," "quantization," etc.):** performed well — these consistently clustered into coherent, correctly-scoped technical topics (VRAM/hardware discussion, Ollama/llama.cpp discussion, RAG/vector-DB discussion), showing the embedding model handles domain-specific ML jargon better than it handles very short, low-context text.

**Metadata integration bug (caught and fixed):** the first attempt at attaching topic/sentiment metadata to ChromaDB chunks silently failed — every single chunk showed `topic: -1, sentiment: 'unknown'`. Root cause: the lookup dictionaries were built using integer row positions as keys, while `chunk_sources` (from Week 2) stores actual Reddit comment IDs — so every lookup missed and fell back to the default value. Fixed by rebuilding the lookup dictionaries keyed on the real comment ID (`dict(zip(comments_clean["id"], comments_clean["topic"]))`), which resolved all 30,508 chunks correctly (0 unmatched).

### Task 4: Integrating Metadata into ChromaDB for Filtered Retrieval

Both `topic` (integer topic number) and `sentiment` (positive/negative/neutral) were attached as metadata to every one of the 30,508 chunks already stored in ChromaDB from Week 2, using `collection.update(ids=..., metadatas=...)` in batches of 5,000.

A `filtered_semantic_search()` function was built on top of the existing Week 2 search, adding an optional `where` clause so results can be narrowed by sentiment and/or topic *before* ranking by semantic similarity.

**Demonstrated result** for the query *"AI replacing jobs"*:
- **Unfiltered:** general mix of comments about AI and job displacement
- **Negative-filtered:** comments framing AI job displacement as a crisis (e.g. concerns about lost purpose, lack of UBI, inadequate legislation)
- **Positive-filtered:** comments framing the same topic pragmatically/optimistically (e.g. productivity gains, transition planning)

This confirms the metadata isn't just attached but functionally useful — the same query returns meaningfully different, correctly-filtered result sets depending on the emotional lens requested.

### Problems Encountered This Week (Chronological)

Five distinct issues came up during this week's work. Each is documented here with what went wrong, how it was diagnosed, and how it was fixed — not just the two most significant ones, but every real problem hit along the way.

**Problem 1: 52% of comments left unclustered by BERTopic**

- **Symptom:** after the initial `fit_transform()` run, topic `-1` (BERTopic's "outlier/unclustered" bucket) contained 9,527 of 18,397 comments — more than every real topic combined.
- **First hypothesis (wrong):** assumed outlier comments were simply short, low-content text that didn't give the clustering algorithm enough to work with.
- **Diagnosis:** tested the hypothesis directly instead of accepting it — compared average character length of outlier vs. clustered comments. Outliers averaged 282 characters, clustered comments averaged 267 — essentially the same, disproving the "short comments" theory.
- **Real cause:** BERTopic's default clustering algorithm (HDBSCAN) is density-based — it only forms a topic when documents pack tightly enough together. Reddit's wide topic diversity means many legitimate, substantive comments are too spread out in embedding space to meet that density threshold, even though they aren't low-quality.
- **Fix:** used `topic_model.reduce_outliers(..., strategy="embeddings")` to reassign every outlier to its nearest real topic by embedding similarity. Outliers went from 9,527 to 0, and topic sizes became much more balanced (roughly 100–450 per topic).

**Problem 2: CUDA crash during sentiment analysis (`AcceleratorError: device-side assert triggered`)**

- **Symptom:** running the sentiment pipeline on a 2,000-comment sample crashed with a CUDA-level error, even though `truncation=True` was already set.
- **Diagnosis:** `truncation=True` alone doesn't guarantee a safe cutoff length on every tokenizer — some report an unreasonably large "max length" internally, so truncation effectively did nothing, and a comment exceeding the model's real position limit crashed the GPU at the hardware level.
- **Fix, attempt 1:** added `max_length=512` explicitly to force a real cutoff. This is the correct fix in principle, but running it immediately after the crash **still failed with the identical error** — because a CUDA "device-side assert" leaves the GPU session itself corrupted; no amount of code-level fixing works until the runtime is genuinely restarted.
- **Fix, attempt 2 (the real fix):** restarted the Colab runtime (Runtime → Restart session), confirmed the restart actually took effect by checking that `sentiment_pipeline` no longer existed in memory (the first restart attempt silently failed to take effect — see Problem 3), then re-ran the corrected code with `max_length=512` on a clean GPU session. This worked cleanly: 2,000 comments in 36.04 seconds.

**Problem 3: Colab restart appeared to not take effect**

- **Symptom:** after clicking "Restart session" and re-running the sentiment cell, the exact same CUDA crash happened again — suggesting the restart hadn't actually cleared anything.
- **Diagnosis:** added a one-line check — `try: print(sentiment_pipeline) except NameError: print("gone")` — to directly verify whether the old pipeline object still existed in memory. It did, confirming the restart genuinely hadn't taken effect (rather than assuming and re-guessing at code fixes).
- **Fix:** explicitly restarted again and waited for Colab's connection indicator to fully cycle (disconnect then reconnect) before running anything else, then confirmed with the same check before proceeding. Once confirmed clean, the `max_length=512` fix worked as expected.

**Problem 4: Sentiment analysis on the full dataset felt like it was hanging**

- **Symptom:** running sentiment across all 18,397 comments produced no visible output for several minutes, making it unclear whether the process was working or stuck.
- **Diagnosis:** the base `pipeline()` call doesn't print any progress by default, so a genuinely-running multi-minute job looks identical to a frozen one from the outside.
- **Fix:** rewrote the loop using `tqdm` to wrap the batches with a live progress bar, and increased batch size from 64 to 128 for better GPU throughput. The full run completed in 447.92 seconds (~41 comments/sec) with visible progress throughout, confirming it had been working correctly the whole time — this was a visibility problem, not an actual bug.

**Problem 5: All chunk metadata silently defaulted to `topic: -1, sentiment: 'unknown'`**

- **Symptom:** after attaching topic/sentiment metadata to all 30,508 ChromaDB chunks, a spot-check of the sample metadata showed every single entry was the fallback default, not a real value — despite the code running without any errors.
- **Diagnosis:** `chunk_sources` (built back in Week 2) stores each chunk's original Reddit comment ID (a string) as its source reference. The metadata lookup dictionaries, however, were built using `.to_dict()` on a reset dataframe index, which uses integer row positions as keys instead — so every lookup like `topic_lookup.get("op1cly9", -1)` failed to find a match and silently fell back to the default, with no error thrown to signal the mismatch.
- **Fix:** rebuilt both lookup dictionaries keyed on the actual comment ID column — `dict(zip(comments_clean["id"], comments_clean["topic"]))` — matching the key type that `chunk_sources` actually contains. Re-ran the metadata attachment; confirmed 0 unmatched chunks out of 30,508, and spot-checked sample metadata to verify real topic numbers and sentiment labels were now present.



| Metric | Value |
|---|---|
| Topics discovered (after outlier reduction) | 15+ coherent topics, 0 unclustered |
| Outliers before fix | 9,527 (52% of dataset) |
| Outliers after fix | 0 |
| Sentiment analysis coverage | All 18,397 comments |
| Manual evaluation accuracy | 83.3% (25/30) |
| Chunks with metadata attached | 30,508 (100%, after bug fix) |

### Known Limitations

- Sentiment model shows a directional bias toward "negative" on emotionally-neutral text about serious topics — worth accounting for when interpreting sentiment distributions, rather than treating them as fully objective.
- Very short comments (<20 characters) can still be assigned to topics that don't meaningfully reflect their content, since there's often too little signal to disambiguate.
- The manual evaluation set (30 comments) is small; a larger labeled set would give a more statistically reliable accuracy estimate.
- Topic numbers are stable only within a given BERTopic run — re-fitting the model (e.g. with new data) would require re-attaching metadata to ChromaDB.


