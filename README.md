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
