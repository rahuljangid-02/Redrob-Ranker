# SRJ Prime — Intelligent Candidate Discovery & Ranking
### Redrob Data & AI Hackathon Submission

**Team:** SRJ Prime | **Team Members:** Sakshi Shekhawat & Rahul Jangid |


---
## What We Built


*A JD-aware hybrid retrieval system that ranks 100,000 candidates in under 5 minutes on CPU —  
no GPU, no external APIs, no keyword gaming.*


---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download models (first time only — requires internet)
python download_models.py

# 3. Run ranking  <-- single reproduce command
python rank.py --candidates ./candidates.jsonl --job-desc ./job_description.txt --out ./team_SRJ_Prime.csv
# 4. Validate output
python validate_submission.py ./team_SRJ_Prime.csv
```

Expected output:
```
[OK] Submission is valid.
     Ready to upload to the hackathon portal.
```

---

## Runtime

| Step | Time | Notes |
|------|------|-------|
| First run — builds embedding cache | ~6-8 hours | One-time only. Saves to `cache/embedding_cache.pkl` |
| Subsequent runs — cache loaded | **~4 minutes** | Within 5-min spec limit |
| Spec limit | 5 minutes | CPU only, 16 GB RAM, no network |

> Pre-computation (embedding cache) is explicitly allowed to exceed the 5-minute window per the submission spec.  
> The **ranking step** that produces `submission.csv` runs in ~4 minutes with cache loaded.

---

## System Overview

```
candidates.jsonl (100,000 profiles)  +  job_description.txt
                        |
          ┌─────────────┴──────────────┐
          │                            │
   [Step 1] Dense Retrieval     [Step 2] BM25 Lexical
   all-MiniLM-L6-v2             rank-bm25 (BM25Okapi)
   Semantic cosine similarity   Production skills weighted 4x
   Embedding cache (instant)    Exact keyword matching
          │                            │
          └─────────────┬──────────────┘
                        │
                [Step 3] Signal Bonus
                Career stability + Response rate
                Education tier + Endorsements
                        │
                [Step 4] JD-Aware Multiplier        <-- key differentiator
                0.40x to 1.20x per candidate
                Encodes JD disqualifiers explicitly
                        │
                [Step 5b] Honeypot Filter
                Profile consistency check
                Claimed exp vs career history
                        │
                [Step 6] Hybrid Score
                alpha*semantic + beta*BM25 + gamma*signal
                x JD_multiplier x consistency_penalty
                        │
                [Step 7] Top-100 -> Cross-Encoder
                ms-marco-MiniLM-L-6-v2
                Pair-wise re-ranking
                        │
                [Step 8] Final Score
                0.40 * hybrid + 0.60 * cross-encoder
                        │
                  team_SRJ_Prime.csv
          100 ranked candidates with reasoning
```

---

## The JD-Aware Multiplier

The Redrob JD explicitly warns:

> *"The right answer is NOT find candidates whose skills section contains the most AI keywords.  
> That is a trap we have explicitly built into the dataset."*

Instead of relying on the model to infer disqualifiers, we encode them directly as score multipliers
derived from the JD text — applied to every candidate before cross-encoder re-ranking.

### Penalties

| Condition | Multiplier | JD Source |
|-----------|:----------:|-----------|
| Junior / intern / entry-level title | **x0.50** | Founding team senior role |
| Research Scientist / AI Research Engineer | **x0.65** | *"Tried twice, didn't work"* |
| Computer Vision / Speech / Robotics title | **x0.60** | *"You'd be re-learning fundamentals"* |
| Recruiter response rate < 0.25 | **x0.55** | *"Not actually available — down-weight"* |
| Response rate 0.25 – 0.35 | **x0.70** | Low availability signal |
| Response rate 0.35 – 0.45 | **x0.85** | Below-average availability |
| Experience < 3.5 years | **x0.60** | Below founding team bar |
| Experience 3.5 – 5.0 years | **x0.82** | Marginal experience band |

### Boosts

| Condition | Multiplier | JD Source |
|-----------|:----------:|-----------|
| Senior ML / AI / SSE(ML) / Staff / Principal title | **x1.12** | Ideal profile match |
| Experience 5 – 9 years | **x1.08** | JD's stated sweet spot |
| 4+ production retrieval skills | **x1.10** | Core JD requirement |
| Response rate >= 0.75 | **x1.05** | High hire probability |

> Multipliers compound. A Senior ML Engineer (x1.12) with 7 yrs (x1.08) and 5 retrieval skills (x1.10)  
> and 80% response rate (x1.05) gets a combined boost of **x1.40** — capped at x1.20.

---

## Honeypot Protection

The spec warns of ~80 honeypot candidates with impossible profiles.  
Our system catches them through **profile consistency checks** — not special-casing:

| Check | What It Detects | Penalty |
|-------|----------------|---------|
| Claimed experience vs career history duration | 8 yrs claimed, 3 yrs of actual jobs | x0.50 |
| High skill count with zero endorsements + zero years used | Expert in 10 skills, 0 evidence | x0.55 |
| Many skills vs very short career | 10+ skills, < 2 years total career | x0.60 |

**Result:** 3,415 inconsistent profiles flagged, 21 hard-penalised.  
Zero honeypots expected in top 100.

---

## Scoring Weights

| Parameter | Value | Role |
|-----------|:-----:|------|
| `alpha` | 0.50 | Semantic (dense retrieval) weight |
| `beta` | 0.20 | BM25 lexical weight |
| `gamma` | 0.30 | Signal bonus weight |
| Final blend | 40/60 | Hybrid vs cross-encoder |
| `--top-k` | 100 | Candidates sent to cross-encoder |

Weight rationale:
- **alpha=0.50** — Semantic fit is the primary signal for JD match
- **beta=0.20** — Reduced from default to resist keyword gaming
- **gamma=0.30** — Elevated because availability and career signals matter per JD
- **60% cross-encoder** — More discriminative for fine-grained ranking in top-100

---

## Results

| Metric | Value |
|--------|-------|
| Runtime (cache loaded, 100K candidates, CPU only) | ~4 minutes |
| Spec compliance | 100% — all 13 validator checks pass |
| Junior / Research-only in top 20 | 0 |
| CV / Speech engineers in top 20 | 0 |
| LOW availability candidates in top 20 | 0 |
| Score spread (rank 1 to rank 100) | 0.545 |
| Unique reasoning strings | 100 / 100 |
| Candidates penalised by JD multiplier | 73,613 / 100,000 |
| Candidates boosted by JD multiplier | 19,387 / 100,000 |
| Inconsistent profiles flagged (honeypot filter) | 3,415 |

### Top 5 Candidates

| Rank | Candidate ID | Score | Profile |
|:----:|-------------|:-----:|---------|
| 1 | CAND_0049978 | 0.9653 | Senior SSE (ML), 6.6 yrs, response rate 71% |
| 2 | CAND_0067866 | 0.9031 | Senior SSE (ML), 6.4 yrs, response rate 79% |
| 3 | CAND_0068985 | 0.9023 | Senior SSE (ML), 5.2 yrs, response rate 88% |
| 4 | CAND_0059795 | 0.8958 | Senior SSE (ML), 5.6 yrs, response rate 75% |
| 5 | CAND_0000981 | 0.8824 | ML Engineer, 6.4 yrs, response rate 55% |

---

## Design Decisions

**Why hybrid retrieval (dense + BM25) instead of pure embeddings?**  
Dense retrieval captures semantic fit but misses exact production terms like "Pinecone", "NDCG", or "FAISS".
BM25 catches exact skill keywords but falls into the keyword-stuffing trap the JD warns about.
Together, they compensate for each other's weaknesses.

**Why a JD multiplier instead of relying on the model?**  
The JD names specific disqualifiers in plain English — junior titles, pure research roles, CV/Speech backgrounds.
Encoding these directly as multipliers is more reliable and interpretable than hoping embeddings infer them.
It also makes our reasoning strings grounded and defensible at Stage 5 review.

**Why cross-encoder only on top-100?**  
Cross-encoders are slow — O(N) inference per candidate pair. Running on 100K candidates would take hours.
Restricting to top-100 from hybrid scoring gives high-precision re-ranking exactly where it matters.

**Why 40/60 blend (hybrid / cross-encoder)?**  
The cross-encoder is more discriminative for fine-grained pair comparison.
Giving it 60% weight spreads scores meaningfully across ranks 2–10,
avoiding the flat clustering seen with equal weighting.

**Why profile consistency check instead of honeypot special-casing?**  
The spec explicitly says not to special-case. A consistency check — verifying claimed experience
matches career history, and that expert skills have evidence — is valid general-purpose quality
filtering that happens to catch honeypots as a side effect.

---

## Project Structure

```
.
├── rank.py                     # Main ranking script — full pipeline
├── download_models.py          # One-time model download (requires internet)
├── validate_submission.py      # Validates submission.csv against spec
├── requirements.txt            # Pinned Python dependencies
├── submission_metadata.yaml    # Team info, approach, compute environment
├── README.md                   # This file
├── .gitignore                  # Excludes dataset, cache, venv, model weights
│
├── candidates.jsonl            # NOT committed — download from hackathon portal
├── job_description.txt         # NOT committed — download from hackathon portal
│
├── cache/                      # Auto-created on first run
│   └── embedding_cache.pkl     # NOT committed — auto-generated (~2 GB)
│
└── team_SRJ_Prime.csv          # Final ranked output
```

---

## Dependencies

```
sentence-transformers==3.0.1
transformers==4.41.2
torch==2.2.2
scikit-learn==1.3.2
numpy==1.24.3
tqdm==4.66.1
pyyaml==6.0.1
rank-bm25==0.2.2
huggingface-hub==0.23.3
scipy==1.11.4
```

Install: `pip install -r requirements.txt`

Models used:
- `all-MiniLM-L6-v2` — embedding model (sentence-transformers)
- `cross-encoder/ms-marco-MiniLM-L-6-v2` — re-ranker (sentence-transformers)

Both are downloaded locally via `python download_models.py` and loaded offline during ranking.

---

## Compute Environment

| Property | Value |
|----------|-------|
| OS | Windows 11 |
| CPU | Intel, 8 cores |
| RAM | 16 GB |
| GPU | None — CPU only |
| Python | 3.11 |
| Network during ranking | None — `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1` |

---

## CLI Reference

```bash
python rank.py \
  --candidates ./candidates.jsonl \     # Path to candidates JSONL file
  --job-desc   ./job_description.txt \  # Path to job description
  --out       ./team_SRJ_Prime.csv \    # Output path
  --alpha      0.50 \                   # Semantic weight (default: 0.50)
  --beta       0.20 \                   # BM25 weight    (default: 0.20)
  --gamma      0.30 \                   # Signal weight  (default: 0.30)
  --top-k      100                      # Cross-encoder candidates (default: 100)
```

---

## AI Tools Declaration

   ChatGPT (OpenAI) was used for architecture discussion and debugging.
   Gemini (Google) was used for idea validation and comparing possible ranking approaches.
   GitHub Copilot was used for autocomplete during development.
   Claude (Anthropic) was used for code review, presentation artifacts.
   No candidate profile data was fed to any external LLM or API during development or ranking.
   All engineering decisions -- weight choices (alpha/beta/gamma), JD multiplier design,
   penalty thresholds, and cross-encoder blend ratio -- were made by the team and can be
   fully defended at the Stage 5 interview.
---



