#!/usr/bin/env python3
"""
Hybrid Candidate Ranking System - JD-Aware Edition 
==============================================================
Combines dense retrieval (semantic embeddings), sparse retrieval (BM25),
cross-encoder re-ranking, signal bonus scoring, and JD-specific
penalty/boost logic tuned for the Redrob Senior AI Engineer JD.

Usage:
    python rank.py --candidates ./candidates.jsonl --job-desc ./job_description.txt --out ./team_SRJ_Prime.csv


"""

import os
import pickle
import re
import csv
import json
import sys
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from sentence_transformers import SentenceTransformer, util, CrossEncoder
from rank_bm25 import BM25Okapi

# ---------------------------------------------
# Constants
# ---------------------------------------------
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CACHE_VERSION   = "v4"   # bumped: Fix 1/2/3 applied, forces cache rebuild

# ---------------------------------------------
# JD-specific profile signals
# (tuned for: Senior AI Engineer, Redrob, founding team)
# ---------------------------------------------

# Titles that are a STRONG fit for this JD
IDEAL_TITLE_KEYWORDS = [
    "senior ml engineer",
    "senior machine learning engineer",
    "ml engineer",
    "machine learning engineer",
    "ai engineer",
    "senior ai engineer",
    "applied scientist",
    "applied ml",
    "senior software engineer ml",
    "senior software engineer (ml)",
    "nlp engineer",
    "search engineer",
    "ranking engineer",
    "recommendation engineer",
    # Fix 1: Staff/Principal/Lead titles - senior roles wrongly ranked low
    "staff machine learning engineer",
    "staff ml engineer",
    "staff ai engineer",
    "staff software engineer",
    "principal ml engineer",
    "principal machine learning engineer",
    "principal ai engineer",
    "lead ml engineer",
    "lead machine learning engineer",
    "lead ai engineer",
]

# Titles that the JD explicitly says are NOT a fit
DISQUALIFIED_TITLE_KEYWORDS = [
    "junior",
    "associate",
    "intern",
    "trainee",
    "entry level",
    "marketing",
    "sales",
    "business analyst",
    "product manager",
    "scrum master",
    "project manager",
    "hr ",
    "recruiter",
]

# Titles that are RESEARCH-only -> JD says "we've tried it twice, didn't work"
RESEARCH_ONLY_TITLE_KEYWORDS = [
    "research scientist",
    "ai research engineer",
    "research engineer",
    "research fellow",
    "phd researcher",
    "postdoc",
]

# Skills the JD explicitly values (production retrieval/ranking signals)
HIGH_VALUE_SKILLS = {
    "sentence-transformers", "sentence transformers",
    "embeddings", "dense retrieval", "hybrid search",
    "vector database", "vector search",
    "pinecone", "weaviate", "qdrant", "milvus", "faiss",
    "opensearch", "elasticsearch",
    "bm25", "sparse retrieval",
    "ndcg", "mrr", "map", "ranking evaluation",
    "learning to rank", "ltr",
    "fine-tuning", "fine tuning", "lora", "qlora", "peft",
    "rag", "retrieval augmented generation",
    "a/b testing", "online evaluation",
    "recommendation system", "search system",
    "pytorch", "tensorflow",
    "mlops", "production ml",
    "xgboost", "lightgbm",
    "transformers", "bert", "llm",
}

# Skills JD says NOT to over-value (keyword-trap skills)
LOW_VALUE_SKILLS = {
    "langchain", "llamaindex", "chatgpt", "prompt engineering",
    "computer vision", "image classification",
    "speech recognition", "tts", "robotics",
    "openai api", "gpt wrapper",
}


class HybridCandidateRanker:
    """
    JD-Aware Hybrid Ranking System.

    Pipeline:
    1. Dense retrieval   -> semantic similarity via sentence-transformers
    2. Sparse retrieval  -> BM25 lexical matching
    3. Signal bonus      -> career stability, platform activity, education
    4. JD penalties      -> title mismatch, experience band, availability
    5. Honeypot check    -> profile consistency (claimed exp vs career history)
    6. Hybrid score      -> weighted combination of 1+2+3, adjusted by 4+5
    7. Cross-encoder     -> pair-wise re-ranking of top-K
    8. Final blend       -> 40/60 hybrid + cross-encoder
    """

    def __init__(
        self,
        dense_model_name: str = "all-MiniLM-L6-v2",
        cross_encoder_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        alpha: float = 0.50,   # semantic weight
        beta: float  = 0.20,   # BM25 weight  (reduced: less keyword gaming)
        gamma: float = 0.30,   # signal weight (increased: availability matters)
        top_k_hybrid: int = 100,
    ):
        print("[1/4] Loading Dense Retrieval Model (Sentence Transformer)...")
        self.dense_model = SentenceTransformer(
            dense_model_name, local_files_only=True
        )

        print("[2/4] Loading Cross-Encoder Model...")
        self.cross_encoder = CrossEncoder(
            cross_encoder_model_name, local_files_only=True
        )

        print("[3/4] Initializing BM25 Vectorizer...")

        print("[4/4] Configuration loaded.")
        self.alpha        = alpha
        self.beta         = beta
        self.gamma        = gamma
        self.top_k_hybrid = top_k_hybrid

    # ---------------------------------------------
    # Text preparation
    # ---------------------------------------------

    def prepare_candidate_text(self, candidate: Dict[str, Any]) -> str:
        """Build a single searchable text from all candidate fields."""
        parts = []

        profile = candidate.get("profile", {})
        for field in ["headline", "summary", "current_title",
                      "current_company", "current_industry"]:
            val = profile.get(field)
            if val:
                parts.append(str(val))

        for job in candidate.get("career_history", []):
            for field in ["title", "company", "description"]:
                val = job.get(field)
                if val:
                    parts.append(str(val))

        for edu in candidate.get("education", []):
            for field in ["degree", "field_of_study", "institution"]:
                val = edu.get(field)
                if val:
                    parts.append(str(val))

        for skill in candidate.get("skills", []):
            name = skill.get("name")
            if name:
                # Repeat high-value skills so BM25 and embeddings weight them
                name_lower = name.lower()
                if any(hv in name_lower for hv in HIGH_VALUE_SKILLS):
                    parts.extend([name] * 4)
                elif any(lv in name_lower for lv in LOW_VALUE_SKILLS):
                    parts.append(name)   # only once - don't over-boost
                else:
                    parts.extend([name] * 2)

        for cert in candidate.get("certifications", []):
            val = cert.get("name")
            if val:
                parts.append(str(val))

        return " ".join(parts)

    # ---------------------------------------------
    # Embedding cache
    # ---------------------------------------------

    def load_or_create_embeddings(
        self,
        candidates: List[Dict],
        candidate_texts: List[str],
        embedding_file: str = "cache/embedding_cache.pkl",
    ) -> np.ndarray:
        os.makedirs("cache", exist_ok=True)
        embedding_dict = {}

        if os.path.exists(embedding_file):
            print("Loading embedding cache...")
            with open(embedding_file, "rb") as f:
                cache_data = pickle.load(f)
            if (cache_data.get("model") != EMBEDDING_MODEL or
                    cache_data.get("version") != CACHE_VERSION):
                print("Cache outdated - rebuilding...")
                os.remove(embedding_file)
            else:
                print("Valid cache found.")
                embedding_dict = cache_data["embeddings"]

        new_ids, new_texts = [], []
        for candidate, text in zip(candidates, candidate_texts):
            cid = candidate["candidate_id"]
            if cid not in embedding_dict:
                new_ids.append(cid)
                new_texts.append(text)

        print(f"Cached candidates : {len(embedding_dict)}")
        print(f"New candidates    : {len(new_ids)}")

        if new_ids:
            print("Generating embeddings for new candidates...")
            new_embeddings = self.dense_model.encode(
                new_texts,
                batch_size=64,
                show_progress_bar=True,
                convert_to_numpy=True,
            )
            for cid, emb in zip(new_ids, new_embeddings):
                embedding_dict[cid] = emb

            with open(embedding_file, "wb") as f:
                pickle.dump({
                    "model": EMBEDDING_MODEL,
                    "version": CACHE_VERSION,
                    "embeddings": embedding_dict,
                }, f)
            print("Cache updated.")

        return np.vstack([
            embedding_dict[c["candidate_id"]] for c in candidates
        ])

    # ---------------------------------------------
    # BM25 lexical scoring
    # ---------------------------------------------

    def compute_lexical_score(
        self, job_text: str, candidate_texts: List[str]
    ) -> List[float]:
        def tokenize(t):
            return re.findall(r"\w+", t.lower())

        bm25  = BM25Okapi([tokenize(t) for t in candidate_texts])
        scores = np.array(bm25.get_scores(tokenize(job_text)), dtype=float)

        if scores.max() > scores.min():
            scores = (scores - scores.min()) / (scores.max() - scores.min())
        else:
            scores = np.zeros_like(scores)

        return scores.tolist()

    # ---------------------------------------------
    # Signal bonus
    # ---------------------------------------------

    AI_SKILLS = {
        "machine learning", "deep learning", "nlp",
        "natural language processing", "llm", "llms",
        "large language model", "fine-tuning", "fine tuning",
        "lora", "qlora", "peft", "rag", "generative ai", "genai",
        "transformers", "bert", "gpt", "tensorflow", "pytorch", "keras",
        "scikit-learn", "sklearn", "xgboost", "lightgbm",
        "vector database", "milvus", "pinecone", "weaviate", "faiss",
        "mlops", "embeddings", "sentence-transformers",
        "recommendation system", "search system", "ranking",
        "retrieval", "bm25", "elasticsearch", "opensearch",
        "statistical modeling", "huggingface", "openai",
    }

    def count_ai_skills(self, candidate: Dict[str, Any]) -> int:
        count = 0
        for skill in candidate.get("skills", []):
            name = re.sub(r"[^a-z0-9+#. ]", " ",
                          skill.get("name", "").lower())
            name = re.sub(r"\s+", " ", name).strip()
            for ai_skill in self.AI_SKILLS:
                if re.search(r"\b" + re.escape(ai_skill) + r"\b", name):
                    count += 1
                    break
        return count

    def count_production_retrieval_skills(self, candidate: Dict[str, Any]) -> int:
        """Count skills that specifically signal production retrieval experience."""
        count = 0
        production_signals = {
            "pinecone", "weaviate", "qdrant", "milvus", "faiss",
            "opensearch", "elasticsearch", "vector database",
            "dense retrieval", "hybrid search", "bm25",
            "sentence-transformers", "embeddings",
            "ndcg", "mrr", "learning to rank", "a/b testing",
            "rag", "retrieval augmented generation",
        }
        for skill in candidate.get("skills", []):
            name = skill.get("name", "").lower().strip()
            for ps in production_signals:
                if ps in name:
                    count += 1
                    break
        return count

    def compute_signal_bonus(self, candidate: Dict[str, Any]) -> float:
        """
        Compute bonus from career signals.
        Max = 0.50 (will be further scaled by JD penalty/boost).
        """
        bonus = 0.0

        # --- Career stability ---
        career_history = candidate.get("career_history", [])
        if career_history:
            avg_duration = np.mean(
                [job.get("duration_months", 0) for job in career_history]
            )
            if avg_duration >= 36:
                bonus += 0.10
            elif avg_duration >= 24:
                bonus += 0.07
            elif avg_duration >= 18:
                bonus += 0.05

        # --- Current role tenure ---
        if career_history and career_history[0].get("is_current"):
            current_duration = career_history[0].get("duration_months", 0)
            if current_duration >= 24:
                bonus += 0.07
            elif current_duration >= 12:
                bonus += 0.04

        # --- Education ---
        education = candidate.get("education", [])
        if education:
            tier_bonus = {
                "tier_1": 0.05, "tier_2": 0.04,
                "tier_3": 0.02, "tier_4": 0.01, "unknown": 0.00,
            }
            bonus += tier_bonus.get(education[0].get("tier", "unknown"), 0.0)

        # --- Skills & endorsements ---
        skills = candidate.get("skills", [])
        total_endorsements = sum(s.get("endorsements", 0) for s in skills if s)
        bonus += min(0.08, total_endorsements * 0.01)

        ai_skill_count = self.count_ai_skills(candidate)
        if ai_skill_count >= 8:
            bonus += 0.05
        elif ai_skill_count >= 5:
            bonus += 0.03
        elif ai_skill_count >= 3:
            bonus += 0.01

        # --- Platform availability (weighted heavily per JD) ---
        signals      = candidate.get("redrob_signals", {})
        response_rate = signals.get("recruiter_response_rate", 0.0)

        if response_rate >= 0.75:
            bonus += 0.10
        elif response_rate >= 0.60:
            bonus += 0.07
        elif response_rate >= 0.45:
            bonus += 0.04
        elif response_rate >= 0.30:
            bonus += 0.01
        # < 0.30 -> no bonus at all

        return min(0.50, bonus)

    # ---------------------------------------------
    # JD-aware penalty / boost  <- NEW
    # ---------------------------------------------

    def compute_jd_multiplier(self, candidate: Dict[str, Any]) -> float:
        """
        Return a multiplier [0.40 ... 1.15] that reflects how well the
        candidate matches THIS specific JD's stated requirements and
        disqualifiers.

        > 1.0  -> boost  (ideal profile)
        < 1.0  -> penalty (JD disqualifier triggered)
        """
        profile = candidate.get("profile", {})
        title   = (profile.get("current_title") or "").lower().strip()
        years   = profile.get("years_of_experience", 0) or 0
        signals = candidate.get("redrob_signals", {})
        response_rate = signals.get("recruiter_response_rate", 0.0)

        multiplier = 1.0

        # -- Hard disqualifiers --------------------------------------
        # 1. Junior / non-engineering titles
        for kw in DISQUALIFIED_TITLE_KEYWORDS:
            if kw in title:
                multiplier *= 0.50
                break

        # 2. Pure research titles (JD says: "tried twice, didn't work")
        for kw in RESEARCH_ONLY_TITLE_KEYWORDS:
            if kw in title:
                multiplier *= 0.65
                break

        # 2b. Fix 2: CV/Speech/Robotics - JD explicitly says NOT a fit
        # "people whose primary expertise is CV, speech, or robotics
        #  without significant NLP/IR exposure - you'd be re-learning fundamentals"
        CV_SPEECH_TITLES = [
            "computer vision engineer",
            "cv engineer",
            "vision engineer",
            "image recognition engineer",
            "speech engineer",
            "speech recognition",
            "robotics engineer",
            "autonomous systems",
        ]
        for kw in CV_SPEECH_TITLES:
            if kw in title:
                multiplier *= 0.60
                break

        # 3. Availability - JD explicitly says down-weight unavailable candidates
        if response_rate < 0.25:
            multiplier *= 0.55
        elif response_rate < 0.35:
            multiplier *= 0.70
        elif response_rate < 0.45:
            multiplier *= 0.85

        # 4. Experience too low for founding-team senior role
        if years < 3.5:
            multiplier *= 0.60
        elif years < 5.0:
            multiplier *= 0.82

        # -- Positive boosts -----------------------------------------
        # 5. Ideal title match
        for kw in IDEAL_TITLE_KEYWORDS:
            if kw in title:
                multiplier *= 1.12
                break

        # 6. Sweet-spot experience band per JD (5-9 years)
        if 5.0 <= years <= 9.0:
            multiplier *= 1.08
        elif 9.0 < years <= 12.0:
            multiplier *= 1.02   # slightly over but still fine

        # 7. Production retrieval skills (what the JD ACTUALLY needs)
        prod_skills = self.count_production_retrieval_skills(candidate)
        if prod_skills >= 4:
            multiplier *= 1.10
        elif prod_skills >= 2:
            multiplier *= 1.05

        # 8. High availability = more likely to actually join
        if response_rate >= 0.75:
            multiplier *= 1.05

        # Cap multiplier range
        return max(0.40, min(1.20, multiplier))

    # ---------------------------------------------
    # Honeypot / profile consistency check
    # ---------------------------------------------

    def compute_honeypot_penalty(self, candidate: Dict[str, Any]) -> float:
        """
        Detect profiles with internally inconsistent signals.
        Returns a penalty multiplier: 1.0 = clean, 0.40 = likely honeypot.

        NOT special-casing honeypots -- checking profile CONSISTENCY.
        A real candidate's career history duration should match claimed experience.
        A real expert's skills should have endorsements or years_used > 0.

        Honeypot examples from spec:
          - 8 years experience at company founded 3 years ago
          - Expert proficiency in 10 skills with 0 years used
        Both fail consistency checks naturally.
        """
        penalty = 1.0
        profile       = candidate.get("profile", {})
        career_history = candidate.get("career_history", [])
        skills         = candidate.get("skills", [])

        # Check 1: claimed years vs actual career history duration
        claimed_years = profile.get("years_of_experience", 0) or 0
        if career_history and claimed_years > 0:
            actual_months = sum(
                job.get("duration_months", 0) for job in career_history
            )
            actual_years = actual_months / 12.0

            if actual_years > 0:
                ratio = claimed_years / (actual_years + 0.01)
                if ratio > 2.5:
                    penalty *= 0.50   # strong inconsistency - likely honeypot
                elif ratio > 1.8:
                    penalty *= 0.70   # moderate inconsistency

        # Check 2: many skills with 0 endorsements AND 0 years used
        if skills:
            zero_signal_skills = sum(
                1 for s in skills
                if s.get("endorsements", 0) == 0
                and s.get("years_used", 0) == 0
            )
            zero_ratio = zero_signal_skills / len(skills)
            if zero_ratio >= 0.90 and len(skills) >= 7:
                penalty *= 0.55   # expert in many skills but zero evidence

        # Check 3: high skill count but very short actual career
        if career_history:
            total_months = sum(j.get("duration_months", 0) for j in career_history)
            if len(skills) >= 10 and total_months < 24:
                penalty *= 0.60   # too many skills for too little experience

        return max(0.40, penalty)

    # ---------------------------------------------
    # Rich reasoning generation
    # ---------------------------------------------

    def generate_reasoning(
        self,
        candidate: Dict[str, Any],
        final_score: float,
        hybrid_score: float,
        ce_score: float,
        jd_multiplier: float,
    ) -> str:
        """
        Generate a 2-sentence reasoning that explains WHY this candidate
        ranks where they do - not just what metadata they have.
        """
        profile       = candidate.get("profile", {})
        title         = profile.get("current_title", "Unknown")
        years         = profile.get("years_of_experience", 0) or 0
        signals       = candidate.get("redrob_signals", {})
        response_rate = signals.get("recruiter_response_rate", 0.0)
        ai_skills     = self.count_ai_skills(candidate)
        prod_skills   = self.count_production_retrieval_skills(candidate)

        # Sentence 1: profile summary + key strengths
        title_lower = title.lower()
        cv_speech = ["computer vision engineer", "cv engineer", "vision engineer",
                     "speech engineer", "robotics engineer"]
        if any(kw in title_lower for kw in IDEAL_TITLE_KEYWORDS):
            fit_label = "strong title fit"
        elif any(kw in title_lower for kw in RESEARCH_ONLY_TITLE_KEYWORDS):
            fit_label = "research-focused profile (production fit uncertain)"
        elif any(kw in title_lower for kw in DISQUALIFIED_TITLE_KEYWORDS):
            fit_label = "junior profile - below seniority bar"
        elif any(kw in title_lower for kw in cv_speech):
            fit_label = "CV/Speech background - JD requires NLP/IR expertise"
        else:
            fit_label = "partial title fit"

        exp_note = ""
        if 5.0 <= years <= 9.0:
            exp_note = f"{years:.1f} yrs in ideal 5-9 yr band"
        elif years < 5.0:
            exp_note = f"{years:.1f} yrs (below 5-yr minimum)"
        else:
            exp_note = f"{years:.1f} yrs (above band, still eligible)"

        skills_note = ""
        if prod_skills >= 4:
            skills_note = f"{prod_skills} production retrieval/ranking skills"
        elif prod_skills >= 2:
            skills_note = f"{prod_skills} retrieval skills"
        else:
            skills_note = f"{ai_skills} AI core skills (retrieval signals limited)"

        sentence1 = (
            f"{title} - {fit_label}; {exp_note}; {skills_note}."
        )

        # Sentence 2: availability + scoring breakdown
        if response_rate >= 0.70:
            avail_note = f"high availability (response rate {response_rate:.0%})"
        elif response_rate >= 0.45:
            avail_note = f"moderate availability (response rate {response_rate:.0%})"
        else:
            avail_note = f"LOW availability (response rate {response_rate:.0%}) - ranked down"

        if jd_multiplier >= 1.10:
            jd_note = "boosted by JD fit"
        elif jd_multiplier >= 0.95:
            jd_note = "neutral JD alignment"
        elif jd_multiplier >= 0.80:
            jd_note = "mild JD mismatch"
        else:
            jd_note = "penalised by JD mismatch"

        sentence2 = (
            f"Semantic={hybrid_score:.3f}, CrossEncoder={ce_score:.3f}; "
            f"{avail_note}; {jd_note}."
        )

        return f"{sentence1} {sentence2}"

    # ---------------------------------------------
    # Main ranking pipeline
    # ---------------------------------------------

    def rank_candidates(
        self,
        job_description: str,
        candidates: List[Dict[str, Any]],
        top_n: int = 100,
    ) -> List[Tuple[str, float, str]]:

        print(f"\n{'='*70}")
        print("HYBRID RANKING PROCESS (JD-Aware Edition)")
        print(f"{'='*70}")

        # Step 1: Prepare texts
        print("\n[Step 1] Preparing candidate texts...")
        candidate_texts = [self.prepare_candidate_text(c) for c in candidates]

        # Step 2: Dense retrieval
        print("\n[Step 2] Dense Retrieval - Computing semantic embeddings...")
        job_embedding = self.dense_model.encode(
            job_description, convert_to_numpy=True
        )
        candidate_embeddings = self.load_or_create_embeddings(
            candidates, candidate_texts
        )
        semantic_scores = util.cos_sim(
            job_embedding, candidate_embeddings
        )[0].cpu().numpy()
        print(f"   Semantic scores: min={semantic_scores.min():.4f}, max={semantic_scores.max():.4f}")

        # Step 3: BM25
        print("\n[Step 3] Sparse Retrieval - Computing BM25 scores...")
        lexical_scores = self.compute_lexical_score(job_description, candidate_texts)
        print(f"   Lexical scores: min={min(lexical_scores):.4f}, max={max(lexical_scores):.4f}")

        # Step 4: Signal bonus
        print("\n[Step 4] Computing signal bonuses...")
        signal_bonuses = [self.compute_signal_bonus(c) for c in candidates]
        print(f"   Signal bonuses: min={min(signal_bonuses):.4f}, max={max(signal_bonuses):.4f}")

        # Step 5: JD multipliers  <- NEW
        print("\n[Step 5] Computing JD-aware penalty/boost multipliers...")
        jd_multipliers = [self.compute_jd_multiplier(c) for c in candidates]
        print(f"   JD multipliers: min={min(jd_multipliers):.4f}, max={max(jd_multipliers):.4f}")
        boosted   = sum(1 for m in jd_multipliers if m > 1.0)
        penalised = sum(1 for m in jd_multipliers if m < 0.90)
        print(f"   Boosted: {boosted} candidates | Penalised: {penalised} candidates")

        # Step 5b: Honeypot / profile consistency check
        print("\n[Step 5b] Computing profile consistency penalties (honeypot filter)...")
        honeypot_penalties = [self.compute_honeypot_penalty(c) for c in candidates]
        flagged = sum(1 for p in honeypot_penalties if p < 0.90)
        hard_flagged = sum(1 for p in honeypot_penalties if p <= 0.50)
        print(f"   Inconsistent profiles flagged: {flagged} (hard: {hard_flagged})")

        # Step 6: Hybrid score
        print("\n[Step 6] Computing hybrid scores...")
        raw_hybrid = np.array([
            (self.alpha * sem + self.beta * lex + self.gamma * sig)
            * jd_mult * hp_penalty
            for sem, lex, sig, jd_mult, hp_penalty in zip(
                semantic_scores, lexical_scores, signal_bonuses,
                jd_multipliers, honeypot_penalties
            )
        ])

        # Normalize to [0, 1]
        if raw_hybrid.max() > raw_hybrid.min():
            hybrid_scores = (raw_hybrid - raw_hybrid.min()) / (
                raw_hybrid.max() - raw_hybrid.min()
            )
        else:
            hybrid_scores = np.zeros_like(raw_hybrid)

        print(f"   Hybrid scores: min={hybrid_scores.min():.4f}, max={hybrid_scores.max():.4f}")

        # Step 7: Select top-K for cross-encoder
        print(f"\n[Step 7] Selecting top {self.top_k_hybrid} candidates for cross-encoder...")
        top_k_indices = np.argsort(hybrid_scores)[-self.top_k_hybrid:][::-1]

        # Step 8: Cross-encoder re-ranking
        print(f"\n[Step 8] Cross-Encoder re-ranking top {len(top_k_indices)} candidates...")
        top_candidates = [candidates[i] for i in top_k_indices]
        top_texts      = [candidate_texts[i] for i in top_k_indices]
        top_hybrid     = hybrid_scores[top_k_indices]
        top_jd_mult    = [jd_multipliers[i] for i in top_k_indices]

        pairs = [[job_description, t] for t in top_texts]
        ce_scores_raw = self.cross_encoder.predict(pairs)

        ce_min, ce_max = ce_scores_raw.min(), ce_scores_raw.max()
        if ce_max > ce_min:
            ce_scores = (ce_scores_raw - ce_min) / (ce_max - ce_min)
        else:
            ce_scores = np.zeros_like(ce_scores_raw)
            
        ce_scores = np.clip(ce_scores, 0.10, 1.0)    

        print(f"   Cross-encoder scores: min={ce_scores.min():.4f}, max={ce_scores.max():.4f}")

        # Step 9: Final score + reasoning
        print("\n[Step 9] Computing final scores and generating reasoning...")
        final_results = []
        for candidate, h_score, ce_score, jd_mult in zip(
            top_candidates, top_hybrid, ce_scores, top_jd_mult
        ):
            # Fix 3: 40/60 blend - cross-encoder is more reliable for fine-grained ranking
            # This spreads scores more meaningfully across ranks 2-10
            final_score = 0.40 * h_score + 0.60 * ce_score

            reasoning = self.generate_reasoning(
                candidate, final_score, h_score, float(ce_score), jd_mult
            )

            final_results.append(
                (candidate["candidate_id"], final_score, reasoning)
            )

        # Sort descending
        final_results.sort(key=lambda x: x[1], reverse=True)
        final_results = final_results[:top_n]

        print(f"\n[OK] Ranking complete. Top scores: {[f'{s:.4f}' for _, s, _ in final_results[:5]]}")
        print(f"{'='*70}\n")

        return final_results

    # ---------------------------------------------
    # Save CSV
    # ---------------------------------------------

    def save_submission(
        self,
        results: List[Tuple[str, float, str]],
        output_path: str,
    ) -> None:
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["candidate_id", "rank", "score", "reasoning"])
            for rank, (cid, score, reasoning) in enumerate(results, start=1):
                writer.writerow([cid, rank, f"{score:.4f}", reasoning])
        print(f"[OK] Submission saved to {output_path}")


# ---------------------------------------------
# Data loaders
# ---------------------------------------------

def load_candidates(jsonl_path: str) -> List[Dict[str, Any]]:
    candidates = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                candidates.append(json.loads(line))
    return candidates


def load_job_description(desc_path: str) -> str:
    with open(desc_path, "r", encoding="utf-8") as f:
        return f.read().strip()


# ---------------------------------------------
# Entry point
# ---------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Hybrid Candidate Ranking System - JD-Aware Edition"
    )
    parser.add_argument("--candidates", type=str, required=True,
                        help="Path to candidates.jsonl")
    parser.add_argument("--job-desc",   type=str, default="job_description.txt",
                        help="Path to job description file")
    parser.add_argument("--out",        type=str, default="team_SRJ_Prime.csv",
                        help="Output CSV path")
    parser.add_argument("--alpha",  type=float, default=0.50,
                        help="Semantic weight (default: 0.50)")
    parser.add_argument("--beta",   type=float, default=0.20,
                        help="BM25 weight (default: 0.20)")
    parser.add_argument("--gamma",  type=float, default=0.30,
                        help="Signal weight (default: 0.30)")
    parser.add_argument("--top-k",  type=int,   default=100,
                        help="Top-K for cross-encoder (default: 100)")
    args = parser.parse_args()

    # Validate paths
    if not Path(args.candidates).exists():
        print(f"Error: Candidates file not found: {args.candidates}")
        sys.exit(1)
    if not Path(args.job_desc).exists():
        print(f"Error: Job description file not found: {args.job_desc}")
        sys.exit(1)

    # Load data
    print("[Loading Data]")
    candidates      = load_candidates(args.candidates)
    job_description = load_job_description(args.job_desc)

    candidate_ids = [c["candidate_id"] for c in candidates]
    print(f"Total candidates : {len(candidate_ids)}")
    print(f"Unique IDs       : {len(set(candidate_ids))}")

    if len(candidate_ids) != len(set(candidate_ids)):
        from collections import Counter
        dupes = [cid for cid, n in Counter(candidate_ids).items() if n > 1]
        print(f"ERROR: Duplicate candidate_ids found: {dupes[:10]}")
        sys.exit(1)

    print(f"[OK] All candidate IDs are unique")
    print(f"Job description length: {len(job_description)} characters\n")

    # Run ranker
    ranker = HybridCandidateRanker(
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
        top_k_hybrid=args.top_k,
    )

    results = ranker.rank_candidates(job_description, candidates, top_n=100)
    ranker.save_submission(results, args.out)

    print(f"\n[Submission Preview]")
    print("Top 10 candidates:")
    for cid, score, reasoning in results[:10]:
        print(f"  {cid}: {score:.4f} - {reasoning}")


if __name__ == "__main__":
    main()