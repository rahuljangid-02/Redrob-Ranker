#!/usr/bin/env python3
"""
validate_submission.py
======================
Validates a submission CSV against the Redrob Hackathon submission spec.

Usage:
    python validate_submission.py ./team_SRJ_Prime.csv

Exit codes:
    0  -- Submission is valid
    1  -- Submission has errors (see output)
"""

import sys
import csv
import os


def validate(filepath):
    errors = []
    warnings = []

    # ── Check 1: File exists ─────────────────────────────────────
    if not os.path.exists(filepath):
        print(f"[ERROR] File not found: {filepath}")
        sys.exit(1)

    # ── Check 2: File extension ──────────────────────────────────
    if not filepath.lower().endswith(".csv"):
        errors.append(f"File must have .csv extension, got: {filepath}")

    # ── Check 3: Read the file ───────────────────────────────────
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            fieldnames = reader.fieldnames or []
    except UnicodeDecodeError:
        errors.append("File is not UTF-8 encoded. Re-save as UTF-8.")
        print_results(errors, warnings)
        return

    # ── Check 4: Required columns in correct order ───────────────
    required_cols = ["candidate_id", "rank", "score", "reasoning"]
    if fieldnames != required_cols:
        errors.append(
            f"Columns must be exactly {required_cols} in that order.\n"
            f"  Got: {fieldnames}"
        )

    # ── Check 5: Exactly 100 rows ────────────────────────────────
    if len(rows) != 100:
        errors.append(f"Must have exactly 100 data rows. Found: {len(rows)}")

    if not rows:
        print_results(errors, warnings)
        return

    # ── Check 6: Parse and validate each row ────────────────────
    candidate_ids = []
    ranks = []
    scores = []
    empty_reasoning = []
    parse_errors = []

    for i, row in enumerate(rows, start=2):  # row 1 = header
        # candidate_id
        cid = row.get("candidate_id", "").strip()
        if not cid:
            parse_errors.append(f"Row {i}: candidate_id is empty")
        else:
            candidate_ids.append(cid)

        # rank
        rank_raw = row.get("rank", "").strip()
        try:
            rank = int(rank_raw)
            ranks.append(rank)
        except ValueError:
            parse_errors.append(f"Row {i}: rank '{rank_raw}' is not an integer")

        # score
        score_raw = row.get("score", "").strip()
        try:
            score = float(score_raw)
            scores.append((rank if rank_raw.isdigit() else 9999, score))
            if not (0.0 <= score <= 1.0):
                parse_errors.append(f"Row {i}: score {score} is outside [0, 1]")
        except ValueError:
            parse_errors.append(f"Row {i}: score '{score_raw}' is not a float")

        # reasoning
        reasoning = row.get("reasoning", "").strip()
        if not reasoning:
            empty_reasoning.append(i)

    errors.extend(parse_errors)

    # ── Check 7: Ranks are exactly 1 to 100, each once ──────────
    if sorted(ranks) != list(range(1, 101)):
        missing = sorted(set(range(1, 101)) - set(ranks))
        dupes = sorted(r for r in ranks if ranks.count(r) > 1)
        if missing:
            errors.append(f"Missing ranks: {missing[:10]}{'...' if len(missing)>10 else ''}")
        if dupes:
            errors.append(f"Duplicate ranks: {list(set(dupes))[:10]}")

    # ── Check 8: No duplicate candidate_ids ─────────────────────
    seen = set()
    dupes = set()
    for cid in candidate_ids:
        if cid in seen:
            dupes.add(cid)
        seen.add(cid)
    if dupes:
        errors.append(f"Duplicate candidate_ids: {list(dupes)[:5]}")

    # ── Check 9: Scores monotonically non-increasing ─────────────
    scores_sorted = [s for _, s in sorted(scores, key=lambda x: x[0])]
    mono_violations = [
        (i + 1, scores_sorted[i], scores_sorted[i + 1])
        for i in range(len(scores_sorted) - 1)
        if scores_sorted[i] < scores_sorted[i + 1]
    ]
    if mono_violations:
        sample = mono_violations[:3]
        errors.append(
            f"Scores are not monotonically non-increasing. "
            f"Found {len(mono_violations)} violation(s). "
            f"First 3: {[(r, round(s1,4), round(s2,4)) for r,s1,s2 in sample]}"
        )

    # ── Check 10: All scores are the same (model not differentiating) ──
    unique_scores = len(set(round(s, 6) for _, s in scores))
    if unique_scores == 1:
        errors.append("All scores are identical -- model is not differentiating candidates.")
    elif unique_scores < 10:
        warnings.append(f"Only {unique_scores} unique score values -- low differentiation.")

    # ── Check 11: Empty reasoning strings ────────────────────────
    if empty_reasoning:
        warnings.append(
            f"Empty reasoning strings at rows: {empty_reasoning[:10]}"
            f"{'...' if len(empty_reasoning) > 10 else ''} "
            f"(optional but strongly recommended per spec)"
        )

    # ── Check 12: candidate_id format ────────────────────────────
    bad_format = [
        cid for cid in candidate_ids
        if not cid.startswith("CAND_") or len(cid) != 12
    ]
    if bad_format:
        errors.append(
            f"Unexpected candidate_id format (expected CAND_XXXXXXX): "
            f"{bad_format[:5]}"
        )

    print_results(errors, warnings, filepath, scores_sorted)


def print_results(errors, warnings, filepath="", scores=None):
    print()
    print("=" * 60)
    print("SUBMISSION VALIDATOR -- Redrob Hackathon")
    print("=" * 60)
    if filepath:
        print(f"File: {filepath}")
    print()

    if errors:
        print(f"[ERRORS] {len(errors)} error(s) found -- submission will be REJECTED:\n")
        for i, e in enumerate(errors, 1):
            print(f"  {i}. {e}")
        print()

    if warnings:
        print(f"[WARNINGS] {len(warnings)} warning(s) -- submission accepted but review recommended:\n")
        for i, w in enumerate(warnings, 1):
            print(f"  {i}. {w}")
        print()

    if scores:
        print("[Score Stats]")
        print(f"  Rank 1   : {scores[0]:.4f}")
        print(f"  Rank 10  : {scores[9]:.4f}")
        print(f"  Rank 50  : {scores[49]:.4f}")
        print(f"  Rank 100 : {scores[99]:.4f}")
        print(f"  Spread   : {scores[0] - scores[99]:.4f}")
        print()

    if not errors:
        print("[OK] Submission is valid.")
        print("     Ready to upload to the hackathon portal.")
    else:
        print("[ERROR] Submission is NOT valid. Fix the errors above and re-run.")

    print("=" * 60)
    print()

    sys.exit(0 if not errors else 1)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python validate_submission.py <path_to_team_SRJ_Prime.csv>")
        print("Example: python validate_submission.py ./team_SRJ_Prime.csv")
        sys.exit(1)

    validate(sys.argv[1])