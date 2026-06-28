import pandas as pd
import json

# -----------------------------
# File paths
# -----------------------------
TOP_100_CSV = "team_SRJ_Prime.csv"          # Your final top 100 CSV
FULL_CANDIDATES_JSONL = "candidates.jsonl"  # Full candidate data
OUTPUT_JSONL = "top_100_candidates.jsonl"   # Output file

# -----------------------------
# Step 1: Load top 100 CSV
# -----------------------------
top_df = pd.read_csv(TOP_100_CSV)

print("Top CSV columns:", top_df.columns.tolist())
print("Top CSV rows:", len(top_df))

# Automatically detect candidate ID column
possible_id_cols = ["candidate_id", "id", "ID", "Candidate_ID", "candidateId"]

top_id_col = None
for col in possible_id_cols:
    if col in top_df.columns:
        top_id_col = col
        break

if top_id_col is None:
    raise ValueError("No candidate ID column found in top 100 CSV.")

# Keep only first 100
top_df = top_df.head(100)

top_ids = top_df[top_id_col].astype(str).tolist()

print("Detected ID column in top CSV:", top_id_col)
print("Total top IDs:", len(top_ids))
print("Unique top IDs:", len(set(top_ids)))

# -----------------------------
# Step 2: Load full candidates JSONL
# -----------------------------
full_candidates = []

with open(FULL_CANDIDATES_JSONL, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            full_candidates.append(json.loads(line))

print("Full candidates loaded:", len(full_candidates))

# -----------------------------
# Step 3: Detect ID column in full data
# -----------------------------
sample_candidate = full_candidates[0]
print("Full candidate keys:", sample_candidate.keys())

full_id_col = None
for col in possible_id_cols:
    if col in sample_candidate:
        full_id_col = col
        break

if full_id_col is None:
    raise ValueError("No candidate ID column found in full candidate JSONL.")

print("Detected ID column in full JSONL:", full_id_col)

# -----------------------------
# Step 4: Create lookup dictionary
# -----------------------------
candidate_lookup = {
    str(candidate[full_id_col]): candidate
    for candidate in full_candidates
}

# -----------------------------
# Step 5: Select top 100 full profiles
# -----------------------------
top_100_full_profiles = []
missing_ids = []

for cid in top_ids:
    if cid in candidate_lookup:
        top_100_full_profiles.append(candidate_lookup[cid])
    else:
        missing_ids.append(cid)

print("Matched candidates:", len(top_100_full_profiles))
print("Missing candidates:", len(missing_ids))

if missing_ids:
    print("Missing IDs:")
    print(missing_ids[:10])

# -----------------------------
# Step 6: Save as JSONL
# -----------------------------
with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
    for candidate in top_100_full_profiles:
        f.write(json.dumps(candidate, ensure_ascii=False) + "\n")

print(f"JSONL file created successfully: {OUTPUT_JSONL}")
