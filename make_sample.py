import json
from pathlib import Path

# Input: full extracted dataset
INPUT_FILE = "candidates.jsonl"

# Output: small sample for Colab/GitHub demo
OUTPUT_FILE = "data/sample_candidates_100.jsonl"

SAMPLE_SIZE = 100

Path("data").mkdir(exist_ok=True)

count = 0

with open(INPUT_FILE, "r", encoding="utf-8") as infile, open(
    OUTPUT_FILE, "w", encoding="utf-8"
) as outfile:
    for line in infile:
        if line.strip():
            # Validate that each line is proper JSON
            candidate = json.loads(line)

            # Write back as JSONL
            outfile.write(json.dumps(candidate, ensure_ascii=False) + "\n")

            count += 1

        if count >= SAMPLE_SIZE:
            break

print(f"Saved {count} candidates to {OUTPUT_FILE}")

if count < SAMPLE_SIZE:
    print("Warning: Less than 100 candidates found in input file.")
