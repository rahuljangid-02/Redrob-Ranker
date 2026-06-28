from sentence_transformers import SentenceTransformer, CrossEncoder

print("Downloading embedding model...")
SentenceTransformer("all-MiniLM-L6-v2")

print("Downloading cross-encoder model...")
CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

print("Models downloaded successfully.")