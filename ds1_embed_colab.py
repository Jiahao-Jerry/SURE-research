"""
Step 1 — BGE-M3 Embeddings (run in Google Colab, T4 GPU)

Upload to Colab:
  - candidate_posts_eng10.jsonl.gz

Download after running:
  - embeddings.npy   (2.2M × 1024 float32)
  - post_ids.npy     (2.2M post IDs, row-aligned to embeddings)

Runtime: ~30 min on T4
"""

# ── Cell 1: Install ───────────────────────────────────────────────
# !pip install -q sentence-transformers

# ── Cell 2: Embed ─────────────────────────────────────────────────
import json, gzip
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from google.colab import files

INPUT_FILE  = "candidate_posts_eng10.jsonl.gz"
MODEL_NAME  = "BAAI/bge-m3"
BATCH_SIZE  = 512

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
if device == "cpu":
    print("⚠️  Switch to T4 GPU: Runtime → Change runtime type → T4 GPU")

print(f"\nLoading model {MODEL_NAME}...")
model = SentenceTransformer(MODEL_NAME, device=device,
                            model_kwargs={"torch_dtype": torch.float16})
model.max_seq_length = 512

print(f"\nReading {INPUT_FILE}...")
texts, post_ids = [], []
with gzip.open(INPUT_FILE, "rt", encoding="utf-8") as f:
    for line in f:
        d = json.loads(line)
        if "text" in d and "post_id" in d:
            texts.append(d["text"])
            post_ids.append(d["post_id"])
print(f"Loaded {len(texts):,} posts")

print(f"\nEncoding in batches of {BATCH_SIZE}...")
embeddings = model.encode(
    texts,
    batch_size=BATCH_SIZE,
    show_progress_bar=True,
    convert_to_numpy=True,
).astype(np.float32)

np.save("embeddings.npy", embeddings)
np.save("post_ids.npy", np.array(post_ids))
print(f"\nSaved embeddings.npy {embeddings.shape} and post_ids.npy")

print("\nDownloading...")
files.download("embeddings.npy")
files.download("post_ids.npy")
print("✅ Done")
