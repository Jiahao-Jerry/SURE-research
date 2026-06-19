import os, json, time
import numpy as np
from bertopic import BERTopic
from bertopic.dimensionality import BaseDimensionalityReduction
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP
from hdbscan import HDBSCAN

SAMPLE_SIZE      = 500_000
INPUT_FILE       = "candidate_posts_eng10.jsonl"   # unzipped local file
INPUT_EMBEDDINGS = "embeddings.npy"
INPUT_POST_IDS   = "post_ids.npy"
OUTPUT_TOPICS    = "post_topics.jsonl"

t0 = time.time()

# ── 1. Load texts ────────────────────────────────────────────────
print("Loading texts...")
texts, pid_list = [], []
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    for line in f:
        d = json.loads(line)
        if "text" in d and "post_id" in d:
            texts.append(d["text"])
            pid_list.append(d["post_id"])
print(f"Loaded {len(texts):,} texts  ({time.time()-t0:.0f}s)")

# ── 2. Load embeddings ───────────────────────────────────────────
print("Loading embeddings...")
embeddings = np.load(INPUT_EMBEDDINGS, mmap_mode="r")  # memory-mapped, don't load all into RAM
post_ids   = np.load(INPUT_POST_IDS, allow_pickle=True)
print(f"Embeddings shape: {embeddings.shape}  ({time.time()-t0:.0f}s)")

# ── 3. Random 500K sample ────────────────────────────────────────
print(f"\nSampling {SAMPLE_SIZE:,} posts...")
np.random.seed(42)
sample_idx        = np.sort(np.random.choice(len(texts), SAMPLE_SIZE, replace=False))
sample_texts      = [texts[i] for i in sample_idx]
sample_embeddings = np.array(embeddings[sample_idx], dtype=np.float32)   # load only sample into RAM
print(f"Sample loaded: {sample_embeddings.shape}  ~{sample_embeddings.nbytes/1e9:.1f} GB  ({time.time()-t0:.0f}s)")

# ── 4. CPU UMAP on sample ────────────────────────────────────────
print("\nRunning CPU UMAP on 500K sample (this is the slow part ~20-35 min)...")
umap_model = UMAP(
    n_neighbors=15,
    n_components=5,
    min_dist=0.0,
    metric="cosine",
    random_state=42,
    low_memory=True,
    verbose=True
)
reduced_sample = umap_model.fit_transform(sample_embeddings)
print(f"UMAP done: {reduced_sample.shape}  ({time.time()-t0:.0f}s)")

# ── 5. HDBSCAN + BERTopic on sample ─────────────────────────────
print("\nClustering...")
hdbscan_model = HDBSCAN(
    min_samples=25,
    min_cluster_size=250,
    metric="euclidean",
    prediction_data=True,  # required for transform() on new batches
    core_dist_n_jobs=-1   # use all CPU cores
)

vectorizer_model = CountVectorizer(
    stop_words="english",
    min_df=50, max_df=0.9,
    max_features=200_000,
    ngram_range=(1, 2)
)

topic_model = BERTopic(
    umap_model=BaseDimensionalityReduction(),
    hdbscan_model=hdbscan_model,
    vectorizer_model=vectorizer_model,
    calculate_probabilities=False,
    verbose=True
)

sample_topics, _ = topic_model.fit_transform(sample_texts, reduced_sample)
print(f"\nDiscovered {len(topic_model.get_topic_info()) - 1} topics  ({time.time()-t0:.0f}s)")

# ── 6. Assign sample topics ──────────────────────────────────────
all_topics = np.full(len(texts), -1, dtype=np.int32)
all_topics[sample_idx] = sample_topics

# ── 7. Transform remaining in batches ───────────────────────────
remaining_idx = np.setdiff1d(np.arange(len(texts)), sample_idx)
print(f"\nTransforming {len(remaining_idx):,} remaining posts in batches...")
BATCH = 50_000

for start in range(0, len(remaining_idx), BATCH):
    batch_idx     = remaining_idx[start:start + BATCH]
    batch_emb     = np.array(embeddings[batch_idx], dtype=np.float32)
    batch_txt     = [texts[i] for i in batch_idx]
    batch_reduced = umap_model.transform(batch_emb)
    batch_topics, _ = topic_model.transform(batch_txt, batch_reduced)
    all_topics[batch_idx] = batch_topics
    done = min(start + BATCH, len(remaining_idx))
    print(f"  {done:,} / {len(remaining_idx):,}  ({time.time()-t0:.0f}s)")

# ── 8. Stats ─────────────────────────────────────────────────────
unique, counts = np.unique(all_topics, return_counts=True)
n_outliers = dict(zip(unique, counts)).get(-1, 0)
print(f"\nOutliers: {n_outliers:,} ({n_outliers/len(texts):.1%})")
print(f"Topics:   {sum(unique != -1)}")
print(topic_model.get_topic_info().head(20))

# ── 9. Save ──────────────────────────────────────────────────────
print(f"\nSaving {OUTPUT_TOPICS}...")
with open(OUTPUT_TOPICS, "w") as f:
    for pid, topic in zip(post_ids, all_topics):
        f.write(json.dumps({"post_id": str(pid), "topic": int(topic)}) + "\n")

topic_model.save("bertopic_model", serialization="safetensors", save_ctfidf=True)
topic_model.get_topic_info().to_csv("topic_info.csv", index=False)

print(f"\n✅ Done in {(time.time()-t0)/60:.1f} minutes")
print(f"   post_topics.jsonl  —  {len(texts):,} rows")
print(f"   topic_info.csv     —  {len(topic_model.get_topic_info())} topics")
print(f"   bertopic_model/    —  saved model")
