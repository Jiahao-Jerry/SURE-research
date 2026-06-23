"""
Logistic Regression Quality Scorer

Inputs:
  - all_labels.csv               (1,000 human labels)
  - embeddings.npy               (2.2M × 1024 BGE-M3 embeddings)
  - post_ids.npy                 (2.2M post IDs, row-aligned to embeddings)
  - post_topics.jsonl            (2.2M posts → BERTopic cluster ID, from ds2_cluster_topics.py)
  - candidate_posts_eng10.jsonl  (post texts)

Outputs:
  - probability_scores.jsonl      (post_id, topic_name, probability_score for all scored posts)
  - candidate_pool_500.jsonl     (top 500 per topic = 8,500 posts)
"""

import csv, json, time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from collections import defaultdict
import os
from config import load_config, get_cluster_to_topic, validate_config

t0 = time.time()

TOP_N_PER_TOPIC = 1000

cfg = load_config()
validate_config(cfg)
CLUSTER_TO_TOPIC = get_cluster_to_topic(cfg)
SELECTED_CLUSTERS = set(CLUSTER_TO_TOPIC.keys())

# ── Fast path: rebuild candidate pool from existing scores ────────
if os.path.exists("probability_scores.jsonl"):
    print("probability_scores.jsonl found — skipping LR training/scoring.")
    print("Loading existing scores...")
    scores     = {}
    post_topic = {}
    with open("probability_scores.jsonl") as f:
        for line in f:
            d = json.loads(line)
            pid = str(d["post_id"])
            scores[pid]     = d["probability_score"]
            post_topic[pid] = d["cluster_id"]
    print(f"  Loaded {len(scores):,} scores")

    print("Loading post texts...")
    eligible_pids = set(scores.keys())
    post_data = {}
    with open("candidate_posts_eng10.jsonl") as f:
        for line in f:
            d   = json.loads(line)
            pid = str(d["post_id"])
            if pid in eligible_pids:
                post_data[pid] = d
    print(f"  Loaded {len(post_data):,} post texts")

    print(f"\nBuilding candidate_pool_{TOP_N_PER_TOPIC}.jsonl (top {TOP_N_PER_TOPIC} per topic)...")
    by_topic = defaultdict(list)
    for pid, score in scores.items():
        topic_name = CLUSTER_TO_TOPIC[post_topic[pid]]
        by_topic[topic_name].append((pid, score))

    total = 0
    with open(f"candidate_pool_{TOP_N_PER_TOPIC}.jsonl", "w") as f:
        for topic_name, pid_scores in sorted(by_topic.items()):
            topN = sorted(pid_scores, key=lambda x: x[1], reverse=True)[:TOP_N_PER_TOPIC]
            for pid, score in topN:
                d = post_data.get(pid, {})
                f.write(json.dumps({
                    "post_id":    pid,
                    "topic_name": topic_name,
                    "cluster_id": post_topic[pid],
                    "text":       d.get("text", ""),
                    "engagement": d.get("engagement", 0),
                    "lr_score":   round(score, 4),
                }, ensure_ascii=False) + "\n")
            total += len(topN)
            print(f"  {topic_name:<30} {len(topN)} posts")

    print(f"\nTotal: {total} posts → candidate_pool_{TOP_N_PER_TOPIC}.jsonl")
    print(f"Done in {(time.time() - t0) / 60:.1f} minutes")
    exit()

# ── Step 1: Load labels ───────────────────────────────────────────
print("Loading labels...")
labeled = {}
with open("all_labels.csv") as f:
    for row in csv.DictReader(f):
        labeled[str(row["post_id"])] = int(row["label"])

pos = sum(v == 1 for v in labeled.values())
neg = sum(v == 0 for v in labeled.values())
print(f"  {len(labeled)} labels  |  pos={pos}  neg={neg}")

# ── Step 2: Load embeddings (memory-mapped) ───────────────────────
print("\nLoading embeddings (mmap)...")
post_ids   = np.load("post_ids.npy", allow_pickle=True).astype(str)
embeddings = np.load("embeddings.npy", mmap_mode="r")
pid_to_idx = {pid: i for i, pid in enumerate(post_ids)}
print(f"  {len(post_ids):,} posts in embeddings")

# ── Step 3: Build training matrix ────────────────────────────────
print("\nBuilding training matrix...")
X_rows, y_rows = [], []
missing = 0
for pid, label in labeled.items():
    idx = pid_to_idx.get(pid)
    if idx is None:
        missing += 1
        continue
    X_rows.append(embeddings[idx])
    y_rows.append(label)

X_train = np.array(X_rows, dtype=np.float32)
y_train = np.array(y_rows, dtype=np.int32)
print(f"  Training matrix: {X_train.shape}  (missing: {missing})")

# ── Step 4: Train logistic regression ────────────────────────────
print("\nTraining logistic regression...")
scaler  = StandardScaler()
X_scaled = scaler.fit_transform(X_train)

clf = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced", random_state=42)
clf.fit(X_scaled, y_train)

y_pred = clf.predict(X_scaled)
print("\nTraining performance:")
print(classification_report(y_train, y_pred, target_names=["low quality", "high quality"]))

# ── Step 5: Load topic assignments ────────────────────────────────
print("\nLoading topic assignments...")
post_topic = {}
with open("post_topics.jsonl") as f:
    for line in f:
        d = json.loads(line)
        cluster = d["topic"]
        if cluster in SELECTED_CLUSTERS:
            post_topic[str(d["post_id"])] = cluster
print(f"  {len(post_topic):,} posts in selected clusters")

# ── Step 6: Load post texts (for output) ──────────────────────────
print("\nLoading post texts...")
post_data = {}
with open("candidate_posts_eng10.jsonl") as f:
    for line in f:
        d = json.loads(line)
        pid = str(d["post_id"])
        if pid in post_topic:
            post_data[pid] = d
print(f"  {len(post_data):,} post texts loaded")

# ── Step 7: Score all posts in selected clusters ──────────────────
print("\nScoring posts...")
BATCH = 50_000
eligible_pids = [pid for pid in post_topic if pid in pid_to_idx]
print(f"  Eligible posts: {len(eligible_pids):,}")

scores = {}
for start in range(0, len(eligible_pids), BATCH):
    batch_pids = eligible_pids[start:start + BATCH]
    batch_idx  = [pid_to_idx[p] for p in batch_pids]
    batch_emb  = np.array(embeddings[batch_idx], dtype=np.float32)
    batch_scaled = scaler.transform(batch_emb)
    probs = clf.predict_proba(batch_scaled)[:, 1]
    for pid, score in zip(batch_pids, probs):
        scores[pid] = float(score)
    print(f"  Scored {min(start + BATCH, len(eligible_pids)):,} / {len(eligible_pids):,}")

# ── Step 8: Save probability_scores.jsonl ────────────────────────────
print("\nSaving probability_scores.jsonl...")
with open("probability_scores.jsonl", "w") as f:
    for pid, score in scores.items():
        topic_name = CLUSTER_TO_TOPIC[post_topic[pid]]
        f.write(json.dumps({
            "post_id":       pid,
            "topic_name":    topic_name,
            "cluster_id":    post_topic[pid],
            "probability_score": round(score, 4),
        }) + "\n")
print(f"  Saved {len(scores):,} scores")

# ── Step 9: Top N per topic → candidate_pool_N.jsonl ──────────────
print(f"\nBuilding candidate_pool_{TOP_N_PER_TOPIC}.jsonl (top {TOP_N_PER_TOPIC} per topic)...")
by_topic = defaultdict(list)
for pid, score in scores.items():
    topic_name = CLUSTER_TO_TOPIC[post_topic[pid]]
    by_topic[topic_name].append((pid, score))

total = 0
with open(f"candidate_pool_{TOP_N_PER_TOPIC}.jsonl", "w") as f:
    for topic_name, pid_scores in sorted(by_topic.items()):
        topN = sorted(pid_scores, key=lambda x: x[1], reverse=True)[:TOP_N_PER_TOPIC]
        for pid, score in topN:
            d = post_data.get(pid, {})
            f.write(json.dumps({
                "post_id":       pid,
                "topic_name":    topic_name,
                "cluster_id":    post_topic[pid],
                "text":          d.get("text", ""),
                "engagement":    d.get("engagement", 0),
                "lr_score":      round(score, 4),
            }, ensure_ascii=False) + "\n")
        total += len(topN)
        print(f"  {topic_name:<30} {len(topN)} posts")

print(f"\nTotal: {total} posts → candidate_pool_{TOP_N_PER_TOPIC}.jsonl")
print(f"\nDone in {(time.time() - t0) / 60:.1f} minutes")
