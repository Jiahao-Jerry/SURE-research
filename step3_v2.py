import csv, json, re
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import time

t0 = time.time()

# ── Config ─────────────────────────────────────────────────────────
LABEL_FILES     = ["labeled_300.csv", "progress_194.csv", "progress_300.csv", "progress_40.csv"]
EMBEDDINGS_FILE = "embeddings.npy"
POST_IDS_FILE   = "post_ids.npy"
POSTS_FILE      = "candidate_posts_eng10.jsonl"
TOPICS_FILE     = "post_topics_gpu.jsonl"
OUTPUT_FILE     = "candidate_pool_500.jsonl"

TOP_N_PER_TOPIC = 500

SELECTED_TOPICS = {0, 10, 11, 12, 15, 20, 23, 33, 39, 42, 48, 49, 51, 52, 63, 78, 83}

TOPIC_NAMES = {
    0:  "Pets",
    10: "Music",
    11: "Books & Reading",
    12: "Climate & Energy",
    15: "Sports",
    20: "Birds & Nature",
    23: "Food & Cooking",
    33: "Gardening & Plants",
    39: "Movies & Film",
    42: "Autism & Communication",
    48: "Video Games",
    49: "Space & Astronomy",
    51: "Ancient History",
    52: "Fitness & Gym",
    63: "Politics",
    78: "Higher Education",
    83: "Economy & Jobs",
}

def has_link(text):
    return bool(re.search(
        r'https?://'                   # explicit URLs
        r'|www\.'                      # www. prefix
        r'|\b\S+\.(com|org|net|io|co|bandcamp|substack|patreon|github|app'
        r'|edu|gov|me|ly|uk|fm|tv|info|biz|store|shop|link|bio)\b',
        text, re.IGNORECASE
    ))

def is_english(text):
    try:
        from langdetect import detect
        return detect(text) == 'en'
    except:
        return False

def is_long_enough(text):
    return len(text.strip()) >= 200

# ── Step 1: Load all labels (deduplicate by post_id) ───────────────
print("Loading labels from all files...")
labeled = {}
for fname in LABEL_FILES:
    count = 0
    with open(fname) as f:
        for row in csv.DictReader(f):
            label = str(row.get('label', '')).strip()
            if label in ('0', '1'):
                labeled[str(row['post_id'])] = int(label)
                count += 1
    print(f"  {fname}: loaded")

pos = sum(1 for v in labeled.values() if v == 1)
neg = sum(1 for v in labeled.values() if v == 0)
print(f"  Total unique labeled: {len(labeled)} | pos={pos} neg={neg}")

# ── Step 2: Load post_ids & embeddings ─────────────────────────────
print("\nLoading post IDs and embeddings (mmap)...")
post_ids  = np.load(POST_IDS_FILE, allow_pickle=True).astype(str)
embeddings = np.load(EMBEDDINGS_FILE, mmap_mode='r')
pid_to_idx = {pid: i for i, pid in enumerate(post_ids)}
print(f"  {len(post_ids):,} total posts in embeddings")

# ── Step 3: Build training matrix ──────────────────────────────────
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
print(f"  Training matrix: {X_train.shape}  (missing embeddings: {missing})")

# ── Step 4: Train logistic regression ──────────────────────────────
print("\nTraining logistic regression...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)

clf = LogisticRegression(
    C=1.0,
    max_iter=1000,
    class_weight='balanced',
    random_state=42
)
clf.fit(X_scaled, y_train)

y_pred = clf.predict(X_scaled)
print("\nTraining performance:")
print(classification_report(y_train, y_pred, target_names=['low quality', 'high quality']))
print(f"  Done in {time.time()-t0:.0f}s")

# ── Step 5: Load topic assignments ─────────────────────────────────
print("\nLoading topic assignments...")
post_topic = {}
with open(TOPICS_FILE) as f:
    for line in f:
        d = json.loads(line)
        t = d['topic']
        if t in SELECTED_TOPICS:
            post_topic[str(d['post_id'])] = t
print(f"  {len(post_topic):,} posts in selected topics")

# ── Step 6: Load post texts ────────────────────────────────────────
print("\nLoading post texts...")
post_data = {}
with open(POSTS_FILE) as f:
    for line in f:
        d = json.loads(line)
        pid = str(d['post_id'])
        if pid in post_topic:
            post_data[pid] = d
print(f"  {len(post_data):,} post texts loaded")

# ── Step 7: Filter eligible posts ──────────────────────────────────
print("\nFiltering eligible posts...")
eligible_pids = []
for pid in post_topic:
    text = post_data.get(pid, {}).get('text', '')
    if has_link(text):            continue
    if not is_long_enough(text):  continue
    if not is_english(text):      continue
    if pid not in pid_to_idx:     continue
    eligible_pids.append(pid)

print(f"  Eligible after filters: {len(eligible_pids):,}")

# ── Step 8: Score all eligible posts ──────────────────────────────
print("\nScoring posts in batches...")
BATCH = 50_000
scores = {}

for start in range(0, len(eligible_pids), BATCH):
    batch_pids = eligible_pids[start:start+BATCH]
    batch_idx  = [pid_to_idx[p] for p in batch_pids]
    batch_emb  = np.array(embeddings[batch_idx], dtype=np.float32)
    batch_scaled = scaler.transform(batch_emb)
    probs = clf.predict_proba(batch_scaled)[:, 1]
    for pid, score in zip(batch_pids, probs):
        scores[pid] = float(score)
    print(f"  Scored {min(start+BATCH, len(eligible_pids)):,} / {len(eligible_pids):,}")

# ── Step 9: Select top 50 per topic ───────────────────────────────
print(f"\nSelecting top {TOP_N_PER_TOPIC} posts per topic...")

from collections import defaultdict
import heapq

topic_scores = defaultdict(list)
for pid, score in scores.items():
    t = post_topic[pid]
    topic_scores[t].append((score, pid))

final_posts = []
for t in sorted(SELECTED_TOPICS):
    top = heapq.nlargest(TOP_N_PER_TOPIC, topic_scores[t], key=lambda x: x[0])
    print(f"  Topic {t:3d} ({TOPIC_NAMES[t]}): {len(top)} posts | "
          f"top score={top[0][0]:.3f}  min score={top[-1][0]:.3f}")
    for score, pid in top:
        d = post_data[pid]
        final_posts.append({
            "post_id":          pid,
            "topic":            t,
            "topic_name":       TOPIC_NAMES[t],
            "quality_score":    round(score, 4),
            "text":             d.get('text', ''),
            "engagement":       d.get('like_count', d.get('engagement', 0)),
        })

# ── Step 10: Save ──────────────────────────────────────────────────
print(f"\nSaving {OUTPUT_FILE}...")
with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    for post in final_posts:
        f.write(json.dumps(post, ensure_ascii=False) + '\n')

score_arr = np.array([p['quality_score'] for p in final_posts])
print(f"\nDone in {(time.time()-t0)/60:.1f} min")
print(f"  Total posts saved: {len(final_posts)}")
print(f"  Score mean:        {score_arr.mean():.3f}")
print(f"  Score median:      {np.median(score_arr):.3f}")
print(f"  Score >0.7:        {(score_arr > 0.7).sum()}")
print(f"  Score >0.9:        {(score_arr > 0.9).sum()}")
