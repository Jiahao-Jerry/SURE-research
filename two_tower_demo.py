import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.feature_extraction.text import TfidfVectorizer

# ── Config  
DATA_FILE    = "100_posts.jsonl"
TFIDF_DIM    = 150      # TF-IDF vocabulary size
EMBED_DIM    = 128      # shared embedding space dimension
BATCH_SIZE   = 32
EPOCHS       = 25
LR           = 1e-3
N_USERS      = 500      # synthetic users for training
TEMPERATURE  = 0.07     # InfoNCE temperature

CATEGORIES = [
    "Technology & Innovation",
    "Science & Nature",
    "Health & Medicine",
    "Business & Economics",
    "Environment & Sustainability",
    "Psychology & Behavior",
    "Global Issues & Politics",
    "Society & Culture",
]
N_CATS = len(CATEGORIES)


# Load posts and building item features
def load_posts(path):
    posts = []
    with open(path) as f:
        for line in f:
            posts.append(json.loads(line))
    return posts


def build_item_features(posts):
    texts = [p["text"] for p in posts]
    cats  = [p["category"] for p in posts]

    tfidf = TfidfVectorizer(max_features=TFIDF_DIM, sublinear_tf=True) # TF-IDF (Ter Frequency Inverse Document Frequency)
    tfidf_mat = tfidf.fit_transform(texts).toarray().astype(np.float32) # 3000 by 200 matrix

    cat_ids = [CATEGORIES.index(c) if c in CATEGORIES else 0 for c in cats]
    cat_onehot = np.zeros((len(posts), N_CATS), dtype=np.float32)
    for i, cid in enumerate(cat_ids):
        cat_onehot[i, cid] = 1.0

    features = np.concatenate([tfidf_mat, cat_onehot], axis=1)
    return features, cat_ids


# ── Synthetic Users 
def make_user_prefs(n_users):
    return np.random.dirichlet(np.ones(N_CATS) * 0.4, size=n_users).astype(np.float32)


def make_pairs(user_prefs, cat_ids, pairs_per_user=3):
    """Multiple (user, item) positive pairs per user, sampled proportional to preference."""
    cat_to_items = {c: [] for c in range(N_CATS)}
    for idx, c in enumerate(cat_ids):
        cat_to_items[c].append(idx)

    pairs = []
    for uid, prefs in enumerate(user_prefs):
        for _ in range(pairs_per_user):
            for _ in range(10):                      # retry if category is empty
                c = np.random.choice(N_CATS, p=prefs)
                if cat_to_items[c]:
                    pairs.append((uid, random.choice(cat_to_items[c])))
                    break
    return pairs


# ── Model
class ItemTower(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), #256 numbers
            nn.ReLU(), #negative zeroed
            nn.Dropout(0.1), #drop 10%
            nn.Linear(256, EMBED_DIM), #compress down to 64
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


class UserTower(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(N_CATS, 64), #64 numbers
            nn.ReLU(),
            nn.Linear(64, EMBED_DIM),
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


# ── Loss 
def infonce_loss(u_emb, i_emb):
    """
    Symmetric InfoNCE: each row i is a positive pair, all others are negatives.
    Diagonal of the logit matrix = positive scores.
    """
    logits = torch.matmul(u_emb, i_emb.T) / TEMPERATURE   # [B, B]
    labels = torch.arange(len(u_emb), device=u_emb.device)
    loss_u = F.cross_entropy(logits,   labels)
    loss_i = F.cross_entropy(logits.T, labels)
    return (loss_u + loss_i) / 2


# ── Training
def train(item_feats, cat_ids):
    item_tower = ItemTower(item_feats.shape[1])
    user_tower = UserTower()
    opt = torch.optim.Adam(
        list(item_tower.parameters()) + list(user_tower.parameters()), lr=LR
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    item_tensor  = torch.tensor(item_feats)
    user_prefs   = make_user_prefs(N_USERS)
    user_tensor  = torch.tensor(user_prefs)

    print(f"Training  |  items={len(cat_ids)}  users={N_USERS}  "
          f"embed_dim={EMBED_DIM}  epochs={EPOCHS}")
    print(f"{'Epoch':>6}  {'Loss':>8}")
    print("─" * 18)

    for epoch in range(EPOCHS):
        pairs = make_pairs(user_prefs, cat_ids)
        random.shuffle(pairs)

        epoch_loss, n_batches = 0.0, 0

        for start in range(0, len(pairs) - BATCH_SIZE, BATCH_SIZE):
            batch   = pairs[start : start + BATCH_SIZE]
            u_idx   = [p[0] for p in batch]
            i_idx   = [p[1] for p in batch]

            u_emb = user_tower(user_tensor[u_idx])
            i_emb = item_tower(item_tensor[i_idx])

            loss = infonce_loss(u_emb, i_emb)
            opt.zero_grad()
            loss.backward()
            opt.step()

            epoch_loss += loss.item()
            n_batches  += 1

        scheduler.step()

        if (epoch + 1) % 5 == 0:
            print(f"{epoch+1:>6}  {epoch_loss/n_batches:>8.4f}")

    return item_tower, user_tower


# ── Index & Retrieval
def build_index(item_tower, item_feats):
    """Pre-compute all item embeddings once for fast retrieval."""
    with torch.no_grad():
        embeddings = item_tower(torch.tensor(item_feats)).numpy()
    return embeddings


def retrieve(pref_dict, user_tower, item_index, posts, n=3, candidate_pool=20):
    """
    pref_dict: {category_name: weight, ...}  — weights need not sum to 1.

    Steps:
      1. Compute user embedding from preference weights.
      2. Score all items, take top `candidate_pool` aligned candidates.
      3. Randomly sample `n` from that pool — so results vary each call
         while still being preference-aligned.
    """
    pref_vec = np.zeros(N_CATS, dtype=np.float32)
    for cat, w in pref_dict.items():
        if cat in CATEGORIES:
            pref_vec[CATEGORIES.index(cat)] = w
    if pref_vec.sum() > 0:
        pref_vec /= pref_vec.sum()

    with torch.no_grad():
        u_emb = user_tower(torch.tensor(pref_vec[None])).numpy()[0]

    scores     = item_index @ u_emb
    candidates = np.argsort(scores)[::-1][:candidate_pool]
    chosen     = random.sample(list(candidates), min(n, len(candidates)))

    return [
        {"category": posts[idx]["category"],
         "text": posts[idx]["text"],
         "score": float(scores[idx])}
        for idx in chosen
    ]


# ── Demo  ──
DEMO_USERS = [
    {
        "name": "Science & Tech Enthusiast",
        "prefs": {"Science & Nature": 0.5, "Technology & Innovation": 0.4,
                  "Health & Medicine": 0.1},
    },
    {
        "name": "Policy & Economics Reader",
        "prefs": {"Business & Economics": 0.45, "Global Issues & Politics": 0.4,
                  "Society & Culture": 0.15},
    }
]


def run_demo(user_tower, item_index, posts):
    print("\n" + "=" * 65)
    print("  TWO-TOWER RETRIEVAL DEMO")
    print("=" * 65)

    for user in DEMO_USERS:
        recs = retrieve(user["prefs"], user_tower, item_index, posts, n=3)

        print(f"\n▶  User: {user['name']}")
        print(f"   Preferences: {user['prefs']}")
        print(f"   3 Recommended Posts:\n")

        for i, r in enumerate(recs, 1):
            print(f"   ── Recommendation {i} ──────────────────────────────────")
            print(f"   Category : {r['category']}")
            print(f"   Score    : {r['score']:.3f}")
            print(f"   Content  :")
            print(f"   {r['text']}")
            print()

    print("=" * 65)


# ── Main 
if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    print("Loading posts...")
    posts = load_posts(DATA_FILE)
    print(f"  {len(posts)} posts loaded\n")

    print("Extracting item features (TF-IDF + category one-hot)...")
    item_feats, cat_ids = build_item_features(posts)
    print(f"  Feature dim: {item_feats.shape[1]}  "
          f"({TFIDF_DIM} TF-IDF + {N_CATS} category)\n")

    item_tower, user_tower = train(item_feats, cat_ids)

    print("\nBuilding item index (pre-computing embeddings)...")
    item_index = build_index(item_tower, item_feats)
    print(f"  Index shape: {item_index.shape}")

    run_demo(user_tower, item_index, posts)

    torch.save({
        "item_tower": item_tower.state_dict(),
        "user_tower": user_tower.state_dict(),
        "item_embeddings": item_index,
        "categories": CATEGORIES,
    }, "two_tower_model.pt")
    print("Model saved → two_tower_model.pt")
