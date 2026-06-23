# SURE-Research

Two pipelines for building and analyzing a curated Bluesky post dataset:

- **Dataset pipeline** — filters 2.2M raw posts down to 2,550 high-quality, topic-diverse posts annotated with 7 style dimensions
- **SAE pipeline** — trains a Sparse Autoencoder on style-shift vectors to discover interpretable writing-style features

All topics and style axes are defined in `pipeline_config.yaml`. Adding or changing a topic or axis requires editing that file only — no code changes.

---

## Repository Structure

```
SURE-Research/
│
├── pipeline_config.yaml       # All topics and axes defined here
├── config.py                  # Loads pipeline_config.yaml; used by all scripts
│
├── Dataset Pipeline
│   ├── ds1_embed_colab.py     # [Colab T4]  BGE-M3 embeddings of 2.2M posts
│   ├── ds2_cluster_topics.py  # [Local]     BERTopic clustering → 17 topics
│   ├── ds3_lr_score.py        # [Local]     Logistic regression quality scoring
│   ├── ds4_agent_qc.py        # [Local]     Claude Haiku quality ranking
│   ├── ds5_label_axes.py      # [Local]     Claude Haiku style axis annotation
│   ├── all_labels.csv         # 1,000 human-labeled posts for LR training
│   └── 2550_posts.jsonl       # Final dataset — 2,550 posts with style axes
│
└── SAE Pipeline
    ├── sae1_gen_rewrites.py   # [Local]     Generate 7-axis rewrites via Claude Haiku
    ├── sae2_embed_colab.py    # [Colab T4]  Extract Qwen2.5-3B layer-23 diff vectors
    ├── sae3_train.py          # [Local]     Train Sparse Autoencoder on diff vectors
    └── sae4_labeler.py        # [Local]     Generate interactive HTML feature viewer
```

---

## Dataset Pipeline

Filters 2.2M raw Bluesky posts down to 2,550 curated posts with style annotations.

```
candidate_posts_eng10.jsonl.gz  (~2.2M pre-filtered Bluesky posts)
        │
        ▼  ds1_embed_colab.py       [Colab T4, ~2h]
        BGE-M3 embeddings (1024-dim)
        → embeddings.npy, post_ids.npy
        │
        ▼  ds2_cluster_topics.py    [Local, ~30min]
        BERTopic → 362 clusters → 17 hand-picked topics
        → post_topics.jsonl
        │
        ▼  ds3_lr_score.py          [Local, ~5min]
        Logistic regression trained on 1,000 human labels
        Scores all posts in 17 topics, keeps top 1,000 per topic
        → probability_scores.jsonl, candidate_pool_1000.jsonl
        │
        ▼  ds4_agent_qc.py          [Local, ~45min]
        Claude Haiku scores each post 0.0–1.0 on quality
        Keeps top 150 per topic
        → rankings.jsonl, 2550_posts.jsonl
        │
        ▼  ds5_label_axes.py        [Local, ~20min]
        Claude Haiku annotates each post with 7 style axes
        → 2550_posts.jsonl (updated in-place with axes_json)
```

### 17 Topics

Ancient History, Birds & Nature, Books & Reading, Capitalism, Climate & Energy,
Economy & Jobs, Fitness & Gym, Food & Cooking, Gardening & Plants, Higher Education,
Movies & Film, Music, Pets, Politics, Space & Astronomy, Sports, Video Games

### 7 Style Axes (`axes_json` in `2550_posts.jsonl`)

| Axis | 0 → 1 |
|---|---|
| `reading_level` | simple vocabulary → academic / complex |
| `background` | no prior knowledge assumed → expert knowledge assumed |
| `abstract_concrete` | vague general claims → specific facts / numbers |
| `tone` | analytical / neutral → emotional / charged |
| `humor` | earnest → witty / humorous |
| `narrativity` | pure argument → story / anecdote |
| `grounding` | direct statement → analogy / example-driven |

### Quality Rubric (used in `ds4_agent_qc.py`)

**High quality (0.7–1.0):** shares specific knowledge, insight, or a well-reasoned opinion; uses concrete facts, examples, or real observations.

**Low quality (0.0–0.3):** personal feelings with no substance; list format; low-effort commentary.

---

## SAE Pipeline

Trains a Sparse Autoencoder on style-shift vectors to find interpretable style features.

```
2550_posts.jsonl
        │
        ▼  sae1_gen_rewrites.py     [Local, ~25min + API cost ~$2]
        Claude Haiku rewrites each post on each axis (up and down)
        → sae_clean_rewrites.json  (N posts × 7 axes)
        │
        ▼  sae2_embed_colab.py      [Colab T4, ~30min]
        Load Qwen2.5-3B, extract layer-23 activations via forward hook
        Compute diff_vector = emb(rewrite) − emb(original)
        Sign-normalize + L2-normalize
        → diff_vectors_N.npy, axis_labels_N.npy, directions_N.npy, post_ids_N.npy
        │
        ▼  sae3_train.py            [Local, ~5min]
        Train SAE: encoder 2048→64 (ReLU), decoder 64→2048
        L1 sparsity penalty (coef=0.05), 600 epochs
        Compute lift per feature per axis to categorize features
        → sae_qwen_model.pt, sae_qwen_results.png
        │
        ▼  sae4_labeler.py          [Local, ~1min]
        Load SAE model, compute activations, build interactive HTML viewer
        → sae_feature_labeler.html  (open in browser — no server needed)
```

### Key Design Choices

| Choice | Reason |
|---|---|
| Qwen2.5-3B layer 23 | Mid-to-late layer where stylistic signal is strongest; linear probe hits 91% accuracy on sign-normalized diff vectors |
| Sign-normalize "down" vectors | Flips direction so all vectors point "increase axis"; prevents up/down pairs from cancelling inside the SAE |
| L2-normalize all vectors | Removes magnitude differences so the SAE learns directions, not scales |
| L1 coef = 0.05 | Sparse enough to specialize features, not so aggressive that features die (0.10 killed 41/64 features) |
| 64 features | Enough to cover 7 axes with room for partial-overlap and novel candidates |

### SAE Feature Categories

| Category | Meaning |
|---|---|
| `confirms_axis` | lift ≥ 0.30 — feature strongly tracks one style axis |
| `partial_overlap` | lift 0.15–0.30 — feature weakly tracks one axis |
| `novel_candidate` | lift < 0.15 — feature captures something beyond the 7 axes |
| `dead` | never activates — killed by L1 penalty |

---

## Configuration

All topics and axes live in `pipeline_config.yaml`. Every script reads from it via `config.py`.

### Adding a new topic

1. Find the BERTopic cluster ID in `topic_info.csv` by inspecting top keywords
2. Add one entry to `pipeline_config.yaml`:

```yaml
topics:
  - cluster_id: 88
    name: "Philosophy"
```

3. Delete `probability_scores.jsonl` and re-run from `ds3_lr_score.py` downward

### Adding a new style axis

Add one entry to `pipeline_config.yaml`:

```yaml
axes:
  - name: "formality"
    short: "formal"
    color: "#aabbcc"
    label_description: "0=casual and conversational, 1=formal register"
    rewrite_up: "more formal register — use professional vocabulary and complete sentences"
    rewrite_down: "more casual and conversational — use contractions and informal phrasing"
```

Then re-run `ds5_label_axes.py` (dataset side) and `sae1_gen_rewrites.py` onward (SAE side).

---

## Raw Data

`candidate_posts_eng10.jsonl.gz` (~2.2M posts) is pre-filtered from the Bluesky firehose:
- ≥ 10 engagement, English only, ≥ 100 characters, ≥ 18 words
- No URLs, no replies / reposts / quotes, no `@`-starting posts, max 3 hashtags

Not included in the repo due to size.
