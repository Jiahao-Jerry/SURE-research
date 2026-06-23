"""
Step 1 (Local) — Generate 4,900 rewrites (700 posts × 7 axes) using Claude Haiku.

Input:  2550_posts.jsonl  (from github.com/Jiahao-Jerry/SURE)
Output: sae_clean_rewrites.json  (list of 700 post entries, each with 7 axis rewrites)

Post selection: up to 1,190 posts (17 × 70). First 700 are already cached;
only the next 29 per topic (~493 new posts × 7 = ~3,451 rewrites) are generated.
Posts ranked by mid-range axis score count within each topic.

Cost estimate (new only): ~3,451 rewrites × $0.00062 ≈ $2.14
Run time (new only):      ~18 minutes
"""

import json, time, random
import numpy as np
import pandas as pd
from pathlib import Path
from anthropic import Anthropic
from config import load_config, get_axis_names, get_axis_defs, validate_config

POSTS_FILE = "2550_posts.jsonl"

# ── Config ────────────────────────────────────────────────────────
N_PER_TOPIC    = 70            # posts per topic (17 × 70 = 1190; first 41 already cached)
CACHE_FILE     = "sae_clean_rewrites.json"
SEED           = 42

client = Anthropic()

cfg        = load_config()
validate_config(cfg)
AXIS_NAMES = get_axis_names(cfg)
AXIS_DEFS  = get_axis_defs(cfg)

def make_prompt(text: str, axis: str, direction: str,
                current_score: float, target_score: float) -> str:
    desc = AXIS_DEFS[axis][direction]
    return f"""Rewrite this social media post so that it is {desc}.

Current {axis} score: {current_score:.2f}/1.0
Target  {axis} score: {target_score:.2f}/1.0  (shift of {abs(target_score-current_score):.2f})

RULES:
- Keep the EXACT same claim, facts, opinion and stance — do not add or remove information.
- Only change delivery on this one dimension.
- Keep roughly the same length (±20%).
- Return ONLY the rewritten post text, nothing else.

ORIGINAL POST:
{text}"""

def generate_rewrite(text, axis, direction, current_score, target_score, retries=3):
    for attempt in range(retries):
        try:
            r = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=400,
                messages=[{"role": "user", "content":
                           make_prompt(text, axis, direction, current_score, target_score)}]
            )
            return r.content[0].text.strip()
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    return None

# ── Load corpus ───────────────────────────────────────────────────
print("Loading corpus...")
rows = []
with open(POSTS_FILE) as f:
    for line in f:
        rows.append(json.loads(line))
df = pd.DataFrame(rows)
df["post_id"] = df["post_id"].astype(str)

def parse_axes(row):
    axes = json.loads(row) if isinstance(row, str) else row
    return {ax: axes[ax]["score"] if isinstance(axes.get(ax), dict) else axes.get(ax)
            for ax in AXIS_NAMES}

axes_df = df["axes_json"].apply(parse_axes).apply(pd.Series).astype(float)
df = pd.concat([df, axes_df], axis=1)
print(f"  {len(df)} posts loaded")

# ── Load existing cache ───────────────────────────────────────────
# Format: list of {post_id, original, rewrites: {axis: {direction, rewrite}}}
# Internal flat dict for fast lookup: (post_id, axis) -> entry
cache_list: list = []
cache: dict = {}   # (post_id, axis) -> rewrite str, for fast duplicate check

if Path(CACHE_FILE).exists():
    with open(CACHE_FILE) as f:
        raw = json.load(f)
    if isinstance(raw, dict) and "__axis_names__" in raw:
        cache_list = raw.get("posts", [])
        for entry in cache_list:
            for ax, r in entry.get("rewrites", {}).items():
                cache[(entry["post_id"], ax)] = r["rewrite"]
    elif isinstance(raw, list):
        cache_list = raw
        for entry in cache_list:
            for ax, r in entry.get("rewrites", {}).items():
                cache[(entry["post_id"], ax)] = r["rewrite"]
    else:
        # Migrate old flat-dict format on the fly
        for key, r in raw.items():
            cache[(r["post_id"], r["axis"])] = r["rewrite"]

print(f"  {len(cache)} rewrites already cached")

# ── Select N_PER_TOPIC posts per topic ───────────────────────────
def mid_range_count(row):
    return sum(1 for ax in AXIS_NAMES if 0.2 <= row[ax] <= 0.8)

df["mid_count"] = df.apply(mid_range_count, axis=1)

selected_ids = []
rng = random.Random(SEED)
for topic, group in df.groupby("topic_name"):
    group_sorted = group.sort_values("mid_count", ascending=False)
    selected_ids += group_sorted["post_id"].iloc[:N_PER_TOPIC].tolist()

print(f"\nSelected {len(selected_ids)} posts ({N_PER_TOPIC} per topic × {df['topic_name'].nunique()} topics)")

# ── Build work list ───────────────────────────────────────────────
todo = []
for pid in selected_ids:
    row = df[df["post_id"] == pid].iloc[0]
    for ax in AXIS_NAMES:
        score     = float(row[ax])
        direction = "up" if score < 0.5 else "down"
        target    = min(score + 0.40, 1.0) if direction == "up" else max(score - 0.40, 0.0)
        if (pid, ax) not in cache:
            todo.append({
                "post_id": pid, "axis": ax,
                "direction": direction, "current": score, "target": target,
                "original": row["text"],
            })

total_needed = len(selected_ids) * len(AXIS_NAMES)
already_done = total_needed - len(todo)
print(f"Total pairs needed:  {total_needed}  ({len(selected_ids)} posts × {len(AXIS_NAMES)} axes)")
print(f"Already cached:      {already_done}")
print(f"To generate:         {len(todo)}")
print(f"Estimated cost:      ${len(todo) * 0.00062:.2f}")

# ── Generate ──────────────────────────────────────────────────────
def save_cache():
    # Rebuild list from flat cache dict
    by_post: dict = {}
    for (pid, ax), rewrite_text in cache.items():
        if pid not in by_post:
            rows = df[df["post_id"] == pid]
            orig = rows.iloc[0]["text"] if not rows.empty else ""
            by_post[pid] = {"post_id": pid, "original": orig, "rewrites": {}}
        rows = df[df["post_id"] == pid]
        if not rows.empty:
            score = float(rows.iloc[0][ax])
            direction = "up" if score < 0.5 else "down"
        else:
            direction = "up"
        by_post[pid]["rewrites"][ax] = {"direction": direction, "rewrite": rewrite_text}
    # Embed axis list so step2 on Colab doesn't need config.py
    output = {"__axis_names__": AXIS_NAMES, "posts": list(by_post.values())}
    with open(CACHE_FILE, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

print(f"\nGenerating rewrites...")
failed = 0
for i, item in enumerate(todo):
    rewrite = generate_rewrite(
        item["original"], item["axis"], item["direction"],
        item["current"], item["target"]
    )
    if rewrite:
        cache[(item["post_id"], item["axis"])] = rewrite
    else:
        failed += 1

    if (i + 1) % 50 == 0:
        save_cache()
        print(f"  {i+1}/{len(todo)} done  |  {failed} failed  |  cached {len(cache)}")

save_cache()
print(f"\nDone. Total cached: {len(cache)}  |  Failed: {failed}")
print(f"Cache saved → {CACHE_FILE}")
