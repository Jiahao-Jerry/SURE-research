"""
LLM Agent Quality Check and Ranking

Input:  candidate_pool_500.jsonl (500 posts per topic, 8,500 total)
Output:
  - rankings.jsonl    — every post with its agent score, sorted by score
  - 2550_posts.jsonl  — top 150 per topic (17 × 150 = 2,550 posts)

Each post is scored 0.0–1.0 by Claude Haiku. Top 150 per topic are kept.
"""

import json, os, time
from anthropic import Anthropic
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict

INPUT_FILE      = "candidate_pool_500.jsonl"
RANKINGS_FILE   = "rankings.jsonl"
OUTPUT_FILE     = "2550_posts.jsonl"
CHECKPOINT_FILE = "agent_qc_checkpoint.json"

TOP_N       = 150
BATCH_SIZE  = 10
MAX_WORKERS = 15

client = Anthropic()

SYSTEM_PROMPT = """You are a content quality rater. Score each post from 0.0 to 1.0.

A HIGH QUALITY post (score 0.7–1.0) shares specific knowledge, insight, or a well-reasoned opinion, gives the reader something to think about, and uses concrete facts, examples, or observations in English.

A LOW QUALITY post (score 0.0–0.3) is personal feelings or reactions with no substance, a list, not in English, or low-effort commentary.

Scores between 0.3 and 0.7 are for posts that partially meet the criteria.

For each numbered post, reply with ONLY a JSON array of decimal scores between 0.0 and 1.0.
Example for 3 posts: [0.9, 0.4, 0.1]"""


def score_batch(batch, attempt=1):
    numbered = "\n\n".join(f"[{i+1}] {p['text'][:600]}" for i, p in enumerate(batch))
    try:
        resp = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=100,
            temperature=0,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": f"Score these {len(batch)} posts:\n\n{numbered}"}]
        )
        scores = json.loads(resp.content[0].text.strip())
        if (len(scores) == len(batch) and
                all(isinstance(s, (int, float)) and 0.0 <= float(s) <= 1.0 for s in scores)):
            return [float(s) for s in scores]
    except Exception as e:
        if attempt < 3:
            time.sleep(2 ** attempt)
            return score_batch(batch, attempt + 1)
        print(f"  Batch failed after 3 attempts: {e}")
    return [0.5] * len(batch)


def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE) as f:
            return json.load(f)
    return {}


def save_checkpoint(cp):
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(cp, f)


def main():
    posts = []
    with open(INPUT_FILE) as f:
        for line in f:
            posts.append(json.loads(line))
    print(f"Loaded {len(posts)} posts")

    checkpoint = load_checkpoint()
    print(f"Checkpoint: {len(checkpoint)} already scored")

    todo = [p for p in posts if str(p["post_id"]) not in checkpoint]
    print(f"Remaining: {len(todo)} posts")

    batches = [todo[i:i+BATCH_SIZE] for i in range(0, len(todo), BATCH_SIZE)]
    processed = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(score_batch, b): b for b in batches}
        for future in as_completed(futures):
            batch = futures[future]
            scores = future.result()
            for post, score in zip(batch, scores):
                checkpoint[str(post["post_id"])] = score
            processed += len(batch)
            if processed % 500 == 0 or processed == len(batches):
                save_checkpoint(checkpoint)
                done = sum(1 for p in posts if str(p["post_id"]) in checkpoint)
                print(f"  [{done}/{len(posts)}] scored")

    save_checkpoint(checkpoint)
    print(f"\nAll scoring done.")

    # ── Save rankings ─────────────────────────────────────────────
    print(f"\nSaving {RANKINGS_FILE}...")
    with open(RANKINGS_FILE, "w") as f:
        for p in sorted(posts, key=lambda x: checkpoint.get(str(x["post_id"]), 0), reverse=True):
            f.write(json.dumps({
                "post_id":     p["post_id"],
                "topic_name":  p["topic_name"],
                "agent_score": checkpoint.get(str(p["post_id"])),
                "lr_score":    p["lr_score"],
                "engagement":  p["engagement"],
                "text":        p["text"],
            }, ensure_ascii=False) + "\n")
    print(f"  Saved {len(posts)} ranked posts")

    # ── Top 150 per topic ─────────────────────────────────────────
    print(f"\nSelecting top {TOP_N} per topic → {OUTPUT_FILE}...")
    by_topic = defaultdict(list)
    for p in posts:
        by_topic[p["topic_name"]].append((p, checkpoint.get(str(p["post_id"]), 0)))

    total = 0
    with open(OUTPUT_FILE, "w") as f:
        for topic_name, topic_posts in sorted(by_topic.items()):
            top = sorted(topic_posts, key=lambda x: x[1], reverse=True)[:TOP_N]
            for p, score in top:
                f.write(json.dumps({
                    "post_id":     p["post_id"],
                    "topic_name":  p["topic_name"],
                    "text":        p["text"],
                    "engagement":  p["engagement"],
                    "lr_score":    p["lr_score"],
                    "agent_score": score,
                }, ensure_ascii=False) + "\n")
            total += len(top)
            print(f"  {topic_name:<30} {len(top)} posts  "
                  f"(top: {top[0][1]:.2f}, cutoff: {top[-1][1]:.2f})")

    print(f"\nTotal: {total} posts → {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
