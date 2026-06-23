"""
Label 2,550 posts with 7 style axes using Claude Haiku.

Input:  2550_posts.jsonl
Output: 2550_posts.jsonl  (adds axes_json field in-place)
        label_axes_checkpoint.json  (resume support)
"""

import json, os, time
from anthropic import Anthropic
from concurrent.futures import ThreadPoolExecutor, as_completed
from config import load_config, build_label_prompt, validate_config

INPUT_FILE      = "2550_posts.jsonl"
CHECKPOINT_FILE = "label_axes_checkpoint.json"
BATCH_SIZE      = 5
MAX_WORKERS     = 20

client = Anthropic()

cfg = load_config()
validate_config(cfg)
SYSTEM_PROMPT = build_label_prompt(cfg)


def score_batch(batch, attempt=1):
    numbered = "\n\n".join(f"[{i+1}] {p['text'][:500]}" for i, p in enumerate(batch))
    try:
        resp = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1500,
            temperature=0,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content":
                f"Score these {len(batch)} posts:\n\n{numbered}"}]
        )
        raw = resp.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        scores = json.loads(raw.strip())
        if len(scores) == len(batch):
            return scores
    except Exception as e:
        if attempt < 3:
            time.sleep(2 ** attempt)
            return score_batch(batch, attempt + 1)
        print(f"  Batch failed: {e}")
    return [None] * len(batch)


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
    print(f"Checkpoint: {len(checkpoint)} already labeled")

    todo = [p for p in posts if str(p["post_id"]) not in checkpoint]
    print(f"Remaining: {len(todo)} posts")

    batches = [todo[i:i+BATCH_SIZE] for i in range(0, len(todo), BATCH_SIZE)]
    processed = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(score_batch, b): b for b in batches}
        for future in as_completed(futures):
            batch = futures[future]
            scores = future.result()
            for post, axes in zip(batch, scores):
                if axes is not None:
                    checkpoint[str(post["post_id"])] = axes
            processed += len(batch)
            if processed % 50 == 0 or processed == len(batches):
                save_checkpoint(checkpoint)
                done = len([p for p in posts if str(p["post_id"]) in checkpoint])
                print(f"  [{done}/{len(posts)}] labeled")

    save_checkpoint(checkpoint)
    print(f"\nLabeling done. Writing output...")

    # Merge axes back into posts
    missing = 0
    with open(INPUT_FILE, "w") as f:
        for p in posts:
            axes = checkpoint.get(str(p["post_id"]))
            if axes is None:
                missing += 1
            p["axes_json"] = axes
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"Done. {len(posts) - missing} posts labeled, {missing} missing.")
    print(f"Output: {INPUT_FILE}")


if __name__ == "__main__":
    main()
