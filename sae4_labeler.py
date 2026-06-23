"""
Generate sae_feature_labeler.html — interactive viewer for SAE features.

For each feature shows:
  - Lift across all 7 axes (bar chart)
  - Category, density
  - Top 10 examples with highest activation (original + rewrite + axis + direction)
"""

import json, numpy as np, torch, torch.nn as nn, glob
from collections import Counter
from config import load_config, get_axis_names, get_axis_colors, validate_config

# ── Config ────────────────────────────────────────────────────────
MODEL_FILE   = "sae_qwen_model.pt"
CACHE_FILE   = "sae_clean_rewrites.json"
OUT_HTML     = "sae_feature_labeler.html"

N_FEATURES   = 64
CONFIRM_LIFT = 0.30
PARTIAL_LIFT = 0.15
TOP_EXAMPLES = 10

cfg        = load_config()
validate_config(cfg)
AXIS_NAMES = get_axis_names(cfg)

AXIS_COLORS = get_axis_colors(cfg)

# Auto-detect diff vector files
diff_files = sorted(glob.glob("diff_vectors_*.npy"))
if not diff_files:
    raise FileNotFoundError("No diff_vectors_*.npy found. Run step2 first.")
n_str    = diff_files[-1].split("_")[-1].replace(".npy", "")
DIFF_FILE = diff_files[-1]
AXIS_FILE = f"axis_labels_{n_str}.npy"
DIR_FILE  = f"directions_{n_str}.npy"
ID_FILE   = f"post_ids_{n_str}.npy"

# ── Load SAE ──────────────────────────────────────────────────────
class SAE(nn.Module):
    def __init__(self, input_dim, n_features):
        super().__init__()
        self.W_enc = nn.Parameter(torch.empty(n_features, input_dim))
        self.b_enc = nn.Parameter(torch.zeros(n_features))
        self.W_dec = nn.Parameter(torch.empty(input_dim, n_features))
        self.b_dec = nn.Parameter(torch.zeros(input_dim))
    def encode(self, x):
        return torch.relu(x @ self.W_enc.T + self.b_enc)

model = SAE(2048, N_FEATURES)
model.load_state_dict(torch.load(MODEL_FILE, map_location="cpu"))
model.eval()

# ── Load diff vectors ─────────────────────────────────────────────
diff  = np.load(DIFF_FILE).astype(np.float32)
axis  = np.load(AXIS_FILE, allow_pickle=True).astype(str)
dirs  = np.load(DIR_FILE,  allow_pickle=True).astype(str)
pids  = np.load(ID_FILE,   allow_pickle=True).astype(str)

# Same preprocessing as sae3_train.py
diff[dirs == "down"] *= -1
norms = np.linalg.norm(diff, axis=1, keepdims=True).clip(min=1e-8)
diff  = diff / norms

# ── Get activations ───────────────────────────────────────────────
with torch.no_grad():
    acts = model.encode(torch.from_numpy(diff)).numpy()  # (4900, 64)

print(f"Activations: {acts.shape}  dead={(acts.max(0)==0).sum()}")

# ── Load rewrite cache ────────────────────────────────────────────
with open(CACHE_FILE) as f:
    raw = json.load(f)
cache = {p["post_id"]: p for p in raw}  # post_id -> {original, rewrites}

# ── Compute per-feature stats ─────────────────────────────────────
features = []
for f in range(N_FEATURES):
    feat    = acts[:, f]
    density = float((feat > 0).mean())

    if density < 0.01:
        features.append({"f": f, "density": 0.0, "category": "dead",
                         "lifts": {ax: 0.0 for ax in AXIS_NAMES},
                         "best_axis": None, "best_lift": 0.0, "examples": []})
        continue

    # Lift per axis
    lifts = {}
    for ax in AXIS_NAMES:
        mask   = axis == ax
        p_on   = float((feat[mask]  > 0).mean()) if mask.sum()  > 0 else 0.0
        p_off  = float((feat[~mask] > 0).mean()) if (~mask).sum() > 0 else 0.0
        lifts[ax] = round(p_on - p_off, 4)

    best_ax   = max(lifts, key=lambda k: abs(lifts[k]))
    best_lift = lifts[best_ax]
    cat = ("confirms_axis"   if abs(best_lift) >= CONFIRM_LIFT else
           "partial_overlap" if abs(best_lift) >= PARTIAL_LIFT else
           "novel_candidate")

    # Top examples by activation value
    top_idx = np.argsort(feat)[::-1][:TOP_EXAMPLES]
    examples = []
    for idx in top_idx:
        if feat[idx] <= 0:
            break
        pid      = pids[idx]
        ax_name  = axis[idx]
        direction = dirs[idx]
        post     = cache.get(pid, {})
        original = post.get("original", "")
        rewrite  = post.get("rewrites", {}).get(ax_name, {}).get("rewrite", "")
        examples.append({
            "pid":       pid,
            "axis":      ax_name,
            "direction": direction,
            "activation": round(float(feat[idx]), 4),
            "original":  original,
            "rewrite":   rewrite,
        })

    features.append({
        "f":         f,
        "density":   round(density, 4),
        "category":  cat,
        "lifts":     lifts,
        "best_axis": best_ax,
        "best_lift": round(best_lift, 4),
        "examples":  examples,
    })

# Sort: confirms first (by lift), then partial, then novel, then dead
cat_order = {"confirms_axis": 0, "partial_overlap": 1, "novel_candidate": 2, "dead": 3}
features_sorted = sorted(features, key=lambda x: (cat_order[x["category"]], -abs(x["best_lift"])))

cats = Counter(f["category"] for f in features)
print(f"Categories: {dict(cats)}")

# ── Build HTML ────────────────────────────────────────────────────
CAT_COLORS = {
    "confirms_axis":   "#5cb85c",
    "partial_overlap": "#e6a817",
    "novel_candidate": "#6c8ebf",
    "dead":            "#555",
}

features_json = json.dumps(features_sorted, ensure_ascii=False)
axis_colors_json = json.dumps(AXIS_COLORS)
cat_colors_json  = json.dumps(CAT_COLORS)

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>SAE Feature Labeler</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: #0f1117; color: #e8eaf0; font-family: 'Segoe UI', sans-serif; display: flex; height: 100vh; overflow: hidden; }}

  /* Left panel */
  #sidebar {{ width: 280px; min-width: 280px; background: #1e2130; border-right: 1px solid #2a2d3e; display: flex; flex-direction: column; }}
  #sidebar-header {{ padding: 14px 16px; border-bottom: 1px solid #2a2d3e; font-size: 13px; color: #aaa; }}
  #sidebar-header strong {{ color: #e8eaf0; font-size: 15px; display: block; margin-bottom: 4px; }}
  #feature-list {{ overflow-y: auto; flex: 1; }}
  .feat-item {{ padding: 10px 16px; cursor: pointer; border-bottom: 1px solid #23263a; transition: background 0.15s; }}
  .feat-item:hover {{ background: #252840; }}
  .feat-item.active {{ background: #2e3355; border-left: 3px solid #6c8ebf; }}
  .feat-item .feat-title {{ font-size: 13px; font-weight: 600; }}
  .feat-item .feat-sub {{ font-size: 11px; color: #aaa; margin-top: 2px; }}
  .cat-badge {{ display: inline-block; padding: 1px 6px; border-radius: 3px; font-size: 10px; font-weight: 700; margin-left: 6px; }}

  /* Right panel */
  #detail {{ flex: 1; overflow-y: auto; padding: 24px 32px; }}
  #detail h2 {{ font-size: 20px; margin-bottom: 6px; }}
  #detail .meta {{ font-size: 13px; color: #aaa; margin-bottom: 20px; }}
  .section-title {{ font-size: 13px; font-weight: 700; color: #aaa; text-transform: uppercase; letter-spacing: 0.08em; margin: 24px 0 10px; }}

  /* Lift chart */
  .lift-row {{ display: flex; align-items: center; margin-bottom: 7px; }}
  .lift-label {{ width: 160px; font-size: 12px; color: #ccc; }}
  .lift-bar-bg {{ flex: 1; height: 18px; background: #2a2d3e; border-radius: 3px; position: relative; }}
  .lift-bar {{ height: 100%; border-radius: 3px; transition: width 0.3s; }}
  .lift-val {{ width: 52px; text-align: right; font-size: 12px; color: #ccc; padding-left: 8px; }}

  /* Examples */
  .example-card {{ background: #1a1d2e; border: 1px solid #2a2d3e; border-radius: 6px; padding: 14px 16px; margin-bottom: 12px; }}
  .example-meta {{ font-size: 11px; color: #888; margin-bottom: 8px; }}
  .example-meta span {{ color: #ccc; font-weight: 600; }}
  .text-label {{ font-size: 11px; color: #888; margin-bottom: 4px; }}
  .text-block {{ font-size: 13px; line-height: 1.55; color: #d8dae8; background: #13151f; border-radius: 4px; padding: 10px 12px; margin-bottom: 10px; white-space: pre-wrap; }}
  .text-block.rewrite {{ border-left: 3px solid #6c8ebf; }}
  .activation-badge {{ display: inline-block; background: #2e3355; color: #8ab4f8; padding: 2px 8px; border-radius: 12px; font-size: 11px; font-weight: 700; }}
</style>
</head>
<body>

<div id="sidebar">
  <div id="sidebar-header">
    <strong>SAE Feature Labeler</strong>
    {N_FEATURES} features · {len(axis)} pairs · layer 23
  </div>
  <div id="feature-list"></div>
</div>

<div id="detail">
  <p style="color:#555; margin-top: 40px; text-align:center;">Select a feature from the left panel.</p>
</div>

<script>
const FEATURES      = {features_json};
const AXIS_COLORS   = {axis_colors_json};
const CAT_COLORS    = {cat_colors_json};
const AXIS_NAMES    = {json.dumps(AXIS_NAMES)};
const CONFIRM_LIFT  = {CONFIRM_LIFT};
const PARTIAL_LIFT  = {PARTIAL_LIFT};

// ── Build sidebar ────────────────────────────────────────────────
const list = document.getElementById('feature-list');
FEATURES.forEach((feat, i) => {{
  const item = document.createElement('div');
  item.className = 'feat-item';
  item.dataset.idx = i;
  const catColor = CAT_COLORS[feat.category] || '#555';
  const axLabel  = feat.best_axis ? feat.best_axis.replace('_',' ') : '—';
  item.innerHTML = `
    <div class="feat-title">
      F${{feat.f}}
      <span class="cat-badge" style="background:${{catColor}}22;color:${{catColor}}">
        ${{feat.category === 'confirms_axis' ? '✓' : feat.category === 'partial_overlap' ? '~' : feat.category === 'dead' ? '✗' : '?'}}
      </span>
    </div>
    <div class="feat-sub">${{axLabel}}  lift=${{feat.best_lift.toFixed(3)}}  density=${{feat.density.toFixed(3)}}</div>
  `;
  item.addEventListener('click', () => showFeature(i));
  list.appendChild(item);
}});

// ── Show feature detail ───────────────────────────────────────────
function showFeature(idx) {{
  document.querySelectorAll('.feat-item').forEach(el => el.classList.remove('active'));
  document.querySelector(`.feat-item[data-idx="${{idx}}"]`).classList.add('active');

  const feat = FEATURES[idx];
  const catColor = CAT_COLORS[feat.category] || '#555';
  const maxLift  = Math.max(...Object.values(feat.lifts).map(Math.abs), 0.01);

  // Lift bars
  let liftHtml = '';
  AXIS_NAMES.forEach(ax => {{
    const lift    = feat.lifts[ax] || 0;
    const pct     = Math.abs(lift) / maxLift * 100;
    const color   = AXIS_COLORS[ax] || '#6c8ebf';
    const isBest  = ax === feat.best_axis;
    liftHtml += `
      <div class="lift-row">
        <div class="lift-label" style="${{isBest ? 'color:#fff;font-weight:700' : ''}}">${{ax.replace(/_/g,' ')}}</div>
        <div class="lift-bar-bg">
          <div class="lift-bar" style="width:${{pct}}%;background:${{color}};opacity:${{isBest ? 1 : 0.5}}"></div>
        </div>
        <div class="lift-val">${{lift >= 0 ? '+' : ''}}${{lift.toFixed(3)}}</div>
      </div>`;
  }});

  // Examples
  let exHtml = '';
  if (feat.examples.length === 0) {{
    exHtml = '<p style="color:#555">No active examples.</p>';
  }} else {{
    feat.examples.forEach(ex => {{
      const axColor = AXIS_COLORS[ex.axis] || '#6c8ebf';
      exHtml += `
        <div class="example-card">
          <div class="example-meta">
            post ${{ex.pid}} ·
            <span style="color:${{axColor}}">${{ex.axis.replace(/_/g,' ')}}</span> ·
            direction: <span>${{ex.direction}}</span> ·
            activation: <span class="activation-badge">${{ex.activation.toFixed(4)}}</span>
          </div>
          <div class="text-label">ORIGINAL</div>
          <div class="text-block">${{escHtml(ex.original)}}</div>
          <div class="text-label">REWRITE (${{ex.direction}} ${{ex.axis.replace(/_/g,' ')}})</div>
          <div class="text-block rewrite">${{escHtml(ex.rewrite)}}</div>
        </div>`;
    }});
  }}

  document.getElementById('detail').innerHTML = `
    <h2>Feature F${{feat.f}}</h2>
    <div class="meta">
      <span class="cat-badge" style="background:${{catColor}}22;color:${{catColor}};font-size:12px">${{feat.category}}</span>
      &nbsp; best axis: <strong>${{feat.best_axis || '—'}}</strong> &nbsp;
      lift: <strong>${{feat.best_lift.toFixed(4)}}</strong> &nbsp;
      density: <strong>${{feat.density.toFixed(4)}}</strong>
    </div>
    <div class="section-title">Lift across axes</div>
    ${{liftHtml}}
    <div class="section-title">Top ${{feat.examples.length}} activating examples</div>
    ${{exHtml}}
  `;
}}

function escHtml(s) {{
  return String(s)
    .replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')
    .replace(/"/g,'&quot;').replace(/'/g,'&#39;');
}}

// Auto-select first feature
if (FEATURES.length > 0) showFeature(0);
</script>
</body>
</html>
"""

with open(OUT_HTML, "w", encoding="utf-8") as f:
    f.write(html)

print(f"Saved → {OUT_HTML}")
import subprocess; subprocess.Popen(["open", OUT_HTML])
