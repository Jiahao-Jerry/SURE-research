import yaml

def load_config(path="pipeline_config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)

def get_cluster_to_topic(cfg):
    return {t["cluster_id"]: t["name"] for t in cfg["topics"]}

def get_axis_names(cfg):
    return [a["name"] for a in cfg["axes"]]

def get_axis_short(cfg):
    return [a["short"] for a in cfg["axes"]]

def get_axis_defs(cfg):
    return {a["name"]: {"up": a["rewrite_up"], "down": a["rewrite_down"]}
            for a in cfg["axes"]}

def get_axis_colors(cfg):
    return {a["name"]: a["color"] for a in cfg["axes"]}

def build_label_prompt(cfg):
    axes = cfg["axes"]
    keys = "\n".join(f'  "{a["name"]}": {{"score": 0.0-1.0}},' for a in axes)
    descs = "\n".join(f'- {a["name"]}: {a["label_description"]}' for a in axes)
    return f"""You are a writing style analyst. For each numbered post, score {len(axes)} style dimensions and return ONLY a JSON array.

Each element must be an object with exactly these keys:
{{
{keys}
}}

Dimension guidelines:
{descs}

Reply ONLY with a valid JSON array of objects, one per post."""

def validate_config(cfg):
    assert len(cfg.get("topics", [])) > 0, "No topics defined in pipeline_config.yaml"
    assert len(cfg.get("axes", [])) > 0, "No axes defined in pipeline_config.yaml"
    for t in cfg["topics"]:
        assert "cluster_id" in t and "name" in t, f"Topic missing cluster_id or name: {t}"
    for a in cfg["axes"]:
        for key in ("name", "short", "color", "label_description", "rewrite_up", "rewrite_down"):
            assert key in a, f"Axis '{a.get('name','?')}' missing field: {key}"
