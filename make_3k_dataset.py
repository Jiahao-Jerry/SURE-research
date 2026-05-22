import json
import re
from collections import defaultdict

CATEGORIES = {
    "Technology & Innovation": [
        "artificial intelligence", "machine learning", "neural network",
        "large language model", "llm", "chatgpt", "open source", "cybersecurity",
        "data privacy", "cloud computing", "quantum computing", "semiconductor",
        "digital platform", "e-scooter", "explainable ai", "open banking",
        "nonprofit cyber", "open protocol", "electronic frontier", "huggingface",
        "openai", "softbank", "bluesky data", "data scraping", "programming",
        "developer", "software", "algorithm", "automation", "robot", "chip",
        "internet", "digital", "startup", "api", "app", "platform",
    ],
    "Science & Nature": [
        "baleen", "species", "evolution", "biology", "physics", "chemistry",
        "astronomy", "space", "planet", "star", "fossil", "gene", "dna",
        "protein", "cell", "atom", "ocean", "ecosystem", "wildlife", "geology",
        "neuroscience", "brain", "particle", "nasa", "telescope", "denisovan",
        "biomathematics", "soil organic carbon", "carbon cycling", "megadelta",
        "streamflow", "bioimage", "crispr", "discovery", "scientist", "research",
        "experiment", "laboratory", "nature", "species",
    ],
    "Health & Medicine": [
        "health", "disease", "treatment", "vaccine", "medicine", "doctor",
        "hospital", "mental health", "therapy", "patient", "drug", "medication",
        "surgery", "cancer", "diabetes", "heart", "nutrition", "diet", "sleep",
        "exercise", "virus", "bacteria", "immune", "anxiety", "depression",
        "symptom", "diagnosis", "clinical", "pharmaceutical", "mortality",
        "adhd", "multidrug resistance", "amputation", "prostate", "kidney",
        "physician", "mask mandate", "road traffic", "health informatics",
        "reproductive health", "periprosthetic", "hydroquinine", "long covid",
        "dialysis", "hsd11b2", "breast cancer", "physical activity", "walk",
        "autism", "fragrance cognition", "public involvement research",
    ],
    "Business & Economics": [
        "economy", "market", "stock", "investment", "startup", "entrepreneur",
        "revenue", "profit", "gdp", "inflation", "interest rate", "trade",
        "supply chain", "company", "corporate", "business", "finance", "bank",
        "venture capital", "ipo", "merger", "productivity", "labor", "workforce",
        "unemployment", "recession", "growth", "tariff", "export", "import",
        "market cap", "quantitative analysis", "imf", "shipping costs",
        "return-to-office", "hybrid work", "databricks", "openai valuation",
        "defense budget", "agribusiness", "gdpr",
    ],
    "Environment & Sustainability": [
        "climate change", "global warming", "carbon", "emission", "renewable",
        "solar", "wind energy", "fossil fuel", "sustainability", "environment",
        "pollution", "plastic", "deforestation", "biodiversity", "conservation",
        "green", "electric vehicle", "net zero", "recycling", "water",
        "drought", "flood", "wildfire", "coral reef", "glacier", "heat pump",
        "ipcc", "ecological", "trees", "forest", "co2", "greenhouse",
        "1.5c", "climate policy", "energy transition", "urban forestry",
    ],
    "Psychology & Behavior": [
        "psychology", "behavior", "cognitive", "bias", "habit", "motivation",
        "creativity", "decision", "emotion", "mindset", "perception", "memory",
        "attention", "stress", "trauma", "resilience", "empathy", "influence",
        "persuasion", "social psychology", "heuristic", "consciousness",
        "critical thinking", "logical fallacy", "emotional intelligence",
        "fact check", "media credibility", "financial resilience",
        "computational psychiatry", "gentrification",
    ],
    "Global Issues & Politics": [
        "war", "conflict", "peace", "diplomacy", "sanction", "refugee",
        "human rights", "geopolitics", "nato", "united nations", "foreign policy",
        "nuclear", "terrorism", "corruption", "election", "government",
        "policy", "legislation", "regulation", "democracy", "authoritarianism",
        "constitution", "ukraine", "hezbollah", "disinformation",
        "defense budget", "military spending", "camp century",
    ],
    "Society & Culture": [
        "society", "culture", "community", "social", "inequality", "race",
        "gender", "education", "school", "university", "poverty", "justice",
        "rights", "history", "tradition", "migration", "immigration", "identity",
        "diversity", "inclusion", "family", "generation", "youth", "aging",
        "urban", "rural", "journalism", "media", "netflix", "film",
        "communism", "religion", "ideology", "foster care", "disability",
        "evidence-based", "policymaking", "research culture",
    ],
}

CATEGORY_PRIORITY = [
    "Health & Medicine",
    "Science & Nature",
    "Technology & Innovation",
    "Environment & Sustainability",
    "Business & Economics",
    "Global Issues & Politics",
    "Psychology & Behavior",
    "Society & Culture",
]

NOISE = [
    r"\b(d&d|dnd|warlock|wizard|paladin|elf|dwarf|orc|bard|hexer)\b",
    r"\b(mtg|magic the gathering|commander deck|planeswalker|mana|counterspell)\b",
    r"\b(minecraft|fortnite|roblox|pokemon|genshin|valorant|elden ring|skyrim|baldur)\b",
    r"\bstalker 2\b",
    r"\b(anime|manga|waifu|otaku|jujutsu|naruto|one piece|demon slayer|headcanon)\b",
    r"\b(touchdown|quarterback|hat trick|offside|pitcher|shortstop)\b",
    r"\b(astrology|zodiac|mercury retrograde|birth chart|leo rising)\b",
    r"\b(crypto pump|to the moon|hodl|ape in|nft drop|whitelist|memecoin)\b",
    r"\b(warbler|firecrest|birdwatch|twitching)\b",
    r"\b(impersonating @|block and report|paypal scam)\b",
    r"(click here|subscribe now|buy now|coupon|discount)",
    r"bit\.ly|tinyurl|amzn\.to|goo\.gl|bityl\.co|ecomerit\.ie",
    r"^(hello|hi|hey|greetings).{0,80}(bluesky|bsky|follow me|this is my account)",
    r"^this is the (bsky|bluesky) account for",
    r"^intro post",
    r"^hi everyone[!.]?\s*(i'?m|check out|i am|are you interested)",
    r"\b(postdoctoral.*position|fully.funded position|phd position available)\b",
    r"^assistant professor.*@",
    r"^⏰ reminder",
    r"\bbright & early at \d",
    r"resist\.bot/petitions",
    r"\b(butterfly tour|birds of paradise.*tour)\b",
    r"^📢 virtual demonstration",
    r"\bbeing a republican.{0,60}(bad for|your health)\b",
    r"\b(pete hegseth dropped a bombshell)\b",
    r"\b(rachel maddow|grifter from msnbc)\b",
    r"\b(maga is not crying|maga is crying|trump pp|own the libs)\b",
    r"\b(edolf muskler)\b",
    r"trump.{0,30}tariffs.{0,50}intentionally raise prices",
    r"^🚨.{0,20}breaking news",
    r"\b(released from prison|helped hitman)\b",
    r"netflix star.*escaped.*prison.*smuggling",
    r"^time for an irrelevant moan",
    r"^good morning.*ms daisy",
    r"(nepali.*gpt.*chatgpt)",
    r"^calling (the|for) nepali",
    r"^we're a lab group based at",
    r"^a thread on (some|my) topics of my.*(career|research)",
    r"^can we do a #.*role.?call",
    r"^inspiring meeting last week",
    r"always great to join @.{5,50}to talk the latest",
    r"^reynolds: '(there are|we need|insists)",
    r"^i saw a post yesterday about the prices",
    r"l\.smartnews\.com",
    r"\b(this guy is a war monger)\b",
    r"^i swear the concept of government.{0,30}(sides|parties) is dumb",
    r"^asus (zenbook|vivobook).{0,30}review",
    r"^i read that more than \d+% of content on linkedin was ai",
    r"^seems like it'?s dropping every day.{0,30}(all because|invasion)",
    r"^having just re-read the introduction to.{0,80}(eds|ed\.)",
    r"\b(deathconsciousness)\b",
    r"(world.leading 3 percent.*gdp|economy is defying predictions)",
    r"\b(stain only had 48%|blue dot resistance)\b",
    r"^i enjoy owning.*shares\. despite",
    r"^finally got a chance to play with duckdb",
    r"tupped.*kindly written a sample email",
    r"^1/ hey @hf\.co no\.",
    r"^there are still opportunities for (msc|phd) projects",
    r"^never in my life have i done a benchmarking study",
    r"^excellent resource for.{0,60}courses",
]

def is_noisy(text):
    t = text.lower()
    for pat in NOISE:
        if re.search(pat, t, re.I):
            return True
    return False

def classify(text):
    t = text.lower()
    scores = defaultdict(float)
    for cat, kws in CATEGORIES.items():
        for kw in kws:
            if kw in t:
                scores[cat] += 0.5 + len(kw) / 20.0
    if not scores:
        return "Society & Culture"
    best = max(scores.values())
    for cat in CATEGORY_PRIORITY:
        if scores.get(cat, 0) >= best * 0.85:
            return cat
    return max(scores, key=scores.get)

def main():
    seen = set()
    selected = []

    with open("master_dataset.jsonl") as f:
        for line in f:
            obj = json.loads(line.strip())
            text = obj.get("text", "")

            # Hard filters
            if obj.get("reply_to"):
                continue
            if text.startswith("@"):
                continue
            if len(text) < 200:
                continue
            if len(re.findall(r"#\w+", text)) > 5:
                continue
            if len(re.findall(r"https?://", text)) > 1:
                continue
            if is_noisy(text):
                continue

            # Text dedup
            key = text[:100].strip().lower()
            if key in seen:
                continue
            seen.add(key)

            obj["category"] = classify(text)
            selected.append(obj)

            if len(selected) == 3000:
                break

    print(f"Selected: {len(selected)}")
    dist = defaultdict(int)
    for o in selected:
        dist[o["category"]] += 1
    for cat, n in sorted(dist.items(), key=lambda x: -x[1]):
        print(f"  {n:4d}  {cat}")

    with open("ted3k_dataset.jsonl", "w") as f:
        for o in selected:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
    print("Written to ted3k_dataset.jsonl")

if __name__ == "__main__":
    main()
