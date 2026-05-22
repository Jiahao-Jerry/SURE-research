import json
import re
from collections import defaultdict

# ── TED-style categories ──────────────────────────────────────────────────
# Higher-specificity keywords get higher weight via length bonus in classify_category().
# Categories are tried in priority order for tie-breaking.
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

CATEGORIES = {
    "Technology & Innovation": [
        "artificial intelligence", "machine learning", "neural network",
        "large language model", "llm", "chatgpt", "open source", "cybersecurity",
        "data privacy", "cloud computing", "quantum computing", "semiconductor",
        "digital platform", "e-scooter algorithm", "gbfs data",
        "explainable ai", "open banking", "nonprofit cyber", "open protocol",
        "electronic frontier foundation", "huggingface", "openai",
        "softbank tender", "bluesky data", "data scraping ai",
        "cloud computing energy", "condensed matter physics",
    ],
    "Science & Nature": [
        "baleen whale", "evolution", "neuroscience", "astronomy",
        "fossil", "gene editing", "crispr", "dna", "protein",
        "cell biology", "atom", "ocean", "ecosystem", "geology",
        "particle physics", "nasa", "telescope", "denisovan", "biomathematics",
        "soil organic carbon", "carbon cycling", "megadelta", "streamflow",
        "megadeltas", "bioimage", "reproducibility bioimage",
        "journal of the american medical association", "far right youtube",
        "obsidian md", "regional universities",
    ],
    "Health & Medicine": [
        "drug resistance", "multidrug resistance", "clinical trial",
        "mental health", "therapy outcome", "vaccine", "surgery outcome",
        "cancer research", "diabetes", "heart disease",
        "nutrition", "sleep research", "exercise benefit",
        "virus", "bacteria", "immune", "anxiety treatment", "depression",
        "diagnosis", "pharmaceutical", "mortality rate", "adhd",
        "amputation risk", "prostate cancer", "kidney disease",
        "transgender health camhs", "equine therapy veteran",
        "physician associate nhs", "mask mandate covid",
        "road traffic injury", "health informatics",
        "reproductive health", "periprosthetic joint infection",
        "hydroquinine bacteria", "air pollution long covid",
        "greener dialysis", "hsd11b2", "breast cancer clinical trial",
        "long term health benefits exercise", "10-15 minutes of exercise",
        "walk more live longer", "physical activity",
        "islamophobia mental health",
        "autism late diagnosis",
        "bhattacharya", "fragrance cognition",
        # Misclassified as Society & Culture
        "patient & public involvement", "patient public involvement research",
        "incorporating patient", "public involvement in research",
    ],
    "Business & Economics": [
        "inflation data", "interest rate", "supply chain",
        "venture capital", "gdp", "labor market", "unemployment",
        "productivity", "imf analysis", "tariff impact",
        "canadian exports", "vietnam exports",
        "minimum wage effect", "market cap",
        "nwsl attendance", "wall street inflation",
        "shipping costs inflation", "return-to-office",
        "hybrid work productivity", "databricks", "snowflake",
        "quantitative analysis stocks",
        "climate investment funds world bank",
        "net zero productivity investment",
        "gdpr data", "trump tax law workers",
        "deportation algorithm corporate watch",
        "saudi taxi market",
        "brazil bitcoin reserves",
        "gm tariff market cap",
        "china defense budget",
        "uk agribusiness",
    ],
    "Environment & Sustainability": [
        "climate change", "global warming", "carbon emission",
        "renewable energy", "solar power", "wind energy",
        "fossil fuel subsid", "sustainability",
        "pollution", "plastic waste", "deforestation", "biodiversity",
        "electric vehicle", "net zero", "recycling",
        "water scarcity", "drought", "wildfire", "coral reef", "glacier",
        "industrial heat pump", "ipcc report", "ecological threat",
        "1.5c warming", "trees counteract climate", "eth zurich trees",
        "reform uk climate", "epa climate attitudes",
        "trains not planes gatwick", "energy protest policy",
        "ukraine lithium electric vehicle",
        "milliband net zero", "greenland camp century",
    ],
    "Psychology & Behavior": [
        "cognitive bias", "habit formation", "decision making",
        "emotional intelligence", "mindset", "perception", "memory research",
        "attention span", "stress resilience", "empathy",
        "social influence", "persuasion", "heuristic", "consciousness",
        "critical thinking", "logical fallacy",
        "fact checking content creators",
        "financial resilience", "ux psychology", "product psychology",
        "bellingcat far right", "sensory archaeology",
        "planning anxiety computational psychiatry",
        "computational psychiatry",
    ],
    "Global Issues & Politics": [
        "war conflict", "peace negotiation", "foreign policy", "nuclear",
        "terrorism financing", "corruption",
        "election interference", "government policy",
        "authoritarianism", "constitution", "geopolitics",
        "hezbollah cease-fire", "ukraine aid congress",
        "disinformation elections", "19th century disinfo",
        "austerity government spending",
        "security clearance", "camp century cold war",
        "criminal justice violence data",
        "charter school atlas network",
        "government data transparency corruption",
        "censorship germany",
        # Military/defense budget
        "defense budget", "defense spending", "military spending",
        "china defense budget", "china military budget",
        "ukraine aid", "congress ukraine",
        "camp century greenland",
    ],
    "Society & Culture": [
        "social inequality", "gender equality", "education reform",
        "poverty", "human rights", "democracy", "freedom of speech",
        "history", "migration", "diversity inclusion",
        "aging population", "media literacy", "journalism standards",
        "disabled musicians", "foster care",
        "academic peer review culture",
        "young women discrimination",
        "communism classless", "vh1 late night",
        "tiktok election romania",
        "evidence-based policymaking",
        "patient public involvement research",
    ],
}

# ── Hard-disqualify patterns ───────────────────────────────────────────────
NOISE_SIGNALS = [
    # Gaming
    r"\b(d&d|dnd|dungeon master|warlock|wizard|paladin|elf|dwarf|orc|bard|hexer)\b",
    r"\b(mtg|magic the gathering|commander deck|planeswalker|mana|counterspell|edh)\b",
    r"\b(minecraft|fortnite|roblox|pokemon|genshin|valorant|elden ring|skyrim|baldur)\b",
    r"\bstalker 2\b",
    # Anime/fandom
    r"\b(anime|manga|waifu|otaku|jujutsu|naruto|one piece|demon slayer|headcanon|fandom)\b",
    # Sports play-by-play
    r"\b(touchdown|quarterback|dribble|hat trick|offside|pitcher|shortstop|lamine yamal)\b",
    r"\b(i support (liverpool|arsenal|chelsea|manchester|celtic|rangers) fc)\b",
    # Astrology
    r"\b(astrology|zodiac|mercury retrograde|birth chart|leo rising|scorpio moon|virgo sun)\b",
    # Crypto gambling language
    r"\b(crypto pump|to the moon|hodl|ape in|nft drop|whitelist|memecoin)\b",
    r"\b(bitcoin.*100k|100k.*bitcoin|6 million.*banana|crypto bro.*banana)\b",
    # Birdwatching logs
    r"\b(warbler|firecrest|twitching)\b",
    # Personal fundraising
    r"\b(fundraise\.|cancer research uk.*swim|swim \d+k challenge|another swim done)\b",
    # Scam warnings
    r"\b(impersonating @|block and report|paypal scam|always check the profil)\b",
    # Spam / marketing
    r"(click here|subscribe now|buy now|coupon|discount|affiliate)",
    r"bit\.ly|tinyurl|amzn\.to|goo\.gl|bityl\.co|ecomerit\.ie",
    r"\b(ecomerit|ecoaudit)\b",
    # Intro / self-intro posts
    r"^(hello|hi|hey|greetings).{0,80}(bluesky|bsky|account|follow me|this is my|i'm a.*researcher)",
    r"^this is the (bsky|bluesky) account for",
    r"^intro post",
    r"^hi everyone[!.]?\s*(i'?m|check out|i am)",
    # SCA / historical re-enactment
    r"\b(sca|medieval combat|fiore.*1409)\b",
    # Job postings / recruitment
    r"\b(postdoctoral.*position|fully.funded position|phd position|lab technician.*position)\b",
    r"\b(deadline \d+ (dec|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov))\b",
    r"^assistant professor.*@",
    r"\bwe are.{0,30}(hiring|recruiting)\b",
    r"\b(we're a lab group based at|our lab focuses on)\b",
    # Conference / event promo
    r"^(join us|join me).{0,60}(hear|listen|next week|webinar|session|thurs|tomorrow)",
    r"^⏰ reminder",
    r"\bbright & early at \d",
    # Petition links
    r"resist\.bot/petitions",
    # Butterfly / ecotour ads
    r"\b(butterfly tour|birds of paradise.*tour)\b",
    # Virtual demonstration activist
    r"^📢 virtual demonstration",
    # Partisan rants
    r"\bbeing a republican.{0,60}(bad for|your health|deaths)\b",
    r"\bgop districts.{0,60}(bad for|deaths|health)\b",
    r"\b(pete hegseth dropped a bombshell|hegseth.*fox.*earlier)\b",
    r"\b(rachel maddow|grifter from msnbc)\b",
    r"\b(maga is not crying|maga is crying|trump pp|own the libs)\b",
    r"\b(edolf muskler|elon.*dismantling the government)\b",
    r"trump.{0,30}tariffs.{0,50}intentionally raise prices",
    r"(from gasoline to groceries.{0,40}trump|trump.{0,40}dangerous tariff threats)",
    r"\bwe need to keep putting pressure on.{0,20}🗑️",
    r"\b(stain only had 48%|blue dot resistance|ms daisy and my crew)\b",
    r"(world.leading 3 percent.*gdp|gdp.*world.leading 3 percent|economy is defying predictions)",
    # Breaking news / crime / prison stories
    r"^🚨.{0,20}breaking news",
    r"\b(released from prison|helped hitman|convicted for murder|sentenced to)\b",
    r"netflix star.*escaped.*prison.*smuggling",
    # Personal commentary / rant
    r"^time for an irrelevant moan",
    r"^good morning.*ms daisy",
    r"(continuing.*blue dot resistance)",
    # Duplicate story seeds
    r"(nepali.*gpt.*chatgpt|nepali gpt.*10k.*conversations)",
    r"^calling (the|for) nepali",
    # Lab / account intro
    r"^we're a lab group based at",
    r"^hi everyone.{0,50}are you interested in",
    # Awareness-month activist posts (not educational insight)
    r"^(november|october|january|february|march|april|may|june|july|august|september|december) is (islamophobia|domestic violence|autism|cancer|mental health)",
    # Personal career/research threads
    r"^a thread on (some|my) topics of my.*(career|research)",
    # Social role-call posts
    r"^can we do a #.*role.?call",
    # Meeting / conference attendance announcements
    r"^inspiring meeting last week",
    # Job listings / graduate recruitment
    r"\b(two postdoctoral|fully.funded positions? supported by|postdoctoral.*lab technician)\b",
    r"^there are still opportunities for (msc|phd|msc/phd) projects",
    # Personal research update (no insight)
    r"^never in my life have i done a benchmarking study",
    # Podcast / talk show promos
    r"always great to join @.{5,50}to talk the latest",
    # Politician soundbite quotes
    r"^reynolds: '(there are|we need|insists)",
    # Vague price commentary
    r"^i saw a post yesterday about the prices",
    # Resource listings
    r"^excellent resource for.{0,60}courses",
    # HuggingFace AI consent objection (personal, no insight)
    r"^1/ hey @hf\.co no\.",
    # GDPR copy-paste instructions
    r"tupped.*kindly written a sample email",
    # Stock picking
    r"^i enjoy owning.*shares\. despite",
    r"^finally got a chance to play with duckdb",
    # Abbreviation-heavy unclear posts
    r"^imp\. to study transnational, cascade effects",
    # Sports news misrouted
    r"lamine yamal",
    # Product reviews
    r"^asus (zenbook|vivobook|zenfone|zenscreen).{0,30}review",
    r"^(it'?s not everyday that i'?m excited about a new piece of tech)",
    # LinkedIn / social media snark (no insight)
    r"^i read that more than \d+% of content on linkedin was ai",
    # Partisan emotional rant about economy
    r"^seems like it'?s dropping every day.{0,30}(all because|invasion)",
    # Personal re-reading academic intro (not educational for general audience)
    r"^having just re-read the introduction to.{0,80}(eds|ed\.)",
    # Partisan SmartNews article links with editorializing
    r"l\.smartnews\.com",
    r"\b(this guy is a war monger|neo con|neocon warmonger)\b",
    # Simplistic political "both sides" commentary
    r"^i swear the concept of government.{0,30}(sides|parties) is dumb",
    r"(body of this bird is basically wall st|right wing or left wing it'?s all part of the same bird)",
    # Music / album retrospective (not educational)
    r"^revisiting .{0,50}(album|has given me|gave me).{0,50}(joy|feels|emotions)",
    r"\b(deathconsciousness|godspeed you|swans|mogwai|portishead)\b",
]

# Patterns that reduce score heavily but don't hard-disqualify
ANNOUNCEMENT_PATTERNS = [
    r"\b(new paper|our new paper|my new paper|our latest paper|just published|just launched)\b",
    r"\b(i am happy to share|delighted to share|proud to share|proud to be coauthor)\b",
    r"\b(we are recruiting|now recruiting|take the survey|share this survey)\b",
    r"\b(new report from us|we've published|we published)\b",
    r"\b(launching our report|lunching our report)\b",
    r"\b(check our review paper|check out our report)\b",
    r"\b(new feed for|i have started a new feed)\b",
    r"\b(must-follow|join their)\b",
    r"\b(visit.*for more|learn more at|read more at|read the full report at)\b",
]

# Same-event fingerprints — only keep the first occurrence
STORY_FINGERPRINTS = [
    # Ukraine $24B — match regardless of word order
    r"\$24 billion.{0,80}ukraine|ukraine.{0,80}\$24 billion",
    r"biden.{0,80}\$24 billion|24 billion.{0,80}ukraine",
    r"\$106 billion.{0,60}ukraine|\$175 billion.{0,60}ukraine",
    # GM tariff
    r"general motors.{0,80}tariff|gm.{0,50}(lost|lose).{0,50}market cap",
    # NepaliGPT
    r"nepali.{0,10}gpt.{0,60}chatgpt",
    # OpenAI SoftBank
    r"softbank.{0,60}openai.{0,60}\$1\.5 billion",
    # Bhattacharya (3 posts on same person/topic — keep only first)
    r"bhattacharya.{0,80}(santa clara|antibodies|herd immunity|nih director|flawed|debunked)",
    r"(herd immunity.{0,40}author|santa clara antibodies study)",
    # HuggingFace AI consent (two near-identical posts)
    r"hey @hf\.co.{0,30}(no|consent|object)",
]

# TED-quality positive signals
QUALITY_PATTERNS = [
    (r"\b(research|study|studies|findings|data|evidence|analysis)\b", 3),
    (r"\b(discovered|reveals?|shows?|demonstrates?|indicates?|suggests?|found that)\b", 2),
    (r"\b(because|therefore|however|although|despite|whereas|thus|yet|nonetheless)\b", 2),
    (r"\b(\d+[\.,]?\d*\s*(%|percent|billion|million|trillion))\b", 3),
    (r"\b(historically|century|decades?|years? ago|since \d{4})\b", 2),
    (r"\b(fascinating|remarkable|surprising|counterintuitive|paradox|irony)\b", 2),
    (r"\b(innovation|breakthrough|discovery|insight|impact|implication|consequence)\b", 2),
    (r"\b(according to|scientists?|researchers?|experts?|studies show|analysis shows)\b", 3),
    (r"\b(published in|peer.reviewed|clinical trial|cohort study)\b", 3),
    (r"\b(most people|we often|humans tend|turns out|the reality is|it turns out)\b", 2),
    (r"\b(policy|legislation|reform|regulation|governance|public health)\b", 2),
    (r"\b(risk|benefit|challenge|opportunity|trade-off|unintended consequence)\b", 1),
    (r"\d+\s*(percent|%)\s*(of|more|less|higher|lower|increase|decrease)", 3),
    (r"\$([\d,.]+)\s*(billion|million|trillion)", 2),
    # Genuine insight markers
    (r"\b(explains?|illustrates?|reveals?|highlights?|underscores?|demonstrates?)\b", 1),
    (r"\b(what we know|what the data shows|new evidence|surprising finding)\b", 3),
]


def classify_category(text: str) -> str:
    text_lower = text.lower()
    scores = defaultdict(float)
    for cat, keywords in CATEGORIES.items():
        for kw in keywords:
            if kw in text_lower:
                # Longer keywords are more specific = higher weight
                scores[cat] += 0.5 + len(kw) / 20.0

    if not scores:
        return "Society & Culture"

    max_score = max(scores.values())
    # Among tied categories, prefer higher-priority ones
    for cat in CATEGORY_PRIORITY:
        if scores.get(cat, 0) >= max_score * 0.85:
            return cat
    return max(scores, key=scores.get)


def score_post(text: str) -> float:
    text_lower = text.lower()

    for pattern in NOISE_SIGNALS:
        if re.search(pattern, text_lower, re.I):
            return -1.0

    score = 0.0
    score += len(text) / 100.0

    for pattern, weight in QUALITY_PATTERNS:
        matches = len(re.findall(pattern, text_lower))
        score += min(matches, 3) * weight

    # Penalize announcements of own work
    for pattern in ANNOUNCEMENT_PATTERNS:
        if re.search(pattern, text_lower, re.I):
            score -= 2.5

    score -= len(re.findall(r"#\w+", text)) * 0.5
    score -= len(re.findall(r"[\U0001F300-\U0001FFFF]", text)) * 0.3
    score -= len(re.findall(r"https?://", text)) * 1.5
    score -= len(re.findall(r"\b[A-Z]{4,}\b", text)) * 0.5
    score -= len(re.findall(r"@[\w.]+", text)) * 0.5

    if re.search(r"[.!?]\s*$", text.strip()):
        score += 1.5

    return score


def get_story_fingerprint(text: str) -> str | None:
    text_lower = text.lower()
    for fp in STORY_FINGERPRINTS:
        if re.search(fp, text_lower):
            return fp
    return None


def deduplicate(posts: list) -> list:
    seen_text = set()
    seen_story = set()
    unique = []
    for s, obj in posts:
        # Exact text dedup
        key = re.sub(r"\s+", " ", obj["text"][:130]).strip().lower()
        if key in seen_text:
            continue
        # Same-story dedup
        story = get_story_fingerprint(obj["text"])
        if story and story in seen_story:
            continue
        seen_text.add(key)
        if story:
            seen_story.add(story)
        unique.append((s, obj))
    return unique


def main():
    posts = []

    with open("master_dataset.jsonl") as f:
        for line in f:
            obj = json.loads(line.strip())
            text = obj.get("text", "")

            if obj.get("reply_to"):
                continue
            if text.startswith("@"):
                continue
            if len(text) < 260:
                continue
            if len(re.findall(r"#\w+", text)) > 4:
                continue
            if len(re.findall(r"https?://", text)) > 1:
                continue

            s = score_post(text)
            if s < 0:
                continue

            posts.append((s, obj))

    print(f"Scored candidates: {len(posts)}")
    posts.sort(key=lambda x: -x[0])
    posts = deduplicate(posts)
    print(f"After dedup: {len(posts)}")

    # Select top 100 with category diversity (max 15 per category)
    selected = []
    cat_counts = defaultdict(int)
    CAT_LIMIT = 15

    for score, obj in posts:
        correct_cat = classify_category(obj["text"])
        if cat_counts[correct_cat] >= CAT_LIMIT:
            continue
        new_obj = {k: v for k, v in obj.items() if k != "category"}
        new_obj["category"] = correct_cat
        new_obj["quality_score"] = round(score, 3)
        selected.append(new_obj)
        cat_counts[correct_cat] += 1
        if len(selected) == 100:
            break

    # Fill remaining without cap if somehow short
    if len(selected) < 100:
        used_uris = {o["uri"] for o in selected}
        for score, obj in posts:
            if obj["uri"] in used_uris:
                continue
            correct_cat = classify_category(obj["text"])
            new_obj = {k: v for k, v in obj.items() if k != "category"}
            new_obj["category"] = correct_cat
            new_obj["quality_score"] = round(score, 3)
            selected.append(new_obj)
            if len(selected) == 100:
                break

    print(f"\nSelected: {len(selected)} posts")
    print("\nCategory distribution:")
    dist = defaultdict(int)
    for o in selected:
        dist[o["category"]] += 1
    for cat, count in sorted(dist.items(), key=lambda x: -x[1]):
        print(f"  {count:3d}  {cat}")

    print("\nTop 15 scoring posts:")
    for o in selected[:15]:
        print(f"  [{o['quality_score']:5.2f}] [{o['category']}] {repr(o['text'][:200])}")
        print()

    print("\nBottom 15 scoring posts:")
    for o in selected[-15:]:
        print(f"  [{o['quality_score']:5.2f}] [{o['category']}] {repr(o['text'][:200])}")
        print()

    with open("top100_ted_posts.jsonl", "w") as f:
        for o in selected:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")

    print("Written to top100_ted_posts.jsonl")


if __name__ == "__main__":
    main()
