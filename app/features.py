import json
import math
import re
import sqlite3
from collections import Counter
from PIL import Image, ImageOps, UnidentifiedImageError
from .config import MATCH_MIN_SCORE
from .database import db
SYNONYMS = {
    "rucksack": "backpack", "knapsack": "backpack", "purse": "wallet",
    "cellphone": "phone", "mobile": "phone", "smartphone": "phone",
    "spectacles": "glasses", "eyeglasses": "glasses", "sneakers": "shoes",
    "trainers": "shoes", "jumper": "sweater", "hoodie": "sweater",
    "grey": "gray", "navy": "blue", "crimson": "red", "scarlet": "red",
    "beige": "tan", "cream": "white", "dark": "dark", "light": "light",
    "symbol": "logo", "emblem": "logo", "insignia": "logo",
    "scratched": "scratch", "scratches": "scratch", "stickered": "sticker",
}
STOP = {"the", "and", "with", "for", "from", "near", "was", "were", "have", "has", "its", "this", "that", "item", "found", "lost", "nearby", "at", "in", "on", "a", "an", "of", "to", "my", "is", "it"}

def tokens(text: str) -> list[str]:
    raw = re.findall(r"[a-z0-9]+", text.lower())
    return [SYNONYMS.get(t, t) for t in raw if t not in STOP and len(t) > 1]

def text_vector(text: str) -> Counter:
    ts = tokens(text)
    terms = Counter(ts)
    terms.update(f"{a}_{b}" for a, b in zip(ts, ts[1:]))
    return terms

def cosine_text(a: str, b: str) -> float:
    va, vb = text_vector(a), text_vector(b)
    if not va or not vb:
        return 0.0
    dot = sum(v * vb.get(k, 0) for k, v in va.items())
    denom = math.sqrt(sum(v*v for v in va.values()) * sum(v*v for v in vb.values()))
    return dot / denom if denom else 0.0

def extract_image_features(raw: bytes) -> tuple[dict, Image.Image]:
    try:
        im = Image.open(__import__('io').BytesIO(raw))
        im = ImageOps.exif_transpose(im).convert("RGB")
        im.thumbnail((256, 256))
    except (UnidentifiedImageError, OSError, ValueError) as e:
        raise ValueError("The uploaded file is not a readable image.") from e
    # 8 bins per RGB channel plus grayscale histogram preserve coarse colour/tonal cues.
    hist = im.histogram()
    rgb = []
    for channel in range(3):
        part = hist[channel * 256:(channel + 1) * 256]
        rgb.extend(sum(part[i:i + 32]) for i in range(0, 256, 32))
    total = max(sum(rgb[:8]), 1)
    rgb = [v / total for v in rgb]
    gray_hist = im.convert("L").histogram()
    gray = [sum(gray_hist[i:i + 16]) / max(im.width * im.height, 1) for i in range(0, 256, 16)]
    tiny = im.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
    px = list(tiny.tobytes())
    bits = [1 if px[y*9+x] > px[y*9+x+1] else 0 for y in range(8) for x in range(8)]
    features = {"rgb": rgb, "gray": gray, "ahash": bits, "extractor": "histogram-ahash-v1"}
    return features, im

def histogram_similarity(a: list[float], b: list[float]) -> float:
    # Histograms are normalized; intersection is robust to modest lighting changes.
    return max(0.0, min(1.0, sum(min(x, y) for x, y in zip(a, b))))

def image_similarity(a: dict, b: dict) -> float:
    colour = histogram_similarity(a["rgb"], b["rgb"])
    tone = histogram_similarity(a["gray"], b["gray"])
    ahash = 1 - sum(x != y for x, y in zip(a["ahash"], b["ahash"])) / 64
    return max(0.0, min(1.0, 0.62 * colour + 0.20 * tone + 0.18 * ahash))

def report_dict(r: sqlite3.Row) -> dict:
    return {k: r[k] for k in ("id", "report_type", "title", "description", "category", "colour", "brand", "contact_method", "location", "event_date", "status", "created_at")} | {"image_url": f"/uploads/{r['image_path']}"}

def candidate_matches(report_id: int) -> list[dict]:
    with db() as c:
        query = c.execute("SELECT * FROM reports WHERE id=?", (report_id,)).fetchone()
        if not query:
            raise KeyError("Report not found")
        opposite = "FOUND" if query["report_type"] == "LOST" else "LOST"
        rows = c.execute("SELECT * FROM reports WHERE report_type=? AND status IN ('OPEN','UNDER_REVIEW') AND id<>? ORDER BY created_at DESC", (opposite, report_id)).fetchall()
    qfeatures = json.loads(query["image_features"])
    qtext = " ".join([query["title"], query["description"], query["colour"], query["brand"]])
    results = []
    for r in rows:
        tfeatures = json.loads(r["image_features"])
        img = image_similarity(qfeatures, tfeatures)
        ttext = " ".join([r["title"], r["description"], r["colour"], r["brand"]])
        txt = cosine_text(qtext, ttext)
        cat = float(query["category"].lower() == r["category"].lower())
        # Category is supporting evidence rather than a hard filter.
        final = 0.55 * img + 0.37 * txt + 0.08 * cat
        results.append({"report": report_dict(r), "image_score": round(img, 4), "text_score": round(txt, 4), "category_score": cat, "final_score": round(final, 4)})
    results = [result for result in results if result["final_score"] >= MATCH_MIN_SCORE]
    results = sorted(results, key=lambda x: x["final_score"], reverse=True)[:20]
    # Return the persistent review-state row used by the PATCH review endpoint.
    with db() as c:
        for result in results:
            other_id = result["report"]["id"]
            lost_id, found_id = (report_id, other_id) if query["report_type"] == "LOST" else (other_id, report_id)
            saved = c.execute("SELECT id,status FROM matches WHERE lost_report_id=? AND found_report_id=?", (lost_id, found_id)).fetchone()
            result["match_id"] = saved["id"] if saved else None
            result["match_status"] = saved["status"] if saved else "SUGGESTED"
    return results
