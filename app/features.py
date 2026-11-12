"""Refined version for reducing similarity between non similar objects."""

"""
Matching / similarity module (drop-in replacement for features.py).

Public functions, arguments and return shapes are unchanged:
    tokens, text_vector, cosine_text, extract_image_features, histogram_similarity,
    image_similarity, report_dict, candidate_matches   (+ optional warm_up())

What this version does
  * Image evidence : CLIP embeddings of the whole photo AND of the main object
                     (background removed with rembg), blended with the colour/tone/hash baseline.
  * Text evidence  : Sentence-BERT meaning similarity blended with the word-count baseline.
  * Cross-modal    : CLIP text-vs-photo check (lost description vs found photo, and vice versa).
  * Time           : a found item dated before the loss is penalised; a quick find is rewarded.
  * Location       : similar locations give a small bonus (no penalty, because items get moved).
  * Safe fallbacks : if rembg / sentence-transformers are missing or cannot load, each part is
                     skipped and the module keeps working with what is available.
  * Older stored reports (no CLIP / object vectors) still work with the baseline.

candidate_matches() returns the same keys as before, plus extra ones:
    cross_score, time_adjustment, location_adjustment, attribute_adjustment.

A high score means "possible candidate". It is not proof of ownership.
"""
import difflib
import io
import json
import logging
import math
import re
import sqlite3
import threading
from collections import Counter
from datetime import datetime

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from .config import MATCH_MIN_SCORE
from .database import db

log = logging.getLogger(__name__)

# ================================================================ tunable settings
TEXT_MODEL_NAME = "all-MiniLM-L6-v2"
IMAGE_MODEL_NAME = "clip-ViT-B-32"
SEGMENT_MODEL = "u2net"            # rembg model used to find the main object

# ---- object isolation (rembg)
SEGMENT_MAX_SIDE = 512             # photo is shrunk to this before segmentation (speed)
SEGMENT_MIN_FRAC = 0.03            # object smaller than 3% of the photo -> segmentation ignored
SEGMENT_MAX_FRAC = 0.92            # object covering >92% -> nothing to remove, ignored
SEGMENT_PAD = 0.04                 # margin kept around the object

# ---- image score = CLIP_WEIGHT * clip + (1 - CLIP_WEIGHT) * histogram baseline
CLIP_WEIGHT = 0.80
CLIP_OBJ_SHARE = 0.65              # when both photos have an object crop: 65% object, 35% whole photo
CLIP_COS_FLOOR, CLIP_COS_CEIL = 0.55, 0.95   # stretch image-image cosine to 0..1 (related-but-different objects score low)

# ---- text score = SEMANTIC_WEIGHT * sbert + (1 - SEMANTIC_WEIGHT) * word-count cosine
SEMANTIC_WEIGHT = 0.70
SBERT_COS_FLOOR = 0.30         # same kind of item (headphones/earbuds) must not score as a match

# ---- cross-modal (CLIP text vs photo). Text-image cosines are small, so use a lower band.
CROSS_COS_FLOOR, CROSS_COS_CEIL = 0.18, 0.33
CLIP_TEXT_MAX_CHARS = 200          # CLIP's text side only reads ~77 tokens

# ---- final fusion. With cross-modal evidence available / not available:
FUSION_WITH_CROSS = {"image": 0.45, "text": 0.30, "cross": 0.17, "category": 0.08}
FUSION_NO_CROSS = {"image": 0.55, "text": 0.37, "category": 0.08}      # previous weights

# ---- time plausibility (days = found date - lost date)
TIME_SLACK_DAYS = 1                # dates are approximate: allow 1 day of slack
TIME_FOUND_BEFORE_LOST = -0.15     # found clearly before it was lost
TIME_QUICK_FIND_DAYS, TIME_QUICK_FIND_BONUS = 7, 0.03
TIME_LATE_DAYS, TIME_LATE_PENALTY = 30, -0.05
TIME_VERY_LATE_DAYS, TIME_VERY_LATE_PENALTY = 90, -0.10

# ---- location closeness (bonus only: a lost item is often moved before it is found)
LOCATION_STRONG, LOCATION_STRONG_BONUS = 0.80, 0.04
LOCATION_PARTIAL, LOCATION_PARTIAL_BONUS = 0.50, 0.02

# ---- bonuses only count when the item evidence is already plausible:
#      gate = 0 at/below ADJ_GATE_LOW, 1 at/above ADJ_GATE_HIGH. Penalties are never gated.
ADJ_GATE_LOW, ADJ_GATE_HIGH = 0.40, 0.60

# ---- look-alike protection: facts the reporters stated that contradict each other
CATEGORY_MISMATCH_PENALTY = -0.06   # different categories (neither is "other")
BRAND_MISMATCH_PENALTY = -0.10      # both brands given and different
COLOUR_MISMATCH_PENALTY = -0.05     # both colours given and no colour word in common

TEXT_CACHE_MAX = 4096

# ================================================================ text preprocessing (baseline)
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
STOP = {"the", "and", "with", "for", "from", "near", "was", "were", "have", "has", "its",
        "this", "that", "item", "found", "lost", "nearby", "at", "in", "on", "a", "an", "of",
        "to", "my", "is", "it"}


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def tokens(text: str) -> list[str]:
    raw = re.findall(r"[a-z0-9]+", text.lower())
    return [SYNONYMS.get(t, t) for t in raw if t not in STOP and len(t) > 1]


def text_vector(text: str) -> Counter:
    ts = tokens(text)
    terms = Counter(ts)
    terms.update(f"{a}_{b}" for a, b in zip(ts, ts[1:]))
    return terms


# ================================================================ model loading (lazy, safe)
_models: dict = {}
_failed: set = set()
_model_lock = threading.Lock()
_segmenter = {"session": None, "failed": False}


def _get_model(name: str):
    """Load a sentence-transformers model once; None (remembered) if it cannot be loaded."""
    if name in _models:
        return _models[name]
    if name in _failed:
        return None
    with _model_lock:
        if name in _models:
            return _models[name]
        if name in _failed:
            return None
        try:
            from sentence_transformers import SentenceTransformer
            _models[name] = SentenceTransformer(name)
            log.info("Loaded similarity model %s", name)
        except Exception as exc:  # missing package, no internet, out of memory ...
            _failed.add(name)
            log.warning("Could not load %s (%s). Falling back to the baseline method.", name, exc)
            return None
    return _models[name]


def _get_segmenter():
    """Load the rembg session once; None (remembered) if rembg is unavailable."""
    if _segmenter["session"] is not None:
        return _segmenter["session"]
    if _segmenter["failed"]:
        return None
    with _model_lock:
        if _segmenter["session"] is not None:
            return _segmenter["session"]
        if _segmenter["failed"]:
            return None
        try:
            from rembg import new_session
            _segmenter["session"] = new_session(SEGMENT_MODEL)
            log.info("Loaded object segmentation model %s", SEGMENT_MODEL)
        except Exception as exc:
            _segmenter["failed"] = True
            log.warning("Object isolation unavailable (%s). Using whole photos only.", exc)
            return None
    return _segmenter["session"]


def warm_up() -> None:
    """Optional: call once at start-up so the first request is not slow."""
    _get_model(TEXT_MODEL_NAME)
    _get_model(IMAGE_MODEL_NAME)
    _get_segmenter()


# ================================================================ text similarity
_text_cache: dict = {}
_clip_text_cache: dict = {}
_cache_lock = threading.Lock()


def _normalise_text(text: str) -> str:
    return " ".join(text.lower().split())


def _cached_encode(model_name: str, cache: dict, texts: list[str], prefix: str = ""):
    """Batch-encode texts with a normalised-embedding model, using a cache. None on failure."""
    model = _get_model(model_name)
    if model is None:
        return None
    keys = [_normalise_text(t) for t in texts]
    with _cache_lock:
        known = {k: cache[k] for k in set(keys) if k in cache}
    missing = [k for k in dict.fromkeys(keys) if k not in known]
    if missing:
        try:
            vecs = model.encode([prefix + k for k in missing], batch_size=32,
                                convert_to_numpy=True, normalize_embeddings=True)
        except Exception as exc:
            log.warning("Text encoding with %s failed (%s).", model_name, exc)
            return None
        for k, v in zip(missing, vecs):
            known[k] = v
        with _cache_lock:
            if len(cache) + len(missing) > TEXT_CACHE_MAX:
                cache.clear()
            cache.update({k: known[k] for k in missing})
    return [known[k] for k in keys]


def _text_embeddings(texts: list[str]):
    """One normalised SBERT vector per text (batch-encoded, cached), or None."""
    return _cached_encode(TEXT_MODEL_NAME, _text_cache, texts)


def _clip_text_embeddings(texts: list[str]):
    """One normalised CLIP text vector per text (for the cross-modal score), or None."""
    short = [t[:CLIP_TEXT_MAX_CHARS] for t in texts]
    return _cached_encode(IMAGE_MODEL_NAME, _clip_text_cache, short, prefix="a photo of ")


def _lexical_cosine(a: str, b: str) -> float:
    va, vb = text_vector(a), text_vector(b)
    if not va or not vb:
        return 0.0
    dot = sum(v * vb.get(k, 0) for k, v in va.items())
    denom = math.sqrt(sum(v * v for v in va.values()) * sum(v * v for v in vb.values()))
    return dot / denom if denom else 0.0


def _semantic_cosine(a: str, b: str):
    if not a.strip() or not b.strip():
        return 0.0
    vecs = _text_embeddings([a, b])
    if vecs is None:
        return None
    cos = float(np.dot(vecs[0], vecs[1]))
    return _clamp01((cos - SBERT_COS_FLOOR) / (1.0 - SBERT_COS_FLOOR))


def cosine_text(a: str, b: str) -> float:
    lexical = _lexical_cosine(a, b)
    semantic = _semantic_cosine(a, b)
    if semantic is None:
        return lexical
    return _clamp01(SEMANTIC_WEIGHT * semantic + (1 - SEMANTIC_WEIGHT) * lexical)


# ================================================================ image features
def _object_crop(im: Image.Image):
    """Cut out the main object (rembg), crop to it and place it on white. None if not reliable."""
    session = _get_segmenter()
    if session is None:
        return None
    try:
        from rembg import remove
        work = im.copy()
        work.thumbnail((SEGMENT_MAX_SIDE, SEGMENT_MAX_SIDE))
        cut = remove(work, session=session).convert("RGBA")
        mask = np.asarray(cut.split()[-1]) > 127
        frac = float(mask.mean())
        if frac < SEGMENT_MIN_FRAC or frac > SEGMENT_MAX_FRAC:
            return None                      # nothing found, or the "object" is the whole photo
        ys, xs = np.where(mask)
        pad_x = int(SEGMENT_PAD * work.width)
        pad_y = int(SEGMENT_PAD * work.height)
        box = (max(xs.min() - pad_x, 0), max(ys.min() - pad_y, 0),
               min(xs.max() + 1 + pad_x, work.width), min(ys.max() + 1 + pad_y, work.height))
        crop = cut.crop(box)
        canvas = Image.new("RGB", crop.size, (255, 255, 255))
        canvas.paste(crop, mask=crop.split()[-1])
        return canvas
    except Exception as exc:
        log.warning("Object isolation failed (%s); using the whole photo.", exc)
        return None


def _clip_embedding(im: Image.Image):
    model = _get_model(IMAGE_MODEL_NAME)
    if model is None:
        return None
    try:
        vec = model.encode(im, convert_to_numpy=True, normalize_embeddings=True)
        return [round(float(v), 5) for v in vec]
    except Exception as exc:
        log.warning("CLIP encoding failed (%s); storing baseline features only.", exc)
        return None


def extract_image_features(raw: bytes) -> tuple[dict, Image.Image]:
    try:
        im = Image.open(io.BytesIO(raw))
        im = ImageOps.exif_transpose(im).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as e:
        raise ValueError("The uploaded file is not a readable image.") from e
    full = im.copy()                 # kept for CLIP / segmentation
    im.thumbnail((256, 256))
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
    bits = [1 if px[y * 9 + x] > px[y * 9 + x + 1] else 0 for y in range(8) for x in range(8)]
    features = {"rgb": rgb, "gray": gray, "ahash": bits, "extractor": "histogram-ahash-v1"}

    clip_full = _clip_embedding(full)
    if clip_full is not None:
        features["clip"] = clip_full                       # whole photo (same key as before)
        features["extractor"] = "clip-vit-b32+histogram-v2"
        obj = _object_crop(full)
        if obj is not None:
            obj_vec = _clip_embedding(obj)
            if obj_vec is not None:
                features["clip_obj"] = obj_vec             # main object, background removed
                features["extractor"] = "clip-vit-b32+rembg+histogram-v3"
    return features, im


# ================================================================ image similarity
def histogram_similarity(a: list[float], b: list[float]) -> float:
    # Histogram intersection divided by total histogram mass, giving a true 0..1 score.
    # The RGB histogram holds three channels (mass 3); the grayscale one holds one (mass 1).
    overlap = sum(min(x, y) for x, y in zip(a, b))
    mass = max(min(sum(a), sum(b)), 1e-9)
    return _clamp01(overlap / mass)


def _baseline_image_similarity(a: dict, b: dict) -> float:
    colour = histogram_similarity(a["rgb"], b["rgb"])
    tone = histogram_similarity(a["gray"], b["gray"])
    ahash = 1 - sum(x != y for x, y in zip(a["ahash"], b["ahash"])) / 64
    return _clamp01(0.62 * colour + 0.20 * tone + 0.18 * ahash)


def _scaled_cosine(va, vb):
    if not va or not vb or len(va) != len(vb):
        return None
    va, vb = np.asarray(va, dtype=float), np.asarray(vb, dtype=float)
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    if denom == 0:
        return None
    return _clamp01((float(np.dot(va, vb)) / denom - CLIP_COS_FLOOR) / (CLIP_COS_CEIL - CLIP_COS_FLOOR))


def _clip_similarity(a: dict, b: dict):
    full = _scaled_cosine(a.get("clip"), b.get("clip"))
    obj = _scaled_cosine(a.get("clip_obj"), b.get("clip_obj"))
    if full is None and obj is None:
        return None                      # older records without CLIP vectors
    if obj is None:
        return full
    if full is None:
        return obj
    return CLIP_OBJ_SHARE * obj + (1 - CLIP_OBJ_SHARE) * full


def image_similarity(a: dict, b: dict) -> float:
    baseline = _baseline_image_similarity(a, b)
    clip = _clip_similarity(a, b)
    if clip is None:
        return baseline
    return _clamp01(CLIP_WEIGHT * clip + (1 - CLIP_WEIGHT) * baseline)


# ================================================================ cross-modal (text <-> photo)
def _photo_vector(features: dict):
    vec = features.get("clip_obj") or features.get("clip")
    if not vec:
        return None
    vec = np.asarray(vec, dtype=float)
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm else None


def _cross_modal_score(query_text_vec, query_feats: dict, other_text_vec, other_feats: dict):
    """Average of (query text vs other photo) and (other text vs query photo), scaled to 0..1."""
    sims = []
    if query_text_vec is not None:
        other_photo = _photo_vector(other_feats)
        if other_photo is not None and len(other_photo) == len(query_text_vec):
            sims.append(float(np.dot(query_text_vec, other_photo)))
    if other_text_vec is not None:
        query_photo = _photo_vector(query_feats)
        if query_photo is not None and len(query_photo) == len(other_text_vec):
            sims.append(float(np.dot(other_text_vec, query_photo)))
    if not sims:
        return None
    return _clamp01((float(np.mean(sims)) - CROSS_COS_FLOOR) / (CROSS_COS_CEIL - CROSS_COS_FLOOR))


# ================================================================ time and location signals
def _parse_date(value):
    if not value:
        return None
    text = str(value).strip()
    try:
        return datetime.fromisoformat(text).date()
    except ValueError:
        pass
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%m/%d/%Y", "%d %b %Y", "%d %B %Y"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


def time_adjustment(lost_date, found_date) -> float:
    """Signed score adjustment from how the lost and found dates relate. 0 if a date is unknown."""
    if lost_date is None or found_date is None:
        return 0.0
    days = (found_date - lost_date).days
    if days < -TIME_SLACK_DAYS:
        return TIME_FOUND_BEFORE_LOST        # found before it was lost: very unlikely
    if days <= TIME_QUICK_FIND_DAYS:
        return TIME_QUICK_FIND_BONUS
    if days <= TIME_LATE_DAYS:
        return 0.0
    if days <= TIME_VERY_LATE_DAYS:
        return TIME_LATE_PENALTY
    return TIME_VERY_LATE_PENALTY


_LOCATION_STOP = {"the", "near", "in", "at", "of", "on", "outside", "inside", "by", "to"}
_LOCATION_GENERIC = {"block", "building", "hall", "floor", "room", "campus", "area", "lab"}


def _location_tokens(text) -> set:
    return {t for t in re.findall(r"[a-z0-9]+", (text or "").lower()) if t not in _LOCATION_STOP}


def _same_token(x: str, y: str) -> bool:
    """Equal, or a close spelling (typo) for words of 5+ letters. Short codes like 'a'/'b' must match exactly."""
    if x == y:
        return True
    return len(x) >= 5 and len(y) >= 5 and difflib.SequenceMatcher(None, x, y).ratio() >= 0.85


def location_similarity(a, b) -> float:
    """0..1 closeness of two free-text locations. 0 if either is empty."""
    ta, tb = _location_tokens(a), _location_tokens(b)
    if not ta or not tb:
        return 0.0
    matched_a = {x for x in ta if any(_same_token(x, y) for y in tb)}
    matched_b = {y for y in tb if any(_same_token(x, y) for x in ta)}
    union = len(ta) + len(tb) - min(len(matched_a), len(matched_b))
    jaccard = min(len(matched_a), len(matched_b)) / union
    # One location name fully inside the other ("library" inside "central library").
    small_matched, small = (matched_a, ta) if len(ta) <= len(tb) else (matched_b, tb)
    contained = 1.0 if small_matched == small and (small - _LOCATION_GENERIC) else 0.0
    return max(jaccard, contained)


def location_adjustment(a, b) -> float:
    sim = location_similarity(a, b)
    if sim >= LOCATION_STRONG:
        return LOCATION_STRONG_BONUS
    if sim >= LOCATION_PARTIAL:
        return LOCATION_PARTIAL_BONUS
    return 0.0


# ================================================================ stated-attribute contradictions
def _word_set(text) -> set:
    return {SYNONYMS.get(t, t) for t in re.findall(r"[a-z0-9]+", (text or "").lower()) if t not in STOP}


def attribute_adjustment(query, other) -> float:
    """Penalty when stated category / brand / colour clearly disagree. Unknown values never penalise."""
    adj = 0.0
    qc, oc = (query["category"] or "").strip().lower(), (other["category"] or "").strip().lower()
    if qc and oc and qc != oc and "other" not in (qc, oc):
        adj += CATEGORY_MISMATCH_PENALTY
    qb, ob = _word_set(query["brand"]), _word_set(other["brand"])
    if qb and ob and not (qb & ob):
        adj += BRAND_MISMATCH_PENALTY
    qcol, ocol = _word_set(query["colour"]), _word_set(other["colour"])
    if qcol and ocol and not (qcol & ocol):
        adj += COLOUR_MISMATCH_PENALTY
    return adj


# ================================================================ reports and candidates
def report_dict(r: sqlite3.Row) -> dict:
    return {k: r[k] for k in ("id", "report_type", "title", "description", "category", "colour", "brand", "contact_method", "location", "event_date", "status", "created_at")} | {"image_url": f"/uploads/{r['image_path']}"}


def _combine(img: float, txt: float, cat: float, cross):
    w = FUSION_NO_CROSS if cross is None else FUSION_WITH_CROSS
    base = w["image"] * img + w["text"] * txt + w["category"] * cat
    if cross is not None:
        base += w["cross"] * cross
    return base


def candidate_matches(report_id: int) -> list[dict]:
    with db() as c:
        query = c.execute("SELECT * FROM reports WHERE id=?", (report_id,)).fetchone()
        if not query:
            raise KeyError("Report not found")
        opposite = "FOUND" if query["report_type"] == "LOST" else "LOST"
        rows = c.execute("SELECT * FROM reports WHERE report_type=? AND status IN ('OPEN','UNDER_REVIEW') AND id<>? ORDER BY created_at DESC", (opposite, report_id)).fetchall()
    qfeatures = json.loads(query["image_features"])
    qtext = " ".join([query["title"], query["description"], query["colour"], query["brand"]])
    ttexts = [" ".join([r["title"], r["description"], r["colour"], r["brand"]]) for r in rows]
    qdate = _parse_date(query["event_date"])
    query_is_lost = query["report_type"] == "LOST"

    clip_text = None
    if rows:
        _text_embeddings([qtext] + ttexts)              # one batch; later calls hit the cache
        clip_text = _clip_text_embeddings([qtext] + ttexts)

    results = []
    for i, (r, ttext) in enumerate(zip(rows, ttexts)):
        tfeatures = json.loads(r["image_features"])
        img = image_similarity(qfeatures, tfeatures)
        txt = cosine_text(qtext, ttext)
        cat = float(query["category"].lower() == r["category"].lower())
        cross = None
        if clip_text is not None:
            cross = _cross_modal_score(clip_text[0], qfeatures, clip_text[i + 1], tfeatures)

        rdate = _parse_date(r["event_date"])
        lost_d, found_d = (qdate, rdate) if query_is_lost else (rdate, qdate)
        t_adj = time_adjustment(lost_d, found_d)
        l_adj = location_adjustment(query["location"], r["location"])

        base = _combine(img, txt, cat, cross)
        a_adj = attribute_adjustment(query, r)
        # Time/location bonuses only count when the item evidence is already plausible.
        gate = _clamp01((base - ADJ_GATE_LOW) / (ADJ_GATE_HIGH - ADJ_GATE_LOW))
        t_adj = t_adj * gate if t_adj > 0 else t_adj
        l_adj = l_adj * gate
        final = _clamp01(base + t_adj + l_adj + a_adj)
        results.append({
            "report": report_dict(r), "image_score": round(img, 4), "text_score": round(txt, 4),
            "category_score": cat, "final_score": round(final, 4),
            "cross_score": None if cross is None else round(cross, 4),
            "time_adjustment": round(t_adj, 4), "location_adjustment": round(l_adj, 4),
            "attribute_adjustment": round(a_adj, 4),
        })
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