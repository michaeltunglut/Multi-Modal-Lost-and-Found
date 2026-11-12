"""
Evaluate the INITIAL (baseline) and REFINED matching modules separately on the same labelled
data and save every score, for the project report.

Setup
  app/features_initial.py   the initial baseline module (provided)
  app/features.py           the refined module
  app/evaluate_versions.py  this file

Run from the project root (it never touches your real database or uploads):

    python -m app.evaluate_versions --csv data/items.csv --images data/images
    python -m app.evaluate_versions --csv data/items.csv --images data/images --version initial
    python -m app.evaluate_versions --csv data/items.csv --images data/images --version refined

CSV columns (first four are required; the rest are optional but make the test realistic,
because the refined version also uses category, colour, brand, location and date):
    item_id,type,description,image_file,title,category,colour,brand,location,event_date
type = lost or found. The same item_id on a lost row and a found row = the SAME item.
Include look-alike items with DIFFERENT item_ids (headphones vs earbuds): they are the real test.

Output (default folder evaluation_results/)
    scores_<version>.csv      every lost x found pair with all component scores and the rank
    summary_<version>.json    metrics, timings and settings of that version
    summary.csv / summary.md  both versions side by side (built from the summary_*.json present)
    run_info.json             library versions, date, dataset size
    comparison.png            bar chart (only if matplotlib is installed)
"""
import argparse
import csv
import importlib
import importlib.metadata as md
import json
import os
import platform
import sqlite3
import statistics
import tempfile
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

os.environ.setdefault("LOST_FOUND_DATA", tempfile.mkdtemp(prefix="lf_eval_"))   # never use real data

MODULES = {"initial": "features_initial", "refined": "features"}
SETTING_NAMES = ["CLIP_WEIGHT", "CLIP_OBJ_SHARE", "CLIP_COS_FLOOR", "CLIP_COS_CEIL", "SEMANTIC_WEIGHT",
                 "SBERT_COS_FLOOR", "CROSS_COS_FLOOR", "CROSS_COS_CEIL", "FUSION_WITH_CROSS",
                 "FUSION_NO_CROSS", "ADJ_GATE_LOW", "ADJ_GATE_HIGH"]
SCHEMA = """
CREATE TABLE reports(id INTEGER PRIMARY KEY AUTOINCREMENT, report_type, title, description, category,
    colour, brand, contact_method, location, event_date, status, created_at, image_path, image_features);
CREATE TABLE matches(id INTEGER PRIMARY KEY AUTOINCREMENT, lost_report_id, found_report_id, status);
"""


# ------------------------------------------------------------------ data
def load_rows(csv_path):
    rows = []
    with open(csv_path, newline="", encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            r = {k.strip(): (v or "").strip() for k, v in r.items() if k}
            r["type"] = r["type"].lower()
            r.setdefault("title", "")
            if not r["title"]:
                r["title"] = " ".join(r["description"].split()[:5])
            rows.append(r)
    for need in ("item_id", "type", "description", "image_file"):
        if rows and need not in rows[0]:
            raise SystemExit(f"CSV is missing the column '{need}'.")
    if not any(r["type"] == "lost" for r in rows) or not any(r["type"] == "found" for r in rows):
        raise SystemExit("The CSV needs at least one 'lost' and one 'found' row.")
    return rows


def new_database():
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)

    @contextmanager
    def fake_db():
        yield conn
    return conn, fake_db


# ------------------------------------------------------------------ metrics
def _rank(true_score, other_scores):
    """Pessimistic rank: ties count against the true match, so tied scores never flatter a version."""
    return 1 + sum(1 for s in other_scores if s >= true_score)


def rank_metrics(pairs_by_lost, key, k=3):
    p1 = rk = rr = n = 0
    for pairs in pairs_by_lost.values():
        true = [p for p in pairs if p["same_item"]]
        if not true:
            continue
        t = true[0]
        if t[key] is None:
            continue
        others = [p[key] for p in pairs if p is not t and p[key] is not None]
        rank = _rank(t[key], others)
        n += 1
        p1 += rank == 1
        rk += rank <= k
        rr += 1.0 / rank
    if n == 0:
        return {"precision_at_1": None, "recall_at_3": None, "mrr": None, "n_queries": 0}
    return {"precision_at_1": p1 / n, f"recall_at_{k}": rk / n, "mrr": rr / n, "n_queries": n}


def pair_auc(all_pairs, key):
    """Chance that a random TRUE pair scores higher than a random FALSE pair (ties count 0.5)."""
    t = [p[key] for p in all_pairs if p["same_item"] and p[key] is not None]
    f = [p[key] for p in all_pairs if not p["same_item"] and p[key] is not None]
    if not t or not f:
        return None
    wins = sum(1.0 if a > b else 0.5 if a == b else 0.0 for a in t for b in f)
    return wins / (len(t) * len(f))


# ------------------------------------------------------------------ one version
def evaluate_version(name, rows, images_dir, cutoff):
    module = importlib.import_module(f".{MODULES[name]}", package=__package__)
    conn, fake_db = new_database()
    module.db = fake_db                 # the module reads reports from our temporary database
    module.MATCH_MIN_SCORE = -1.0       # return every candidate so every pair gets a score

    t0 = time.perf_counter()
    ids, n_clip, n_obj = {}, 0, 0
    for i, r in enumerate(rows):
        raw = (Path(images_dir) / r["image_file"]).read_bytes()
        feats, _ = module.extract_image_features(raw)
        n_clip += "clip" in feats
        n_obj += "clip_obj" in feats
        cur = conn.execute(
            "INSERT INTO reports(report_type,title,description,category,colour,brand,contact_method,location,"
            "event_date,status,created_at,image_path,image_features) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (r["type"].upper(), r["title"], r["description"], r.get("category", ""), r.get("colour", ""),
             r.get("brand", ""), "test", r.get("location", ""), r.get("event_date", ""), "OPEN",
             f"2026-01-01T00:00:{i % 60:02d}", r["image_file"], json.dumps(feats)))
        ids[cur.lastrowid] = r
    t_extract = time.perf_counter() - t0

    t0 = time.perf_counter()
    all_pairs, by_lost = [], {}
    for rid, r in ids.items():
        if r["type"] != "lost":
            continue
        results = module.candidate_matches(rid)
        pairs = []
        for res in results:
            fr = ids[res["report"]["id"]]
            pairs.append({
                "lost_item_id": r["item_id"], "lost_file": r["image_file"],
                "found_item_id": fr["item_id"], "found_file": fr["image_file"],
                "same_item": r["item_id"] == fr["item_id"],
                "image_score": res["image_score"], "text_score": res["text_score"],
                "category_score": res["category_score"], "cross_score": res.get("cross_score"),
                "time_adjustment": res.get("time_adjustment"),
                "location_adjustment": res.get("location_adjustment"),
                "attribute_adjustment": res.get("attribute_adjustment"),
                "final_score": res["final_score"]})
        for p in pairs:       # rank of this found item by the final score (pessimistic ties)
            p["rank_by_final"] = _rank(p["final_score"], [q["final_score"] for q in pairs if q is not p])
        by_lost[rid] = pairs
        all_pairs.extend(pairs)
    t_score = time.perf_counter() - t0

    true_final = [p["final_score"] for p in all_pairs if p["same_item"]]
    false_final = [p["final_score"] for p in all_pairs if not p["same_item"]]
    tp = sum(1 for s in true_final if s >= cutoff)
    fp = sum(1 for s in false_final if s >= cutoff)
    summary = {
        "version": name,
        "module": MODULES[name],
        "n_reports": len(rows),
        "n_lost": sum(r["type"] == "lost" for r in rows),
        "n_found": sum(r["type"] == "found" for r in rows),
        "n_true_pairs": len(true_final),
        "n_false_pairs": len(false_final),
        "final": rank_metrics(by_lost, "final_score"),
        "image_only": rank_metrics(by_lost, "image_score"),
        "text_only": rank_metrics(by_lost, "text_score"),
        "auc": {"final": pair_auc(all_pairs, "final_score"), "image": pair_auc(all_pairs, "image_score"),
                "text": pair_auc(all_pairs, "text_score")},
        "mean_true_final": statistics.mean(true_final) if true_final else None,
        "mean_false_final": statistics.mean(false_final) if false_final else None,
        "min_true_final": min(true_final) if true_final else None,
        "max_false_final": max(false_final) if false_final else None,
        "cutoff": cutoff,
        "true_pairs_above_cutoff": tp, "false_pairs_above_cutoff": fp,
        "precision_at_cutoff": tp / (tp + fp) if tp + fp else None,
        "recall_at_cutoff": tp / len(true_final) if true_final else None,
        "seconds_per_photo_features": t_extract / len(rows),
        "seconds_per_lost_query": t_score / max(summary_count(rows), 1),
        "photos_with_clip_vector": n_clip, "photos_with_object_crop": n_obj,
        "settings": {n: getattr(module, n) for n in SETTING_NAMES if hasattr(module, n)}
                    or {"fusion": "0.55*image + 0.37*text + 0.08*category"},
    }
    return summary, all_pairs


def summary_count(rows):
    return sum(r["type"] == "lost" for r in rows)


# ------------------------------------------------------------------ saving
PAIR_COLUMNS = ["lost_item_id", "lost_file", "found_item_id", "found_file", "same_item", "rank_by_final",
                "image_score", "text_score", "category_score", "cross_score", "time_adjustment",
                "location_adjustment", "attribute_adjustment", "final_score"]


def save_scores(path, pairs):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=PAIR_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for p in sorted(pairs, key=lambda p: (str(p["lost_item_id"]), p["rank_by_final"])):
            w.writerow(p)


def _fmt(v, pct=False):
    if v is None:
        return "n/a"
    return f"{v * 100:.1f}%" if pct else f"{v:.3f}"


ROWS = [  # label, getter, percent?
    ("Precision@1 (final score)", lambda s: s["final"]["precision_at_1"], True),
    ("Recall@3 (final score)", lambda s: s["final"].get("recall_at_3"), True),
    ("MRR (final score)", lambda s: s["final"]["mrr"], False),
    ("Precision@1 (image only)", lambda s: s["image_only"]["precision_at_1"], True),
    ("MRR (image only)", lambda s: s["image_only"]["mrr"], False),
    ("Precision@1 (text only)", lambda s: s["text_only"]["precision_at_1"], True),
    ("MRR (text only)", lambda s: s["text_only"]["mrr"], False),
    ("AUC (final score)", lambda s: s["auc"]["final"], False),
    ("AUC (image only)", lambda s: s["auc"]["image"], False),
    ("AUC (text only)", lambda s: s["auc"]["text"], False),
    ("Mean score, true pairs", lambda s: s["mean_true_final"], False),
    ("Mean score, false pairs", lambda s: s["mean_false_final"], False),
    ("Highest false-pair score", lambda s: s["max_false_final"], False),
    ("Lowest true-pair score", lambda s: s["min_true_final"], False),
    ("Precision at cutoff", lambda s: s["precision_at_cutoff"], True),
    ("Recall at cutoff", lambda s: s["recall_at_cutoff"], True),
    ("False pairs above cutoff", lambda s: s["false_pairs_above_cutoff"], None),
    ("True pairs above cutoff", lambda s: s["true_pairs_above_cutoff"], None),
    ("Seconds per photo (features)", lambda s: s["seconds_per_photo_features"], False),
    ("Seconds per lost query", lambda s: s["seconds_per_lost_query"], False),
]


def build_combined(out):
    summaries = {}
    for name in ("initial", "refined"):
        p = out / f"summary_{name}.json"
        if p.exists():
            summaries[name] = json.loads(p.read_text())
    if not summaries:
        return summaries
    names = list(summaries)
    with open(out / "summary.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["metric"] + names)
        for label, get, pct in ROWS:
            w.writerow([label] + [(_fmt(get(summaries[n]), pct) if pct is not None else get(summaries[n])) for n in names])
    lines = ["| Metric | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for label, get, pct in ROWS:
        vals = [(_fmt(get(summaries[n]), pct) if pct is not None else str(get(summaries[n]))) for n in names]
        lines.append(f"| {label} | " + " | ".join(vals) + " |")
    first = summaries[names[0]]
    lines += ["", f"Pairs: {first['n_true_pairs']} true, {first['n_false_pairs']} false "
                  f"({first['n_lost']} lost x {first['n_found']} found reports). Cutoff = {first['cutoff']}.",
              "Ties are ranked pessimistically (a tie counts against the true match)."]
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        metrics = [("P@1", lambda s: s["final"]["precision_at_1"]), ("Recall@3", lambda s: s["final"].get("recall_at_3")),
                   ("MRR", lambda s: s["final"]["mrr"]), ("AUC", lambda s: s["auc"]["final"])]
        width = 0.8 / len(names)
        fig, ax = plt.subplots(figsize=(7, 4))
        for j, n in enumerate(names):
            vals = [(g(summaries[n]) or 0) for _, g in metrics]
            ax.bar([x + j * width for x in range(len(metrics))], vals, width, label=n)
        ax.set_xticks([x + width * (len(names) - 1) / 2 for x in range(len(metrics))])
        ax.set_xticklabels([m for m, _ in metrics])
        ax.set_ylim(0, 1.05)
        ax.set_title("Initial vs refined matching (final score)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / "comparison.png", dpi=150)
    except Exception:
        pass
    return summaries


def package_versions():
    out = {}
    for pkg in ("numpy", "Pillow", "torch", "sentence-transformers", "rembg", "onnxruntime", "fastapi"):
        try:
            out[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            out[pkg] = "not installed"
    return out


# ------------------------------------------------------------------ main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default="data/items.csv")
    ap.add_argument("--images", default="data/images")
    ap.add_argument("--out", default="evaluation_results")
    ap.add_argument("--version", choices=["initial", "refined", "both"], default="both")
    ap.add_argument("--cutoff", type=float, default=0.60, help="app display cutoff (default 0.60)")
    a = ap.parse_args(argv)

    rows = load_rows(a.csv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    versions = ["initial", "refined"] if a.version == "both" else [a.version]

    for name in versions:
        print(f"\n=== Evaluating {name} ({MODULES[name]}.py) ===")
        summary, pairs = evaluate_version(name, rows, a.images, a.cutoff)
        if name == "refined" and summary["photos_with_clip_vector"] < summary["n_reports"]:
            print("WARNING: the refined module ran WITHOUT CLIP for some photos (models not installed or not "
                  "loaded). Those results reflect the fallback, not the refined method. Fix this before using them.")
        save_scores(out / f"scores_{name}.csv", pairs)
        (out / f"summary_{name}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        f = summary["final"]
        print(f"  Precision@1 {_fmt(f['precision_at_1'], True)} | Recall@3 {_fmt(f.get('recall_at_3'), True)} | "
              f"MRR {_fmt(f['mrr'])} | AUC {_fmt(summary['auc']['final'])}")
        print(f"  mean score: true pairs {_fmt(summary['mean_true_final'])}, false pairs {_fmt(summary['mean_false_final'])}")
        print(f"  at cutoff {a.cutoff}: {summary['true_pairs_above_cutoff']} true and "
              f"{summary['false_pairs_above_cutoff']} false pairs shown")

    (out / "run_info.json").write_text(json.dumps({
        "date": datetime.now().isoformat(timespec="seconds"), "python": platform.python_version(),
        "platform": platform.platform(), "packages": package_versions(), "csv": str(a.csv),
        "images": str(a.images), "reports": len(rows), "cutoff": a.cutoff}, indent=2), encoding="utf-8")
    summaries = build_combined(out)
    print(f"\nSaved to {out.resolve()}")
    if len(summaries) > 1:
        print((out / "summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
