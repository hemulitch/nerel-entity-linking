from __future__ import annotations
import argparse
import json
import math
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from collections import defaultdict

import numpy as np

def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)

def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


ID_RE = re.compile(r"(Q\d+|P\d+)")

def canonical_id(x: Any) -> Optional[str]:
    if x is None:
        return None
    s = str(x).strip()
    if not s:
        return None
    if s.lower() == "null" or s.upper() == "NULL":
        return None
    m = ID_RE.search(s)
    if not m:
        return None
    return m.group(1)


@dataclass
class BM25Index:
    qids: List[str]
    bm25: Any


WORD_RE = re.compile(r"[0-9A-Za-zА-Яа-яЁё_]+")

def normalize_text(s: str) -> str:
    return s.replace("Ё", "Е").replace("ё", "е").lower()

def tokenize(s: str) -> List[str]:
    return WORD_RE.findall(normalize_text(s))

def load_bm25_index(bm25_dir: Path) -> BM25Index:
    pkl = bm25_dir / "bm25_index.pkl"
    with pkl.open("rb") as f:
        return pickle.load(f)

def make_query(ex: Dict[str, Any], use_context: bool) -> str:
    mention = ex["mention_text"]
    if not use_context:
        return mention
    ctx = ex["context_full"].replace("[M]", " ").replace("[/M]", " ")
    return f"{mention} {ctx}"

def predict_top1(idx: BM25Index, query: str) -> Tuple[str, float]:
    q_toks = tokenize(query)
    scores = np.asarray(idx.bm25.get_scores(q_toks), dtype=float)
    best_i = int(scores.argmax())
    return idx.qids[best_i], float(scores[best_i])


# metrics
def micro_accuracy(gold: List[str], pred: List[str]) -> float:
    return sum(int(g == p) for g, p in zip(gold, pred)) / max(1, len(gold))

def macro_accuracy_by_type(gold: List[str], pred: List[str], types: List[str]) -> Tuple[float, Dict[str, Dict[str, float]]]:
    tot = defaultdict(int)
    cor = defaultdict(int)
    for g, p, t in zip(gold, pred, types):
        tot[t] += 1
        cor[t] += int(g == p)
    per_type = {t: {"n": tot[t], "acc": cor[t] / tot[t]} for t in tot}
    macro = sum(v["acc"] for v in per_type.values()) / max(1, len(per_type))
    return macro, per_type

def null_precision_recall(gold: List[str], pred: List[str]) -> Dict[str, float]:
    tp = sum(1 for g, p in zip(gold, pred) if g == "NULL" and p == "NULL")
    fp = sum(1 for g, p in zip(gold, pred) if g != "NULL" and p == "NULL")
    fn = sum(1 for g, p in zip(gold, pred) if g == "NULL" and p != "NULL")
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": prec, "recall": rec, "f1": f1}

def acc_on_mask(gold: List[str], pred: List[str], mask: List[bool]) -> float:
    idx = [i for i, m in enumerate(mask) if m]
    if not idx:
        return 0.0
    return sum(int(gold[i] == pred[i]) for i in idx) / len(idx)


# core eval
def run_split(
    idx: BM25Index,
    split_path: Path,
    use_context: bool,
    threshold: Optional[float],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[float]]:
    gold, pred, types = [], [], []
    is_nested_mask, has_inner_mask = [], []
    best_scores = []

    rows_out = []

    for ex in read_jsonl(split_path):
        g = canonical_id(ex.get("gold_qid")) or "NULL"
        query = make_query(ex, use_context=use_context)
        qid_best, score_best = predict_top1(idx, query)

        p = qid_best
        if threshold is not None and score_best < threshold:
            p = "NULL"

        gold.append(g)
        pred.append(p)
        types.append(ex["entity_type"])
        is_nested = bool(ex.get("is_nested", False))
        has_inner = bool(ex.get("inner_mentions"))
        is_nested_mask.append(is_nested)
        has_inner_mask.append(has_inner)
        best_scores.append(score_best)

        rows_out.append({
            "mention_id": ex["mention_id"],
            "doc_id": ex["doc_id"],
            "entity_type": ex["entity_type"],
            "surface": ex["mention_text"],
            "gold_qid": g,
            "pred_qid": p,
            "bm25_best_qid": qid_best,
            "bm25_best_score": float(score_best),
            "is_nested": is_nested,
            "has_inner_mentions": has_inner,
        })

    micro = micro_accuracy(gold, pred)
    macro, per_type = macro_accuracy_by_type(gold, pred, types)
    null_stats = null_precision_recall(gold, pred)

    metrics = {
        "n": len(gold),
        "micro_accuracy": micro,
        "macro_accuracy": macro,
        "null_stats": null_stats,
        "per_type_accuracy": per_type,
        "threshold": threshold,
        "use_context": use_context,
        "nested_accuracy_inner_mentions": acc_on_mask(gold, pred, is_nested_mask),
        "non_nested_accuracy": acc_on_mask(gold, pred, [not x for x in is_nested_mask]),
        "outer_mentions_accuracy_has_inner": acc_on_mask(gold, pred, has_inner_mask),
        "no_inner_mentions_accuracy": acc_on_mask(gold, pred, [not x for x in has_inner_mask]),
    }
    return metrics, rows_out, best_scores


def calibrate_threshold(best_scores: List[float], gold: List[str], best_qids: List[str]) -> Dict[str, Any]:
    """
    We calibrate threshold to maximize micro accuracy on dev.
    For a given threshold t:
      pred = NULL if score < t else best_qid
    """
    scores = np.asarray(best_scores, dtype=float)
    # grid: 0..100 percentiles
    qs = np.quantile(scores, np.linspace(0, 1, 101))
    thresholds = sorted(set(float(x) for x in qs))

    best = {"threshold": None, "micro_accuracy": -1.0}

    curve = []
    for t in thresholds:
        pred = ["NULL" if s < t else q for s, q in zip(best_scores, best_qids)]
        micro = sum(int(p == g) for p, g in zip(pred, gold)) / max(1, len(gold))
        curve.append({"threshold": float(t), "micro_accuracy": float(micro)})

        if micro > best["micro_accuracy"]:
            best = {"threshold": float(t), "micro_accuracy": float(micro)}

    return {"best": best, "curve": curve}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed_dir", type=str, default="data/processed")
    ap.add_argument("--split", type=str, required=True, choices=["dev", "test"])
    ap.add_argument("--bm25_dir", type=str, required=True, help="e.g. data/kb/bm25_all")
    ap.add_argument("--out_dir", type=str, default="runs/bm25_top1")

    ap.add_argument("--use_context", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--threshold", type=float, default=None, help="If set: apply this threshold (no calibration).")
    ap.add_argument("--calibrate_on_dev", action="store_true", help="Calibrate threshold on dev and save calibration.json")
    ap.add_argument("--calibration_path", type=str, default=None, help="Load threshold from calibration.json")
    args = ap.parse_args()

    processed_dir = Path(args.processed_dir)
    split_path = processed_dir / f"{args.split}.jsonl"
    idx = load_bm25_index(Path(args.bm25_dir))

    # decide threshold
    threshold = args.threshold
    calib_info = None

    if args.calibration_path is not None:
        calib = json.loads(Path(args.calibration_path).read_text(encoding="utf-8"))
        threshold = float(calib["best"]["threshold"])
        print(f"[load] threshold from {args.calibration_path}: {threshold:.6f}")

    if args.calibrate_on_dev:
        dev_path = processed_dir / "dev.jsonl"

        # compute best_qid and best_score on dev once
        gold_dev = []
        best_scores_dev = []
        best_qids_dev = []
        for ex in read_jsonl(dev_path):
            g = canonical_id(ex.get("gold_qid")) or "NULL"
            q = make_query(ex, use_context=args.use_context)
            qid_best, score_best = predict_top1(idx, q)
            gold_dev.append(g)
            best_scores_dev.append(score_best)
            best_qids_dev.append(qid_best)

        calib_info = calibrate_threshold(best_scores_dev, gold_dev, best_qids_dev)
        threshold = float(calib_info["best"]["threshold"])
        print(f"[calib] best threshold={threshold:.6f} | dev micro={calib_info['best']['micro_accuracy']:.4f}")

    metrics, preds_rows, _ = run_split(idx, split_path, use_context=args.use_context, threshold=threshold)

    out_dir = Path(args.out_dir) / args.split
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "metrics.json", metrics)
    write_jsonl(out_dir / "predictions.jsonl", preds_rows)

    if calib_info is not None:
        write_json(Path(args.out_dir) / "dev" / "calibration.json", calib_info)

    print(f"\n[BM25 TOP1:{args.split}] kb_size={len(idx.qids)} use_context={args.use_context} threshold={threshold}")
    print(f"  n={metrics['n']}")
    print(f"  micro_accuracy={metrics['micro_accuracy']:.4f}")
    print(f"  macro_accuracy={metrics['macro_accuracy']:.4f}")
    ns = metrics["null_stats"]
    print(f"  NULL precision={ns['precision']:.4f} recall={ns['recall']:.4f} f1={ns['f1']:.4f}")
    print(f"  inner(is_nested) acc={metrics['nested_accuracy_inner_mentions']:.4f} | non_nested acc={metrics['non_nested_accuracy']:.4f}")
    print(f"  outer(has_inner) acc={metrics['outer_mentions_accuracy_has_inner']:.4f} | no_inner acc={metrics['no_inner_mentions_accuracy']:.4f}")
    print(f"[saved] -> {out_dir/'metrics.json'}")


if __name__ == "__main__":
    main()
