"""
Step 2: Surface-form frequency baseline for Entity Linking.

Idea:
- Build dictionary from TRAIN: normalized_surface -> most frequent QID
- Predict on DEV/TEST:
    if surface seen -> argmax_qid
    else -> NULL (default)

Also produces diagnostics:
- micro/macro accuracy
- per-type accuracy
- seen vs unseen accuracy + coverage
- nested vs non-nested accuracy
- NULL precision/recall
- top ambiguous surfaces in train (for analysis section)

Inputs:
- data/processed/train.jsonl
- data/processed/dev.jsonl or test.jsonl

Outputs (by default):
- runs/surface_baseline/<split>/metrics.json
- runs/surface_baseline/<split>/predictions.jsonl
- runs/surface_baseline/train_surface_ambiguity.csv
"""

from __future__ import annotations
import argparse
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


WORD_RE = re.compile(r"[0-9A-Za-zА-Яа-яЁё_]+")


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


def normalize_surface(s: str) -> str:
    """
    Normalization for surface forms:
    - lower
    - ё -> е
    - keep only word tokens, join with spaces
    """
    s = s.replace("Ё", "Е").replace("ё", "е").lower()
    toks = WORD_RE.findall(s)
    return " ".join(toks).strip()


def micro_accuracy(gold: List[str], pred: List[str]) -> float:
    if not gold:
        return 0.0
    return sum(1 for g, p in zip(gold, pred) if g == p) / len(gold)


def macro_accuracy_by_type(gold: List[str], pred: List[str], types: List[str]) -> Tuple[float, Dict[str, Dict[str, float]]]:
    """
    Macro accuracy defined as average accuracy across entity types.
    Returns:
      macro, per_type_stats[type] = {n, acc}
    """
    by_type_total = defaultdict(int)
    by_type_correct = defaultdict(int)

    for g, p, t in zip(gold, pred, types):
        by_type_total[t] += 1
        by_type_correct[t] += int(g == p)

    per_type = {}
    for t in by_type_total:
        n = by_type_total[t]
        acc = by_type_correct[t] / n if n else 0.0
        per_type[t] = {"n": n, "acc": acc}

    macro = sum(v["acc"] for v in per_type.values()) / len(per_type) if per_type else 0.0
    return macro, per_type


def null_precision_recall(gold: List[str], pred: List[str]) -> Dict[str, float]:
    """
    Treat NULL as the positive class and compute precision/recall/F1.
    """
    tp = sum(1 for g, p in zip(gold, pred) if g == "NULL" and p == "NULL")
    fp = sum(1 for g, p in zip(gold, pred) if g != "NULL" and p == "NULL")
    fn = sum(1 for g, p in zip(gold, pred) if g == "NULL" and p != "NULL")

    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) else 0.0

    return {"tp": tp, "fp": fp, "fn": fn, "precision": prec, "recall": rec, "f1": f1}


def build_surface_dictionary(train_path: Path) -> Tuple[Dict[str, str], Dict[str, Counter], Counter]:
    """
    Returns:
      surface_to_best_qid: normalized_surface -> argmax QID
      surface_to_counter: normalized_surface -> Counter(QID)
      global_qid_counts: Counter(QID)
    """
    surface_to_counter: Dict[str, Counter] = defaultdict(Counter)
    global_counts = Counter()

    for ex in read_jsonl(train_path):
        surface = normalize_surface(ex["mention_text"])
        qid = ex["gold_qid"]
        if not surface:
            continue
        surface_to_counter[surface][qid] += 1
        global_counts[qid] += 1

    surface_to_best_qid: Dict[str, str] = {}
    for s, c in surface_to_counter.items():
        surface_to_best_qid[s] = c.most_common(1)[0][0]

    return surface_to_best_qid, surface_to_counter, global_counts


def save_ambiguity_report(surface_to_counter: Dict[str, Counter], out_csv: Path, topn: int = 100) -> None:
    """
    Save top ambiguous surfaces:
    - by number of distinct (non-NULL) QIDs
    - also compute purity = top_count / total_count
    """
    rows = []
    for surface, counter in surface_to_counter.items():
        total = sum(counter.values())
        top_qid, top_count = counter.most_common(1)[0]
        qids_nonnull = [q for q in counter.keys() if q != "NULL"]
        distinct_nonnull = len(set(qids_nonnull))
        purity = top_count / total if total else 0.0

        ent = 0.0
        for _, cnt in counter.items():
            p = cnt / total
            ent -= p * math.log(p + 1e-12)

        rows.append({
            "surface": surface,
            "total_mentions_train": total,
            "top_qid": top_qid,
            "top_count": top_count,
            "purity": purity,
            "distinct_nonnull_qids": distinct_nonnull,
            "entropy": ent,
        })

    rows_sorted = sorted(rows, key=lambda r: (r["distinct_nonnull_qids"], -r["entropy"], -r["total_mentions_train"]), reverse=True)
    rows_sorted = rows_sorted[:topn]

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8") as f:
        headers = list(rows_sorted[0].keys()) if rows_sorted else ["surface"]
        f.write(",".join(headers) + "\n")
        for r in rows_sorted:
            vals = []
            for h in headers:
                v = r.get(h)
                if isinstance(v, str):
                    vv = v.replace('"', '""')
                    vals.append(f'"{vv}"')
                else:
                    vals.append(str(v))
            f.write(",".join(vals) + "\n")


def evaluate(
    split_path: Path,
    surface_to_best_qid: Dict[str, str],
    unknown_strategy: str = "NULL",
    global_most_common_qid: str = "NULL",
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Evaluate baseline on a split.
    Returns:
      metrics dict, predictions rows
    """
    gold, pred, types = [], [], []
    seen_flags = []
    nested_flags = []

    preds_rows = []

    for ex in read_jsonl(split_path):
        surface = normalize_surface(ex["mention_text"])
        is_seen = surface in surface_to_best_qid

        if is_seen:
            p = surface_to_best_qid[surface]
        else:
            p = global_most_common_qid if unknown_strategy == "global" else "NULL"

        g = ex["gold_qid"]
        t = ex["entity_type"]
        is_nested = bool(ex.get("is_nested", False))

        gold.append(g)
        pred.append(p)
        types.append(t)
        seen_flags.append(is_seen)
        nested_flags.append(is_nested)

        preds_rows.append({
            "mention_id": ex["mention_id"],
            "doc_id": ex["doc_id"],
            "entity_type": t,
            "surface_raw": ex["mention_text"],
            "surface_norm": surface,
            "gold_qid": g,
            "pred_qid": p,
            "is_seen": is_seen,
            "is_nested": is_nested,
        })

    micro = micro_accuracy(gold, pred)
    macro, per_type = macro_accuracy_by_type(gold, pred, types)
    null_pr = null_precision_recall(gold, pred)

    n = len(gold)
    seen_n = sum(1 for x in seen_flags if x)
    unseen_n = n - seen_n

    def acc_on(mask: List[bool]) -> float:
        idx = [i for i, m in enumerate(mask) if m]
        if not idx:
            return 0.0
        return sum(1 for i in idx if gold[i] == pred[i]) / len(idx)

    seen_acc = acc_on(seen_flags)
    unseen_acc = acc_on([not x for x in seen_flags])

    nested_acc = acc_on(nested_flags)
    non_nested_acc = acc_on([not x for x in nested_flags])

    metrics = {
        "micro_accuracy": micro,
        "macro_accuracy": macro,
        "n": n,
        "seen_coverage": (seen_n / n) if n else 0.0,
        "seen_accuracy": seen_acc,
        "unseen_accuracy": unseen_acc,
        "nested_accuracy": nested_acc,
        "non_nested_accuracy": non_nested_acc,
        "null_stats": null_pr,
        "per_type_accuracy": per_type,
        "unknown_strategy": unknown_strategy,
        "global_most_common_qid": global_most_common_qid,
    }

    return metrics, preds_rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed_dir", type=str, default="data/processed")
    ap.add_argument("--train_split", type=str, default="train", choices=["train"])
    ap.add_argument("--split", type=str, required=True, choices=["dev", "test"])
    ap.add_argument("--out_dir", type=str, default="runs/surface_baseline")
    ap.add_argument("--unknown_strategy", type=str, default="NULL", choices=["NULL", "global"],
                    help="What to predict for unseen surfaces: NULL (default) or global most frequent QID in train.")
    ap.add_argument("--save_ambiguity", action="store_true", help="Save train surface ambiguity CSV (recommended)")
    ap.add_argument("--top_ambiguous", type=int, default=100)
    args = ap.parse_args()

    processed_dir = Path(args.processed_dir)
    train_path = processed_dir / f"{args.train_split}.jsonl"
    split_path = processed_dir / f"{args.split}.jsonl"

    if not train_path.exists():
        raise FileNotFoundError(f"Missing train: {train_path}")
    if not split_path.exists():
        raise FileNotFoundError(f"Missing split: {split_path}")

    surface_to_best, surface_to_counter, global_counts = build_surface_dictionary(train_path)
    global_most_common_qid = global_counts.most_common(1)[0][0] if global_counts else "NULL"

    if args.save_ambiguity:
        out_csv = Path(args.out_dir) / "train_surface_ambiguity.csv"
        save_ambiguity_report(surface_to_counter, out_csv, topn=args.top_ambiguous)
        print(f"[saved] ambiguity report -> {out_csv}")

    metrics, preds_rows = evaluate(
        split_path=split_path,
        surface_to_best_qid=surface_to_best,
        unknown_strategy=args.unknown_strategy,
        global_most_common_qid=global_most_common_qid,
    )

    out_dir = Path(args.out_dir) / args.split
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "metrics.json", metrics)
    write_jsonl(out_dir / "predictions.jsonl", preds_rows)

    print(f"\n[SURFACE BASELINE:{args.split}]")
    print(f"  n={metrics['n']}")
    print(f"  micro_accuracy={metrics['micro_accuracy']:.4f}")
    print(f"  macro_accuracy={metrics['macro_accuracy']:.4f}")
    print(f"  seen_coverage={metrics['seen_coverage']:.4f} | seen_acc={metrics['seen_accuracy']:.4f} | unseen_acc={metrics['unseen_accuracy']:.4f}")
    print(f"  nested_acc={metrics['nested_accuracy']:.4f} | non_nested_acc={metrics['non_nested_accuracy']:.4f}")
    ns = metrics["null_stats"]
    print(f"  NULL precision={ns['precision']:.4f} recall={ns['recall']:.4f} f1={ns['f1']:.4f}")
    print(f"[saved] metrics -> {out_dir / 'metrics.json'}")
    print(f"[saved] preds   -> {out_dir / 'predictions.jsonl'}")


if __name__ == "__main__":
    main()
