from __future__ import annotations
import argparse
import json
import pickle
import re
from collections import defaultdict
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


def normalize_text(s: str) -> str:
    s = s.replace("Ё", "Е").replace("ё", "е").lower()
    return s


def tokenize(s: str) -> List[str]:
    s = normalize_text(s)
    return WORD_RE.findall(s)


@dataclass
class BM25Index:
    qids: List[str]
    bm25: Any 


def load_bm25(path: Path) -> BM25Index:
    with path.open("rb") as f:
        return pickle.load(f)


def topk_qids(idx: BM25Index, query: str, k: int) -> List[str]:
    q_tokens = tokenize(query)
    scores = idx.bm25.get_scores(q_tokens)
    k = min(k, len(scores))
    top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [idx.qids[i] for i in top_idx]


def compute_recall_at_k(golds: List[str], cand_lists: List[List[str]], ks: List[int]) -> Dict[str, float]:
    # считаем только для non-NULL
    pairs = [(g, c) for g, c in zip(golds, cand_lists) if g != "NULL"]
    if not pairs:
        return {f"recall@{k}": 0.0 for k in ks}

    out = {}
    for k in ks:
        hit = 0
        for g, cands in pairs:
            if g in cands[:k]:
                hit += 1
        out[f"recall@{k}"] = hit / len(pairs)
    out["n_non_null"] = len(pairs)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed_dir", type=str, default="data/processed")
    ap.add_argument("--split", type=str, required=True, choices=["dev", "test"])
    ap.add_argument("--bm25_dir", type=str, default="data/kb/bm25")
    ap.add_argument("--k_list", type=int, nargs="+", default=[1, 5, 10, 50, 100])
    ap.add_argument("--use_context", action="store_true", default=True)
    ap.add_argument("--out_dir", type=str, default="runs/bm25_candidates")
    args = ap.parse_args()

    split_path = Path(args.processed_dir) / f"{args.split}.jsonl"
    idx = load_bm25(Path(args.bm25_dir) / "bm25_index.pkl")

    golds = []
    cand_lists = []
    nested_flags = []
    types = []

    # for subgroup recalls
    sub = {
        "nested": {"g": [], "c": []},
        "non_nested": {"g": [], "c": []},
    }

    for ex in read_jsonl(split_path):
        mention = ex["mention_text"]
        context = ex["context_full"].replace("[M]", " ").replace("[/M]", " ")

        query = f"{mention} {context}" if args.use_context else mention

        kmax = max(args.k_list)
        cands = topk_qids(idx, query=query, k=kmax)

        g = ex["gold_qid"]
        is_nested = bool(ex.get("is_nested", False))

        golds.append(g)
        cand_lists.append(cands)
        nested_flags.append(is_nested)
        types.append(ex["entity_type"])

        key = "nested" if is_nested else "non_nested"
        sub[key]["g"].append(g)
        sub[key]["c"].append(cands)

    overall = compute_recall_at_k(golds, cand_lists, args.k_list)
    nested = compute_recall_at_k(sub["nested"]["g"], sub["nested"]["c"], args.k_list)
    non_nested = compute_recall_at_k(sub["non_nested"]["g"], sub["non_nested"]["c"], args.k_list)

    metrics = {
        "split": args.split,
        "use_context": args.use_context,
        "kb_size": len(idx.qids),
        "overall": overall,
        "nested": nested,
        "non_nested": non_nested,
    }

    out_dir = Path(args.out_dir) / args.split
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "candidate_recall.json", metrics)

    print(f"\n[BM25 CANDIDATES:{args.split}] kb_size={len(idx.qids)} use_context={args.use_context}")
    for k in args.k_list:
        print(f"  overall recall@{k}: {overall.get(f'recall@{k}', 0.0):.4f}")
    print(f"  n_non_null={overall.get('n_non_null', 0)}")
    for k in args.k_list:
        print(f"  nested  recall@{k}: {nested.get(f'recall@{k}', 0.0):.4f} | non_nested recall@{k}: {non_nested.get(f'recall@{k}', 0.0):.4f}")

    print(f"[saved] -> {out_dir/'candidate_recall.json'}")


if __name__ == "__main__":
    main()
