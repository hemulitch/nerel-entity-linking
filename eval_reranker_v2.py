#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# -------- IO --------
def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
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

# -------- canonical ids --------
ID_RE = re.compile(r"(Q\d+|P\d+)")
WORD_RE = re.compile(r"[0-9A-Za-zА-Яа-яЁё_]+")

def canonical_id(x: Any) -> Optional[str]:
    if x is None:
        return None
    s = str(x).strip()
    if not s:
        return None
    if s.lower() == "null":
        return None
    m = ID_RE.search(s)
    return m.group(1) if m else None

def normalize_surface(s: str) -> str:
    s = str(s).replace("Ё", "Е").replace("ё", "е").lower()
    toks = WORD_RE.findall(s)
    return " ".join(toks).strip()

# -------- surface dict (seen/unseen diagnostics) --------
def build_surface_set(train_path: Path) -> set[str]:
    s = set()
    for ex in read_jsonl(train_path):
        surf = normalize_surface(ex.get("mention_text", ""))
        if surf:
            s.add(surf)
    return s

def build_surface_dict(train_path: Path) -> Dict[str, str]:
    counts: Dict[str, Dict[str, int]] = {}
    for ex in read_jsonl(train_path):
        g = canonical_id(ex.get("gold_qid")) or "NULL"
        surf = normalize_surface(ex.get("mention_text", ""))
        if not surf:
            continue
        counts.setdefault(surf, {})
        counts[surf][g] = counts[surf].get(g, 0) + 1
    out = {}
    for surf, d in counts.items():
        out[surf] = max(d.items(), key=lambda kv: kv[1])[0]
    return out

# -------- KB texts --------
def build_entity_text(rec: Dict[str, Any], max_aliases: int = 10) -> str:
    label = rec.get("label", "") or ""
    desc = rec.get("description", "") or ""
    aliases = rec.get("aliases", []) or []
    aliases = aliases[:max_aliases]
    parts = []
    if label:
        parts.append(label)
    if aliases:
        parts.append(" ; ".join(aliases))
    if desc:
        parts.append(desc)
    return "\n".join(parts).strip()

def load_kb_texts(kb_path: Path, max_aliases: int = 10) -> Dict[str, str]:
    kb = {}
    for rec in read_jsonl(kb_path):
        qid = rec.get("qid")
        if not qid:
            continue
        kb[qid] = build_entity_text(rec, max_aliases=max_aliases) or qid
    return kb

NULL_TEXT = "NULL_CANDIDATE: нет подходящей сущности (NIL / Wikidata:NULL)."

# -------- BM25 --------
@dataclass
class BM25Index:
    qids: List[str]
    bm25: Any

def load_bm25_index(bm25_dir: Path) -> BM25Index:
    with (bm25_dir / "bm25_index.pkl").open("rb") as f:
        return pickle.load(f)

def tokenize_for_bm25(s: str) -> List[str]:
    s = str(s).replace("Ё", "Е").replace("ё", "е").lower()
    return WORD_RE.findall(s)

def bm25_topk(idx: BM25Index, query: str, k: int) -> List[str]:
    q_toks = tokenize_for_bm25(query)
    scores = np.asarray(idx.bm25.get_scores(q_toks), dtype=float)
    if k >= len(scores):
        top = np.argsort(scores)[::-1]
    else:
        top = np.argpartition(scores, -k)[-k:]
        top = top[np.argsort(scores[top])[::-1]]
    return [idx.qids[i] for i in top]

def make_query(ex: Dict[str, Any], use_context: bool, use_inner_mentions: bool, inner_max: int = 5) -> str:
    mention = ex.get("mention_text", "")
    if use_context:
        q = ex.get("context_full", "")
    else:
        q = f"[M]{mention}[/M]"
    if use_inner_mentions:
        inn = ex.get("inner_mentions", []) or []
        inn = [str(x) for x in inn[:inner_max] if str(x).strip()]
        if inn:
            q = q + "\nINNER_MENTIONS: " + " ; ".join(inn)
    return q

def pick_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

# -------- metrics --------
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

def acc_on_mask(gold: List[str], pred: List[str], mask: List[bool]) -> float:
    idx = [i for i, m in enumerate(mask) if m]
    if not idx:
        return 0.0
    return sum(int(gold[i] == pred[i]) for i in idx) / len(idx)

# -------- main --------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed_dir", type=str, default="data/processed")
    ap.add_argument("--split", type=str, required=True, choices=["dev", "test"])
    ap.add_argument("--bm25_dir", type=str, required=True)
    ap.add_argument("--kb_path", type=str, required=True)
    ap.add_argument("--model_dir", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="runs/reranker_v2_eval")

    ap.add_argument("--candidate_k", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--max_length", type=int, default=256)

    ap.add_argument("--use_context", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--use_inner_mentions", action=argparse.BooleanOptionalAction, default=False)

    ap.add_argument("--add_surface_candidate", action="store_true")
    ap.add_argument("--max_aliases", type=int, default=10)

    ap.add_argument("--device", type=str, default="auto")
    args = ap.parse_args()

    processed_dir = Path(args.processed_dir)
    split_path = processed_dir / f"{args.split}.jsonl"
    train_path = processed_dir / "train.jsonl"

    idx = load_bm25_index(Path(args.bm25_dir))
    kb_text = load_kb_texts(Path(args.kb_path), max_aliases=args.max_aliases)
    kb_text["NULL"] = NULL_TEXT  # add explicit NULL candidate

    surface_set = build_surface_set(train_path)
    surface_dict = None
    if args.add_surface_candidate:
        surface_dict = build_surface_dict(train_path)
        print(f"[surface_dict] size={len(surface_dict)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
    device = pick_device(args.device)
    model.to(device)
    model.eval()

    gold, pred, types = [], [], []
    seen_mask, nested_mask, has_inner_mask = [], [], []

    out_rows = []

    def unique_keep_order(xs: List[str]) -> List[str]:
        seen = set()
        out = []
        for x in xs:
            if x in seen:
                continue
            seen.add(x)
            out.append(x)
        return out

    for ex in tqdm(list(read_jsonl(split_path)), desc=f"rerank_v2:{args.split}"):
        g = canonical_id(ex.get("gold_qid")) or "NULL"

        q = make_query(ex, use_context=args.use_context, use_inner_mentions=args.use_inner_mentions)

        cands = bm25_topk(idx, query=q, k=args.candidate_k)
        # add surface candidate (hybrid)
        if surface_dict is not None:
            surf = normalize_surface(ex.get("mention_text", ""))
            q_surf = canonical_id(surface_dict.get(surf)) if surf else None
            if q_surf:
                cands = [q_surf] + cands

        # add NULL candidate
        cands = ["NULL"] + cands
        cands = unique_keep_order(cands)[:args.candidate_k]

        cand_qids = [cq for cq in cands if cq in kb_text]
        cand_texts = [kb_text[cq] for cq in cand_qids]

        # score candidates
        scores_all = []
        for i in range(0, len(cand_qids), args.batch_size):
            sub_q = [q] * min(args.batch_size, len(cand_qids) - i)
            sub_c = cand_texts[i:i + args.batch_size]
            enc = tokenizer(
                sub_q, sub_c,
                padding=True, truncation=True, max_length=args.max_length,
                return_tensors="pt"
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            with torch.no_grad():
                logits = model(**enc).logits
            # binary: score = logit for class=1
            sub_scores = logits[:, 1].detach().cpu().numpy().tolist()
            scores_all.extend(sub_scores)

        j = int(np.argmax(np.asarray(scores_all, dtype=float)))
        p = cand_qids[j]

        gold.append(g)
        pred.append(p)
        types.append(ex.get("entity_type", "UNK"))

        surf = normalize_surface(ex.get("mention_text", ""))
        is_seen = surf in surface_set if surf else False
        is_nested = bool(ex.get("is_nested", False))
        has_inner = bool(ex.get("inner_mentions"))

        seen_mask.append(is_seen)
        nested_mask.append(is_nested)
        has_inner_mask.append(has_inner)

        out_rows.append({
            "mention_id": ex.get("mention_id"),
            "doc_id": ex.get("doc_id"),
            "entity_type": ex.get("entity_type"),
            "surface": ex.get("mention_text"),
            "gold_qid": g,
            "pred_qid": p,
            "best_score": float(scores_all[j]),
            "is_seen_surface": is_seen,
            "is_nested": is_nested,
            "has_inner_mentions": has_inner,
        })

    micro = micro_accuracy(gold, pred)
    macro, per_type = macro_accuracy_by_type(gold, pred, types)

    metrics = {
        "split": args.split,
        "micro_accuracy": micro,
        "macro_accuracy": macro,
        "candidate_k": args.candidate_k,
        "use_context": args.use_context,
        "use_inner_mentions": args.use_inner_mentions,
        "add_surface_candidate": bool(surface_dict is not None),
        "nested_accuracy_inner_mentions": acc_on_mask(gold, pred, nested_mask),
        "non_nested_accuracy": acc_on_mask(gold, pred, [not x for x in nested_mask]),
        "seen_surface_coverage": sum(seen_mask) / max(1, len(seen_mask)),
        "seen_surface_accuracy": acc_on_mask(gold, pred, seen_mask),
        "unseen_surface_accuracy": acc_on_mask(gold, pred, [not x for x in seen_mask]),
        "outer_mentions_accuracy_has_inner": acc_on_mask(gold, pred, has_inner_mask),
        "no_inner_mentions_accuracy": acc_on_mask(gold, pred, [not x for x in has_inner_mask]),
        "per_type_accuracy": per_type,
    }

    out_dir = Path(args.out_dir) / args.split
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "metrics.json", metrics)
    write_jsonl(out_dir / "predictions.jsonl", out_rows)

    print(f"\n[RERANKER_V2:{args.split}] micro={micro:.4f} macro={macro:.4f} device={device}")
    print(f"  seen_cov={metrics['seen_surface_coverage']:.4f} | seen_acc={metrics['seen_surface_accuracy']:.4f} | unseen_acc={metrics['unseen_surface_accuracy']:.4f}")
    print(f"  nested_acc={metrics['nested_accuracy_inner_mentions']:.4f} | non_nested_acc={metrics['non_nested_accuracy']:.4f}")
    print(f"  outer(has_inner)_acc={metrics['outer_mentions_accuracy_has_inner']:.4f} | no_inner_acc={metrics['no_inner_mentions_accuracy']:.4f}")
    print(f"[saved] -> {out_dir/'metrics.json'}")

if __name__ == "__main__":
    main()
