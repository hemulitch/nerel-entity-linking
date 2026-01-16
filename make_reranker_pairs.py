#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import random
import re
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from tqdm import tqdm


# ---- I/O ----
def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ---- canonical QID/PID ----
ID_RE = re.compile(r"(Q\d+|P\d+)")

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


# ---- surface dict (optional hybrid candidates) ----
WORD_RE = re.compile(r"[0-9A-Za-zА-Яа-яЁё_]+")

def normalize_surface(s: str) -> str:
    s = str(s).replace("Ё", "Е").replace("ё", "е").lower()
    toks = WORD_RE.findall(s)
    return " ".join(toks).strip()

def build_surface_dict(train_processed: Path) -> Dict[str, str]:
    # surface_norm -> most frequent qid in train
    counts: Dict[str, Dict[str, int]] = {}
    for ex in read_jsonl(train_processed):
        g = canonical_id(ex.get("gold_qid")) or "NULL"
        surf = normalize_surface(ex.get("mention_text", ""))
        if not surf:
            continue
        if surf not in counts:
            counts[surf] = {}
        counts[surf][g] = counts[surf].get(g, 0) + 1

    out = {}
    for surf, d in counts.items():
        best = max(d.items(), key=lambda kv: kv[1])[0]
        out[surf] = best
    return out


# ---- KB texts ----
def build_entity_text(rec: Dict[str, Any], max_aliases: int = 20) -> str:
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

def load_kb_texts(kb_path: Path, max_aliases: int = 20) -> Dict[str, str]:
    kb = {}
    for rec in read_jsonl(kb_path):
        qid = rec.get("qid")
        if not qid:
            continue
        kb[qid] = build_entity_text(rec, max_aliases=max_aliases) or qid
    return kb


# ---- BM25 index (pickle) ----
@dataclass
class BM25Index:
    qids: List[str]
    bm25: Any  # BM25Okapi

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
    if not use_context:
        q = mention
    else:
        ctx = ex.get("context_full", "").replace("[M]", " ").replace("[/M]", " ")
        q = f"{mention} {ctx}"

    if use_inner_mentions:
        inn = ex.get("inner_mentions", []) or []
        inn = [str(x) for x in inn[:inner_max] if str(x).strip()]
        if inn:
            q = q + " INNER: " + " ; ".join(inn)
    return q


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed_dir", type=str, default="data/processed")
    ap.add_argument("--bm25_dir", type=str, required=True)
    ap.add_argument("--kb_path", type=str, required=True)
    ap.add_argument("--out_path", type=str, default="data/reranker/train_pairs.jsonl")

    ap.add_argument("--candidate_k", type=int, default=100)
    ap.add_argument("--neg_per_pos", type=int, default=5)
    ap.add_argument("--max_train_mentions", type=int, default=None, help="Debug: limit number of train mentions (non-NULL).")
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--use_context", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--use_inner_mentions", action=argparse.BooleanOptionalAction, default=False)

    ap.add_argument("--add_surface_candidate", action="store_true",
                    help="Hybrid candidates: add train surface-dict top-1 to BM25 candidate list.")
    ap.add_argument("--force_add_gold", action="store_true",
                    help="Ensure gold QID is included among candidates (training convenience).")

    ap.add_argument("--max_aliases", type=int, default=20)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    processed_dir = Path(args.processed_dir)
    train_path = processed_dir / "train.jsonl"
    out_path = Path(args.out_path)

    idx = load_bm25_index(Path(args.bm25_dir))
    kb_text = load_kb_texts(Path(args.kb_path), max_aliases=args.max_aliases)
    kb_keys = list(kb_text.keys())

    surface_dict = None
    if args.add_surface_candidate:
        surface_dict = build_surface_dict(train_path)
        print(f"[surface_dict] size={len(surface_dict)}")

    total_seen = 0
    used = 0
    skipped_null = 0
    skipped_no_kb = 0

    def unique_keep_order(xs: List[str]) -> List[str]:
        seen = set()
        out = []
        for x in xs:
            if x in seen:
                continue
            seen.add(x)
            out.append(x)
        return out

    out_rows = []

    for ex in tqdm(read_jsonl(train_path), desc="make_pairs"):
        g = canonical_id(ex.get("gold_qid")) or "NULL"
        if g == "NULL":
            skipped_null += 1
            continue
        if g not in kb_text:
            skipped_no_kb += 1
            continue

        query = make_query(ex, use_context=args.use_context, use_inner_mentions=args.use_inner_mentions)

        cands = bm25_topk(idx, query=query, k=args.candidate_k)

        # optional: add surface candidate (train dict)
        if surface_dict is not None:
            surf = normalize_surface(ex.get("mention_text", ""))
            if surf and surf in surface_dict:
                total_seen += 1
                q_surf = canonical_id(surface_dict[surf]) or "NULL"
                if q_surf != "NULL":
                    cands = [q_surf] + cands

        if args.force_add_gold and g not in cands:
            cands = [g] + cands

        cands = unique_keep_order(cands)
        # keep size bounded
        cands = cands[:args.candidate_k]

        # positives
        out_rows.append({
            "query": query,
            "candidate": kb_text[g],
            "label": 1,
            "gold_qid": g,
            "cand_qid": g,
            "mention_id": ex.get("mention_id"),
            "entity_type": ex.get("entity_type"),
            "is_nested": bool(ex.get("is_nested", False)),
            "has_inner_mentions": bool(ex.get("inner_mentions")),
        })

        # negatives: hard negatives from candidates
        neg_pool = [q for q in cands if q != g and q in kb_text]
        if len(neg_pool) >= args.neg_per_pos:
            negs = rng.sample(neg_pool, k=args.neg_per_pos)
        else:
            negs = list(neg_pool)
            # fill with random negatives from KB
            while len(negs) < args.neg_per_pos:
                q = rng.choice(kb_keys)
                if q != g:
                    negs.append(q)

        for nq in negs:
            out_rows.append({
                "query": query,
                "candidate": kb_text[nq],
                "label": 0,
                "gold_qid": g,
                "cand_qid": nq,
                "mention_id": ex.get("mention_id"),
                "entity_type": ex.get("entity_type"),
                "is_nested": bool(ex.get("is_nested", False)),
                "has_inner_mentions": bool(ex.get("inner_mentions")),
            })

        used += 1
        if args.max_train_mentions is not None and used >= args.max_train_mentions:
            break

    write_jsonl(out_path, out_rows)

    print(f"[saved] pairs -> {out_path} lines={len(out_rows)}")
    print(f"[train_mentions_used] {used} (non-NULL)")
    print(f"[skipped_null] {skipped_null}")
    print(f"[skipped_no_kb_text] {skipped_no_kb}")
    if surface_dict is not None:
        print(f"[surface_seen_mentions_counted] {total_seen}")


if __name__ == "__main__":
    main()
