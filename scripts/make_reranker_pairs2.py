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
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
from tqdm import tqdm

# ---------- IO ----------
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

# ---------- canonical ids ----------
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

# ---------- surface dict (optional candidate) ----------
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

# ---------- KB texts ----------
def build_entity_text(rec: Dict[str, Any], max_aliases: int = 10) -> str:
    label = rec.get("label", "") or ""
    desc = rec.get("description", "") or ""
    aliases = rec.get("aliases", []) or []
    aliases = aliases[:max_aliases]
    parts = []
    if label:
        parts.append(label)
    # aliases часто шумят — поэтому делаем их ограниченно
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

# ---------- BM25 ----------
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
    if use_context:
        # ВАЖНО: НЕ убираем маркеры [M]...[/M] — пусть модель видит “фокус”
        ctx = ex.get("context_full", "")
        q = f"{ctx}"
    else:
        q = f"[M]{mention}[/M]"

    if use_inner_mentions:
        inn = ex.get("inner_mentions", []) or []
        inn = [str(x) for x in inn[:inner_max] if str(x).strip()]
        if inn:
            q = q + "\nINNER_MENTIONS: " + " ; ".join(inn)
    return q

def unique_keep_order(xs: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in xs:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out

# ---------- main ----------
NULL_TEXT = "NULL_CANDIDATE: нет подходящей сущности (NIL / Wikidata:NULL)."

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed_dir", type=str, default="data/processed")
    ap.add_argument("--bm25_dir", type=str, required=True)
    ap.add_argument("--kb_path", type=str, required=True)
    ap.add_argument("--out_path", type=str, default="data/reranker/train_pairs_v2.jsonl")

    ap.add_argument("--candidate_k", type=int, default=100)
    ap.add_argument("--neg_per_pos", type=int, default=5)
    ap.add_argument("--max_train_mentions", type=int, default=None, help="debug: limit mentions")
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--use_context", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--use_inner_mentions", action=argparse.BooleanOptionalAction, default=False)

    ap.add_argument("--add_surface_candidate", action="store_true")
    ap.add_argument("--force_add_gold", action="store_true",
                    help="ensure gold in candidates for training (recommended)")

    ap.add_argument("--max_aliases", type=int, default=10)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    processed_dir = Path(args.processed_dir)
    train_path = processed_dir / "train.jsonl"

    idx = load_bm25_index(Path(args.bm25_dir))
    kb_text = load_kb_texts(Path(args.kb_path), max_aliases=args.max_aliases)
    kb_keys = list(kb_text.keys())

    surface_dict = None
    if args.add_surface_candidate:
        surface_dict = build_surface_dict(train_path)
        print(f"[surface_dict] size={len(surface_dict)}")

    out_rows = []

    used = 0
    skipped_no_kb = 0

    for ex in tqdm(read_jsonl(train_path), desc="make_pairs_v2"):
        g = canonical_id(ex.get("gold_qid")) or "NULL"
        query = make_query(ex, use_context=args.use_context, use_inner_mentions=args.use_inner_mentions)

        # BM25 candidates (even for NULL mentions, to create hard negatives)
        cands = bm25_topk(idx, query=query, k=args.candidate_k)

        # add surface candidate (hybrid)
        if surface_dict is not None:
            surf = normalize_surface(ex.get("mention_text", ""))
            if surf and surf in surface_dict:
                q_surf = canonical_id(surface_dict[surf]) or "NULL"
                if q_surf != "NULL":
                    cands = [q_surf] + cands

        # Always include NULL as a candidate in the training universe
        # (it will be positive only when gold is NULL)
        cands = ["NULL"] + cands

        # Ensure gold is present (important for learning!)
        if args.force_add_gold and g != "NULL" and g not in cands:
            cands = [g] + cands

        cands = unique_keep_order(cands)[:args.candidate_k]

        # Helper: get text for candidate
        def cand_text(qid: str) -> Optional[str]:
            if qid == "NULL":
                return NULL_TEXT
            return kb_text.get(qid)

        # Positive example
        if g == "NULL":
            out_rows.append({
                "query": query,
                "candidate": NULL_TEXT,
                "label": 1,
                "gold_qid": "NULL",
                "cand_qid": "NULL",
                "mention_id": ex.get("mention_id"),
                "entity_type": ex.get("entity_type"),
                "is_nested": bool(ex.get("is_nested", False)),
                "has_inner_mentions": bool(ex.get("inner_mentions")),
            })
            # negatives: pick from candidates (excluding NULL) + random fallback
            neg_pool = [q for q in cands if q != "NULL" and q in kb_text]
            negs = rng.sample(neg_pool, k=min(args.neg_per_pos, len(neg_pool)))
            while len(negs) < args.neg_per_pos:
                q = rng.choice(kb_keys)
                negs.append(q)
            for nq in negs:
                out_rows.append({
                    "query": query,
                    "candidate": kb_text[nq],
                    "label": 0,
                    "gold_qid": "NULL",
                    "cand_qid": nq,
                    "mention_id": ex.get("mention_id"),
                    "entity_type": ex.get("entity_type"),
                    "is_nested": bool(ex.get("is_nested", False)),
                    "has_inner_mentions": bool(ex.get("inner_mentions")),
                })
        else:
            # --- non-NULL: need KB text ---
            if g not in kb_text:
                skipped_no_kb += 1
                continue

            # positive
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

            # !!! ALWAYS add NULL as an explicit negative
            out_rows.append({
                "query": query,
                "candidate": NULL_TEXT,
                "label": 0,
                "gold_qid": g,
                "cand_qid": "NULL",
                "mention_id": ex.get("mention_id"),
                "entity_type": ex.get("entity_type"),
                "is_nested": bool(ex.get("is_nested", False)),
                "has_inner_mentions": bool(ex.get("inner_mentions")),
            })

            # other negatives (hard negatives) — exclude NULL and gold
            remaining = max(0, args.neg_per_pos - 1)

            neg_pool = [q for q in cands if q != g and q != "NULL" and q in kb_text]
            negs = rng.sample(neg_pool, k=min(remaining, len(neg_pool)))

            # fill up if not enough
            while len(negs) < remaining:
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

    out_path = Path(args.out_path)
    write_jsonl(out_path, out_rows)

    print(f"[saved] pairs -> {out_path} lines={len(out_rows)}")
    print(f"[mentions_used] {used}")
    print(f"[skipped_no_kb_text(non-null)] {skipped_no_kb}")

if __name__ == "__main__":
    main()
