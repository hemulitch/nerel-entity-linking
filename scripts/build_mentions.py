#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step 1 (NEREL -> mention-level EL dataset):
- optionally downloads raw NEREL splits from HuggingFace
- parses entities + links
- creates mention-level jsonl: one line = one mention with gold_qid (Q... or NULL)
- computes nested features (is_nested, nesting_level, inner_mentions)
- saves dataset stats to CSV

Usage examples:
1) Download + preprocess:
   python step1_build_mentions.py --download --raw_dir data/raw --out_dir data/processed

2) If you already have raw jsonl in data/raw:
   python step1_build_mentions.py --raw_dir data/raw --out_dir data/processed

Output:
- data/processed/train.jsonl, dev.jsonl, test.jsonl
- data/processed/dataset_stats.csv
- data/processed/type_dist_<split>.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from collections import Counter, defaultdict

# Optional progress bar
try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
    def tqdm(x, **kwargs):  # type: ignore
        return x


# Optional download
try:
    import requests  # type: ignore
except Exception:  # pragma: no cover
    requests = None


HF_BASE_URL_DEFAULT = "https://huggingface.co/datasets/iluvvatar/NEREL/resolve/main/data"


@dataclass(frozen=True)
class Entity:
    ent_id: str
    ent_type: str
    start: int
    stop: int
    provided_text: str


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def parse_entity_line(line: str) -> Entity:
    """
    Example:
      "T1\tPERSON 60 71\tБарак Обама"
    """
    parts = line.split("\t", 2)
    if len(parts) < 3:
        raise ValueError(f"Bad entity line: {line!r}")

    ent_id = parts[0].strip()
    meta = parts[1].strip().split()
    if len(meta) < 3:
        raise ValueError(f"Bad entity meta: {parts[1]!r} from {line!r}")

    ent_type = meta[0]
    start = int(meta[1])
    stop = int(meta[2])
    provided_text = parts[2]
    return Entity(ent_id=ent_id, ent_type=ent_type, start=start, stop=stop, provided_text=provided_text)


def parse_link_line(line: str) -> Tuple[str, str]:
    """
    Example:
      "N1\tReference T11 Wikidata:Q13133\tМишель Обама"
      "N18\tReference T7 Wikidata:NULL\t"
    Returns:
      (ent_id, qid_or_NULL)
    """
    parts = line.split("\t")
    if len(parts) < 2:
        raise ValueError(f"Bad link line: {line!r}")

    meta = parts[1].strip().split()
    if len(meta) < 3:
        raise ValueError(f"Bad link meta: {parts[1]!r} from {line!r}")

    ent_id = meta[1].strip()
    link = meta[2].strip()  # "Wikidata:Q..." or "Wikidata:NULL"
    if ":" in link:
        _, tail = link.split(":", 1)
    else:
        tail = link
    qid = tail.strip()
    if not qid:
        qid = "NULL"
    return ent_id, qid


def build_char_window(text: str, start: int, stop: int, window: int) -> Tuple[str, str, str, str]:
    """
    Returns: left, mention, right, context_full = left + [M]mention[/M] + right
    """
    left_start = max(0, start - window)
    right_stop = min(len(text), stop + window)

    left = text[left_start:start]
    mention = text[start:stop]
    right = text[stop:right_stop]
    context_full = f"{left}[M]{mention}[/M]{right}"
    return left, mention, right, context_full


def compute_nested_features(entities: List[Entity], doc_text: str) -> Dict[str, Dict[str, Any]]:
    """
    For each entity ent_id:
      is_nested: whether it is contained in any other entity span
      nesting_level: number of containers
      inner_mentions: list[str] of inner entity surface texts (sorted by offset)
    """
    # Sort entities by (start, stop) to make inner list ordered
    ents_sorted = sorted(entities, key=lambda e: (e.start, e.stop))

    feats: Dict[str, Dict[str, Any]] = {}

    for e in ents_sorted:
        containers = 0
        inner: List[Tuple[int, str]] = []

        for o in ents_sorted:
            if o.ent_id == e.ent_id:
                continue

            # o contains e
            if o.start <= e.start and o.stop >= e.stop and (o.start, o.stop) != (e.start, e.stop):
                containers += 1

            # o is inside e
            if e.start <= o.start and o.stop <= e.stop and (o.start, o.stop) != (e.start, e.stop):
                inner_text = doc_text[o.start:o.stop]
                inner.append((o.start, inner_text))

        inner_sorted = [t for _, t in sorted(inner, key=lambda x: x[0])]

        feats[e.ent_id] = {
            "is_nested": containers > 0,
            "nesting_level": containers,
            "inner_mentions": inner_sorted,
        }

    return feats


def download_file(url: str, out_path: Path) -> None:
    if requests is None:
        raise RuntimeError("requests is not installed. Either install it (pip install requests) or download files manually.")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0))
        downloaded = 0
        with out_path.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                f.write(chunk)
                downloaded += len(chunk)
                if total > 0:
                    pct = 100.0 * downloaded / total
                    print(f"\r  {out_path.name}: {pct:5.1f}% ({downloaded}/{total} bytes)", end="")
        print()


def ensure_raw_download(raw_dir: Path, base_url: str, splits: List[str]) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    for split in splits:
        path = raw_dir / f"{split}.jsonl"
        if path.exists():
            print(f"[skip] raw exists: {path}")
            continue
        url = f"{base_url}/{split}.jsonl"
        print(f"[download] {url}")
        download_file(url, path)


def build_processed_split(
    raw_path: Path,
    out_path: Path,
    split_name: str,
    context_window: int,
    include_only_linked: bool,
    min_mention_len: int,
    seed: int = 42,
    max_docs: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Builds mention-level dataset for one split and returns stats dict.
    """
    rng = random.Random(seed)

    docs = 0
    mentions = 0
    null_count = 0
    nested_count = 0
    unique_qids: set[str] = set()
    type_counter = Counter()
    mention_len_sum = 0
    mismatch_count = 0
    out_rows: List[Dict[str, Any]] = []

    for idx, doc in enumerate(tqdm(read_jsonl(raw_path), desc=f"process:{split_name}")):
        if max_docs is not None and docs >= max_docs:
            break

        text = doc.get("text", "")
        if not isinstance(text, str) or not text:
            continue

        raw_doc_id = doc.get("id")
        if raw_doc_id is None:
            doc_id = f"{split_name}_{idx}"
        else:
            doc_id = f"{split_name}_{raw_doc_id}"

        docs += 1

        # parse entities
        entities_raw = doc.get("entities", []) or []
        entities: List[Entity] = []
        for line in entities_raw:
            try:
                e = parse_entity_line(line)
                # sanity: offsets within bounds
                if not (0 <= e.start < e.stop <= len(text)):
                    continue
                entities.append(e)
            except Exception:
                continue

        if not entities:
            continue

        # parse links: ent_id -> qid
        links_raw = doc.get("links", []) or []
        link_map: Dict[str, str] = {}
        for line in links_raw:
            try:
                ent_id, qid = parse_link_line(line)
                link_map[ent_id] = qid
            except Exception:
                continue

        nested_feats = compute_nested_features(entities, text)

        for e in entities:
            if include_only_linked and e.ent_id not in link_map:
                continue

            gold_qid = link_map.get(e.ent_id, "NULL")
            left, mention_text, right, context_full = build_char_window(text, e.start, e.stop, window=context_window)

            # filter empty/too short mentions
            if len(mention_text.strip()) < min_mention_len:
                continue

            # mismatch check: provided_text vs slice (not fatal, but track)
            provided = e.provided_text
            if provided and provided != mention_text:
                mismatch_count += 1

            feats = nested_feats.get(e.ent_id, {"is_nested": False, "nesting_level": 0, "inner_mentions": []})

            row = {
                "doc_id": doc_id,
                "mention_id": f"{doc_id}_{e.ent_id}",
                "entity_type": e.ent_type,
                "start": e.start,
                "stop": e.stop,
                "mention_text": mention_text,
                "gold_qid": gold_qid,  # "Q..." or "NULL"
                "context_left": left,
                "context_right": right,
                "context_full": context_full,
                "is_nested": bool(feats["is_nested"]),
                "nesting_level": int(feats["nesting_level"]),
                "inner_mentions": list(feats["inner_mentions"]),
            }
            out_rows.append(row)

            # stats update
            mentions += 1
            type_counter[e.ent_type] += 1
            mention_len_sum += len(mention_text)

            if gold_qid == "NULL":
                null_count += 1
            else:
                unique_qids.add(gold_qid)

            if row["is_nested"]:
                nested_count += 1

    write_jsonl(out_path, out_rows)

    stats = {
        "split": split_name,
        "docs": docs,
        "mentions": mentions,
        "unique_qids": len(unique_qids),
        "null_rate": (null_count / mentions) if mentions else 0.0,
        "nested_rate": (nested_count / mentions) if mentions else 0.0,
        "avg_mention_len": (mention_len_sum / mentions) if mentions else 0.0,
        "mismatch_rate_entities_vs_slice": (mismatch_count / mentions) if mentions else 0.0,
    }

    # save type distribution for this split
    type_csv = out_path.parent / f"type_dist_{split_name}.csv"
    with type_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["entity_type", "count"])
        for t, c in type_counter.most_common():
            w.writerow([t, c])

    # also save a small random sample file (useful for sanity-check)
    sample_path = out_path.parent / f"samples_{split_name}.txt"
    sample_k = min(10, len(out_rows))
    if sample_k > 0:
        sample = rng.sample(out_rows, k=sample_k)
        with sample_path.open("w", encoding="utf-8") as f:
            for r in sample:
                f.write(f"mention_id: {r['mention_id']}\n")
                f.write(f"type: {r['entity_type']} | gold_qid: {r['gold_qid']}\n")
                f.write(f"mention: {r['mention_text']!r}\n")
                f.write(f"context: {r['context_full'][:400].replace('\\n',' ')}\n")
                f.write("-" * 80 + "\n")

    return stats


def write_stats_csv(stats_list: List[Dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "split",
        "docs",
        "mentions",
        "unique_qids",
        "null_rate",
        "nested_rate",
        "avg_mention_len",
        "mismatch_rate_entities_vs_slice",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for s in stats_list:
            w.writerow(s)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", type=str, default="data/raw", help="Directory with raw train/dev/test.jsonl")
    ap.add_argument("--out_dir", type=str, default="data/processed", help="Output directory for processed jsonl")
    ap.add_argument("--splits", type=str, nargs="+", default=["train", "dev", "test"])
    ap.add_argument("--context_window", type=int, default=200, help="Chars to the left/right of mention")
    ap.add_argument("--min_mention_len", type=int, default=1, help="Minimum mention length after strip")
    ap.add_argument("--include_only_linked", action="store_true", default=True, help="Keep only entities present in links[]")
    ap.add_argument("--download", action="store_true", help="Download raw splits from HuggingFace")
    ap.add_argument("--hf_base_url", type=str, default=HF_BASE_URL_DEFAULT)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_docs", type=int, default=None, help="Debug option: limit docs per split")
    args = ap.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.download:
        ensure_raw_download(raw_dir, args.hf_base_url, args.splits)

    stats_list = []
    for split in args.splits:
        raw_path = raw_dir / f"{split}.jsonl"
        if not raw_path.exists():
            raise FileNotFoundError(f"Missing raw split: {raw_path}. Use --download or put the file there.")
        out_path = out_dir / f"{split}.jsonl"

        print(f"\n[build] split={split} raw={raw_path} -> {out_path}")
        stats = build_processed_split(
            raw_path=raw_path,
            out_path=out_path,
            split_name=split,
            context_window=args.context_window,
            include_only_linked=args.include_only_linked,
            min_mention_len=args.min_mention_len,
            seed=args.seed,
            max_docs=args.max_docs,
        )
        stats_list.append(stats)

        print(
            f"[stats:{split}] docs={stats['docs']} mentions={stats['mentions']} "
            f"unique_qids={stats['unique_qids']} null_rate={stats['null_rate']:.3f} "
            f"nested_rate={stats['nested_rate']:.3f} avg_mention_len={stats['avg_mention_len']:.1f} "
            f"mismatch_rate={stats['mismatch_rate_entities_vs_slice']:.3f}"
        )

    stats_csv = out_dir / "dataset_stats.csv"
    write_stats_csv(stats_list, stats_csv)
    print(f"\n[saved] dataset stats -> {stats_csv}")
    print(f"[saved] type distributions -> {out_dir}/type_dist_<split>.csv")
    print(f"[saved] sanity samples -> {out_dir}/samples_<split>.txt")


if __name__ == "__main__":
    main()
