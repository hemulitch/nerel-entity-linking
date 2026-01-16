from __future__ import annotations
import argparse
import json
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple
from rank_bm25 import BM25Okapi


WORD_RE = re.compile(r"[0-9A-Za-zА-Яа-яЁё_]+")


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def normalize_text(s: str) -> str:
    s = s.replace("Ё", "Е").replace("ё", "е").lower()
    return s


def tokenize(s: str) -> List[str]:
    s = normalize_text(s)
    return WORD_RE.findall(s)


@dataclass
class BM25Index:
    qids: List[str]
    bm25: BM25Okapi


def build_entity_text(rec: Dict[str, Any], max_aliases: int = 30) -> str:
    label = rec.get("label", "") or ""
    desc = rec.get("description", "") or ""
    aliases = rec.get("aliases", []) or []
    aliases = aliases[:max_aliases]
    parts = [label]
    if aliases:
        parts.append(" ; ".join(aliases))
    if desc:
        parts.append(desc)
    return "\n".join(p for p in parts if p).strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kb", type=str, default="data/kb/entities.jsonl")
    ap.add_argument("--out_dir", type=str, default="data/kb/bm25")
    args = ap.parse_args()

    kb_path = Path(args.kb)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = list(read_jsonl(kb_path))
    qids = [r["qid"] for r in rows]
    texts = [build_entity_text(r) for r in rows]
    tokenized = [tokenize(t) for t in texts]

    bm25 = BM25Okapi(tokenized)
    idx = BM25Index(qids=qids, bm25=bm25)

    with (out_dir / "bm25_index.pkl").open("wb") as f:
        pickle.dump(idx, f)

    # сохраним и тексты (полезно для дебага)
    with (out_dir / "corpus.jsonl").open("w", encoding="utf-8") as f:
        for q, t in zip(qids, texts):
            f.write(json.dumps({"qid": q, "text": t}, ensure_ascii=False) + "\n")

    print(f"[saved] bm25 index -> {out_dir/'bm25_index.pkl'} docs={len(qids)}")


if __name__ == "__main__":
    main()
