#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

import requests
from tqdm import tqdm

# вытаскиваем только корректные ID
ID_RE = re.compile(r"(Q\d+|P\d+)")

def canonical_id(s: str) -> Optional[str]:
    """
    Приводим к каноническому виду:
    - если NULL -> None
    - если в строке встречается Q123 или P123 -> возвращаем его
    - иначе -> None (шум)
    """
    if s is None:
        return None
    s = str(s).strip()
    if not s:
        return None
    if s.upper() == "NULL":
        return None
    m = ID_RE.search(s)
    if not m:
        return None
    return m.group(1)


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


def load_existing_kb(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        return {}
    kb = {}
    for row in read_jsonl(path):
        kb[row["qid"]] = row
    return kb


def collect_qids(processed_dir: Path, qids_from: str) -> List[str]:
    splits = ["train"] if qids_from == "train" else ["train", "dev", "test"]
    qids: Set[str] = set()
    dropped = 0

    for sp in splits:
        p = processed_dir / f"{sp}.jsonl"
        for ex in read_jsonl(p):
            raw = ex.get("gold_qid", "")
            cid = canonical_id(raw)
            if cid is None:
                if str(raw).strip().upper() != "NULL":
                    dropped += 1
                continue
            qids.add(cid)

    if dropped:
        print(f"[warn] dropped malformed non-NULL gold ids (during qid collection): {dropped}")

    return sorted(qids)


@dataclass
class WikidataClient:
    endpoint: str
    user_agent: str
    sleep: float = 0.2
    timeout: int = 90
    max_retries: int = 5

    def query(self, sparql: str) -> dict:
        headers = {
            "Accept": "application/sparql-results+json",
            "User-Agent": self.user_agent,
        }

        # POST предпочтительнее GET (меньше проблем с длиной URL)
        for attempt in range(self.max_retries):
            r = requests.post(self.endpoint, data={"query": sparql}, headers=headers, timeout=self.timeout)

            if r.status_code in (429, 503):
                time.sleep(1.0 + attempt * 1.5)
                continue

            if r.status_code == 400:
                # если вдруг ещё раз словим 400 — выведем кусок запроса
                snippet = sparql[:800].replace("\n", " ")
                raise RuntimeError(f"Wikidata SPARQL 400. Query snippet: {snippet}")

            r.raise_for_status()
            return r.json()

        raise RuntimeError(f"Failed to query Wikidata after {self.max_retries} retries")

    def fetch_batch(self, qids: List[str]) -> Dict[str, Dict[str, Any]]:
        if not qids:
            return {}

        # здесь qids уже канонические (Q\d+|P\d+)
        values = " ".join(f"wd:{qid}" for qid in qids)

        sparql = f"""
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
PREFIX wikibase: <http://wikiba.se/ontology#>
PREFIX bd: <http://www.bigdata.com/rdf#>

SELECT ?item ?itemLabel ?itemDescription ?altLabel WHERE {{
  VALUES ?item {{ {values} }}
  OPTIONAL {{
    ?item skos:altLabel ?altLabel .
    FILTER (lang(?altLabel) = "ru" || lang(?altLabel) = "en")
  }}
  SERVICE wikibase:label {{
    bd:serviceParam wikibase:language "ru,en".
  }}
}}
"""

        data = self.query(sparql)
        rows = data.get("results", {}).get("bindings", [])

        out: Dict[str, Dict[str, Any]] = {qid: {"qid": qid, "label": "", "description": "", "aliases": []} for qid in qids}

        for b in rows:
            uri = b["item"]["value"]
            qid = uri.rsplit("/", 1)[-1]

            label = b.get("itemLabel", {}).get("value", "") or ""
            desc = b.get("itemDescription", {}).get("value", "") or ""
            alt = b.get("altLabel", {}).get("value", "") or ""

            rec = out.setdefault(qid, {"qid": qid, "label": "", "description": "", "aliases": []})
            if label and not rec["label"]:
                rec["label"] = label
            if desc and not rec["description"]:
                rec["description"] = desc
            if alt:
                rec["aliases"].append(alt)

        # dedup aliases
        for qid, rec in out.items():
            seen = set()
            uniq = []
            for a in rec["aliases"]:
                if a in seen:
                    continue
                seen.add(a)
                uniq.append(a)
            rec["aliases"] = uniq

        time.sleep(self.sleep)
        return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed_dir", type=str, default="data/processed")
    ap.add_argument("--out_kb", type=str, default="data/kb/entities_all.jsonl")
    ap.add_argument("--qids_from", type=str, default="all", choices=["train", "all"])
    ap.add_argument("--endpoint", type=str, default="https://query.wikidata.org/sparql")
    ap.add_argument("--batch_size", type=int, default=30)  # можно 30, чтобы запросы были короче
    ap.add_argument("--sleep", type=float, default=0.2)
    ap.add_argument("--user_agent", type=str, default="nerel-el-course-project/0.1 (contact: you@example.com)")
    args = ap.parse_args()

    processed_dir = Path(args.processed_dir)
    out_kb = Path(args.out_kb)
    out_kb.parent.mkdir(parents=True, exist_ok=True)

    qids = collect_qids(processed_dir, args.qids_from)
    print(f"[qids] total={len(qids)} (qids_from={args.qids_from})")

    existing = load_existing_kb(out_kb)
    missing = [q for q in qids if q not in existing]
    print(f"[kb] existing={len(existing)} missing={len(missing)}")

    client = WikidataClient(endpoint=args.endpoint, user_agent=args.user_agent, sleep=args.sleep)

    for i in tqdm(range(0, len(missing), args.batch_size), desc="wikidata"):
        batch = missing[i:i + args.batch_size]
        got = client.fetch_batch(batch)
        existing.update(got)

    # гарантируем непустые label (иначе BM25 документы будут пустыми)
    rows = []
    for q in qids:
        rec = existing.get(q, {"qid": q, "label": "", "description": "", "aliases": []})
        if not rec.get("label"):
            rec["label"] = q
        rows.append(rec)

    write_jsonl(out_kb, rows)
    print(f"[saved] kb -> {out_kb} rows={len(rows)}")


if __name__ == "__main__":
    main()
