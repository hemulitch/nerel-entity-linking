## NEREL Entity Linking (RU)

Lightweight baseline pipeline for mention-level Entity Linking on the NEREL dataset:
- Build processed mention data from raw NEREL JSONL
- Build a compact KB from Wikidata
- Create a BM25 index and evaluate candidate recall
- Run BM25 Top-1 with NIL calibration
- Optional: surface-form baseline, cross-encoder reranker

### Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 1) Data preparation

Option A — auto-download raw splits from HuggingFace and preprocess:

```bash
python scripts/build_mentions.py --download --raw_dir data/raw --out_dir data/processed
```

Option B — manual download:

```bash
wget -O data/raw/dev.jsonl  https://huggingface.co/datasets/iluvvatar/NEREL/resolve/main/data/dev.jsonl
wget -O data/raw/test.jsonl https://huggingface.co/datasets/iluvvatar/NEREL/resolve/main/data/test.jsonl
wget -O data/raw/train.jsonl https://huggingface.co/datasets/iluvvatar/NEREL/resolve/main/data/train.jsonl

python scripts/build_mentions.py --raw_dir data/raw --out_dir data/processed
```

Outputs:
- `data/processed/{train,dev,test}.jsonl`
- `data/processed/dataset_stats.csv`, `type_dist_<split>.csv` and small sample files

### 2) Build KB from Wikidata

```bash
python scripts/build_kb_wikidata.py \
  --processed_dir data/processed \
  --qids_from all \
  --out_kb data/kb/entities_all.jsonl
```

Notes:
- Be polite to Wikidata. You can tune `--batch_size` and `--sleep`, and set a descriptive `--user_agent`.

### 3) Build BM25 index

```bash
python scripts/build_bm25_index.py \
  --kb data/kb/entities_all.jsonl \
  --out_dir data/kb/bm25_all
```

Saves `bm25_index.pkl` and `corpus.jsonl`.

### 4) Candidate recall (BM25@K)

```bash
python scripts/eval_bm25_candidates.py \
  --processed_dir data/processed \
  --split dev \
  --bm25_dir data/kb/bm25_all \
  --k_list 1 5 10 50 100 \
  --use_context \
  --out_dir runs/bm25_candidates
```

### 5) BM25 Top‑1 with NIL calibration

Calibrate on dev:

```bash
python scripts/bm25_top1_nil.py \
  --processed_dir data/processed \
  --split dev \
  --bm25_dir data/kb/bm25_all \
  --out_dir runs/bm25_top1_all \
  --calibrate_on_dev
```

Run on test with saved threshold:

```bash
python scripts/bm25_top1_nil.py \
  --processed_dir data/processed \
  --split test \
  --bm25_dir data/kb/bm25_all \
  --out_dir runs/bm25_top1_all \
  --calibration_path runs/bm25_top1_all/dev/calibration.json
```

### Surface-form baseline (optional)

```bash
python scripts/surface_baseline.py --split dev  --processed_dir data/processed --save_ambiguity
python scripts/surface_baseline.py --split test --processed_dir data/processed
```

### Cross‑encoder reranker 

1) Create training pairs:

```bash
python scripts/make_reranker_pairs.py \
  --processed_dir data/processed \
  --bm25_dir data/kb/bm25_all \
  --kb_path data/kb/entities_all.jsonl \
  --out_path data/reranker/train_pairs.jsonl \
  --candidate_k 100 \
  --max_train_mentions 2000 \
  --neg_per_pos 5 \
  --force_add_gold \
  --add_surface_candidate
  --use_inner_mentions
```

2) Train:

```bash
python scripts/train_reranker.py \
  --train_pairs data/reranker/train_pairs.jsonl \
  --model_name cointegrated/rubert-tiny2 \
  --output_dir runs/reranker_tiny
  --batch_size 16
  --epochs 1
```

3) Evaluate:

```bash
# Dev 
python scripts/eval_reranker.py \
  --processed_dir data/processed \
  --split dev \
  --bm25_dir data/kb/bm25_all \
  --kb_path data/kb/entities_all.jsonl \
  --model_dir runs/reranker_tiny \
  --out_dir runs/reranker_eval \
  --candidate_k 100 \
  --add_surface_candidate \
  --use_inner_mentions \
  --out_dir runs/reranker_tiny

# Test 
python scripts/eval_reranker.py \
  --processed_dir data/processed \
  --split test \
  --bm25_dir data/kb/bm25_all \
  --kb_path data/kb/entities_all.jsonl \
  --model_dir runs/reranker_tiny \
  --out_dir runs/reranker_eval \
  --candidate_k 100 \
  --add_surface_candidate \
  --use_inner_mentions \
  --out_dir runs/reranker_tiny
```

### Outputs layout
- `runs/surface_baseline/<split>/{metrics.json,predictions.jsonl}`
- `runs/bm25_candidates/<split>/candidate_recall.json`
- `runs/bm25_top1_all/<split>/{metrics.json,predictions.jsonl}` and `runs/bm25_top1_all/dev/calibration.json`
- `runs/reranker_eval/<split>/{metrics.json,predictions.jsonl}` (+ optional `calibration.json` for dev)

### Notes
- Python 3.9+ recommended (uses `argparse.BooleanOptionalAction`).
- If `torch` installation fails on your platform, follow the official PyTorch install selector.
- NEREL data courtesy of the `iluvvatar/NEREL` dataset on HuggingFace.
