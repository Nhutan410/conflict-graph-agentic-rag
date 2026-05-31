# SCRIPTS_GUIDE.md
# Hướng dẫn đầy đủ các scripts cần chạy — RESE006 Pipeline

> Đọc file này để biết **chạy gì, theo thứ tự nào, input là gì, output ra đâu**.
> Mọi lệnh đều chạy từ thư mục gốc `RESE006/` với conda env `rese006`.

---

## QUICK START (3 bước)

```bash
# 1. Kích hoạt môi trường
conda activate rese006

# 2. Preprocess data (chỉ cần chạy 1 lần)
python data/preprocess.py --max_queries 10

# 3. Chạy pipeline trên 1 query
python scripts/run_pipeline.py --query_id q_002 --output outputs/result.json
```

---

## CẤU TRÚC THƯ MỤC

```
RESE006/
├── .env                    ← OPENAI_API_KEY
├── configs/
│   └── default.yaml        ← hyperparameters pipeline
├── data/
│   ├── raw/
│   │   └── RAMDocs_test.jsonl          ← dataset gốc (500 queries)
│   ├── preprocessed/
│   │   ├── queries.jsonl               ← 500 queries đã normalize
│   │   ├── documents.jsonl             ← 500 × N docs đã format
│   │   ├── claims.jsonl                ← claims extracted bởi LLM
│   │   └── metadata.jsonl              ← ground truth (CHỈ evaluation)
│   ├── preprocess.py       ← Script 1: tạo preprocessed data
│   └── loaders.py          ← utility: load data vào Pydantic models
├── src/                    ← toàn bộ pipeline logic
│   ├── schema.py           ← Pydantic models (Document, Claim LoopResult, ...)
│   ├── extractor.py        ← Phase 1: LLM claim extraction
│   ├── embedder.py         ← Phase 2: BGE-M3 embedding + canonical merge
│   ├── graph_builder.py    ← Phase 3: NLI evidence graph
│   ├── retriever.py        ← Phase 4: BM25 + Dense + RRF hybrid retrieval
│   ├── conflict_zone.py    ← Phase 5: credibility arbitration + factoid localization
│   ├── iterative_loop.py   ← Phase 6: targeted retrieval loop
│   ├── generator.py        ← Phase 7: grounded answer generation
│   └── pipeline.py         ← orchestrator: nối tất cả 7 phases
├── scripts/
│   └── run_pipeline.py     ← Script 2: chạy pipeline end-to-end
├── evaluation/
│   └── retrieval_eval.py   ← utility: tính metrics (Recall@k, MRR, F1, ...)
└── tests/
    └── test_smoke.py       ← Script 3: smoke tests (không cần OpenAI/GPU)
```

---

## THIẾT LẬP MÔI TRƯỜNG

### Bước 0.1 — Tạo conda env (chỉ cần 1 lần)

```bash
conda env create -f environment.yml
conda activate rese006
```

**`environment.yml` cài gì:**
- Python 3.10
- faiss-cpu (từ conda-forge, Mac compatible)
- pip packages: torch, transformers, sentence-transformers, openai, pydantic, networkx, rank-bm25, pyyaml, numpy

### Bước 0.2 — Cấu hình API Key

```bash
# File .env đã có sẵn, hoặc tạo thủ công:
echo "OPENAI_API_KEY=sk-proj-..." > .env
```

Script tự đọc `.env` khi chạy — không cần `export`.

### Bước 0.3 — Kiểm tra môi trường

```bash
python -c "from src.pipeline import ConflictAwarePipeline; print('OK')"
```

---

## SCRIPT 1 — `data/preprocess.py`

### Mục đích

Chuyển đổi `RAMDocs_test.jsonl` (raw format) → 4 files preprocessed theo pipeline schema.

Sử dụng **gpt-4o-mini** để extract atomic factual claims (không phải sentence split đơn thuần).

### Khi nào cần chạy

- Lần đầu setup project
- Khi muốn tái tạo claims với model khác
- Khi dataset thay đổi

### Input

| File | Mô tả |
|---|---|
| `data/raw/RAMDocs_test.jsonl` | 500 queries, mỗi query gồm question + N documents (correct/misinfo/noise) |

**Format mỗi dòng RAMDocs:**
```json
{
  "question": "Who directed Lahu Ke Do Rang?",
  "documents": [
    {"text": "...", "type": "correct", "answer": "Mahesh Bhatt"},
    {"text": "...", "type": "misinfo", "answer": "Raj Kapoor"}
  ],
  "gold_answers": ["Mahesh Bhatt"],
  "wrong_answers": ["Raj Kapoor"]
}
```

### Output

| File | Dòng | Mô tả | Feed vào pipeline? |
|---|---|---|---|
| `data/preprocessed/queries.jsonl` | 500 | `{query_id, user_query}` | Có — run_pipeline.py đọc |
| `data/preprocessed/documents.jsonl` | 500 | `{query_id, documents[]}` | Có — khi dùng `--use_documents` |
| `data/preprocessed/claims.jsonl` | ~5,000–15,000 | `Claim[]` từ LLM | Có — input Phase 2 |
| `data/preprocessed/metadata.jsonl` | 500 | gold_answers, doc_labels | **KHÔNG** — chỉ evaluation |

**Format mỗi dòng `claims.jsonl`:**
```json
{
  "claim_id":           "c_q002_d00_s01",
  "claim_text":         "Lahu Ke Do Rang was directed by Mahesh Bhatt.",
  "evidence":           "Lahu Ke Do Rang was directed by Mahesh Bhatt.",
  "doc_id":             "d_d0d46c",
  "source_id":          "s_q002_00",
  "claim_embedding":    null,
  "retrieval_relevance": 1.0,
  "claim_confidence":   0.8764,
  "credibility_score":  null,
  "is_representative":  true,
  "merged_claim_ids":   null
}
```

> `claim_embedding: null` và `credibility_score: null` — được compute bởi pipeline (Phase 2a, Phase 5).

### Cách chạy

```bash
# Chạy toàn bộ 500 queries với gpt-4o-mini (mất ~10-15 phút, ~$0.50)
python data/preprocess.py

# Chỉ chạy 10 queries đầu (test nhanh, ~1 phút)
python data/preprocess.py --max_queries 10

# Không dùng LLM (sentence-split, nhanh nhưng quality thấp)
python data/preprocess.py --no_llm

# Bỏ qua noise documents (chỉ giữ correct + misinfo)
python data/preprocess.py --skip_noise

# Custom output directory
python data/preprocess.py --output_dir /path/to/output
```

### Tất cả arguments

| Argument | Default | Ý nghĩa |
|---|---|---|
| `--input` | `data/raw/RAMDocs_test.jsonl` | Path dataset gốc |
| `--output_dir` | `data/preprocessed` | Thư mục output |
| `--max_queries` | `None` (tất cả) | Giới hạn số queries xử lý |
| `--skip_noise` | `False` | Bỏ qua noise documents |
| `--no_llm` | `False` | Dùng sentence-split thay vì LLM |
| `--model` | `gpt-4o-mini` | OpenAI model cho extraction |

### Xử lý bên trong

```
RAMDocs_test.jsonl
    │
    ├─► build_query()          → {query_id, user_query}
    │
    ├─► build_documents()      → list[{doc_id, source_id, text, retrieval_score, metadata}]
    │       Truncate nếu > 3000 chars
    │
    ├─► extract_claims_llm()   → list[claim_dict]
    │       1. is_boilerplate() — filter Wikipedia navigation, edit links, URLs
    │       2. clean_doc_text() — bỏ artifacts, normalize
    │       3. gpt-4o-mini prompt → atomic factual statements
    │       4. Filter claims < 4 words hoặc > 40 words
    │
    ├─► run_phase2_canonical()  → list[claim_dict] (deduplicated)
    │       Merge claims với Jaccard overlap > 0.55 VÀ cùng temporal/numerical values
    │       Giữ riêng claims có số/năm khác nhau (nguồn gốc conflict)
    │
    └─► build_metadata()       → ground truth record
```

### Ghi chú quan trọng

- **`claims.jsonl` là input chính của pipeline** — bỏ qua Phase 1 LLM extraction khi đã có file này
- `metadata.jsonl` chứa `doc_type` (correct/misinfo/noise) — **không feed vào pipeline** để tránh leak ground truth
- Preprocessing tự flush file sau mỗi query — có thể xem tiến độ bằng `wc -l data/preprocessed/claims.jsonl`

---

## SCRIPT 2 — `scripts/run_pipeline.py`

### Mục đích

Chạy toàn bộ **7-phase conflict-aware pipeline** trên 1 query từ RAMDocs data. Tự đọc `.env`, load config, load data, chạy pipeline, lưu kết quả.

### Khi nào cần chạy

- Chạy pipeline trên query cụ thể để xem kết quả
- Batch run nhiều queries để evaluation
- Demo hoặc debug từng query

### Input

| Nguồn | Mô tả |
|---|---|
| `data/preprocessed/queries.jsonl` | Để tìm query_id → user_query |
| `data/preprocessed/claims.jsonl` | Claims đã extracted (skip Phase 1) |
| `data/preprocessed/documents.jsonl` | Documents (chỉ khi dùng `--use_documents`) |
| `configs/default.yaml` | Hyperparameters pipeline |
| `.env` | `OPENAI_API_KEY` |

### Output

JSON file (hoặc stdout) với cấu trúc:

```json
{
  "query_id": "q_002",
  "query": "Who are the directors of the film Lahu Ke Do Rang?",
  "resolved": false,
  "iterations_run": 1,
  "final_answer": "The film was directed by Mahesh Bhatt...",
  "n_validated_claims": 1,
  "n_conflict_localizations": 2,
  "conflict_localizations": [
    {
      "claim_i_id": "c_q002_d00_s01",
      "claim_j_id": "c_q002_d03_s05",
      "slot": "temporal",
      "value_i": "1979",
      "value_j": "1974",
      "conflict_intensity": 0.5,
      "credibility_i": 1.94,
      "credibility_j": -8.49
    }
  ],
  "validated_claims": [
    {
      "claim_id": "c_q002_d00_s01",
      "doc_id": "d_d0d46c",
      "text": "Lahu Ke Do Rang was directed by Mahesh Bhatt.",
      "retrieval_relevance": 0.73,
      "claim_confidence": 0.88
    }
  ]
}
```

### Cách chạy

```bash
# Chạy 1 query cụ thể, in ra stdout
python scripts/run_pipeline.py --query_id q_002

# Lưu kết quả vào file
python scripts/run_pipeline.py --query_id q_002 --output outputs/q002.json

# Chạy query đầu tiên (mặc định)
python scripts/run_pipeline.py

# Giới hạn số claims (test nhanh, Phase 3 NLI nhẹ hơn)
python scripts/run_pipeline.py --query_id q_002 --limit 10

# Chạy từ raw documents (Phase 1 LLM extraction, bỏ qua claims.jsonl)
python scripts/run_pipeline.py --query_id q_002 --use_documents

# Dùng config tùy chỉnh
python scripts/run_pipeline.py --config configs/my_config.yaml

# Chỉ hiện WARNING trở lên (ít log hơn)
python scripts/run_pipeline.py --query_id q_002 --log_level WARNING
```

### Tất cả arguments

| Argument | Default | Ý nghĩa |
|---|---|---|
| `--query_id` | query đầu tiên | Query ID cần chạy, ví dụ `q_002` |
| `--limit` | `None` | Giới hạn số claims load (bỏ qua Phase 3 nặng) |
| `--output` | `None` (stdout) | Path file JSON để lưu kết quả |
| `--config` | `configs/default.yaml` | Path YAML config |
| `--use_documents` | `False` | Chạy từ raw docs (Phase 1 extraction) thay vì claims |
| `--log_level` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR` |

### Luồng thực thi bên trong

```
run_pipeline.py
    │
    ├─► Đọc .env → OPENAI_API_KEY vào os.environ
    ├─► Load config từ configs/default.yaml → PipelineConfig
    ├─► ConflictAwarePipeline(config) → khởi tạo tất cả components
    │
    ├─► [Nếu --use_documents]
    │       load_documents_for_query(query_id) → list[Document]
    │       pipeline.run(documents=docs) → Phase 1 LLM extraction
    │
    ├─► [Mặc định — preprocessed claims]
    │       load_claims(query_id) → list[Claim] (embedding=None)
    │       pipeline.run(claims=claims)
    │           ├── Phase 2a: ClaimEmbedder (bge-m3) → embeddings 1024-dim
    │           ├── Phase 3:  ClaimGraphBuilder (DeBERTa NLI) → nx.DiGraph
    │           ├── Phase 4:  HybridRetriever (BM25+Dense+RRF) → top-k claims
    │           ├── Phase 5:  ConflictZoneAnalyzer → credibility scores + localizations
    │           ├── Phase 6:  IterativeLoop → targeted queries, update graph
    │           └── Phase 7:  AnswerGenerator (gpt-4o-mini) → final_answer
    │
    └─► Lưu/print kết quả JSON
```

### Ví dụ kết quả thực tế (q_002)

```
Query  : Who are the directors of the film "Lahu Ke Do Rang"?
Gold   : Mahesh Bhatt
Misinfo: Raj Kapoor

resolved     = False
iterations   = 1
validated    = 1 claim (Mahesh Bhatt, cred = +1.94)
suppressed   = 21 claims (Raj Kapoor docs, cred = -8.49)
conflicts    = 2 conflict localizations [temporal]

Answer: "The film 'Lahu Ke Do Rang' (1979) was directed by Mahesh Bhatt.
         Conflicting sources were suppressed by credibility arbitration."
```

---

## SCRIPT 3 — `tests/test_smoke.py`

### Mục đích

Kiểm tra toàn bộ codebase không bị broken mà **không cần OpenAI API** và **không cần GPU**. Dùng mock objects và unit embeddings.

### Khi nào cần chạy

- Sau mỗi thay đổi code
- Trước khi commit
- Để verify môi trường setup đúng

### Input

Không cần file data — tất cả dùng mock objects tạo trong test.

### Output

Pytest report — pass/fail mỗi test case.

### Cách chạy

```bash
# Chạy tất cả 35 tests
python -m pytest tests/test_smoke.py

# Verbose output (thấy tên từng test)
python -m pytest tests/test_smoke.py -v

# Chỉ chạy 1 nhóm test
python -m pytest tests/test_smoke.py -k "TestCredibilityArbitrator"
python -m pytest tests/test_smoke.py -k "TestClaimCanonical"
python -m pytest tests/test_smoke.py -k "TestEvidenceSelector"

# Dừng khi gặp failure đầu tiên
python -m pytest tests/test_smoke.py -x

# Tóm tắt ngắn gọn (không verbose)
python -m pytest tests/test_smoke.py -q
```

### 35 test cases và ý nghĩa

| Nhóm | Test | Kiểm tra gì |
|---|---|---|
| **TestSchemaValidation** (7) | `test_document_creation` | Document Pydantic model valid |
| | `test_claim_creation` | Claim Pydantic model valid |
| | `test_edge_validation_valid` | Edge với relation hợp lệ |
| | `test_edge_validation_invalid` | Edge với relation sai → ValidationError |
| | `test_loop_result_creation` | LoopResult có final_answer |
| | `test_conflict_localization` | ConflictLocalization fields đúng |
| | `test_factoid_slots` | FactoidSlots nullable fields |
| **TestAdapter** (4) | `test_claim_from_record_basic` | `claim_text` → `text` mapping đúng |
| | `test_claim_from_record_with_embedding` | `claim_embedding` → `embedding` mapping |
| | `test_doc_from_record_basic` | `source_id` → `source` mapping |
| | `test_claim_from_record_defaults` | `source_credibility` default = -1.0 |
| **TestClaimCanonical** (5) | `test_canonicalize_identical_embeddings_merges` | Claims identical → merge thành 1 |
| | `test_canonicalize_orthogonal_embeddings_keeps_both` | Claims khác xa → giữ 2 |
| | `test_canonicalize_high_sim_different_numbers_keeps_both` | Sim cao nhưng khác số → giữ 2 (conflict) |
| | `test_canonicalize_single_claim` | 1 claim → return nguyên |
| | `test_canonicalize_missing_embedding_raises` | Thiếu embedding → ValueError |
| **TestPairGenerator** (4) | `test_pair_generator_high_similarity` | Sim cao → tạo pair |
| | `test_pair_generator_low_similarity` | Sim thấp → không tạo pair |
| | `test_pair_generator_single_claim` | 1 claim → 0 pairs |
| | `test_pair_generator_multiple_claims` | N claims → đúng số pairs |
| **TestCredibilityArbitrator** (5) | `test_arbitration_returns_all_nodes` | Output có đủ claim_ids |
| | `test_arbitration_contradiction_reduces_score` | Contradiction → score âm |
| | `test_arbitration_support_increases_score` | Support → score dương |
| | `test_arbitration_empty_graph` | Graph rỗng → dict rỗng |
| | `test_arbitration_convergence` | Converge sau ≤ max_iterations |
| **TestConflictQueryFormulator** (5) | `test_temporal_slot_query` | Temporal slot → "Which year is correct?" |
| | `test_numerical_slot_query` | Numerical slot → "Find authoritative evidence" |
| | `test_entity_slot_query` | Entity slot → "Disambiguate entity conflict" |
| | `test_location_slot_query` | Location slot → "Verify location" |
| | `test_unknown_slot_query` | Unknown slot → "Fact-check" |
| **TestEvidenceSelector** (5) | `test_select_returns_at_most_max_claims` | Không vượt max_claims |
| | `test_select_empty_returns_empty` | Input rỗng → output rỗng |
| | `test_select_orders_by_credibility` | Sort theo credibility giảm dần |
| | `test_select_uses_credibility_scores_override` | Override credibility_scores dict |
| | `test_select_with_negative_relevance` | retrieval_relevance âm → treat as 0 |

---

## SCRIPT 4 — `data/loaders.py` (utility, không chạy trực tiếp)

### Mục đích

Chứa các hàm load và convert data từ `data/preprocessed/` sang Pydantic models. Được import bởi `scripts/run_pipeline.py`.

### Các hàm chính

```python
# Load claims cho 1 query cụ thể → list[Claim]
claims = load_claims("q_002")

# Load claims của tất cả queries → list[Claim]
all_claims = load_all_claims()

# Load danh sách queries → list[dict]
queries = load_queries()

# Load documents của 1 query → list[Document]
docs = load_documents_for_query("q_002")

# Load metadata (ground truth) → dict[query_id → record]
meta = load_metadata()
```

### Field mapping khi load claims

| `claims.jsonl` field | Pydantic `Claim` field | Ghi chú |
|---|---|---|
| `claim_text` | `text` | Rename |
| `claim_embedding` | `embedding` | Rename, `null` → `None` |
| `retrieval_relevance` | `retrieval_relevance` | Giữ nguyên |
| `claim_confidence` | `claim_confidence` | Giữ nguyên |
| `source_id` | (không map) | Bỏ qua |
| `evidence` | (không map) | Bỏ qua |
| `credibility_score` | (không map) | Computed tại Phase 5 |
| *(không có)* | `source_credibility` | Set `-1.0` |

---

## MODULE 5 — `evaluation/retrieval_eval.py` (utility)

### Mục đích

Tính các metrics đánh giá retrieval và generation. Import trong evaluation scripts.

### Các hàm

```python
from evaluation.retrieval_eval import (
    recall_at_k,         # Recall@k cho 1 query
    mean_reciprocal_rank,# MRR cho 1 query
    ndcg_at_k,           # nDCG@k cho 1 query
    answer_f1,           # Token-level F1 giữa prediction và gold
    answer_exact_match,  # Exact match
    evaluate_retrieval,  # Aggregate Recall@k, MRR, nDCG trên toàn dataset
    evaluate_generation, # Aggregate F1, EM trên toàn dataset
)
```

### Ví dụ dùng

```python
from evaluation.retrieval_eval import evaluate_generation
from data.loaders import load_metadata

# Load ground truth
meta = load_metadata()
gold = {qid: rec["gold_answers"][0] for qid, rec in meta.items() if rec["gold_answers"]}

# Load pipeline results
import json
results = [json.loads(l) for l in open("outputs/batch_results.jsonl")]

# Tính metrics
metrics = evaluate_generation(results, gold)
print(metrics)  # {"Answer_F1": 0.72, "Answer_EM": 0.41}
```

---

## THỨ TỰ CHẠY KHUYẾN NGHỊ

### Lần đầu setup

```bash
# 1. Tạo môi trường
conda env create -f environment.yml
conda activate rese006

# 2. Kiểm tra setup
python -m pytest tests/test_smoke.py -q

# 3. Preprocess 10 queries để test nhanh
python data/preprocess.py --max_queries 10

# 4. Chạy pipeline thử 1 query
python scripts/run_pipeline.py --query_id q_002 --log_level WARNING --output outputs/test.json

# 5. Preprocess toàn bộ 500 queries (background, mất ~15 phút)
nohup python data/preprocess.py > logs/preprocess.log 2>&1 &
```

### Mỗi lần thay đổi code

```bash
# Chạy tests trước khi làm gì khác
python -m pytest tests/test_smoke.py -q
```

### Chạy pipeline trên nhiều queries

```bash
# Chạy batch bằng shell loop
for qid in q_000 q_001 q_002 q_003 q_004; do
    python scripts/run_pipeline.py \
        --query_id $qid \
        --log_level WARNING \
        --output outputs/${qid}.json
done
```

---

## HYPERPARAMETERS — `configs/default.yaml`

```yaml
# Models
embedding_model: "BAAI/bge-m3"           # Phase 2, 4 — embed claims + query
nli_model: "cross-encoder/nli-deberta-v3-base" # Phase 3 — NLI edge labeling
llm_model: "gpt-4o-mini"                  # Phase 1, 5, 7 — extraction + generation

# Phase 2b — Claim Canonical
canonical_threshold: 0.92  # cosine sim để candidate merge (cao = merge ít hơn)

# Phase 3 — Evidence Graph
pair_similarity_threshold: 0.3  # cosine sim để tạo NLI pair (thấp = nhiều pairs hơn)
nli_edge_threshold: 0.5         # NLI score dưới ngưỡng → bỏ edge

# Phase 4 — Hybrid Retrieval
retrieval_top_k: 10     # số claims trả về
retrieval_alpha: 0.5    # dense weight (0.5 = BM25 và Dense ngang nhau)
min_conflict_pairs: 1   # tối thiểu 1 contradiction pair trong top-k

# Phase 5 — Conflict Zone
arbitration_max_iter: 10        # vòng lặp arbitration tối đa
arbitration_convergence: 0.01   # dừng khi max_delta < 0.01
arbitration_damping: 0.85       # damping tránh oscillation
credibility_threshold: 0.0      # score > 0 → validated, ≤ 0 → suppressed

# Phase 6 — Iterative Loop
max_loop_iterations: 3      # tối đa 3 vòng lặp
intensity_threshold: 0.1    # conflict_intensity < 0.1 → coi là resolved
min_intensity_reduction: 0.1 # giảm < 0.1 giữa iterations → dừng

# Phase 7 — Generation
max_evidence_claims: 10  # số claims tối đa đưa vào LLM prompt

seed: 42     # reproducibility
device: "cpu"  # "cpu" hoặc "cuda"
```

---

## TROUBLESHOOTING

### `ModuleNotFoundError: No module named 'src'`

```bash
# Đảm bảo chạy từ thư mục RESE006/
cd /Users/nguyenan/Project/RESE006
python scripts/run_pipeline.py ...
```

### `No claims found for query_id=q_XXX`

Preprocessing chưa chạy đến query đó. Kiểm tra:
```bash
wc -l data/preprocessed/claims.jsonl
grep -c "q_XXX" data/preprocessed/claims.jsonl
```

### `openai.AuthenticationError`

Key trong `.env` sai hoặc expired:
```bash
cat .env  # kiểm tra key
```

### `RuntimeError: Call index() before retrieve()`

Claims chưa có embedding → pipeline chưa chạy Phase 2. Không cần làm gì — pipeline tự embed khi `run()`.

### Pylance báo `Import "networkx" could not be resolved`

VS Code đang dùng sai Python interpreter. Mở Command Palette → **Python: Select Interpreter** → chọn `/opt/anaconda3/envs/rese006/bin/python`.

---

## TÓM TẮT NHANH

| Script | Lệnh | Thời gian | Cần OpenAI? |
|---|---|---|---|
| Preprocess 10 queries | `python data/preprocess.py --max_queries 10` | ~2 phút | Có |
| Preprocess 500 queries | `python data/preprocess.py` | ~15 phút | Có |
| Preprocess (mock, không LLM) | `python data/preprocess.py --no_llm` | ~10 giây | Không |
| Chạy pipeline 1 query | `python scripts/run_pipeline.py --query_id q_002` | ~30-60 giây | Có |
| Chạy pipeline (mock LLM) | inject `pipeline.answer_gen._client = lambda p: "..."` | ~20 giây | Không |
| Chạy tests | `python -m pytest tests/test_smoke.py -q` | ~3 giây | Không |

---

*RESE006 — Conflict-Aware Agentic RAG | Team RESE006 | 2026*


