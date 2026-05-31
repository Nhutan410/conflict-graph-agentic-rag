# Conflict-Aware Agentic RAG — Giải Thích Pipeline Toàn Diện

> Tài liệu này giải thích **từng bước** toàn bộ pipeline `ConflictAwarePipeline` (Method 1), cách mỗi chỉ số được tính, và phân tích chi tiết output thực tế của query `q_008: "When was General Bryan born?"`.

---

## Mục lục

1. [Tổng quan kiến trúc](#1-tổng-quan-kiến-trúc)
2. [Dữ liệu đầu vào](#2-dữ-liệu-đầu-vào)
3. [Phase 1 — Claim Extraction](#3-phase-1--claim-extraction)
4. [Phase 2 — Embedding & Canonicalization](#4-phase-2--embedding--canonicalization)
5. [Phase 3 — Evidence Graph](#5-phase-3--evidence-graph)
6. [Phase 4 — Hybrid Retrieval](#6-phase-4--hybrid-retrieval)
7. [Phase 5 — Conflict Zone Analysis](#7-phase-5--conflict-zone-analysis)
8. [Phase 6 — Iterative Loop](#8-phase-6--iterative-loop)
9. [Phase 7 — Answer Generation](#9-phase-7--answer-generation)
10. [Output Schema](#10-output-schema)
11. [Phân tích thực tế: q_008 "When was General Bryan born?"](#11-phân-tích-thực-tế-q_008)
12. [Tại sao resolved=false và validated_claims=0?](#12-tại-sao-resolvedfalse-và-validated_claims0)
13. [Cách tính từng chỉ số trong output](#13-cách-tính-từng-chỉ-số-trong-output)

---

## 1. Tổng quan kiến trúc

```
Input: query + documents/claims
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│  Phase 1: Claim Extraction (LLM)                            │
│  Document text → Atomic factual claims                      │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  Phase 2: Embedding & Canonicalization                      │
│  Claims → Dense vectors (BGE-M3)                            │
│  Merge duplicates, preserve temporal/numerical divergence   │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  Phase 3: Evidence Graph Construction                       │
│  Claims → NLI pairs → nx.DiGraph                            │
│  Edge types: support | contradiction | neutral              │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  Phase 4: Hybrid Retrieval (BM25 + Dense + RRF)             │
│  Query → Top-k relevant claims, balanced by conflict pairs  │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  Phase 5: Conflict Zone Analysis                            │
│  5.1 Credibility Arbitration (ArbGraph-style propagation)   │
│  5.2 Factoid Decomposition + Conflict Localization          │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  Phase 6: Iterative Loop                                    │
│  Formulate targeted queries → retrieve new docs → update    │
│  Stop when: resolved | max_iter | no new docs               │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  Phase 7: Answer Generation (LLM)                           │
│  Validated claims → grounded final answer                   │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
Output: LoopResult (resolved, validated_claims,
        conflict_localizations, final_answer)
```

Pipeline được cài đặt tại `src/pipeline.py:ConflictAwarePipeline.run()`.

---

## 2. Dữ liệu đầu vào

### 2.1 Queries — `data/preprocessed/queries.jsonl`

Mỗi dòng là một câu hỏi:
```json
{"query_id": "q_008", "user_query": "When was General Bryan born?"}
```

### 2.2 Claims — `data/preprocessed/claims.jsonl`

Mỗi dòng là một atomic claim đã được extract sẵn từ document:
```json
{
  "claim_id": "c_q008_d00_s00",
  "claim_text": "General Bryan was born in 1900.",
  "doc_id": "d_xxxxx",
  "source_id": "s_q008_00",
  "retrieval_relevance": 1.0,
  "claim_confidence": 0.85,
  "credibility_score": null
}
```

**Naming convention của claim_id:**
```
c _ q008 _ d00 _ s00
│   │      │     └── sentence index trong document
│   │      └──────── document index (d00 = document đầu tiên)
│   └─────────────── query id
└─────────────────── "claim"
```

Khi chạy `python scripts/run_pipeline.py --query_id q_008`, script đọc claims này qua `data/loaders.py:load_claims("q_008")` và **bỏ qua Phase 1** (không cần gọi LLM để extract lại).

---

## 3. Phase 1 — Claim Extraction

> **File:** `src/extractor.py:ClaimExtractor`
> **Skip khi:** dùng preprocessed claims (mặc định khi chạy `run_pipeline.py`)

### Mục đích
Chuyển raw document text → danh sách các **atomic factual claims** — mỗi claim là một sự kiện đơn lẻ có thể verify độc lập.

### Cách hoạt động
```python
# Prompt gửi cho GPT-4o-mini:
SYSTEM: "Extract atomic factual claims from text..."
USER:   "Extract all atomic claims... Return ONLY a JSON array..."
        "Document: {document_text}"
```

LLM trả về danh sách chuỗi. Mỗi chuỗi được wrap thành object `Claim`:
```python
Claim(
    claim_id = f"c{counter:06d}",   # e.g. "c000001"
    doc_id   = document.doc_id,
    text     = claim_text,
    source_credibility = document.credibility_score,
)
```

Claim bị bỏ qua nếu dài hơn 35 từ (quá dài → không atomic).

**Retry logic:** Exponential backoff, tối đa 3 lần (`2^0=1s`, `2^1=2s`, `2^2=4s`).

---

## 4. Phase 2 — Embedding & Canonicalization

> **Files:** `src/embedder.py:ClaimEmbedder`, `src/embedder.py:ClaimCanonical`

### 4.1 Embedding (ClaimEmbedder)

Model: `BAAI/bge-m3` (multilingual, 1024-dim)

```python
embeddings = model.encode(texts, normalize_embeddings=True)
# normalize_embeddings=True → L2 norm = 1 → cosine sim = dot product
claim.embedding = emb.tolist()   # list[float], len=1024
```

Embedding vector được dùng ở:
- Phase 3: tính cosine similarity để tạo candidate pairs
- Phase 4: dense retrieval

### 4.2 Canonicalization (ClaimCanonical)

**Vấn đề cần giải quyết:** Nhiều documents có thể nói cùng một sự kiện → duplicate claims làm nhiễu graph.

**Logic:**
```
Với mọi cặp (claim_i, claim_j):
  cos_sim = embedding_i · embedding_j   (đã normalize)

  Nếu cos_sim >= 0.92 (canonical_threshold):
    val_i = temporal/numerical tokens trong claim_i
    val_j = temporal/numerical tokens trong claim_j

    Nếu val_i và val_j có giá trị KHÁC nhau:
      → KHÔNG merge (VD: "born in 1900" vs "born in 1905")
    Ngược lại:
      → MERGE (union-find), giữ claim có index nhỏ hơn
```

**Tại sao quan trọng:** Nếu merge nhầm "born in 1900" và "born in 1905", ta mất đi xung đột quan trọng nhất. Bước này bảo tồn những divergence có giá trị.

---

## 5. Phase 3 — Evidence Graph

> **File:** `src/graph_builder.py`

### 5.1 Pair Generation (PairGenerator)

Thay vì pair tất cả O(n²) claims, chỉ pair những claim có:
```
cosine_similarity(embedding_i, embedding_j) >= 0.3  (pair_similarity_threshold)
```

Điều này đảm bảo chỉ pair những claims nói về cùng chủ đề → tiết kiệm tài nguyên NLI.

### 5.2 NLI Inference (NLIInference)

Model: `cross-encoder/nli-deberta-v3-base`

Với mỗi pair (claim_A, claim_B):
```python
result = nli_pipeline({"text": claim_A, "text_pair": claim_B})
# result: {"label": "contradiction", "score": 0.99}

if score < 0.5 (nli_edge_threshold):
    relation = "neutral"    # quá uncertain → không thêm edge
else:
    relation = LABEL_MAP[label]
    # "entailment" → "support"
    # "contradiction" → "contradiction"
    # "neutral" → "neutral"
```

### 5.3 Graph Construction

```
Graph = nx.DiGraph

Nodes: mỗi claim_id là một node
       attributes: {text, embedding, retrieval_relevance, ...}

Edges: mỗi NLI pair có score >= threshold
       attributes: {relation, nli_score, source="nli_model"}
```

**Ví dụ cho q_008:**
```
c_q008_d00_s00 ("born in 1900")
        │
        │  contradiction, score=0.97
        ▼
c_q008_d01_s00 ("born in 1905")
```

---

## 6. Phase 4 — Hybrid Retrieval

> **File:** `src/retriever.py`

### 6.1 HybridRetriever — Công thức tính điểm

```
claim_relevance_score(q, r_i) = α * dense_score + (1-α) * bm25_score + rrf_score
```

Với `α = 0.5` (retrieval_alpha từ config):

**BM25 score:**
```python
bm25_scores = BM25Okapi(tokenized_claims).get_scores(tokenized_query)
bm25_norm   = bm25_scores / (bm25_scores.max() + 1e-9)    # normalize về [0,1]
```

**Dense score:**
```python
q_emb        = model.encode([query], normalize_embeddings=True)
dense_scores = emb_matrix @ q_emb.T    # cosine similarity
dense_norm   = (dense_scores + 1) / 2  # [-1,1] → [0,1]
```

**RRF (Reciprocal Rank Fusion) — k=60:**
```python
# Với mỗi retriever (BM25 và Dense), cộng điểm RRF theo rank:
rrf_score[i] += 1 / (60 + rank + 1)

# Claim ở rank 1: 1/61 ≈ 0.0164
# Claim ở rank 10: 1/70 ≈ 0.0143
# Claim ở rank 100: 1/160 ≈ 0.0063
```

**Final hybrid score:**
```
hybrid[i] = 0.5 * dense_norm[i] + 0.5 * bm25_norm[i] + rrf_scores[i]
```

Top 10 claims (`retrieval_top_k`) được chọn.

### 6.2 BalancedTopKSelector

Sau khi rank, selector đảm bảo **ít nhất 1 conflict pair** (`min_conflict_pairs`) có mặt trong top-k. Nếu thiếu, nó swap claim ít quan trọng nhất để đưa "conflict partner" vào.

**Lý do:** Một RAG system bình thường sẽ chỉ lấy claims tương đồng nhau → không thấy conflict. Pipeline này chủ động đưa conflict vào để xử lý.

---

## 7. Phase 5 — Conflict Zone Analysis

> **File:** `src/conflict_zone.py`

Đây là **trái tim** của pipeline — nơi phát hiện và định lượng mâu thuẫn.

### 7.1 Credibility Arbitration (CredibilityArbitrator)

Cảm hứng từ **ArbGraph**: propagate credibility signals qua evidence graph.

**Công thức tính credibility score:**
```
credibility_score(r_i) = Σ [nli_score(r_j→r_i) * (credibility(r_j) + 1)]   [quan hệ support]
                        - Σ [nli_score(r_k→r_i) * (credibility(r_k) + 1)]   [quan hệ contradiction]
```

**Iterative update (tối đa 10 vòng, damping=0.85):**
```python
# Khởi tạo
scores = {node: 0.0 for node in graph.nodes}

for iteration in range(10):
    new_scores = copy(scores)

    for node in graph.nodes:
        support_sum     = 0.0
        contradict_sum  = 0.0

        for pred in graph.predecessors(node):
            rel       = edge_data["relation"]
            nli_score = edge_data["nli_score"]
            src_cred  = max(scores[pred], 0.0) + 1.0   # +1 tránh zero

            if rel in ("support", "entailment"):
                support_sum += nli_score * src_cred
            elif rel == "contradiction":
                contradict_sum += nli_score * src_cred

        raw = support_sum - contradict_sum
        # Damping: blend giữa score mới và cũ, tránh oscillation
        new_scores[node] = 0.85 * raw + 0.15 * scores[node]

    # Kiểm tra hội tụ
    max_delta = max(|new_scores[n] - scores[n]| for n in scores)
    if max_delta < 0.01:
        break   # đã ổn định
    scores = new_scores
```

**Ý nghĩa của kết quả:**
- `score > 0`: claim được **nhiều claims khác support** → validated
- `score = 0`: claim không có incoming edges đáng kể → neutral
- `score < 0`: claim bị **nhiều claims khác contradict** → suppressed

**Threshold phân loại:** `credibility_threshold = 0.0`
- `score > 0.0` → **validated**
- `score <= 0.0` → **suppressed**

### 7.2 Factoid Decomposition (FactoidDecomposer)

Với mỗi contradiction pair trong graph, decompose từng claim thành các **typed slots**:

```python
class FactoidSlots:
    temporal       # Năm, ngày tháng (regex: \b(1[0-9]{3}|20[0-9]{2})\b)
    numerical      # Số liệu (regex: \b\d+([.,]\d+)?(%|km|kg|...)?\b)
    entity_subject # Chủ thể (LLM)
    entity_object  # Đối tượng (LLM)
    relation       # Quan hệ (LLM)
    location       # Địa điểm (LLM)
```

**Strategy:**
- `temporal` và `numerical`: rule-based regex (nhanh, deterministic)
- `entity_subject`, `entity_object`, `relation`, `location`: gọi GPT-4o-mini

**Ví dụ decompose claim "General Bryan was born in 1900 in Virginia":**
```json
{
  "temporal": "1900",
  "numerical": null,
  "entity_subject": "General Bryan",
  "entity_object": null,
  "relation": "born in",
  "location": "Virginia"
}
```

### 7.3 Conflict Localization

Với mỗi contradiction pair (claim_i, claim_j):

```python
for field_name in ["temporal", "numerical", "entity_subject",
                   "entity_object", "relation", "location"]:
    val_i = slots_i[field_name]
    val_j = slots_j[field_name]

    # Bỏ qua nếu cả hai null
    if val_i is None and val_j is None:
        continue

    total_slots += 1

    # Conflict nếu cả hai có giá trị VÀ khác nhau
    if val_i != val_j and val_i is not None and val_j is not None:
        conflict_slots.append(field_name)
```

**Conflict Intensity:**
```
conflict_intensity = len(conflict_slots) / total_slots
```

Ví dụ: 2 slots có giá trị, 1 bị conflict → intensity = 1/2 = 0.5

**ConflictLocalization object:**
```python
ConflictLocalization(
    claim_i_id        = "c_q008_d00_s00",
    claim_j_id        = "c_q008_d01_s00",
    slot              = "temporal",        # slot đầu tiên bị conflict
    value_i           = "1900",
    value_j           = "1905",
    conflict_intensity = 0.75,             # 3 slots conflict / 4 total slots
    credibility_i     = 0.0,              # từ Phase 5.1
    credibility_j     = -0.9998...,       # từ Phase 5.1
)
```

---

## 8. Phase 6 — Iterative Loop

> **File:** `src/iterative_loop.py:IterativeLoop`

### 8.1 Vòng lặp chính

```python
for iteration in range(max_iterations=3):

    # Phân tích conflict trong top-k claims hiện tại
    analysis = conflict_analyzer.analyze(graph, top_k_claims)
    max_intensity = max(loc.conflict_intensity for loc in localizations)

    # Stopping Condition A: đã giải quyết xong
    if max_intensity < intensity_threshold (0.1):
        break   # RESOLVED

    # Stopping Condition C: không còn tiến bộ
    if iteration > 0 and (prev_intensity - max_intensity) < 0.1:
        break   # CONVERGED

    # Formulate targeted query cho conflict nặng nhất
    for loc in localizations[:2]:
        targeted_query = formulate(original_query, loc)
        # VD: "Verify: When was General Bryan born?
        #      Source A claims 1900, Source B claims 1905.
        #      [temporal conflict] Which year is correct?"

    # Retrieve new docs + update graph (nếu có updater)
    if updater and new_docs:
        graph, all_claims = updater.update(graph, new_docs, all_claims)
    else:
        break   # No new documents — stopping

# Final conflict analysis
final_analysis = conflict_analyzer.analyze(graph, top_k_claims)
```

### 8.2 ConflictQueryFormulator

Tạo targeted query theo slot type:

| Slot | Template |
|------|----------|
| `temporal` | `"Verify: {query} Source A claims {val_i}, Source B claims {val_j}. [temporal conflict] Which year is correct?"` |
| `numerical` | `"Verify numerical claim: {query} One source says {val_i}, another says {val_j}. Find authoritative evidence."` |
| `entity_object/subject` | `"Disambiguate entity conflict: {query} Conflicting entities: '{val_i}' vs '{val_j}'."` |
| `location` | `"Verify location: {query} Source A: {val_i}, Source B: {val_j}."` |

### 8.3 Tiêu chí kết thúc loop

| Condition | Trigger | Kết quả |
|-----------|---------|---------|
| A — Resolved | `max_intensity < 0.1` | `resolved = True` |
| B — Max iterations | `iterations == 3` | `resolved = False` |
| C — No new docs | `updater is None or new_docs empty` | Dừng sớm |
| D — No progress | `intensity_reduction < 0.1` | Dừng sớm |

### 8.4 resolved được tính như thế nào?

```python
is_resolved = (
    not final_analysis.conflict_localizations          # không còn conflict nào
    OR
    all(loc.conflict_intensity < 0.1                   # tất cả conflict đều nhỏ
        for loc in final_analysis.conflict_localizations)
)
```

### 8.5 validated_claims được lọc như thế nào?

```python
validated_claims = [
    c for c in top_k_claims
    if c.claim_id in final_analysis.validated_claim_ids  # credibility > 0.0
]
validated_claims.sort(
    key=lambda c: credibility_scores[c.claim_id],
    reverse=True    # claim tin cậy nhất lên đầu
)
```

---

## 9. Phase 7 — Answer Generation

> **File:** `src/generator.py`

### 9.1 Evidence Selector

Ranking score cho từng validated claim:
```
evidence_score = 0.6 * (0.7 * credibility_norm + 0.3 * source_credibility)
              + 0.4 * retrieval_relevance

credibility_norm = clamp((credibility + 2.0) / 4.0, 0, 1)
# Shift và scale để map credibility [-2, 2] → [0, 1]
```

### 9.2 Trường hợp không có validated claims

```python
if not selected_claims:
    return (
        "Insufficient evidence to answer this question reliably. "
        "The retrieved documents do not contain enough verified information."
    )
```

Đây chính xác là `final_answer` của q_008.

### 9.3 Prompt khi có validated claims

```
You are a precise, factual assistant...

Question: {query}

Verified Claims (ordered by confidence):
1. [doc_id] claim text
2. [doc_id] claim text
...

Unresolved Conflicts:
- [temporal] '1900' (confidence: 0.00) vs '1905' (confidence: -1.00)

Instructions: Ground every statement... present both perspectives...

Answer:
```

---

## 10. Output Schema

```python
class LoopResult:
    query_id              : str
    iterations_run        : int
    resolved              : bool
    validated_claims      : list[Claim]
    conflict_localizations: list[ConflictLocalization]
    final_answer          : Optional[str]
```

Output JSON từ `run_pipeline.py` thêm các field thống kê:
```json
{
  "query_id": "...",
  "query": "...",
  "resolved": false,
  "iterations_run": 1,
  "final_answer": "...",
  "n_validated_claims": 0,
  "n_conflict_localizations": 9,
  "conflict_localizations": [...],
  "validated_claims": [...]
}
```

---

## 11. Phân tích thực tế: q_008

### Query
```
"When was General Bryan born?"
```

### Tình huống dữ liệu

Dữ liệu cho q_008 có **2 documents** (d00 và d01) mô tả General Bryan, nhưng **mâu thuẫn nhau**:

| Claim | Doc | Nội dung |
|-------|-----|---------|
| `c_q008_d00_s00` | d00 | General Bryan sinh năm **1900** |
| `c_q008_d01_s00` | d01 | General Bryan sinh năm **1905** |
| `c_q008_d00_s06` | d00 | Bryan học tại **Virginia Military Institute** |
| `c_q008_d00_s07` | d00 | Bryan tốt nghiệp West Point năm **1918** |
| `c_q008_d01_s09` | d01 | Bryan thuộc **Class of 1923** |
| `c_q008_d01_s11` | d01 | Bryan nhận commission năm **1921** |

### Luồng thực tế qua pipeline

```
1. Load preprocessed claims cho q_008
   → Embedding: BGE-M3 encode tất cả claims

2. Phase 3 — Build graph:
   → PairGenerator: pair claims có cosine_sim >= 0.3
   → NLIInference: d00_s00 vs d01_s00 → "contradiction", score=0.97
     (vì "born in 1900" mâu thuẫn "born in 1905")

3. Phase 4 — Retrieval:
   → HybridRetriever: top 10 claims liên quan nhất đến "When was born"
   → BalancedTopKSelector: đảm bảo conflict pairs có mặt

4. Phase 5 — Conflict Zone:
   → CredibilityArbitrator: không node nào có credibility > 0
     (vì tất cả nodes đều có contradiction edges, không có support)
   → Factoid localization: phát hiện 9 conflict pairs

5. Phase 6 — Iterative Loop:
   → Iteration 1: max_intensity = 0.75 > 0.1 → cần tiếp tục
   → Formulate targeted query: "Verify: When was General Bryan born?
      Source A claims 1900, Source B claims 1905. [temporal conflict]"
   → updater = None → không có external retrieval → BREAK

6. Final analysis: vẫn còn conflicts, 0 validated claims

7. Phase 7 — Generation:
   → selected_claims = [] (0 validated)
   → Return hardcoded "Insufficient evidence..." message
```

---

## 12. Tại sao resolved=false và validated_claims=0?

### Lý do 1: Tất cả credibility scores đều ≤ 0

Nhìn vào credibility scores trong output:
```
credibility_i: 0.0       (c_q008_d00_s00 — "born 1900")
credibility_j: -0.9999   (c_q008_d01_s00 — "born 1905")
credibility_j: -1.0268   (c_q008_d01_s09 — "Class of 1923")
credibility_j: -1.9914   (c_q008_d00_s07 — "West Point 1918")
credibility_j: -3.9322   (c_q008_d01_s11 — "commission 1921")
```

**Tại sao tất cả <= 0?** Vì không có claim nào nhận được nhiều "support" từ claims khác. Tất cả các edges quan trọng đều là `contradiction`. Arbitration propagation không tìm được "winner" rõ ràng.

Threshold là `credibility_threshold = 0.0`:
```python
validated = [cid for cid, score in cred_scores.items() if score > 0.0]
```
Không có claim nào vượt ngưỡng → `validated_claims = []`.

### Lý do 2: Loop dừng sớm sau 1 iteration

```python
# Trong IterativeLoop.run():
if self.updater and new_docs:
    graph, all_claims = updater.update(...)
else:
    logger.info("[Loop] No new documents — stopping")
    break   ← dừng ở đây
```

Pipeline hiện tại không có external retrieval system thật (`updater=None`). Loop chỉ chạy được 1 iteration, không có cơ hội tìm thêm evidence để phá vỡ xung đột.

### Lý do 3: max_intensity = 0.75 >> threshold = 0.1

```
max_intensity (0.75) > intensity_threshold (0.1)
→ Condition A (resolved) KHÔNG thỏa mãn
→ resolved = False
```

---

## 13. Cách tính từng chỉ số trong output

### `resolved`

```python
resolved = (
    not conflict_localizations               # không còn conflict
    or
    all(loc.conflict_intensity < 0.1         # hoặc tất cả nhỏ hơn threshold
        for loc in conflict_localizations)
)
```

**q_008:** `conflict_localizations` có 9 items, max intensity = 0.75 → `resolved = False`

---

### `iterations_run`

Đếm số vòng trong `IterationLog`. Mỗi iteration = 1 lần phân tích conflict zone.

**q_008:** Chỉ 1 iteration vì loop break do không có new documents.

---

### `n_validated_claims`

```python
n_validated_claims = len(result.validated_claims)
# = len([c for c in top_k_claims if credibility_score[c.claim_id] > 0.0])
```

**q_008:** `0` vì không có claim nào có credibility > 0.

---

### `n_conflict_localizations`

```python
n_conflict_localizations = len(result.conflict_localizations)
```

Bằng số cặp (claim_i, claim_j) trong graph có `relation = "contradiction"` VÀ có ít nhất 1 slot bị conflict sau decomposition.

**q_008:** `9` — toàn bộ là các contradiction edges trong graph.

---

### `conflict_intensity`

```python
conflict_intensity = len(conflict_slots) / total_slots
```

| Conflict pair | total_slots | conflict_slots | intensity |
|---------------|-------------|----------------|-----------|
| d00_s00 vs d01_s00 | 4 | 3 (temporal + 2 others) | **0.75** |
| d00_s02 vs d01_s09 | 2 | 1 | **0.50** |
| d00_s03 vs d00_s07 | 5 | 2 | **0.40** |
| d00_s03 vs d01_s09 | 2 | 1 | **0.50** |
| d00_s03 vs d01_s11 | 4 | 1 | **0.25** |
| d00_s05 vs d00_s06 | 4 | 1 | **0.25** |
| d00_s06 vs d00_s07 | 5 | 2 | **0.40** |
| d00_s07 vs d01_s11 | 5 | 2 | **0.40** |
| d01_s09 vs d01_s11 | 2 | 1 | **0.50** |

---

### `credibility_i` và `credibility_j`

Output trực tiếp từ `CredibilityArbitrator.compute()`. Ví dụ:

```
credibility(c_q008_d01_s11) = -3.932

Tại sao thấp nhất?
→ claim "received his commission" bị mâu thuẫn bởi nhiều claims khác
  (d00_s03 [served during WWII], d01_s09 [Class of 1923])
→ Không có claim nào support nó
→ Nhiều contradiction incoming edges → score cộng dồn rất âm
```

```
credibility(c_q008_d00_s00) = 0.0

Tại sao bằng 0?
→ Claim này có trong graph nhưng không có incoming edges
  (là nguồn gốc của conflict, không bị contradict lại bởi claim nào khác
   trong top-k selection)
→ support_sum = 0, contradict_sum = 0 → score = 0
```

---

### `slot` — Loại mâu thuẫn

| Slot | Ý nghĩa | Detect bằng |
|------|---------|------------|
| `temporal` | Thời gian mâu thuẫn (năm, ngày) | Regex |
| `numerical` | Số liệu mâu thuẫn | Regex |
| `entity_subject` | Chủ thể khác nhau | LLM |
| `entity_object` | Đối tượng khác nhau | LLM |
| `relation` | Quan hệ khác nhau (born in vs attended) | LLM |
| `location` | Địa điểm khác nhau | LLM |

**Lưu ý:** Mỗi ConflictLocalization chỉ báo cáo **slot đầu tiên** bị conflict (không phải tất cả). `conflict_intensity` mới phản ánh tổng số slots bị conflict.

---

## Tóm tắt toàn bộ luồng cho q_008

```
Query: "When was General Bryan born?"
         │
         ▼
Load claims q_008 (preprocessed, skip Phase 1)
  → c_q008_d00_s00: "born in 1900"  (doc 0)
  → c_q008_d01_s00: "born in 1905"  (doc 1)
  → + 10 claims khác về cùng người
         │
         ▼
Phase 2: Embed với BGE-M3
  → Mỗi claim → vector 1024-dim (normalize)
  → Canonical: "born 1900" vs "born 1905" có temporal khác → KHÔNG merge
         │
         ▼
Phase 3: Build evidence graph
  → PairGenerator: 12 claims → ~20 pairs có sim >= 0.3
  → NLI: phát hiện nhiều contradiction edges
  → Graph: nhiều nodes mâu thuẫn nhau, ít support
         │
         ▼
Phase 4: Retrieval
  → Query "When was General Bryan born?" → top 10 claims liên quan nhất
  → BalancedTopKSelector: đảm bảo conflict pairs vào top-k
         │
         ▼
Phase 5.1: Credibility Arbitration (10 iterations)
  → Propagate qua contradiction edges
  → Không có winner rõ ràng → tất cả scores <= 0
  → validated = [] (threshold = 0.0)
         │
         ▼
Phase 5.2: Factoid Localization
  → 9 contradiction pairs → decompose slots → 9 ConflictLocalization
  → Conflict quan trọng nhất: temporal "1900" vs "1905" (intensity=0.75)
         │
         ▼
Phase 6: Iterative Loop
  → max_intensity=0.75 > 0.1 → cần retrieve thêm
  → Targeted query: "Verify: born in 1900 or 1905?"
  → updater=None → không có new docs → BREAK sau 1 iteration
         │
         ▼
Phase 7: Generation
  → validated_claims = [] → không đủ evidence
  → final_answer = "Insufficient evidence..."
         │
         ▼
Output:
  resolved = False        ← còn conflicts chưa giải quyết
  iterations_run = 1      ← loop dừng sớm
  n_validated_claims = 0  ← không claim nào vượt credibility threshold
  n_conflict_localizations = 9  ← 9 mâu thuẫn được phát hiện
  final_answer = "Insufficient evidence..."
```

**Kết luận:** Pipeline hoạt động **đúng** — nó **từ chối đoán bừa** năm sinh khi bằng chứng mâu thuẫn nhau, thay vì chọn đại 1900 hoặc 1905. Đây chính là hành vi cốt lõi của Conflict-Aware RAG: thà không trả lời còn hơn trả lời sai.
