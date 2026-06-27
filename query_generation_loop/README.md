# Conflict-State-Guided Iterative Retrieval Loop

## 1. Vai trò trong pipeline

Module này nằm giữa conflict detection và LLM generation:

```text
claims + edges → conflict_regions
                        ↓
       ┌─ ITERATIVE LOOP ──────────────────────────────┐
       │  conflict_regions                               │
       │  → sinh targeted query                          │
       │  → retrieve thêm evidence                       │
       │  → re-analyze conflict state                    │
       │  → lặp hoặc dừng                                │
       │  → output: claims_final                        │
       └────────────────────────────────────────────────┘
                        ↓
       claims_final → LLM generate answer + Evaluation
```

## 2. Luồng hoạt động

Mỗi query được xử lý qua vòng lặp:

1. Nhận danh sách **conflict records** từ edges.json (cặp claim có `relation_type = "contradiction"`)
2. Mỗi vòng lặp:
   - Lọc unresolved conflicts, infer state (Resolvable / Contextual / Underdetermined)
   - Chọn top conflicts → sinh **targeted query** (BM25 retrieval)
   - Phân loại claim mới: ủng hộ claim_i / claim_j / neutral
   - Cập nhật conflict intensity
   - Kiểm tra điều kiện dừng
3. Output: **claims_final** — danh sách claims đã phân loại (selected / competing / excluded) + answer policy

## 3. Input / Output

### Input (từ các module trước)

| File | Định dạng | Nội dung |
|------|-----------|----------|
| `factoid_claims.jsonl` | JSONL | Claims có factoid_features (slot, value, entity...) |
| `edges.json` | JSON array | NLI edges với relation_type, relation_confidence, nli_features |
| `queries.jsonl` | JSONL | query_id → user_query mapping |
| `conflict_regions.jsonl` | JSONL | (optional) Region-based conflict grouping |
| `claim_graph_edges.jsonl` | JSONL | (optional) Edge features cho Phase 3 re-analysis |

### Output

| File | Nội dung |
|------|----------|
| `claims_final_{qid}.json` | Claim selection mỗi query: selected, competing, excluded + answer policy + confidence |
| `claims_final_all.json` | Tổng hợp tất cả queries, dùng làm input cho LLM generation |

## 4. Targeted queries pipeline sinh ra

Mỗi conflict có thể sinh 1 trong 5 loại query:

| Query type | Dùng cho state | Mục đích |
|------------|----------------|----------|
| `context_disambiguation` | Contextual | Làm rõ ngữ cảnh của claim |
| `evidence_coverage_expansion` | Underdetermined | Mở rộng evidence |
| `claim_i_verification` | Resolvable | Verify claim thứ nhất |
| `claim_j_verification` | Resolvable | Verify claim thứ hai |
| `comparison` | — | So sánh trực tiếp 2 claims |

Query được chọn dựa trên scoring: `conflict_intensity × 0.4 + slot_importance × 0.25 + query_specificity × 0.2 + state_probability × 0.15`.

## 5. Khi nào vòng lặp dừng

Dừng ngay khi gặp 1 trong các điều kiện:

1. **state_stable** — State không đổi sau 2 vòng liên tiếp (hay gặp nhất)
2. **no_new_claims** — Không retrieve được claim mới nào
3. **coverage_gain_below_threshold** — Claim mới < 10% kỳ vọng
4. **all_conflicts_resolved** — Tất cả conflicts đã xử lý xong
5. **max_iterations_reached** — Chạy đủ 10 vòng

Trung bình mỗi query chạy 1-3 vòng. Pipeline deterministic, không gọi LLM.

## 6. State inference

Conflict state được xác định bằng heuristic dựa trên:

| Signal | Weight |
|--------|--------|
| Base score | 0.20 (threshold) |
| Confidence gap | 0.08 (small) / 0.20 (large) |
| Context completeness | < 0.5 → low |
| Evidence coverage | < 0.4 → low |
| Relation confidence | 0.4-0.75 moderate / > 0.80 high |
| Contradiction probability | > 0.70 → high |

Kết quả: **Resolvable** (có thể giải quyết), **Contextual** (cần ngữ cảnh), **Underdetermined** (chưa đủ evidence).

## 7. Batch run

Chạy tất cả queries có factoid claims, tổng hợp output 1 file duy nhất:

```powershell
Remove-Item outputs\tmp_batch -Recurse -Force -ErrorAction SilentlyContinue
$tmp = "outputs\tmp_batch"
for ($i = 0; $i -le 313; $i++) {
  $qid = "q_{0:D3}" -f $i
  $d = "$tmp\$qid"
  New-Item -ItemType Directory -Path $d -Force > $null
  python -m query_generation_loop.pipeline --query_id $qid --mode methodology_baseline --output "$d\result.json" --claims_final_output "$d\claims.json" 2>&1 | Out-Null
}
python -c "
import json, glob
all_claims = []
for fpath in sorted(glob.glob('outputs/tmp_batch/q_*/claims.json')):
    try:
        with open(fpath, encoding='utf-8') as f:
            all_claims.append(json.load(f))
    except: pass
with open('outputs/claims_final_all.json', 'w', encoding='utf-8') as f:
    json.dump(all_claims, f, ensure_ascii=False, indent=2)
print(f'Done: {len(all_claims)} queries -> outputs/claims_final_all.json')
"
```

**Output:** `outputs/claims_final_all.json` (~5-10 phút, deterministic).

## 8. Tests

```bash
pytest query_generation_loop/tests -q
```
