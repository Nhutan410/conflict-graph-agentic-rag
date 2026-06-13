# Schema cho Pipeline "Conflict-State-Guided Retrieval for Conflict-Aware RAG" (bản Revised)

File: `conflict_state_guided_rag_schema.json` — JSON Schema (draft 2020-12), gồm **56 definitions** trong `$defs`.

## Nguyên tắc trích xuất feature

Feature chỉ được lấy từ **các mục/heading chính (module 1–13 và mục 17)**. Phần **16 ("Tổng hợp schema theo từng level")** chỉ là bảng tổng hợp lại — nếu một schema/feature ở mục 16 trùng với feature đã nêu ở module tương ứng thì giữ (pass), nhưng **không** lấy thêm feature/field nào chỉ xuất hiện riêng trong mục 16.

Vì vậy, so với bản trước, các phần sau đã được **loại bỏ** vì chỉ xuất hiện ở mục 16, không được định nghĩa ở module chính nào:

- `DocumentEvidence`, `RetrievalScores`, `DocumentMetadata` (chỉ ở mục 16.1 — Module 1 chỉ nói về kỹ thuật Hybrid fusion/RRF, không định nghĩa schema document).
- `NormalizedClaim` (cấu trúc subject/predicate/object/polarity/time/location/condition/scope/modality/certainty — chỉ ở mục 16.2). Module 3 (mục 5.3) chỉ định nghĩa **factoid features**: Number, Entity, Temporal, Negation, Verb -> giữ lại dưới dạng `FactoidFeatures`.
- `Provenance` (doc_id/passage_id/source_span — chỉ ở mục 16.2).
- `temporal_specificity`, `entity_specificity` trong `ClaimFeatures` (chỉ ở mục 16.2; Module 4 mục 6.2 chỉ có 4 feature: retrieval_relevance, claim_confidence, claim_evidence_coverage, context_completeness).
- `RelationFeatures` / `relation_features` trong `Edge` (semantic_similarity, temporal_overlap, predicate_compatibility, scope_overlap, condition_overlap, polarity_mismatch, modality_mismatch... — chỉ ở mục 16.3, khác với Edge Features thực sự của Module 5 ở mục 7.3/7.4).
- `claim_cluster_ids` trong `ConflictRegion` (chỉ ở mục 16.5; có thể suy ra bằng cách lọc `Cluster.conflict_region_id`).

## Map module -> schema (theo đúng thứ tự pipeline trong tài liệu revise)

| # | Module | Schema chính |
| --- | --- | --- |
| 1 | Initial Evidence Retrieval | (chỉ là kỹ thuật Hybrid fusion + RRF, không có schema riêng) |
| 2 | Atomic Claim Extraction | `Claim.claim_text` |
| 3 | Factoid Claim Extraction (mục 5.3) | `FactoidFeatures` (Number, Entity, Temporal, Negation, Verb) + `FactoidEntityMention`, `FactoidNumberValue`, `FactoidVerb` |
| 4 | Claim Representation & Feature Initialization (mục 6.2) | `ClaimFeatures` (retrieval_relevance, claim_confidence, claim_evidence_coverage, context_completeness), `Claim` (node-level, gộp Module 2+3+4 + `ClaimIdAlignment`) |
| 5 | Relation Inference & Claim Graph Construction (mục 7.3, 7.4) | `NLIEdgeFeatures`, `ClaimPairFeatures` (Entity/Attribute/Number/Temporal/Negation/Aggregate), `Edge` |
| 6 | Conflict Region Detection (mục 8.2) | `RegionDetection` |
| 7 | Claim Cluster Construction (mục 9.1, 9.6) | `ClusterFeatures`, `Cluster` |
| 8 | Conflict Region Encoding & Feature Aggregation (mục 10.1–10.13) | `RegionFeatures`, `AttentionDiagnostics`, `ConflictRegionEncoding`, `ConflictRegion` |
| 9 | Dynamic Feature Update (mục 11.3) | `FeatureUpdateLogEntry`, `FeatureUpdateLog` |
| 10 | Conflict State Classification (mục 12.2–12.5) | `ConflictStateClassificationInput`, `ConflictStateClassificationOutput`, `StateProbabilities`, `StateRationale`, `ConflictState` |
| 11 | Conflict-State-Guided Targeted Query Generation (mục 13.2–13.6) | `StateToQueryMapping`, `TargetedQueryGenerationInput`, `ConflictingClaimRef`, `TargetedQuery`, `TargetedQueryGenerationOutput`, `QueryType` |
| 12 | Iterative Reasoning–Retrieval Loop (mục 14.3–14.6) | `IterationInput`, `TargetedQueryRef`, `IterationOutput`, `GraphUpdate`, `UpdatedConflictRegion`, `StoppingCriteria` |
| 13 | Final Evidence Selection and Answer Generation (mục 15.3–15.5) | `FinalAnswerInput`, `ResolvedGraph`, `FinalAnswerOutput` |
| §17 | Claim ID / Entity / Temporal / Context-slot / Feature-update alignment | `ClaimIdAlignment` (17.1), `EntityAlignment` (17.2), `TemporalAlignment` (17.3), `ContextSlotAlignmentNote` (17.4) |
| Tổng hợp graph | — | `ConflictGraph` (gom Claim/Edge/Cluster/ConflictRegion theo `graph_id`, dùng cho `g_001`/`g_003` được tham chiếu ở mục 14.3 và 15.3) |

## Lưu ý quan trọng

- Chỉ phản ánh **bản revise**, không merge với pipeline cũ trong `PIPELINE_EXPLAINED.md` (file này thực ra chưa upload thành công — chỉ file `.docx` revise được nhận).
- `claim_text` (atomic claim, Module 2) **không bị thay thế** bởi `factoid_features` (Module 3, mục 5.3) — cả hai cùng tồn tại trong `Claim`.
- `ea_alignment_score` (trong `AggregateEdgeFeatures`, mục 7.4) là **gate** cho `negation_conflict_score`, `number_mismatch`, `temporal_order_mismatch`; còn `conflict_intensity_score` thì **không** bị gate — đúng theo công thức gốc.
- `Edge` hiện chỉ gồm `nli_features` (mục 7.3) + `claim_pair_features` (mục 7.4) — không còn `relation_features` tổng hợp kiểu mục 16.3.
- Các enum (`RelationType`, `ConflictState`, `QueryType`, `TemporalRelation`, `TemporalGranularity`) được định nghĩa tập trung để đảm bảo Context Slot Alignment (mục 17.4).

## Cách dùng

```python
import json, jsonschema
schema = json.load(open("conflict_state_guided_rag_schema.json"))
region_schema = {**schema, "$ref": "#/$defs/ConflictRegion"}
jsonschema.validate(instance=my_region_payload, schema=region_schema)
```
