import json, glob, sys

sys.path.insert(0, ".")
from query_generation_loop.pipeline import load_queries

queries = load_queries("data/confict_rag/conflict_region_query_claims.json")
results = []
for fpath in sorted(glob.glob("outputs/tmp_batch/q_*/result.json")):
    try:
        with open(fpath, encoding="utf-8") as f:
            d = json.load(f)
        qid = d["query_id"]
        iters = d.get("num_conflicts_processed", 0)
        results.append({
            "query_id": qid,
            "question": queries.get(qid, d.get("user_query", "")),
            "num_claims": d.get("num_claims", 0),
            "num_contradiction_edges": d.get("num_query_contradiction_edges", 0),
            "num_conflicts_processed": iters,
            "num_generated_queries": d.get("num_generated_queries", 0),
            "state_distribution": d.get("state_distribution", {}),
            "resolution_status": d.get("final_resolution_status", ""),
            "confidence": d.get("final_answer_confidence", 0),
            "answer_quality": d.get("answer_quality_score", 0),
        })
    except Exception as e:
        print(f"FAIL: {fpath} - {e}")

summary = {
    "mode": "methodology_baseline",
    "total_queries": len(results),
    "total_with_edges": sum(1 for r in results if r["num_contradiction_edges"] > 0),
    "total_with_conflicts_processed": sum(1 for r in results if r["num_conflicts_processed"] > 0),
    "resolution_breakdown": {},
    "queries": results,
}
for r in results:
    s = r["resolution_status"]
    summary["resolution_breakdown"][s] = summary["resolution_breakdown"].get(s, 0) + 1

with open("outputs/batch_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

res = summary["resolution_breakdown"]
print(f"Done: {len(results)} queries")
print(f"  Conflicts: {summary['total_with_edges']} queries have edges, {summary['total_with_conflicts_processed']} processed")
for k, v in sorted(res.items()):
    print(f"  {k}: {v} ({v*100//len(results)}%)")
