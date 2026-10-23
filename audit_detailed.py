import json
from core.complete_pipeline import CompletePipeline

def run_detailed_audit():
    pipeline = CompletePipeline()
    results = []
    
    with open("quality_audit.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            prompt = item["prompt"]
            
            # Use a timeout/error handler wrapper for the run
            try:
                report = pipeline.run(prompt, session_id=None)
                
                res = {
                    "id": item["id"],
                    "category": item["scenario_type"],
                    "expected": item["expected_action"],
                    "initial_action": (report.get("jev_style") or {}).get("decision"),
                    "second_action": (report.get("jev_style_recheck") or {}).get("decision"),
                    "final_status": report.get("final_status"),
                    "infra_error": report.get("final_status") == "ERROR"
                }
            except Exception as e:
                res = {
                    "id": item["id"],
                    "category": item["scenario_type"],
                    "expected": item["expected_action"],
                    "initial_action": "EXCEPTION",
                    "second_action": None,
                    "final_status": "ERROR",
                    "infra_error": True
                }
            results.append(res)
            print(f"Processed {item['id']}")
            
    return results

if __name__ == "__main__":
    data = run_detailed_audit()
    print("\n--- DETAILED AUDIT RESULTS ---")
    for r in data:
        print(f"{r['id']} | {r['category']:<20} | Exp: {r['expected']:<8} | Init: {str(r['initial_action']):<10} | 2nd: {str(r['second_action']):<10} | Final: {r['final_status']:<12} | Infra: {r['infra_error']}")
