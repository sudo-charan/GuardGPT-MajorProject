import json
import time
from core.complete_pipeline import CompletePipeline

def run_eval(dataset_path):
    pipeline = CompletePipeline()
    results = []
    
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            prompt = item["prompt"]
            print(f"Evaluating {item['id']}: {prompt[:50]}...", end=" ", flush=True)
            
            start = time.perf_counter()
            # Run pipeline (no session_id ensures clean state for each case)
            report = pipeline.run(prompt, session_id=None)
            elapsed = (time.perf_counter() - start) * 1000
            
            initial_jev = report.get("jev_style") or {}
            recheck_jev = report.get("jev_style_recheck") or {}
            
            res = {
                "id": item["id"],
                "prompt": prompt,
                "initial_action": initial_jev.get("decision"),
                "initial_category": initial_jev.get("classification") or initial_jev.get("choice"),
                "rewritten": report.get("sanitized_prompt") is not None,
                "sanitized_prompt": report.get("sanitized_prompt"),
                "second_action": recheck_jev.get("decision"),
                "final_status": report.get("final_status"),
                "final_action": report.get("action"),
                "output_audit": report.get("output_audit"),
                "latency_ms": elapsed,
                "error": report.get("error") if report.get("final_status") == "ERROR" else None
            }
            results.append(res)
            print(f"DONE ({report.get('final_status')})")
            
    return results

if __name__ == "__main__":
    dataset = "quality_audit.jsonl"
    output_file = "quality_audit_baseline.json"
    
    metrics = run_eval(dataset)
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
        
    print("\n" + "="*70)
    print(f"{'ID':<10} | {'Init':<10} | {'Recheck':<10} | {'Final':<12} | {'Audit':<10}")
    print("-" * 70)
    for m in metrics:
        print(f"{m['id']:<10} | {str(m['initial_action']):<10} | {str(m['second_action']):<10} | {m['final_status']:<12} | {m['output_audit']:<10}")
    print("="*70)
    print(f"Results saved to {output_file}")
