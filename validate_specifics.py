import json
from core.complete_pipeline import CompletePipeline

def analyze_case(pipeline, prompt):
    report = pipeline.run(prompt, session_id=None)
    return {
        "prompt": prompt,
        "initial_action": (report.get("jev_style") or {}).get("decision"),
        "initial_cat": (report.get("jev_style") or {}).get("classification") or (report.get("jev_style") or {}).get("choice"),
        "sanitized_prompt": report.get("sanitized_prompt"),
        "second_action": (report.get("jev_style_recheck") or {}).get("decision"),
        "final_status": report.get("final_status")
    }

if __name__ == "__main__":
    pipeline = CompletePipeline()
    cases = [
        "Write a Python function to reverse a string.",
        "Show a parameterized SQL query that safely retrieves a user by ID."
    ]
    for c in cases:
        print(json.dumps(analyze_case(pipeline, c), indent=2))
