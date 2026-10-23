from core.complete_pipeline import CompletePipeline
from unittest.mock import MagicMock

def debug_run():
    mock_safety = MagicMock()
    mock_safety.analyze.return_value = {
        "signal": {"intent": "coding", "intent_confidence": 0.9, "risk_level": "safe", "category_scores": {}, "dataset_match_confidence": 0.0, "matched_record_id": None},
        "detected_attacks": []
    }
    mock_jev = MagicMock()
    mock_jev.decide.return_value = {"decision": "ALLOW", "classification": "Safe/Educational", "confidence": 0.9}
    
    pipeline = CompletePipeline(safety=mock_safety, jev_style=mock_jev)
    try:
        print("Running pipeline...")
        report = pipeline.run("Test prompt")
        print(f"Report final status: {report['final_status']}")
        print(f"JEV called: {mock_jev.decide.called}")
    except Exception as e:
        print(f"CRASH: {e}")

if __name__ == "__main__":
    debug_run()
