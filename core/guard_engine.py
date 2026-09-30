"""Compatibility facade: legacy callers use the same audited pipeline as the CLI."""
from dataclasses import dataclass
from typing import Optional
import uuid
from core.decision_engine import DecisionOutput

@dataclass
class EngineResponse:
    allowed: bool
    intent: str
    risk_level: str
    decision: DecisionOutput
    llama_response: Optional[str] = None
    blocked_message: Optional[str] = None
    error: Optional[str] = None

class GuardEngine:
    def __init__(self, session_id=None):
        from core.complete_pipeline import CompletePipeline
        self.session_id = session_id or uuid.uuid4().hex
        self._pipeline = CompletePipeline()

    def startup(self):
        from core.safety_service import resources
        resources()

    def process(self, prompt):
        report = self._pipeline.run(prompt, session_id=self.session_id)
        decision = DecisionOutput(
            allowed=report["allowed"], intent=report["intent"], risk_level=report["risk_level"],
            user_message=report.get("user_message", ""), technical_reason="; ".join(report["reasons"]),
            category_scores=report.get("category_scores", {}),
            reason_codes=report["reasons"], action=report["action"],
            final_status=report["final_status"], sanitized_prompt=report["sanitized_prompt"],
            dataset_match_confidence=report.get("dataset_match_confidence", 0.0),
            matched_record_id=report.get("matched_record_id"), audit_id=report["audit_id"])
        return EngineResponse(
            allowed=report["allowed"], intent=report["intent"], risk_level=report["risk_level"],
            decision=decision,
            llama_response=report.get("response") if report["allowed"] else None,
            blocked_message=report.get("response") or report.get("user_message") if not report["allowed"] else None,
            error=report.get("error"))

    def new_conversation(self):
        self._pipeline.sessions.pop(self.session_id, None)
        self.session_id = uuid.uuid4().hex

    def print_status(self):
        from main import status
        status()

    def run_interactive(self):
        print("GuardGPT: type /exit to quit or /new for a new session.")
        while True:
            try:
                prompt = input("You: ")
            except (EOFError, KeyboardInterrupt):
                break
            if prompt.strip() == "/exit":
                break
            if prompt.strip() == "/new":
                self.new_conversation()
                continue
            print_response(self.process(prompt))

def print_response(response):
    print(response.llama_response or response.blocked_message or response.error)
    print(f"[{response.decision.action} | {response.decision.final_status}]")
