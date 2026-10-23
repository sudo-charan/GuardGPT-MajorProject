import ast
import importlib
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class PhaseTwoStructureTests(unittest.TestCase):
    def test_core_modules_import_and_parse(self):
        modules = (
            "core.conversation_guard",
            "core.safety_service",
            "core.complete_pipeline",
        )
        for name in modules:
            with self.subTest(module=name):
                module = importlib.import_module(name)
                source = Path(module.__file__).read_text(encoding="utf-8")
                ast.parse(source)

    def test_only_one_active_definition_of_each_phase_two_method(self):
        targets = {
            "core/conversation_guard.py": ("ConversationGuard", "update_behavioral_state"),
            "core/safety_service.py": ("SafetyService", "finalize_turn"),
            "core/complete_pipeline.py": ("CompletePipeline", "_process"),
        }
        for relative_path, (class_name, method_name) in targets.items():
            with self.subTest(path=relative_path, method=method_name):
                tree = ast.parse((ROOT / relative_path).read_text(encoding="utf-8"))
                cls = next(
                    node for node in tree.body
                    if isinstance(node, ast.ClassDef) and node.name == class_name
                )
                matches = [
                    node for node in cls.body
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and node.name == method_name
                ]
                self.assertEqual(len(matches), 1)

        safety_tree = ast.parse((ROOT / "core/safety_service.py").read_text(encoding="utf-8"))
        safety_class = next(node for node in safety_tree.body if isinstance(node, ast.ClassDef) and node.name == "SafetyService")
        self.assertTrue(any(isinstance(node, ast.FunctionDef) and node.name == "apply_history" for node in safety_class.body))


if __name__ == "__main__":
    unittest.main()
