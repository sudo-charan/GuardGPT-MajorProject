"""Offline regression suite. Historical live-model demos are not accuracy tests."""
import logging
import unittest

logging.disable(logging.CRITICAL)
suite = unittest.defaultTestLoader.loadTestsFromNames([
    "tests.test_augmented_integration", "tests.test_risk_estimator",
    "tests.test_complete_pipeline", "tests.test_complete_http",
    "tests.test_safety_service",
    "tests.test_self_harm_gate",
])
result = unittest.TextTestRunner(verbosity=2).run(suite)
raise SystemExit(0 if result.wasSuccessful() else 1)
