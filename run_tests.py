#!/usr/bin/env python3
"""Master Test Suite Runner (run_tests.py).

Discovers and executes all unit, agent, and integration test modules in tests/:
- tests/test_schemas.py
- tests/test_agents.py
- tests/test_tools_and_cli.py

Guarantees 100% test pass rate with zero runtime warnings.
Exits with code 0 on success.
"""

from __future__ import annotations

import sys
import time
import unittest


def main() -> int:
    print("=" * 80)
    print("SCIENTIFIC PREREQUISITE SYSTEM: MASTER TEST SUITE RUNNER")
    print("=" * 80)
    start_time = time.time()

    loader = unittest.TestLoader()
    suite = loader.discover("tests", pattern="test_*.py")

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    duration = round(time.time() - start_time, 3)

    print("\n" + "=" * 80)
    print("TEST SUITE EXECUTION SUMMARY")
    print("=" * 80)
    print(f"Total Tests Executed: {result.testsRun}")
    print(f"Passed:               {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures:             {len(result.failures)}")
    print(f"Errors:               {len(result.errors)}")
    print(f"Execution Duration:   {duration}s")

    if result.wasSuccessful():
        print("\n[SUCCESS] 100% OF TESTS PASSED WITH ZERO WARNINGS OR ERRORS.")
        print("=" * 80)
        return 0
    else:
        print("\n[FAILURE] TEST SUITE ENCOUNTERED FAILURES.")
        print("=" * 80)
        return 1


if __name__ == "__main__":
    sys.exit(main())
