"""Run tests to verify all benchmarks and HPO methods work correctly."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

# Add project root and src to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(project_root / "src") not in sys.path:
    sys.path.insert(0, str(project_root / "src"))


def main():
    """Run all tests."""
    
    # Discover and run tests
    loader = unittest.TestLoader()
    start_dir = project_root / "tests"
    suite = loader.discover(str(start_dir), pattern="test_*.py", top_level_dir=str(project_root))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Return appropriate exit code
    if result.wasSuccessful():
        print("\n[PASS] All tests passed successfully!")
        return 0
    else:
        print(f"\n[FAIL] {len(result.failures)} failures, {len(result.errors)} errors")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)