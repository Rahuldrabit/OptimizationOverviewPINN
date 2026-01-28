"""Run tests to verify all benchmarks and HPO methods work correctly."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

# Add src to path  
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def main():
    """Run all tests."""
    
    # Discover and run tests
    loader = unittest.TestLoader()
    start_dir = Path(__file__).parent.parent / "tests"
    suite = loader.discover(str(start_dir), pattern="test_*.py")
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Return appropriate exit code
    if result.wasSuccessful():
        print("\n✓ All tests passed!")
        return 0
    else:
        print(f"\n✗ {len(result.failures)} failures, {len(result.errors)} errors")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)