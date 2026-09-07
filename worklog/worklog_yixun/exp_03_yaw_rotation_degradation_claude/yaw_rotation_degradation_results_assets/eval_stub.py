"""Stub evaluator for launcher fault tests: --stub-exit N exits with N without writing any output (N=0 => provenance must fail)."""
import sys
code = int(sys.argv[sys.argv.index("--stub-exit") + 1]) if "--stub-exit" in sys.argv else 1
print("stub evaluator, exiting", code); sys.exit(code)
