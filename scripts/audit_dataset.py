#!/usr/bin/env python3
"""CLI script to audit ConvoGuard dataset files, schema integrity, and label distributions.

Usage:
    python scripts/audit_dataset.py --data-root "data/raw" --output "artifacts/data_audit/summary.json"
"""

import sys
from pathlib import Path

# Ensure src/ is on sys.path when running script directly from repo root
repo_root = Path(__file__).resolve().parent.parent
src_dir = repo_root / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from convoguard.data.audit import main_cli  # noqa: E402

if __name__ == "__main__":
    main_cli()
