#!/usr/bin/env python3
"""
Run the regime analysis pipeline and persist outputs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ofz_curve.regime_analysis import (  # noqa: E402
    CORE_PATH,
    DEFAULT_FINDINGS_PATH,
    DEFAULT_OUTPUT_DIR,
    MACRO_PATH,
    run_regime_analysis,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run regime analysis for OFZ thesis data.")
    parser.add_argument("--core", type=Path, default=CORE_PATH, help="Path to monthly NS factors csv.")
    parser.add_argument(
        "--macro",
        type=Path,
        default=MACRO_PATH,
        help="Path to monthly macro expectations csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for CSV and JSON outputs.",
    )
    parser.add_argument(
        "--findings-path",
        type=Path,
        default=DEFAULT_FINDINGS_PATH,
        help="Markdown report path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_regime_analysis(
        core_path=args.core,
        macro_path=args.macro,
        output_dir=args.output_dir,
        findings_path=args.findings_path,
    )
    payload = {
        "comparison_path": result["comparison_path"],
        "break_tests_path": result["break_tests_path"],
        "summary_path": result["summary_path"],
        "findings_path": result["findings_path"],
        "core_last_date": result["summary"]["input_data"]["core_last_date"],
        "macro_last_date": result["summary"]["input_data"]["macro_last_date"],
        "model_last_date": result["summary"]["input_data"]["model_last_date"],
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

