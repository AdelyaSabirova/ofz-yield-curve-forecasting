#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ofz_curve.policy_path_gap import (  # noqa: E402
    DEFAULT_FIGURE_PATH,
    DEFAULT_FINDINGS_PATH,
    DEFAULT_OFFICIAL_CURVE_PATH,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SURVEY_WORKBOOK_PATH,
    run_policy_path_gap_analysis,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Построить разрыв доходностей 5Y и 10Y относительно опросной 24-месячной траектории ключевой ставки."
    )
    parser.add_argument("--survey-workbook-path", type=Path, default=DEFAULT_SURVEY_WORKBOOK_PATH)
    parser.add_argument("--official-curve-path", type=Path, default=DEFAULT_OFFICIAL_CURVE_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--findings-path", type=Path, default=DEFAULT_FINDINGS_PATH)
    parser.add_argument("--figure-path", type=Path, default=DEFAULT_FIGURE_PATH)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_policy_path_gap_analysis(
        survey_workbook_path=args.survey_workbook_path,
        official_curve_path=args.official_curve_path,
        output_dir=args.output_dir,
        findings_path=args.findings_path,
        figure_path=args.figure_path,
    )
    print(
        json.dumps(
            {
                "survey_path_start": result["metadata"]["coverage"]["survey_path_start"],
                "survey_path_end": result["metadata"]["coverage"]["survey_path_end"],
                "summary_csv": result["metadata"]["artifacts"]["summary_csv"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
