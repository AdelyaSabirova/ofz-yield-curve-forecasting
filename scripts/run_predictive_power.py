from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ofz_curve.predictive_power import (
    DEFAULT_CORE_PATH,
    DEFAULT_FINDINGS_PATH,
    DEFAULT_MACRO_PATH,
    DEFAULT_OFFICIAL_CURVE_PATH,
    DEFAULT_OUTPUT_DIR,
    run_predictive_power,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Пересчитать блок добавочной прогностической силы для инфляции и средней ключевой ставки."
    )
    parser.add_argument("--core-path", type=Path, default=DEFAULT_CORE_PATH)
    parser.add_argument("--macro-path", type=Path, default=DEFAULT_MACRO_PATH)
    parser.add_argument("--official-curve-path", type=Path, default=DEFAULT_OFFICIAL_CURVE_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--findings-path", type=Path, default=DEFAULT_FINDINGS_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_predictive_power(
        core_path=args.core_path,
        macro_path=args.macro_path,
        official_curve_path=args.official_curve_path,
        output_dir=args.output_dir,
        findings_path=args.findings_path,
    )
    summary = result["summary_metadata"]
    coverage = summary["input_coverage"]
    print(
        "Predictive-power run complete: "
        f"panel {coverage['panel_start']}..{coverage['panel_end']}, "
        f"inflation actuals to {coverage['inflation_actual_end']}, "
        f"key rate actuals to {coverage['key_rate_actual_end']}."
    )
    print(f"Inflation comparison: {summary['artifacts']['inflation_model_comparison_csv']}")
    print(f"Key rate comparison: {summary['artifacts']['key_rate_model_comparison_csv']}")
    print(f"DM tests: {summary['artifacts']['dm_tests_csv']}")
    print(f"Key rate tau1 increment: {summary['artifacts']['key_rate_tau1_increment_csv']}")
    print(f"Key rate tau1 DM tests: {summary['artifacts']['key_rate_tau1_dm_tests_csv']}")
    print(f"Key rate benchmark comparison: {summary['artifacts']['key_rate_curve_benchmark_comparison_csv']}")
    print(f"Key rate benchmark DM tests: {summary['artifacts']['key_rate_curve_benchmark_dm_tests_csv']}")
    print(f"Key rate benchmark summary: {summary['artifacts']['key_rate_curve_benchmark_summary_csv']}")
    print(f"Key rate market-controls comparison: {summary['artifacts']['key_rate_market_controls_comparison_csv']}")
    print(f"Key rate market-controls DM tests: {summary['artifacts']['key_rate_market_controls_dm_tests_csv']}")
    print(f"Key rate market-controls summary: {summary['artifacts']['key_rate_market_controls_summary_csv']}")
    print(f"Findings: {summary['artifacts']['findings_markdown']}")


if __name__ == "__main__":
    main()
