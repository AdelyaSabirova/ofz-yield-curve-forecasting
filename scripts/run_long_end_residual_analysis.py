from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ofz_curve.long_end_residual import (  # noqa: E402
    DEFAULT_CORE_PATH,
    DEFAULT_FINDINGS_PATH,
    DEFAULT_FIGURE_PATH,
    DEFAULT_MACRO_PATH,
    DEFAULT_OFFICIAL_CURVE_PATH,
    DEFAULT_OUTPUT_DIR,
    run_long_end_residual_analysis,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Построить остаточный компонент длинного конца кривой ОФЗ относительно старого режима."
    )
    parser.add_argument("--core-path", type=Path, default=DEFAULT_CORE_PATH)
    parser.add_argument("--macro-path", type=Path, default=DEFAULT_MACRO_PATH)
    parser.add_argument("--official-curve-path", type=Path, default=DEFAULT_OFFICIAL_CURVE_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--findings-path", type=Path, default=DEFAULT_FINDINGS_PATH)
    parser.add_argument("--figure-path", type=Path, default=DEFAULT_FIGURE_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_long_end_residual_analysis(
        core_path=args.core_path,
        macro_path=args.macro_path,
        official_curve_path=args.official_curve_path,
        output_dir=args.output_dir,
        findings_path=args.findings_path,
        figure_path=args.figure_path,
    )
    coverage = result["metadata"]["coverage"]
    artifacts = result["metadata"]["artifacts"]
    print(
        "Анализ остаточного наклона длинного конца завершён: "
        f"панель {coverage['panel_start']}..{coverage['panel_end']}, "
        f"старый режим до {coverage['old_regime_train_end']}."
    )
    print(f"Ряд остатков: {artifacts['residual_series_csv']}")
    print(f"Сводка по режимам: {artifacts['summary_csv']}")
    print(f"Коэффициенты: {artifacts['coefficients_csv']}")
    print(f"Рисунок: {artifacts['figure']}")
    print(f"Выводы: {artifacts['findings_markdown']}")


if __name__ == "__main__":
    main()
