from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ofz_curve.cbr_zero_curve import (  # noqa: E402
    DEFAULT_END_DATE,
    DEFAULT_REPORT_START_DATE,
    DEFAULT_START_DATE,
    DEFAULT_FIGURE_PATH,
    run_cbr_zero_curve_validation,
)
from ofz_curve.paths import (  # noqa: E402
    CBR_ZERO_CURVE_DAILY_CSV,
    CBR_ZERO_CURVE_MONTHLY_CSV,
    CORE_DAILY_NS_FACTORS_CSV,
    OFFICIAL_CURVE_VALIDATION_DIR,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Загрузить официальный архив КБД Банка России и сопоставить его с собственной кривой ОФЗ."
    )
    parser.add_argument("--core-daily-path", type=Path, default=CORE_DAILY_NS_FACTORS_CSV)
    parser.add_argument("--raw-daily-path", type=Path, default=CBR_ZERO_CURVE_DAILY_CSV)
    parser.add_argument("--monthly-path", type=Path, default=CBR_ZERO_CURVE_MONTHLY_CSV)
    parser.add_argument("--output-dir", type=Path, default=OFFICIAL_CURVE_VALIDATION_DIR)
    parser.add_argument("--figure-path", type=Path, default=DEFAULT_FIGURE_PATH)
    parser.add_argument("--start-date", type=str, default=DEFAULT_START_DATE.strftime("%Y-%m-%d"))
    parser.add_argument("--report-start-date", type=str, default=DEFAULT_REPORT_START_DATE.strftime("%Y-%m-%d"))
    parser.add_argument("--end-date", type=str, default=DEFAULT_END_DATE.strftime("%Y-%m-%d"))
    parser.add_argument("--refresh", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_cbr_zero_curve_validation(
        core_daily_path=args.core_daily_path,
        raw_daily_path=args.raw_daily_path,
        monthly_path=args.monthly_path,
        output_dir=args.output_dir,
        figure_path=args.figure_path,
        start_date=args.start_date,
        report_start_date=args.report_start_date,
        end_date=args.end_date,
        refresh=args.refresh,
    )

    metadata = result["metadata"]
    archive = metadata["official_archive"]
    comparison = metadata["comparison"]
    print(
        "Сопоставление с официальной КБД завершено: "
        f"архив {archive['start_date']}..{archive['end_date']}, "
        f"перекрытие {comparison['overlap_start']}..{comparison['overlap_end']}."
    )
    print(f"Дневной архив: {archive['raw_daily_csv']}")
    print(f"Месячный архив: {archive['monthly_csv']}")
    print(f"Дневное сопоставление: {comparison['artifacts']['daily_comparison_csv']}")
    print(f"Сводная таблица: {comparison['artifacts']['summary_csv']}")
    print(f"Итоговый рисунок: {comparison['artifacts']['summary_figure']}")


if __name__ == "__main__":
    main()
