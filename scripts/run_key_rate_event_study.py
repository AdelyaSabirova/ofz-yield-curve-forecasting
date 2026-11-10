from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ofz_curve.key_rate_event_study import (  # noqa: E402
    CBR_KEY_RATE_CALENDAR_URL,
    DEFAULT_FINDINGS_PATH,
    DEFAULT_FIGURE_PATH,
    DEFAULT_MODEL_FACTORS_PATH,
    DEFAULT_OFFICIAL_CURVE_PATH,
    DEFAULT_OUTPUT_DIR,
    run_key_rate_event_study,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Построить событийное исследование реакции кривой ОФЗ вокруг решений Банка России по ставке."
    )
    parser.add_argument("--official-curve-path", type=Path, default=DEFAULT_OFFICIAL_CURVE_PATH)
    parser.add_argument("--model-factors-path", type=Path, default=DEFAULT_MODEL_FACTORS_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--findings-path", type=Path, default=DEFAULT_FINDINGS_PATH)
    parser.add_argument("--figure-path", type=Path, default=DEFAULT_FIGURE_PATH)
    parser.add_argument("--calendar-url", default=CBR_KEY_RATE_CALENDAR_URL)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_key_rate_event_study(
        official_curve_path=args.official_curve_path,
        model_factors_path=args.model_factors_path,
        output_dir=args.output_dir,
        findings_path=args.findings_path,
        figure_path=args.figure_path,
        calendar_url=args.calendar_url,
    )
    coverage = result["metadata"]["coverage"]
    artifacts = result["metadata"]["artifacts"]
    print(
        "Событийное исследование по решениям Банка России завершено: "
        f"события {coverage['events_start']}..{coverage['events_end']}, "
        f"регулярных заседаний {coverage['regular_meetings']}."
    )
    print(f"Таблица событий: {artifacts['events_csv']}")
    print(f"Таблица реакций: {artifacts['reactions_csv']}")
    print(f"Сводка: {artifacts['summary_csv']}")
    print(f"Рисунок: {artifacts['figure']}")
    print(f"Выводы: {artifacts['findings_markdown']}")


if __name__ == "__main__":
    main()
