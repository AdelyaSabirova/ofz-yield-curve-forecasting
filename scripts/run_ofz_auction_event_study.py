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

from ofz_curve.ofz_auction_study import run_ofz_auction_event_study


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Построить событийное исследование реакции кривой ОФЗ вокруг результатов аукционов Минфина."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data" / "processed" / "ofz_auction_study",
        help="Директория для сохранения артефактов.",
    )
    parser.add_argument(
        "--findings-path",
        type=Path,
        default=ROOT / "docs" / "p4_ofz_auction_findings.md",
        help="Путь к markdown-файлу с короткими выводами.",
    )
    parser.add_argument(
        "--figure-path",
        type=Path,
        default=ROOT / "assets" / "ofz_auction_study.png",
        help="Путь к итоговому рисунку.",
    )
    parser.add_argument(
        "--analysis-start-date",
        default="2022-02-01",
        help="Начальная дата анализа для событийной выборки.",
    )
    parser.add_argument(
        "--crawl-start-date",
        default="2022-01-01",
        help="Начальная дата, до которой канал Минфина листается при сборе аукционных сообщений.",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=420,
        help="Максимальное число страниц канала Минфина для обхода.",
    )
    args = parser.parse_args()

    result = run_ofz_auction_event_study(
        output_dir=args.output_dir,
        findings_path=args.findings_path,
        figure_path=args.figure_path,
        analysis_start_date=args.analysis_start_date,
        crawl_start_date=args.crawl_start_date,
        max_pages=args.max_pages,
    )

    print(
        json.dumps(
            {
                "events": result["metadata"]["coverage"]["event_count"],
                "start_date": result["metadata"]["coverage"]["start_date"],
                "end_date": result["metadata"]["coverage"]["end_date"],
                "summary_csv": result["metadata"]["artifacts"]["summary_csv"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
