from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from nelson_siegel_svensson import NelsonSiegelCurve


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ofz_curve.cbr_zero_curve import (  # noqa: E402
    build_curve_comparison,
    build_monthly_cbr_zero_curve,
    normalize_cbr_zero_curve_table,
    summarize_curve_comparison,
)


def test_normalize_cbr_zero_curve_table() -> None:
    columns = pd.MultiIndex.from_tuples(
        [
            ("Дата", "Дата"),
            ("Срок до погашения, лет", "0,25"),
            ("Срок до погашения, лет", "1"),
            ("Срок до погашения, лет", "10"),
        ]
    )
    table = pd.DataFrame(
        [
            ["20.03.2026", 1203, 1295, 1432],
            ["19.03.2026", 1211, 1308, 1433],
        ],
        columns=columns,
    )

    normalized = normalize_cbr_zero_curve_table(table)

    assert normalized.columns[0] == "date"
    assert {"y0_25", "y1_00", "y10_00"} <= set(normalized.columns)
    assert normalized["date"].dt.strftime("%Y-%m-%d").tolist() == ["2026-03-19", "2026-03-20"]
    assert np.isclose(normalized.loc[1, "y0_25"], 12.03)
    assert np.isclose(normalized.loc[0, "y10_00"], 14.33)


def test_build_monthly_cbr_zero_curve() -> None:
    daily = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-30", "2026-01-31", "2026-02-27"]),
            "y0_25": [10.0, 10.2, 10.5],
            "y1_00": [11.0, 11.2, 11.5],
        }
    )

    monthly = build_monthly_cbr_zero_curve(daily)

    assert monthly["date"].dt.strftime("%Y-%m-%d").tolist() == ["2026-01-01", "2026-02-01"]
    assert monthly["curve_source_date"].dt.strftime("%Y-%m-%d").tolist() == ["2026-01-31", "2026-02-27"]
    assert np.isclose(monthly.loc[0, "y0_25"], 10.2)


def test_curve_comparison_summary_zero_error() -> None:
    dates = pd.to_datetime(["2024-01-02", "2024-01-03"])
    curve_1 = NelsonSiegelCurve(beta0=0.12, beta1=-0.04, beta2=0.02, tau=3.6)
    curve_2 = NelsonSiegelCurve(beta0=0.13, beta1=-0.03, beta2=0.015, tau=3.5)

    core_daily = pd.DataFrame(
        {
            "date": dates,
            "beta0": [0.12, 0.13],
            "beta1": [-0.04, -0.03],
            "beta2": [0.02, 0.015],
            "tau1": [3.6, 3.5],
            "success": [True, True],
        }
    )
    official_daily = pd.DataFrame(
        {
            "date": dates,
            "y0_25": [curve_1(0.25) * 100.0, curve_2(0.25) * 100.0],
            "y1_00": [curve_1(1.0) * 100.0, curve_2(1.0) * 100.0],
        }
    )

    comparison = build_curve_comparison(
        core_daily,
        official_daily,
        maturities=(0.25, 1.0),
        report_start_date="2024-01-01",
    )
    summary = summarize_curve_comparison(comparison)

    assert len(comparison) == 4
    assert np.allclose(summary["rmse_pp"], 0.0)
    assert np.allclose(summary["median_abs_error_pp"], 0.0)
    assert np.allclose(summary["correlation"], 1.0)


def test_cbr_zero_curve_cli_smoke(tmp_path: Path) -> None:
    dates = pd.to_datetime(["2024-01-02", "2024-01-03"])
    curve = NelsonSiegelCurve(beta0=0.10, beta1=-0.03, beta2=0.015, tau=3.2)

    core_daily_path = tmp_path / "daily_ns_factors_core.csv"
    official_daily_path = tmp_path / "cbr_zero_curve_daily.csv"
    monthly_path = tmp_path / "monthly_cbr_zero_curve.csv"
    output_dir = tmp_path / "official_curve_validation"
    figure_path = tmp_path / "official_curve_validation_summary.png"

    pd.DataFrame(
        {
            "date": dates,
            "curve_source_date": dates,
            "beta0": [0.10, 0.10],
            "beta1": [-0.03, -0.03],
            "beta2": [0.015, 0.015],
            "tau1": [3.2, 3.2],
            "n_bonds": [10, 10],
            "rmse_pp": [0.1, 0.1],
            "rmse_pp_static": [0.1, 0.1],
            "success": [True, True],
            "error_message": [np.nan, np.nan],
        }
    ).to_csv(core_daily_path, index=False)

    pd.DataFrame(
        {
            "date": dates,
            "y0_25": [curve(0.25) * 100.0, curve(0.25) * 100.0],
            "y0_50": [curve(0.50) * 100.0, curve(0.50) * 100.0],
            "y0_75": [curve(0.75) * 100.0, curve(0.75) * 100.0],
            "y1_00": [curve(1.0) * 100.0, curve(1.0) * 100.0],
            "y2_00": [curve(2.0) * 100.0, curve(2.0) * 100.0],
            "y3_00": [curve(3.0) * 100.0, curve(3.0) * 100.0],
            "y5_00": [curve(5.0) * 100.0, curve(5.0) * 100.0],
            "y7_00": [curve(7.0) * 100.0, curve(7.0) * 100.0],
            "y10_00": [curve(10.0) * 100.0, curve(10.0) * 100.0],
            "y15_00": [curve(15.0) * 100.0, curve(15.0) * 100.0],
            "y20_00": [curve(20.0) * 100.0, curve(20.0) * 100.0],
            "y30_00": [curve(30.0) * 100.0, curve(30.0) * 100.0],
        }
    ).to_csv(official_daily_path, index=False)

    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "run_cbr_zero_curve_validation.py"),
        "--core-daily-path",
        str(core_daily_path),
        "--raw-daily-path",
        str(official_daily_path),
        "--monthly-path",
        str(monthly_path),
        "--output-dir",
        str(output_dir),
        "--figure-path",
        str(figure_path),
        "--start-date",
        "2024-01-02",
        "--report-start-date",
        "2024-01-02",
        "--end-date",
        "2024-01-03",
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)

    assert "Сопоставление с официальной КБД завершено" in completed.stdout
    summary_csv = output_dir / "official_vs_model_summary.csv"
    metadata_json = output_dir / "official_vs_model_summary.json"

    assert monthly_path.exists()
    assert summary_csv.exists()
    assert metadata_json.exists()
    assert figure_path.exists()

    summary = pd.read_csv(summary_csv)
    metadata = json.loads(metadata_json.read_text(encoding="utf-8"))

    assert set(summary["maturity_years"]) == {1.0, 3.0, 5.0, 10.0}
    assert metadata["comparison"]["overlap_start"] == "2024-01-02"
    assert metadata["comparison"]["overlap_end"] == "2024-01-03"
