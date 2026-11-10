from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ofz_curve.regime_analysis import run_regime_analysis  # noqa: E402


def _build_synthetic_inputs(tmp_path: Path) -> tuple[Path, Path]:
    rng = np.random.default_rng(42)
    dates = pd.date_range("2013-09-01", "2026-02-01", freq="MS")
    n_obs = len(dates)
    break_idx = int(np.where(dates >= pd.Timestamp("2022-02-01"))[0][0])

    common_trend = np.cumsum(rng.normal(0.0, 0.002, n_obs))
    post_shift = np.zeros(n_obs)
    post_shift[break_idx:] = 0.015

    beta0 = 0.09 + common_trend + post_shift + rng.normal(0.0, 0.004, n_obs)
    beta1 = -0.05 + 0.35 * common_trend - 0.010 * post_shift + rng.normal(0.0, 0.01, n_obs)
    beta2 = -0.08 - 0.25 * common_trend - 0.012 * post_shift + rng.normal(0.0, 0.01, n_obs)
    cpi_mom = 0.005 + 0.25 * np.diff(beta0, prepend=beta0[0]) + rng.normal(0.0, 0.003, n_obs)
    key_rate = 0.06 + 0.4 * beta0 + 0.015 * post_shift + rng.normal(0.0, 0.004, n_obs)
    tau1 = 4.8 - 1.2 * beta0 - 0.5 * post_shift + rng.normal(0.0, 0.2, n_obs)

    core = pd.DataFrame(
        {
            "date": dates,
            "beta0": beta0,
            "beta1": beta1,
            "beta2": beta2,
            "tau1": tau1,
            "curve_source_date": dates + pd.offsets.MonthEnd(0),
        }
    )
    core.loc[core["date"] == "2014-08-01", "beta1"] = 12.0
    core.loc[core["date"] == "2014-08-01", "beta2"] = -12.0

    macro = pd.DataFrame(
        {
            "date": dates,
            "cpi_mom": cpi_mom,
            "cpi_yoy": 0.05 + 6 * cpi_mom,
            "key_rate": key_rate,
            "analyst_cpi_2026": np.nan,
            "analyst_key_rate_2026": np.nan,
            "usd_rub": 35 + np.cumsum(rng.normal(0.0, 0.2, n_obs)),
            "budget_signal": (dates >= pd.Timestamp("2022-02-01")).astype(int),
            "breakeven_5y": np.nan,
            "breakeven_10y": np.nan,
        }
    )

    core_path = tmp_path / "monthly_ns_factors_core.csv"
    macro_path = tmp_path / "monthly_expectations_panel.csv"
    core.to_csv(core_path, index=False)
    macro.to_csv(macro_path, index=False)
    return core_path, macro_path


def test_regime_analysis_smoke(tmp_path: Path) -> None:
    core_path, macro_path = _build_synthetic_inputs(tmp_path)
    output_dir = tmp_path / "regime_outputs"
    findings_path = tmp_path / "p3_regime_findings.md"

    result = run_regime_analysis(
        core_path=core_path,
        macro_path=macro_path,
        output_dir=output_dir,
        findings_path=findings_path,
    )

    comparison_path = Path(result["comparison_path"])
    break_tests_path = Path(result["break_tests_path"])
    summary_path = Path(result["summary_path"])

    assert comparison_path.exists()
    assert break_tests_path.exists()
    assert summary_path.exists()
    assert findings_path.exists()

    comparison = pd.read_csv(comparison_path)
    break_tests = pd.read_csv(break_tests_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    findings_text = findings_path.read_text(encoding="utf-8")

    assert {"table", "regime_id", "target", "metric", "value"}.issubset(comparison.columns)
    assert {"test_name", "target", "p_value", "reject_5pct"}.issubset(break_tests.columns)
    assert "regimes" in summary
    assert summary["input_data"]["core_last_date"] == "2026-02-01"
    assert "Краткие выводы" in findings_text
    assert "2014-08-01" in findings_text

