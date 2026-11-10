import json
from pathlib import Path

import pandas as pd

from ofz_curve.predictive_power import run_predictive_power


ROOT = Path(__file__).resolve().parents[2]
CORE_PATH = ROOT / "data/processed/core_2026/monthly_ns_factors_core.csv"
MACRO_PATH = ROOT / "data/processed/macro_2026/monthly_expectations_panel.csv"
OFFICIAL_CURVE_PATH = ROOT / "data/processed/cbr_zero_curve/monthly_cbr_zero_curve.csv"


def test_predictive_power_full_run(tmp_path):
    output_dir = tmp_path / "predictive_outputs"
    findings_path = tmp_path / "p4_predictive_findings.md"

    result = run_predictive_power(
        core_path=CORE_PATH,
        macro_path=MACRO_PATH,
        official_curve_path=OFFICIAL_CURVE_PATH,
        output_dir=output_dir,
        findings_path=findings_path,
    )

    inflation_path = output_dir / "inflation_model_comparison.csv"
    key_rate_path = output_dir / "key_rate_model_comparison.csv"
    dm_path = output_dir / "dm_tests.csv"
    key_rate_tau1_path = output_dir / "key_rate_tau1_increment.csv"
    key_rate_tau1_dm_path = output_dir / "key_rate_tau1_dm_tests.csv"
    curve_benchmark_path = output_dir / "key_rate_curve_benchmark_comparison.csv"
    curve_benchmark_dm_path = output_dir / "key_rate_curve_benchmark_dm_tests.csv"
    curve_benchmark_summary_path = output_dir / "key_rate_curve_benchmark_summary.csv"
    market_control_path = output_dir / "key_rate_market_controls_comparison.csv"
    market_control_dm_path = output_dir / "key_rate_market_controls_dm_tests.csv"
    market_control_summary_path = output_dir / "key_rate_market_controls_summary.csv"
    summary_path = output_dir / "predictive_summary.json"

    for path in [
        inflation_path,
        key_rate_path,
        dm_path,
        key_rate_tau1_path,
        key_rate_tau1_dm_path,
        curve_benchmark_path,
        curve_benchmark_dm_path,
        curve_benchmark_summary_path,
        market_control_path,
        market_control_dm_path,
        market_control_summary_path,
        summary_path,
        findings_path,
    ]:
        assert path.exists(), f"Missing artifact: {path}"

    inflation_df = pd.read_csv(inflation_path)
    key_rate_df = pd.read_csv(key_rate_path)
    dm_df = pd.read_csv(dm_path)
    key_rate_tau1_df = pd.read_csv(key_rate_tau1_path)
    key_rate_tau1_dm_df = pd.read_csv(key_rate_tau1_dm_path)
    curve_benchmark_df = pd.read_csv(curve_benchmark_path)
    curve_benchmark_dm_df = pd.read_csv(curve_benchmark_dm_path)
    curve_benchmark_summary_df = pd.read_csv(curve_benchmark_summary_path)
    market_control_df = pd.read_csv(market_control_path)
    market_control_dm_df = pd.read_csv(market_control_dm_path)
    market_control_summary_df = pd.read_csv(market_control_summary_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    findings_text = findings_path.read_text(encoding="utf-8")
    data = result["data"]

    assert set(inflation_df["horizon_months"]) == {1, 3, 6, 12}
    assert set(key_rate_df["horizon_months"]) == {1, 3, 6, 12}
    assert {"all", "pre_2022_02", "post_2022_02"} <= set(inflation_df["split"])
    assert {"all", "pre_2022_02", "post_2022_02"} <= set(key_rate_df["split"])
    assert {"all", "pre_2022_02", "post_2022_02"} <= set(key_rate_tau1_df["split"])
    assert {"all", "pre_2022_02", "post_2022_02"} <= set(curve_benchmark_df["split"])
    assert {"all", "pre_2022_02", "post_2022_02"} <= set(curve_benchmark_summary_df["split"])
    assert {"all", "post_2022_02", "stabilization_2024_plus"} <= set(market_control_df["split"])
    assert {"all", "post_2022_02", "stabilization_2024_plus"} <= set(market_control_summary_df["split"])
    assert (inflation_df["status"] == "ok").any()
    assert (key_rate_df["status"] == "ok").any()
    assert (dm_df["status"] == "ok").any()
    assert (key_rate_tau1_df["status"] == "ok").any()
    assert (key_rate_tau1_dm_df["status"] == "ok").any()
    assert (curve_benchmark_df["status"] == "ok").any()
    assert (curve_benchmark_dm_df["status"] == "ok").any()
    assert (market_control_df["status"] == "ok").any()
    assert (market_control_dm_df["status"] == "ok").any()
    assert {"simple_spreads", "official_shape"} <= set(curve_benchmark_df["benchmark_name"])
    assert {"simple_spreads", "official_shape"} <= set(curve_benchmark_summary_df["benchmark_name"])
    assert "Инфляция + рост долга сектора госуправления" in set(market_control_summary_df["model_label"])
    assert "Инфляция + доля нерезидентов" in set(market_control_summary_df["model_label"])
    assert "Инфляция + все наблюдаемые контроли" in set(
        market_control_df["model_label"]
    )
    assert "Инфляция + 10-летняя инфляционная компенсация" in set(market_control_summary_df["model_label"])
    assert "Инфляция + breakeven 10Y" not in set(market_control_summary_df["model_label"])

    assert summary["input_coverage"]["panel_end"] == "2026-02-01"
    assert summary["input_coverage"]["inflation_actual_end"] == "2026-02-01"
    assert summary["input_coverage"]["key_rate_actual_end"] == "2026-02-01"
    assert summary["input_coverage"]["expectations_start"] == "2023-09-01"
    assert "2026-02-01" in findings_text
    assert "Добавочная прогностическая сила факторов кривой" in findings_text
    assert "tau1" in findings_text
    assert "спред" in findings_text
    assert "официальной КБД" in findings_text
    assert "долговых и инвестиционных показателей" in findings_text
    assert "нерезидентов" in findings_text
    assert "10-летняя инфляционная компенсация" in findings_text
    assert "breakeven 10Y" not in findings_text
    assert "8` из `12" in findings_text

    required_lag_columns = {
        "cpi_yoy_lag1",
        "ofz_nonresident_share_lag1",
        "general_government_debt_growth_12m_lag1",
    }
    assert required_lag_columns <= set(data.columns)
    row = data.loc[data["date"] == pd.Timestamp("2026-02-01")].iloc[0]
    prev_row = data.loc[data["date"] == pd.Timestamp("2026-01-01")].iloc[0]
    assert abs(row["cpi_yoy_lag1"] - prev_row["cpi_yoy"]) < 1e-12
    assert abs(row["ofz_nonresident_share_lag1"] - prev_row["ofz_nonresident_share"]) < 1e-12
    assert abs(row["general_government_debt_growth_12m_lag1"] - prev_row["general_government_debt_growth_12m"]) < 1e-12

    assert result["summary_metadata"]["artifacts"]["inflation_model_comparison_csv"].endswith(
        "inflation_model_comparison.csv"
    )
    assert result["summary_metadata"]["artifacts"]["key_rate_tau1_increment_csv"].endswith(
        "key_rate_tau1_increment.csv"
    )
    assert result["summary_metadata"]["artifacts"]["key_rate_curve_benchmark_summary_csv"].endswith(
        "key_rate_curve_benchmark_summary.csv"
    )
    assert result["summary_metadata"]["artifacts"]["key_rate_market_controls_summary_csv"].endswith(
        "key_rate_market_controls_summary.csv"
    )
