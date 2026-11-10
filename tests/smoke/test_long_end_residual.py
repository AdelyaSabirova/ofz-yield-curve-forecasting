import pandas as pd
from pathlib import Path

from ofz_curve.long_end_residual import run_long_end_residual_analysis


ROOT = Path(__file__).resolve().parents[2]
CORE_PATH = ROOT / "data/processed/core_2026/monthly_ns_factors_core.csv"
MACRO_PATH = ROOT / "data/processed/macro_2026/monthly_expectations_panel.csv"
OFFICIAL_CURVE_PATH = ROOT / "data/processed/cbr_zero_curve/monthly_cbr_zero_curve.csv"


def test_long_end_residual_full_run(tmp_path):
    output_dir = tmp_path / "long_end_residual"
    findings_path = tmp_path / "p4_long_end_residual_findings.md"
    figure_path = tmp_path / "long_end_slope_residual.png"

    result = run_long_end_residual_analysis(
        core_path=CORE_PATH,
        macro_path=MACRO_PATH,
        official_curve_path=OFFICIAL_CURVE_PATH,
        output_dir=output_dir,
        findings_path=findings_path,
        figure_path=figure_path,
    )

    summary_path = output_dir / "long_end_residual_summary.csv"
    residual_series_path = output_dir / "long_end_residual_series.csv"
    coefficients_path = output_dir / "long_end_residual_coefficients.csv"
    metadata_path = output_dir / "long_end_residual_metadata.json"

    for path in [summary_path, residual_series_path, coefficients_path, metadata_path, findings_path, figure_path]:
        assert path.exists(), f"Missing artifact: {path}"

    summary_df = pd.read_csv(summary_path)
    findings_text = findings_path.read_text(encoding="utf-8")

    official_post = summary_df.loc[
        (summary_df["target_name"] == "official")
        & (summary_df["spec_name"] == "macro_only")
        & (summary_df["regime_id"] == "post_2022_02")
    ].iloc[0]
    official_post_full = summary_df.loc[
        (summary_df["target_name"] == "official")
        & (summary_df["spec_name"] == "macro_plus_market_structure")
        & (summary_df["regime_id"] == "post_2022_02")
    ].iloc[0]

    assert official_post["sample_end"] == "2026-02-01"
    assert official_post_full["sample_end"] == "2025-05-01"
    assert official_post["mean_residual_pp"] > 2.0
    assert official_post["positive_share"] > 0.9
    assert bool(official_post["significant_positive_5pct"])
    assert official_post_full["mean_residual_pp"] > 2.0
    assert "остаточный компонент длинного конца" in findings_text
    assert "2026-02" in findings_text
    assert "2025-05" in findings_text
    assert result["metadata"]["coverage"]["panel_end"] == "2026-02-01"
