from pathlib import Path

import pandas as pd

from ofz_curve.policy_path_gap import run_policy_path_gap_analysis


ROOT = Path(__file__).resolve().parents[2]
SURVEY_WORKBOOK_PATH = ROOT / "data/raw/macro/expectations/cbr_macro_survey_full.xlsx"
OFFICIAL_CURVE_PATH = ROOT / "data/processed/cbr_zero_curve/monthly_cbr_zero_curve.csv"


def test_policy_path_gap_full_run(tmp_path):
    output_dir = tmp_path / "policy_path_gap"
    findings_path = tmp_path / "p4_policy_path_gap_findings.md"
    figure_path = tmp_path / "policy_path_gap.png"

    result = run_policy_path_gap_analysis(
        survey_workbook_path=SURVEY_WORKBOOK_PATH,
        official_curve_path=OFFICIAL_CURVE_PATH,
        output_dir=output_dir,
        findings_path=findings_path,
        figure_path=figure_path,
    )

    panel_path = output_dir / "policy_path_gap_panel.csv"
    summary_path = output_dir / "policy_path_gap_summary.csv"
    metadata_path = output_dir / "policy_path_gap_metadata.json"

    for path in [panel_path, summary_path, metadata_path, findings_path, figure_path]:
        assert path.exists(), f"Missing artifact: {path}"

    summary_df = pd.read_csv(summary_path)
    findings_text = findings_path.read_text(encoding="utf-8")

    stable_2 = summary_df.loc[
        (summary_df["gap_column"] == "nominal_gap_2y_pp")
        & (summary_df["regime_id"] == "stabilization_2024_plus")
    ].iloc[0]

    assert stable_2["sample_start"] == "2024-01-01"
    assert stable_2["sample_end"] == "2026-03-01"
    assert stable_2["mean_gap_pp"] > 2.0
    assert bool(stable_2["significant_positive_5pct"])
    assert "2-летняя официальная доходность" in findings_text
    assert "2024--2026" in findings_text
    assert result["metadata"]["coverage"]["survey_path_start"] == "2021-05-01"
