import numpy as np
import pandas as pd
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from build_expectations_panel import (
    _extract_general_government_debt_from_sheet,
    _parse_ofz_nonresident_table,
)


def test_parse_ofz_nonresident_table_normalizes_share():
    table = pd.DataFrame(
        {
            "Date": ["2025-11-30", "2025-12-31", "2026-01-31"],
            "Nonresidents’ OFZ holdings, bln roubles": [963.0, 1008.0, 1033.0],
            "OFZ market size, bln roubles": [29207.0, 29983.0, 30168.0],
            "Nonresidents’ market share, %": [3.3, 3.4, 3.4],
        }
    )

    parsed = _parse_ofz_nonresident_table(table)

    assert list(parsed.columns) == [
        "date",
        "ofz_nonresident_holdings_bln",
        "ofz_market_size_bln",
        "ofz_nonresident_share",
    ]
    assert parsed["date"].dt.strftime("%Y-%m-%d").tolist() == ["2025-11-01", "2025-12-01", "2026-01-01"]
    assert np.isclose(parsed.loc[0, "ofz_nonresident_share"], 0.033)
    assert np.isclose(parsed.loc[2, "ofz_nonresident_holdings_bln"], 1033.0)


def test_extract_general_government_debt_from_sheet_builds_12m_growth():
    month_dates = pd.date_range("2013-01-01", periods=15, freq="MS")
    sheet = pd.DataFrame(np.nan, index=range(40), columns=range(len(month_dates) + 1), dtype=object)
    sheet.iloc[4, 1:] = month_dates
    sheet.iloc[31, 0] = "General government"
    debt_values = np.linspace(4000.0, 5400.0, num=len(month_dates))
    sheet.iloc[31, 1:] = debt_values

    parsed = _extract_general_government_debt_from_sheet(sheet)

    assert list(parsed.columns) == ["date", "general_government_debt_bln", "general_government_debt_growth_12m"]
    assert parsed["date"].iloc[0].strftime("%Y-%m-%d") == "2013-09-01"
    expected_last_bln = debt_values[-1] / 1000.0
    assert np.isclose(parsed["general_government_debt_bln"].iloc[-1], expected_last_bln)

    last_row = parsed.dropna(subset=["general_government_debt_growth_12m"]).iloc[-1]
    current_idx = month_dates.get_loc(last_row["date"])
    current_debt = debt_values[current_idx] / 1000.0
    previous_debt = debt_values[current_idx - 12] / 1000.0
    expected_growth = np.log(current_debt) - np.log(previous_debt)
    assert np.isclose(last_row["general_government_debt_growth_12m"], expected_growth)
