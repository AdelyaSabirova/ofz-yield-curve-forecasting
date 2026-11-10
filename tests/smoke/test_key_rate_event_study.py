from pathlib import Path

import pandas as pd
import pytest

from ofz_curve.key_rate_event_study import (
    classify_decision_from_title,
    parse_key_rate_calendar,
    run_key_rate_event_study,
)


ROOT = Path(__file__).resolve().parents[2]
OFFICIAL_CURVE_PATH = ROOT / "data/raw/market/cbr_zero_curve/cbr_zero_curve_daily.csv"
MODEL_FACTORS_PATH = ROOT / "data/processed/core_2026/daily_ns_factors_core.csv"

pytestmark = pytest.mark.filterwarnings(
    "ignore:Precision loss occurred in moment calculation due to catastrophic cancellation.*:RuntimeWarning"
)


CALENDAR_HTML = """
<div data-tabs-tab="t1">2021</div>
<div data-tabs-tab="t9">2022</div>
<div data-tabs-tab="t11">2024</div>
<div data-tabs-tab="t12">2025</div>
<div data-tabs-content="t1">
  <div class="calendar-main-events">
    <div class="main-events_day">
      <div class="date">12 February 2021</div>
      <div class="main-events">
        <div class="main-event">
          <div class="title"><span>Bank of Russia Board of Directors’ key rate meeting</span></div>
          <div class="info"><a href="/eng/press/pr/?file=12022021_133000Key_eng.htm">Press release</a></div>
        </div>
      </div>
    </div>
  </div>
</div>
<div data-tabs-content="t9">
  <div class="calendar-main-events">
    <div class="main-events_day">
      <div class="date">29 April 2022</div>
      <div class="main-events">
        <div class="main-event">
          <div class="title"><span>Bank of Russia Board of Directors’ key rate meeting</span></div>
          <div class="info"><a href="/eng/press/pr/?file=29042022_133000Key_eng.htm">Press release</a></div>
        </div>
      </div>
    </div>
  </div>
</div>
<div data-tabs-content="t11">
  <div class="calendar-main-events">
    <div class="main-events_day">
      <div class="date">22 March 2024</div>
      <div class="main-events">
        <div class="main-event">
          <div class="title"><span>Bank of Russia Board of Directors’ key rate meeting</span></div>
          <div class="info"><a href="/eng/press/pr/?file=22032024_133000key_e.htm">Press release</a></div>
        </div>
      </div>
    </div>
  </div>
</div>
<div data-tabs-content="t12">
  <div class="calendar-main-events">
    <div class="main-events_day">
      <div class="date">6 June 2025</div>
      <div class="main-events">
        <div class="main-event">
          <div class="title"><span>Bank of Russia Board of Directors’ key rate meeting</span></div>
          <div class="info"><a href="/eng/press/pr/?file=06062025_133000key_e.htm">Press release</a></div>
        </div>
      </div>
    </div>
  </div>
</div>
<div data-tabs-tab="t3">2019</div>
<div data-tabs-content="t3">
  <table class="data">
    <tr><th>Date</th><th>Event</th></tr>
    <tr>
      <td>8 February</td>
      <td>Board of Directors’ monetary policy meeting <a href="/eng/press/PR/?file=08022019_133000Key_eng.htm">Press release</a></td>
    </tr>
  </table>
</div>
"""


TITLE_MAP = {
    "https://www.cbr.ru/eng/press/pr/?file=12022021_133000Key_eng.htm": "Bank of Russia raises the key rate by 25 bp to 4.50% p.a.",
    "https://www.cbr.ru/eng/press/pr/?file=29042022_133000Key_eng.htm": "Bank of Russia cuts the key rate by 300 bp to 14.00% p.a.",
    "https://www.cbr.ru/eng/press/pr/?file=22032024_133000key_e.htm": "The Bank of Russia keeps the key rate at 16.00% p.a.",
    "https://www.cbr.ru/eng/press/pr/?file=06062025_133000key_e.htm": "Bank of Russia cuts the key rate by 100 bp to 20.00% p.a.",
    "https://www.cbr.ru/eng/press/PR/?file=08022019_133000Key_eng.htm": "Bank of Russia keeps the key rate at 7.75% p.a.",
}


def _title_loader(url: str) -> str:
    return TITLE_MAP[url]


def test_parse_key_rate_calendar_handles_new_and_old_layouts():
    events = parse_key_rate_calendar(CALENDAR_HTML)
    assert events.shape[0] == 5
    assert events["event_date"].min().strftime("%Y-%m-%d") == "2019-02-08"
    assert events["event_date"].max().strftime("%Y-%m-%d") == "2025-06-06"
    assert events["is_unscheduled"].sum() == 0


def test_key_rate_event_study_end_to_end_with_local_curves(tmp_path):
    output_dir = tmp_path / "key_rate_event_study"
    findings_path = tmp_path / "p4_key_rate_event_findings.md"
    figure_path = tmp_path / "key_rate_event_study.png"

    result = run_key_rate_event_study(
        official_curve_path=OFFICIAL_CURVE_PATH,
        model_factors_path=MODEL_FACTORS_PATH,
        output_dir=output_dir,
        findings_path=findings_path,
        figure_path=figure_path,
        calendar_html=CALENDAR_HTML,
        title_loader=_title_loader,
    )

    events_path = output_dir / "key_rate_meeting_events.csv"
    reactions_path = output_dir / "key_rate_event_reactions.csv"
    summary_path = output_dir / "key_rate_event_summary.csv"
    metadata_path = output_dir / "key_rate_event_metadata.json"

    for path in [events_path, reactions_path, summary_path, metadata_path, findings_path, figure_path]:
        assert path.exists(), f"Missing artifact: {path}"

    events_df = pd.read_csv(events_path, parse_dates=["event_date"])
    summary_df = pd.read_csv(summary_path)
    findings_text = findings_path.read_text(encoding="utf-8")

    assert set(events_df["decision_type"]) == {"raise", "cut", "keep"}
    assert events_df["is_regular_meeting"].all()
    assert {"pre_2022_02", "transition_2022_2023", "stabilization_2024_plus"} <= set(summary_df["regime_id"])
    assert {"official", "model"} <= set(summary_df["source_name"])
    assert "реакция" in findings_text.lower()
    assert result["metadata"]["coverage"]["regular_meetings"] == 5


def test_classify_decision_from_title():
    assert classify_decision_from_title("Bank of Russia keeps the key rate at 21.00% p.a.") == "keep"
    assert classify_decision_from_title("Bank of Russia cuts key rate by 100 bp to 17.00% p.a.") == "cut"
    assert classify_decision_from_title("The Bank of Russia increases the key rate by 100 b.p. to 9.50% p.a.") == "raise"
