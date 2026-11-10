from pathlib import Path

import pandas as pd

from ofz_curve import ofz_auction_study as auction_module
from ofz_curve.ofz_auction_study import load_ofz_auction_events, run_ofz_auction_event_study


ROOT = Path(__file__).resolve().parents[2]
OFFICIAL_CURVE_PATH = ROOT / "data/raw/market/cbr_zero_curve/cbr_zero_curve_daily.csv"
MODEL_FACTORS_PATH = ROOT / "data/processed/core_2026/daily_ns_factors_core.csv"


def _message(post_id: int, date_time: str, text: str) -> str:
    return f"""
    <div class="tgme_widget_message_wrap">
      <div class="tgme_widget_message">
        <a class="tgme_widget_message_date" href="https://t.me/minfin/{post_id}">
          <time datetime="{date_time}"></time>
        </a>
        <div class="tgme_widget_message_text">{text}</div>
      </div>
    </div>
    """


PAGE_MAP = {
    "https://t.me/s/minfin": "\n".join(
        [
            _message(
                110,
                "2025-11-12T15:00:00+00:00",
                "Минфин России на аукционе разместил два выпуска ОФЗ на общую сумму 240,000 млрд рублей "
                "На аукционе по размещению ОФЗ-ПД выпуска № 26248 с погашением 16 мая 2040 года было продано бумаг "
                "на общую сумму 140,000 млрд рублей. Выручка от аукциона - 110,000 млрд рублей.",
            ),
            _message(
                109,
                "2024-04-10T15:00:00+00:00",
                "Минфин России на аукционах разместил два выпуска ОФЗ на общую сумму 150,000 млрд рублей "
                "На аукционе по размещению ОФЗ-ПД выпуска № 26246 с погашением 12 марта 2036 года было продано бумаг "
                "на общую сумму 90,000 млрд рублей. Выручка от аукциона - 80,000 млрд рублей.",
            ),
        ]
    ),
    "https://t.me/s/minfin?before=109": "\n".join(
        [
            _message(
                90,
                "2023-06-14T15:00:00+00:00",
                "Минфин России на аукционах разместил два выпуска ОФЗ на общую сумму 120,000 млрд рублей "
                "На аукционе по размещению ОФЗ-ПД выпуска № 26241 с погашением 17 ноября 2032 года было продано бумаг "
                "на общую сумму 37,323 млрд рублей. Выручка от аукциона - 35,474 млрд рублей.",
            ),
            _message(
                89,
                "2022-11-23T15:00:00+00:00",
                "Минфин России на аукционах разместил три выпуска ОФЗ на общую сумму 60,000 млрд рублей "
                "На первом аукционе по размещению ОФЗ-ПД выпуска № 26240 с погашением 30 июля 2036 года было продано бумаг "
                "на общую сумму 20,000 млрд рублей. Выручка от аукциона - 18,000 млрд рублей.",
            ),
            _message(
                88,
                "2022-10-25T14:00:00+00:00",
                "Минфин России на аукционе 26 октября разместит три выпуска ОФЗ и продолжит проведение еженедельных аукционов.",
            ),
        ]
    ),
    "https://t.me/s/minfin?before=89": _message(
        88,
        "2022-01-15T15:00:00+00:00",
        "Минфин России опубликовал комментарий по долговому рынку.",
    ),
    "https://t.me/s/minfin?before=88": _message(
        70,
        "2021-12-15T15:00:00+00:00",
        "Минфин России подвел итоги работы за год.",
    ),
}


def _fake_fetch(url: str, **_: object) -> str:
    return PAGE_MAP[url]


def test_load_ofz_auction_events_parses_results_and_skips_announcements(monkeypatch):
    monkeypatch.setattr(auction_module, "_fetch_text", _fake_fetch)
    events = load_ofz_auction_events(start_date="2022-01-01", max_pages=5)

    assert events.shape[0] == 4
    assert events["event_date"].min().strftime("%Y-%m-%d") == "2022-11-23"
    assert events["event_date"].max().strftime("%Y-%m-%d") == "2025-11-12"
    assert events["total_nominal_bln"].tolist() == [60.0, 120.0, 150.0, 240.0]


def test_run_ofz_auction_event_study_end_to_end(monkeypatch, tmp_path):
    monkeypatch.setattr(auction_module, "_fetch_text", _fake_fetch)

    output_dir = tmp_path / "ofz_auction_study"
    findings_path = tmp_path / "p4_ofz_auction_findings.md"
    figure_path = tmp_path / "ofz_auction_study.png"

    result = run_ofz_auction_event_study(
        official_curve_path=OFFICIAL_CURVE_PATH,
        model_factors_path=MODEL_FACTORS_PATH,
        output_dir=output_dir,
        findings_path=findings_path,
        figure_path=figure_path,
        analysis_start_date="2022-02-01",
        crawl_start_date="2022-01-01",
        max_pages=5,
    )

    events_path = output_dir / "ofz_auction_events.csv"
    monthly_path = output_dir / "ofz_auction_monthly.csv"
    reactions_path = output_dir / "ofz_auction_event_reactions.csv"
    summary_path = output_dir / "ofz_auction_event_summary.csv"
    metadata_path = output_dir / "ofz_auction_metadata.json"

    for path in [events_path, monthly_path, reactions_path, summary_path, metadata_path, findings_path, figure_path]:
        assert path.exists(), f"Missing artifact: {path}"

    summary_df = pd.read_csv(summary_path)
    findings_text = findings_path.read_text(encoding="utf-8")

    assert {"official", "model"} <= set(summary_df["source_name"])
    assert {"transition_2022_2023", "stabilization_2024_plus"} <= set(summary_df["regime_id"])
    assert {"low", "high"} <= set(summary_df["volume_bucket"])
    assert "аукцион" in findings_text.lower()
    assert result["metadata"]["coverage"]["event_count"] == 4
