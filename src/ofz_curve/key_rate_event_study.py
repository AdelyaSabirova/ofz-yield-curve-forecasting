from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
import time
from typing import Any, Callable
from urllib.parse import urljoin

from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
from nelson_siegel_svensson import NelsonSiegelCurve
import pandas as pd
import requests
from scipy.stats import ttest_1samp

from .paths import (
    CBR_ZERO_CURVE_DAILY_CSV,
    CORE_DAILY_NS_FACTORS_CSV,
    KEY_RATE_EVENT_STUDY_DIR,
    KEY_RATE_EVENT_STUDY_FIGURE,
    PROJECT_ROOT,
)


CBR_KEY_RATE_CALENDAR_URL = "https://www.cbr.ru/eng/dkp/cal_mp/"
DEFAULT_TIMEOUT_SECONDS = 30
DEFAULT_RETRIES = 3
DEFAULT_RETRY_SLEEP_SECONDS = 1.5
EVENT_HORIZONS_DAYS = (1, 3, 5)
DEFAULT_EVENT_START_DATE = pd.Timestamp("2019-01-01")

DEFAULT_OFFICIAL_CURVE_PATH = CBR_ZERO_CURVE_DAILY_CSV
DEFAULT_MODEL_FACTORS_PATH = CORE_DAILY_NS_FACTORS_CSV
DEFAULT_OUTPUT_DIR = KEY_RATE_EVENT_STUDY_DIR
DEFAULT_FINDINGS_PATH = PROJECT_ROOT / "docs" / "p4_key_rate_event_findings.md"
DEFAULT_FIGURE_PATH = KEY_RATE_EVENT_STUDY_FIGURE

DATE_COLUMN = "date"

REGIME_SPECS = (
    ("pre_2022_02", "До февраля 2022 года", pd.Timestamp("2019-01-01"), pd.Timestamp("2022-01-31")),
    ("transition_2022_2023", "Переходный режим 2022--2023", pd.Timestamp("2022-02-01"), pd.Timestamp("2023-12-31")),
    ("stabilization_2024_plus", "Устойчивый режим 2024--2026", pd.Timestamp("2024-01-01"), None),
)


@dataclass(frozen=True)
class CurveSpec:
    name: str
    label: str


CURVE_SPECS = (
    CurveSpec(name="official", label="Официальная КБД Банка России"),
    CurveSpec(name="model", label="Авторская кривая Нельсона--Сигеля"),
)


def _normalize_text(value: str | None) -> str:
    return " ".join((value or "").replace("\xa0", " ").split())


def _normalize_text_lower(value: str | None) -> str:
    return _normalize_text(value).lower()


def _fetch_text(
    url: str,
    *,
    session: requests.Session | None = None,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    retries: int = DEFAULT_RETRIES,
) -> str:
    local_session = session or requests.Session()
    headers = {"User-Agent": "nir-adelya-key-rate-event-study/1.0"}

    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            response = local_session.get(url, headers=headers, timeout=timeout)
            response.raise_for_status()
            return response.text
        except Exception as error:  # noqa: BLE001
            last_error = error
            if attempt == retries:
                break
            time.sleep(DEFAULT_RETRY_SLEEP_SECONDS * attempt)
    raise RuntimeError(f"Не удалось загрузить страницу {url}") from last_error


def _parse_calendar_date(date_text: str, year_hint: int) -> pd.Timestamp | None:
    candidate = _normalize_text(date_text)
    if not re.search(r"\b20\d{2}\b", candidate):
        candidate = f"{candidate} {year_hint}"
    stamp = pd.to_datetime(candidate, errors="coerce", dayfirst=True)
    if pd.isna(stamp):
        stamp = pd.to_datetime(candidate, errors="coerce")
    if pd.isna(stamp):
        return None
    return pd.Timestamp(stamp).normalize()


def _parse_new_style_events(panel: BeautifulSoup, *, year_hint: int, base_url: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for day in panel.select(".main-events_day"):
        date_el = day.select_one(".date")
        if date_el is None:
            continue
        date_text = _normalize_text(date_el.get_text(" ", strip=True))
        event_date = _parse_calendar_date(date_text, year_hint)
        if event_date is None:
            continue
        for event in day.select(".main-event"):
            event_text = _normalize_text(event.get_text(" ", strip=True))
            lowered = _normalize_text_lower(event_text)
            if "key rate meeting" not in lowered and "monetary policy meeting" not in lowered:
                continue
            press_release_url = None
            for link in event.select("a[href]"):
                href = link.get("href")
                if not href:
                    continue
                link_text = _normalize_text_lower(link.get_text(" ", strip=True))
                if "press release" in link_text or "/press/pr/" in href.lower():
                    press_release_url = urljoin(base_url, href)
                    break
            rows.append(
                {
                    "event_date": event_date,
                    "event_date_text": date_text,
                    "calendar_year": year_hint,
                    "raw_event_text": event_text,
                    "press_release_url": press_release_url,
                    "is_unscheduled": "unscheduled" in lowered,
                }
            )
    return rows


def _parse_old_style_events(panel: BeautifulSoup, *, year_hint: int, base_url: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in panel.select("table.data tr")[1:]:
        cells = row.find_all("td")
        if not cells:
            continue
        date_text = _normalize_text(cells[0].get_text(" ", strip=True))
        event_text = _normalize_text(row.get_text(" ", strip=True))
        lowered = _normalize_text_lower(event_text)
        if "key rate meeting" not in lowered and "monetary policy meeting" not in lowered:
            continue
        event_date = _parse_calendar_date(date_text, year_hint)
        if event_date is None:
            continue
        press_release_url = None
        for link in row.select("a[href]"):
            href = link.get("href")
            if not href:
                continue
            link_text = _normalize_text_lower(link.get_text(" ", strip=True))
            if "press release" in link_text or "/press/pr/" in href.lower():
                press_release_url = urljoin(base_url, href)
                break
        rows.append(
            {
                "event_date": event_date,
                "event_date_text": date_text,
                "calendar_year": year_hint,
                "raw_event_text": event_text,
                "press_release_url": press_release_url,
                "is_unscheduled": "unscheduled" in lowered,
            }
        )
    return rows


def parse_key_rate_calendar(html: str, *, calendar_url: str = CBR_KEY_RATE_CALENDAR_URL) -> pd.DataFrame:
    soup = BeautifulSoup(html, "html.parser")
    year_tabs = {
        tab.get("data-tabs-tab"): int(_normalize_text(tab.get_text(" ", strip=True)))
        for tab in soup.select("[data-tabs-tab]")
        if _normalize_text(tab.get_text(" ", strip=True)).isdigit()
    }

    rows: list[dict[str, Any]] = []
    for tab_key, year in sorted(year_tabs.items(), key=lambda item: item[1]):
        panel = soup.select_one(f'[data-tabs-content="{tab_key}"]')
        if panel is None:
            continue
        rows.extend(_parse_new_style_events(panel, year_hint=year, base_url=calendar_url))
        rows.extend(_parse_old_style_events(panel, year_hint=year, base_url=calendar_url))

    events = pd.DataFrame(rows)
    if events.empty:
        return pd.DataFrame(
            columns=[
                "event_date",
                "event_date_text",
                "calendar_year",
                "raw_event_text",
                "press_release_url",
                "is_unscheduled",
            ]
        )

    events = (
        events.drop_duplicates(subset=["event_date", "press_release_url", "is_unscheduled"])
        .sort_values("event_date")
        .reset_index(drop=True)
    )
    return events


def classify_decision_from_title(title: str | None) -> str:
    normalized = _normalize_text_lower(title)
    if any(
        marker in normalized
        for marker in (
            "keeps the key rate",
            "keeps key rate",
            "leaves the key rate",
            "leaves key rate",
            "maintains the key rate",
            "maintains key rate",
        )
    ):
        return "keep"
    if any(
        marker in normalized
        for marker in (
            "cuts the key rate",
            "cuts key rate",
            "reduces the key rate",
            "reduces key rate",
            "lowers the key rate",
            "lowers key rate",
        )
    ):
        return "cut"
    if any(
        marker in normalized
        for marker in (
            "raises the key rate",
            "raises key rate",
            "increases the key rate",
            "increases key rate",
            "hikes the key rate",
            "hikes key rate",
        )
    ):
        return "raise"
    return "unknown"


def _extract_resulting_rate(title: str | None) -> float | None:
    normalized = _normalize_text(title)
    match = re.search(r"\b(?:to|at)\s+(\d+(?:\.\d+)?)%", normalized)
    if not match:
        return None
    return float(match.group(1))


def _safe_one_sided_ttest(sample: pd.Series, *, alternative: str) -> float | None:
    cleaned = sample.dropna()
    if cleaned.shape[0] < 2:
        return None
    return float(ttest_1samp(cleaned, popmean=0.0, alternative=alternative).pvalue)


def load_key_rate_meeting_events(
    *,
    calendar_url: str = CBR_KEY_RATE_CALENDAR_URL,
    session: requests.Session | None = None,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    retries: int = DEFAULT_RETRIES,
    analysis_end_date: pd.Timestamp | str | None = None,
    analysis_start_date: pd.Timestamp | str | None = DEFAULT_EVENT_START_DATE,
    title_loader: Callable[[str], str] | None = None,
) -> pd.DataFrame:
    html = _fetch_text(calendar_url, session=session, timeout=timeout, retries=retries)
    events = parse_key_rate_calendar(html, calendar_url=calendar_url)
    if analysis_start_date is not None:
        start_stamp = pd.Timestamp(analysis_start_date).normalize()
        events = events.loc[events["event_date"] >= start_stamp].copy()
    if analysis_end_date is not None:
        end_stamp = pd.Timestamp(analysis_end_date).normalize()
        events = events.loc[events["event_date"] <= end_stamp].copy()

    local_session = session or requests.Session()
    loaded_titles: list[str | None] = []
    for url in events["press_release_url"]:
        if not isinstance(url, str) or not url:
            loaded_titles.append(None)
            continue
        if title_loader is not None:
            loaded_titles.append(_normalize_text(title_loader(url)))
            continue
        page_html = _fetch_text(url, session=local_session, timeout=timeout, retries=retries)
        soup = BeautifulSoup(page_html, "html.parser")
        title_tag = soup.select_one("h1") or soup.select_one("title")
        loaded_titles.append(_normalize_text(title_tag.get_text(" ", strip=True) if title_tag else ""))

    events = events.copy()
    events["press_release_title"] = loaded_titles
    events["decision_type"] = events["press_release_title"].map(classify_decision_from_title)
    events["resulting_key_rate_pct"] = events["press_release_title"].map(_extract_resulting_rate)
    events["has_press_release"] = events["press_release_url"].notna()
    events["is_regular_meeting"] = ~events["is_unscheduled"]
    return events.reset_index(drop=True)


def _build_model_curve_daily(core_path: Path) -> pd.DataFrame:
    core = pd.read_csv(core_path, parse_dates=[DATE_COLUMN]).sort_values(DATE_COLUMN).reset_index(drop=True)

    def model_yield(row: Any, maturity: float) -> float:
        curve = NelsonSiegelCurve(
            beta0=float(row.beta0),
            beta1=float(row.beta1),
            beta2=float(row.beta2),
            tau=float(row.tau1),
        )
        return float(curve(maturity) * 100)

    core["y1_00"] = [model_yield(row, 1.0) for row in core[["beta0", "beta1", "beta2", "tau1"]].itertuples(index=False)]
    core["y10_00"] = [model_yield(row, 10.0) for row in core[["beta0", "beta1", "beta2", "tau1"]].itertuples(index=False)]
    core["slope_10_1"] = core["y10_00"] - core["y1_00"]
    core["source_name"] = "model"
    core["source_label"] = "Авторская кривая Нельсона--Сигеля"
    return core[[DATE_COLUMN, "y1_00", "y10_00", "slope_10_1", "source_name", "source_label"]]


def _load_official_curve_daily(official_curve_path: Path) -> pd.DataFrame:
    official = pd.read_csv(official_curve_path, parse_dates=[DATE_COLUMN]).sort_values(DATE_COLUMN).reset_index(drop=True)
    official["slope_10_1"] = official["y10_00"] - official["y1_00"]
    official["source_name"] = "official"
    official["source_label"] = "Официальная КБД Банка России"
    return official[[DATE_COLUMN, "y1_00", "y10_00", "slope_10_1", "source_name", "source_label"]]


def _assign_regime(event_date: pd.Timestamp) -> tuple[str, str] | tuple[None, None]:
    for regime_id, regime_label, start_date, end_date in REGIME_SPECS:
        if event_date < start_date:
            continue
        if end_date is not None and event_date > end_date:
            continue
        return regime_id, regime_label
    return None, None


def build_key_rate_event_reactions(
    meetings: pd.DataFrame,
    *,
    official_curve_path: Path = DEFAULT_OFFICIAL_CURVE_PATH,
    model_factors_path: Path = DEFAULT_MODEL_FACTORS_PATH,
    regular_only: bool = True,
    horizons: tuple[int, ...] = EVENT_HORIZONS_DAYS,
) -> pd.DataFrame:
    meetings = meetings.copy().sort_values("event_date").reset_index(drop=True)
    if regular_only:
        meetings = meetings.loc[meetings["is_regular_meeting"]].copy()

    official = _load_official_curve_daily(official_curve_path)
    model = _build_model_curve_daily(model_factors_path)
    curves = pd.concat([official, model], ignore_index=True)

    analysis_end = curves[DATE_COLUMN].max()
    meetings = meetings.loc[meetings["event_date"] <= analysis_end].copy()
    meetings[["regime_id", "regime_label"]] = meetings["event_date"].apply(
        lambda value: pd.Series(_assign_regime(pd.Timestamp(value)))
    )
    meetings = meetings.dropna(subset=["regime_id"]).reset_index(drop=True)

    rows: list[dict[str, Any]] = []
    for curve_spec in CURVE_SPECS:
        curve = curves.loc[curves["source_name"] == curve_spec.name].sort_values(DATE_COLUMN).reset_index(drop=True)
        date_series = curve[DATE_COLUMN]
        for event in meetings.itertuples(index=False):
            position_mask = date_series >= event.event_date
            if not position_mask.any():
                continue
            base_idx = int(position_mask.idxmax())
            base_row = curve.iloc[base_idx]
            for horizon in horizons:
                if base_idx + horizon >= len(curve):
                    continue
                future_row = curve.iloc[base_idx + horizon]
                rows.append(
                    {
                        "source_name": curve_spec.name,
                        "source_label": curve_spec.label,
                        "event_date": event.event_date,
                        "aligned_curve_date": base_row[DATE_COLUMN],
                        "future_curve_date": future_row[DATE_COLUMN],
                        "calendar_year": int(event.calendar_year),
                        "regime_id": event.regime_id,
                        "regime_label": event.regime_label,
                        "decision_type": event.decision_type,
                        "press_release_title": event.press_release_title,
                        "press_release_url": event.press_release_url,
                        "is_unscheduled": bool(event.is_unscheduled),
                        "horizon_days": int(horizon),
                        "base_y1_pp": float(base_row["y1_00"]),
                        "base_y10_pp": float(base_row["y10_00"]),
                        "base_slope_pp": float(base_row["slope_10_1"]),
                        "future_y1_pp": float(future_row["y1_00"]),
                        "future_y10_pp": float(future_row["y10_00"]),
                        "future_slope_pp": float(future_row["slope_10_1"]),
                        "dy1_pp": float(future_row["y1_00"] - base_row["y1_00"]),
                        "dy10_pp": float(future_row["y10_00"] - base_row["y10_00"]),
                        "dslope_pp": float(future_row["slope_10_1"] - base_row["slope_10_1"]),
                    }
                )
    return pd.DataFrame(rows)


def summarize_key_rate_event_reactions(reactions: pd.DataFrame) -> pd.DataFrame:
    if reactions.empty:
        return pd.DataFrame()

    summary_rows: list[dict[str, Any]] = []
    for curve_spec in CURVE_SPECS:
        source_name = curve_spec.name
        source_label = curve_spec.label
        source_subset = reactions.loc[reactions["source_name"] == source_name].copy()
        if source_subset.empty:
            continue
        for regime_id, regime_label, _, _ in REGIME_SPECS:
            regime_subset = source_subset.loc[source_subset["regime_id"] == regime_id].copy()
            if regime_subset.empty:
                continue
            for horizon in EVENT_HORIZONS_DAYS:
                horizon_subset = regime_subset.loc[regime_subset["horizon_days"] == horizon].copy()
                if horizon_subset.empty:
                    continue
                for decision_type, decision_label in (
                    ("all", "Все регулярные заседания"),
                    ("keep", "Сохранение ставки"),
                    ("cut", "Снижение ставки"),
                    ("raise", "Повышение ставки"),
                ):
                    sample = horizon_subset if decision_type == "all" else horizon_subset.loc[
                        horizon_subset["decision_type"] == decision_type
                    ]
                    if sample.empty:
                        continue
                    p_slope = _safe_one_sided_ttest(sample["dslope_pp"], alternative="greater")
                    p_y1 = _safe_one_sided_ttest(sample["dy1_pp"], alternative="less")
                    p_y10 = _safe_one_sided_ttest(sample["dy10_pp"], alternative="greater")
                    summary_rows.append(
                        {
                            "source_name": source_name,
                            "source_label": source_label,
                            "regime_id": regime_id,
                            "regime_label": regime_label,
                            "decision_type": decision_type,
                            "decision_label": decision_label,
                            "horizon_days": int(horizon),
                            "n_events": int(sample["event_date"].nunique()),
                            "sample_start": sample["event_date"].min().strftime("%Y-%m-%d"),
                            "sample_end": sample["event_date"].max().strftime("%Y-%m-%d"),
                            "mean_dy1_pp": float(sample["dy1_pp"].mean()),
                            "mean_dy10_pp": float(sample["dy10_pp"].mean()),
                            "mean_dslope_pp": float(sample["dslope_pp"].mean()),
                            "median_dslope_pp": float(sample["dslope_pp"].median()),
                            "share_slope_positive": float((sample["dslope_pp"] > 0).mean()),
                            "p_value_slope_positive": p_slope,
                            "p_value_y1_negative": p_y1,
                            "p_value_y10_positive": p_y10,
                            "significant_slope_positive_5pct": bool(p_slope is not None and p_slope < 0.05),
                        }
                    )
    return pd.DataFrame(summary_rows)


def plot_key_rate_event_summary(summary_df: pd.DataFrame, figure_path: Path) -> Path:
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    official = summary_df.loc[
        (summary_df["source_name"] == "official")
        & (summary_df["decision_type"] == "all")
        & (summary_df["regime_id"].isin(["pre_2022_02", "transition_2022_2023", "stabilization_2024_plus"]))
    ].copy()
    model = summary_df.loc[
        (summary_df["source_name"] == "model")
        & (summary_df["decision_type"] == "all")
        & (summary_df["regime_id"].isin(["transition_2022_2023", "stabilization_2024_plus"]))
    ].copy()

    regime_labels = {
        "pre_2022_02": "До 2022-02",
        "transition_2022_2023": "2022--2023",
        "stabilization_2024_plus": "2024--2026",
    }

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6))

    official_h1 = official.loc[official["horizon_days"] == 1].copy()
    ordered_h1 = official_h1.set_index("regime_id").reindex(["pre_2022_02", "transition_2022_2023", "stabilization_2024_plus"])
    x_positions = range(len(ordered_h1))
    bar_width = 0.24

    axes[0].bar(
        [x - bar_width for x in x_positions],
        ordered_h1["mean_dy1_pp"],
        width=bar_width,
        color="#457b9d",
        label="1Y",
    )
    axes[0].bar(
        x_positions,
        ordered_h1["mean_dy10_pp"],
        width=bar_width,
        color="#1d3557",
        label="10Y",
    )
    axes[0].bar(
        [x + bar_width for x in x_positions],
        ordered_h1["mean_dslope_pp"],
        width=bar_width,
        color="#e76f51",
        label="10Y-1Y",
    )
    axes[0].axhline(0.0, color="#444444", linewidth=1.0, linestyle=":")
    axes[0].set_xticks(list(x_positions))
    axes[0].set_xticklabels([regime_labels[idx] for idx in ordered_h1.index])
    axes[0].set_title("Средняя реакция через 1 торговый день\nпо официальной КБД")
    axes[0].set_ylabel("п.п.")
    axes[0].legend(frameon=False, loc="upper left")
    axes[0].grid(alpha=0.25, linestyle=":")

    official_lines = official[["regime_id", "horizon_days", "mean_dslope_pp"]].copy()
    model_lines = model[["regime_id", "horizon_days", "mean_dslope_pp"]].copy()
    for regime_id, color, linestyle in (
        ("transition_2022_2023", "#6c757d", "-"),
        ("stabilization_2024_plus", "#c1121f", "-"),
    ):
        official_regime = official_lines.loc[official_lines["regime_id"] == regime_id].sort_values("horizon_days")
        model_regime = model_lines.loc[model_lines["regime_id"] == regime_id].sort_values("horizon_days")
        axes[1].plot(
            official_regime["horizon_days"],
            official_regime["mean_dslope_pp"],
            color=color,
            linestyle=linestyle,
            linewidth=2.2,
            marker="o",
            label=f"Официальная КБД, {regime_labels[regime_id]}",
        )
        axes[1].plot(
            model_regime["horizon_days"],
            model_regime["mean_dslope_pp"],
            color=color,
            linestyle=":",
            linewidth=2.0,
            marker="s",
            label=f"Авторская кривая, {regime_labels[regime_id]}",
        )
    axes[1].axhline(0.0, color="#444444", linewidth=1.0, linestyle=":")
    axes[1].set_xticks(list(EVENT_HORIZONS_DAYS))
    axes[1].set_xlabel("Горизонт, торговых дней")
    axes[1].set_ylabel("п.п.")
    axes[1].set_title("Среднее изменение наклона 10Y-1Y")
    axes[1].legend(
        frameon=False,
        fontsize=7.6,
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        columnspacing=1.2,
        handlelength=2.3,
        handletextpad=0.6,
    )
    axes[1].grid(alpha=0.25, linestyle=":")

    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(figure_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return figure_path


def build_key_rate_event_findings_markdown(
    meetings: pd.DataFrame,
    summary_df: pd.DataFrame,
    figure_path: Path,
    output_dir: Path,
) -> str:
    def pick(source_name: str, regime_id: str, horizon_days: int, decision_type: str = "all") -> pd.Series:
        mask = (
            (summary_df["source_name"] == source_name)
            & (summary_df["regime_id"] == regime_id)
            & (summary_df["horizon_days"] == horizon_days)
            & (summary_df["decision_type"] == decision_type)
        )
        return summary_df.loc[mask].iloc[0]

    official_pre_h1 = pick("official", "pre_2022_02", 1)
    official_transition_h1 = pick("official", "transition_2022_2023", 1)
    official_stabilization_h1 = pick("official", "stabilization_2024_plus", 1)
    official_stabilization_h3 = pick("official", "stabilization_2024_plus", 3)
    model_stabilization_h3 = pick("model", "stabilization_2024_plus", 3)
    official_keep_h3 = pick("official", "stabilization_2024_plus", 3, decision_type="keep")

    def fmt(value: float | None) -> str:
        if value is None or pd.isna(value):
            return "н/д"
        return f"{float(value):.2f}"

    def fmt_p(value: float | None) -> str:
        if value is None or pd.isna(value):
            return "н/д"
        if float(value) < 0.001:
            return "< 0.001"
        return f"= {float(value):.3f}"

    regular_meetings = meetings.loc[meetings["is_regular_meeting"]].copy()
    lines = [
        "# Поток 4в. Реакция кривой ОФЗ вокруг решений по ставке",
        "",
        "## Что сделано",
        "- Собран официальный календарь регулярных решений Банка России по ключевой ставке за 2019--2026 годы.",
        "- Для каждого заседания измерено изменение 1-летней, 10-летней доходности и наклона `10Y-1Y` на горизонтах 1, 3 и 5 торговых дней.",
        "- Внеочередные решения 2022 года сохранены в таблице событий, но исключены из основного сравнения, чтобы кризисные эпизоды не смешивались с обычным каналом передачи сигнала.",
        "",
        "## Главный результат",
        f"- До февраля 2022 года среднее изменение наклона `10Y-1Y` по официальной КБД через 1 торговый день было близко к нулю: `{fmt(official_pre_h1['mean_dslope_pp'])}` п.п.",
        f"- В переходном режиме 2022--2023 годов устойчивой закономерности не видно: через 1 день средний сдвиг наклона составляет `{fmt(official_transition_h1['mean_dslope_pp'])}` п.п.",
        f"- В режиме 2024--2026 годов картина меняется: через 1 торговый день 1-летняя доходность в среднем меняется на `{fmt(official_stabilization_h1['mean_dy1_pp'])}` п.п., 10-летняя --- на `{fmt(official_stabilization_h1['mean_dy10_pp'])}` п.п., а наклон `10Y-1Y` увеличивается на `{fmt(official_stabilization_h1['mean_dslope_pp'])}` п.п. при одностороннем `p {fmt_p(official_stabilization_h1['p_value_slope_positive'])}`.",
        f"- На горизонте 3 дней средний прирост наклона по официальной КБД сохраняется на уровне `{fmt(official_stabilization_h3['mean_dslope_pp'])}` п.п., а по авторской кривой --- `{fmt(model_stabilization_h3['mean_dslope_pp'])}` п.п.",
        f"- Наиболее заметный описательный сдвиг в режиме 2024--2026 приходится на заседания с сохранением ставки: через 3 дня наклон по официальной КБД растет в среднем на `{fmt(official_keep_h3['mean_dslope_pp'])}` п.п.",
        "",
        "## Как это читать",
        "- В 2024--2026 годах короткий конец чаще реагирует быстрее и слабее тянет вниз длинный конец, чем это было в старой конфигурации рынка.",
        "- Из-за этого вокруг обычных решений по ставке наклон `10Y-1Y` уже не сглаживается, а, наоборот, чаще растет.",
        "- Такой результат согласуется с идеей, что в длинном конце усилилась роль дополнительного бюджетно-долгового и риск-премиального давления.",
        "",
        "## Покрытие событий",
        f"- Всего регулярных заседаний в основном сравнении: `{int(regular_meetings.shape[0])}`.",
        f"- Переходный режим 2022--2023: `{int(regular_meetings.loc[regular_meetings['event_date'].between('2022-02-01', '2023-12-31')].shape[0])}` заседаний.",
        f"- Устойчивый режим 2024--2026: `{int(regular_meetings.loc[regular_meetings['event_date'] >= pd.Timestamp('2024-01-01')].shape[0])}` заседаний, из которых в реакциях `h=1..5` участвует `{int(summary_df.loc[(summary_df['source_name'] == 'official') & (summary_df['regime_id'] == 'stabilization_2024_plus') & (summary_df['decision_type'] == 'all') & (summary_df['horizon_days'] == 1), 'n_events'].iloc[0])}`.",
        "",
        "## Артефакты",
        f"- Таблица событий: `{output_dir / 'key_rate_meeting_events.csv'}`",
        f"- Таблица реакций: `{output_dir / 'key_rate_event_reactions.csv'}`",
        f"- Сводка по режимам и типам решений: `{output_dir / 'key_rate_event_summary.csv'}`",
        f"- Рисунок: `{figure_path}`",
    ]
    return "\n".join(lines)


def run_key_rate_event_study(
    *,
    official_curve_path: Path = DEFAULT_OFFICIAL_CURVE_PATH,
    model_factors_path: Path = DEFAULT_MODEL_FACTORS_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    findings_path: Path = DEFAULT_FINDINGS_PATH,
    figure_path: Path = DEFAULT_FIGURE_PATH,
    calendar_url: str = CBR_KEY_RATE_CALENDAR_URL,
    session: requests.Session | None = None,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    retries: int = DEFAULT_RETRIES,
    calendar_html: str | None = None,
    title_loader: Callable[[str], str] | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    findings_path.parent.mkdir(parents=True, exist_ok=True)

    official_curve = pd.read_csv(official_curve_path, parse_dates=[DATE_COLUMN]).sort_values(DATE_COLUMN).reset_index(drop=True)
    analysis_end = official_curve[DATE_COLUMN].max().normalize()

    if calendar_html is None:
        meetings = load_key_rate_meeting_events(
            calendar_url=calendar_url,
            session=session,
            timeout=timeout,
            retries=retries,
            analysis_end_date=analysis_end,
            analysis_start_date=DEFAULT_EVENT_START_DATE,
            title_loader=title_loader,
        )
    else:
        meetings = parse_key_rate_calendar(calendar_html, calendar_url=calendar_url)
        meetings = meetings.loc[
            (meetings["event_date"] >= DEFAULT_EVENT_START_DATE)
            & (meetings["event_date"] <= analysis_end)
        ].copy().reset_index(drop=True)
        meetings["press_release_title"] = meetings["press_release_url"].map(
            lambda url: _normalize_text(title_loader(url)) if title_loader is not None and isinstance(url, str) else None
        )
        meetings["decision_type"] = meetings["press_release_title"].map(classify_decision_from_title)
        meetings["resulting_key_rate_pct"] = meetings["press_release_title"].map(_extract_resulting_rate)
        meetings["has_press_release"] = meetings["press_release_url"].notna()
        meetings["is_regular_meeting"] = ~meetings["is_unscheduled"]

    reactions = build_key_rate_event_reactions(
        meetings,
        official_curve_path=official_curve_path,
        model_factors_path=model_factors_path,
        regular_only=True,
        horizons=EVENT_HORIZONS_DAYS,
    )
    summary_df = summarize_key_rate_event_reactions(reactions)
    figure_path = plot_key_rate_event_summary(summary_df, figure_path)

    events_output = output_dir / "key_rate_meeting_events.csv"
    reactions_output = output_dir / "key_rate_event_reactions.csv"
    summary_output = output_dir / "key_rate_event_summary.csv"

    meetings.to_csv(events_output, index=False)
    reactions.to_csv(reactions_output, index=False)
    summary_df.to_csv(summary_output, index=False)

    findings_text = build_key_rate_event_findings_markdown(
        meetings=meetings,
        summary_df=summary_df,
        figure_path=figure_path,
        output_dir=output_dir,
    )
    findings_path.write_text(findings_text, encoding="utf-8")

    metadata = {
        "coverage": {
            "events_start": meetings["event_date"].min().strftime("%Y-%m-%d"),
            "events_end": meetings["event_date"].max().strftime("%Y-%m-%d"),
            "regular_meetings": int(meetings["is_regular_meeting"].sum()),
            "unscheduled_meetings": int(meetings["is_unscheduled"].sum()),
            "official_curve_end": analysis_end.strftime("%Y-%m-%d"),
        },
        "artifacts": {
            "events_csv": str(events_output),
            "reactions_csv": str(reactions_output),
            "summary_csv": str(summary_output),
            "figure": str(figure_path),
            "findings_markdown": str(findings_path),
        },
    }
    metadata_path = output_dir / "key_rate_event_metadata.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    metadata["artifacts"]["metadata_json"] = str(metadata_path)

    return {
        "meetings": meetings,
        "reactions": reactions,
        "summary": summary_df,
        "metadata": metadata,
        "findings_text": findings_text,
    }
