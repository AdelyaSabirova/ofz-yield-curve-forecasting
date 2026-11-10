from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
import time
from typing import Any

from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
from nelson_siegel_svensson import NelsonSiegelCurve
import pandas as pd
import requests
from scipy.stats import ttest_1samp

from .paths import (
    CBR_ZERO_CURVE_DAILY_CSV,
    CORE_DAILY_NS_FACTORS_CSV,
    OFZ_AUCTION_STUDY_DIR,
    OFZ_AUCTION_STUDY_FIGURE,
    PROJECT_ROOT,
)


TELEGRAM_MINFIN_CHANNEL_URL = "https://t.me/s/minfin"
DEFAULT_TIMEOUT_SECONDS = 30
DEFAULT_RETRIES = 3
DEFAULT_RETRY_SLEEP_SECONDS = 1.5
EVENT_HORIZONS_DAYS = (1, 3, 5)
DEFAULT_ANALYSIS_START_DATE = pd.Timestamp("2022-02-01")
DEFAULT_CRAWL_START_DATE = pd.Timestamp("2022-01-01")
DEFAULT_MAX_PAGES = 420

DEFAULT_OFFICIAL_CURVE_PATH = CBR_ZERO_CURVE_DAILY_CSV
DEFAULT_MODEL_FACTORS_PATH = CORE_DAILY_NS_FACTORS_CSV
DEFAULT_OUTPUT_DIR = OFZ_AUCTION_STUDY_DIR
DEFAULT_FINDINGS_PATH = PROJECT_ROOT / "docs" / "p4_ofz_auction_findings.md"
DEFAULT_FIGURE_PATH = OFZ_AUCTION_STUDY_FIGURE

DATE_COLUMN = "date"

REGIME_SPECS = (
    ("transition_2022_2023", "Переходный режим 2022--2023", pd.Timestamp("2022-02-01"), pd.Timestamp("2023-12-31")),
    ("stabilization_2024_plus", "Устойчивый режим 2024--2026", pd.Timestamp("2024-01-01"), None),
)

VOLUME_BUCKETS = (
    ("all", "Все аукционы"),
    ("low", "Низкий объем размещения"),
    ("high", "Высокий объем размещения"),
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
    headers = {"User-Agent": "nir-adelya-ofz-auction-study/1.0"}

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


def _parse_russian_number(value: str | None) -> float | None:
    if value is None:
        return None
    cleaned = value.replace(" ", "").replace("\u202f", "").replace(",", ".")
    if not cleaned:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def _parse_amount_bln(match: re.Match[str] | None) -> float | None:
    if match is None:
        return None
    value = _parse_russian_number(match.group(1))
    if value is None:
        return None
    unit = match.group(2).lower()
    if unit.startswith("млн"):
        return value / 1000.0
    return value


def _extract_post_id(post_url: str | None) -> int | None:
    if not isinstance(post_url, str):
        return None
    match = re.search(r"/(\d+)(?:\?|$)", post_url)
    if not match:
        return None
    return int(match.group(1))


def _extract_message_datetime(message: BeautifulSoup) -> pd.Timestamp | None:
    time_tag = message.select_one("time[datetime]")
    if time_tag is not None and time_tag.has_attr("datetime"):
        return pd.Timestamp(time_tag["datetime"]).tz_convert(None).normalize()
    date_anchor = message.select_one("a.tgme_widget_message_date")
    if date_anchor is not None and date_anchor.has_attr("data-content"):
        stamp = pd.to_datetime(date_anchor["data-content"], errors="coerce")
        if not pd.isna(stamp):
            return pd.Timestamp(stamp).normalize()
    return None


def _extract_message_text(message: BeautifulSoup) -> str:
    text_block = message.select_one("div.tgme_widget_message_text")
    if text_block is not None:
        return _normalize_text(text_block.get_text(" ", strip=True))
    return _normalize_text(message.get_text(" ", strip=True))


def _parse_auction_result_text(message_text: str) -> dict[str, Any] | None:
    normalized = _normalize_text(message_text)
    lowered = _normalize_text_lower(normalized)
    if "минфин россии на аукцион" not in lowered:
        return None
    if "разместил" not in lowered or "разместит" in lowered or "офз" not in lowered:
        return None

    total_nominal_match = re.search(
        r"на общую сумму\s+([\d\s,]+)\s*(млрд|млн)\s*руб",
        lowered,
    )
    total_nominal_bln = _parse_amount_bln(total_nominal_match)

    drpa_match = re.search(
        r"размещенный объем выпуска\s*[--]\s*([\d\s,]+)\s*(млрд|млн)\.?\s*руб",
        lowered,
    )
    drpa_placed_bln = _parse_amount_bln(drpa_match)

    issue_numbers = sorted(set(re.findall(r"№\s*([0-9]{5,})", normalized)))
    revenue_matches = re.finditer(
        r"выручка(?: от аукциона| от размещения)?\s*[--]\s*([\d\s,]+)\s*(млрд|млн)\s*руб",
        lowered,
    )
    revenue_values = [value for value in (_parse_amount_bln(match) for match in revenue_matches) if value is not None]

    additional_placement = "дополнительного размещения" in lowered or "дрпа" in lowered
    total_with_drpa_bln = None
    if total_nominal_bln is not None:
        total_with_drpa_bln = total_nominal_bln + (drpa_placed_bln or 0.0)

    return {
        "total_nominal_bln": total_nominal_bln,
        "drpa_placed_bln": drpa_placed_bln,
        "total_with_drpa_bln": total_with_drpa_bln,
        "issue_count": len(issue_numbers),
        "issue_numbers": ",".join(issue_numbers),
        "revenue_total_bln": float(sum(revenue_values)) if revenue_values else None,
        "has_additional_placement": bool(additional_placement),
        "raw_message_text": normalized,
    }


def load_ofz_auction_events(
    *,
    channel_url: str = TELEGRAM_MINFIN_CHANNEL_URL,
    session: requests.Session | None = None,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    retries: int = DEFAULT_RETRIES,
    start_date: pd.Timestamp | str | None = DEFAULT_CRAWL_START_DATE,
    end_date: pd.Timestamp | str | None = None,
    max_pages: int = DEFAULT_MAX_PAGES,
) -> pd.DataFrame:
    local_session = session or requests.Session()
    start_stamp = pd.Timestamp(start_date).normalize() if start_date is not None else None
    end_stamp = pd.Timestamp(end_date).normalize() if end_date is not None else None

    rows: list[dict[str, Any]] = []
    before_id: int | None = None
    seen_post_ids: set[int] = set()

    for _page in range(max_pages):
        page_url = channel_url if before_id is None else f"{channel_url}?before={before_id}"
        html = _fetch_text(page_url, session=local_session, timeout=timeout, retries=retries)
        soup = BeautifulSoup(html, "html.parser")
        messages = soup.select("div.tgme_widget_message_wrap")
        if not messages:
            break

        page_rows: list[dict[str, Any]] = []
        oldest_post_id: int | None = None
        oldest_message_date: pd.Timestamp | None = None

        for message in messages:
            date_anchor = message.select_one("a.tgme_widget_message_date[href]")
            if date_anchor is None:
                continue
            post_url = date_anchor["href"]
            post_id = _extract_post_id(post_url)
            if post_id is None or post_id in seen_post_ids:
                continue
            message_date = _extract_message_datetime(message)
            if message_date is None:
                continue

            seen_post_ids.add(post_id)
            oldest_post_id = post_id if oldest_post_id is None else min(oldest_post_id, post_id)
            oldest_message_date = (
                message_date if oldest_message_date is None else min(oldest_message_date, message_date)
            )

            if end_stamp is not None and message_date > end_stamp:
                continue

            message_text = _extract_message_text(message)
            parsed = _parse_auction_result_text(message_text)
            if parsed is None:
                continue
            if start_stamp is not None and message_date < start_stamp:
                continue

            page_rows.append(
                {
                    "event_date": message_date,
                    "post_url": post_url,
                    "post_id": post_id,
                    **parsed,
                }
            )

        rows.extend(page_rows)
        if oldest_post_id is None:
            break
        if before_id == oldest_post_id:
            break
        if start_stamp is not None and oldest_message_date is not None and oldest_message_date < start_stamp:
            break

        before_id = oldest_post_id
        time.sleep(0.05)

    events = pd.DataFrame(rows)
    if events.empty:
        return pd.DataFrame(
            columns=[
                "event_date",
                "post_url",
                "post_id",
                "total_nominal_bln",
                "drpa_placed_bln",
                "total_with_drpa_bln",
                "issue_count",
                "issue_numbers",
                "revenue_total_bln",
                "has_additional_placement",
                "raw_message_text",
            ]
        )

    events = events.drop_duplicates(subset=["event_date", "post_id"]).sort_values("event_date").reset_index(drop=True)
    return events


def _assign_regime(event_date: pd.Timestamp) -> tuple[str, str] | tuple[None, None]:
    for regime_id, regime_label, start_date, end_date in REGIME_SPECS:
        if event_date < start_date:
            continue
        if end_date is not None and event_date > end_date:
            continue
        return regime_id, regime_label
    return None, None


def _assign_volume_buckets(events: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    events = events.copy()
    medians: dict[str, float] = {}
    events["volume_bucket"] = "all"
    events["volume_label"] = "Все аукционы"

    for regime_id, regime_label, _, _ in REGIME_SPECS:
        mask = events["regime_id"] == regime_id
        if not mask.any():
            continue
        current = events.loc[mask, "total_with_drpa_bln"].dropna()
        if current.empty:
            continue
        median_value = float(current.median())
        medians[regime_id] = median_value
        high_mask = mask & (events["total_with_drpa_bln"] >= median_value)
        low_mask = mask & (events["total_with_drpa_bln"] < median_value)
        events.loc[high_mask, "volume_bucket"] = "high"
        events.loc[high_mask, "volume_label"] = "Высокий объем размещения"
        events.loc[low_mask, "volume_bucket"] = "low"
        events.loc[low_mask, "volume_label"] = "Низкий объем размещения"

    return events, medians


def aggregate_monthly_ofz_auctions(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()
    monthly = events.copy()
    monthly[DATE_COLUMN] = monthly["event_date"].dt.to_period("M").dt.to_timestamp()

    def _sum_with_nan(series: pd.Series) -> float | None:
        value = series.sum(min_count=1)
        if pd.isna(value):
            return None
        return float(value)

    grouped = (
        monthly.groupby(DATE_COLUMN, as_index=False)
        .agg(
            auction_count=("post_id", "nunique"),
            total_nominal_bln=("total_nominal_bln", _sum_with_nan),
            total_with_drpa_bln=("total_with_drpa_bln", _sum_with_nan),
            total_drpa_bln=("drpa_placed_bln", _sum_with_nan),
            mean_nominal_bln=("total_nominal_bln", "mean"),
            mean_with_drpa_bln=("total_with_drpa_bln", "mean"),
            max_nominal_bln=("total_with_drpa_bln", "max"),
            additional_placement_count=("has_additional_placement", "sum"),
        )
        .sort_values(DATE_COLUMN)
        .reset_index(drop=True)
    )
    return grouped


def _build_model_curve_daily(model_factors_path: Path) -> pd.DataFrame:
    core = pd.read_csv(model_factors_path, parse_dates=[DATE_COLUMN]).sort_values(DATE_COLUMN).reset_index(drop=True)

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


def build_ofz_auction_event_reactions(
    auction_events: pd.DataFrame,
    *,
    official_curve_path: Path = DEFAULT_OFFICIAL_CURVE_PATH,
    model_factors_path: Path = DEFAULT_MODEL_FACTORS_PATH,
    horizons: tuple[int, ...] = EVENT_HORIZONS_DAYS,
) -> pd.DataFrame:
    if auction_events.empty:
        return pd.DataFrame()

    official = _load_official_curve_daily(official_curve_path)
    model = _build_model_curve_daily(model_factors_path)
    curves = pd.concat([official, model], ignore_index=True)

    events = auction_events.copy().sort_values("event_date").reset_index(drop=True)
    events[["regime_id", "regime_label"]] = events["event_date"].apply(lambda value: pd.Series(_assign_regime(value)))
    events = events.dropna(subset=["regime_id"]).reset_index(drop=True)
    events, regime_medians = _assign_volume_buckets(events)

    analysis_end = curves[DATE_COLUMN].max()
    events = events.loc[events["event_date"] <= analysis_end].copy()
    rows: list[dict[str, Any]] = []

    for curve_spec in CURVE_SPECS:
        curve = curves.loc[curves["source_name"] == curve_spec.name].sort_values(DATE_COLUMN).reset_index(drop=True)
        date_series = curve[DATE_COLUMN]
        for event in events.itertuples(index=False):
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
                        "regime_id": event.regime_id,
                        "regime_label": event.regime_label,
                        "volume_bucket": event.volume_bucket,
                        "volume_label": event.volume_label,
                        "post_id": int(event.post_id),
                        "post_url": event.post_url,
                        "total_nominal_bln": float(event.total_nominal_bln) if pd.notna(event.total_nominal_bln) else None,
                        "drpa_placed_bln": float(event.drpa_placed_bln) if pd.notna(event.drpa_placed_bln) else None,
                        "total_with_drpa_bln": (
                            float(event.total_with_drpa_bln) if pd.notna(event.total_with_drpa_bln) else None
                        ),
                        "regime_median_with_drpa_bln": regime_medians.get(event.regime_id),
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


def _safe_one_sided_ttest(sample: pd.Series, *, alternative: str) -> float | None:
    cleaned = sample.dropna()
    if cleaned.shape[0] < 2:
        return None
    return float(ttest_1samp(cleaned, popmean=0.0, alternative=alternative).pvalue)


def summarize_ofz_auction_event_reactions(reactions: pd.DataFrame) -> pd.DataFrame:
    if reactions.empty:
        return pd.DataFrame()

    summary_rows: list[dict[str, Any]] = []
    for curve_spec in CURVE_SPECS:
        current_source = reactions.loc[reactions["source_name"] == curve_spec.name].copy()
        if current_source.empty:
            continue
        for regime_id, regime_label, _, _ in REGIME_SPECS:
            current_regime = current_source.loc[current_source["regime_id"] == regime_id].copy()
            if current_regime.empty:
                continue
            for horizon in EVENT_HORIZONS_DAYS:
                current_horizon = current_regime.loc[current_regime["horizon_days"] == horizon].copy()
                if current_horizon.empty:
                    continue
                for volume_bucket, volume_label in VOLUME_BUCKETS:
                    sample = current_horizon if volume_bucket == "all" else current_horizon.loc[
                        current_horizon["volume_bucket"] == volume_bucket
                    ]
                    if sample.empty:
                        continue
                    p_slope = _safe_one_sided_ttest(sample["dslope_pp"], alternative="greater")
                    summary_rows.append(
                        {
                            "source_name": curve_spec.name,
                            "source_label": curve_spec.label,
                            "regime_id": regime_id,
                            "regime_label": regime_label,
                            "volume_bucket": volume_bucket,
                            "volume_label": volume_label,
                            "horizon_days": int(horizon),
                            "n_events": int(sample["event_date"].nunique()),
                            "sample_start": sample["event_date"].min().strftime("%Y-%m-%d"),
                            "sample_end": sample["event_date"].max().strftime("%Y-%m-%d"),
                            "mean_total_with_drpa_bln": float(sample["total_with_drpa_bln"].mean()),
                            "mean_dy1_pp": float(sample["dy1_pp"].mean()),
                            "mean_dy10_pp": float(sample["dy10_pp"].mean()),
                            "mean_dslope_pp": float(sample["dslope_pp"].mean()),
                            "median_dslope_pp": float(sample["dslope_pp"].median()),
                            "share_slope_positive": float((sample["dslope_pp"] > 0).mean()),
                            "p_value_slope_positive": p_slope,
                            "significant_slope_positive_5pct": bool(p_slope is not None and p_slope < 0.05),
                        }
                    )
    return pd.DataFrame(summary_rows)


def plot_ofz_auction_summary(summary_df: pd.DataFrame, figure_path: Path) -> Path:
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    official_h5 = summary_df.loc[
        (summary_df["source_name"] == "official")
        & (summary_df["horizon_days"] == 5)
        & (summary_df["volume_bucket"].isin(["low", "high"]))
    ].copy()
    official_h5 = official_h5.set_index(["regime_id", "volume_bucket"])

    model_high = summary_df.loc[
        (summary_df["volume_bucket"] == "high")
        & (summary_df["regime_id"].isin(["transition_2022_2023", "stabilization_2024_plus"]))
        & (summary_df["source_name"].isin(["official", "model"]))
    ].copy()

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.6))

    regimes = ["transition_2022_2023", "stabilization_2024_plus"]
    regime_labels = {"transition_2022_2023": "2022--2023", "stabilization_2024_plus": "2024--2026"}
    x_positions = range(len(regimes))
    bar_width = 0.34
    low_values = [official_h5.loc[(regime_id, "low"), "mean_dslope_pp"] for regime_id in regimes]
    high_values = [official_h5.loc[(regime_id, "high"), "mean_dslope_pp"] for regime_id in regimes]

    axes[0].bar([x - bar_width / 2 for x in x_positions], low_values, width=bar_width, color="#9db4c0", label="Низкий объем")
    axes[0].bar([x + bar_width / 2 for x in x_positions], high_values, width=bar_width, color="#c65d4b", label="Высокий объем")
    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[0].set_xticks(list(x_positions))
    axes[0].set_xticklabels([regime_labels[item] for item in regimes], fontsize=10)
    axes[0].set_ylabel("Изменение наклона 10Y-1Y, п.п.")
    axes[0].set_title("Официальная КБД, горизонт 5 дней")
    axes[0].legend(frameon=False, fontsize=9)
    axes[0].grid(axis="y", alpha=0.25)

    for source_name, source_label, color in (
        ("official", "Официальная КБД", "#1d3557"),
        ("model", "Авторская кривая", "#457b9d"),
    ):
        for regime_id, line_style in (("transition_2022_2023", "--"), ("stabilization_2024_plus", "-")):
            subset = model_high.loc[
                (model_high["source_name"] == source_name) & (model_high["regime_id"] == regime_id)
            ].sort_values("horizon_days")
            axes[1].plot(
                subset["horizon_days"],
                subset["mean_dslope_pp"],
                marker="o",
                linestyle=line_style,
                linewidth=2,
                color=color,
                label=f"{source_label}, {regime_labels[regime_id]}",
            )
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_xticks(list(EVENT_HORIZONS_DAYS))
    axes[1].set_xlabel("Горизонт, торговые дни")
    axes[1].set_ylabel("Изменение наклона 10Y-1Y, п.п.")
    axes[1].set_title("Крупные аукционы")
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].grid(axis="y", alpha=0.25)

    fig.suptitle("Реакция кривой ОФЗ вокруг результатов аукционов Минфина")
    fig.tight_layout()
    fig.savefig(figure_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return figure_path


def _build_findings_markdown(
    auction_events: pd.DataFrame,
    summary_df: pd.DataFrame,
    *,
    findings_path: Path,
) -> Path:
    findings_path.parent.mkdir(parents=True, exist_ok=True)

    def get_row(source: str, regime: str, bucket: str, horizon: int) -> pd.Series:
        current = summary_df.loc[
            (summary_df["source_name"] == source)
            & (summary_df["regime_id"] == regime)
            & (summary_df["volume_bucket"] == bucket)
            & (summary_df["horizon_days"] == horizon)
        ]
        if current.empty:
            raise KeyError((source, regime, bucket, horizon))
        return current.iloc[0]

    official_high_2024_h3 = get_row("official", "stabilization_2024_plus", "high", 3)
    official_high_2024_h5 = get_row("official", "stabilization_2024_plus", "high", 5)
    official_low_2024_h5 = get_row("official", "stabilization_2024_plus", "low", 5)
    official_high_2022_h5 = get_row("official", "transition_2022_2023", "high", 5)
    model_high_2024_h3 = get_row("model", "stabilization_2024_plus", "high", 3)

    text = "\n".join(
        [
            "# Поток 4г. Реакция кривой ОФЗ вокруг аукционов Минфина",
            "",
            f"- Источник событий: официальный канал Минфина России в Telegram, собраны результаты {auction_events.shape[0]} аукционов с {auction_events['event_date'].min().strftime('%Y-%m-%d')} по {auction_events['event_date'].max().strftime('%Y-%m-%d')}.",
            "- Крупные аукционы определялись отдельно внутри каждого режима как размещения не ниже медианного совокупного объема с учетом дополнительного размещения после аукциона.",
            f"- В режиме 2024--2026 по официальной КБД после крупных аукционов наклон 10Y-1Y растет в среднем на {official_high_2024_h3['mean_dslope_pp']:.2f} п.п. через три торговых дня и на {official_high_2024_h5['mean_dslope_pp']:.2f} п.п. через пять дней.",
            f"- После менее крупных аукционов в том же режиме среднее изменение наклона через пять торговых дней остается отрицательным и составляет {official_low_2024_h5['mean_dslope_pp']:.2f} п.п.",
            f"- В переходном режиме 2022--2023 даже крупные аукционы не давали такого эффекта: среднее изменение наклона по официальной КБД на горизонте пяти дней составляло {official_high_2022_h5['mean_dslope_pp']:.2f} п.п.",
            f"- По авторской кривой в 2024--2026 после крупных аукционов средний прирост наклона на горизонте трех дней достигает {model_high_2024_h3['mean_dslope_pp']:.2f} п.п.",
            "- Интерпретация: в устойчивом режиме 2024--2026 долговое давление проявляется не мгновенно, а в течение нескольких торговых дней после крупных размещений, прежде всего через более слабую динамику длинного конца.",
        ]
    )
    findings_path.write_text(text + "\n", encoding="utf-8")
    return findings_path


def run_ofz_auction_event_study(
    *,
    channel_url: str = TELEGRAM_MINFIN_CHANNEL_URL,
    official_curve_path: Path = DEFAULT_OFFICIAL_CURVE_PATH,
    model_factors_path: Path = DEFAULT_MODEL_FACTORS_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    findings_path: Path = DEFAULT_FINDINGS_PATH,
    figure_path: Path = DEFAULT_FIGURE_PATH,
    analysis_start_date: pd.Timestamp | str = DEFAULT_ANALYSIS_START_DATE,
    crawl_start_date: pd.Timestamp | str = DEFAULT_CRAWL_START_DATE,
    max_pages: int = DEFAULT_MAX_PAGES,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    auction_events = load_ofz_auction_events(
        channel_url=channel_url,
        start_date=crawl_start_date,
        end_date=pd.read_csv(official_curve_path, parse_dates=[DATE_COLUMN])[DATE_COLUMN].max(),
        max_pages=max_pages,
    )
    if auction_events.empty:
        raise RuntimeError("Не удалось извлечь аукционные события ОФЗ из официального канала Минфина.")

    analysis_start_stamp = pd.Timestamp(analysis_start_date).normalize()
    auction_events = auction_events.loc[auction_events["event_date"] >= analysis_start_stamp].reset_index(drop=True)
    auction_events[["regime_id", "regime_label"]] = auction_events["event_date"].apply(
        lambda value: pd.Series(_assign_regime(value))
    )
    auction_events = auction_events.dropna(subset=["regime_id"]).reset_index(drop=True)
    auction_events, regime_medians = _assign_volume_buckets(auction_events)

    monthly = aggregate_monthly_ofz_auctions(auction_events)
    reactions = build_ofz_auction_event_reactions(
        auction_events=auction_events,
        official_curve_path=official_curve_path,
        model_factors_path=model_factors_path,
    )
    summary = summarize_ofz_auction_event_reactions(reactions)
    if summary.empty:
        raise RuntimeError("Сводка по реакции кривой вокруг аукционов оказалась пустой.")

    figure_path = plot_ofz_auction_summary(summary, figure_path)
    findings_path = _build_findings_markdown(auction_events, summary, findings_path=findings_path)

    events_path = output_dir / "ofz_auction_events.csv"
    monthly_path = output_dir / "ofz_auction_monthly.csv"
    reactions_path = output_dir / "ofz_auction_event_reactions.csv"
    summary_path = output_dir / "ofz_auction_event_summary.csv"
    metadata_path = output_dir / "ofz_auction_metadata.json"

    auction_events.to_csv(events_path, index=False)
    monthly.to_csv(monthly_path, index=False)
    reactions.to_csv(reactions_path, index=False)
    summary.to_csv(summary_path, index=False)

    metadata = {
        "source": {"channel_url": channel_url, "type": "official_telegram_html"},
        "coverage": {
            "event_count": int(auction_events.shape[0]),
            "start_date": auction_events["event_date"].min().strftime("%Y-%m-%d"),
            "end_date": auction_events["event_date"].max().strftime("%Y-%m-%d"),
        },
        "volume_medians_bln": regime_medians,
        "artifacts": {
            "events_csv": str(events_path),
            "monthly_csv": str(monthly_path),
            "reactions_csv": str(reactions_path),
            "summary_csv": str(summary_path),
            "findings_md": str(findings_path),
            "figure_png": str(figure_path),
        },
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    return {
        "auction_events": auction_events,
        "monthly": monthly,
        "reactions": reactions,
        "summary": summary,
        "metadata": metadata,
        "figure_path": figure_path,
        "findings_path": findings_path,
    }
