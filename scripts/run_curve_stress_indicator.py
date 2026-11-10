#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "processed" / "curve_stress_indicator"
FIG_PATH = ROOT / "assets" / "curve_stress_indicator.png"


def zscore(series: pd.Series, mask: pd.Series) -> pd.Series:
    base = series[mask & series.notna()]
    return (series - base.mean()) / base.std(ddof=0)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)

    gap = pd.read_csv(ROOT / "data" / "processed" / "policy_path_gap" / "policy_path_gap_panel.csv")
    residual = pd.read_csv(ROOT / "data" / "processed" / "long_end_residual" / "long_end_residual_series.csv")
    macro = pd.read_csv(ROOT / "data" / "processed" / "macro_2026" / "monthly_expectations_panel.csv")
    auctions = pd.read_csv(ROOT / "data" / "processed" / "ofz_auction_study" / "ofz_auction_monthly.csv")

    gap["date"] = pd.to_datetime(gap["date"])
    residual["date"] = pd.to_datetime(residual["date"])
    macro["date"] = pd.to_datetime(macro["date"])
    auctions["date"] = pd.to_datetime(auctions["date"])

    residual = residual[
        (residual["target_name"] == "official")
        & (residual["spec_name"] == "macro_only")
    ][["date", "residual_pp"]].rename(columns={"residual_pp": "long_slope_residual_pp"})

    auctions["auction_volume_bln"] = auctions["total_with_drpa_bln"].fillna(auctions["total_nominal_bln"])
    auctions = auctions[["date", "auction_volume_bln", "auction_count"]]

    panel = (
        gap[["date", "nominal_gap_2y_pp"]]
        .merge(residual, on="date", how="outer")
        .merge(macro[["date", "breakeven_5y", "ofz_nonresident_share", "general_government_debt_growth_12m"]], on="date", how="outer")
        .merge(auctions, on="date", how="outer")
        .sort_values("date")
    )
    panel["auction_volume_3m_bln"] = panel["auction_volume_bln"].fillna(0).rolling(3, min_periods=1).sum()

    post_2022 = panel["date"] >= pd.Timestamp("2022-02-01")
    panel["z_policy_gap"] = zscore(panel["nominal_gap_2y_pp"], post_2022)
    panel["z_long_residual"] = zscore(panel["long_slope_residual_pp"], post_2022)
    panel["z_breakeven_5y"] = zscore(panel["breakeven_5y"], post_2022)
    panel["z_debt_growth"] = zscore(panel["general_government_debt_growth_12m"], post_2022)
    panel["z_auction_pressure"] = zscore(panel["auction_volume_3m_bln"], post_2022)

    components = [
        "z_policy_gap",
        "z_long_residual",
        "z_breakeven_5y",
        "z_debt_growth",
        "z_auction_pressure",
    ]
    panel["curve_stress_index"] = panel[components].mean(axis=1, skipna=True)
    panel["stress_label"] = pd.cut(
        panel["curve_stress_index"],
        bins=[-np.inf, 0.0, 0.5, np.inf],
        labels=["ниже среднего фона", "умеренное напряжение", "повышенное напряжение"],
    )

    panel.to_csv(OUT_DIR / "curve_stress_indicator.csv", index=False)

    regimes = [
        ("pre_2022_02", "До февраля 2022 года", "2018-04-01", "2022-01-01"),
        ("transition_2022_2023", "Переходный режим 2022--2023", "2022-02-01", "2023-12-01"),
        ("stabilization_2024_plus", "Устойчивый режим 2024--2026", "2024-01-01", "2026-03-01"),
    ]
    rows = []
    for regime_id, label, start, end in regimes:
        sub = panel[(panel["date"] >= start) & (panel["date"] <= end)]
        rows.append(
            {
                "regime_id": regime_id,
                "regime_label": label,
                "sample_start": sub["date"].min().date() if not sub.empty else "",
                "sample_end": sub["date"].max().date() if not sub.empty else "",
                "n_obs": int(sub["curve_stress_index"].notna().sum()),
                "mean_index": sub["curve_stress_index"].mean(),
                "median_index": sub["curve_stress_index"].median(),
                "share_elevated_stress": (sub["curve_stress_index"] >= 0.5).mean(),
                "mean_policy_gap_pp": sub["nominal_gap_2y_pp"].mean(),
                "mean_long_residual_pp": sub["long_slope_residual_pp"].mean(),
                "mean_auction_3m_bln": sub["auction_volume_3m_bln"].mean(),
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT_DIR / "curve_stress_summary.csv", index=False)

    top_months = (
        panel[panel["date"] >= "2024-01-01"]
        .sort_values("curve_stress_index", ascending=False)
        .head(8)
        [[
            "date",
            "curve_stress_index",
            "nominal_gap_2y_pp",
            "long_slope_residual_pp",
            "breakeven_5y",
            "general_government_debt_growth_12m",
            "auction_volume_3m_bln",
        ]]
    )
    top_months.to_csv(OUT_DIR / "curve_stress_top_months.csv", index=False)

    fig, axes = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True, gridspec_kw={"height_ratios": [1.1, 1]})
    plot_panel = panel[panel["date"] >= "2021-05-01"].copy()
    axes[0].plot(plot_panel["date"], plot_panel["curve_stress_index"], color="#1f4e79", linewidth=2.2)
    axes[0].axhline(0.5, color="#b03a2e", linewidth=1, linestyle="--")
    axes[0].axhline(0.0, color="#d68910", linewidth=1, linestyle=":")
    axes[0].axvspan(pd.Timestamp("2022-02-01"), pd.Timestamp("2023-12-31"), color="#d6eaf8", alpha=0.35)
    axes[0].axvspan(pd.Timestamp("2024-01-01"), pd.Timestamp("2026-03-31"), color="#fdebd0", alpha=0.35)
    axes[0].set_ylabel("индекс, z")
    axes[0].set_title("Индикатор напряжения кривой ОФЗ")
    axes[0].grid(alpha=0.25)

    recent = panel[panel["date"] >= "2024-01-01"].copy()
    axes[1].plot(recent["date"], recent["z_policy_gap"], label="разрыв 2Y к ожиданиям ставки", linewidth=1.8)
    axes[1].plot(recent["date"], recent["z_long_residual"], label="остаточный наклон 10Y-1Y", linewidth=1.8)
    axes[1].plot(recent["date"], recent["z_auction_pressure"], label="аукционное давление, 3 мес.", linewidth=1.8)
    axes[1].plot(recent["date"], recent["z_debt_growth"], label="рост госдолга", linewidth=1.8)
    axes[1].set_ylabel("z")
    axes[1].set_title("Компоненты индикатора в устойчивом режиме 2024--2026")
    axes[1].grid(alpha=0.25)
    axes[1].legend(loc="upper left", ncols=2, fontsize=8)

    fig.tight_layout()
    fig.savefig(FIG_PATH, dpi=220)
    plt.close(fig)

    print(f"panel: {OUT_DIR / 'curve_stress_indicator.csv'}")
    print(f"summary: {OUT_DIR / 'curve_stress_summary.csv'}")
    print(f"top months: {OUT_DIR / 'curve_stress_top_months.csv'}")
    print(f"figure: {FIG_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
