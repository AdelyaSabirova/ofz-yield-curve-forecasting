#!/usr/bin/env python3
"""
Estimate a recursively identified structural VECM block and export IRF artifacts.
"""

from __future__ import annotations

from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.vector_ar.vecm import VECM


ROOT = Path(__file__).resolve().parents[1]
CORE_PATH = ROOT / "data" / "processed" / "core_2026" / "monthly_ns_factors_core.csv"
MACRO_PATH = ROOT / "data" / "processed" / "macro_2026" / "monthly_expectations_panel.csv"
OUT_DIR = ROOT / "data" / "processed" / "svar_outputs"
FIG_DIR = ROOT / "assets"

MODEL_ORDER = ["cpi_mom", "key_rate", "beta0", "beta1", "beta2"]
SHOCKS_TO_PLOT = ["cpi_mom", "key_rate", "beta0", "beta1"]
MACRO_SHOCKS_TO_PLOT = ["cpi_mom", "key_rate"]
CURVE_SHOCKS_TO_PLOT = ["beta0", "beta1"]
RESPONSES_TO_PLOT = ["cpi_mom", "key_rate", "beta0", "beta1", "beta2"]
HORIZONS = [1, 3, 6, 12]
MAX_HORIZON = 12
BREAK_DATE = pd.Timestamp("2022-02-01")

LABELS = {
    "cpi_mom": "инфляция м/м",
    "key_rate": "ключевая ставка",
    "beta0": "уровень кривой",
    "beta1": "наклон кривой",
    "beta2": "кривизна",
}

SHOCK_LABELS = {
    "cpi_mom": "инфляционный шок",
    "key_rate": "шок ключевой ставки",
    "beta0": "шок уровня кривой",
    "beta1": "шок наклона кривой",
    "beta2": "шок кривизны",
}


def load_panel() -> pd.DataFrame:
    core = pd.read_csv(CORE_PATH, parse_dates=["date"])
    macro = pd.read_csv(MACRO_PATH, parse_dates=["date"])
    merged = core.merge(macro, on="date", how="inner").sort_values("date")
    panel = merged[["date", "tau1", *MODEL_ORDER]].dropna().copy()
    outlier_mask = (panel["beta1"].abs() > 1.0) | (panel["beta2"].abs() > 1.0)
    panel = panel.loc[~outlier_mask].set_index("date").sort_index()
    return panel


def fit_vecm(frame: pd.DataFrame):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return VECM(
            frame[MODEL_ORDER],
            k_ar_diff=1,
            coint_rank=1,
            deterministic="co",
        ).fit()


def build_identification_table(result, frame: pd.DataFrame) -> pd.DataFrame:
    chol = np.linalg.cholesky(np.asarray(result.sigma_u, dtype=float))
    rows = []
    for idx, shock in enumerate(MODEL_ORDER):
        rows.append(
            {
                "order": idx + 1,
                "shock": shock,
                "shock_label": SHOCK_LABELS[shock],
                "contemporaneously_ordered_after": ", ".join(MODEL_ORDER[:idx]) or "none",
                "shock_size_pp": float(chol[idx, idx] * 100.0),
                "sample_start": frame.index.min().strftime("%Y-%m-%d"),
                "sample_end": frame.index.max().strftime("%Y-%m-%d"),
                "n_obs": int(len(frame)),
            }
        )
    return pd.DataFrame(rows)


def build_irf_table(result, frame: pd.DataFrame) -> pd.DataFrame:
    irf = result.irf(MAX_HORIZON)
    orth_irfs = np.asarray(irf.orth_irfs, dtype=float)
    rows = []
    for response_idx, response in enumerate(MODEL_ORDER):
        for shock_idx, shock in enumerate(MODEL_ORDER):
            path = orth_irfs[:, response_idx, shock_idx]
            peak_horizon = int(np.abs(path).argmax())
            for horizon in HORIZONS:
                rows.append(
                    {
                        "response": response,
                        "response_label": LABELS[response],
                        "shock": shock,
                        "shock_label": SHOCK_LABELS[shock],
                        "horizon": horizon,
                        "response_pp": float(path[horizon] * 100.0),
                        "cumulative_12m_pp": float(path[: MAX_HORIZON + 1].sum() * 100.0),
                        "peak_horizon": peak_horizon,
                        "peak_response_pp": float(path[peak_horizon] * 100.0),
                        "sample_start": frame.index.min().strftime("%Y-%m-%d"),
                        "sample_end": frame.index.max().strftime("%Y-%m-%d"),
                        "n_obs": int(len(frame)),
                    }
                )
    return pd.DataFrame(rows)


def plot_irf(result, output_path: Path, shocks: list[str], title: str) -> None:
    irf = result.irf(MAX_HORIZON)
    orth_irfs = np.asarray(irf.orth_irfs, dtype=float) * 100.0
    stderr = np.asarray(irf.stderr(orth=True), dtype=float) * 100.0
    lower95 = orth_irfs - 1.96 * stderr
    upper95 = orth_irfs + 1.96 * stderr
    lower68 = orth_irfs - 1.00 * stderr
    upper68 = orth_irfs + 1.00 * stderr
    x = np.arange(MAX_HORIZON + 1)
    fig, axes = plt.subplots(
        len(RESPONSES_TO_PLOT),
        len(shocks),
        figsize=(9, 12),
        sharex=True,
    )
    if len(shocks) == 1:
        axes = axes.reshape(-1, 1)
    for row_idx, response in enumerate(RESPONSES_TO_PLOT):
        response_idx = MODEL_ORDER.index(response)
        for col_idx, shock in enumerate(shocks):
            shock_idx = MODEL_ORDER.index(shock)
            ax = axes[row_idx, col_idx]
            y = orth_irfs[:, response_idx, shock_idx]
            ax.fill_between(
                x,
                lower95[:, response_idx, shock_idx],
                upper95[:, response_idx, shock_idx],
                color="#d8dee7",
                alpha=0.85,
                linewidth=0,
                label="95% интервал",
            )
            ax.fill_between(
                x,
                lower68[:, response_idx, shock_idx],
                upper68[:, response_idx, shock_idx],
                color="#9fb3c8",
                alpha=0.55,
                linewidth=0,
                label="68% интервал",
            )
            ax.plot(x, y, color="#0f4c81", linewidth=2.0, label="отклик")
            ax.axhline(0.0, color="#3f3f3f", linewidth=0.9, linestyle="--", alpha=0.9)
            ax.grid(alpha=0.25)
            if row_idx == 0:
                ax.set_title(SHOCK_LABELS[shock], fontsize=10)
            if col_idx == 0:
                ax.set_ylabel(f"{LABELS[response]}, п.п.", fontsize=9)
            if row_idx == len(RESPONSES_TO_PLOT) - 1:
                ax.set_xlabel("месяцы после шока")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=9,
        bbox_to_anchor=(0.5, 0.0),
    )
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=[0, 0.025, 1, 0.97])
    fig.savefig(output_path, dpi=240)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    panel = load_panel()
    result = fit_vecm(panel)

    identification = build_identification_table(result, panel)
    irf_table = build_irf_table(result, panel)

    identification.to_csv(OUT_DIR / "svar_identification.csv", index=False)
    irf_table.to_csv(OUT_DIR / "svar_irf_selected.csv", index=False)
    plot_irf(
        result,
        FIG_DIR / "svar_structural_irf_macro.png",
        MACRO_SHOCKS_TO_PLOT,
        "Структурные отклики на инфляционный шок и шок ключевой ставки",
    )
    plot_irf(
        result,
        FIG_DIR / "svar_structural_irf_curve.png",
        CURVE_SHOCKS_TO_PLOT,
        "Структурные отклики на шоки уровня и наклона кривой",
    )

    post_panel = panel.loc[panel.index >= BREAK_DATE]
    if len(post_panel) >= 24:
        post_result = fit_vecm(post_panel)
        build_irf_table(post_result, post_panel).to_csv(
            OUT_DIR / "svar_irf_selected_post_2022.csv",
            index=False,
        )

    print(
        {
            "n_obs": int(len(panel)),
            "sample_start": panel.index.min().strftime("%Y-%m-%d"),
            "sample_end": panel.index.max().strftime("%Y-%m-%d"),
            "macro_figure": str(FIG_DIR / "svar_structural_irf_macro.png"),
            "curve_figure": str(FIG_DIR / "svar_structural_irf_curve.png"),
        }
    )


if __name__ == "__main__":
    main()
