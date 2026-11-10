"""
Rebuild macro block artifacts for the thesis:
- monthly merged dataset (NSS factors + CPI),
- lag selection / Johansen / VECM diagnostics,
- Granger tests (levels + differences stability-check),
- FEVD fallback from VAR(levels),
- figures for Appendix B (image9, image10, image11).
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.vector_ar.vecm import VECM, coint_johansen


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ofz_curve.paths import (  # noqa: E402
    FIXED_TAU1_FACTORS_CSV,
    RUSSIA_CPI_CSV,
)


OUT_DIR = ROOT / "data" / "processed" / "vecm_outputs"
IMG_DIR = ROOT / "assets"
OUT_DIR.mkdir(parents=True, exist_ok=True)
IMG_DIR.mkdir(parents=True, exist_ok=True)


VARS = ["beta0", "beta1", "beta2", "cpi_mom"]


def _load_data() -> pd.DataFrame:
    nss = pd.read_csv(FIXED_TAU1_FACTORS_CSV)
    cpi = pd.read_csv(RUSSIA_CPI_CSV)

    nss["date"] = pd.to_datetime(nss["date"], errors="coerce")
    cpi["date"] = pd.to_datetime(cpi["date"], errors="coerce")
    nss = nss.dropna(subset=["date"]).set_index("date").sort_index()
    cpi = cpi.dropna(subset=["date"]).set_index("date").sort_index()

    # Normalize both to month start to avoid month-end/start merge artifacts.
    nss_ms = nss[["beta0", "beta1", "beta2"]].resample("MS").last()
    cpi_ms = cpi[["cpi_mom", "cpi_yoy"]].copy()
    cpi_ms.index = cpi_ms.index.to_period("M").to_timestamp(how="start")

    data = nss_ms.join(cpi_ms, how="inner").dropna()
    return data


def _adf_diagnostics(data: pd.DataFrame) -> pd.DataFrame:
    from statsmodels.tsa.stattools import adfuller

    rows: list[dict[str, float | str]] = []
    for col in VARS:
        res = adfuller(data[col].dropna(), autolag="AIC")
        rows.append(
            {
                "variable": col,
                "adf_stat": float(res[0]),
                "p_value": float(res[1]),
                "is_stationary_5pct": bool(res[1] <= 0.05),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "adf_diagnostics.csv", index=False)
    return out


def _granger_table(data: pd.DataFrame, lag: int, title: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    columns = list(data.columns)
    lag = max(1, min(int(lag), 4))
    for target in columns:
        for source in columns:
            if source == target:
                continue
            pair = data[[target, source]].dropna()
            result = grangercausalitytests(pair, maxlag=lag, verbose=False)
            p_val = float(result[lag][0]["ssr_ftest"][1])
            rows.append(
                {
                    "direction": f"{source} -> {target}",
                    "lag": lag,
                    "p_value": p_val,
                    "significant_5pct": p_val < 0.05,
                }
            )
    out = pd.DataFrame(rows).sort_values(["p_value", "direction"]).reset_index(drop=True)
    out.to_csv(OUT_DIR / f"granger_{title}.csv", index=False)
    return out


def _plot_image9(data: pd.DataFrame) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(14, 11))
    data["beta0"].plot(ax=axes[0, 0], title="beta0 (уровень)", color="#1f77b4")
    data["beta1"].plot(ax=axes[0, 1], title="beta1 (наклон)", color="#d62728")
    data["beta2"].plot(ax=axes[1, 0], title="beta2 (кривизна)", color="#2ca02c")
    data["cpi_mom"].plot(ax=axes[1, 1], title="Инфляция cpi_mom (м/м)", color="#ff7f0e")
    data["cpi_yoy"].plot(ax=axes[2, 0], title="Инфляция cpi_yoy (г/г)", color="#9467bd")

    corr = data[VARS].corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, ax=axes[2, 1], fmt=".2f")
    axes[2, 1].set_title("Корреляционная матрица")

    for ax in axes.flat:
        if ax is not axes[2, 1]:
            ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(IMG_DIR / "image9.png", dpi=220)
    plt.close(fig)


def _plot_image10(vecm_irf, columns: list[str]) -> None:
    periods = vecm_irf.irfs.shape[0]
    n = len(columns)
    fig, axes = plt.subplots(n, n, figsize=(15, 11), sharex=True)
    if n == 1:
        axes = np.array([[axes]])

    x = np.arange(periods)
    for i in range(n):
        for j in range(n):
            ax = axes[i, j]
            series = vecm_irf.irfs[:, i, j]
            ax.plot(x, series, color="#1f77b4", linewidth=1.8)
            ax.axhline(0.0, color="black", linestyle="--", linewidth=0.8, alpha=0.7)
            ax.set_title(f"{columns[j]} -> {columns[i]}", fontsize=9)
            ax.grid(alpha=0.2)
            if i == n - 1:
                ax.set_xlabel("месяцы")
            if j == 0:
                ax.set_ylabel("отклик")
    fig.suptitle("IRF на основе VECM (уровни, ортогонализированные шоки)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(IMG_DIR / "image10.png", dpi=220)
    plt.close(fig)


def _plot_image11(fevd_table: pd.DataFrame) -> None:
    horizons = sorted(fevd_table["horizon"].unique().tolist())
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.flatten()
    for idx, horizon in enumerate(horizons):
        ax = axes[idx]
        current = fevd_table[fevd_table["horizon"] == horizon]
        labels = current["shock"].tolist()
        values = current["share_pct"].to_numpy(dtype=float)
        ax.pie(values, labels=labels, autopct="%1.1f%%", startangle=90)
        ax.set_title(f"FEVD для cpi_mom, горизонт {horizon} мес.")
    fig.tight_layout()
    fig.savefig(IMG_DIR / "image11.png", dpi=220)
    plt.close(fig)


def main() -> None:
    warnings.filterwarnings("ignore", category=FutureWarning, module="statsmodels.tsa.stattools")
    data = _load_data()
    levels = data[VARS].dropna()
    levels.to_csv(OUT_DIR / "levels_dataset.csv")

    _adf_diagnostics(levels)

    # Lag selection in levels.
    var_select = VAR(levels)
    lag_res = var_select.select_order(maxlags=12)
    lag_table = lag_res.summary().data
    lag_df = pd.DataFrame(lag_table[1:], columns=lag_table[0])
    lag_df.to_csv(OUT_DIR / "lag_selection_levels.csv", index=False)
    p_levels = int(lag_res.aic) if lag_res.aic is not None else 1
    p_levels = max(1, p_levels)

    # Johansen rank.
    k_ar_diff = max(0, p_levels - 1)
    johansen = coint_johansen(levels, det_order=0, k_ar_diff=k_ar_diff)
    trace_df = pd.DataFrame(
        {
            "r_leq": np.arange(len(johansen.lr1)),
            "trace_stat": johansen.lr1,
            "crit_90": johansen.cvt[:, 0],
            "crit_95": johansen.cvt[:, 1],
            "crit_99": johansen.cvt[:, 2],
        }
    )
    maxeig_df = pd.DataFrame(
        {
            "r_eq": np.arange(len(johansen.lr2)),
            "maxeig_stat": johansen.lr2,
            "crit_90": johansen.cvm[:, 0],
            "crit_95": johansen.cvm[:, 1],
            "crit_99": johansen.cvm[:, 2],
        }
    )
    trace_df["reject_95"] = trace_df["trace_stat"] > trace_df["crit_95"]
    maxeig_df["reject_95"] = maxeig_df["maxeig_stat"] > maxeig_df["crit_95"]
    trace_df.to_csv(OUT_DIR / "johansen_trace.csv", index=False)
    maxeig_df.to_csv(OUT_DIR / "johansen_maxeig.csv", index=False)

    coint_rank = 0
    for _, row in trace_df.iterrows():
        if bool(row["reject_95"]):
            coint_rank = int(row["r_leq"]) + 1
        else:
            break
    coint_rank = max(1, min(coint_rank, len(VARS) - 1))

    # Fit VECM (primary model).
    vecm = VECM(levels, k_ar_diff=k_ar_diff, coint_rank=coint_rank, deterministic="co")
    vecm_res = vecm.fit()

    alpha_df = pd.DataFrame(vecm_res.alpha, index=VARS, columns=[f"ec{j+1}" for j in range(coint_rank)])
    beta_df = pd.DataFrame(vecm_res.beta, index=VARS, columns=[f"ec{j+1}" for j in range(coint_rank)])
    alpha_df.to_csv(OUT_DIR / "vecm_alpha.csv")
    beta_df.to_csv(OUT_DIR / "vecm_beta.csv")

    resid = pd.DataFrame(vecm_res.resid, columns=VARS, index=levels.index[-len(vecm_res.resid) :])
    lb_rows: list[dict[str, object]] = []
    for col in VARS:
        lb = acorr_ljungbox(resid[col], lags=[10], return_df=True)
        p_val = float(lb["lb_pvalue"].iloc[-1])
        lb_rows.append({"variable": col, "lb_pvalue_lag10": p_val, "no_autocorr_5pct": p_val > 0.05})
    pd.DataFrame(lb_rows).to_csv(OUT_DIR / "vecm_ljungbox.csv", index=False)

    # Granger: levels (main) and differences (stability check).
    _granger_table(levels, lag=p_levels, title="levels")
    diff = levels.diff().dropna()
    diff.columns = [f"d_{c}" for c in diff.columns]
    _granger_table(diff, lag=min(p_levels, 4), title="diff")

    # FEVD fallback from VAR(levels), same variables and lag.
    var_fit = VAR(levels).fit(maxlags=p_levels, trend="c")
    fevd = var_fit.fevd(24)
    cpi_idx = VARS.index("cpi_mom")
    fevd_rows: list[dict[str, object]] = []
    for horizon in [1, 6, 12, 24]:
        h_idx = horizon - 1
        for shock_idx, shock in enumerate(VARS):
            share = float(fevd.decomp[cpi_idx, h_idx, shock_idx] * 100.0)
            fevd_rows.append({"horizon": horizon, "response": "cpi_mom", "shock": shock, "share_pct": share})
    fevd_df = pd.DataFrame(fevd_rows)
    fevd_df.to_csv(OUT_DIR / "fevd_cpi_mom.csv", index=False)

    # Figures for Appendix B
    _plot_image9(data)
    vecm_irf = vecm_res.irf(24)
    _plot_image10(vecm_irf, VARS)
    _plot_image11(fevd_df)

    summary = {
        "n_levels_obs": int(len(levels)),
        "period_start": str(levels.index.min().date()),
        "period_end": str(levels.index.max().date()),
        "p_levels_aic": int(p_levels),
        "k_ar_diff": int(k_ar_diff),
        "coint_rank_trace_95": int(coint_rank),
        "vecm_det": "co",
        "fevd_source": "VAR(levels) fallback",
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
