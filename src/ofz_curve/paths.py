"""
Централизованные пути проекта.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"

DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_PROCESSED_DIR = DATA_DIR / "processed"
DATA_INTERIM_DIR = DATA_DIR / "interim"

OFZ_HISTORY_CSV = DATA_RAW_DIR / "market" / "ofz" / "ofz_full_history.csv"
CBR_ZERO_CURVE_DAILY_CSV = DATA_RAW_DIR / "market" / "cbr_zero_curve" / "cbr_zero_curve_daily.csv"
BRENT_XLSX = DATA_RAW_DIR / "macro" / "oil" / "DCOILBRENTEU.xlsx"
USD_XLSX = DATA_RAW_DIR / "macro" / "usd" / "RC_F01_01_2012_T06_12_2025.xlsx"
CBR_IPC_XLSX = DATA_RAW_DIR / "macro" / "cbr" / "ipc_mes_05-2025.xlsx"
CBR_INFLATION_KEY_RATE_XLSX = (
    DATA_RAW_DIR
    / "macro"
    / "cbr"
    / "Инфляция и ключевая ставка Банка России_F17_09_2013_T06_06_2025.xlsx"
)

DYNAMIC_TAU_PARAMS_CSV = DATA_PROCESSED_DIR / "dynamic_tau1_daily_params.csv"
NSS_FACTORS_CSV = DATA_PROCESSED_DIR / "nss_nls_factors_20250408_102558.csv"
RUSSIA_CPI_CSV = DATA_PROCESSED_DIR / "russia_cpi_data.csv"
CORE_DAILY_NS_FACTORS_CSV = DATA_PROCESSED_DIR / "core_2026" / "daily_ns_factors_core.csv"
FIXED_TAU1_FACTORS_CSV = DATA_PROCESSED_DIR / "core_2026" / "daily_ns_factors_fixed_tau1.csv"
CBR_ZERO_CURVE_MONTHLY_CSV = DATA_PROCESSED_DIR / "cbr_zero_curve" / "monthly_cbr_zero_curve.csv"
OFFICIAL_CURVE_VALIDATION_DIR = DATA_PROCESSED_DIR / "official_curve_validation"
LONG_END_RESIDUAL_DIR = DATA_PROCESSED_DIR / "long_end_residual"
POLICY_PATH_GAP_DIR = DATA_PROCESSED_DIR / "policy_path_gap"
KEY_RATE_EVENT_STUDY_DIR = DATA_PROCESSED_DIR / "key_rate_event_study"
OFZ_AUCTION_STUDY_DIR = DATA_PROCESSED_DIR / "ofz_auction_study"

ASSETS_DIR = PROJECT_ROOT / "assets"
LONG_END_RESIDUAL_FIGURE = ASSETS_DIR / "long_end_slope_residual.png"
POLICY_PATH_GAP_FIGURE = ASSETS_DIR / "policy_path_gap.png"
KEY_RATE_EVENT_STUDY_FIGURE = ASSETS_DIR / "key_rate_event_study.png"
OFZ_AUCTION_STUDY_FIGURE = ASSETS_DIR / "ofz_auction_study.png"
