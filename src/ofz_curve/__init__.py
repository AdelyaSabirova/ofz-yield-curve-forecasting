"""
Пакет функций для анализа кривых доходности ОФЗ
"""

from .config import *
from .data_loading import *
from .tau1_optimization import *
from .ns_calibration import *
from .forecasting import *
from .ml_models import *
from .visualization import *
from .analysis import *
from .cpi_loaders import *
from .cbr_zero_curve import *
from .long_end_residual import *
from .policy_path_gap import *
from .key_rate_event_study import *
from .ofz_auction_study import *
from .var_tests import *

__all__ = [
    # config
    'FILENAME', 'OIL_PATH', 'USD_PATH', 'CLIP_QUANTILE_LOW', 'CLIP_QUANTILE_HIGH',
    'MIN_DATA_POINTS_FOR_FIT', 'TAU0_INITIAL_GUESS', 'TAU1_SEARCH_BOUNDS',
    'TAU1_OPTIM_SAMPLE_DATES', 'TAU1_OPTIM_GRID_SIZE', 'TAU1_OPTIM_SEED',
    'VAR_MAX_LAGS', 'IRF_PERIODS', 'COLLINEARITY_THRESHOLD', 'TRADING_DAYS_PER_YEAR',
    'MATURITIES_FOR_PREMIUM', 'EVAL_MATURITIES_YEARS', 'FORECAST_HORIZONS_DAYS',
    'OUT_OF_SAMPLE_START_RATIO', 'FIGURE_SIZE', 'SUBPLOT_SIZE',

    # data_loading
    'load_ofz_data', 'load_oil_data', 'load_usd_data', 'prepare_factors_for_vecm',

    # tau1_optimization
    'optimize_tau1',

    # ns_calibration
    'calibrate_ns_for_tau1',

    # forecasting
    'forecast_factors_ar1', 'forecast_factors_var_level', 'forecast_factors_var_dl',
    'forecast_factors_vecm', 'analyze_irf',

    # ml_models
    'prepare_ml_features_for_factors', 'train_catboost_factor_models',
    'forecast_factors_catboost', 'compare_ml_vs_econometric', 'compare_ml_vs_econometric_all',

    # visualization
    'plot_rmse_comparison', 'plot_irf_yields', 'plot_yield_forecast_rmse',
    'plot_factor_forecasts_combined',

    # analysis
    'merge_factor_forecasts', 'calculate_yield_forecasts', 'calculate_rmse_summary', 'analyze_tau1_results',

    # cpi_loaders
    'get_cbr_inflation_data', 'load_monthly_inflation', 'load_annual_inflation',

    # cbr_zero_curve
    'download_cbr_zero_curve_history', 'ensure_cbr_zero_curve_daily',
    'build_monthly_cbr_zero_curve', 'build_curve_comparison',
    'summarize_curve_comparison', 'run_cbr_zero_curve_validation',

    # long_end_residual
    'load_long_end_panel', 'evaluate_long_end_residuals',
    'run_long_end_residual_analysis',

    # policy_path_gap
    'build_monthly_survey_panel', 'build_policy_path_gap_panel', 'summarize_policy_path_gap',
    'run_policy_path_gap_analysis',

    # key_rate_event_study
    'parse_key_rate_calendar', 'classify_decision_from_title',
    'load_key_rate_meeting_events', 'build_key_rate_event_reactions',
    'summarize_key_rate_event_reactions', 'run_key_rate_event_study',

    # ofz_auction_study
    'load_ofz_auction_events', 'aggregate_monthly_ofz_auctions',
    'build_ofz_auction_event_reactions', 'summarize_ofz_auction_event_reactions',
    'run_ofz_auction_event_study',

    # var_tests
    'adf_test', 'granger_causality_test', 'interpret_johansen_result'
]
