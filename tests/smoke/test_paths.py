from ofz_curve.paths import (
    BRENT_XLSX,
    CBR_INFLATION_KEY_RATE_XLSX,
    DYNAMIC_TAU_PARAMS_CSV,
    FIXED_TAU1_FACTORS_CSV,
    OFZ_HISTORY_CSV,
    RUSSIA_CPI_CSV,
    USD_XLSX,
)


def test_required_data_paths_exist():
    assert OFZ_HISTORY_CSV.exists(), f"Missing file: {OFZ_HISTORY_CSV}"
    assert BRENT_XLSX.exists(), f"Missing file: {BRENT_XLSX}"
    assert USD_XLSX.exists(), f"Missing file: {USD_XLSX}"
    assert CBR_INFLATION_KEY_RATE_XLSX.exists(), f"Missing file: {CBR_INFLATION_KEY_RATE_XLSX}"
    assert DYNAMIC_TAU_PARAMS_CSV.exists(), f"Missing file: {DYNAMIC_TAU_PARAMS_CSV}"
    assert RUSSIA_CPI_CSV.exists(), f"Missing file: {RUSSIA_CPI_CSV}"
    assert FIXED_TAU1_FACTORS_CSV.exists(), f"Missing file: {FIXED_TAU1_FACTORS_CSV}"
