from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _read(rel_path: str) -> str:
    nb_path = ROOT / rel_path
    assert nb_path.exists(), f"Missing notebook: {nb_path}"
    return nb_path.read_text(encoding="utf-8")


def test_final_refactored_notebook_contract():
    raw = _read("notebooks/active/final_refactored.ipynb")
    assert "from functions" not in raw
    assert "import functions" not in raw
    assert "latex_vkr/" not in raw


def test_cpi_notebook_contract():
    raw = _read("notebooks/active/get_cpi_data.ipynb")
    assert "from functions.cpi_loaders" not in raw
    assert "to_csv('russia_cpi_data.csv')" not in raw
    assert 'output_filename = "russia_cpi_data.csv"' not in raw
    assert "output_filename = 'russia_cpi_data.csv'" not in raw


def test_var_notebook_contract():
    raw = _read("notebooks/active/var_analysis.ipynb")
    assert "from functions.var_tests" not in raw
    assert "pd.read_csv('params_best_tau1_')" not in raw
    assert 'pd.read_csv("params_best_tau1_' not in raw
    assert "pd.read_csv('params_best_tau1.csv')" not in raw
    assert 'pd.read_csv("params_best_tau1.csv")' not in raw
    assert "pd.read_csv('daily_ns_factors_fixed_tau1.csv')" not in raw
    assert 'pd.read_csv("daily_ns_factors_fixed_tau1.csv")' not in raw
    assert 'pd.read_csv("russia_cpi_data.csv")' not in raw
    assert "pd.read_csv('russia_cpi_data.csv')" not in raw
