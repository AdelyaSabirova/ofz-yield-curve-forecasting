def test_import_ofz_curve_package():
    import ofz_curve  # noqa: F401


def test_import_key_modules():
    import ofz_curve.analysis  # noqa: F401
    import ofz_curve.key_rate_event_study  # noqa: F401
    import ofz_curve.ofz_auction_study  # noqa: F401
    import ofz_curve.long_end_residual  # noqa: F401
    import ofz_curve.policy_path_gap  # noqa: F401
    import ofz_curve.cpi_loaders  # noqa: F401
    import ofz_curve.data_loading  # noqa: F401
    import ofz_curve.dyn_tau_analysis  # noqa: F401
    import ofz_curve.forecasting  # noqa: F401
    import ofz_curve.paths  # noqa: F401
    import ofz_curve.var_tests  # noqa: F401
