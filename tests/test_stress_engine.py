"""Unit tests for stress engine mathematical components."""

import numpy as np

from stress_engine.backtest import (
    calculate_var_exceptions,
    christoffersen_independence_test,
    kupiec_pof_test,
    run_full_var_backtest,
)


def test_calculate_var_exceptions():
    returns = np.array([-0.05, -0.01, 0.02, -0.04, 0.01], dtype=np.float64)
    var = -0.03
    exceptions = calculate_var_exceptions(returns, var)
    expected = np.array([True, False, False, True, False], dtype=np.bool_)
    assert np.array_equal(exceptions, expected)


def test_kupiec_perfect_calibration():
    # 1,000 observations with exactly 50 breaches (5.0% empirical on 95% VaR)
    exceptions = np.zeros(1000, dtype=np.bool_)
    exceptions[:50] = True

    stat, p_val, reject = kupiec_pof_test(exceptions, confidence_level=0.95)
    assert np.isclose(stat, 0.0, atol=1e-2)
    assert p_val > 0.90
    assert not reject


def test_kupiec_severe_underestimation():
    # 1,000 observations with 150 breaches (15% empirical on 95% VaR)
    exceptions = np.zeros(1000, dtype=np.bool_)
    exceptions[:150] = True

    stat, p_val, reject = kupiec_pof_test(exceptions, confidence_level=0.95)
    assert stat > 20.0
    assert p_val < 0.001
    assert reject


def test_christoffersen_independence_clustering():
    # Heavy clustering: all 10 exceptions occur sequentially
    exceptions = np.zeros(200, dtype=np.bool_)
    exceptions[50:60] = True

    stat, p_val, reject = christoffersen_independence_test(exceptions)
    assert stat > 3.841  # Chi-squared critical value for df=1 at 95%
    assert p_val < 0.05
    assert reject


def test_full_backtest_dataclass():
    returns = np.random.default_rng(42).normal(0.0, 0.01, size=500)
    var = -0.02
    result = run_full_var_backtest(returns, var, confidence_level=0.95)

    assert result.total_observations == 500
    assert isinstance(result.empirical_rate, float)
    assert isinstance(result.kupiec_reject, bool)


def test_monte_carlo_tensor_geometry():
    from stress_engine.monte_carlo import run_comprehensive_stress_engine

    # Run lightweight smoke simulation (100 paths, 10 days)
    df_results, master_tensor = run_comprehensive_stress_engine(n_paths=100, n_days=10)

    # 81 parameter nodes
    assert len(df_results) == 81
    assert master_tensor.shape == (81, 100, 10)
    assert master_tensor.dtype == np.float32

    # Drawdown probabilities must strictly live in [0.0, 1.0]
    assert (df_results["Distress_Probability"] >= 0.0).all()
    assert (df_results["Distress_Probability"] <= 1.0).all()
