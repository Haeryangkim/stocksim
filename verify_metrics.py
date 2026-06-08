"""Verification harness for PortfolioBacktest metric calculations.

Each test feeds the backtester synthetic price data whose theoretical
metrics are known by construction, then compares the computed values
against the analytical truth. Goal: catch systematic errors before
fixing anything, with particular focus on installment (DCA) semantics.

Run inside the Docker container:
    docker exec stocksim_app python /app/verify_metrics.py
"""
import math
import sys

import numpy as np
import pandas as pd

sys.path.append('/app')
sys.path.append('.')

from backtester import PortfolioBacktest, RISK_FREE_RATE


# ---------- helpers ----------

def bdates(start, n):
    return pd.bdate_range(start=start, periods=n)


def constant_growth_df(daily_return, n_days, tickers=('A',), start='2020-01-01', base=100.0):
    idx = bdates(start, n_days)
    factor = (1 + daily_return) ** np.arange(n_days)
    return pd.DataFrame({t: base * factor for t in tickers}, index=idx)


def synth_bt(asset_prices, bench_prices, weights_dict, **kw):
    """Construct a PortfolioBacktest with prices pre-injected (no network)."""
    bt = PortfolioBacktest(
        weights_dict,
        asset_prices.index[0], asset_prices.index[-1],
        benchmark='__SYNTH__',
        **kw,
    )
    bt.prices = asset_prices.copy()
    bt.bench_prices = bench_prices.copy()
    bt.benchmark_is_inflation = False
    bt.benchmark_label = '__SYNTH__'
    bt.start_date_adjusted = False
    bt.fetch_data = lambda: None
    return bt


class Recorder:
    def __init__(self):
        self.results = []

    def check(self, label, got, want, tol=1e-6, kind='ratio'):
        diff = abs(got - want)
        ok = diff < tol
        self.results.append((label, ok))
        sym = 'OK  ' if ok else 'FAIL'
        if kind == 'pct':
            print(f"  [{sym}] {label}: got={got:+.4%}  want={want:+.4%}  diff={diff:.2e}")
        else:
            print(f"  [{sym}] {label}: got={got:.6f}  want={want:.6f}  diff={diff:.2e}")
        return ok

    def note(self, msg):
        print(f"        {msg}")

    def summary(self):
        passed = sum(1 for _, ok in self.results if ok)
        total = len(self.results)
        print("\n" + "=" * 70)
        print(f"SUMMARY: {passed}/{total} checks passed")
        fails = [name for name, ok in self.results if not ok]
        if fails:
            print("\nFailing checks:")
            for name in fails:
                print(f"  - {name}")
        print("=" * 70)
        return passed == total


# ---------- tests ----------

def t1_constant_growth_lump_sum(R):
    """Constant daily return → CAGR = (1+r)^252 - 1, vol = 0, MDD = 0."""
    print("\n[T1] Constant daily growth, lump-sum (single asset)")
    n = 252
    daily = 0.001
    prices = constant_growth_df(daily, n, tickers=['A'])
    bench = constant_growth_df(daily, n, tickers=['B'])['B']
    bt = synth_bt(prices, bench, {'A': 1.0})
    bt.run()

    # We have n trading days but pct_change drops first → n-1 daily returns.
    # CAGR is annualized on calendar-time so it lines up with MWR/IRR.
    n_ret = n - 1
    idx = prices.index[1:]  # what portfolio_returns will see
    years = (idx[-1] - idx[0]).days / 365.25
    expected_cum = (1 + daily) ** n_ret
    expected_cagr = expected_cum ** (1 / years) - 1

    R.check("CAGR matches (1+r)^252 - 1", bt.cagr, expected_cagr, tol=1e-6)
    R.check("Cumulative TWR", bt.cumulative_returns.iloc[-1], expected_cum, tol=1e-6)
    R.check("Volatility (zero-vol series)", bt.volatility, 0.0, tol=1e-12)
    R.check("MaxDD (monotone rising)", bt.max_drawdown, 0.0, tol=1e-12)
    R.check("Beta=1 when port==bench", bt.beta, 1.0, tol=1e-9)
    R.check("Alpha=0 when port==bench", bt.alpha, 0.0, tol=1e-9)
    R.check("MWR == TWR CAGR for lump-sum", bt.mwr, bt.cagr, tol=1e-4)


def t2_known_drawdown(R):
    """Construct a series with a clean 30% drawdown from a known peak."""
    print("\n[T2] Constructed -30% drawdown")
    idx = bdates('2020-01-01', 300)
    n = len(idx)
    p = np.empty(n)
    # Phase A (0..99): rise 100->150
    p[:100] = 100.0 * (1.5) ** (np.arange(100) / 99)
    # Phase B (100..199): drop linearly to 0.7 * 150 = 105
    p[100:200] = 150.0 * (1 - 0.30 * (np.arange(100) / 99))
    # Phase C (200..): flat at 105
    p[200:] = 150.0 * 0.70
    df = pd.DataFrame({'A': p}, index=idx)
    bench = pd.Series(p, index=idx, name='B')
    bt = synth_bt(df, bench, {'A': 1.0})
    bt.run()

    R.check("MaxDD ≈ -30%", bt.max_drawdown, -0.30, tol=2e-3)


def t3_dca_twr_invariance(R):
    """DCA should NOT change time-weighted CAGR, since daily returns are
    computed from market-only moves (cash flow subtracted out by design)."""
    print("\n[T3] DCA invariance of TWR / CAGR / Vol / MDD")
    n = 252 * 3
    prices = constant_growth_df(0.0005, n, tickers=['A'])
    bench = constant_growth_df(0.0005, n, tickers=['B'])['B']

    lump = synth_bt(prices, bench, {'A': 1.0}, initial_capital=10000)
    lump.run()

    dca = synth_bt(prices, bench, {'A': 1.0}, initial_capital=10000,
                   installment_amount=500, installment_frequency='Monthly')
    dca.run()

    R.check("CAGR identical (lump vs DCA)", dca.cagr, lump.cagr, tol=1e-12)
    R.check("Vol identical (lump vs DCA)", dca.volatility, lump.volatility, tol=1e-12)
    R.check("TWR cumulative identical",
            dca.cumulative_returns.iloc[-1], lump.cumulative_returns.iloc[-1], tol=1e-12)

    invested_dca = dca.invested_capitals.iloc[-1]
    roi_displayed = dca.portfolio_value.iloc[-1] / invested_dca - 1
    twr_total = dca.cumulative_returns.iloc[-1] - 1

    R.note(f"App displays 'Total Return (ROI)' = final/invested - 1 = {roi_displayed:+.4%}")
    R.note(f"TWR cumulative total return                              = {twr_total:+.4%}")
    R.note("Divergence is expected for DCA (MWR-style vs TWR). Display labels")
    R.note("should make this distinction clearer.")


def t4_benchmark_dca_handling(R):
    """Two benchmark series:
      bench_value     — pure lump-sum on initial_capital (market reference)
      bench_value_dca — same cash-flow schedule as portfolio (DCA-matched)
    In a constant-rate market the DCA-matched benchmark must end up at
    EXACTLY the portfolio's value (same returns + same flows).
    """
    print("\n[T4] Benchmark has both lump-sum and DCA-matched lines")
    n = 252 * 2
    prices = constant_growth_df(0.0005, n, tickers=['A'])
    bench = constant_growth_df(0.0005, n, tickers=['B'])['B']

    bt = synth_bt(prices, bench, {'A': 1.0}, initial_capital=10000,
                  installment_amount=1000, installment_frequency='Monthly')
    bt.run()

    n_ret = n - 1
    idx_ret = prices.index[1:]
    years = (idx_ret[-1] - idx_ret[0]).days / 365.25
    expected_market_cum = (1 + 0.0005) ** n_ret
    expected_market_cagr = expected_market_cum ** (1 / years) - 1
    expected_bench_lump = 10000 * expected_market_cum

    R.check("Bench TWR CAGR = pure market CAGR", bt.bench_cagr, expected_market_cagr, tol=1e-6)
    R.check("Bench TWR cumulative = pure market cumulative",
            bt.bench_cumulative.iloc[-1], expected_market_cum, tol=1e-6)
    R.check("bench_value = initial * market_cumulative (lump-sum, no DCA)",
            bt.bench_value.iloc[-1], expected_bench_lump, tol=1e-4)
    R.check("bench_value_dca == portfolio_value (same returns, same flows)",
            bt.bench_value_dca.iloc[-1], bt.portfolio_value.iloc[-1], tol=1e-4)
    R.note(f"  → portfolio with DCA = ${bt.portfolio_value.iloc[-1]:,.2f}, "
           f"bench DCA-matched = ${bt.bench_value_dca.iloc[-1]:,.2f}, "
           f"bench lump-sum    = ${bt.bench_value.iloc[-1]:,.2f}, "
           f"invested          = ${bt.invested_capitals.iloc[-1]:,.2f}")


def t5_inflation_benchmark_semantics(R):
    """Inflation benchmark must also be lump-sum-only (the value 10k would
    have if it had merely kept pace with CPI)."""
    print("\n[T5] Inflation benchmark with DCA → still lump-sum baseline")
    n = 252 * 2
    prices = constant_growth_df(0.0005, n, tickers=['A'])
    cpi_daily = (1.03) ** (1 / 252) - 1
    cpi = constant_growth_df(cpi_daily, n, tickers=['CPI'])['CPI']

    bt = synth_bt(prices, cpi, {'A': 1.0}, initial_capital=10000,
                  installment_amount=1000, installment_frequency='Monthly')
    bt.run()

    n_ret = n - 1
    expected_lump = 10000 * (1 + cpi_daily) ** n_ret
    R.check("Inflation bench_value (lump) = initial * inflation_cumulative",
            bt.bench_value.iloc[-1], expected_lump, tol=1e-4)
    # DCA-matched inflation bench should equal total invested grown at CPI
    R.note(f"  → lump  = ${bt.bench_value.iloc[-1]:,.2f} (10k preserved purchasing power)")
    R.note(f"  → DCA   = ${bt.bench_value_dca.iloc[-1]:,.2f} (all 34k contributions, inflation-adjusted)")
    R.note(f"  → port  = ${bt.portfolio_value.iloc[-1]:,.2f}")


def t6_beta_alpha_when_equal(R):
    """Random returns, portfolio identical to benchmark → Beta=1, Alpha=0."""
    print("\n[T6] Beta/Alpha sanity (port == bench)")
    np.random.seed(42)
    n = 500
    rets = np.random.normal(0.0005, 0.01, n)
    idx = bdates('2020-01-01', n + 1)
    series = 100 * np.cumprod(np.r_[1.0, 1 + rets])
    prices = pd.DataFrame({'A': series}, index=idx)
    bench = pd.Series(series, index=idx, name='B')
    bt = synth_bt(prices, bench, {'A': 1.0})
    bt.run()

    R.check("Beta = 1", bt.beta, 1.0, tol=1e-9)
    R.check("Alpha = 0", bt.alpha, 0.0, tol=1e-9)


def t7_beta_no_ddof_bias(R):
    """End-to-end check: when port = 2 * bench (perfectly correlated, scale 2),
    the computed Beta must equal 2 exactly (within float precision)."""
    print("\n[T7] Beta of a 2x-leveraged synthetic = 2.000")
    np.random.seed(7)
    n = 500
    bench_rets = np.random.normal(0.0005, 0.01, n)
    # Make prices whose returns are exactly 2x the bench (clamp lower bound)
    port_rets = np.clip(2.0 * bench_rets, -0.99, None)
    idx = bdates('2020-01-01', n + 1)
    bench_series = 100 * np.cumprod(np.r_[1.0, 1 + bench_rets])
    port_series = 100 * np.cumprod(np.r_[1.0, 1 + port_rets])
    prices = pd.DataFrame({'A': port_series}, index=idx)
    bench = pd.Series(bench_series, index=idx, name='B')
    bt = synth_bt(prices, bench, {'A': 1.0})
    bt.run()

    R.note(f"True beta (construction) = 2.0; code Beta = {bt.beta:.6f}")
    R.check("Beta = 2.0 (no ddof bias)", bt.beta, 2.0, tol=1e-9)


def t8_volatility_annualization(R):
    """Daily returns sampled from N(0, σ_d²); annualized vol ≈ σ_d * √252."""
    print("\n[T8] Volatility annualization")
    np.random.seed(11)
    n = 2000
    sigma_d = 0.012
    rets = np.random.normal(0.0, sigma_d, n)
    idx = bdates('2015-01-01', n + 1)
    series = 100 * np.cumprod(np.r_[1.0, 1 + rets])
    prices = pd.DataFrame({'A': series}, index=idx)
    bench = pd.Series(series, index=idx, name='B')
    bt = synth_bt(prices, bench, {'A': 1.0})
    bt.run()

    expected = sigma_d * math.sqrt(252)
    R.note(f"Realized vol*sqrt(252): {bt.volatility:.6f}  expected: {expected:.6f}")
    # Sample noise tolerance
    R.check("Vol annualized", bt.volatility, expected, tol=0.01)


def t9_multi_asset_rebalanced_cagr(R):
    """Two assets w/ different constant returns, rebalanced monthly to 50/50.
    Daily portfolio return = 0.5*r1 + 0.5*r2 (after rebalance reset every
    period). With CONSTANT daily returns, even no-rebalance keeps weights
    drifting but daily return per asset is constant, so portfolio return
    asymptotes to a weighted-by-allocation rate. We test the rebalanced
    case where weights are exactly 0.5/0.5 each period."""
    print("\n[T9] Multi-asset rebalanced, mixed constant returns")
    n = 252
    r1 = 0.001
    r2 = 0.0005
    idx = bdates('2020-01-01', n)
    s1 = 100 * (1 + r1) ** np.arange(n)
    s2 = 100 * (1 + r2) ** np.arange(n)
    prices = pd.DataFrame({'A': s1, 'B': s2}, index=idx)
    bench = pd.Series(s1, index=idx, name='X')
    bt = synth_bt(prices, bench, {'A': 0.5, 'B': 0.5}, rebalance='Monthly')
    bt.run()

    # With monthly rebalance and constant daily returns, daily portfolio return
    # is exactly 0.5*r1 + 0.5*r2 every day (the weights reset back to 50/50
    # each rebalance, and within a month the drift is small but nonzero).
    approx_daily = 0.5 * r1 + 0.5 * r2
    idx_ret = prices.index[1:]
    years = (idx_ret[-1] - idx_ret[0]).days / 365.25
    approx_cum = (1 + approx_daily) ** (n - 1)
    approx_cagr = approx_cum ** (1 / years) - 1
    R.note(f"Approx CAGR (compounded over {years:.3f} cal-years): {approx_cagr:.6%}")
    R.note(f"Realized CAGR:                                       {bt.cagr:.6%}")
    R.check("Realized CAGR within 1% of approx", bt.cagr, approx_cagr, tol=1e-2)


def t12_rolling_mwr_matches_full_window(R):
    """The MWR at the earliest rolling start date should match bt.mwr (same
    cash flows over same window). Same for benchmark MWR against an
    independently-computed lump-sum benchmark IRR."""
    print("\n[T12] Rolling-start MWR at earliest date == full-window MWR")
    n = 252 * 3
    prices = constant_growth_df(0.0005, n, tickers=['A'])
    bench = constant_growth_df(0.0005, n, tickers=['B'])['B']

    bt = synth_bt(prices, bench, {'A': 1.0}, initial_capital=10000,
                  installment_amount=500, installment_frequency='Monthly')
    bt.run()
    rs = bt.rolling_start_analysis()

    R.check("rolling 'Portfolio MWR' column exists when DCA active",
            float('Portfolio MWR' in rs.columns), 1.0, tol=0.5)
    earliest_mwr = rs['Portfolio MWR'].iloc[0]
    R.check("rolling MWR at earliest start ≈ bt.mwr", earliest_mwr, bt.mwr, tol=5e-3)
    R.note(f"  → rolling MWR[0]={earliest_mwr:+.4%}, bt.mwr={bt.mwr:+.4%}")


def t11_mwr_diverges_from_twr_with_timing(R):
    """DCA into a market that returned a lot up-front then went flat:
    the late contributions earn nothing, so MWR < TWR.
    Reverse case (flat then up): MWR > TWR.
    """
    print("\n[T11] MWR vs TWR diverges with return timing")
    n_half = 252
    n = n_half * 2
    idx = bdates('2020-01-01', n)
    # First half: 0.1%/day. Second half: 0%/day.
    rets_front = np.r_[np.full(n_half, 0.001), np.zeros(n_half)]
    series_front = 100 * np.cumprod(np.r_[1.0, 1 + rets_front[:-1]])
    prices_front = pd.DataFrame({'A': series_front}, index=idx)
    bench_front = pd.Series(series_front, index=idx, name='B')

    front = synth_bt(prices_front, bench_front, {'A': 1.0}, initial_capital=10000,
                     installment_amount=500, installment_frequency='Monthly')
    front.run()

    R.note(f"Returns frontloaded (up early, flat late):  TWR={front.cagr:+.4%}  MWR={front.mwr:+.4%}")
    R.check("MWR < TWR when returns are front-loaded with DCA",
            float(front.mwr < front.cagr), 1.0, tol=0.5)

    # Reverse: flat first half, up second half
    rets_back = np.r_[np.zeros(n_half), np.full(n_half, 0.001)]
    series_back = 100 * np.cumprod(np.r_[1.0, 1 + rets_back[:-1]])
    prices_back = pd.DataFrame({'A': series_back}, index=idx)
    bench_back = pd.Series(series_back, index=idx, name='B')

    back = synth_bt(prices_back, bench_back, {'A': 1.0}, initial_capital=10000,
                    installment_amount=500, installment_frequency='Monthly')
    back.run()

    R.note(f"Returns back-loaded (flat early, up late): TWR={back.cagr:+.4%}  MWR={back.mwr:+.4%}")
    R.check("MWR > TWR when returns are back-loaded with DCA",
            float(back.mwr > back.cagr), 1.0, tol=0.5)


def t10_invested_capital_accounting(R):
    """Verify total invested = initial_capital + N*installments."""
    print("\n[T10] Invested-capital accounting")
    n = 252 * 2
    prices = constant_growth_df(0.0005, n, tickers=['A'])
    bench = constant_growth_df(0.0005, n, tickers=['B'])['B']

    bt = synth_bt(prices, bench, {'A': 1.0}, initial_capital=10000,
                  installment_amount=500, installment_frequency='Monthly')
    bt.run()

    # Count month boundaries in the return series (which is len n-1).
    dates = bt.portfolio_returns.index
    n_month_starts = 0
    for i in range(1, len(dates)):
        if dates[i].month != dates[i - 1].month:
            n_month_starts += 1

    expected_invested = 10000 + 500 * n_month_starts
    actual = bt.invested_capitals.iloc[-1]
    R.note(f"Detected {n_month_starts} month boundaries, expected invested = {expected_invested}")
    R.check("Total invested", actual, expected_invested, tol=0.5)


def main():
    R = Recorder()
    tests = [
        t1_constant_growth_lump_sum,
        t2_known_drawdown,
        t3_dca_twr_invariance,
        t4_benchmark_dca_handling,
        t5_inflation_benchmark_semantics,
        t6_beta_alpha_when_equal,
        t7_beta_no_ddof_bias,
        t8_volatility_annualization,
        t9_multi_asset_rebalanced_cagr,
        t10_invested_capital_accounting,
        t11_mwr_diverges_from_twr_with_timing,
        t12_rolling_mwr_matches_full_window,
    ]
    for t in tests:
        try:
            t(R)
        except Exception as e:
            print(f"  [ERROR] {t.__name__}: {type(e).__name__}: {e}")
            R.results.append((t.__name__, False))
    R.summary()


if __name__ == "__main__":
    main()
