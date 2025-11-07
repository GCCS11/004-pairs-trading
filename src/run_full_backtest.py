"""
Run complete backtesting pipeline on all data splits.
"""

import pandas as pd
from data_loader import DataLoader
from data_preprocessing import DataPreprocessor
from pair_selection import PairSelector
from kalman_hedge_ratio import HedgeRatioKalmanFilter
from kalman_signal_generation import SignalKalmanFilter
from backtesting import PairsTradingBacktest


def run_pipeline(train_data, test_data, val_data):
    """Run complete pipeline on all data splits."""

    # Step 1: Get cointegration parameters from TRAINING data only
    print("=" * 70)
    print("STEP 1: Pair Selection (Training Data)")
    print("=" * 70)
    selector = PairSelector(train_data)
    selector.calculate_correlation()
    eg_results = selector.engle_granger_test()
    joh_results = selector.johansen_test()

    initial_beta = eg_results['hedge_ratio']
    evec = joh_results['eigenvectors'][:, 0]

    # Step 2: Apply Kalman Filter #1 to all datasets
    print("\n" + "=" * 70)
    print("STEP 2: Kalman Filter #1 - Dynamic Hedge Ratios")
    print("=" * 70)

    results = {}

    for name, data in [('train', train_data), ('test', test_data), ('val', val_data)]:
        print(f"\nProcessing {name} set...")

        # Hedge ratios
        kf_hedge = HedgeRatioKalmanFilter(initial_beta=initial_beta, initial_P=1.0, Q=1e-2, R=0.1)
        hedge_results = kf_hedge.filter_series(data)

        # VECM spread
        vecm_spread = data.iloc[:, 0] * evec[0] + data.iloc[:, 1] * evec[1]

        # Signals
        kf_signal = SignalKalmanFilter(
            initial_spread=vecm_spread.iloc[0],
            initial_P=1.0,
            alpha=0.99,
            Q=1e-3,
            R=1e-2
        )
        signal_results = kf_signal.filter_spread_series(vecm_spread)
        signal_results = kf_signal.generate_signals(signal_results, entry_threshold=0.75, exit_threshold=0.3)

        n_trades = len(signal_results[signal_results['signal'].isin(['LONG', 'SHORT'])])
        print(f"  {n_trades} trade entries")

        results[name] = {
            'prices': data,
            'hedge_ratios': hedge_results,
            'signals': signal_results
        }

    # Step 3: Backtest on all datasets
    print("\n" + "=" * 70)
    print("STEP 3: Backtesting")
    print("=" * 70)

    backtest = PairsTradingBacktest(initial_capital=1_000_000)
    all_metrics = {}

    for name in ['train', 'test', 'val']:
        print(f"\n{name.upper()} SET:")

        bt_results = backtest.run_backtest(
            results[name]['prices'],
            results[name]['signals'],
            results[name]['hedge_ratios']
        )

        metrics = backtest.calculate_metrics(bt_results)
        all_metrics[name] = metrics

        print(f"  Return: {metrics['total_return']:.2f}%")
        print(f"  Sharpe: {metrics['sharpe_ratio']:.2f}")
        print(f"  Max DD: {metrics['max_drawdown']:.2f}%")
        print(f"  Trades: {metrics['n_trades']}")
        print(f"  Costs: ${metrics['total_costs']:,.2f}")

        # Save results
        bt_results.to_csv(f'data/processed/backtest_{name}.csv')
        results[name]['signals'].to_csv(f'data/processed/signals_{name}.csv')
        results[name]['hedge_ratios'].to_csv(f'data/processed/hedge_ratios_{name}.csv')

        # Plot
        backtest.plot_results(bt_results,
                              title=f"Pairs Trading: HD-LOW ({name.capitalize()} Set)",
                              save_path=f'reports/figures/backtest_{name}.png')

    # Summary comparison
    print("\n" + "=" * 70)
    print("PERFORMANCE SUMMARY")
    print("=" * 70)
    print(f"\n{'Set':<12} {'Return':<12} {'Sharpe':<10} {'Max DD':<12} {'Trades':<10}")
    print("-" * 60)

    for name in ['train', 'test', 'val']:
        m = all_metrics[name]
        print(f"{name.upper():<12} {m['total_return']:>10.2f}% {m['sharpe_ratio']:>9.2f} "
              f"{m['max_drawdown']:>10.2f}% {m['n_trades']:>9}")

    return results, all_metrics


if __name__ == "__main__":
    print("Loading data...")
    loader = DataLoader(tickers=['HD', 'LOW'])
    data = loader.load_data()

    preprocessor = DataPreprocessor(data)
    train, test, val = preprocessor.load_splits()

    results, metrics = run_pipeline(train, test, val)

    print("\n" + "=" * 70)
    print("COMPLETE! All results saved to data/processed/ and reports/figures/")
    print("=" * 70)