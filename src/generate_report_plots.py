"""
Generate all plots required for the executive report.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import io


def plot_spread_evolution_all_sets(save_path='reports/figures/spread_evolution_all_sets.png'):
    """Plot VECM spread evolution across all data sets."""
    from data_loader import DataLoader
    from data_preprocessing import DataPreprocessor
    from pair_selection import PairSelector

    # Suppress prints temporarily
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()

    loader = DataLoader(['HD', 'LOW'])
    data = loader.load_data()

    prep = DataPreprocessor(data)
    train, test, val = prep.split_data()

    selector = PairSelector(train)
    selector.engle_granger_test()
    joh = selector.johansen_test()
    evec = joh['eigenvectors'][:, 0]

    sys.stdout = old_stdout  # Restore prints

    # Calculate spreads for all sets
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    datasets = [('Train', train, 'darkblue'),
                ('Test', test, 'darkred'),
                ('Val', val, 'darkgreen')]

    for idx, (name, dataset, color) in enumerate(datasets):
        spread = dataset.iloc[:, 0] * evec[0] + dataset.iloc[:, 1] * evec[1]

        axes[idx].plot(spread.index, spread, linewidth=1, color=color)
        axes[idx].axhline(y=0, color='black', linestyle='--', linewidth=1)
        axes[idx].axhline(y=spread.mean() + 2*spread.std(), color='red',
                         linestyle='--', linewidth=1, alpha=0.5, label='±2σ')
        axes[idx].axhline(y=spread.mean() - 2*spread.std(), color='red',
                         linestyle='--', linewidth=1, alpha=0.5)
        axes[idx].set_ylabel('VECM Spread')
        axes[idx].set_title(f'{name} Set: Spread Evolution (VECM)')
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)

    axes[-1].set_xlabel('Date')
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_hedge_ratio_evolution_all_sets(save_path='reports/figures/hedge_ratio_all_sets.png'):
    """Plot dynamic hedge ratio evolution across all sets."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    datasets = [('Train', 'darkblue'), ('Test', 'darkred'), ('Val', 'darkgreen')]

    for idx, (name, color) in enumerate(datasets):
        hedge_df = pd.read_csv(f'data/processed/hedge_ratios_{name.lower()}.csv',
                               index_col='date', parse_dates=True)

        axes[idx].plot(hedge_df.index, hedge_df['hedge_ratio'],
                      linewidth=1.5, color=color)
        axes[idx].set_ylabel('Hedge Ratio (β)')
        axes[idx].set_title(f'{name} Set: Dynamic Hedge Ratio Evolution')
        axes[idx].grid(True, alpha=0.3)

    axes[-1].set_xlabel('Date')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_first_eigenvector_evolution(save_path='reports/figures/eigenvector_evolution.png'):
    """Plot first eigenvector values through time using rolling windows."""
    from data_loader import DataLoader
    from statsmodels.tsa.vector_ar.vecm import coint_johansen

    loader = DataLoader(['HD', 'LOW'])

    # Suppress loader prints
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    data = loader.load_data()
    sys.stdout = old_stdout

    # Calculate rolling eigenvectors
    window = 252  # 1 year
    eigenvector_1 = []
    eigenvector_2 = []
    dates = []

    for i in range(window, len(data)):
        window_data = data.iloc[i-window:i]
        try:
            result = coint_johansen(window_data, det_order=0, k_ar_diff=1)
            evec = result.evec[:, 0]
            eigenvector_1.append(evec[0])
            eigenvector_2.append(evec[1])
            dates.append(data.index[i])
        except:
            continue

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))

    ax1.plot(dates, eigenvector_1, linewidth=1.5, color='darkblue', label='HD coefficient')
    ax1.set_ylabel('Eigenvector Component 1')
    ax1.set_title('First Eigenvector Evolution (Rolling 1-Year Window)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(dates, eigenvector_2, linewidth=1.5, color='darkgreen', label='LOW coefficient')
    ax2.set_ylabel('Eigenvector Component 2')
    ax2.set_xlabel('Date')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_correlation_over_time(save_path='reports/figures/rolling_correlation.png'):
    """Plot rolling correlation between HD and LOW."""
    from data_loader import DataLoader

    loader = DataLoader(['HD', 'LOW'])

    # Suppress loader prints
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    data = loader.load_data()
    sys.stdout = old_stdout

    # Calculate rolling correlation
    window = 252
    rolling_corr = data['HD'].rolling(window).corr(data['LOW'])

    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(rolling_corr.index, rolling_corr, linewidth=1.5, color='darkblue')
    ax.axhline(y=0.7, color='red', linestyle='--', linewidth=1,
              label='Threshold (0.7)', alpha=0.5)
    ax.set_xlabel('Date')
    ax.set_ylabel('Correlation')
    ax.set_title('Rolling Correlation: HD vs LOW (1-Year Window)')
    ax.set_ylim([0, 1])
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    print("Generating report plots.")

    plot_spread_evolution_all_sets()
    print("  - Spread evolution")

    plot_hedge_ratio_evolution_all_sets()
    print("  - Hedge ratio evolution")

    plot_first_eigenvector_evolution()
    print("  - Eigenvector evolution")

    plot_correlation_over_time()
    print("  - Rolling correlation")

    print("\nAll plots generated in reports/figures/")