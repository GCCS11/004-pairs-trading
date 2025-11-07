"""
Performance Analysis and Visualization
Comprehensive analysis tools for strategy evaluation.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


class PerformanceAnalyzer:
    """Analyze and visualize trading strategy performance."""

    def __init__(self):
        """Initialize performance analyzer."""
        self.results = {}
        self.metrics = {}

    def load_results(self, dataset_names=['train', 'test', 'val']):
        """Load backtest results from all datasets."""
        for name in dataset_names:
            self.results[name] = pd.read_csv(
                f'data/processed/backtest_{name}.csv',
                index_col='date',
                parse_dates=True
            )

    def compare_equity_curves(self, save_path='reports/figures/equity_curves_comparison.png'):
        """Plot equity curves for all datasets."""
        fig, ax = plt.subplots(figsize=(14, 7))

        colors = {'train': 'darkblue', 'test': 'darkred', 'val': 'darkgreen'}

        for name, results in self.results.items():
            ax.plot(results.index, results['portfolio_value'],
                   label=f'{name.capitalize()} Set',
                   linewidth=2, color=colors[name])

        ax.axhline(y=1_000_000, color='black', linestyle='--',
                  linewidth=1, alpha=0.5, label='Initial Capital')
        ax.set_xlabel('Date')
        ax.set_ylabel('Portfolio Value ($)')
        ax.set_title('Equity Curves: Train/Test/Validation Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {save_path}")

    def plot_drawdowns(self, save_path='reports/figures/drawdowns_comparison.png'):
        """Plot drawdown analysis for all datasets."""
        fig, axes = plt.subplots(3, 1, figsize=(14, 10))

        colors = {'train': 'darkblue', 'test': 'darkred', 'val': 'darkgreen'}

        for idx, (name, results) in enumerate(self.results.items()):
            returns = results['returns'].dropna()
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max * 100

            axes[idx].fill_between(drawdown.index, drawdown, 0,
                                  color=colors[name], alpha=0.3)
            axes[idx].plot(drawdown.index, drawdown,
                          color=colors[name], linewidth=1.5)
            axes[idx].set_ylabel('Drawdown (%)')
            axes[idx].set_title(f'{name.capitalize()} Set Drawdown')
            axes[idx].grid(True, alpha=0.3)
            axes[idx].axhline(y=0, color='black', linewidth=1, alpha=0.5)

        axes[-1].set_xlabel('Date')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {save_path}")

    def plot_trade_distribution(self, save_path='reports/figures/trade_distribution.png'):
        """Plot distribution of returns per trade."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        for idx, (name, results) in enumerate(self.results.items()):
            # Get trade returns
            trades = results[results['signal'].isin(['EXIT_LONG', 'EXIT_SHORT'])].copy()

            if len(trades) > 0:
                # Calculate returns between consecutive exits
                trade_returns = trades['portfolio_value'].pct_change().dropna() * 100

                axes[idx].hist(trade_returns, bins=20, edgecolor='black', alpha=0.7)
                axes[idx].axvline(x=0, color='red', linestyle='--', linewidth=2)
                axes[idx].axvline(x=trade_returns.mean(), color='green',
                                linestyle='--', linewidth=2, label=f'Mean: {trade_returns.mean():.2f}%')
                axes[idx].set_xlabel('Return per Trade (%)')
                axes[idx].set_ylabel('Frequency')
                axes[idx].set_title(f'{name.capitalize()} Set')
                axes[idx].legend()
                axes[idx].grid(True, alpha=0.3)
            else:
                axes[idx].text(0.5, 0.5, 'No completed trades',
                             ha='center', va='center', transform=axes[idx].transAxes)
                axes[idx].set_title(f'{name.capitalize()} Set')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {save_path}")

    def plot_metrics_comparison(self, metrics_dict, save_path='reports/figures/metrics_comparison.png'):
        """Create bar chart comparing key metrics across datasets."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        datasets = list(metrics_dict.keys())

        # Total Return
        returns = [metrics_dict[d]['total_return'] for d in datasets]
        axes[0, 0].bar(datasets, returns, color=['darkblue', 'darkred', 'darkgreen'])
        axes[0, 0].set_ylabel('Return (%)')
        axes[0, 0].set_title('Total Return')
        axes[0, 0].grid(True, alpha=0.3, axis='y')

        # Sharpe Ratio
        sharpes = [metrics_dict[d]['sharpe_ratio'] for d in datasets]
        axes[0, 1].bar(datasets, sharpes, color=['darkblue', 'darkred', 'darkgreen'])
        axes[0, 1].set_ylabel('Sharpe Ratio')
        axes[0, 1].set_title('Sharpe Ratio')
        axes[0, 1].axhline(y=0, color='black', linewidth=1)
        axes[0, 1].grid(True, alpha=0.3, axis='y')

        # Max Drawdown
        drawdowns = [metrics_dict[d]['max_drawdown'] for d in datasets]
        axes[1, 0].bar(datasets, drawdowns, color=['darkblue', 'darkred', 'darkgreen'])
        axes[1, 0].set_ylabel('Max Drawdown (%)')
        axes[1, 0].set_title('Maximum Drawdown')
        axes[1, 0].grid(True, alpha=0.3, axis='y')

        # Number of Trades
        trades = [metrics_dict[d]['n_trades'] for d in datasets]
        axes[1, 1].bar(datasets, trades, color=['darkblue', 'darkred', 'darkgreen'])
        axes[1, 1].set_ylabel('Number of Trades')
        axes[1, 1].set_title('Trading Activity')
        axes[1, 1].grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {save_path}")

    def create_metrics_table(self, metrics_dict, save_path='reports/figures/metrics_table.png'):
        """Create a clean table of all metrics."""
        # Prepare data
        data = []
        for name in ['train', 'test', 'val']:
            m = metrics_dict[name]
            data.append([
                name.capitalize(),
                f"{m['total_return']:.2f}%",
                f"{m['sharpe_ratio']:.2f}",
                f"{m['sortino_ratio']:.2f}",
                f"{m['calmar_ratio']:.2f}",
                f"{m['max_drawdown']:.2f}%",
                m['n_trades'],
                f"${m['total_costs']:,.0f}"
            ])

        columns = ['Set', 'Return', 'Sharpe', 'Sortino', 'Calmar', 'Max DD', 'Trades', 'Costs']

        fig, ax = plt.subplots(figsize=(12, 3))
        ax.axis('tight')
        ax.axis('off')

        table = ax.table(cellText=data, colLabels=columns,
                        cellLoc='center', loc='center',
                        colColours=['lightgray']*8)
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)

        # Color code returns
        for i in range(1, 4):
            if float(data[i-1][1].strip('%')) > 50:
                table[(i, 1)].set_facecolor('#90EE90')  # Light green
            elif float(data[i-1][1].strip('%')) < 20:
                table[(i, 1)].set_facecolor('#FFB6C6')  # Light red

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {save_path}")


if __name__ == "__main__":
    print("Creating comprehensive performance analysis...\n")

    # Load metrics
    import json
    metrics = {}
    for name in ['train', 'test', 'val']:
        backtest_df = pd.read_csv(f'data/processed/backtest_{name}.csv',
                                  index_col='date', parse_dates=True)

        # Recalculate metrics
        from backtesting import PairsTradingBacktest
        bt = PairsTradingBacktest()
        metrics[name] = bt.calculate_metrics(backtest_df)

    # Create analyzer
    analyzer = PerformanceAnalyzer()
    analyzer.load_results()

    # Generate all plots
    analyzer.compare_equity_curves()
    analyzer.plot_drawdowns()
    analyzer.plot_trade_distribution()
    analyzer.plot_metrics_comparison(metrics)
    analyzer.create_metrics_table(metrics)
