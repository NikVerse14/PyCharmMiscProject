# portfolio_analysis_fixed.py
import os
import hashlib
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import networkx as nx
from scipy.optimize import minimize
import time

# Matplotlib defaults
plt.rcParams["figure.autolayout"] = True
plt.rcParams["axes.grid"] = True


# ============================================================================
# Configuration & Data Classes
# ============================================================================

@dataclass
class PortfolioConfig:
    """Configuration for portfolio analysis."""
    start_date: str = "2023-01-01"
    end_date: str = "2024-12-31"
    tickers: List[str] = None
    cache_dir: Path = Path("./data_cache")
    max_weight_per_asset: float = 0.4
    trading_days_per_year: int = 252

    def __post_init__(self):
        if self.tickers is None:
            # Default tickers - you can change these to ASX small caps
            self.tickers = [
                "AAPL", "MSFT", "META", "NVDA", "GOOGL", "JPM",
                "AMZN", "V", "WMT", "TSLA", "JNJ", "PG"
            ]
        self.cache_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class PortfolioMetrics:
    """Container for portfolio performance metrics."""
    annual_return: float
    annual_volatility: float
    sharpe_ratio: float
    max_drawdown: float
    weights: pd.Series

    def __str__(self):
        return (
            f"Annual Return: {self.annual_return:.1%}\n"
            f"Annual Volatility: {self.annual_volatility:.1%}\n"
            f"Sharpe Ratio: {self.sharpe_ratio:.2f}\n"
            f"Max Drawdown: {self.max_drawdown:.1%}"
        )


# ============================================================================
# Data Management
# ============================================================================

class DataManager:
    """Handles all data downloading and caching operations."""

    def __init__(self, config: PortfolioConfig):
        self.config = config

    def get_price_data(self, use_cache: bool = True, safe_mode: bool = False) -> pd.DataFrame:
        """
        Download or load cached price data.
        FIXED: Handles 'Close' column when auto_adjust=True and rate limiting
        """
        cache_file = self._get_cache_path()

        if use_cache and cache_file.exists():
            print(f"• Loading cached data from {cache_file}")
            cached_data = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            if not cached_data.empty:
                return cached_data
            else:
                print("• Cached data is empty, re-downloading...")

        print(f"• Downloading data for {len(self.config.tickers)} stocks...")

        try:
            if not safe_mode:
                # Download all at once
                data = yf.download(
                    self.config.tickers,
                    start=self.config.start_date,
                    end=self.config.end_date,
                    auto_adjust=True,
                    progress=False,
                    group_by="ticker"
                )

                # Extract prices - when auto_adjust=True, we get 'Close' not 'Adj Close'
                prices = self._extract_prices(data)
            else:
                # Safe mode: download one by one with delays
                prices = self._safe_download()

            prices = prices.dropna(how="all").dropna(axis=0)

            if prices.empty:
                print("❌ No data downloaded! Generating sample data for demonstration...")
                prices = self._generate_sample_data()

            if use_cache and not prices.empty:
                prices.to_csv(cache_file)
                print(f"• Data cached to {cache_file}")

            print(f"• Loaded {len(prices)} trading days for {len(prices.columns)} stocks.")
            return prices

        except Exception as e:
            if "rate limit" in str(e).lower():
                print(f"• Rate limited! Generating sample data for demonstration...")
                return self._generate_sample_data()
            else:
                print(f"• Download failed: {e}")
                print("• Generating sample data for demonstration...")
                return self._generate_sample_data()

    def _extract_prices(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract close prices from yfinance data structure."""
        if len(self.config.tickers) == 1:
            # Single ticker - simple columns
            if 'Close' in data.columns:
                return data[['Close']].rename(columns={'Close': self.config.tickers[0]})
            return data

        if isinstance(data.columns, pd.MultiIndex):
            # Multiple tickers with MultiIndex
            prices_list = []
            for ticker in self.config.tickers:
                for col_order in [(ticker, 'Close'), ('Close', ticker)]:
                    try:
                        series = data[col_order]
                        series.name = ticker  # FIXED: Use .name instead of .rename()
                        prices_list.append(series)
                        break
                    except KeyError:
                        continue
                else:
                    print(f"• Warning: couldn't find price for {ticker}")
            return pd.concat(prices_list, axis=1) if prices_list else pd.DataFrame()
        else:
            # Multiple tickers but simple columns (shouldn't happen but handle it)
            if 'Close' in data.columns:
                return data[['Close']]
            return data

    def _safe_download(self) -> pd.DataFrame:
        """Download tickers one by one with delays to avoid rate limits."""
        cols = []
        for i, ticker in enumerate(self.config.tickers):
            print(f"  - Downloading {ticker} ({i + 1}/{len(self.config.tickers)})")

            try:
                d = yf.download(
                    ticker,
                    start=self.config.start_date,
                    end=self.config.end_date,
                    auto_adjust=True,
                    progress=False
                )

                if not d.empty and 'Close' in d.columns:
                    series = d['Close'].copy()
                    series.name = ticker  # FIXED: Use .name instead of .rename()
                    cols.append(series)
                else:
                    print(f"• Warning: no data for {ticker}")

                # Rate limit protection
                if i < len(self.config.tickers) - 1:
                    time.sleep(1)  # Wait 1 second between downloads

            except Exception as e:
                print(f"• Failed to download {ticker}: {e}")
                continue

        return pd.concat(cols, axis=1) if cols else pd.DataFrame()

    def _generate_sample_data(self) -> pd.DataFrame:
        """Generate realistic sample stock data for demonstration when real data fails."""
        np.random.seed(42)  # For reproducible results

        # Create date range
        start = pd.to_datetime(self.config.start_date)
        end = pd.to_datetime(self.config.end_date)
        dates = pd.date_range(start, end, freq='D')
        # Filter to business days only
        dates = dates[dates.weekday < 5]

        n_days = len(dates)

        # Generate realistic price paths using geometric Brownian motion
        price_data = {}

        stock_params = {
            # (initial_price, annual_return, volatility)
            'AAPL': (150, 0.15, 0.25),
            'MSFT': (300, 0.12, 0.22),
            'META': (200, 0.10, 0.30),
            'NVDA': (400, 0.20, 0.35),
            'GOOGL': (120, 0.11, 0.24),
            'JPM': (140, 0.08, 0.20),
            'AMZN': (130, 0.13, 0.28),
            'V': (220, 0.14, 0.21),
            'WMT': (160, 0.06, 0.18),
            'TSLA': (200, 0.25, 0.45),
            'JNJ': (170, 0.07, 0.16),
            'PG': (150, 0.05, 0.15)
        }

        for ticker in self.config.tickers:
            if ticker in stock_params:
                initial_price, mu, sigma = stock_params[ticker]
            else:
                # Default parameters for any missing tickers
                initial_price, mu, sigma = (100, 0.10, 0.25)

            # Convert to daily parameters
            dt = 1 / 252  # Trading days per year
            daily_return = mu * dt
            daily_vol = sigma * np.sqrt(dt)

            # Generate random returns
            returns = np.random.normal(daily_return, daily_vol, n_days)

            # Create price series using cumulative product
            price_series = [initial_price]
            for i in range(n_days - 1):
                price_series.append(price_series[-1] * (1 + returns[i]))

            price_data[ticker] = price_series

        # Create DataFrame
        df = pd.DataFrame(price_data, index=dates)

        print(f"• Generated sample data from {df.index[0].date()} to {df.index[-1].date()}")
        return df

    def _get_cache_path(self) -> Path:
        """Generate cache filename based on config."""
        key = f"{self.config.start_date}|{self.config.end_date}|{'|'.join(self.config.tickers)}"
        md5 = hashlib.md5(key.encode("utf-8")).hexdigest()[:12]
        filename = f"prices_{self.config.start_date}_{self.config.end_date}_{md5}.csv"
        return self.config.cache_dir / filename


# ============================================================================
# Analysis Components
# ============================================================================

class ReturnAnalyzer:
    """Analyzes returns and correlations."""

    def __init__(self, prices: pd.DataFrame, config: PortfolioConfig):
        self.prices = prices
        self.config = config
        self.returns = self._calculate_returns()

    def _calculate_returns(self) -> pd.DataFrame:
        """Calculate simple daily returns."""
        return self.prices.pct_change().dropna()

    def get_correlation_matrix(self) -> pd.DataFrame:
        """Get correlation matrix of returns."""
        return self.returns.corr()

    def get_risk_return_profile(self) -> pd.DataFrame:
        """Calculate annualized risk/return profile for all assets."""
        profile = pd.DataFrame({
            "annual_return": self.returns.mean() * self.config.trading_days_per_year,
            "annual_volatility": self.returns.std() * np.sqrt(self.config.trading_days_per_year),
        })
        profile["sharpe_ratio"] = profile["annual_return"] / profile["annual_volatility"]
        return profile


# ============================================================================
# Portfolio Optimization
# ============================================================================

class PortfolioOptimizer:
    """Handles portfolio optimization strategies."""

    def __init__(self, returns: pd.DataFrame, config: PortfolioConfig):
        self.returns = returns
        self.config = config
        self.mean_returns = returns.mean() * config.trading_days_per_year
        self.cov_matrix = returns.cov() * config.trading_days_per_year
        self.n_assets = len(self.mean_returns)

    def optimize(self, strategy: str = "min_variance") -> pd.Series:
        """Optimize portfolio weights using specified strategy."""
        if strategy == "min_variance":
            objective = self._variance_objective
        elif strategy == "max_sharpe":
            objective = self._sharpe_objective
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        # Ensure feasible bounds
        cap = self.config.max_weight_per_asset
        if self.n_assets * cap < 1.0:
            cap = 1.0

        bounds = [(0.0, cap) for _ in range(self.n_assets)]
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        initial = np.ones(self.n_assets) / self.n_assets

        result = minimize(
            objective,
            initial,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-9}
        )

        if not result.success:
            print(f"• Optimization warning: {result.message}")
            return pd.Series(initial, index=self.returns.columns)

        return pd.Series(result.x, index=self.returns.columns)

    def _variance_objective(self, weights: np.ndarray) -> float:
        return float(weights.T @ self.cov_matrix @ weights)

    def _sharpe_objective(self, weights: np.ndarray) -> float:
        ret = float(weights.T @ self.mean_returns)
        vol = np.sqrt(float(weights.T @ self.cov_matrix @ weights))
        return -(ret / vol) if vol > 0 else 0.0


# ============================================================================
# Performance Evaluation
# ============================================================================

class PerformanceEvaluator:
    """Evaluates portfolio performance."""

    def __init__(self, returns: pd.DataFrame, config: PortfolioConfig):
        self.returns = returns
        self.config = config

    def evaluate_portfolio(self, weights: pd.Series) -> PortfolioMetrics:
        """Calculate comprehensive portfolio metrics."""
        portfolio_returns = self.returns @ weights

        # Annualized metrics
        annual_return = portfolio_returns.mean() * self.config.trading_days_per_year
        annual_vol = portfolio_returns.std() * np.sqrt(self.config.trading_days_per_year)
        sharpe_ratio = (annual_return / annual_vol) if annual_vol > 0 else 0.0

        # Drawdown calculation
        cumulative = (1.0 + portfolio_returns).cumprod()
        rolling_max = cumulative.cummax()
        drawdowns = (cumulative / rolling_max) - 1.0
        max_drawdown = drawdowns.min()

        return PortfolioMetrics(
            annual_return=float(annual_return),
            annual_volatility=float(annual_vol),
            sharpe_ratio=float(sharpe_ratio),
            max_drawdown=float(max_drawdown),
            weights=weights
        )


# ============================================================================
# Main Analysis Pipeline
# ============================================================================

class FinancialAnalysisPipeline:
    """Main pipeline orchestrating the entire analysis."""

    def __init__(self, config: Optional[PortfolioConfig] = None):
        self.config = config or PortfolioConfig()
        self.data_manager = DataManager(self.config)

    def run_analysis(self, use_cache: bool = True, safe_mode: bool = False) -> Dict:
        print("=" * 60)
        print("FINANCIAL PORTFOLIO ANALYSIS")
        print("=" * 60)
        print(f"Period: {self.config.start_date} to {self.config.end_date}")
        print(f"Analyzing {len(self.config.tickers)} stocks\n")

        # Step 1: Data Collection
        print("Step 1: Data Collection")
        prices = self.data_manager.get_price_data(use_cache=use_cache, safe_mode=safe_mode)

        if prices.empty:
            print("ERROR: No price data retrieved. Check tickers and date range.")
            return {}

        # Step 2: Return Analysis
        print("\nStep 2: Return Analysis")
        analyzer = ReturnAnalyzer(prices, self.config)
        corr = analyzer.get_correlation_matrix()
        profile = analyzer.get_risk_return_profile()

        # Step 3: Portfolio Optimization
        print("\nStep 3: Portfolio Optimization")
        optimizer = PortfolioOptimizer(analyzer.returns, self.config)

        portfolios = {
            "Min Variance": optimizer.optimize("min_variance"),
            "Max Sharpe": optimizer.optimize("max_sharpe"),
        }

        # Step 4: Performance Evaluation
        print("\nStep 4: Performance Evaluation")
        evaluator = PerformanceEvaluator(analyzer.returns, self.config)

        for name, weights in portfolios.items():
            metrics = evaluator.evaluate_portfolio(weights)
            print(f"\n{name} Portfolio:")
            print("  Top holdings:")
            for stock, weight in weights.sort_values(ascending=False).head(3).items():
                print(f"    - {stock}: {weight:.1%}")
            print(metrics)

        # Step 5: Create visualization
        print("\nStep 5: Creating Performance Chart")
        self._plot_performance(analyzer.returns, portfolios)

        print("\n" + "=" * 60)
        print("ANALYSIS COMPLETE")
        print("=" * 60)

        return {
            "prices": prices,
            "returns": analyzer.returns,
            "correlation_matrix": corr,
            "portfolios": portfolios,
            "risk_return_profile": profile,
        }

    def _plot_performance(self, returns: pd.DataFrame, portfolios: Dict[str, pd.Series]):
        """Plot portfolio performance comparison."""
        plt.figure(figsize=(12, 6))

        for name, weights in portfolios.items():
            portfolio_returns = returns @ weights
            cumulative = (1.0 + portfolio_returns).cumprod()
            plt.plot(cumulative.index, cumulative.values, label=name, linewidth=2)

        plt.title("Portfolio Performance Comparison")
        plt.xlabel("Date")
        plt.ylabel("Cumulative Return")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()


# ============================================================================
# Script entry point
# ============================================================================

if __name__ == "__main__":
    # Example: Run with default settings
    pipeline = FinancialAnalysisPipeline()
    results = pipeline.run_analysis(use_cache=True, safe_mode=True)  # safe_mode=True to avoid rate limits

    # Example: ASX Small Cap Analysis (uncomment to use)
    # asx_config = PortfolioConfig(
    #     start_date="2023-01-01",
    #     end_date="2024-12-31",
    #     tickers=["WEB.AX", "NEA.AX", "NXT.AX", "MP1.AX", "EML.AX"],
    #     max_weight_per_asset=0.4
    # )
    # asx_pipeline = FinancialAnalysisPipeline(asx_config)
    # asx_results = asx_pipeline.run_analysis(use_cache=True, safe_mode=True)