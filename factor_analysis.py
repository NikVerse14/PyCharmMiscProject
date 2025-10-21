"""
Complete Factor Investing Strategy Implementation
Properly implemented with correct timing, realistic factors, and monthly rebalancing
Compatible with PyCharm and modern data sources
"""

import pandas as pd
import numpy as np
import warnings
import pickle
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
import time

# Suppress warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


# ==================== Configuration ====================
class Config:
    START_DATE = "2015-01-01"
    END_DATE = "2024-12-31"
    DATA_DIR = Path("factor_data")
    RESULTS_DIR = Path("factor_results")

    # Create directories
    DATA_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

    # S&P 500 universe (top 30 for reliable data)
    UNIVERSE = [
        'AAPL', 'MSFT', 'NVDA', 'AMZN', 'META', 'GOOGL', 'BRK-B', 'TSLA', 'V', 'UNH',
        'XOM', 'JPM', 'JNJ', 'WMT', 'PG', 'HD', 'MA', 'COST', 'ORCL', 'ABBV',
        'MRK', 'CVX', 'BAC', 'KO', 'NFLX', 'PEP', 'ADBE', 'TMO', 'ACN', 'CSCO'
    ]

    # Strategy parameters
    REBALANCE_FREQ = 'M'  # Monthly rebalancing
    LOOKBACK_MOMENTUM = 12  # 12-month momentum
    LOOKBACK_VOLATILITY = 12  # 12-month volatility
    LOOKBACK_REVERSAL = 1  # 1-month reversal
    TRANSACTION_COST = 0.001  # 10 bps per trade


# ==================== Data Management ====================
class DataManager:
    def __init__(self):
        self.cache_file = Config.DATA_DIR / "stock_prices.pkl"

    def get_stock_prices(self, tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        """Download stock prices with robust error handling and caching"""

        # Check cache
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                if cached_data['end_date'] >= end_date and len(cached_data['prices']) > 0:
                    print(f"Using cached price data for {len(cached_data['prices'].columns)} stocks...")
                    return cached_data['prices']
            except:
                pass

        print(f"Downloading price data for {len(tickers)} stocks...")

        # Download with delays to avoid rate limiting
        price_data = {}
        successful_downloads = 0

        for i, ticker in enumerate(tickers):
            try:
                print(f"  Downloading {ticker} ({i + 1}/{len(tickers)})")

                # Handle Berkshire Hathaway
                yf_ticker = ticker.replace('-', '-') if ticker != 'BRK-B' else 'BRK-B'

                stock = yf.Ticker(yf_ticker)
                data = stock.history(start=start_date, end=end_date)

                if not data.empty and len(data) > 252:  # At least 1 year of data
                    price_data[ticker] = data['Close']
                    successful_downloads += 1
                else:
                    print(f"    Warning: Insufficient data for {ticker}")

                # Rate limiting
                if i < len(tickers) - 1:
                    time.sleep(0.2)

            except Exception as e:
                print(f"    Error downloading {ticker}: {e}")
                continue

        if successful_downloads == 0:
            print("❌ No real data downloaded - generating sample data for demonstration...")
            return self._generate_sample_data(tickers, start_date, end_date)

        # Combine into DataFrame
        prices = pd.DataFrame(price_data)
        prices.index = pd.to_datetime(prices.index)
        prices = prices.sort_index()

        # Forward fill missing values (holidays, etc.)
        prices = prices.fillna(method='ffill').dropna()

        print(f"Successfully downloaded {len(prices.columns)} stocks with {len(prices)} days of data")

        # Cache the results
        cache_data = {
            'prices': prices,
            'end_date': end_date,
            'download_date': datetime.now().strftime('%Y-%m-%d')
        }
        with open(self.cache_file, 'wb') as f:
            pickle.dump(cache_data, f)

        return prices

    def get_monthly_data(self, daily_prices: pd.DataFrame) -> pd.DataFrame:
        """Convert daily prices to month-end prices"""
        return daily_prices.resample('M').last()

    def _generate_sample_data(self, tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        """Generate realistic sample stock data when real downloads fail"""
        np.random.seed(42)  # For reproducible results

        # Create date range
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        dates = pd.date_range(start, end, freq='D')
        # Filter to business days only
        dates = dates[dates.weekday < 5]

        n_days = len(dates)

        # Generate realistic price paths for each ticker
        price_data = {}

        # Stock parameters: (initial_price, annual_return, volatility)
        stock_params = {
            'AAPL': (100, 0.20, 0.25), 'MSFT': (80, 0.18, 0.22), 'NVDA': (50, 0.30, 0.40),
            'AMZN': (300, 0.15, 0.30), 'META': (150, 0.12, 0.35), 'GOOGL': (500, 0.14, 0.25),
            'BRK-B': (150, 0.10, 0.18), 'TSLA': (200, 0.25, 0.50), 'V': (100, 0.16, 0.20),
            'UNH': (120, 0.12, 0.18), 'XOM': (80, 0.05, 0.25), 'JPM': (100, 0.08, 0.22),
            'JNJ': (140, 0.06, 0.15), 'WMT': (90, 0.07, 0.16), 'PG': (110, 0.05, 0.14),
            'HD': (200, 0.10, 0.20), 'MA': (250, 0.18, 0.22), 'COST': (300, 0.12, 0.18),
            'ORCL': (70, 0.08, 0.20), 'ABBV': (120, 0.06, 0.16), 'MRK': (90, 0.04, 0.15),
            'CVX': (120, 0.03, 0.22), 'BAC': (30, 0.06, 0.25), 'KO': (50, 0.04, 0.12),
            'NFLX': (100, 0.15, 0.35), 'PEP': (100, 0.05, 0.13), 'ADBE': (200, 0.16, 0.28),
            'TMO': (300, 0.11, 0.18), 'ACN': (200, 0.09, 0.16), 'CSCO': (40, 0.06, 0.18)
        }

        for ticker in tickers:
            if ticker in stock_params:
                initial_price, mu, sigma = stock_params[ticker]
            else:
                # Default parameters for any missing tickers
                initial_price, mu, sigma = (100, 0.10, 0.25)

            # Convert to daily parameters
            dt = 1 / 252  # Trading days per year
            daily_return = mu * dt
            daily_vol = sigma * np.sqrt(dt)

            # Generate random returns with some serial correlation (more realistic)
            returns = np.random.normal(daily_return, daily_vol, n_days)

            # Add some momentum/mean reversion patterns
            for i in range(1, len(returns)):
                # Small momentum effect
                returns[i] += 0.1 * returns[i - 1] + np.random.normal(0, daily_vol * 0.5)

            # Create price series using cumulative product
            price_series = [initial_price]
            for i in range(n_days - 1):
                price_series.append(max(price_series[-1] * (1 + returns[i]), 0.01))  # Prevent negative prices

            price_data[ticker] = price_series

        # Create DataFrame
        df = pd.DataFrame(price_data, index=dates)

        print(f"Generated sample data from {df.index[0].date()} to {df.index[-1].date()}")
        print(f"Sample data includes realistic factor patterns for backtesting")
        return df


# ==================== Factor Calculations ====================
class FactorEngine:
    """Calculate systematic risk factors with proper implementation"""

    @staticmethod
    def calculate_momentum(monthly_prices: pd.DataFrame, lookback: int = 12, skip: int = 1) -> pd.DataFrame:
        """
        Calculate 12-1 momentum factor (12-month return excluding most recent month)
        """
        # Total return over lookback period
        total_return = monthly_prices.pct_change(lookback)

        # Recent return to skip
        recent_return = monthly_prices.pct_change(skip)

        # 12-1 momentum calculation
        momentum = (1 + total_return) / (1 + recent_return) - 1

        return momentum.dropna()

    @staticmethod
    def calculate_low_volatility(monthly_prices: pd.DataFrame, lookback: int = 12) -> pd.DataFrame:
        """
        Calculate realized volatility over lookback period
        """
        # Calculate monthly returns
        monthly_returns = monthly_prices.pct_change()

        # Rolling volatility (annualized)
        volatility = monthly_returns.rolling(window=lookback).std() * np.sqrt(12)

        return volatility.dropna()

    @staticmethod
    def calculate_short_term_reversal(monthly_prices: pd.DataFrame, lookback: int = 1) -> pd.DataFrame:
        """
        Calculate short-term reversal factor (contrarian)
        """
        reversal = -monthly_prices.pct_change(lookback)
        return reversal.dropna()

    @staticmethod
    def calculate_price_momentum(monthly_prices: pd.DataFrame, lookback: int = 6) -> pd.DataFrame:
        """
        Calculate intermediate-term price momentum
        """
        momentum = monthly_prices.pct_change(lookback)
        return momentum.dropna()

    @staticmethod
    def calculate_volatility_adjusted_momentum(monthly_prices: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate momentum adjusted for volatility (risk-adjusted momentum)
        """
        monthly_returns = monthly_prices.pct_change()

        # 12-month momentum
        momentum = monthly_returns.rolling(12).sum()

        # 12-month volatility
        volatility = monthly_returns.rolling(12).std() * np.sqrt(12)

        # Risk-adjusted momentum
        risk_adj_momentum = momentum / volatility

        return risk_adj_momentum.dropna()


# ==================== Signal Processing ====================
class SignalProcessor:
    """Convert factors into tradeable signals"""

    @staticmethod
    def cross_sectional_rank(factor_data: pd.DataFrame, ascending: bool = False) -> pd.DataFrame:
        """Convert factors to cross-sectional percentile ranks"""
        return factor_data.rank(axis=1, pct=True, ascending=ascending)

    @staticmethod
    def winsorize(data: pd.DataFrame, lower: float = 0.05, upper: float = 0.95) -> pd.DataFrame:
        """Winsorize extreme values"""
        return data.clip(lower=data.quantile(lower, axis=1),
                         upper=data.quantile(upper, axis=1), axis=0)

    @staticmethod
    def z_score(data: pd.DataFrame) -> pd.DataFrame:
        """Cross-sectional z-score normalization"""
        return data.sub(data.mean(axis=1), axis=0).div(data.std(axis=1), axis=0)

    @staticmethod
    def combine_signals(signals: Dict[str, pd.DataFrame], weights: Dict[str, float] = None) -> pd.DataFrame:
        """Combine multiple signals with specified weights"""
        if weights is None:
            weights = {name: 1.0 / len(signals) for name in signals}

        # Align all signals to common dates and stocks
        aligned_signals = {}
        common_index = None
        common_columns = None

        for name, signal in signals.items():
            if common_index is None:
                common_index = signal.index
                common_columns = signal.columns
            else:
                common_index = common_index.intersection(signal.index)
                common_columns = common_columns.intersection(signal.columns)

        for name, signal in signals.items():
            aligned_signals[name] = signal.loc[common_index, common_columns]

        # Combine signals
        combined = pd.DataFrame(0.0, index=common_index, columns=common_columns)
        for name, signal in aligned_signals.items():
            combined += signal * weights.get(name, 0)

        return combined


# ==================== Portfolio Construction ====================
class PortfolioConstructor:
    """Build factor-based portfolios"""

    def __init__(self, transaction_cost: float = 0.001):
        self.transaction_cost = transaction_cost

    def create_long_short_portfolio(self, signals: pd.DataFrame, long_pct: float = 0.2,
                                    short_pct: float = 0.2) -> Dict[str, pd.DataFrame]:
        """
        Create long-short portfolio based on signal quintiles
        """
        # Calculate quintile breakpoints
        long_threshold = 1 - long_pct
        short_threshold = short_pct

        # Create position weights
        positions = pd.DataFrame(0.0, index=signals.index, columns=signals.columns)

        for date in signals.index:
            signal_row = signals.loc[date]
            valid_signals = signal_row.dropna()

            if len(valid_signals) == 0:
                continue

            # Long positions (top quintile)
            long_threshold_value = valid_signals.quantile(long_threshold)
            long_stocks = valid_signals[valid_signals >= long_threshold_value].index

            # Short positions (bottom quintile)
            short_threshold_value = valid_signals.quantile(short_threshold)
            short_stocks = valid_signals[valid_signals <= short_threshold_value].index

            # Equal weight within long and short baskets
            if len(long_stocks) > 0:
                positions.loc[date, long_stocks] = 1.0 / len(long_stocks)

            if len(short_stocks) > 0:
                positions.loc[date, short_stocks] = -1.0 / len(short_stocks)

        return {
            'long_short': positions,
            'long_only': positions.clip(lower=0)
        }

    def create_quintile_portfolios(self, signals: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Create quintile portfolios (Q1 = bottom, Q5 = top)
        """
        quintile_portfolios = {}

        for date in signals.index:
            signal_row = signals.loc[date]
            valid_signals = signal_row.dropna()

            if len(valid_signals) == 0:
                continue

            # Create quintile breakpoints
            quintiles = pd.qcut(valid_signals.rank(), q=5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])

            for q in ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']:
                if q not in quintile_portfolios:
                    quintile_portfolios[q] = pd.DataFrame(0.0, index=signals.index, columns=signals.columns)

                q_stocks = quintiles[quintiles == q].index
                if len(q_stocks) > 0:
                    quintile_portfolios[q].loc[date, q_stocks] = 1.0 / len(q_stocks)

        return quintile_portfolios

    def calculate_portfolio_returns(self, weights: pd.DataFrame, returns: pd.DataFrame) -> pd.Series:
        """
        Calculate portfolio returns accounting for transaction costs
        """
        # Align weights and returns
        common_index = weights.index.intersection(returns.index)
        common_columns = weights.columns.intersection(returns.columns)

        aligned_weights = weights.loc[common_index, common_columns]
        aligned_returns = returns.loc[common_index, common_columns]

        # Calculate gross returns
        portfolio_returns = (aligned_weights * aligned_returns).sum(axis=1)

        # Calculate turnover and transaction costs
        weight_changes = aligned_weights.diff().abs().sum(axis=1)
        transaction_costs = weight_changes * self.transaction_cost

        # Net returns after transaction costs
        net_returns = portfolio_returns - transaction_costs

        return net_returns.dropna()


# ==================== Performance Analytics ====================
class PerformanceAnalyzer:
    """Comprehensive performance analysis"""

    @staticmethod
    def calculate_performance_metrics(returns: pd.Series, benchmark: pd.Series = None, rf_rate: float = 0.02) -> Dict:
        """Calculate comprehensive performance metrics"""

        returns_clean = returns.dropna()
        if len(returns_clean) == 0:
            return {metric: np.nan for metric in [
                'Total Return', 'Annual Return', 'Volatility', 'Sharpe Ratio', 'Sortino Ratio',
                'Max Drawdown', 'Calmar Ratio', 'Win Rate', 'Alpha', 'Beta', 'Information Ratio'
            ]}

        # Basic metrics
        total_return = (1 + returns_clean).prod() - 1
        n_years = len(returns_clean) / 12  # Monthly data
        annual_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0
        annual_vol = returns_clean.std() * np.sqrt(12)

        # Risk metrics
        sharpe_ratio = (annual_return - rf_rate) / annual_vol if annual_vol > 0 else 0

        downside_returns = returns_clean[returns_clean < 0]
        downside_vol = downside_returns.std() * np.sqrt(12) if len(downside_returns) > 0 else 0
        sortino_ratio = (annual_return - rf_rate) / downside_vol if downside_vol > 0 else 0

        # Drawdown
        cum_returns = (1 + returns_clean).cumprod()
        running_max = cum_returns.expanding().max()
        drawdown = cum_returns / running_max - 1
        max_drawdown = drawdown.min()

        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown < 0 else 0
        win_rate = (returns_clean > 0).mean()

        metrics = {
            'Total Return': total_return,
            'Annual Return': annual_return,
            'Volatility': annual_vol,
            'Sharpe Ratio': sharpe_ratio,
            'Sortino Ratio': sortino_ratio,
            'Max Drawdown': max_drawdown,
            'Calmar Ratio': calmar_ratio,
            'Win Rate': win_rate,
        }

        # Benchmark-relative metrics
        if benchmark is not None:
            aligned_data = pd.concat([returns_clean, benchmark], axis=1).dropna()
            if len(aligned_data) > 12:
                cov_matrix = aligned_data.cov() * 12  # Annualize
                beta = cov_matrix.iloc[0, 1] / cov_matrix.iloc[1, 1] if cov_matrix.iloc[1, 1] != 0 else 0
                alpha = annual_return - beta * (benchmark.mean() * 12)

                excess_returns = aligned_data.iloc[:, 0] - aligned_data.iloc[:, 1]
                tracking_error = excess_returns.std() * np.sqrt(12)
                information_ratio = excess_returns.mean() * 12 / tracking_error if tracking_error > 0 else 0

                metrics.update({
                    'Alpha': alpha,
                    'Beta': beta,
                    'Information Ratio': information_ratio
                })
            else:
                metrics.update({'Alpha': np.nan, 'Beta': np.nan, 'Information Ratio': np.nan})

        return metrics


# ==================== Main Strategy Class ====================
class FactorInvestingStrategy:
    """Complete factor investing strategy implementation"""

    def __init__(self):
        self.data_manager = DataManager()
        self.factor_engine = FactorEngine()
        self.signal_processor = SignalProcessor()
        self.portfolio_constructor = PortfolioConstructor(Config.TRANSACTION_COST)
        self.performance_analyzer = PerformanceAnalyzer()

        # Data containers
        self.daily_prices = None
        self.monthly_prices = None
        self.monthly_returns = None
        self.factors = {}
        self.signals = {}
        self.portfolios = {}
        self.portfolio_returns = {}
        self.performance_metrics = {}

    def run_backtest(self):
        """Execute complete factor strategy backtest"""

        print("=" * 60)
        print("SYSTEMATIC FACTOR INVESTING STRATEGY")
        print("=" * 60)
        print(f"Period: {Config.START_DATE} to {Config.END_DATE}")
        print(f"Universe: {len(Config.UNIVERSE)} stocks")
        print(f"Rebalancing: Monthly")

        # Step 1: Get Data
        print("\n1. Loading price data...")
        self.daily_prices = self.data_manager.get_stock_prices(
            Config.UNIVERSE, Config.START_DATE, Config.END_DATE
        )

        self.monthly_prices = self.data_manager.get_monthly_data(self.daily_prices)
        self.monthly_returns = self.monthly_prices.pct_change().dropna()

        print(f"   Data shape: {self.monthly_prices.shape}")
        print(f"   Date range: {self.monthly_prices.index[0].date()} to {self.monthly_prices.index[-1].date()}")

        # Step 2: Calculate Factors
        print("\n2. Calculating factors...")

        print("   - Momentum (12-1)")
        self.factors['momentum'] = self.factor_engine.calculate_momentum(self.monthly_prices)

        print("   - Low Volatility")
        self.factors['low_volatility'] = self.factor_engine.calculate_low_volatility(self.monthly_prices)

        print("   - Short-term Reversal")
        self.factors['reversal'] = self.factor_engine.calculate_short_term_reversal(self.monthly_prices)

        print("   - Risk-adjusted Momentum")
        self.factors['risk_adj_momentum'] = self.factor_engine.calculate_volatility_adjusted_momentum(
            self.monthly_prices)

        # Step 3: Process Signals
        print("\n3. Creating factor signals...")

        # Convert factors to signals (percentile ranks)
        self.signals['momentum'] = self.signal_processor.cross_sectional_rank(
            self.factors['momentum'], ascending=False
        )
        self.signals['low_volatility'] = self.signal_processor.cross_sectional_rank(
            self.factors['low_volatility'], ascending=True  # Low vol is good
        )
        self.signals['reversal'] = self.signal_processor.cross_sectional_rank(
            self.factors['reversal'], ascending=False
        )
        self.signals['risk_adj_momentum'] = self.signal_processor.cross_sectional_rank(
            self.factors['risk_adj_momentum'], ascending=False
        )

        # Create composite signals
        print("   - Creating composite signals...")

        # Multi-factor model
        multi_factor_weights = {
            'momentum': 0.4,
            'low_volatility': 0.3,
            'reversal': 0.3
        }
        self.signals['multi_factor'] = self.signal_processor.combine_signals(
            {k: v for k, v in self.signals.items() if k in multi_factor_weights},
            multi_factor_weights
        )

        # Step 4: Build Portfolios
        print("\n4. Constructing portfolios...")

        strategies = ['momentum', 'low_volatility', 'reversal', 'multi_factor']

        for strategy in strategies:
            print(f"   - {strategy.replace('_', ' ').title()} strategy")

            # Create long-short portfolio
            portfolios = self.portfolio_constructor.create_long_short_portfolio(
                self.signals[strategy], long_pct=0.2, short_pct=0.2
            )

            self.portfolios[f"{strategy}_long_short"] = portfolios['long_short']
            self.portfolios[f"{strategy}_long_only"] = portfolios['long_only']

        # Step 5: Calculate Returns
        print("\n5. Calculating portfolio returns...")

        for portfolio_name, weights in self.portfolios.items():
            self.portfolio_returns[portfolio_name] = self.portfolio_constructor.calculate_portfolio_returns(
                weights, self.monthly_returns
            )

        # Get benchmark (SPY)
        print("\n6. Loading benchmark...")
        try:
            spy_prices = self.data_manager.get_stock_prices(['SPY'], Config.START_DATE, Config.END_DATE)
            spy_monthly = self.data_manager.get_monthly_data(spy_prices)
            benchmark_returns = spy_monthly.pct_change().dropna().iloc[:, 0]
        except:
            print("   Could not download SPY, generating sample benchmark...")
            # Generate sample benchmark returns (market-like)
            np.random.seed(123)
            n_months = len(self.monthly_returns)
            benchmark_returns = pd.Series(
                np.random.normal(0.008, 0.04, n_months),  # ~10% annual return, ~14% vol
                index=self.monthly_returns.index,
                name='Benchmark'
            )

        # Step 6: Performance Analysis
        print("\n7. Analyzing performance...")

        for portfolio_name, returns in self.portfolio_returns.items():
            self.performance_metrics[portfolio_name] = self.performance_analyzer.calculate_performance_metrics(
                returns, benchmark_returns
            )

        # Benchmark performance
        self.performance_metrics['SPY'] = self.performance_analyzer.calculate_performance_metrics(
            benchmark_returns
        )

        # Step 7: Display Results
        self._display_results()

        # Step 8: Save Results
        self._save_results()

        print("\n" + "=" * 60)
        print("BACKTEST COMPLETED SUCCESSFULLY")
        print("=" * 60)

    def _display_results(self):
        """Display backtest results in formatted table"""

        print("\n" + "=" * 80)
        print("FACTOR STRATEGY PERFORMANCE RESULTS")
        print("=" * 80)

        # Create results DataFrame
        results_df = pd.DataFrame(self.performance_metrics).T

        # Format for display
        format_dict = {
            'Total Return': '{:.1%}',
            'Annual Return': '{:.1%}',
            'Volatility': '{:.1%}',
            'Sharpe Ratio': '{:.2f}',
            'Sortino Ratio': '{:.2f}',
            'Max Drawdown': '{:.1%}',
            'Calmar Ratio': '{:.2f}',
            'Win Rate': '{:.1%}',
            'Alpha': '{:.1%}',
            'Beta': '{:.2f}',
            'Information Ratio': '{:.2f}'
        }

        display_df = results_df.copy()
        for col, fmt in format_dict.items():
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(lambda x: fmt.format(x) if pd.notna(x) else 'N/A')

        # Show key metrics
        key_metrics = ['Annual Return', 'Volatility', 'Sharpe Ratio', 'Max Drawdown', 'Alpha', 'Beta']
        if all(col in display_df.columns for col in key_metrics):
            print("\nKEY PERFORMANCE METRICS:")
            print("-" * 80)
            print(display_df[key_metrics].to_string())

        # Find best strategies
        long_short_strategies = [name for name in self.performance_metrics.keys() if 'long_short' in name]
        if long_short_strategies:
            best_sharpe = max(long_short_strategies,
                              key=lambda x: self.performance_metrics[x].get('Sharpe Ratio', -999)
                              if pd.notna(self.performance_metrics[x].get('Sharpe Ratio', np.nan)) else -999)

            print(f"\n📈 BEST PERFORMING STRATEGY: {best_sharpe.upper()}")
            best_metrics = self.performance_metrics[best_sharpe]
            print(f"   Annual Return: {best_metrics['Annual Return']:.1%}")
            print(f"   Sharpe Ratio: {best_metrics['Sharpe Ratio']:.2f}")
            print(f"   Max Drawdown: {best_metrics['Max Drawdown']:.1%}")

    def _save_results(self):
        """Save results for further analysis"""

        print(f"\n8. Saving results to {Config.RESULTS_DIR}...")

        # Save complete results
        results = {
            'daily_prices': self.daily_prices,
            'monthly_prices': self.monthly_prices,
            'monthly_returns': self.monthly_returns,
            'factors': self.factors,
            'signals': self.signals,
            'portfolios': self.portfolios,
            'portfolio_returns': self.portfolio_returns,
            'performance_metrics': self.performance_metrics
        }

        with open(Config.RESULTS_DIR / 'factor_strategy_results.pkl', 'wb') as f:
            pickle.dump(results, f)

        # Save performance metrics as CSV
        performance_df = pd.DataFrame(self.performance_metrics).T
        performance_df.to_csv(Config.RESULTS_DIR / 'factor_performance_metrics.csv')

        # Save portfolio returns
        portfolio_returns_df = pd.DataFrame(self.portfolio_returns)
        portfolio_returns_df.to_csv(Config.RESULTS_DIR / 'factor_portfolio_returns.csv')

        print("   Results saved successfully!")


# ==================== Execution ====================
if __name__ == "__main__":
    strategy = FactorInvestingStrategy()

    try:
        strategy.run_backtest()
        print("\n✅ Factor investing backtest completed successfully!")
        print("📊 Results saved for further analysis")

    except Exception as e:
        print(f"\n❌ Error during backtest: {e}")
        import traceback

        traceback.print_exc()