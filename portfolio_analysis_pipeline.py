# portfolio_analysis_pipeline.py

import time
import hashlib
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Dict
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import networkx as nx
from scipy.optimize import minimize

plt.rcParams["figure.autolayout"] = True
plt.rcParams["axes.grid"] = True


# ----------------------------- Config -----------------------------
@dataclass
class PortfolioConfig:
    start_date: str = "2023-01-01"
    end_date: str = "2024-12-31"
    tickers: Optional[List[str]] = None
    cache_dir: Path = Path("./data_cache")
    max_weight_per_asset: float = 0.4
    trading_days_per_year: int = 252

    def __post_init__(self):
        if self.tickers is None:
            self.tickers = [
                "AAPL", "MSFT", "META", "NVDA", "GOOGL", "JPM",
                "AMZN", "V", "WMT", "TSLA", "JNJ", "PG"
            ]
        self.cache_dir.mkdir(parents=True, exist_ok=True)


# -------------------------- Data Manager --------------------------
class DataManager:
    """Download prices with caching and safe-mode fallbacks."""

    def __init__(self, config: PortfolioConfig):
        self.config = config

    def get_price_data(self, use_cache: bool = True, safe_mode: bool = True) -> pd.DataFrame:
        cache_file = self._cache_path()
        if use_cache and cache_file.exists():
            print(f"• Loading cached data from {cache_file}")
            df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            if not df.empty:
                return df

        print(f"• Downloading data for {len(self.config.tickers)} tickers...")

        def _pick_series(df: pd.DataFrame, t: str) -> pd.Series:
            # MultiIndex
            if isinstance(df.columns, pd.MultiIndex):
                for field in ("Adj Close", "Close"):
                    try:
                        s = df[(t, field)]
                        s.name = t
                        return s
                    except KeyError:
                        pass
                    try:
                        s = df[(field, t)]
                        s.name = t
                        return s
                    except KeyError:
                        pass
                raise KeyError(f"No Close/Adj Close for {t}")
            # Single index
            for field in ("Adj Close", "Close"):
                if field in df.columns:
                    return df[field].rename(t)
            raise KeyError("No Close/Adj Close column present")

        if not safe_mode:
            raw = yf.download(
                self.config.tickers,
                start=self.config.start_date,
                end=self.config.end_date,
                auto_adjust=True,
                progress=False,
                group_by="ticker",
            )
            series = []
            for t in self.config.tickers:
                try:
                    series.append(_pick_series(raw, t))
                except KeyError:
                    print(f"  • Warning: missing price for {t}, skipping.")
            prices = pd.concat(series, axis=1) if series else pd.DataFrame()
        else:
            cols = []
            for i, t in enumerate(self.config.tickers):
                print(f"  - {t} ({i+1}/{len(self.config.tickers)})")
                try:
                    d = yf.download(
                        t,
                        start=self.config.start_date,
                        end=self.config.end_date,
                        auto_adjust=True,
                        progress=False,
                    )
                    if "Adj Close" in d.columns:
                        cols.append(d["Adj Close"].rename(t))
                    elif "Close" in d.columns:
                        cols.append(d["Close"].rename(t))
                    else:
                        print(f"    • No Close/Adj Close for {t}, skipped.")
                except Exception as e:
                    print(f"    • Download failed for {t}: {e}")
                time.sleep(1)
            prices = pd.concat(cols, axis=1) if cols else pd.DataFrame()

        prices = prices.dropna(how="all").dropna()
        if prices.empty:
            raise RuntimeError("No price data downloaded.")

        if use_cache:
            prices.to_csv(cache_file)
            print(f"• Data cached to {cache_file}")
        print(f"• Loaded {len(prices)} trading days for {len(prices.columns)} tickers.")
        return prices

    def _cache_path(self) -> Path:
        key = f"{self.config.start_date}|{self.config.end_date}|{'|'.join(self.config.tickers)}"
        md5 = hashlib.md5(key.encode("utf-8")).hexdigest()[:12]
        return self.config.cache_dir / f"prices_{self.config.start_date}_{self.config.end_date}_{md5}.csv"


# ------------------------- Return Analyzer ------------------------
class ReturnAnalyzer:
    def __init__(self, prices: pd.DataFrame, config: PortfolioConfig):
        self.prices = prices
        self.config = config
        self.returns = self.prices.pct_change().dropna()

    def correlation(self) -> pd.DataFrame:
        return self.returns.corr()

    def risk_return_profile(self) -> pd.DataFrame:
        tdpy = self.config.trading_days_per_year
        prof = pd.DataFrame({
            "annual_return": self.returns.mean() * tdpy,
            "annual_volatility": self.returns.std() * np.sqrt(tdpy),
        })
        prof["sharpe_ratio"] = prof["annual_return"] / prof["annual_volatility"]
        return prof

    def top_correlations(self, n=5):
        corr = self.correlation()
        pairs = []
        cols = corr.columns
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                pairs.append((cols[i], cols[j], float(corr.iloc[i, j])))
        pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        return pairs[:n]


# ----------------------- Portfolio Optimizer ----------------------
class PortfolioOptimizer:
    def __init__(self, returns: pd.DataFrame, config: PortfolioConfig):
        self.returns = returns
        self.config = config
        tdpy = config.trading_days_per_year
        self.mu = returns.mean() * tdpy
        self.cov = returns.cov() * tdpy
        self.n = len(self.mu)

    def optimize(self, strategy: str) -> pd.Series:
        if strategy == "min_variance":
            obj = lambda w: float(w.T @ self.cov @ w)
        elif strategy == "max_sharpe":
            def obj(w):
                vol = np.sqrt(float(w.T @ self.cov @ w))
                ret = float(w.T @ self.mu)
                return -(ret / vol) if vol > 0 else 0.0
        else:
            raise ValueError("strategy must be 'min_variance' or 'max_sharpe'")

        cap = self.config.max_weight_per_asset
        if self.n * cap < 1.0:  # keep feasible
            cap = 1.0
        bounds = [(0.0, cap)] * self.n
        cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        w0 = np.ones(self.n) / self.n

        res = minimize(obj, w0, method="SLSQP", bounds=bounds, constraints=cons)
        if not res.success:
            print(f"• Optimization warning: {res.message}")
            return pd.Series(w0, index=self.returns.columns)
        return pd.Series(res.x, index=self.returns.columns)


# ---------------------- Performance Evaluator ---------------------
@dataclass
class PortfolioMetrics:
    annual_return: float
    annual_volatility: float
    sharpe_ratio: float
    max_drawdown: float
    weights: pd.Series

    def __str__(self):
        return (f"Annual Return: {self.annual_return:.1%}\n"
                f"Annual Volatility: {self.annual_volatility:.1%}\n"
                f"Sharpe Ratio: {self.sharpe_ratio:.2f}\n"
                f"Max Drawdown: {self.max_drawdown:.1%}")


class PerformanceEvaluator:
    def __init__(self, returns: pd.DataFrame, config: PortfolioConfig):
        self.returns = returns
        self.config = config

    def evaluate(self, w: pd.Series) -> PortfolioMetrics:
        tdpy = self.config.trading_days_per_year
        port = self.returns @ w
        ann_ret = float(port.mean() * tdpy)
        ann_vol = float(port.std() * np.sqrt(tdpy))
        sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0
        curve = (1 + port).cumprod()
        mdd = float((curve / curve.cummax() - 1).min())
        return PortfolioMetrics(ann_ret, ann_vol, sharpe, mdd, w)


# --------------------------- Visualization ------------------------
class PortfolioVisualizer:
    def create_dashboard(self, corr: pd.DataFrame, metrics: Dict[str, PortfolioMetrics],
                         profile: pd.DataFrame, returns: pd.DataFrame):
        fig = plt.figure(figsize=(20, 14))

        # Correlation heatmap
        ax = plt.subplot(2, 3, 1)
        im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xticks(range(len(corr.columns))); ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=9)
        ax.set_yticks(range(len(corr.columns))); ax.set_yticklabels(corr.columns, fontsize=9)
        ax.set_title("Correlation Matrix")

        # Weights comparison
        ax = plt.subplot(2, 3, 2)
        names = list(metrics.keys())
        first = metrics[names[0]].weights
        x = np.arange(len(first))
        width = 0.8 / max(1, len(names))
        for i, (name, m) in enumerate(metrics.items()):
            ax.bar(x + (i - (len(names) - 1) / 2) * width, m.weights.values, width, label=name)
        ax.set_xticks(x); ax.set_xticklabels(first.index, rotation=45, ha="right", fontsize=9)
        ax.set_title("Portfolio Weights"); ax.legend(fontsize=9)

        # Risk-return scatter
        ax = plt.subplot(2, 3, 3)
        ax.scatter(profile["annual_volatility"], profile["annual_return"], s=80, alpha=0.7, label="Assets")
        for idx, row in profile.iterrows():
            ax.annotate(idx, (row["annual_volatility"], row["annual_return"]), xytext=(4, 4),
                        textcoords="offset points", fontsize=8)
        for name, m in metrics.items():
            ax.scatter(m.annual_volatility, m.annual_return, s=180, label=f"{name} Portfolio")
        ax.set_title("Risk–Return"); ax.legend(fontsize=9)

        # Correlation network
        ax = plt.subplot(2, 3, 4)
        G = nx.Graph(); cols = corr.columns; th = 0.3
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                c = float(corr.iloc[i, j])
                if abs(c) > th:
                    G.add_edge(cols[i], cols[j], weight=abs(c))
        if G.number_of_edges() > 0:
            pos = nx.spring_layout(G, seed=42)
            nx.draw(G, pos, with_labels=True, node_color="lightblue", ax=ax)
        ax.set_title(f"Correlation Network (|r|>{th})")

        # Cumulative returns
        ax = plt.subplot(2, 3, 5)
        for name, m in metrics.items():
            curve = (1 + (returns @ m.weights)).cumprod()
            ax.plot(curve.index, curve.values, label=name)
        ax.set_title("Portfolio Performance"); ax.legend(fontsize=9)

        # Metrics bars
        ax = plt.subplot(2, 3, 6)
        cats = ["Return", "Vol", "Sharpe", "MaxDD"]
        x = np.arange(len(cats)); width = 0.8 / max(1, len(metrics))
        for i, (name, m) in enumerate(metrics.items()):
            vals = [m.annual_return * 100, m.annual_volatility * 100, m.sharpe_ratio, abs(m.max_drawdown) * 100]
            ax.bar(x + (i - (len(metrics) - 1) / 2) * width, vals, width, label=name)
        ax.set_xticks(x); ax.set_xticklabels(cats)
        ax.set_title("Metrics"); ax.legend(fontsize=9)

        plt.tight_layout(); plt.show()


# ------------------------------ Pipeline --------------------------
class FinancialAnalysisPipeline:
    def __init__(self, config: Optional[PortfolioConfig] = None):
        self.config = config or PortfolioConfig()
        self.data = DataManager(self.config)
        self.viz = PortfolioVisualizer()

    def run(self, use_cache=True, safe_mode=True):
        print("=" * 60)
        print("FINANCIAL CORRELATION & PORTFOLIO ANALYSIS")
        print("=" * 60)
        print(f"Period: {self.config.start_date} to {self.config.end_date}")
        print(f"Analyzing {len(self.config.tickers)} stocks\n")

        prices = self.data.get_price_data(use_cache=use_cache, safe_mode=safe_mode)
        analyzer = ReturnAnalyzer(prices, self.config)
        corr = analyzer.correlation()
        profile = analyzer.risk_return_profile()
        for a, b, c in analyzer.top_correlations():
            print(f"{a} <-> {b}: {c:.3f}")

        opt = PortfolioOptimizer(analyzer.returns, self.config)
        portfolios = {
            "Min Variance": opt.optimize("min_variance"),
            "Max Sharpe": opt.optimize("max_sharpe"),
        }

        evalr = PerformanceEvaluator(analyzer.returns, self.config)
        metrics = {name: evalr.evaluate(w) for name, w in portfolios.items()}

        for name, m in metrics.items():
            print(f"\n{name} Portfolio\n{m}")
            print("Top holdings:\n", m.weights.sort_values(ascending=False).head(3))

        self.viz.create_dashboard(corr, metrics, profile, analyzer.returns)

        # Save weights
        pd.DataFrame({k: v for k, v in portfolios.items()}).to_csv("portfolio_weights.csv")
        print("\n• Saved weights -> portfolio_weights.csv")

        return {"prices": prices, "returns": analyzer.returns, "metrics": metrics}


# ------------------------------ Runner ----------------------------
if __name__ == "__main__":
    pipeline = FinancialAnalysisPipeline()
    pipeline.run(use_cache=True, safe_mode=True)