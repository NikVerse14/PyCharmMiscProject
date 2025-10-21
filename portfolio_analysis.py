# analysis.py
from dataclasses import dataclass
from typing import List, Dict
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize
import matplotlib.pyplot as plt

plt.rcParams["figure.autolayout"] = True
plt.rcParams["axes.grid"] = True


# ----------------------------- config -----------------------------
@dataclass
class Config:
    start: str = "2023-01-01"
    end: str = "2024-12-31"
    tickers: List[str] = None
    max_weight: float = 0.4
    tdpy: int = 252  # trading days per year

    def __post_init__(self):
        if self.tickers is None:
            self.tickers = [
                "AAPL",
                "MSFT",
                "META",
                "NVDA",
                "GOOGL",
                "JPM",
                "AMZN",
                "V",
                "WMT",
                "TSLA",
                "JNJ",
                "PG",
            ]


# --------------------- yfinance-safe downloader -------------------
def fetch_prices(tickers: List[str], start: str, end: str) -> pd.DataFrame:
    """Download adjusted close if available, otherwise close. Robust to
    MultiIndex shapes returned by yfinance."""
    data = yf.download(
        tickers, start=start, end=end, auto_adjust=True, progress=False, group_by="ticker"
    )

    def pick(df: pd.DataFrame, t: str) -> pd.Series:
        if isinstance(df.columns, pd.MultiIndex):
            # try (field, ticker) then (ticker, field)
            for field in ("Adj Close", "Close"):
                try:
                    return df[(field, t)].rename(t)
                except KeyError:
                    try:
                        return df[(t, field)].rename(t)
                    except KeyError:
                        pass
            raise KeyError(f"No Close/Adj Close found for {t}")
        else:
            # single-ticker single-index
            for field in ("Adj Close", "Close"):
                if field in df.columns:
                    return df[field].rename(t)
            raise KeyError("No Close/Adj Close column present")

    series = []
    for t in tickers:
        try:
            s = pick(data, t)
            series.append(s)
        except KeyError:
            print(f"• Warning: skipping {t} (no price column found)")
    prices = pd.concat(series, axis=1).dropna(how="all").dropna()
    if prices.empty:
        raise RuntimeError("No price data downloaded.")
    return prices


# -------------------------- analysis bits -------------------------
def daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change().dropna()


def resolve_weight_cap(max_weight: float, asset_count: int) -> tuple[float, bool]:
    """Return a feasible per-asset cap and whether it was adjusted."""
    if asset_count <= 0:
        raise ValueError("asset_count must be positive.")
    if max_weight <= 0:
        raise ValueError("max_weight must be positive.")
    if asset_count * max_weight >= 1.0 or np.isclose(asset_count * max_weight, 1.0):
        return max_weight, False
    return 1.0 / asset_count, True


def optimize_weights(returns: pd.DataFrame, strategy: str, max_w: float, tdpy: int) -> pd.Series:
    mu = returns.mean() * tdpy
    cov = returns.cov() * tdpy
    n = returns.shape[1]

    cap, adjusted = resolve_weight_cap(max_w, n)
    if adjusted:
        print(
            f"• Adjusted max_weight from {max_w:.1%} to {cap:.1%} "
            f"to satisfy feasibility for {n} assets."
        )
    bounds = [(0.0, cap)] * n
    cons = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)
    w0 = np.ones(n) / n

    def var_obj(w):
        return float(w.T @ cov @ w)

    def neg_sharpe(w):
        vol = np.sqrt(float(w.T @ cov @ w))
        ret = float(w.T @ mu)
        return -(ret / vol) if vol > 0 else 0.0

    obj = var_obj if strategy == "min_var" else neg_sharpe
    res = minimize(obj, w0, method="SLSQP", bounds=bounds, constraints=cons, options={"ftol": 1e-9})
    if not res.success:
        print("• Optimization warning:", res.message)
    return pd.Series(res.x if res.success else w0, index=returns.columns)


def metrics(returns: pd.Series, tdpy: int) -> Dict[str, float]:
    ann_ret = float(returns.mean() * tdpy)
    ann_vol = float(returns.std() * np.sqrt(tdpy))
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0
    curve = (1 + returns).cumprod()
    mdd = float((curve / curve.cummax() - 1.0).min())
    return dict(annual_return=ann_ret, annual_vol=ann_vol, sharpe=sharpe, max_dd=mdd)


# ----------------------------- main -------------------------------
def main():
    cfg = Config()
    print(f"Downloading {len(cfg.tickers)} tickers...")
    prices = fetch_prices(cfg.tickers, cfg.start, cfg.end)
    rets = daily_returns(prices)
    print("Days:", len(rets), "Assets:", rets.shape[1])

    w_min = optimize_weights(rets, "min_var", cfg.max_weight, cfg.tdpy)
    w_ms = optimize_weights(rets, "max_sharpe", cfg.max_weight, cfg.tdpy)

    port_min = rets @ w_min
    port_ms = rets @ w_ms

    m_min = metrics(port_min, cfg.tdpy)
    m_ms = metrics(port_ms, cfg.tdpy)

    print("\nMin-Variance top weights:")
    print(w_min.sort_values(ascending=False).head(5).apply(lambda x: f"{x:.1%}"))
    print(
        f"Return {m_min['annual_return']:.1%}, Vol {m_min['annual_vol']:.1%}, Sharpe {m_min['sharpe']:.2f}, MaxDD {m_min['max_dd']:.1%}"
    )

    print("\nMax-Sharpe top weights:")
    print(w_ms.sort_values(ascending=False).head(5).apply(lambda x: f"{x:.1%}"))
    print(
        f"Return {m_ms['annual_return']:.1%}, Vol {m_ms['annual_vol']:.1%}, Sharpe {m_ms['sharpe']:.2f}, MaxDD {m_ms['max_dd']:.1%}"
    )

    # quick performance plot
    plt.figure()
    (1 + port_min).cumprod().plot(label="Min Var", linewidth=2)
    (1 + port_ms).cumprod().plot(label="Max Sharpe", linewidth=2)
    plt.title("Portfolio Performance")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
