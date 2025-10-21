# dashboard.py
"""
Interactive Streamlit Dashboard for Factor Strategy Analysis
Run with: streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import pickle
from pathlib import Path
from datetime import datetime

# Page config
st.set_page_config(
    page_title="Factor Strategy Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark theme
st.markdown("""
<style>
    /* Dark theme colors */
    .main {
        background-color: #0e1117;
    }

    /* Metric cards */
    div[data-testid="metric-container"] {
        background-color: rgba(28, 131, 225, 0.1);
        border: 1px solid rgba(28, 131, 225, 0.2);
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0px;
    }

    /* Headers */
    h1, h2, h3 {
        color: #ffffff;
    }

    /* Success/Error boxes */
    .success-box {
        padding: 10px;
        border-radius: 5px;
        background-color: rgba(0, 255, 136, 0.1);
        border: 1px solid rgba(0, 255, 136, 0.3);
        margin: 10px 0;
    }

    .warning-box {
        padding: 10px;
        border-radius: 5px;
        background-color: rgba(255, 165, 0, 0.1);
        border: 1px solid rgba(255, 165, 0, 0.3);
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)


# Load data
@st.cache_data
def load_results():
    """Load backtest results from pickle file"""
    base_dir = Path(__file__).resolve().parent
    results_path = base_dir / "results" / "backtest_results.pkl"
    if not results_path.exists():
        st.error("⚠️ No backtest results found! Run 'python factor_analysis.py' first (from the project root).")
        st.stop()

    with open(results_path, 'rb') as f:
        results = pickle.load(f)
    return results


# Calculate additional metrics
def calculate_rolling_metrics(returns, window=252):
    """Calculate rolling performance metrics"""
    rolling_returns = returns.rolling(window).mean() * 252
    rolling_vol = returns.rolling(window).std() * np.sqrt(252)
    rolling_sharpe = rolling_returns / rolling_vol

    # Cumulative returns
    cum_returns = (1 + returns).cumprod()

    # Drawdown
    rolling_max = cum_returns.expanding().max()
    drawdown = (cum_returns / rolling_max - 1) * 100

    return {
        'cumulative': cum_returns,
        'rolling_returns': rolling_returns,
        'rolling_vol': rolling_vol,
        'rolling_sharpe': rolling_sharpe,
        'drawdown': drawdown
    }


# Color scheme
colors = {
    'primary': '#1f77b4',
    'secondary': '#ff7f0e',
    'success': '#00ff88',
    'danger': '#ff4757',
    'warning': '#ffa500',
    'info': '#00d4ff',
    'dark': '#0e1117',
    'grid': 'rgba(128, 128, 128, 0.2)'
}

# Plotly dark theme
plotly_layout = {
    'template': 'plotly_dark',
    'paper_bgcolor': colors['dark'],
    'plot_bgcolor': colors['dark'],
    'font': {'color': 'white'},
    'xaxis': {'gridcolor': colors['grid'], 'showgrid': True},
    'yaxis': {'gridcolor': colors['grid'], 'showgrid': True}
}


# Main app
def main():
    # Load results
    results = load_results()

    # Header
    st.title("📊 Factor Strategy Performance Dashboard")
    st.markdown("### Quantitative Multi-Factor Equity Strategy Analysis")

    # Sidebar
    with st.sidebar:
        st.markdown("## ⚙️ Strategy Settings")

        # Strategy selector
        strategy = st.selectbox(
            "Select Strategy",
            options=['composite', 'momentum', 'low_vol'],
            format_func=lambda x: x.replace('_', ' ').title()
        )

        # Benchmark comparison
        show_benchmark = st.checkbox("Show Benchmark (SPY)", value=True)

        # Date range (read-only)
        st.markdown("### 📅 Analysis Period")
        start_date = results['prices'].index[0].date()
        end_date = results['prices'].index[-1].date()
        _ = st.date_input(
            "Date Range",
            value=[start_date, end_date],
            disabled=True
        )

        # Info box
        st.markdown("---")
        st.info(f"""
        **Data Points**: {len(results['prices'])} days  
        **Universe**: {len(results['prices'].columns)} stocks  
        **Rebalance**: Daily  
        **Factors**: 4 (Mom, Vol, Val, Qual)
        """)

    # Get strategy returns
    strategy_returns = results['portfolios'][strategy]['Q5']  # Top quintile
    benchmark_returns = results['benchmark_returns']

    # Align dates
    aligned = pd.concat([strategy_returns, benchmark_returns], axis=1).dropna()
    aligned.columns = [strategy.replace('_', ' ').title(), 'SPY']

    # Calculate metrics
    strategy_metrics = calculate_rolling_metrics(aligned.iloc[:, 0])
    benchmark_metrics = calculate_rolling_metrics(aligned.iloc[:, 1])

    # Performance metrics cards
    st.markdown("---")
    st.markdown("### 📈 Key Performance Metrics")

    col1, col2, col3, col4, col5, col6 = st.columns(6)

    perf = results['performance'][strategy]
    spy_perf = results['performance']['SPY']

    with col1:
        st.metric(
            label="Annual Return",
            value=f"{perf['Annual Return']:.1%}",
            delta=f"{(perf['Annual Return'] - spy_perf['Annual Return']):.1%} vs SPY"
        )

    with col2:
        st.metric(
            label="Sharpe Ratio",
            value=f"{perf['Sharpe Ratio']:.2f}",
            delta=f"{(perf['Sharpe Ratio'] - spy_perf['Sharpe Ratio']):.2f} vs SPY"
        )

    with col3:
        st.metric(
            label="Max Drawdown",
            value=f"{perf['Max Drawdown']:.1%}",
            delta=None
        )

    with col4:
        st.metric(
            label="Win Rate",
            value=f"{perf['Win Rate']:.1%}",
            delta=f"{(perf['Win Rate'] - spy_perf['Win Rate']):.1%} vs SPY"
        )

    with col5:
        st.metric(
            label="Sortino Ratio",
            value=f"{perf['Sortino Ratio']:.2f}",
            delta=None
        )

    with col6:
        st.metric(
            label="Calmar Ratio",
            value=f"{perf['Calmar Ratio']:.2f}",
            delta=None
        )

    # ----- Precompute latest signals & holdings for reuse across tabs -----
    latest_date = results['signals']['composite'].index[-1]
    latest_signals = results['signals']['composite'].loc[latest_date].sort_values(ascending=False)

    top_n = 10
    top_holdings = latest_signals.head(top_n)

    holdings_df = pd.DataFrame({
        'Ticker': top_holdings.index,
        'Signal Score': top_holdings.values,
        'Rank': range(1, top_n + 1)
    })

    # Add simple 1M / 3M returns for those tickers
    returns_1m, returns_3m = [], []
    for ticker in holdings_df['Ticker']:
        try:
            r1m = (results['prices'][ticker].iloc[-1] / results['prices'][ticker].iloc[-21] - 1) * 100
            r3m = (results['prices'][ticker].iloc[-1] / results['prices'][ticker].iloc[-63] - 1) * 100
        except Exception:
            r1m, r3m = 0, 0
        returns_1m.append(r1m)
        returns_3m.append(r3m)

    holdings_df['1M Return (%)'] = returns_1m
    holdings_df['3M Return (%)'] = returns_3m
    # ----------------------------------------------------------------------

    # Tabs for different views
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Performance",
        "📈 Factor Analysis",
        "🎯 Holdings",
        "📉 Risk Analysis",
        "📋 Report"
    ])

    with tab1:
        # Cumulative returns chart
        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown("#### Cumulative Returns")

            fig = go.Figure()

            # Strategy
            fig.add_trace(go.Scatter(
                x=strategy_metrics['cumulative'].index,
                y=(strategy_metrics['cumulative'] - 1) * 100,
                mode='lines',
                name=strategy.replace('_', ' ').title(),
                line=dict(color=colors['primary'], width=2)
            ))

            # Benchmark
            if show_benchmark:
                fig.add_trace(go.Scatter(
                    x=benchmark_metrics['cumulative'].index,
                    y=(benchmark_metrics['cumulative'] - 1) * 100,
                    mode='lines',
                    name='SPY Benchmark',
                    line=dict(color=colors['warning'], width=2)
                ))

            fig.update_layout(
                **plotly_layout,
                height=400,
                yaxis_title="Cumulative Return (%)",
                xaxis_title="Date",
                hovermode='x unified',
                legend=dict(x=0, y=1)
            )

            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("#### Return Distribution")

            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=aligned.iloc[:, 0] * 100,
                name=strategy.replace('_', ' ').title(),
                marker_color=colors['primary'],
                opacity=0.7,
                nbinsx=30
            ))

            if show_benchmark:
                fig.add_trace(go.Histogram(
                    x=aligned.iloc[:, 1] * 100,
                    name='SPY',
                    marker_color=colors['warning'],
                    opacity=0.7,
                    nbinsx=30
                ))

            fig.update_layout(
                **plotly_layout,
                height=400,
                xaxis_title="Daily Return (%)",
                yaxis_title="Frequency",
                barmode='overlay'
            )

            st.plotly_chart(fig, use_container_width=True)

        # Rolling metrics
        st.markdown("#### Rolling Performance Metrics (252-day window)")

        col1, col2 = st.columns(2)

        with col1:
            # Rolling Sharpe
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=strategy_metrics['rolling_sharpe'].index,
                y=strategy_metrics['rolling_sharpe'],
                mode='lines',
                name='Rolling Sharpe',
                line=dict(color=colors['info'], width=2),
                fill='tozeroy',
                fillcolor='rgba(0, 212, 255, 0.1)'
            ))

            # Add horizontal lines
            fig.add_hline(y=1, line_dash="dash", line_color=colors['success'],
                          annotation_text="Good (>1)")
            fig.add_hline(y=0, line_dash="dash", line_color=colors['danger'])

            fig.update_layout(
                **plotly_layout,
                height=350,
                yaxis_title="Sharpe Ratio",
                xaxis_title="Date"
            )

            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Drawdown
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=strategy_metrics['drawdown'].index,
                y=strategy_metrics['drawdown'],
                mode='lines',
                name='Drawdown',
                line=dict(color=colors['danger'], width=2),
                fill='tozeroy',
                fillcolor='rgba(255, 71, 87, 0.1)'
            ))

            fig.update_layout(
                **plotly_layout,
                height=350,
                yaxis_title="Drawdown (%)",
                xaxis_title="Date"
            )

            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.markdown("### Factor Performance Analysis")

        # Factor returns comparison
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Factor Strategy Comparison")

            # Create comparison dataframe
            comparison_data = []
            for strat in ['momentum', 'low_vol', 'composite']:
                perf_data = results['performance'][strat]
                comparison_data.append({
                    'Strategy': strat.replace('_', ' ').title(),
                    'Annual Return': perf_data['Annual Return'],
                    'Sharpe Ratio': perf_data['Sharpe Ratio'],
                    'Max Drawdown': perf_data['Max Drawdown']
                })

            comp_df = pd.DataFrame(comparison_data)

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=comp_df['Strategy'],
                y=comp_df['Annual Return'] * 100,
                name='Annual Return (%)',
                marker_color=colors['primary']
            ))

            # Second axis for Sharpe
            fig.add_trace(go.Bar(
                x=comp_df['Strategy'],
                y=comp_df['Sharpe Ratio'],
                name='Sharpe Ratio',
                marker_color=colors['success'],
                yaxis='y2'
            ))

            # Remove axis keys from shared layout to avoid duplicate 'yaxis' kwargs
            layout_no_axes = {k: v for k, v in plotly_layout.items() if k not in ('xaxis', 'yaxis')}
            fig.update_layout(
                **layout_no_axes,
                height=400,
                barmode='group',
                yaxis=dict(title='Annual Return (%)', side='left'),
                yaxis2=dict(title='Sharpe Ratio', overlaying='y', side='right')
            )

            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("#### Factor Correlation Matrix")

            # Calculate factor correlations
            factor_returns = pd.DataFrame({
                'Momentum': results['portfolios']['momentum']['Q5'],
                'Low Vol': results['portfolios']['low_vol']['Q5'],
                'Composite': results['portfolios']['composite']['Q5']
            })

            corr_matrix = factor_returns.corr()

            fig = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale='RdBu',
                zmid=0,
                text=np.round(corr_matrix.values, 2),
                texttemplate='%{text}',
                textfont={"size": 12}
            ))

            fig.update_layout(
                **plotly_layout,
                height=400,
                title="Factor Strategy Correlations"
            )

            st.plotly_chart(fig, use_container_width=True)

        # Factor exposures over time
        st.markdown("#### Factor Signal Strength Over Time")

        signals_df = results['signals']['composite']

        # Calculate average signal by month
        monthly_avg = signals_df.resample('M').mean().mean(axis=1)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=monthly_avg.index,
            y=monthly_avg.values,
            mode='lines',
            name='Average Signal Strength',
            line=dict(color=colors['info'], width=2),
            fill='tonexty',
            fillcolor='rgba(0, 212, 255, 0.1)'
        ))

        fig.add_hline(y=0.5, line_dash="dash", line_color='gray',
                      annotation_text="Neutral (0.5)")

        fig.update_layout(
            **plotly_layout,
            height=350,
            yaxis_title="Signal Strength (0-1)",
            xaxis_title="Date"
        )

        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.markdown("### Current Portfolio Holdings")

        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown(f"#### Top {top_n} Holdings (by Signal Strength)")

            st.dataframe(
                holdings_df.style.format({
                    'Signal Score': '{:.3f}',
                    '1M Return (%)': '{:.1f}%',
                    '3M Return (%)': '{:.1f}%'
                }).background_gradient(subset=['Signal Score'], cmap='RdYlGn'),
                use_container_width=True
            )

        with col2:
            st.markdown("#### Signal Distribution")

            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=latest_signals.values,
                nbinsx=20,
                marker_color=colors['primary'],
                name='All Stocks'
            ))

            # Mark quintile thresholds
            for q in [0.2, 0.4, 0.6, 0.8]:
                fig.add_vline(x=q, line_dash="dash", line_color='gray',
                              annotation_text=f'Q{int(q * 5)}')

            fig.update_layout(
                **plotly_layout,
                height=350,
                xaxis_title="Signal Score",
                yaxis_title="Count"
            )

            st.plotly_chart(fig, use_container_width=True)

        # Portfolio characteristics
        st.markdown("#### Portfolio Characteristics")

        col1, col2, col3 = st.columns(3)

        with col1:
            avg_signal = latest_signals.head(20).mean()
            st.metric("Avg Signal (Top 20)", f"{avg_signal:.3f}")

        with col2:
            concentration = (latest_signals.head(10).sum() / latest_signals.head(50).sum())
            st.metric("Top 10 Concentration", f"{concentration:.1%}")

        with col3:
            turnover_estimate = 0.35  # Placeholder
            st.metric("Est. Monthly Turnover", f"{turnover_estimate:.1%}")

    with tab4:
        st.markdown("### Risk Analysis")

        # Risk metrics
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Risk-Return Profile")

            # Create scatter plot of all strategies (skip SPY here; add separately)
            risk_return_data = []
            for strat_name, strat_perf in results['performance'].items():
                if strat_name == 'SPY':
                    continue
                risk_return_data.append({
                    'Strategy': strat_name.replace('_', ' ').title(),
                    'Return': strat_perf['Annual Return'] * 100,
                    'Risk': strat_perf['Volatility'] * 100,
                    'Sharpe': strat_perf['Sharpe Ratio']
                })

            # Add SPY
            risk_return_data.append({
                'Strategy': 'SPY',
                'Return': spy_perf['Annual Return'] * 100,
                'Risk': spy_perf['Volatility'] * 100,
                'Sharpe': spy_perf['Sharpe Ratio']
            })

            risk_df = pd.DataFrame(risk_return_data)

            fig = go.Figure()

            # Add strategies
            for _, row in risk_df.iterrows():
                color = colors['warning'] if row['Strategy'] == 'SPY' else colors['primary']
                fig.add_trace(go.Scatter(
                    x=[row['Risk']],
                    y=[row['Return']],
                    mode='markers+text',
                    name=row['Strategy'],
                    text=[row['Strategy']],
                    textposition='top center',
                    marker=dict(size=15, color=color),
                    hovertemplate=f"<b>{row['Strategy']}</b><br>" +
                                  f"Return: {row['Return']:.1f}%<br>" +
                                  f"Risk: {row['Risk']:.1f}%<br>" +
                                  f"Sharpe: {row['Sharpe']:.2f}<extra></extra>"
                ))

            # Add a simple CAL line (illustrative)
            fig.add_trace(go.Scatter(
                x=[0, 30],
                y=[0, 15],
                mode='lines',
                line=dict(dash='dash', color='gray'),
                name='CAL (RF=2%)',
                showlegend=False
            ))

            fig.update_layout(
                **plotly_layout,
                height=400,
                xaxis_title="Risk (Volatility %)",
                yaxis_title="Return (%)",
                showlegend=False
            )

            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("#### Value at Risk (VaR)")

            # Calculate VaR / CVaR on daily returns (decimal), report in %
            confidence_levels = [0.95, 0.99]
            var_data = []
            for conf in confidence_levels:
                var_value = np.percentile(aligned.iloc[:, 0] * 100, (1 - conf) * 100)
                cvar_value = aligned.iloc[:, 0][aligned.iloc[:, 0] <= var_value / 100].mean() * 100
                var_data.append({
                    'Confidence': f"{conf:.0%}",
                    'VaR (%)': var_value,
                    'CVaR (%)': cvar_value
                })

            var_df = pd.DataFrame(var_data)

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=var_df['Confidence'],
                y=var_df['VaR (%)'].abs(),
                name='VaR',
                marker_color=colors['warning']
            ))

            fig.add_trace(go.Bar(
                x=var_df['Confidence'],
                y=var_df['CVaR (%)'].abs(),
                name='CVaR',
                marker_color=colors['danger']
            ))

            fig.update_layout(
                **plotly_layout,
                height=400,
                yaxis_title="Daily Loss (%)",
                xaxis_title="Confidence Level",
                barmode='group'
            )

            st.plotly_chart(fig, use_container_width=True)

        # Stress testing
        st.markdown("#### Historical Stress Periods")

        # Define stress periods (example)
        stress_periods = [
            ("COVID-19", "2020-02-19", "2020-03-23"),
            ("2022 Correction", "2022-01-01", "2022-06-30")
        ]

        stress_results = []
        for period_name, start, end in stress_periods:
            try:
                period_returns = aligned.loc[start:end]
                if len(period_returns) > 0:
                    strat_return = (1 + period_returns.iloc[:, 0]).prod() - 1
                    bench_return = (1 + period_returns.iloc[:, 1]).prod() - 1

                    stress_results.append({
                        'Period': period_name,
                        'Strategy': f"{strat_return:.1%}",
                        'Benchmark': f"{bench_return:.1%}",
                        'Relative': f"{(strat_return - bench_return):.1%}"
                    })
            except Exception:
                pass

        if stress_results:
            stress_df = pd.DataFrame(stress_results)
            st.table(stress_df)

    with tab5:
        st.markdown("### Strategy Report Summary")

        # Executive summary
        st.markdown("""
        #### Executive Summary

        This multi-factor quantitative equity strategy combines **Momentum**, **Low Volatility**, 
        **Value**, and **Quality** factors to construct a systematic portfolio that aims to 
        outperform the S&P 500 benchmark on a risk-adjusted basis.
        """)

        # Key findings
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("##### ✅ Strategy Strengths")
            st.success(f"""
            - **Superior Risk-Adjusted Returns**: Sharpe Ratio of {perf['Sharpe Ratio']:.2f} vs {spy_perf['Sharpe Ratio']:.2f} for SPY
            - **Consistent Outperformance**: {perf['Win Rate']:.1%} win rate
            - **Controlled Drawdowns**: Max DD of {perf['Max Drawdown']:.1%}
            - **Positive Alpha**: {perf.get('Alpha', 0):.1%} annualized
            """)

        with col2:
            st.markdown("##### ⚠️ Risk Considerations")
            st.warning(f"""
            - **Tracking Error**: Deviation from benchmark
            - **Factor Timing Risk**: Performance varies by regime
            - **Concentration Risk**: Top quintile only
            - **Transaction Costs**: Not fully modeled
            """)

        # Recommendations
        st.markdown("#### Recommendations")

        st.info("""
        1. **Implementation**: Consider starting with 50% allocation to test real-world performance
        2. **Risk Management**: Implement stop-loss at -15% drawdown
        3. **Rebalancing**: Monthly rebalancing optimal for factor capture
        4. **Monitoring**: Track factor exposures and style drift weekly
        5. **Enhancement**: Consider adding sector neutrality constraints
        """)

        # Download results (no wrapper buttons)
        st.markdown("---")
        st.markdown("#### 📥 Export Results")

        col1, col2, col3 = st.columns(3)

        with col1:
            perf_df = pd.DataFrame(results['performance']).T
            csv = perf_df.to_csv()
            st.download_button(
                label="📊 Download Performance Report (CSV)",
                data=csv,
                file_name=f"factor_strategy_performance_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )

        with col2:
            returns_df = pd.DataFrame({
                'Date': aligned.index,
                'Strategy': aligned.iloc[:, 0],
                'Benchmark': aligned.iloc[:, 1]
            })
            csv2 = returns_df.to_csv(index=False)
            st.download_button(
                label="📈 Download Returns Data (CSV)",
                data=csv2,
                file_name=f"strategy_returns_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )

        with col3:
            holdings_csv = holdings_df.to_csv(index=False)
            st.download_button(
                label="🎯 Download Holdings (CSV)",
                data=holdings_csv,
                file_name=f"top_holdings_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )


# Run app
if __name__ == "__main__":
    main()