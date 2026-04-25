import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from backtester import PortfolioBacktest
import warnings
import os
import json

warnings.filterwarnings('ignore')

st.set_page_config(page_title="Stock Portfolio Backtester", layout="wide")

STATS_FILE = "usage_stats.json"

def load_stats():
    if os.path.exists(STATS_FILE):
        try:
            with open(STATS_FILE, "r") as f:
                return json.load(f)
        except:
            pass
    return {"page_views": 0, "executions": 0, "likes": 0}

def save_stats(stats):
    try:
        with open(STATS_FILE, "w") as f:
            json.dump(stats, f)
    except:
        pass

if 'visited' not in st.session_state:
    st.session_state.visited = True
    stats = load_stats()
    stats["page_views"] += 1
    save_stats(stats)


@st.cache_data
def load_krx_tickers():
    try:
        import FinanceDataReader as fdr
        df_krx = fdr.StockListing('KRX')
        def map_krx(m):
            if m == 'KOSPI': return '.KS'
            elif pd.isna(m) or m == '': return '.KS'
            else: return '.KQ'
        df_krx['YF_Ticker'] = df_krx['Code'] + df_krx['Market'].apply(map_krx)
        df_krx['Display'] = "[" + df_krx['YF_Ticker'] + "] " + df_krx['Name'] + " (KRX)"
        df_krx = df_krx.drop_duplicates(subset=['YF_Ticker'])
        return df_krx[['Name', 'YF_Ticker', 'Display']]
    except Exception as e:
        return pd.DataFrame()

st.title("Python Stock Backtester")
st.markdown("Replicating comprehensive quantitative analysis reports using Streamlit & yfinance.")

st.sidebar.header("Portfolio Configuration")

PREDEFINED_PORTFOLIOS = {
    "Custom": {"tickers": "AAPL, TLT, SPY", "weights": "40, 40, 20"},
    "Comprehensive Optimized (2004+)": {"tickers": "QQQ, GLD, SHY, IEF, DVY, TLT", "weights": "24.8, 36.9, 28.1, 4.9, 2.7, 2.6"},
    "Low Drawdown Growth (1990+)": {"tickers": "QQQ, TLT, GLD, VTI", "weights": "30, 45, 10, 15"},
    "60/40 Portfolio (Stocks/Bonds)": {"tickers": "SPY, TLT", "weights": "60, 40"},
    "All Weather (Ray Dalio)": {"tickers": "VTI, TLT, IEF, GLD, DBC", "weights": "30, 40, 15, 7.5, 7.5"},
    "Permanent Portfolio (Harry Browne)": {"tickers": "SPY, TLT, SHY, GLD", "weights": "25, 25, 25, 25"},
    "Golden Butterfly": {"tickers": "VTI, IWN, TLT, SHY, GLD", "weights": "20, 20, 20, 20, 20"},
    "Korean Blue Chips (Samsung/SkHynix)": {"tickers": "005930.KS, 000660.KS", "weights": "50, 50"}
}

st.sidebar.subheader("Strategy Selection")
selected_strategy = st.sidebar.selectbox("Choose a Strategy", list(PREDEFINED_PORTFOLIOS.keys()), help="이미 구성된 유명 투자 전략을 선택해 바로 테스트해볼 수 있습니다. Custom 선택 시 직접 종목과 비중을 입력할 수 있습니다.")
rebalance = st.sidebar.selectbox("Rebalance Frequency", ["None", "Monthly", "Annually"], index=1, help="시장 변화로 틀어진 자산 비중을 지정한 주기마다 원래 비중으로 되돌리는(사고파는) 리밸런싱을 의미합니다.")

# Ticker Search
st.sidebar.subheader("Ticker Search (KRX Only)")
df_krx = load_krx_tickers()

if not df_krx.empty:
    search_query = st.sidebar.selectbox("Search Korean Company Name", options=[""] + df_krx['Display'].tolist(), help="한국 주식의 회사명이나 코드를 검색하여 yfinance 호환 티커를 찾을 수 있습니다.")
    if search_query:
        selected_row = df_krx[df_krx['Display'] == search_query]
        if not selected_row.empty:
            selected_ticker = selected_row.iloc[0]['YF_Ticker']
            st.sidebar.info(f"Ticker for yfinance: **{selected_ticker}**\n\n*(Copy and paste into Tickers field below)*")

# Dynamic inputs for tickers and weights
st.sidebar.subheader("Assets")
tickers_input = st.sidebar.text_input("Tickers (comma separated)", PREDEFINED_PORTFOLIOS[selected_strategy]["tickers"], help="투자할 종목의 티커를 쉼표(,)로 구분해서 적어주세요. (예: AAPL, SPY, 005930.KS)")
weights_input = st.sidebar.text_input("Weights (comma separated)", PREDEFINED_PORTFOLIOS[selected_strategy]["weights"], help="티커 순서에 맞춰 목표 투자 비중(%)을 쉼표로 적어주세요. 총합이 항상 100이어야 합니다.")

tickers = [t.strip() for t in tickers_input.split(",")]
try:
    weights = [float(w.strip())/100.0 for w in weights_input.split(",")]
except ValueError:
    st.sidebar.error("Weights must be numbers")
    weights = []

start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime('2018-01-01'), min_value=pd.to_datetime('1990-01-01'), help="투자를 시작할(백테스트 시작) 기준일입니다. 상장 전인 종목이 있다면 자동으로 해당 종목 상장일로 연기됩니다.")
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime('today'), min_value=pd.to_datetime('1990-01-01'), help="투자를 종료할 시점입니다.")
initial_capital = st.sidebar.number_input("Initial Capital", min_value=1000, value=10000, step=1000, help="포트폴리오에 최초 편입할 자본금입니다. (계산 편의상 달러로 표기됩니다)")
installment_amount = st.sidebar.number_input("Installment Amount", min_value=0, value=0, step=100, help="적립식으로 매 기간마다 투입할 추가 금액입니다.")
installment_frequency = st.sidebar.selectbox("Installment Frequency", ["None", "Monthly", "Annually"], help="지정한 적립 투자 금액을 매월 투입할지, 매년 투입할지 결정합니다.")
BENCHMARK_OPTIONS = {
    "Inflation (CPI)": "Inflation (CPI)",
    "SPY (S&P 500)": "SPY",
    "QQQ (Nasdaq 100)": "QQQ",
    "VTI (Total US Market)": "VTI",
    "ACWI (World)": "ACWI",
    "Custom ticker...": "__CUSTOM__",
}
benchmark_choice = st.sidebar.selectbox(
    "Benchmark",
    list(BENCHMARK_OPTIONS.keys()),
    index=0,
    help="포트폴리오 성과를 비교할 기준입니다. 기본값은 미국 물가(CPI, FRED 데이터). 인덱스 ETF나 직접 입력한 티커와도 비교할 수 있습니다.",
)
if BENCHMARK_OPTIONS[benchmark_choice] == "__CUSTOM__":
    benchmark = st.sidebar.text_input("Custom Benchmark Ticker", "SPY")
else:
    benchmark = BENCHMARK_OPTIONS[benchmark_choice]

if len(tickers) != len(weights):
    st.sidebar.error("Number of tickers and weights must match.")
elif not np.isclose(sum(weights), 1.0):
    st.sidebar.error(f"Weights must sum to 100%. Currently sum to {sum(weights)*100:.1f}%")
else:
    if st.sidebar.button("Run Backtest"):
        tickers_weights = dict(zip(tickers, weights))
        
        with st.spinner("Fetching data and running backtest..."):
            bt = PortfolioBacktest(tickers_weights, start_date, end_date, initial_capital, benchmark, rebalance, installment_amount, installment_frequency)
            try:
                bt.run()
                success = True
            except Exception as e:
                st.error(f"An error occurred: {e}")
                success = False
                
        if success:
            stats = load_stats()
            stats["executions"] += 1
            save_stats(stats)
            st.success("Backtest complete!")
            
            if hasattr(bt, 'start_date_adjusted') and bt.start_date_adjusted:
                st.warning(f"⚠️ **상장일 알림**: 설정하신 시작일({start_date}) 당시에 상장되지 않은 종목이 포함되어 있습니다. 가장 늦게 상장된 종목의 데이터가 존재하는 **{bt.actual_start_date}** 부터 백테스트가 자동으로 조절되어 실행되었습니다.")
            
            # --- Overview & Allocation ---
            st.header("1. Portfolio Overview & Allocation", help="백테스트의 최종 자산 평가 금액, 누적 원금, 순수익 요약과 포트폴리오의 비중 파이를 보여줍니다.")
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Investment Parameters**")
                st.write(f"- **Initial Balance:** ${bt.initial_capital:,.2f}")
                if bt.installment_amount > 0 and bt.installment_frequency != 'None':
                    st.write(f"- **Total Invested:** ${bt.invested_capitals.iloc[-1]:,.2f}")
                st.write(f"- **Final Balance:** ${bt.portfolio_value.iloc[-1]:,.2f}")
                st.write(f"- **Total Profit:** ${bt.portfolio_value.iloc[-1] - bt.invested_capitals.iloc[-1]:,.2f}")
                st.write(f"- **Total Return (ROI):** {(bt.portfolio_value.iloc[-1] / bt.invested_capitals.iloc[-1] - 1)*100:.2f}%")
                st.write(f"- **Rebalancing:** {rebalance}")
                if bt.installment_amount > 0 and bt.installment_frequency != 'None':
                    st.write(f"- **Installment:** ${bt.installment_amount:,.2f} ({bt.installment_frequency})")
                st.write(f"- **Time Period:** {start_date} to {end_date}")
            with col2:
                fig, ax = plt.subplots(figsize=(4, 4))
                ax.pie(weights, labels=tickers, autopct='%1.1f%%', startangle=90)
                ax.axis('equal')
                st.pyplot(fig)
                
            # --- Performance & Returns ---
            st.header("2. Performance & Returns", help="투자 기간 동안 포트폴리오 가치와 벤치마크(기준 시장) 가치의 누적 상승 추이를 시각적인 그래프로 비교합니다.")
            
            # Growth Chart
            bench_label = getattr(bt, 'benchmark_label', benchmark)
            st.subheader("Portfolio Growth")
            growth_df = pd.DataFrame({
                'Portfolio': bt.portfolio_value,
                f'Benchmark ({bench_label})': bt.bench_value
            })
            st.line_chart(growth_df)
            if getattr(bt, 'benchmark_is_inflation', False):
                st.caption("벤치마크가 물가(CPI)이므로 '벤치마크'는 동일 원금을 물가상승률로만 불렸을 때의 가상 잔고입니다. 포트폴리오가 이 선을 넘으면 실질 수익(real return)이 (+)입니다.")
            
            # --- Risk & Return Metrics ---
            st.header("3. Risk & Return Metrics")
            st.markdown("""
            * **CAGR** (Compound Annual Growth Rate): 투자 기간 동안의 연평균 복리 수익률입니다.
            * **Daily Volatility (Ann.)** (연환산 변동성): 포트폴리오 수익률이 얼마나 출렁이는지를 나타내는 위험 지표입니다. 낮을수록 안정적입니다.
            * **Sharpe Ratio** (샤프 지수): 1단위 위험을 감수할 때 얻을 수 있는 초과 수익률입니다. 높을수록 좋습니다.
            * **Max Drawdown** (MDD, 최대 낙폭): 직전 최고점(전고점) 대비 최대로 하락한 비율을 의미하며, 해당 포트폴리오의 가장 큰 손실 위험을 나타냅니다.
            * **Beta** (베타): 시장(벤치마크)의 움직임에 얼마나 민감하게 반응하는지를 수치화한 것입니다. 1보다 크면 시장보다 변동이 크다는 뜻입니다.
            * **Alpha** (알파): 벤치마크 대비 포트폴리오의 실질적인 초과 수익률입니다. 높을수록 좋습니다.
            """)
            if getattr(bt, 'benchmark_is_inflation', False):
                beta_port, alpha_port = "N/A", "N/A"
                beta_help = "벤치마크가 CPI(물가)라 시장 민감도 지표인 Beta/Alpha는 의미가 없어 표시하지 않습니다."
            else:
                beta_port, alpha_port = f"{bt.beta:.2f}", f"{bt.alpha:.2%}"
                beta_help = None

            metrics_df = pd.DataFrame({
                'Metric': ['CAGR', 'Daily Volatility (Ann.)', 'Sharpe Ratio', 'Max Drawdown', 'Beta', 'Alpha'],
                'Portfolio': [f"{bt.cagr:.2%}", f"{bt.volatility:.2%}", f"{bt.sharpe:.2f}", f"{bt.max_drawdown:.2%}", beta_port, alpha_port],
                f'Benchmark ({bench_label})': [f"{bt.bench_cagr:.2%}", f"{bt.bench_volatility:.2%}", f"{bt.bench_sharpe:.2f}", f"{bt.bench_max_drawdown:.2%}", "1.00", "0.00%"]
            })
            st.table(metrics_df.set_index('Metric'))
            if beta_help:
                st.caption(beta_help)
            
            # --- Drawdowns Analysis ---
            st.header("4. Drawdowns Analysis", help="직전 최고점(전고점) 대비 자산이 얼마나 하락했는지(손실폭)를 보여주는 낙폭 차트입니다. 그래프가 아래로 패인 구간이 경제 위기나 하락장 구간입니다.")
            port_cummax = bt.cumulative_returns.cummax()
            port_drawdown = (bt.cumulative_returns / port_cummax - 1.0) * 100
            
            bench_cummax = bt.bench_cumulative.cummax()
            bench_drawdown = (bt.bench_cumulative / bench_cummax - 1.0) * 100
            
            dd_df = pd.DataFrame({
                'Portfolio Drawdown (%)': port_drawdown,
                f'Benchmark ({bench_label}) Drawdown (%)': bench_drawdown
            })
            st.line_chart(dd_df)

            # --- Start-Date Sensitivity ---
            st.header(
                "5. Start-Date Sensitivity",
                help="'언제 투자를 시작했느냐'에 따라 현재(종료일)까지의 수익률이 얼마나 달라지는지를 보여줍니다. 각 월초를 가상의 시작 시점으로 삼아 종료일까지의 CAGR/총수익률을 계산해 선으로 잇습니다. 선이 가파르면 타이밍 민감도가 크고, 평평하면 둔감합니다.",
            )
            try:
                sensitivity = bt.rolling_start_analysis()
            except Exception as e:
                sensitivity = None
                st.info(f"Start-date sensitivity 계산 중 오류: {e}")

            if sensitivity is not None and not sensitivity.empty:
                metric_choice = st.radio(
                    "Metric",
                    ["CAGR (연평균 수익률)", "Total Return (총 수익률)"],
                    index=0,
                    horizontal=True,
                    key="sensitivity_metric",
                )
                if metric_choice.startswith("CAGR"):
                    plot_df = sensitivity[['Portfolio CAGR', 'Benchmark CAGR']].rename(
                        columns={'Benchmark CAGR': f'Benchmark CAGR ({bench_label})'}
                    )
                else:
                    plot_df = sensitivity[['Portfolio Total Return', 'Benchmark Total Return']].rename(
                        columns={'Benchmark Total Return': f'Benchmark Total Return ({bench_label})'}
                    )
                st.line_chart(plot_df)

                st.caption(
                    f"각 X축 지점은 '이 날 투자를 시작했다면' 의 가상 시작일이며, 종료일({end_date})까지의 성과를 의미합니다. "
                    "포트폴리오 전략(비중/리밸런싱)은 동일하게 유지되고 시작 시점만 달라집니다."
                )

                # Summary stats
                best = sensitivity['Portfolio CAGR'].idxmax()
                worst = sensitivity['Portfolio CAGR'].idxmin()
                colA, colB, colC = st.columns(3)
                colA.metric(
                    "Best Start (CAGR)",
                    best.strftime('%Y-%m-%d'),
                    f"{sensitivity.loc[best, 'Portfolio CAGR']:.2%}",
                )
                colB.metric(
                    "Worst Start (CAGR)",
                    worst.strftime('%Y-%m-%d'),
                    f"{sensitivity.loc[worst, 'Portfolio CAGR']:.2%}",
                )
                colC.metric(
                    "Median Start CAGR",
                    "—",
                    f"{sensitivity['Portfolio CAGR'].median():.2%}",
                )

                with st.expander("Raw data (start date × return)"):
                    display_df = sensitivity.copy()
                    display_df['Portfolio CAGR'] = display_df['Portfolio CAGR'].map(lambda x: f"{x:.2%}")
                    display_df['Benchmark CAGR'] = display_df['Benchmark CAGR'].map(lambda x: f"{x:.2%}")
                    display_df['Portfolio Total Return'] = display_df['Portfolio Total Return'].map(lambda x: f"{x:.2%}")
                    display_df['Benchmark Total Return'] = display_df['Benchmark Total Return'].map(lambda x: f"{x:.2%}")
                    display_df['Years Held'] = display_df['Years Held'].map(lambda x: f"{x:.2f}")
                    st.dataframe(display_df)
            else:
                st.info("데이터 기간이 너무 짧아 시작일별 민감도를 계산할 수 없습니다.")

            # --- Asset Level ---
            st.header("6. Asset-Level Analysis", help="포트폴리오 구성 자산들의 일간 수익률 상관관계를 색상으로 표기한 히트맵입니다. 상관계수가 음수(파란색)이거나 낮은 자산끼리 섞이면 위험 분산(헤지) 효과가 커집니다.")
            st.subheader("Asset Correlations (Daily Returns)")
            returns = bt.prices.pct_change().dropna()
            corr = returns.corr()
            
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1, ax=ax)
            st.pyplot(fig)

# Display usage stats in sidebar
current_stats = load_stats()
st.sidebar.markdown("---")
st.sidebar.caption(f"Total Views: {current_stats.get('page_views', 0)} | Total Executions: {current_stats.get('executions', 0)}")

# Like button
already_liked = st.session_state.get('liked', False)
like_count = current_stats.get('likes', 0)
like_label = f"👍 Liked  ·  {like_count}" if already_liked else f"👍 Like  ·  {like_count}"
if st.sidebar.button(like_label, disabled=already_liked, use_container_width=True, help="이 페이지가 마음에 들면 눌러주세요. 같은 세션에서는 한 번만 카운트됩니다."):
    st.session_state.liked = True
    fresh = load_stats()
    fresh['likes'] = fresh.get('likes', 0) + 1
    save_stats(fresh)
    st.rerun()
