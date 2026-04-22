import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from backtester import PortfolioBacktest
import warnings

warnings.filterwarnings('ignore')

st.set_page_config(page_title="Stock Portfolio Backtester", layout="wide")

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
benchmark = st.sidebar.text_input("Benchmark Ticker", "SPY", help="내 포트폴리오의 성과와 비교할 기준 지수(시장 평균 등)입니다. 기본값은 미국 S&P 500 ETF(SPY)입니다.")

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
            st.subheader("Portfolio Growth")
            growth_df = pd.DataFrame({
                'Portfolio': bt.portfolio_value,
                'Benchmark': bt.bench_value
            })
            st.line_chart(growth_df)
            
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
            metrics_df = pd.DataFrame({
                'Metric': ['CAGR', 'Daily Volatility (Ann.)', 'Sharpe Ratio', 'Max Drawdown', 'Beta', 'Alpha'],
                'Portfolio': [f"{bt.cagr:.2%}", f"{bt.volatility:.2%}", f"{bt.sharpe:.2f}", f"{bt.max_drawdown:.2%}", f"{bt.beta:.2f}", f"{bt.alpha:.2%}"],
                'Benchmark': [f"{bt.bench_cagr:.2%}", f"{bt.bench_volatility:.2%}", f"{bt.bench_sharpe:.2f}", f"{bt.bench_max_drawdown:.2%}", "1.00", "0.00%"]
            })
            st.table(metrics_df.set_index('Metric'))
            
            # --- Drawdowns Analysis ---
            st.header("4. Drawdowns Analysis", help="직전 최고점(전고점) 대비 자산이 얼마나 하락했는지(손실폭)를 보여주는 낙폭 차트입니다. 그래프가 아래로 패인 구간이 경제 위기나 하락장 구간입니다.")
            port_cummax = bt.portfolio_value.cummax()
            port_drawdown = (bt.portfolio_value / port_cummax - 1.0) * 100
            
            bench_cummax = bt.bench_value.cummax()
            bench_drawdown = (bt.bench_value / bench_cummax - 1.0) * 100
            
            dd_df = pd.DataFrame({
                'Portfolio Drawdown (%)': port_drawdown,
                'Benchmark Drawdown (%)': bench_drawdown
            })
            st.line_chart(dd_df)
            
            # --- Asset Level ---
            st.header("5. Asset-Level Analysis", help="포트폴리오 구성 자산들의 일간 수익률 상관관계를 색상으로 표기한 히트맵입니다. 상관계수가 음수(파란색)이거나 낮은 자산끼리 섞이면 위험 분산(헤지) 효과가 커집니다.")
            st.subheader("Asset Correlations (Daily Returns)")
            returns = bt.prices.pct_change().dropna()
            corr = returns.corr()
            
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1, ax=ax)
            st.pyplot(fig)
