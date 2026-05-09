import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from backtesting import Backtest, Strategy
import FinanceDataReader as fdr
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ============================================================
# [0. 페이지 설정 - 모바일 대응]
# ============================================================
st.set_page_config(
    layout="wide",
    page_title="VTS 통합 트레이딩 시스템",
    initial_sidebar_state="auto",  # 모바일에선 자동 접힘
    menu_items={'About': "Vitacho Trading System - 모바일 지원"}
)

# 모바일 친화 CSS
st.markdown("""
<style>
    /* 모바일에서 폰트/패딩 최적화 */
    @media (max-width: 768px) {
        .block-container { padding: 1rem 0.5rem !important; }
        h1 { font-size: 1.5rem !important; }
        h2 { font-size: 1.2rem !important; }
        h3 { font-size: 1.05rem !important; }
        .stTabs [data-baseweb="tab-list"] { gap: 4px; }
        .stTabs [data-baseweb="tab"] { padding: 8px 10px; font-size: 0.85rem; }
        .stDataFrame { font-size: 0.8rem; }
    }
    /* 표 가독성 */
    .stDataFrame { border-radius: 8px; }
    /* 메트릭 카드 */
    [data-testid="stMetricValue"] { font-size: 1.3rem; }
</style>
""", unsafe_allow_html=True)


# ============================================================
# [1. 백테스트 전략]
# ============================================================
class VitachoStrategy(Strategy):
    def init(self):
        self.sma_fast = self.I(lambda x: pd.Series(x).rolling(10).mean(), self.data.Close)
        self.sma_mid = self.I(lambda x: pd.Series(x).rolling(25).mean(), self.data.Close)
        self.sma_slow = self.I(lambda x: pd.Series(x).rolling(75).mean(), self.data.Close)

    def next(self):
        if self.sma_fast[-1] > self.sma_mid[-1] > self.sma_slow[-1]:
            if self.data.Close[-1] > self.sma_fast[-1] and not self.position:
                self.buy()
        elif self.position and self.data.Close[-1] < self.sma_mid[-1]:
            self.position.close()


# ============================================================
# [2. 보조지표 계산 함수]
# ============================================================
def calc_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = -delta.where(delta < 0, 0).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def calc_bollinger(series, period=20, std=2):
    ma = series.rolling(period).mean()
    sd = series.rolling(period).std()
    return ma + std * sd, ma, ma - std * sd


def calc_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    sig = macd.ewm(span=signal, adjust=False).mean()
    return macd, sig, macd - sig


# ============================================================
# [3. Plotly 인터랙티브 차트 함수 - 모바일 친화]
# ============================================================
def make_advanced_chart(df, ticker_name, currency="$"):
    """이평선 + 볼린저밴드 + 거래량 + RSI + MACD + 매매 신호 마커"""
    df = df.copy()
    df['SMA10'] = df['Close'].rolling(10).mean()
    df['SMA25'] = df['Close'].rolling(25).mean()
    df['SMA75'] = df['Close'].rolling(75).mean()
    df['BB_U'], df['BB_M'], df['BB_L'] = calc_bollinger(df['Close'])
    df['RSI'] = calc_rsi(df['Close'])
    df['MACD'], df['SIG'], df['HIST'] = calc_macd(df['Close'])

    # 매수/매도 신호 포인트 추출
    buy_signals = []
    sell_signals = []
    for i in range(1, len(df)):
        if pd.notna(df['SMA75'].iloc[i]):
            f, m, s = df['SMA10'].iloc[i], df['SMA25'].iloc[i], df['SMA75'].iloc[i]
            pf, pm, ps = df['SMA10'].iloc[i-1], df['SMA25'].iloc[i-1], df['SMA75'].iloc[i-1]
            # 정배열 진입 (골든크로스 시그널)
            if (f > m > s) and not (pf > pm > ps):
                buy_signals.append((df.index[i], df['Low'].iloc[i] * 0.98))
            # 역배열 진입 (데드크로스 시그널)
            elif (f < m < s) and not (pf < pm < ps):
                sell_signals.append((df.index[i], df['High'].iloc[i] * 1.02))

    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.5, 0.15, 0.18, 0.17],
        subplot_titles=(f"{ticker_name} 가격 차트", "거래량", "RSI (14)", "MACD")
    )

    # 1. 캔들스틱
    fig.add_trace(go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
        name="가격", increasing_line_color='#ef4444', decreasing_line_color='#3b82f6'
    ), row=1, col=1)

    # 볼린저밴드
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_U'], line=dict(color='rgba(150,150,150,0.3)', width=1), name='BB Upper', showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_L'], line=dict(color='rgba(150,150,150,0.3)', width=1), fill='tonexty', fillcolor='rgba(150,150,150,0.08)', name='BB', showlegend=False), row=1, col=1)

    # 이평선
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA10'], line=dict(color='#f59e0b', width=1.5), name='10일선'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA25'], line=dict(color='#10b981', width=1.5), name='25일선'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA75'], line=dict(color='#8b5cf6', width=2), name='75일선'), row=1, col=1)

    # 매수/매도 신호 마커
    if buy_signals:
        bx, by = zip(*buy_signals)
        fig.add_trace(go.Scatter(x=list(bx), y=list(by), mode='markers',
                                  marker=dict(symbol='triangle-up', size=14, color='#22c55e', line=dict(color='white', width=1)),
                                  name='매수 신호'), row=1, col=1)
    if sell_signals:
        sx, sy = zip(*sell_signals)
        fig.add_trace(go.Scatter(x=list(sx), y=list(sy), mode='markers',
                                  marker=dict(symbol='triangle-down', size=14, color='#dc2626', line=dict(color='white', width=1)),
                                  name='매도 신호'), row=1, col=1)

    # 2. 거래량
    colors = ['#ef4444' if c >= o else '#3b82f6' for c, o in zip(df['Close'], df['Open'])]
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], marker_color=colors, name='거래량', showlegend=False), row=2, col=1)

    # 3. RSI
    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], line=dict(color='#06b6d4', width=1.5), name='RSI', showlegend=False), row=3, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=3, col=1)

    # 4. MACD
    macd_colors = ['#ef4444' if v >= 0 else '#3b82f6' for v in df['HIST']]
    fig.add_trace(go.Bar(x=df.index, y=df['HIST'], marker_color=macd_colors, name='Hist', showlegend=False), row=4, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], line=dict(color='#f59e0b', width=1.3), name='MACD', showlegend=False), row=4, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['SIG'], line=dict(color='#8b5cf6', width=1.3), name='Signal', showlegend=False), row=4, col=1)

    fig.update_layout(
        height=750,
        xaxis_rangeslider_visible=False,
        hovermode='x unified',
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template='plotly_white',
        dragmode='pan',
    )
    fig.update_xaxes(rangeslider_visible=False)
    fig.update_yaxes(title_text=f"가격 ({currency})", row=1, col=1)
    fig.update_yaxes(title_text="거래량", row=2, col=1)
    fig.update_yaxes(title_text="RSI", row=3, col=1, range=[0, 100])
    fig.update_yaxes(title_text="MACD", row=4, col=1)

    return fig, df


# ============================================================
# [4. 나스닥 100 종목 자동 수집]
# ============================================================
@st.cache_data(ttl=3600)
def get_nasdaq_100():
    try:
        url = 'https://en.wikipedia.org/wiki/Nasdaq-100'
        dfs = pd.read_html(url)
        for df in dfs:
            if 'Ticker' in df.columns:
                return df['Ticker'].tolist()
            elif 'Symbol' in df.columns:
                return df['Symbol'].tolist()
    except:
        pass
    return ["AAPL","MSFT","NVDA","AMZN","META","GOOGL","GOOG","AVGO","TSLA","COST","PEP","CSCO","TMUS","ADBE","TXN","CMCSA","AMGN","INTU","ISRG","QCOM","AMD","SBUX","GILD","INTC","BKNG","MDLZ","ADI","ADP","VRTX","REGN","LRCX","MU","MELI","CSX","PANW","SNPS","CDNS","KLAC","PYPL","MAR","ORLY","MNST","NXPI","KHC","FTNT","CHTR","CTAS","KDP","DXCM","ABNB","MCHP","PAYX","PCAR","LULU","IDXX","CPRT","MRVL","ODFL","BIIB","EA","FAST","CTSH","WBA","BKR","VRSK","CSGP","ROST","FANG","DLTR","ANSS","EBAY","ALGN","ILMN","WBD"]


# ============================================================
# [5. 종목 분석 함수 (한국/미국 통합 로직)]
# ============================================================
def analyze_kr_stock(ticker_code, ticker_name, market_suffix, sl_pct, tp_pct):
    try:
        yf_ticker = yf.Ticker(f"{ticker_code}{market_suffix}")
        df = yf_ticker.history(period="1y")
        if df.empty or len(df) < 80: return None

        curr = df['Close'].iloc[-1]
        fast = df['Close'].rolling(10).mean().iloc[-1]
        mid = df['Close'].rolling(25).mean().iloc[-1]
        slow = df['Close'].rolling(75).mean().iloc[-1]
        rsi = calc_rsi(df['Close']).iloc[-1]
        score = curr / slow if slow != 0 else 0

        is_buy_cond = (fast > mid > slow) and (curr > fast)
        is_sell_cond = (fast < mid < slow)

        if is_buy_cond:
            status, reason = "🟢 매수 가능", "10>25>75일 정배열 및 10일선 지지"
        elif is_sell_cond:
            status, reason = "🔴 매도/회피", "완전 역배열 (지속 하락 추세)"
        elif fast > mid > slow:
            status, reason = "🟡 관망", "정배열이나 단기선 이탈 (조정/눌림목)"
        else:
            status, reason = "🟡 관망", "이동평균선 혼조세 (방향성 부재)"

        return {
            "종목명": ticker_name,
            "현재가": f"{curr:,.0f} 원",
            "이평강도": round(score, 4),
            "RSI": round(rsi, 1) if pd.notna(rsi) else 0,
            "매매판단": status,
            "분석사유": reason,
            "손절가": f"{curr * (1 - sl_pct/100):,.0f} 원",
            "익절가": f"{curr * (1 + tp_pct/100):,.0f} 원"
        }
    except: return None


def analyze_us_stock(ticker_code, sl_pct, tp_pct):
    try:
        yf_ticker = yf.Ticker(ticker_code)
        df = yf_ticker.history(period="1y")
        if df.empty or len(df) < 80: return None

        curr = df['Close'].iloc[-1]
        fast = df['Close'].rolling(10).mean().iloc[-1]
        mid = df['Close'].rolling(25).mean().iloc[-1]
        slow = df['Close'].rolling(75).mean().iloc[-1]
        rsi = calc_rsi(df['Close']).iloc[-1]
        score = curr / slow if slow != 0 else 0

        is_buy_cond = (fast > mid > slow) and (curr > fast)
        is_sell_cond = (fast < mid < slow)

        if is_buy_cond:
            status, reason = "🟢 매수 가능", "정배열 상승 추세 및 10일 이평선 지지"
        elif is_sell_cond:
            status, reason = "🔴 매도/회피", "완전 역배열 하락 지속"
        elif fast > mid > slow:
            status, reason = "🟡 관망", "정배열 상태이나 10일선 아래 (단기 조정)"
        else:
            status, reason = "🟡 관망", "이동평균선 꼬임 (방향성 대기)"

        try:
            name = yf_ticker.info.get('shortName', ticker_code)
        except:
            name = ticker_code

        return {
            "종목명": name,
            "현재가": f"$ {curr:,.2f}",
            "이평강도": round(score, 4),
            "RSI": round(rsi, 1) if pd.notna(rsi) else 0,
            "매매판단": status,
            "분석사유": reason,
            "손절가": f"$ {curr * (1 - sl_pct/100):,.2f}",
            "익절가": f"$ {curr * (1 + tp_pct/100):,.2f}"
        }
    except: return None


# ============================================================
# [6. 결과 표시 헬퍼 - 모바일 친화]
# ============================================================
def show_results(df_res, label_kr=True):
    buy_df = df_res[df_res['매매판단'] == "🟢 매수 가능"]
    hold_df = df_res[df_res['매매판단'] == "🟡 관망"]
    sell_df = df_res[df_res['매매판단'] == "🔴 매도/회피"]

    # 모바일에서 한눈에 보이도록 메트릭 표시
    c1, c2, c3 = st.columns(3)
    c1.metric("🟢 매수", f"{len(buy_df)}개")
    c2.metric("🟡 관망", f"{len(hold_df)}개")
    c3.metric("🔴 회피", f"{len(sell_df)}개")

    st.write("---")
    st.subheader(f"🟢 매수 가능 종목 ({len(buy_df)}개)")
    if not buy_df.empty:
        st.dataframe(buy_df, use_container_width=True, height=min(400, 50 + 35 * len(buy_df)))
    else: st.info("없음")

    with st.expander(f"🟡 관망 요망 종목 ({len(hold_df)}개)"):
        if not hold_df.empty: st.dataframe(hold_df, use_container_width=True)
        else: st.info("없음")

    with st.expander(f"🔴 매도/회피 종목 ({len(sell_df)}개)"):
        if not sell_df.empty: st.dataframe(sell_df, use_container_width=True)
        else: st.info("없음")


# ============================================================
# [7. 메인 UI]
# ============================================================
st.title("⚡ VTS 통합 트레이딩 시스템")
st.caption("📱 모바일 지원 | 📊 인터랙티브 차트 | 🎯 자동 매매 신호")

st.sidebar.header("⚙️ 매매 원칙 설정")
sl_pct = st.sidebar.slider("손절선 (%)", 1.0, 30.0, 5.0, 0.5)
tp_pct = st.sidebar.slider("익절선 (%)", 1.0, 50.0, 20.0, 0.5)
st.sidebar.markdown("---")
st.sidebar.info("📱 모바일에서는 좌상단 `>` 버튼으로 사이드바를 여세요.")

tab1, tab2, tab3, tab4 = st.tabs([
    "🇰🇷 국내 스캐너",
    "🇺🇸 해외 스캐너",
    "📊 정밀 차트 분석",
    "📖 매뉴얼"
])

# ===== 탭 1: 한국 주식 =====
with tab1:
    st.subheader("국내 전 종목 분석 리포트")
    market_select = st.radio("시장 선택", ["코스피(KOSPI)", "코스닥(KOSDAQ)"], horizontal=True)
    market_code = "KOSPI" if "KOSPI" in market_select else "KOSDAQ"
    market_suffix = ".KS" if market_code == "KOSPI" else ".KQ"
    scan_limit = st.slider("스캔할 종목 수 (상위 시총순)", 20, 300, 100, 10)

    if st.button("🔍 한국 시장 분석 시작", use_container_width=True):
        with st.spinner(f'{market_code} 종목 목록을 가져오는 중...'):
            krx_df = fdr.StockListing(market_code)
            ticker_list = krx_df[['Code', 'Name']].values.tolist()

        if ticker_list:
            results = []
            prog = st.progress(0)
            with st.spinner(f'상위 {scan_limit}개 종목 스캔 중...'):
                for i, (t_code, t_name) in enumerate(ticker_list[:scan_limit]):
                    info = analyze_kr_stock(t_code, t_name, market_suffix, sl_pct, tp_pct)
                    if info: results.append({"티커": t_code, **info})
                    prog.progress((i + 1) / scan_limit)

            if results:
                df_res = pd.DataFrame(results).sort_values(by="이평강도", ascending=False).reset_index(drop=True)
                df_res.index = df_res.index + 1
                show_results(df_res)
        else:
            st.error("종목 리스트를 가져오지 못했습니다.")

# ===== 탭 2: 해외 주식 =====
with tab2:
    st.subheader("🇺🇸 미국 주요 주식 스캐너 (나스닥 100)")
    use_nasdaq_100 = st.checkbox("나스닥 100 편입 종목 전체 스캔", value=True)

    if use_nasdaq_100:
        us_tickers = get_nasdaq_100()
        st.info(f"나스닥 100에 편입된 {len(us_tickers)}개 우량 기업 분석")
    else:
        us_tickers_input = st.text_input("티커를 직접 입력 (쉼표 구분)", "AAPL, MSFT, NVDA, GOOGL, AMZN, META, TSLA")
        us_tickers = [t.strip() for t in us_tickers_input.split(',')]

    if st.button("🔍 미국 시장 스캔 시작", use_container_width=True):
        results_us = []
        prog_us = st.progress(0)
        with st.spinner(f'미국 주식 {len(us_tickers)}개 분석 중...'):
            for i, t_code in enumerate(us_tickers):
                info = analyze_us_stock(t_code, sl_pct, tp_pct)
                if info: results_us.append({"티커": t_code, **info})
                prog_us.progress((i + 1) / len(us_tickers))

        if results_us:
            df_res_us = pd.DataFrame(results_us).sort_values(by="이평강도", ascending=False).reset_index(drop=True)
            df_res_us.index = df_res_us.index + 1
            show_results(df_res_us)

            buy_count = len(df_res_us[df_res_us['매매판단'] == "🟢 매수 가능"])
            st.write("---")
            if buy_count > 0:
                st.success(f"💡 미국 우량 기술주 중 상승 추세 진입 종목 {buy_count}개 확인. 글로벌 유동성 확장에 대비하여 편입 고려.")
            else:
                st.warning("💡 나스닥 우량주 중 상승 추세 종목 없음. 단기 조정 의심, 현금 비중 확대 권장.")

# ===== 탭 3: 정밀 차트 분석 (인라인) =====
with tab3:
    st.subheader("📊 단일 종목 인터랙티브 정밀 분석")
    st.caption("이평선 + 볼린저밴드 + 거래량 + RSI + MACD + 자동 매매 신호 마커")

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        chart_market = st.selectbox("시장", ["🇺🇸 미국", "🇰🇷 한국 (KOSPI)", "🇰🇷 한국 (KOSDAQ)"])
    with col2:
        chart_period = st.selectbox("기간", ["6mo", "1y", "2y", "5y"], index=1)
    with col3:
        run_bt = st.checkbox("백테스트 포함", value=True)

    if chart_market == "🇺🇸 미국":
        chart_ticker = st.text_input("티커 입력", "NVDA").strip().upper()
        full_ticker = chart_ticker
        currency = "$"
    else:
        chart_ticker = st.text_input("종목 코드 입력 (예: 005930)", "005930").strip()
        suffix = ".KS" if "KOSPI" in chart_market else ".KQ"
        full_ticker = f"{chart_ticker}{suffix}"
        currency = "₩"

    if st.button("📈 분석 실행", use_container_width=True):
        with st.spinner(f'{chart_ticker} 데이터 분석 중...'):
            try:
                df = yf.download(full_ticker, period=chart_period, progress=False, auto_adjust=False)
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                df = df.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume'])

                if df.empty or len(df) < 80:
                    st.error("데이터가 충분하지 않습니다. 다른 종목/기간을 시도하세요.")
                else:
                    # 현재 상태 메트릭
                    curr = df['Close'].iloc[-1]
                    prev = df['Close'].iloc[-2]
                    chg_pct = (curr - prev) / prev * 100
                    fast = df['Close'].rolling(10).mean().iloc[-1]
                    mid = df['Close'].rolling(25).mean().iloc[-1]
                    slow = df['Close'].rolling(75).mean().iloc[-1]
                    rsi_val = calc_rsi(df['Close']).iloc[-1]

                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("현재가", f"{currency}{curr:,.2f}", f"{chg_pct:+.2f}%")
                    m2.metric("이평강도", f"{curr/slow:.3f}" if slow else "N/A")
                    m3.metric("RSI(14)", f"{rsi_val:.1f}")
                    if fast > mid > slow:
                        trend = "🟢 정배열" if curr > fast else "🟡 조정"
                    elif fast < mid < slow:
                        trend = "🔴 역배열"
                    else:
                        trend = "🟡 혼조"
                    m4.metric("추세", trend)

                    # 인터랙티브 차트
                    fig, df_with_ind = make_advanced_chart(df, chart_ticker, currency)
                    st.plotly_chart(fig, use_container_width=True, config={
                        'displayModeBar': True,
                        'displaylogo': False,
                        'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
                        'scrollZoom': True,
                    })

                    # 손절/익절 가이드
                    st.write("---")
                    st.subheader("🎯 매매 가이드")
                    g1, g2, g3 = st.columns(3)
                    g1.metric("진입가 (현재가)", f"{currency}{curr:,.2f}")
                    g2.metric(f"손절가 (-{sl_pct}%)", f"{currency}{curr*(1-sl_pct/100):,.2f}")
                    g3.metric(f"익절가 (+{tp_pct}%)", f"{currency}{curr*(1+tp_pct/100):,.2f}")

                    # 백테스트 (인라인 결과)
                    if run_bt:
                        st.write("---")
                        st.subheader("🧪 전략 백테스트 결과")
                        with st.spinner("백테스트 실행 중..."):
                            bt = Backtest(df, VitachoStrategy, cash=10000, commission=.0015)
                            stats = bt.run()
                            s = stats.to_dict()

                            b1, b2, b3, b4 = st.columns(4)
                            b1.metric("최종 자산", f"${s.get('Equity Final [$]', s.get('Equity Final', 0)):,.2f}")
                            b2.metric("총 수익률", f"{s.get('Return [%]', 0):.2f}%")
                            b3.metric("Buy&Hold 수익률", f"{s.get('Buy & Hold Return [%]', 0):.2f}%")
                            b4.metric("최대 낙폭", f"{s.get('Max. Drawdown [%]', 0):.2f}%")

                            b5, b6, b7, b8 = st.columns(4)
                            b5.metric("승률", f"{s.get('Win Rate [%]', 0):.1f}%")
                            b6.metric("거래 횟수", f"{int(s.get('# Trades', 0))}")
                            b7.metric("샤프 비율", f"{s.get('Sharpe Ratio', 0):.2f}")
                            b8.metric("수익팩터", f"{s.get('Profit Factor', 0):.2f}")

                            with st.expander("전체 백테스트 통계 보기"):
                                stat_df = pd.DataFrame(
                                    [(k, str(v)) for k, v in s.items() if not k.startswith('_')],
                                    columns=['지표', '값']
                                )
                                st.dataframe(stat_df, use_container_width=True, hide_index=True)
            except Exception as e:
                st.error(f"분석 중 오류: {e}")

# ===== 탭 4: 매뉴얼 =====
with tab4:
    st.header("📖 Vitacho 트레이딩 시스템 로직")
    st.markdown("""
    이 시스템은 유명 트레이더 **'테스타(Testa)'**의 철학인 **"생존 우선, 손실 제한, 추세 순응"**을 기반으로 설계된 퀀트 스캐너입니다.
    시장의 휩쏘(가짜 돌파)를 피하고 확실한 대추세(정배열)에만 탑승하는 것을 목표로 합니다.
    """)

    st.write("---")
    st.subheader("1. 핵심 지표 (Moving Averages)")
    st.markdown("""
    - **단기선 (10일):** 단기 매수세, 진입 타점 기준
    - **중기선 (25일):** 추세의 1차 지지선
    - **장기선 (75일):** 시장 대추세를 결정짓는 생명선
    """)

    st.subheader("2. 종목 필터링 기준")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.success("🟢 매수 가능")
        st.write("- 10>25>75 정배열\n- 종가가 10일선 위\n- → 주도주/상승장")
    with col2:
        st.warning("🟡 관망")
        st.write("- 정배열이나 10일선 이탈\n- 또는 이평선 꼬임\n- → 조정/횡보")
    with col3:
        st.error("🔴 매도/회피")
        st.write("- 10<25<75 역배열\n- 매도세 압도적\n- → 절대 금지")

    st.write("---")
    st.subheader("3. 보조 지표 (정밀 차트 탭)")
    st.markdown("""
    - **볼린저밴드 (20, 2σ):** 변동성 범위와 과매수/과매도 영역 시각화
    - **RSI (14):** 70 이상 과매수, 30 이하 과매도. 이평선과 함께 보조 판단
    - **MACD (12, 26, 9):** 추세 전환 시점 포착. 히스토그램이 양수→음수 전환 시 주의
    - **자동 매매 신호 마커:** 차트에 정배열 진입(▲), 역배열 진입(▼) 자동 표시
    """)

    st.subheader("4. 이평강도 (Trend Strength Score)")
    st.markdown("""
    - **계산식:** `현재가 ÷ 75일 이동평균선`
    - **1.0 이상**일수록 75일선 위에서 강하게 뻗어나가는 대장주
    """)

    st.subheader("5. 리스크 관리")
    st.markdown("""
    - 사이드바에서 손절선/익절선 % 설정 → 자동으로 타겟 가격 제시
    - HTS/MTS에서 해당 가격에 **자동 매도(스탑로스)** 설정 권장
    """)

    st.write("---")
    st.subheader("📱 모바일 사용 팁")
    st.markdown("""
    - 좌상단 `>` 버튼으로 사이드바(설정) 열기
    - 차트는 두 손가락으로 확대/축소 가능
    - 표는 좌우 스크롤 지원
    - 관망/회피 종목은 접혀있음 → 클릭하여 펼치기
    """)
