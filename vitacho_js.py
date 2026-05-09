import streamlit as st
import pandas as pd
import yfinance as yf
from backtesting import Backtest, Strategy
import FinanceDataReader as fdr

#[1. 전략 로직 (백테스트용)]
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

#[나스닥 100 종목 자동 수집 함수]
@st.cache_data
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
    return["AAPL","MSFT","NVDA","AMZN","META","GOOGL","GOOG","AVGO","TSLA","COST","PEP","CSCO","TMUS","ADBE","TXN","CMCSA","AMGN","INTU","ISRG","QCOM","AMD","SBUX","GILD","INTC","BKNG","MDLZ","ADI","ADP","VRTX","REGN","LRCX","MU","MELI","CSX","PANW","SNPS","CDNS","KLAC","PYPL","MAR","ORLY","MNST","NXPI","KHC","FTNT","CHTR","CTAS","KDP","DXCM","ABNB","MCHP","PAYX","PCAR","LULU","IDXX","CPRT","MRVL","ODFL","BIIB","EA","FAST","CTSH","WBA","BKR","VRSK","CSGP","ROST","FANG","DLTR","ANSS","EBAY","ALGN","ILMN","WBD"]

#[2. 국내 종목 분석 함수 (분석사유 추가)]
def analyze_kr_stock(ticker_code, ticker_name, market_suffix, sl_pct, tp_pct):
    try:
        yf_ticker = yf.Ticker(f"{ticker_code}{market_suffix}")
        df = yf_ticker.history(period="1y")
        if df.empty or len(df) < 80: return None

        curr = df['Close'].iloc[-1]
        fast, mid, slow = df['Close'].rolling(10).mean().iloc[-1], df['Close'].rolling(25).mean().iloc[-1], df['Close'].rolling(75).mean().iloc[-1]
        score = curr / slow if slow != 0 else 0
        
        is_buy_cond = (fast > mid > slow) and (curr > fast)
        is_sell_cond = (fast < mid < slow)
        
        if is_buy_cond: 
            status = "🟢 매수 가능"
            reason = "10>25>75일 정배열 및 10일선 지지"
        elif is_sell_cond: 
            status = "🔴 매도/회피"
            reason = "완전 역배열 (지속 하락 추세)"
        elif fast > mid > slow:
            status = "🟡 관망"
            reason = "정배열이나 단기선(10일) 이탈로 인한 조정/눌림목"
        else: 
            status = "🟡 관망"
            reason = "이동평균선 혼조세 (방향성 부재)"
        
        return {
            "종목명": ticker_name, 
            "현재가": f"{curr:,.0f} 원",
            "이평강도": round(score, 4),
            "매매판단": status,
            "분석사유": reason,
            "손절가": f"{curr * (1 - sl_pct/100):,.0f} 원",
            "익절가": f"{curr * (1 + tp_pct/100):,.0f} 원"
        }
    except: return None

#[3. 해외 종목 분석 함수 (분석사유 추가)]
def analyze_us_stock(ticker_code, sl_pct, tp_pct):
    try:
        yf_ticker = yf.Ticker(ticker_code)
        df = yf_ticker.history(period="1y")
        if df.empty or len(df) < 80: return None

        curr = df['Close'].iloc[-1]
        fast, mid, slow = df['Close'].rolling(10).mean().iloc[-1], df['Close'].rolling(25).mean().iloc[-1], df['Close'].rolling(75).mean().iloc[-1]
        score = curr / slow if slow != 0 else 0
        
        is_buy_cond = (fast > mid > slow) and (curr > fast)
        is_sell_cond = (fast < mid < slow)
        
        if is_buy_cond: 
            status = "🟢 매수 가능"
            reason = "정배열 상승 추세 및 10일 이평선 지지"
        elif is_sell_cond: 
            status = "🔴 매도/회피"
            reason = "완전 역배열 하락 지속"
        elif fast > mid > slow:
            status = "🟡 관망"
            reason = "정배열 상태이나 10일선 아래 위치 (단기 조정)"
        else: 
            status = "🟡 관망"
            reason = "이동평균선 꼬임 (방향성 대기)"
        
        name = yf_ticker.info.get('shortName', ticker_code)

        return {
            "종목명": name, 
            "현재가": f"$ {curr:,.2f}",
            "이평강도": round(score, 4),
            "매매판단": status,
            "분석사유": reason,
            "손절가": f"$ {curr * (1 - sl_pct/100):,.2f}",
            "익절가": f"$ {curr * (1 + tp_pct/100):,.2f}"
        }
    except: return None

#[4. 대시보드 메인]
st.set_page_config(layout="wide", page_title="Vitacho Trading System")
st.title("⚡ VTS 통합 트레이딩 시스템")

st.sidebar.header("매매 원칙 설정")
sl_pct = st.sidebar.slider("손절선 (%)", 1.0, 30.0, 5.0, 0.5)
tp_pct = st.sidebar.slider("익절선 (%)", 1.0, 50.0, 20.0, 0.5)

# 탭 3개로 확장
tab1, tab2, tab3 = st.tabs(["🇰🇷 국내 시장 스캐너", "🇺🇸 해외 시장 (나스닥 100)", "📖 전략 로직 및 매뉴얼"])

# ===== 탭 1: 한국 주식 =====
with tab1:
    st.subheader("국내 전 종목 분석 리포트")
    market_select = st.radio("시장 선택",["코스피(KOSPI)", "코스닥(KOSDAQ)"])
    market_code = "KOSPI" if "KOSPI" in market_select else "KOSDAQ"
    market_suffix = ".KS" if market_code == "KOSPI" else ".KQ"
    
    if st.button("한국 시장 분석 시작"):
        with st.spinner(f'{market_code} 종목 목록을 가져오는 중...'):
            krx_df = fdr.StockListing(market_code)
            ticker_list = krx_df[['Code', 'Name']].values.tolist()
        
        if ticker_list:
            results =[]
            prog = st.progress(0)
            scan_limit = 100 
            with st.spinner(f'상위 {scan_limit}개 종목 스캔 중...'):
                for i, (t_code, t_name) in enumerate(ticker_list[:scan_limit]):
                    info = analyze_kr_stock(t_code, t_name, market_suffix, sl_pct, tp_pct)
                    if info: results.append({"티커": t_code, **info})
                    prog.progress((i + 1) / scan_limit)
            
            if results:
                df_res = pd.DataFrame(results).sort_values(by="이평강도", ascending=False).reset_index(drop=True)
                df_res.index = df_res.index + 1 
                
                buy_df = df_res[df_res['매매판단'] == "🟢 매수 가능"]
                hold_df = df_res[df_res['매매판단'] == "🟡 관망"]
                sell_df = df_res[df_res['매매판단'] == "🔴 매도/회피"]
                
                st.write("---")
                st.subheader(f"🟢 매수 가능 종목 ({len(buy_df)}개)")
                if not buy_df.empty: st.dataframe(buy_df, width='stretch')
                else: st.info("없음")

                st.subheader(f"🟡 관망 요망 종목 ({len(hold_df)}개)")
                if not hold_df.empty: st.dataframe(hold_df, width='stretch')
                else: st.info("없음")

                st.subheader(f"🔴 매도/회피 종목 ({len(sell_df)}개)")
                if not sell_df.empty: st.dataframe(sell_df, width='stretch')
                else: st.info("없음")
        else: st.error("종목 리스트를 가져오지 못했습니다.")

# ===== 탭 2: 해외 주식 (나스닥 100 특화) =====
with tab2:
    st.subheader("🇺🇸 미국 주요 주식 스캐너 (나스닥 100 집중)")
    
    use_nasdaq_100 = st.checkbox("나스닥 100 (NASDAQ 100) 편입 종목 전체 스캔하기", value=True)
    
    if use_nasdaq_100:
        us_tickers = get_nasdaq_100()
        st.info(f"현재 나스닥 100에 편입된 {len(us_tickers)}개 우량 기업을 대상으로 분석합니다.")
    else:
        default_us_tickers = "AAPL, MSFT, NVDA, GOOGL, AMZN, META, TSLA"
        us_tickers_input = st.text_input("분석할 티커를 직접 입력 (쉼표로 구분)", default_us_tickers)
        us_tickers =[t.strip() for t in us_tickers_input.split(',')]
    
    if st.button("미국 시장 스캔 시작"):
        results_us =[]
        prog_us = st.progress(0)
        
        with st.spinner(f'미국 주식 {len(us_tickers)}개 분석 중... (약 20~30초 소요)'):
            for i, t_code in enumerate(us_tickers):
                info = analyze_us_stock(t_code, sl_pct, tp_pct)
                if info: results_us.append({"티커": t_code, **info})
                prog_us.progress((i + 1) / len(us_tickers))
        
        if results_us:
            df_res_us = pd.DataFrame(results_us).sort_values(by="이평강도", ascending=False).reset_index(drop=True)
            df_res_us.index = df_res_us.index + 1 
            
            buy_df_us = df_res_us[df_res_us['매매판단'] == "🟢 매수 가능"]
            hold_df_us = df_res_us[df_res_us['매매판단'] == "🟡 관망"]
            sell_df_us = df_res_us[df_res_us['매매판단'] == "🔴 매도/회피"]
            
            st.write("---")
            st.subheader(f"🟢 매수 가능 종목 ({len(buy_df_us)}개)")
            if not buy_df_us.empty: st.dataframe(buy_df_us, width='stretch')
            else: st.info("없음")

            st.subheader(f"🟡 관망 요망 종목 ({len(hold_df_us)}개)")
            if not hold_df_us.empty: st.dataframe(hold_df_us, width='stretch')
            else: st.info("없음")

            st.subheader(f"🔴 매도/회피 종목 ({len(sell_df_us)}개)")
            if not sell_df_us.empty: st.dataframe(sell_df_us, width='stretch')
            else: st.info("없음")
            
            st.write("---")
            st.write("### 💡 글로벌 투자 시사점")
            if len(buy_df_us) > 0: 
                st.success(f"미국 우량 기술주 중 상승 추세에 진입한 {len(buy_df_us)}개 종목이 확인되었습니다. 글로벌 유동성 확장에 대비하여 편입을 고려하세요.")
            else: 
                st.warning("나스닥 우량주 중 상승 추세인 종목이 없습니다. 글로벌 시장 전체의 단기 조정을 의심하고 현금 비중을 확대하세요.")
    
    st.write("---")
    st.subheader("📊 단일 종목 정밀 백테스트 (차트 확인용)")
    bt_ticker = st.text_input("백테스트할 티커 한 개 입력", "NVDA")
    if st.button("백테스트 실행"):
        with st.spinner(f'{bt_ticker} 백테스트 진행 중...'):
            df = yf.download(bt_ticker, start="2023-01-01", progress=False)
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            df = df.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume'])
            
            bt = Backtest(df, VitachoStrategy, cash=10000, commission=.0015)
            stats = bt.run()
            s = stats.to_dict()
            st.table(pd.DataFrame({"지표":["최종 자산 (USD)", "총 수익률(%)"], "값":[f"$ {s.get('Equity Final', 0):,.2f}", f"{s.get('Return [%]', 0):.2f}%"]}))
            bt.plot(filename='chart.html')
            st.components.v1.html(open('chart.html', 'r', encoding='utf-8').read(), height=800)

# ===== 탭 3: 전략 로직 및 매뉴얼 =====
with tab3:
    st.header("📖 Vitacho 트레이딩 시스템 로직 (테스타 스타일)")
    
    st.markdown("""
    이 시스템은 유명 트레이더 **'테스타(Testa)'**의 철학인 **"생존 우선, 손실 제한, 추세 순응"**을 기반으로 설계된 퀀트 스캐너입니다.  
    시장의 휩쏘(가짜 돌파)를 피하고 확실한 대추세(정배열)에만 탑승하는 것을 목표로 합니다.
    """)

    st.write("---")
    
    st.subheader("1. 핵심 지표 (Moving Averages)")
    st.markdown("""
    * **단기선 (10일 이동평균선):** 단기적인 매수세와 진입 타점을 잡는 기준선.
    * **중기선 (25일 이동평균선):** 추세의 1차 지지선 및 익절/손절의 주요 기준선.
    * **장기선 (75일 이동평균선):** 시장의 대추세를 결정짓는 생명선.
    """)

    st.subheader("2. 종목 필터링 기준 (매매 판단)")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.success("🟢 매수 가능 (Buy)")
        st.write("""
        - **조건 1:** 10일선 > 25일선 > 75일선 (완벽한 정배열)
        - **조건 2:** 현재 종가가 10일선 위에 위치
        - **해석:** 대추세가 상승 중이며, 단기적으로도 지지를 받고 있는 **주도주/상승장** 구간입니다.
        """)
        
    with col2:
        st.warning("🟡 관망 (Hold/Wait)")
        st.write("""
        - **조건:** 정배열이지만 현재가가 10일선 아래로 무너졌거나, 이평선이 서로 꼬여있는 상태.
        - **해석:** 추세가 쉬어가는 **조정/눌림목** 구간이거나 방향성이 없는 횡보장입니다. 신규 매수를 피하고 기다립니다.
        """)
        
    with col3:
        st.error("🔴 매도/회피 (Sell/Avoid)")
        st.write("""
        - **조건:** 10일선 < 25일선 < 75일선 (완전한 역배열)
        - **해석:** 매도세가 압도적인 **하락장**입니다. 절대 매수해서는 안 되는 종목입니다.
        """)

    st.write("---")
    
    st.subheader("3. 이평강도 (Trend Strength Score)")
    st.markdown("""
    표의 정렬 기준이 되는 **이평강도**는 현재 시장에서 '어떤 종목이 가장 힘이 좋은가'를 나타냅니다.
    * **계산식:** `현재가 ÷ 75일 이동평균선`
    * 수치가 **1.0 이상**일수록 75일선 위에서 강하게 뻗어나가는 대장주임을 의미합니다.
    """)

    st.subheader("4. 리스크 관리 (손절/익절 자동화)")
    st.markdown("""
    테스타의 핵심 원칙은 **"예측이 틀렸을 때 빠르게 잘라내는 것"**입니다.
    * 왼쪽 사이드바에서 손절선(기본 5%)과 익절선(기본 20%)을 설정하면, 시스템이 실시간 현재가를 반영하여 **정확한 타겟 가격**을 표에 제시합니다.
    * HTS/MTS에서 해당 가격에 **'자동 매도(스탑로스)'**를 걸어두어 뇌동매매를 방지하십시오.
    """)

    st.write("---")
    with st.expander("💻 실제 시스템에 적용된 Python 핵심 코드 보기"):
        st.code('''
        # 현재가 및 이동평균선 계산
        curr = df['Close'].iloc[-1]
        fast = df['Close'].rolling(10).mean().iloc[-1]
        mid = df['Close'].rolling(25).mean().iloc[-1]
        slow = df['Close'].rolling(75).mean().iloc[-1]
        
        # 1. 이평강도(스코어) 계산
        score = curr / slow
        
        # 2. 매수 판단 로직 (정배열 + 10일선 지지)
        is_buy_cond = (fast > mid > slow) and (curr > fast)
        
        # 3. 매도 판단 로직 (역배열)
        is_sell_cond = (fast < mid < slow)
        ''', language='python')