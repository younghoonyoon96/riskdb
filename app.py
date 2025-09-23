# -*- coding: utf-8 -*-
"""
RiskDash — 멀티페이지 구성 (기업/업종/외국인 · 인덱스 · FinBERT · LLM 요약+채팅)

주요 수정:
- AG Grid(기사수 표) 폰트/행높이/컬럼폭 확대 + 다크테마 고정(balham-dark)
- 워드클라우드 한글 폰트 자동탐지 후 font_path 적용 (깨짐 해결)
"""

from __future__ import annotations
import os
from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sqlalchemy import create_engine, text
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, JsCode
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import plotly.io as pio
from pathlib import Path
from PIL import Image

# === 로고 경로 ===
ASSETS_DIR = Path(__file__).parent / "assets"
LOGO_PATH = ASSETS_DIR / "R2_logo.png"

# ======== 다크 테마 기본값 ========
pio.templates.default = "plotly_dark"
st.set_page_config(
    page_title="RiskDash",
    layout="wide",
    page_icon=str(LOGO_PATH) if LOGO_PATH.exists() else None  # 브라우저 탭 아이콘
)

# ======== (전역) 다크 스타일 + AgGrid 폰트 확대 CSS 주입 ========
def inject_global_css():
    st.markdown(
        """
        <style>
          /* 페이지 베이스 톤(다크) */
          .stApp { background-color: #0f1116; }
          .stMarkdown, .stText, .stCaption, .stCode, .stHeader { color: #e6e6e6; }

          /* AG Grid 다크 테마 폰트/행 높이/여백 */
          .ag-theme-balham-dark {
            --ag-foreground-color: #e6e6e6;
            --ag-background-color: #141821;
            --ag-header-foreground-color: #e6e6e6;
            --ag-header-background-color: #1b2030;
            --ag-odd-row-background-color: #141821;
            --ag-row-hover-color: #1f2635;
            font-size: 15px;              /* ← 숫자 작게 보이는 문제 개선 */
          }
          .ag-theme-balham-dark .ag-header-cell {
            font-weight: 700;
            font-size: 15px;
          }
          .ag-theme-balham-dark .ag-cell {
            line-height: 1.4;
            padding-top: 6px;
            padding-bottom: 6px;
          }

          /* 표 컨테이너 여백 */
          .block-container { padding-top: 1.2rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )

inject_global_css()

# -----------------------------
# (선택) Gemini — 없으면 해당 페이지에서 안내만 표시
# -----------------------------
try:
    from google import genai
    from google.genai import types as gtypes
    from pydantic import BaseModel
except Exception:
    genai = None
    gtypes = None
    BaseModel = None

# -----------------------------
# 공용 유틸/캐시
# -----------------------------
@st.cache_resource
def get_engine():
    """PostgreSQL 엔진. DATABASE_URL 미설정 시 안내 후 중단."""
    url = os.getenv("DATABASE_URL")
    if not url:
        try:
            url = st.secrets["DATABASE_URL"]
        except Exception:
            url = None
    if not url:
        st.error(
            "DATABASE_URL이 설정되지 않았습니다.\n"
            "터미널에서 export 하거나(.zshrc), 프로젝트의 .streamlit/secrets.toml에 DATABASE_URL을 넣어주세요."
        )
        st.stop()
    return create_engine(url, pool_pre_ping=True, future=True)

@st.cache_data(ttl=120)
def read_df(sql: str, params: dict | None = None) -> pd.DataFrame:
    eng = get_engine()
    with eng.connect() as conn:
        return pd.read_sql_query(text(sql), conn, params=params or {})

def fmt_company(ticker: str, name_map: dict[str, str]) -> str:
    nm = name_map.get(ticker)
    return f"{nm} ({ticker})" if nm and nm != ticker else ticker

def _rerun():
    if hasattr(st, "rerun"): st.rerun()
    elif hasattr(st, "experimental_rerun"): st.experimental_rerun()

# -----------------------------
# 워드클라우드/한국어 처리 유틸
# -----------------------------
def _get_korean_font_path() -> str | None:
    # macOS / Windows / Linux 후보
    cands = [
        "/System/Library/Fonts/AppleSDGothicNeo.ttc",                 # macOS
        "C:/Windows/Fonts/malgun.ttf", "C:/Windows/Fonts/malgunbd.ttf",  # Windows
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",            # Ubuntu (nanum)
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",     # Noto CJK
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    ]
    for p in cands:
        if os.path.exists(p):
            return p
    return None

_KR_FONT = _get_korean_font_path()

_DEFAULT_STOPS = set(["기사","속보","단독","종합","영상","포토","외","…","무"])

def simple_tokenize_korean(titles: list[str]) -> str:
    import re
    tokens = []
    for t in titles:
        t = str(t)
        t = re.sub(r"http\S+|www\.\S+", " ", t)
        t = re.sub(r"[^0-9A-Za-z가-힣\s]", " ", t)
        for w in t.split():
            w = w.strip()
            if len(w) >= 2 and w not in _DEFAULT_STOPS:
                tokens.append(w)
    return " ".join(tokens)

def render_wordcloud(titles: list[str], sentiment: str):
    """다크테마에 맞춘 색 + 한글 폰트 고정"""
    color_map = {
        "positive": ("#2ecc71", "white"),
        "negative": ("#e74c3c", "white"),
        "neutral": ("#ffffff", "#1e1e1e"),
    }
    word_color, bg_color = color_map.get(sentiment, ("#3498db", "white"))

    def mono_color_func(*args, **kwargs):
        return word_color

    text = simple_tokenize_korean(titles)
    wc = WordCloud(
        width=1100, height=550, background_color=bg_color,
        font_path=_KR_FONT,              # ★ 한글 폰트 지정 (없으면 None → 경고)
        regexp=r"[A-Za-z가-힣0-9]+"
    ).generate(text)
    wc.recolor(color_func=mono_color_func)

    if _KR_FONT is None:
        st.warning("서버에서 한글 폰트를 찾지 못했습니다. 워드클라우드가 깨지면 `sudo apt-get install fonts-nanum` 후 재실행 하세요.", icon="⚠️")

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

# -----------------------------
# PD 단위/스무딩 유틸
# -----------------------------
def pd_scale(df: pd.DataFrame, cols: list[str], unit: str, smooth_n: int) -> pd.DataFrame:
    x = df.sort_values("date").copy()
    factor = 10_000.0 if unit.startswith("bp") else 100.0
    for c in cols:
        if c in x.columns:
            x[c] = (pd.to_numeric(x[c], errors="coerce") * factor).rolling(smooth_n, min_periods=1).mean()
    return x

# -----------------------------
# 외국인 보유비율 정규화 유틸
# -----------------------------
def normalize_ratio_to_pct(x) -> float | None:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    try:
        v = float(x)
    except Exception:
        return None
    if v < 0:
        return None
    if v <= 1.0:      return round(v * 100.0, 4)
    if v <= 100.0:    return round(v, 4)
    if v <= 10000.0:  return round(v / 100.0, 4)
    return round(v, 4)

def series_to_pct(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    if x.dropna().empty:
        return x
    ref = x.dropna().quantile(0.9)
    if not np.isfinite(ref):
        ref = x.dropna().max()
    if ref <= 1.5:        y = x * 100.0
    elif ref <= 100.0:    y = x
    elif ref <= 10000.0:  y = x / 100.0
    else:                 y = x
    return y.round(4)

# -----------------------------
# 상단 타이틀 & 글로벌 컨트롤바  (로고 포함 헤더)
# -----------------------------
def render_topbar():
    # 헤더 스타일 (다크)
    st.markdown("""
    <style>
      .topbar { display:flex; align-items:center; gap:14px; margin-bottom:8px; }
      .topbar .title { font-size:28px; font-weight:800; margin:0; color:#e6e6e6; }
      .topbar .sub { color:#9aa4b2; margin:2px 0 0 0; }
      .topbar .btn-wrap { margin-left:auto; min-width:220px; }
      .topbar img { border-radius:10px; }
    </style>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns([0.13, 0.62, 0.25], gap="small")

    with c1:
        if LOGO_PATH.exists():
            st.image(str(LOGO_PATH), use_container_width=True)
        else:
            st.write("")  # 로고 없을 때 자리만 차지

    with c2:
        st.markdown("<div class='topbar'>"
                    "<div>"
                    "<div class='title'>📊 RiskDash</div>"
                    "<div class='sub'>KOSPI/KOSDAQ · PD(EWMA) · 업종평균 · 외국인 · 인덱스 · FinBERT · Gemini</div>"
                    "</div>"
                    "</div>", unsafe_allow_html=True)

    with c3:
        st.markdown("<div class='btn-wrap'></div>", unsafe_allow_html=True)
        if st.button("🔄 데이터 새로고침", use_container_width=True):
            try: read_df.clear()
            except Exception: pass
            try: st.cache_data.clear()
            except Exception: pass
            _rerun()

# 호출
render_topbar()
st.divider()

# -----------------------------
# 페이지 선택
# -----------------------------
st.markdown("### 🔎 섹션 선택")
if "page" not in st.session_state:
    st.session_state["page"] = "기업/업종/외국인"

page = st.radio(
    "섹션 선택",
    options=["기업/업종/외국인", "인덱스", "FinBERT", "LLM 종합의견"],
    index=["기업/업종/외국인", "인덱스", "FinBERT", "LLM 종합의견"].index(st.session_state["page"]),
    horizontal=True,
    label_visibility="collapsed",
    key="page_radio_main",
)
st.session_state["page"] = page
st.divider()

# -----------------------------
# 메타 로드 (회사/기간 범위)
# -----------------------------
companies = read_df("""
    SELECT ticker, company_name, market, ksic_mid_code, ksic_mid_name
    FROM companies
    ORDER BY company_name NULLS LAST, ticker
""")
name_map = dict(zip(companies["ticker"], companies["company_name"])) if not companies.empty else {}

minmax = read_df("SELECT MIN(date) AS min_d, MAX(date) AS max_d FROM pd_daily")
if not minmax.empty and pd.notna(minmax.loc[0, "max_d"]):
    default_end = pd.to_datetime(minmax.loc[0, "max_d"]).date()
    default_start = default_end - timedelta(days=180)
else:
    default_end = date.today()
    default_start = default_end - timedelta(days=180)

# -----------------------------
# 사이드바 — 전역 필터
# -----------------------------

with st.sidebar:
    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH), use_container_width=True)
    st.header("전역 필터")

st.sidebar.header("전역 필터")
markets = sorted(companies["market"].dropna().unique().tolist()) if not companies.empty else []
sel_markets = st.sidebar.multiselect("시장", markets, default=markets)

comp_by_mkt = companies[companies["market"].isin(sel_markets)] if sel_markets else companies
ind_options = (
    comp_by_mkt[["ksic_mid_code", "ksic_mid_name"]]
    .drop_duplicates()
    .sort_values(["ksic_mid_code", "ksic_mid_name"])
)
ind_labels = ind_options.apply(lambda r: f"{r['ksic_mid_code'] or '-'} · {r['ksic_mid_name'] or '-'}", axis=1).tolist()
ind_pairs = list(zip(ind_labels, ind_options["ksic_mid_code"].tolist(), ind_options["ksic_mid_name"].tolist()))
sel_inds = st.sidebar.multiselect("업종(중분류)", options=[lab for lab, _, _ in ind_pairs], default=None)

def _filter_companies(df: pd.DataFrame) -> pd.DataFrame:
    x = df
    if sel_markets:
        x = x[x["market"].isin(sel_markets)]
    if sel_inds:
        sel_codes = [code for lab, code, _ in ind_pairs if lab in sel_inds]
        x = x[x["ksic_mid_code"].isin(sel_codes)]
    return x

comp_filtered = _filter_companies(companies)
all_tickers = comp_filtered["ticker"].tolist()
sel_tickers = st.sidebar.multiselect(
    "종목(여러 개)", options=all_tickers, default=all_tickers[:5] if all_tickers else [],
    format_func=lambda t: fmt_company(t, name_map),
)

focus_ticker = st.sidebar.selectbox(
    "대표 종목(드릴다운)", options=sel_tickers if sel_tickers else all_tickers,
    index=0 if (sel_tickers or all_tickers) else None,
    format_func=lambda t: fmt_company(t, name_map) if t else "",
)

start_date = st.sidebar.date_input("시작일", default_start)
end_date   = st.sidebar.date_input("종료일", default_end)

st.sidebar.subheader("PD 표시 옵션")
pd_unit = st.sidebar.radio("단위", ["bp(베이시스포인트)", "%(퍼센트)"], index=0, horizontal=True)
smooth_n = st.sidebar.slider("스무딩(이동평균 일수)", 1, 30, 5, 1)
y_scale  = st.sidebar.selectbox("Y축 스케일", ["linear", "log"], index=0)

st.sidebar.subheader("인덱스 변동성")
rv_window = st.sidebar.slider("RV 윈도우(일)", 10, 60, 20, 2)

# -----------------------------
# 렌더: 기업/업종/외국인
# -----------------------------
def render_company_page():
    st.subheader("📈 기업 PD(EWMA) · 🏭 업종 평균 · 🌏 외국인 흐름")

    # 1) 기업 PD(EWMA)
    if focus_ticker:
        pdf = read_df(
            """
            SELECT p.date::date AS date, p.ticker,
                   p.pd_raw_avg_ewma,
                   c.company_name, c.market, c.ksic_mid_code, c.ksic_mid_name
            FROM pd_daily p
            JOIN companies c ON c.ticker = p.ticker
            WHERE p.date BETWEEN :start AND :end AND p.ticker = :t
            ORDER BY p.date
            """,
            {"start": start_date, "end": end_date, "t": focus_ticker},
        )
    else:
        pdf = pd.DataFrame()

    if pdf.empty or pd.to_numeric(pdf["pd_raw_avg_ewma"], errors="coerce").isna().all():
        st.info("선택한 기간에 EWMA PD 데이터가 없습니다.")
    else:
        plot_df = pd_scale(pdf, ["pd_raw_avg_ewma"], pd_unit, smooth_n).rename(columns={"pd_raw_avg_ewma": "PD(EWMA)"})
        ttl = f"{fmt_company(focus_ticker, name_map)} — PD(EWMA) ({'bp' if pd_unit.startswith('bp') else '%'})"
        fig = px.line(plot_df, x="date", y="PD(EWMA)", title=ttl)
        fig.update_yaxes(title=f"PD [{'bp' if pd_unit.startswith('bp') else '%'}]", type=y_scale)
        fig.update_layout(height=360, margin=dict(l=10, r=10, t=55, b=5))
        st.plotly_chart(fig, use_container_width=True, key="pg_cmp_pd")

    st.markdown("---")

    # 2) 업종 평균
    focus_ind_code = None; focus_ind_name = None
    if not pdf.empty and "ksic_mid_code" in pdf.columns:
        focus_ind_code = pdf["ksic_mid_code"].iloc[0]
        focus_ind_name = pdf["ksic_mid_name"].iloc[0]
    if not focus_ind_code:
        _row = companies.loc[companies["ticker"] == focus_ticker]
        if not _row.empty:
            focus_ind_code = _row["ksic_mid_code"].iloc[0]
            focus_ind_name = _row["ksic_mid_name"].iloc[0]
    if sel_inds:
        first_lab = sel_inds[0]
        for lab, code, nm in ind_pairs:
            if lab == first_lab:
                focus_ind_code, focus_ind_name = code, nm
                break

    if focus_ind_code:
        ind_df = read_df(
            """
            SELECT date::date AS date, ksic_mid_code, ksic_mid_name, industry_pd_avg
            FROM mv_industry_pd_daily
            WHERE ksic_mid_code = :k AND date BETWEEN :s AND :e
            ORDER BY date
            """,
            {"k": focus_ind_code, "s": start_date, "e": end_date},
        )
        if ind_df.empty:
            st.info("선택한 기간에 업종 평균 데이터가 없습니다. (물질화뷰 갱신 필요)")
        else:
            plot_ind = pd_scale(ind_df.rename(columns={"industry_pd_avg": "pd_avg"}), ["pd_avg"], pd_unit, smooth_n)
            fig_ind = px.line(plot_ind, x="date", y="pd_avg",
                              title=f"{focus_ind_name or focus_ind_code} — 업종 평균 PD ({'bp' if pd_unit.startswith('bp') else '%'})")
            fig_ind.update_yaxes(title=f"Industry PD [{'bp' if pd_unit.startswith('bp') else '%'}]", type=y_scale)
            fig_ind.update_layout(height=300, margin=dict(l=10, r=10, t=55, b=5))
            st.plotly_chart(fig_ind, use_container_width=True, key="pg_cmp_ind_pd")
    else:
        st.info("업종을 식별할 수 없습니다. (종목 또는 업종을 선택하세요)")

    st.markdown("---")

    # 3) 외국인 보유/플로우
    if focus_ticker:
        try:
            ff = read_df(
                """
                SELECT date::date AS date, ticker, company_name, market,
                       foreign_ratio, foreign_shares, delta_shares, delta_ratio, flow_label
                FROM v_foreign_flows_daily
                WHERE ticker = :t AND date BETWEEN :s AND :e
                ORDER BY date
                """,
                {"t": focus_ticker, "s": start_date, "e": end_date},
            )
        except Exception:
            base = read_df(
                """
                SELECT date::date AS date, ticker, foreign_ratio, foreign_shares
                FROM foreign_holdings_daily
                WHERE ticker = :t AND date BETWEEN :s AND :e
                ORDER BY date
                """,
                {"t": focus_ticker, "s": start_date, "e": end_date},
            )
            base["delta_shares"] = pd.to_numeric(base["foreign_shares"], errors="coerce").diff()
            base["delta_ratio"] = pd.to_numeric(base["foreign_ratio"], errors="coerce").diff()
            base["flow_label"] = np.where(base["delta_shares"] > 0, "매수",
                                    np.where(base["delta_shares"] < 0, "매도", "변화없음"))
            base["company_name"] = name_map.get(focus_ticker, focus_ticker)
            try:
                base["market"] = companies.set_index("ticker")["market"].to_dict().get(focus_ticker)
            except Exception:
                base["market"] = None
            ff = base

        if ff.empty:
            st.info("외국인 보유 데이터가 없습니다.")
        else:
            c1, c2 = st.columns(2)
            with c1:
                ff_plot = ff.copy()
                ff_plot["foreign_ratio_pct"] = series_to_pct(ff_plot["foreign_ratio"])
                fig_fr = px.line(
                    ff_plot,
                    x="date",
                    y="foreign_ratio_pct",
                    title=f"{fmt_company(focus_ticker, name_map)} — 외국인 보유비율(%)"
                )
                fig_fr.update_layout(height=300, margin=dict(l=10, r=10, t=55, b=5))
                fig_fr.update_yaxes(title="외국인 보유비율(%)")
                st.plotly_chart(fig_fr, use_container_width=True, key="pg_cmp_fore_ratio")
            with c2:
                bars = ff.dropna(subset=["delta_shares"]).copy()
                bars["flow"] = np.where(bars["delta_shares"] >= 0, "매수", "매도")
                fig_fl = px.bar(
                    bars, x="date", y="delta_shares", color="flow",
                    title="전일 대비 순매수/순매도(주)",
                    color_discrete_map={"매수": "#1f77b4", "매도": "#ff7f0e"},
                    category_orders={"flow": ["매수", "매도"]},
                )
                fig_fl.add_hline(y=0, line_dash="dot", opacity=0.4)
                fig_fl.update_layout(height=300, margin=dict(l=10, r=10, t=55, b=5), legend_title=None)
                st.plotly_chart(fig_fl, use_container_width=True, key="pg_cmp_fore_flow")
    else:
        st.info("종목을 선택하세요.")

# -----------------------------
# 렌더: 인덱스
# -----------------------------
def render_index_page():
    st.subheader("📈 코스피/코스닥 — 수익률 & 실현변동성(RV)")
    idx = read_df(
        """
        SELECT date::date AS date, index_name, close, return
        FROM market_index_daily
        WHERE date BETWEEN :s AND :e
        ORDER BY index_name, date
        """,
        {"s": start_date, "e": end_date},
    )
    if idx.empty:
        st.info("인덱스 데이터가 없습니다.")
        return

    fig_ret = px.line(
        idx, x="date", y="return", color="index_name", title="일간 수익률",
        color_discrete_map={"KOSPI": "#1f77b4", "KOSDAQ": "#ff7f0e"},
        category_orders={"index_name": ["KOSPI", "KOSDAQ"]},
    )
    fig_ret.update_layout(height=280, margin=dict(l=10, r=10, t=55, b=5), legend_title=None)
    st.plotly_chart(fig_ret, use_container_width=True, key="pg_idx_ret")

    rv = idx.sort_values(["index_name", "date"]).copy()
    rv["rv"] = rv.groupby("index_name")["return"].rolling(rv_window, min_periods=rv_window//2).std(ddof=0).reset_index(level=0, drop=True)
    rv = rv.dropna(subset=["rv"])
    if rv.empty:
        st.info("변동성 계산을 위한 데이터가 부족합니다.")
    else:
        fig_rv = px.line(
            rv, x="date", y="rv", color="index_name", title=f"{rv_window}일 실현변동성(RV)",
            color_discrete_map={"KOSPI": "#1f77b4", "KOSDAQ": "#ff7f0e"},
            category_orders={"index_name": ["KOSPI", "KOSDAQ"]},
        )
        fig_rv.update_layout(height=280, margin=dict(l=10, r=10, t=55, b=5), legend_title=None)
        st.plotly_chart(fig_rv, use_container_width=True, key="pg_idx_rv")

# -----------------------------
# 렌더: FinBERT
# -----------------------------
def render_finbert_page():
    st.subheader("📰 FinBERT — 경기심리 게이지 & 기사수/워드클라우드")
    fb = read_df(
        """
        SELECT date::date AS date,
               finbert_net_sentiment,
               finbert_expected_value,
               finbert_pos_ratio
        FROM finbert_index_daily
        WHERE date BETWEEN :s AND :e
        ORDER BY date
        """,
        {"s": start_date, "e": end_date},
    )
    if fb.empty or fb["finbert_expected_value"].dropna().empty:
        st.info("해당 기간의 FinBERT 인덱스 데이터가 없습니다.")
        return

    fbev = fb.dropna(subset=["finbert_expected_value"])
    last = fbev.iloc[-1]
    cur  = float(last["finbert_expected_value"])
    prev = float(fbev.iloc[-2]["finbert_expected_value"]) if len(fbev) >= 2 else None

    neutral_eps = 0.05
    if   cur >=  neutral_eps: label = "긍정"
    elif cur <= -neutral_eps: label = "부정"
    else:                     label = "중립"

    display_date = None
    sel_dict = None

    col_g, col_tbl = st.columns([1.2, 1], gap="medium")

    with col_g:
        fig_g = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=cur,
            delta=(dict(reference=prev, valueformat=".3f",
                        increasing={"color": "#2ecc71"}, decreasing={"color": "#e74c3c"}) if prev is not None else None),
            title={"text": f"FinBERT 기대값 — {label}"},
            gauge=dict(
                axis=dict(range=[-1, 1], tickvals=[-1,-0.5,0,0.5,1]),
                bar=dict(thickness=0.25),
                steps=[
                    {"range": [-1.0, -neutral_eps], "color": "#f5b7b1"},
                    {"range": [-neutral_eps, neutral_eps], "color": "#d5d8dc"},
                    {"range": [neutral_eps, 1.0], "color": "#abebc6"},
                ],
                threshold=dict(line=dict(color="black", width=3), thickness=0.8, value=cur),
            ),
            number=dict(suffix="  (−1 ~ +1)", valueformat=".3f"),
        ))
        fig_g.update_layout(height=270, margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig_g, use_container_width=True, key=f"pg_fb_gauge_{pd.to_datetime(last['date']).date()}_{start_date}_{end_date}")

    with col_tbl:
        counts = read_df(
            """
            SELECT date::date AS date, n_pos, n_neu, n_neg
            FROM v_finbert_daily_counts
            WHERE date BETWEEN :s AND :e
            ORDER BY date
            """,
            {"s": start_date, "e": end_date},
        )
        if counts.empty:
            st.info("해당 기간의 일별 기사수 데이터가 없습니다.")
        else:
            counts["date"] = pd.to_datetime(counts["date"]).dt.date
            display_date = pd.to_datetime(last["date"]).date()
            row = counts.loc[counts["date"] == display_date]
            if row.empty:
                row = counts.iloc[[-1]]
                display_date = row["date"].iloc[0]

            n_pos = int(row["n_pos"].iloc[0]); n_neu = int(row["n_neu"].iloc[0]); n_neg = int(row["n_neg"].iloc[0])
            tbl = pd.DataFrame([
                {"감성": "긍정", "기사수": n_pos, "sent": "positive"},
                {"감성": "중립", "기사수": n_neu, "sent": "neutral"},
                {"감성": "부정", "기사수": n_neg, "sent": "negative"},
            ])

            st.markdown(f"**📅 {display_date} 기사수** (행 클릭 → 워드클라우드)")

            # 행 스타일: 감성별 배경
            row_style = JsCode(
                """
                function(params) {
                  if (params.data.감성 === '긍정') { return { backgroundColor: '#183a2a', fontWeight: 700 }; }
                  else if (params.data.감성 === '부정') { return { backgroundColor: '#3a1a1a', fontWeight: 700 }; }
                  else { return { backgroundColor: '#1c2230', fontWeight: 700 }; }
                }
                """
            )

            gb = GridOptionsBuilder.from_dataframe(tbl[["감성","기사수"]])
            gb.configure_default_column(resizable=True, filter=False, sortable=False)
            gb.configure_selection(selection_mode='single', use_checkbox=False)
            gb.configure_grid_options(
                rowHeight=42, headerHeight=38,
                suppressMovableColumns=True, getRowStyle=row_style
            )
            gb.configure_column("감성", width=120, cellStyle={'fontWeight': '700'})
            gb.configure_column("기사수", width=140, type=["numericColumn"],
                                cellStyle={'fontWeight': '800', 'textAlign': 'right'})

            grid = AgGrid(
                tbl[["감성","기사수"]],
                gridOptions=gb.build(),
                update_mode=GridUpdateMode.SELECTION_CHANGED,
                fit_columns_on_grid_load=True,     # ← 자동 폭 맞춤
                allow_unsafe_jscode=True,
                height=170,
                theme="balham-dark",               # ← 다크 테마
                key=f"pg_fb_counts_{display_date}",
            )

            selected_row = grid.get("selected_rows", None)
            if isinstance(selected_row, list) and len(selected_row) > 0:
                sel_dict = selected_row[0]
            elif isinstance(selected_row, pd.DataFrame) and not selected_row.empty:
                sel_dict = selected_row.iloc[0].to_dict()

    if display_date and (sel_dict is not None):
        sel_label = sel_dict.get("감성")
        label2sent = {"긍정": "positive", "중립": "neutral", "부정": "negative"}
        sel_sent = label2sent.get(sel_label)
        st.caption(f"**{display_date} · {sel_label}** 기사 워드클라우드（제목에서 키워드 추출）")
        news = read_df(
            """
            SELECT title FROM finbert_news_titles
            WHERE date = :d AND predicted_sentiment = :s
            ORDER BY 1
            """,
            {"d": display_date, "s": sel_sent},
        )
        if news.empty:
            st.info(f"{display_date} — {sel_label} 기사 없음")
        else:
            render_wordcloud(news["title"].tolist(), sel_sent)
    else:
        st.caption("표에서 **행을 클릭**하면 워드클라우드가 표시됩니다.")

# -----------------------------
# (선택) Gemini 연동 — 요약 + 채팅
# -----------------------------
if BaseModel is not None:
    class Opinion(BaseModel):
        ticker: str
        name: str
        market: str
        period: str
        stance: str
        confidence: float
        summary: str
        reasons: list[str]
        risks: list[str]
        watch_items: list[str] | None = []
        next_actions: list[str] | None = []
else:
    Opinion = None

@st.cache_resource
def get_gemini_client():
    if genai is None:
        return None
    key = (getattr(st, "secrets", {}).get("GEMINI_API_KEY") if hasattr(st, "secrets") else None) or os.getenv("GEMINI_API_KEY")
    if not key:
        return None
    try:
        return genai.Client(api_key=key)
    except Exception:
        return None

GEMINI_MODEL = (getattr(st, "secrets", {}).get("GEMINI_MODEL") if hasattr(st, "secrets") else None) or "gemini-2.5-flash"

SYS_PROMPT = (
    "너는 한국 주식시장의 데이터 기반 리서치 어시스턴트다. "
    "주어진 수치(PD[bp,%], 업종 평균, 외국인 매수/매도 흐름, 인덱스 변동성, FinBERT 기대값)만 근거로 "
    "관찰 중심 의견을 제시해라. 개인 맞춤형 조언은 금지. "
    "foreign_ratio는 0~1 비율 → 퍼센트로 말할 땐 100을 곱해 '%'를 붙인다. bp로 표현 금지."
)

CHAT_SYS_PROMPT = (
    "너는 금융 교육용 어시스턴트다. 공개 지식/일반 상식(+검색 사용 시 외부자료)을 바탕으로 답하라. "
    "DB 수치는 참고일 뿐. foreign_ratio는 0~1 비율(0.55=55%). 퍼센트 표기만, bp 금지. "
    "쉽고 간결하게 답하라."
)

def _load_context_for_llm(ticker: str, s: date, e: date) -> dict:
    meta = read_df("SELECT ticker, company_name, market, ksic_mid_code, ksic_mid_name FROM companies WHERE ticker = :t", {"t": ticker})
    if meta.empty:
        return {}
    m = meta.iloc[0]
    pddf = read_df("""
        SELECT date::date AS date, pd_raw_avg_ewma
        FROM pd_daily
        WHERE ticker = :t AND date BETWEEN :s AND :e
        ORDER BY date
    """, {"t": ticker, "s": s, "e": e})
    pd_ctx = {}
    if not pddf.empty:
        srs = pd.to_numeric(pddf["pd_raw_avg_ewma"], errors="coerce").dropna()
        if not srs.empty:
            last = float(srs.iloc[-1]); base_idx = max(0, len(srs) - 31); base = float(srs.iloc[base_idx])
            pd_ctx = {"last_pd_ewma_bp": round(last * 10_000, 2), "delta_30d_bp": round((last - base) * 10_000, 2)}
    ind_ctx = {}
    if pd.notna(m["ksic_mid_code"]):
        ind = read_df("""
            SELECT date::date AS date, industry_pd_avg
            FROM mv_industry_pd_daily
            WHERE ksic_mid_code = :k AND date BETWEEN :s AND :e
            ORDER BY date
        """, {"k": m["ksic_mid_code"], "s": s, "e": e})
        srs = pd.to_numeric(ind.get("industry_pd_avg", pd.Series(dtype=float)), errors="coerce").dropna()
        if not srs.empty:
            ind_ctx = {"industry_last_bp": round(float(srs.iloc[-1]) * 10_000, 2)}
    flows = read_df("""
        SELECT date::date AS date, foreign_ratio::float AS fr, foreign_shares::float AS fs
        FROM foreign_holdings_daily
        WHERE ticker = :t AND date BETWEEN :s AND :e
        ORDER BY date
    """, {"t": ticker, "s": s, "e": e})
    flow_ctx = {}
    if not flows.empty:
        flows["delta_shares"] = pd.to_numeric(flows["fs"], errors="coerce").diff()
        fr_last_series = pd.to_numeric(flows["fr"], errors="coerce").dropna()
        fr_last_pct = normalize_ratio_to_pct(fr_last_series.iloc[-1]) if not fr_last_series.empty else None
        flow_ctx = {
            "foreign_ratio_last_pct": fr_last_pct,
            "net_flow_5d_shares": int(pd.to_numeric(flows["delta_shares"], errors="coerce").dropna().tail(5).sum())
                if pd.to_numeric(flows["delta_shares"], errors="coerce").dropna().size else None,
        }
    idx = read_df("""
        SELECT date::date AS date, index_name, return::float AS ret
        FROM market_index_daily
        WHERE date BETWEEN :s AND :e
        ORDER BY index_name, date
    """, {"s": s, "e": e})
    idx_ctx = {}
    if not idx.empty:
        rv = idx.sort_values(["index_name", "date"]).copy()
        rv["rv20"] = rv.groupby("index_name")["ret"].rolling(20, min_periods=10).std(ddof=0).reset_index(level=0, drop=True)
        sub = rv.dropna(subset=["rv20"])
        if not sub.empty:
            kospi  = sub.loc[sub["index_name"] == "KOSPI", "rv20"].mean()
            kosdaq = sub.loc[sub["index_name"] == "KOSDAQ", "rv20"].mean()
            idx_ctx = {"rv20_kospi": round(float(kospi), 4) if pd.notna(kospi) else None,
                       "rv20_kosdaq": round(float(kosdaq), 4) if pd.notna(kosdaq) else None}
    fb = read_df("""
        SELECT finbert_expected_value::float AS ev
        FROM finbert_index_daily
        WHERE date BETWEEN :s AND :e
        ORDER BY date
    """, {"s": s, "e": e})
    fb_ctx = {"finbert_ev_last": round(float(fb["ev"].dropna().iloc[-1]), 3)} if (not fb.empty and fb["ev"].dropna().size) else {}
    return {
        "ticker": m["ticker"], "name": m["company_name"], "market": m["market"],
        "industry_code": m["ksic_mid_code"], "industry_name": m["ksic_mid_name"],
        "period": f"{s} ~ {e}", "pd": pd_ctx, "industry_pd": ind_ctx,
        "foreign": flow_ctx, "index": idx_ctx, "finbert": fb_ctx,
    }

def _format_ctx_for_chat(ctx: dict) -> str:
    pd_last = ctx.get("pd", {}).get("last_pd_ewma_bp")
    pd_delta = ctx.get("pd", {}).get("delta_30d_bp")
    ind_last = ctx.get("industry_pd", {}).get("industry_last_bp")
    fr_pct = ctx.get("foreign", {}).get("foreign_ratio_last_pct")
    nf5 = ctx.get("foreign", {}).get("net_flow_5d_shares")
    rv_k = ctx.get("index", {}).get("rv20_kospi")
    rv_q = ctx.get("index", {}).get("rv20_kosdaq")
    fbev = ctx.get("finbert", {}).get("finbert_ev_last")
    return (
        f"- PD(EWMA) 최근: {pd_last} bp, 30일 변화: {pd_delta} bp\n"
        f"- 업종 평균 PD 최근: {ind_last} bp\n"
        f"- 외국인 보유비율: {fr_pct}% (최근), 5영업일 순매수: {nf5} 주\n"
        f"- 지수 RV20: KOSPI={rv_k}, KOSDAQ={rv_q}\n"
        f"- FinBERT 기대값: {fbev} (-1~+1)\n"
    )

def render_llm_page():
    st.subheader("🤖 Gemini — 투자 종합의견 & 질의응답")

    if genai is None or gtypes is None or Opinion is None:
        st.warning("google-genai / pydantic 패키지가 필요합니다. 설치 후 새로고침하세요.")
        st.code("pip install -U google-genai pydantic", language="bash")
        return

    if "gemini_opinion" not in st.session_state: st.session_state["gemini_opinion"] = None
    if "gemini_ctx_sig" not in st.session_state: st.session_state["gemini_ctx_sig"] = None
    if "chat_messages" not in st.session_state: st.session_state["chat_messages"] = []

    ctx_sig = (str(focus_ticker or ""), str(start_date), str(end_date))

    st.markdown("### 📌 요약(내부 데이터 기반)")
    c_left, c_right = st.columns([1, 2])
    with c_left:
        temp = st.slider("창의성(temperature)", 0.0, 1.0, 0.3, 0.1, key="gem_temp")
        run  = st.button("🔎 요약 생성", use_container_width=True, key="gem_run")
        clear = st.button("🧹 요약 지우기", use_container_width=True, key="gem_clear")

    if clear:
        st.session_state["gemini_opinion"] = None
        st.session_state["gemini_ctx_sig"] = None
        try: st.rerun()
        except Exception: pass

    if run:
        cli = get_gemini_client()
        if cli is None:
            st.warning("GEMINI_API_KEY가 설정되지 않았거나 클라이언트 생성 실패.")
            return
        if not focus_ticker:
            st.info("종목을 먼저 선택하세요."); return

        ctx = _load_context_for_llm(focus_ticker, start_date, end_date)
        if not ctx:
            st.info("컨텍스트가 비었습니다. 기간/데이터를 확인하세요."); return

        user_prompt = (
            f"[종목] {ctx['name']} ({ctx['ticker']}, {ctx['market']})\n"
            f"[기간] {ctx['period']}\n"
            f"[업종] {ctx.get('industry_name')} (코드 {ctx.get('industry_code')})\n"
            f"{_format_ctx_for_chat(ctx)}\n"
            "위 내부 데이터만 근거로 '투자 종합의견'을 JSON으로 작성해라. "
            "추정이나 외부지식/뉴스를 인용하지 말라. "
            "foreign_ratio는 퍼센트(%)로만 표기하고 절대 bp로 표기하지 말라."
        )

        with st.spinner("Gemini가 요약을 생성 중…"):
            try:
                resp = cli.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=f"{SYS_PROMPT}\n\n{user_prompt}",
                    config=gtypes.GenerateContentConfig(
                        temperature=float(temp),
                        response_mime_type="application/json",
                        response_schema=Opinion,
                        thinking_config=gtypes.ThinkingConfig(thinking_budget=0),
                        max_output_tokens=800,
                    ),
                )
                op = resp.parsed
                op_dict = op.model_dump() if hasattr(op, "model_dump") else (op.dict() if hasattr(op, "dict") else None)
                st.session_state["gemini_opinion"] = op_dict
                st.session_state["gemini_ctx_sig"] = ctx_sig
            except Exception as e:
                st.error(f"Gemini 요약 호출 오류: {e}")

    with c_right:
        op_data = st.session_state.get("gemini_opinion")
        created_sig = st.session_state.get("gemini_ctx_sig")
        if op_data and created_sig != ctx_sig:
            st.info("기간/종목이 변경되었습니다. 아래 결과는 이전 컨텍스트 기준입니다. [요약 생성]을 다시 눌러 갱신하세요.")
        if not op_data:
            st.caption("아직 요약을 생성하지 않았습니다. [요약 생성] 버튼을 눌러 만들어보세요.")
        else:
            stance = op_data.get("stance", "중립"); conf = op_data.get("confidence", 0.0)
            stance_color = {"매수": "green", "중립": "gray", "매도": "red"}.get(stance, "gray")
            st.markdown(f"### 🎯 종합의견: <span style='color:{stance_color}'>{stance}</span> (신뢰도 {conf:.0%})", unsafe_allow_html=True)
            st.markdown(f"**요약**: {op_data.get('summary','')}")
            cA, cB = st.columns(2)
            with cA:
                st.markdown("#### ✅ 근거")
                for r in (op_data.get("reasons") or []): st.markdown(f"- {r}")
            with cB:
                st.markdown("#### ⚠️ 리스크")
                for r in (op_data.get("risks") or []): st.markdown(f"- {r}")
            st.caption("※ 본 내용은 정보 제공 목적이며 투자 권유가 아닙니다.")

    st.divider()

    # ---- 채팅
    st.markdown("### 💬 챗봇")
    chat_cols = st.columns([1, 2.2, 1])
    with chat_cols[0]:
        temp_chat = st.slider("창의성", 0.0, 1.0, 0.3, 0.1, key="chat_temp")
        use_search = st.toggle("🔍 웹/검색 사용", value=True, help="켜면 Google 검색을 활용해 더 광범위한 정보로 답합니다.")
    with chat_cols[2]:
        if st.button("🧹 채팅 초기화", use_container_width=True):
            st.session_state["chat_messages"] = []
            try: st.rerun()
            except Exception: pass

    st.caption("예시 질문:")
    ex1, ex2, ex3, ex4 = st.columns(4)
    if ex1.button("외국인 순매수/순매도가 지수에 주는 영향은?"):
        st.session_state["chat_messages"].append({"role": "user", "content": "외국인 순매수·순매도가 한국 주식시장(특히 KOSPI)에 어떤 영향을 주나요? 역사적 패턴과 예외도 알려줘."})
    if ex2.button("삼성전자 기업 개요와 경쟁우위는?"):
        st.session_state["chat_messages"].append({"role": "user", "content": "삼성전자의 주요 사업부, 경쟁우위, 리스크를 최근 동향까지 포함해 간단히 설명해줘."})
    if ex3.button("메모리 사이클과 주가의 관계는?"):
        st.session_state["chat_messages"].append({"role": "user", "content": "메모리(DDR/HBM) 업황 사이클이 반도체 대형주 주가와 어떻게 연결되는지 개념적으로 설명해줘."})
    if ex4.button("KOSDAQ과 KOSPI의 차이?"):
        st.session_state["chat_messages"].append({"role": "user", "content": "KOSDAQ과 KOSPI의 차이(상장 요건, 업종 구성, 변동성)를 초보자도 이해하기 쉽게 알려줘."})

    for m in st.session_state["chat_messages"]:
        st.chat_message(m["role"]).markdown(m["content"])

    user_msg = st.chat_input("질문을 입력하세요…")
    if user_msg:
        st.session_state["chat_messages"].append({"role": "user", "content": user_msg})

    if len(st.session_state["chat_messages"]) and st.session_state["chat_messages"][-1]["role"] == "user":
        last_user = st.session_state["chat_messages"][-1]["content"]
        cli = get_gemini_client()
        if cli is None:
            st.warning("GEMINI_API_KEY가 설정되지 않았거나 클라이언트 생성 실패.")
            return
        ctx = _load_context_for_llm(focus_ticker, start_date, end_date) if focus_ticker else None
        tools = [gtypes.Tool(google_search=gtypes.GoogleSearch())] if use_search else None
        ctx_line = ""
        if ctx:
            ctx_line = (
                f"(참고) 우리 DB: {ctx['name']} {ctx['ticker']} "
                f"PD(EWMA) 최근 {ctx['pd'].get('last_pd_ewma_bp')}bp, "
                f"외국인 보유비율 {ctx['foreign'].get('foreign_ratio_last_pct')}%, "
                f"FinBERT EV {ctx['finbert'].get('finbert_ev_last')}."
            )
        full_prompt = (
            f"{CHAT_SYS_PROMPT}\n\n{ctx_line}\n\n사용자 질문: {last_user}\n\n"
            "응답 규칙:\n"
            "- 개념/기업/매크로 질문은 공개 지식 위주로 설명하고 필요한 경우 우리 DB는 참고 수치로만 인용\n"
            "- 수치 언급 시 단위 명시(PD=bp, 외국인 보유비율=%), foreign_ratio는 절대 bp로 말하지 말 것\n"
            "- 초보자도 이해하기 쉽게 간결하게 답변\n"
        )
        with st.spinner("답변 생성 중…"):
            try:
                resp = cli.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=full_prompt,
                    config=gtypes.GenerateContentConfig(
                        temperature=float(st.session_state.get("chat_temp", 0.3)),
                        tools=tools,
                        max_output_tokens=1024,
                    ),
                )
                text = getattr(resp, "text", None) or "죄송해요. 지금은 답변을 생성하지 못했어요."
                st.session_state["chat_messages"].append({"role": "assistant", "content": text})
                st.chat_message("assistant").markdown(text)
            except Exception as e:
                err = f"오류: {e}"
                st.session_state["chat_messages"].append({"role": "assistant", "content": err})
                st.chat_message("assistant").markdown(err)

    st.caption("※ 본 채팅은 정보 제공 목적이며 투자 권유가 아닙니다.")

# -----------------------------
# 라우팅
# -----------------------------
st.markdown("---")
if st.session_state["page"] == "기업/업종/외국인":
    render_company_page()
elif st.session_state["page"] == "인덱스":
    render_index_page()
elif st.session_state["page"] == "FinBERT":
    render_finbert_page()
elif st.session_state["page"] == "LLM 종합의견":
    render_llm_page()

# 공통 하단 안내
st.markdown("---")
st.caption("※ 단위: PD는 사이드바에서 %(퍼센트) 또는 bp(베이시스포인트)로 전환 가능 · 스무딩은 이동평균 적용")
st.caption("※ 업종 평균은 ksic_mid_code로 집계하고 화면에는 ksic_mid_name을 표시합니다.")
st.caption("※ 외국인 순매수/순매도는 전일 대비 foreign_shares 차이로 계산합니다. (보유비율 추이는 정규화된 %로 표시)")
