# -*- coding: utf-8 -*-
"""
바뀐 스키마 + 인덱스(kp,kd_index.csv) 적재 (업서트)
- companies / pd_daily / foreign_holdings_daily / market_index_daily
- 마지막에 mv_industry_pd_daily REFRESH
- 입력된 CSV에 따라 자동 실행 (mode 불필요)

예시:
python load_data.py --db postgresql+psycopg://riskuser:riskpass@127.0.0.1:5432/riskdb \
    --kp /home/ubuntu/riskdash/kp.csv \
    --kd /home/ubuntu/riskdash/kd.csv \
    --index /home/ubuntu/riskdash/kd_index.csv
"""
import argparse
import io
import re
import unicodedata
import pandas as pd
from sqlalchemy import create_engine, text

# ================================
# 유틸: 정규화/타입 변환
# ================================
def zfill_ticker(x, width: int = 6):
    """숫자형/문자형 혼재 종목코드를 0패딩 6자리로 통일"""
    try:
        if pd.isna(x):
            return None
        return str(int(str(x).strip())).zfill(width)
    except Exception:
        s = str(x).strip()
        return s.zfill(width) if s.isdigit() else (s if s else None)

_WEIRD_WS = re.compile(r"[\u00A0\u200B\u200C\u200D\uFEFF]")

def canon_header(s: str) -> str:
    """헤더를 소문자+언더스코어로 정규화 (한글 보존)"""
    if s is None:
        return ""
    t = unicodedata.normalize("NFKC", str(s))
    t = _WEIRD_WS.sub("", t).strip().lower()
    # 영숫자/한글/괄호/공백만 남기고 구분자는 언더스코어
    t = re.sub(r"[^\w\s\(\)가-힣]", " ", t).replace("-", " ").replace("/", " ")
    t = re.sub(r"\s+", "_", t)
    t = re.sub(r"_+", "_", t).strip("_")
    return t

def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    """데이터프레임 컬럼명을 canon_header 규칙으로 정규화"""
    d = df.copy()
    d.columns = [canon_header(c) for c in d.columns]
    return d

def to_date(s):   return pd.to_datetime(s, errors="coerce").dt.date
def to_f01(s):    return pd.to_numeric(s, errors="coerce").clip(0, 1)
def to_ratio(s):  return pd.to_numeric(s, errors="coerce").clip(0, 100)   # 0~100 범위 비율(%) 가정
def to_int64(s):  return pd.to_numeric(s, errors="coerce").round().astype("Int64")

# ================================
# 빌드: 테이블별 가공
# ================================
def build_companies(df: pd.DataFrame, market_label: str) -> pd.DataFrame:
    d = df.copy()
    code_col = next((c for c in d.columns if "종목코드" in c), None)
    name_col = next((c for c in d.columns if c in ("회사명", "종목명", "한글종목명", "기업명", "company_name")), None)
    # KSIC 컬럼 후보 (원본마다 다를 수 있어 prefix 기반 매칭)
    ksic_code = next((c for c in d.columns if c.startswith("통계청_한국표준산업분류_코드_11차")), None)
    ksic_name = next((c for c in d.columns if c.startswith("통계청_한국표준산업분류_11차")), None)

    out = pd.DataFrame()
    out["ticker"] = d[code_col].apply(zfill_ticker) if code_col else None
    out["company_name"] = d[name_col] if name_col else None
    out["company_name"] = (
        out["company_name"]
        .astype(str)
        .str.strip()
        .replace({"": None, "nan": None, "None": None})
    )
    out.loc[out["company_name"].isna(), "company_name"] = out["ticker"]
    out["market"] = market_label
    out["ksic_mid_code"] = d[ksic_code].astype(str).str.strip() if ksic_code else None
    out["ksic_mid_name"] = d[ksic_name].astype(str).str.strip() if ksic_name else None
    out = out.dropna(subset=["ticker"]).drop_duplicates(subset=["ticker"])
    return out[["ticker", "company_name", "market", "ksic_mid_code", "ksic_mid_name"]]

def build_pd(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    date_col = next((c for c in d.columns if c == "날짜"), None)
    code_col = next((c for c in d.columns if "종목코드" in c), None)
    out = pd.DataFrame()
    out["date"]   = to_date(d[date_col]) if date_col else None
    out["ticker"] = d[code_col].apply(zfill_ticker) if code_col else None
    # 원본 헤더 다양성 대응
    col_raw   = next((c for c in d.columns if c in ("pd_raw_avg", "pd_avg", "raw_avg")), None)
    col_ewma  = next((c for c in d.columns if c in ("pd_raw_avg_ewma", "pd_ewma_avg", "ewma_pd_avg")), None)
    col_kal   = next((c for c in d.columns if c in ("pd_raw_avg_kalman", "pd_kalman_avg", "kalman_pd_avg")), None)
    out["pd_raw_avg"]        = to_f01(d[col_raw])  if col_raw else None
    out["pd_raw_avg_ewma"]   = to_f01(d[col_ewma]) if col_ewma else None
    out["pd_raw_avg_kalman"] = to_f01(d[col_kal])  if col_kal else None
    out = out.dropna(subset=["date", "ticker"]).sort_values(["date", "ticker"])
    return out[["date", "ticker", "pd_raw_avg", "pd_raw_avg_ewma", "pd_raw_avg_kalman"]]

def build_foreign(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    date_col = next((c for c in d.columns if c == "날짜"), None)
    code_col = next((c for c in d.columns if "종목코드" in c), None)
    out = pd.DataFrame()
    out["date"]   = to_date(d[date_col]) if date_col else None
    out["ticker"] = d[code_col].apply(zfill_ticker) if code_col else None
    # CSV에서는 보통 % 단위가 들어오므로 0~100 범위로 클리핑만 수행
    out["foreign_ratio"]  = to_ratio(d.get("foreign_ratio", None))
    out["foreign_shares"] = to_int64(d.get("foreign_shares", None))
    out = out.dropna(subset=["date", "ticker"]).sort_values(["date", "ticker"])
    return out[["date", "ticker", "foreign_ratio", "foreign_shares"]]

def build_index(idx: pd.DataFrame) -> pd.DataFrame:
    d = idx.copy()
    d["date"] = to_date(d["날짜"])
    # 정규화 후 보통 'kospi_종가', 'kosdaq_종가' 형태
    kospi_col  = next((c for c in d.columns if c == "kospi_종가" or c.endswith("kospi_종가")), "kospi_종가")
    kosdaq_col = next((c for c in d.columns if c == "kosdaq_종가" or c.endswith("kosdaq_종가")), "kosdaq_종가")
    long = pd.melt(d, id_vars=["date"], value_vars=[kospi_col, kosdaq_col],
                   var_name="name", value_name="close")
    long["index_name"] = long["name"].str.replace("_종가", "", regex=False).str.upper()
    long = long.drop(columns=["name"]).sort_values(["index_name", "date"])
    # 일간 수익률
    long["return"] = long.groupby("index_name")["close"].pct_change().fillna(0.0)
    return long[["date", "index_name", "close", "return"]]

# ================================
# 업서트: 대용량 안전 COPY + ON CONFLICT
# ================================
def upsert(engine, df: pd.DataFrame, table: str, pk_cols):
    """psycopg3 COPY를 이용해 임시테이블로 적재 후 ON CONFLICT 업서트"""
    if df is None or df.empty:
        print(f"[SKIP] empty: {table}")
        return
    cols = list(df.columns)
    col_list = ", ".join(f'"{c}"' for c in cols)
    conflict = ", ".join(f'"{c}"' for c in pk_cols)
    updates  = ", ".join(f'"{c}" = EXCLUDED."{c}"' for c in cols if c not in pk_cols)

    with engine.begin() as conn:
        # 임시테이블 생성
        conn.execute(text(f'CREATE TEMP TABLE tmp_{table} AS SELECT * FROM {table} WITH NO DATA;'))

        # COPY로 빠르게 적재
        raw = conn.connection  # psycopg3 raw connection
        buf = io.StringIO()
        df.to_csv(buf, index=False)
        buf.seek(0)
        with raw.cursor() as cur:
            with cur.copy(f'COPY tmp_{table} ({col_list}) FROM STDIN WITH (FORMAT CSV, HEADER TRUE)') as cp:
                cp.write(buf.getvalue())

        # ON CONFLICT 업서트
        sql = f"""
        INSERT INTO {table} ({col_list})
        SELECT {col_list} FROM tmp_{table}
        ON CONFLICT ({conflict}) DO UPDATE SET {updates};
        """
        conn.execute(text(sql))
        conn.execute(text(f"DROP TABLE IF EXISTS tmp_{table};"))
    print(f"[UPSERT] {table}: {len(df):,} rows")

# ================================
# main
# ================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True, help="Postgres 연결 URL (예: postgresql+psycopg://user:pass@host:5432/db)")
    ap.add_argument("--kp", help="KOSPI CSV 경로")
    ap.add_argument("--kd", help="KOSDAQ CSV 경로")
    ap.add_argument("--index", help="코스피/코스닥 인덱스 CSV 경로")
    args = ap.parse_args()

    eng = create_engine(args.db, pool_pre_ping=True, future=True)

    # KOSPI
    if args.kp:
        kp = normalize_cols(pd.read_csv(args.kp, encoding="utf-8-sig"))
        comp = build_companies(kp, "KOSPI")
        upsert(eng, comp, "companies", pk_cols=["ticker"])
        pdd = build_pd(kp)
        upsert(eng, pdd, "pd_daily", pk_cols=["date", "ticker"])
        fh  = build_foreign(kp)
        upsert(eng, fh, "foreign_holdings_daily", pk_cols=["date", "ticker"])

    # KOSDAQ
    if args.kd:
        kd = normalize_cols(pd.read_csv(args.kd, encoding="utf-8-sig"))
        comp = build_companies(kd, "KOSDAQ")
        upsert(eng, comp, "companies", pk_cols=["ticker"])
        pdd = build_pd(kd)
        upsert(eng, pdd, "pd_daily", pk_cols=["date", "ticker"])
        fh  = build_foreign(kd)
        upsert(eng, fh, "foreign_holdings_daily", pk_cols=["date", "ticker"])

    # Index
    if args.index:
        idx = normalize_cols(pd.read_csv(args.index, encoding="utf-8-sig"))
        midx = build_index(idx)
        upsert(eng, midx, "market_index_daily", pk_cols=["date", "index_name"])

    # 업종 평균 뷰 리프레시
    try:
        with eng.begin() as conn:
            conn.execute(text("REFRESH MATERIALIZED VIEW mv_industry_pd_daily;"))
        print("[REFRESH] mv_industry_pd_daily")
    except Exception as e:
        print("[WARN] mv_industry_pd_daily 리프레시 실패:", e)

    print("[DONE] 적재 완료")

if __name__ == "__main__":
    main()