# -*- coding: utf-8 -*-
"""
바뀐 스키마 + 인덱스(kp,kd_index.csv)까지 적재 (업서트)
- companies / pd_daily / foreign_holdings_daily / market_index_daily
- 마지막에 mv_industry_pd_daily REFRESH
"""
import argparse, io, re, unicodedata
import pandas as pd
from sqlalchemy import create_engine, text

# ===== 유틸 =====
def zfill_ticker(x, width=6):
    try:
        if pd.isna(x):
            return None
        return str(int(str(x).strip())).zfill(width)
    except Exception:
        s = str(x).strip()
        return s.zfill(width) if s.isdigit() else (s if s else None)

_WEIRD_WS = re.compile(r"[\u00A0\u200B\u200C\u200D\uFEFF]")
def canon_header(s: str) -> str:
    if s is None: return ""
    t = unicodedata.normalize("NFKC", str(s))
    t = _WEIRD_WS.sub("", t).strip().lower()
    t = re.sub(r"[^\w\s\(\)가-힣]", " ", t).replace("-", " ").replace("/", " ")
    t = re.sub(r"\s+", "_", t)
    t = re.sub(r"_+", "_", t).strip("_")
    return t
def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy(); d.columns = [canon_header(c) for c in d.columns]; return d

def to_date(s):   return pd.to_datetime(s, errors="coerce").dt.date
def to_f01(s):    return pd.to_numeric(s, errors="coerce").clip(0, 1)
def to_ratio(s):  return pd.to_numeric(s, errors="coerce").clip(0, 100)
def to_int64(s):  return pd.to_numeric(s, errors="coerce").round().astype("Int64")

# ===== 빌드 =====
def build_companies(df: pd.DataFrame, market_label: str) -> pd.DataFrame:
    d = df.copy()
    code_col = next((c for c in d.columns if "종목코드" in c), None)
    name_col = next((c for c in d.columns if c in ("회사명","종목명","한글종목명","기업명","company_name")), None)
    # KSIC 컬럼은 괄호/언더스코어 모두 커버
    ksic_code = next((c for c in d.columns if c.startswith("통계청_한국표준산업분류_코드_11차")), None)
    ksic_name = next((c for c in d.columns if c.startswith("통계청_한국표준산업분류_11차")), None)

    out = pd.DataFrame()
    out["ticker"] = d[code_col].apply(zfill_ticker)
    out["company_name"] = d[name_col] if name_col else None
    out["company_name"] = out["company_name"].astype(str).str.strip().replace({"":None,"nan":None,"None":None})
    out.loc[out["company_name"].isna(), "company_name"] = out["ticker"]
    out["market"] = market_label                           # ★ 여기!
    out["ksic_mid_code"] = d[ksic_code].astype(str).str.strip() if ksic_code else None
    out["ksic_mid_name"] = d[ksic_name].astype(str).str.strip() if ksic_name else None
    out = out.dropna(subset=["ticker"]).drop_duplicates(subset=["ticker"])
    return out[["ticker","company_name","market","ksic_mid_code","ksic_mid_name"]]

def build_pd(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    date_col = next((c for c in d.columns if c == "날짜"), None)
    code_col = next((c for c in d.columns if "종목코드" in c), None)
    out = pd.DataFrame()
    out["date"]   = to_date(d[date_col])
    out["ticker"] = d[code_col].apply(zfill_ticker)
    # 원본 헤더 대응: PD_raw_avg, PD_raw_avg_ewma, PD_raw_avg_kalman
    col_raw   = next((c for c in d.columns if c in ("pd_raw_avg","pd_avg","raw_avg")), None)
    col_ewma  = next((c for c in d.columns if c in ("pd_raw_avg_ewma","pd_ewma_avg","ewma_pd_avg")), None)
    col_kal   = next((c for c in d.columns if c in ("pd_raw_avg_kalman","pd_kalman_avg","kalman_pd_avg")), None)
    out["pd_raw_avg"]        = to_f01(d[col_raw])  if col_raw else None
    out["pd_raw_avg_ewma"]   = to_f01(d[col_ewma]) if col_ewma else None
    out["pd_raw_avg_kalman"] = to_f01(d[col_kal])  if col_kal else None
    out = out.dropna(subset=["date","ticker"]).sort_values(["date","ticker"])
    return out[["date","ticker","pd_raw_avg","pd_raw_avg_ewma","pd_raw_avg_kalman"]]

def build_foreign(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    date_col = next((c for c in d.columns if c == "날짜"), None)
    code_col = next((c for c in d.columns if "종목코드" in c), None)
    out = pd.DataFrame()
    out["date"]   = to_date(d[date_col])
    out["ticker"] = d[code_col].apply(zfill_ticker)
    out["foreign_ratio"]  = to_ratio(d.get("foreign_ratio", None))
    out["foreign_shares"] = to_int64(d.get("foreign_shares", None))
    out = out.dropna(subset=["date","ticker"]).sort_values(["date","ticker"])
    return out[["date","ticker","foreign_ratio","foreign_shares"]]

def build_index(idx: pd.DataFrame) -> pd.DataFrame:
    d = idx.copy()
    d["date"] = to_date(d["날짜"])
    # 정규화 과정에서 'KOSPI_종가' -> 'kospi_종가' 로 들어왔을 수 있음
    kospi_col  = next((c for c in d.columns if c.endswith("kospi_종가") or c == "kospi_종가" or c == "KOSPI_종가".lower()), "kospi_종가")
    kosdaq_col = next((c for c in d.columns if c.endswith("kosdaq_종가") or c == "kosdaq_종가" or c == "KOSDAQ_종가".lower()), "kosdaq_종가")
    long = pd.melt(d, id_vars=["date"], value_vars=[kospi_col, kosdaq_col],
                   var_name="name", value_name="close")
    long["index_name"] = long["name"].str.replace("_종가","",regex=False).str.upper()
    long = long.drop(columns=["name"]).sort_values(["index_name","date"])
    long["return"] = long.groupby("index_name")["close"].pct_change().fillna(0.0)
    return long[["date","index_name","close","return"]]

# ===== 업서트 =====
def upsert(engine, df: pd.DataFrame, table: str, pk_cols):
    if df is None or df.empty:
        print(f"[SKIP] empty: {table}"); return
    cols = list(df.columns)
    col_list = ", ".join(f'"{c}"' for c in cols)
    conflict = ", ".join(f'"{c}"' for c in pk_cols)
    updates  = ", ".join(f'"{c}" = EXCLUDED."{c}"' for c in cols if c not in pk_cols)
    with engine.begin() as conn:
        conn.execute(text(f'CREATE TEMP TABLE tmp_{table} AS SELECT * FROM {table} WITH NO DATA;'))
        raw = conn.connection
        buf = io.StringIO(); df.to_csv(buf, index=False); buf.seek(0)
        with raw.cursor() as cur:
            with cur.copy(f'COPY tmp_{table} ({col_list}) FROM STDIN WITH (FORMAT CSV, HEADER TRUE)') as cp:
                cp.write(buf.getvalue())
        sql = f"""
        INSERT INTO {table} ({col_list})
        SELECT {col_list} FROM tmp_{table}
        ON CONFLICT ({conflict}) DO UPDATE SET {updates};
        """
        conn.execute(text(sql))
        conn.execute(text(f"DROP TABLE IF EXISTS tmp_{table};"))
    print(f"[UPSERT] {table}: {len(df):,} rows")

# ===== main =====
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--kp", required=True)
    ap.add_argument("--kd", required=True)
    ap.add_argument("--index", required=True, help="코스피/코스닥 인덱스 CSV 경로 (파일명에 쉼표 있으면 경로 전체를 따옴표로)")
    args = ap.parse_args()

    eng = create_engine(args.db, pool_pre_ping=True, future=True)

    kp  = normalize_cols(pd.read_csv(args.kp, encoding="utf-8-sig"))
    kd  = normalize_cols(pd.read_csv(args.kd, encoding="utf-8-sig"))
    idx = normalize_cols(pd.read_csv(args.index, encoding="utf-8-sig"))

    print("KP cols (normalized):", list(kp.columns))
    print("KD cols (normalized):", list(kd.columns))
    print("IDX cols (normalized):", list(idx.columns))

    comp = pd.concat([
        build_companies(kp, "KOSPI"),
        build_companies(kd, "KOSDAQ"),
    ], ignore_index=True).drop_duplicates(subset=["ticker"])
    upsert(eng, comp, "companies", pk_cols=["ticker"])

    pdd = pd.concat([build_pd(kp), build_pd(kd)], ignore_index=True)
    upsert(eng, pdd, "pd_daily", pk_cols=["date","ticker"])

    fh  = pd.concat([build_foreign(kp), build_foreign(kd)], ignore_index=True)
    upsert(eng, fh, "foreign_holdings_daily", pk_cols=["date","ticker"])

    midx = build_index(idx)
    upsert(eng, midx, "market_index_daily", pk_cols=["date","index_name"])

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