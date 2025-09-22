# -*- coding: utf-8 -*-
"""
바뀐 스키마 + 인덱스(kp,kd_index.csv) 적재 (업서트)
- companies / pd_daily / foreign_holdings_daily / market_index_daily
- 마지막에 mv_industry_pd_daily REFRESH
- 입력된 CSV에 따라 자동 실행 (mode 불필요)
"""
import argparse, io, re, unicodedata
import pandas as pd
from sqlalchemy import create_engine, text

# ===== 기존 유틸 함수들 (zfill_ticker, canon_header, normalize_cols, build_* 등은 그대로) =====
# (생략 - 기존 코드 그대로 유지)

# ===== 업서트 함수 =====
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
    ap.add_argument("--db", required=True, help="Postgres 연결 URL")
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
        upsert(eng, pdd, "pd_daily", pk_cols=["date","ticker"])
        fh  = build_foreign(kp)
        upsert(eng, fh, "foreign_holdings_daily", pk_cols=["date","ticker"])

    # KOSDAQ
    if args.kd:
        kd = normalize_cols(pd.read_csv(args.kd, encoding="utf-8-sig"))
        comp = build_companies(kd, "KOSDAQ")
        upsert(eng, comp, "companies", pk_cols=["ticker"])
        pdd = build_pd(kd)
        upsert(eng, pdd, "pd_daily", pk_cols=["date","ticker"])
        fh  = build_foreign(kd)
        upsert(eng, fh, "foreign_holdings_daily", pk_cols=["date","ticker"])

    # Index
    if args.index:
        idx = normalize_cols(pd.read_csv(args.index, encoding="utf-8-sig"))
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