# -*- coding: utf-8 -*-
"""
FinBERT 인덱스 CSV → finbert_index_daily 업서트
CSV 컬럼(예시):
- date
- finbert_net_sentiment
- finbert_expected_value      # 기대값: +1*긍정 + 0*중립 -1*부정
- finbert_pos_ratio           # [0,1]

사용:
  python 04_load_finbert.py --db "$DATABASE_URL" --csv "/path/to/finbert_index.csv"
"""

import argparse, io, re, unicodedata
import pandas as pd
from sqlalchemy import create_engine, text

_WEIRD_WS = re.compile(r"[\u00A0\u200B\u200C\u200D\uFEFF]")

def canon_header(s: str) -> str:
    t = unicodedata.normalize("NFKC", str(s))
    t = _WEIRD_WS.sub("", t).strip().lower()
    t = re.sub(r"[^\w\s\-\./]", " ", t)
    t = t.replace("-", "_").replace("/", "_")
    t = re.sub(r"\s+", "_", t).strip("_")
    return t

def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d.columns = [canon_header(c) for c in d.columns]
    return d

def to_date(s):  return pd.to_datetime(s, errors="coerce").dt.date
def to_float(s): return pd.to_numeric(s, errors="coerce")

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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db",  required=True)
    ap.add_argument("--csv", required=True)
    args = ap.parse_args()

    eng = create_engine(args.db, pool_pre_ping=True, future=True)
    df  = normalize_cols(pd.read_csv(args.csv, encoding="utf-8-sig"))

    # 컬럼 매핑(헤더 변형 대응)
    date_col = next((c for c in df.columns if c in ("date","날짜")), None)
    ns_col   = next((c for c in df.columns if c in ("finbert_net_sentiment","net_sentiment")), None)
    ev_col   = next((c for c in df.columns if c in ("finbert_expected_value","expected_value")), None)
    pr_col   = next((c for c in df.columns if c in ("finbert_pos_ratio","pos_ratio")), None)

    if not date_col:
        raise ValueError("CSV에 date(또는 날짜) 컬럼이 필요합니다.")
    if not any([ns_col, ev_col, pr_col]):
        raise ValueError("finbert_* 컬럼이 필요합니다.")

    out = pd.DataFrame()
    out["date"]                   = to_date(df[date_col])
    if ns_col: out["finbert_net_sentiment"]  = to_float(df[ns_col])
    if ev_col: out["finbert_expected_value"] = to_float(df[ev_col]).clip(-1, 1)
    if pr_col: out["finbert_pos_ratio"]      = to_float(df[pr_col]).clip(0, 1)

    out = out.dropna(subset=["date"]).sort_values("date")
    out = out[["date","finbert_net_sentiment","finbert_expected_value","finbert_pos_ratio"]]

    upsert(eng, out, "finbert_index_daily", pk_cols=["date"])
    print("[DONE] FinBERT 적재 완료")

if __name__ == "__main__":
    main()