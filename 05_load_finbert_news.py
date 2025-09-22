# -*- coding: utf-8 -*-
"""
FinBERT 뉴스 타이틀 CSV → finbert_news_titles 적재
CSV 예시 컬럼:
- date
- title
- predicted_sentiment  # 'positive'/'neutral'/'negative' 또는 '긍정'/'중립'/'부정'

사용:
  python 05_load_finbert_news.py --db "$DATABASE_URL" --csv "/path/to/finbert_news.csv"
"""
import argparse, io, re, unicodedata, hashlib
import pandas as pd
from sqlalchemy import create_engine, text

_WEIRD_WS = re.compile(r"[\u00A0\u200B\u200C\u200D\uFEFF]")

def canon_header(s: str) -> str:
    t = unicodedata.normalize("NFKC", str(s))
    t = _WEIRD_WS.sub("", t).strip().lower()
    t = re.sub(r"[^\w\s\-\./가-힣]", " ", t)
    t = t.replace("-", "_").replace("/", "_")
    t = re.sub(r"\s+", "_", t).strip("_")
    return t

def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy(); d.columns = [canon_header(c) for c in d.columns]; return d

def to_date(s):  return pd.to_datetime(s, errors="coerce").dt.date

def map_sentiment(x: str) -> str | None:
    if x is None: return None
    s = str(x).strip().lower()
    mapping = {
        "positive": "positive", "pos": "positive", "긍정": "positive",
        "neutral": "neutral", "neu": "neutral", "중립": "neutral",
        "negative": "negative", "neg": "negative", "부정": "negative",
    }
    return mapping.get(s)

def upsert(engine, df: pd.DataFrame, table: str):
    if df is None or df.empty:
        print(f"[SKIP] empty: {table}"); return
    cols = list(df.columns)
    col_list = ", ".join(f'"{c}"' for c in cols)
    with engine.begin() as conn:
        conn.execute(text(f'CREATE TEMP TABLE tmp_{table} AS SELECT * FROM {table} WITH NO DATA;'))
        raw = conn.connection
        buf = io.StringIO(); df.to_csv(buf, index=False); buf.seek(0)
        with raw.cursor() as cur:
            with cur.copy(f'COPY tmp_{table} ({col_list}) FROM STDIN WITH (FORMAT CSV, HEADER TRUE)') as cp:
                cp.write(buf.getvalue())
        # 중복 방지: 같은 (date, title, predicted_sentiment) 조합만 새로 insert
        sql = f"""
        INSERT INTO {table} ({col_list})
        SELECT {col_list} FROM tmp_{table} t
        WHERE NOT EXISTS (
            SELECT 1 FROM {table} x
            WHERE x.date = t.date
              AND x.title = t.title
              AND x.predicted_sentiment = t.predicted_sentiment
        );
        """
        conn.execute(text(sql))
        conn.execute(text(f"DROP TABLE IF EXISTS tmp_{table};"))
    print(f"[INSERT] {table}: +{len(df):,} rows (중복 제외)")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--csv", required=True)
    args = ap.parse_args()

    eng = create_engine(args.db, pool_pre_ping=True, future=True)
    df  = normalize_cols(pd.read_csv(args.csv, encoding="utf-8-sig"))

    # 필드 찾기
    date_col = next((c for c in df.columns if c in ("date","날짜")), None)
    title_col = next((c for c in df.columns if c in ("title","제목")), None)
    ps_col = next((c for c in df.columns if c in ("predicted_sentiment","감성","감성분류")), None)
    if not (date_col and title_col and ps_col):
        raise ValueError("CSV에 date, title, predicted_sentiment(또는 대응 한글 컬럼)가 필요합니다.")

    out = pd.DataFrame()
    out["date"] = to_date(df[date_col])
    out["title"] = df[title_col].astype(str).str.strip()
    out["predicted_sentiment"] = df[ps_col].map(map_sentiment)

    out = out.dropna(subset=["date","title","predicted_sentiment"]).drop_duplicates()
    upsert(eng, out, "finbert_news_titles")
    print("[DONE] finbert_news_titles 적재 완료")

if __name__ == "__main__":
    main()