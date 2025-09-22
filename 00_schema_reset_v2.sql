-- 00_schema_reset_v3.sql (핵심 변경: companies.market 추가)
DROP MATERIALIZED VIEW IF EXISTS mv_industry_pd_daily;
DROP VIEW IF EXISTS v_foreign_flows_daily;

DROP TABLE IF EXISTS market_index_daily CASCADE;
DROP TABLE IF EXISTS foreign_holdings_daily CASCADE;
DROP TABLE IF EXISTS pd_daily CASCADE;
DROP TABLE IF EXISTS companies CASCADE;

CREATE TABLE companies (
  ticker         VARCHAR(12) PRIMARY KEY,
  company_name   TEXT        NOT NULL,
  market         TEXT        NOT NULL,  -- ★ KOSPI | KOSDAQ
  ksic_mid_code  TEXT,
  ksic_mid_name  TEXT,
  created_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE pd_daily (
  date               DATE NOT NULL,
  ticker             VARCHAR(12) NOT NULL,
  pd_raw_avg         DOUBLE PRECISION,
  pd_raw_avg_ewma    DOUBLE PRECISION,
  pd_raw_avg_kalman  DOUBLE PRECISION,
  CONSTRAINT ck_pd_raw_avg        CHECK (pd_raw_avg        IS NULL OR (pd_raw_avg        BETWEEN 0 AND 1)),
  CONSTRAINT ck_pd_raw_avg_ewma   CHECK (pd_raw_avg_ewma   IS NULL OR (pd_raw_avg_ewma   BETWEEN 0 AND 1)),
  CONSTRAINT ck_pd_raw_avg_kalman CHECK (pd_raw_avg_kalman IS NULL OR (pd_raw_avg_kalman BETWEEN 0 AND 1)),
  PRIMARY KEY (date, ticker),
  FOREIGN KEY (ticker) REFERENCES companies(ticker)
);
CREATE INDEX ix_pd_daily_date   ON pd_daily(date);
CREATE INDEX ix_pd_daily_ticker ON pd_daily(ticker);

CREATE TABLE foreign_holdings_daily (
  date            DATE NOT NULL,
  ticker          VARCHAR(12) NOT NULL,
  foreign_ratio   DOUBLE PRECISION,
  foreign_shares  BIGINT,
  CONSTRAINT ck_foreign_ratio CHECK (foreign_ratio IS NULL OR (foreign_ratio BETWEEN 0 AND 100)),
  CONSTRAINT ck_foreign_shares_nonneg CHECK (foreign_shares IS NULL OR (foreign_shares >= 0)),
  PRIMARY KEY (date, ticker),
  FOREIGN KEY (ticker) REFERENCES companies(ticker)
);
CREATE INDEX ix_fh_daily_date_ticker ON foreign_holdings_daily(date, ticker);

CREATE TABLE market_index_daily (
  date        DATE NOT NULL,
  index_name  TEXT NOT NULL,
  close       DOUBLE PRECISION,
  return      DOUBLE PRECISION,
  PRIMARY KEY (date, index_name)
);
CREATE INDEX ix_mkt_idx_dt ON market_index_daily(date, index_name);

CREATE MATERIALIZED VIEW mv_industry_pd_daily AS
SELECT
  p.date,
  c.ksic_mid_code,
  c.ksic_mid_name,
  AVG(COALESCE(p.pd_raw_avg_ewma, p.pd_raw_avg_kalman, p.pd_raw_avg)) AS industry_pd_avg
FROM pd_daily p
JOIN companies c USING (ticker)
GROUP BY p.date, c.ksic_mid_code, c.ksic_mid_name;
CREATE INDEX ix_mv_industry_pd_daily ON mv_industry_pd_daily(date, ksic_mid_code);

CREATE VIEW v_foreign_flows_daily AS
SELECT
  f.date, f.ticker, c.company_name, c.market,  -- ★ market 포함해 두면 필터 편함
  c.ksic_mid_code, c.ksic_mid_name,
  f.foreign_ratio, f.foreign_shares,
  (f.foreign_shares - LAG(f.foreign_shares) OVER (PARTITION BY f.ticker ORDER BY f.date)) AS delta_shares,
  (f.foreign_ratio  - LAG(f.foreign_ratio)  OVER (PARTITION BY f.ticker ORDER BY f.date)) AS delta_ratio,
  CASE
    WHEN (f.foreign_shares - LAG(f.foreign_shares) OVER (PARTITION BY f.ticker ORDER BY f.date)) > 0 THEN '매수'
    WHEN (f.foreign_shares - LAG(f.foreign_shares) OVER (PARTITION BY f.ticker ORDER BY f.date)) < 0 THEN '매도'
    ELSE '변화없음'
  END AS flow_label
FROM foreign_holdings_daily f
JOIN companies c USING (ticker);