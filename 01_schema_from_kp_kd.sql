-- companies: 종목 기본정보
CREATE TABLE IF NOT EXISTS companies (
  ticker         VARCHAR(12) PRIMARY KEY,
  company_name   TEXT NOT NULL,
  market         TEXT NOT NULL,
  ksic_mid_code  TEXT,
  ksic_mid_name  TEXT,
  created_at     TIMESTAMPTZ DEFAULT now()
);

-- pd_daily: 일별 부도확률
CREATE TABLE IF NOT EXISTS pd_daily (
  date            DATE NOT NULL,
  ticker          VARCHAR(12) NOT NULL,
  market          TEXT,
  pd_raw          DOUBLE PRECISION,
  pd_ewma         DOUBLE PRECISION,
  pd_kalman       DOUBLE PRECISION,
  pd_raw_avg      DOUBLE PRECISION,
  pd_ewma_avg     DOUBLE PRECISION,
  pd_kalman_avg   DOUBLE PRECISION,
  pd_hat          DOUBLE PRECISION,
  PRIMARY KEY (date, ticker)
);

-- 외국인 보유
CREATE TABLE IF NOT EXISTS foreign_holdings_daily (
  date            DATE NOT NULL,
  ticker          VARCHAR(12) NOT NULL,
  foreign_ratio   DOUBLE PRECISION,
  foreign_shares  BIGINT,
  PRIMARY KEY (date, ticker)
);

-- 지수 일별 종가/수익률
CREATE TABLE IF NOT EXISTS market_index_daily (
  date        DATE NOT NULL,
  index_name  TEXT NOT NULL,
  close       DOUBLE PRECISION,
  return      DOUBLE PRECISION,
  PRIMARY KEY (date, index_name)
);