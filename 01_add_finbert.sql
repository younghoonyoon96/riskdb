-- FinBERT 인덱스(일별) 저장 테이블 (존재하면 생성 안 함)
CREATE TABLE IF NOT EXISTS finbert_index_daily (
  date                    DATE PRIMARY KEY,        -- 기준일
  finbert_net_sentiment   DOUBLE PRECISION,        -- 순감성(원자료)
  finbert_expected_value  DOUBLE PRECISION,        -- 기대값: [-1, 1]
  finbert_pos_ratio       DOUBLE PRECISION,        -- 긍정 비율: [0, 1]
  CONSTRAINT ck_f_expval CHECK (finbert_expected_value IS NULL OR (finbert_expected_value BETWEEN -1 AND 1)),
  CONSTRAINT ck_f_posrat CHECK (finbert_pos_ratio      IS NULL OR (finbert_pos_ratio      BETWEEN 0  AND 1))
);

-- 조회 성능(기간 쿼리용)
CREATE INDEX IF NOT EXISTS ix_finbert_index_daily_date ON finbert_index_daily(date);