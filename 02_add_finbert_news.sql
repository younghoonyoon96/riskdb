-- 원문: 날짜/제목/예측감성 보관
CREATE TABLE IF NOT EXISTS finbert_news_titles (
  id BIGSERIAL PRIMARY KEY,
  date DATE NOT NULL,
  title TEXT NOT NULL,
  predicted_sentiment TEXT NOT NULL
    CHECK (predicted_sentiment IN ('positive','neutral','negative'))
);

CREATE INDEX IF NOT EXISTS ix_finbert_news_date ON finbert_news_titles(date);
CREATE INDEX IF NOT EXISTS ix_finbert_news_sentiment ON finbert_news_titles(predicted_sentiment);

-- 일별 감성별 기사수 집계 뷰
CREATE OR REPLACE VIEW v_finbert_daily_counts AS
SELECT
  date,
  COUNT(*) FILTER (WHERE predicted_sentiment='positive') AS n_pos,
  COUNT(*) FILTER (WHERE predicted_sentiment='neutral')  AS n_neu,
  COUNT(*) FILTER (WHERE predicted_sentiment='negative') AS n_neg
FROM finbert_news_titles
GROUP BY date
ORDER BY date;