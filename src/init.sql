\c jobs

CREATE TABLE job_history(
    id SERIAL PRIMARY KEY,
    job_type TEXT NOT NULL,
    start_at TIMESTAMP NOT NULL,
    end_at TIMESTAMP,
    record_count INTEGER,
    duration_estimate INTEGER,
    CONSTRAINT valid_date_range CHECK (end_at IS NULL OR end_at >= start_at)
);

CREATE INDEX idx_job_history_job_type ON job_history (job_type);
CREATE INDEX idx_job_history_start_at ON job_history (start_at);
CREATE INDEX idx_job_history_end_at ON job_history (end_at);
