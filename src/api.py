import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.predictor import Predictor

app = FastAPI(title="Job Duration Predictor API")
host = os.getenv("PREDICTOR_API_DB_HOST", "127.0.0.1")
port = os.getenv("PREDICTOR_API_DB_PORT", "5433")
log_level = os.getenv("PREDICTOR_API_LOG_LEVEL", "warn")
predictor = Predictor(
    {"db_host": host, "db_port": port, "log_level": log_level}
)
success_status = "success"
failure_status = "fail"


class NewJob(BaseModel):
    start_at: str
    record_count: int


class Count(BaseModel):
    count: int


@app.get("/health")
async def health_check():
    return {
        "status": success_status,
        "healthy": True,
    }


@app.get("/job-history-length")
async def job_history_length():
    return {"status": success_status, "count": predictor.job_history_length()}


@app.post("/insert-bogus-job-history")
async def insert_bogus_job_history(count: Count):
    original_count = predictor.job_history_length()
    if original_count == 0:
        predictor.insert_bogus_job_history(count.count)
        new_count = predictor.job_history_length()
        return {
            "status": success_status,
            "old_count": original_count,
            "insert_count": count.count,
            "new_count": new_count,
        }
    else:
        raise HTTPException(status_code=400, detail="Job history is not empty")


@app.post("/delete-all-records")
async def delete_all_records():
    original_count = predictor.job_history_length()
    if original_count > 0:
        predictor.clear_job_history()
        deleted = original_count
        new_count = predictor.job_history_length()
    else:
        deleted = 0
        new_count = 0
    return {
        "status": success_status,
        "old_count": original_count,
        "delete_count": deleted,
        "new_count": new_count,
    }


@app.post("/new-job")
async def new_job(job: NewJob):
    original_count = predictor.job_history_length()
    id = predictor.insert_new_job(job.start_at, job.record_count)
    new_count = predictor.job_history_length()
    return {
        "status": success_status,
        "original_count": original_count,
        "insert_id": id,
        "new_count": new_count,
    }


@app.post("/train")
async def train():
    predictor.train()
    return {"status": "success"}


@app.post("/predict")
async def predict():
    result = predictor.predict()
    if not result:
        result = []
    return {
        "status": "success",
        "explanation": result["explanation"],
        "predictions": result["predictions"],
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
