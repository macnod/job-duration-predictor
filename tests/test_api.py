import unittest
from fastapi.testclient import TestClient
from datetime import datetime
from src.api import app
from src.predictor import Predictor


class APITest(unittest.TestCase):
    """Tests the Job Duration Predictor API endpoints."""

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.client.post("/delete-all-records")

    def test_health_endpoint(self):
        response = self.client.get("/health")
        self.assertEqual(
            response.status_code, 200, "/health endpoint returns 200"
        )
        data = response.json()
        self.assertEqual(
            data["status"], "success", "/health endpoint status is 'success'"
        )
        self.assertTrue(
            data["healthy"], "/health endpoint healthy values is True"
        )

    def test_job_history_length(self):
        response = self.client.get("/job-history-length")
        self.assertEqual(
            response.status_code,
            200,
            "/job-history-length endpoint returns 200",
        )
        data = response.json()
        self.assertEqual(
            data["status"],
            "success",
            "/job-history-length endpoint status is 'success'",
        )
        self.assertIsInstance(
            data["count"], int, "/job-history-length count value is an integer"
        )
        self.assertGreaterEqual(
            data["count"], 0, "/job-history-length count value is 0"
        )

    def test_new_job(self):
        start_at = Predictor.dt_to_string(datetime.now())
        job_data = {"start_at": start_at, "record_count": 1000}
        response = self.client.post("/new-job", json=job_data)
        self.assertEqual(
            response.status_code, 200, "Post to /new-job returns 200"
        )
        data = response.json()
        self.assertEqual(
            data["status"],
            "success",
            "Post to /new-job return 'success' status",
        )
        self.assertIsInstance(
            data["insert_id"], int, "/new-job insert_id is an integer"
        )
        self.assertEqual(
            data["new_count"],
            data["original_count"] + 1,
            "job count increased by 1",
        )

    def test_flow(self):
        bogus_record_count = 1000
        job1_record_count = 1000
        job2_record_count = 10000
        # Clear the job history table
        response = self.client.post("/delete-all-records")
        data = response.json()
        self.assertEqual(
            data["status"],
            "success",
            "Post to /delete-all-records returns 'success' status",
        )
        # Check job history length
        response = self.client.get("/job-history-length")
        self.assertEqual(
            response.json()["count"], 0, "Job history length is now 0"
        )
        # Insert some bogus records
        count = {"count": bogus_record_count}
        response = self.client.post("/insert-bogus-job-history", json=count)
        self.assertEqual(
            response.status_code,
            200,
            "Post to /insert-bogus-job-history return 200",
        )
        data = response.json()
        self.assertEqual(
            data["status"],
            "success",
            "Post to /insert-bogous-job-history return 'success' status",
        )
        self.assertIsInstance(
            data["old_count"],
            int,
            "/insert-bogus-job-history old_count is an integer",
        )
        self.assertEqual(
            data["insert_count"],
            bogus_record_count,
            f"Inserted {bogus_record_count} records",
        )
        self.assertEqual(
            data["new_count"],
            data["old_count"] + bogus_record_count,
            f"Count increased by {bogus_record_count}",
        )
        # Check job history length
        response = self.client.get("/job-history-length")
        self.assertEqual(
            response.json()["count"],
            bogus_record_count,
            "/job-history-length count is correct",
        )
        # Train
        response = self.client.post("/train")
        self.assertEqual(
            response.status_code, 200, "Post to /train returns 200"
        )
        data = response.json()
        self.assertEqual(data["status"], "success")
        # Add a new job
        start_at = Predictor.dt_to_string(datetime.now())
        job_data = {"start_at": start_at, "record_count": job1_record_count}
        response = self.client.post("/new-job", json=job_data)
        job1_id = response.json()["insert_id"]
        # Predict
        response = self.client.post("/predict")
        self.assertEqual(
            response.status_code, 200, "Post to /predict returns 200"
        )
        job1 = response.json()["predictions"][0]
        self.assertEqual(
            job1["id"], job1_id, "/predict response includes expected Job ID"
        )
        # Add a second job
        start_at = Predictor.dt_to_string(datetime.now())
        job_data = {"start_at": start_at, "record_count": job2_record_count}
        response = self.client.post("/new-job", json=job_data)
        job2_id = response.json()["insert_id"]
        # Predict for the second job
        response = self.client.post("/predict")
        self.assertEqual(
            response.status_code, 200, "Second post to /predict returns 200"
        )
        job2 = response.json()["predictions"][0]
        self.assertEqual(
            job2["id"],
            job2_id,
            "Second post to /predict returns expected Job ID",
        )
        self.assertGreater(
            job2["duration_estimate"],
            job1["duration_estimate"],
            "Job 2 correctly predicted to take longer than Job 1",
        )


if __name__ == "__main__":
    breakpoint()
    unittest.main()
