import unittest
import subprocess
import time
import requests
from fastapi.testclient import TestClient
from datetime import datetime
from src.api import app  # Import the FastAPI app from your api.py
from src.predictor import Predictor
from subprocess import Popen
from typing import Optional, Callable, Any


class APITest(unittest.TestCase):
    """Tests the Job Duration Predictor API endpoints."""

    @classmethod
    def report(cls, action: Callable, message: str) -> Any:
        print(f"  - {message}", end="")
        mark = time.time()
        result = action()
        elapsed_time = time.time() - mark
        print(f": {elapsed_time:.2f} seconds")
        return result

    @classmethod
    def docker_compose(cls, up: bool) -> None:
        """Helper method to run docker compose commands.

        Arguments:
        up: True if you want to bring up the containers, False if you want to
            shut them down. Turns into "up" or "down" after "docker compose"

        """
        Predictor.docker_compose(cls.compose_file, cls.docker_project, up, True)

    @classmethod
    def start_api_service(cls) -> Optional[Popen]:
        # Start the FastAPI service
        process = subprocess.Popen(
            ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        # Wait for the service to be ready
        cls.base_url = "http://127.0.0.1:8000"
        max_attempts = 10
        for attempt in range(max_attempts):
            try:
                response = requests.get(f"{cls.base_url}/health")
                if (
                    response.status_code == 200
                    and response.json().get("status") == "success"
                ):
                    return process
            except requests.ConnectionError:
                time.sleep(2)
        return None

    @classmethod
    def setUpClass(cls):
        print("\nInitializing API tests")
        cls.options = Predictor.default_options()
        cls.options["db_port"] = 5433
        cls.options["log_level"] = "warn"
        cls.options["model"] = ""
        # Start the database container
        cls.compose_file = "tests/docker-compose.yaml"
        cls.docker_project = "predictor-test"
        cls.report(
            lambda: cls.docker_compose(True), "Starting database container"
        )
        cls.compose_file = "tests/docker-compose.yaml"
        cls.docker_project = "api-test"
        cls.db_url = Predictor.make_db_url(cls.options)
        cls.engine = cls.report(
            lambda: Predictor.wait_for_engine(cls.db_url),
            "Waiting for database",
        )
        if not cls.engine:
            cls.tearDownClass()
            raise RuntimeError("Database did not start")
        cls.api_process = cls.report(
            lambda: cls.start_api_service(), "Starting the API service"
        )
        if not cls.api_process:
            raise RuntimeError("API service failed to start")
        # Initialize TestClient
        cls.client = TestClient(app)

    @classmethod
    def tearDownClass(cls):
        # Shut down the service
        try:
            cls.client.post("/shutdown")
        except Exception:
            pass  # Service may already be down

        # Ensure the process is terminated
        if cls.api_process:
            cls.api_process.terminate()
            cls.api_process.wait(timeout=10)

        # Stop the database container
        cls.docker_compose(False)

    def test_health_endpoint(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "success")
        self.assertTrue(data["healthy"])

    def test_job_history_length(self):
        response = self.client.get("/job-history-length")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "success")
        self.assertIsInstance(data["count"], int)
        self.assertGreaterEqual(data["count"], 0)

    def test_new_job(self):
        start_at = Predictor.dt_to_string(datetime.now())
        job_data = {"start_at": start_at, "record_count": 1000}
        response = self.client.post("/new-job", json=job_data)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "success")
        self.assertIsInstance(data["insert_id"], int)
        self.assertEqual(data["new_count"], data["original_count"] + 1)

    def test_flow(self):
        """Performs these functions, in order:

            - Clears the job history table

            - Inserts 1000 bogus job records

            - Checks the size of the job history table

            - Trains a model on the bogus jobs

            - Adds a new job1 that, unlike the bogus jobs, has just started
              running and has no end_at date.

            - Predicts the duration of job1

            - Adds a second job2

            - Predicts the duration of job2

            - Checks that the duration prediction for job2 is greater than the
              duration prediction of job1. This should be so because job2
              processes 10 times the number of records as job1.

        """
        bogus_record_count = 1000
        job1_record_count = 1000
        job2_record_count = 10000
        # Clear the job history table
        response = self.client.post("/delete-all-records")
        data = response.json()
        self.assertEqual(data["status"], "success")
        # Check job history length
        response = self.client.get("/job-history-length")
        self.assertEqual(response.json()["count"], 0)
        # Insert some bogus records
        count = {"count": bogus_record_count}
        response = self.client.post("/insert-bogus-job-history", json=count)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "success")
        self.assertIsInstance(data["old_count"], int)
        self.assertEqual(data["insert_count"], bogus_record_count)
        self.assertEqual(
            data["new_count"], data["old_count"] + bogus_record_count
        )
        # Check job history length
        response = self.client.get("/job-history-length")
        self.assertEqual(response.json()["count"], bogus_record_count)
        # Train
        response = self.client.post("/train")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "success")
        # Add a new job
        start_at = Predictor.dt_to_string(datetime.now())
        job_data = {"start_at": start_at, "record_count": job1_record_count}
        response = self.client.post("/new-job", json=job_data)
        job1_id = response.json()["insert_id"]
        # Predict
        response = self.client.post("/predict")
        self.assertEqual(response.status_code, 200)
        job1 = response.json()["predictions"][0]
        self.assertEqual(job1["id"], job1_id)
        # Add a second job
        start_at = Predictor.dt_to_string(datetime.now())
        job_data = {"start_at": start_at, "record_count": job2_record_count}
        response = self.client.post("/new-job", json=job_data)
        job2_id = response.json()["insert_id"]
        # Predict for the second job
        response = self.client.post("/predict")
        self.assertEqual(response.status_code, 200)
        job2 = response.json()["predictions"][0]
        self.assertEqual(job2["id"], job2_id)
        self.assertGreater(job2["duration_estimate"], job1["duration_estimate"])


if __name__ == "__main__":
    unittest.main()
