import unittest
from src.predictor import Predictor
from sqlalchemy import text
from datetime import datetime, timedelta
from time import time
from typing import Any, Callable


class PredictorTest(unittest.TestCase):
    """Tests the Predictor class, which predicts the duration of running jobs
    based on the history of similar job runs.

    """

    @classmethod
    def report(cls, action: Callable, message: str) -> Any:
        print(f"  - {message}", end="")
        mark = time()
        result = action()
        elapsed_time = time() - mark
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
    def setUpClass(cls):
        print("\nInitializing predictor tests")
        cls.epoch = datetime(1970, 1, 1)
        cls.options = Predictor.default_options()
        cls.options["db_port"] = 5433
        cls.options["log_level"] = "warn"
        cls.options["model"] = ""
        # Start the database container
        cls.compose_file = "tests/docker-compose.yaml"
        cls.docker_project = "predictor-test"
        cls.report(
            lambda: cls.docker_compose(True),
            "Starting database container"
        )
        cls.db_url = Predictor.make_db_url(cls.options)
        cls.engine = cls.report(
            lambda: Predictor.wait_for_engine(cls.db_url),
            "Waiting for the database"
        )
        if not cls.engine:
            cls.tearDownClass()
            raise RuntimeError("Database did not start")
        # Initialize the Predictor instance
        cls.predictor = cls.report(
            lambda: Predictor(cls.options),
            "Instantiating predictor"
        )
        # Insert some bogus records into the job history
        cls.initial_job_count = 1000
        cls.report(
            lambda: cls.predictor.insert_bogus_job_history(
                cls.initial_job_count
            ),
            "Inserting {:d} records into the database".format(
                cls.initial_job_count)
        )

    @classmethod
    def tearDownClass(cls):
        # Stop the database container
        cls.docker_compose(False)

    def unixtime(self, dt: datetime) -> int:
        """Returns the time represented by dt as unix time (the number of
        seconds elapsed since 1970-01-01.

        """
        return int((dt - self.epoch).total_seconds())

    def test_init(self):
        self.assertEqual(self.predictor.config["estimators"], 100)
        self.assertEqual(self.predictor.config["job_type"], "default-job")
        self.assertEqual(
            self.predictor.db_url, Predictor.make_db_url(self.options)
        )

    def test_fetch_jobs(self):
        df = self.predictor.fetch_jobs(historical=True)
        self.assertEqual(len(df), self.initial_job_count)
        df = self.predictor.fetch_jobs(historical=False)
        self.assertEqual(len(df), 0)

    def test_model(self):
        # We start out with no model
        self.assertIsNone(self.predictor.model)
        # Successful training creates a model
        self.assertTrue(self.predictor.train())
        self.assertIsNotNone(self.predictor.model)
        # Insert a new job so that we can test prediction
        start_at = datetime.now()
        start_at_string = Predictor.dt_to_string(start_at)
        job_id = self.predictor.insert_new_job(start_at_string, 1000)
        # Ensure job was inserted
        self.assertIsNotNone(job_id)
        df = self.predictor.fetch_jobs(historical=True)
        self.assertEqual(len(df), self.initial_job_count)
        df = self.predictor.fetch_jobs(historical=False)
        self.assertEqual(len(df), 1)
        # Predict. This will look for jobs with null end_at and null
        # duration_estimate, compute duration estimates for those,
        # update the records in the database, and return a dict with
        # job IDs for keys.
        result = self.predictor.predict()
        self.assertTrue("explanation" in result)
        self.assertTrue("predictions" in result)
        predictions = result["predictions"]
        self.assertEqual(len(predictions), 1)
        prediction = predictions[0]
        self.assertTrue("id" in prediction)
        self.assertEqual(prediction["id"], job_id)
        self.assertTrue("duration_estimate" in prediction)
        self.assertTrue("projected_completion" in prediction)
        self.assertGreater(
            prediction["projected_completion"], Predictor.dt_to_string(start_at)
        )
        self.assertGreater(prediction["duration_estimate"], 100)
        self.assertGreater(300, prediction["duration_estimate"])
        # Compute the projected completion time here and make sure that it is
        # about the same as what came back in the prediction.
        computed = (
            start_at + timedelta(seconds=prediction["duration_estimate"])
        ).replace(microsecond=0)
        self.assertTrue(
            abs(
                self.unixtime(computed)
                - self.unixtime(
                    Predictor.dt_from_string(prediction["projected_completion"])
                )
            )
            < 2
        )
        # Now, check the database directly, to ensure the job was updated
        # with the predicted duration.
        sql = text(
            """
            select duration_estimate
            from job_history
            where id = :job_id
        """
        )
        with self.engine.connect() as conn:
            result = conn.execute(sql, {"job_id": job_id})
            duration_estimate = result.scalar()
            self.assertEqual(duration_estimate, prediction["duration_estimate"])
