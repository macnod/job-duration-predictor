import time
import unittest
from src.predictor import Predictor
from sqlalchemy import create_engine, text
from datetime import datetime, timedelta


class PredictorTest(unittest.TestCase):
    """Tests the Predictor class, which predicts the duration of running jobs
    based on the history of similar job runs.

    """

    @classmethod
    def docker_compose(cls, up: bool) -> None:
        """Helper method to run docker compose commands.

        Arguments:
        up: True if you want to bring up the containers, False if you want to
            shut them down. Turns into "up" or "down" after "docker compose"

        """
        Predictor.docker_compose(cls.compose_file, cls.docker_project, up, True)
        # up_or_down: str = "up" if up else "down"
        # command = [
        #     "docker",
        #     "compose",
        #     "--file",
        #     cls.compose_file,
        #     "--project-name",
        #     cls.docker_project,
        #     "--progress",
        #     "quiet",
        #     up_or_down,
        # ]
        # if up_or_down == "up":
        #     command.append("--detach")
        # subprocess.run(command, check=True)

    @classmethod
    def make_db_url(cls):
        return "postgresql://{}:{}@{}:{}/{}".format(
            cls.db_user,
            cls.db_pass,
            cls.db_host,
            cls.db_port,
            cls.db_name,
        )

    @classmethod
    def setUpClass(cls):
        cls.epoch = datetime(1970, 1, 1)
        # Database connection
        cls.db_user = "jobs-user"
        cls.db_pass = "jobs-user-password"
        cls.db_host = "127.0.0.1"
        cls.db_port = "5433"
        cls.db_name = "jobs"
        # Start the database container
        cls.compose_file = "tests/docker-compose.yaml"
        cls.docker_project = "predictor-test"
        cls.docker_compose(True)
        cls.db_url = cls.make_db_url()
        max_attempts = 10
        for attempt in range(max_attempts):
            try:
                engine = create_engine(cls.db_url)
                with engine.connect() as conn:
                    conn.execute(text("SELECT 1"))
                break
            except Exception as e:
                if attempt >= 7:
                    print(f"Failed attempt #{attempt}: {e}")
                time.sleep(2)
        else:
            cls.tearDownClass()
            raise RuntimeError("Database did not start in time")
        # Initialize the Predictor instance
        cls.predictor = Predictor()
        # Insert some bogus records into the job history
        cls.initial_job_count = 1000
        cls.predictor.insert_bogus_job_history(cls.initial_job_count)

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
        self.assertEqual(self.predictor.db_url, self.make_db_url())
        self.assertIsNone(self.predictor.model)

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
        start_at_string = start_at.strftime("%Y-%m-%d %H:%M:%S")
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
        predictions = self.predictor.predict()
        self.assertEqual(len(predictions), 1)
        self.assertTrue(job_id in predictions)
        prediction = predictions[job_id]
        self.assertTrue("duration_estimate" in prediction)
        self.assertTrue("projected_completion" in prediction)
        self.assertGreater(prediction["projected_completion"], start_at)
        self.assertGreater(prediction["duration_estimate"], 100)
        self.assertGreater(300, prediction["duration_estimate"])
        # Compute the projected completion time here and make sure that it is
        # about the same as what came back in the prediction.
        computed = (
            start_at + timedelta(seconds=prediction["duration_estimate"])
        ).replace(microsecond=0)
        self.assertTrue(
            abs(self.unixtime(computed)
                - self.unixtime(prediction['projected_completion']))
            < 2)
        # Now, check the database directly, to ensure the job was updated
        # with the predicted duration.
        sql = text(
            """
            select duration_estimate
            from job_history
            where id = :job_id
        """
        )
        engine = create_engine(self.db_url)
        with engine.connect() as conn:
            result = conn.execute(sql, {"job_id": job_id})
            duration_estimate = result.scalar()
            self.assertEqual(duration_estimate, prediction["duration_estimate"])
