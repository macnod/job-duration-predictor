import holidays
import json
import logging
import numpy as np
import os
import pandas as pd
import random
import subprocess
import sys

from argparse import ArgumentParser
from datetime import datetime, timedelta
from pandas.core.frame import DataFrame
from pandas.core.series import Series
from sklearn.ensemble import RandomForestRegressor
from sqlalchemy import create_engine, text
from sqlalchemy.exc import DatabaseError
from sqlalchemy.sql.elements import TextClause
from typing import Optional, Union, Any, Tuple

# Set up logging
logging.basicConfig(
    level=logging.WARN,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


class Predictor:
    def __init__(self, command_line_options: Optional[dict[str, Any]] = None):
        self.config = self.build_config(command_line_options)
        self.db_url = self.make_db_url()
        if self.config["no_db"]:
            self.engine = None
        else:
            self.engine = create_engine(self.db_url)
        self.model = None
        self.us_holidays = holidays.US()

    @staticmethod
    def docker_compose(
        compose_file: str, project: str, up: bool, quiet: bool
    ) -> None:
        up_or_down: str = "up" if up else "down"
        progress = "quiet" if quiet else "auto"
        command = [
            "docker",
            "compose",
            "--file",
            compose_file,
            "--project-name",
            project,
            "--progress",
            progress,
            up_or_down,
        ]
        if up_or_down == "up":
            command.append("--detach")
        subprocess.run(command, check=True)

    @staticmethod
    def default_options() -> dict[str, Any]:
        return {
            "db_user": "jobs-user",
            "db_pass": "jobs-user-password",
            "db_host": "127.0.0.1",
            "db_port": 5433,
            "db_name": "jobs",
            "estimators": 100,
            "random_state": 42,
            "job_type": "default-job",
            "recency_weight": 0.01,
            "training_row_limit": 10000,
            "no_db": False,
            "start": None,
            "record_count": None,
        }

    @staticmethod
    def command_line_options() -> dict[str, Any]:
        parser = ArgumentParser(
            prog="Predictor",
            description="Predicts job durations based on history",
        )
        shortcuts = {
            "db_user": "-u",
            "db_pass": "-P",
            "db_port": "-p",
            "db_name": "-d",
            "estimators": "-e",
            "random_state": "-r",
            "job_type": "-j",
            "recency_weight": "-w",
            "training_row_limit": "-l",
        }
        for key, value in Predictor.default_options().items():
            longname = f"--{key}"
            shortcut = shortcuts.get(key)
            if key == "no_db":
                parser.add_argument(
                    "--no_db", action="store_true", default=False
                )
            else:
                if shortcut:
                    parser.add_argument(shortcut, longname)
                else:
                    parser.add_argument(longname)
        parser.add_argument("command")
        options = parser.parse_args()
        return {k: v for k, v in vars(options).items() if v is not None}

    def make_db_url(self) -> str:
        return "postgresql://{}:{}@{}:{}/{}".format(
            self.config["db_user"],
            self.config["db_pass"],
            self.config["db_host"],
            self.config["db_port"],
            self.config["db_name"],
        )

    def build_config(
        self, command_line_options: Optional[dict[str, Any]] = None
    ) -> dict[str, Any]:
        defaults = Predictor.default_options()
        config = {}
        options = command_line_options or dict()
        for key in defaults:
            if key in options:
                config[key] = options[key]
            else:
                env_var = f"PREDICTOR_{key.upper()}"
                config[key] = os.getenv(env_var, defaults[key])
        return config

    def job_query(self, historical: bool) -> TextClause:
        """Generate SQL for retrieving jobs. If historical is True, then
        historical jobs are retrieved. Otherwise, currently-running jobs
        are retrieved"""

        if historical:
            query = """
                SELECT
                    jh1.id,
                    jh1.job_type,
                    jh1.start_at,
                    jh1.record_count,
                    (
                        SELECT COUNT(*)
                        FROM job_history jh2
                        WHERE jh2.job_type = jh1.job_type
                          AND jh2.id != jh1.id
                          AND jh2.start_at <= jh1.end_at
                          AND (
                              jh2.end_at is null
                              OR jh2.end_at >= jh1.start_at
                          )
                    ) AS overlapping_job_count,
                    EXTRACT(EPOCH FROM (jh1.end_at - jh1.start_at))
                      AS duration,
                    (
                        EXTRACT(HOUR FROM jh1.start_at)
                        + EXTRACT(MINUTE FROM jh1.start_at) / 60.0
                    ) AS time_of_day,
                    EXTRACT(DOW FROM jh1.start_at) + 1 AS weekday,
                    EXTRACT(MONTH FROM jh1.start_at) AS month
                FROM job_history jh1
                WHERE
                    jh1.job_type = :job_type
                    AND jh1.end_at is not null
                ORDER BY jh1.start_at DESC
                LIMIT :limit
            """

        else:
            query = """
                SELECT
                    jh1.id,
                    jh1.job_type,
                    jh1.start_at,
                    jh1.record_count,
                    (
                        SELECT COUNT(*)
                        FROM job_history jh2
                        WHERE jh2.job_type = jh1.job_type
                        AND jh2.id != jh1.id
                        AND (jh2.end_at is null OR jh2.end_at >= jh1.start_at)
                    ) AS overlapping_job_count,
                    (
                        EXTRACT(HOUR FROM jh1.start_at)
                        + EXTRACT(MINUTE FROM jh1.start_at) / 60.0
                    ) AS time_of_day,
                    EXTRACT(DOW FROM jh1.start_at) + 1 AS weekday,
                    EXTRACT(MONTH FROM jh1.start_at) AS month
                FROM job_history jh1
                WHERE
                    jh1.job_type = :job_type
                    AND jh1.end_at is null
                    AND jh1.duration_estimate is null
                ORDER BY jh1.start_at DESC
                LIMIT :limit
            """

        replacement = "not null" if historical else "null"
        return text(query.replace(":NULL_OR_NOTNULL", replacement))

    def fetch_jobs(self, historical: bool) -> DataFrame:
        """Retrieve job history data for the specified job type."""
        query = self.job_query(historical=historical)
        try:
            with self.engine.connect() as conn:
                df = pd.read_sql_query(
                    query,
                    conn,
                    params={
                        "job_type": self.config["job_type"],
                        "limit": self.config["training_row_limit"],
                    },
                )
            logger.info(
                f"Fetched {len(df)} records for job_type "
                + self.config["job_type"]
            )
            return df
        except Exception as e:
            logger.error(f"Error fetching run history: {e}")
            raise

    def engineer_features(self, df: DataFrame) -> Tuple[DataFrame, list]:
        """Engineer features for training or prediction."""
        df = df.copy()
        df["holiday"] = (
            df["start_at"].apply(lambda x: x in self.us_holidays).astype(int)
        )
        # Select features
        features = [
            "time_of_day",
            "weekday",
            "month",
            "holiday",
            "overlapping_job_count",
            "record_count",
        ]

        return df, features

    def compute_recency_weights(self, df: DataFrame) -> Series:
        """Calculate sample weights based on recency using exponential decay."""
        current_time = datetime.now()
        time_diffs = (current_time - df["start_at"]).dt.total_seconds() / (
            24 * 3600
        )  # Convert to days
        weights = np.exp(-self.config["recency_weight"] * time_diffs)
        return weights

    def train(self) -> bool:
        """Train the model on run history data."""
        df = self.fetch_jobs(historical=True)
        if df.empty:
            logger.warning("No data available for training")
            return False

        df, features = self.engineer_features(df)

        X = df[features]
        y = df["duration"]
        weights = self.compute_recency_weights(df)

        self.model = RandomForestRegressor(
            n_estimators=self.config["estimators"],
            random_state=self.config["random_state"],
        )
        self.model.fit(X, y, sample_weight=weights)
        logger.info("Model trained successfully")
        return True

    def predict(self) -> Optional[dict[str, Union[datetime, int]]]:
        """Predict job duration. This function retrieves from the job_history
        table the jobs that have started but have not yet completed and have not
        yet been assigned a duration estimate. For each job, the function
        predicts the job's duration and updates the job's duration_estimate
        field with the prediction.

        If no jobs require estimates, this function returns None.

        If the model has not been trained, this function raises a ValueError
        exception.

        """
        result = {}
        df = self.fetch_jobs(historical=False)
        if df.empty:
            logger.info("No jobs need duration predictions")
            return
        if self.model is None:
            logger.error("Model not trained")
            raise ValueError("Model not trained")
        df, features = self.engineer_features(df)
        X = df[features]
        predictions = self.model.predict(X)
        id_list = df["id"]
        start_at_list = df["start_at"]
        for i, prediction in enumerate(predictions):
            id = int(id_list[i])
            start_at = start_at_list[i]
            rounded = round(prediction)
            message = f"Predicted job {id} duration: {rounded} seconds"
            logger.info(message)
            self.update_duration_estimate(id, rounded)
            result[id] = {
                "duration_estimate": rounded,
                "projected_completion": start_at + timedelta(seconds=rounded),
            }
        return result

    def update_duration_estimate(self, id: int, prediction: int) -> None:
        """Updates the job_history row specified via the id parameter with the
        the value of the prediction parameter. After calling this function, the
        job should have a duration_estimate value in the database.

        """
        sql = text(
            """
            update job_history
            set duration_estimate = :duration_estimate
            where id = :id
        """
        )
        with self.engine.connect() as conn:
            with conn.begin() as trans:
                try:
                    conn.execute(
                        sql, {"duration_estimate": prediction, "id": id}
                    )
                except DatabaseError as e:
                    trans.rollback()
                    raise Exception(f"Database error: {e}")

    def insert_bogus_job_history(self, num_rows: int) -> None:
        """Inserts num_rows bogus rows into the database. These rows represent
        jobs that have run in the past. The function makes a feeble attempt to
        overlap some of the jobs, and adjusts their durations based on overlap
        and on the number of records that the job is processing. This function
        is helpful when testing training.
        """
        end_time = datetime.now() - timedelta(days=2)
        max_time_span = timedelta(seconds=3600 * num_rows)
        base_time = end_time - max_time_span
        query = text(
            """
            INSERT INTO job_history (job_type, start_at, end_at, record_count)
            VALUES (:job_type, :start_at, :end_at, :record_count)
        """
        )
        with self.engine.connect() as conn:
            with conn.begin() as trans:
                jobs = []
                # Generate job data
                for i in range(num_rows):
                    progress = i / num_rows
                    start_at = base_time + (end_time - base_time) * progress
                    offset_seconds = random.randint(-3600, 3600)
                    start_at += timedelta(seconds=offset_seconds)
                    record_count = random.randint(100, 10000)
                    base_duration_seconds = record_count / 5.0
                    overlapping_jobs = self.overlapping(
                        jobs,
                        start_at,
                        start_at + timedelta(seconds=base_duration_seconds),
                    )
                    overlap_count = len(overlapping_jobs) + 1
                    effective_duration = base_duration_seconds * overlap_count
                    effective_duration = max(60, effective_duration)
                    end_at = start_at + timedelta(seconds=effective_duration)
                    jobs.append(
                        {
                            "start_at": start_at,
                            "end_at": end_at,
                            "record_count": record_count,
                        }
                    )
                    try:
                        conn.execute(
                            query,
                            {
                                "job_type": self.config["job_type"],
                                "start_at": start_at,
                                "end_at": end_at,
                                "record_count": record_count,
                            },
                        )
                    except DatabaseError as e:
                        trans.rollback()
                        raise Exception(f"Database error: {e}")

    def overlapping(
        self,
        jobs: list[dict[str, Union[datetime, int]]],
        start_at: datetime,
        end_at: datetime,
    ) -> list[dict[str, Union[datetime, int]]]:
        """Given a list of jobs, this function return a subset of the list
        containing jobs that overlap with the given start_at and end_at times.

        """
        return [
            job
            for job in jobs
            if job["start_at"] <= start_at < job["end_at"]
            or job["start_at"] < end_at <= job["end_at"]
            or (start_at <= job["start_at"] and end_at >= job["end_at"])
        ]

    def insert_new_job(self, start: str, record_count: int) -> int:
        """Inserts a new job into the job_history table. The new job will not
        have null end_at and duration_estimate values. Returns the ID of the job
        that was inserted.

        Arguments:
        start: A string in the format 'YYYY-MM-DD hh:mm:ss' that indicates the
           job's start time.
        record_count: An integer that represents the number of records the job
           will process.

        """
        start_at = datetime.strptime(start, "%Y-%m-%d %H:%M:%S")
        sql = text(
            """
            INSERT INTO job_history (job_type, start_at, record_count)
            VALUES (:job_type, :start_at, :record_count)
            RETURNING id
        """
        )
        with self.engine.connect() as conn:
            with conn.begin() as trans:
                try:
                    result = conn.execute(
                        sql,
                        {
                            "job_type": self.config["job_type"],
                            "start_at": start_at,
                            "record_count": record_count,
                        },
                    )
                    job_id = result.scalar()
                    return job_id
                except DatabaseError as e:
                    trans.rollback()
                    raise Exception(f"Database error: {e}")

    def clear_job_history(self) -> None:
        """Deletes every row from the job_history table."""
        sql = text("delete from job_history")
        with self.engine.connect() as conn:
            with conn.begin() as trans:
                try:
                    conn.execute(sql)
                except DatabaseError as e:
                    trans.rollback()
                    raise Exception(f"Database error: {e}")


if __name__ == "__main__":
    options = Predictor.command_line_options()
    match options["command"]:
        case "instance-test":
            pred = Predictor(options)
            print(
                json.dumps(
                    {"config": pred.config, "db_url": pred.db_url}, indent=2
                )
            )
        case "insert-bogus-job-history":
            pred = Predictor(options)
            pred.insert_bogus_job_history(1000)
            df = pred.fetch_jobs(historical=True)
            print(f"Inserted {len(df)} rows")
        case "clear-job-history":
            pred = Predictor(options)
            pred.clear_history()
            df = pred.fetch_jobs(historical=True)
            print(f"Remaining records: {len(df)}")
        case "start-database":
            Predictor.docker_compose(
                "tests/docker-compose.yaml",
                "predictor-command-line",
                True,
                False,
            )
        case "stop-database":
            Predictor.docker_compose(
                "tests/docker-compose.yaml",
                "predictor-command-line",
                False,
                False,
            )
        case "train":
            pred = Predictor(options)
            pred.train()
            print("Training complete.")
        case "new-job":
            pred = Predictor(options)
            start = pred.config["start"]
            count = int(pred.config["record_count"])
            id = pred.insert_new_job(start, count)
            print(f"Inserted new job record with ID {id}")
        case "predict":
            pred = Predictor(options)
            result = pred.predict()
            if result:
                print(json.dumps(result, indent=2))
            else:
                print("No records needed predictions")
        case _:
            print(f"Unknown command {options['command']}")
