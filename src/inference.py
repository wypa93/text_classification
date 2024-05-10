from pyspark.sql.functions import struct, length, udf, col, round
from pyspark.sql.types import (
    StringType,
    FloatType,
    StructType,
    StructField,
    IntegerType,
)
from mlflow.store.artifact.models_artifact_repo import ModelsArtifactRepository
import mlflow.transformers
import pyspark.sql.functions as F
from pyspark.sql import SparkSession
from pyspark.sql import DataFrame
import mlflow
import os
import logging
import datetime

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logging.getLogger().setLevel(logging.WARNING)


class ModelRunner:
    def __init__(
        self,
        model_uri,
        input_table_name,
        output_table_name,
        text_col,
        date_col,
        timedelta=7,
    ):
        """
        Initializes ModelRunner with model and table information
        Args:
            model_uri (str): URI of the model
            input_table_name (str): name of the input table
            output_table_name (str): name of the output table
            text_col (str): name of text column used as inference
            date_col (str): name of date column used to define start date
            timedelta (int): # days back to run the inference on, 7 days as default
        """
        self.model_uri = model_uri
        self.input_table_name = input_table_name
        self.output_table_name = output_table_name
        self.text_col = text_col
        self.date_col = date_col
        self.spark = None
        self.model_artifact = None
        self.timedelta = timedelta

    def load_spark_session(self):
        """
        Loads or creates a SparkSession if not already loaded
        """
        if not self.spark:
            self.spark = SparkSession.builder.appName(
                "safety_classification_inference"
            ).getOrCreate()

    def predict_wrapper(self):
        """
        Loads the MLflow model artifact and returns predict function
        Returns:
            predict (function): The predict function that generates inference on text data and returns
                               the label and score.
        """
        model_artifact = mlflow.transformers.load_model(self.model_uri)

        def predict(text: str) -> tuple:
            """
            Generates inference on text data and returns label and score
            Args:
                text (str): text narrative to generate prediction
            Returns:
                label (int): predicted class label
                score (float): label probability
            """
            try:
                if not isinstance(text, str):
                    raise ValueError("Input text must be a string with non-zero length")

                prediction = model_artifact(text, truncation=True)[0]
                label = int(prediction["label"].replace("LABEL_", ""))
                score = prediction["score"]
            except (KeyError, ValueError) as e:
                label = 99
                score = 0.00
            return label, score

        return predict

    def run(self):
        """
        Runs the model inference on the input table and stores predictions
        """
        self.load_spark_session()
        predict = self.predict_wrapper()  # get prediction function
        predict_udf = udf(
            predict,
            StructType(
                [  # define as udf
                    StructField("label", IntegerType(), False),
                    StructField("score", FloatType(), False),
                ]
            ),
        )
        table = self.spark.table(self.input_table_name)  # load input data table
        table = table.filter(
            col(self.date_col)
            >= (
                datetime.datetime.today().date()
                - datetime.timedelta(days=self.timedelta)
            )
        )  # filter rows in date range
        # update table with predictions, creation date, and model_uri
        table = (
            table.withColumn("failure_type_prediction", predict_udf(col("Narrative")))
            .withColumn("label", col("failure_type_prediction.label"))
            .withColumn("score", col("failure_type_prediction.score"))
            .drop("failure_type_prediction")
            .withColumn("score", F.round(col("score"), 3))
            .withColumn("rec_create_date", F.current_date())
            .withColumn("rec_create_by", F.lit(self.model_uri))
        )
        table.write.mode("append").saveAsTable(
            self.output_table_name
        )  # append predictions to output table


if __name__ == "__main__":

    # start logging
    logging.info("initating inference run")

    # load params from yaml config file
    params = {
        "model_uri": f"models:/roberta_classifier/16",
        "input_table_name": "text_analytics.im_03_18",
        "output_table_name": "text_analytics.mlops_inference",
        "text_col": "Narrative",
        "date_col": "rec_create_date",
        "timedelta": 7,
    }

    # set up environment and install requirements
    logging.info("installing dependencies")
    local_path = ModelsArtifactRepository(params["model_uri"]).download_artifacts(
        ""
    )  # download model from remote registry
    requirements_path = os.path.join(local_path, "requirements.txt")
    if not os.path.exists(requirements_path):
        dbutils.fs.put("file:" + requirements_path, "", True)
    %pip install -r $requirements_path

    # pass parameters and run model inference
    model_runner = ModelRunner(**params)
    logging.info(
        f"Starting the inference process on table: {model_runner.input_table_name}\t column: {model_runner.text_col}\t timedelta: {model_runner.timedelta}"
    )
    model_runner.run()
    logging.info(
        f"Successfully completed inference run and stored predictions to {model_runner.output_table_name}"
    )