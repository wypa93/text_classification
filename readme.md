## Project: Multi-label Text Classification

This project consists of two Python scripts for training, validating, and deploying a text classification model to identify safety hazards from warranty claims and complaints.

**train.py**

This script defines a class `HazardousFailureTypeClassifier` for training and validating the model. Key functionalities include:

* Data preprocessing using a custom `DataProcessor` class.
* Model training and validation within the `HazardousFailureTypeClassifier` class.
* Logging training parameters and metrics.
* Registering the trained model as a pipeline on MLflow.

**inference.py**

This script defines a class `ModelRunner` for loading a pre-trained model and performing inference on new data. Key functionalities include:

* Loading a pre-trained model artifact from MLflow.
* Defining a User Defined Function (UDF) for applying model predictions to new data using Spark SQL.
* Updating the input data table with predictions and saving the results to a separate output table.

**Overall Workflow**

1. **Data Preprocessing (train.py):**
   - Load training data from a Spark table.
   - Split data into training, validation, and testing sets.
   - Tokenize text data using a pre-trained tokenizer.

2. **Model Training and Validation (train.py):**
   - Initialize model, tokenizer, optimizer, scheduler, and loss function.
   - Train the model for a specified number of epochs.
   - Validate the model performance.
   - Log training metrics and parameters.

3. **Model Deployment (inference.py):**
   - Load the pre-trained model from MLflow.
   - Use a Spark UDF to apply model predictions to new data.
   - Update the input data table with predictions and save the results.

**Getting Started**

* Refer to the individual scripts (`train.py` and `inference.py`) for detailed implementation and configuration.
* This project utilizes libraries like `transformers`, `pandas`, `torch`, `pyspark.sql`, and `mlflow`. Ensure these are installed in your environment.

**Additional Notes**

* The specific pre-trained model used (e.g., Roberta) and number of classes are configurable within the code.
* This is a high-level overview, and the scripts contain detailed comments for further understanding.
