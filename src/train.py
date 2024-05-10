!pip install azure-storage-blob
!pip install optimum
!pip install torch
dbutils.library.restartPython()

# import general purpose libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import textwrap
import pyodbc
from azure.storage.blob import BlobServiceClient
import azure.core
import os
import mlflow
import re
import nltk
import random
import json
import warnings
from tqdm import tqdm
warnings.filterwarnings("ignore")

# import AIML libraries
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AdamW, RobertaForSequenceClassification, RobertaTokenizer, RobertaConfig
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
import torch.optim as optim
from torch.nn import CrossEntropyLoss
import torch.optim.lr_scheduler as lr_scheduler
import os

import mlflow
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup, pipeline
import torch.nn.functional as F
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm

# import pyspark related libraries
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, TimestampType, BinaryType, BooleanType, DateType, DecimalType, DoubleType, FloatType, ByteType, LongType, ShortType
import pyspark.sql.functions as f
from pyspark.sql.types import *
from pyspark.sql.functions import *

# launch spark session
spark = SparkSession.builder.getOrCreate()


class CustomDataset(Dataset):
    """
    A class used to create a custom dataset for tokenized data.

    Args:
        encodings (dict): The tokenized encodings.
        labels (np.array): The corresponding labels for the encodings.
    """
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = torch.tensor(labels, dtype=torch.long)  # Change dtype to long for multi-class

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)


class DataProcessor:
    """
    Class for data preprocessing that loads data from a Spark table, splits it into train, validation, and test sets, and tokenizes the data.

    Methods:
        init(self, train_tbl_params, tokenizer_params, base_model='roberta-base', max_length=256, train_size=0.5, random_state=42): Initializes the DataProcessor with the given parameters.

        load_data(self): Loads training data for X, y from a Spark table.

        split_data(self, X, y): Splits the loaded data into training, validation, and testing sets.

        tokenize_data(self, X_train, X_val, X_test, y_train, y_val, y_test): Tokenizes the input data.

        run_processor(self): Runs the load_data, split_data, and tokenize_data methods in sequence. 
    """
    def __init__(self, train_tbl_params, tokenizer_params, base_model='roberta-base', max_length=256, train_size=0.5, random_state=42):

        """
        Initializes the DataProcessor with the necessary parameters.

        Args:
            train_tbl_params : dict Parameters for the Spark table. Includes table name and column names. 
            tokenizer_params : dict Parameters for the tokenizer. 
            base_model : str, optional The base model for the tokenizer. Default is 'roberta-base'. 
            max_length : int, optional The maximum length for the tokenizer. Default is 256. 
            train_size : float, optional The size of the training set. Default is 0.5. 
            random_state : int, optional The random state for data splitting. Default is 42. 
            tokenizer : transformers.PreTrainedTokenizer The tokenizer for the input data.
        """
        self.train_tbl_params = train_tbl_params
        self.base_model = base_model
        self.max_length = max_length
        self.train_size = train_size
        self.random_state = random_state
        self.tokenizer = RobertaTokenizer.from_pretrained(self.base_model)
        self.tokenizer_params = tokenizer_params

    def load_data(self):
        """
        Loads training data for X,y from a Spark table.

        Returns:
            X (list) : Input features from the spark table.
            y (np.array) : Target variable from the spark table.
            num_classes (int) : Number of unique classes in the target variable.
        """
        train_tbl_name = self.train_tbl_params['train_tbl_name']
        X_col = self.train_tbl_params['X_col']
        y_col = self.train_tbl_params['y_col']
        data = spark.table(train_tbl_name).select(col(X_col),col(y_col)).toPandas()
        X = data[X_col].tolist()
        y = np.array(data[y_col].tolist())
        num_classes = len(np.unique(y))
        return X, y, num_classes
        
    def split_data(self, X,y):
        """
        Splits the loaded data into training, validation, and testing sets.

        Args
            X (list): Input features.
            y (np.array): Target variable.

        Returns
            X_train (list): Input features for training.
            X_val (list): Input features for validation.
            X_test (list): Input features for testing.
            y_train (np.array): Target variable for training.
            y_val (np.array): Target variable for validation.
            y_test (np.array): Target variable for testing.
        """
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=self.train_size, random_state=self.random_state)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=self.random_state)
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def tokenize_data(self, X_train, X_val, X_test, y_train, y_val, y_test):
        """
        Tokenizes the input data.

        Args:
            X_train (list): Training input features.
            X_val (list): Validation input features.
            X_test (list): Test input features.
            y_train (np.array): Training target variable.
            y_val (np.array): Validation target variable.
            y_test (np.array): Test target variable.

        Returns:
            train_dataset (CustomDataset): Tokenized training data.
            val_dataset (CustomDataset): Tokenized validation data.
            test_dataset (CustomDataset): Tokenized test data.
        """
        train_encodings = self.tokenizer(X_train, **self.tokenizer_params)
        val_encodings = self.tokenizer(X_val, **self.tokenizer_params)
        test_encodings = self.tokenizer(X_test, **self.tokenizer_params)
        
        train_dataset = CustomDataset(train_encodings, y_train)
        val_dataset = CustomDataset(val_encodings, y_val)
        test_dataset = CustomDataset(test_encodings, y_test)
        
        return train_dataset, val_dataset, test_dataset

    def run_processor(self):
        """
        Runs sequences to obtain train, test, and validation data.

        Returns:
            train_dataset (CustomDataset): Tokenized training data.
            val_dataset (CustomDataset): Tokenized validation data.
            test_dataset (CustomDataset): Tokenized test data.
            num_classes (int): Number of unique classes in the target variable.
        """
        X, y, num_classes = self.load_data()
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(X, y)
        train_dataset, val_dataset, test_dataset = self.tokenize_data(X_train, X_val, X_test, y_train, y_val, y_test)
        return train_dataset, val_dataset, test_dataset, num_classes


class HazardousFailureTypeClassifier:
    """
    Trains and validates multi-class classifiers to identify safety hazards from warranty claims and complaints.

    Methods: 
        train(epoch): Trains the model for one epoch and returns the running loss. 
        validate(epoch): Validates the model and returns the predictions and true labels. 
        test(): Tests the model and returns the predictions and true labels. 
        log_training_params(): Logs the training and model parameters.
        log_metrics(epoch, predictions, true_labels, mode, loss): Logs the metrics for one epoch. 
        register_model(): Registers the model on MLflow. 
        run_training(): Runs the training and validation process for num_epochs times
    """
    def __init__(self, model, tokenizer, optimizer, scheduler, criterion, device, train_dataset, val_dataset, test_dataset, run_name='run_name', batch_size=8, num_epochs=20, early_stopping_patience=3):
        """
        Initializes the HazardousFailureTypeClassifier with the necessary parameters.

        Args:
            model: The model to be trained.
            tokenizer: The tokenizer to be used for preprocessing the data.
            optimizer: torch optimizer including weight decay
            scheduler: learning rate scheduler used for regularization
            criterion: torch loss function
            device: The device to train the model on (cpu or gpu).
            train_dataset: The dataset to be used for training.
            val_dataset: The dataset to be used for validation.
            test_dataset: The dataset to be used for testing.
            run_name: The name of the run (default is current date and time).
            batch_size: The size of the batch for training (default is 8).
            num_epochs: The number of epochs for training (default is 20).
            early_stopping_patience: The number of epochs with no improvement after which training will be stopped (default is 3).
        """
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.device = device
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.run_name = run_name
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.early_stopping_patience = early_stopping_patience
        self.best_val_loss = float('inf')
        self.no_improvement_counter = 0

    def train(self, epoch):
        """
        Trains the model for one epoch and returns the running loss.
        Args:
            epoch: The current epoch number.

        Returns:
            running_loss: The total loss of the model at the current epoch.
        """
        self.model.train()
        train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        running_loss = 0.0
        predictions = []
        true_labels = []
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f'Epoch {epoch + 1}/{self.num_epochs}')):
            self.optimizer.zero_grad()
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            outputs = self.model(input_ids, attention_mask=attention_mask)
            loss = self.criterion(outputs.logits, labels)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
            predictions.extend(torch.argmax(F.softmax(outputs.logits, dim=1), dim=1).cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
        # calculate training loss
        train_loss = running_loss / len(train_loader)
        return predictions, true_labels, train_loss

    def validate(self, epoch):
        """
        Validates the model and returns the predictions and true labels.

        Args:
            epoch: The current epoch number.

        Returns:
            predictions: The model's predictions on the validation dataset.
            true_labels: The true labels of the validation dataset.
        """
        self.model.eval()
        val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)
        predictions = []
        true_labels = []
        running_loss = 0.0
        with torch.no_grad():
            for val_batch in tqdm(val_loader, desc=f'Validation Epoch {epoch + 1}'):
                val_input_ids = val_batch['input_ids'].to(self.device)
                val_attention_mask = val_batch['attention_mask'].to(self.device)
                val_labels = val_batch['labels'].to(self.device)
                val_outputs = self.model(val_input_ids, attention_mask=val_attention_mask)
                val_logits = val_outputs.logits
                loss = self.criterion(val_logits, val_labels)
                running_loss += loss.item()
                predictions.extend(torch.argmax(F.softmax(val_logits, dim=1), dim=1).cpu().numpy())
                true_labels.extend(val_labels.cpu().numpy())
        # calcute validation loss
        val_loss = running_loss / len(val_loader)
        return predictions, true_labels, val_loss

    def test(self):
        """
        Tests the model and returns the predictions and true labels.

        Returns:
            predictions: The model's predictions on the test dataset.
            true_labels: The true labels of the test dataset.
        """
        self.model.eval()
        test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)
        predictions = []
        true_labels = []
        with torch.no_grad():
            for test_batch in tqdm(test_loader, desc='Test Epoch'):
                test_input_ids = test_batch['input_ids'].to(self.device)
                test_attention_mask = test_batch['attention_mask'].to(self.device)
                test_labels = test_batch['labels'].to(self.device)
                test_outputs = self.model(test_input_ids, attention_mask=test_attention_mask,labels=test_labels)
                test_logits = test_outputs.logits
                self.scheduler.step(test_outputs.loss)
                predictions.extend(torch.argmax(F.softmax(test_logits, dim=1), dim=1).cpu().numpy())
                true_labels.extend(test_labels.cpu().numpy())
        return predictions, true_labels

    def log_training_params(self):
        """
        Logs the training and model parameters
        """
        mlflow.pytorch.log_model(self.model, "model")
        mlflow.log_param("tokenizer_type", type(self.tokenizer).__name__)
        mlflow.log_param("tokenizer_vocab_size", self.tokenizer.vocab_size)
        mlflow.log_param("batch_size", self.batch_size)
        mlflow.log_param("optimizer", self.optimizer)
        mlflow.log_param("loss_fn", self.criterion)
        mlflow.log_param("scheduler", self.scheduler)
        mlflow.log_param("num_epochs", self.num_epochs)
        mlflow.log_param("early_stopping_patience", self.early_stopping_patience)
        mlflow.log_param("batch_size", self.batch_size)

    def log_metrics(self, epoch, true_labels, predictions, loss, mode):
        """
        Logs the metrics for one epoch.

        Args:
            epoch: current training epoch
            true_labels: correct lables reflecting ground truth.
            predictions: model predictions.
            loss: train or validation loss.
            mode: set as 'train', 'val', or 'test'.
        """
        accuracy = accuracy_score(true_labels, predictions)
        precision = precision_score(true_labels, predictions, average='micro')
        recall = recall_score(true_labels, predictions, average='micro')
        f1 = f1_score(true_labels, predictions, average='micro')
        mlflow.log_metric(f"{mode}_accuracy", accuracy, step=epoch)
        mlflow.log_metric(f"{mode}_precision", precision, step=epoch)
        mlflow.log_metric(f"{mode}_recall", recall, step=epoch)
        mlflow.log_metric(f"{mode}_f1_score", f1, step=epoch)
        if mode in ('train','val'):
            mlflow.log_metric(f"{mode}_loss", loss, step=epoch)
        elif mode == 'test':
            clf_report = classification_report(true_labels, predictions, digits=2)
            mlflow.log_artifacts(clf_report,'clf_report.json')
        else:
            print('you must select a correct mode to log metrics. Use train, val, or test')
            pass

    def register_model(self):
        """
        Registers the model on MLflow.
        """
        self.model.tie_weights()
        text_classification_pipeline = pipeline(
            task="text-classification",
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device,
            max_length=256,
            padding='max_length',
            truncation=True,
        )
        mlflow.transformers.log_model(
            transformers_model=text_classification_pipeline,
            registered_model_name='roberta_classifier',
            artifact_path="classifier_pipeline",
            input_example="hello world I am a sample safety hazard narrative",
        )

    def run_training(self):
        """
        Runs the training and validtion process for num_epochs times.
        """
        with mlflow.start_run(run_name=self.run_name) as run:
            # log training parameters
            self.log_training_params()
            # train and validate model for n epochs
            for epoch in range(self.num_epochs):
                train_predictions, train_labels, train_loss = self.train(epoch)
                self.log_metrics(epoch, train_labels, train_predictions,mode='train', loss=train_loss)
                predictions, true_labels, val_loss = self.validate(epoch)
                self.log_metrics(epoch, true_labels, predictions,mode='val', loss=val_loss)
                # apply early stopping if performance does not improve after n epochs
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.no_improvement_counter = 0
                else:
                    self.no_improvement_counter += 1
                    if self.no_improvement_counter >= self.early_stopping_patience:
                        print(f'\nEarly stopping after no improvement for {self.early_stopping_patience} epochs.')
                        mlflow.log_param("termination_reason", "no_improvement")
                        break
            # get test predictions and compute metrics
            test_predictions, test_labels = self.test()
            self.log_metrics(epoch, test_labels, test_predictions,mode='test', loss=None)
            # register model as pipeline
            self.register_model()

if __name__ == '__main__':
    # fetch hyper parameters
    tokenizer_params = {
        'max_length': 256,
        'padding': 'max_length',
        'return_tensors': 'pt',
        'truncation': True
    }
    train_tbl_params = {
        'train_tbl_name':'hive_metastore.text_analytics.tbl_train_cse_l3',
        'X_col':'text',
        'y_col':'target_class'
    }
    training_params = {
        'run_name':f'Hazardous_Failure_Type_Roberta_CSE_Overall:{datetime.now().strftime("%Y%m%d-%H%M%S")}',
        'batch_size':16, 
        'num_epochs':20, 
        'early_stopping_patience':2,
        'train_size':0.5,
        'random_state':42,
    }
    model_params = {
        'base_model':'roberta-base',
        'hidden_dropout_prob':0.2,
        'attention_probs_dropout_prob':0.2,
        'classifier_dropout':0.5,
    }
    # pre-process training, validation, and testing data
    preprocessor = DataProcessor(
          train_tbl_params,
          tokenizer_params,
          base_model='roberta-base',
          max_length=tokenizer_params['max_length'], 
          train_size=training_params['train_size'], 
          random_state=training_params['random_state'],
          )
    train_dataset, val_dataset, test_dataset, num_classes = preprocessor.run_processor()
    # configer model parameters
    configuration = RobertaConfig.from_pretrained(model_params['base_model'])
    configuration.hidden_dropout_prob = model_params['hidden_dropout_prob']
    configuration.attention_probs_dropout_prob = model_params['attention_probs_dropout_prob']
    configuration.classifier_dropout = model_params['classifier_dropout']
    configuration.num_labels = num_classes
    # instantiate training artifacts like model, tokenizer, scheduler, optimizer, criterion
    model = RobertaForSequenceClassification.from_pretrained(model_params['base_model'],config=configuration)
    tokenizer = RobertaTokenizer.from_pretrained(model_params['base_model'])
    optimizer = AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=2, min_lr=1e-5, verbose=True)
    criterion = CrossEntropyLoss()
    # set to device if GPU is available
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    # instantiate training object
    classifier = HazardousFailureTypeClassifier(model, tokenizer, optimizer, scheduler, criterion, device, train_dataset, val_dataset, test_dataset,run_name=training_params['run_name'], batch_size=training_params['batch_size'], num_epochs=training_params['num_epochs'], early_stopping_patience=training_params['early_stopping_patience'])
    # run training process
    classifier.run_training()