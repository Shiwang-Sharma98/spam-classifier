import os
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

if __name__=='__main__':
    obj = DataIngestion() # we ahve to initiate the data Ingestion
    train_data_path,test_data_path = obj.initiate_data_ingestion() # initiating train and test data path
    print(train_data_path,test_data_path)
    data_transf = DataTransformation()
    train_arr,test_arr,obj_path=data_transf.initiate_data_transformation(train_data_path,test_data_path)

    model_trainer = ModelTrainer()

    model_trainer.initate_model_training(train_arr,test_arr)