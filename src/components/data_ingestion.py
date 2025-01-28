import os
import sys  
import pandas as pd
from src.logger import logging
from src.exception import CustomException
from sklearn.model_selection import train_test_split
from dataclasses import dataclass


## initializing the data ingestion configuration
## output of data ingestion is training and test data
@dataclass
class DataIngestionconfig:
    train_data_path = os.path.join('artifacts','train.csv') # the path where we want to save train data
    test_data_path = os.path.join('artifacts','test.csv')
    raw_data_path = os.path.join('artifacts','raw.csv') # gemstone file

## create data ingestion class

class DataIngestion:
    def __init__(self): # this is constructor
        self.ingestion_config = DataIngestionconfig()

    def initiate_data_ingestion(self):
        logging.info('data ingestion method starts')

        try:
           
            df = pd.read_csv(os.path.join('notebooks/data','spam.csv'))
            logging.info('dataset read as pandas dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True) ## by default folder location we are given
            df.to_csv(self.ingestion_config.raw_data_path,index=False) # entire dataFrame being converted to raw data

            logging.info('raw data is been created')

            train_set,test_set = train_test_split(df,test_size=0.20,random_state=2)

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True) # header contains info of all the  coulumns
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True) # convertion of train and test to csv


            logging.info('ingestion of data is completed')

            return(

                self.ingestion_config.train_data_path, 
                self.ingestion_config.test_data_path # we are returning the train path and the test path 
            )


        
        except Exception as e: # we are calling custom exception here
            logging.info('error occured in data ingestion stage')
            raise CustomException(e,sys) 