import os
import sys
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = {
            'preprocessing_obj_file_path': os.path.join('artifacts', 'vectorizer.pkl')
        }
        nltk.download('punkt')
        nltk.download('stopwords')
        self.ps = PorterStemmer()

    def transform_text(self, text):
        try:
            text = text.lower()
            text = nltk.word_tokenize(text)
            
            y = []
            for i in text:
                if i.isalnum():
                    y.append(i)
            
            text = y[:]
            y.clear()
            
            for i in text:
                if i not in stopwords.words('english') and i not in string.punctuation:
                    y.append(i)
            
            text = y[:]
            y.clear()
            
            for i in text:
                y.append(self.ps.stem(i))
            
            return " ".join(y)
        
        except Exception as e:
            raise CustomException(e, sys)

    def get_data_transformer_object(self):
        try:
            # Define TFIDF Vectorizer
            tfidf = TfidfVectorizer(max_features=3000)
            
            return tfidf
        
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logging.info("Read train and test data completed")
            logging.info("Obtaining preprocessing object")
            
            preprocessing_obj = self.get_data_transformer_object()
            
            target_column_name = 'v1'  # Assuming 'label' is your target column
            text_column = 'v2'  # Assuming 'text' is your feature column
            
            # Apply text transformation
            train_df[text_column] = train_df[text_column].apply(self.transform_text)
            test_df[text_column] = test_df[text_column].apply(self.transform_text)
            
            # Splitting into features and target
            input_feature_train_df = train_df[text_column]
            target_feature_train_df = train_df[target_column_name]
            
            input_feature_test_df = test_df[text_column]
            target_feature_test_df = test_df[target_column_name]
            
            # Transform using TFIDF
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)
            
            train_arr = np.column_stack((input_feature_train_arr.toarray(), target_feature_train_df))
            test_arr = np.column_stack((input_feature_test_arr.toarray(), target_feature_test_df))
            
            logging.info("Applying preprocessing object on training and testing datasets.")
            
            save_object(
                file_path=self.data_transformation_config['preprocessing_obj_file_path'],
                obj=preprocessing_obj
            )
            
            logging.info("Preprocessing pickle file saved")
            
            return (
                train_arr,
                test_arr,
                self.data_transformation_config['preprocessing_obj_file_path']
            )
            
        except Exception as e:
            raise CustomException(e, sys)