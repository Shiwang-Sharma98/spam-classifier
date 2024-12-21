import os
import sys
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
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
            logging.error(f"Error in text preprocessing: {str(e)}")
            raise CustomException(e, sys)

    def get_data_transformer_object(self):
        try:
            tfidf = TfidfVectorizer(
                max_features=None,
                ngram_range=(1, 3),
                min_df=2,
                max_df=0.95,
                strip_accents='unicode',
                analyzer='word',
                token_pattern=r'\w{1,}',
                stop_words='english',
                sublinear_tf=True
            )
            
            return tfidf
        
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            # Read data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            # Log initial data information
            logging.info(f"Training data shape: {train_df.shape}")
            logging.info(f"Test data shape: {test_df.shape}")
            logging.info("\nLabel distribution in training data:")
            logging.info(train_df['v1'].value_counts())
            
            # Ensure labels are properly encoded
            label_map = {'ham': 0, 'spam': 1}
            train_df['v1'] = train_df['v1'].map(label_map)
            test_df['v1'] = test_df['v1'].map(label_map)
            
            logging.info("\nEncoded label distribution in training data:")
            logging.info(train_df['v1'].value_counts())
            
            # Log some sample transformations
            logging.info("\nSample text transformations:")
            for idx, row in train_df.head(3).iterrows():
                original = row['v2']
                transformed = self.transform_text(original)
                logging.info(f"\nOriginal: {original}")
                logging.info(f"Transformed: {transformed}")
            
            # Apply text transformation
            train_df['v2'] = train_df['v2'].apply(self.transform_text)
            test_df['v2'] = test_df['v2'].apply(self.transform_text)
            
            # Get features and target
            input_feature_train_df = train_df['v2']
            target_feature_train_df = train_df['v1']
            
            input_feature_test_df = test_df['v2']
            target_feature_test_df = test_df['v1']
            
            # Get preprocessor
            preprocessing_obj = self.get_data_transformer_object()
            
            # Transform using TFIDF
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)
            
            # Log vectorizer information
            logging.info(f"\nVectorizer vocabulary size: {len(preprocessing_obj.vocabulary_)}")
            logging.info(f"Training features shape: {input_feature_train_arr.shape}")
            logging.info(f"Testing features shape: {input_feature_test_arr.shape}")
            
            # Check sparsity
            train_sparsity = 1.0 - (input_feature_train_arr.nnz / 
                                   float(input_feature_train_arr.shape[0] * input_feature_train_arr.shape[1]))
            logging.info(f"Training data sparsity: {train_sparsity:.4f}")
            
            # Create final arrays
            train_arr = np.column_stack((input_feature_train_arr.toarray(), target_feature_train_df))
            test_arr = np.column_stack((input_feature_test_arr.toarray(), target_feature_test_df))
            
            # Save preprocessor
            save_object(
                file_path=self.data_transformation_config['preprocessing_obj_file_path'],
                obj=preprocessing_obj
            )
            
            logging.info("Preprocessing object saved")
            
            return (
                train_arr,
                test_arr,
                self.data_transformation_config['preprocessing_obj_file_path']
            )
            
        except Exception as e:
            raise CustomException(e, sys)