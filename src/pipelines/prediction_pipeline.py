import os
import sys
import pandas as pd
import numpy as np
from src.exception import CustomException
from src.utils import load_object
from src.logger import logging
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string

class PredictPipeline:
    def __init__(self):
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

    def predict(self, text):
        try:
            # Load the preprocessor and model
            preprocessor_path = os.path.join('artifacts', 'vectorizer.pkl')
            model_path = os.path.join('artifacts', 'model.pkl')
            
            logging.info("Loading preprocessor and model")
            preprocessor = load_object(file_path=preprocessor_path)
            model = load_object(file_path=model_path)
            
            # Preprocess the text using the same transformation as training
            logging.info("Preprocessing text")
            processed_text = self.transform_text(text)
            logging.info(f"Processed text: {processed_text}")
            
            # Transform the processed text using vectorizer
            logging.info("Vectorizing text")
            transformed_text = preprocessor.transform([processed_text])
            logging.info(f"Vectorized shape: {transformed_text.shape}")
            
            # Make prediction
            logging.info("Making prediction")
            prediction = model.predict(transformed_text)
            
            return prediction[0]
            
        except Exception as e:
            logging.error(f"Error in prediction pipeline: {str(e)}")
            raise CustomException(e, sys)

class CustomData:
    def __init__(self, text: str):
        self.text = text

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                "text": [self.text]
            }
            
            df = pd.DataFrame(custom_data_input_dict)
            logging.info('Dataframe Gathered')
            return df
            
        except Exception as e:
            raise CustomException(e, sys)

def test_pipeline():
    # Test cases
    test_messages = [
        "WINNER!! As a valued network customer you have been selected to receivea £900 prize reward! To claim call 09061701461. Claim code KL341. Valid 12 hours only.",
        "Hi, how are you doing? Want to meet for coffee?",
        "URGENT! You have won a 1 week FREE membership in our £100,000 Prize Jackpot! Txt the word: CLAIM to No: 81010 T&C www.dbuk.net LCCLTD POBOX 4403LDNW1A7RW18",
        "Hey, I'll be there in 10 minutes."
    ]
    
    pred_pipeline = PredictPipeline()
    
    logging.info("Starting test predictions")
    
    for idx, text in enumerate(test_messages, 1):
        try:
            prediction = pred_pipeline.predict(text)
            processed_text = pred_pipeline.transform_text(text)
            print(f"\nTest {idx}:")
            print(f"Original: {text}")
            print(f"Processed: {processed_text}")
            print(f"Prediction: {'spam' if prediction == 1 else 'ham'}")
        except Exception as e:
            print(f"Error occurred for test {idx}: {e}")

if __name__ == "__main__":
    test_pipeline()