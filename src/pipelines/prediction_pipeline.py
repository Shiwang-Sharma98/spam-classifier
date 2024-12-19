import os
import sys
import pandas as pd
import numpy as np
from src.exception import CustomException
from src.utils import load_object
from src.logger import logging

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, text):
        try:
            # Load the preprocessor and model
            preprocessor_path = os.path.join('artifacts', 'vectorizer.pkl')
            model_path = os.path.join('artifacts', 'model.pkl')
            
            preprocessor = load_object(file_path=preprocessor_path)
            model = load_object(file_path=model_path)
            
            # Transform the text using vectorizer
            transformed_text = preprocessor.transform([text])
            
            # Make prediction
            prediction = model.predict(transformed_text)
            
            return prediction[0]
            
        except Exception as e:
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

# Example usage
if __name__ == "__main__":
    # Test with a sample message
    text = "WINNER!! As a valued network customer you have been selected to receivea £900 prize reward! To claim call 09061701461. Claim code KL341. Valid 12 hours only."
    
    pred_pipeline = PredictPipeline()
    custom_data = CustomData(text)
    
    try:
        prediction = pred_pipeline.predict(text)
        print(f"Prediction: {'spam' if prediction == 1 else 'ham'}")
    except Exception as e:
        print(f"Error occurred: {e}")
        