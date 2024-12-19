import os
import sys
from dataclasses import dataclass
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_model

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initate_model_training(self, train_array, test_array):
        try:
            logging.info("Splitting training and test input data")
            
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            # Define model
            model = MultinomialNB()
            
            logging.info("Model Training Started")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions and evaluate
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            train_accuracy = accuracy_score(y_train, y_train_pred)
            test_accuracy = accuracy_score(y_test, y_test_pred)
            
            logging.info(f"Train Accuracy: {train_accuracy}")
            logging.info(f"Test Accuracy: {test_accuracy}")
            
            logging.info("Model Training Complete")
            
            # Save the trained model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=model
            )
            
            logging.info("Model pickle file saved")
            
            # Return test accuracy
            return test_accuracy
            
        except Exception as e:
            raise CustomException(e, sys)