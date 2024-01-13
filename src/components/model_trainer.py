import pandas as pd 
import numpy as np 
from src.logger.logging import logging
from src.exception.exeption import CustomException

import os
import sys 
from dataclasses import dataclass
from pathlib import Path

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from src.utils.utils import save_object, evaluate_model

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def initiate_model_training(self, train_array, test_array):
        try:
            logging.info("splitting dependent and independent variable from train and test")
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            
            models = {
                'LinearRegression': LinearRegression(),
                'Lasso': Lasso(),
                'Ridge': Ridge(),
                'Elasticnet': ElasticNet()
            }
            
            model_report:dict = evaluate_model(X_train,y_train,X_test,y_test,models)
            print(model_report)
            print('\n==============================================================================\n')
            logging.info(f'Model report: {model_report}')
            
            # TO get the best model score from dict
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            
            best_model = models[best_model_name]
            
            print(f'Best Model Found, Model Name: {best_model_name}, R2 Score: {best_model_score}')
            print('\n==================================================================================\n')
            logging.info(f'Best Model Found, Model Name: {best_model_name}, R2 Score: {best_model_score}')
            
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )
            
        except Exception as e:
            logging.info("Exception occure in Model training")
            raise CustomException(e, sys)


