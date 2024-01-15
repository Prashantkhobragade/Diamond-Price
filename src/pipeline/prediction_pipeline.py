import os
import sys

import pandas as pd
from src.logger.logging import logging
from src.exception.exeption import CustomException

from src.utils.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass
    
    def predict(self, features):
        try:
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
            model_path = os.path.join("artifacts", "model.pkl")
            
            preprocessor = load_object(preprocessor_path)
            model = load_object(model_path)
            
            scaled_feat = preprocessor.transform(features)
            pred = model.predict(scaled_feat)
            
            return pred
        
        except Exception as e:
            raise CustomException(e, sys)
        
class CustomData:
    def __init__(self):
        pass
    
    def get_data_as_dataframe(self):
        pass