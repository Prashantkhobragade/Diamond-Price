import pandas as pd 
import numpy as np 
from src.logger.logging import logging
from src.exception.exeption import CustomException

import os
import sys 
from dataclasses import dataclass
from pathlib import Path

from sklearn.model_selection import train_test_split
@dataclass
class DataIngestionConfig:
    raw_data_path:str = os.path.join("artifacts","raw.csv")
    train_data_path:str = os.path.join("artifacts", "train.csv")
    test_data_path:str = os.path.join("artifacts", "test.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
    
    def initiate_data_ingestion(self):
        logging.info("Data ingestion started")
        try:
            
            data = pd.read_csv("data\Gemstone_Price_Data.csv")
            logging.info("reading data from data dir")
            
            os.makedirs(os.path.dirname(os.path.join(self.ingestion_config.raw_data_path)), exist_ok = True)
            data.to_csv(self.ingestion_config.raw_data_path, index=False)
            logging.info("saved raw data in artifacts")
            
            train_data, test_data = train_test_split(data, test_size=0.25, random_state=42)
            logging.info("train and test completed")
            
            train_data.to_csv(self.ingestion_config.train_data_path, index=False)
            test_data.to_csv(self.ingestion_config.test_data_path, index=False)
            logging.info("data ingestion completed")
            
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
            
            
        except Exception as e:
            raise CustomException(e, sys)
        
        
if __name__ == "__main__":
    obj = DataIngestion()
    obj.initiate_data_ingestion()


