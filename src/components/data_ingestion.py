import os
import sys
import pandas as pd
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging
from src.utils import cleaning_data_process

from sklearn.model_selection import train_test_split

@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts','train.csv')
    test_data_path: str=os.path.join('artifacts','test.csv')
    raw_data_path: str=os.path.join('artifacts','data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        try:
            df = pd.read_csv('data/baby_data.csv')
            logging.info('Reading dataset successfully.')

            cleaned_data = cleaning_data_process(data = df)

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            cleaned_data.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info('Initiating train test split')
            train_set, test_set = train_test_split(cleaned_data, test_size=0.2, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info('Ingestion process is successfull')

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )
        except Exception as e:
            raise CustomException(e,sys)

if __name__ == '__main__':
    obj = DataIngestion()
    obj.initiate_data_ingestion()
