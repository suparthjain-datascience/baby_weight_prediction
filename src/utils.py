import sys
import os

import pandas as pd
import numpy as np
from src.exception import CustomException

def cleaning_data_process(data):
    try:
        cleaned_data = data.drop('ID', axis=1)

        numerical_cols = ['GAINED', 'VISITS', 'MAGE', 'FEDUC', 'MEDUC', 'TOTALP', 'BDEAD', 'TERMS', 'WEEKS', 'CIGNUM',
                          'DRINKNUM', 'BWEIGHT']

        all_columns = cleaned_data.columns
        categorical_cols = [col for col in all_columns if col not in numerical_cols]

        columns_to_check = numerical_cols + categorical_cols
        cleaned_data = cleaned_data.dropna(subset=columns_to_check)

        mapping = {
            'C': 1,  # Cubans
            'M': 2,  # Mexicans
            'N': 0,  # No
            'O': 3,  # Colombians
            'P': 4,  # Peruvians
            'S': 5,  # Salvadorans
            'U': 6  # Guatemalans
        }

        cleaned_data['HISPMOM'] = cleaned_data['HISPMOM'].replace(mapping)
        cleaned_data['HISPDAD'] = cleaned_data['HISPDAD'].replace(mapping)

        return cleaned_data
    except Exception as e:
        raise CustomException(e,sys)