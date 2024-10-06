import sys
import os
import pandas as pd

from src.exception import CustomException
from src.logger import logging
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = os.path.join('artifacts','model.pkl')
            preprocessor_path = os.path.join('artifacts','preprocessor.pkl')

            logging.info('Before Loading')

            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)

            logging.info('After Loading')

            scaled_data = preprocessor.transform(features)

            predictions = model.predict(scaled_data)

            return predictions
        except Exception as e:
            raise CustomException(e,sys)

class CustomData:
    numerical_columns = ['GAINED', 'VISITS', 'MAGE', 'FEDUC', 'MEDUC', 'TOTALP', 'BDEAD', 'TERMS', 'WEEKS', 'CIGNUM',
                         'DRINKNUM', 'BWEIGHT']
    categorical_columns = ['SEX', 'MARITAL', 'FAGE', 'LOUTCOME', 'RACEMOM', 'RACEDAD', 'HISPMOM', 'HISPDAD', 'ANEMIA',
                           'CARDIAC', 'ACLUNG', 'DIABETES', 'HERPES', 'HYDRAM', 'HEMOGLOB', 'HYPERCH', 'HYPERPR',
                           'ECLAMP', 'CERVIX', 'PINFANT', 'PRETERM', 'RENAL', 'RHSEN', 'UTERINE']

    def __init__(
            self,
            GAINED: float,
            VISITS: int,
            MAGE: int,
            FAGE: int,
            TOTALP: int,
            BDEAD: int,
            TERMS: int,
            WEEKS: int,
            CIGNUM: int,
            DRINKNUM: int,
            SEX: str,
            MARITAL: str,
            RACEMOM: str,
            RACEDAD: str,
            HISPMOM: str,
            HISPDAD: str,
            ANEMIA: str,
            CARDIAC: str,
            ACLUNG: str,
            DIABETES: str,
            HERPES: str,
            HYDRAM: str,
            HEMOGLOB: str,
            HYPERCH: str,
            HYPERPR: str,
            ECLAMP: str,
            CERVIX: str,
            PINFANT: str,
            PRETERM: str,
            RENAL: str,
            RHSEN: str,
            UTERINE: str
    ):
        self.gained = GAINED
        self.visits = VISITS
        self.m_age = MAGE
        self.f_age = FAGE
        self.total_p = TOTALP
        self.b_dead = BDEAD
        self.terms = TERMS
        self.weeks = WEEKS
        self.cig_num = CIGNUM
        self.drink_num = DRINKNUM
        self.sex = SEX,
        self.marital = MARITAL,
        self.race_mom = RACEMOM,
        self.race_dad = RACEDAD,
        self.hisp_mom = HISPMOM,
        self.hisp_dad = HISPDAD,
        self.anemia = ANEMIA,
        self.cardiac = CARDIAC,
        self.aclung = ACLUNG,
        self.diabetes = DIABETES,
        self.herpes = HERPES,
        self.hydram = HYDRAM,
        self.hyperpr = HYPERPR,
        self.haemoglob = HEMOGLOB,
        self.hyperch = HYPERCH,
        self.eclamp = ECLAMP,
        self.cervix = CERVIX,
        self.pinfant = PINFANT,
        self.preterm = PRETERM,
        self.renal = RENAL,
        self.rhsen = RHSEN,
        self.uterine = UTERINE

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "gained": [self.gained],
                "visits": [self.visits],
                "m_age": [self.m_age],
                "f_age": [self.f_age],
                "total_p": [self.total_p],
                "b_dead": [self.b_dead],
                "terms": [self.terms],
                "weeks": [self.weeks],
                "cig_num": [self.cig_num],
                "drink_num": [self.drink_num],
                "sex": [self.sex],
                "marital": [self.marital],
                "race_mom": [self.race_mom],
                "race_dad": [self.race_dad],
                "hisp_mom": [self.hisp_mom],
                "hisp_dad": [self.hisp_dad],
                "anemia": [self.anemia],
                "cardiac": [self.cardiac],
                "aclung": [self.aclung],
                "diabetes": [self.diabetes],
                "herpes": [self.herpes],
                "hydram": [self.hydram],
                "hyperch": [self.hyperch],
                "hyperpr": [self.hyperpr],
                "haemoglob": [self.haemoglob],
                "eclamp": [self.eclamp],
                "cervix": [self.cervix],
                "pinfant": [self.pinfant],
                "preterm": [self.preterm],
                "renal": [self.renal],
                "rhsen": [self.rhsen],
                "uterine": [self.uterine]
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)
