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
            model_path = os.path.join('artifacts', 'model.pkl')
            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')

            logging.info('Before Loading')

            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)

            logging.info('After Loading')

            scaled_data = preprocessor.transform(features)

            predictions = model.predict(scaled_data)

            return predictions
        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
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
        self.mage = MAGE
        self.fage = FAGE
        self.totalp = TOTALP
        self.bdead = BDEAD
        self.terms = TERMS
        self.weeks = WEEKS
        self.cignum = CIGNUM
        self.drinknum = DRINKNUM
        self.sex = SEX,
        self.marital = MARITAL,
        self.racemom = RACEMOM,
        self.racedad = RACEDAD,
        self.hispmom = HISPMOM,
        self.hispdad = HISPDAD,
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
                "GAINED": [self.gained],
                "VISITS": [self.visits],
                "MAGE": [self.mage],
                "FAGE": [self.fage],
                "TOTALP": [self.totalp],
                "BDEAD": [self.bdead],
                "TERMS": [self.terms],
                "WEEKS": [self.weeks],
                "CIGNUM": [self.cignum],
                "DRINKNUM": [self.drinknum],
                "SEX": [self.sex],
                "MARITAL": [self.marital],
                "RACEMOM": [self.racemom],
                "RACEDAD": [self.racedad],
                "HISPMOM": [self.hispmom],
                "HISPDAD": [self.hispdad],
                "ANEMIA": [self.anemia],
                "CARDIAC": [self.cardiac],
                "ACLUNG": [self.aclung],
                "DIABETES": [self.diabetes],
                "HERPES": [self.herpes],
                "HYDRAM": [self.hydram],
                "HEMOGLOB": [self.haemoglob],
                "HYPERCH": [self.hyperch],
                "HYPERPR": [self.hyperpr],
                "ECLAMP": [self.eclamp],
                "CERVIX": [self.cervix],
                "PINFANT": [self.pinfant],
                "PRETERM": [self.preterm],
                "RENAL": [self.renal],
                "RHSEN": [self.rhsen],
                "UTERINE": [self.uterine]
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)
