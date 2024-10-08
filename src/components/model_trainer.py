import sys
import os
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import evaluate_model, compare_and_save_model, load_hyperparameter_grids, hyperparameter_tuning


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')

    yaml_file_path = os.path.join('config', 'hyperparameters.yaml')
    param_grids = load_hyperparameter_grids(yaml_file_path)


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_arr, test_arr):
        try:
            logging.info('Splitting train and test input data')
            X_train, y_train, X_test, y_test = (
                train_arr[:, : -1],
                train_arr[:, -1],
                test_arr[:, : -1],
                test_arr[:, -1]
            )

            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "KNearest Neighbor": KNeighborsRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }

            best_model_score = -float('inf')
            best_model_name = None
            best_model = None

            logging.info(f"Param Grids:", self.model_trainer_config.param_grids)

            for model_name, model in models.items():
                logging.info(f"Training {model_name} with hyperparameter tuning")

                # Perform hyperparameter tuning using the grid from YAML
                tuned_model = hyperparameter_tuning(
                    model=model,
                    param_grid=self.model_trainer_config.param_grids[model_name],
                    X_train=X_train,
                    y_train=y_train
                )

                # Evaluate the tuned model
                model_report = evaluate_model(
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_test,
                    y_test=y_test,
                    models={model_name: tuned_model}
                )

                # Find the best model based on R2 score
                model_score = model_report[model_name]

                if model_score > best_model_score:
                    best_model_score = model_score
                    best_model_name = model_name
                    best_model = tuned_model

            # if best_model_score < 0.5:
            #     raise CustomException('No best model found')

            logging.info(f"Best model after tuning is {best_model_name} with R2 score: {best_model_score}")

            # Compare the best model with existing saved model and save if better
            compare_and_save_model(
                file_path=self.model_trainer_config.trained_model_file_path,
                new_model=best_model,
                X_test=X_test,
                y_test=y_test,
                new_model_score=best_model_score
            )

            # Final prediction with the best model
            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)

            return r2_square
        except Exception as e:
            raise CustomException(e, sys)

            pass
        except Exception as e:
            raise CustomException(e, sys)
