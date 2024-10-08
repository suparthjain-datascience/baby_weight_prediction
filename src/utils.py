import os
import sys
import dill
from sklearn.metrics import r2_score
from src.exception import CustomException
from src.logger import logging
import pickle
import yaml
from sklearn.model_selection import GridSearchCV


def load_object(file_path):
    """
    Loads the object from the specified file path using dill.
    """
    try:
        if os.path.exists(file_path):
            with open(file_path, 'rb') as file_obj:
                logging.info(f'Loading object from {file_path}')
                return pickle.load(file_obj)
        else:
            logging.info(f'No object found at {file_path}')
            return None
    except Exception as e:
        raise CustomException(e, sys)


def cleaning_data_process(data):
    try:
        cols_to_remove = ['ID', 'LOUTCOME', 'FEDUC', 'MEDUC']
        cleaned_data = data.drop(cols_to_remove, axis=1)

        numerical_cols = ['GAINED', 'VISITS', 'MAGE', 'FAGE', 'TOTALP', 'BDEAD', 'TERMS', 'WEEKS', 'CIGNUM',
                          'DRINKNUM']

        all_columns = cleaned_data.columns

        categorical_cols = [col for col in all_columns if col not in numerical_cols and col != 'BWEIGHT']

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
        raise CustomException(e, sys)


def get_numerical_and_categorical_columns(data):
    try:
        numerical_cols = ['GAINED', 'VISITS', 'MAGE', 'TOTALP', 'BDEAD', 'TERMS', 'WEEKS', 'CIGNUM',
                          'DRINKNUM', 'BWEIGHT']
        all_columns = data.columns
        categorical_cols = [col for col in all_columns if col not in numerical_cols]

        return (
            numerical_cols,
            categorical_cols
        )
    except Exception as e:
        raise CustomException(e, sys)


def save_object(file_path, obj):
    """
    Saves the object to the specified file path using dill.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)
        logging.info(f'Object saved to {file_path}')
    except Exception as e:
        raise CustomException(e, sys)


def evaluate_model(X_train, y_train, X_test, y_test, models):
    """
    Evaluates a list of models on train and test sets and returns their test R2 scores.

    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        models: A dictionary of models to be trained and evaluated

    Returns:
        A dictionary with model names as keys and their R2 test scores as values.
    """
    try:
        report = {}
        for model_name, model in models.items():
            logging.info(f'Evaluating {model_name}')
            # Fit the model
            model.fit(X_train, y_train)

            # Predict on train and test sets
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # Calculate R2 scores
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            # Log the scores
            logging.info(f'{model_name} Train R2 score: {train_model_score}')
            logging.info(f'{model_name} Test R2 score: {test_model_score}')

            # Store the test score in the report
            report[model_name] = test_model_score

        return report
    except Exception as e:
        logging.error(f'Error occurred while evaluating models: {str(e)}')
        raise CustomException(e, sys)


def compare_and_save_model(file_path, new_model, X_test, y_test, new_model_score):
    """
    Loads an existing model (if it exists), compares it with the new model,
    and saves the new model only if it performs better.

    Args:
        file_path: The path where the model is saved/loaded from.
        new_model: The new model to be compared and potentially saved.
        X_test: Test features for evaluation.
        y_test: Test labels for evaluation.
        new_model_score: R2 score of the new model.
    """
    try:
        # Load the existing model if it exists
        existing_model = load_object(file_path)

        if existing_model is not None:
            logging.info('An existing model is found. Comparing with the new model...')

            # Predict using the existing model
            existing_pred = existing_model.predict(X_test)
            existing_r2_score = r2_score(y_test, existing_pred)

            logging.info(f'Existing model R2 score: {existing_r2_score}')
            logging.info(f'New model R2 score: {new_model_score}')

            # Compare the new model with the existing one
            if new_model_score > existing_r2_score:
                logging.info('New model is better. Saving the new model...')
                save_object(file_path=file_path, obj=new_model)
            else:
                logging.info('Existing model performs better. No need to save the new model.')
        else:
            logging.info('No existing model found. Saving the new model...')
            save_object(file_path=file_path, obj=new_model)

    except Exception as e:
        logging.error(f'Error during model comparison and saving: {str(e)}')
        raise CustomException(e, sys)


def load_hyperparameter_grids(yaml_file_path):
    """
    Load hyperparameter grids from a YAML file.
    Returns:
        A dictionary with model names as keys and hyperparameter grids as values.
    """
    try:
        with open(yaml_file_path, 'r') as file:
            param_grids = yaml.safe_load(file)
        return param_grids
    except Exception as e:
        raise CustomException(e, sys)


def hyperparameter_tuning(model, param_grid, X_train, y_train):
    """
    Performs hyperparameter tuning using GridSearchCV.

    Args:
        model: The model to tune.
        param_grid: Hyperparameter grid to search over.
        X_train: Training features.
        y_train: Training labels.

    Returns:
        The best estimator (model with the best hyperparameters).
    """
    try:
        search = GridSearchCV(model, param_grid, cv=3, n_jobs=-1, verbose=2)

        search.fit(X_train, y_train)

        return search.best_estimator_
    except Exception as e:
        raise CustomException(e, sys)
