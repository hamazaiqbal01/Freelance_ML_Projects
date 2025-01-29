import os 
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor 
from sklearn.ensemble import (
    AdaBoostRegressor, 
    GradientBoostingRegressor, 
    RandomForestRegressor, 
)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor 

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models  # ✅ Import missing function

@dataclass
class ModelTrainingConfig: 
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainingConfig()  # ✅ Fixed missing parentheses

    def initate_model_trainer(self, train_array, test_array, preprocessor_path):
        try:
            logging.info("Splitting training and test input data")

            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            # ✅ Correct Indentation
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }

            # ✅ Define Hyperparameter Grid
            param_grid = {
                "Random Forest": {"n_estimators": [50, 100, 200]},
                "Decision Tree": {"max_depth": [3, 5, 10]},
                "Gradient Boosting": {"learning_rate": [0.01, 0.1, 0.2]},
                "Linear Regression": {},  # No hyperparameters to tune
                "XGBRegressor": {"n_estimators": [50, 100, 200]},
                "CatBoosting Regressor": {"depth": [6, 8, 10]},
                "AdaBoost Regressor": {"n_estimators": [50, 100, 200]},
            }

            # ✅ Fix function call by passing `param_grid`
            model_report: dict = evaluate_models(
                X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                models=models, param=param_grid  # ✅ Fix: Added `param`
            )

            # ✅ Get best model
            best_model_score = max(model_report.values())
            best_model_name = max(model_report, key=model_report.get)
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found")

            logging.info(f"Best model found: {best_model_name} with score {best_model_score}")

            # ✅ Save best model
            save_object(self.model_trainer_config.trained_model_file_path, best_model)

            # ✅ Predict and return R² score
            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)

            return r2_square

        except Exception as e:  # ✅ Correct Indentation of `except`
            raise CustomException(e, sys)
