import logging
from abc import ABC, abstractmethod
#rom pyexpat import model
from typing import Any

import joblib
import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Setup logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# Abstract Base Class for Model Building Strategy
class ModelBuildingStrategy(ABC):
    @abstractmethod
    def build_and_train_model(self, X_train: pd.DataFrame, y_train: pd.DataFrame) -> RegressorMixin:
        """
        Abstract method to build and train a model.

        Parameters:
        X_train (pd.DataFrame): The training data features.
        y_train (pd.Series): The training data labels/target.

        Returns:
        RegressorMixin: A trained scikit-learn model instance.
        """
        pass


# Concrete Strategy for Linear Regression using scikit-learn
class LinearRegressionStrategy(ModelBuildingStrategy):
    def build_and_train_model(self, X_train: pd.DataFrame, y_train: pd.DataFrame) -> Pipeline:
        """
        Builds and trains a linear regression model using scikit-learn.

        Parameters:
        X_train (pd.DataFrame): The training data features.
        y_train (pd.Series): The training data labels/target.

        Returns:
        Pipeline: A scikit-learn pipeline with a trained Linear Regression model.
        """
        # Ensure the inputs are of the correct type
        # if not isinstance(X_train, pd.DataFrame):
        #     raise TypeError("X_train must be a pandas DataFrame.")
        # if not isinstance(y_train, pd.Series):
        #     raise TypeError("y_train must be a pandas Series.")

        logging.info("Initializing Logistic Regression model with scaling.")

        # Creating a pipeline with standard scaling and logistic regression
        pipeline = Pipeline(
            [
               # ("scaler", StandardScaler()),  # Feature scaling
                ("model", LogisticRegression()),  # Logistic regression model
            ]
        )

        logging.info("Training Logistic Regression model.")
        pipeline.fit(X_train, y_train)  # Fit the pipeline to the training data

        logging.info("Model training completed.")
        return pipeline


# Context Class for Model Building
class ModelBuilder:
    def __init__(self, strategy: ModelBuildingStrategy):
        """
        Initializes the ModelBuilder with a specific model building strategy.

        Parameters:
        strategy (ModelBuildingStrategy): The strategy to be used for model building.
        """
        self._strategy = strategy

    def set_strategy(self, strategy: ModelBuildingStrategy):
        """
        Sets a new strategy for the ModelBuilder.

        Parameters:
        strategy (ModelBuildingStrategy): The new strategy to be used for model building.
        """
        logging.info("Switching model building strategy.")
        self._strategy = strategy

    def build_model(self, X_train: pd.DataFrame, y_train: pd.DataFrame) -> RegressorMixin:
        """
        Executes the model building and training using the current strategy.

        Parameters:
        X_train (pd.DataFrame): The training data features.
        y_train (pd.Series): The training data labels/target.

        Returns:
        RegressorMixin: A trained scikit-learn model instance.
        """
        logging.info("Building and training the model using the selected strategy.")
        return self._strategy.build_and_train_model(X_train, y_train)


# Example usage
if __name__ == "__main__":
    # Example DataFrame (replace with actual data loading)
    X_train=pd.read_csv(r'C:\Users\91954\OneDrive\Desktop\Data_Science\Diabetes_Prediction_System\Preprocessed_data\preprocessed_cleaned_X_train.csv')
    y_train=pd.read_csv(r'C:\Users\91954\OneDrive\Desktop\Data_Science\Diabetes_Prediction_System\Cleaned_data\cleaned_y_train.csv')


    # Example usage of Linear Regression Strategy
    model_builder = ModelBuilder(LinearRegressionStrategy())
    trained_model = model_builder.build_model(X_train, y_train)
    print(trained_model.named_steps['model'].coef_)  # Print model coefficients
    #model.save(r'C:\Users\91954\OneDrive\Desktop\Data_Science\Diabetes_Prediction_System\Models\linear_regression_model.pkl')
    joblib.dump(trained_model, r'C:\Users\91954\OneDrive\Desktop\Data_Science\Diabetes_Prediction_System\Models\linear_regression_model.pkl')

    pass
