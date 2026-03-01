from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class FeatureEngineeringStrategy(ABC):
    @abstractmethod
    def engineer_features(self, df: pd.DataFrame, target_column: str):
        pass



class log_transformation(FeatureEngineeringStrategy):
    def __init__(self, features):
        self.features = features

    def engineer_features(self, df: pd.DataFrame, target_column: str):
        logging.info(f"Applying log transformation to features: {self.features}")
        df_transformed = df.copy()
        for feature in self.features:
            df_transformed[feature] = np.log1p(df[feature])  # log1p handles log(0) by calculating log(1+x) 
        logging.info("Log transformation completed.")
        return df_transformed

class Standardization(FeatureEngineeringStrategy):
    def __init__(self, features):
        self.features = features

    def engineer_features(self, df: pd.DataFrame, target_column: str):
        logging.info(f"Applying standardization to features: {self.features}")
        scaler = StandardScaler()
        df_scaled = df.copy()
        df_scaled[self.features] = scaler.fit_transform(df[self.features])
        logging.info("Standardization completed.")
        return df_scaled

class Normalization(FeatureEngineeringStrategy):    
    def __init__(self, features):
        self.features = features

    def engineer_features(self, df: pd.DataFrame, target_column: str):
        logging.info(f"Applying normalization to features: {self.features}")
        scaler = MinMaxScaler()
        df_normalized = df.copy()
        df_normalized[self.features] = scaler.fit_transform(df[self.features])
        logging.info("Normalization completed.")
        return df_normalized        

class one_hot_encoding(FeatureEngineeringStrategy):
    def __init__(self, features):
        self.features = features

    def engineer_features(self, df: pd.DataFrame, target_column: str):
        logging.info(f"Applying one-hot encoding to features: {self.features}")
        encoder = OneHotEncoder(sparse=False, drop='first')
        encoded_features = encoder.fit_transform(df[self.features])
        encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(self.features))
        df_encoded = pd.concat([df.drop(columns=self.features), encoded_df], axis=1)
        logging.info("One-hot encoding completed.")
        return df_encoded
    
    
class FeatureEngineering:
    def __init__(self, strategy):
        self.strategy = strategy

    def set_strategy(self, strategy):
        self.strategy = strategy

    def engineer_features(self, df: pd.DataFrame, target_column: str):
        return self.strategy.engineer_features(df, target_column)
        