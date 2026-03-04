from abc import ABC, abstractmethod
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class FeatureEngineeringStrategy(ABC):
    @abstractmethod
    def engineer_features(self, df: pd.DataFrame):
        pass



class log_transformation(FeatureEngineeringStrategy):
    def __init__(self, features):
        self.features = features

    def engineer_features(self, df: pd.DataFrame):
        logging.info(f"Applying log transformation to features: {self.features}")
        df_transformed = df.copy()
        for feature in self.features:
            df_transformed[feature] = np.log1p(df[feature])  # log1p handles log(0) by calculating log(1+x) 
        logging.info("Log transformation completed.")
        return df_transformed

class Standardization(FeatureEngineeringStrategy):
    def __init__(self, features):
        self.features = features

    def engineer_features(self, df: pd.DataFrame):
        logging.info(f"Applying standardization to features: {self.features}")
        scaler = StandardScaler()
        df_scaled = df.copy()
        scale_pipeline=Pipeline([('scaler', StandardScaler())])
        df_scaled[self.features] = scale_pipeline.fit_transform(df[self.features])
        logging.info("Standardization completed.")
        return df_scaled,scale_pipeline

class Normalization(FeatureEngineeringStrategy):    
    def __init__(self, features):
        self.features = features

    def engineer_features(self, df: pd.DataFrame):
        logging.info(f"Applying normalization to features: {self.features}")
        scaler = MinMaxScaler()
        df_normalized = df.copy()
        df_normalized[self.features] = scaler.fit_transform(df[self.features])
        logging.info("Normalization completed.")
        return df_normalized        

class one_hot_encoding(FeatureEngineeringStrategy):
    def __init__(self, features):
        self.features = features

    def engineer_features(self, df: pd.DataFrame):
        logging.info(f"Applying one-hot encoding to features: {self.features}")
        #cat_df=df.select_dtypes(exclude=[np.number])
        cat_pipeline=Pipeline([('onehot', OneHotEncoder(handle_unknown='ignore',drop='first'))])
        df_encoded=cat_pipeline.fit_transform(df[self.features])
        df_encoded=pd.DataFrame(df_encoded.toarray(), columns=cat_pipeline.named_steps['onehot'].get_feature_names_out(self.features))
        df_final=pd.concat([df.drop(columns=self.features), df_encoded], axis=1)    
        return df_final,cat_pipeline

class FeatureEngineering:
    def __init__(self, strategy):
        self.strategy = strategy

    def set_strategy(self, strategy):
        self.strategy = strategy

    def engineer_features(self, df: pd.DataFrame):
        return self.strategy.engineer_features(df)
        

if __name__ == "__main__":

    clean_file_path=r"C:\Users\91954\OneDrive\Desktop\Data_Science\Diabetes_Prediction_System\Cleaned_data"

    preprocessed_data=r"C:\Users\91954\OneDrive\Desktop\Data_Science\Diabetes_Prediction_System\Preprocessed_data"

    for file in ['cleaned_X_train.csv']:
        if file.endswith(".csv" ):
            df = pd.read_csv(os.path.join(clean_file_path, file))
            #df.drop(columns=['diabetes'], inplace=True)
            print(f"\nApplying feature engineering to {file}...")
            cat_pipeline=Pipeline([('onehot', OneHotEncoder(handle_unknown='ignore',drop='first'))])
            scale_pipeline=Pipeline([('scaler', StandardScaler())])
            cat_features=df.select_dtypes(exclude=[np.number]).columns.to_list()
            num_features=df.select_dtypes(include=[np.number]).columns.to_list()
            
            preprocessor=ColumnTransformer(transformers=[('cat', cat_pipeline, cat_features), ('num', scale_pipeline, num_features)])
            pre_df=preprocessor.fit_transform(df)
            df_final=pd.DataFrame(pre_df, columns=preprocessor.get_feature_names_out())
            joblib.dump(preprocessor, r'C:\Users\91954\OneDrive\Desktop\Data_Science\Diabetes_Prediction_System\Pipelines\preprocessor.pkl')
            df_final.to_csv(os.path.join(preprocessed_data, f'preprocessed_{file}'), index=False)
            print(f"Feature engineering completed for {file}")

    pass