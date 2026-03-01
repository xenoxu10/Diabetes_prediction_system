from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging

from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class DataSplittingStrategy(ABC):
    @abstractmethod
    def split_data(self,df: pd.DataFrame, target_column: str):
        pass

class SimpleTrainTestSplitStrategy(DataSplittingStrategy):
    def __init__(self,test_size=0.2,random_state=42):
        self.test_size=test_size
        self.random_state=random_state
    
    def split_data(self, df, target_column):
        logging.info("Performing simple train-test split.")
        X = df.drop(columns=[target_column])
        y = df[target_column]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)
        return X_train, X_test, y_train, y_test
    

if __name__ == "__main__":
    pass
