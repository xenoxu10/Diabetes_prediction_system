import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from abc import ABC, abstractmethod


class BivariateAnalysisStrategy(ABC):
    @abstractmethod
    def analyze(self,df: pd.DataFrame, feature1: str, feature2: str):
        pass

###########################################################################


class NumericalVSNumericalAnalysis(BivariateAnalysisStrategy):

    def analyze(self, df, feature1, feature2):
        
        plt.figure(figsize=(10,6))
        sns.scatterplot(x=feature1, y=feature2, data=df)
        plt.title(f"{feature1} vs {feature2}")
        plt.xlabel(feature1)
        plt.ylabel(feature2)
        plt.show()

    
class NumericalVsCategoricalAnalysis(BivariateAnalysisStrategy):
    def analyze(self, df, feature1, feature2):
        
        plt.figure(figsize=(10,6))
        sns.boxplot(x=feature1, y=feature2, data=df)
        plt.title(f"{feature1} vs {feature2}")
        plt.xlabel(feature1)
        plt.ylabel(feature2)
        plt.show()


class BivariateAnalyzer:

    def __init__(self,strategy: BivariateAnalysisStrategy):
        self._strategy=strategy
    
    def set_strategy(self,strategy: BivariateAnalysisStrategy):
        self._strategy=strategy
    
    def execute_strategy(self,df: pd.DataFrame, feature1: str, feature2: str):
        self._strategy.analyze(df, feature1, feature2)


if __name__== "__main__":
    pass