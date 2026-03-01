import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

class MultivariateAnalysisStrategy(ABC):

    @abstractmethod
    def analyze(self,df: pd.DataFrame):
        self.generate_correlation_heatmap(df)
        self.generate_pairplot(df)
    
    @abstractmethod
    def generate_correlation_heatmap(self,df: pd.DataFrame):
        pass    
        
    @abstractmethod

    def generate_pairplot(self,df: pd.DataFrame):
        pass


class SimppleMultivariateAnalysis(MultivariateAnalysisStrategy):

    def generate_correlation_heatmap(self, df):
        plt.figure(figsize=(10, 6))
        sns.heatmap(df.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
        plt.title('Correlation Heatmap')
        plt.show()
    def generate_pairplot(self, df):
        sns.pairplot(df)
        plt.suptitle('Pair Plot', y=1.02)
        plt.show()


if __name__ == "__main__":
    pass