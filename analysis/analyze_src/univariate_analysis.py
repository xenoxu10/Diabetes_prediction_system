import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from abc import ABC, abstractmethod

class UnivariateAnalysisStrategy(ABC):
    @abstractmethod
    def analyze(self,df: pd.DataFrame):
        self.plot_histograms(df)
        self.plot_boxplots(df)

    @abstractmethod
    def plot_histograms(self,df: pd.DataFrame):
        pass

    @abstractmethod
    def plot_boxplots(self,df: pd.DataFrame):
        pass

class SimpleUnivariateAnalysis(UnivariateAnalysisStrategy):

    def plot_histograms(self, df):
        df.hist(figsize=(12, 10), bins=20, color='skyblue', edgecolor='black')
        plt.suptitle('Histograms of Features', y=1.02)
        plt.show()

    def plot_boxplots(self, df):
        plt.figure(figsize=(12, 10))
        for i, column in enumerate(df.columns, 1):
            plt.subplot(4, 4, i)
            sns.boxplot(y=df[column], color='lightcoral')
            plt.title(f'Box Plot of {column}')
        plt.tight_layout()
        plt.show()      

if __name__ == "__main__":      
    pass