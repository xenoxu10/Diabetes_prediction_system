from abc import abstractmethod, ABC
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


class MissingValueAnalysisStrategy(ABC):
    @abstractmethod

    def analyze(self,df: pd.DataFrame):

        self.identify_missing_values(df)
        self.visualize_missing_values(df)
    

    @abstractmethod

    def identify_missing_values(self,df: pd.DataFrame):
        pass

    def visualize_missing_values(self,df: pd.DataFrame):
       pass


class SimpleMissingValueAnalysis(MissingValueAnalysisStrategy):

    def identify_missing_values(self,df: pd.DataFrame):
        print("\nMissing Values Count:")
        print(df.isnull().sum())

    def visualize_missing_values(self,df: pd.DataFrame):
        plt.figure(figsize=(10, 6))
        sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
        plt.title('Missing Values Heatmap')
        plt.show()


if __name__ == "__main__":
    pass