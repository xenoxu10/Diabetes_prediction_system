from abc import ABC, abstractmethod

import pandas as pd


class DataInspectionStrategy():
    @abstractmethod
    def inspect(self,df: pd.DataFrame):
        pass

###########################################################



class DataTypesInspectionStrategy(DataInspectionStrategy):

    def inspect(self,df: pd.DataFrame):
        print("\nData Types and Non-null Counts:")
        print(df.info())

###########################################################


class SummaryStatisticsInspectionStrategy(DataInspectionStrategy):

    def inspect(self,df: pd.DataFrame):
        print("\nSummary Statistics (Numerical Features):")
        print(df.describe())
        print("\nSummary Statistics (Categorical Features):")
        print(df.describe(include=["O"]))


class DataInspector:

    def __init__(self, strategy: DataInspectionStrategy):
        self._strategy = strategy

    def set_strategy(self,strategy: DataInspectionStrategy):
        self._strategy = strategy
    
    def execute_strategy(self,df:pd.DataFrame):
        self._strategy.inspect(df)
    

if __name__ == "__main__":

    # df=pd.DataFrame([
    #     [10,20,30],
    #     [40,50,60]],columns=["A","B","C"]
    # )

    # inpector=DataInspector(DataTypesInspectionStrategy())
    # inpector.execute_strategy(df)
    pass



 
