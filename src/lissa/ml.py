import numpy as np
from sklearn.preprocessing import power_transform
import pandas as pd
from typing import List, Tuple

def EntriesPerPump(data: pd.DataFrame) -> np.array:
    '''
        Returns the number of records for each pump.
    '''
    return data["Well Run"].value_counts()[data["Well Run"].unique()].to_numpy()


def Splitter(pumpList: np.ndarray | list, proportion: float , entireData: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray,pd.DataFrame, np.ndarray]:
    '''
        Splits the dataset considering a fraction of pumps (and not a fraction of data). The pumps are chosen randomly.
    '''

    pumpsIndex = np.linspace(0,len(pumpList)-1,len(pumpList)) #there are 57 pumps in the original dataset.

    #chooses randomly as a combination of pump indexes
    trainIndex = np.random.choice(
        pumpsIndex, 
        round(len(pumpList)*proportion), #approximatedly, 75% of the dataset is considered as trainset
        replace=False #an already selected pump cannot be rechosen
        ).astype(int)

    testIndex = pumpsIndex[np.isin(pumpsIndex,trainIndex,invert=True)].astype(int) #pumps indexes that are not in trainIndex

    #np.array(pumpList)

    entireData.fillna(0,inplace=True)

    #define training data
    trainPumpData = entireData.loc[
        entireData["Well Run"].isin(pumpList[trainIndex])
    ] 


    testPumpData = entireData.loc[
        entireData["Well Run"].isin(pumpList[testIndex])
    ]

    X_train = trainPumpData#.drop(columns=["Failure","Well Run","Well_down"])
    #y_train = trainPumpData["Failure"]
    X_train = X_train.sort_values(["Well Run","time"])

    

    X_test = testPumpData#.drop(columns=["time", "Failure","Well_down"])
    #y_test = testPumpData["Failure"]
    X_test = X_test.sort_values(["Well Run","time"])

    return X_train, EntriesPerPump(X_train), X_test, EntriesPerPump(X_test)


def NewHeaderApplier(string: str, exportData: pd.DataFrame) -> list:
    '''
        Search for a fragment of string in columns, and returns the ones containing the aforementioned string.
    '''
    return [
        colName for colName 
        in list(exportData) 
        if string in colName
        ]



def BoxCoxProccess(data: pd.DataFrame,columnName: str | list ) -> np.ndarray:
    '''
        Returns an array with the yeo-johnson transform of the specified column name. 

        This continuous reshaping is needed due to the power_transform implementation.
    '''
    return power_transform(data[columnName].to_numpy().reshape(-1,1)).reshape(1,-1)[0]