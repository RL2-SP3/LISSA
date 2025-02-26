import numpy as np
from sklearn.preprocessing import power_transform
import pandas as pd
from typing import List, Tuple

from hmmlearn import hmm

from matplotlib import pyplot as plt

def EntriesPerPump(entireData: pd.DataFrame, pumpList: list, defIndex: np.ndarray) -> np.array:
    '''
        Returns the number of records for each pump.
    '''
    #return data["Well Run"].value_counts()[data["Well Run"].unique()].to_numpy()

    exportData = pd.DataFrame(columns=list(entireData))

    modelPlay = np.array([])
    for pump in pumpList[defIndex]:
        pumpData = entireData.loc[entireData["Well Run"]==pump].copy()

        pumpData["Shutdown"] = (pumpData["Well_down"] != pumpData["Well_down"].shift(1).fillna(pumpData["Well_down"].iloc[-1])) #differentiates well down blocks

        pumpData["CumShut"] = pumpData["Shutdown"].cumsum() #group names them

        testSize =pumpData.groupby("CumShut") #groupby execution

        blockSize = testSize.size() #gets size of each group
        real = blockSize[testSize["Well_down"].last()==0] #selects only online runs

        modelData = pumpData.loc[pumpData["Well_down"]==0].copy()
   
    
        modelPlay = np.concat([modelPlay,real.to_numpy()])
        exportData = pd.concat([exportData,modelData],axis=0)

    return exportData, modelPlay.astype(int)


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
  
    X_train, trainLength = EntriesPerPump(entireData, pumpList, trainIndex)
    X_test, testLength = EntriesPerPump(entireData, pumpList, testIndex)
    return X_train, trainLength, X_test, testLength


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


# def HiddenMarkovModel(modelData,PCAData,n,seed,X_train,trainLength,totalLength,Headers,outputHeaders):
#     model = hmm.GaussianHMM(n_components=n,random_state=seed)

#     if type(X_train[Headers]) == pd.Series:
#         reshapedData = X_train[Headers].to_numpy().reshape(-1,1)
#         model.fit(reshapedData,trainLength)
#         modelData[outputHeaders] = model.predict(modelData[Headers].to_numpy().reshape(-1,1),totalLength)+1;
#     else:
#         reshapedData = X_train[Headers].to_numpy()
#         model.fit(reshapedData,trainLength)
#         modelData[outputHeaders] = model.predict(modelData[Headers].to_numpy(),totalLength)+1;
    
#     print(np.log(model.aic(reshapedData)))

#     PCAData[outputHeaders] = 0
#     PCAData.loc[modelData[outputHeaders].index,outputHeaders] = modelData[outputHeaders]
#     return model


def StateConversion(distribution,n):
    stateOrder = np.argsort(distribution)+1
    stateOrder = np.insert(np.flip(stateOrder),0,0)
    return dict(zip(stateOrder,range(0,n+1)))

def HiddenMarkovModel(X_train, trainLength, mainSeed, n,covar_type="full",algorithm="viterbi"):
    model = hmm.GaussianHMM(
        n_components=n,
        covariance_type=covar_type,
        random_state=mainSeed,
        algorithm=algorithm,
        n_iter = 100,
        tol=0.01)

    if type(X_train) == pd.Series:
        reshapedData = X_train.to_numpy().reshape(-1,1)
        model.fit(reshapedData,trainLength)
    else:
        reshapedData = X_train.to_numpy()
        model.fit(reshapedData,trainLength)


    return model


def PostProcessing(model, PCAData, modelData,inputHeader, outputHeader, totalLength):
    
    if type(modelData[inputHeader])==pd.Series:
        totalReshaped = modelData[inputHeader].to_numpy().reshape(-1,1)
        modelData[outputHeader] = model.predict(totalReshaped,totalLength)+1;
    else:
        totalReshaped = modelData[inputHeader].to_numpy()
        modelData[outputHeader] = model.predict(totalReshaped,totalLength)+1;


    print(np.log(model.aic(totalReshaped)))
    PCAData[outputHeader] = 0
    PCAData.loc[modelData[outputHeader].index,outputHeader] = modelData[outputHeader]
