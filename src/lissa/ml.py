import numpy as np
from sklearn.preprocessing import power_transform
import pandas as pd
from typing import List, Tuple

from scipy.stats import norm

from hmmlearn import hmm

from matplotlib import pyplot as plt

from sklearn.mixture import GaussianMixture

def EntriesPerPump(entireData: pd.DataFrame, pumpList: list, defIndex: np.ndarray) -> Tuple[pd.DataFrame,np.array]:
    '''
        Returns the data related of online pumps and it's number of records for each pump.
    '''
    modelPlay = np.array([])
    for pump in pumpList[defIndex]:
        pumpData = entireData.loc[entireData["Well Run"]==pump].copy()

        #pumpData["Shutdown"] = (pumpData["Well_down"] != pumpData["Well_down"].shift(1).fillna(pumpData["Well_down"].iloc[-1])) #differentiates well down blocks

        wellDown = pumpData["Well_down"].to_numpy()

        pumpData["Shutdown"] = (wellDown != np.roll(wellDown,1)) #differentiates well down blocks

        pumpData["CumShut"] = pumpData["Shutdown"].cumsum() #group names them

        testSize =pumpData.groupby("CumShut") #groupby execution

        blockSize = testSize.size() #gets size of each group
        real = blockSize[testSize["Well_down"].last()==0] #selects only online runs

        modelData = pumpData.loc[pumpData["Well_down"]==0].copy()
   
    
        modelPlay = np.concatenate([modelPlay,real.to_numpy()])
        if pump == pumpList[defIndex][0]:
            exportData = modelData
        else:
            exportData = pd.concat([exportData,modelData],axis=0)

    return exportData.infer_objects(), modelPlay.astype(int)


def Splitter(
        pumpList: np.ndarray | list, 
        proportion: float , 
        entireData: pd.DataFrame
        ) -> Tuple[pd.DataFrame, np.ndarray,pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]:
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

    totalLength = np.concatenate([trainLength,testLength])

    modelData = pd.concat([X_train,X_test])


    return X_train, trainLength, X_test, testLength, modelData, totalLength


def NewHeaderApplier(string: str, exportData: pd.DataFrame) -> list:
    '''
        Search for a fragment of string in columns, and returns the ones containing the aforementioned string.
    '''
    return [
        colName for colName 
        in list(exportData) 
        if string in colName
        ]



def StateConversion(distribution: np.ndarray ,n: int) -> dict:
    '''
        The HMM does the state classification randomly, this means that the state 0 not necessarily occurs most. 
        Therefore, this function reorganizes the states, with the higher number meaning the lowest probability state in distribution.
    '''
    stateOrder = np.argsort(distribution)+1
    stateOrder = np.insert(np.flip(stateOrder),0,0)
    return dict(zip(stateOrder,range(0,n+1)))



def check_data(data):
    if type(data) == pd.Series:
        return data.to_numpy().reshape(-1,1)
    else:
        return data.to_numpy()



def PostProcessing(
        model:          hmm.BaseHMM | GaussianMixture, 
        originalData:   pd.Series | pd.DataFrame, 
        modelData:      pd.Series | pd.DataFrame,
        inputHeader:    str, 
        outputHeader:   str, 
        totalLength:    np.ndarray, 
        ):
    
    '''
        Generates the states related to the model and the dataset, setting them on the original dataframe.
        Also, if verbose, print the AIC and BIC for comparision.
    '''
    
    totalReshaped = check_data(modelData[inputHeader])
    
    originalData.set_index(keys="Well Run",append=True,inplace=True)
    modelData.set_index(keys="Well Run",append=True,inplace=True)

    originalData[outputHeader] = 0
    if type(model) == hmm.BaseHMM:
        originalData.loc[modelData.index,outputHeader] =  model.predict(totalReshaped,totalLength)+1
    else:
        originalData.loc[modelData.index,outputHeader] =  model.predict(totalReshaped)+1

    originalData.reset_index(level="Well Run",inplace=True)
    modelData.reset_index(level="Well Run",inplace=True)

    return originalData,modelData
    


