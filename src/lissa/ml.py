import numpy as np
from sklearn.preprocessing import power_transform
import pandas as pd
from typing import List, Tuple

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

        pumpData["Shutdown"] = (pumpData["Well_down"] != pumpData["Well_down"].shift(1).fillna(pumpData["Well_down"].iloc[-1])) #differentiates well down blocks

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



def BoxCoxProccess(data: pd.DataFrame,columnName: str | list ) -> np.ndarray:
    '''
        Returns an array with the yeo-johnson transform of the specified column name. 

        This continuous reshaping is needed due to the power_transform implementation.
    '''
    return power_transform(data[columnName].to_numpy().reshape(-1,1)).reshape(1,-1)[0]


def StateConversion(distribution: np.ndarray ,n: int) -> dict:
    '''
        The HMM does the state classification randomly, this means that the state 0 not necessarily occurs most. 
        Therefore, this function reorganizes the states, with the higher number meaning the lowest probability state in distribution.
    '''
    stateOrder = np.argsort(distribution)+1
    stateOrder = np.insert(np.flip(stateOrder),0,0)
    return dict(zip(stateOrder,range(0,n+1)))

def HMMTrainer(
        X_train:        pd.DataFrame, 
        trainLength:    np.ndarray, 
        model:          hmm.BaseHMM
        ):
    
    reshapedData = X_train.to_numpy()
    if type(X_train) == pd.Series:
        reshapedData = reshapedData.reshape(-1,1)

    model.fit(reshapedData,trainLength)

    return model



def PostProcessing(
        model:          hmm.BaseHMM, 
        originalData:   pd.Series | pd.DataFrame, 
        modelData:      pd.Series | pd.DataFrame,
        inputHeader:    str, 
        outputHeader:   str, 
        totalLength:    np.ndarray, 
        verbose=True
        ):
    
    '''
        Generates the states related to the model and the dataset, setting them on the original dataframe.
        Also, if verbose, print the AIC and BIC for comparision.
    '''
    
    totalReshaped = modelData[inputHeader].to_numpy()

    if type(modelData[inputHeader])==pd.Series:
        totalReshaped = totalReshaped.reshape(-1,1)

    modelData[outputHeader] = model.predict(totalReshaped,totalLength)+1;

    if verbose:
        print("AIC: " + str(model.aic(totalReshaped))+ " BIC: " + str(model.bic(totalReshaped)))
    
    
    if originalData.index.unique().shape[0] == originalData.index.shape[0]:
        originalData[outputHeader] = 0
        originalData.loc[modelData[outputHeader].index,outputHeader] = modelData[outputHeader]
        return originalData
    else:
        originalData_ = originalData.reset_index()
        modelData_ = modelData.reset_index()

        # Realizar merge com base em chaves que permitem identificar a linha
        merged = originalData_.merge(
            modelData_[[originalData.index.name, 'Well Run', outputHeader]],
            on=[originalData.index.name, 'Well Run'],  # ajuste conforme suas chaves
            how='left').fillna(0)

        
        return merged.set_index(originalData.index.names)


def GaussianMixtureFit(
        data:           pd.Series | pd.DataFrame ,
        n_components:   int,
        seed:           int,
        verbose=True
        ):
    
    '''
        Fits a Gaussian Mixture Model into provided data
    '''
    

    reshapedData = data.to_numpy()
    if type(data) == pd.Series:
        reshapedData = reshapedData.reshape(-1,1)
    
    gmm = GaussianMixture(n_components=n_components, random_state=seed,covariance_type="full")
    gmm.fit(reshapedData)

    if verbose:
        print("GMM AIC: " + str(gmm.aic(reshapedData)))
        print("GMM BIC: " + str(gmm.bic(reshapedData)))

    return gmm




'''
    These functions might be deprecated:
'''


# def GaussianHiddenMarkovModel(
#         X_train:        pd.Series | pd.DataFrame , 
#         trainLength:    np.ndarray, 
#         mainSeed:       int, 
#         n:              int,
#         covar_type="full",
#         algorithm="viterbi"
#         ) -> hmm.GaussianHMM:
    
#     '''
#         This function defines a model and reshapes if the input is a pd.Series.
#         It allows the code to be more easier to deal without knowing hmmlearn.
#     '''
    
#     model = hmm.GaussianHMM(
#         n_components=n,
#         covariance_type=covar_type,
#         random_state=mainSeed,
#         algorithm=algorithm,
#         n_iter = 100,
#         tol=0.01)

#     reshapedData = X_train.to_numpy()
    
#     if type(X_train) == pd.Series:
#         reshapedData = reshapedData.reshape(-1,1)
        
#     model.fit(reshapedData,trainLength)
    
#     return model


# def StandardMarkovModel(n,seed, gmm):
#     model = hmm.GaussianHMM(
#         n_components=n,
#         random_state=seed,
#         covariance_type="full",
#         init_params="st",
#         #algorithm="map"
#         )
   
#     model.means_ = gmm.means_
#     model.covars__ = gmm.covariances_

#     return model
