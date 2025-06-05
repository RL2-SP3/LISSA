import pandas as pd

from sklearn.decomposition import PCA, FastICA

import lissa.processing as pro

import numpy as np

#apenas cabeçalhos operacionais, ou seja, aqueles que vão efetivamente na PCA
def operationalHeader(exportData):
    # vibrationHeaders = [
    #     'ESP Vibration X',
    #     'ESP Vibration Y'
    # ]    
    # return list(set(pro.relevantHeader(exportData))-set(vibrationHeaders))
    return pro.relevantHeader(exportData)



def ApplyPCA(exportData, n):   
    inputData = exportData[operationalHeader(exportData)].loc[exportData["Well_down"]==0].fillna(0)
    pca = PCA(n_components=n)
    pca.fit(inputData)
    return pca

def ApplyICA(exportData, n):   
    inputData = exportData[operationalHeader(exportData)].loc[exportData["Well_down"]==0].fillna(0)
    ica = FastICA(n_components=n)
    ica.fit(inputData)
    return ica
  

def ReducePCA(pca, n, Headers,entireData):
    PCAHeaders = [str(i) for i in range(0,n)]
    pcat = pd.DataFrame(pca.components_.T,columns=PCAHeaders)
    pcat.index = Headers
    PCAData = entireData[Headers] @ pcat

    return PCAData
    #PCAData.to_csv("data/PCA/PCAtotal.csv",index=False)

def ReduceICA(ica, n, Headers,entireData):
    ICAHeaders = [str(i) for i in range(0,n)]
    icat = pd.DataFrame(ica.components_.T,columns=ICAHeaders)
    icat.index = Headers
    ICAData = entireData[Headers] @ icat

    return ICAData
    #PCAData.to_csv("data/PCA/PCAtotal.csv",index=False)

def ExportPCAData(PCAData,entireData):
    notNumericalHeaders = [
        "Well Run",
        "Failure",
        "Well_down",
        "Well aligned to Train A",
        "Well aligned to Train B",
        #,
        'Failure Info',
        'Pump Info'
        ]
    
    inputHeaders = [
        'Water Cut @ 20degC - 1 atm',
        'Choke Opening'
    ]

    vibrationHeaders = [
        #'ESP Vibration X',
        #'ESP Vibration Y',
        'ESP Vibration Module'

    ]
    PCAHeaders = list(PCAData)
    PCAData = pd.concat([PCAData,entireData[notNumericalHeaders+inputHeaders+vibrationHeaders]],axis=1)
    PCAData["Radius"] = PCAData[PCAHeaders].pow(2).sum(axis=1).pow(1/2)
    
    #PCAData.to_csv("data/PCA/PCAtotal.csv",index=True)

    return PCAData
