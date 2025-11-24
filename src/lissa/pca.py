import pandas as pd

import lissa.processing as pro

class Transformations():
    def __init__(self):
        pass


def AutoModelApplier(model, exportData: pd.DataFrame):
    '''
        Set correctly the data to be fit, considering only relevant headers and data to the transformation
    '''
    #apenas cabeçalhos operacionais, ou seja, aqueles que vão efetivamente na PCA

    inputData = exportData[pro.relevantHeader(exportData)].loc[exportData["Well_down"]==0].fillna(0) 
    model.fit(inputData)
    return model

def AutoReduce(
        model,
        n:                  int,
        originalHeaders:    list,
        entireData:         pd.DataFrame
        ): #-> Tuple[TransformClass, list]
    '''
        Similar to fit_transform method, but allows the model just to be fitted, not necessarily transformed.

        Indeed, it could be directly fit_transform
    '''

    transformHeaders = [str(i) for i in range(0,n)]
    transformedTransposed = pd.DataFrame(model.components_.T,columns=transformHeaders)
    transformedTransposed.index = originalHeaders
    transformedData = entireData[originalHeaders] @ transformedTransposed
    return transformedData, transformHeaders

def ExportPCAData(
        PCAData:    pd.DataFrame,
        entireData: pd.DataFrame,
        export=False,
        path="data/PCA/PCAtotal.csv"
        ) -> pd.DataFrame:
    '''
        Integrates the transformed data with the previous non transformed data.
        
        By default, does not exports PCA Data to CSV file
    '''

    #depending on the approach, these headers might change.
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
        #'ESP Vibration Module'

    ]

    PCAData["Radius"] = PCAData.pow(2).sum(axis=1).pow(1/2)
    PCAData = pd.concat([PCAData,entireData[notNumericalHeaders+inputHeaders+vibrationHeaders]],axis=1)
    
    
    if export:
        PCAData.to_csv(path,index=True)

    return PCAData


'''
Deprecated functions
'''

# def ApplyPCA(exportData, n):   
#     pca = PCA(n_components=n)
#     AutoModelApplier(PCA(n_components=n),exportData)
#     return pca

# def ApplyICA(exportData, n):   
#     ica = FastICA(n_components=n)
#     AutoModelApplier(ica,exportData)
#     return ica

# def ReducePCA(pca, n, originalHeaders,entireData):
#    return AutoReduce(pca,n,entireData=entireData,originalHeaders=originalHeaders)

# def ReduceICA(ica, n, originalHeaders,entireData):
#     return AutoReduce(ica,n,entireData=entireData,originalHeaders=originalHeaders)