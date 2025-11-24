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
        model:          hmm.BaseHMM | GaussianMixture, 
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

    
    originalData.set_index(keys="Well Run",append=True,inplace=True)
    modelData.set_index(keys="Well Run",append=True,inplace=True)

    originalData[outputHeader] = 0
    if type(model) == hmm.BaseHMM:
        originalData.loc[modelData.index,outputHeader] =  model.predict(totalReshaped,totalLength)+1
    else:
        originalData.loc[modelData.index,outputHeader] =  model.predict(totalReshaped)+1

    originalData.reset_index(level="Well Run",inplace=True)
    modelData.reset_index(level="Well Run",inplace=True)

    if verbose:
        print("AIC: " + str(model.aic(totalReshaped))+ " BIC: " + str(model.bic(totalReshaped)))

    return originalData,modelData
    

class GaussianMixtureModel:

    DEFAULT_FIG_PARAMS = {
            "fig_size"          :   (20,10),
            "text_size"         :   10,
            "title_size"        :   16,
            "tick_size"         :   10,
            "label_size"        :   10,
            "tight_layout_pad"  :   1,
            "dpi"               :   600
        }
    
    DEFAULT_LABEL_PARAMS = {
            "x_label"           :   "X Label",
            "y_label"           :   "Y Label",
            "plot_title"        :   "Title of the Plot",
            "colorbar_title"    :   "Colorbar title",
            "legend_title"      :   "Legend title"
            }

    DEFAULT_GAUSSIAN_MIXTURE_MODEL_PARAMS = {
        "histogram_name" : "Histogram",
        "gaussian_name"  : "Gaussian",
        "figsize"        : (20,10)
    }

    DEFAULT_SAVE_PARAMS = {
            "save_figure"       :    False,
            "dir"               :    "./",
            "file_name"         :    "image"            
        }
    
    DEFAULT_DICTS_PARAMS ={
            "portuguese_dictionary" :   "translation_to_portuguese",
            "units"                 :   "measurement_units",
            "dict_path"             :   "./dictionaries/dictionaries.json",
            "headers_path"          :   "./dictionaries/new_headers.json"
        }

    params = {**DEFAULT_GAUSSIAN_MIXTURE_MODEL_PARAMS}

    def __init__(self,data,**kwargs):

        self.params = {**self.params, **kwargs}
       
        verbose = kwargs.pop("verbose", False)
        self.gmm = GaussianMixture(**kwargs)

        reshaped_data = data.to_numpy()
        if type(data) == pd.Series:
            reshaped_data = reshaped_data.reshape(-1,1)
        
        
        self.gmm.fit(reshaped_data)

        if verbose:
            print("GMM AIC: " + str(self.gmm.aic(reshaped_data)))
            print("GMM BIC: " + str(self.gmm.bic(reshaped_data)))

    def __getattr__(self, name):
        return getattr(self.gmm,name)

    def plot_gmm(self,limits  =   (0,17)):
               
        self.stds = np.sqrt(self.gmm.covariances_).flatten()
        self.weights = self.gmm.weights_
        self.means = self.gmm.means_

        # Criando um range de valores para plotar as distribuições
        x = np.linspace(limits[0], np.max(self.data), 1000)


        # Plotando histograma dos dados originais
        self.figure = plt.figure(figsize=self.params["figsize"])
        self.ax = self.figure.add_subplot(1,1,1)
        
        self.ax.hist(self.data, bins=100, density=True, alpha=0.5, label=self.params["histogram_name"])
        self.ax.set_xlim(limits[0],limits[1])

        # Plotando cada gaussiana individualmente
        for i in range(self.gmm.means_.shape[0]):
            self.ax.plot(x, self.weights[i] * norm.pdf(x, self.means[i], self.stds[i]), label=self.params["gaussian_name"] + f"{i+1}")

        self.figure.legend()
        self.figure.suptitle(self.params["plot_title"],fontsize=self.params["title_size"])
        self.ax.set_xlabel(self.params["x_label"])
        self.ax.set_ylabel(self.params["y_label"])
        self.figure.show()


