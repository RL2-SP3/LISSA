import numpy as np
from sklearn.preprocessing import power_transform


def EntriesPerPump(data):
    return data["Well Run"].value_counts()[data["Well Run"].unique()].to_numpy()

def Splitter(pumpList,proportion,entireData):
    pumpsIndex = np.linspace(0,len(pumpList)-1,len(pumpList)) #there are 57 pumps in this dataset.

    #chooses randomly as a combination of pump indexes
    trainIndex = np.random.choice(
        pumpsIndex, 
        round(len(pumpList)*proportion), #approximatedly, 75% of the dataset is considered as trainset
        replace=False #an already selected pump cannot be rechosen
        ).astype(int)

    testIndex = pumpsIndex[np.isin(pumpsIndex,trainIndex,invert=True)].astype(int) #pumps indexes that are not in trainIndex

    np.array(pumpList)

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



#busca um pré-header em uma lista de cabeçalhos 
#e.g.: vc quer procurar MACD na lista e ele selciona apenas os cabeçalhos que contem MACD)
def NewHeaderApplier(string,exportData):
    return [
        colName for colName 
        in list(exportData) 
        if string in colName
        ]



def BoxCoxProccess(data,columnName):
    return power_transform(data[columnName].to_numpy().reshape(-1,1)).reshape(1,-1)[0]