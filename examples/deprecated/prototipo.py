import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import hmmlearn.hmm as hmm

import lissa as li


PCAData = pd.read_csv("../data/PCA/PCAtotal.csv",delimiter=",").fillna(0)
PCAData.rename(columns={"Unnamed: 0":"time"},inplace=True)

mainSeed = 971215
np.random.seed(mainSeed)

PCAHeaders = [str(i) for i in range(0,8)]

pumpList = PCAData["Well Run"].unique()

complexVib = PCAData["ESP Vibration X"]+1j*PCAData["ESP Vibration Y"]

PCAData["VibMod"] = np.absolute(complexVib)


X_train, trainLength, X_test, testLength, modelData, totalLength = li.Splitter(pumpList=pumpList,proportion=0.75,entireData=PCAData)

nr = 3
nv = 2

radiusData = PCAData["Radius"].loc[PCAData["Well_down"]==0]
vibData= PCAData["VibMod"].loc[(PCAData["VibMod"]!=0) & (PCAData["Well_down"]==0)].apply(np.log1p)


modelGMMradius = li.GaussianMixtureFit(radiusData,nr,seed=mainSeed)
modelGMMvibs = li.GaussianMixtureFit(vibData,nv,seed=mainSeed)

modelRadius = hmm.GaussianHMM(
    n_components=nr,
    random_state=mainSeed,
    covariance_type="full",
    init_params="st"
    )

modelVib = hmm.GaussianHMM(
    n_components=nv,
    random_state=mainSeed,
    covariance_type="full",
    init_params="st",
    )

modelRadius.means_ = modelGMMradius.means_
modelRadius.covars__ = modelGMMradius.covariances_

modelVib.means_ = modelGMMvibs.means_
modelVib.covars_ = modelGMMvibs.covariances_



modelRadius = li.HMMTrainer(X_train["Radius"],trainLength,modelRadius)
modelVib = li.HMMTrainer(X_train["VibMod"].apply(np.log1p),trainLength,modelVib)


li.PostProcessing(modelRadius, PCAData, modelData,"Radius", "State Radius", totalLength)
li.PostProcessing(modelVib, PCAData, modelData,"VibMod", "State Mod", totalLength)

PCAData["State Radius"] = PCAData["State Radius"].map(li.StateConversion(modelRadius.get_stationary_distribution(),nr))
PCAData["State Mod"] = PCAData["State Mod"].map(li.StateConversion(modelVib.get_stationary_distribution(),nv))

PCAData.loc[(PCAData["VibMod"]==0) & (PCAData["Well_down"]==0),"State Mod"] = 3


medianList = np.array([])

for pump in pumpList:
    pumpData = PCAData.loc[(PCAData["Well Run"]==pump) & (PCAData["Well_down"]==0) ]#.copy()
    last = pumpData.shape[0]
    pumpMedian = pumpData.loc[pumpData.index[last-24:last-1],"State Mod"].median()
    medianList = np.append(medianList,pumpMedian)


cmap = plt.get_cmap('Oranges', 4)
plt.figure(figsize=(7,5))

states,counts = np.unique(medianList,return_counts=True)

plt.bar(states,counts,width=0.3, color=[cmap(i) for i in range(1,cmap.N)])
plt.xticks(states,np.int64(states))
plt.ylabel("Counts")
plt.xlabel("States")
plt.title("Vibration Modulus Persistence Histogram")
plt.savefig("../imagens_gerais/vibration_histogram")

medianList = np.array([])

for pump in pumpList:
    pumpData = PCAData.loc[(PCAData["Well Run"]==pump) & (PCAData["Well_down"]==0) ]#.copy()
    last = pumpData.shape[0]
    pumpMedian = pumpData.loc[pumpData.index[last-24:last-1],"State Radius"].median()
    medianList = np.append(medianList,pumpMedian)

cmap = plt.get_cmap('Oranges', 4)
plt.figure(figsize=(7,5))
states,counts = np.unique(medianList,return_counts=True)
plt.bar(states,counts,width=0.3,color=[cmap(i) for i in range(1,cmap.N)])
plt.xticks(states,np.int64(states))
plt.ylabel("Counts")
plt.xlabel("States")
plt.title("Radius Persistence Histogram")
plt.savefig("../imagens_gerais/radius_histogram")