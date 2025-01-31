import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

import statsmodels.api as sm

import seaborn as sns
import matplotlib.colors as mcolors

#directed imported from analysis.py
def CorrGraphGen(corrAnalysis,plotHeaders,pump):
    dataCorrelation = corrAnalysis[plotHeaders].corr(method='pearson')

    mask = np.triu(np.ones_like(dataCorrelation, dtype=bool)) # Generate a mask for the upper triangle
    f, ax = plt.subplots(figsize=(11, 9)) # Set up the matplotlib figure
    cmap = sns.diverging_palette(230, 20, as_cmap=True) # Generate a custom diverging colormap
    sns.heatmap(dataCorrelation, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5}) # Draw the heatmap with the mask and correct aspect ratio

   

    #plt.savefig("imagens/comFiltro/correlacoes/corr_"+pump+".png")
    plt.savefig("imagens/PCAcorr/corr_"+pump+".png")
    plt.close() 



def FigureComponents(pca,string,Headers):

    PCAt = pca.components_.T

    plt.figure(figsize=(10, 5))
    plt.imshow(PCAt, interpolation='nearest',aspect='auto',cmap='bwr')

    for i in range(PCAt.shape[0]):
        for j in range(PCAt.shape[1]):
            plt.text(j, i, f'{PCAt[i, j]:.2f}', ha='center', va='center', color='black')


    plt.yticks(ticks=range(0,len(Headers)),labels=Headers)
    plt.title(string)
    plt.colorbar()

    plt.show()
    print(pca.explained_variance_ratio_.cumsum())


   
#realiza os qq plots dos dados e das colunas que forem necessárias 
def QQPlots(data,relevantHeaders):

    n = np.ceil(np.sqrt(len(relevantHeaders))).astype(int)
    fig, axs = plt.subplots(n,n,figsize=(20,20))
    axs = axs.ravel()

    i = 0
    for prop in relevantHeaders:
        sm.qqplot(data[prop], line='s',ax=axs[i])
        axs[i].title.set_text(prop)
        i +=1

    for j in range(i,n*n):
        axs[j].remove()

    return fig,axs

def PCAComponentsPlot(pumpData,pump,PCAHeaders):

    fig, axs = plt.subplots(3,1, figsize=(40,17))

    pumpData[PCAHeaders].plot(ax=axs[0])
    axs[0].legend(loc='upper left',bbox_to_anchor=(1, 1),fontsize=15)



    norm = mcolors.Normalize(vmin=-6, vmax=6)

    its = axs[1].pcolor(pumpData[PCAHeaders].T,cmap='hsv', norm=norm)

    axs[1].grid(axis="y",linewidth=0.5,color="black")

    pumpData[PCAHeaders].pow(2).sum(axis=1).pow(1/2).plot(ax=axs[2])

    if pumpData.loc[pumpData["Failure"]==True].shape[0] != 0:
        failureX = pumpData.index.get_loc(pumpData.loc[pumpData["Failure"]==True].index[0])
        
        axs[1].axvline(x=failureX, color='red', linestyle='--', linewidth=2)
        axs[0].axvline(pumpData.loc[pumpData["Failure"]==True].index[0], color='red', linestyle='--', linewidth=2)
        axs[2].axvline(pumpData.loc[pumpData["Failure"]==True].index[0], color='red', linestyle='--', linewidth=2)


    fig.colorbar(its,ax=axs[1],orientation='horizontal',shrink=0.5)

    fig.suptitle("PCA Data of " + pump,fontsize=20);

    return fig,axs


def HMMPicture(pumpData,pump,PCAHeaders,cmap,n):

    pumpData["time"] = pd.to_datetime(pumpData["time"])
    pumpData.set_index("time",inplace=True)

    pumpData = pumpData.asfreq('h',fill_value=0)

    pumpData["Shutdown"] = pumpData["Well_down"] != pumpData["Well_down"].shift(-1).fillna(pumpData["Well_down"].iloc[-1])


    fig, axs = plt.subplots(2,1, figsize=(60,20))

    pumpData[PCAHeaders].plot(ax=axs[0])
    for state in range(0,n+1):
            color = cmap(state)  # Pega uma cor automática para cada estado
            axs[0].fill_between(pumpData.index,np.min(pumpData[PCAHeaders]), np.max(pumpData[PCAHeaders]), where=(pumpData["State Gaussian"] == state), 
                            color=color, alpha=0.3, label=f"State {state}")

            axs[0].legend(loc='upper left',bbox_to_anchor=(1, 1),fontsize=20)

    pumpData[['Water Cut @ 20degC - 1 atm', 'Choke Opening']].plot(ax=axs[1])
    axs[1].legend(loc='upper left',bbox_to_anchor=(1, 1),fontsize=20)

    if pumpData.loc[pumpData["Failure"]==True].shape[0] != 0:
            #failureX = pumpData.index.get_loc(pumpData.loc[pumpData["Failure"]==True].index[0])
            axs[0].axvline(x=pumpData.loc[pumpData["Failure"]==True].index[0], color='red', linestyle='--', linewidth=2)
            axs[1].axvline(x=pumpData.loc[pumpData["Failure"]==True].index[0], color='red', linestyle='--', linewidth=2)


    fig.suptitle("HMM: " + pump,fontsize=20);
    plt.figtext(0.5, 0.47, pumpData["Pump Info"].iloc[0],fontsize=10,va="center",ha="center")
    plt.figtext(0.5, 0.45, pumpData["Failure Info"].iloc[0],fontsize=10,va="center",ha="center")


    plt.tight_layout()
    return fig,axs