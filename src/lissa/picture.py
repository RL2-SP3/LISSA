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
    
    if n > 1:   
        axs = axs.ravel()
        i = 0
        for prop in relevantHeaders:
            sm.qqplot(data[prop], line='s',ax=axs[i])
            axs[i].title.set_text(prop)
            i +=1

        for j in range(i,n*n):
            axs[j].remove()
    else:
         sm.qqplot(data, line='s',ax=axs)

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


def OverFill(pumpData,Headers,State,n,ax):
    cmap = plt.get_cmap('Oranges', n+1)
    for state in range(0,n+1):
            color = cmap(state)  # Pega uma cor automática para cada estado
            ax.fill_between(pumpData.index,np.min(pumpData[Headers]), np.max(pumpData[Headers]), where=(pumpData[State] == state), 
                            color=color, alpha=0.3, label=f"State {state}")
            ax.legend(loc='upper left',bbox_to_anchor=(1, 1),fontsize=20)


def HMMPicture(pumpData,pump,PCAHeaders,n):

    pumpData["time"] = pd.to_datetime(pumpData["time"])
    pumpData.set_index("time",inplace=True)

    pumpData = pumpData.asfreq('h',fill_value=0)

    n_g = 3
    fig, axs = plt.subplots(n_g,1, figsize=(60,30))

    pumpData[PCAHeaders].plot(ax=axs[0])
    OverFill(pumpData,PCAHeaders,"State Gaussian",n[0],axs[0])

    vibeHeader = ["ESP Vibration X","ESP Vibration Y"]

    pumpData[vibeHeader].plot(ax=axs[1])
    OverFill(pumpData,vibeHeader,"State Vib",n[1],axs[1])
    
    esotericHeader = ['Water Cut @ 20degC - 1 atm', 'Choke Opening']
    pumpData[esotericHeader].plot(ax=axs[2])
    OverFill(pumpData,esotericHeader,"State Tot",n[2],axs[2])



    axs[2].legend(loc='upper left',bbox_to_anchor=(1, 1),fontsize=20)

    for i in range(0,n_g):
        if pumpData.loc[pumpData["Failure"]==True].shape[0] != 0:
            axs[i].axvline(x=pumpData.loc[pumpData["Failure"]==True].index[0], color='red', linestyle='--', linewidth=5)


    fig.suptitle("HMM: " + pump,fontsize=20);
    plt.figtext(0.5, 0.32, pumpData["Pump Info"].iloc[0],fontsize=10,va="center",ha="center")
    plt.figtext(0.5, 0.3, pumpData["Failure Info"].iloc[0],fontsize=10,va="center",ha="center")


    plt.tight_layout()
    return fig,axs


def HMMPicture(pumpData,pump,PCAHeaders, props,states, numberOfStates):

    pumpData["time"] = pd.to_datetime(pumpData["time"])
    pumpData.set_index("time",inplace=True)
    pumpData = pumpData.asfreq('h',fill_value=0)


    n_g = len(props)    
    fig, axs = plt.subplots(n_g,1, figsize=(60,30))

    if n_g == 1:
        pumpData[props[0]].plot(ax=axs)
        OverFill(pumpData,props[0],states[0],numberOfStates[0],axs)
        axs.legend(loc='upper left',bbox_to_anchor=(1, 1),fontsize=20)
        if pumpData.loc[pumpData["Failure"]==True].shape[0] != 0:
            axs.axvline(x=pumpData.loc[pumpData["Failure"]==True].index[0], color='red', linestyle='--', linewidth=5)
    else:
        for i in range(0,n_g):
            pumpData[props[i]].plot(ax=axs[i])
            OverFill(pumpData,props[i],states[i],numberOfStates[i],axs[i])
            axs[i].legend(loc='upper left',bbox_to_anchor=(1, 1),fontsize=20)
            if pumpData.loc[pumpData["Failure"]==True].shape[0] != 0:
                axs[i].axvline(x=pumpData.loc[pumpData["Failure"]==True].index[0], color='red', linestyle='--', linewidth=5)
        
    fig.suptitle("HMM: " + pump,fontsize=20);
    plt.figtext(0.5, 0.32, pumpData["Pump Info"].iloc[0],fontsize=10,va="center",ha="center")
    plt.figtext(0.5, 0.3, pumpData["Failure Info"].iloc[0],fontsize=10,va="center",ha="center")


    plt.tight_layout()
    return fig,axs