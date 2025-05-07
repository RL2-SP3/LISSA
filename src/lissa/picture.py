import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

import statsmodels.api as sm

import seaborn as sns
import matplotlib.colors as mcolors


from scipy.stats import norm

#directed imported from analysis.py
def CorrGraphGen(corrAnalysis,plotHeaders,pump):
    dataCorrelation = corrAnalysis[plotHeaders].corr(method='pearson')

    mask = np.triu(np.ones_like(dataCorrelation, dtype=bool)) # Generate a mask for the upper triangle
    f, ax = plt.subplots(figsize=(11, 9)) # Set up the matplotlib figure
    cmap = sns.diverging_palette(230, 20, as_cmap=True) # Generate a custom diverging colormap
    sns.heatmap(dataCorrelation, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5}) # Draw the heatmap with the mask and correct aspect ratio

   

    #plt.savefig("imagens/comFiltro/correlacoes/corr_"+pump+".png")
    plt.savefig("imagens/PCAcorr/corr_"+pump+".jpg")
    plt.close() 



def FigureComponents(pca,string,Headers,typemethod="pca"):

    PCAt = pca.components_.T

    plt.figure(figsize=(10, 5))

    plt.imshow(PCAt, interpolation='nearest',aspect='auto',cmap='bwr')

    for i in range(PCAt.shape[0]):
        for j in range(PCAt.shape[1]):
            plt.text(j, i, f'{PCAt[i, j]:.2f}', ha='center', va='center', color='black',fontsize=10)


    plt.yticks(ticks=range(0,len(Headers)),labels=Headers)
    plt.title(string,fontsize=16)
    

    
    if typemethod=="pca":
        plt.colorbar(label="Correlation")
        plt.xlabel("PCA Components",fontsize=13)
        plt.ylabel("Original Data Features",fontsize=13)
        plt.savefig("../imagens_gerais/PCA_matrix.jpg", bbox_inches='tight')
        plt.tight_layout()
        plt.show()
        print(pca.explained_variance_ratio_.cumsum())
    else:
        plt.colorbar(label="Weights")
        plt.xlabel("ICA Components",fontsize=13)
        plt.ylabel("Original Data Features",fontsize=13)
        plt.savefig("../imagens_gerais/PCA_matrix.jpg", bbox_inches='tight')
        plt.tight_layout()
        plt.show()


   
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


def HMMPicture(pumpData,pump, props,states, numberOfStates, measures, figsize=(60,30),posLeg = 0.5):

    pumpData["time"] = pd.to_datetime(pumpData["time"])
    pumpData.set_index("time",inplace=True)
    pumpData = pumpData.asfreq('h',fill_value=0)


    n_g = len(props)    
    fig, axs = plt.subplots(n_g,1, figsize=figsize,sharex=True)

    if n_g == 1:
        pumpData[props[0]].plot(ax=axs)
        OverFill(pumpData,props[0],states[0],numberOfStates[0],axs)
        axs.legend(loc='upper left',bbox_to_anchor=(1, 1),fontsize=20)
        axs.tick_params(axis='both',which="both", labelsize="20")
        axs.set_xlabel("time",fontsize=20)
        axs.set_ylabel(measures,fontsize=20)
        if pumpData.loc[pumpData["Failure"]==True].shape[0] != 0:
            axs.axvline(x=pumpData.loc[pumpData["Failure"]==True].index[0], color='red', linestyle='--', linewidth=5)
    else:
        for i in range(0,n_g):
            pumpData[props[i]].plot(ax=axs[i])
            OverFill(pumpData,props[i],states[i],numberOfStates[i],axs[i])
            axs[i].legend(loc='upper left',bbox_to_anchor=(1, 1),fontsize=20)
            axs[i].tick_params(axis='both',which="both", labelsize="20")
            axs[i].set_xlabel("time",fontsize=20)
            axs[i].set_ylabel(measures[i],fontsize=20)
            if pumpData.loc[pumpData["Failure"]==True].shape[0] != 0:
                axs[i].axvline(x=pumpData.loc[pumpData["Failure"]==True].index[0], color='red', linestyle='--', linewidth=5)
        
    fig.suptitle("HMM: " + pump,fontsize=25);
    #plt.figtext(0.5, posLeg + 0.02, pumpData["Pump Info"].iloc[0],fontsize=10,va="center",ha="center")
    #plt.figtext(0.5, posLeg, pumpData["Failure Info"].iloc[0],fontsize=10,va="center",ha="center")


    plt.tight_layout()
    return fig,axs


def GaussianMixturePlot(data,gmm,strings,figsize=(7,5)):
    model = gmm
    means = model.means_.flatten()
    stds = np.sqrt(model.covariances_).flatten()
    weights = model.weights_

    # Criando um range de valores para plotar as distribuições
    x = np.linspace(0, np.max(data), 1000)


    fig = plt.figure(figsize=figsize)
    # Plotando histograma dos dados originais
    plt.hist(data, bins=100, density=True, alpha=0.5, label=strings[0])

    # Plotando cada gaussiana individualmente
    for i in range(model.means_.shape[0]):
        plt.plot(x, weights[i] * norm.pdf(x, means[i], stds[i]), label=f"{strings[1]} {i+1}")

    # Plotando a soma das gaussianas
    pdf = np.exp(gmm.score_samples(x.reshape(-1, 1)))
    plt.plot(x, pdf, label="Soma das Gaussianas", color="blue", linestyle="dashed")

    plt.legend()
    plt.title(strings[2])
    plt.xlabel(strings[3])
    plt.ylabel(strings[4])
    path = "../imagens_gerais/gmm_"+data.name+".jpg"
    plt.savefig(path)
    plt.show()


    return gmm, fig



def PCAComparisionPlot(pumpData,originalData,pump,PCAHeaders,originalHeaders):

    plt.rcParams['font.size'] = 15.0
    fig, axs = plt.subplots(2,1, figsize=(20,10),sharex=True)
    

    pumpData[PCAHeaders].plot(ax=axs[1])
    originalData[originalHeaders].plot(ax=axs[0])
    axs[0].legend(loc='upper left',bbox_to_anchor=(1, 1),fontsize=10)
    axs[1].legend(loc='upper left',bbox_to_anchor=(1, 1),fontsize=10)



    if pumpData.loc[pumpData["Failure"]==True].shape[0] != 0:       
        axs[0].axvline(pumpData.loc[pumpData["Failure"]==True].index[0], color='red', linestyle='--', linewidth=5)
        axs[1].axvline(pumpData.loc[pumpData["Failure"]==True].index[0], color='red', linestyle='--', linewidth=5)

    
    fig.suptitle("PCA Data of " + pump,fontsize=20);
    plt.tight_layout()
    factor = 0.5
    plt.rcParams.update({
        'font.size': plt.rcParams['font.size'] * factor,
    })
    
    return fig,axs


def ZScorePlot(totalData,pump,Headers):
    pumpData =  totalData.loc[totalData["Well Run"]==pump]

    fig, axs = plt.subplots(1,1,figsize=(20,7))
    pumpData[Headers].plot(ax=axs)
    axs.legend(loc='upper left',bbox_to_anchor=(1, 1),fontsize=15)

    if pumpData.loc[pumpData["Failure"]==True].shape[0] != 0:
        axs.axvline(pumpData.loc[pumpData["Failure"]==True].index[0], color='red', linestyle='--', linewidth=5)
        

    fig.suptitle("Modified Z-score data of " + pump,fontsize=20);
    
    return fig,axs