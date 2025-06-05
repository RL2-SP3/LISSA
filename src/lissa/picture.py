import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

import statsmodels.api as sm

import seaborn as sns
import matplotlib.colors as mcolors

from sklearn.mixture import GaussianMixture
from scipy.stats import norm
from math import ceil, sqrt

from scipy.stats import norm

def Traducao():
    return {
    'VSD power frequency': "Frequência do AVV",
    'ESP motor temperature': "Temperatura do motor da ESP",
    'ESP intake Pressure':"Pressão na Entrada da ESP",
    'Water Cut @ 20degC - 1 atm':"Fração de água @ ",
    'ESP intake temperature':"Temperatura na Entrada da ESP", 
    'ESP discharge pressure':"Pressão de Descarga da ESP", 
    'Choke Opening':"Abertura da Válvula",
    'Well head pressure':"Pressão na Cabeça do Poço",#
    'Well head Temperature' : "Temperatura na Cabeça do Poço",
    'Well aligned to Train A': "Poço Alinhado a A",
    'Well aligned to Train B': "Poço Alinhado a b",
    'ESP Motor Voltage': "Tensão do Motor da ESP",
    'ESP Vibration X': "Vibração da ESP em X",
    'ESP Vibration Y': "Vibração da ESP em Y",
    'Well Run': "Corrida do Poço",
    'Well_down': "Poço desativado",
    'Pump Info': "Informação da bomba",
    'Failure Info': "Informação da Falha",
    'Failure': "Falha",
    'Current Mean': "Média da corrente",
    "":"",
    "ESP Vibration Module":"Módulo da Vibração na ESP"
    }

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



def FigureComponents(
        pca, Headers, plotName, 
        listOfNames = ["PCA - Measures and Variance Components", "Correlation","PCA Components", "Original Data Features"],
        savePath = "../imagens_gerais/",
        english = True):

    PCAt = pca.components_.T

    plt.figure(figsize=(10, 5))

    plt.imshow(PCAt, interpolation='nearest',aspect='auto',cmap='bwr')

    for i in range(PCAt.shape[0]):
        for j in range(PCAt.shape[1]):
            plt.text(j, i, f'{PCAt[i, j]:.2f}', ha='center', va='center', color='black',fontsize=10)

    if english:
        plt.yticks(ticks=range(0,len(Headers)),labels=Headers)
    else:
        mapa = Traducao()
        newHeaders = [mapa[item] for item in Headers]
        plt.yticks(ticks=range(0,len(Headers)),labels=newHeaders)
    
    plt.title(listOfNames[0],fontsize=16)
    plt.colorbar(label=listOfNames[1])
    plt.xlabel(listOfNames[2],fontsize=13)
    plt.ylabel(listOfNames[3],fontsize=13)
    plt.savefig(savePath+plotName, bbox_inches='tight')
    plt.tight_layout()
    return plt.gcf(), plt.gca()
    

   
#realiza os qq plots dos dados e das colunas que forem necessárias 
def QQPlots(data,relevantHeaders,lineType="s", english=True,titleFontsize=20,ydist=1):

    n = np.ceil(np.sqrt(len(relevantHeaders))).astype(int)
    fig, axs = plt.subplots(n,n,figsize=(20,20))

    if not english:
        mapa = Traducao()
        fig.suptitle("Gráfico Q-Q",fontsize=titleFontsize, y=ydist)
    else:
        fig.suptitle("Q-Q Plot",fontsize=titleFontsize,y=ydist)
        
    
    if n > 1:   
        axs = axs.ravel()
        i = 0
        for prop in relevantHeaders:
            sm.qqplot(data[prop], line=lineType,ax=axs[i])
            if not english:
                axs[i].title.set_text(mapa[prop])
                axs[i].set_xlabel("Quantil teórico")
                axs[i].set_ylabel("Quantil da amostra")
            else:
                axs[i].title.set_text(prop)
            i +=1

        for j in range(i,n*n):
            axs[j].remove()
    else:
         sm.qqplot(data, line=lineType,ax=axs)

    fig.tight_layout(pad=1.1)

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
    plt.xlim(0,17)

    # Plotando cada gaussiana individualmente
    for i in range(model.means_.shape[0]):
        plt.plot(x, weights[i] * norm.pdf(x, means[i], stds[i]), label=f"{strings[1]} {i+1}")

    # Plotando a soma das gaussianas
    #pdf = np.exp(gmm.score_samples(x.reshape(-1, 1)))
    #plt.plot(x, pdf, label="Soma das Gaussianas", color="blue", linestyle="dashed")

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


def ZScorePlot(totalData,pump,Headers,english = True):
    pumpData =  totalData.loc[totalData["Well Run"]==pump]

    fig, axs = plt.subplots(1,1,figsize=(20,7))
    pumpData[Headers].plot(ax=axs)

    if pumpData.loc[pumpData["Failure"]==True].shape[0] != 0:
        axs.axvline(pumpData.loc[pumpData["Failure"]==True].index[0], color='red', linestyle='--', linewidth=5)
    

    if not english:
        axs.legend([Traducao()[item] for item in Headers],loc='upper left',bbox_to_anchor=(1, 1),fontsize=15)
        axs.set_xlabel("Tempo")
        fig.suptitle("Gráfico do Z-score " + pump,fontsize=20);
    else:
        fig.suptitle("Modified Z-score data of " + pump,fontsize=20);           

        
    return fig,axs



def PlotGMMMarginals(gmm: GaussianMixture, X: np.ndarray, bins=50):
    n_features = X.shape[1]
    n_components = gmm.means_.shape[0]

    s = ceil(sqrt(n_features))
    fig, axes = plt.subplots(s, s, figsize=(5 * s, 4 * s))
    axes = axes.flatten()

    for i in range(n_features):
        ax = axes[i]
        x = X[:, i]
        ax.hist(x, bins=bins, density=True, alpha=0.5, label='Data hist')

        xmin, xmax = x.min(), x.max()
        x_grid = np.linspace(xmin, xmax, 1000)
        total_pdf = np.zeros_like(x_grid)

        for k in range(n_components):
            mean = gmm.means_[k, i]
            var = gmm.covariances_[k]
            if gmm.covariance_type == 'full':
                var = var[i, i]
            elif gmm.covariance_type == 'diag':
                var = var[i]
            elif gmm.covariance_type == 'spherical':
                var = var
            std = np.sqrt(var)

            weight = gmm.weights_[k]
            pdf_k = weight * norm.pdf(x_grid, loc=mean, scale=std)
            total_pdf += pdf_k
            ax.plot(x_grid, pdf_k, '--', label=f'Comp {k+1}')

        ax.plot(x_grid, total_pdf, '-', color='black', label='GMM Total')
        ax.set_title(f'Feature {i+1}')
        ax.legend()

    # Remove subplots extras se s² > n_features
    for j in range(n_features, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()
    fig.savefig("../imagens_gerais/multiple_gmm.jpeg")



def PlotHMMProbs(data,model):

    if type(data)==pd.Series:
        totalReshaped = data.to_numpy().reshape(-1,1)
    else:
        totalReshaped = data.to_numpy()
    
    df = pd.DataFrame(model.predict_proba(totalReshaped))

    # Índices de tempo
    time = np.arange(len(df))

    # Criar figura
    fig, ax = plt.subplots(figsize=(20, 6))

    # Cores para os estados
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c','#4cff2c','#fcff2c']

    # Gráfico de área empilhada com transparência (alpha)
    ax.stackplot(time, df.T, labels=df.columns, colors=colors, alpha=0.5)

    # Configurações do gráfico
    ax.set_xlabel("Tempo")
    ax.set_ylabel("Probabilidade Empilhada (%)")
    ax.set_title("Probabilidades do HMM (100% Stacked Area Chart)")
    ax.legend(title="Estados", loc='upper left')
    plt.ylim(0, 1)  # Mantém o eixo Y de 0 a 1 (100%)

    # Exibir o gráfico
    plt.show()

