import pandas as pd
from matplotlib import pyplot as plt

from matplotlib import figure, axes
import numpy as np

import statsmodels.api as sm

import seaborn as sns
import matplotlib.colors as mcolors

from sklearn.mixture import GaussianMixture
import scipy.stats as ss
from math import ceil, sqrt

from scipy.stats import norm

def Traducao(
        Header: list, 
        english=False):
    '''
        If exists, translate a property from english to portuguese.
    '''
    dicionario =  {
    'VSD power frequency': "Frequência do AVV",
    "ESP motor Current - phase A":"Corrente do motor - Fase A",
    "ESP motor Current - phase B":"Corrente do motor - Fase B",
    "ESP motor Current - phase C":"Corrente do motor - Fase C",
    'ESP discharge temperature sensor':'Sensor de temperatura na descarga da BCS',
    'ESP differential pressure':"Pressão diferencial da BCS",
    'ESP motor temperature': "Temperatura do motor da BCS",
    'ESP intake Pressure':"Pressão na Entrada da BCS",
    'Water Cut @ 20degC - 1 atm':"Fração de água @ ",
    'ESP intake temperature':"Temperatura na Entrada da BCS", 
    'ESP discharge pressure':"Pressão de Descarga da BCS", 
    'Choke Opening':"Abertura da Válvula",
    'Well head pressure':"Pressão na Cabeça do Poço",#
    'Well head Temperature' : "Temperatura na Cabeça do Poço",
    'Well aligned to Train A': "Poço Alinhado a A",
    'Well aligned to Train B': "Poço Alinhado a b",
    'ESP Motor Voltage': "Tensão do Motor da BCS",
    'ESP Vibration X': "Vibração da BCS em X",
    'ESP Vibration Y': "Vibração da BCS em Y",
    'Well Run': "Corrida do Poço",
    'Well_down': "Poço desativado",
    'Pump Info': "Informação da bomba",
    'Failure Info': "Informação da Falha",
    'Failure': "Falha",
    'Current Mean': "Média da corrente",
    'ESP Current Module': "Módulo da Corrente na BCS",
    "":"",
    "ESP Vibration Module":"Módulo da Vibração na BCS"
    }

    if Header in list(dicionario) and english == False:
        return dicionario[Header]
    else:
        return Header
    

def Measures(Header:list):
    '''
        Links a property to it's measure.
    '''
    dicionario =  {
    'VSD power frequency': "Hz",
    'ESP motor temperature': "ºC",
    'ESP intake Pressure':"Bar",
    'Water Cut @ 20degC - 1 atm':"%",
    'ESP intake temperature':"ºC", 
    'ESP discharge pressure':"Bar", 
    'Choke Opening':"%",
    'Well head pressure':"Bar",#
    'Well head Temperature' : "ºC",
    'Well aligned to Train A': "bool",
    'Well aligned to Train B': "bool",
    'ESP Motor Voltage': "V",
    'ESP Vibration X': "g",
    'ESP Vibration Y': "g",
    'Well Run': "str",
    'Well_down': "str",
    'Pump Info': "str",
    'Failure Info': "str",
    'Failure': "bool",
    'Current Mean': "A",
    "":"",
    "ESP motor Current - phase A":"A",
    "ESP motor Current - phase B":"A",
    "ESP motor Current - phase C":"A",
    'ESP discharge temperature sensor':'ºC',
    'ESP differential pressure':"Bar",
    "ESP Vibration Module":"g"
    }

    if Header in list(dicionario):
        return dicionario[Header]
    else:
        return Header


def FigureComponents(
        model       ,   #sklearn.transformation -> but this is not a class
        Headers     :   list, 
        plotName    :   str, 
        listOfNames =   ["PCA - Measures and Variance Components", "Correlation","PCA Components", "Original Data Features"],
        savePath    =   "../imagens_gerais/",
        english     =   True
        ): #-> Tuple[fig, axs]
    
    '''
        Plot the associated matrix from a transformation model. 

        Example:

        X = USV^t -> U is ploted
    '''

    transposedComponents = model.components_.T

    plt.figure(figsize=(10, 5))

    plt.imshow(transposedComponents, interpolation='nearest',aspect='auto',cmap='bwr')

    for i in range(transposedComponents.shape[0]):
        for j in range(transposedComponents.shape[1]):
            plt.text(j, i, f'{transposedComponents[i, j]:.2f}', ha='center', va='center', color='black',fontsize=10)

    
    plt.yticks(ticks=range(0,len(Headers)),labels=[Traducao(item,english) for item in Headers])
    
    
    plt.title(listOfNames[0],fontsize=16)
    plt.colorbar(label=listOfNames[1])
    plt.xlabel(listOfNames[2],fontsize=13)
    plt.ylabel(listOfNames[3],fontsize=13)
    plt.savefig(savePath+plotName, bbox_inches='tight')
    plt.tight_layout()
    return plt.gcf(), plt.gca()
    

   
#realiza os qq plots dos dados e das colunas que forem necessárias 
def QQPlots(
        data                :   pd.DataFrame | np.ndarray,
        relevantHeaders     :   list, 
        title               =   "QQ",
        lineType            =   "s", 
        english             =   True,
        titleFontsize       =   20,
        ydist               =   1,
        generalFontSize     =   15, 
        distO               =   norm,
        figSizeSt           =   (40,15),
        layout              =   (2,5)
        
        ): #-> Tuple[fig, axs]

    '''
        Generates QQ plot, measuring the desired data distribution relative to another distribution, with gaussian as default.
    '''

    plt.rcParams["font.size"]=generalFontSize
    n = len(relevantHeaders)
    # if not (n % 2):
    #     n = len(relevantHeaders) + 1


    
    fig, axs = plt.subplots(layout[0],layout[1],figsize=figSizeSt)    
    
    # n = np.ceil(np.sqrt(len(relevantHeaders))).astype(int)
    # fig, axs = plt.subplots(n,n,figsize=(20,20))

    fig.suptitle(title,fontsize=titleFontsize, y=ydist)    
    
    if n > 1:   
        axs = axs.ravel()
        i = 0
        for prop in relevantHeaders:
            sm.qqplot(data[prop], line=lineType,ax=axs[i],dist=distO)
            if not english:
                axs[i].title.set_text(Traducao(prop))
                axs[i].set_xlabel("Quantil teórico")
                axs[i].set_ylabel("Quantil da amostra")
            else:
                axs[i].title.set_text(prop)
            i +=1

        # for j in range(i,n*n):
        #     axs[j].remove()
        if (len(relevantHeaders) % 2):
            axs[n].remove()
    else:
         sm.qqplot(data, line=lineType,ax=axs)

    fig.tight_layout(pad=1.2)

    return fig,axs


def PumpPlot(
        pumpData    :   pd.DataFrame,
        Headers     :   list,
        axs         :   axes.Axes,
        titleS      =   "",
        english     =   False,
        legendTextSize = 20
        ):
    
    '''
        Given a pump data, returns the plot of properties listed in Headers.
    '''

    pumpData[Headers].plot(ax=axs)

    if titleS == "":
        axs.legend([Traducao(item,english) for item in Headers],loc='upper left',bbox_to_anchor=(1, 1),fontsize=legendTextSize)
    else:
        axs.legend([Traducao(item,english) for item in Headers],loc='upper left',bbox_to_anchor=(1, 1),fontsize=legendTextSize, title=titleS)


    if pumpData.loc[pumpData["Failure"]==True].shape[0] != 0:
        axs.axvline(pumpData.loc[pumpData["Failure"]==True].index[0], color='red', linestyle='--', linewidth=2)


def TimeSeriesColored(
        pumpData    :   pd.DataFrame,
        Headers     :   list,
        fig         :   figure.Figure,
        axs         :   axes.Axes
        ):
    
    '''
        Given a pump data, returns the color map of properties listed in Headers.
    '''

    its = axs.pcolor(pumpData[Headers].T,cmap='hsv', norm=norm)

    axs.grid(axis="y",linewidth=0.5,color="black")
    norm = mcolors.Normalize(vmin=-6, vmax=6)

    fig.colorbar(its,ax=axs,orientation='horizontal',shrink=0.5)
    if pumpData.loc[pumpData["Failure"]==True].shape[0] != 0:
        failureX = pumpData.index.get_loc(pumpData.loc[pumpData["Failure"]==True].index[0])
        
        axs.axvline(x=failureX, color='red', linestyle='--', linewidth=2)


def TimeSeriesL2(
        pumpData    : pd.DataFrame,
        Headers     : list,
        axs         : axes.Axes
        ):
    '''
        Given a pump data, returns the L2 Norm of properties listed in Headers.
    '''

    pumpData[Headers].pow(2).sum(axis=1).pow(1/2).plot(ax=axs)

    if pumpData.loc[pumpData["Failure"]==True].shape[0] != 0:
        axs.axvline(pumpData.loc[pumpData["Failure"]==True].index[0], color='red', linestyle='--', linewidth=2)


def OverFill(
        pumpData        : pd.DataFrame,
        Headers         : list,
        State           : str,
        n               : int,
        ax              : axes.Axes,
        chosenPallete   = 'Oranges',
        gnrlFont        = 20,
        english=True
        ):
    
    '''
        Fills the time-series with an transparent mask of each color representing one state. 
    '''

    cmap = plt.get_cmap(chosenPallete, n+1)
    if english:
        stateLabel = "State"
    else:
        stateLabel = "Estado"

    for state in range(0,n+1):
            color = cmap(state)  # Pega uma cor automática para cada estado
        

            ax.fill_between(pumpData.index,
                            np.min(pumpData[Headers]), 
                            np.max(pumpData[Headers]), 
                            where=(pumpData[State] == state), 
                            color=color, 
                            alpha=0.3, 
                            label=stateLabel + " " +str(state) 
                            )
            h, l = ax.get_legend_handles_labels()
            ax.legend(h,[Traducao(item) for item in l], loc='upper left',bbox_to_anchor=(1, 1),fontsize=gnrlFont)


def PlotHMMSeries(
        pumpData        : pd.DataFrame,
        axs             : axes.Axes,
        Headers         : list,
        states          : str,
        numberOfStates  : int,
        measures        : str,
        gnrlFont        = 20,
        english=False):
    
    '''
        Plot the time series of chosen headers and fill with the respective state colors.
    '''
    
    PumpPlot(pumpData,Headers,axs,english,legendTextSize=gnrlFont)
    
    OverFill(pumpData,Headers,states,numberOfStates,axs,english=english,gnrlFont=gnrlFont)

    axs.tick_params(axis='both',which="both", labelsize=str(gnrlFont))
    axs.set_xlabel("time",fontsize=gnrlFont)
    axs.set_ylabel(measures,fontsize=gnrlFont)
        
    

def GaussianMixturePlot(
        data    :   np.ndarray,
        gmm     :   GaussianMixture,
        strings :   list,
        limits  =   (0,17),
        figsize =   (7,5),
        path    =   "../imagens_gerais/"
        ):
    
    '''
        Generates the Gaussian Mixture Model plot on provided data.
    '''
    model = gmm
    means = model.means_.flatten()
    stds = np.sqrt(model.covariances_).flatten()
    weights = model.weights_

    # Criando um range de valores para plotar as distribuições
    x = np.linspace(limits[0], np.max(data), 1000)


    fig = plt.figure(figsize=figsize)
    # Plotando histograma dos dados originais
    plt.hist(data, bins=100, density=True, alpha=0.5, label=strings[0])
    plt.xlim(limits[0],limits[1])

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
    plt.savefig(path + "gmm_"+data.name+".jpg",bbox_inches='tight')
    plt.show()


    return gmm, fig

def Histogram(
        data        :   pd.DataFrame,
        title       :   str,
        binsN       =   50,
        figsizeT    =   (20,18),
        english     =   False,
        layout      =   (2,5)
        ):
    '''
        Plots the histogram of provided data. It uses hist as base, but it adapted to translation
    '''


    axes = data.hist(bins=binsN,figsize=figsizeT,layout=layout)

    i = 0

    if english:
        xlabel_ = "Values"
        ylabel_ = "Count"
    else:
        xlabel_ = "Valores"
        ylabel_ = "Contagem"

    for ax in axes.flatten():
        prop = ax.get_title()
        ax.set_title(Traducao(prop,english))
        ax.set_xlabel(xlabel_ + " ["+ Measures(prop) +"]" )  # Define a legenda com o nome da coluna
        ax.set_ylabel(ylabel_)
        i += 1


    plt.suptitle(title,fontsize=20)
    plt.tight_layout(pad=1.3)

    #isso não é bom, mas tbm não encontrei outra solução
    return plt.gcf(), plt.gca()



# Functions in the scope of the thesis - you might not want to use them:

def ComparisionPlot(originalData,newData,title,newHeaders,originalHeaders, 
                    english=True,
                    figsizeT=(20,10),
                    padF=1.08,
                    generalFontSize = 15):

    plt.rcParams['font.size'] = generalFontSize
    plt.rcParams.update({'font.size': plt.rcParams['font.size']})
    fig, axs = plt.subplots(2,1, figsize=figsizeT,sharex=True)

       
    
    fig.suptitle(title,fontsize=generalFontSize*1.5);

    if not english:
        PumpPlot(originalData,originalHeaders,axs[0],"Dados Padronizados",legendTextSize=generalFontSize)
        PumpPlot(newData,newHeaders,axs[1],"Dados Transformados",legendTextSize=generalFontSize)
    else:
        PumpPlot(originalData,originalHeaders,axs[0],"Standardized Data",legendTextSize=generalFontSize)
        PumpPlot(newData,newHeaders,axs[1],"Transformed Data",legendTextSize=generalFontSize)    
    
    
    plt.tight_layout(pad=padF)

    
    
    return fig,axs


def ZScorePlot(totalData,pump,Headers,english = True):
    pumpData =  totalData.loc[totalData["Well Run"]==pump]

    fig, axs = plt.subplots(1,1,figsize=(20,7))
    PumpPlot(pumpData,Headers,axs)
    
    if not english:
        axs.set_xlabel("Tempo")
        fig.suptitle("Gráfico do Z-score " + pump,fontsize=20);
    else:
        fig.suptitle("Modified Z-score data of " + pump,fontsize=20);           

        
    return fig,axs

#this function is deprecated, but maintained, since there is some notebook that might use it.
def PCAComponentsPlot(pumpData,pump,Headers):

    fig, axs = plt.subplots(3,1, figsize=(40,17))

    PumpPlot(pumpData,Headers,axs[0])
    TimeSeriesColored(pumpData,Headers,fig,axs[1])
    TimeSeriesL2(pumpData,Headers,axs[2])
    
    fig.suptitle("PCA Data of " + pump,fontsize=20);
    
    return fig,axs

def HMMPicture(
        pumpData,
        pump,
        Headers,
        states, 
        numberOfStates,
        measures, 
        figsize=(60,30),
        #posLeg = 0.5,
        english=True,
        titleFont = 25,
        gnrlFont = 20
        ):

    fig, axs = plt.subplots(1,1, figsize=figsize,sharex=True)

    PlotHMMSeries(pumpData,axs,Headers,states,numberOfStates,measures,english=english,gnrlFont=gnrlFont)
    
    # for i in range(0,n_g):
    #     pass
        
    fig.suptitle("HMM: " + pump,fontsize=titleFont);
    #plt.figtext(0.5, posLeg + 0.02, pumpData["Pump Info"].iloc[0],fontsize=10,va="center",ha="center")
    #plt.figtext(0.5, posLeg, pumpData["Failure Info"].iloc[0],fontsize=10,va="center",ha="center")


    plt.tight_layout()
    return fig,axs




#GPTed codes - they might be useful, but were not used:

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
    fig.savefig("../imagens_gerais/multiple_gmm.jpeg",bbox_inches='tight')



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


#directed imported from analysis.py
def CorrGraphGen(corrAnalysis,plotHeaders,pump):
    dataCorrelation = corrAnalysis[plotHeaders].corr(method='pearson')

    mask = np.triu(np.ones_like(dataCorrelation, dtype=bool)) # Generate a mask for the upper triangle
    f, ax = plt.subplots(figsize=(11, 9)) # Set up the matplotlib figure
    cmap = sns.diverging_palette(230, 20, as_cmap=True) # Generate a custom diverging colormap
    sns.heatmap(dataCorrelation, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5}) # Draw the heatmap with the mask and correct aspect ratio

   

    #plt.savefig("imagens/comFiltro/correlacoes/corr_"+pump+".png")
    plt.savefig("imagens/PCAcorr/corr_"+pump+".jpg",bbox_inches='tight')
    plt.close() 
