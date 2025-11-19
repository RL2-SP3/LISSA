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

from pathlib import Path

import json


class LissaFigure():

    DEFAULT_FIG_PARAMS = {
            "fig_size"          :   (20,10),
            "text_size"         :   10,
            "title_size"        :   16,
            "tick_size"         :   10,
            "label_size"        :   10,
            "color_pallette"    :   "Oranges",
            "alpha"             :   0.3
        }
    
    DEFAULT_LABEL_PARAMS = {
            "x_label"           :   "X Label",
            "y_label"           :   "Y Label",
            "plot_title"        :   "Title of the Plot",
            "colorbar_title"    :   "Colorbar title",
            "legend_title"      :   "Legend title"
            }
    
    DEFAULT_SAVE_PARAMS = {
            "save_figure"       :    False,
            "dir"               :    "./",
            "file_name"         :    "image"            
        }
    
    DEFAULT_DICTS_PARAMS ={
            "portuguese_dictionary" :   "translation_to_portuguese",
            "units"                 :   "measurement_units"
        }
    
    params = {
            **DEFAULT_FIG_PARAMS,
            **DEFAULT_LABEL_PARAMS,
            **DEFAULT_SAVE_PARAMS,
            **DEFAULT_DICTS_PARAMS
            }
    

    def __init__(
            self,
            data,
            properties,
            **kwargs
            ):
        
        self.params = {**self.params, **kwargs}
        self.properties = properties
        self.properties_translated = properties.copy()
        self.data = data
        self.measures = {}

        dictionaries_dir = Path(__file__).resolve().parent / "dictionaries"
        
        with (dictionaries_dir/"dictionaries.json").open() as dictionary:
            self.dict_of_dicts = json.load(dictionary)

    def plot(self):
        self.figure = plt.figure(figsize=self.params["fig_size"])
        
    def load_parameters(self,file_path):
        with Path(file_path).open() as dictionary:
            self.params.update(json.load(dictionary))

    def _mapping(self,dict_name):
        if dict_name not in self.dict_of_dicts:
            print(f"Dictionary {dict_name} not found. Returning empty dict!")
            return {}
        else: 
            return {
                prop:self.dict_of_dicts[dict_name][prop] 
                for prop in self.properties 
                if prop in self.dict_of_dicts[dict_name]
                }
        
    def translate(self,dict_name):
        self.properties_translated = [
                self.dict_of_dicts[dict_name][prop] 
                for prop in self.properties 
                if prop in self.dict_of_dicts[dict_name]
                ]

    def set_measures(self):
        self.measures = self._mapping(self.params["units"])      
            
    def save_fig(self):
        if self.params["save_figure"]:
            plt.savefig(Path(self.params["dir"]) / self.params["file_name"], bbox_inches='tight')
        else:
            print("This figure was not saved!")

    def __repr__(self):
        return f"LissaFigure(properties={self.properties}, fig_size={self.params['fig_size']})"
        


class MatrixPlot(LissaFigure):
    def plot(self):
        super().plot()
        self.ax = self.figure.add_subplot(1,1,1)
        self.ax.imshow(self.data, interpolation='nearest',aspect='auto',cmap='bwr')

        n,m = self.data.shape

        for i in range(n):
            for j in range(m):
                plt.text(j, i, f'{self.data[i, j]:.2f}', 
                        ha='center', 
                        va='center', 
                        color='black',
                        fontsize=self.params["text_size"]
                        )
    
        plt.yticks(
            ticks=range(0,n),
            labels=[prop for prop in self.properties_translated],
            fontsize=self.params["tick_size"])
        
        plt.xticks(
            ticks=range(0,len(m)),
            labels=range(0,len(m)),
            fontsize=self.params["tick_size"])
        
        
        plt.title(self.params["title"],fontsize=self.params["title_size"])

        plt.colorbar(label=self.params["colorbar_title"])

        plt.xlabel(self.params["x_label"],fontsize=self.params["title_size"])
        plt.ylabel(self.params["y_label"],fontsize=self.params["title_size"])

        plt.tight_layout()

        return plt.gcf(), plt.gca()




class QuantileQuantilePlot(LissaFigure):
    '''
    Generates QQ plot, measuring the desired data distribution relative to another distribution, with gaussian as default.
    '''

    def __init__(self,
                 data,
                 line_type = "s",
                 layout = (2,5),
                 y_dist = 1                 
                 ):
        super().__init__()
        
        self.figure, self.axs = plt.subplots(
            layout[0],
            layout[1],
            figsize=self.params["fig_size"])
        
        self.figure.suptitle(
            self.params["plot_title"],
            fontsize=self.params["text_size"], 
            y=y_dist)
        
        self.n = len(data.columns)

        self.data = data

        self.line_type = line_type
        self.dist_to_title_foot = y_dist

    def plot(self):        
        if self.n > 1:   
            self.axs = self.axs.ravel()
            for index,prop in enumerate(self.properties):
                sm.qqplot(self.data[prop], line=self.line_type,ax=self.axs[index],dist=self.dist_to_title_foot)
                self.axs[index].title.set_text(self.properties_translated[index])
                self.axs[index].set_xlabel(self.params["x_label"])
                self.axs[index].set_ylabel(self.params["y_label"])
                
            if (len(self.properties) % 2):
                self.axs[self.n].remove()
        else:
            sm.qqplot(self.data, line=self.lineType,ax=self.axs)

        self.figure.tight_layout(pad=1.2)

        plt.show()

class GaussianMixturePlot(LissaFigure):
    def __init__(
            self,
            data,
            model
            ):
        super().__init__()
        self.data = data
        self.model = model
        self.stds = np.sqrt(model.covariances_).flatten()
        self.weights = model.weights_

        self.params["histogram_name"] = "Histogram"
        self.params["gaussian_name"] = "Gaussian"

    def plot(
            self,
            limits  =   (0,17)
            ):
        
        # Criando um range de valores para plotar as distribuições
        x = np.linspace(limits[0], np.max(self.data), 1000)


        # Plotando histograma dos dados originais
        self.ax = self.figure.add_subplots(1,1,1)
        self.ax.hist(self.data, bins=100, density=True, alpha=0.5, label=self.params["histogram_name"])
        self.ax.xlim(limits[0],limits[1])

        # Plotando cada gaussiana individualmente
        for i in range(self.model.means_.shape[0]):
            self.ax.plot(x, self.weights[i] * norm.pdf(x, self.means[i], self.stds[i]), label=self.params["gaussian_name"] + f"{i+1}")

        self.figure.legend()
        self.figure.title(self.params["plot_title"],fontsize=self.params["title_size"])
        self.ax.xlabel(self.params["x_label"])
        self.ax.ylabel(self.params["y_label"])
        self.figure.show()


class EquipmentPlot(LissaFigure):
    '''
        Given a pump data, returns the plot of properties listed in Headers.
    '''
    def __init__(self,equipment_data):
        super().__init__()
        self.ax = self.figure.add_subplot(1,1,1)
        equipment_data[self.properties].plot(ax=self.ax)

        self.ax.legend(
            self.properties_translated,
            loc='upper left',
            bbox_to_anchor=(1, 1),
            fontsize = self.params["font_size"],
            title = self.params["legend_title"]
            )
        
        self.data = equipment_data
    
    def failure_reference(self,time_entry):
        self.ax.axvline(
            time_entry, 
            color='red', 
            linestyle='--', 
            linewidth=2
            )
        
    def classification_boundaries(self,n,state_name):
        cmap = plt.get_cmap(self.params["color_pallette"], n+1)

                
        for state in self.data[state_name].unique():
            color = cmap(state)  # Pega uma cor automática para cada estado
        
            self.ax.fill_between(
                self.equipment_data.index,
                np.min(self.data[self.properties]),
                np.max(self.data[self.properties]),
                where=(self.data[state_name] == state),
                color=color,
                alpha=self.params["alpha"],
                label= state_name + " " +str(state)
                )
            
            
            # h, l = ax.get_legend_handles_labels()
            # ax.legend(h,[Traducao(item, english) for item in l], loc='upper left',bbox_to_anchor=(1, 1),fontsize=gnrlFont)



class Histogram(LissaFigure):
    def __init__(self,data,**kwargs):
        super().__init__()
        self.data = data
        self.params.update(kwargs)
    def plot(self):

        axes = self.data.hist(
            bins=self.bins,
            figsize=self.params["fig_size"],
            layout=self.params["layout"]
            )

        for index,ax in enumerate(axes.flatten()):
            ax.set_title(self.properties_translated[index])
            ax.set_xlabel(self.params["x_label"] + " ["+ self.measures[index] +"]" )
            ax.set_ylabel(self.params["y_label"])
            
        plt.suptitle(self.params["plot_title"],fontsize=20)
        plt.tight_layout(pad=1.3)

        #isso não é bom, mas tbm não encontrei outra solução
        return plt.gcf(), plt.gca()




