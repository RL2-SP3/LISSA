import pandas as pd
from matplotlib import pyplot as plt

from matplotlib import axes
import numpy as np

import statsmodels.api as sm

from scipy.stats.distributions import norm

from math import ceil, sqrt

from pathlib import Path

import json



class LissaFigure:

    DEFAULT_FIG_PARAMS = {
            "fig_size"          :   (20,10),
            "text_size"         :   10,
            "title_size"        :   16,
            "tick_size"         :   10,
            "label_size"        :   10,
            "color_pallette"    :   "Oranges",
            "alpha"             :   0.3,
            "layout"            :   (2,5),
            "y_dist"            :   1,
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
    
    DEFAULT_QQ_PARAMS = {
            "line_type"         :   "s"
    }

    DEFAULT_HIST_PARAMS = {
            "bins"              :   20
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

    def __init__(
            self,
            **kwargs
            ):
        
        self.params = {
            **self.DEFAULT_FIG_PARAMS,
            **self.DEFAULT_LABEL_PARAMS,
            **self.DEFAULT_SAVE_PARAMS,
            **self.DEFAULT_DICTS_PARAMS,
            **self.DEFAULT_QQ_PARAMS,
            **self.DEFAULT_HIST_PARAMS,
            **kwargs
            }
        
        self.measures = {}

        #dictionaries_dir = Path(__file__).resolve().parent / "dictionaries"
        
        self.dict_of_dicts = self._load_dict(self.params["dict_path"])

        headers = self._load_dict(self.params["headers_path"])
        
        self.numerical_properties = headers["numerical_headers"]
        self.non_numerical_properties = headers["non_numerical_headers"]       

        self.numerical_properties_translated = self.numerical_properties.copy()
        self.non_numerical_properties_translated = self.non_numerical_properties.copy()


    def _load_dict(self,path):
        with (path).open() as f:
            return json.load(f)
    
    def __repr__(self):
        return f"LissaFigure({self.params})"
    
    def load_parameters(self,file_path):
        dictionary = self._load_dict(Path(file_path))

        self.params.update(json.load(dictionary))

    def _mapping(self,dict_name, headers):
        if dict_name not in self.dict_of_dicts:
            print(f"Dictionary {dict_name} not found. Returning empty dict!")
            return {}
        else: 
            return {
                prop:self.dict_of_dicts[dict_name][prop] 
                for prop in headers 
                if prop in self.dict_of_dicts[dict_name]
                }
        
    def set_translation(self):
        self.numerical_properties_translated = self._mapping(self.params["portuguese_dictionary"],self.numerical_properties)
        self.non_numerical_properties_translated = self._mapping(self.params["portuguese_dictionary"],self.non_numerical_properties)

    def set_measures(self):
        self.measures = self._mapping(self.params["units"],self.numerical_properties)      
            
    def save_fig(self):
        if self.params["save_figure"]:
            self.figure.savefig(Path(self.params["dir"]) / self.params["file_name"], bbox_inches='tight')
        else:
            print("This figure was not saved!")
    
    def set_figure(self):
            
        self.figure = plt.figure(figsize=self.params["fig_size"])


    def matrix_plot(
            self,
            data : np.ndarray
            ):
        
        
        self.ax = self.figure.add_subplot(1,1,1)
        self.ax.imshow(data, interpolation='nearest',aspect='auto',cmap='bwr')

        n,m = self.data.shape

        for i in range(n):
            for j in range(m):
                plt.text(j, i, f'{data[i, j]:.2f}', 
                        ha='center', 
                        va='center', 
                        color='black',
                        fontsize=self.params["text_size"]
                        )
    
        self.ax.set_yticks(
            ticks=range(n),
            labels=[prop for prop in self.numerical_properties_translated],
            fontsize=self.params["tick_size"])
        
        self.ax.set_xticks(
            ticks=range(m),
            labels=range(m),
            fontsize=self.params["tick_size"])
        
        self.figure.colorbar(label=self.params["colorbar_title"])

        self._set_axes_texts(self.ax,title=None)
        self._finalize()
        
    #TODO
    def histogram(self):

        self.axs = self.data[self.numerical_properties].hist(
            bins=self.params["bins"],
            figsize=self.params["fig_size"],
            layout=self.params["layout"]
            )

        for index,ax in enumerate(self.axs.flatten()):
            self._set_axes_texts(
                ax,
                title=self.numerical_properties_translated[index],
                measure_X=self.measures[index]
                )
        
        self._finalize()
         

    def quantile_quantile_plot(self,distribution=norm):

        self.figure, self.axs = plt.subplots(
            self.params["layout"][0],
            self.params["layout"][1],
            figsize=self.params["fig_size"])
        
        
        self.axs = self.axs.ravel()
        for index,ax in enumerate(self.axs):
            sm.qqplot(
                self.data[self.numerical_properties[index]], 
                line=self.params["line_type"],
                ax=ax,
                dist=distribution
                )
            self._set_axes_texts(index,self.numerical_properties_translated[index])
    
        n = len(self.numerical_properties)    
        if (n > 1) and (n % 2):   
            self.axs[n-1].remove()
        
        self._finalize()

    def time_series_plot(self):
        self.ax = self.figure.add_subplot(1,1,1)
        self.data[self.numerical_properties].plot(ax=self.ax)

        self.ax.legend(
            self.numerical_properties_translated,
            loc='upper left',
            bbox_to_anchor=(1, 1),
            fontsize = self.params["text_size"],
            title = self.params["legend_title"]
            )
        
        self._set_axes_texts(self.ax)
        self._finalize()

    def failure_reference(self,time_entry):
        
        if hasattr(self,"ax"):
            self.ax.axvline(
                time_entry, 
                color='red', 
                linestyle='--', 
                linewidth=2
                )
        else:
            raise ReferenceError("No ax set!")
                
    def classification_boundaries(self,n,state_name):
        cmap = plt.get_cmap(self.params["color_pallette"], n+1)

                
        for state in self.data[state_name].unique():
            color = cmap(state)  # Pega uma cor automática para cada estado
            data_numpy = self.data[self.numerical_properties].to_numpy()
        
            self.ax.fill_between(
                self.data.index,
                data_numpy.min(),
                data_numpy.max(),
                where=(self.data[state_name] == state),
                color=color,
                alpha=self.params["alpha"],
                label= state_name + " " +str(state)
                )           
            
            # h, l = ax.get_legend_handles_labels()
            # ax.legend(h,[Traducao(item, english) for item in l], loc='upper left',bbox_to_anchor=(1, 1),fontsize=gnrlFont)
    
    
    def _set_axes_texts(self,ax,title = None,measure_X=None,measure_Y=None):
        x_label =  self.params["x_label"]
        y_label =  self.params["y_label"]
        
        if title:
            ax.set_title(title)
        if measure_X:
            x_label = self.params["x_label"] + "[" + measure_X + "]"
        if measure_Y:
            y_label = self.params["y_label"] + "[" + measure_X + "]"
        
        ax.set_xlabel(x_label,fontsize=self.params["title_size"])
        ax.set_ylabel(y_label,fontsize=self.params["title_size"])


    def _finalize(self):
        self.figure.suptitle(self.params["plot_title"],fontsize=self.params["title_size"],y=self.params["y_dist"])
        self.figure.tight_layout(pad=self.params["tight_layout_pad"])
        self.figure.set_dpi(self.params["dpi"])
