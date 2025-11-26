import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.cm as cm

import numpy as np

from statsmodels.api import qqplot
from scipy.stats.distributions import norm
from math import ceil, sqrt
from pathlib import Path

import json

import warnings




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
            "dpi"               :   600,
            "color_scale"       :   "tab10",
            "log_scale_y"       :   False,
            "log_scale_x"       :   False
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
    
    DEFAULT_FAILURE_PARAMS = {
            "linewidth"             :   2,
            "linestyle"             :   '--',
            "linecolor"             :   "red",
            "state_name"            :   "State"
    }

    def __init__(
            self,
            path = None,
            **kwargs
            ):
        
        self.params = {
            **self.DEFAULT_FIG_PARAMS,
            **self.DEFAULT_LABEL_PARAMS,
            **self.DEFAULT_SAVE_PARAMS,
            **self.DEFAULT_DICTS_PARAMS,
            **self.DEFAULT_QQ_PARAMS,
            **self.DEFAULT_HIST_PARAMS,
            **self.DEFAULT_FAILURE_PARAMS,
            **kwargs
            }
        
        if path is not None:
            dictionary = self._load_dict(path)
            self.params.update(dictionary)
               
        self.dict_of_dicts = self._load_dict(self.params["dict_path"])

        headers = self._load_dict(self.params["headers_path"])
        
        self.numerical_properties = headers["numerical_headers"]
        self.non_numerical_properties = headers["non_numerical_headers"]       

        self.numerical_properties_translated = self.numerical_properties.copy()
        self.non_numerical_properties_translated = self.non_numerical_properties.copy()


    def _load_dict(self,path):
        with (Path(path)).open() as f:
            return json.load(f)
    
    def __repr__(self):
        return f"LissaFigure({self.params})"

    def _mapping(self,dict_name, headers):
        if dict_name not in self.dict_of_dicts:
            warnings.warn(f"Dictionary {dict_name} not found. Returning empty dict!")
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
        return self

    def set_measures(self,no_scale=False):
        if not no_scale:
            self.measures = self._mapping(self.params["units"],self.numerical_properties)
        else:
            self.measures = {prop:"-" for prop in self.numerical_properties}
        
        return self
            
    def save_fig(self):
        if bool(self.params["save_figure"]) and hasattr(self,"figure"):
            self.figure.savefig(Path(self.params["dir"]) / self.params["file_name"], bbox_inches='tight')
        else:
            warnings.warn("This figure was not saved!")

        return self
    
    def set_figure(self):
        if hasattr(self,"figure"):
            plt.close(self.figure)
        
        self.figure, self.axs = plt.subplots(
            self.params["layout"][0],
            self.params["layout"][1],
            figsize=self.params["fig_size"])
        
        return self
        

    def _select_ax_to_plot(self,index=None):
        if index == None:
            return self.axs
        else:
            return (self.axs.ravel())[index]
    

    def matrix_plot(
            self,
            data : pd.DataFrame,
            index=None
            ):
        
        ax = self._select_ax_to_plot(index)
        
        ax.imshow(data, interpolation='nearest',aspect='auto',cmap='bwr')

        n,m = data.shape

        for i in range(n):
            for j in range(m):
                ax.text(j, i, f'{data.iloc[i, j]:.2f}', 
                        ha='center', 
                        va='center', 
                        color='black',
                        fontsize=self.params["text_size"]
                        )
    
        ax.set_yticks(
            ticks=range(n),
            labels=[self.numerical_properties_translated[prop] for prop in data.index],
            fontsize=self.params["tick_size"])
        
        ax.set_xticks(
            ticks=range(m),
            labels=range(m),
            fontsize=self.params["tick_size"])
        
        self.figure.colorbar(ax.images[0], ax=ax, label=self.params["colorbar_title"])

        self._set_axes_texts(ax,title=None)
        self._finalize()
        
    def histogram(self,data):
        for index, ax in enumerate(self.axs.ravel()):

            prop = self.numerical_properties[index]
            translated_prop = self.numerical_properties_translated[prop]


            data[prop].hist(
                bins=self.params["bins"],
                figsize=self.params["fig_size"],
                layout=self.params["layout"],
                ax=ax
                )
            
            self._set_axes_texts(
                ax,
                title=translated_prop,
                measure_X=self.measures[prop]
                )
        
        self._finalize()
         

    def quantile_quantile_plot(self,data,distribution=norm):       
        
        for index,ax in enumerate(self.axs.ravel()):
            prop = self.numerical_properties[index]
            translated_prop = self.numerical_properties_translated[prop]

            qqplot(
                data[prop], 
                line=self.params["line_type"],
                ax=ax,
                dist=distribution
                )
            self._set_axes_texts(ax,translated_prop)
    
        n = len(self.numerical_properties)    
        if (n > 1) and (n % 2):   
            self.axs[n-1].remove()
        
        self._finalize()

    def time_series_plot(self,data : pd.DataFrame,index=None):
        cmap = plt.get_cmap(self.params["color_scale"])
        plt.rcParams["axes.prop_cycle"] = plt.cycler(color=cmap(np.linspace(0,1,len(self.numerical_properties))))

        ax = self._select_ax_to_plot(index)

        lines = data[self.numerical_properties].plot(
            ax=ax,
            logy=bool(self.params["log_scale_y"]),
            fontsize=self.params["text_size"])

        translated = [
            self.numerical_properties_translated[c] 
            for c in self.numerical_properties
]

        leg = ax.legend(
            handles = lines.lines,
            labels = translated,
            loc='upper left',
            bbox_to_anchor=(1, 1),
            fontsize = self.params["text_size"],
            )
        
        leg.set_title(self.params["legend_title"], prop={"size": self.params["text_size"]})
        
        
        self._set_axes_texts(ax)
        self._finalize()

        self.time_flag = True

        return self


    def failure_reference(self,time_entry,index=None):
        if not hasattr(self,"time_flag"):
            raise ReferenceError("No time-series plot!")

        if hasattr(self,"figure"):
            ax = self._select_ax_to_plot(index)
            ax.axvline(
                time_entry, 
                color       =   self.params["linecolor"],
                linestyle   =   self.params["linestyle"],
                linewidth   =   self.params["linewidth"]
                )
        else:
            raise ReferenceError("No figure set!")
        
        return self
                
    def classification_boundaries(self, n, states_vector, index=None):

        if not hasattr(self, "time_flag"):
            raise ReferenceError("No time-series plot!")

        ax = self._select_ax_to_plot(index)

        cmap = plt.get_cmap(self.params["color_pallette"], n + 1)

        
        lines = ax.get_lines()
        
        if not lines:
            raise RuntimeError("No data plotted on this axis; cannot infer x-axis.")

        x = lines[0].get_xdata()

        
        ymin, ymax = ax.get_ylim()

        
        states = states_vector.to_numpy()

        for state in np.unique(states):
            color = cmap(state)
            mask = (states == state)

            ax.fill_between(
                x,
                ymin,
                ymax,
                where=mask,
                color=color,
                alpha=self.params["alpha"],
                label=self.params["state_name"] + f" {state}"
            )

            # h, l = ax.get_legend_handles_labels()
            # ax.legend(h,[Traducao(item, english) for item in l], loc='upper left',bbox_to_anchor=(1, 1),fontsize=gnrlFont)

        return self
    
    
    def _set_axes_texts(self,ax,title = None,measure_X=None,measure_Y=None):
        
        if title:
            ax.set_title(title)
                
        x_label =  self.params["x_label"] + ("[" + str(measure_X) + "]") * (measure_X is not None)
        y_label =  self.params["y_label"] + ("[" + str(measure_Y) + "]") * (measure_Y is not None)
                
        ax.set_xlabel(x_label,fontsize=self.params["label_size"])
        ax.set_ylabel(y_label,fontsize=self.params["label_size"])


    def _finalize(self):
        self.figure.suptitle(self.params["plot_title"],fontsize=self.params["title_size"],y=self.params["y_dist"])
        self.figure.tight_layout(pad=self.params["tight_layout_pad"])
        self.figure.set_dpi(self.params["dpi"])