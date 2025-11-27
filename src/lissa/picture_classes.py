import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.cm as cm

from matplotlib.axes import Axes

import numpy as np

from statsmodels.api import qqplot
from scipy.stats.distributions import norm
from math import ceil, sqrt
from pathlib import Path

import json

import warnings




class LissaFigure:

    DEFAULT_RCPARAMS = {
            "figure.figsize":      (20, 10),
            "font.size":           10,
            "figure.titlesize":    16,
            "xtick.labelsize":     10,
            "ytick.labelsize":     10,
            "axes.labelsize":      10,
            "figure.dpi":          600
}

    DEFAULT_FIG_PARAMS = {
            "color_pallette"    :   "Oranges",
            "alpha"             :   0.3,
            "layout"            :   (2,5),
            "y_dist"            :   1,
            "tight_layout_pad"  :   1,
            "color_scale"       :   "tab10",
            "log_scale_y"       :   False,
            "log_scale_x"       :   False
        }
    
    DEFAULT_PLOT_PARAMS = {
            "x_label"           :   "X Label",
            "y_label"           :   "Y Label",
            "plot_title"        :   "Title of the Plot",
            "colorbar_title"    :   "Colorbar title",
            "legend_title"      :   "Legend title",
            "line_type"         :   "s",
            "bins"              :   20,
            "state_name"        :   "State"
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
            path = None,
            **kwargs
            ):
        
        self.params = {
            **self.DEFAULT_FIG_PARAMS,
            **self.DEFAULT_SAVE_PARAMS,
            **self.DEFAULT_DICTS_PARAMS,
            **kwargs
            }
        
        plt.rcParams.update(self.DEFAULT_RCPARAMS)
        
        if path is not None:
            dictionary = self._load_dict(path)
            self.params.update(dictionary["li_params"])
            plt.rcParams.update(dictionary["rc_params"])
               
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
            self.params["layout"][1])
        
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
                        va='center'
                        )
    
        ax.set_yticks(ticks=range(n),labels=[self.numerical_properties_translated[prop] for prop in data.index])
        
        ax.set_xticks(ticks=range(m),labels=range(m))
        
        self.figure.colorbar(ax.images[0], ax=ax, label=self.params["colorbar_title"])

        return self
        
    def histogram(self,data, limits=None):
        for index, ax in enumerate(self.axs.ravel()):

            prop = self.numerical_properties[index]
            translated_prop = self.numerical_properties_translated[prop]


            data[prop].hist(
                bins=self.params["bins"],
                ax=ax
                )
            
            if limits is not None:
                ax.set_xlim(limits)
            
            self.set_axes_texts(
                ax,
                title=translated_prop,
                measure_X=self.measures[prop]
                ) 
            
        return self

    def quantile_quantile_plot(self,data,distribution=norm):       
        
        view_of_axes = self.axs.ravel()

        for index,ax in enumerate(view_of_axes):
            prop = self.numerical_properties[index]
            translated_prop = self.numerical_properties_translated[prop]

            qqplot(
                data[prop], 
                line=self.params["line_type"],
                ax=ax,
                dist=distribution
                )
            self.set_axes_texts(ax,translated_prop)
    
        n = len(self.numerical_properties)    
        if (n > 1) and (n % 2):   
            self.axs[n-1].remove()

        return self
    

    def time_series_plot(self,data : pd.DataFrame,index=None):
        cmap = plt.get_cmap(self.params["color_scale"])
        plt.rcParams["axes.prop_cycle"] = plt.cycler(color=cmap(np.linspace(0,1,len(self.numerical_properties))))

        ax = self._select_ax_to_plot(index)

        lines = data[self.numerical_properties].plot(
            ax=ax,
            logy=bool(self.params["log_scale_y"]),
            sharex=True
            )

        translated = [
            self.numerical_properties_translated[c] 
            for c in self.numerical_properties
]

        leg = ax.legend(
            handles = lines.lines,
            labels = translated,
            loc='upper left',
            bbox_to_anchor=(1, 1)
            )
        
        leg.set_title(self.params["legend_title"])
        
        
        self.set_axes_texts(ax)
        self.finalize()

        self.time_flag = True

        return self


    def failure_reference(
            self,
            time_entry,
            linestyle="--",
            linewidth=2,
            index=None):
        if not hasattr(self,"time_flag"):
            raise ReferenceError("No time-series plot!")

        if hasattr(self,"figure"):
            ax = self._select_ax_to_plot(index)
            ax.axvline(
                time_entry,
                linestyle = linestyle, 
                linewidth=linewidth)
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
    
    
    def set_axes_texts(self,entry,title = None,measure_X=None,measure_Y=None):
        if isinstance(entry,Axes):
            ax = entry
        else:
            ax = self.axs.ravel()[entry]
        
        if title:
            ax.set_title(title)
                
        x_label =  self.params["x_label"] + (" [" + str(measure_X) + "]") * (measure_X is not None)
        y_label =  self.params["y_label"] + (" [" + str(measure_Y) + "]") * (measure_Y is not None)
                
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

        return self


    def finalize(self):
        self.figure.suptitle(self.params["plot_title"],y=self.params["y_dist"])
        self.figure.tight_layout(pad=self.params["tight_layout_pad"])

        return self