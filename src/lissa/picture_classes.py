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


from sklearn.mixture import GaussianMixture


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
            "log_scale_x"       :   False,
            "colormap"          :   "bwr"
        }
    
    DEFAULT_PLOT_PARAMS = {
            "x_label"           :   "X Label",
            "y_label"           :   "Y Label",
            "plot_title"        :   "Title of the Plot",
            "colorbar_title"    :   "Colorbar title",
            "legend_title"      :   "Legend title",
            "line_type"         :   "s",
            "bins"              :   20,
            "state_name"        :   "State",
            "text_color"        :   "black",
            "failure_line_color":   "red"
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
            **self.DEFAULT_PLOT_PARAMS,
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
            
    def save_fig(self,path=None):
        if hasattr(self,"figure"):
            if path == None:
                self.figure.savefig(self.params["dir"]+ self.params["file_name"], bbox_inches='tight')
            else:
                self.figure.savefig(path, bbox_inches='tight')
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
    
    def set_legends(self,index=None,legend_title=None,columns=None):
        ax = self._select_ax_to_plot(index)
        
        h, l = ax.get_legend_handles_labels()

        if columns == None:
            columns = [
                self.numerical_properties_translated.get(c,c)
                for c in l 
                ]

        leg = ax.legend(
            handles = h,
            labels = columns,
            loc='upper left',
            bbox_to_anchor=(1, 1)
            )
        
        if legend_title == None:
            leg_title = self.params["legend_title"]
        else:
            leg_title = legend_title

        leg.set_title(leg_title)

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
        
        ax.imshow(data, interpolation='nearest',aspect='auto',cmap=self.params["color_scale"])

        n,m = data.shape

        for i in range(n):
            for j in range(m):
                ax.text(j, i, f'{data.iloc[i, j]:.2f}', 
                        ha='center', 
                        va='center',
                        color=self.params["text_color"]
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

    def quantile_quantile_plot(
            self,
            data,
            distribution=norm, 
            columns = None
            ):       
        
        view_of_axes = self.axs.ravel()

        if columns == None:
            columns_to_plot = self.numerical_properties
            translation = self.numerical_properties_translated
        else:
            columns_to_plot = columns

        for index,ax in enumerate(view_of_axes):
            prop = columns_to_plot[index]
            
            qqplot(
                data[prop], 
                line=self.params["line_type"],
                ax=ax,
                dist=distribution
                )
            
            if columns is None:
                translated_prop = translation[prop]
                self.set_axes_texts(entry=ax,title=translated_prop)
            else:
                self.set_axes_texts(entry=ax,title=prop)
    
        n = len(columns_to_plot)    
        if (n > 1) and (n % 2):   
            self.axs[n-1].remove()

        return self
    

    def time_series_plot(
            self,
            data : pd.DataFrame | pd.Series ,
            index=None,
            columns=None,
            line_color = None,
            line_width = 0.7

            ):
        cmap = plt.get_cmap(self.params["color_scale"])
        plt.rcParams["axes.prop_cycle"] = plt.cycler(color=cmap(np.linspace(0,1,len(self.numerical_properties))))

        ax = self._select_ax_to_plot(index)

        if columns == None:
            columns = self.numerical_properties

        plot_kwargs = {
            "ax" : ax,
            "logy" : bool(self.params["log_scale_y"]),
            "sharex" : True,
            "linewidth":line_width
        }

        if line_color is not None:
            plot_kwargs["color"] = line_color

        if type(data) == pd.DataFrame:
            data[columns].plot(**plot_kwargs)        
        else:
            data.plot(**plot_kwargs)        

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
                linewidth=linewidth,
                color=self.params["failure_line_color"])
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

        return self
    
    def plot_gmm_histogram(
            self,
            model_gmm : GaussianMixture,
            data,
            limits = (0,17),
            index=None
            ):
        
        stds = np.sqrt(model_gmm.covariances_).flatten()
        weights = model_gmm.weights_
        means = model_gmm.means_

        # Criando um range de valores para plotar as distribuições
        x = np.linspace(limits[0], limits[1], 1000)

        ax = self._select_ax_to_plot(index)
        
        n,bins,patches = ax.hist(data, bins=100, density=True, alpha=0.5,label="Histogram")
        ax.set_xlim(limits[0],limits[1])


        # Plotando cada gaussiana individualmente
        for i in range(means.shape[0]):
            ax.plot(x, weights[i] * norm.pdf(x, means[i], stds[i]),label=f"Gaussian {i}")
                

        return self
    
    
    def set_axes_texts(self,entry=None,title = None,measure_X=None,measure_Y=None):

        if entry == None:
            ax = self.axs
        else:
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


    def finalize(self,title=None):
        if title == None:
            title = self.params["plot_title"]
        self.figure.suptitle(title,y=self.params["y_dist"])
        self.figure.tight_layout(pad=self.params["tight_layout_pad"])

        return self