# LISSA - Layer of Integration with Scikit-learn and Signal Analysis 

**LISSA** is a Python library for **standardized, reproducible scientific plotting**, with a strong focus on **time series visualization**, **comparative analysis**, and **declarative configuration via JSON**.

Its main goal is to **separate visualization logic from analysis code**, allowing plots to be configured, reused, and audited without modifying plotting scripts.

A secondary goal is to provide examples of how to apply PCA methods, HMM methods and doing a general data analysis of the Equinor provided data of the main project. You can check it in **examples** folder.

---

## Motivation

In data analysis and signal processing projects, it is common that:

- plotting code is tightly coupled to analysis logic;
- visual parameters are duplicated across scripts;
- reproducibility depends on undocumented global `matplotlib` state;
- style changes require touching multiple files.

**LISSA** addresses these issues by:

- encapsulating figure state in a single class;
- centralizing plot configuration in JSON files;
- using a fluent (chainable) API;
- leveraging `matplotlib.rcParams` in a controlled way.

---

## Key Features

- Declarative configuration via **JSON**
- Fluent interface (`.method().method().method()`)
- Native support for **time series**
- Direct integration with **pandas** and **matplotlib**
- Optional label translation and unit handling
- Centralized control via `rcParams`
- Clear separation between:
  - configuration
  - figure creation
  - plotting methods
  - finalization/export

---

## Project Structure

```text
lissafigure/
├── lissa_figure.py          # Main LISSA class
├── defaults.py              # Default parameter dictionaries
├── utils.py                 # Utility functions
├── dictionaries/
│   ├── dictionaries.json    # Translations and units
│   └── new_headers.json
├── plots/
│   └── example_plot.json    # Plot configuration
└── README.md
```

---
## Core Concept: JSON-Driven Configuration

Most visual behavior is defined in an external JSON file:

```
{
  "rc_params": {
    "figure.figsize": [18, 10],
    "font.size": 12,
    "axes.titlesize": 16,
    "legend.fontsize": 14
  },
  "plot_title": "Time Series Example",
  "log_scale_y": false
}
```


This file is loaded directly by the LISSA class and applied consistently.

---

## Basic Usage Example

```
import lissafigure as li

fig = (
    li.LissaFigure("./plots/time_series_plot.json")
      .set_translation()
      .set_figure()
      .time_series_plot(data, index=0)
      .set_axes_texts(0, "Signal", "Time", "bar")
      .finalize()
)
```

---

## Fluent Interface

All major methods return self, enabling chained calls:

```
(fig
 .set_figure()
 .time_series_plot(raw_data, 0)
 .time_series_plot(filtered_data, 1)
 .finalize())
```

---


## Main Methods
```set_figure()```

- Creates the figure and axes

- Applies rcParams

- Sets layout and DPI

```time_series_plot(data, index, ...)```

- Plots time series on a selected axis

- Supports logarithmic scaling

- Uses pandas plotting backend

```set_axes_texts(ax_or_idx, ...)```

- Sets title, labels, and units

- Accepts either an Axes object or an axis index

```finalize()```

- Applies tight_layout

- Sets global title

- Saves the figure (if enabled)

---

## Translation and Units

If configured, LissaFigure can:

- translate column names automatically;

- attach physical units to axis labels;

- switch between languages without modifying code.

---

## Design Philosophy

- Configuration over code

- Explicit state control

- Reproducible graphics

- Minimal plotting boilerplate

- Low coupling with analysis logic

## Dependencies

- Python ≥ 3.9

- matplotlib

- pandas

- numpy

---

## Project Status

- Functional and actively used

- Ongoing simplification

- Progressive migration toward rcParams

- Public API still evolving

## Roadmap

- Automatic API documentation

- Extended examples

- JSON schema validation

- Additional plot types