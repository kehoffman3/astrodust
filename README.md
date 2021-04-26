# Astrodust

A package for predicting the distribution of dust particles in protoplanetary disk based off our paper "Multi-Output Random Forest Regression to Emulatethe Earliest Stages of Planet Formation".

## Installation

Astrodust can be installed from PyPI via pip: 

`pip install astrodust`

## Pretrained Models

The package requires two pretrained models, a random forest regression model and XGBoost classifier. These can be downloaded beforehand from [Zenodo](https://zenodo.org/record/4662910#.YGx_bGRue3I) and placed in the current working directory in a `models` directory. Otherwise the package will prompt to automatically download them when the `DustModel` is instaniated.

## Documentation

The documentation for the package is located [here](https://kehoffman3.github.io/astrodust/docs/astrodust.html). A demonstration code notebook is also [available](https://github.com/kehoffman3/astrodust/blob/master/demo/demo.ipynb), or can be viewed online [here](https://kehoffman3.github.io/astrodust/docs/demo.html).

Particle sizes for each bin for the input and output are included as a reference in our [wiki](https://github.com/kehoffman3/astrodust/wiki/Particle-Sizes).

