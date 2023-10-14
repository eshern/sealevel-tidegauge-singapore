# sealevel-tidegauge-singapore
Mean Sea Level Prediction with Time Series Model for Singapore

Climate change is an urgent global issue that affects ecosystems, communities, and economies. This is evident from shifts in temperature, changes in precipitation patterns, and rising sea levels, all of which underscore the need for immediate understanding and action.

This project aims to explore the impact of global warming on Singapore's climate evidence from the change in rate of sea level rising trend. Several hypothesis testing conducted on monthly Mean Sea Level Tide Gauge Data collected by the Permanent Service for Mean Sea Level (PSMSL). And a time series forecast model built based on past observations, and used in predicting near future trend. 

Objectives: 

1. To identify trends and seasonal patterns, 
2. To determine the significance of changes in sea level conditions with statistical hypothesis testing for trending 
3. To build a time series model (SARIMA) based on past observations and predict near future trend, with underlying seasonality preserved.

### To create the environment from the .yml file, use the command:

conda env create -f environment.yml

### After creating the environment, activate it with:

conda activate timeseries_sealevel

### Then, start Jupyter notebook with:

jupyter notebook

### To export a raw data file in .csv with selected Station ID

python helpers.py

### To deactivate the active environment, use

conda deactivate

