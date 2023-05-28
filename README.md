# cars_project

## Project for my master's thesis

#### Contains 4 stages:
* data scraping from otomoto:
  - script otomoto_scraping.py
  - collects all the possible information about each offer

* data cleaning
  - notebook data_cleaning.ipynb
  - basic transformations and cleaning to the dataset

* modelling of price with 4 different models
  - notebook models.ipynb
  - feature engineering and onehotencoding
  - used 4 different models: linear regression, ridge regression, decision trees, random forest
  - hyperparameters tuning with RandomizedSearchCV
  - model validation with CV

* Monte Carlo simluations for performance assessment and identify differences between models
  - scripts: simulation1/2/3.py and notebook simulation_analysis.ipynb
  - during each simulation, target variable is generated from different linear model. Then after target variable is generated, every considered model is being fitted to 95% of this data and performance is assesed on 5% (test set). Each simulation took 1000 iterations (results are not added as they weight too much). 
  - each simulation (1/2/3) had different level of not linear relationships. This was achieved by adding additional variable which adds non-linear relations and random variable from shifted gamma distribution. In last one (3rd) to strenghten the non linearity, parameter of linear model which was assigned to the new variable was multiplied 3 times. 
  - results with characteristics (RMSE and bias) were calcualted and presented on plots in simulation_analysis.ipynb notebook
