# Kaggle's Zillow Home Value Prediction 
This repo is my solution for [Zillow's Home Value Prediction Competition](https://www.kaggle.com/c/zillow-prize-1/) available on Kaggle. This kaggle competition asks competitors to predict the log error between **Zestimate**(the Zillow's statistical models that estimates the value of a property sold on Zillow's website) and the **actual sale price**. Submissions are evaluated based on **Mean Absolute Error** between the predicted log error and the actual log error.

# Structure

- `Zillow_Exploratory_Data_Analysis.ipynb` performs detailed exploratory data analysis on each feature within the dataset.

- `Zillow_Feature_Engineering.ipynb` performs feature engineeering, including detecting missing patterns, extracting datetime, aggregating based on regions, years, and so on. Data is saved as .csv format for later modeling.

- `Zillow_Modeling_Catboost.ipynb` constructs CatBoost models on top of the dataset derived from the earlier feature engineering process.

- `Zillow_Modeling_LGB.ipynb` constructs LightGBM models on top of the dataset derived from the earlier feature engineering process. Hyperparameters are tuned via bayesian hyperparameter optimization appraoch. 

- `Stacking.ipynb` performs random stacking on top of the two results from Catboost and LightGBM models. 

# Result

The following table demonstrates the performance of the models on the hidden private test set for this competition.


| Model | Private Leaderboard Score | Private Leaderboard Ranking | Percentile (Top) |
| :---: | :---:| :---: | :---: |
| LightGBM | 0.07540 | 760 / 3770 | 20.2% |
| CatBoost | 0.07514 | 250 / 3770 | 6.6% |
| **Stacking** | **0.07505** | **120 / 3770** | **3.2%** |

CatBoost, without tuning much parameter, showcases stroing performance, while LightGBM performs poor even if I perform some sort of parameter tuning. The performace on LightGBM model could have been improved if I worked on tuning hyperparameter more.

In the meantime, stacking is one powerful method to combine the "opinion" from two models and thus climb up the leaderboard. With random weights assigned, it turns out that **0.7 on CatBoost and 0.3 on LightGBM gives the best Result**. 