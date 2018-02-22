# Housing Price Prediction

# ============================================================================================
# 1) Fetch the data from URL to tar, and extract
# ============================================================================================
import os
import tarfile
from six.moves import urllib

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = os.path.join("dataset", "housing")               # dataset\housing
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

os.getcwd()     # 'C:\\Users\\angsharma'
os.chdir(r"C:\Users\angsharma\Documents\Angela\Career\Package\housing_price_prediction")

def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)                           # Create folder if not exist
        
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)       # URL to local path
    housing_tgz= tarfile.open(tgz_path)                     # Open tar file
    housing_tgz.extractall(path=housing_path)               # Extract all tar file
    housing_tgz.close()                                     # Make sure to close file
    
fetch_housing_data()
   
# ============================================================================================    
# 2) Load data
# ============================================================================================
import time
startime = time.time()

import pandas as pd

def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)
    
housing = load_housing_data()

# ============================================================================================    
# 3) Quick look at data structure
# ============================================================================================   
housing.head()                              # Top 5 rows
housing.info()                              # Data type and missing values?
housing.describe()                          # Mean, std, min, max etc

housing["ocean_proximity"].value_counts()   # Counts by categories

# ============================================================================================    
# 4) Data visualization
# ============================================================================================   
housing.hist(bins=50, figsize=(15,12))      # age capped (50)
                                            # median income non-USD and capped (15)
                                            # house value capped (500k)
                                            
                                            # attribute different scale --> feature scaling
                                            # tail heavy --> hard to detect pattern 
                                            #            --> transform to bell shape
                                            
# ============================================================================================    
# 4a) Create test set (Random)
# ============================================================================================ 
import numpy as np
from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42) # random

housing["income_cat"] = np.ceil(housing["median_income"] / 1.5)
housing["income_cat"].value_counts()
housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace = True) # > 5 replace w/ 5

# ============================================================================================    
# 4b) Create test set (Stratified Sampling) --> important to have instances from each stratrum
# ============================================================================================ 
from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

housing["income_cat"].value_counts() / len(housing)
strat_test_set["income_cat"].value_counts() / len(strat_test_set)
strat_train_set["income_cat"].value_counts() / len(strat_train_set)

# ============================================================================================    
# 5) Remove income cat to restore dataset back to original state
# ============================================================================================ 
for set_ in (strat_test_set, strat_train_set):
    set_.drop("income_cat", axis=1, inplace=True)              
                                            
# ============================================================================================    
# 6) Visual geographic data --> price is very much related to location 
# ============================================================================================ 
import matplotlib.pyplot as plt   
    
housing = strat_train_set.copy()

housing.plot(kind="scatter", x="longitude", y="latitude")
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1) #density of data points
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
             s=housing["population"]/100, # radius (population)
             label="population",                  
             figsize=(10,7),
             c="median_house_value",      # color (house value)
             cmap=plt.get_cmap("jet"), 
             colorbar=True)

# ============================================================================================    
# 7) Correlations + Pandas Plotting (Scatter_Matrix)
# ============================================================================================ 
corr_matrix = housing.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)

from pandas.plotting import scatter_matrix
attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]

scatter_matrix(housing[attributes], figsize=(12,8))

# ============================================================================================    
# 8) Experiment with attribute combinations
# ============================================================================================ 
housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"] = housing["population"]/housing["households"]

corr_matrix = housing.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)

# ============================================================================================    
# 9) Prepare the data for machine learning algorithms
# ---------------------------------------------------
# a) Separate predictors and labels
# ============================================================================================ 
housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()

# ============================================================================================    
# b) Data cleaning
# ----------------
# i) Imputer - ML algorithms cannot work with missing features
# ============================================================================================ 
from sklearn.preprocessing import Imputer
imputer = Imputer(strategy="median")

# ============================================================================================    
# Since median can only be computed on numerical values, we need to create a copy of data with
# only numerical values
# ============================================================================================ 
housing_num = housing.drop("ocean_proximity", axis=1)
imputer.fit(housing_num)                # simply compute median for numerical fields
 
imputer.statistics_
housing_num.median().values

X = imputer.transform(housing_num)      # use the trained imputer to transform training set
housing_tr = pd.DataFrame(X, columns=housing_num.columns) # transform np array back as pd df

# ============================================================================================    
# ii) Handling Text and Categorical Attributes - Factorize, OneHotEncoder
# ============================================================================================ 
housing_cat = housing["ocean_proximity"]
housing_cat.head(10)

housing_cat_encoded, housing_categories = housing_cat.factorize()           # Factorize
                                                                  
housing_cat_encoded[:10]    # maps category to integers
housing_categories          # list categories

from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder()                                                   # OneHotEncoder                                              
housing_cat_1hot = encoder.fit_transform(housing_cat_encoded.reshape(-1,1)) # reshape(-1,1)                                                                  
housing_cat_1hot.toarray()

#encoder = LabelBinarizer()
#housing_cat_1hot = encoder.fit_transform(housing_cat)

from sklearn.preprocessing import LabelBinarizer
from sklearn.base import TransformerMixin 
class MyLabelBinarizer(TransformerMixin):
    def __init__(self, *args, **kwargs):
        self.encoder = LabelBinarizer(*args, **kwargs)
    def fit(self, x, y=0):
        self.encoder.fit(x)
        return self
    def transform(self, x, y=0):
        return self.encoder.transform(x)

encoder = MyLabelBinarizer()
housing_cat_1hot = encoder.fit_transform(housing_cat)

# ============================================================================================    
# 10) Custom Transformers - Add additional combined attributes
# ============================================================================================ 
from sklearn.base import BaseEstimator, TransformerMixin

rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6    # column number

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):    # BaseEstimator (baseclass)
                                                                   # get_params(), set_params
    def __init__(self, add_bedrooms_per_room=True):                
        self.add_bedrooms_per_room = add_bedrooms_per_room      
        
    def fit(self, X, y=None):                      # X - numpy array of shape (training set)
        return self                                # y - numpy array of shape (target values)
    
    def transform(self, X, y=None):                             # TransformerMixin (transform)
        rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
        population_per_household = X[:, population_ix] / X[:, household_ix]
        if self.add_bedrooms_per_room:             # Hyperparameter - help ML algorithms?
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]
            
        
attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing.values)    # Transform df to numpy array

# ============================================================================================    
# 11) Custom Transformers - feed Pandas dataframe directly into pipeline
# ============================================================================================ 
class DataFrameSelector(BaseEstimator, TransformerMixin):
    
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X[self.attribute_names].values

# ============================================================================================    
# 12) Feature Scaling - Usually only applied to predictors
# --------------------------------------------------------
#  a) Normalization - values are shifted and rescaled end up ranging 0 to 1
#                   - (value - min value) / (max value - min value)
#                   - MinMaxScaler
# 
#  b) Standardization - (value - mean) / variance
#                     - could be problematic since some ML expects 0 - 1
#                     - much less affected by outliners
#                     - StandardScaler
# ============================================================================================ 

# ============================================================================================    
# 13) Transformation Pipeline
# ============================================================================================ 
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

num_attribs = list(housing_num)     # list
cat_attribs = ["ocean_proximity"]   # list

num_pipeline = Pipeline([("selector", DataFrameSelector(num_attribs)),
                         ("imputer", Imputer(strategy="median")),       # handle missing values
                         ("attribs_adder", CombinedAttributesAdder()),  # add attributer
                         ("std_scaler", StandardScaler())])             

cat_pipeline = Pipeline([("selector", DataFrameSelector(cat_attribs)),
                         ("cat_encoder", MyLabelBinarizer())])

    
from sklearn.pipeline import FeatureUnion

full_pipeline = FeatureUnion(transformer_list=[("num_pipeline", num_pipeline),
                                               ("cat_pipeline", cat_pipeline)])

housing_prepared = full_pipeline.fit_transform(housing)

# ============================================================================================    
# 14) Select and Train a Model
# ============================================================================================ 
# ============================================================================================    
# Train model (Linear regression)
# ============================================================================================ 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)       # Train model - Predictor & Target/Labels

housing_predictions = lin_reg.predict(housing_prepared)

lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
lin_rmse     # 68628.198198489234

# ============================================================================================    
# Train model (Decision Tree Regressor)
# ============================================================================================ 
from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor()
tree_reg.fit(housing_prepared, housing_labels)

housing_predictions = tree_reg.predict(housing_prepared)

tree_mse = mean_squared_error(housing_labels, housing_predictions)
tree_rmse = np.sqrt(tree_mse)
tree_rmse   # 0

# ============================================================================================    
# Train model (Random Forest Regressor)
# ============================================================================================ 
from sklearn.ensemble import RandomForestRegressor

forest_reg = RandomForestRegressor()
forest_reg.fit(housing_prepared, housing_labels)

housing_prediction = forest_reg.predict(housing_prepared)

forest_mse = mean_squared_error(housing_labels, housing_prediction)
forest_rmse = np.sqrt(forest_mse)
forest_rmse # 21957.475988088001

# ============================================================================================    
#  Better evalation using cross validation (k-fold validation)
#  - split the training set into 10 subsets, called folds
#  - trains and evaluates the decison tree model 10 times
# ============================================================================================ 
from sklearn.model_selection import cross_val_score

def display_scores(scores):
    print("Scores", scores)
    print("Score mean", scores.mean())
    print("Score STD", scores.std())

scores = cross_val_score(tree_reg, housing_prepared, housing_labels, 
                         scoring="neg_mean_squared_error", cv=10)

tree_rsme_scores = np.sqrt(-scores)
display_scores(tree_rsme_scores)
# Scores [ 70573.49231195  66373.50645769  71594.22430533  69176.13657549
# 70694.8161215   74979.43049516  72692.31748349  69923.61022424
# 76248.85017124  69511.41171589]
# Score mean 71176.7795862
# Score STD 2732.60502598

scores = cross_val_score(lin_reg, housing_prepared, housing_labels,
                         scoring="neg_mean_squared_error", cv=10)

lin_rmse_scores = np.sqrt(-scores)
display_scores(lin_rmse_scores)
# Scores [ 66782.73843989  66960.118071    70347.95244419  74739.57052552
# 68031.13388938  71193.84183426  64969.63056405  68281.61137997
# 71552.91566558  67665.10082067]
# Score mean 69052.4613635
# Score STD 2731.6740018

scores = cross_val_score(forest_reg, housing_prepared, housing_labels,
                         scoring="neg_mean_squared_error", cv=10)
forest_rmse_score = np.sqrt(-scores)
display_scores(forest_rmse_score)
# Scores [ 52269.2721895   49917.57441642  53401.20991116  55645.11108538
# 51850.25825171  54319.20754323  51303.18502142  51024.48787638
# 56128.59641516  52773.63226848]
# Score mean 52863.2534979
# Score STD 1915.4450744

# ============================================================================================    
#  15) Save Models
# ============================================================================================ 
from sklearn.externals import joblib

joblib.dump(forest_reg, "forest_reg.pkl")           # save model

my_model_loaded = joblib.load("forest_reg.pkl")     # retrieve model

# ============================================================================================    
#  16) Fine tuning model
#   a) Grid Search
#   -  n_estimators is the number of trees used in the forest
#   -  max_features is the number of features to consier while looking for a split
#
#   b) Randomized Search (if hyper parameter search space is large)
# ============================================================================================ 
from sklearn.model_selection import GridSearchCV

param_grid = [{"n_estimators":[3, 10, 30], "max_features":[2, 4, 6, 8]},
              {"n_estimators":[3, 10], "max_features":[2, 3, 4, 8], "bootstrap":[False]},]         # type of ensemble learning

forest_reg = RandomForestRegressor()

grid_search = GridSearchCV(forest_reg, param_grid, cv=5, scoring="neg_mean_squared_error")            
    
grid_search.fit(housing_prepared, housing_labels)

grid_search.best_params_
grid_search.best_estimator_

cv_res = grid_search.cv_results_
for mean_score, params in zip(cv_res["mean_test_score"], cv_res["params"]):
    print(np.sqrt(-mean_score), params)

# ============================================================================================    
#  17) Ensemble Methods
# ============================================================================================ 
feature_importances = grid_search.best_estimator_.feature_importances_
feature_importances

# Display importance scores next to corresponding attribute names
extra_attribs = ["rooms_per_hhold", "pop_per_hhold", "bedrooms_per_room"]
cat_encoder = cat_pipeline.named_steps["cat_encoder"]
cat_one_hot_attribs = list(housing_categories)

attributes = num_attribs + extra_attribs + cat_one_hot_attribs

sorted(zip(feature_importances, attributes), reverse=True)

# ============================================================================================    
#  18) Evaluate model on the test set
# ============================================================================================ 
final_model = grid_search.best_estimator_

x_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"].copy()

x_test_prepared = full_pipeline.transform(x_test)

final_predictions = final_model.predict(x_test_prepared)

final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
final_rmse