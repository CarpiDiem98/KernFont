# import numpy as numpy
# from pprint import pprint
from progettoKerning.library.dataframe import concatDataFramePlus
from path_font import path_kerning_file


# from sklearn.datasets import make_regression
# from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

# from autosklearn.regression import AutoSklearnRegressor

# from autosklearn.automl import AutoML
from autosklearn.regression import AutoSklearnRegressor
# from autosklearn.constants import REGRESSION
import pandas as pd 

kern_path_list = path_kerning_file('../Dataset/train/UFO')
letter = ['A', 'T', 'F', 'V', 'P', 'Y', 'W', 'D', 'R', 'S',
          'period', 'comma', 'v', 'f', 't', 'c', 'k', 'l', 'o', 'r', 'd', 'b']


# df_complete = concatDataFramePlus(kern_path_list, letter)
# df_complete.to_csv('font.csv')
df_complete = pd.read_csv('./font.csv')
X = df_complete[['peso', 'squadratura']]
y = df_complete.iloc[:, 3:].fillna(0.0)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

# reg = AutoML(time_left_for_this_task=2*60, per_run_time_limit=30)
reg = AutoSklearnRegressor(time_left_for_this_task=2*60, per_run_time_limit=20)
reg.fit(X_train, y_train)


#dopo ore non ha finito di eseguire, forse è una cosa troppo lunga dopo vediamo