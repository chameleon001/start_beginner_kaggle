
#%%

# Statoil/C-CORE 필사 커널
# https://kaggle-kr.tistory.com/19?category=868316
# https://www.kaggle.com/c/statoil-iceberg-classifier-challenge/data

# D:\GitHub\kaggle_dataset\statoil-iceberg-classifier-challenge\data\processed

#%%

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt


# %%
%matplotlib inline

# %%
import plotly.offline as py
import plotly.graph_objs as go
from plotly import tools

# %%
py.init_notebook_mode(connected=True)

# %%

#home
# data_path = "D:\GitHub\kaggle_dataset\statoil-iceberg-classifier-challenge\data\processed"
#company
data_path = "D:\Chameleon\pytorch\data\statoil-iceberg-classifier-challenge\data\processed"
train = pd.read_json(data_path + "/train.json")
test = pd.read_json(data_path + "/test.json")
train.head()
#%%
train.inc_angle = train.inc_angle.replace('na',0)
train_inc_angle = train.inc_angle.astype(float).fillna(0.0)
print("done!")

# %%
train.head()

# %%
