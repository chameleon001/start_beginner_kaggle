
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
#generate the training data
#create 3 bands having hh, hv and avg of both

X_band_1=np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in train["band_1"]])
X_band_2=np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in train["band_2"]])
X_train = np.concatenate([X_band_1[:, :, :, np.newaxis], 
                         X_band_2[:, :, :, np.newaxis],
                         ((X_band_1+X_band_2)/2)[:, :, :, np.newaxis]], 
                        axis=-1)

X_band_test_1=np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in test["band_1"]])
X_band_test_2=np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in test["band_2"]])
X_test = np.concatenate([X_band_test_1[:, :, :, np.newaxis]
                          , X_band_test_2[:, :, :, np.newaxis]
                         , ((X_band_test_1+X_band_test_2)/2)[:, :, :, np.newaxis]], axis=-1)
#%%
# https://plot.ly/api/

def plot_contour_2d(band1, band2, label):
   
    fig = tools.make_subplots(rows=1, cols=2,  specs=[[{'is_3d': True}, {'is_3d': True}]])

    fig.append_trace(dict(type='surface', z=band1, colorscale='RdBu',
                          scene='scene1', showscale=False), 1, 1)
    
    fig.append_trace(dict(type='surface', z=band2, colorscale='RdBu',
                          scene='scene2', showscale=False), 1, 2)

    fig['layout'].update(title='3D surface plot for "{}" (left is from band1, right is from band2)'.format(label), titlefont=dict(size=30), height=800, width=1200)

    py.iplot(fig)

    fig, ax = plt.subplots(1, 2, figsize=(16, 10))
    ax[0].imshow(X_band_1[num, :, :])
    ax[0].set_title('Image from band_1', fontsize=15)
    ax[1].imshow(X_band_2[num, :, :])
    ax[1].set_title('Image from band_2', fontsize=15)
    plt.show()

#%%


num = 0
label = 'iceberg' if (train['is_iceberg'].values[num] == 1) else'ship'
plot_contour_2d(X_band_1[num,:,:], X_band_2[num,:,:], label)



# %%
#Import Keras.
from matplotlib import pyplot
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Input, Flatten, Activation
from keras.layers import GlobalMaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import Concatenate
from keras.models import Model
from keras import initializers
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping

# %%
