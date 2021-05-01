import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense

data = pd.read_csv('weatherAUS.csv')
data.dropna(subset=['RainTomorrow'], inplace=True)
data['RainTomorrow'] = pd.factorize(data['RainTomorrow'])[0]
target = data['RainTomorrow']
data = data.drop(['RainTomorrow'], axis=1)

cat = ['Location',
       'WindGustDir',
       'WindDir9am',
       'WindDir3pm',
       'RainToday',
       'Date']
dig = [i for i in list(data) if i not in cat]

percent_nan = data.isna().sum()/data.shape[0]
for ind, d in zip(percent_nan.index, percent_nan):
    if ind in dig and d != 0:
        data[ind].fillna(data[ind].mean(), inplace=True)
data.fillna(method='ffill', inplace=True)

for c in cat:
    data['{}_dig'.format(c)] = pd.factorize(data[c])[0]

cat_dig = ['{}_dig'.format(c) for c in cat]
features = dig + cat_dig

drop_features = []
for feature, corr in zip(data[features].corrwith(target).index, data[features].corrwith(target).values):
    if np.abs(corr) < 0.05:
        drop_features.append(feature)
for d in dig:
  data[d] = (data[d] - data[d].mean()) / data[d].std()

new_features = [feature for feature in features if feature not in drop_features]
xtrain, xtest, ytrain, ytest = train_test_split(
    data[new_features], target, random_state=40, test_size=0.2, shuffle=False)

device_name = tf.test.gpu_device_name()
if "GPU" not in device_name:
    print("GPU device not found")
print('Found GPU at: {}'.format(device_name))

k = 30

r_xtrain = [xtrain.iloc[i:i+k] for i in range(xtrain.shape[0]-k-1)]
r_ytrain = [ytrain.iloc[i+k] for i in range(ytrain.shape[0]-k-1)]

r_xtest = [xtest.iloc[i:i+k] for i in range(xtest.shape[0]-k-1)]
r_ytest = [ytest.iloc[i+k] for i in range(xtest.shape[0]-k-1)]

r_xtrain = np.array(r_xtrain)
r_ytrain = np.array(r_ytrain)

r_xtest = np.array(r_xtest)
r_ytest = np.array(r_ytest)

with tf.device('/cpu:0'):
    model = models.Sequential()

    model.add(layers.GRU(32, input_shape=(None, r_xtrain.shape[-1]), recurrent_dropout=0.2))
    model.add(layers.Dense(1))
    model.compile(optimizer=Adam(amsgrad=True), loss='mse', metrics='accuracy')
    history = model.fit(x=r_xtrain,
                        y=r_ytrain,
                        epochs=15,
                        validation_data=(r_xtest, r_ytest)
                        )
