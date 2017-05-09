import numpy as np
import tensorflow as tf
from tensorflow.contrib import learn
from tensorflow.contrib.learn.python import SKCompat
import scipy.io as sio
from sklearn import metrics

# the data should be organized as [date, open, close, low, high, volume]
data = sio.loadmat('daily_data.mat')['data']

data_len = data.shape[0]

# test data length
train_len = round(3 * data_len / 4)

x_train = data[0:train_len, :, :]
# start with just predicting the opening value the next day
y_train = data[1:train_len + 1, :, 1]

x_test = data[train_len + 1:-1, :, :]
# start with just predicting the opening value the next day
y_test = data[train_len + 2:, :, 1]

# let's start by just flattening the data
x_train = np.reshape(x_train, (train_len, -1))
x_test = np.reshape(x_test, (len(y_test), -1))

# Specify that all features have real-value data
feature_columns = [tf.contrib.layers.real_valued_column("", dimension=x_train.shape[1])]

classifier = SKCompat(learn.DNNRegressor(label_dimension=y_train.shape[1], feature_columns=feature_columns,
                                         hidden_units=[100, 50, 20]))
classifier.fit(x_train, y_train, steps=100000, batch_size=100)
score = metrics.r2_score(y_test, classifier.predict(x_test)['scores'])
accuracy = metrics.mean_squared_error(y_test, classifier.predict(x_test)['scores'])
# score= np.linalg.norm(y_test-classifier.predict(x_test)['scores'])/np.linalg.norm(y_test)
print("Score: %f, Accuracy: %f" % (score, accuracy))
