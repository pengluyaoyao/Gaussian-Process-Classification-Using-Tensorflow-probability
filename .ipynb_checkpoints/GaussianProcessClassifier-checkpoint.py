import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import os
import itertools
import sys
from sklearn.metrics import f1_score
import time
from TFBinaryGPClassifier import TFBinaryGPClassifier

tfb = tfp.bijectors
tfd = tfp.distributions
tfk = tfp.math.psd_kernels

##### Prepare data ########

data = pd.read_csv('sonar.csv', header=None)
data = data.values
X = data[:,0:60].astype(float)
Y = data[:,60]

encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)

X_train, X_test, y_train, y_test = train_test_split(X, encoded_Y, test_size=0.2, random_state=42)

##### Define kernel: RBF with same length scale ########

constrain_positive = tfb.Shift(np.finfo(np.float64).tiny)(tfb.Softplus())

amplitude = tfp.util.TransformedVariable(
    initial_value=1.,
    bijector=constrain_positive,
    name='amplitude',
    dtype=np.float64)

length_scale = tfp.util.TransformedVariable(
    initial_value=1.,
    bijector=constrain_positive,
    name='length_scale',
    dtype=np.float64)

kernel = tfk.ExponentiatedQuadratic(amplitude, length_scale)

##### Fit model ########
start = time.time()

BGPC = TFBinaryGPClassifier(kernel=kernel).fit(X=X_train, y=y_train)
probs_preds = BGPC.predict_proba(X=X_test)
labels_preds = BGPC.predict(X=X_test)

end =time.time()

print('Accuracy and F1 accuracy are %f %f' %(np.mean(labels_preds==y_test),f1_score(labels_preds, y_test)), 'in %f seconds' %(end-start))
print('Optmized amplitude and length scale are %f, %f' %(BGPC.kernel.amplitude, BGPC.kernel.length_scale))