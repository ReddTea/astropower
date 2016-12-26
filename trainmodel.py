from keras.models import Sequential
from keras.layers import MaxoutDense, Convolution2D, pooling, MaxPooling2D
import pandas as pd
import scipy as sp
import glob
from astropy.io import fits

# train data
GZ2 = pd.read_csv('GalaxyZoo2.csv')
r1 = GZ2["t01_smooth_or_features_a01_smooth_debiased"]
r2 = GZ2["t01_smooth_or_features_a02_features_or_disk_debiased"]
r3 = GZ2["t01_smooth_or_features_a03_star_or_artifact_debiased"]
y_train = [1]*len(r1)
for i in range(len(r1)):
    maxprob = max(r1[i],r2[i],r3[i])
    if r1[i]==maxprob :
        y_train[i] = 1
    elif r2[i]==maxprob:
        y_train[i] = 2
    elif r3[i]==maxprob:
        y_train[i] = 3
y_train = sp.array(y_train)

files =  glob.glob("./kerasfits/*.fits")
x_train = sp.array([])
for pathfile in files:
    hdulist = fits.open(pathfile)
    data = hdulist[0].data
    x_train = sp.array((x_train, data))
x_train = sp.delete(x_train,0)




model = Sequential()
model.add(Convolution2D(16,10,10,border_mode='same',input_shape=(1,207,207)))
model.add(pooling.MaxPooling2D(pool_size=(2,2),border_mode='same')) #98x98

model.add(Convolution2D(32,9,9,border_mode='same'))
model.add(pooling.MaxPooling2D(pool_size=(2,2),border_mode='same'))# 45x45

model.add(Convolution2D(64,6,6,border_mode='same'))
model.add(pooling.MaxPooling2D(pool_size=(2,2),border_mode='same'))#20x20

model.add(Convolution2D(128,5,5,border_mode='same'))
model.add(pooling.MaxPooling2D(pool_size=(2,2),border_mode='same')) #8x8

model.add(Convolution2D(256,3,3,border_mode='same'))
#6x6

model.add(Convolution2D(256,3,3,border_mode='same'))#4x4
model.add(MaxPooling2D(pool_size=(2,2),border_mode='same')) #2x2

model.add(MaxoutDense(2048))

model.compile(loss='categorical_crossentropy', optimizer='sgd')

model.fit(x_train, y_train, batch_size=32, nb_epoch=1)

from sklearn.externals import joblib
joblib.dump(model,"model.pkl")
