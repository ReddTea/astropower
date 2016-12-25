from keras.models import Sequential
from keras.layers import Dense, Convolution2D, pooling

model = Sequential()
model.add(Convolution2D(16,10,10,border_mode='same'),input_shape=(207,207))
model.add(pooling.MaxPooling2D(pool_size=(2,2),border_mode='same')) #98x98

model.add(Convolution2D(32,9,9,border_mode='same'))
model.add(pooling.MaxPooling2D(pool_size=(2,2),border_mode='same'))# 45x45

model.add(Convolution2D(64,6,6,border_mode='same'),input_shape=(207,207))
model.add(pooling.MaxPooling2D(pool_size=(2,2),border_mode='same'))#20x20

model.add(Convolution2D(128,5,5,border_mode='same'),input_shape=(207,207))
model.add(pooling.MaxPooling2D(pool_size=(2,2),border_mode='same')) #8x8

model.add(Convolution2D(256,3,3,border_mode='same'),input_shape=(207,207))
#6x6

model.add(Convolution2D(256,3,3,border_mode='same'),input_shape=(207,207))#4x4
model.add(pooling.MaxPooling2D(pool_size=(2,2),border_mode='same')) #2x2   
