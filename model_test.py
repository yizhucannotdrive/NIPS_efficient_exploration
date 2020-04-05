from keras.models import Sequential
from keras.layers import Dense, Reshape, Activation
from keras.optimizers import Adam
from keras.activations import softmax
from  keras import backend as K

LEARNING_RATE = 0.001
ACTIONS_DIM = 2
OBSERVATIONS_DIM = 4
def softMaxAxis1(x, axis = 1):
    ndim = K.ndim(x)
    if ndim >= 2:
        e = K.exp(x)
        s = K.sum(e, axis=axis, keepdims=True)
        return e / s
    else:
        raise ValueError('Cannot apply softmax to a tensor that is 1D. '
                         'Received input: %s' % x)

model = Sequential()
model.add(Dense(16, input_shape=(OBSERVATIONS_DIM,), activation='relu'))
model.add(Dense(16, input_shape=(OBSERVATIONS_DIM,), activation='relu'))
model.add(Dense(ACTIONS_DIM * OBSERVATIONS_DIM, activation='linear'))
model.add(Reshape((ACTIONS_DIM, OBSERVATIONS_DIM)))
model.add(Activation(softMaxAxis1))
model.summary()