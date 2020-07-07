import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, concatenate, GlobalMaxPool1D, GlobalAveragePooling1D
from tensorflow.keras import optimizers, losses, activations, models
from tensorflow.keras.layers import Conv1D, MaxPooling1D

def get_1D_CNN_model():
    nclass = 6
    inp = tf.keras.Input(shape=(430, 1))
    img_1 = Conv1D(16, kernel_size=5, activation=activations.relu, padding="valid")(inp)
    img_1 = Conv1D(16, kernel_size=5, activation=activations.relu, padding="valid")(img_1)
    img_1 = MaxPooling1D(pool_size=2)(img_1)
    img_1 = Dropout(rate=0.1)(img_1)
    img_1 = Conv1D(32, kernel_size=3, activation=activations.relu, padding="valid")(img_1)
    img_1 = Conv1D(32, kernel_size=3, activation=activations.relu, padding="valid")(img_1)
    img_1 = MaxPooling1D(pool_size=2)(img_1)
    img_1 = Dropout(rate=0.1)(img_1)
    img_1 = Conv1D(32, kernel_size=3, activation=activations.relu, padding="valid")(img_1)
    img_1 = Conv1D(32, kernel_size=3, activation=activations.relu, padding="valid")(img_1)
    img_1 = MaxPooling1D(pool_size=2)(img_1)
    img_1 = Dropout(rate=0.1)(img_1)
    img_1 = Conv1D(256, kernel_size=3, activation=activations.relu, padding="valid")(img_1)
    img_1 = Conv1D(256, kernel_size=3, activation=activations.relu, padding="valid")(img_1)
    img_1 = GlobalMaxPool1D()(img_1)
    img_1 = Dropout(rate=0.2)(img_1)

    dense_1 = Dense(64, activation=activations.relu, name="dense_1")(img_1)
    dense_1 = Dense(64, activation=activations.relu, name="dense_2")(dense_1)
    dense_1 = Dense(nclass, activation=activations.softmax, name="dense_3_patterns")(dense_1)

    model = models.Model(inputs=inp, outputs=dense_1)
    opt = optimizers.Adam(0.001)

    model.compile(optimizer=opt, loss=losses.sparse_categorical_crossentropy, metrics=['acc'])
    #model.summary()
    return model

def get_1D_CNN_model_stack():
    "build the model for training"
    model = Sequential()
    model.add(Conv1D(filters=16, kernel_size=5, activation='relu', padding="valid", input_shape=(430, 1)))
    model.add(Conv1D(filters=16, kernel_size=5, activation='relu', padding='valid'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.1))
    model.add(Conv1D(filters=32, kernel_size=3, activation='relu', padding='valid'))
    model.add(Conv1D(filters=32, kernel_size=3, activation='relu', padding='valid'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.1))
    model.add(Conv1D(filters=32, kernel_size=3, activation='relu', padding='valid'))
    model.add(Conv1D(filters=32, kernel_size=3, activation='relu', padding='valid'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.1))
    model.add(Conv1D(filters=256, kernel_size=3, activation='relu', padding='valid'))
    model.add(Conv1D(filters=256, kernel_size=3, activation='relu', padding='valid'))
    model.add(GlobalMaxPool1D())
    model.add(Dropout(0.2))

    model.add(Dense(units=64, activation='relu', name='dense_1'))
    model.add(Dense(units=64, activation='relu', name='dense-2'))
    model.add(Dense(6, activation='softmax', name='dense_3_patterns'))

    model.compile(optimizer='adam', loss=losses.sparse_categorical_crossentropy, metrics=['acc'])
    model.summary()
    return model

