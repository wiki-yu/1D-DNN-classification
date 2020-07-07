import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, concatenate, GlobalMaxPool1D, GlobalAveragePooling1D
from tensorflow.keras import optimizers, losses, activations, models
from tensorflow.keras.layers import Conv1D, MaxPooling1D
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from io import open

from ventmap.raw_utils import extract_raw
import pva_models


def load_waveform_data():
    """load flow & pressure waveforms into list respectively"""
    flows = []
    pressures = []
    flow_list = []
    pressure_list = []

    for (dir_path, dir_names, file_names) in os.walk('./waveforms'):
        for file in file_names:
            csv_path = os.path.join(dir_path, file)
            print("waveforms path: " + csv_path)
            generator = extract_raw(open(csv_path), False)
            for breath in generator:
                # breath data is output in dictionary format
                flow, pressure = breath['flow'], breath['pressure']
                flows += flow
                pressures += pressure
                flow_list.append(flow)
                pressure_list.append(pressure)

    return flow_list, pressure_list

def load_label_data():
    """load the label data corresponding to waveforms"""
    labels = []
    for (dir_path, dir_names, file_names) in os.walk('./labels'):
        for file in file_names:
            csv_path = os.path.join(dir_path, file)
            print("label path: " + csv_path)
            df = pd.read_csv(csv_path)
            df_filter = df[['dbl', 'mt', 'bs', 'co', 'su']]
            labels.append(df_filter)
    df_total = pd.concat(labels, axis=0, ignore_index=True)
    label_arr = df_total.to_numpy()

    # check if the pva type is unique in each row
    test = (df_total == 0).sum(axis=1) < 4
    count = 0
    for index, value in test.items():
        if value:
            count += 1
    print("pva type not unique amount:" + str(count))

    return label_arr

def waveform_trim(pressure_list):
    """unify the length of wall the waveform sections"""
    len_list = []
    for pressure in pressure_list:
        len_list.append(len(pressure))
    len_trim = int(np.percentile(len_list, 90))

    return len_trim

def check_notation(notation, i):
    """specify the labels based on the raw notation data"""
    if notation[i][0] == 1:
        label = 1 #"dbl"
    elif notation[i][1] == 1:
        label = 2 #"mt"
    elif notation[i][2] == 1:
        label = 3 #"bs"
    elif notation[i][3] == 1:
        label = 4 #"co"
    elif notation[i][4] == 1:
        label = 5 #"su"
    else:
        label = 0 #"normal"

    return label

def plot_sample_data(flows, pressures):
    """plot the sample waveform data of flow & pressure"""
    plt.figure(figsize=(20, 5))
    plt.plot(flows[:2000], label='Flow')
    plt.plot(pressures[:2000], label='Pressure')
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.show()

def load_train_test_data():
    """generate train and test data based on the waveforms & labels"""
    waveforms = []
    labels = []

    flow_list, pressure_list = load_waveform_data()
    label_arr = load_label_data()
    len_trim = waveform_trim(pressure_list)

    for i, pressure in enumerate(pressure_list):
        flow = flow_list[i]
        if len(pressure) < len_trim:
            pressure_ext = np.concatenate([pressure, np.zeros(len_trim-len(pressure))])
            pressure_ext = pressure_ext.tolist()
            flow_ext = np.concatenate([flow, np.zeros(len_trim-len(flow))])
            flow_ext = flow_ext.tolist()
            pressure_flow = pressure_ext + flow_ext
            waveforms.append(pressure_flow)
            label = check_notation(label_arr, i)
            labels.append(label)
        else:
            pressure_ext = pressure[:len_trim]
            flow_ext = flow[:len_trim]
            pressure_flow = pressure_ext + flow_ext
            waveforms.append(pressure_flow)
            label = check_notation(label_arr, i)
            labels.append(label)

    comb_wav_label = np.concatenate((np.matrix(waveforms), np.matrix(labels).T), axis=1)
    df = pd.DataFrame(comb_wav_label)
    X = np.array(df[list(range(len_trim * 2))].values)[..., np.newaxis]
    y = np.array(df[len_trim * 2].values).astype(np.int8)

    # separate the datasets into the training and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    return X_train, X_test, y_train, y_test

def plot_segments(waveforms, labels, title):
    """present sample waveform pieces and corresponding labels"""
    plt.figure(figsize=(12, 12))
    col_num = 5
    row_num = 5
    signal_nums = 25
    k = 4

    for i in range(signal_nums):
        plt.subplot(row_num, col_num, i+1)
        plt.plot(waveforms[i + k*signal_nums]) # pay attention to the range
        plt.title(labels[i + k*signal_nums])
        plt.xticks([])
        plt.yticks([])
    plt.suptitle(title, size=20)
    plt.show()

def plot_train_history(history, title):
    """plot training result"""
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(loss))

    plt.figure()
    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.grid(True)
    plt.title(title)
    plt.legend()
    plt.show()

def main():
    "the main function"
    X_train, X_test, y_train, y_test = load_train_test_data()
    # Select a model you prefer to use
    model = pva_models.get_1D_CNN_model_stack()

    file_path = "pva_classification.h5"

    checkpoint = ModelCheckpoint(file_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    early = EarlyStopping(monitor="val_acc", mode="max", patience=5, verbose=1)
    redonplat = ReduceLROnPlateau(monitor="val_acc", mode="max", patience=3, verbose=2)

    callbacks_list = [checkpoint, early, redonplat]
    multi_step_history = model.fit(X_train, y_train, epochs=30, verbose=2, callbacks=callbacks_list, validation_split=0.1)
    # model.save("breaths_model.h5")
    model.load_weights(file_path)

    pred_test = model.predict(X_test)
    print("the first predict: {}".format(np.shape(pred_test)))
    pred_test = np.argmax(pred_test, axis=-1)
    print("the second predict: {}".format(np.shape(pred_test)))

    f1 = f1_score(y_test, pred_test, average="macro")
    print("Test f1 score : %s "% f1)

    acc = accuracy_score(y_test, pred_test)
    print("Test accuracy score : %s "% acc)
    plot_train_history(multi_step_history, "loss")

if __name__ == '__main__':
    main()