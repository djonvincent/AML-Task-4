import numpy as np
from pathlib import Path
from argparse import ArgumentParser
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import balanced_accuracy_score, classification_report
from load_data import load_data
from features import power_features, PLF, onset_features, EMG_power

N_SPLITS=3 # 3 subjects in train data

parser = ArgumentParser()
parser.add_argument('--test', dest='test', default=False)
parser.add_argument('--predict', dest='predict', default=True)
args = parser.parse_args()

def main():
    #DATAFRAME
    x_train, y_train, x_test = load_data()

    #FFT
    x_train = fft(x_train)
    x_test = fft(x_test)

    #RESHAPE
    x_train = np.reshape(x_train, (x_train.shape[0], -1))
    x_test = np.reshape(x_test, (x_test.shape[0], -1))

    #ANALYZE
    class_weights = analyze(y_train)

    #SET VAL DATA ASIDE
    if args.test:
        x_val = x_train[0:21600, :]
        y_val = y_train[0:21600]
        x_train = x_train[21600:, :]
        y_train = y_train[21600:]

    #SCALE DATA
    if args.test:
        x_train, x_val = scale(x_train, x_val)
    if args.predict:
        x_train, x_test = scale(x_train, x_test)

    #CREATE SHIFTED DATA
    x_train_pre = np.concatenate((np.zeros(shape=(1, x_train.shape[1])), x_train[0:-1]), axis=0)
    x_train_post = np.concatenate((x_train[1:], np.zeros(shape=(1, x_train.shape[1]))), axis=0)

    x_test_pre = np.concatenate((np.zeros(shape=(1, x_train.shape[1])), x_test[0:-1]), axis=0)
    x_test_post = np.concatenate((x_test[1:], np.zeros(shape=(1, x_train.shape[1]))), axis=0)

    if args.test:
        x_val_pre = np.concatenate((np.zeros(shape=(1, x_train.shape[1])), x_val[0:-1]), axis=0)
        x_val_post = np.concatenate((x_val[1:], np.zeros(shape=(1, x_train.shape[1]))), axis=0)

    NN_model = NN(x_train.shape[1])

    if args.test:
        for i in range(10):
            NN_model.fit([x_train_pre, x_train, x_train_post], y_train, batch_size=32, epochs=1, class_weight=class_weights)

            y_val_predict = np.argmax(NN_model.predict([x_val_pre, x_val, x_val_post]), axis=1)
            BMAC = balanced_accuracy_score(y_val, y_val_predict)
            print('the validation BMAC score is: {}'.format(BMAC))

    if args.predict:
        NN_model.fit([x_train_pre, x_train, x_train_post], y_train, batch_size=32, epochs=1, class_weight=class_weights)
        y_pred = np.argmax(NN_model.predict([x_test_pre, x_test, x_test_post]), axis=1)

        y_pred_ids = np.hstack((
        np.arange(y_pred.size).reshape(-1, 1),
        y_pred.reshape(-1, 1) + 1 # Labels start at 1
        ))

        np.savetxt(fname='y_test.csv', header='Id,y', delimiter=',', X=y_pred_ids,
            fmt=['%d', '%d'], comments='')


def fft(data):
    fft_data = np.abs(np.fft.fft(data))
    return fft_data


def NN(input_shape):
    print('creating NN')
    model_input_1 = keras.Input(shape=input_shape)
    model_input_2 = keras.Input(shape=input_shape)
    model_input_3 = keras.Input(shape=input_shape)

    model_input = keras.Input(shape=input_shape)
    x = keras.layers.Dense(4, activation='relu')(model_input)
    #x = keras.layers.Dropout(0.6)(x)
    #x = keras.layers.Dense(64, activation='relu')(x)
    #x = keras.layers.Dropout(0.6)(x)
    #x = keras.layers.Dense(32, activation='relu')(x)
    x = keras.layers.Dropout(0.6)(x)
    output = keras.layers.Dense(3, activation='softmax')(x)
    model = keras.Model(inputs = model_input, outputs = output)
    #model.summary()

    output_1 = model(model_input_1)
    output_2 = model(model_input_2)
    output_3 = model(model_input_3)
    output_average = keras.layers.average([output_1, output_2, output_3])
    neighbour_model = keras.Model(inputs=[model_input_1, model_input_2, model_input_3], outputs=output_average)
    neighbour_model.summary()
    neighbour_model.compile(optimizer='adam', loss=keras.losses.SparseCategoricalCrossentropy())
    #keras.utils.plot_model(neighbour_model, "neighbour_model.png", show_shapes=True)
    return neighbour_model


def analyze(y_train):
    print('analyzing data...')
    unique, counts = np.unique(y_train, return_counts=True)
    print("the samples are distributed in the following way: {}".format(counts))
    invverse_counts = 1/counts
    invverse_counts = invverse_counts / np.sqrt(np.sum(invverse_counts**2))
    class_weights = {0: invverse_counts[0], 1: invverse_counts[1], 2: invverse_counts[2]}
    return class_weights



def scale(x_train_raw, x_test_raw):
    print('scaling data...')
    scaler = StandardScaler().fit(x_train_raw)
    x_train = scaler.transform(x_train_raw)
    x_test = scaler.transform(x_test_raw)
    return x_train, x_test


if __name__ == "__main__":
    main()