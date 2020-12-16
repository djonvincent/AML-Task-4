import numpy as np
from pathlib import Path
from argparse import ArgumentParser
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import balanced_accuracy_score, classification_report
from load_data import load_data
from features import power_features, PLF, onset_features, EMG_power
from sklearn.model_selection import KFold

N_SPLITS=3 # 3 subjects in train data

parser = ArgumentParser()
parser.add_argument('--mode', choices=['val', 'test'], default='val')
args = parser.parse_args()

def main():
    #DATAFRAME
    x_train, y_train, x_test = load_data()

    #FFT for each 2s
    fft_train = fft(x_train.reshape(-1, 2, 256)).reshape(x_train.shape[0], -1)
    fft_test = fft(x_test.reshape(-1, 2, 256)).reshape(x_test.shape[0], -1)
    x_train = fft_train
    x_test = fft_test

    #ANALYZE
    class_weights = analyze(y_train)

    #SCALE DATA
    if args.mode == 'test':
        x_train, x_test = scale(x_train, x_test)

        #CREATE SHIFTED DATA
        x_train_pre = np.concatenate((np.zeros(shape=(1, x_train.shape[1])), x_train[0:-1]), axis=0)
        x_train_post = np.concatenate((x_train[1:], np.zeros(shape=(1, x_train.shape[1]))), axis=0)

        x_test_pre = np.concatenate((np.zeros(shape=(1, x_train.shape[1])), x_test[0:-1]), axis=0)
        x_test_post = np.concatenate((x_test[1:], np.zeros(shape=(1, x_train.shape[1]))), axis=0)
        NN_model = NN(x_train.shape[1])
        NN_model.fit([x_train_pre, x_train, x_train_post], y_train, batch_size=32, epochs=1, class_weight=class_weights)
        y_pred = np.argmax(NN_model.predict([x_test_pre, x_test, x_test_post]), axis=1)

        y_pred_ids = np.hstack((
        np.arange(y_pred.size).reshape(-1, 1),
        y_pred.reshape(-1, 1) + 1 # Labels start at 1
        ))

        np.savetxt(fname='y_test.csv', header='Id,y', delimiter=',', X=y_pred_ids,
            fmt=['%d', '%d'], comments='')



    elif args.mode == 'val':
        kf = KFold(n_splits=3, shuffle=False)
        for train_idx, test_idx in kf.split(x_train):
            scores = []
            x_train_kf = x_train[train_idx]
            y_train_kf = y_train[train_idx]
            x_test_kf = x_train[test_idx]
            y_test_kf = y_train[test_idx]
            # Scale data
            x_train_kf, x_test_kf = scale(x_train_kf, x_test_kf)
            # Created shifted data
            x_train_kf_pre = np.concatenate(
                (np.zeros((1, x_train.shape[1])), x_train_kf[:-1]),
                axis=0
            )
            x_train_kf_post = np.concatenate(
                (x_train_kf[1:], np.zeros((1, x_train.shape[1]))),
                axis=0
            )
            x_test_kf_pre = np.concatenate(
                (np.zeros((1, x_train.shape[1])), x_test_kf[:-1]),
                axis=0
            )
            x_test_kf_post = np.concatenate(
                (x_test_kf[1:], np.zeros((1, x_train.shape[1]))),
                axis=0
            )
            # Run 10 models and take average BMAC
            for i in range(3):
                NN_model = NN(x_train.shape[1])
                NN_model.fit(
                    [x_train_kf_pre, x_train_kf, x_train_kf_post],
                    y_train_kf, batch_size=32, epochs=1,
                    class_weight=class_weights, verbose=0
                )

                y_pred = np.argmax(
                    NN_model.predict([x_test_kf_pre, x_test_kf, x_test_kf_post]),
                    axis=1
                )
                BMAC = balanced_accuracy_score(y_test_kf, y_pred)
                scores.append(BMAC)
            print(f'The validation BMAC score is: {sum(scores)/len(scores)}')

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
    #neighbour_model.summary()
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
