import pandas as pd
import numpy as np
from pathlib import Path

def load_data():
    x0TrainPath = Path('data', 'train_eeg1.csv')
    x1TrainPath = Path('data', 'train_eeg2.csv')
    x2TrainPath = Path('data', 'train_emg.csv')
    x0TestPath = Path('data', 'test_eeg1.csv')
    x1TestPath = Path('data', 'test_eeg2.csv')
    x2TestPath = Path('data', 'test_emg.csv')
    yTrainPath = Path('data', 'train_labels.csv')
    
    assert all([p.exists() for p in [x0TrainPath, x1TrainPath, x2TrainPath,
        x0TestPath, x1TestPath, x2TestPath, yTrainPath]]), 'Wrong data path.'

    print('loading dataframes...')
    x0_train = pd.read_csv(x0TrainPath, index_col=0).values
    x1_train = pd.read_csv(x1TrainPath, index_col=0).values
    x2_train = pd.read_csv(x2TrainPath, index_col=0).values
    # Labels start at 1
    y_train = pd.read_csv(yTrainPath, index_col=0).values.squeeze() - 1
    x0_test = pd.read_csv(x0TestPath, index_col=0).values
    x1_test = pd.read_csv(x1TestPath, index_col=0).values
    x2_test = pd.read_csv(x2TestPath, index_col=0).values

    x_train = np.hstack((
        np.expand_dims(x0_train, 1),
        np.expand_dims(x1_train, 1),
        np.expand_dims(x2_train, 1)
    ))
    x_test = np.hstack((
        np.expand_dims(x0_test, 1),
        np.expand_dims(x1_test, 1),
        np.expand_dims(x2_test, 1)
    ))
    
    return x_train, y_train, x_test
