import numpy as np
from argparse import ArgumentParser
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from scipy.ndimage.filters import uniform_filter1d
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import balanced_accuracy_score, classification_report
from sklearn.svm import SVC
from load_data import load_data
from features import power_features, PLF, onset_features

N_SPLITS=3 # 3 subjects in train data

parser = ArgumentParser()
parser.add_argument('--test', dest='test', action='store_true', default=False)
parser.add_argument('--finetune', dest='finetune', action='store_true', default=False)
args = parser.parse_args()

#DATAFRAME
x_train, y_train, x_test = load_data()

def preprocess(x_raw, y, x_test_raw, k_features):
    #FEATURE EXTRACTION
    print('extracting features...')
    pf = power_features(x_raw[:, :2])
    pf_test = power_features(x_test_raw[:, :2])

    plf = PLF(x_raw[:, :2])
    plf_test = PLF(x_test_raw[:, :2])

    f_train = np.hstack((pf.reshape(-1, 10), plf.reshape(plf.shape[0], 1)))
    f_test = np.hstack((pf_test.reshape(-1, 10), plf_test.reshape(plf_test.shape[0], 1)))

    #SMOOTHING
    n = 7
    f_train = uniform_filter1d(f_train, axis=0, size=n, mode='nearest')
    f_test = uniform_filter1d(f_test, axis=0, size=n, mode='nearest')

    #SCALING
    scaler = StandardScaler().fit(f_train)
    f_train = scaler.transform(f_train)
    f_test = scaler.transform(f_test)

    #FEATURE SELECTION
    #x, x_test = kBest(x, y, x_test, f_classif, k_features)

    return f_train, f_test

def CV_score(x, y, model, k_features, report=False):
    kf = KFold(n_splits=N_SPLITS, shuffle=False)
    total_score = 0
    y_pred_all = np.zeros(y.size, dtype=int)
    for train_idx, test_idx in kf.split(x):
        x_train_kf, x_test_kf = preprocess(
            x[train_idx], y[train_idx], x[test_idx], k_features)
        model.fit(x_train_kf, y[train_idx])
        y_pred = model.predict(x_test_kf)
        y_pred_all[test_idx] = y_pred
        total_score += balanced_accuracy_score(y[test_idx], y_pred)
    if report:
        print(classification_report(y, y_pred_all))
    return total_score/N_SPLITS

if args.test:
    x_train, y_train, x_test = preprocess(x_train, y_train, x_test, args.k)

    y_test = np.concatenate((
        np.arange(y_test.size).reshape(-1, 1),
        y_test.reshape(-1, 1)), axis=1)
    np.savetxt(fname='y_test.csv', header='Id,y', delimiter=',', X=y_test,
            fmt=['%d', '%.5f'], comments='')

elif args.finetune:
    scores = []
    np.savetxt(fname='finetuning.csv', header='', delimiter=',',
            X=scores, fmt=[], comments='')
else:
    model = SVC(class_weight='balanced', kernel='rbf')
    avg_score = CV_score(x_train, y_train, model, 10)
    print('Average BMAC:', avg_score)

