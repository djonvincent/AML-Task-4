import numpy as np
from argparse import ArgumentParser
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import balanced_accuracy_score, classification_report
from load_data import load_data

N_SPLITS=3 # 3 subjects in train data

parser = ArgumentParser()
parser.add_argument('--test', dest='test', action='store_true', default=False)
parser.add_argument('--finetune', dest='finetune', action='store_true', default=False)
args = parser.parse_args()

#DATAFRAME
x_train, y_train, x_test = load_data()

def preprocess(x_raw, y_raw, x_test_raw, k_features):
    #SCALING
    scaler = StandardScaler().fit(x)
    x = scaler.transform(x)
    x_test = scaler.transform(x_test)

    #FEATURE SELECTION
    x, x_test = kBest(x, y, x_test, f_classif, k_features)

    return x, y, x_test

def CV_score(x, y, model, k_features, report=False):
    kf = KFold(n_splits=N_SPLITS, shuffle=False)
    total_score = 0
    y_pred_all = np.zeros(y.size, dtype=int)
    for train_idx, test_idx in kf.split(x):
        x_train_kf, y_train_kf, x_test_kf = preprocess(
            x[train_idx], y[train_idx], x[test_idx], k_features)
        model.fit(x_train_kf, y_train_kf)
        y_pred = model.predict(x_test_kf)
        y_pred_all[test_idx] = y_pred
        total_score += balanced_accuracy_score(y[test_idx], y_pred)
    if report:
        print(classification_report(y, y_pred_all)
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
    model = None
    avg_score = CV_score(x_train, y_train, model)
    print('Average BMAC:', avg_score)
